from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from core.raft_stereo import RAFTStereo
from core.disparity_upsampler import DisparityUpsampler

from evaluate_stereo import *
from evaluate_upsample import *
import core.stereo_datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def validate_upsampler(model, valid_loader, args):
    model.eval()
    results = {}
    total_epe = 0
    total_pixels = 0
    
    with torch.no_grad():
        for _, *data_blob in valid_loader:
            image1, image2, flow_gt, valid_gt = [x.cuda() for x in data_blob]
            
            # Downsample the ground truth flow for input to the model
            flow_low_res = F.interpolate(flow_gt, scale_factor=1/2, mode='area')
            
            # Run the upsampler
            upsampled_flow = model(image1, flow_low_res)
            
            # Calculate metrics
            epe = torch.sum((upsampled_flow - flow_gt)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid_gt.view(-1) > 0.5]
            
            if epe.numel() > 0:
                total_epe += epe.sum().item()
                total_pixels += epe.numel()
    
    avg_epe = total_epe / max(total_pixels, 1)
    
    # Compute additional metrics
    results = {
        'epe': avg_epe,
        '1px': 100.0 * (epe < 1).float().mean().item() if epe.numel() > 0 else 0,
        '3px': 100.0 * (epe < 3).float().mean().item() if epe.numel() > 0 else 0,
        '5px': 100.0 * (epe < 5).float().mean().item() if epe.numel() > 0 else 0,
    }
    
    # Optional: compare to a simple bilinear upsampling baseline
    bilinear_upsampled = F.interpolate(flow_low_res, scale_factor=2, mode='bilinear', align_corners=False)
    bilinear_epe = torch.sum((bilinear_upsampled - flow_gt)**2, dim=1).sqrt()
    bilinear_epe = bilinear_epe.view(-1)[valid_gt.view(-1) > 0.5]
    
    if bilinear_epe.numel() > 0:
        results['bilinear_epe'] = bilinear_epe.mean().item()
        results['relative_improvement'] = 100.0 * (1.0 - avg_epe / results['bilinear_epe'])
    
    return results

def disparity_to_image(disparity, max_disp=None):
    """Convert disparity to RGB image for visualization
    
    Args:
        disparity: Tensor of shape [B, 1, H, W] or [1, H, W]
        max_disp: Maximum disparity value for normalization (if None, uses max value in tensor)
    
    Returns:
        RGB image of shape [B, 3, H, W] or [3, H, W]
    """
    if disparity.dim() == 3:
        disparity = disparity.unsqueeze(0)
    
    batch_size, _, height, width = disparity.shape
    
    # Normalize disparity to [0,1] range
    if max_disp is None:
        max_disp = disparity.max()
    
    # Apply colormap (using typical disparity coloring - from blue to red)
    norm_disp = disparity / (max_disp + 1e-6)  # Add epsilon to avoid division by zero
    
    # Create RGB image using a color mapping
    # Blue (far) to Red (near)
    rgb = torch.zeros(batch_size, 3, height, width, device=disparity.device)
    
    # Red channel (increases with disparity)
    rgb[:, 0] = norm_disp.squeeze(1)
    
    # Green channel (parabolic response - highest in middle range)
    rgb[:, 1] = 1.0 - 4.0 * (norm_disp.squeeze(1) - 0.5)**2
    
    # Blue channel (decreases with disparity)
    rgb[:, 2] = 1.0 - norm_disp.squeeze(1)
    
    # Clip values to [0,1] range
    rgb = torch.clamp(rgb, 0, 1)
    
    # If input was 3D tensor, return 3D tensor
    if disparity.shape[0] == 1:
        rgb = rgb.squeeze(0)
    
    return rgb

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    print(flow_preds.shape)
    print(flow_gt.shape)
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='cos', div_factor=25, final_div_factor=10000)

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def masked_loss(pred, target, valid_mask):
    # Ensure valid_mask has same shape as pred and target
    if valid_mask.shape != pred.shape:
        valid_mask = valid_mask.unsqueeze(1)
    
    # Normalize mask to sum to 1 to maintain scale
    norm_factor = valid_mask.sum() + 1e-8
    
    # L1 loss on valid pixels only
    loss = (torch.abs(pred - target) * valid_mask).sum() / norm_factor
    
    # Calculate metrics
    with torch.no_grad():
        epe = torch.sum((pred - target)**2, dim=1).sqrt()
        # Only consider valid pixels for metrics
        valid_epe = epe.view(-1)[valid_mask.view(-1) > 0.5]
        metrics = {
            'epe': valid_epe.mean().item() if len(valid_epe) > 0 else float('inf'),
            '1px': (valid_epe < 1).float().mean().item() if len(valid_epe) > 0 else 0.0,
            '3px': (valid_epe < 3).float().mean().item() if len(valid_epe) > 0 else 0.0,
            '5px': (valid_epe < 5).float().mean().item() if len(valid_epe) > 0 else 0.0,
        }
    
    return loss, metrics

def train(args):

    SCALE = 4

    # There is a naming convention mismatch where RAFT Stereo refers to flow even though it is calcualting disparity
    # This is a holdover from RAFT calculating flow
    # Here we are upsampling disparity, naming conventions are slightly mixed where it intermingles with RAFT code
    
    # Initialize the DisparityUpsampler model
    model = DisparityUpsampler(
        image_dim=3,  
        feature_dim=256, 
        scale=SCALE  # TODO make this a command line argument
    ).cuda()
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()

    validation_frequency = 100

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    # print("Loading Data")
    # single_batch = next(iter(train_loader))
    # _, *data_blob = single_batch
    print("Data loaded")
    while should_keep_training:
        print("Training...")
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
        # for i_batch in tqdm(range(10000)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]
            flow[valid.unsqueeze(1)<0.5] = -300
            
            flow_low_res = F.interpolate(flow, scale_factor=1.0/SCALE, mode='area')

            flow_low_res = flow_low_res
            
            assert model.training
            upsampled_flow = model(image1, flow_low_res) # Calling it flow but actually disparity
            assert model.training

            # loss, metrics = sequence_loss(upsampled_flow, flow, valid)
            loss, metrics = masked_loss(upsampled_flow, flow, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # Visualization 
            # if i_batch % 100 == 0:
            #     idx = 0  # First image in batch
            #     # Create visualization grid with low-res input, upsampled prediction, and ground truth
            #     with torch.no_grad():
            #         # Get shapes
            #         low_res_shape = flow_low_res[idx].shape[-2:]  # Height, width
            #         up_res_shape = upsampled_flow[idx].shape[-2:]
            #         gt_res_shape = flow[idx].shape[-2:]
                    
            #         # Choose which resolution to use (original ground truth is often best)
            #         target_shape = gt_res_shape
                    
            #         # Resize all to target shape
            #         vis_low_res = F.interpolate(flow_low_res[idx:idx+1], size=target_shape, mode='bilinear', align_corners=False)
                    
            #         # Only resize upsampled if needed
            #         if up_res_shape != target_shape:
            #             vis_up_flow = F.interpolate(upsampled_flow[idx:idx+1], size=target_shape, mode='bilinear', align_corners=False)
            #         else:
            #             vis_up_flow = upsampled_flow[idx:idx+1]
                    
            #         # Create visualization
            #         vis_imgs = torch.cat([
            #             disparity_to_image(vis_low_res[0]),  # Remove batch dimension
            #             disparity_to_image(vis_up_flow[0]),
            #             disparity_to_image(flow[idx])
            #         ], dim=2)
            #     logger.writer.add_image(f'flow_vis/sample_{i_batch}', vis_imgs, global_batch_num)
  
            logger.push(metrics)


            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(f'checkpoints/{total_steps + 1}_upsampler_{4}x_{args.name}.pth')
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                results = validate_middlebury(model)                     
                logger.write_dict(results)

                model.train()
                # model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path(f'checkpoints/{total_steps + 1}_upsampler_{4}x_{args.name}.pth')
            logging.info(f"Saving file {save_path.absolute()}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    save_path = Path(f'checkpoints/{total_steps + 1}_upsampler_{4}x_{args.name}.pth')
    torch.save(model.state_dict(), save_path)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='trainwgt', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['middlebury_2014'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.00015, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=4000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256], help="size of the random image crops used during training.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)