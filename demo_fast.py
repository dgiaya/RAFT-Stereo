import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from fast_stereo import FASTStereo
from disparity_upsampler import DisparityUpsampler
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import random
import cv2


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def extract_patch(image, i, j, patch_h, patch_w):
        """
        Extract a patch from the image given the top-left corner (i, j) and the dimensions (patch_h, patch_w).

        Args:
            image (torch.Tensor): The input image tensor of shape (C, H, W).
            i (int): The starting row index of the patch.
            j (int): The starting column index of the patch.
            patch_h (int): Height of the extracted patch.
            patch_w (int): Width of the extracted patch.
        Returns:
            patch (torch.Tensor): The extracted patch of shape (C, patch_h, patch_w).
        """
        patch = image[:, i:i+patch_h, j:j+patch_w]
        return patch

def random_patch(image, min_dim, max_dim):
    """
    Randomly extract a patch from the image.
    Args:
        image (torch.Tensor): The input image tensor of shape (C, H, W).
        min_dim (int): Minimum dimension of the patch.
        max_dim (int): Maximum dimension of the patch.
    Returns:
        patch (torch.Tensor): The extracted patch of shape (C, patch_h, patch_w).
        i (int): The starting row index of the patch.
        j (int): The starting column index of the patch.
        patch_h (int): Height of the extracted patch.
        patch_w (int): Width of the extracted patch.
    """

    # Handle batch dimension if present
    if len(image.shape) == 4:  # [B, C, H, W]
        _, _, H_img, W_img = image.shape
    elif len(image.shape) == 3:  # [C, H, W]
        _, H_img, W_img = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    patch_h = random.randint(min_dim, min(max_dim, H_img))
    patch_w = random.randint(min_dim, min(max_dim, W_img))
    i = random.randint(0, H_img - patch_h)
    j = random.randint(0, W_img - patch_w)

    # Extract patch based on image dimensionality
    if len(image.shape) == 4:
        patch = image[:, :, i:i+patch_h, j:j+patch_w]
    else:
        patch = image[:, i:i+patch_h, j:j+patch_w]
        
    return patch, i, j, patch_h, patch_w


def combine_images(disp_raft, disp_patch, i, j, box_thickness=2):
    # Convert tensors to numpy if needed
    if isinstance(disp_raft, torch.Tensor):
        disp_raft = disp_raft.cpu().numpy().squeeze()
    if isinstance(disp_patch, torch.Tensor):
        disp_patch = disp_patch.cpu().numpy().squeeze()
    
    # Upsample the RAFT disparity by 4x
    h, w = disp_raft.shape
    upsampled_disp = cv2.resize(disp_raft, (w*4, h*4), interpolation=cv2.INTER_NEAREST)
    
    # Get patch dimensions
    patch_h, patch_w = disp_patch.shape
    
    # Create a copy of the upsampled map
    combined_disp = upsampled_disp.copy()
    
    # Insert the patch at the specified location
    # Make sure i and j are scaled to match the upsampled coordinates
    i_upsampled = i
    j_upsampled = j
    
    # Ensure we stay within bounds
    # if i_upsampled + patch_h > combined_disp.shape[0]:
    #     patch_h = combined_disp.shape[0] - i_upsampled
    # if j_upsampled + patch_w > combined_disp.shape[1]:
    #     patch_w = combined_disp.shape[1] - j_upsampled
    
    # Insert the patch
    combined_disp[i_upsampled:i_upsampled+patch_h, j_upsampled:j_upsampled+patch_w] = disp_patch[:patch_h, :patch_w]
    
    # Convert to a color image for drawing the box
    # Normalizing between 0-1 for matplotlib
    norm_disp = (combined_disp - combined_disp.min()) / (combined_disp.max() - combined_disp.min())
    color_map = plt.get_cmap('jet')
    colored_disp = (color_map(norm_disp) * 255).astype(np.uint8)
    
    # Draw the box
    # Format of coordinates for rectangle: (x, y, width, height)
    cv2.rectangle(
        colored_disp, 
        (j_upsampled, i_upsampled), 
        (j_upsampled + patch_w, i_upsampled + patch_h), 
        (255, 0, 0), 
        box_thickness
    )
    
    return colored_disp

def demo(args):
    raft_model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    raft_model.load_state_dict(torch.load(args.raft_ckpt))

    upsampler_model = DisparityUpsampler(image_dim=args.patch_dim, feature_dim=args.upsampler_feat_dim, scale=args.downsample_scale)
    upsampler_model.load_state_dict(torch.load(args.upsample_ckpt)) 

    model = FASTStereo(raft_model, upsampler_model, freeze_raft=True, freeze_upsampler=True)

    # model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # Determine a random patch of interest
            _, i, j, patch_h, patch_w = random_patch(image1, args.patch_size, args.patch_size)

            # padder = InputPadder(image1.shape, divis_by=32)
            # image1, image2 = padder.pad(image1, image2)

            disp_raw, disp_raft, disp_patch, profiling_info = model(image1, image2, i, j, size=args.patch_size, iters=args.valid_iters, test_mode=True, profile=True)
            # flow_up = padder.unpad(flow_up).squeeze()

            file_stem = imfile1.split('/')[-2]
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp_raft.cpu().numpy().squeeze())
            # Find global min/max across both images to use consistent scaling
            min_val = min(disp_raft.min().item(), disp_patch.min().item())
            max_val = max(disp_raft.max().item(), disp_patch.max().item())

            # Save with consistent scaling
            plt.imsave(output_directory / f"{file_stem}_raft.png", -disp_raft.cpu().numpy().squeeze(), 
                    cmap='jet', vmin=-max_val, vmax=-min_val)
            plt.imsave(output_directory / f"{file_stem}_patch.png", -disp_patch.cpu().numpy().squeeze(), 
                    cmap='jet', vmin=-max_val, vmax=-min_val)
            
            combined_image = combine_images(disp_raft, disp_patch, i, j)
            plt.imsave(output_directory / f"{file_stem}_combined.png", combined_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raft_ckpt', help="raft checkpoint", required=True)
    parser.add_argument('--upsample_ckpt', help="upsample checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/Test/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/Test/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_fast_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=7, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--patch_dim', type=int, default=3, help="Channels of high resolution patch")
    parser.add_argument('--patch_size', type=int, default=256, help="Dimension of high resolution patch")
    parser.add_argument('--downsample_scale', type=int, default=4, help="Downsample image before RAFT")
    parser.add_argument('--upsampler_feat_dim', type=int, default=256, help="Feature dimension of the upsampler")

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg_cuda", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=2, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
