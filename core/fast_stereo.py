import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8
import time
import random


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class FASTStereo(nn.Module):
    def __init__(self, raft_model, upsampler_model, freeze_raft=False, freeze_upsampler=False):
        super(FASTStereo, self).__init__()
        self.raft = raft_model.module
        self.upsampler = upsampler_model

        if freeze_raft:
            for param in self.raft.parameters():
                param.requires_grad = False
                
        if freeze_upsampler:
            for param in self.upsampler.parameters():
                param.requires_grad = False 
    
    def extract_patch(self, image, i, j, patch_h, patch_w):
        """
        Extract a patch from the image given the top-left corner (i, j) and the dimensions (patch_h, patch_w).
        """
        if len(image.shape) == 4:  # [B, C, H, W]
            patch = image[:, :, i:i+patch_h, j:j+patch_w]
        elif len(image.shape) == 3:  # [C, H, W]
            patch = image[:, i:i+patch_h, j:j+patch_w]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
            
        return patch

    def forward(self, image1, image2, i, j, size=256, iters=12, scale=4, test_mode=False, profile=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        image1_lowres = F.interpolate(image1, scale_factor=1.0/scale, mode='area')
        image2_lowres = F.interpolate(image2, scale_factor=1.0/scale, mode='area')

        patch = self.extract_patch(image1, i, j, size, size)

        print(f'Processing Image: {image1.shape} Iterations: {iters}')
        
        if profile:
            torch.cuda.synchronize()
            start_raft = time.perf_counter()
        
        disp_raw, disp_raft, profiling_info = self.raft(image1_lowres, image2_lowres, iters=iters, test_mode=test_mode, profile=profile)
        
        if profile:
            torch.cuda.synchronize()
            end_raft = time.perf_counter()
            raft_time = end_raft - start_raft
        
        if profile:
            torch.cuda.synchronize()
            start_upsampler = time.perf_counter()
            
        # Extract a patch from RAFT disparity map that corresponds to our full resolution patch
        disp_raft_patch = self.extract_patch(disp_raft, i//4, j//4, size//4, size//4)
        
        disp_patch = self.upsampler(patch, disp_raft_patch)

        if profile:
            torch.cuda.synchronize()
            end_upsampler = time.perf_counter()
            upsampler_time = end_upsampler-start_upsampler
        
        
        if test_mode:
            if profile:
                profiling_info = {
                    "image_size": image1.shape[2]*image1.shape[3],
                    "raft_time": raft_time,
                    "upsampler_time": upsampler_time
                }
                return disp_raw, disp_raft, disp_patch, profiling_info
        
            return disp_raw, disp_raft, disp_patch
        
        return disp_raw, disp_raft, disp_patch
