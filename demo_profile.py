import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd


DEVICE = 'cuda'

IMAGE_SIZES = [
    (240, 320),
    (480, 640),
    (600, 800),
    (720, 1280),
    (1024, 1280),
    (1080, 1920),
    (1440, 2048),
    (1536, 2730),
    (1920, 2560)
]


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def generate_image(height, width):
    img = torch.rand(1, 3, height, width) * 255
    return img.to(DEVICE).float()

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        profiling_data = []

        for height, width in tqdm(IMAGE_SIZES):
            for i in range(10):
                image1 = generate_image(height, width)
                image2 = generate_image(height, width)

                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)

                # Discard the optical flow prediction and keep profiling information
                _, profiling_info = model(image1, image2, iters=args.valid_iters, test_mode=True, profile=True)

                # Save profiling information
                profiling_data.append(profiling_info)

    df = pd.DataFrame(profiling_data)

    grouped = df.groupby("image_size")

    # Step 3: Compute mean and std
    mean_df = grouped.mean()
    std_df = grouped.std()

    # Step 4: Plot with error bars
    plt.figure(figsize=(10, 6))

    for column in df.columns:
        if column != "image_size":
            plt.errorbar(
                mean_df.index/1e6,
                mean_df[column],
                yerr=std_df[column],
                label=column,
                capsize=3,
                marker='o',
                linestyle='--'
            )

    plt.xlabel("Image Size (Megapixels)")
    plt.ylabel("Time (seconds)")
    plt.title("Profiling RAFT Stereo Inference Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("profiling_summary.png", dpi=300)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
