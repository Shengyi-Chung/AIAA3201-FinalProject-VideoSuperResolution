'''
This script evaluates the performance of the trained BasicVSR model.
It calculates PSNR and SSIM for the center frame of each sequence and saves
visual comparison results.

Key Features:
1. Metric Calculation: Computes PSNR and SSIM for the center frame of each
    sequence to quantify reconstruction quality.
2. Visual Comparison: Generates side-by-side images (Model Output vs. Ground Truth)
    to assess perceptual clarity and detail restoration.
3. Automated Directory Management: Automatically creates and stores all
    outputs in a chosen directory for organized post-training analysis.

Usage:
python eval_vsr.py --checkpoint-path ./weights/basicvsr_stage1.pth --save-dir ./evalresult/basicvsr --csv-out ./evalresult/basicvsr_stage1_metrics.csv
'''

import argparse
import csv
import os
from math import log10
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model_basicvsr import BasicVSR
from vsr_dataset import Vimeo90KDataset

def calculate_psnr(img1, img2):
    """Calculates PSNR between two images in [0, 1] range."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: 
        return 100
    return 20 * log10(1.0 / mse.item()**0.5)


def calculate_ssim(img1, img2):
    """Calculates SSIM between two images in [0, 1] range."""
    img1_np = img1.permute(1, 2, 0).detach().cpu().numpy()
    img2_np = img2.permute(1, 2, 0).detach().cpu().numpy()
    return float(structural_similarity(img1_np, img2_np, data_range=1.0, channel_axis=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BasicVSR checkpoint on Vimeo-90K val split")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_stage1.pth"),
        help="Path to the model checkpoint (.pth)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("./evalresult"),
        help="Directory to save visual comparisons",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV path to save per-sequence PSNR/SSIM",
    )
    parser.add_argument(
        "--val-data-root",
        type=Path,
        default=Path("/home/schung760/shared_data/project1/val"),
        help="Root path of the validation split",
    )
    return parser.parse_args()

def evaluate():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_data_root = str(args.val_data_root)
    checkpoint_path = str(args.checkpoint_path)
    save_dir = str(args.save_dir)
    
    os.makedirs(save_dir, exist_ok=True)

    # Dynamically adjust Dataset path, because the folder is called val_sharp
    dataset = Vimeo90KDataset(val_data_root, seq_length=7)
    # Force override dataset internal path to match val directory
    dataset.hr_root = os.path.join(val_data_root, 'val_sharp')
    dataset.lr_root = os.path.join(val_data_root, 'val_sharp_bicubic', 'X4')
    # Rescan the correct path
    dataset.samples = []
    all_folders = sorted(os.listdir(dataset.hr_root))
    for folder in all_folders:
        if os.path.isdir(os.path.join(dataset.hr_root, folder)):
            dataset.samples.append(folder)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Load model
    model = BasicVSR(spynet_path='weights/spynet.pth').to(device)
    
    checkpoint = torch.load(checkpoint_path)
    
    # --- Fix 2: Compatible with different save formats ---
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If state_dict was directly saved
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    avg_psnr = 0
    avg_ssim = 0
    num_samples = len(loader)
    print(f"Starting evaluation on {num_samples} sequences...")

    csv_rows = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            lrs = batch['lr'].to(device)
            hrs = batch['hr'].to(device)
            
            # Inference
            output = model(lrs) # [1, 7, 3, 720, 1280]
            
            # Evaluate the center frame (index 3 for a 7-frame sequence)
            mid = 3 
            current_psnr = calculate_psnr(output[0, mid], hrs[0, mid])
            current_ssim = calculate_ssim(output[0, mid], hrs[0, mid])
            avg_psnr += current_psnr
            avg_ssim += current_ssim
            csv_rows.append([i, current_psnr, current_ssim])
            
            # Save visual comparison every 5 sequences
            if i % 5 == 0:
                # Stack Output and HR horizontally for easy comparison
                comparison = torch.cat([output[0, mid], hrs[0, mid]], dim=2)
                save_path = os.path.join(save_dir, f"result_seq_{i:03d}_psnr_{current_psnr:.2f}_ssim_{current_ssim:.4f}.png")
                save_image(comparison, save_path)
                print(f"Saved visual comparison: {save_path}")

    final_psnr = avg_psnr / num_samples
    final_ssim = avg_ssim / num_samples

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sequence_idx', 'psnr', 'ssim'])
            writer.writerows(csv_rows)
            # Append overall averages for easy reference
            writer.writerow([])
            writer.writerow(['average', f"{final_psnr:.4f}", f"{final_ssim:.4f}"])

    print("\n" + "="*30)
    print(f"Evaluation Complete!")
    print(f"Final Average PSNR: {final_psnr:.2f} dB")
    print(f"Final Average SSIM: {final_ssim:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results saved in: {os.path.abspath(save_dir)}")
    print("="*30)

if __name__ == "__main__":
    evaluate()