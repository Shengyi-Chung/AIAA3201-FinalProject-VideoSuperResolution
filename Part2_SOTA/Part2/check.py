'''This is a test and debug tool, the content is not important'''


"""
Integration test script to verify the compatibility of:
1. UNetDiscriminatorSN (model_discriminator.py)
2. PerceptualLoss & GANLoss (loss_gan.py)
3. BasicVSRNet (model_basicvsr.py)
"""

import torch
import torch.nn as nn
from model_basicvsr import BasicVSR
from model_discriminator import UNetDiscriminatorSN
from loss_gan import PerceptualLoss, GANLoss

def test_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # --- 1. Test Generator (BasicVSR) ---
    print("\n[Test 1] Generator Forward...")
    # Input dimension: (Batch=1, Time=3, Channel=3, H=64, W=64)
    dummy_lr = torch.randn(1, 3, 3, 64, 64).to(device)
    net_g = BasicVSR(spynet_path='weights/spynet_20210409-c6c1bd09.pth').to(device)
    srs = net_g(dummy_lr)
    print(f"SR Output Shape: {srs.shape}") # Expected: (1, 3, 3, 256, 256)
    assert srs.shape == (1, 3, 3, 256, 256), "Generator output shape mismatch!"

    # --- 2. Test Discriminator (UNetDiscriminatorSN) ---
    print("\n[Test 2] Discriminator Forward...")
    # Randomly select one SR frame for testing
    sr_frame = srs[:, 0, :, :, :] 
    net_d = UNetDiscriminatorSN().to(device)
    d_out = net_d(sr_frame)
    print(f"Discriminator Output Shape: {d_out.shape}")
    # UNet discriminator usually outputs a map with the same resolution as input or reduced
    assert d_out.dim() == 4, "Discriminator output should be 4D (N, 1, H, W)"

    # --- 3. Test Loss Functions ---
    print("\n[Test 3] Loss Functions...")
    # Make sure the local VGG weight path is correct
    try:
        cri_percep = PerceptualLoss(model_path='weights/vgg19-dcbb9e9d.pth').to(device)
        dummy_gt_frame = torch.randn(1, 3, 256, 256).to(device)
        l_percep = cri_percep(sr_frame, dummy_gt_frame)
        print(f"Perceptual Loss: {l_percep.item():.4f}")
    except Exception as e:
        print(f"Perceptual Loss Error: {e}")
        print("Tip: Check if 'weights/vgg19-dcbb9e9d.pth' exists.")

    cri_gan = GANLoss(gan_type='vanilla').to(device)
    l_gan = cri_gan(d_out, target_is_real=True)
    print(f"GAN Loss: {l_gan.item():.4f}")

    # --- 4. Test Backward Pass ---
    print("\n[Test 4] Backward Pass...")
    total_loss = l_percep + l_gan
    total_loss.backward()
    print("Backward pass successful!")

    print("\n✅ All systems GO! Your GAN pipeline is ready.")

if __name__ == "__main__":
    # Before execution, ensure the weights folder has the corresponding pth files
    test_pipeline()