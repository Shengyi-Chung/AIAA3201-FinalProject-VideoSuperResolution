"""
Discriminator architecture for Perceptual Enhancement (Stage 2).
Based on the UNet design from Real-ESRGAN to provide pixel-wise 
adversarial guidance, forcing the generator to recover realistic 
textures and sharp edges. 

Key features:
1. UNet structure: Captures both global semantics and local details.
2. Spectral Normalization (SN): Stabilizes GAN training and prevents mode collapse.
3. Skip connections: Improves gradient flow for finer texture discrimination.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class UNetDiscriminatorSN(nn.Module):
    """UNet Discriminator with Spectral Normalization."""

    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        
        # --- Encoder / Downsampling ---
        self.conv0 = spectral_norm(nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1))
        
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1))
        
        # --- Decoder / Upsampling ---
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1))
        
        # --- Output layers ---
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Downsampling
        feat0 = self.lrelu(self.conv0(x))
        feat1 = self.lrelu(self.conv1(feat0))
        feat2 = self.lrelu(self.conv2(feat1))
        feat3 = self.lrelu(self.conv3(feat2))
        
        # Upsampling with Skip Connections
        # Level 3 to 2
        out = self.upsample(feat3)
        out = self.lrelu(self.conv4(out))
        if self.skip_connection:
            out = out + feat2
            
        # Level 2 to 1
        out = self.upsample(out)
        out = self.lrelu(self.conv5(out))
        if self.skip_connection:
            out = out + feat1
            
        # Level 1 to 0
        out = self.upsample(out)
        out = self.lrelu(self.conv6(out))
        if self.skip_connection:
            out = out + feat0
            
        # Final decision map
        out = self.lrelu(self.conv7(out))
        out = self.lrelu(self.conv8(out))
        out = self.conv9(out)
        
        return out