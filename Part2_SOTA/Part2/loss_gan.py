"""
Loss functions for Generative Adversarial Network (GAN) training.
Includes:
1. Perceptual Loss: Uses a pre-trained VGG19 network to compare high-level 
   features of SR and GT images.
2. GAN Loss: Standard Vanilla or LSGAN loss for the adversarial game 
   between Generator and Discriminator.
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19
import os

class PerceptualLoss(nn.Module):
    """VGG19 based Perceptual Loss with local weight loading."""
    def __init__(self, model_path='weights/vgg19-dcbb9e9d.pth', layer_idx=35):
        super(PerceptualLoss, self).__init__()
        
        # 1. Create model structure without weights
        vgg_model = vgg19(weights=None)
        
        # 2. Check and load local weight file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weight file not found: {model_path}. Please ensure it is manually uploaded to this path.")
            
        print(f"Loading VGG19 weights from local path: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        vgg_model.load_state_dict(state_dict)
        
        # 3. Extract feature extraction layer (features part)
        vgg_features = vgg_model.features
        self.vgg = nn.Sequential(*list(vgg_features.children())[:layer_idx]).eval()
        
        # 4. Freeze all parameters, not participate in training
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()

        # ImageNet normalization parameters expected by VGG (can be done here if not in Dataloader)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, gt):
        # Normalize input from [0, 1] to VGG expected distribution
        sr = (sr - self.mean) / self.std
        gt = (gt - self.mean) / self.std
        
        sr_feat = self.vgg(sr)
        gt_feat = self.vgg(gt)
        return self.criterion(sr_feat, gt_feat)

class GANLoss(nn.Module):
    """Adversarial Loss for GAN training."""
    def __init__(self, gan_type='vanilla'):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        if gan_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f'GAN type {gan_type} is not supported.')

    def get_target_label(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input)
        else:
            return torch.zeros_like(input)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        return self.criterion(input, target_label)