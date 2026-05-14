import torch
import torch.nn as nn
import torch.nn.functional as F
import os

'''
This script defines the complete SpyNet architecture for Optical Flow estimation.
SpyNet computes motion vectors between two frames by using a spatial pyramid.
The flow_warp function uses the estimated flow to align frames spatially.
'''

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        # Corresponds to .0.conv, .1.conv, etc. in weights
        self.basic_module = nn.ModuleList([
            nn.Sequential(nn.Conv2d(8, 32, 7, 1, 3), nn.ReLU(inplace=False)),
            nn.Sequential(nn.Conv2d(32, 64, 7, 1, 3), nn.ReLU(inplace=False)),
            nn.Sequential(nn.Conv2d(64, 32, 7, 1, 3), nn.ReLU(inplace=False)),
            nn.Sequential(nn.Conv2d(32, 16, 7, 1, 3), nn.ReLU(inplace=False)),
            nn.Sequential(nn.Conv2d(16, 2, 7, 1, 3))
        ])

    def forward(self, x):
        for layer in self.basic_module:
            x = layer(x)
        return x

class SpyNet(nn.Module):
    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        # SpyNet contains 6 pyramid sub-networks
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        
        if load_path and os.path.exists(load_path):
            # 1. First load the original weights
            checkpoint = torch.load(load_path, map_location='cpu')
            state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
            
            # 2. Core: Create mapping dictionary
            # Code error shows Missing: .0.0.weight, but Unexpected is .0.conv.weight
            # This means we need to replace .conv. with .0.
            new_state_dict = {}
            for k, v in state_dict.items():
                # This line handles all level nesting differences
                new_key = k.replace('.conv.', '.0.')
                new_state_dict[new_key] = v
            
            # 3. Load the modified dictionary
            self.load_state_dict(new_state_dict)
            print(f"SpyNet weights loaded and keys mapped from {load_path}")

        # Predefined mean/std for input normalization
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def flow_warp(self, x, flow):
        """Warp feature maps according to optical flow offset"""
        B, C, H, W = x.size()
        
        # Force check if flow size matches x size, if not, perform interpolation alignment
        if flow.size(2) != H or flow.size(3) != W:
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)

        # Generate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=x.device), 
            torch.arange(0, W, device=x.device), 
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2).float()  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)   # [B, H, W, 2]
        
        # Compute sampling grid: grid + flow
        vgrid = grid + flow.permute(0, 2, 3, 1)

        # Normalize to [-1, 1] range, which is required by grid_sample
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
        
        return F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, ref, supp):
        """Calculate Optical Flow between two frames"""
        ref = (ref - self.mean) / self.std
        supp = (supp - self.mean) / self.std

        ref_list = [ref]
        supp_list = [supp]

        # Build image pyramid (5-layer downsampling)
        for _ in range(5):
            ref_list.append(F.avg_pool2d(ref_list[-1], kernel_size=2, stride=2))
            supp_list.append(F.avg_pool2d(supp_list[-1], kernel_size=2, stride=2))

        ref_list = ref_list[::-1]
        supp_list = supp_list[::-1]

        B, _, H, W = ref_list[0].size()
        device = ref.device
        flow = torch.zeros(B, 2, H, W).to(device)

        # Compute and refine flow layer by layer
        for i in range(6):
            if i != 0:
                # Upsample flow from previous layer
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            
            # --- Core fix code: force alignment of dimensions ---
            # If upsampled flow shape differs from current pyramid layer (off by 1 pixel), interpolate to align
            if flow.shape[2:] != ref_list[i].shape[2:]:
                flow = F.interpolate(flow, size=ref_list[i].shape[2:], mode='bilinear', align_corners=True)
            # --------------------------------

            warped_supp = self.flow_warp(supp_list[i], flow)
            
            # Double-check warped_supp dimensions (just to be safe)
            if warped_supp.shape[2:] != ref_list[i].shape[2:]:
                warped_supp = F.interpolate(warped_supp, size=ref_list[i].shape[2:], mode='bilinear', align_corners=True)

            flow_input = torch.cat([ref_list[i], warped_supp, flow], dim=1)
            flow = flow + self.basic_module[i](flow_input)

        return flow