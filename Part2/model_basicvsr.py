import torch
import torch.nn as nn
import torch.nn.functional as F
from model_spynet import SpyNet

'''
This script implements the BasicVSR model for Video Super-Resolution.
Features:
1. Bidirectional Propagation: Processes frames in both forward and backward 
   directions to aggregate temporal information.
2. Optical Flow Alignment: Uses SpyNet to warp previous/future features 
   to the current frame's coordinate.
3. Residual Reconstruction: Reconstructs high-resolution frames using fused features.
'''

class ResidualBlockNoBN(nn.Module):
    """Simple Residual Block without Batch Normalization for stability in SR."""
    def __init__(self, num_feat=64):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class BasicVSR(nn.Module):
    def __init__(self, spynet_path, num_feat=64, num_block=30):
        super(BasicVSR, self).__init__()
        self.num_feat = num_feat

        # 1. Optical Flow Network (SpyNet)
        self.spynet = SpyNet(load_path=spynet_path)

        # 2. Propagation Branches
        self.backward_resblocks = nn.ModuleList([ResidualBlockNoBN(num_feat * 2) for _ in range(num_block)])
        self.forward_resblocks = nn.ModuleList([ResidualBlockNoBN(num_feat * 2) for _ in range(num_block)])
        
        # 3. Feature Extractors & Reconstruction Layers
        self.feat_extract = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        
        # Upsampling (PixelShuffle x4)
        self.upsample1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upsample2 = nn.Conv2d(num_feat, 3 * 4, 3, 1, 1)
        self.pix_shuffle = nn.PixelShuffle(2)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lrs):
        """
        Args:
            lrs (Tensor): Input LR sequence [b, t, 3, h, w]
        Returns:
            Tensor: Output HR sequence [b, t, 3, 4h, 4w]
        """
        b, t, c, h, w = lrs.size()

        # Step 1: Feature Extraction
        # [b*t, 3, h, w] -> [b*t, c, h, w] -> [b, t, c, h, w]
        feats = self.lrelu(self.feat_extract(lrs.view(-1, c, h, w)))
        feats = feats.view(b, t, -1, h, w)

        # Step 2: Backward Propagation
        outputs_back = []
        back_feat = torch.zeros(b, self.num_feat, h, w).to(lrs.device)
        for i in range(t - 1, -1, -1):
            curr_frame = lrs[:, i, :, :, :]
            if i < t - 1:
                next_frame = lrs[:, i + 1, :, :, :]
                flow = self.spynet(curr_frame, next_frame) # Compute flow: current -> next
                back_feat = self.spynet.flow_warp(back_feat, flow) # Warp next hidden state to current
            
            # Concatenate current feature and warped previous hidden state
            back_feat = torch.cat([feats[:, i, :, :, :], back_feat], dim=1)
            for block in self.backward_resblocks:
                back_feat = block(back_feat)
            
            # Keep only num_feat channels for next iteration
            back_feat = back_feat[:, :self.num_feat, :, :]
            outputs_back.append(back_feat)
        outputs_back = outputs_back[::-1] # Reverse to 0 -> t-1

        # Step 3: Forward Propagation
        outputs_forward = []
        forward_feat = torch.zeros(b, self.num_feat, h, w).to(lrs.device)
        for i in range(0, t):
            curr_frame = lrs[:, i, :, :, :]
            if i > 0:
                prev_frame = lrs[:, i - 1, :, :, :]
                flow = self.spynet(curr_frame, prev_frame) # Compute flow: current -> previous
                forward_feat = self.spynet.flow_warp(forward_feat, flow)
            
            forward_feat = torch.cat([feats[:, i, :, :, :], forward_feat], dim=1)
            for block in self.forward_resblocks:
                forward_feat = block(forward_feat)
            
            forward_feat = forward_feat[:, :self.num_feat, :, :]
            outputs_forward.append(forward_feat)

        # Step 4: Fusion and Upsampling
        outputs = []
        for i in range(t):
            # Combine forward and backward features
            combined = torch.cat([outputs_forward[i], outputs_back[i]], dim=1)
            out = self.lrelu(self.fusion(combined))
            
            # x2 Upsampling
            out = self.pix_shuffle(self.upsample1(out))
            out = self.lrelu(out)
            # x2 Upsampling -> Total x4
            out = self.pix_shuffle(self.upsample2(out))
            outputs.append(out)

        return torch.stack(outputs, dim=1)