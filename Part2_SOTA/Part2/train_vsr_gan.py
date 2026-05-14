import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F

# Import modules
from model_basicvsr import BasicVSR
from model_discriminator import UNetDiscriminatorSN
from loss_gan import PerceptualLoss, GANLoss
from vsr_dataset import Vimeo90KDataset 

# [Optimization 1] Define Charbonnier Loss, which is mentioned to be better than standard L1 in the paper
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize generator
    net_g = BasicVSR(spynet_path='weights/spynet.pth').to(device)
    
    pretrained_path = 'weights/basicvsr_stage1.pth'
    if os.path.exists(pretrained_path):
        print(f"Loading Stage 1 weights from {pretrained_path}...")
        checkpoint_g = torch.load(pretrained_path, map_location=device)
        net_g.load_state_dict(checkpoint_g)
    
    # 2. Initialize discriminator
    net_d = UNetDiscriminatorSN().to(device)
    
    # 3. Initialize loss functions
    cri_pix = CharbonnierLoss().to(device) # [Modified] Use Charbonnier Loss
    cri_perceptual = PerceptualLoss(model_path='weights/vgg19-dcbb9e9d.pth').to(device)
    cri_gan = GANLoss(gan_type='vanilla').to(device)

    # 4. [Optimization 2] Separate parameters, set lower learning rate for SPyNet
    spynet_params = []
    main_params = []
    for name, param in net_g.named_parameters():
        if 'spynet' in name:
            spynet_params.append(param)
        else:
            main_params.append(param)

    # SPyNet learning rate set to 1/4 or lower of other modules
    optimizer_g = optim.Adam([
        {'params': main_params, 'lr': 1e-4},
        {'params': spynet_params, 'lr': 2.5e-5} 
    ], betas=(0.9, 0.99))
    
    optimizer_d = optim.Adam(net_d.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # [Optimization 3] Introduce cosine annealing scheduler
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=100, eta_min=1e-7)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=100, eta_min=1e-7)

    # 5. Load data (can enable temporal flipping in Dataset)
    dataset = Vimeo90KDataset(
        data_root='/home/schung760/shared_data/project1',
        split='train',
        seq_length=7 # Keep 7 frames, but can do mirroring flip augmentation inside Dataset
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    print("--- Starting Stage 2 Optimized GAN Training ---")

    for epoch in range(100):
        for i, batch in enumerate(dataloader):
            lrs = batch['lr'].to(device)
            gts = batch['hr'].to(device)
            
            # [Optimization 4] Mirrored flip augmentation (manual flipping at data level, achieves 14-frame effect)
            # This step significantly increases VRAM pressure, can skip if OOM or reduce original sequence length
            # lrs = torch.cat([lrs, lrs.flip(1)], dim=1)
            # gts = torch.cat([gts, gts.flip(1)], dim=1)

            # --- Optimize generator net_g ---
            for p in net_d.parameters(): p.requires_grad = False
            optimizer_g.zero_grad()
            
            srs = net_g(lrs) 
            l_g_pix = cri_pix(srs, gts)

            t = srs.size(1)
            idx = torch.randint(0, t, (1,)).item()
            sr_frame = srs[:, idx]
            gt_frame = gts[:, idx]

            l_g_percep = cri_perceptual(sr_frame, gt_frame)
            pred_g_fake = net_d(sr_frame)
            l_g_gan = cri_gan(pred_g_fake, target_is_real=True)

            l_g_total = l_g_pix * 1.0 + l_g_percep * 1.0 + l_g_gan * 0.1
            l_g_total.backward()
            optimizer_g.step()

            # --- Optimize discriminator net_d ---
            for p in net_d.parameters(): p.requires_grad = True
            optimizer_d.zero_grad()
            l_d_real = cri_gan(net_d(gt_frame), target_is_real=True)
            l_d_fake = cri_gan(net_d(sr_frame.detach()), target_is_real=False)
            l_d_total = (l_d_real + l_d_fake) / 2
            l_d_total.backward()
            optimizer_d.step()

        # Update learning rates
        scheduler_g.step()
        scheduler_d.step()
        
        # Save every Epoch
        torch.save(net_g.state_dict(), f'weights/basicvsr_gan_optimized_epoch_{epoch}.pth')

if __name__ == '__main__':
    train()