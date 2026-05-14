'''
This script implements Stage 1 training for BasicVSR.
Goal: Feature Alignment and Reconstruction using Pixel-wise Loss (L1).

Key Features:
1. Loads pre-trained SpyNet weights to ensure stable optical flow estimation.
2. Uses L1 Loss to maximize PSNR and ensure structural/color accuracy.
3. Implements Gradient Clipping to prevent gradient explosion in RNN structures.
4. Redirects all Checkpoints and Logs to storage drives to save root partition space.
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import modules
from vsr_dataset import Vimeo90KDataset
from model_basicvsr import BasicVSR

def train():
    # --- 1. Environment and path configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_name = "basicvsr_stage1_l1"
    
    # Ensure save path is on a drive with sufficient space
    base_path = "/home/schung760/my_storage_1T"
    save_dir = os.path.join(base_path, "checkpoints", exp_name)
    log_dir = os.path.join(base_path, "logs", exp_name)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # --- 2. Hyperparameter settings ---
    batch_size = 1        # Reduce to 2 if GPU OOM
    num_epochs = 50       # Stage 1 usually needs more epochs to converge
    lr = 2e-4             # Initial learning rate
    seq_length = 7        # Sequence length during training (Vimeo90K Septuplet has 7 frames)
    
    # --- 3. Data loading ---
    # Modify data_root according to the actual path of Vimeo90K on your server
    data_root = "/home/schung760/shared_data/project1/train"
    
    train_dataset = Vimeo90KDataset(data_root, seq_length=seq_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

    # --- 4. Initialize model, loss function, and optimizer ---
    # Automatically load the fixed SpyNet weights
    model = BasicVSR(spynet_path='weights/spynet.pth').to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Learning rate schedule: halve every 20 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    writer = SummaryWriter(log_dir)

    print(f"--- Stage 1 Training Started ---")
    print(f"Device: {device}")
    print(f"Checkpoints: {save_dir}")
    print(f"Logs: {log_dir}")

    # --- 5. Training loop ---
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for batch in pbar:
            # batch['lr']: [B, T, 3, H, W]
            # batch['hr']: [B, T, 3, 4H, 4W]
            lrs = batch['lr'].to(device)
            hrs = batch['hr'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(lrs) # [B, T, 3, 4H, 4W]
            
            # Calculate loss
            loss = criterion(outputs, hrs)
            
            # Backward pass
            loss.backward()
            
            # --- Critical step: Gradient clipping ---
            # BasicVSR has recurrent structure, prone to gradient explosion, limit norm to 0.01
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            
            optimizer.step()
            
            # Record and update progress bar
            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(L1_Loss=f"{loss.item():.4f}", LR=f"{optimizer.param_groups[0]['lr']:.2e}")
            
            if global_step % 10 == 0:
                writer.add_scalar('Train/Loss_L1', loss.item(), global_step)

        # Update learning rate after each Epoch
        scheduler.step()
        
        # --- 6. Save Checkpoint ---
        # Save latest model
        torch.save(model.state_dict(), os.path.join(save_dir, "latest.pth"))
        
        # Save a backup every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

    writer.close()
    print("Training Complete!")

if __name__ == "__main__":
    train()