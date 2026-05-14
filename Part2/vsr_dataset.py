import os
import torch
from torch.utils import data as data
from PIL import Image
from torchvision.transforms import ToTensor

'''
This script defines the Vimeo90KDataset class for Video Super-Resolution (VSR).
It is specifically adapted for a flattened Vimeo-90K dataset structure where 
frame sequences are stored directly within first-level numbered folders.

Key Features:
1. Flattened Structure Support: Handles directories where frames are located 
   at 'train_sharp/[folder_id]/[frame_id].png' instead of the standard 
   multi-level Vimeo-90K hierarchy.
2. Zero-Padded Frame Mapping: Maps frame indices to an 8-digit zero-padded 
   naming convention (e.g., '00000000.png' to '00000006.png') to match 
   custom dataset exports.
3. Sequential Loading: Loads a continuous sequence of T frames (default 7) 
   to support recurrent propagation and temporal alignment in models like BasicVSR.
4. Memory Management: Designed to work with high-resolution (720p) outputs 
   by supporting flexible sequence lengths and batch sizes to prevent OOM.
5. Path Validation: Includes an initialization check to ensure folders 
   actually contain the starting frame ('00000000.png') before adding to the index.
'''

class Vimeo90KDataset(data.Dataset):
    def __init__(self, data_root, split='train', seq_length=7):
        self.data_root = data_root
        self.split = split
        self.seq_length = seq_length
        
        # Locate HR and LR root directories
        self.hr_root = os.path.join(data_root, split, f'{split}_sharp')
        self.lr_root = os.path.join(data_root, split, f'{split}_sharp_bicubic', 'X4')
        
        self.samples = []
        
        if not os.path.exists(self.hr_root):
            print(f"Error: HR path not found at {self.hr_root}")
            return
        if not os.path.exists(self.lr_root):
            print(f"Error: LR path not found at {self.lr_root}")
            return

        # Scan logic: traverse folders like 000, 001, etc. under train_sharp
        all_folders = sorted(os.listdir(self.hr_root))
        for folder in all_folders:
            folder_path = os.path.join(self.hr_root, folder)
            if os.path.isdir(folder_path):
                # Check if the first frame image exists in this folder
                if os.path.exists(os.path.join(folder_path, '00000000.png')):
                    self.samples.append(folder)
        
        print(f"Dataset initialized: found {len(self.samples)} sequences.")

    def __getitem__(self, index):
        folder_name = self.samples[index]
        hr_frames = []
        lr_frames = []
        
        for i in range(self.seq_length):
            # Image naming convention: 00000000.png, 00000001.png ...
            img_name = f'{i:08d}.png'
            
            hr_img_path = os.path.join(self.hr_root, folder_name, img_name)
            lr_img_path = os.path.join(self.lr_root, folder_name, img_name)
            
            with Image.open(hr_img_path) as hr_img:
                hr_frames.append(ToTensor()(hr_img.convert('RGB')))
            with Image.open(lr_img_path) as lr_img:
                lr_frames.append(ToTensor()(lr_img.convert('RGB')))
            
        return {
            'lr': torch.stack(lr_frames), # [T, 3, H, W]
            'hr': torch.stack(hr_frames)  # [T, 3, 4H, 4W]
        }

    def __len__(self):
        return len(self.samples)