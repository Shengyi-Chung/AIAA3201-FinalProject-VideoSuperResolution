'''
This script performs Video Super-Resolution on a sequence of ANY length.
It automatically counts the number of frames in the folder and processes 
them using BasicVSR's recurrent propagation.
'''

import os
import torch
from model_basicvsr import BasicVSR
from torchvision.utils import save_image
from PIL import Image
from torchvision.transforms import ToTensor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Configuration ---
    target_folder = "000" 
    lr_root = "/home/schung760/shared_data/project1/val/val_sharp_bicubic/X4"
    checkpoint_path = "/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_gan.pth"
    output_base_dir = "./vsrgan_result"
    
    input_dir = os.path.join(lr_root, target_folder)
    output_dir = os.path.join(output_base_dir, target_folder)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Auto-detect number of frames
    # Scan all .png files and sort
    all_imgs = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    num_frames = len(all_imgs)
    print(f"Found {num_frames} frames in folder '{target_folder}'.")

    # 2. Load model
    model = BasicVSR(spynet_path='weights/spynet.pth').to(device)
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Read all frames
    lr_frames = []
    for img_name in all_imgs:
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        lr_frames.append(ToTensor()(img))

    # 4. Perform inference
    with torch.no_grad():
        # Stack into [1, T, 3, H, W], T can be any number at this time
        input_tensor = torch.stack(lr_frames).unsqueeze(0).to(device)
        
        print(f"Processing {num_frames} frames... (This may take longer)")
        # BasicVSR's internal forward function will automatically loop based on T size
        output = model(input_tensor) 
        
        # 5. Save all results
        for i, img_name in enumerate(all_imgs):
            save_image(output[0, i], os.path.join(output_dir, img_name))
            
    print(f"Success! Processed {num_frames} frames.")

if __name__ == "__main__":
    main()