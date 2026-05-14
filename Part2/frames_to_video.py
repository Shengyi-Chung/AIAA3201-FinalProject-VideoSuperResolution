'''
This script converts an image sequence into a video file.
Input: Manually provided directory path.
Output: Saved in the './visual' directory.
Usage:
python frames_to_video.py
'''

import cv2
import os

def images_to_video(input_dir, output_path, fps=24):
    """Core function: Convert image sequence in folder to original .mp4 video"""
    if not os.path.exists(input_dir):
        print(f"Skipped: Path does not exist -> {input_dir}")
        return False

    # Get and sort images
    images = sorted([img for img in os.listdir(input_dir) if img.endswith(".png")])
    if not images:
        print(f"Skipped: No images -> {input_dir}")
        return False

    # Read first image to get resolution
    sample_img = cv2.imread(os.path.join(input_dir, images[0]))
    h, w, _ = sample_img.shape
    
    # Use mp4v codec (saves original data directly without H.264 conversion)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Writing: {os.path.basename(output_path)} ...")
    for filename in images:
        img = cv2.imread(os.path.join(input_dir, filename))
        out.write(img)

    out.release()
    return True

def main():
    # 1. Sequence ID input
    seq_id = "000"  # You can change this ID to process different video sequences, e.g., "001", "002", ..., "010"
    
    # 2. Configure root directory paths (ensure these paths are correct on your server)
    base_lr = "/home/schung760/shared_data/project1/val/val_sharp_bicubic/X4"
    base_gt = "/home/schung760/shared_data/project1/val/val_sharp"
    base_sr = "vsrgan_result"
    
    # 3. Create output subdirectory visual/000
    output_dir = os.path.join("visual", seq_id)
    os.makedirs(output_dir, exist_ok=True)

    # 4. Execute tasks
    tasks = [
        (os.path.join(base_lr, seq_id), os.path.join(output_dir, f"{seq_id}_LR.mp4")),
        (os.path.join(base_gt, seq_id), os.path.join(output_dir, f"{seq_id}_GT.mp4")),
        (os.path.join(base_sr, seq_id), os.path.join(output_dir, f"{seq_id}_SR.mp4"))
    ]

    print(f"\n--- Processing started ---")
    for in_path, out_path in tasks:
        if images_to_video(in_path, out_path):
            print(f"Generated: {out_path}")
    
    print(f"\n All done. Right-click 'visual/{seq_id}' folder on the left in VS Code and download to view locally.")

if __name__ == "__main__":
    main()