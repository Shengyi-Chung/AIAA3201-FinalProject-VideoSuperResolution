import os
'''Combine three comparison videos in the visual folder into one
Input: Manually provide three video paths
Output: Saved in visual folder
Usage:
python visual_compare.py
'''
def create_triple_comparison():
    # Set three sources
    # Path example: LR (Bicubic), SR (Your result), GT (Ground Truth)
    v1 = "/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/visual/000_result.mp4"
    v2 = "/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/visual/val_sharp_bicubic_X4_000.mp4"
    v3 = "/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/visual/val_val_sharp_000.mp4"
    
    output = "visual/triple_comparison.mp4"
    # Check if files exist
    for v in [v1, v2, v3]:
        if not os.path.exists(v):
            print(f"Error: Cannot find file {v}")
            return

    # --- Build FFmpeg command ---
    # When using f-string, [0:v] in FFmpeg filters doesn't need special escaping,
    # but if the command contains { }, it needs to be written as {{ }}.
    # Here we added scale filter to ensure three videos have consistent height (unified to 720p)
    filter_str = (
        "[0:v]scale=-1:720,drawtext=text='Input (LR)':fontcolor=white:fontsize=40:x=10:y=10[v0]; "
        "[1:v]scale=-1:720,drawtext=text='Ours (BasicVSR)':fontcolor=white:fontsize=40:x=10:y=10[v1]; "
        "[2:v]scale=-1:720,drawtext=text='Target (GT)':fontcolor=white:fontsize=40:x=10:y=10[v2]; "
        "[v0][v1][v2]hstack=inputs=3"
    )

    cmd = f'ffmpeg -i {v1} -i {v2} -i {v3} -filter_complex "{filter_str}" -c:v libx264 -pix_fmt yuv420p {output} -y'

    print("Executing command...")
    print(cmd) # Print for easy debugging
    
    ret = os.system(cmd)
    
    if ret == 0:
        print(f"\nSuccess! Comparison video saved to: {output}")
    else:
        print("\nComposition failed, please check if FFmpeg is installed and paths are correct.")

if __name__ == "__main__":
    create_triple_comparison()