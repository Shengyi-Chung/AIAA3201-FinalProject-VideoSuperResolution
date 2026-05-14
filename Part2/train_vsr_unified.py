"""
Part 2: SOTA VSR Pipeline - Unified Training Manager

This script provides a high-level interface for:
1. Stage 1: BasicVSR training with L1/Charbonnier loss (Feature Alignment)
2. Stage 2: BasicVSR + GAN training with Perceptual Loss (Perceptual Enhancement)

Usage:
  python train_vsr_unified.py --stage 1 --config config.yaml
  python train_vsr_unified.py --stage 2 --pretrained weights/basicvsr_stage1.pth
"""

import argparse
import os
import sys
import yaml
from pathlib import Path


def setup_directories(config):
    """Create necessary directories for checkpoints, logs, and results."""
    dirs = ['checkpoint_dir', 'log_dir', 'result_dir', 'weight_dir']
    for key in dirs:
        if key in config:
            os.makedirs(config[key], exist_ok=True)
    return config


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config():
    """Return default configuration."""
    return {
        'data_root': '/path/to/vimeo90k',
        'batch_size': 2,
        'num_workers': 4,
        'num_epochs': 50,
        'lr': 2e-4,
        'weight_decay': 0,
        'grad_clip': 5.0,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'result_dir': './results',
        'weight_dir': './weights',
        'device': 'cuda',
        'seed': 42,
    }


def train_stage1(config):
    """
    Stage 1: Train BasicVSR with L1/Charbonnier loss.
    
    Goal: Feature Alignment & Reconstruction
    - Maximize PSNR (typical: 28-32 dB)
    - Ensure structural color accuracy
    - Preserve optical flow stability
    """
    print("=" * 60)
    print("🟢 STAGE 1: Feature Alignment & Reconstruction")
    print("=" * 60)
    print(f"📦 Config: {config}")
    
    try:
        from train_vsr_stage1 import train
        config = setup_directories(config)
        train(config)
        print("✅ Stage 1 training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Stage 1 training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def train_stage2(config, pretrained_path):
    """
    Stage 2: Train BasicVSR with GAN + Perceptual Loss.
    
    Goal: Perceptual Enhancement
    - Recover realistic textures
    - Sharp edges and fine details
    - Visual quality > PSNR (typical: 27-30 dB but much better perception)
    - Requires Stage 1 pretrained weights
    """
    print("=" * 60)
    print("🟠 STAGE 2: Perceptual Enhancement (GAN Training)")
    print("=" * 60)
    print(f"📦 Config: {config}")
    print(f"📥 Loading pretrained Stage 1: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        print(f"❌ Pretrained checkpoint not found: {pretrained_path}")
        print("   Please complete Stage 1 training first!")
        return False
    
    try:
        from train_vsr_gan import train
        config['pretrained_path'] = pretrained_path
        config = setup_directories(config)
        train(config)
        print("✅ Stage 2 training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Stage 2 training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_inference(checkpoint_path, input_dir, output_dir):
    """Run inference on a video sequence."""
    print("=" * 60)
    print("🔵 INFERENCE: Video Super-Resolution")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from inference_vsr import main
        print(f"📥 Input:  {input_dir}")
        print(f"📤 Output: {output_dir}")
        main(checkpoint_path, input_dir, output_dir)
        print("✅ Inference completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation(checkpoint_path, data_root, output_csv):
    """Evaluate model on validation set."""
    print("=" * 60)
    print("🟣 EVALUATION: Metrics Calculation (PSNR, SSIM)")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    try:
        from eval_vsr import evaluate
        print(f"📥 Checkpoint: {checkpoint_path}")
        print(f"📊 Output CSV: {output_csv}")
        evaluate(checkpoint_path, data_root, output_csv)
        print("✅ Evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Part 2: SOTA VSR Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Stage 1
  python train_vsr_unified.py --stage 1 --config config.yaml
  
  # Train Stage 2 (requires Stage 1 checkpoint)
  python train_vsr_unified.py --stage 2 --pretrained weights/basicvsr_stage1.pth
  
  # Run inference
  python train_vsr_unified.py --infer --checkpoint weights/basicvsr_gan.pth \\
    --input-dir ./val_data/000 --output-dir ./results/000
  
  # Evaluate
  python train_vsr_unified.py --eval --checkpoint weights/basicvsr_gan.pth \\
    --data-root /path/to/vimeo90k --output-csv ./metrics.csv
        """
    )
    
    parser.add_argument('--stage', type=int, choices=[1, 2], 
                        help='Training stage: 1 (L1) or 2 (GAN)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--pretrained', type=str,
                        help='Path to pretrained checkpoint (for Stage 2)')
    
    parser.add_argument('--infer', action='store_true',
                        help='Run inference mode')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation mode')
    parser.add_argument('--checkpoint', type=str,
                        help='Model checkpoint path')
    parser.add_argument('--input-dir', type=str,
                        help='Input video frame directory (for inference)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (for inference)')
    parser.add_argument('--data-root', type=str,
                        help='Data root directory (for evaluation)')
    parser.add_argument('--output-csv', type=str,
                        help='Output CSV path for metrics (for evaluation)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute requested task
    if args.stage:
        print("\n" + "=" * 60)
        print("🎬 PART 2: SOTA Video Super-Resolution Pipeline")
        print("=" * 60 + "\n")
        
        if args.stage == 1:
            success = train_stage1(config)
            if not success:
                sys.exit(1)
                
        elif args.stage == 2:
            if not args.pretrained:
                parser.error("--stage 2 requires --pretrained argument")
            success = train_stage2(config, args.pretrained)
            if not success:
                sys.exit(1)
    
    elif args.infer:
        if not args.checkpoint or not args.input_dir or not args.output_dir:
            parser.error("--infer requires --checkpoint, --input-dir, --output-dir")
        success = run_inference(args.checkpoint, args.input_dir, args.output_dir)
        if not success:
            sys.exit(1)
    
    elif args.eval:
        if not args.checkpoint or not args.data_root or not args.output_csv:
            parser.error("--eval requires --checkpoint, --data-root, --output-csv")
        success = run_evaluation(args.checkpoint, args.data_root, args.output_csv)
        if not success:
            sys.exit(1)
    
    else:
        parser.print_help()
        print("\n⚠️  Please specify --stage, --infer, or --eval")


if __name__ == '__main__':
    main()
