# Part 1: Parameters & Configuration Guide

## 🔧 Complete Parameter Reference

### Training Parameters (`train_srcnn.py`)

#### Dataset & I/O
```
--project1-root PATH
  Type: str
  Default: "/home/schung760/shared_data/project1"
  Description: Root path containing train/ and val/ folders
  Example: /data/vimeo90k

--save-dir PATH
  Type: str
  Default: "./checkpoints"
  Description: Directory to save model checkpoints
  Effect: Checkpoints saved to {save-dir}/srcnn_best.pt

--max-train-pairs INT
  Type: int
  Default: None (use all)
  Description: Limit number of training image pairs
  Effect: Useful for quick testing
  Example: --max-train-pairs 200 → Use first 200 pairs

--max-val-pairs INT
  Type: int
  Default: None (use all)
  Description: Limit number of validation pairs
  Effect: Speeds up validation loop
  Example: --max-val-pairs 100 → Use first 100 pairs

--max-train-patches INT
  Type: int
  Default: None (extract all)
  Description: Limit patches extracted from training set
  Effect: Reduces memory usage
  Example: --max-train-patches 100000 → Extract 100k patches

--max-val-patches INT
  Type: int
  Default: None (extract all)
  Description: Limit patches extracted from validation set
  Example: --max-val-patches 10000 → Extract 10k patches
```

#### SRCNN Architecture
```
--patch-size INT
  Type: int
  Default: 33
  Range: 20-50
  Description: Input patch size (H×W)
  Note: Larger = bigger receptive field but slower
  Effect on output: output_size = input_size - (receptive_field - 1)
  Example: 33 → outputs 21×21

--label-size INT
  Type: int
  Default: 21
  Range: 15-30
  Description: Output patch size (target size)
  Constraint: label_size < patch_size
  Formula: output_size = (patch_size - 9 - 1 + 2) - 5 + 1 = patch_size - 12
  For patch_size=33: output = 21 ✓

--stride INT
  Type: int
  Default: 14
  Range: 5-20
  Description: Patch extraction stride
  Effect: Higher stride → fewer patches → faster training but less data
  Recommended: stride = (patch_size - label_size) / 2 = (33-21)/2 = 6? (but 14 used empirically)
  Example values:
    - stride=5  → dense patches, more training samples
    - stride=14 → sparser patches, fewer samples
    - stride=20 → very sparse, minimal samples
```

#### Training Hyperparameters
```
--epochs INT
  Type: int
  Default: 10
  Range: 1-500
  Description: Number of training epochs
  Timing: Each epoch ≈ 1-2 minutes on GPU
  Recommendation:
    - Quick test: 1-5 epochs
    - Standard: 50-200 epochs
    - Full: 200+ epochs
  Convergence: Usually plateaus around epoch 100-150

--batch-size INT
  Type: int
  Default: 32
  Range: 4-256
  Description: Training batch size
  Memory requirements:
    - batch=8  → 2-3 GB VRAM
    - batch=32 → 4-6 GB VRAM
    - batch=64 → 8-12 GB VRAM
  GPU recommendations:
    - 4GB VRAM:  batch=8
    - 8GB VRAM:  batch=16-32
    - 16GB VRAM: batch=32-64

--lr FLOAT
  Type: float
  Default: 1e-4 (0.0001)
  Range: 1e-6 to 1e-2
  Description: Learning rate for Adam optimizer
  Effect: Too high → divergence; Too low → slow convergence
  Recommendations:
    - Default (1e-4): Works for most cases
    - High variation: Try 1e-5 (more stable)
    - Fast training: Try 5e-4 (riskier)
  Plateau behavior: If loss plateaus, reduce to 1e-5

--num-workers INT
  Type: int
  Default: 4
  Range: 0-16
  Description: Number of data loading workers
  Effect: More workers → faster loading but higher CPU usage
  Recommendations:
    - num_workers = 2-4 × num_cpus (for GPU)
    - num_workers = 0 on CPU machines
  When to increase:
    - If training ~100% GPU utilization but slow iteration
    - If CPU has many cores (8+)
```

#### Loss & Optimization
```
[implicit] Optimizer: Adam
  Learning rate: --lr value
  Beta1: 0.9
  Beta2: 0.999
  Epsilon: 1e-8

[implicit] Loss function: L1 (Mean Absolute Error)
  Formula: L = mean(|pred - target|)
  Why L1? More robust than MSE for SR
```

---

### Evaluation Parameters (`eval.py`)

#### Dataset
```
--project1-root PATH
  Type: str
  Default: "/home/schung760/shared_data/project1"
  Description: Root path to dataset

--split STR
  Type: str
  Choices: ["train", "val"]
  Default: "val"
  Description: Which split to evaluate on
  Recommendation: Use "val" for fair comparison
```

#### Model & Inference
```
--srcnn-ckpt PATH
  Type: str
  Required: True
  Description: Path to SRCNN checkpoint
  Example: checkpoints/srcnn_best.pt
  Effect: If not found, error

--device STR
  Type: str
  Default: "cuda" if available else "cpu"
  Description: Computation device
```

#### Methods to Evaluate
```
[implicit] Methods: bicubic, lanczos, srcnn
  All three are always evaluated
  No option to select subset
```

#### Output
```
--csv-out PATH
  Type: str
  Default: "results_part1.csv"
  Description: Output CSV file
  Content: method, psnr_y, ssim_y, fps
```

#### Limits
```
--max-pairs INT
  Type: int
  Default: None
  Description: Limit number of image pairs to evaluate
  Example: --max-pairs 50 → Evaluate on first 50 pairs
```

---

### Temporal Baseline Parameters (`temporal_baseline.py`)

#### Dataset
```
--test-root PATH
  Type: str
  Default: "/home/schung760/shared_data/project1/vimeo_super_resolution_test"
  Description: Path to test sequences
```

#### Temporal Parameters
```
--weights FLOAT FLOAT FLOAT
  Type: list[float]
  Default: [0.25, 0.5, 0.25]
  Description: Temporal fusion weights [prev, center, next]
  Constraint: Sum should ≈ 1.0
  Examples:
    - [0.25, 0.5, 0.25]: Equal-ish weighting
    - [0.0, 1.0, 0.0]:  No temporal fusion
    - [0.2, 0.6, 0.2]:  Center-focused
    - [1/3, 1/3, 1/3]: Uniform weighting

--sigma FLOAT
  Type: float
  Default: 1.0
  Range: 0.1-3.0
  Description: Gaussian blur sigma for unsharp masking
  Effect: Higher → stronger blur → less sharpening
  Recommendations:
    - sigma=0.5: Subtle sharpening
    - sigma=1.0: Default (balanced)
    - sigma=2.0: Strong blur (risk of artifacts)

--amount FLOAT
  Type: float
  Default: 0.6
  Range: 0.0-2.0
  Description: Sharpening strength for unsharp masking
  Formula: sharp = img + amount × (img - blur)
  Effect: Higher → more aggressive sharpening
  Recommendations:
    - amount=0.3: Subtle
    - amount=0.6: Default (noticeable)
    - amount=1.5: Aggressive (risk of halos)
```

#### Output
```
--csv-out PATH
  Type: str
  Default: "temporal_metrics.csv"
  Description: Output CSV file
  Content: sequence, frame, avg_psnr, avg_ssim, unsharp_psnr, unsharp_ssim
```

#### Limits
```
--max-seqs INT
  Type: int
  Default: None
  Description: Limit number of sequences to evaluate
  Example: --max-seqs 10 → First 10 sequences only
```

---

### Inference Parameters (`infer_srcnn.py`)

#### Dataset & Model
```
--project1-root PATH
  Type: str
  Default: "/home/schung760/shared_data/project1"
  Description: Root to dataset

--split STR
  Type: str
  Choices: ["train", "val"]
  Default: "val"
  Description: Which split to inference on

--srcnn-ckpt PATH
  Type: str
  Default: (hardcoded to common path)
  Description: SRCNN checkpoint path
```

#### Output
```
--out-dir PATH
  Type: str
  Default: "./outputs_srcnn_preview"
  Description: Output directory for preview images
  Files generated:
    - {idx}_lr.png: Upsampled input
    - {idx}_sr.png: SRCNN prediction
    - {idx}_gt.png: Ground truth
    - {idx}_compare.png: Side-by-side
```

#### Limits
```
--max-pairs INT
  Type: int
  Default: 8
  Description: Number of image pairs to visualize
  Example: --max-pairs 20 → Generate 20 sets of preview images
```

---

### Reporting Parameters (`report_summary.py`)

```
--spatial-csv PATH
  Type: str
  Required: True
  Description: Path to spatial results CSV
  Expected format: results_part1.csv

--temporal-csv PATH
  Type: str
  Required: True
  Description: Path to temporal metrics CSV
  Expected format: temporal_metrics.csv

--out PATH
  Type: str
  Required: True
  Description: Output markdown file
  Default suggestion: report_summary.md
```

---

## 📊 Preset Configurations

### Configuration 1: Quick Test (2 minutes)
```bash
python train_srcnn.py \
  --project1-root /data/vimeo90k \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --max-train-pairs 100 \
  --max-val-pairs 50 \
  --max-train-patches 2000 \
  --max-val-patches 500 \
  --save-dir checkpoints
```

### Configuration 2: Light Training (20 minutes)
```bash
python train_srcnn.py \
  --project1-root /data/vimeo90k \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 2 \
  --max-train-pairs 500 \
  --max-val-pairs 200 \
  --max-train-patches 10000 \
  --max-val-patches 2000 \
  --save-dir checkpoints
```

### Configuration 3: Standard Training (2-4 hours)
```bash
python train_srcnn.py \
  --project1-root /data/vimeo90k \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 4 \
  --max-train-pairs 2000 \
  --max-val-pairs 500 \
  --max-train-patches 50000 \
  --max-val-patches 10000 \
  --save-dir checkpoints
```

### Configuration 4: Full Training (6-10 hours)
```bash
python train_srcnn.py \
  --project1-root /data/vimeo90k \
  --epochs 200 \
  --batch-size 32 \
  --num-workers 4 \
  --num-workers 4 \
  --save-dir checkpoints
  # Uses all data, no max limits
```

### Configuration 5: Aggressive Learning (higher risk)
```bash
python train_srcnn.py \
  --project1-root /data/vimeo90k \
  --epochs 100 \
  --batch-size 64 \
  --lr 5e-4 \
  --save-dir checkpoints
  # Larger batch, higher LR - may diverge but trains fast
```

### Configuration 6: Conservative Learning (more stable)
```bash
python train_srcnn.py \
  --project1-root /data/vimeo90k \
  --epochs 300 \
  --batch-size 16 \
  --lr 1e-5 \
  --save-dir checkpoints
  # Lower LR, smaller batch - stable but slow
```

---

## 🎯 Parameter Tuning Guidelines

### If Training is Too Slow
1. Reduce `--num-workers` (if already high)
2. Increase `--batch-size` (if VRAM allows)
3. Reduce `--max-train-patches`
4. Reduce `--epochs` for testing

### If GPU Memory is Full
1. Reduce `--batch-size` (halve it)
2. Reduce `--max-train-patches`
3. Increase `--num-workers` (paradoxically helps CPU-GPU balance)

### If Training Diverges (loss → inf)
1. Reduce `--lr` by 10× (1e-4 → 1e-5)
2. Reduce `--batch-size`
3. Check data for NaN values

### If Training Plateaus (loss stuck)
1. Reduce `--lr` by 2-5×
2. Increase `--epochs` to give more iterations
3. Check if model has converged (PSNR saturated?)

### If Validation Loss is Much Higher Than Training
1. Data mismatch (train ≠ val distribution)
2. Reduce `--lr` slightly to prevent overfitting
3. Add `--max-train-pairs` limit to match complexity

---

## 📈 Recommended Starting Point

```bash
# For most users:
python train_srcnn.py \
  --project1-root /path/to/project1 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --save-dir checkpoints
```

Then evaluate:
```bash
python eval.py \
  --project1-root /path/to/project1 \
  --split val \
  --srcnn-ckpt checkpoints/srcnn_best.pt \
  --csv-out results.csv
```

---

## ⚙️ Advanced: Manual Hyperparameter Search

```bash
# Try different learning rates
for lr in 1e-5 1e-4 5e-4; do
  python train_srcnn.py \
    --lr $lr \
    --epochs 30 \
    --save-dir checkpoints_lr_${lr}
done

# Evaluate each
python eval.py \
  --srcnn-ckpt checkpoints_lr_1e-5/srcnn_best.pt \
  --csv-out results_lr_1e-5.csv
```

---

## 📝 Quick Reference Card

```
TRAINING SPEED: --batch-size ↑, --num-workers ↑
TRAINING QUALITY: --epochs ↑, --lr careful tuning
MEMORY: --batch-size ↓, --max-train-patches ↓
STABILITY: --lr ↓, --batch-size ↓
CONVERGENCE: --epochs ↑, --lr tuned
```

---

See README.md for complete documentation.
