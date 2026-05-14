# Part 1: Architecture & Dependency Diagrams

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Part 1: Baseline SR                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐         ┌──────────────────┐          │
│  │   Vimeo90K      │         │  Pre-trained     │          │
│  │   Dataset       │         │  Weights         │          │
│  │                 │         │  (optional)      │          │
│  │ • train_sharp   │         │                  │          │
│  │ • train_lr_4x   │         │ • srcnn_best.pt  │          │
│  │ • val_sharp     │         │                  │          │
│  │ • test (public) │         └──────────────────┘          │
│  └────────┬────────┘                │                       │
│           │                          │                       │
│           └──────────────┬───────────┘                       │
│                          │                                   │
│                   ┌──────▼─────────┐                        │
│                   │  dataset.py    │                        │
│                   │                │                        │
│                   │ • Load images  │                        │
│                   │ • Extract      │                        │
│                   │   patches      │                        │
│                   │ • Convert to Y │                        │
│                   └──────┬─────────┘                        │
│                          │                                   │
│           ┌──────────────┼──────────────┐                    │
│           │              │              │                    │
│     ┌─────▼─────┐  ┌────▼─────┐  ┌────▼─────┐             │
│     │ train_    │  │ eval.py   │  │ temporal_│             │
│     │ srcnn.py  │  │           │  │ baseline │             │
│     │           │  │ Methods:  │  │ .py      │             │
│     │ SRCNN     │  │ • Bicubic │  │          │             │
│     │ Training  │  │ • Lanczos │  │ Methods: │             │
│     │           │  │ • SRCNN   │  │ • Avg    │             │
│     │ 3-layer   │  │           │  │ • Unsharp│             │
│     │ CNN       │  │ Metrics:  │  │          │             │
│     │           │  │ • PSNR/   │  │ Metrics: │             │
│     └─────┬─────┘  │   SSIM    │  │ • PSNR/  │             │
│           │        │ • FPS     │  │   SSIM   │             │
│           │        └────┬──────┘  └────┬─────┘             │
│           │             │              │                    │
│     ┌─────▼────────────┐│              │                    │
│     │checkpoints/      ││              │                    │
│     │srcnn_best.pt     │└──────┬───────┘                    │
│     │(55k params)      │       │                            │
│     └────────┬─────────┘       │                            │
│              │                 │                            │
│              │    ┌────────────▼────────────┐               │
│              │    │   results_part1.csv     │               │
│              │    │   temporal_metrics.csv  │               │
│              │    └────────────┬────────────┘               │
│              │                 │                            │
│         ┌────▼─────────────────▼─────────┐                 │
│         │   report_summary.py            │                 │
│         │                                │                 │
│         │ Aggregates CSVs into markdown  │                 │
│         └────────────┬────────────────────┘                │
│                      │                                     │
│              ┌───────▼─────────┐                           │
│              │report_summary.md│                           │
│              │                 │                           │
│              │ • Spatial table │                           │
│              │ • Temporal table│                           │
│              │ • Improvement %│                           │
│              └─────────────────┘                           │
│                                                             │
│  [Optional] infer_srcnn.py                                 │
│             └─ outputs_srcnn_preview/                      │
│                  {idx}_lr/sr/gt/compare.png                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Diagram

### Training Phase
```
Vimeo90K Dataset
    │
    ├─→ Train Split
    │   └─→ dataset.py (SRCNNPatchDataset)
    │       └─→ Extract 33×33 patches (stride=14)
    │           └─→ train_srcnn.py
    │               ├─→ Forward: Patch → SRCNN → 21×21 output
    │               ├─→ Loss: L1(output - target)
    │               ├─→ Backward: Update weights (Adam, lr=1e-4)
    │               └─→ Checkpoint: srcnn_best.pt (best val loss)
    │
    └─→ Val Split
        └─→ dataset.py (SRCNNPatchDataset)
            └─→ Extract patches
                └─→ Evaluate on val loss
                    └─→ Save if best
```

### Evaluation Phase
```
Vimeo90K Dataset (Val/Test)
    │
    ├─→ Bicubic Interpolation
    │   └─→ Upsample LR 4× → eval.py
    │       └─→ Compare with HR using PSNR/SSIM
    │           └─→ results_part1.csv (row: bicubic)
    │
    ├─→ Lanczos Resampling
    │   └─→ Upsample LR 4× → eval.py
    │       └─→ Compare with HR using PSNR/SSIM
    │           └─→ results_part1.csv (row: lanczos)
    │
    └─→ SRCNN Model
        ├─→ Load srcnn_best.pt
        ├─→ Upsample LR 4× → eval.py
        ├─→ Run through SRCNN patches
        └─→ Compare with HR using PSNR/SSIM
            └─→ results_part1.csv (row: srcnn)
```

### Temporal Fusion Phase
```
Vimeo90K Test Set (7-frame sequences)
    │
    ├─→ For each center frame [i-1, i, i+1]
    │   │
    │   ├─→ Upsample each 4× using Bicubic
    │   │
    │   ├─→ Temporal Weighted Average
    │   │   output[i] = 0.25×[i-1] + 0.5×[i] + 0.25×[i+1]
    │   │
    │   ├─→ Compare with HR → PSNR/SSIM
    │   │
    │   └─→ Optionally apply Unsharp Mask
    │       └─→ sharp = img + 0.6×(img - blur)
    │           └─→ Compare with HR → PSNR/SSIM
    │
    └─→ Aggregate all frames
        └─→ temporal_metrics.csv (MEAN row)
```

---

## 🔀 Execution Flow Diagram

```
START
  │
  ├─→ [Check] Dataset exists?
  │   └─→ No: ERROR - Download Vimeo90K
  │
  ├─→ [Option 1] Train SRCNN from scratch
  │   │
  │   └─→ train_srcnn.py
  │       ├─→ dataset.py loads train patches
  │       ├─→ Loop epochs=1 to 200:
  │       │   ├─→ Batch training
  │       │   ├─→ Validate on val split
  │       │   ├─→ Save best checkpoint
  │       │   └─→ Early stopping?
  │       └─→ checkpoints/srcnn_best.pt
  │
  ├─→ [Check] srcnn_best.pt exists?
  │   ├─→ No: ERROR or go to Option 1
  │   └─→ Yes: Continue
  │
  ├─→ eval.py (Spatial Baselines)
  │   ├─→ Load srcnn_best.pt
  │   ├─→ For each method in [bicubic, lanczos, srcnn]:
  │   │   ├─→ Load images from val split
  │   │   ├─→ Apply upsampling
  │   │   ├─→ Compute PSNR/SSIM/FPS
  │   │   └─→ Append to CSV
  │   └─→ results_part1.csv
  │
  ├─→ temporal_baseline.py (Temporal Fusion)
  │   ├─→ For each 7-frame sequence:
  │   │   ├─→ For each center frame i:
  │   │   │   ├─→ Extract [i-1, i, i+1]
  │   │   │   ├─→ Apply weighted average
  │   │   │   ├─→ Compute PSNR/SSIM
  │   │   │   ├─→ Optionally apply unsharp
  │   │   │   └─→ Append to CSV
  │   │   └─→ Next sequence
  │   └─→ temporal_metrics.csv
  │
  ├─→ [Optional] infer_srcnn.py (Visual Preview)
  │   ├─→ Load srcnn_best.pt
  │   ├─→ For each of max_pairs images:
  │   │   ├─→ Load LR and HR
  │   │   ├─→ Run SRCNN inference
  │   │   ├─→ Save comparison images
  │   │   └─→ Next image
  │   └─→ outputs_srcnn_preview/{idx}_{lr,sr,gt,compare}.png
  │
  ├─→ report_summary.py (Generate Summary)
  │   ├─→ Load results_part1.csv
  │   ├─→ Load temporal_metrics.csv
  │   ├─→ Format as markdown tables
  │   ├─→ Compute improvement percentages
  │   └─→ report_summary.md
  │
  └─→ END: All results ready
```

---

## 🔗 Dependency Chain

```
LEVEL 0: External Dependencies
├─ pytorch
├─ opencv-python (cv2)
├─ numpy
└─ scikit-image

LEVEL 1: Core Module
├─ dataset.py
│   └─ Uses: LEVEL 0
└─ train_srcnn.py
    ├─ Defines: SRCNN class
    └─ Uses: dataset.py, LEVEL 0

LEVEL 2: Evaluation & Analysis
├─ eval.py
│   ├─ Uses: train_srcnn.py (SRCNN class)
│   ├─ Uses: dataset.py
│   └─ Creates: results_part1.csv
│
├─ temporal_baseline.py
│   ├─ Uses: LEVEL 0 (no Part1 files)
│   └─ Creates: temporal_metrics.csv
│
└─ infer_srcnn.py
    ├─ Uses: train_srcnn.py (SRCNN class)
    ├─ Uses: dataset.py
    └─ Creates: outputs_srcnn_preview/

LEVEL 3: Reporting
├─ report_summary.py
│   ├─ Reads: results_part1.csv
│   ├─ Reads: temporal_metrics.csv
│   └─ Creates: report_summary.md
│
└─ run_part1_report.sh
    └─ Orchestrates: All above in sequence
```

---

## 📈 Data Size Progression

```
Vimeo90K Dataset (original)
├─ Train: ~30GB (64,612 images)
├─ Val: ~1GB (3,782 images)
└─ Test: ~0.5GB (1,881 images)

After Processing (in memory):
├─ Train patches (33×33): ~1.5M patches
├─ Val patches: ~500K patches
└─ Test frames (7-frame sequences): ~1.5M frames

Model Checkpoints:
├─ srcnn_best.pt: ~220KB (55k params)
└─ Other checkpoints: Similar size

CSV Outputs:
├─ results_part1.csv: ~1KB (3-5 rows)
├─ temporal_metrics.csv: ~50KB (1000+ rows)
└─ report_summary.md: ~10KB
```

---

## ⏱️ Timing Breakdown

```
Training SRCNN (depends on data size)
├─ Epoch 1 (quick test):        2-5 minutes
├─ Epoch 50 (half training):    50-100 minutes
└─ Epoch 200 (full training):   200-400 minutes

Evaluation (eval.py)
├─ Bicubic (very fast):         1-3 minutes
├─ Lanczos (faster):            2-5 minutes
└─ SRCNN (slowest):             10-20 minutes
Total spatial:                  15-30 minutes

Temporal Baseline
├─ Weighted average:            5-10 minutes
└─ + Unsharp masking:           <1 minute
Total temporal:                 5-10 minutes

Visualization & Reporting
├─ infer_srcnn.py (8 images):   2-5 minutes
├─ report_summary.py:           <1 minute
└─ Total utility:               5 minutes

TOTAL END-TO-END: 4-7 hours (depending on dataset size)
```

---

## 🎯 Key Metrics Definitions

### PSNR (Peak Signal-to-Noise Ratio)
```
MSE = mean((pred - gt)^2)
PSNR = 10 * log10(MAX^2 / MSE)     [dB]

Where:
- MAX = 1.0 (normalized to [0,1])
- Higher = Better
- Typical range: 20-40 dB
- Part1 baseline: 25-28 dB
```

### SSIM (Structural Similarity Index)
```
SSIM(x, y) = (2μx μy + C1)(2σxy + C2) / ((μx^2 + μy^2 + C1)(σx^2 + σy^2 + C2))

Where:
- μx, μy = luminance
- σx, σy = contrast
- σxy = covariance
- Range: [-1, 1] (typically [0, 1])
- Higher = Better
- Part1 baseline: 0.68-0.78
```

### FPS (Frames Per Second)
```
FPS = num_samples / total_time_seconds

Where:
- Bicubic: 500+ FPS (very fast)
- SRCNN: 50-100 FPS (moderate)
- Useful for real-time applications?
```

---

## 🔄 Workflow States

```
┌─────────────────────────┐
│  Initial State (START)  │
│                         │
│ - No checkpoints        │
│ - No results            │
│ - No report             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  After Training         │
│                         │
│ ✓ srcnn_best.pt         │
│ - No results            │
│ - No report             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  After Spatial Eval     │
│                         │
│ ✓ srcnn_best.pt         │
│ ✓ results_part1.csv     │
│ - No report             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  After Temporal Eval    │
│                         │
│ ✓ srcnn_best.pt         │
│ ✓ results_part1.csv     │
│ ✓ temporal_metrics.csv  │
│ - No report             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  After Report Gen       │
│                         │
│ ✓ srcnn_best.pt         │
│ ✓ results_part1.csv     │
│ ✓ temporal_metrics.csv  │
│ ✓ report_summary.md     │
│                         │
│ FINAL STATE (READY)     │
└─────────────────────────┘
```

---

## 💾 Critical Files Checklist

**Must Have**:
- ✅ `dataset.py` — Without this, training/eval cannot load data
- ✅ `train_srcnn.py` — Required for training or for SRCNN class import

**Should Have**:
- ✅ `checkpoints/srcnn_best.pt` — Can skip training with pre-trained
- ✅ `eval.py` — For spatial evaluation
- ✅ `temporal_baseline.py` — For temporal evaluation

**Nice to Have**:
- ☑️ `infer_srcnn.py` — For visualization only
- ☑️ `report_summary.py` — For formatting only
- ☑️ `run_part1_report.sh` — Convenience script

---

## 🚀 Start Here

1. **Setup**: Install dependencies, download Vimeo90K
2. **Train** (optional): `python train_srcnn.py --epochs 200 ...`
3. **Evaluate**: 
   - `python eval.py ...`
   - `python temporal_baseline.py ...`
4. **Report**: `python report_summary.py ...`
5. **Review**: Open `report_summary.md`

**See QUICKSTART.md for command examples.**
