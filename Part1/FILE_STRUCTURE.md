# Part 1: File Structure & Usage Matrix

## 📁 Complete File Breakdown

### Core Training Files

#### 📄 `dataset.py`
**Purpose**: Dataset loader for Vimeo90K paired images (LR ↔ HR)

**Key Functions**:
- `default_split_paths()`: Returns train/val split paths
- `build_frame_pairs()`: Builds aligned (LR, HR) tuples
- `SRCNNPatchDataset`: Extracts overlapping patches from images
- `load_y_channel()`, `load_bgr()`: Image I/O with Y-channel conversion

**Key Classes**:
```
SplitPaths
├─ hr_root: Path to HR images
└─ lr_bicubic_root: Path to LR upsampled images

SRCNNPatchDataset
├─ patch_size: Input patch size (default=33)
├─ label_size: Output patch size (default=21)
├─ stride: Extraction stride (default=14)
└─ max_patches: Max patches to extract (default=None)
```

**Used By**:
- `train_srcnn.py` → Creates DataLoader from SRCNNPatchDataset
- `eval.py` → Loads full images for comparison

**Dependencies**:
- `pathlib.Path`
- `cv2` (OpenCV)
- `numpy`
- `torch.utils.data.Dataset`

---

#### 📄 `train_srcnn.py`
**Purpose**: Train SRCNN model from scratch

**Key Sections**:
```
1. SRCNN class definition (3-layer CNN)
2. Argument parser for hyperparameters
3. Data loading (train + validation splits)
4. Training loop with validation
5. Checkpoint saving (best + latest)
```

**SRCNN Architecture**:
```
Input (1×33×33)
  ↓
Conv2D(1→64, 9×9, pad=0) + ReLU  [features extraction]
  ↓
Conv2D(64→32, 1×1, pad=0) + ReLU [non-linear mapping]
  ↓
Conv2D(32→1, 5×5, pad=0)         [reconstruction]
  ↓
Output (1×21×21)

Parameters: ~55,000
```

**Hyperparameters**:
| Parameter | Default | Effect |
|-----------|---------|--------|
| `patch-size` | 33 | Input patch size |
| `label-size` | 21 | Output patch size |
| `stride` | 14 | Patch extraction stride |
| `batch-size` | 32 | Batch size |
| `lr` | 1e-4 | Learning rate (Adam) |
| `epochs` | 10 | Training epochs |

**Output**: `checkpoints/srcnn_best.pt`

**Used By**:
- `eval.py` → Loads trained model for inference
- `infer_srcnn.py` → Loads for visual preview

**Dependencies**:
- `torch`, `torch.nn`, `torch.optim`
- `dataset.py` → SRCNNPatchDataset
- `pathlib.Path`

---

### Evaluation Files

#### 📄 `eval.py`
**Purpose**: Evaluate and compare spatial upsampling methods

**Methods Evaluated**:
1. **Bicubic**: Classical interpolation (OpenCV)
2. **Lanczos**: Higher-order interpolation (OpenCV)
3. **SRCNN**: Trained deep learning model

**Metrics Computed** (on Y channel):
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index
- **FPS**: Frames per second (throughput)

**Key Functions**:
- `psnr_y()`: Compute PSNR on Y channel
- `ssim_y()`: Compute SSIM on Y channel
- `load_srcnn()`: Load trained model from checkpoint
- `degrade_then_upsample()`: Apply interpolation methods

**Output**: `results_part1.csv`
```
method,psnr_y,ssim_y,fps
bicubic,25.234,0.680,520.5
lanczos,25.892,0.702,310.2
srcnn,28.512,0.781,75.3
```

**Used By**:
- `report_summary.py` → Reads for summary generation
- Standalone evaluation

**Dependencies**:
- `eval.py` itself (metric functions)
- `train_srcnn.py` → SRCNN class
- `dataset.py` → File loading utilities
- `torch`, `numpy`, `cv2`, `skimage.metrics`

---

#### 📄 `temporal_baseline.py`
**Purpose**: Evaluate temporal fusion methods

**Methods**:
1. **Temporal Weighted Average**: Multi-frame averaging
   ```
   output[i] = Σ(w[j] × frame[i+j-center])
   Default weights: [0.25, 0.5, 0.25]
   ```

2. **+ Unsharp Masking**: Edge enhancement
   ```
   sharp = img + amount × (img - blur(img))
   ```

**Key Functions**:
- `psnr_y()`, `ssim_y()`: Metric computation
- `temporal_weighted_average()`: Apply weighted averaging
- `unsharp_mask()`: Apply sharpening filter
- `bgr_to_y_float01()`: RGB → Y channel conversion

**Output**: `temporal_metrics.csv`
```
sequence,avg_psnr,avg_ssim,unsharp_psnr,unsharp_ssim
seq_001,29.234,0.805,29.512,0.812
seq_002,28.891,0.798,29.145,0.805
...
MEAN,29.156,0.801,29.423,0.808
```

**Used By**:
- `report_summary.py` → Reads for summary generation
- Standalone evaluation

**Dependencies**:
- `numpy`, `cv2`, `skimage.metrics`
- Metric functions (same as eval.py)

---

#### 📄 `infer_srcnn.py`
**Purpose**: Generate visual preview of SRCNN predictions

**Outputs** (per image pair):
- `{idx}_lr.png`: Input (upsampled LR)
- `{idx}_sr.png`: SRCNN output
- `{idx}_gt.png`: Ground truth (HR)
- `{idx}_compare.png`: Side-by-side comparison

**Key Functions**:
- `y_to_bgr()`: Y channel → BGR conversion
- `center_crop()`: Crop to match output size
- Inference loop with batch processing

**Output**: `outputs_srcnn_preview/` directory

**Used By**:
- Manual visualization
- Quality assessment
- Paper figures

**Dependencies**:
- `train_srcnn.py` → SRCNN class
- `dataset.py` → load_y_channel(), load_bgr()
- `torch`, `cv2`, `numpy`

---

### Reporting

#### 📄 `report_summary.py`
**Purpose**: Generate markdown summary from evaluation CSVs

**Inputs**:
- `results_part1.csv` (spatial methods)
- `temporal_metrics.csv` (temporal methods)

**Output**: `report_summary.md`
```markdown
# Part1 Result Summary

## Spatial Baseline
| Method | PSNR(Y) | SSIM(Y) | FPS |
|---|---:|---:|---:|
| bicubic | 25.2341 | 0.6803 | 520.45 |
| srcnn | 28.5123 | 0.7812 | 75.30 |

## Temporal Baseline (Mean)
- temporal_avg: PSNR=29.1564, SSIM=0.8014
- temporal_avg_unsharp: PSNR=29.4232, SSIM=0.8078

## Summary
SRCNN outperforms Bicubic by 3.28 dB (PSNR)
Temporal fusion adds ~0.9 dB improvement
```

**Used By**:
- Final presentation
- Paper comparison table

**Dependencies**:
- `csv` module
- `pathlib.Path`

---

#### 📄 `run_part1_report.sh`
**Purpose**: Orchestrate full pipeline with one command

**Steps**:
1. Train SRCNN (if not exists)
2. Evaluate spatial baselines → results_part1.csv
3. Evaluate temporal baseline → temporal_metrics.csv
4. Generate summary → report_summary.md
5. Generate visual preview → outputs_srcnn_report/

**Output Files**:
```
checkpoints/srcnn_best.pt          [trained model]
results_part1.csv                  [spatial metrics]
temporal_metrics.csv               [temporal metrics]
report_summary.md                  [formatted report]
outputs_srcnn_report/              [preview images]
```

**Used By**:
- Full pipeline execution
- Batch processing

**Dependencies**:
- All other Part1 files

---

## 🔗 Dependency Graph

```
dataset.py
    ├── Used by: train_srcnn.py, eval.py
    └── Uses: cv2, numpy, torch.utils.data

train_srcnn.py
    ├── Creates: checkpoints/srcnn_best.pt
    └── Uses: dataset.py, torch, argparse

eval.py
    ├── Reads: checkpoints/srcnn_best.pt
    ├── Reads: dataset paths
    ├── Creates: results_part1.csv
    └── Uses: train_srcnn.py, dataset.py, torch, cv2, skimage

temporal_baseline.py
    ├── Creates: temporal_metrics.csv
    └── Uses: cv2, numpy, skimage

infer_srcnn.py
    ├── Reads: checkpoints/srcnn_best.pt
    ├── Reads: dataset paths
    ├── Creates: outputs_srcnn_preview/
    └── Uses: train_srcnn.py, dataset.py, torch, cv2

report_summary.py
    ├── Reads: results_part1.csv
    ├── Reads: temporal_metrics.csv
    ├── Creates: report_summary.md
    └── Uses: csv, pathlib

run_part1_report.sh
    └── Calls: all of the above in sequence
```

---

## 📋 Stage-by-Stage Usage

### Stage 1: SRCNN Training
```
dataset.py → train_srcnn.py → checkpoints/srcnn_best.pt
```

**Files Required**:
- `dataset.py` ✓
- `train_srcnn.py` ✓

**Files Generated**:
- `checkpoints/srcnn_best.pt`

---

### Stage 2: Spatial Evaluation
```
eval.py reads checkpoints/srcnn_best.pt + dataset → results_part1.csv
```

**Files Required**:
- `eval.py` ✓
- `train_srcnn.py` ✓ (for SRCNN class)
- `dataset.py` ✓ (for utilities)
- `checkpoints/srcnn_best.pt` ✓ (from Stage 1)

**Files Generated**:
- `results_part1.csv`

---

### Stage 3: Temporal Evaluation
```
temporal_baseline.py reads dataset → temporal_metrics.csv
```

**Files Required**:
- `temporal_baseline.py` ✓
- Dataset (vimeo_super_resolution_test)

**Files Generated**:
- `temporal_metrics.csv`

---

### Stage 4: Visual Preview
```
infer_srcnn.py reads checkpoints/srcnn_best.pt + dataset → outputs_srcnn_preview/
```

**Files Required**:
- `infer_srcnn.py` ✓
- `train_srcnn.py` ✓ (for SRCNN class)
- `dataset.py` ✓ (for utilities)
- `checkpoints/srcnn_best.pt` ✓ (from Stage 1)

**Files Generated**:
- `outputs_srcnn_preview/{idx}_lr.png`
- `outputs_srcnn_preview/{idx}_sr.png`
- `outputs_srcnn_preview/{idx}_gt.png`
- `outputs_srcnn_preview/{idx}_compare.png`

---

### Stage 5: Report Generation
```
report_summary.py reads CSVs → report_summary.md
```

**Files Required**:
- `report_summary.py` ✓
- `results_part1.csv` ✓ (from Stage 2)
- `temporal_metrics.csv` ✓ (from Stage 3)

**Files Generated**:
- `report_summary.md`

---

## 🔄 Execution Workflow

```
START
  ↓
[Optional] Run train_srcnn.py
  ↓ checkpoints/srcnn_best.pt
eval.py ──→ results_part1.csv
  ↓
temporal_baseline.py ──→ temporal_metrics.csv
  ↓
[Optional] infer_srcnn.py ──→ outputs_srcnn_preview/
  ↓
report_summary.py ──→ report_summary.md
  ↓
END
```

---

## 📊 File Statistics

| File | Lines | Purpose | Type |
|------|-------|---------|------|
| `dataset.py` | ~150 | Data loading | Core |
| `train_srcnn.py` | ~200 | Model training | Core |
| `eval.py` | ~200 | Method evaluation | Evaluation |
| `temporal_baseline.py` | ~150 | Temporal fusion | Evaluation |
| `infer_srcnn.py` | ~120 | Visual preview | Utility |
| `report_summary.py` | ~100 | Report generation | Utility |
| `run_part1_report.sh` | ~50 | Pipeline orchestration | Script |

**Total**: ~1,000 lines of code

---

## ✅ Checklist for Running Part1

- [ ] Environment set up (PyTorch, OpenCV, etc.)
- [ ] Vimeo90K dataset downloaded and organized
- [ ] `dataset.py` paths updated for your system
- [ ] `train_srcnn.py` run (or checkpoint provided)
- [ ] `eval.py` run and `results_part1.csv` generated
- [ ] `temporal_baseline.py` run and metrics generated
- [ ] `report_summary.py` run for final report
- [ ] Results reviewed in `report_summary.md`

---

## 🎯 Next Steps

After completing Part1:
1. ✅ Understand classical vs. deep learning SR trade-offs
2. ✅ Establish baseline performance metrics
3. ✅ See benefits of temporal fusion
4. 👉 **Proceed to Part2_SOTA** for state-of-the-art methods (BasicVSR + GAN)
