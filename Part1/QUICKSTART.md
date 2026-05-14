# Part 1: Quick Start Guide

## ⚡ 5-Minute Quick Start

### Prerequisites
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install opencv-python scikit-image numpy
```

### Data Setup
```bash
# Ensure Vimeo90K is in correct structure:
/path/to/project1/
├── train/
│   ├── train_sharp/
│   └── train_sharp_bicubic/X4/
├── val/
│   ├── val_sharp/
│   └── val_sharp_bicubic/X4/
└── vimeo_super_resolution_test/
```

### Run Everything (30 seconds to 1 hour depending on data size)

```bash
# Navigate to Part1
cd Part1

# Quick test (2 minutes)
bash run_part1_report.sh --quick

# Full evaluation (depends on dataset size)
bash run_part1_report.sh
```

### Check Results
```bash
# View summary
cat report_summary.md

# View detailed metrics
cat results_part1.csv
cat temporal_metrics.csv
```

---

## 🎯 Common Use Cases

### "I just want to evaluate baseline methods without training"

```bash
# If you already have checkpoints/srcnn_best.pt:
python eval.py \
  --project1-root /path/to/project1 \
  --split val \
  --srcnn-ckpt checkpoints/srcnn_best.pt \
  --csv-out results.csv

# View results
cat results.csv
```

**Time**: 5-10 minutes (depending on dataset size)

---

### "I want to train SRCNN from scratch"

```bash
# Quick test on small subset (2 minutes)
python train_srcnn.py \
  --project1-root /path/to/project1 \
  --epochs 1 \
  --batch-size 8 \
  --max-train-pairs 100 \
  --max-val-pairs 50 \
  --save-dir checkpoints

# Full training (2-10 hours)
python train_srcnn.py \
  --project1-root /path/to/project1 \
  --epochs 200 \
  --batch-size 32 \
  --num-workers 4 \
  --save-dir checkpoints
```

**Outputs**: 
- `checkpoints/srcnn_best.pt` (pre-trained SRCNN)
- Training can continue from checkpoint

---

### "I want to evaluate temporal methods"

```bash
# Weighted averaging with unsharp masking
python temporal_baseline.py \
  --test-root /path/to/vimeo_super_resolution_test \
  --weights 0.25 0.5 0.25 \
  --sigma 1.0 \
  --amount 0.6 \
  --csv-out temporal_metrics.csv

# View results
cat temporal_metrics.csv
```

**Time**: 5-10 minutes

---

### "I want to visualize SRCNN predictions"

```bash
# Generate preview images
python infer_srcnn.py \
  --project1-root /path/to/project1 \
  --split val \
  --srcnn-ckpt checkpoints/srcnn_best.pt \
  --max-pairs 8 \
  --out-dir outputs_preview

# Images saved in outputs_preview/
```

**Outputs**:
- `outputs_preview/0_lr.png`
- `outputs_preview/0_sr.png`
- `outputs_preview/0_gt.png`
- `outputs_preview/0_compare.png`

---

## ❓ FAQ

### Q: I'm getting "FileNotFoundError: Path not found"
**A**: Check your dataset path:
```bash
# Verify train data exists
ls /path/to/project1/train/train_sharp/
ls /path/to/project1/train/train_sharp_bicubic/X4/

# Update path in command if needed
python eval.py --project1-root /correct/path/to/project1 ...
```

---

### Q: Training is very slow (1 sample/sec or slower)
**A**: CPU bottleneck. Increase `num-workers`:
```bash
# Default: num-workers=4
# Try: num-workers=8 or 12 (2-4x number of CPU cores)
python train_srcnn.py --num-workers 12 ...
```

---

### Q: Out of memory (OOM) during training
**A**: Reduce batch size:
```bash
# Default: batch-size=32
# Try: batch-size=8 or 16
python train_srcnn.py --batch-size 8 ...

# Also reduce patches per image
--max-train-patches 2000 --max-val-patches 400
```

---

### Q: My GPU is not being used (training on CPU instead)
**A**: Verify PyTorch installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

### Q: Can I continue training from a checkpoint?
**A**: Yes! Training script automatically resumes:
```bash
# If checkpoints/srcnn_best.pt exists and --epochs > current_epoch,
# training continues from that checkpoint
python train_srcnn.py \
  --project1-root /path/to/project1 \
  --epochs 500 \
  --save-dir checkpoints
```

---

### Q: How do I use my own video dataset?
**A**: Modify `dataset.py`:
```python
# Change this function:
def default_split_paths(project1_root: str | Path):
    # Update paths to match your data structure
    return {
        "train": SplitPaths(
            hr_root=root / "my_data" / "hr",
            lr_bicubic_root=root / "my_data" / "lr",
        ),
        ...
    }
```

---

### Q: What's the difference between Bicubic and Lanczos?
**A**:
| Method | Speed | Quality | Edge Preservation |
|--------|-------|---------|-------------------|
| Bicubic | Very fast (500+ FPS) | Good | Slight blur |
| Lanczos | Fast (300+ FPS) | Better | Better edges |
| SRCNN | Slow (50-100 FPS) | Best | Sharp details |

Use Bicubic for speed, SRCNN for best quality.

---

### Q: Why does SRCNN perform slightly worse on some images?
**A**: Reasons:
1. **Trained on patches**: Loses global context
2. **Fixed input size**: Only sees 33×33 receptive field
3. **No temporal info**: Single frame inference
4. **Training data**: Vimeo90K may not match your test distribution

**Solution**: Use Part2 (BasicVSR) for better results.

---

### Q: Can I combine spatial (SRCNN) + temporal (averaging)?
**A**: Not directly in current code, but you can manually:
```python
# 1. Apply SRCNN to each frame
sr_frames = [apply_srcnn(frame) for frame in lr_frames]

# 2. Apply temporal averaging
output = temporal_weighted_average(sr_frames, center_idx, weights)

# 3. Apply unsharp masking
final = unsharp_mask(output)
```

---

### Q: How to evaluate on only specific sequences?
**A**: Modify `max-pairs` parameter:
```bash
# Evaluate on only 5 image pairs
python eval.py --max-pairs 5 ...

# Or modify dataset.py to filter by sequence name
```

---

### Q: What batch size should I use?
**A**: Rule of thumb:
```
GPU VRAM        Recommended batch-size
4 GB            8-16
8 GB            16-32
16 GB           32-64
24 GB+          64-128
```

If OOM, halve the batch size.

---

### Q: Can I visualize training curves?
**A**: Training logs are printed to console. For plotting:
```bash
# Redirect to file
python train_srcnn.py ... > train.log

# Parse and plot (use your favorite tool)
```

Better: Check training progress by monitoring csv:
```bash
watch -n 5 'tail results_part1.csv'
```

---

### Q: How to evaluate on test set?
**A**: 
```bash
python eval.py \
  --project1-root /path/to/project1 \
  --split test \
  --srcnn-ckpt checkpoints/srcnn_best.pt \
  --csv-out results_test.csv
```

If no "test" split, modify `dataset.py`:
```python
return {
    "train": ...,
    "val": ...,
    "test": SplitPaths(
        hr_root=root / "test_sharp",
        lr_bicubic_root=root / "test_sharp_bicubic/X4",
    ),
}
```

---

## 🔧 Command Reference

### Training
```bash
python train_srcnn.py \
  --project1-root PATH \
  --epochs 200 \
  --batch-size 32 \
  --lr 1e-4 \
  --patch-size 33 \
  --label-size 21 \
  --stride 14 \
  --num-workers 4 \
  --save-dir checkpoints
```

### Evaluation (Spatial)
```bash
python eval.py \
  --project1-root PATH \
  --split val \
  --srcnn-ckpt checkpoints/srcnn_best.pt \
  --csv-out results.csv
```

### Evaluation (Temporal)
```bash
python temporal_baseline.py \
  --test-root PATH/vimeo_super_resolution_test \
  --weights 0.25 0.5 0.25 \
  --sigma 1.0 \
  --amount 0.6 \
  --csv-out temporal_metrics.csv
```

### Inference (Visual Preview)
```bash
python infer_srcnn.py \
  --project1-root PATH \
  --split val \
  --srcnn-ckpt checkpoints/srcnn_best.pt \
  --max-pairs 8 \
  --out-dir outputs_preview
```

### Generate Report
```bash
python report_summary.py \
  --spatial-csv results_part1.csv \
  --temporal-csv temporal_metrics.csv \
  --out report_summary.md
```

---

## 📊 Expected Runtimes

| Operation | Time | Notes |
|-----------|------|-------|
| Train SRCNN (quick) | 2 min | epoch=1, pairs=200 |
| Train SRCNN (full) | 2-10 hrs | epochs=200, all data |
| Evaluate spatial | 5-20 min | Depends on dataset size |
| Evaluate temporal | 5-20 min | 3-frame weighted avg |
| Generate preview | 2-5 min | 8 image pairs |
| Generate report | <1 min | CSV aggregation |

---

## 💡 Pro Tips

1. **Test on small subset first**
   ```bash
   --max-train-pairs 100 --max-val-pairs 50
   ```

2. **Monitor GPU usage**
   ```bash
   # On Linux:
   watch -n 1 nvidia-smi
   ```

3. **Run in background**
   ```bash
   nohup python train_srcnn.py ... > train.log 2>&1 &
   ```

4. **Compare methods easily**
   ```bash
   # Run eval on same split for fair comparison
   python eval.py --split val --csv-out results.csv
   ```

---

**Next Steps**: 
- After Part1, proceed to **Part2_SOTA** for state-of-the-art video SR
- See Part1/README.md for full documentation
