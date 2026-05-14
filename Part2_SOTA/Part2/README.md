# Part 2: Video Super-Resolution using BasicVSR + GAN (SOTA)

## Overview

This Part 2 implements a two-stage video super-resolution (VSR) framework based on **BasicVSR** architecture combined with **GAN** training. The approach achieves significant quality improvements through:

- **Stage 1**: BasicVSR training with pixel-level loss (L1/Charbonnier)
- **Stage 2**: Fine-tuning with adversarial training (GAN) + perceptual loss

**Key Results:**
- Stage 1 PSNR: 28-32 dB (clean, sharp edges)
- Stage 2 PSNR: 27-30 dB, SSIM: 0.75-0.85 (realistic textures, perceptually pleasing)

---

## Architecture Overview

### BasicVSR Network

```
Input: 7 LR Frames [B, 7, 3, H, W]
         ↓
    ┌────────────────────────────────────┐
    │  Backward Propagation (T-1 → 0)    │ ← Optical Flow Estimation (SpyNet)
    │  - Align features using flow       │ ← Feature Warping
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │  Forward Propagation (0 → T-1)     │ ← Refine previous results
    │  - Cascade propagation              │
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │  Feature Fusion & Upsampling       │ ← 4x PixelShuffle
    └────────────────────────────────────┘
         ↓
Output: 7 SR Frames [B, 7, 3, 4H, 4W]
```

### Two-Stage Training Strategy

| Stage | Goal | Loss Function | Output PSNR | Characteristics |
|-------|------|---------------|-------------|-----------------|
| **Stage 1** | Learn pixel-level reconstruction | L1 / Charbonnier | 28-32 dB | Sharp, clean, but may lack details |
| **Stage 2** | Add perceptual realism | Charbonnier + Perceptual + GAN | 27-30 dB | Realistic textures, visually pleasing |

---

## File Structure

### Core Model Files

| File | Purpose | Key Class/Function |
|------|---------|-------------------|
| `model_basicvsr.py` | Main VSR network | `BasicVSR` - bidirectional propagation |
| `model_spynet.py` | Optical flow estimation | `SpyNet` - 6-layer pyramid network |
| `model_discriminator.py` | GAN discriminator | `UNetDiscriminatorSN` - UNet with spectral norm |
| `loss_gan.py` | Loss functions | `PerceptualLoss`, `CharbonnierLoss`, `GANLoss` |

### Data & Training

| File | Purpose | Usage |
|------|---------|-------|
| `vsr_dataset.py` | Vimeo90K dataset loader | Loads 7-frame sequences with HR/LR pairs |
| `train_vsr_stage1.py` | Stage 1 training script | `python train_vsr_stage1.py` |
| `train_vsr_gan.py` | Stage 2 GAN training | `python train_vsr_gan.py` (requires Stage 1 checkpoint) |

### Inference & Evaluation

| File | Purpose | Usage |
|------|---------|-------|
| `inference_vsr.py` | Run SR on any video length | `python inference_vsr.py` |
| `eval_vsr.py` | Evaluate on validation set | `python eval_vsr.py` |
| `check.py` | Debug forward passes | Test model shapes and losses |

### Utilities

| File | Purpose | Usage |
|------|---------|-------|
| `frames_to_video.py` | Convert frame sequences to MP4 | `python frames_to_video.py` |
| `visual_compare.py` | Create side-by-side comparison videos | `python visual_compare.py` |
| `basicvsr_net.py` | Alternative implementation (optional) | Reference implementation |

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- torch, torchvision
- opencv-python (cv2)
- tensorboard (logging)
- tqdm (progress bars)
- Pillow

### 1. Prepare Data

Download Vimeo90K dataset and organize as:

```
dataset_root/
├── train/
│   ├── train_sharp/          # HR frames (720p)
│   │   ├── 000/
│   │   │   ├── 00000000.png
│   │   │   ├── 00000001.png
│   │   │   └── ... (7 frames per sequence)
│   │   ├── 001/
│   │   └── ... (more sequences)
│   └── train_sharp_bicubic/
│       └── X4/               # LR frames (180p)
│           ├── 000/
│           ├── 001/
│           └── ...
└── val/
    ├── val_sharp/
    └── val_sharp_bicubic/X4/
```

### 2. Stage 1: Basic Training

Train BasicVSR with pixel-level loss:

```bash
python train_vsr_stage1.py
```

**Configuration:**
- Batch size: 1 (adjust based on GPU memory)
- Learning rate: 2e-4 with StepLR scheduler
- Epochs: 50
- Gradient clipping: 0.01 (critical for RNN stability)

**Output:**
- Checkpoint: `weights/basicvsr_stage1.pth`
- Logs: `logs/stage1/` (TensorBoard)

**Expected Results:**
- PSNR converges to ~28-32 dB
- Loss decreases smoothly

### 3. Stage 2: GAN Fine-tuning

Fine-tune with adversarial and perceptual loss:

```bash
python train_vsr_gan.py
```

**Important:** Stage 2 requires Stage 1 checkpoint. The script automatically loads `weights/basicvsr_stage1.pth`.

**Configuration:**
- Load checkpoint from Stage 1
- Generator LR: 1e-4 (main modules), 2.5e-5 (SpyNet)
- Discriminator LR: 1e-4
- Cosine annealing scheduler
- GAN loss weight: 0.1 (balanced with pixel loss)

**Output:**
- Checkpoint: `weights/basicvsr_gan.pth`
- Periodic saves: `weights/basicvsr_gan_optimized_epoch_{N}.pth`

**Expected Results:**
- PSNR: 27-30 dB (slight drop due to GAN trade-off)
- SSIM: 0.75-0.85
- Visually pleasing textures

---

## Training Details

### Stage 1 Pipeline

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        lrs = batch['lr']        # [B, 7, 3, H, W]
        gts = batch['hr']        # [B, 7, 3, 4H, 4W]
        
        # Forward
        srs = model(lrs)         # [B, 7, 3, 4H, 4W]
        
        # Loss
        loss = L1Loss(srs, gts)
        
        # Backward with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
```

**Why Gradient Clipping?**
- BasicVSR has recurrent connections (RNN-like)
- Without clipping, gradients explode over 7 frames
- Norm threshold 0.01 empirically prevents instability

### Stage 2 Pipeline

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Alternating optimization
        
        # --- Optimize Generator ---
        loss_g_pix = PixelLoss(sr, gt)
        loss_g_percep = PerceptualLoss(sr_frame, gt_frame)
        loss_g_gan = GANLoss(discriminator(sr_frame), real=True)
        
        loss_g = 1.0*loss_g_pix + 1.0*loss_g_percep + 0.1*loss_g_gan
        loss_g.backward()
        
        # --- Optimize Discriminator ---
        loss_d_real = GANLoss(discriminator(gt_frame), real=True)
        loss_d_fake = GANLoss(discriminator(sr_frame.detach()), real=False)
        
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
```

**Loss Weights Explanation:**
- Pixel loss (1.0): Preserve overall structure
- Perceptual loss (1.0): Match intermediate features (VGG19 layer 35)
- GAN loss (0.1): Add realistic textures without over-emphasis

---

## Key Components

### SpyNet: Optical Flow Estimation

```
Input: Two consecutive frames [B, 3, H, W]
         ↓
[Level 6] Small resolution (H/64, W/64)  ← Start
  ↓ upsample & refine
[Level 5] H/32, W/32
  ↓ upsample & refine
...
[Level 1] H/2, W/2
  ↓ upsample & refine
[Level 0] Original resolution (H, W)  ← Final flow
```

**Why Spatial Pyramid?**
- Coarse-to-fine: Large motions captured at low res
- Refinement: Local details added at high res
- Efficient: Fewer parameters than dense optical flow

### Feature Warping

```python
# Grid sampling with bilinear interpolation
flow = spynet(ref, supp)  # [B, 2, H, W]
warped = grid_sample(supp, flow)

# Alignment: Warp supp features to match ref's viewpoint
# Result: Temporally aligned feature maps for fusion
```

### Bidirectional Propagation

```
Frame 0 ← 1 ← 2 ← 3 ← 4 ← 5 ← 6  (Backward)
  ↓ fusion
Frame 0 → 1 → 2 → 3 → 4 → 5 → 6  (Forward)

Backward: Reference frame is frame 0, propagate backward
Forward:  Refine features using forward context
Result:   Bidirectional awareness of temporal structure
```

### Perceptual Loss (Stage 2)

```
VGG19 Pre-trained (frozen)
  ↓
Extract Layer 35 features (mid-high level)
  ↓
L1 distance between sr_frame and gt_frame features
  ↓
Loss = mean(|vgg(sr) - vgg(gt)|)

Why Layer 35?
- Low layer (early): Pixel-level details (already covered by L1)
- Mid layer (35): Texture, edges, objects
- High layer (late): Semantic meaning (too abstract)
```

### Discriminator (Stage 2)

```
UNet Architecture with Spectral Normalization
  ↓
Input: Single SR/GT frame [B, 3, 256, 256]
  ↓
Encoder: 5 levels downsampling
  ↓
Bottleneck: Deepest features
  ↓
Decoder: 5 levels upsampling with skip connections
  ↓
Output: Patch-based classification [B, 1, 16, 16]

Why Spectral Norm?
- Stabilizes GAN training
- Prevents discriminator from becoming too powerful
- Ensures smooth gradient flow to generator
```

---

## Troubleshooting

### OOM (Out of Memory)

**Solution 1: Reduce batch size**
```python
# train_vsr_stage1.py, line 37
batch_size = 1  # Already minimal
```

**Solution 2: Reduce sequence length**
```python
# vsr_dataset.py, line 24
seq_length = 5  # Instead of 7 (fewer frames per batch)
```

**Solution 3: Use gradient accumulation**
```python
# Add to training loop
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### NaN Loss (Gradient Instability)

**Likely Cause:** Optical flow becomes invalid or gradients explode

**Solution:**
```python
# Check gradient clipping (Stage 1)
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

# Use Charbonnier loss instead of L1
# More robust to outliers in recurrent networks
```

### Checkpoint Loading Errors

**Error:** `KeyError: 'model_state_dict'`

**Solution:**
```python
# Check checkpoint format
checkpoint = torch.load('weights/basicvsr_stage1.pth')

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
```

### Inference on Long Sequences

**Input:** Video with any number of frames (not limited to 7)

```python
# inference_vsr.py automatically handles arbitrary lengths
python inference_vsr.py
# Output: vsrgan_result/000/ (SR frames)
```

---

## Inference

### Single Video Sequence

```bash
python inference_vsr.py
```

**Configuration (edit inference_vsr.py):**
```python
target_folder = "000"  # Input sequence folder
lr_root = "/path/to/val/val_sharp_bicubic/X4"
checkpoint_path = "weights/basicvsr_gan.pth"  # Or Stage 1 checkpoint
output_base_dir = "./vsrgan_result"
```

**Output:**
```
vsrgan_result/
└── 000/
    ├── 00000000.png  (SR frame)
    ├── 00000001.png
    └── ... (all SR frames)
```

### Create Comparison Videos

**Combine LR, SR, GT into one video:**

```bash
python frames_to_video.py
```

**Output:**
```
visual/000/
├── 000_LR.mp4  (Input - Bicubic upsampled)
├── 000_SR.mp4  (Result - Our model)
└── 000_GT.mp4  (Reference - Ground truth)
```

**Create side-by-side comparison:**

```bash
python visual_compare.py
```

**Output:**
```
visual/triple_comparison.mp4
```

---

## Evaluation

### Quantitative Metrics

```bash
python eval_vsr.py \
    --val_data_root /path/to/val \
    --checkpoint_path weights/basicvsr_gan.pth \
    --save_dir evalresult/
```

**Output:** CSV file with per-frame PSNR/SSIM

```
sequence_id, psnr_mean, ssim_mean, ...
000, 29.34, 0.78, ...
001, 30.12, 0.80, ...
```

### Visual Inspection

1. Open `vsrgan_result/000/` frames in image viewer
2. Compare with ground truth in `val/val_sharp/000/`
3. Look for:
   - Sharp edges and details
   - Realistic texture
   - Artifact-free regions

### Expected Performance

| Metric | Stage 1 | Stage 2 | GT |
|--------|---------|---------|-----|
| **PSNR (dB)** | 28-32 | 27-30 | - |
| **SSIM** | 0.80-0.85 | 0.75-0.85 | 1.0 |
| **Visual Quality** | Sharp, clean | Realistic textures | - |

---

## Advanced Configuration

### Learning Rate Schedules

**Stage 1: StepLR**
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# Halve LR every 20 epochs
```

**Stage 2: CosineAnnealingLR**
```python
scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_g, T_max=100, eta_min=1e-7
)
# Smoothly decay to near-zero over 100 epochs
```

### Hyperparameter Tuning

**Pixel Loss Weight (Stage 2):**
- Default: 1.0
- Increase (2.0): More pixel-accurate, but less realistic
- Decrease (0.5): More texture, but possible artifacts

**Perceptual Loss Weight (Stage 2):**
- Default: 1.0
- Increase (2.0): Stronger feature matching
- Decrease (0.5): Allow more deviation

**GAN Loss Weight (Stage 2):**
- Default: 0.1
- Increase (0.5): More aggressive adversarial training
- Decrease (0.01): Conservative, safer training

---

## Dataset: Vimeo90K

**Description:**
- 90,000 video sequences
- 7 frames per sequence
- Multiple resolutions (480p-720p)
- Diverse content (animation, live action, etc.)

**Format (Flattened):**
```
train_sharp/
├── 000/
│   ├── 00000000.png (Frame 0)
│   ├── 00000001.png (Frame 1)
│   └── ...
│   └── 00000006.png (Frame 6)
├── 001/
└── ... (10000 sequences)

train_sharp_bicubic/X4/
├── 000/
│   ├── 00000000.png (LR version)
│   └── ...
└── 001/
```

**Why 7 Frames?**
- Captures temporal context without excessive memory
- Vimeo90K standard (septuplet)
- Balance between temporal awareness and computational cost

---

## References

**BasicVSR Paper:**
- Channels-Last Network: A method for fast inference with low memory
- Recurrent Architecture: Temporal propagation for video quality

**GAN Training Tips:**
- Spectral Normalization: Stabilizes discriminator
- Alternating Optimization: Fair competition between G and D
- Lower GAN weight: Prevents mode collapse

**Vimeo90K Dataset:**
- Original publication: "Video Enhancement with Task-Oriented Flow"
- 90,000 sequences, 7 frames each

---

## File Sizes & Training Time

### Checkpoint Sizes
- Stage 1: ~50 MB (BasicVSR weights)
- Stage 2: ~50 MB (BasicVSR) + ~5 MB (Discriminator)
- VGG19 (Perceptual Loss): ~550 MB

### Training Duration
| Stage | GPU (V100) | GPU (A100) | Iterations |
|-------|-----------|-----------|-----------|
| **Stage 1** | ~12-15 hrs | ~6-8 hrs | 50 epochs × 1000 samples |
| **Stage 2** | ~8-10 hrs | ~4-5 hrs | 100 epochs × 1000 samples |

### Inference Time
| Task | Time (per video) |
|------|------------------|
| 7-frame sequence | ~2-3 seconds (V100) |
| Single frame | ~0.3-0.5 seconds |

---

## Next Steps

1. **Start Training:**
   ```bash
   python train_vsr_stage1.py
   ```

2. **Monitor Progress:**
   ```bash
   tensorboard --logdir=logs/stage1
   ```

3. **After Stage 1 Completes:**
   ```bash
   python train_vsr_gan.py
   ```

4. **Evaluate Results:**
   ```bash
   python eval_vsr.py
   ```

5. **Create Visualizations:**
   ```bash
   python frames_to_video.py
   python visual_compare.py
   ```

---

## Troubleshooting & FAQ

### Q: How do I resume training from a checkpoint?

A: Edit the training script to load checkpoint at start:
```python
checkpoint = torch.load('weights/basicvsr_stage1.pth')
model.load_state_dict(checkpoint)
# Training will continue from this point
```

### Q: Can I use this for 4K video?

A: With 8 GPU memory:
- Resize input to 512×512 (instead of 720p)
- Reduce batch size to 1 (already minimum)
- Use gradient accumulation

### Q: What if my dataset has different frame lengths?

A: `inference_vsr.py` handles arbitrary lengths:
```python
# Auto-detects frame count
all_imgs = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
num_frames = len(all_imgs)  # Works for any number
```

### Q: How do I use pre-trained models?

A: Place checkpoint in `weights/` and load:
```python
model = BasicVSR(spynet_path='weights/spynet.pth')
checkpoint = torch.load('weights/basicvsr_gan.pth')
model.load_state_dict(checkpoint)
model.eval()
```

---

## License & Citation

This Part 2 implementation is based on:
- **BasicVSR:** Propagate better by recurrent network (CVPR 2021)
- **Real-ESRGAN:** Practical Blind Real-World Super-Resolution (ICCV 2021)

For academic use, please cite:
```
@inproceedings{chan2021basicvsr,
  title={BasicVSR: The Search for Essential Components in Video Super-Resolution},
  author={Chan, Kelvin CK and others},
  booktitle={CVPR},
  year={2021}
}
```

---

## Contact & Support

For questions or issues, refer to:
1. `📖_START_HERE.md` - Navigation guide
2. `FILE_USAGE_MATRIX.md` - Detailed file dependencies
3. `DEPENDENCY_MAP.md` - Architecture diagrams
4. `check.py` - Debug individual components

**Last Updated:** 2024
**Framework:** PyTorch
**Python Version:** 3.8+
