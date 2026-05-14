# Part 2: Module Dependency Map & Execution Flow

## 🏗️ Architecture Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        🎯 Stage 1: Feature Alignment                     │
│                    (L1 Loss / Charbonnier Loss Training)                 │
└─────────────────────────────────────────────────────────────────────────┘

                                    train_vsr_stage1.py
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ↓                       ↓                       ↓
            vsr_dataset.py        model_basicvsr.py        model_spynet.py
          (Data Loading:           (BasicVSR Model)        (Optical Flow)
           Vimeo90K)              ├─ ResidualBlockNoBN      ├─ BasicModule
                                  ├─ Bidirectional Prop    ├─ 6-layer pyramid
                                  └─ Feature fusion        └─ Flow warping


                            ↓↓↓ Training Loop ↓↓↓
                                    
            Input: LR frames [B, T, 3, H, W]
                    ↓
            SpyNet: Optical Flow Estimation
                    ↓
            Backward Propagation: LR → HR (features)
                    ↓
            Forward Propagation: LR → HR (features)  
                    ↓
            Fusion & Reconstruction
                    ↓
            PixelShuffle (4x upsampling)
                    ↓
            Output: SR frames [B, T, 3, 4H, 4W]
                    ↓
            CharbonnierLoss.forward()
                    ↓
            Backpropagation & Optimization
                    ↓
            ✅ Checkpoint: basicvsr_stage1.pth (PSNR: 28-32 dB)


┌─────────────────────────────────────────────────────────────────────────┐
│                    🎯 Stage 2: Perceptual Enhancement                    │
│            (GAN Loss + Perceptual Loss + Pixel Loss Training)            │
└─────────────────────────────────────────────────────────────────────────┘

                                 train_vsr_gan.py
                                        │
                ┌───────────────────────┼────────────────────────┐
                │                       │                        │
                ↓                       ↓                        ↓
        vsr_dataset.py        model_basicvsr.py      model_discriminator.py
        (Data Loading)        (Generator G)          (Discriminator D)
                              + basicvsr_stage1.pth  ├─ UNet Architecture
                                                     ├─ Spectral Norm
                                                     └─ Skip Connections
                                │
                                ↓
                            loss_gan.py
                        ├─ CharbonnierLoss (pixel)
                        ├─ PerceptualLoss (VGG19)
                        └─ GANLoss (adversarial)


                        ↓↓↓ GAN Training Loop ↓↓↓

    For each batch:
    
    1️⃣ Generator (G) step:
       Input: LR frames → G.forward() → SR output
       Loss = CharbonnierLoss(SR, HR)
            + λ_perceptual × PerceptualLoss(SR, HR)
            + λ_gan × GANLoss(D(SR), True)
       Backprop & optimize G

    2️⃣ Discriminator (D) step:
       Real: Loss = GANLoss(D(HR), True)
       Fake: Loss = GANLoss(D(SR), False)
       Total_D_Loss = Loss_Real + Loss_Fake
       Backprop & optimize D

    ↓ Repeat for all batches
    
    ✅ Checkpoint: basicvsr_gan.pth (PSNR: 27-30 dB, SSIM: 0.75-0.85)


┌─────────────────────────────────────────────────────────────────────────┐
│                      🎯 Inference & Evaluation                           │
└─────────────────────────────────────────────────────────────────────────┘

    inference_vsr.py              eval_vsr.py
         │                              │
         ├─ Load checkpoint ◄──────────┤
         │                              │
         ├─ For each sequence:          │
         │  ├─ Load LR frames           │
         │  ├─ Forward pass             │
         │  └─ Save SR output           │
         │                              ├─ Load validation data
         └─ Output: SR videos           │
                                        ├─ Calculate PSNR
                                        ├─ Calculate SSIM
                                        └─ Output: CSV + comparison images
```

---

## 📋 File Dependency Matrix

| 文件 | 依赖于 | 被依赖于 | 阶段 |
|------|--------|---------|------|
| `model_basicvsr.py` | model_spynet.py | train_vsr_stage1.py, train_vsr_gan.py, inference_vsr.py | Both |
| `model_spynet.py` | torch | model_basicvsr.py | Both |
| `model_discriminator.py` | torch | train_vsr_gan.py | 2 |
| `loss_gan.py` | torch, torchvision | train_vsr_stage1.py (CharbonnierLoss), train_vsr_gan.py | Both |
| `vsr_dataset.py` | torch, PIL | train_vsr_stage1.py, train_vsr_gan.py, eval_vsr.py | Both |
| `train_vsr_stage1.py` | ✓ All models & losses | - | 1 |
| `train_vsr_gan.py` | ✓ All models & losses | - | 2 |
| `inference_vsr.py` | model_basicvsr.py | - | Both |
| `eval_vsr.py` | model_basicvsr.py, vsr_dataset.py | - | Both |
| `train_vsr_unified.py` | All scripts | - | Bridge |

---

## 🔄 Execution Flow with Data

### **Stage 1: Training Flow**

```
START
  ↓
[Load config.yaml]
  ↓
[Initialize model_basicvsr.py + load spynet.pth]
  ↓
[Load vsr_dataset.py → Vimeo90K/train]
  ↓
FOR each epoch:
  ↓
  FOR each batch:
    ↓
    [Load T=7 LR frames: [B, 7, 3, H, W]]
    [Load T=7 HR frames: [B, 7, 3, 4H, 4W]]
    ↓
    [model_basicvsr.forward(LR_frames)]
      ├─ model_spynet.forward(frame_t, frame_t+1) → optical flow
      ├─ Backward propagation with flow warping
      ├─ Forward propagation with flow warping
      └─ Output: SR_frames [B, 7, 3, 4H, 4W]
    ↓
    [loss_gan.CharbonnierLoss(SR_frames, HR_frames)]
    ↓
    [Optimizer.backward() + Optimizer.step()]
    ↓
    [Log loss to TensorBoard]
  ↓
  [Save checkpoint every save_freq epochs]
  ↓
  [Evaluate on validation set → PSNR/SSIM]
↓
SAVE: checkpoints/basicvsr_stage1.pth
↓
END
```

### **Stage 2: Training Flow**

```
START
  ↓
[Load config.yaml]
  ↓
[Load model_basicvsr.py + basicvsr_stage1.pth]  ⭐ CRITICAL
[Initialize model_discriminator.py]
  ↓
[Load vsr_dataset.py → Vimeo90K/train]
  ↓
FOR each epoch:
  ↓
  FOR each batch:
    ↓
    [Load T=7 LR frames & HR frames]
    ↓
    ┌─── GENERATOR STEP ───┐
    │                      │
    │ [model_basicvsr      │
    │  .forward(LR_frames)]│
    │  → SR_frames         │
    │                      │
    │ [Loss calculation]   │
    │  + CharbonnierLoss   │
    │  + PerceptualLoss    │
    │  + GANLoss           │
    │                      │
    │ [Backprop G]         │
    │                      │
    └──────────────────────┘
           ↓
    ┌─── DISCRIMINATOR STEP ───┐
    │                          │
    │ [Real sample]            │
    │ model_discriminator      │
    │ .forward(HR_frames)      │
    │ → output_real            │
    │                          │
    │ [Fake sample]            │
    │ model_discriminator      │
    │ .forward(SR_frames)      │
    │ → output_fake            │
    │                          │
    │ [Loss calculation]       │
    │  GANLoss(real) +         │
    │  GANLoss(fake)           │
    │                          │
    │ [Backprop D]             │
    │                          │
    └──────────────────────────┘
           ↓
    [Log losses]
  ↓
  [Save checkpoint]
  ↓
  [Evaluate]
↓
SAVE: weights/basicvsr_gan.pth
↓
END
```

---

## 🎯 Data Flow Visualization

### **Input Data Format**

```
Vimeo90K Dataset Structure:
├── train/
│   ├── 00000/
│   │   ├── 00000000.png  (LR low-res)
│   │   ├── 00000001.png
│   │   ├── ...
│   │   ├── 00000006.png
│   │   ├── 00000000.png  (HR high-res) [duplicated, see vsr_dataset.py]
│   │   └── ...
│   ├── 00001/
│   ├── ...
│   └── 89999/

Dataset.__getitem__(idx) returns:
{
    'lr': Tensor of shape [T=7, 3, H, W]       # Low-res: 64×64 or similar
    'hr': Tensor of shape [T=7, 3, 4H, 4W]    # High-res: 256×256 or similar
    'folder_idx': int
}

DataLoader output (batch):
{
    'lr': Tensor [B=2, T=7, 3, H, W]
    'hr': Tensor [B=2, T=7, 3, 4H, 4W]
}
```

### **Model Output Dimensions**

```
Input:  [B, T, C, H, W] = [2, 7, 3, 64, 64]
         │   │  │   │  └─ Width
         │   │  │   └──── Height
         │   │  └──────── Color channels
         │   └─────────── Time steps (7 frames)
         └──────────────── Batch size

BasicVSR Processing:
├─ Feature Extract: [B, T, 3, H, W] → [B, T, 64, H, W]
├─ Backward Propagation:
│  └─ State h_b: [B, 128, H, W] (accumulated)
├─ Forward Propagation:
│  └─ State h_f: [B, 128, H, W] (accumulated)
├─ Fusion: [B, 128, H, W] → [B, 64, H, W]
├─ Upsampling x2: [B, 64, H, W] → [B, 64, 2H, 2W]
├─ Upsampling x2: [B, 64, 2H, 2W] → [B, 3, 4H, 4W]

Output: [B, T, 3, 4H, 4W] = [2, 7, 3, 256, 256]

Loss Calculation:
├─ SR: [B×T, 3, 4H, 4W]
├─ HR: [B×T, 3, 4H, 4W]
└─ Loss: Scalar (averaged over batch & time)
```

---

## 🔐 Key Weight Dependencies

```
Pre-trained Weights Required:
├─ weights/spynet.pth
│  ├─ Used in: model_basicvsr.py
│  ├─ Usage: SpyNet(load_path='weights/spynet.pth')
│  ├─ Format: PyTorch .pth checkpoint
│  └─ Size: ~500 MB
│
└─ weights/vgg19-dcbb9e9d.pth (for Stage 2 only)
   ├─ Used in: loss_gan.py
   ├─ Usage: PerceptualLoss(model_path='weights/vgg19-dcbb9e9d.pth')
   ├─ Format: PyTorch .pth checkpoint
   └─ Size: ~500 MB

Generated Checkpoints:
├─ checkpoints/basicvsr_stage1.pth
│  ├─ Generated by: train_vsr_stage1.py
│  ├─ Used in: train_vsr_gan.py (load as pretrained)
│  ├─ Size: ~1.2 GB
│  └─ PSNR: 28-32 dB
│
└─ weights/basicvsr_gan.pth
   ├─ Generated by: train_vsr_gan.py
   ├─ Used in: inference_vsr.py, eval_vsr.py
   ├─ Size: ~1.2 GB
   └─ PSNR: 27-30 dB (SSIM improved)
```

---

## 📊 Loss Function Computation Graph

### **Stage 1**

```
SR_output [B, T, 3, 4H, 4W]
         ↓
    CharbonnierLoss.forward()
         ├─ diff = SR_output - HR_target
         ├─ loss_map = sqrt(diff^2 + eps^2) - eps
         └─ scalar_loss = mean(loss_map)
         ↓
    BACKWARD PASS
         ├─ .backward()
         ├─ Gradient computation
         ├─ Gradient clipping (max_norm=5.0)
         └─ Optimizer.step()
```

### **Stage 2**

```
┌──────────────────────────────────────────────────┐
│ G_loss = λ_pix × L_pix                           │
│        + λ_perc × L_perc                         │
│        + λ_gan × L_gan                           │
└──────────────────────────────────────────────────┘

SR_output [B, T, 3, 4H, 4W]
    │
    ├─→ [1] CharbonnierLoss(SR, HR)
    │       → scalar: L_pix
    │
    ├─→ [2] PerceptualLoss.forward(SR, HR)
    │       ├─ VGG19(SR) → [B, 512, H', W']
    │       ├─ VGG19(HR) → [B, 512, H', W']
    │       └─ L1(feat_SR, feat_HR) → scalar: L_perc
    │
    ├─→ [3] D(SR) → validity_fake
    │       GANLoss(validity_fake, True)
    │       → scalar: L_gan
    │
    └─→ Total Loss = 1.0×L_pix + 1.0×L_perc + 0.1×L_gan

D_loss = L_gan_real + L_gan_fake
    ├─ Real: D(HR) → validity_real
    │         GANLoss(validity_real, True)
    └─ Fake: D(SR) → validity_fake
             GANLoss(validity_fake, False)
```

---

## 🔗 Import Dependency Chain

```
train_vsr_stage1.py imports:
├─ vsr_dataset.Vimeo90KDataset
├─ model_basicvsr.BasicVSR
├─ loss_gan.CharbonnierLoss
└─ pytorch, tensorboard, etc.

model_basicvsr.py imports:
├─ model_spynet.SpyNet
├─ torch.nn.PixelShuffle
└─ torch.nn.functional

train_vsr_gan.py imports:
├─ vsr_dataset.Vimeo90KDataset
├─ model_basicvsr.BasicVSR
├─ model_discriminator.UNetDiscriminatorSN
├─ loss_gan.CharbonnierLoss
├─ loss_gan.PerceptualLoss
├─ loss_gan.GANLoss
└─ pytorch, tensorboard, etc.

eval_vsr.py imports:
├─ model_basicvsr.BasicVSR
├─ vsr_dataset.Vimeo90KDataset
├─ torch
└─ skimage.metrics
```

---

## ✅ Checklist for Running Complete Pipeline

```
Stage 1 Setup:
☐ Verify weights/spynet.pth exists
☐ Edit config.yaml with correct data_root
☐ Run: python train_vsr_unified.py --stage 1 --config config.yaml
☐ Monitor: tensorboard --logdir logs/
☐ Output: checkpoints/basicvsr_stage1.pth ✓

Stage 2 Setup (Optional):
☐ Verify weights/vgg19-dcbb9e9d.pth exists
☐ Update config.yaml (adjust gan_loss_weight, etc.)
☐ Run: python train_vsr_unified.py --stage 2 \
         --pretrained checkpoints/basicvsr_stage1.pth \
         --config config.yaml
☐ Monitor: tensorboard --logdir logs/
☐ Output: weights/basicvsr_gan.pth ✓

Inference:
☐ Run: python train_vsr_unified.py --infer \
         --checkpoint weights/basicvsr_gan.pth \
         --input-dir ./val_data/000 \
         --output-dir ./results/000
☐ Output: SR video frames in results/000/ ✓

Evaluation:
☐ Run: python train_vsr_unified.py --eval \
         --checkpoint weights/basicvsr_gan.pth \
         --data-root /path/to/vimeo90k/test \
         --output-csv ./results/metrics.csv
☐ Output: CSV with PSNR/SSIM metrics ✓
```

---

**图表生成日期**: 2026-05-14  
**版本**: v1.0
