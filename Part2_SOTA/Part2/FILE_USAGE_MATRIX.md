# Part 2: 文件使用对照表

## 🎯 按训练阶段分类

### 🟢 **Stage 1: 特征对齐与重建** (仅用L1损失)

| 文件 | 角色 | 必需性 |
|------|------|--------|
| **train_vsr_stage1.py** ⭐ | 主训练脚本 | ✅ 必需 |
| **model_basicvsr.py** ⭐ | 生成器网络 | ✅ 必需 |
| **model_spynet.py** ⭐ | 光流估计 | ✅ 必需 |
| **vsr_dataset.py** ⭐ | 数据加载 | ✅ 必需 |
| **loss_gan.py** | CharbonnierLoss | ✅ 必需 (仅用一个) |
| **weights/spynet.pth** ⭐ | 预训练权重 | ✅ 必需 |

**Stage 1 输出**: `checkpoints/basicvsr_stage1.pth`

**数据流**:
```
train_vsr_stage1.py
├─ 导入 model_basicvsr.py
│  └─ 导入 model_spynet.py (加载 spynet.pth)
├─ 导入 vsr_dataset.py (加载Vimeo90K数据)
├─ 导入 loss_gan.py (使用CharbonnierLoss)
└─ 循环训练
   ├─ 读取[LR, HR] from vsr_dataset.py
   ├─ 前向 model_basicvsr.py
   ├─ 计算 CharbonnierLoss
   └─ 反向更新权重
```

---

### 🟠 **Stage 2: 感知增强 (GAN训练)**

| 文件 | 角色 | 必需性 |
|------|------|--------|
| **train_vsr_gan.py** ⭐ | 主训练脚本 | ✅ 必需 |
| **model_basicvsr.py** ⭐ | 生成器 | ✅ 必需 |
| **model_spynet.py** ⭐ | 光流估计 | ✅ 必需 |
| **model_discriminator.py** ⭐ | 判别器 | ✅ 必需 |
| **vsr_dataset.py** ⭐ | 数据加载 | ✅ 必需 |
| **loss_gan.py** ⭐ | 三个损失函数 | ✅ 必需 (全部用) |
| **checkpoints/basicvsr_stage1.pth** ⭐ | Stage 1预训练权重 | ✅ 必需 |
| **weights/spynet.pth** ⭐ | SpyNet权重 | ✅ 必需 |
| **weights/vgg19-dcbb9e9d.pth** ⭐ | VGG19权重 | ✅ 必需 |

**Stage 2 输出**: `weights/basicvsr_gan.pth`

**关键依赖链**:
```
train_vsr_gan.py
├─ 加载 checkpoints/basicvsr_stage1.pth ⭐
│  (这是从Stage 1继承的预训练权重)
├─ 导入 model_basicvsr.py
│  └─ 加载 weights/spynet.pth
├─ 导入 model_discriminator.py
├─ 导入 vsr_dataset.py
├─ 导入 loss_gan.py
│  ├─ CharbonnierLoss (像素级)
│  ├─ PerceptualLoss (需要 vgg19-dcbb9e9d.pth) ⭐
│  └─ GANLoss (对抗)
└─ 循环训练
   ├─ Generator step: G优化
   └─ Discriminator step: D优化
```

---

## 🔵 **Stage外的辅助文件** (推理/评估/工具)

### **推理脚本**

| 文件 | 用途 | 何时使用 | 依赖 |
|------|------|---------|------|
| **inference_vsr.py** ⭐ | 对任意长度视频进行超分 | ✅ 训练后推理 | model_basicvsr.py, spynet |
| | | 输入: 低分辨率帧序列 | |
| | | 输出: 高分辨率帧序列 | |
| | | 支持递归传播处理 | |

**使用方式**:
```bash
python inference_vsr.py \
  --checkpoint weights/basicvsr_gan.pth \
  --input-dir ./val_data/000 \
  --output-dir ./results/000
```

---

### **评估脚本**

| 文件 | 用途 | 何时使用 | 输出 |
|------|------|---------|------|
| **eval_vsr.py** ⭐ | 计算PSNR/SSIM指标 | ✅ 训练完评估 | CSV表格 + 对比图像 |
| | | 支持批量验证集评估 | |
| | | 生成可视化对比 | |

**使用方式**:
```bash
python eval_vsr.py \
  --checkpoint weights/basicvsr_gan.pth \
  --data-root /path/to/vimeo90k/test \
  --csv-out ./results/metrics.csv
```

---

### **可视化工具**

| 文件 | 用途 | 何时使用 | 功能 |
|------|------|---------|------|
| **visual_compare.py** | 并排对比输出 | ✅ 推理后检查效果 | 生成[LR-SR-HR]对比图 |
| **frames_to_video.py** | 帧序列转视频 | ✅ 最后呈现结果 | 生成MP4/AVI视频文件 |

---

### **调试工具**

| 文件 | 用途 | 何时使用 | 诊断内容 |
|------|------|---------|---------|
| **check.py** | 检查环境和数据 | ✅ 遇到问题时 | ✓ SpyNet权重加载 |
| | | 培训前验证 | ✓ 数据范围检查 |
| | | | ✓ 模型结构验证 |

**使用方式**:
```bash
python check.py --mode check_spynet
python check.py --mode check_data --data-root /path/to/vimeo90k
python check.py --mode check_model
```

---

### **备用文件**

| 文件 | 用途 | 关系 |
|------|------|------|
| **basicvsr_net.py** | BasicVSR的备用实现 | 可选，与model_basicvsr.py功能相同 |

---

## 📊 完整文件使用矩阵

```
文件名                      Stage 1    Stage 2    推理    评估    其他
─────────────────────────────────────────────────────────────────────────
train_vsr_stage1.py          ✅         -         -      -      -
train_vsr_gan.py             -          ✅         -      -      -
model_basicvsr.py            ✅         ✅        ✅      ✅      -
model_spynet.py              ✅         ✅        ✅      ✅      -
model_discriminator.py        -         ✅         -      -      -
vsr_dataset.py               ✅         ✅         -      ✅      -
loss_gan.py                  ✅         ✅         -      -      -
inference_vsr.py             -          -         ✅      -      -
eval_vsr.py                  -          -         -      ✅      -
visual_compare.py            -          -         ✅      ✅      ✅
frames_to_video.py           -          -         ✅      ✅      ✅
check.py                     -          -         -      -      ✅ (调试)
basicvsr_net.py              -          -         -      -      ✅ (备用)

权重文件:
weights/spynet.pth           ✅         ✅        ✅      ✅      -
weights/vgg19-dcbb9e9d.pth   -          ✅         -      -      -
checkpoints/basicvsr_stage1  -          ✅         -      -      -
```

---

## 🎯 典型工作流程与文件使用

### **场景 1: 从零开始训练**

```
Step 1: 准备数据和权重
├─ ✅ 下载 Vimeo90K → vsr_dataset.py 读取
├─ ✅ 下载 spynet.pth → model_spynet.py 加载

Step 2: Stage 1 训练 (3-5小时)
├─ 运行: python train_vsr_stage1.py
├─ 使用: model_basicvsr.py, model_spynet.py, loss_gan.py, vsr_dataset.py
├─ 生成: checkpoints/basicvsr_stage1.pth
└─ [可选] 运行 eval_vsr.py 验证

Step 3: Stage 2 训练 (5-8小时)
├─ 准备: 下载 vgg19-dcbb9e9d.pth
├─ 运行: python train_vsr_gan.py
├─ 输入: checkpoints/basicvsr_stage1.pth
├─ 使用: model_basicvsr.py, model_discriminator.py, loss_gan.py
├─ 生成: weights/basicvsr_gan.pth
└─ [可选] 运行 eval_vsr.py 验证

Step 4: 推理新视频
├─ 运行: python inference_vsr.py
├─ 使用: weights/basicvsr_gan.pth, model_basicvsr.py
└─ 生成: 超分辨率视频帧

Step 5: 可视化和呈现
├─ 运行: python visual_compare.py (对比)
├─ 运行: python frames_to_video.py (转视频)
└─ 生成: 最终演示视频
```

### **场景 2: 只做推理 (不训练)**

```
输入: 预训练的 weights/basicvsr_gan.pth

Step 1: 推理
├─ 运行: python inference_vsr.py
└─ 使用: model_basicvsr.py, model_spynet.py

Step 2: 可视化
├─ 运行: python visual_compare.py
├─ 运行: python frames_to_video.py
└─ 输出: 超分辨率视频
```

### **场景 3: 只做评估 (不训练)**

```
输入: 预训练的 weights/basicvsr_gan.pth + 测试集

Step 1: 评估指标
├─ 运行: python eval_vsr.py
├─ 使用: model_basicvsr.py, vsr_dataset.py
└─ 输出: metrics.csv

Step 2: 对比可视化
├─ 运行: python visual_compare.py
└─ 输出: 对比图像
```

### **场景 4: 遇到问题**

```
✅ 使用 check.py 诊断:
├─ check_spynet: 验证SpyNet权重
├─ check_data: 验证Vimeo90K数据
└─ check_model: 验证模型结构

✅ 使用 QUICKSTART.md 查故障
```

---

## 🔗 文件依赖关键链

### **Stage 1 最少依赖**
```
train_vsr_stage1.py
  ├─ model_basicvsr.py
  │  └─ model_spynet.py (⚠️ 需要 weights/spynet.pth)
  ├─ vsr_dataset.py (⚠️ 需要 Vimeo90K数据)
  ├─ loss_gan.py (仅用 CharbonnierLoss)
  └─ [可选] eval_vsr.py
```

### **Stage 2 最少依赖**
```
train_vsr_gan.py
  ├─ checkpoints/basicvsr_stage1.pth ⭐ (必须有!)
  ├─ model_basicvsr.py
  │  └─ model_spynet.py (⚠️ weights/spynet.pth)
  ├─ model_discriminator.py
  ├─ vsr_dataset.py (⚠️ Vimeo90K数据)
  ├─ loss_gan.py (全部三个损失)
  │  └─ PerceptualLoss (⚠️ weights/vgg19-dcbb9e9d.pth)
  └─ [可选] eval_vsr.py
```

### **推理最少依赖**
```
inference_vsr.py
  ├─ weights/basicvsr_gan.pth
  ├─ model_basicvsr.py
  │  └─ model_spynet.py (⚠️ weights/spynet.pth)
  └─ [可选] visual_compare.py, frames_to_video.py
```

---

## ✅ 文件完整清单

### **必需文件** (13个)
```
核心模型:
  ✅ model_basicvsr.py
  ✅ model_spynet.py
  ✅ model_discriminator.py

数据处理:
  ✅ vsr_dataset.py

损失函数:
  ✅ loss_gan.py

训练脚本:
  ✅ train_vsr_stage1.py
  ✅ train_vsr_gan.py

推理评估:
  ✅ inference_vsr.py
  ✅ eval_vsr.py

工具脚本:
  ✅ visual_compare.py
  ✅ frames_to_video.py
  ✅ check.py

备用:
  ✅ basicvsr_net.py (可选备用)
```

### **必需权重** (3个)
```
✅ weights/spynet.pth (Stage 1 & 2 都需)
✅ weights/vgg19-dcbb9e9d.pth (仅Stage 2)
✅ checkpoints/basicvsr_stage1.pth (Stage 2的输入)
```

### **必需数据**
```
✅ Vimeo90K 训练集 (vsr_dataset.py读取)
✅ Vimeo90K 测试集 (eval_vsr.py评估用)
```

---

## 💡 快速查询

**Q: 我只想做Stage 1，需要哪些文件?**
```
✅ train_vsr_stage1.py (主脚本)
✅ model_basicvsr.py
✅ model_spynet.py
✅ vsr_dataset.py
✅ loss_gan.py (仅CharbonnierLoss)
✅ weights/spynet.pth
✅ Vimeo90K 数据
```

**Q: 我只想做Stage 2，需要哪些文件?**
```
✅ train_vsr_gan.py (主脚本)
✅ model_basicvsr.py
✅ model_spynet.py
✅ model_discriminator.py
✅ vsr_dataset.py
✅ loss_gan.py (全部三个损失)
✅ checkpoints/basicvsr_stage1.pth ⭐ (必须从Stage 1来!)
✅ weights/spynet.pth
✅ weights/vgg19-dcbb9e9d.pth
✅ Vimeo90K 数据
```

**Q: 我只想推理，需要哪些文件?**
```
✅ inference_vsr.py
✅ model_basicvsr.py
✅ model_spynet.py
✅ weights/basicvsr_gan.pth (已训练模型)
✅ weights/spynet.pth
✅ 低分辨率视频帧
```

**Q: 我只想评估，需要哪些文件?**
```
✅ eval_vsr.py
✅ model_basicvsr.py
✅ model_spynet.py
✅ vsr_dataset.py
✅ weights/basicvsr_gan.pth (已训练模型)
✅ weights/spynet.pth
✅ Vimeo90K 测试集
```

---

**版本**: v1.0  
**日期**: 2026-05-14
