# Part 2: SOTA Reproduction - File Structure & Organization

## 📌 目标概览
根据论文要求实现现代VSR框架，包含：
1. **Feature Alignment & Reconstruction**: BasicVSR (双向传播 + 光流对齐)
2. **Perceptual Enhancement**: Real-ESRGAN (GAN + 感知损失)

---

## 📂 文件分类与说明

### 🔴 **核心模型架构** (Model Architecture)

#### 1. `model_basicvsr.py` 
- **功能**: 实现BasicVSR核心模型
- **关键特性**:
  - 双向特征传播 (Backward/Forward Propagation)
  - SpyNet光流对齐 (Optical Flow Alignment)
  - 残差重建 (Residual Reconstruction)
  - 4倍上采样 (PixelShuffle x4)
- **输入**: 低分辨率视频序列 [B, T, 3, H, W]
- **输出**: 高分辨率视频 [B, T, 3, 4H, 4W]
- **论文参考**: BasicVSR (Huang et al., CVPR 2021) [6]

#### 2. `model_spynet.py`
- **功能**: 空间金字塔光流网络
- **关键特性**:
  - 6层金字塔结构 (Multi-scale pyramid)
  - 从粗到细的光流估计
  - 权重加载适配 (Handles .conv. to .0. mapping)
- **用途**: 在BasicVSR中用于特征对齐
- **输入**: 两帧图像对
- **输出**: 光流 [B, 2, H, W]

#### 3. `model_discriminator.py`
- **功能**: UNet判别器（用于GAN训练）
- **关键特性**:
  - UNet架构（编码器-解码器）
  - 谱归一化 (Spectral Normalization) - 稳定训练
  - 跳连接 (Skip Connections) - 改进梯度流
  - 像素级对抗指导
- **用途**: Stage 2 GAN训练中判别真假
- **论文参考**: Real-ESRGAN UNet Discriminator

---

### 🟠 **损失函数** (Loss Functions)

#### 4. `loss_gan.py`
- **功能**: GAN训练所需的各类损失函数
- **包含模块**:
  - **PerceptualLoss**: 基于VGG19的特征级损失
    - 提取中层特征 (layer 35)
    - ImageNet标准化
    - 解决"回归到平均"问题（模糊）
  - **GANLoss**: 对抗损失
    - Vanilla GAN Loss
    - LSGAN可选
- **输入**: 生成图像 vs 真实图像
- **输出**: 标量损失值
- **论文参考**: Real-ESRGAN (Wang et al., CVPR 2021) [8]

---

### 🟡 **数据处理** (Data Handling)

#### 5. `vsr_dataset.py`
- **功能**: Vimeo90K数据集加载
- **关键特性**:
  - 支持扁平化目录结构
  - 连续T帧加载（默认7帧）
  - 8位补零帧编号映射 (00000000.png ~ 00000006.png)
  - 内存优化（高分辨率720p输出）
- **分割**: train/test/val split
- **输出**: 低分辨率序列 + 高分辨率GT序列

---

### 🟢 **训练脚本** (Training Scripts)

#### 6. `train_vsr_stage1.py` ⭐ **必须先运行**
- **阶段**: Stage 1 - 特征对齐与重建
- **目标**: 最大化PSNR，确保结构与颜色准确性
- **损失函数**:
  - L1 Loss 或 Charbonnier Loss (推荐)
  - Charbonnier更稳定，对异常值敏感性低
- **关键优化**:
  - 冻结SpyNet权重（保证稳定光流）
  - 梯度裁剪（防止RNN梯度爆炸）
  - 分离参数学习率（SpyNet用更低LR）
- **预期结果**: 高PSNR，锐边但略显平滑
- **输出**: `basicvsr_stage1.pth`

#### 7. `train_vsr_gan.py` ⭐ **基于Stage 1继续训练**
- **阶段**: Stage 2 - 感知增强（GAN训练）
- **目标**: 恢复纹理，生成逼真输出
- **损失函数组合**:
  - Charbonnier Loss (像素级)
  - Perceptual Loss (特征级，VGG19)
  - GAN Loss (对抗训练)
- **训练策略**:
  - 加载Stage 1预训练权重
  - 两阶段学习率调度
  - 对生成器和判别器交替训练
- **预期结果**: 锐边+纹理恢复，PSNR可能略低但视觉质量显著提升
- **输出**: `basicvsr_gan.pth`

---

### 🔵 **推理与评估** (Inference & Evaluation)

#### 8. `inference_vsr.py`
- **功能**: 对任意长度的视频序列进行推理
- **特性**:
  - 自动检测帧数
  - 支持任意序列长度
  - 递归传播处理
  - 逐帧保存结果
- **使用方式**:
  ```bash
  python inference_vsr.py --checkpoint basicvsr_gan.pth --input-dir ./000 --output-dir ./vsrgan_result/000
  ```

#### 9. `eval_vsr.py`
- **功能**: 评估模型性能
- **指标计算**:
  - **PSNR**: 峰值信噪比（越高越好，但GAN模型可能偏低）
  - **SSIM**: 结构相似度（感知质量）
- **输出**:
  - CSV格式指标表
  - 对比图像（模型输出 vs. GT）
  - 可视化对比
- **使用方式**:
  ```bash
  python eval_vsr.py --checkpoint basicvsr_stage1.pth --save-dir ./evalresult/basicvsr --csv-out ./results.csv
  ```

---

### 🟣 **工具脚本** (Utility Scripts)

#### 10. `visual_compare.py`
- **功能**: 生成并排可视化对比
- **用途**: 验证纹理恢复、边界锐度等感知质量

#### 11. `frames_to_video.py`
- **功能**: 将输出帧序列转换为视频文件
- **支持格式**: MP4, AVI 等
- **用途**: 生成最终演示视频

#### 12. `check.py`
- **功能**: 调试与检查脚本
- **用途**: 验证模型结构、权重形状等

#### 13. `basicvsr_net.py`
- **备用模型实现**（可选）

---

## 🚀 **完整训练流程**

### **第一步：Stage 1 训练** (3-5小时)
```bash
# 原始帧分辨率下的特征对齐
python train_vsr_stage1.py \
  --data-root /path/to/vimeo90k \
  --batch-size 2 \
  --num-epochs 50 \
  --save-dir ./checkpoints/stage1
```
- 输出: `checkpoints/stage1/basicvsr_stage1.pth`
- 预期PSNR: 28-32 dB

### **第二步：Stage 2 训练** (5-8小时)
```bash
# 基于Stage 1 微调，加入GAN和感知损失
python train_vsr_gan.py \
  --pretrained ./checkpoints/stage1/basicvsr_stage1.pth \
  --data-root /path/to/vimeo90k \
  --batch-size 1 \
  --num-epochs 30 \
  --save-dir ./checkpoints/stage2
```
- 输出: `weights/basicvsr_gan.pth`
- 预期PSNR: 27-30 dB (略低但视觉质量更好)
- 预期SSIM: 0.75-0.85 (视觉效果更优)

### **第三步：推理**
```bash
python inference_vsr.py \
  --checkpoint ./weights/basicvsr_gan.pth \
  --input-dir ./val_data/000 \
  --output-dir ./vsrgan_result/000
```

### **第四步：评估**
```bash
python eval_vsr.py \
  --checkpoint ./weights/basicvsr_gan.pth \
  --data-root /path/to/vimeo90k \
  --save-dir ./evalresult \
  --csv-out ./metrics.csv
```

---

## 📊 **性能指标对比**

| 方法 | PSNR (dB) | SSIM | 特点 |
|------|-----------|------|------|
| **Part 1 (SRCNN)** | 25-27 | 0.65-0.70 | 快速但模糊 |
| **Stage 1 (L1)** | 28-32 | 0.72-0.78 | 锐利, 高PSNR |
| **Stage 2 (GAN)** | 27-30 | 0.75-0.85 | 纹理丰富, 逼真 |

---

## 🔑 **关键概念解析**

### **1. 双向传播 (Bidirectional Propagation)**
- 从第一帧向前传播特征
- 从最后一帧向后传播特征
- 融合得到最终高质量输出

### **2. 光流对齐 (Optical Flow Alignment)**
- SpyNet估计帧间运动
- 使用光流扭曲特征到当前帧
- 处理镜头运动和物体运动

### **3. Charbonnier Loss**
```
L_charbonnier(x, y) = sqrt(||x - y||^2 + eps^2) - eps
```
优点：对异常值（如快速运动）更鲁棒

### **4. 感知损失 (Perceptual Loss)**
- 不直接比较像素
- 比较VGG19提取的特征空间
- 更符合人眼感知

### **5. GAN损失解决的问题**
- **回归到平均 (Regression-to-the-Mean)**: L1/MSE损失倾向生成模糊平均图像
- **解决方案**: GAN强制生成器产生与真实数据分布一致的锐利纹理

---

## 📝 **文件检查清单**

- [ ] `model_basicvsr.py` - BasicVSR核心模型 ✓
- [ ] `model_spynet.py` - 光流网络 ✓
- [ ] `model_discriminator.py` - 判别器 ✓
- [ ] `loss_gan.py` - 损失函数 ✓
- [ ] `vsr_dataset.py` - 数据加载 ✓
- [ ] `train_vsr_stage1.py` - Stage 1训练 ✓
- [ ] `train_vsr_gan.py` - Stage 2训练 ✓
- [ ] `inference_vsr.py` - 推理脚本 ✓
- [ ] `eval_vsr.py` - 评估脚本 ✓
- [ ] `visual_compare.py` - 可视化对比 ✓
- [ ] `frames_to_video.py` - 帧转视频 ✓

---

## 🎯 **下一步建议**

1. **验证数据加载**: 运行 `check.py` 确保Vimeo90K数据正确加载
2. **Stage 1微调**: 根据显存调整batch_size，监控梯度
3. **Stage 2对齐**: 确保加载Stage 1预训练权重
4. **定期评估**: 每5个epoch在验证集上运行`eval_vsr.py`
5. **可视化对比**: 使用`visual_compare.py`监控纹理恢复质量

---

## 📚 **论文参考**

- [6] BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond (CVPR 2021)
- [7] BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment (CVPR 2022)
- [8] Real-ESRGAN: Practical Blind Real-World Super-Resolution with Generative Adversarial Networks (CVPR 2021)

---

**编译日期**: 2026-05-14  
**版本**: v1.0
