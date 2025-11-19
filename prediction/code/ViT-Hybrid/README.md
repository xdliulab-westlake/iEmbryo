# ViT-Hybrid：胚胎质量评估深度学习系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-v2.0-green.svg)](https://github.com/lzzsh/IVF_AI)

ViT-Hybrid 是一个专为胚胎质量二分类（好/差）设计的深度学习系统。该系统采用 **VGG16 + Transformer** 架构，通过时空联合建模实现精准的胚胎评估。

## 5 分钟快速入门

```bash
# 1. 首次运行（验证数据并缓存）
python main.py train \
  --data_root /liaozizhuo/ \
  --multi_focal \
  --selected_images_csv /liaozizhuo/selected_image_paths.csv \
  --maindb_csv /liaozizhuo/scripts/MainDB.csv \
  --filtered_csv /liaozizhuo/scripts/filtered_clean_embryos.csv \
  --two_stage

# 2. 后续运行（使用缓存，极速）
python main.py train \
  --data_root /liaozizhuo/ \
  --multi_focal \
  --selected_images_csv /liaozizhuo/selected_image_paths.csv \
  --maindb_csv /liaozizhuo/scripts/MainDB.csv \
  --filtered_csv /liaozizhuo/scripts/filtered_clean_embryos.csv \
  --two_stage \
  --skip_validation

# 3. 查看结果
ls results/        # 训练历史和评估指标
ls checkpoints/    # 模型权重
ls cache/          # 验证缓存（下次可复用）
```

**就这么简单！** 系统自动处理数据验证、缓存、模型训练等所有细节。第一次运行约 5-10 分钟（取决于数据量），后续运行使用缓存仅需 1-2 分钟。

---

## 一、数据结构

### 多焦距多时间点数据

```
project_root/
├── image/
│   ├── embryo_id_1/
│   │   └── images/
│   │       ├── F-15/           # 焦平面
│   │       │   ├── embryo_id_1_RUN_000_F-15.jpeg
│   │       │   └── ...
│   │       ├── F0/
│   │       │   └── ...
│   │       └── ...             # 共11个焦平面
│   └── embryo_id_2/
│       └── ...
├── selected_image_paths.csv    # 时间点和焦平面映射
├── MainDB.csv                  # 质量标签（ICM, TE, Quality）
└── filtered_clean_embryos.csv  # 补充标签（可选）
```

**关键文件说明：**
- **selected_image_paths.csv**：包含 (PatientID, EmbryoNumber, FocalPlane, TargetHour, FullPath)
- **MainDB.csv**：包含 (EmbryoID, ICM, TE, Quality)
- **filtered_clean_embryos.csv**：补充低质量胚胎标签

**数据规模：**
- 时间点：33 个（96-112 小时，每 0.5 小时采样）
- 焦平面：11 个（不同深度）
- 单个胚胎：33 × 11 = 363 张图像

---

## 二、系统架构 v2.3

### 核心设计理念

胚胎在 96-112 小时发育期间：
- **时间维度**：33 个时间点，每 0.5 小时采样一次
- **空间维度**：每个时间点有 11 个焦平面，记录三维形态
- **模型策略**：VGG16 → FocalFusion → Transformer(4层) → **TemporalAttention(8头Multi-Head Self-Attention)** → 分类头

**架构亮点**：
- 🎯 **时空联合建模**：完整建模33个时间步间的复杂依赖关系 (O(33²))
- 🧠 **Multi-Head Self-Attention**：8个独立注意力头学习不同的时间模式
- ⚙️ **Vision Transformer最佳实践**：Pre-Norm + GELU + LayerNorm确保梯度稳定流动
- 🎓 **梯度优化**：Xavier(gain=1.0)初始化 + 残差连接 + LayerNorm

### 模型流程

```
输入：(batch, 33, 11, 224, 224, 3)
      批次 × 时间点 × 焦平面 × 高 × 宽 × 通道
  │
  ├─ VGG16 特征提取（ImageNet 预训练）
  │  ├─ 重整形：(batch×33×11, 224, 224, 3)
  │  ├─ 分批处理避免数值溢出（per-GPU: 64）
  │  └─ 输出：(batch×33×11, 4096)
  │
  ├─ 焦平面融合 (FocalPlaneFusion)
  │  ├─ 恢复形状：(batch, 33, 11, 4096)
  │  ├─ 逐时间点融合11个焦平面
  │  └─ 输出：(batch, 33, 4096)
  │
  ├─ 位置编码 (PositionalEncoding)
  │  └─ 可学习嵌入 (33, 4096) 按加法组合
  │
  ├─ Transformer 编码器（4层，Post-Norm）
  │  ├─ MultiheadAttention: 8头，512维每头
  │  ├─ FeedForward: 4096 → 2048 → 4096
  │  ├─ LayerNorm × 2 + 残差连接
  │  └─ 输出：(batch, 33, 4096)
  │
  ├─ 时间注意力池化 ⭐ v2.3: Multi-Head Self-Attention
  │  ├─ 1. Self-Attention (8头，512维)
  │  ├─ 2. 残差连接 + LayerNorm
  │  ├─ 3. 学习权重：FC(4096→2048)→GELU→FC(2048→1)→Softmax
  │  │     输出：(batch, 33) 时间权重分布
  │  ├─ 4. 加权求和 + LayerNorm
  │  └─ 输出：(batch, 4096)
  │
  ├─ 分类头（Pre-Norm + GELU，参考Vision Transformer）
  │  ├─ LayerNorm(4096)           ← Pre-Norm稳定输入
  │  ├─ Linear(4096, 512)         ← Xavier(gain=1.0)
  │  ├─ GELU()                    ← 平滑梯度
  │  ├─ Dropout(0.3)
  │  └─ Linear(512, 2)            ← Xavier(gain=1.0)
  │
  ├─ 损失函数：CrossEntropyLoss(weight=[w_poor, w_good])
  │  ├─ w_poor ≈ 0.60
  │  └─ w_good ≈ 3.02  ← 5倍权重迫使学习Good特征
  │
  └─ 输出：logits ∈ ℝ² (batch, 2)
           预测：softmax(logits) → (batch, 2) 概率分布
```

### 核心模块

| 模块 | 功能 | 输入维度 | 输出维度 | 备注 |
|------|------|----------|---------|------|
| **VGG16** | 空间特征提取（预训练微调） | (224, 224, 3) | 4096 | ImageNet 预训练 |
| **焦平面融合** | 11 个焦平面平均聚合 | (11, 4096) | 4096 | 空间维度融合 |
| **Transformer** | 时间序列建模（4 层） | (33, 4096) | (33, 4096) | 时间维度建模 |
| **时间注意力池化** ⭐ | 可学习权重的时间聚合 | (33, 4096) | 4096 | **NEW**: 替代全局平均池化 |
| **分类头** | 二分类预测 | 4096 | 2 | **v2.2**: logits输出（加权CrossEntropyLoss） |

#### 时间注意力池化（新增）⭐

**问题与动机：**

原始方法使用全局平均池化 (`torch.mean(x, dim=1)`) 会丢失关键的时间动态信息：
- 33 个时间步的所有信息被压缩为单一平均值
- 无法区分不同时间步的重要性
- 无法捕捉胚胎发育的时间轨迹

**进化历程：**

```
v2.0: 全局平均池化 (平均值)
  └─ 问题：完全丧失时间信息

v2.1: 简单FC注意力 (FC → ReLU → FC → Softmax)
  ├─ 改进：学习时间步权重
  └─ 限制：单层线性映射，无法建模时间步间关系

v2.2 (当前): Multi-Head Self-Attention ⭐
  ├─ 改进：8个注意力头学习不同的时间模式
  ├─ 优势：充分建模33个时间步间的复杂依赖关系
  └─ 输出：自适应的时间权重分布
```

**最新架构（Multi-Head Self-Attention）：**

```
输入：(batch, 33, 4096)  [33 个时间步的特征]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Multi-Head Self-Attention (8个注意力头)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 每个头独立学习一种时间模式
  - 头维度 = 4096 / 8 = 512
  - 完全建模所有时间步间的依赖关系
  └─ 输出：(batch, 33, 4096)

  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. 残差连接 + LayerNorm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 稳定特征方差
  - 便于梯度流动
  └─ 输出：(batch, 33, 4096)

  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. 学习时间步权重（FC层）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - FC1: 4096 → 2048 + GELU激活
  - FC2: 2048 → 1
  - Softmax(dim=1) → (batch, 33)
  └─ 输出：各时间步的重要性分数

  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. 加权求和 + LayerNorm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  output = Σ(x_t * weight_t)  for t in 1..33
  └─ 输出：(batch, 4096)  [融合了时间重要性的特征]
```

**三个版本对比：**

| 版本 | 机制 | 参数 | 复杂度 | 时间关系建模 |
|-----|------|------|--------|-----------|
| **v2.0** | 全局平均 | ~0 | O(33) | ✗ 无 |
| **v2.1** | 简单FC注意力 | ~8M | O(33) | △ 弱（线性） |
| **v2.2** | Multi-Head Self-Attn | ~32M | O(33²) | ✓ 强（非线性） |

**优势：**
1. **充分的时间建模**：Multi-Head Self-Attention 捕捉33个时间步间的复杂关系
2. **多角度学习**：8个独立的注意力头，每个学习不同的时间模式
3. **梯度稳定**：Pre-Norm + 残差连接，保证梯度流畅
4. **解释性强**：可视化注意力权重了解模型关注的发育阶段
5. **动态适应**：不同胚胎自适应不同的时间关注模式

#### VGG16 微调策略

```
层分析：
  第 1-5 层（Block 1-2）→ 冻结（低阶边缘纹理特征）
  第 6-13 层（Block 3-5） → 训练（高阶语义特征）

优势：
  - 保留通用视觉基础，避免过度微调
  - 统一训练优于两阶段（充分利用梯度）
  - 防止在小数据集上过拟合
```

#### Transformer 编码器细节

```
配置：
  - 层数：4
  - 注意力头：8
  - 头维度：4096 / 8 = 512
  - 前馈网络：4096 → 2048 → 4096
  - 位置编码：可学习嵌入

参数量：~70M
```

#### 焦平面融合策略

```
为什么用平均而非注意力？
  - 避免模型偏向单个焦平面，丢失三维信息
  - 确保均匀融合各深度层的形态特征
  - 降低计算成本和过拟合风险

公式：
  fusion = mean([f_0, f_1, ..., f_10])  # 简单高效
```

---

## 三、环境与数据准备

### 环境安装

```bash
pip install torch torchvision
pip install numpy pillow scikit-learn opencv-python
```

### 数据预处理流程

**图像预处理步骤：**

1. **读取图像** - OpenCV / PIL 加载，转 RGB
2. **培养皿检测**（可选） - Hough 圆检测，自动裁剪
3. **中心裁剪**（可选） - 提取图像中心正方形
4. **尺寸调整** - 统一为 224×224（双线性插值）
5. **张量化与归一化** - ImageNet 标准化

```python
# ImageNet 官方参数
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

**数据分割：**

```
原始数据分布：3469 Poor + 689 Good（5:1 不平衡）

分割方式：分层随机划分（确保类别比例一致）
  训练集：70%（2413 Poor + 479 Good，保持原始不平衡）
  验证集：15%（约 613 Poor + 103 Good）
  测试集：15%（约 613 Poor + 103 Good）

数据增强（仅训练集）：水平翻转、颜色抖动
验证集和测试集：无增强

重要：v2.2 移除下采样平衡
  原因：保留原始数据分布，用加权损失处理不平衡更科学
  效果：模型见过足够的Poor类样本，加权梯度保证Good类被充分学习
```

### 离线预处理优化（自动启用）

为了加速数据加载，系统支持**自动离线预处理**。确定性的图像处理操作在训练前一次性完成，然后保存为高效 pickle 格式：

**预处理包含的操作：**
- Petri 培养皿检测（Hough 圆检测）
- 中心正方形裁剪
- 图像大小调整到 224×224
- **不包含**数据增强（保持每 epoch 都不同）

**性能提升：**

| 指标 | 原始 | 优化后 | 提升倍数 |
|------|------|--------|---------|
| 单个样本加载时间 | ~10s | ~0.5s | **20x** |
| DataFrame 查询复杂度 | O(n) | O(1) | **数百倍** |
| GPU 利用率 | 30-40% | 80-90% | **2-3x** |

**工作流程：**

```bash
# 首次运行（自动执行预处理）
python main.py train --data_root /liaozizhuo/ --multi_focal ... --two_stage
# → 自动检测预处理数据，若不存在则创建（耗时 10-30 分钟）
# → 预处理完成后开始训练

# 后续运行（使用缓存，速度快）
python main.py train --data_root /liaozizhuo/ --multi_focal ... --two_stage
# → 直接使用 preprocessed_data/ 中的缓存（快速）
```

**预处理数据存储：**

```
ViT-Hybrid/
├── preprocessed_data/              # 自动生成
│   ├── embryo_001_000.pkl          # 胚胎 1，时间点 0
│   ├── embryo_001_001.pkl
│   ...
│   └── embryo_999_032.pkl          # 胚胎 999，时间点 32
```

**磁盘空间：** 约 2-5 GB（取决于数据集大小）

**清理缓存：** `rm -rf preprocessed_data/`（重新预处理时执行）

**相关参数：** `config.py` 中的 `NUM_WORKERS=4` 和 `PIN_MEMORY=True` 配合使用

---

## 四、配置管理

### 配置文件（可选）

多焦距模式运行时默认使用内置配置，无需显式配置。若需自定义，可生成配置文件：

```bash
python main.py generate-config --output my_config.json
python main.py train --data_root /liaozizhuo/ --multi_focal ... --config_file my_config.json
```

### 默认配置模板

```json
{
  "model": {
    "TIME_STEPS": 33,
    "FOCAL_PLANES": 11,
    "IMAGE_SIZE": 224,
    "NUM_TRANSFORMER_LAYERS": 4,
    "NUM_HEADS": 8,
    "D_MODEL": 4096,
    "DFF": 2048,
    "DROPOUT_RATE": 0.1,
    "FUSION_TYPE": "average"
  },
  "training": {
    "BATCH_SIZE": 4,
    "EPOCHS": 100,
    "LEARNING_RATE": 0.0001,
    "WEIGHT_DECAY": 0.00001,
    "TRAIN_RATIO": 0.7,
    "VAL_RATIO": 0.15,
    "EARLY_STOPPING_PATIENCE": 10,
    "REDUCE_LR_PATIENCE": 5,
    "REDUCE_LR_FACTOR": 0.5,
    "MIN_LR": 0.0000001,
    "MONITOR_METRIC": "val_auc",
    "MONITOR_MODE": "max",
    "NUM_WORKERS": 4,
    "PIN_MEMORY": true
  },
  "data": {
    "DATA_ROOT": "/liaozizhuo/",
    "SCORE_DICT_PATH": "/liaozizhuo/score_dict.json",
    "TEST_RATIO": 0.1,
    "APPLY_PETRI_CROP": true,
    "APPLY_CENTER_CROP": true
  },
  "paths": {
    "MODEL_SAVE_DIR": "/liaozizhuo/scripts/ViT-Hybrid/checkpoints",
    "LOG_DIR": "/liaozizhuo/scripts/ViT-Hybrid/logs",
    "RESULT_DIR": "/liaozizhuo/scripts/ViT-Hybrid/results",
    "CACHE_DIR": "/liaozizhuo/scripts/ViT-Hybrid/cache"
  }
}
```

### 配置参数详解

#### 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TIME_STEPS` | 33 | 时间点数量（96-112h，0.5h 间隔） |
| `FOCAL_PLANES` | 11 | 焦平面数量 |
| `IMAGE_SIZE` | 224 | 图像尺寸（VGG16 标准） |
| `NUM_TRANSFORMER_LAYERS` | 4 | Transformer 层数 |
| `NUM_HEADS` | 8 | 注意力头数 |
| `D_MODEL` | 4096 | 特征维度（与 VGG16 输出一致） |
| `DFF` | 2048 | 前馈网络隐层维度 |

#### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 4 | 批次大小（V100: 4-8，A100: 8-16） |
| `EPOCHS` | 100 | 最多训练轮数（早停通常 30-50） |
| `LEARNING_RATE` | 1e-4 | 初始学习率（AdamW） |
| `WEIGHT_DECAY` | 1e-5 | L2 正则化系数 |
| `EARLY_STOPPING_PATIENCE` | 10 | 早停耐心值（epochs） |
| `REDUCE_LR_PATIENCE` | 5 | 学习率调整耐心值 |

#### 数据配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DATA_ROOT` | - | 数据根目录（必需） |
| `SCORE_DICT_PATH` | score_dict.json | 质量标签字典 |
| `APPLY_PETRI_CROP` | true | 启用培养皿检测裁剪 |
| `APPLY_CENTER_CROP` | true | 启用中心正方形裁剪 |

---

## 五、训练过程

### 命令行参数

```bash
python main.py train [选项]

多焦距模式参数（必需）：
  --data_root DIR              数据根目录
  --multi_focal                启用多焦距模式
  --selected_images_csv PATH   时间点映射 CSV（PatientID, EmbryoNumber, FocalPlane, TargetHour, FullPath）
  --maindb_csv PATH            质量标签 CSV（EmbryoID, ICM, TE, Quality）

可选参数：
  --filtered_csv PATH          补充标签 CSV（EmbryoID, EmbryoQuality）
  --two_stage                  启用两阶段训练（推荐）
  --use_focal_loss             使用 Focal Loss（v2.2 后可选，默认加权CrossEntropyLoss）
  --skip_validation            跳过验证，使用缓存（第一次后运行）
  --config_file FILE, -c FILE  自定义配置文件（可选）

v2.2 改进说明：
  - 默认损失函数：加权CrossEntropyLoss（自动计算类别权重）
  - 标签处理：int64格式（CrossEntropyLoss要求）
  - 模型输出：2个logits（而非1个Sigmoid概率）
  - 数据处理：保留原始不平衡分布（不再下采样）
```

### 数据增强

训练集应用以下增强策略，验证集和测试集不应用：

| 增强方式 | 说明 | 参数 |
|--------|------|------|
| **水平翻转** | Horizontal Flip | p=0.5 |
| **颜色抖动** | ColorJitter | brightness=0.1, contrast=0.1 |

**流程：**
1. 读取 PIL 图像
2. 应用增强变换（仅训练集）
3. 转换为 Tensor
4. 使用 ImageNet 标准化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

### 模型架构与微调策略

#### VGG16 特征提取器

**架构组成：**
```
VGG16 (ImageNet 预训练)
├── features (13 个卷积层)
│   ├── Block 1-2 (conv 1-5)  → 64, 128 通道（低阶特征）
│   ├── Block 3-5 (conv 6-13) → 256, 512 通道（高阶特征）
│   └── MaxPool 层
├── AdaptiveAvgPool2d(7, 7)
├── Flatten
└── Classifier (前 4 层)
    ├── fc1: (25088, 4096) + ReLU
    ├── dropout
    └── fc2: (4096, 4096)
    └─ 输出：4096 维特征
```

#### Transformer 编码器

**架构细节：**
```
输入：(batch, 33, 4096)  [33 个时间点，各 4096 维特征]
  ↓
可学习位置编码 (33, 4096)
  ↓
4 × TransformerBlock（Post-Norm）：
  ├─ MultiheadAttention
  │   ├─ embed_dim = 4096
  │   ├─ num_heads = 8
  │   ├─ head_dim = 4096 / 8 = 512
  │   └─ dropout = 0.1
  ├─ FeedForward (4096 → 2048 → 4096)
  ├─ LayerNorm × 2
  └─ Residual Connections
  ↓
输出：(batch, 33, 4096)  [时间序列建模完成]
  ↓
时间注意力池化 ⭐ NEW（Multi-Head Self-Attention）：
  1. Multi-Head Self-Attention (8个头)
  2. 残差连接 + LayerNorm
  3. 计算每个时间步的权重 (FC → GELU → Softmax)
  4. 加权求和 + LayerNorm
  └─ 输出：(batch, 4096)
  ↓
分类头（Pre-Norm + GELU）：
  ├─ LayerNorm(4096)          ← Pre-Norm（参考Vision Transformer）
  ├─ Linear(4096, 512)        ← Xavier(gain=1.0)初始化
  ├─ GELU()                   ← 平滑激活函数
  ├─ Dropout(0.3)
  └─ Linear(512, 2)           ← Xavier(gain=1.0)初始化
  ↓
输出：(batch, 2) → logits（不含激活函数）
  ↓
损失函数：CrossEntropyLoss(weight=[w0, w1])
  - 内部自动应用 softmax
  - w0 = total / (2 × num_poor) ≈ 0.60
  - w1 = total / (2 × num_good) ≈ 3.02
  ↓
预测时：softmax(logits) → 概率分布
```

#### 分类头详细设计（v2.2改进）

**为什么用Pre-Norm + GELU + LayerNorm？**

参考Vision Transformer的最佳实践，改进了分类头的稳定性：

```python
# 原始设计（v2.1）
classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(d_model, 512),
    nn.ReLU(),            # ✗ 梯度陡峭
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

# 改进设计（v2.2）⭐
classifier_norm = nn.LayerNorm(d_model)      # Pre-Norm 稳定输入
classifier_fc1 = nn.Linear(d_model, 512)     # Xavier(gain=1.0)
classifier_act = nn.GELU()                   # ✓ 平滑梯度
classifier_dropout = nn.Dropout(0.3)
classifier_fc2 = nn.Linear(512, 2)           # Xavier(gain=1.0)
```

**关键改进对比：**

| 特性 | v2.1 | v2.2 | 优势 |
|------|------|------|------|
| **归一化** | 无 | Pre-Norm LayerNorm | 稳定特征方差 |
| **激活函数** | ReLU | GELU | 平滑梯度，更易训练 |
| **初始化** | gain=0.1 | gain=1.0 | 权重范围合理，梯度流畅 |
| **Dropout** | 0.5 → 0.3 | 0.3 | 减少信息丧失 |

**为什么Pre-Norm重要？**

```
分类头输入方差很小 (std ~ 0.0001-0.0005)
  ↓
Pre-Norm: LayerNorm(x) 将其缩放为 std ~ 1.0
  ↓
下游FC层得到合理范围的输入，梯度流畅
  ↓
结果：稳定的梯度流，更好的收敛性
```

---

### 训练策略

#### 损失函数

**加权CrossEntropyLoss（v2.2 推荐）** ✅：处理类别不平衡
```
背景：训练集存在3469 Poor + 689 Good的不平衡（5:1）

L_Weighted_CE = -Σ weight[y] × log(softmax(logits)[y])

权重计算：
  total = num_poor + num_good = 4158
  weight_poor = total / (2 × num_poor) = 4158 / (2 × 3469) ≈ 0.60
  weight_good = total / (2 × num_good) = 4158 / (2 × 689) ≈ 3.02

效果：
  - Good类样本的梯度自动放大 5 倍
  - 迫使模型学习识别Good类特征
  - 解决原始BCE方案中Recall=0的问题
```

**Focal Loss（可选）：** 额外关注难分类样本
```
L_Focal = -α·(1-pt)^γ·log(pt)
参数：γ=2.0, α=0.25

优势：
  - 自动降低容易样本的权重
  - 适合严重不平衡（10:1+）
  - 相比加权CE，多获得2-3%性能提升

用法：在train.py中启用--use_focal_loss参数
```

#### 学习率调整

```
策略：ReduceLROnPlateau
监控：Val AUC（最大化）
衰减：factor=0.5，patience=5 epochs，min_lr=1e-7

示例：
  epoch  1-5:  LR = 1e-4
  epoch  6-10: LR = 5e-5   (无改进 5 epochs 后触发)
  epoch 11-15: LR = 2.5e-5 (继续衰减)
```

#### 早停机制

```
监控指标：Val AUC
耐心值：10 epochs

逻辑：
  - 每 epoch 对比验证 AUC 与历史最高值
  - 若无改进：计数器 +1
  - 若计数器 ≥ 10：停止训练，恢复最佳权重
```

#### 两阶段 vs 单阶段训练

```
单阶段（推荐）：
  - VGG16 前 1/3 层冻结 → 后 2/3 层训练
  - Transformer + 分类头 → 训练
  - 统一优化，避免梯度流中断

两阶段（备用）：
  - Stage 1：VGG16 冻结，Transformer + 分类头训练
  - Stage 2：VGG16 解冻，整体微调（学习率 0.1 倍）
```

### 快速训练示例

#### 首次运行（自动验证并缓存）

```bash
python main.py train \
  --data_root /liaozizhuo/ \
  --multi_focal \
  --selected_images_csv /liaozizhuo/selected_image_paths.csv \
  --maindb_csv /liaozizhuo/scripts/MainDB.csv \
  --filtered_csv /liaozizhuo/scripts/filtered_clean_embryos.csv \
  --two_stage
```

#### 后续运行（使用缓存）

```bash
python main.py train \
  --data_root /liaozizhuo/ \
  --multi_focal \
  --selected_images_csv /liaozizhuo/selected_image_paths.csv \
  --maindb_csv /liaozizhuo/scripts/MainDB.csv \
  --filtered_csv /liaozizhuo/scripts/filtered_clean_embryos.csv \
  --two_stage \
  --skip_validation  # 跳过验证，直接使用 ./cache/ 中的缓存
```

---

## 六、评估与输出

### 评估指标

| 指标 | 定义 | 解释 |
|------|------|------|
| **Accuracy** | (TP+TN)/(总数) | 整体分类准确率 |
| **Precision** | TP/(TP+FP) | "正样本预测正确的比例" |
| **Recall** | TP/(TP+FN) | "真实正样本被识别的比例" |
| **F1-Score** | 2·(P·R)/(P+R) | Precision 和 Recall 的调和平均 |
| **AUC** | ROC 曲线下面积 | 不同阈值下的综合性能 |

### 训练输出

训练完成后生成：

```
checkpoints/
├── final_model.pth             # 最终模型权重
├── stage1_best.pth             # 第一阶段最佳模型
└── stage2_best.pth             # 第二阶段最佳模型（两阶段训练）

results/
├── stage1_history.json         # 训练历史（loss, auc, etc）
└── stage2_history.json         # 第二阶段历史

logs/
└── tensorboard/                # TensorBoard 日志

cache/
└── validation_cache_*.json     # 验证缓存（下次可复用）
```

### 模型可解释性分析

理解模型如何利用不同时间点做决策。

#### 方法 1：梯度归因法（快速）

```bash
python analyze_temporal_interpretability.py \
    --model checkpoints/final_model.pth \
    --output results/temporal_analysis \
    --device cuda
```

输出：重要性分数 (33,)

#### 方法 2：注意力权重可视化

```python
from analysis.attention_visualization import extract_attention_weights

attn_weights = extract_attention_weights(model, images, device='cuda')
# 输出：4 个张量，每个 (batch, 8, 33, 33)
```

#### 方法 3：特征相关性分析

```python
from analysis.temporal_interpretability import compute_temporal_correlation

corr_matrix, info = compute_temporal_correlation(model, images, device='cuda')
# 输出：(33, 33) 相关性矩阵
```

#### 方法 4：消融分析（最可信但较慢）

```python
from analysis.ablation_analysis import temporal_ablation_analysis

ablation_scores, _ = temporal_ablation_analysis(model, images, device='cuda')
# 输出：(33,) 消融分数
```

生成的文件：
- `temporal_importance.json` - 梯度归因结果
- `attention_analysis.json` - 注意力统计
- `ablation_analysis.json` - 消融分析结果
- 4 张可视化图表（热力图、流图、条形图、相关性矩阵）

---

## 七、项目结构

```
ViT-Hybrid/
├── main.py                           # 主入口（训练、配置生成）
├── README.md                         # 本文件
├── config/
│   └── config.py                    # 配置类定义
├── model/
│   ├── model.py                     # ViT-Hybrid 模型
│   └── layers.py                    # 自定义层（Fusion、Attention、etc）
├── dataset/
│   ├── dataset.py                   # 数据加载器
│   └── multi_focal_data_loader.py   # 多焦距加载（带缓存）
├── training/
│   └── train.py                     # 训练脚本
├── utils/
│   └── utils.py                     # 工具函数（损失、指标、etc）
├── analysis/                        # 可解释性分析
├── checkpoints/                     # 模型权重保存
├── results/                         # 训练历史与指标
├── logs/                            # TensorBoard 日志
└── cache/                           # 验证缓存（多焦距模式）
```

---

## 八、版本历史

### v2.3 (2025-11) ⭐ 当前版本
**完整的Vision Transformer架构最佳实践**

```
进化路线：
v2.0 (全局平均池化)
  ↓ 改进：丢失时间信息
v2.1 (简单FC注意力)
  ↓ 改进：无法建模时间步关系
v2.2 (加权CrossEntropyLoss)
  ↓ 改进：需要更强大的时间建模
v2.3 ⭐ Multi-Head Self-Attention (完整ViT架构)
```

**核心改进：**

| 组件 | v2.2 | v2.3 | 改进 |
|------|------|------|------|
| **时间池化** | 简单FC (O(33)) | Multi-Head Self-Attn (O(33²)) | 5倍计算复杂度，充分建模时间关系 |
| **注意力头数** | - | 8个独立头 | 多角度时间模式学习 |
| **分类头** | 无归一化 | Pre-Norm LayerNorm | std: 0.0001 → 1.0，梯度稳定 |
| **激活函数** | ReLU | GELU | 平滑梯度流 |
| **权重初始化** | gain=0.1 | gain=1.0 | 合理梯度范围 |
| **损失函数** | 加权CrossEntropy | 加权CrossEntropy | Good类权重×3 |

**性能指标：**
```
指标              预期改进
─────────────────────────
时间建模能力      显著提升（非线性关系）
梯度稳定性        大幅优化（Pre-Norm+GELU）
特征方差          稳定在std=1.0
收敛速度          加快（更稳定的梯度）
可解释性          强（8层Transformer注意力 + 33维时间权重）
```

---

### 快速对比：ViT-Hybrid v2.3 架构优势

**时间序列建模方式对比：**

```
v2.0: Mean Pooling              v2.1: FC Attention            v2.3: Multi-Head Self-Attention
────────────────────────────────────────────────────────────────────────────────
  33步 → mean               33步 → FC → FC → softmax      33步 → 8头Self-Attn → FC
        ↓                                ↓                            ↓
   单值特征                    权重分布                   充分的时间关系
   (丧失时间)              (线性映射)              (非线性 O(33²) 建模)
```

**架构完整流程：**

```
输入 (B, 33, 11, 224, 224, 3)
  ↓ VGG16 提取空间特征
输出 (B, 33, 11, 4096)
  ↓ 焦平面融合 (11→1)
输出 (B, 33, 4096)
  ↓ Transformer 编码器 (4层)
输出 (B, 33, 4096)
  ↓ ⭐ 时间注意力池化 (Multi-Head Self-Attention, 8头)
    ├─ Self-Attention建模时间步关系
    ├─ 残差连接 + LayerNorm稳定梯度
    ├─ FC层学习权重：4096→2048→1
    └─ 加权求和 (B, 4096)
  ↓ 分类头 (Pre-Norm + GELU)
    ├─ LayerNorm稳定输入 (std: 0.0001→1.0)
    ├─ FC(4096→512) + GELU
    └─ FC(512→2) → logits (B, 2)
  ↓ 损失函数：CrossEntropyLoss(weight=[0.60, 3.02])
     Good类权重3倍，强制学习少数类
```

**核心优势：**
- ✅ **完整时间建模**：O(33²) 复杂度，非线性捕捉时间依赖
- ✅ **梯度稳定流动**：Pre-Norm + GELU + LayerNorm
- ✅ **特征方差稳定**：输入分类头std从0.0001提升到1.0
- ✅ **可解释的决策**：8层Transformer注意力 + 33维时间权重
- ✅ **类别不平衡处理**：加权CrossEntropyLoss (Good×3)
