# Multi-Task TMFI-Net: 胚胎质量评估模型

**完全基于TMFI-Net的原始架构和损失函数进行多任务学习的胚胎质量评估系统**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [核心特点](#核心特点)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [环境配置](#环境配置)
  - [数据准备](#数据准备)
  - [训练模型](#训练模型)
  - [推理示例](#推理示例)
- [配置说明](#配置说明)
- [技术细节](#技术细节)
- [参考文献](#参考文献)

---

## 🎯 核心特点

### ✅ 100% 保留TMFI-Net原始架构

- **Video Swin Transformer** - 3D时空特征提取
- **Gate模块** - 多维度注意力机制（通道、时间、空间）
- **DecoderConvUp** - FPN结构 + 层次解码器
- **所有卷积配置和上采样策略** - 与TMFI-Net完全一致

### ✅ 完全复用TMFI-Net损失函数

- **KL Divergence Loss** - 分布对齐
- **Correlation Coefficient Loss** - 相关性约束
- **Similarity Loss** - 相似度度量
- **组合损失函数** - 可灵活配置权重

### ✅ 多任务学习框架

- **主任务**: 胚胎质量二分类 (Good/Poor)
- **辅助任务**: Saliency prediction (TMFI-Net原始输出)
- **联合优化**: 支持端到端训练或单独训练

### ✅ 多焦距数据处理

- 支持 **average/max/middle/learned** 四种焦平面融合策略
- 自动将多焦距数据转换为Video Swin Transformer输入格式
- 高效的数据预处理和缓存机制

---

## 🏗️ 模型架构

### 整体架构图

```
多焦距胚胎数据 (B, T=33, F=11, C=3, H=224, W=224)
    ↓
┌───────────────────────────────────────────┐
│  Focal Plane Fusion                       │
│  (average/max/middle/learned)             │
└───────────────────────────────────────────┘
    ↓
Video输入 (B, C=3, T=33, H=224, W=224)
    ↓
┌───────────────────────────────────────────┐
│  Video Swin Transformer (TMFI-Net)        │
│  - 3D Spatiotemporal Feature Extraction   │
│  - Multi-level features [y1, y2, y3, y4]  │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│  DecoderConvUp (TMFI-Net)                 │
│  - FPN (Semantic-Guided Encoder)          │
│  - Gate Modules (Multi-Dim Attention)     │
│  - Hierarchical Decoder                   │
└───────────────────────────────────────────┘
    ↓
    ├─→ Saliency Map (B, H, W)
    │   └─→ TMFI-Net Saliency Loss (KLDiv + CC + Sim)
    │
    └─→ Features (y1, y2, y3, y4)
        └─→ Global Pool → Classifier → Logits (B, 2)
            └─→ CrossEntropyLoss
```

### 核心组件

#### 1. Backbone - Video Swin Transformer

```python
# TMFI-Net原始backbone
self.backbone = SwinTransformer3D(
    pretrained=pretrain,
    patch_size=(2,4,4),
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=(8,7,7)
)
```

#### 2. Decoder - DecoderConvUp

```python
# TMFI-Net原始decoder，包含：
# - Gate模块 (Multi-Dimensional Attention)
# - FPN结构 (Feature Pyramid Network)
# - 层次解码器 (Hierarchical Decoder)
self.decoder = DecoderConvUp()

# 输出:
# - saliency_map: (B, H, W) - TMFI-Net原始输出
# - features: (y1, y2, y3, y4) - 用于分类的中间特征
```

#### 3. Classification Head

```python
# 新增：分类头（唯一的新组件）
self.classifier = nn.Sequential(
    nn.Linear(192 * 4, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
```

### 输入输出

| 项目 | 形状 | 说明 |
|------|------|------|
| **输入** | (B, T=33, F=11, C=3, H=224, W=224) | 多焦距时序胚胎图像 |
| **输出1** | (B, H, W) | Saliency Map - 显示模型关注区域 |
| **输出2** | (B, 2) | Logits - Good/Poor分类结果 |

---

## 📁 项目结构

```
ViT-Hybrid/
├── README.md                       # 本文档
├── FINAL_SUMMARY.md               # 技术实现总结
│
├── model/                          # 模型定义
│   ├── tmfi_embryo.py             # Multi-Task TMFI-Net主模型
│   │   ├── Gate                   # TMFI-Net Gate模块
│   │   ├── DecoderConvUp          # TMFI-Net Decoder
│   │   └── EmbryoQualityModel     # 多任务模型
│   ├── swin_transformer.py        # Video Swin Transformer
│   └── __init__.py
│
├── training/                       # 训练脚本
│   ├── train_tmfi.py              # 主训练脚本
│   └── __init__.py
│
├── dataset/                        # 数据加载
│   ├── dataset.py                 # 数据集类和预处理
│   ├── multi_focal_data_loader.py # 多焦距数据加载器
│   ├── focal_fusion.py            # 焦平面融合
│   └── __init__.py
│
├── utils/                          # 工具函数
│   ├── saliency_losses.py         # TMFI-Net损失函数
│   │   ├── kldiv_loss()
│   │   ├── cc_loss()
│   │   ├── similarity_loss()
│   │   └── get_tmfi_loss()        # 组合损失
│   ├── utils.py                   # 训练工具
│   │   ├── AverageMeter           # 指标追踪
│   │   ├── EarlyStopping
│   │   └── calculate_metrics()
│   └── __init__.py
│
├── config/                         # 配置文件
│   ├── config.py                  # 模型和训练配置
│   └── __init__.py
│
├── optimization/                   # 优化工具
│   └── pytorch_optimization.py    # PyTorch性能优化
│
├── generate_score_dict.py         # 数据预处理工具
│
├── checkpoints/                    # 模型权重（训练后生成）
│   ├── tmfi_embryo_best.pth
│   └── tmfi_embryo_final.pth
│
└── results/                        # 训练结果（训练后生成）
    ├── tmfi_embryo_history.json
    └── attention_maps/
        └── attention_epoch_*.pt
```

---

## 🚀 快速开始

### 环境配置

#### 1. Python环境

```bash
# 创建虚拟环境（推荐）
conda create -n tmfi python=3.8
conda activate tmfi

# 或使用 venv
python -m venv tmfi_env
source tmfi_env/bin/activate  # Linux/Mac
# tmfi_env\Scripts\activate  # Windows
```

#### 2. 安装依赖

```bash
# PyTorch (根据CUDA版本选择)
# CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 或 CPU版本
pip install torch torchvision

# 其他依赖
pip install pandas numpy scikit-learn tqdm opencv-python pillow
```

### 数据准备

#### 1. 数据格式要求

项目需要以下数据文件：

**MainDB.csv** - 胚胎质量标签
```csv
EmbryoID,ICM,TE,Quality
E001,A,A,Good
E002,B,C,Poor
...
```

**selected_image_paths.csv** - 多焦距图像路径
```csv
EmbryoID,Timepoint,FocalPlane,ImagePath
E001,0,0,/path/to/E001_t0_f0.jpg
E001,0,1,/path/to/E001_t0_f1.jpg
...
```

#### 2. 生成评分字典（可选）

```bash
python generate_score_dict.py \
    --maindb /path/to/MainDB.csv \
    --output score_dict.json \
    --image_dir /path/to/images
```

### 训练模型

#### 基本训练

```bash
python -m training.train_tmfi \
    --maindb /path/to/MainDB.csv \
    --selected_images /path/to/selected_image_paths.csv \
    --fusion_type average \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4
```

#### 完整参数说明

```bash
python -m training.train_tmfi \
    --maindb /path/to/MainDB.csv \              # 必需：标签文件
    --selected_images /path/to/paths.csv \      # 必需：图像路径
    --filtered_clean /path/to/filtered.csv \    # 可选：额外标签
    --fusion_type average \                     # 焦平面融合策略
    --epochs 100 \                              # 训练轮数
    --batch_size 4 \                            # 批次大小
    --lr 1e-4 \                                 # 学习率
    --num_workers 4 \                           # 数据加载线程
    --use_saliency_loss False \                 # 是否使用saliency loss
    --saliency_weight 0.1                       # saliency loss权重
```

#### 焦平面融合策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `average` | 平均所有焦平面 | 默认，特征平滑 |
| `max` | 最大池化 | 突出显著特征 |
| `middle` | 选择中间焦平面(F=5) | 快速推理 |
| `learned` | 可学习权重融合 | 自动优化 |

### 推理示例

#### Python代码

```python
import torch
from model import create_tmfi_embryo_model, FocalFusionWrapper

# 1. 创建模型
config = {
    'num_classes': 2,
    'pretrained_path': None  # 可选：预训练权重
}

tmfi_model = create_tmfi_embryo_model(config)
model = FocalFusionWrapper(
    tmfi_model=tmfi_model,
    fusion_type='average',
    num_focal_planes=11
)

# 2. 加载训练好的权重
model.load_state_dict(torch.load('checkpoints/tmfi_embryo_best.pth'))
model.eval()
model.to('cuda')

# 3. 准备数据
# data: (B, T=33, F=11, C=3, H=224, W=224)
data = torch.randn(1, 33, 11, 3, 224, 224).cuda()

# 4. 推理
with torch.no_grad():
    saliency_map, logits = model(data)

    # 分类结果
    probs = torch.softmax(logits, dim=1)  # (1, 2)
    prediction = torch.argmax(logits, dim=1)  # 0=Poor, 1=Good

    print(f"Prediction: {'Good' if prediction.item() == 1 else 'Poor'}")
    print(f"Confidence: {probs[0, prediction].item():.4f}")

    # Saliency map
    # saliency_map: (1, 224, 224) - 显示模型关注的空间位置
```

#### 批量预测

```python
from dataset import create_multi_focal_data_loaders

# 创建数据加载器
_, _, test_loader = create_multi_focal_data_loaders(
    multi_focal_df=test_df,
    embryo_to_label=test_labels,
    config=data_config
)

# 批量预测
predictions = []
confidences = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        saliency_map, logits = model(inputs)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        confidences.extend(probs[:, 1].cpu().numpy())

# 保存结果
results_df = pd.DataFrame({
    'EmbryoID': test_ids,
    'Prediction': predictions,
    'Confidence': confidences
})
results_df.to_csv('predictions.csv', index=False)
```

---

## ⚙️ 配置说明

### config/config.py

```python
class ModelConfig:
    """模型配置"""
    TIME_STEPS = 33              # 时间点数量 (96-112h, 0.5h间隔)
    FOCAL_PLANES = 11            # 焦平面数量
    IMAGE_SIZE = 224             # 图像大小
    NUM_CLASSES = 2              # 分类数量 (Good/Poor)
    FUSION_TYPE = 'average'      # 焦平面融合策略

    # Video Swin Transformer配置
    PATCH_SIZE = (2, 4, 4)
    EMBED_DIM = 96
    DEPTHS = [2, 2, 6, 2]
    NUM_HEADS = [3, 6, 12, 24]
    WINDOW_SIZE = (8, 7, 7)

class TrainingConfig:
    """训练配置"""
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # 早停配置
    EARLY_STOPPING_PATIENCE = 10

    # 学习率调度
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7

    # Saliency loss（当有GT saliency map时）
    USE_SALIENCY_LOSS = False
    SALIENCY_WEIGHT = 0.1

    # 损失函数权重（TMFI-Net配置）
    KLDIV_COEFF = 1.0
    CC_COEFF = -1.0        # 负值表示最大化相关性
    SIM_COEFF = -1.0       # 负值表示最大化相似度

class DataConfig:
    """数据配置"""
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    NUM_WORKERS = 4
    PIN_MEMORY = True

    # 数据增强
    USE_AUGMENTATION = True
    ROTATION_RANGE = 10
    ZOOM_RANGE = 0.1
```

---

## 🔬 技术细节

### 1. TMFI-Net架构复用

本项目**100%保留**了TMFI-Net的所有核心组件：

#### Gate模块 (Multi-Dimensional Attention)

```python
class Gate(nn.Module):
    """
    TMFI-Net原始Gate模块
    多维度注意力：通道 × 时间 × 空间
    """
    def __init__(self, in_plane):
        super(Gate, self).__init__()
        self.gate = nn.Conv3d(
            in_plane, in_plane,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=in_plane,
            bias=False
        )

    def forward(self, x):
        return x * torch.sigmoid(self.gate(x))
```

#### DecoderConvUp (FPN + Hierarchical Decoder)

```python
class DecoderConvUp(nn.Module):
    """
    TMFI-Net原始Decoder
    - FPN (Feature Pyramid Network)
    - Gate modules
    - Hierarchical decoder
    """
    def forward(self, y4, y3, y2, y1):
        # ... TMFI-Net原始实现 ...

        # 输出saliency map和中间特征
        saliency_map = z0.view(z0.size(0), z0.size(3), z0.size(4))
        return saliency_map, (y1, y2, y3, y4)
```

### 2. 损失函数

#### TMFI-Net原始损失函数

所有损失函数从`TMFI-Net/loss.py`和`TMFI-Net/utils1.py`完全复制：

```python
# KL Divergence Loss
def kldiv_loss(pred_map, gt_map):
    """分布对齐损失"""
    # 完全复制自TMFI-Net
    ...

# Correlation Coefficient Loss
def cc_loss(pred_map, gt_map):
    """相关性损失（最大化）"""
    # 完全复制自TMFI-Net
    ...

# Similarity Loss
def similarity_loss(pred_map, gt_map):
    """相似度损失（最大化）"""
    # 完全复制自TMFI-Net
    ...

# TMFI-Net组合损失
def get_tmfi_loss(pred_map, gt_map,
                  use_kldiv=True, use_cc=True,
                  kldiv_coeff=1.0, cc_coeff=-1.0):
    """TMFI-Net原始组合损失"""
    loss = 0.0
    if use_kldiv:
        loss += kldiv_coeff * kldiv_loss(pred_map, gt_map)
    if use_cc:
        loss += cc_coeff * cc_loss(pred_map, gt_map)
    return loss
```

#### 多任务学习损失

```python
# 当前配置（无GT saliency map）
total_loss = CrossEntropyLoss(logits, labels)

# 未来配置（有GT saliency map）
cls_loss = CrossEntropyLoss(logits, labels)
sal_loss = get_tmfi_loss(saliency_map, gt_saliency)
total_loss = cls_loss + 0.1 * sal_loss
```

### 3. 焦平面融合

#### FocalPlaneFusion模块

```python
class FocalPlaneFusion(nn.Module):
    """
    将多焦距数据转换为Video Swin输入格式
    """
    def forward(self, x):
        # x: (B, T=33, F=11, C=3, H=224, W=224)

        if self.fusion_type == 'average':
            x_fused = x.mean(dim=2)  # (B, T, C, H, W)
        elif self.fusion_type == 'max':
            x_fused = x.max(dim=2)[0]
        elif self.fusion_type == 'middle':
            x_fused = x[:, :, 5, :, :, :]  # 选择F=5
        elif self.fusion_type == 'learned':
            weights = torch.softmax(self.focal_weights, dim=0)
            x_fused = (x * weights.view(1,1,F,1,1,1)).sum(dim=2)

        # 转换为Video Swin格式
        x_fused = x_fused.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return x_fused
```

### 4. 性能优化

#### 数据加载优化

```python
# 1. 离线预处理
preprocessor = OfflineImagePreprocessor()
preprocessor.preprocess_all_images(image_paths, output_dir)

# 2. 缓存验证结果
validator = MultiFocalDataValidator(use_cache=True)

# 3. 平衡采样
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
```

#### 训练优化

```python
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 学习率调度
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# 3. 早停
early_stopping = EarlyStopping(patience=10, mode='max')
```

### 5. 训练输出

#### 模型文件

```
checkpoints/
├── tmfi_embryo_best.pth      # 验证集最佳模型
└── tmfi_embryo_final.pth     # 最终模型
```

#### 训练历史

```json
// results/tmfi_embryo_history.json
{
  "train_loss": [0.693, 0.612, ...],
  "val_loss": [0.650, 0.587, ...],
  "val_auc": [0.65, 0.72, ...],
  "best_epoch": 45,
  "best_auc": 0.857
}
```

#### 注意力图

```
results/attention_maps/
├── attention_epoch_009.pt    # 每10个epoch保存
├── attention_epoch_019.pt
└── attention_epoch_099.pt
```

---

## 📊 性能指标

### 评估指标

训练过程中自动计算以下指标：

| 指标 | 说明 |
|------|------|
| **AUC-ROC** | 接受者操作特征曲线下面积（主要指标） |
| **Accuracy** | 准确率 |
| **Precision** | 精确率 |
| **Recall** | 召回率 |
| **F1 Score** | F1分数 |
| **Confusion Matrix** | 混淆矩阵 |

### 查看训练历史

```python
import json
import matplotlib.pyplot as plt

# 加载训练历史
with open('results/tmfi_embryo_history.json', 'r') as f:
    history = json.load(f)

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_auc'])
plt.title('Validation AUC')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## 🐛 常见问题

### 1. CUDA out of memory

**解决方案**:
- 减小batch size: `--batch_size 2`
- 减少num_workers: `--num_workers 2`
- 使用梯度累积

### 2. 训练loss不下降

**检查**:
- 数据是否正确加载
- 学习率是否合适（尝试`--lr 1e-3`）
- 是否存在数据不平衡（查看类别分布）

### 3. 数据加载慢

**优化**:
- 使用离线预处理: `enable_preprocessing=True`
- 增加num_workers: `--num_workers 8`
- 启用缓存: `use_cache=True`

---

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{liao2025multitask,
  title={Multi-Task TMFI-Net for Embryo Quality Assessment},
  author={Liao, Zizhuo},
  journal={GitHub},
  year={2025},
  url={https://github.com/yourusername/ViT-Hybrid}
}
```

### 相关论文

1. **TMFI-Net**: Transformer-Based Multi-Scale Feature Integration Network for Video Saliency Prediction (IEEE TCSVT 2023)
2. **Video Swin Transformer**: Liu et al., CVPR 2022

---

## 📄 License

本项目基于 MIT License 开源。详见 [LICENSE](LICENSE) 文件。

---

## 👥 联系方式

- **作者**: Liao Zizhuo
- **项目**: Multi-Task TMFI-Net
- **版本**: 1.0.0
- **日期**: 2025-12-22

---

## 🙏 致谢

- 感谢 TMFI-Net 原作者提供的优秀架构
- 感谢 Video Swin Transformer 团队的开源贡献
- 感谢所有贡献者和使用者

---

**注**: 更多技术实现细节请参考 [FINAL_SUMMARY.md](FINAL_SUMMARY.md)
