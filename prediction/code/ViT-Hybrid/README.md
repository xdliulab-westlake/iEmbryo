# 🧠 ViT-Hybrid 模型架构文档（多焦平面融合版）

**任务目标：** 胚胎好/坏二分类  
**模型类型：** ViT-Hybrid（VGG16 + Transformer）  
**版本：** v2.0（多焦平面融合）  
**设计者：** Liao Zizhuo（2025）

---

## 一、设计动机

胚胎在 0–112 小时的发育过程中，其形态变化呈现明显的时间连续性；而在每个时间点，不同焦平面（focal depths）共同记录了胚胎的三维形态特征。为了在有限样本下充分利用这些时空信息，本模型结合卷积神经网络（CNN）与 Transformer 的优势：使用 **VGG16** 提取空间特征，并通过 **Transformer** 建模时间维度变化，同时对所有焦平面进行**多平面信息融合**（而非单平面选择），以最大限度保留胚胎结构的整体信息，构建时空联合特征表达，实现胚胎好坏分类。

---

## 二、数据结构

| 维度 | 含义 |
|------|------|
| `time_steps = 33` | 时间点（0–112 小时，每 0.5h 采样一次） |
| `focal_planes = 11` | 每个时间点的焦平面数量 |
| `image_shape = (224, 224, 3)` | 单张图像的尺寸 |
| **输入张量** | `(batch_size, 33, 11, 224, 224, 3)` |
| **输出标签** | 二分类：好胚胎 (1) / 坏胚胎 (0) |

---

## 三、模型结构概览
输入：(33 × 11 × 224 × 224 × 3)
│
├── VGG16（共享权重）
│       ↓
│   每张图像 → 512维特征向量
│
├── 多焦平面融合（Focal Fusion）
│       ↓
│   将11个焦平面平均融合 → 每个时间点输出1个512维特征
│
├── Transformer 编码器（Temporal Transformer）
│       ↓
│   建模0–112小时的时序变化 → 输出33×512时序特征
│
├── 全局平均池化（Global Average Pooling）
│
├── 全连接层（Dense 256, ReLU 激活） + Dropout(0.3)
│
└── 输出层（Dense 1, Sigmoid 激活） → 胚胎好坏概率

---

## 四、模块说明

### 1️⃣ VGG16 卷积特征提取模块  
- 使用预训练的 **VGG16（ImageNet 权重）**，移除顶层全连接部分，仅保留卷积主体和全局平均池化层。  
- 所有时间点与焦平面图像共用同一权重，实现高效共享特征提取。  
- 输出每张图像 512 维特征向量，为后续时序建模提供高层语义特征。  
- 训练初期冻结 VGG16 参数，待 Transformer 收敛后再解冻高层卷积模块进行联合微调。

### 2️⃣ 多焦平面融合模块（Focal Fusion）  
- 输入：单一时间点的 11 张焦平面特征 `(11, 512)`  
- 采用**平均融合策略**对所有焦平面特征进行聚合，确保不同深度信息均被利用。  
- 输出：单个时间点的 512 维特征向量 `(1, 512)`  
- 此操作在保持整体结构信息的同时避免注意力偏置，提高特征稳定性。

### 3️⃣ Transformer 编码器（Temporal Transformer）  
- 输入：33×512 的时间序列特征矩阵。  
- 添加可学习的位置编码，显式建模时间顺序。  
- 包含 4 层 Transformer 块，每层由多头自注意力（8 heads）、前馈网络（1024 单元）、Layer Normalization 和残差连接组成。  
- 该模块学习胚胎在发育过程中形态变化的全局动态模式。

### 4️⃣ 分类头（Classification Head）  
- 将 Transformer 输出经全局平均池化（GlobalAveragePooling1D）得到整体时序特征向量。  
- 通过两层全连接网络进行非线性映射：  
  - Dense(256, ReLU 激活) + Dropout(0.3)  
  - Dense(1, Sigmoid 激活)  
- 最终输出值 `p ∈ [0,1]` 表示胚胎为“好”的概率。

---

## 五、训练与优化配置

| 项目 | 设置 |
|------|------|
| **损失函数 (Loss Function)** | Binary Cross-Entropy；若类别不平衡则使用 Focal Loss (γ=2.0, α=0.25) |
| **优化器 (Optimizer)** | AdamW（learning_rate = 1e-4，weight_decay = 1e-5，β₁=0.9，β₂=0.999，clipnorm = 1.0） |
| **评价指标 (Metrics)** | Accuracy、AUC、Precision、Recall、F1-score |
| **学习率策略 (Scheduler)** | Cosine Decay 或 ReduceLROnPlateau |
| **早停策略 (EarlyStopping)** | monitor='val_auc'，patience=10 |
| **Batch Size** | 4–8（取决于 GPU 显存） |
| **数据增强 (Augmentation)** | 随机翻转、旋转、亮度对比调整、轻微缩放与裁剪 |
| **训练策略** | 采用两阶段训练：① 冻结 VGG16，仅训练 Transformer + 分类头；② 解冻 VGG16 高层卷积层联合微调 |

---

## 六、模型优势与特性

| 模块 | 功能 | 说明 |
|------|------|------|
| **VGG16** | 提取空间结构特征 | 捕捉胚胎形态学特征与细胞结构信息 |
| **多焦平面融合** | 保留所有焦平面信息 | 综合不同深度层的三维信息，提高模型稳健性 |
| **Transformer** | 捕捉发育时间动态 | 建模胚胎发育过程中的连续变化模式 |
| **分类头** | 输出胚胎质量概率 | 可直接用于质量预测与阈值判定 |

---

## 七、架构总结

ViT-Hybrid（多焦平面融合版）是一种结合卷积与注意力机制的时空建模框架，通过 VGG16 捕捉空间语义特征、焦平面融合提取深度结构信息、Transformer 编码发育时序规律，最终实现胚胎好坏的二分类判定。模型在结构上兼顾了卷积网络的稳健性与 Transformer 的全局建模能力，适用于小样本、高时序分辨率的胚胎发育分析任务。在训练策略与优化配置上，结合 AdamW 优化器、BCE/Focal Loss、AUC 早停与轻量数据增强，能够实现高稳定性、高可解释性与高准确度的胚胎质量预测。

---

## 八、代码实现（PyTorch）

### 项目结构

```
ViT-Hybrid/
├── README.md              # 本文档
├── layers.py              # 自定义层模块
├── model.py               # 模型架构定义
├── dataset.py             # 数据加载器
├── config.py              # 配置文件
├── utils.py               # 工具函数
├── train.py               # 训练脚本
├── checkpoints/           # 模型保存目录
├── logs/                  # 训练日志
└── results/               # 结果输出
```

### 环境依赖

```bash
pip install torch torchvision
pip install numpy pillow
pip install scikit-learn
```

### 数据组织格式

```
data_root/
├── good/
│   ├── embryo_001/
│   │   ├── time_000/
│   │   │   ├── focal_00.jpg
│   │   │   ├── focal_01.jpg
│   │   │   └── ...
│   │   ├── time_001/
│   │   └── ...
│   └── embryo_002/
└── poor/
    └── embryo_003/
```

### 快速开始

#### 1. 修改配置

编辑 `config.py`：

```python
class DataConfig:
    DATA_ROOT = '/path/to/your/embryo/data'  # 修改为实际数据路径
```

#### 2. 单阶段训练

```bash
python train.py --data_root /path/to/data
```

#### 3. 两阶段训练（推荐）

```bash
python train.py --data_root /path/to/data --two_stage
```

#### 4. 使用Focal Loss（适用于类别不平衡）

```bash
python train.py --data_root /path/to/data --two_stage --use_focal_loss
```

### 模型使用示例

#### 训练模型

```python
from model import create_model
from config import get_model_config
import torch

# 创建模型
config = get_model_config()
model = create_model(config)

# 查看模型结构
print(model)

# 测试输入输出
batch_size = 2
dummy_input = torch.randn(batch_size, 33, 11, 3, 224, 224)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # (2, 1)
```

#### 加载预训练模型

```python
import torch
from model import create_model

model = create_model()
model.load_state_dict(torch.load('checkpoints/final_model.pth'))
model.eval()

# 进行预测
with torch.no_grad():
    output = model(input_data)
    prediction = (output > 0.5).float()
```

### 配置参数说明

#### 模型配置（config.py - ModelConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TIME_STEPS` | 33 | 时间点数量 |
| `FOCAL_PLANES` | 11 | 焦平面数量 |
| `IMAGE_SIZE` | 224 | 图像尺寸 |
| `NUM_TRANSFORMER_LAYERS` | 4 | Transformer层数 |
| `NUM_HEADS` | 8 | 注意力头数 |
| `D_MODEL` | 512 | 特征维度 |
| `DFF` | 1024 | 前馈网络维度 |
| `DROPOUT_RATE` | 0.1 | Dropout率 |

#### 训练配置（config.py - TrainingConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 4 | 批次大小 |
| `EPOCHS` | 100 | 训练轮数 |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `WEIGHT_DECAY` | 1e-5 | 权重衰减 |
| `EARLY_STOPPING_PATIENCE` | 10 | 早停耐心值 |
| `REDUCE_LR_PATIENCE` | 5 | 学习率调整耐心值 |

### 训练输出

训练过程会生成以下文件：

- `checkpoints/stage1_best.pth` - 第一阶段最佳模型
- `checkpoints/stage2_best.pth` - 第二阶段最佳模型（两阶段训练）
- `checkpoints/final_model.pth` - 最终模型
- `results/stage1_history.json` - 第一阶段训练历史
- `results/stage2_history.json` - 第二阶段训练历史（两阶段训练）
- `logs/` - TensorBoard日志

### 模型评估指标

训练过程中会输出以下指标：

- **Accuracy**: 分类准确率
- **AUC**: ROC曲线下面积
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数

### GPU加速

代码自动检测GPU可用性：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

如需指定GPU：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0号GPU
```

### 图像预处理

数据加载器自动进行以下预处理操作：

1. **培养皿检测与裁剪** (Petri Dish Crop)
   - 使用Hough圆检测算法自动检测培养皿边界
   - 通过多参数组合确保高检测率(>96%)
   - 自动生成圆形掩膜并应用到图像

2. **中心正方形裁剪** (Center Square Crop)
   - 从图像中心裁剪正方形区域
   - 移除图像边界的无关信息

3. **尺寸调整**
   - 统一调整为224×224分辨率

4. **数据增强** (训练集)
   - 随机水平翻转
   - 随机亮度调整
   - 随机对比度调整

可在config.py中控制预处理选项：

```python
APPLY_PETRI_CROP = True    # 启用培养皿裁剪
APPLY_CENTER_CROP = True   # 启用中心裁剪
```

### 注意事项

1. 数据路径必须按照指定格式组织
2. 图像命名格式：`time_XXX/focal_XX.jpg`
3. 推荐使用两阶段训练策略
4. 类别不平衡时建议使用Focal Loss
5. 根据GPU显存调整batch_size
6. 训练前确保有足够的磁盘空间保存检查点
7. 图像预处理会自动进行，无需手动处理