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
- 使用预训练的 **VGG16（ImageNet 权重，STORK风格实现）**：该模型在初次运行时自动从 PyTorch torchvision 库下载。无需手动提供预训练权重。
- 架构遵循STORK标准：`features + avgpool(7x7) + flatten + classifier[:4]`
- 输出每张图像 **4096 维**特征向量（来自原始VGG16分类器的FC层）
- 所有时间点与焦平面图像共用同一权重，实现高效共享特征提取。
- **微调策略采用STORK最优方案**：冻结前1/3层，解冻后2/3层进行联合训练（无需两阶段训练）

### 2️⃣ 多焦平面融合模块（Focal Fusion）
- 输入：单一时间点的 11 张焦平面特征 `(11, 4096)`
- 采用**平均融合策略**对所有焦平面特征进行聚合，确保不同深度信息均被利用。
- 输出：单个时间点的 **4096 维**特征向量 `(1, 4096)`
- 此操作在保持整体结构信息的同时避免注意力偏置，提高特征稳定性。

### 3️⃣ Transformer 编码器（Temporal Transformer）
- 输入：33×**4096** 的时间序列特征矩阵。
- 添加可学习的位置编码，显式建模时间顺序。
- 包含 4 层 Transformer 块，每层由多头自注意力（8 heads × 512维）、前馈网络（2048 单元）、Layer Normalization 和残差连接组成。
- 该模块学习胚胎在发育过程中形态变化的全局动态模式。

### 4️⃣ 分类头（Classification Head）
- 将 Transformer 输出经全局平均池化（GlobalAveragePooling1D）得到整体时序特征向量 (4096D)。
- 通过三层全连接网络进行非线性映射：
  - Dropout(0.5) + Linear(4096, 512) + ReLU + Dropout(0.3)
  - Linear(512, 1) + Sigmoid
- 最终输出值 `p ∈ [0,1]` 表示胚胎为"好"的概率。

---


## 五、完整的预处理到评估流程

### 📋 数据预处理流程

#### 5.1 图像加载与标准化

**输入:** 原始胚胎显微镜图像 (任意分辨率，RGB格式)

**处理步骤:**

1. **读取图像**
   - 使用 OpenCV (cv2.imread) 或 PIL 加载
   - 转换为 RGB 颜色空间 (OpenCV默认为BGR)

2. **培养皿检测与裁剪** (可选，默认启用)
   - 使用 **Hough 圆检测** 找到圆形培养皿边界
   - 参数范围: 半径 15%-45% 图像对角线
   - 遍历多组参数组合确保鲁棒性: (60,40), (70,45), (50,35), (80,50), (90,55)
   - 取检测到的最大圆形作为培养皿

3. **中心正方形裁剪** (可选，默认启用)
   - 从裁剪后的图像提取中心正方形区域
   - 目的: 进一步标准化输入

4. **调整尺寸到 224×224**
   - 使用双线性插值 (cv2.INTER_LINEAR)
   - 这是 VGG16 的标准输入尺寸

5. **转换为张量并归一化**
   ```python
   # ImageNet 标准化参数
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]

   # 处理流程
   img_tensor = ToTensor()(img_ndarray)  # [0, 1] 范围
   img_normalized = Normalize(mean, std)(img_tensor)  # ImageNet 标准化
   ```

   这些是 **ImageNet 数据集的官方统计参数**，与 VGG16 预训练时使用的标准化一致

#### 5.2 数据增强 (仅用于训练集)

**增强策略:**

| 增强方法 | 参数 | 用途 |
|---------|------|------|
| **随机水平翻转** | p=0.5 | 提高空间不变性 |
| **颜色抖动** | brightness=0.1, contrast=0.1 | 增强光照鲁棒性 |

**验证集与测试集:** 无数据增强，仅进行标准化处理

### 📊 数据划分

```
原始数据集
├── 训练集:   70% (用数据增强)
├── 验证集:   15% (用于早停与学习率调整)
└── 测试集:   15% (最终性能评估)
```

**划分策略:** 分层随机划分 (stratified split)
- 确保各类别 (好/差胚胎) 在三个集合中比例一致
- 使用 sklearn 的 train_test_split with stratify 参数

---

## 六、训练过程详解

### 6.1 训练配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| **批次大小** | 4 | batch_size (取决于GPU显存) |
| **总轮数** | 100 | 最多训练轮数 |
| **学习率** | 1e-4 | 初始学习率 |
| **权重衰减** | 1e-5 | L2 正则化系数 |
| **损失函数** | BCE Loss 或 Focal Loss | 二分类交叉熵或焦点损失 |

### 6.2 损失函数

**Binary Cross-Entropy (BCE) Loss:**
```
L_BCE = -[y·log(p) + (1-y)·log(1-p)]
```
- 默认使用
- 适用于样本较平衡的情况

**Focal Loss** (可选):
```
L_Focal = -α·(1-pt)^γ·log(pt)
```
- 参数: γ=2.0, α=0.25
- 用于处理类别不平衡情况
- 通过降低简单样本权重，专注困难样本

### 6.3 优化器与学习率策略

**优化器:** AdamW
```
参数:
- lr = 1e-4 (初始学习率)
- weight_decay = 1e-5 (权重衰减)
- β₁ = 0.9 (一阶矩估计系数)
- β₂ = 0.999 (二阶矩估计系数)
```

**学习率调整:** ReduceLROnPlateau
```
- mode: 'max' (监控指标最大化，即AUC)
- factor: 0.5 (学习率乘以0.5)
- patience: 5 epochs (5个epoch无改进则调整)
- min_lr: 1e-7 (最小学习率下界)
```

**示例:** 如果AUC 5个epoch未改进，学习率从1e-4 → 5e-5 → 2.5e-5 ...

### 6.4 早停策略 (Early Stopping)

```
监控指标: Val AUC (验证集AUC)
耐心值: 10 epochs (10个epoch无改进则停止)
```

- 每epoch计算验证集AUC
- 若AUC未超过历史最高值：counter += 1
- 若counter ≥ 10：停止训练，恢复最佳模型

### 6.5 VGG16 微调策略 (STORK 风格)

**单阶段统一训练方案:**

```
初始状态:
├── VGG16 前 1/3 层: 冻结 (requires_grad=False)
├── VGG16 后 2/3 层: 训练 (requires_grad=True)
├── Transformer: 训练
└── 分类头: 训练
```

**原理:**
- 前1/3层学习 ImageNet 的通用低阶特征 (边缘、纹理)，无需微调
- 后2/3层学习高阶语义特征，需要适应胚胎域特征
- 统一训练优于两阶段，避免梯度流中断

---

## 七、评估与测试

### 7.1 评估指标

在训练、验证、测试阶段计算以下指标：

| 指标 | 定义 | 解释 |
|------|------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 整体分类准确率 |
| **Precision** | TP/(TP+FP) | "正样本预测正确的比例" |
| **Recall** | TP/(TP+FN) | "真实正样本被识别的比例" |
| **F1-Score** | 2·(Precision·Recall)/(Precision+Recall) | Precision和Recall的调和平均数 |
| **AUC** | ROC曲线下面积 | 不同阈值下性能的综合度量 |

**其中:**
- TP (True Positive): 正确预测为"好" 的好胚胎
- TN (True Negative): 正确预测为"差" 的差胚胎
- FP (False Positive): 错误预测为"好" 的差胚胎
- FN (False Negative): 错误预测为"差" 的好胚胎

### 7.2 分类阈值

```
预测概率 p ∈ [0, 1]

默认阈值 = 0.5:
  p ≥ 0.5 → 预测为 "好" (1)
  p <  0.5 → 预测为 "差" (0)
```

可根据实际需求调整阈值以平衡精确率和召回率。

### 7.3 验证集评估 (训练过程中)

在每个 epoch 后评估：

```python
# 伪代码
for epoch in range(num_epochs):
    # ... 训练 ...

    # 验证
    val_loss, val_metrics = validate(model, val_loader)

    # 检查是否需要调整学习率
    scheduler.step(val_metrics['auc'])

    # 检查是否需要早停
    early_stopping(val_metrics['auc'], model)

    # 记录历史
    history['val_auc'].append(val_metrics['auc'])

    if early_stopping.early_stop:
        break
```

### 7.4 测试集最终评估

模型训练完成后，在 **完全未见过的测试集** 上进行最终评估：

```python
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        test_preds.extend(outputs.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# 计算最终指标
test_metrics = calculate_metrics(test_labels, test_preds)
print(f"Test AUC: {test_metrics['auc']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
```

### 7.5 训练历史保存

每个阶段的训练历史保存为 JSON：

```json
{
  "train_loss": [0.6931, 0.6245, ..., 0.4523],
  "val_loss": [0.6812, 0.6111, ..., 0.4678],
  "val_auc": [0.5234, 0.6123, ..., 0.8756]
}
```

可用于：
- 绘制学习曲线
- 分析过拟合/欠拟合情况
- 调试训练过程

---

## 八、代码实现（PyTorch）

### 项目结构

```
ViT-Hybrid/
├── main.py                      # 主入口点
├── README.md                    # 项目文档
├── model/                       # 模型相关
│   ├── __init__.py
│   ├── layers.py               # 自定义层（FocalPlaneFusion、PositionalEncoding、TransformerBlock）
│   └── model.py                # 模型架构（VGG16FeatureExtractor、ViTHybridModel、create_model）
├── dataset/                     # 数据处理
│   ├── __init__.py
│   └── dataset.py              # 数据加载器（EmbryoDataset、ImagePreprocessor、create_data_loaders）
├── config/                      # 配置管理
│   ├── __init__.py
│   └── config.py               # 配置类（ModelConfig、TrainingConfig、DataConfig、PathConfig）
├── utils/                       # 工具函数
│   ├── __init__.py
│   └── utils.py                # 工具（FocalLoss、EarlyStopping、calculate_metrics等）
├── training/                    # 训练
│   ├── __init__.py
│   └── train.py                # 训练脚本（两阶段训练、早停、学习率调度）
├── checkpoints/                 # 模型保存目录
├── logs/                        # 训练日志
└── results/                     # 结果输出
```

### 环境依赖

```bash
pip install torch torchvision
pip install numpy pillow
pip install scikit-learn
```

### 数据组织格式

支持两种数据结构:

**方式1 (BiLSTM风格-推荐):**
```
data_root/
├── image/
│   ├── embryo_001.jpg
│   ├── embryo_002.jpg
│   └── ...
└── score_dict.json (包含quality标签)
```

**方式2 (时间序列风格):**
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
        └── ...
```

score_dict.json格式示例:
```json
{
  "embryo_001.jpg": {"quality": 1},
  "embryo_002.jpg": {"quality": 0},
  "embryo_003.jpg": {"quality": 1}
}
```

### 快速开始

#### 0. 生成评分字典（数据准备）

如果使用 BiLSTM 风格的数据（单张图像 + score_dict.json），需要先生成评分字典：

```bash
python generate_score_dict.py --maindb MainDB.csv --image_dir ./image -o score_dict.json
```

**参数说明：**
- `--maindb`: MainDB.csv 文件路径（必需）
- `--image_dir`: 胚胎图像目录（可选，用于验证图像存在）
- `--filtered`: filtered_clean_embryos.csv 文件路径（可选）
- `--output`: 输出文件路径（默认：score_dict.json）

**数据处理过程：**
1. 读取 MainDB.csv 中的胚胎数据
2. 从 ICM/TE 评分计算 BS_prime 分数
3. 映射质量标签 (good=1, poor=0)
4. 筛选有有效质量标签的胚胎
5. 生成 JSON 格式的评分字典

**输出示例：**
```json
{
  "embryo_001": {
    "bs_score": 3.5,
    "quality": 1
  },
  "embryo_002": {
    "bs_score": 6.2,
    "quality": 0
  }
}
```

将生成的 `score_dict.json` 放在数据目录中：
```
data_root/
  ├── image/
  │   ├── embryo_001.jpg
  │   ├── embryo_002.jpg
  │   └── ...
  └── score_dict.json
```

#### 1. 生成配置文件

使用 main.py 生成配置模板：

```bash
python main.py generate-config --output my_config.json
```

#### 2. 配置文件设置

生成的 `my_config.json` 配置文件需编辑以下必需字段：

```json
{
  "data": {
    "DATA_ROOT": "/path/to/your/embryo/data",
    "SCORE_DICT_PATH": "score_dict.json"
  },
  "training": {
    "BATCH_SIZE": 4,
    "EPOCHS": 100
  }
}
```

#### 3. 单阶段训练

```bash
python main.py train --data_root /path/to/data
```

#### 4. 两阶段训练（推荐）

```bash
python main.py train --data_root /path/to/data --two_stage
```

#### 5. 使用Focal Loss（适用于类别不平衡）

```bash
python main.py train --data_root /path/to/data --two_stage --use_focal_loss
```

#### 6. 多焦距多时间点训练（Multi-Focal Mode）

**背景**：从 `selected_image_paths.csv` 中加载多焦距多时间点的完整胚胎数据，支持完整性检查（必须有33个时间点）。

**数据准备**：
1. 生成 `selected_image_paths.csv`（包含所有时间点和焦距的图像路径）
2. 准备 `MainDB.csv`（包含 EmbryoID, ICM, TE, Quality）
3. 准备 `filtered_clean_embryos.csv`（补充低质量胚胎）

**训练命令**：

```bash
python main.py train \
  --data_root /liaozizhuo/ \
  --multi_focal \
  --selected_images_csv /liaozizhuo/selected_image_paths.csv \
  --maindb_csv /liaozizhuo/scripts/MainDB.csv \
  --filtered_csv /liaozizhuo/scripts/filtered_clean_embryos.csv \
  --two_stage
```

**多焦距模式的工作流**：

```
selected_image_paths.csv
  (PatientID, EmbryoNumber, FocalPlane, TargetHour, FullPath)
           ↓
加载质量标签
  (MainDB.csv + filtered_clean_embryos.csv)
           ↓
验证完整性
  (检查每个胚胎是否有完整的33个时间点)
           ↓
聚合焦距
  (对每个胚胎平均所有焦距和时间点的图像)
           ↓
创建数据加载器
  (train/val/test 分层划分)
           ↓
训练二分类模型
```

**完整性检查详情**：
- **要求**：每个胚胎必须有完整的33个时间点（96-112小时，每0.5小时采样）
- **验证**：检查所有图像文件是否存在
- **拒绝**：时间点不完整或缺少图像的胚胎会被跳过
- **统计**：自动输出验证统计（总胚胎数、有效胚胎数、拒绝原因）

**示例输出**：
```
MULTI-FOCAL DATA VALIDATION SUMMARY
================================================================================
Total embryos: 150
Valid embryos: 142
Rejected embryos: 8
Success rate: 94.7%

Rejection reasons:
  Incomplete timepoints: 5
  Missing image files: 3

Final dataset: 4686 images from 142 embryos
```

#### 7. 使用自定义配置文件

修改 encoder 层数和注意力头数量：

```bash
# 编辑 my_config.json，修改：
# "NUM_TRANSFORMER_LAYERS": 6,  # 改为 6 层 encoder
# "NUM_HEADS": 16,              # 改为 16 个注意力头

python main.py train --data_root /path/to/data --config_file my_config.json --two_stage
```

### 完整示例工作流

```bash
# 1. 生成评分字典（数据准备）
python generate_score_dict.py --maindb MainDB.csv --image_dir ./image -o score_dict.json

# 2. 生成配置文件
python main.py generate-config --output my_config.json

# 3. 编辑 my_config.json，设置数据路径和训练参数
# 编辑以下字段：
# - data.DATA_ROOT: /path/to/embryo/data
# - training.BATCH_SIZE: 4 (根据GPU内存调整)
# - training.EPOCHS: 100
# - model.NUM_TRANSFORMER_LAYERS: 4 (可选)
# - model.NUM_HEADS: 8 (可选)

# 4. 执行两阶段训练
python main.py train \
  --data_root /path/to/embryo/data \
  --config_file my_config.json \
  --two_stage \
  --use_focal_loss

# 5. 查看训练结果
# 结果保存在 results/ 目录中
```

### 多焦距训练完整工作流

```bash
# 1. 生成 selected_image_paths.csv（如果还没有）
# 使用 BiLSTM/select_TimePoint_time_lapse.py 或类似脚本

# 2. 准备数据文件
# - selected_image_paths.csv (包含所有时间点和焦距)
# - MainDB.csv (包含 EmbryoID, ICM, TE, Quality)
# - filtered_clean_embryos.csv (补充低质量胚胎)

# 3. 生成配置文件
python main.py generate-config --output multi_focal_config.json

# 4. 编辑 multi_focal_config.json（可选）
# 调整 BATCH_SIZE, EPOCHS, LEARNING_RATE 等参数

# 5. 执行多焦距训练（含完整性验证）
python main.py train \
  --data_root /liaozizhuo/ \
  --multi_focal \
  --selected_images_csv /liaozizhuo/selected_image_paths.csv \
  --maindb_csv /liaozizhuo/scripts/MainDB.csv \
  --filtered_csv /liaozizhuo/scripts/filtered_clean_embryos.csv \
  --config_file multi_focal_config.json \
  --two_stage

# 6. 查看训练结果
# 结果保存在 results/ 目录中
# 包括：loss 曲线、evaluation metrics、最终模型权重
```

### 训练流程

main.py 自动执行以下8个步骤：

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: 环境设置 (ENVIRONMENT SETUP)                        │
│ - 创建 checkpoints/, logs/, results/ 目录                   │
│ - 检测 GPU/CPU 可用性                                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: 配置加载 (CONFIGURATION LOADING)                    │
│ - 加载模型配置 (time_steps, focal_planes等)                 │
│ - 加载训练配置 (batch_size, learning_rate等)                │
│ - 加载数据配置 (data_root, score_dict_path等)               │
│ - 合并命令行参数                                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: 数据准备 (DATA PREPARATION)                         │
│ - 创建数据加载器 (train_loader, val_loader, test_loader)    │
│ - 进行图像预处理 (Petri dish crop, center crop, resize)    │
│ - 应用数据增强 (flip, brightness, contrast)                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: 模型构建 (MODEL BUILDING)                           │
│ - 初始化 VGG16 特征提取器                                   │
│ - 初始化 Transformer 编码器                                 │
│ - 初始化分类头                                              │
│ - 冻结 VGG16 参数                                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: 模型训练 (MODEL TRAINING)                           │
│                                                              │
│ 两阶段训练:                                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ STAGE 1: Transformer训练 (VGG16冻结)                │  │
│  │ - 固定 VGG16 权重                                   │  │
│  │ - 仅训练 Transformer + 分类头                        │  │
│  │ - 持续 epochs/2 轮                                  │  │
│  │ - 使用早停和学习率调整                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                   │                                         │
│                   ▼                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ STAGE 2: VGG16微调 (解冻)                           │  │
│  │ - 解冻 VGG16 高层卷积层                              │  │
│  │ - 以0.1倍学习率联合训练                               │  │
│  │ - 持续 epochs/2 轮                                  │  │
│  │ - 进一步优化所有参数                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                   │                                         │
│ 单阶段训练:                                                 │
│  └─ 仅执行 STAGE 1                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: 模型评估 (MODEL EVALUATION)                         │
│ - 在测试集上评估模型性能                                    │
│ - 计算 Accuracy, Precision, Recall, F1-Score, AUC         │
│ - 输出详细的性能指标                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: 保存模型 (SAVE FINAL MODEL)                         │
│ - 保存最终模型到 checkpoints/final_model.pth               │
│ - 保存训练历史到 results/stage1_history.json              │
│ - 如两阶段则保存 results/stage2_history.json              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: 结果总结 (SUMMARY)                                  │
│ - 展示生成的所有文件                                        │
│ - 输出后续步骤建议                                          │
└─────────────────────────────────────────────────────────────┘
```

详细步骤说明：

| 步骤 | 功能 | 输入 | 输出 |
|------|------|------|------|
| 1 | 环境设置 | 命令行参数 | GPU/CPU设备, 目录结构 |
| 2 | 配置加载 | config类 | 合并后的配置字典 |
| 3 | 数据准备 | 数据路径 | train/val/test加载器 |
| 4 | 模型构建 | 模型配置 | 初始化的模型 |
| 5 | 模型训练 | 数据加载器, 模型 | 训练后的模型 |
| 6 | 模型评估 | 测试数据, 模型 | 性能指标 |
| 7 | 保存模型 | 模型 | 模型文件 (.pth) |
| 8 | 结果总结 | 文件系统 | 训练报告 |

或直接使用训练脚本：

```bash
python training/train.py --data_root /path/to/data --two_stage
```

### 模型使用示例

#### 训练模型

```python
from model.model import create_model
from config.config import get_model_config
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
from model.model import create_model

model = create_model()
model.load_state_dict(torch.load('checkpoints/final_model.pth'))
model.eval()

# 进行预测
with torch.no_grad():
    output = model(input_data)
    prediction = (output > 0.5).float()
```

### 配置参数说明

#### 模型配置（config/config.py - ModelConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TIME_STEPS` | 33 | 时间点数量 |
| `FOCAL_PLANES` | 11 | 焦平面数量 |
| `IMAGE_SIZE` | 224 | 图像尺寸 |
| `NUM_TRANSFORMER_LAYERS` | 4 | Transformer层数 |
| `NUM_HEADS` | 8 | 注意力头数 |
| `D_MODEL` | **4096** | 特征维度（与VGG16输出保持一致，STORK风格） |
| `DFF` | **2048** | 前馈网络维度 |
| `DROPOUT_RATE` | 0.1 | Dropout率 |

#### 训练配置（config/config.py - TrainingConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 4 | 批次大小 |
| `EPOCHS` | 100 | 训练轮数 |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `WEIGHT_DECAY` | 1e-5 | 权重衰减 |
| `TRAIN_RATIO` | 0.7 | 训练集比例（70%） |
| `VAL_RATIO` | 0.15 | 验证集比例（15%）；测试集为 15% |
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

可在config/config.py中控制预处理选项：

```python
APPLY_PETRI_CROP = True    # 启用培养皿裁剪
APPLY_CENTER_CROP = True   # 启用中心裁剪
```

### 配置文件和标签生成

#### 修改配置

编辑 `config/config.py`：

**方式1：BiLSTM风格（单张图像）**
```python
class DataConfig:
    DATA_ROOT = '/path/to/data'      # 数据根目录
    SCORE_DICT_PATH = 'score_dict.json'  # 标签文件路径
```

**方式2：时间序列风格（good/poor目录）**
```python
class DataConfig:
    DATA_ROOT = '/path/to/data'      # 数据根目录（包含good/poor子目录）
    SCORE_DICT_PATH = None           # 不使用score_dict
```

#### 生成score_dict.json

如果使用BiLSTM风格数据，需要创建score_dict.json来标注图像质量。

**方法1：从CSV文件生成**

创建CSV文件 `labels.csv`（格式：filename,quality）：
```csv
filename,quality
embryo_001.jpg,1
embryo_002.jpg,0
embryo_003.jpg,1
```

然后在config/config.py中设置：
```python
class DataConfig:
    DATA_ROOT = '/path/to/data'
    SCORE_DICT_PATH = 'score_dict.json'
```

**方法2：手动创建JSON**

在数据根目录创建 `score_dict.json`：
```json
{
    "embryo_001.jpg": {"quality": 1},
    "embryo_002.jpg": {"quality": 0},
    "embryo_003.jpg": {"quality": 1}
}
```

quality值说明：
- `1` = 好胚胎 (Good embryo)
- `0` = 差胚胎 (Poor embryo)

### 注意事项

1. 数据路径必须按照指定格式组织
2. 图像命名格式：`time_XXX/focal_XX.jpg`
3. 推荐使用两阶段训练策略
4. 类别不平衡时建议使用Focal Loss
5. 根据GPU显存调整batch_size
6. 训练前确保有足够的磁盘空间保存检查点
7. 图像预处理会自动进行，无需手动处理