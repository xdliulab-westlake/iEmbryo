# ViT-Hybrid 模型架构文档（多焦平面融合版）

**任务目标：** 胚胎好/坏二分类
**模型类型：** ViT-Hybrid（VGG16 + Transformer）
**版本：** v2.0（多焦平面融合）
**设计者：** Liao Zizhuo（2025）

---

## 一、设计动机

胚胎在 0–112 小时的发育过程中，其形态变化呈现明显的时间连续性；而在每个时间点，不同焦平面（focal depths）共同记录了胚胎的三维形态特征。为了充分利用这些时空信息，本模型采用：
- **VGG16** 提取空间特征（预训练权重，针对胚胎微调）
- **Transformer** 建模时间维度变化（自注意力机制）
- **多焦平面融合** 聚合所有焦平面信息，避免单平面偏差

构建时空联合特征表达，实现胚胎二分类。

---

## 二、数据结构

| 维度 | 含义 |
|------|------|
| `time_steps = 33` | 时间点（96–112 小时，每 0.5h 采样） |
| `focal_planes = 11` | 每个时间点的焦平面数量 |
| `image_shape = (224, 224, 3)` | 单张图像的尺寸 |
| **输入张量** | `(batch_size, 33, 11, 224, 224, 3)` |
| **输出标签** | 二分类：好胚胎 (1) / 坏胚胎 (0) |

---

## 三、模型架构

### 整体流程

```
输入：(batch, 33, 11, 224, 224, 3)
  │
  ├→ VGG16 特征提取 (33×11 张图像 → 33×11×4096)
  │
  ├→ 焦平面融合 (33×11×4096 → 33×4096)
  │
  ├→ Transformer 编码 (33×4096 → 33×4096)
  │  └─ 4 层 MultiHeadAttention (8 heads)
  │
  ├→ 全局平均池化 (33×4096 → 4096)
  │
  ├→ 分类头 (4096 → 512 → 1)
  │
  └→ 输出：胚胎为"好"的概率 ∈ [0,1]
```

### 模块详解

#### 1. VGG16 卷积特征提取

- **权重来源：** 预训练的 VGG16（ImageNet 权重，PyTorch torchvision 自动下载）
- **架构：** `features + avgpool(7×7) + flatten + classifier[:4]`
  - features: 13层卷积 (conv-relu-pool)
  - avgpool: 7×7 全局平均池化
  - classifier[:4]: 前4层 FC 层（4096 → 4096 → 4096 → 4096）
- **输出维度：** 每张图像 4096 维特征向量
- **参数共享：** 所有 33×11 张图像共用同一权重，实现高效特征提取
- **微调策略：** 冻结前 1/3 层（低阶边缘纹理特征不变），训练后 2/3 层（高阶胚胎特征适应）

#### 2. 焦平面融合模块 (Focal Plane Fusion)

**输入：** 单个时间点的 11 张焦平面特征 `(11, 4096)`
**操作：** 平均池化聚合
```
fusion(f₀, f₁, ..., f₁₀) = (f₀ + f₁ + ... + f₁₀) / 11
```
**输出：** `(1, 4096)` 融合特征向量

**为什么用平均而非注意力？**
- 避免模型过度关注某个焦平面，导致其他信息丢失
- 确保三维结构信息均匀融合
- 降低计算复杂度和过拟合风险

#### 3. Transformer 编码器（时间序列建模）

**输入：** 33 个时间点的特征序列 `(33, 4096)`

**位置编码：** 可学习的位置编码，显式表示时间顺序
```
PE(t) = learnable_embedding(t)  ∈ ℝ^4096
```

**Transformer块结构（共4层）：**
```
MultiHeadAttention(
  query, key, value = x_t
  num_heads = 8
  head_dim = 4096 / 8 = 512
)
  ↓
FeedForward(
  Linear(4096 → 2048, ReLU)
  Linear(2048 → 4096)
)
  ↓
LayerNorm + Residual
```

**每层的操作细节：**
1. 多头自注意力：8 个注意力头，每个 512 维
   - Query/Key/Value: (batch, 33, 4096) → (batch, 8, 33, 512)
   - Attention: softmax(QK^T / √512)
   - 输出合并：(batch, 8, 33, 512) → (batch, 33, 4096)

2. 前馈网络：
   - 隐层：4096 → 2048（降维）
   - 输出层：2048 → 4096（升维）
   - 激活：ReLU

3. 归一化和残差：
   - LayerNorm(x + attention(x))
   - LayerNorm(x + feedforward(x))

**模型容量：**
```
总参数量：
  - VGG16: ~138M（其中微调部分 ~70M）
  - Transformer 4层：~70M
  - 分类头：~1M
  合计：~209M（其中训练参数 ~140M）
```

#### 4. 分类头（Classification Head）

**输入：** (33, 4096) 时间序列特征

**步骤 1：全局平均池化**
```
GAP: (33, 4096) → (4096)
每个维度在 33 个时间点上平均
```

**步骤 2：全连接网络**
```
Dropout(0.5)
  ↓
Linear(4096 → 512) + ReLU
  ↓
Dropout(0.3)
  ↓
Linear(512 → 1) + Sigmoid
  ↓
输出：p ∈ [0, 1]（胚胎为"好"的概率）
```

---

## 四、训练配置与策略

### 超参数设置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| **批次大小** | 4 | 取决于GPU显存（V100: 32GB 可用 ~4-8） |
| **总轮数** | 100 | 最多训练轮数（通常 30-50 epoch 早停） |
| **学习率** | 1e-4 | 初始学习率（AdamW） |
| **权重衰减** | 1e-5 | L2 正则化系数 |
| **优化器** | AdamW | 自适应学习率优化 |

### 损失函数

**Binary Cross-Entropy Loss（默认）：**
```
L_BCE = -[y·log(p) + (1-y)·log(1-p)]

其中 p = sigmoid(logit) ∈ [0, 1]
      y ∈ {0, 1}（标签）
```
适用于样本较平衡的情况。

**Focal Loss（可选）：**
```
L_Focal = -α·(1-pt)^γ·log(pt)

其中 pt = p if y=1 else 1-p
      γ = 2.0（聚焦参数）
      α = 0.25（权重参数）
```
用于处理类别不平衡，通过降低简单样本权重专注困难样本。

### 学习率调整

**ReduceLROnPlateau 策略：**
```
监控指标：Val AUC（验证集AUC）
模式：最大化（mode='max'）
参数：
  - factor: 0.5（每次衰减为原来的 50%）
  - patience: 5 epochs（5 epoch 无改进则调整）
  - min_lr: 1e-7（学习率下界）

示例：
  epoch  1-5: LR = 1e-4（无改进 → 触发）
  epoch  6-10: LR = 5e-5（无改进 → 触发）
  epoch 11-15: LR = 2.5e-5（...）
```

### 早停机制

```
监控指标：Val AUC
耐心值：10 epochs

逻辑：
  - 记录验证集最高 AUC
  - 每 epoch 对比当前 AUC 与历史最高
  - 若无改进：计数器 +1
  - 若计数器 ≥ 10：停止训练，恢复最佳权重
  - 若有改进：计数器归零，保存当前权重
```

### VGG16 微调策略（单阶段）

**初始状态：**
```
VGG16 层分析：
  第 1-5 层（Block 1-2）：边缘、纹理检测器 → 冻结
  第 6-13 层（Block 3-5）：高阶特征提取器 → 训练

Transformer：训练
分类头：训练
```

**微调原理：**
- 前 1/3 层学习的是通用视觉基础（边缘、颜色梯度），ImageNet 预训练足够
- 后 2/3 层学习的是高层语义特征，需要在胚胎数据上适应
- 统一训练优于两阶段（避免梯度流中断，充分利用数据）

---

## 五、数据处理流程

### 图像预处理步骤

1. **读取图像**
   - 使用 OpenCV 或 PIL 加载
   - 转换为 RGB 颜色空间

2. **培养皿检测与裁剪**（可选，默认启用）
   - 使用 Hough 圆检测找到圆形培养皿边界
   - 参数范围：半径 15%-45% 图像对角线
   - 参数组合：(60,40), (70,45), (50,35), (80,50), (90,55)
   - 取最大检测圆作为培养皿

3. **中心正方形裁剪**（可选，默认启用）
   - 从裁剪后的图像提取中心正方形
   - 标准化输入范围

4. **尺寸调整**
   - 调整为 224×224（VGG16 标准输入）
   - 使用双线性插值

5. **张量转换与归一化**
   ```python
   img_tensor = ToTensor()(img_ndarray)  # [0, 1]
   mean = [0.485, 0.456, 0.406]  # ImageNet 官方参数
   std  = [0.229, 0.224, 0.225]
   img_normalized = Normalize(mean, std)(img_tensor)
   ```

### 数据分割

```
训练集：70%（应用数据增强）
验证集：15%（用于早停与学习率调整）
测试集：15%（最终性能评估，完全不见过）

划分策略：分层随机划分（stratified split）
确保各类别（好/差）在三个集合中比例一致
```

### 数据增强（仅用于训练集）

| 增强方法 | 参数 | 作用 |
|---------|------|------|
| 随机水平翻转 | p=0.5 | 增强空间不变性 |
| 颜色抖动 | brightness=0.1, contrast=0.1 | 光照鲁棒性 |

---

## 六、评估指标

### 性能指标定义

| 指标 | 定义 | 解释 |
|------|------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 整体分类准确率 |
| **Precision** | TP/(TP+FP) | "正样本预测正确的比例" |
| **Recall** | TP/(TP+FN) | "真实正样本被识别的比例" |
| **F1-Score** | 2·(Precision·Recall)/(Precision+Recall) | Precision 和 Recall 的调和平均 |
| **AUC** | ROC 曲线下面积 | 不同阈值下性能的综合度量 |

其中：TP（真正样本）、TN（真负样本）、FP（假正样本）、FN（假负样本）

### 分类阈值

默认阈值 = 0.5：
```
p ≥ 0.5 → 预测为"好" (1)
p <  0.5 → 预测为"差" (0)
```

可根据实际需求调整阈值以平衡精确率和召回率。

---

## 七、模型实现细节

### 项目结构

```
ViT-Hybrid/
├── main.py                      # 主入口（训练、配置生成）
├── README.md                    # 本文件
├── config/
│   └── config.py               # 配置类（Model/Training/Data/Path）
├── model/
│   ├── model.py                # ViT-Hybrid 模型定义
│   └── layers.py               # 自定义层（Fusion、Encoding、Block）
├── dataset/
│   ├── dataset.py              # 数据加载器
│   └── multi_focal_data_loader.py  # 多焦距加载器（带缓存）
├── training/
│   └── train.py                # 训练脚本
├── utils/
│   └── utils.py                # 工具函数（损失、早停、指标）
├── analysis/                   # 可解释性分析模块
├── checkpoints/                # 模型保存
├── results/                    # 训练历史与结果
├── logs/                       # TensorBoard 日志
└── cache/                      # 验证缓存
```

### 环境依赖

```bash
pip install torch torchvision
pip install numpy pillow scikit-learn opencv-python
```

### 快速开始

#### 0. 数据准备

确保数据结构如下：

**方式 1（BiLSTM 风格）：**
```
data_root/
├── image/
│   ├── embryo_001.jpg
│   ├── embryo_002.jpg
│   └── ...
└── score_dict.json
```

**方式 2（时间序列风格）：**
```
data_root/
├── good/
│   ├── embryo_001/
│   │   ├── time_000/focal_00.jpg, focal_01.jpg, ...
│   │   ├── time_001/...
│   │   └── ...
│   └── ...
└── poor/
    └── ...
```

#### 1. 生成配置文件

```bash
python main.py generate-config --output config.json
```

编辑 `config.json` 设置数据路径和训练参数。

#### 2. 训练模型

**单阶段训练：**
```bash
python main.py train --data_root /path/to/data --config_file config.json
```

**两阶段训练（推荐）：**
```bash
python main.py train --data_root /path/to/data --config_file config.json --two_stage
```

**使用 Focal Loss（类别不平衡）：**
```bash
python main.py train --data_root /path/to/data --two_stage --use_focal_loss
```

#### 3. 多焦距训练（Multi-Focal Mode）

```bash
python main.py train \
  --data_root /path/to/data \
  --multi_focal \
  --selected_images_csv /path/to/selected_image_paths.csv \
  --maindb_csv /path/to/MainDB.csv \
  --filtered_csv /path/to/filtered_clean_embryos.csv \
  --two_stage
```

**后续运行（使用缓存跳过验证）：**
```bash
python main.py train \
  --data_root /path/to/data \
  --multi_focal \
  --selected_images_csv /path/to/selected_image_paths.csv \
  --maindb_csv /path/to/MainDB.csv \
  --filtered_csv /path/to/filtered_clean_embryos.csv \
  --two_stage \
  --skip_validation
```

验证结果会自动缓存到 `./cache/` 目录，使用 `--skip_validation` 可加速重复运行。

### 训练流程

main.py 自动执行以下步骤：

1. **环境设置** - 创建目录，检测 GPU
2. **配置加载** - 合并命令行参数
3. **数据准备** - 创建数据加载器
4. **模型构建** - 初始化模型，冻结 VGG16 前层
5. **模型训练** - 执行单/双阶段训练
6. **模型评估** - 测试集性能评估
7. **保存模型** - checkpoints/final_model.pth
8. **结果总结** - 生成训练报告

### 模型配置参数

#### ModelConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TIME_STEPS` | 33 | 时间点数量 |
| `FOCAL_PLANES` | 11 | 焦平面数量 |
| `IMAGE_SIZE` | 224 | 图像尺寸 |
| `NUM_TRANSFORMER_LAYERS` | 4 | Transformer 层数 |
| `NUM_HEADS` | 8 | 注意力头数 |
| `D_MODEL` | 4096 | 特征维度（与 VGG16 输出保持一致） |
| `DFF` | 2048 | 前馈网络隐层维度 |
| `DROPOUT_RATE` | 0.1 | Dropout 比率 |

#### TrainingConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 4 | 批次大小 |
| `EPOCHS` | 100 | 最多训练轮数 |
| `LEARNING_RATE` | 1e-4 | 初始学习率 |
| `WEIGHT_DECAY` | 1e-5 | L2 正则化 |
| `EARLY_STOPPING_PATIENCE` | 10 | 早停耐心值 |
| `REDUCE_LR_PATIENCE` | 5 | 学习率调整耐心值 |

### 训练输出

训练完成后生成：

- `checkpoints/final_model.pth` - 最终模型权重
- `results/stage1_history.json` - 第一阶段训练历史
- `results/stage2_history.json` - 第二阶段训练历史（两阶段训练）
- `logs/` - TensorBoard 日志（可选）

---

## 八、模型可解释性分析

### 背景

训练好的模型能进行预测，但内部决策过程是"黑箱"。为理解模型如何利用不同时间点做决策，提供四种互补的分析方法。

### 分析方法

#### 方法 1：梯度归因法（Gradient Attribution）

**原理：** 计算预测输出对输入的梯度 ∂output/∂input，衡量每个时间点的影响程度

```python
from analysis.temporal_interpretability import compute_temporal_importance

importance = compute_temporal_importance(model, images, device='cuda')
# 输出：(33,) 维重要性分数
```

**特点：** 快速、直观、可用于快速筛查

#### 方法 2：注意力权重可视化（Attention Visualization）

**原理：** Transformer 的多头注意力权重直接表示模型在时间点间的信息流动

```python
from analysis.attention_visualization import extract_attention_weights

attn_weights = extract_attention_weights(model, images, device='cuda')
# 输出：4 个张量，每个 (batch, 8, 33, 33)
```

**特点：** 显示时间点间的关联关系、信息流向

#### 方法 3：特征相关性分析（Temporal Correlation）

**原理：** 计算不同时间点特征向量间的余弦相似度，理解发育连续性

```python
from analysis.temporal_interpretability import compute_temporal_correlation

corr_matrix, info = compute_temporal_correlation(model, images, device='cuda')
# 输出：(33, 33) 相关性矩阵
```

**特点：** 量化时间点间的相似性，显示发育阶段

#### 方法 4：消融分析（Ablation Analysis）

**原理：** 逐个遮挡（移除）每个时间点，计算对预测的影响

```python
from analysis.ablation_analysis import temporal_ablation_analysis

ablation_scores, _ = temporal_ablation_analysis(model, images, device='cuda')
# 输出：(33,) 消融分数
```

**特点：** 最直接、最可信（但计算成本高）

### 快速分析

```bash
python analyze_temporal_interpretability.py \
    --model checkpoints/final_model.pth \
    --output results/temporal_analysis \
    --device cuda
```

生成输出：
- `temporal_importance.json` - 梯度归因结果
- `attention_analysis.json` - 注意力统计
- `ablation_analysis.json` - 消融分析结果
- `attention_heatmap.png` - 4 层热力图
- `temporal_correlation.png` - 相关性矩阵
- `temporal_ablation.png` - 消融分数条形图

### 编程接口示例

```python
from analysis.temporal_interpretability import compute_temporal_importance, get_critical_timepoints

# 计算重要性
importance = compute_temporal_importance(model, images, device='cuda')

# 找出最重要的 5 个时间点
critical = get_critical_timepoints(importance, topk=5)
print(f"关键时间点：{critical['hours']} 小时")
print(f"重要性分数：{critical['scores']}")
```

---

## 九、常见问题

**Q: 模型参数量是多少？**
A: 总计 ~209M，其中训练参数 ~140M（VGG16 后 2/3 层 + Transformer + 分类头）

**Q: 训练需要多长时间？**
A: GPU (V100) 上约 30-50 min/epoch，总计 1-2 小时（因早停通常 30-50 epochs）

**Q: 如何调整模型容量？**
A: 修改 config.json 中的 `NUM_TRANSFORMER_LAYERS` 和 `NUM_HEADS`

**Q: 类别不平衡怎么办？**
A: 使用 `--use_focal_loss` 参数

**Q: 如何加速多焦距训练？**
A: 第一次运行后，使用 `--skip_validation` 跳过重复验证，缓存自动保存在 `./cache/` 目录

---
