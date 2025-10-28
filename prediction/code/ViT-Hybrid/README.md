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

## 快速开始

### 1. 环境安装

```bash
pip install torch torchvision
pip install numpy pillow scikit-learn opencv-python
```

### 2. 数据准备

数据结构（多焦距多时间点）：

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

关键文件说明：
- **selected_image_paths.csv**：包含 (PatientID, EmbryoNumber, FocalPlane, TargetHour, FullPath)
- **MainDB.csv**：包含 (EmbryoID, ICM, TE, Quality)
- **filtered_clean_embryos.csv**：补充低质量胚胎标签

### 3. 快速训练

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

#### 后续运行（使用缓存，加速 50-100 倍）

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

#### 处理类别不平衡（可选）

```bash
python main.py train \
  --data_root /liaozizhuo/ \
  --multi_focal \
  --selected_images_csv /liaozizhuo/selected_image_paths.csv \
  --maindb_csv /liaozizhuo/scripts/MainDB.csv \
  --filtered_csv /liaozizhuo/scripts/filtered_clean_embryos.csv \
  --two_stage \
  --use_focal_loss
```

---

## 系统架构

### 核心设计理念

胚胎在 96-112 小时发育期间：
- **时间维度**：33 个时间点，每 0.5 小时采样一次
- **空间维度**：每个时间点有 11 个焦平面，记录三维形态
- **模型策略**：VGG16 提取空间特征 → 焦平面融合 → Transformer 建模时间序列

### 模型流程

```
输入：(batch, 33, 11, 224, 224, 3)
  │
  ├→ VGG16 特征提取
  │   └─ 所有 33×11 张图像共享权重 → 4096 维特征
  │
  ├→ 焦平面融合 (Focal Plane Fusion)
  │   └─ 平均聚合 11 个焦平面 → (33, 4096)
  │
  ├→ Transformer 编码器
  │   └─ 4 层，8 个注意力头，建模时间序列
  │
  ├→ 全局平均池化 + 分类头
  │   └─ (33, 4096) → (4096) → (512) → (1)
  │
  └→ 输出：p ∈ [0, 1]（胚胎为"好"的概率）
```

### 核心模块

| 模块 | 功能 | 输入维度 | 输出维度 |
|------|------|----------|---------|
| **VGG16** | 空间特征提取（预训练微调） | (224, 224, 3) | 4096 |
| **焦平面融合** | 11 个焦平面平均聚合 | (11, 4096) | 4096 |
| **Transformer** | 时间序列建模（4 层） | (33, 4096) | (33, 4096) |
| **分类头** | 二分类预测 | (33, 4096) | 1 |

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

## 配置管理

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
    "DATA_ROOT": "/path/to/embryo/data",
    "SCORE_DICT_PATH": "score_dict.json",
    "TEST_RATIO": 0.1,
    "APPLY_PETRI_CROP": true,
    "APPLY_CENTER_CROP": true
  },
  "paths": {
    "MODEL_SAVE_DIR": "./checkpoints",
    "LOG_DIR": "./logs",
    "RESULT_DIR": "./results",
    "CACHE_DIR": "./cache"
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

## 数据处理流程

### 图像预处理步骤

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

### 数据分割

```
训练集：70%（应用数据增强：水平翻转、颜色抖动）
验证集：15%（无增强，用于早停与学习率调整）
测试集：15%（无增强，最终性能评估）

策略：分层随机划分（确保类别比例一致）
```

---

## 训练策略

### 损失函数

**Binary Cross-Entropy（默认）：** 适用于样本平衡的情况
```
L_BCE = -[y·log(p) + (1-y)·log(1-p)]
```

**Focal Loss（可选）：** 处理类别不平衡
```
L_Focal = -α·(1-pt)^γ·log(pt)
参数：γ=2.0, α=0.25
```

### 学习率调整

```
策略：ReduceLROnPlateau
监控：Val AUC（最大化）
衰减：factor=0.5，patience=5 epochs，min_lr=1e-7

示例：
  epoch  1-5:  LR = 1e-4
  epoch  6-10: LR = 5e-5   (无改进 5 epochs 后触发)
  epoch 11-15: LR = 2.5e-5 (继续衰减)
```

### 早停机制

```
监控指标：Val AUC
耐心值：10 epochs

逻辑：
  - 每 epoch 对比验证 AUC 与历史最高值
  - 若无改进：计数器 +1
  - 若计数器 ≥ 10：停止训练，恢复最佳权重
```

### 两阶段 vs 单阶段训练

```
单阶段（推荐）：
  - VGG16 前 1/3 层冻结 → 后 2/3 层训练
  - Transformer + 分类头 → 训练
  - 统一优化，避免梯度流中断

两阶段（备用）：
  - Stage 1：VGG16 冻结，Transformer + 分类头训练
  - Stage 2：VGG16 解冻，整体微调（学习率 0.1 倍）
```

---

## 评估指标

| 指标 | 定义 | 解释 |
|------|------|------|
| **Accuracy** | (TP+TN)/(总数) | 整体分类准确率 |
| **Precision** | TP/(TP+FP) | "正样本预测正确的比例" |
| **Recall** | TP/(TP+FN) | "真实正样本被识别的比例" |
| **F1-Score** | 2·(P·R)/(P+R) | Precision 和 Recall 的调和平均 |
| **AUC** | ROC 曲线下面积 | 不同阈值下的综合性能 |

---

## 项目结构

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

## 命令行参数

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
  --use_focal_loss             使用 Focal Loss（处理类别不平衡）
  --skip_validation            跳过验证，使用缓存（第一次后运行）
  --config_file FILE, -c FILE  自定义配置文件（可选）
```

---

## 模型可解释性分析

理解模型如何利用不同时间点做决策。

### 方法 1：梯度归因法（快速）

```bash
python analyze_temporal_interpretability.py \
    --model checkpoints/final_model.pth \
    --output results/temporal_analysis \
    --device cuda
```

输出：重要性分数 (33,)

### 方法 2：注意力权重可视化

```python
from analysis.attention_visualization import extract_attention_weights

attn_weights = extract_attention_weights(model, images, device='cuda')
# 输出：4 个张量，每个 (batch, 8, 33, 33)
```

### 方法 3：特征相关性分析

```python
from analysis.temporal_interpretability import compute_temporal_correlation

corr_matrix, info = compute_temporal_correlation(model, images, device='cuda')
# 输出：(33, 33) 相关性矩阵
```

### 方法 4：消融分析（最可信但较慢）

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

## 训练输出

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
```

---

## 常见问题

**Q: 模型需要多少 GPU 显存？**
A: 每个样本约 2-3 GB（V100 32GB 可运行 batch_size=4-8）

**Q: 训练需要多长时间？**
A: GPU (V100) 上约 1-2 小时（30-50 epochs）

**Q: 如何加速多焦距训练？**
A: 第一次运行自动缓存验证结果，后续用 `--skip_validation` 跳过验证

**Q: 模型参数量？**
A: ~209M 总参数，~140M 训练参数

**Q: 如何处理类别不平衡？**
A: 使用 `--use_focal_loss` 参数

**Q: 如何调整模型容量？**
A: 修改 config.json 中的 `NUM_TRANSFORMER_LAYERS` 和 `NUM_HEADS`

---

## 版本历史

### v2.0 (2025-10)
- 多焦平面融合架构
- Transformer 时间序列建模
- VGG16 预训练微调
- 验证缓存机制（多焦距模式）
- 四层可解释性分析工具

---

**ViT-Hybrid v2.0：为胚胎质量评估而设计的时空联合深度学习系统** 🧬
