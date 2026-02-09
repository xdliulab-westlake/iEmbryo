# STORK PyTorch - 胚胎可行性分类与分析

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.9+-green.svg)](https://deepspeed.ai/)

STORK (Specialized Training and Optimization for Reproductive Knowledge) 是面向胚胎可行性二分类的 PyTorch 训练与评估系统。
项目融合了胚胎图像预处理、胚胎级别数据隔离、图像级别平衡策略、批量缓存、GPU 端增强、多模型支持以及图像/胚胎双层评估。

## 关键特性

- 胚胎级别隔离：同一胚胎只出现在一个数据集，避免数据泄漏
- 图像级别平衡：下采样到少数类，训练/验证/测试各自保持 50:50
- 70/15/15 分割：在胚胎隔离前提下分配图像数量
- STORK 专业预处理：培养皿检测 + 统一尺度处理
- 批量缓存：按 chunk 保存 .pt 文件，减少小文件并加速复现
- GPU 增强：Kornia 加速的随机增强 + MixUp/CutMix
- 训练优化：AMP、DeepSpeed、分布式/多 GPU
- 评估与可视化：图像级 + 胚胎级评估、Grad-CAM、HTML/JSON/PDF 报告
- 特征分析：统一 PCA/t-SNE/UMAP 分析管线（见 PCA_ANALYSIS_README.md）

## 5 分钟快速开始

```bash
# 1. 安装依赖
python setup.py
# 或者
pip install -r requirements.txt

# 2. 复制配置并修改路径
cp config/default.yaml config/my_config.yaml
# 编辑 data.data_dir 与 paths.output_dir/model_dir/log_dir/cache_dir

# 3. 开始训练 + 测试评估
python main.py --config config/my_config.yaml
```

提示：默认配置中的路径指向作者本地目录，必须先修改。

## 数据准备

### 数据目录结构

```
/data/your_dataset/
├── 0/                    # Non-viable
│   ├── D2023.09.22_S01009_I4424_P_WELL5_RUN552_F-30.jpeg
│   └── ...
└── 1/                    # Viable
    ├── D2023.08.26_S00892_I4425_P_WELL16_RUN548_F-30.jpeg
    └── ...
```

### 胚胎 ID 提取规则

系统会从文件名中提取胚胎 ID，并剥离 `RUNxx`、`F-xx` 等后缀。
常见格式：

- `D2023.05.31_S00744_I4424_P_WELL12_RUN595` -> `D2023.05.31_S00744_I4424_P_WELL12`
- `D2022.04.15_S00003_I4425_P_8.jpg` -> `D2022.04.15_S00003_I4425_P_8`

## 数据分割与平衡策略

- 胚胎级别隔离：按胚胎 ID 分组后再分配到 train/val/test
- 图像级别平衡：对每个集合下采样到少数类总图像数（50:50）
- 默认分割比例：70/15/15（train/val/test）
- 自动保存映射：`outputs/split_info/embryo_split_mapping.json` 与 `image_split_mapping.csv`

## 配置说明

核心配置在 `config/default.yaml`，常用字段如下：

```yaml
data:
  data_dir: "/path/to/your/data"
  img_size: 224
  num_workers: 8
  apply_petri_crop: true
  cache_chunk_size: 1000
  balance_train_classes: false  # 已废弃

model:
  name: "vgg16"
  pretrained: true
  fine_tune: true

training:
  epochs: 30
  batch_size: 512
  learning_rate: 5e-5
  loss: "cross_entropy"
  label_smoothing: 0.15
  early_stopping:
    enabled: true
    monitor: "val_accuracy"
    patience: 8

optimization:
  use_deepspeed: false
  mixed_precision: true

paths:
  output_dir: "./outputs"
  model_dir: "./models"
  log_dir: "./logs"
  cache_dir: "./batch_cache"
```

## 支持的模型与结构要点

- VGG16 / VGG19
- ResNet50 / ResNet152
- InceptionV3
- EfficientNetB0
- DenseNet121/169/201
- ConvNeXt-Tiny

模型配置通过 `model.name` 控制，优先加载本地预训练权重（若存在），否则使用 torchvision/timm 预训练权重。
分类头会被替换为适配 `num_classes` 的线性层，支持整体微调或仅训练分类头。
VGG 系列使用自定义 3 层 MLP 作为分类头，ConvNeXt 使用 `head.fc`，其他模型替换最后一层。

## 图像预处理与增强

### STORK 预处理（CPU）

- 培养皿检测与裁剪：使用 Hough 圆检测获取有效区域
- 统一尺度：按 `img_size` 进行 resize
- ImageNet 归一化：训练时统一使用标准均值/方差

### 批量缓存

- 预处理后的图像以 uint8 tensor 分块写入 `.pt` 文件
- 训练时再转为 float32 并归一化
- 通过稳定缓存键复用缓存，减少重复预处理

### GPU 端增强（Kornia）

- RandomRotation / Flip / ColorJitter / RandomErasing
- 可选 GaussianNoise / GaussianBlur / ElasticTransform
- MixUp / CutMix 在训练循环中执行

## VGG16 模型结构与微调策略

### VGG16 分类头（BackboneModel 内置）

VGG16 使用自定义 3 层 MLP 分类头，替换原始 classifier：

```
Input: (B, 3, 224, 224)
VGG16 Features -> flatten (25088)
Classifier:
  Linear(25088, 4096) + ReLU + Dropout
  Linear(4096, 2048) + ReLU + Dropout
  Linear(2048, num_classes)
Output: (B, num_classes)
```

### 微调策略（BackboneModel._setup_fine_tuning）

默认策略为“冻结前 1/3 参数、训练后 2/3 参数”，对应实现：

- `fine_tune_layers = None` 时自动计算冻结比例
- `fine_tune_layers > 0` 时，解冻最后 N 个参数
- `fine_tune = false` 时仅训练分类头

示例配置：

```yaml
model:
  name: "vgg16"
  pretrained: true
  fine_tune: true
  fine_tune_layers: null  # 自动策略: 冻结前1/3, 训练后2/3
```

## 训练与评估

常用命令：

```bash
# 默认训练流程（训练 + 测试评估）
python main.py --config config/my_config.yaml

# 调整模型与训练参数
python main.py --model resnet50 --epochs 100 --batch-size 64

# 交叉验证
python main.py --cross-validation

# 只查看配置
python main.py --config config/my_config.yaml --dry-run
```

训练结束后会自动执行：

- 测试集图像级评估
- 胚胎级投票评估（多数投票/软投票）
- Grad-CAM 可视化（可在配置中关闭）
- 模型导出（ONNX/TorchScript）

## 输出内容

```
outputs/
├── evaluation/                 # 测试集评估报告 (HTML/JSON/图表)
├── embryo_evaluation/          # 胚胎级评估结果
├── gradcam/                    # Grad-CAM 可视化
├── split_info/                 # 胚胎/图像分割映射
└── cross_validation/           # 交叉验证结果

models/
├── checkpoints/                # 训练检查点
├── best_models/                # 最佳模型
└── exports/                    # ONNX/TorchScript

logs/
└── *.log                        # 训练日志与历史

batch_cache/
├── train_chunk_*.pt
├── val_chunk_*.pt
├── test_chunk_*.pt
└── metadata.json
```

## 特征分析与可视化

- 统一 PCA/t-SNE/UMAP 管线：`unified_pca_analysis.py`
- 三分类可视化：`visualize_three_classes.py`
- PC 变化序列分析：`analyze_pc_variations.py`

详细说明见：`PCA_ANALYSIS_README.md`

## 项目结构（核心目录）

```
STORK_lzz_torch/
├── config/                # 配置管理与默认参数
├── data/                  # 数据加载、预处理、缓存、GPU增强
│   ├── stork_preprocessing.py
│   ├── parallel_processing.py
│   ├── batch_cache.py
│   └── gpu_transforms.py
├── models/                # 预训练模型封装与选择
├── training/              # 训练器、损失函数、早停与AMP
├── evaluation/            # 图像级/胚胎级评估与报告
├── visualization/         # Grad-CAM、训练曲线、特征空间
├── utils/                 # 日志、数据统计、模型管理
├── main.py                # 训练入口
├── setup.py               # 环境初始化
└── PCA_ANALYSIS_README.md # PCA/UMAP/t-SNE 分析说明
```

## 常见注意事项

- 未安装 Kornia 时，GPU 端增强会自动关闭
- `balance_train_classes` 已废弃，当前采用图像级别平衡逻辑
- 交叉验证默认按胚胎 ID 分组，避免数据泄漏
