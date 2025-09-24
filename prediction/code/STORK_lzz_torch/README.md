# STORK PyTorch - 胚胎分类深度学习系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.9+-green.svg)](https://deepspeed.ai/)
[![Version](https://img.shields.io/badge/version-v2.1-green.svg)](https://github.com/your-repo/stork)

STORK (Specialized Training and Optimization for Reproductive Knowledge) PyTorch版本是一个专为胚胎可行性分类设计的深度学习系统。该系统集成了专业的胚胎图像预处理、多模型支持、分布式训练优化和全面的评估可视化功能。

## 5分钟快速入门

```bash
# 1. 进入项目目录
cd /Users/lzz/Documents/GitHub/IVF_AI/prediction/code/STORK_lzz_torch

# 2. 修改数据路径
cp config/default.yaml config/my_config.yaml
# 编辑 my_config.yaml 中的 data_dir 为你的数据路径

# 3. 开始训练
python main.py --config config/my_config.yaml

# 4. 查看结果
ls outputs/  # 训练结果
tensorboard --logdir logs/tensorboard  # 可视化
```

**就这么简单！** 系统会自动处理GPU配置、缓存创建、模型训练等所有细节。

## 快速开始

### 1. 环境安装

#### 方法一：自动安装（推荐）
```bash
# 进入项目目录
cd /Users/lzz/Documents/GitHub/IVF_AI/prediction/code/STORK_lzz_torch

# 运行环境设置脚本
python setup.py
```

### 2. 数据准备

确保你的数据按以下结构组织：

```
/your/data/directory/
├── 0/                    # 非可行胚胎 (Non-viable)
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── 1/                    # 可行胚胎 (Viable)
    ├── image1.jpg
    ├── image2.png
    └── ...
```

### 3. 配置文件设置

#### 快速配置
```bash
# 复制并修改默认配置
cp config/default.yaml config/my_config.yaml

# 使用文本编辑器修改配置
nano config/my_config.yaml  # 或使用 vim, code 等
```

#### 必须修改的配置项
```yaml
# 数据配置 - 必须修改
data:
  data_dir: "/liaozizhuo/selected_images/train/"  # ← 改为你的数据路径
  num_workers: 30                                 # ← 根据CPU核心数调整

# 训练配置 - 建议调整
training:
  batch_size: 512      # ← 根据GPU内存调整 (512适合大内存GPU)
  epochs: 2000         # ← 根据需要调整
```

#### 配置文件模板
创建你的专用配置文件：
```yaml
# my_config.yaml
data:
  data_dir: "/path/to/your/embryo/data"  # 你的数据目录
  img_size: 224
  train_split: 0.7
  num_workers: 8                         # CPU核心数的1-2倍

  # 批量缓存配置 (重要性能优化)
  cache_chunk_size: 1000

  # STORK专业预处理
  apply_petri_crop: true     # 培养皿检测裁剪
  apply_center_crop: true    # 中心方形裁剪

model:
  name: "vgg16"              # 可选: vgg16, resnet50, resnet152, efficientnetb0
  pretrained: true
  fine_tune: true

training:
  epochs: 100                # 建议100-2000轮
  batch_size: 32             # 根据GPU内存调整: 8, 16, 32, 64
  learning_rate: 0.0001

  # 类别不平衡处理
  loss: "focal"              # focal loss处理不平衡数据
  class_weights: [0.669, 1.978]

# 路径配置
paths:
  output_dir: "./outputs"
  cache_dir: "./batch_cache"   # 缓存目录
```

### 4. 开始训练


#### 常用训练场景
```bash
# 1. 小数据集快速实验
python main.py --config config/my_config.yaml --epochs 50 --batch-size 16

# 2. 生产级训练
python main.py --config config/my_config.yaml --epochs 2000 --batch-size 512

# 3. 更换模型测试
python main.py --model resnet50 --epochs 100

# 4. 指定数据目录
python main.py --data-dir "/path/to/your/data" --epochs 100
```

#### 多GPU训练（自动）
```bash
# 系统自动检测并使用所有可用GPU
python main.py --config config/my_config.yaml

# 手动指定GPU数量（如果需要）
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/my_config.yaml
```

#### 训练监控
```bash
# 实时查看训练日志
tail -f logs/training.log

# TensorBoard可视化（新终端）
tensorboard --logdir logs/tensorboard --port 6006
# 然后访问 http://localhost:6006
```

### 缓存复用策略
- **智能复用**: 相同数据目录 + 相同预处理参数
- **模型无关**: 更换模型类型不影响数据缓存
- **参数独立**: 训练参数变化不影响数据缓存
- **自动重建**: 数据或预处理参数变化时自动创建新缓存

### 存储优化
- **uint8格式**：相比float32节省50%空间
- **分块存储**：每1000张图片一个`.pt`文件，190k图片→193个文件
- **压缩存储**：PyTorch原生压缩，进一步节省空间
- **版本控制**：缓存版本管理，确保兼容性

### 训练优势
- **保持微调能力**：端到端训练，全模型可微调
- **支持数据增强**：实时随机变换，提升泛化能力
- **GPU数据增强**：支持MixUp、CutMix等高级技术
- **极速加载**：5-10倍加载速度提升
- **开发友好**：快速实验迭代，提升研发效率

## 系统功能

### STORK专业预处理
- **培养皿自动检测** - 使用霍夫圆检测算法自动识别和裁剪培养皿区域
- **中心方形裁剪** - 智能裁剪图像中心区域，专为胚胎图像优化
- **渐进式缓存** - 智能缓存机制，第一epoch边训练边缓存，后续epoch显著加速
- **按需加载** - 避免内存溢出，支持大规模数据集训练
- **动态数据增强** - 训练时实时应用随机变换，提升模型泛化能力

### 多模型支持
- **VGG16** - 经典卷积神经网络，适合胚胎图像特征提取
- **ResNet50/152** - 残差网络，更深层次的特征学习
- **EfficientNet** - 高效网络架构，平衡准确性和计算效率
- **微调策略** - 支持层级冻结和渐进式微调

### 性能优化
- **智能多GPU配置** - 自动检测并配置所有可用GPU，无需手动分布式设置
- **DeepSpeed集成** - ZeRO Stage 2内存优化，支持更大批次训练
- **混合精度训练** - FP16训练，减少内存使用，提升训练速度
- **分布式训练** - 自动NCCL后端配置，支持多GPU并行训练
- **梯度累积** - 模拟更大批次训练

### 评估和可视化
- **交叉验证** - K折交叉验证，更可靠的模型评估
- **详细指标报告** - 准确率、精确率、召回率、F1分数、AUC等
- **Grad-CAM可视化** - 模型决策可解释性分析
- **HTML报告生成** - 专业的评估报告，包含图表和统计信息

## 项目结构

```
STORK_lzz_torch/
├── config/                     # 配置管理
│   ├── default.yaml           # 默认配置文件
│   └── config_manager.py      # 配置管理器
├── data/                      # 数据处理模块
│   ├── dataset.py             # PyTorch Dataset
│   ├── transforms.py          # 数据增强
│   ├── stork_preprocessing.py # STORK专业预处理
│   └── parallel_processing.py # 并行处理和缓存
├── models/                    # 模型定义
│   └── backbone.py           # 多模型支持
├── training/                  # 训练模块
│   └── trainer.py            # 训练器主类
├── optimization/              # 优化模块
│   └── deepspeed_config.py   # DeepSpeed配置
├── evaluation/                # 评估模块
│   ├── cross_validation.py   # 交叉验证
│   └── metrics.py            # 评估指标和报告
├── visualization/             # 可视化模块
│   └── gradcam.py            # Grad-CAM实现
├── utils/                     # 工具模块
│   ├── model_manager.py      # 模型管理
│   └── logger.py             # 日志和监控
├── main.py                   # 主程序入口
├── setup.py                  # 环境设置脚本
├── requirements.txt          # 依赖列表
└── README.md                 # 项目文档
```

## 配置文件详解

### 数据配置 (`data`)

```yaml
data:
  data_dir: "/path/to/data"       # 数据目录路径
  img_size: 224                   # 图像尺寸 (224, 299, 384等)
  train_split: 0.7                # 训练集比例 (0.6-0.8)
  num_workers: 30                 # 数据加载进程数（推荐CPU核心数的1-2倍）

  # STORK专业预处理
  apply_petri_crop: true          # 培养皿检测和裁剪
  apply_center_crop: true         # 中心方形裁剪
  normalize: true                 # ImageNet标准化

  # 内存优化配置
  pin_memory: true                # 启用GPU内存固定，加速数据传输
  prefetch_factor: 4              # 预取批次数量
  persistent_workers: true        # 保持worker进程，减少重启开销

  # 数据分割
  split_strategy: "paper_70_30"   # 数据分割策略
  random_state: 42                # 随机种子
```

### 模型配置 (`model`)

```yaml
model:
  name: "vgg16"                   # 模型架构
  num_classes: 2                  # 分类类别数
  pretrained: true                # 使用预训练权重
  fine_tune: true                 # 是否微调
  fine_tune_layers: 18            # 微调层数
```

**支持的模型：**
- `vgg16` - VGG16 (推荐用于胚胎图像)
- `resnet50` - ResNet-50
- `resnet152` - ResNet-152
- `efficientnetb0` - EfficientNet-B0
- `inceptionv3` - Inception-V3

### 训练配置 (`training`)

```yaml
training:
  epochs: 100                     # 训练轮数
  batch_size: 8                   # 批次大小
  learning_rate: 0.0001           # 学习率
  weight_decay: 0.01              # L2正则化
  resume_training: true           # 是否恢复训练

  # 优化器配置
  optimizer: "adamw"              # 优化器: adam, adamw, sgd
  scheduler: "cosine"             # 学习率调度: cosine, step, plateau
  warmup_epochs: 10               # 预热轮数

  # 损失函数
  loss: "focal"                   # 损失函数: focal, cross_entropy
  focal_alpha: 0.25               # Focal Loss alpha
  focal_gamma: 2.0                # Focal Loss gamma

  # 类别权重（针对不平衡数据）
  class_weights:
    - 0.669                       # Class 0权重
    - 1.978                       # Class 1权重
```

### 验证配置 (`validation`)

```yaml
validation:
  cross_validation: false         # 是否使用交叉验证
  n_folds: 4                      # 交叉验证折数
  validation_split: 0.2           # 验证集比例

# 交叉验证详细配置
cross_validation:
  n_folds: 5                      # K折数量
  stratified: true                # 分层抽样
  shuffle: true                   # 是否打乱
  random_state: 42                # 随机种子
  save_fold_models: false         # 是否保存每折模型
```

### 数据增强配置 (`augmentation`)

STORK支持两种数据增强方式：GPU加速增强（推荐）和CPU增强。

#### GPU数据增强（推荐）⚡
```yaml
augmentation:
  enabled: true                   # 是否启用数据增强
  use_gpu_augmentation: true      # 启用GPU数据增强 (需要Kornia)

  # 基础增强
  rotation_range: 15              # 旋转角度范围
  horizontal_flip: true           # 水平翻转
  vertical_flip: false            # 垂直翻转（胚胎图像通常不需要）
  brightness_range: [0.9, 1.1]    # 亮度调整范围

  # GPU特有增强
  gaussian_noise: false           # 高斯噪声
  gaussian_blur: false            # 高斯模糊
  elastic_transform: false        # 弹性变换（医学图像专用）
  random_erasing: false           # 随机擦除

  # 高级增强（MixUp/CutMix）
  mixup_alpha: 0.0               # MixUp强度，0表示禁用
  cutmix_alpha: 0.0              # CutMix强度，0表示禁用
```

#### 性能对比
- **GPU增强**: ~5-10ms/batch，5-10倍加速
- **CPU增强**: ~50-100ms/batch

#### CPU数据增强（备用）
```yaml
augmentation:
  enabled: true
  use_gpu_augmentation: false     # 禁用GPU增强

  rotation_range: 15              # 旋转角度范围
  width_shift_range: 0.1          # 水平偏移范围
  height_shift_range: 0.1         # 垂直偏移范围
  horizontal_flip: true           # 水平翻转
  zoom_range: 0.1                 # 缩放范围
  brightness_range: [0.9, 1.1]    # 亮度调整范围
```

### DeepSpeed优化配置 (`optimization`)

```yaml
optimization:
  use_deepspeed: true             # 是否使用DeepSpeed
  mixed_precision: true           # 混合精度训练
  gradient_checkpointing: true    # 梯度检查点

  # DeepSpeed ZeRO配置
  zero_stage: 2                   # ZeRO阶段 (1, 2, 3)
  offload_optimizer: true         # 优化器状态卸载
  offload_parameters: false       # 参数卸载

  # 内存优化
  max_memory_limit_gb: 70         # 最大内存限制
  enable_memory_growth: true      # 动态内存增长
```

**ZeRO阶段说明：**
- **Stage 1** - 优化器状态分片
- **Stage 2** - 优化器状态+梯度分片 (推荐)
- **Stage 3** - 优化器状态+梯度+参数分片

### 评估配置 (`evaluation`)

```yaml
evaluation:
  save_results: true              # 保存评估结果
  generate_gradcam: true          # 生成Grad-CAM可视化
  gradcam_samples: 10             # Grad-CAM样本数
  save_confusion_matrix: true     # 保存混淆矩阵
  save_roc_curve: true            # 保存ROC曲线
```

### 日志配置 (`logging`)

```yaml
logging:
  log_level: "INFO"               # 日志级别: DEBUG, INFO, WARNING, ERROR
  save_model_every_n_epochs: 50   # 每N轮保存模型
  save_best_model: true           # 保存最佳模型
  tensorboard: true               # 启用TensorBoard
  wandb: false                    # 启用Weights & Biases
```

### 路径配置 (`paths`)

```yaml
paths:
  output_dir: "./outputs"         # 输出目录
  model_dir: "./models"           # 模型保存目录
  log_dir: "./logs"               # 日志目录
  cache_dir: "./cache"            # 缓存目录
```

## 命令行参数

```bash
python main.py [选项]

参数:
  --config CONFIG     配置文件路径 (默认: config/default.yaml)
  --debug             启用调试模式，显示详细配置信息
  --dry-run           仅验证配置，不实际训练
  --data-dir DIR      数据目录路径 (覆盖配置文件)
  --epochs N          训练轮数 (覆盖配置文件)
  --batch-size N      批次大小 (覆盖配置文件)
  --learning-rate LR  学习率 (覆盖配置文件)
  --model MODEL       模型名称 (覆盖配置文件)
```

**示例：**

```bash
# 基本训练
python main.py --config config/my_config.yaml

# 覆盖特定参数
python main.py --config config/my_config.yaml --epochs 200 --batch-size 16

# 调试模式
python main.py --config config/my_config.yaml --debug

# 验证配置
python main.py --config config/my_config.yaml --dry-run

# 使用不同模型
python main.py --model resnet50 --epochs 150
```

## 📁 输出文件结构 v2.1

训练完成后，系统会生成**专业化的**完整文件结构：

```
📁 项目根目录/
├── 🗂️ models/                           # 🏆 专业模型管理
│   ├── 📁 best_models/                  # ⭐ 最佳模型 (新增)
│   │   ├── stork_model_best_val_accuracy.pth    # 最佳模型权重
│   │   └── stork_model_best_val_accuracy_metadata.json  # 详细元数据
│   ├── 📁 checkpoints/                  # 💾 训练检查点
│   │   ├── stork_model_epoch_0050.pth   # 定期检查点
│   │   └── stork_model_epoch_0100.pth   # 包含优化器状态
│   └── 📁 exports/                      # 🚀 生产部署 (新增)
│       ├── stork_model.onnx             # ONNX格式 - 跨平台
│       └── stork_model_scripted.pt      # TorchScript - PyTorch生产
│
├── 📊 outputs/                          # 评估和可视化结果
│   ├── 📁 evaluation/                   # 测试集详细评估
│   │   ├── stork_model_report.html      # 🎯 可视化HTML报告
│   │   ├── stork_model_metrics.json     # 📊 结构化指标数据
│   │   ├── stork_model_summary.csv      # 📋 Excel友好摘要
│   │   ├── confusion_matrix.png         # 🔍 混淆矩阵热力图
│   │   ├── roc_curve.png               # 📈 ROC性能曲线
│   │   └── metrics_distribution.png     # 📊 指标分布图
│   │
│   ├── 📁 cross_validation/             # 交叉验证分析 (可选)
│   │   ├── cv_results.json              # 🔄 K折详细结果
│   │   ├── cv_summary.csv               # 📋 交叉验证摘要
│   │   ├── metrics_boxplot.png          # 📊 指标分布箱线图
│   │   ├── training_history.png         # 📈 训练历史对比
│   │   └── roc_curves.png               # 📈 各折ROC曲线
│   │
│   └── 📁 gradcam/                      # 模型可解释性 (可选)
│       ├── sample_001_gradcam.png       # 🎨 Grad-CAM热力图
│       └── attention_analysis.png       # 🧠 注意力机制分析
│
├── 📝 logs/                             # 训练过程记录
│   ├── training.log                     # 详细训练日志
│   └── 📁 tensorboard/                  # TensorBoard可视化
│       └── events.out.tfevents.*        # 训练过程监控
│
└── 💾 batch_cache/                      # 🚀 智能缓存系统 (v2.1)
    ├── train_chunk_0000.pt              # 训练集缓存块
    ├── train_chunk_0001.pt              # 每块1000张图片
    ├── test_chunk_0000.pt               # 测试集缓存块
    ├── metadata.json                    # 缓存元数据
    └── index.json                       # 🔑 缓存索引键
```

### 分布式训练

**v4.0.0 自动多GPU配置 (推荐)：**
```bash
# 简单启动，自动使用所有可用GPU
python main.py --config config/my_config.yaml
# 系统会自动检测GPU数量并配置分布式训练
```

**传统手动配置方式：**
```bash
# 单机多GPU
python -m torch.distributed.launch --nproc_per_node=2 main.py --config config/distributed.yaml

# 多机多GPU
python -m torch.distributed.launch --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12345 --nproc_per_node=2 main.py
```

### 模型导出

训练完成后，模型会自动导出为多种格式：

```python
# 手动导出
from utils import create_model_manager

model_manager = create_model_manager("./models", "my_model")
model_manager.export_model(model, (3, 224, 224), "onnx")
model_manager.export_model(model, (3, 224, 224), "torchscript")
```

### 交叉验证

```yaml
# 在配置文件中启用
validation:
  cross_validation: true

cross_validation:
  n_folds: 5
  stratified: true
```

### Grad-CAM可视化

```yaml
# 在配置文件中启用
evaluation:
  generate_gradcam: true
  gradcam_samples: 20
```

### 🧠 智能缓存机制
```
./batch_cache/
├── train_chunk_0000.pt      # 1000张预处理图片 (uint8格式)
├── train_chunk_0001.pt      # 智能分块，按需加载
├── test_chunk_0000.pt       # 测试集独立缓存
├── metadata.json            # 📊 缓存统计和版本信息
└── index.json              # 🔑 智能缓存键索引 (v2.1)
```

### 💡 缓存策略优化

**缓存复用条件** (✅ 复用):
```yaml
# 相同配置 - 智能复用
data_dir: "/liaozizhuo/selected_images/train/"  # 相同
img_size: 224                                   # 相同
apply_petri_crop: true                          # 相同
apply_center_crop: true                         # 相同
train_split: 0.7                               # 相同
random_state: 42                               # 相同

# 以下变化不影响缓存复用：
model: "vgg16" → "resnet50"                    # ✅ 模型无关
batch_size: 512 → 256                         # ✅ 训练参数无关
learning_rate: 0.0001 → 0.001                 # ✅ 优化参数无关
```

**缓存重建条件** (🔄 重建):
```yaml
img_size: 224 → 256                           # 🔄 尺寸变化
data_dir: "/path1/" → "/path2/"               # 🔄 数据变化
apply_petri_crop: true → false                # 🔄 预处理变化
train_split: 0.7 → 0.8                       # 🔄 分割比例变化
```

## 📚 更多信息

- **技术支持**: 检查 `logs/` 目录中的详细日志
- **性能调优**: 参考 `optimization/` 模块的DeepSpeed配置
- **模型可解释性**: 使用Grad-CAM功能分析模型决策过程
- **实验追踪**: 启用TensorBoard或Weights & Biases进行实验管理

## 🤝 贡献指南

欢迎提交问题和改进建议！请确保：
1. 代码遵循项目的编码规范
2. 添加适当的测试和文档
3. 提交前运行完整的测试流程

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📈 **版本历史**

### v2.1 (2024-12-XX) - 智能缓存与模型管理革命
- ✨ **智能缓存系统**: 跨运行缓存复用，99%时间节省
- 🏆 **专业模型管理**: 自动最佳模型保存和生产部署导出
- 📦 **多格式导出**: ONNX、TorchScript自动生成
- 🔧 **稳定缓存键**: 基于配置参数的稳定缓存机制
- ⚡ **性能优化**: 大幅提升训练启动和实验迭代速度

### v2.0 (2024-11-XX) - PyTorch原生重构
- 🔄 **完全PyTorch化**: 从TensorFlow迁移到PyTorch
- 🚀 **分布式训练**: 自动多GPU配置和DeepSpeed优化
- 📊 **专业评估**: 详细指标报告和可视化
- 🎨 **Grad-CAM可视化**: 模型可解释性分析

---

**STORK PyTorch v2.1 系统专为胚胎分类优化，集成了最新的深度学习技术、智能缓存系统和专业的医学图像处理算法。现在支持极速实验迭代和生产级部署。** 🧬✨🚀