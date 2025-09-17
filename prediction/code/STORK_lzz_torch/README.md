# STORK PyTorch - 胚胎分类深度学习系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.9+-green.svg)](https://deepspeed.ai/)

STORK (Specialized Training and Optimization for Reproductive Knowledge) PyTorch版本是一个专为胚胎可行性分类设计的深度学习系统。该系统集成了专业的胚胎图像预处理、多模型支持、分布式训练优化和全面的评估可视化功能。

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
cd /path/to/your/project
cd STORK_lzz_torch

# 运行环境设置脚本
python setup.py
```

或手动安装依赖：

```bash
# 安装PyTorch (根据你的CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 安装GPU数据增强依赖 (推荐)
pip install kornia  # GPU加速数据增强
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

复制并修改默认配置：

```bash
# 复制默认配置
cp config/default.yaml config/my_config.yaml

# 编辑配置文件
vim config/my_config.yaml
```

**关键配置项：**

```yaml
# 数据配置
data:
  data_dir: "/path/to/your/data"  # 修改为你的数据路径
  img_size: 224
  train_split: 0.7
  use_cache: true
  num_workers: 8                  # 根据你的CPU核心数调整

# 模型配置
model:
  name: "vgg16"                   # 可选: vgg16, resnet50, resnet152, efficientnetb0

# 训练配置
training:
  epochs: 100                     # 根据需要调整
  batch_size: 8                   # 根据GPU内存调整
  learning_rate: 0.0001
```

### 4. 开始训练

```bash
# 使用默认配置训练（自动多GPU）
python main.py

# 使用自定义配置
python main.py --config config/my_config.yaml

# 调试模式（显示详细配置）
python main.py --config config/my_config.yaml --debug

# 仅验证配置（不实际训练）
python main.py --config config/my_config.yaml --dry-run

# 常用多GPU分布式训练命令（推荐用于生产环境）
PYTHONPATH=/opt/conda/lib/python3.10/site-packages:$PYTHONPATH torchrun --nproc_per_node=2 main.py --config config/my_config.yaml
```

**🚀 多GPU自动配置 (v4.0.0新功能)：**
- 系统自动检测并使用所有可用GPU进行分布式训练
- 无需手动配置分布式环境变量或启动命令
- 支持2×A800-SXM4-80GB等高性能GPU配置
- 自动优化批次大小和内存使用

## 📊 系统功能

### 🔬 STORK专业预处理
- **培养皿自动检测** - 使用霍夫圆检测算法自动识别和裁剪培养皿区域
- **中心方形裁剪** - 智能裁剪图像中心区域，专为胚胎图像优化
- **渐进式缓存** - 智能缓存机制，第一epoch边训练边缓存，后续epoch显著加速
- **按需加载** - 避免内存溢出，支持大规模数据集训练
- **动态数据增强** - 训练时实时应用随机变换，提升模型泛化能力

### 🤖 多模型支持
- **VGG16** - 经典卷积神经网络，适合胚胎图像特征提取
- **ResNet50/152** - 残差网络，更深层次的特征学习
- **EfficientNet** - 高效网络架构，平衡准确性和计算效率
- **微调策略** - 支持层级冻结和渐进式微调

### ⚡ 性能优化
- **智能多GPU配置** - 自动检测并配置所有可用GPU，无需手动分布式设置
- **DeepSpeed集成** - ZeRO Stage 2内存优化，支持更大批次训练
- **混合精度训练** - FP16训练，减少内存使用，提升训练速度
- **分布式训练** - 自动NCCL后端配置，支持多GPU并行训练
- **梯度累积** - 模拟更大批次训练

### 📈 评估和可视化
- **交叉验证** - K折交叉验证，更可靠的模型评估
- **详细指标报告** - 准确率、精确率、召回率、F1分数、AUC等
- **Grad-CAM可视化** - 模型决策可解释性分析
- **HTML报告生成** - 专业的评估报告，包含图表和统计信息

## 📁 项目结构

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

## ⚙️ 配置文件详解

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

## 🖥️ 命令行参数

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

## 📁 输出文件结构

训练完成后，系统会生成以下文件：

```
outputs/
├── models/                     # 模型文件
│   ├── checkpoints/           # 训练检查点
│   ├── best_models/           # 最佳模型
│   └── exports/               # 导出模型 (ONNX, TorchScript)
├── logs/                      # 日志文件
│   ├── training.log           # 训练日志
│   └── tensorboard/           # TensorBoard日志
├── evaluation/                # 评估结果
│   ├── metrics.json           # 详细指标
│   ├── report.html            # HTML报告
│   ├── confusion_matrix.png   # 混淆矩阵
│   └── roc_curve.png          # ROC曲线
├── gradcam/                   # Grad-CAM可视化
│   ├── sample_001.png         # 可视化样本
│   └── attention_analysis.png # 注意力分析
├── cross_validation/          # 交叉验证结果 (如果启用)
│   ├── cv_results.json
│   └── fold_analysis.png
└── preprocessed_cache/        # 渐进式预处理缓存
    ├── train/                 # 训练集缓存
    │   ├── preprocessed_a1b2c3.npy
    │   └── ...
    └── test/                  # 测试集缓存
        ├── preprocessed_d4e5f6.npy
        └── ...
```

## 🔧 高级功能

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

## 🚨 常见问题

### Q: GPU内存不足怎么办？

A: 尝试以下解决方案：
1. 减小 `batch_size`
2. 启用 `mixed_precision: true`
3. 启用 `gradient_checkpointing: true`
4. 使用 `zero_stage: 2` 或 `zero_stage: 3`

## 性能优化

**数据处理优化**：
- 批量缓存系统：自动启用，19万图片→193个文件，加载速度提升5-10倍
- GPU数据增强：启用 `use_gpu_augmentation: true`，Kornia加速数据增强
- STORK专业预处理：培养皿检测+中心裁剪，针对胚胎图像优化

**系统性能优化**：
- 多进程加载：增加 `num_workers` (推荐CPU核心数的1-2倍)
- 存储优化：使用SSD存储缓存目录，提升I/O性能
- 内存优化：启用 `pin_memory: true` 和 `persistent_workers: true`

**训练优化**：
- 混合精度：启用 `mixed_precision: true`，节省GPU内存
- 批次优化：在GPU内存允许范围内增加 `batch_size`
- DeepSpeed：大模型使用 `use_deepspeed: true` 分布式训练

### Q: 如何处理类别不平衡？

A: 配置类别权重：
```yaml
training:
  loss: "focal"  # 或 "weighted_cross_entropy"
  class_weights: [0.5, 2.0]  # 根据你的数据调整
```

### Q: 如何恢复中断的训练？

A: 设置 `resume_training: true`，系统会自动查找最新的检查点。

## 批量缓存系统

STORK使用**BatchCacheManager**，专为大规模胚胎图像训练优化：

**存储结构**：
```
./batch_cache/
├── train_chunk_0000.pt      # 1000张训练图片
├── train_chunk_0001.pt      # 1000张训练图片
├── test_chunk_0000.pt       # 1000张测试图片
├── metadata.json            # 缓存元数据
└── index.json              # 图片路径索引
```

**工作原理**：
- 首次训练时图片分块预处理（STORK管道：培养皿检测→中心裁剪→归一化）
- 每1000张图片保存为一个`.pt`文件，使用PyTorch压缩
- 后续训练直接加载预处理好的张量，跳过图像处理步骤
- 基于图片路径+预处理参数生成唯一缓存键

**性能提升**：
- 文件数量：19万张图片 → 193个文件（减少99.9%）
- 加载速度：5-10倍提升
- 存储效率：PyTorch原生压缩，节省空间
- 跨模型复用：VGG16/ResNet/EfficientNet可共享同一缓存

**缓存管理**：
```bash
# 清理缓存（重新生成时使用）
rm -rf ./batch_cache/
```

**缓存配置**：
```yaml
data:
  cache_chunk_size: 1000         # 每个缓存文件的图片数量

paths:
  cache_dir: "./batch_cache"     # 缓存目录位置
```

**配置建议**：
- 1000张/块（推荐）：平衡文件数量和加载效率
- 更大块（2000+）：文件更少，但内存占用更高
- 更小块（500-）：内存友好，但文件数量增加
- SSD存储：显著提升缓存读写速度
- 大容量磁盘：19万图片约需15-20GB空间

### Q: GPU数据增强需要什么依赖？

A: 安装Kornia库：
```bash
pip install kornia
```
如果没有安装Kornia，系统会自动回退到CPU数据增强。

### Q: 如何使用自己的预处理方法？

A: 可以修改 `data/stork_preprocessing.py` 中的预处理函数，或在配置中禁用STORK预处理：
```yaml
data:
  apply_petri_crop: false
  apply_center_crop: false
```

## 📊 性能基准

### 硬件要求
- **最低配置**: 8GB GPU内存, 16GB系统内存
- **推荐配置**: 16GB+ GPU内存, 32GB+ 系统内存
- **最佳配置**: 多GPU (24GB+), 64GB+ 系统内存

### 训练时间参考
- **小数据集** (1000张): ~30分钟 (单GPU)
- **中等数据集** (10000张): ~3小时 (单GPU)
- **大数据集** (50000张): ~12小时 (单GPU)

*注：使用DeepSpeed和混合精度训练可显著减少训练时间*

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

---

**STORK PyTorch系统专为胚胎分类优化，集成了最新的深度学习技术和专业的医学图像处理算法。** 🧬✨