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

```

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


### 🧠 智能缓存机制
```
./batch_cache/
├── train_chunk_0000.pt      # 1000张预处理图片 (uint8格式)
├── train_chunk_0001.pt      # 智能分块，按需加载
├── test_chunk_0000.pt       # 测试集独立缓存
├── metadata.json            # 📊 缓存统计和版本信息
└── index.json              # 🔑 智能缓存键索引 (v2.1)
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