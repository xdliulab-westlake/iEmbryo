# STORK PyTorch - 胚胎分类深度学习系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.9+-green.svg)](https://deepspeed.ai/)
[![Version](https://img.shields.io/badge/version-v2.1-green.svg)](https://github.com/your-repo/stork)

STORK (Specialized Training and Optimization for Reproductive Knowledge) PyTorch版本是一个专为胚胎可行性二分类设计的深度学习系统。该系统集成了专业的胚胎图像预处理、多模型支持、分布式训练优化和全面的评估可视化功能。

## 🚀 5分钟快速入门

```bash
# 1. 进入项目目录
cd /Users/lzz/Documents/GitHub/IVF_AI/prediction/code/STORK_lzz_torch

# 2. 复制配置文件并修改
cp config/default.yaml config/my_config.yaml
# 编辑 my_config.yaml 中的 data_dir 为你的数据路径

# 3. 开始训练
python main.py --config config/my_config.yaml

# 4. 查看结果
ls outputs/          # 训练结果
tensorboard --logdir logs/tensorboard  # 可视化
```

**就这么简单！** 系统会自动处理GPU配置、缓存创建、模型训练等所有细节。

---

## 一、环境安装

### 方法一：自动安装（推荐）

```bash
cd /Users/lzz/Documents/GitHub/IVF_AI/prediction/code/STORK_lzz_torch
python setup.py
```

### 方法二：手动安装

```bash
pip install -r requirements.txt
```

---

## 二、数据准备

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

### 数据增强

训练集应用以下增强策略，验证集和测试集不应用：

| 增强方式 | 说明 | 参数 |
|--------|------|------|
| **水平翻转** | Horizontal Flip | p=0.5 |
| **竖直翻转** | Vertical Flip | p=0.5 |
| **颜色抖动** | ColorJitter | brightness=0.2, contrast=0.2 |
| **随机旋转** | Random Rotation | 角度范围: ±15° |

---

## 三、配置文件

### 配置文件模板

```yaml
# my_config.yaml
data:
  data_dir: "/path/to/your/embryo/data"  # 你的数据目录
  img_size: 224                          # 输入图像尺寸
  train_split: 0.7                       # 训练集比例
  val_split: 0.15                        # 验证集比例
  num_workers: 8                         # CPU工作进程数

  # 性能优化
  cache_chunk_size: 1000                 # 缓存块大小

  # STORK专业预处理
  apply_petri_crop: true                 # 培养皿检测裁剪
  apply_center_crop: true                # 中心方形裁剪

model:
  name: "vgg16"                          # 可选: vgg16, resnet50, resnet152, efficientnetb0, densenet121等
  pretrained: true                       # 使用预训练权重
  fine_tune: true                        # 启用微调

training:
  epochs: 100                            # 训练轮数（建议100-2000）
  batch_size: 32                         # 批次大小（根据GPU内存调整）
  learning_rate: 0.0001                  # 初始学习率
  weight_decay: 0.0001                   # L2正则化

  # 类别不平衡处理
  loss: "focal"                          # focal loss处理不平衡数据
  class_weights: [0.669, 1.978]          # 类别权重

  # 优化器和学习率
  optimizer: "adamw"                     # adamw, sgd
  scheduler: "cosine"                    # cosine, linear, step

paths:
  output_dir: "./outputs"                # 输出目录
  cache_dir: "./batch_cache"             # 缓存目录
  checkpoint_dir: "./models/checkpoints" # 检查点目录
```

### 配置参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `img_size` | 224 | VGG/ResNet标准输入尺寸 |
| `train_split` | 0.7 | 训练集占比 |
| `batch_size` | 32 | 批次大小（V100: 16-32, A100: 32-64） |
| `learning_rate` | 1e-4 | 初始学习率 |
| `loss` | focal | BCE 或 focal loss |

---

## 四、模型架构

### 支持的模型

| 模型 | 输出维度 | 参数量 | 推荐场景 |
|------|--------|--------|--------|
| **VGG16** | 4096 | ~138M | 经典特征提取，胚胎图像处理 |
| **ResNet50** | 2048 | ~25M | 残差网络，中等复杂度 |
| **ResNet152** | 2048 | ~60M | 深层网络，更强表征能力 |
| **InceptionV3** | 2048 | ~27M | 多尺度特征，复杂模式识别 |
| **EfficientNetB0** | 1280 | ~5.3M | 轻量级，高效计算 |
| **DenseNet121** | 1024 | ~7.9M | 密集连接，特征重用 |
| **DenseNet169** | 1664 | ~12.6M | 中等密集网络 |
| **DenseNet201** | 1920 | ~18.4M | 深层密集网络 |

### VGG16 架构详解

```
输入: (batch, 3, 224, 224)
  ↓
Features (13层卷积块):
  ├─ Block1 (2×Conv3x3-64) + MaxPool
  ├─ Block2 (2×Conv3x3-128) + MaxPool
  ├─ Block3 (3×Conv3x3-256) + MaxPool
  ├─ Block4 (3×Conv3x3-512) + MaxPool
  ├─ Block5 (3×Conv3x3-512) + MaxPool
  ↓
  输出: (batch, 512, 7, 7)

AdaptiveAvgPool2d(7, 7)
  ↓
  (batch, 512, 7, 7) → flatten → (batch, 25088)

Classifier (3层全连接):
  ├─ Linear(25088, 4096) + ReLU + Dropout(0.5)
  ├─ Linear(4096, 4096) + ReLU + Dropout(0.5)
  └─ Linear(4096, 2)  [二分类]
  ↓
输出: (batch, 2) → Softmax → [p_viable, p_non_viable]
```

**参数说明：**
- 总参数: ~138M
- 可训练参数 (微调): ~50M（约36%）
- 输出维度: 4096 (分类前)

### ResNet50 架构详解

```
输入: (batch, 3, 224, 224)
  ↓
Conv7x7 + BatchNorm + ReLU (stride=2)
  ↓
  (batch, 64, 112, 112)

MaxPool3x3 (stride=2)
  ↓
  (batch, 64, 56, 56)

残差层:
  ├─ Layer1 (3×残差块, channels=64) → (batch, 256, 56, 56)
  ├─ Layer2 (4×残差块, channels=128, stride=2) → (batch, 512, 28, 28)
  ├─ Layer3 (6×残差块, channels=256, stride=2) → (batch, 1024, 14, 14)
  ├─ Layer4 (3×残差块, channels=512, stride=2) → (batch, 2048, 7, 7)
  ↓
AdaptiveAvgPool2d(1, 1)
  ↓
  (batch, 2048)

分类头:
  └─ Linear(2048, 2)
  ↓
输出: (batch, 2) → Softmax
```

**参数说明：**
- 总参数: ~25M
- 可训练参数 (微调): ~10M
- 输出维度: 2048 (池化后)

### EfficientNetB0 架构详解

```
输入: (batch, 3, 224, 224)
  ↓
Stem (Conv3x3)
  ├─ Conv(3, 32)
  ↓
  (batch, 32, 112, 112)

Mobile Inverted Bottleneck (9个Block):
  ├─ Block1: 32 → 16 (stride=1) × 1
  ├─ Block2: 16 → 24 (stride=2) × 2
  ├─ Block3: 24 → 40 (stride=2) × 2
  ├─ Block4: 40 → 80 (stride=2) × 3
  ├─ Block5: 80 → 112 (stride=1) × 3
  ├─ Block6: 112 → 192 (stride=2) × 4
  ├─ Block7: 192 → 320 (stride=1) × 1
  ↓
  (batch, 1280, 7, 7)

Head:
  ├─ Conv1x1 → 1280
  ├─ AdaptiveAvgPool2d
  ├─ Linear(1280, 2)
  ↓
输出: (batch, 2) → Softmax
```

**参数说明：**
- 总参数: ~5.3M （轻量级）
- 可训练参数: ~2.1M
- 输出维度: 1280
- MACCs: ~0.4B （高效）

---

## 五、微调策略

### VGG16微调

**冻结策略：** 冻结 Block 1-2 (低阶特征)，训练 Block 3-5 + 分类头

```python
# 冻结 features.0-9 (Block 1-2)
for param in model.features[0:10].parameters():
    param.requires_grad = False

# 训练 features.10-30 (Block 3-5) + classifier
for param in model.features[10:].parameters():
    param.requires_grad = True
```

### ResNet50微充

**冻结策略：** 冻结 Layer1，训练 Layer2-4 + 分类头

```python
# 冻结 layer1
for param in model.layer1.parameters():
    param.requires_grad = False

# 训练 layer2-4 + fc
for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
```

---

## 六、训练过程

### 常用训练命令

```bash
# 1. 快速实验（小数据集）
python main.py --config config/my_config.yaml --epochs 50 --batch-size 16

# 2. 生产训练
python main.py --config config/my_config.yaml --epochs 2000 --batch-size 64

# 3. 更换模型
python main.py --model resnet50 --epochs 100

# 4. 指定数据目录
python main.py --data-dir "/path/to/data" --epochs 100

# 5. 多GPU训练（自动检测）
python main.py --config config/my_config.yaml --use-deepspeed
```

### 训练监控

```bash
# 实时查看训练进度
tensorboard --logdir logs/tensorboard --port 6006
```

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

---

## 七、系统功能

### STORK 专业预处理

- **培养皿自动检测** - 使用Hough圆检测算法自动识别和裁剪培养皿区域
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

- **智能多GPU配置** - 自动检测并配置所有可用GPU
- **DeepSpeed集成** - ZeRO Stage 2内存优化
- **混合精度训练** - FP16训练，减少内存使用
- **分布式训练** - 自动NCCL后端配置
- **梯度累积** - 模拟更大批次训练

### 评估和可视化

- **交叉验证** - K折交叉验证，更可靠的模型评估
- **详细指标报告** - 准确率、精确率、召回率、F1分数、AUC等
- **Grad-CAM可视化** - 模型决策可解释性分析
- **HTML报告生成** - 专业的评估报告，包含图表和统计信息

---

## 八、项目结构

```
STORK_lzz_torch/
├── config/                     # 配置管理
│   ├── default.yaml           # 默认配置文件
│   └── config_manager.py      # 配置管理器
├── data/                       # 数据处理模块
│   ├── dataset.py             # PyTorch Dataset
│   ├── transforms.py          # 数据增强
│   ├── stork_preprocessing.py # STORK专业预处理
│   └── parallel_processing.py # 并行处理和缓存
├── models/                     # 模型定义
│   ├── backbone.py            # 多模型支持
│   └── best_models/           # 最佳模型保存
├── training/                   # 训练模块
│   └── trainer.py             # 训练器主类
├── optimization/               # 优化模块
│   └── deepspeed_config.py    # DeepSpeed配置
├── evaluation/                 # 评估模块
│   ├── cross_validation.py    # 交叉验证
│   └── metrics.py             # 评估指标和报告
├── visualization/              # 可视化模块
│   └── gradcam.py             # Grad-CAM实现
├── utils/                      # 工具模块
│   ├── model_manager.py       # 模型管理
│   └── logger.py              # 日志和监控
├── main.py                     # 主程序入口
├── setup.py                    # 环境设置脚本
├── requirements.txt            # 依赖列表
└── README.md                   # 项目文档
```

---

## 九、评估指标

| 指标 | 定义 | 解释 |
|------|------|------|
| **Accuracy** | (TP+TN)/(总数) | 整体分类准确率 |
| **Precision** | TP/(TP+FP) | "正样本预测正确的比例" |
| **Recall** | TP/(TP+FN) | "真实正样本被识别的比例" |
| **F1-Score** | 2·(P·R)/(P+R) | Precision和Recall的调和平均 |
| **AUC** | ROC曲线下面积 | 不同阈值下的综合性能 |
| **Specificity** | TN/(TN+FP) | "真实负样本被识别的比例" |

---

## 十、输出文件结构

训练完成后，系统会生成完整文件结构：

```
📁 项目根目录/
├── 📁 models/
│   ├── 📁 best_models/                  # ⭐ 最佳模型
│   │   ├── vgg16_best_val_accuracy.pth
│   │   └── vgg16_best_val_accuracy_metadata.json
│   ├── 📁 checkpoints/                  # 💾 训练检查点
│   │   ├── stork_model_epoch_0050.pth
│   │   └── stork_model_epoch_0100.pth
│   └── 📁 exports/                      # 🚀 生产部署
│       ├── stork_model.onnx
│       └── stork_model_scripted.pt
│
├── 📁 outputs/                          # 评估和可视化
│   ├── 📁 evaluation/                   # 测试集详细评估
│   │   ├── stork_model_report.html      # 可视化HTML报告
│   │   ├── stork_model_metrics.json     # 结构化指标数据
│   │   ├── confusion_matrix.png         # 混淆矩阵热力图
│   │   ├── roc_curve.png                # ROC性能曲线
│   │   └── metrics_distribution.png     # 指标分布图
│   │
│   ├── 📁 cross_validation/             # 交叉验证分析（可选）
│   │   ├── cv_results.json
│   │   ├── cv_summary.csv
│   │   ├── metrics_boxplot.png
│   │   ├── training_history.png
│   │   └── roc_curves.png
│   │
│   └── 📁 gradcam/                      # 模型可解释性（可选）
│       ├── sample_001_gradcam.png
│       └── attention_analysis.png
│
├── 📁 logs/                             # 训练过程记录
│   ├── training.log
│   └── 📁 tensorboard/
│       └── events.out.tfevents.*
│
└── 📁 batch_cache/                      # 🚀 智能缓存系统
    ├── train_chunk_0000.pt
    ├── train_chunk_0001.pt
    ├── test_chunk_0000.pt
    ├── metadata.json
    └── index.json
```

---

## 十一、更多信息

- **技术支持**: 检查 `logs/` 目录中的详细日志
- **性能调优**: 参考 `optimization/` 模块的DeepSpeed配置
- **模型可解释性**: 使用Grad-CAM功能分析模型决策过程
- **实验追踪**: 启用TensorBoard进行实验管理

---

## 十二、版本历史

### v2.1 (2024-12)
**关键改进：**
- ✨ 智能缓存系统 - 跨运行缓存复用，99%时间节省
- 🏆 专业模型管理 - 自动最佳模型保存和生产部署导出
- 📦 多格式导出 - ONNX、TorchScript自动生成
- 🔧 稳定缓存键 - 基于配置参数的稳定缓存机制
- ⚡ 性能优化 - 大幅提升训练启动和实验迭代速度

### v2.0 (2024-11)
- 🔄 完全PyTorch化 - 从TensorFlow迁移到PyTorch
- 🚀 分布式训练 - 自动多GPU配置和DeepSpeed优化
- 📊 专业评估 - 详细指标报告和可视化
- 🎨 Grad-CAM可视化 - 模型可解释性分析

---

**STORK PyTorch v2.1 - 为胚胎分类优化，集成最新深度学习技术和专业医学图像处理算法。** 🧬✨🚀
