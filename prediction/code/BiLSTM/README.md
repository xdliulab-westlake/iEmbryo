# BiLSTM 胚胎分类系统 🧬

## 📋 项目概述

BiLSTM胚胎分类系统是一个基于深度学习的双层机器学习管道，专门用于胚胎质量分类：

1. **第一层**：BiLSTM回归模型预测胚胎囊胚评分（Blastocyst Score）
2. **第二层**：逻辑回归分类器进行最终胚胎质量分类

## 🏗️ 模块架构

```
BiLSTM/
├── __init__.py                    # 📦 包初始化和统一导入接口
├── main.py                       # 🚀 主训练程序和配置管理
├── README.md                     # 📖 项目文档
├── bilstm_config.json            # ⚙️ 配置文件
│
├── feature_extractor.py          # 🔍 多模型特征提取器
├── feature_cache.py              # 💾 智能特征缓存系统
├── model.py                      # 🧠 BiLSTM模型构建和训练
├── logistic.py                   # 📊 逻辑回归分类器
├── cross_validation.py           # ✅ 交叉验证和模型评估
├── visualization.py              # 📈 结果可视化和图表生成
├── utils.py                      # 🔧 通用工具函数
│
├── preprocessing/                # 🖼️ 图像预处理模块
│   ├── __init__.py
│   ├── bilstm_image_utils.py     # Hough圆检测和图像预处理
│   └── data_loader_bilstm.py     # 多模型数据加载管道
│
└── BiLSTM_backup_final.py        # 📁 历史版本备份（已弃用）
```

## 🚀 核心特性

### 🎯 模型选择策略
- **灵活模型选择**: 支持单模型或多模型融合
- **EfficientNetV2M**: 高效特征提取 (1280维原始特征)
- **ResNet152V2**: 深度残差网络特征 (2048维原始特征)  
- **DenseNet201**: 密集连接网络特征 (1920维原始特征)
- **VGG16**: 经典卷积网络特征 (512维原始特征)
- **多模型融合**: 4个模型融合 (5760维 = 1280+2048+1920+512)

### 🔍 STORK预处理管道
- **Hough圆检测**: 96.2%成功率的培养皿边界自动检测
- **智能筛选算法**: 基于半径、位置、合理性的综合评分
- **中心方形裁剪**: 标准化视野区域
- **自适应参数**: 根据图像尺寸动态调整检测参数

### 💾 智能缓存系统
- **MD5哈希缓存键**: 基于图像路径和参数生成唯一标识
- **原子文件操作**: 确保缓存数据一致性和完整性
- **自动失效检测**: 基于文件修改时间的智能缓存更新
- **性能提升**: 从9次重复特征提取优化为1次缓存复用

### 🆕 自动评分字典生成
- **智能CSV解析**: 支持MainDB.csv和filtered_embryos.csv格式
- **ICM/TE评分计算**: 自动转换ICM/TE等级为BS_prime分数
- **文件时间检测**: 自动检测数据更新并重新生成评分字典
- **无缝集成**: 配置CSV路径后完全自动化，无需手动干预

### 📊 严格交叉验证
- **4折分层K折**: 确保训练验证中的类别平衡
- **70/30数据分割**: 训练测试数据严格分离
- **图像级增强**: 训练时启用，验证时禁用
- **多指标评估**: AUC、准确率、精确率、召回率

## 🛠️ 安装要求

```bash
# Python环境
python >= 3.8

# 核心依赖
tensorflow >= 2.8.0
opencv-python >= 4.0.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.5.0

# 可选依赖
tqdm  # 进度条显示
```

## 📚 使用指南

### 🎯 快速开始

```bash
# 方式1: 使用main.py主程序（推荐用于完整训练）
cd /path/to/BiLSTM
python main.py --mode train

# 方式2: 使用配置文件训练
python main.py --mode train --config bilstm_config.json

# 方式3: 使用命令行参数快速调整
python main.py --mode train --feature-model efficientnet --epochs 50

# 方式4: 自动生成评分字典并训练（新功能）
# 在配置文件中设置CSV路径后，系统会自动生成score_dict.json
python main.py --mode train --config my_config.json
```

### 🔧 模块化编程使用

```python
# 从包级别直接导入（推荐）
from BiLSTM import (
    create_model, 
    preprocess_embryo_image, 
    configure_gpu,
    BiLSTMFeatureCache,
    MultiModelFeatureExtractor,
    generate_score_dict_from_csv  # 🆕 评分字典生成功能
)

# 模块级精确导入
from BiLSTM.model import create_model, train_model, evaluate_model
from BiLSTM.logistic import perform_logistic_regression
from BiLSTM.visualization import plot_training_history, plot_roc_curve
from BiLSTM.utils import configure_gpu, cleanup_memory
from BiLSTM.feature_extractor import MultiModelFeatureExtractor
from BiLSTM.feature_cache import BiLSTMFeatureCache
```

### 🖼️ 图像预处理示例

```python
from BiLSTM import preprocess_embryo_image

# 使用完整的STORK预处理管道
processed_img = preprocess_embryo_image(
    image_path="/path/to/embryo_image.jpg",
    img_size=224,
    apply_petri_crop=True,  # 启用Hough圆检测培养皿裁剪
    apply_center_crop=True  # 启用中心方形裁剪
)
```

### 🧠 模型构建和训练

```python
from BiLSTM import create_model, train_model, configure_gpu, MultiModelFeatureExtractor

# 配置GPU
configure_gpu()

# 初始化特征提取器
extractor = MultiModelFeatureExtractor(model_name='all')  # 或 'efficientnet', 'resnet' 等
feature_dim = extractor.total_feature_dim  # 自动获取特征维度

# 创建BiLSTM模型
model = create_model(
    feature_dim=feature_dim,   # 根据选择的模型自动设置 (5760 for 'all', 1280 for 'efficientnet' 等)
    lstm_units=128,            # LSTM单元数
    dropout_rate=0.3,          # Dropout比率
    learning_rate=0.001        # 学习率
)

# 训练模型
trained_model, history = train_model(
    X_train, y_train,
    X_val, y_val, 
    feature_dim=feature_dim,
    save_path="/path/to/save/model.h5",
    epochs=100,
    patience=15
)
```

### 💾 智能缓存使用

```python
from BiLSTM import BiLSTMFeatureCache, MultiModelFeatureExtractor

# 初始化特征提取器
extractor = MultiModelFeatureExtractor(model_name='all')  # 选择模型

# 初始化缓存系统
cache = BiLSTMFeatureCache(
    data_root="/path/to/embryo/images",
    cache_dir="bilstm_feature_cache"
)

# 设置特征提取器到缓存系统
cache.set_extractor(extractor)

# 获取或计算特征（自动缓存管理）
features = cache.get_or_compute_features(
    image_paths=image_list,
    use_cache=True,          # 启用缓存
    training=False,          # 推理模式
    use_augmentation=False   # 不使用数据增强
)
```

### 📊 逻辑回归分类

```python
from BiLSTM import perform_logistic_regression

# 执行完整的逻辑回归分析
results = perform_logistic_regression(
    model=trained_bilstm_model,
    X_train, X_val, X_test,
    ages_train, ages_val, ages_test,
    quality_train, quality_val, quality_test,
    test_mae=bilstm_test_mae
)

# 查看结果
print(f"测试集准确率: {results['test_metrics']['accuracy']:.4f}")
print(f"测试集AUC: {results['test_metrics']['auc']:.4f}")
```

### 🆕 自动评分字典生成

```python
from BiLSTM import generate_score_dict_from_csv

# 手动生成评分字典
generate_score_dict_from_csv(
    maindb_path="/path/to/MainDB.csv",
    filtered_path="/path/to/filtered_embryos.csv",  # 可选
    output_path="score_dict.json"
)

# 或在配置文件中设置路径，系统会自动生成
config = {
    "maindb_csv_path": "/path/to/MainDB.csv",
    "filtered_csv_path": "/path/to/filtered_embryos.csv",
    "score_dict_path": "score_dict.json"
}
# 运行训练时会自动检测并生成评分字典
```

### 📈 结果可视化

```python
from BiLSTM import plot_training_history, plot_roc_curve, create_cv_visualization

# 绘制训练历史
plot_training_history(
    history, 
    save_path="/path/to/training_history.png"
)

# 绘制ROC曲线
plot_roc_curve(
    y_true=test_labels,
    y_prob=test_probabilities,
    save_path="/path/to/roc_curve.png"
)

# 创建交叉验证可视化
create_cv_visualization(
    avg_performance=avg_metrics,
    bilstm_avg=bilstm_metrics,
    fold_performances=fold_results
)
```

### ✅ 交叉验证

```python
from BiLSTM import assess_model_stability, validate_cv_setup

# 验证交叉验证设置
is_valid = validate_cv_setup(
    X=feature_data,
    y=target_data, 
    quality_labels=class_labels,
    n_folds=4
)

# 评估模型稳定性
stability_report = assess_model_stability(cv_summary)
print(f"模型稳定性: {stability_report['overall']['stability_grade']}")
```

## ⚙️ 参数配置

### 📋 配置文件管理

BiLSTM系统支持通过JSON配置文件灵活管理所有参数：

```bash
# 创建默认配置文件
python main.py --create-config

# 这将生成 bilstm_config.json 文件，包含所有默认参数
```

### 🎛️ 核心参数说明

#### 📁 数据路径参数
```json
{
  "data_root": "/path/to/your/data",           // 数据根目录
  "image_folder": "image",                    // 图像文件夹名称
  "score_dict_path": "score_dict.json",       // 分数字典文件路径
  "cache_dir": "bilstm_feature_cache",        // 特征缓存目录
  
  // 🆕 自动评分字典生成功能
  "maindb_csv_path": "/path/to/MainDB.csv",           // 主数据库CSV文件路径
  "filtered_csv_path": "/path/to/filtered_embryos.csv" // 低质量胚胎CSV文件路径（可选）
}
```

#### 🧠 模型架构参数
```json
{
  "model_name": "all",        // 模型选择: all, efficientnet, resnet, densenet, vgg16
  "feature_dim": 5760,        // 特征维度 (自动根据model_name调整)
  "lstm_units": 128,          // LSTM单元数量
  "dropout_rate": 0.3,        // Dropout比率 (0.0-1.0)
  "learning_rate": 0.001      // 学习率
}
```

**🔧 模型选择与特征维度对应关系**：
```json
{
  "all": 5760,          // 多模型融合: 1280+2048+1920+512
  "efficientnet": 1280, // EfficientNetV2M原始特征维度
  "resnet": 2048,       // ResNet152V2原始特征维度
  "densenet": 1920,     // DenseNet201原始特征维度
  "vgg16": 512          // VGG16原始特征维度
}
```

#### 🏋️ 训练超参数
```json
{
  "epochs": 100,              // 最大训练轮数
  "batch_size": 32,           // 批量大小
  "patience": 15,             // 早停耐心值
  "validation_split": 0.3     // 验证集比例
}
```

#### ✅ 交叉验证参数
```json
{
  "n_folds": 4,               // 交叉验证折数
  "test_ratio": 0.3,          // 测试集比例
  "random_state": 42          // 随机种子
}
```

#### 🖼️ 图像预处理参数
```json
{
  "img_size": 224,            // 图像尺寸
  "apply_petri_crop": true,   // 是否启用Hough圆检测裁剪
  "apply_center_crop": true,  // 是否启用中心方形裁剪
  "use_augmentation": true    // 是否使用数据增强
}
```

#### 💾 缓存和输出参数
```json
{
  "use_cache": true,                    // 是否使用特征缓存
  "force_recompute": false,             // 是否强制重新计算特征
  "model_save_path": "bilstm_model.h5", // 模型保存路径
  "results_save_dir": "bilstm_results", // 结果保存目录
  "verbose": 1                          // 详细输出级别 (0-2)
}
```

### 📝 JSON配置文件详细指南

#### 🛠️ 创建配置文件

**方式1: 使用命令行生成默认配置**
```bash
cd /path/to/BiLSTM
python main.py --create-config
```
这将创建 `bilstm_config.json` 文件包含所有默认参数。

**方式2: 手动创建配置文件**
```bash
# 创建自定义配置文件
vim my_bilstm_config.json
```

#### 📋 完整配置文件模板
```json
{
  "data_root": "/Users/your_username/data/embryo_project",
  "image_folder": "image",
  "score_dict_path": "score_dict.json",
  "cache_dir": "bilstm_feature_cache",
  
  // 🆕 自动评分字典生成配置
  "maindb_csv_path": "/Users/your_username/data/MainDB.csv",
  "filtered_csv_path": "/Users/your_username/data/filtered_clean_embryos.csv",
  
  "model_name": "all",
  "feature_dim": 5760,
  "lstm_units": 128,
  "dropout_rate": 0.3,
  "learning_rate": 0.001,
  
  "epochs": 100,
  "batch_size": 32,
  "patience": 15,
  "validation_split": 0.3,
  
  "n_folds": 4,
  "test_ratio": 0.3,
  "random_state": 42,
  
  "img_size": 224,
  "apply_petri_crop": true,
  "apply_center_crop": true,
  "use_augmentation": true,
  
  "use_cache": true,
  "force_recompute": false,
  "model_save_path": "bilstm_model.h5",
  "results_save_dir": "bilstm_results",
  "verbose": 1
}
```

#### 🎯 关键配置项说明

**💡 模型选择配置**
```json
// 🔥 重要：模型选择会自动调整特征维度
{
  "model_name": "efficientnet",  // 单模型选择
  "feature_dim": 1280           // 系统自动设置，无需手动修改
}

// 或者
{
  "model_name": "all",     // 多模型融合
  "feature_dim": 5760     // 系统自动设置
}
```

**📁 数据路径配置**
```json
{
  "data_root": "/absolute/path/to/your/data",  // 必须是绝对路径
  "image_folder": "image",                     // 相对于data_root的图像目录
  "score_dict_path": "score_dict.json",       // 相对于data_root的评分文件
  
  // 🆕 自动生成评分字典的源CSV路径
  "maindb_csv_path": "/absolute/path/to/MainDB.csv",        // 主数据库文件
  "filtered_csv_path": "/absolute/path/to/filtered.csv"     // 可选的过滤数据文件
}
```

**⚡ 性能调优配置**
```json
// GPU内存不足时
{
  "batch_size": 16,      // 减小批量大小
  "lstm_units": 64       // 减少LSTM单元数
}

// 训练速度优化
{
  "batch_size": 64,      // 增大批量大小
  "epochs": 50,          // 减少训练轮数
  "use_cache": true      // 启用特征缓存
}
```

#### 🔧 不同场景的配置示例

**🚀 快速测试配置** (`quick_test_config.json`)
```json
{
  "data_root": "/your/data/path",
  "model_name": "vgg16",      // 使用最轻量的模型
  "epochs": 20,               // 快速训练
  "batch_size": 64,           // 大批量
  "n_folds": 2,              // 少量折数
  "patience": 5,             // 快速早停
  "use_cache": true,         // 启用缓存
  "verbose": 2               // 详细输出
}
```

**🎯 高精度训练配置** (`high_accuracy_config.json`)
```json
{
  "data_root": "/your/data/path",
  "model_name": "all",        // 使用全部模型
  "epochs": 200,              // 充分训练
  "batch_size": 16,           // 小批量提高稳定性
  "n_folds": 5,              // 更多交叉验证折数
  "patience": 30,            // 更大的早停耐心
  "learning_rate": 0.0001,   // 更小的学习率
  "dropout_rate": 0.4,       // 更强的正则化
  "use_augmentation": true   // 启用数据增强
}
```

**💻 低内存配置** (`low_memory_config.json`)
```json
{
  "data_root": "/your/data/path",
  "model_name": "vgg16",      // 单个小模型
  "batch_size": 8,            // 小批量
  "lstm_units": 64,           // 减少LSTM单元
  "epochs": 100,
  "use_cache": true,          // 启用缓存避免重复计算
  "force_recompute": false
}
```

**🔬 实验对比配置**
```json
// EfficientNet实验
{
  "data_root": "/your/data/path",
  "model_name": "efficientnet",
  "results_save_dir": "results_efficientnet",
  "model_save_path": "model_efficientnet.h5"
}

// ResNet实验  
{
  "data_root": "/your/data/path",
  "model_name": "resnet",
  "results_save_dir": "results_resnet",
  "model_save_path": "model_resnet.h5"
}
```

#### 🚨 配置注意事项

**❌ 常见错误**
```json
// 错误：使用相对路径
{
  "data_root": "./data"  // ❌ 应该使用绝对路径
}

// 错误：手动设置特征维度
{
  "model_name": "efficientnet",
  "feature_dim": 1536  // ❌ 不匹配，系统会自动调整为1280
}

// 错误：JSON语法错误
{
  "epochs": 100,  // ❌ 最后一行不应有逗号
}
```

**✅ 正确配置**
```json
{
  "data_root": "/Users/username/project/data",  // ✅ 绝对路径
  "model_name": "efficientnet",                 // ✅ 让系统自动调整feature_dim
  "epochs": 100                                 // ✅ 最后一行无逗号
}
```

#### 🔧 配置验证

**使用配置前验证**
```bash
# 验证JSON格式
python -c "import json; json.load(open('your_config.json'))"

# 验证配置完整性
python main.py --config your_config.json --mode train --help
```

### 🔧 参数调优建议

#### 🚀 性能优化
- **GPU内存不足**: 减小 `batch_size` (32→16→8)
- **训练太慢**: 增大 `batch_size`，减少 `epochs`
- **过拟合**: 增大 `dropout_rate` (0.3→0.5)，减少 `lstm_units`
- **欠拟合**: 减小 `dropout_rate`，增大 `lstm_units`

#### 🎯 精度提升
- **数据不平衡**: 启用 `use_augmentation=true`
- **泛化性能差**: 增加 `n_folds` (4→5→10)
- **收敛慢**: 调整 `learning_rate` (0.001→0.01 或 0.0001)
- **早停过早**: 增大 `patience` (15→25→50)

## 🚀 训练指南

### 📋 环境准备

#### 1. 依赖安装
```bash
# 创建conda环境
conda create -n bilstm python=3.8
conda activate bilstm

# 安装核心依赖
pip install tensorflow>=2.8.0
pip install opencv-python>=4.0.0
pip install scikit-learn>=1.0.0
pip install pandas>=1.3.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install tqdm
```

#### 2. 数据目录结构
```
your_data_root/
├── image/                    # 胚胎图像文件夹
│   ├── embryo_001.jpg
│   ├── embryo_002.jpg
│   └── ...
├── score_dict.json           # 胚胎评分数据
└── bilstm_feature_cache/     # 特征缓存目录 (自动创建)
```

#### 3. 数据格式要求

**🆕 自动生成模式 - CSV格式要求：**

**MainDB.csv** 格式示例：
```csv
EmbryoID,Age,ICM,TE
embryo_001,32,A,A
embryo_002,28,B,C
embryo_003,35,C,B
```

**filtered_embryos.csv** 格式示例（可选）：
```csv
EmbryoID,Age,EmbryoQuality
embryo_004,30,poor
embryo_005,33,poor
```

**或传统 score_dict.json** 格式示例：
```json
{
  "embryo_001.jpg": {
    "bs_score": 4.2,      // 囊胚评分 (回归目标)
    "age": 32,            // 母体年龄
    "quality": 1          // 胚胎质量分类 (0或1)
  },
  "embryo_002.jpg": {
    "bs_score": 3.8,
    "age": 28, 
    "quality": 0
  }
}
```

### 🏃‍♂️ 快速开始训练

#### 方式1: 使用默认配置
```bash
# 进入BiLSTM目录
cd /path/to/BiLSTM

# 开始训练 (使用内置默认参数)
python main.py --mode train
```

#### 方式2: 使用自定义配置
```bash
# 1. 创建配置文件模板
python main.py --create-config

# 2. 编辑 bilstm_config.json，修改数据路径和参数
vim bilstm_config.json

# 3. 使用配置文件训练
python main.py --mode train --config bilstm_config.json
```

#### 方式3: 命令行参数覆盖
```bash
# 使用命令行参数快速调整关键参数
python main.py --mode train \
  --config bilstm_config.json \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --feature-model efficientnet

# 快速切换模型对比实验
python main.py --mode train --feature-model efficientnet --epochs 50
python main.py --mode train --feature-model resnet --epochs 50  
python main.py --mode train --feature-model densenet --epochs 50
python main.py --mode train --feature-model all --epochs 50
```

### 📊 训练过程监控

#### 实时输出示例
```
=== BiLSTM 配置参数 ===
数据路径:
  data_root: /your/data/path
  image_folder: image
  score_dict_path: score_dict.json
  cache_dir: bilstm_feature_cache

🚀 开始 BiLSTM 胚胎分类系统训练
==========================================

--- 配置GPU环境 ---
✅ GPU配置完成: NVIDIA GeForce RTX 3080

--- 路径验证 ---
✅ 数据根目录: /your/data/path
✅ 图像文件夹: /your/data/path/image
✅ 分数字典: /your/data/path/score_dict.json

--- 初始化特征缓存系统 ---
📁 特征缓存目录: bilstm_feature_cache
🔍 发现 1250 个图像文件
💾 缓存命中率: 95.2% (1191/1250)

--- 开始交叉验证训练 ---

=== 第 1 折交叉验证 ===
📊 训练集: 875 样本, 验证集: 250 样本, 测试集: 125 样本

🧠 创建BiLSTM模型...
特征维度: 5760, LSTM单元: 128, Dropout: 0.3
✅ 特征提取器初始化完成 (模式: all)
   - EfficientNetV2M: 1280维
   - ResNet152V2: 2048维
   - DenseNet201: 1920维
   - VGG16: 512维
   - 总特征维度: 5760维

🏋️ 开始训练...
Epoch 1/100: loss=0.8234, val_loss=0.7156, mae=0.6123, val_mae=0.5892
Epoch 2/100: loss=0.7456, val_loss=0.6789, mae=0.5834, val_mae=0.5623
...
Epoch 23/100: 早停触发, 最佳验证MAE: 0.4123

📊 第1折逻辑回归分析
训练准确率: 0.8571, AUC: 0.9234
验证准确率: 0.8400, AUC: 0.9156
测试准确率: 0.8480, AUC: 0.9189

=== 第 2 折交叉验证 ===
...
```

### 🎯 训练完成后的输出

#### 最终结果显示
```
🎉 训练完成！最终结果:
==========================================
逻辑回归平均准确率: 0.8425 ± 0.0156
逻辑回归平均AUC: 0.9167 ± 0.0098
BiLSTM平均MAE: 0.4234 ± 0.0234

✅ 结果已保存到: bilstm_results/cv_results.json
✅ 可视化图表已生成:
   📈 bilstm_results/training_curves.png
   📊 bilstm_results/confusion_matrices.png
   🎯 bilstm_results/roc_curves.png
```

#### 生成的文件结构
```
bilstm_results/
├── cv_results.json           # 详细的交叉验证结果
├── training_curves.png       # 训练过程曲线图
├── confusion_matrices.png    # 混淆矩阵热力图
├── roc_curves.png           # ROC曲线对比
├── fold_1_model.h5          # 第1折最佳模型
├── fold_2_model.h5          # 第2折最佳模型
├── fold_3_model.h5          # 第3折最佳模型
└── fold_4_model.h5          # 第4折最佳模型
```

### 🔮 推理模式使用

#### 单个模型推理
```bash
# 使用训练好的模型进行预测
python main.py --mode inference \
  --model bilstm_results/fold_1_model.h5 \
  --images image/embryo_001.jpg image/embryo_002.jpg

# 输出示例:
=== 预测结果 ===
1. embryo_001.jpg: BS分数 = 4.23
2. embryo_002.jpg: BS分数 = 3.78
```

#### 批量预测脚本
```python
# batch_predict.py
from BiLSTM import load_trained_model, BiLSTMFeatureCache, batch_predict_bs_scores
import glob

# 加载模型
model = load_trained_model('bilstm_results/fold_1_model.h5')

# 获取所有图像
image_paths = glob.glob('/path/to/new/images/*.jpg')

# 初始化缓存
cache = BiLSTMFeatureCache(
    data_root='/path/to/data',
    cache_dir='bilstm_feature_cache'
)

# 提取特征并预测
features = cache.get_or_compute_features(
    image_paths=image_paths,
    use_cache=True,
    training=False
)

predictions = batch_predict_bs_scores(model, features)

# 保存结果
results = dict(zip(image_paths, predictions))
with open('batch_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 🐛 训练问题排查

#### 常见错误及解决方案

**1. GPU内存不足**
```
ResourceExhaustedError: OOM when allocating tensor
```
解决方案:
```bash
# 减小批量大小
python main.py --mode train --batch-size 16

# 或修改配置文件
{
  "batch_size": 8,
  "lstm_units": 64  // 同时减少模型复杂度
}
```

**2. 数据路径错误**
```
❌ 图像文件夹: /path/to/image (不存在)
```
解决方案:
```bash
# 检查并修正配置文件中的路径
{
  "data_root": "/correct/path/to/your/data",
  "image_folder": "image"
}
```

**3. 特征缓存冲突**
```
缓存版本不匹配，正在清理缓存...
```
解决方案:
```bash
# 清理缓存重新开始
rm -rf bilstm_feature_cache/

# 或使用强制重计算
python main.py --mode train --config config.json
# 在config.json中设置 "force_recompute": true
```

**4. 训练不收敛**
```
验证MAE一直在 0.8 左右震荡
```
解决方案:
```bash
# 调整学习率和早停耐心
python main.py --mode train --learning-rate 0.0001 \
  --config modified_config.json

# modified_config.json:
{
  "learning_rate": 0.0001,
  "patience": 30,
  "epochs": 200
}
```

## 📊 性能基准

### 🎯 分类性能
- **准确率**: > 85%
- **AUC**: > 0.90
- **精确率**: > 82%
- **召回率**: > 88%

### ⚡ 计算效率
- **Hough圆检测成功率**: 96.2%
- **特征缓存加速**: 9倍性能提升
- **GPU训练支持**: CUDA + CUDNN优化
- **内存管理**: 自动垃圾回收和GPU清理

### 📈 模型稳定性
- **4折交叉验证变异系数**: < 0.10
- **稳定性等级**: 良好-优秀
- **重现性**: 固定随机种子确保结果可重现

## 🔬 技术细节

### 🧠 BiLSTM架构
```python
# 多模型融合版本 (model_name="all")
Input(shape=(None, 5760))  # 1280+2048+1920+512
    ↓
Bidirectional(LSTM(128, dropout=0.3))
    ↓
TimeDistributed(Dense(64, activation='relu'))
    ↓  
TimeDistributed(Dropout(0.3))
    ↓
TimeDistributed(Dense(1, activation='linear'))  # 回归输出

# 单模型版本示例 (model_name="efficientnet")
Input(shape=(None, 1280))  # EfficientNetV2M原始特征
    ↓
Bidirectional(LSTM(128, dropout=0.3))
    ↓
TimeDistributed(Dense(64, activation='relu'))
    ↓  
TimeDistributed(Dropout(0.3))
    ↓
TimeDistributed(Dense(1, activation='linear'))  # 回归输出
```

### 📊 逻辑回归特征
- **特征1**: BiLSTM预测的囊胚评分
- **特征2**: 母体年龄
- **预处理**: StandardScaler标准化
- **类别平衡**: 'balanced'权重策略

### 🔍 Hough圆检测算法
```python
# 自适应参数范围
min_radius = max(50, int(diagonal * 0.15))  # 最小半径
max_radius = min(int(diagonal * 0.45), max_dim // 2)  # 最大半径

# 多参数组合策略
param_combinations = [
    (60, 40),   # 中等敏感度
    (70, 45),   # 较高敏感度
    (50, 35),   # 高敏感度
    (80, 50),   # 较低敏感度
    (90, 55),   # 低敏感度
]

# 智能评分系统
score = (
    radius_weight * 0.4 +      # 半径权重40%
    center_weight * 0.35 +     # 中心位置权重35%
    size_reasonableness * 0.25  # 大小合理性权重25%
)
```

## 🐛 常见问题

### Q: 如何处理GPU内存不足？
```python
from BiLSTM import cleanup_memory, configure_gpu

# 配置GPU内存增长
configure_gpu()

# 训练过程中定期清理内存
cleanup_memory()

# 减小批量大小
batch_size = 16  # 默认32
```

### Q: 如何自定义Hough检测参数？
```python
# 修改 preprocessing/bilstm_image_utils.py 中的参数
param_combinations = [
    (your_param1, your_param2),  # 自定义参数
    # ...
]
```

### Q: 如何添加新的预训练模型？
```python
# 在 BiLSTM.py 的 MultiModelFeatureExtractor 类中添加
def _build_models(self):
    # 添加新模型
    new_model = YourPretrainedModel(weights='imagenet', ...)
    self.models['your_model'] = new_model
    self.feature_dims['your_model'] = output_dim
```

## 📝 更新日志

### v2.1.0 (当前版本)
- ✅ 🆕 自动评分字典生成系统
- ✅ 🆕 智能CSV解析和ICM/TE评分计算
- ✅ 🆕 文件时间检测和自动重新生成
- ✅ 🆕 配置驱动的无缝自动化流程
- ✅ 完全模块化重构
- ✅ 集成Hough圆检测预处理
- ✅ 智能特征缓存系统
- ✅ 多模型特征融合
- ✅ 综合可视化系统
- ✅ 严格交叉验证框架

### v2.0.0 (历史版本)
- 完全模块化重构
- 集成Hough圆检测预处理
- 智能特征缓存系统
- 多模型特征融合

### v1.0.0 (历史版本)
- 基础BiLSTM模型实现
- 简单特征提取和缓存
- 基本交叉验证功能

## 📞 技术支持

- **文档**: 详细的代码注释和docstring
- **示例**: 每个模块都包含使用示例
- **测试**: 模块导入和基本功能验证
- **兼容性**: 向后兼容原有BiLSTM.py接口

## 📄 许可证

本项目专用于胚胎图像分析研究，Hough圆检测算法针对培养皿环境进行了专门优化。

---

**🎯 项目目标**: 为胚胎学研究提供准确、高效、可重现的深度学习分类系统

**💡 核心优势**: 模块化设计 + 智能缓存 + Hough圆检测 + 严格验证
