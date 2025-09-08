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

### 🆕 备份文件风格数据结构
- **结构化数据管理**: 基于EmbryoID和FullPath的标准化数据组织
- **CSV格式胚胎信息**: 支持胚胎质量、年龄等元数据的结构化存储
- **完整路径定位**: 支持任意路径的图像文件精确定位
- **向后兼容**: 完全兼容传统评分字典格式，平滑迁移

#### 🔄 系统架构对比

**传统文件名匹配方式（已弃用）**：
```python
# 基于文件名的简单匹配
image_files = os.listdir(image_folder)
for img_file in image_files:
    if img_file in score_dict:
        # 处理图像和评分
```

**备份文件风格系统（当前推荐）**：
```python
# 基于结构化数据的匹配方式
embryo_df = pd.read_csv(embryo_info_path)
valid_embryos = set(embryo_df["EmbryoID"])
available_embryo_ids = [eid for eid in valid_embryos if eid in score_data]

embryo_metadata = {}
for _, row in embryo_df.iterrows():
    embryo_id = row["EmbryoID"]
    if embryo_id in available_embryo_ids:
        embryo_metadata[embryo_id] = {
            'full_path': row["FullPath"],
            'quality': row["EmbryoQuality"],
            'age': row["Age"],
            'bs_score': score_data[embryo_id]['bs_score']
        }
```

**系统优势对比**：

| 特性 | 传统方式 | 备份文件风格 |
|------|----------|--------------|
| 数据匹配成功率 | 依赖文件名 | 基于结构化ID ✅ |
| 图像路径支持 | 限制单一目录 | 支持任意路径 ✅ |
| 元数据集成 | JSON内嵌 | CSV结构化存储 ✅ |
| 可维护性 | 中等 | 高 ✅ |
| 扩展性 | 有限 | 灵活 ✅ |
| 版本控制 | 困难 | 友好 ✅ |

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

# 方式4: 🆕 backup_final.py完全匹配策略训练（推荐）
# 配置服务器CSV路径后使用与backup_final.py完全相同的数据策略
python main.py --mode train --config bilstm_config.json

# 方式5: 自动生成评分字典并训练（传统功能，向后兼容）
# 在配置文件中设置CSV路径后，系统会自动生成score_dict.json
python main.py --mode train --config legacy_config.json
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

### 🚀 配置文件快速设置指南

#### 1️⃣ 创建配置文件
```bash
# 方式1：使用系统自动创建
cd /path/to/BiLSTM
python main.py --create-config

# 方式2：手动创建
touch bilstm_config.json
```

#### 2️⃣ 基础配置示例
```json
{
  "data_root": "/Users/your_username/data/embryo_project",
  "image_folder": "image",
  "score_dict_path": "score_dict.json",
  "embryo_info_path": "BiLSTM/embryo_info.csv",
  
  "model_name": "efficientnet",
  "epochs": 50,
  "batch_size": 32,
  
  "use_cache": true,
  "verbose": 1
}
```

#### 3️⃣ 验证配置文件
```bash
# 检查JSON格式是否正确
python -c "import json; print('✅ 配置文件格式正确' if json.load(open('bilstm_config.json')) else '❌ 配置文件格式错误')"

# 验证路径是否存在
python main.py --config bilstm_config.json --mode train --help
```

#### 4️⃣ 使用配置文件训练
```bash
# 使用自定义配置训练
python main.py --mode train --config bilstm_config.json

# 配置文件 + 命令行参数覆盖
python main.py --mode train --config bilstm_config.json --epochs 100 --batch-size 16
```

### 🎛️ 核心参数说明

#### 📁 数据路径参数

**🔑 关键配置项（backup_final.py策略）**：
```json
{
  "data_root": "/liaozizhuo/image",                // 服务器图像根目录（必须是绝对路径）
  "cache_dir": "bilstm_feature_cache",             // 特征缓存目录
  
  // 🆕 backup_final.py完全匹配策略（当前推荐）
  "maindb_csv_path": "/liaozizhuo/scripts/MainDB.csv",                      // 主数据库CSV文件路径
  "filtered_csv_path": "/liaozizhuo/scripts/filtered_clean_embryos.csv",    // 胚胎信息CSV文件路径
  "slide_info_path": "/liaozizhuo/scripts/SlidePatientBirthDates.csv",      // 年龄信息CSV文件路径
  
  // 🔄 传统方式（向后兼容，可选）
  "score_dict_path": "score_dict.json",           // 评分字典文件路径
  "embryo_info_path": "BiLSTM/embryo_info.csv"    // 胚胎信息文件路径（传统方式）
}
```

**⚠️ 重要说明**：
- `data_root`: 必须使用绝对路径（如：`/Users/username/project/data`）
- `embryo_info_path`: 系统会根据`score_dict.json`自动生成此文件
- `maindb_csv_path`: 如果为空，跳过自动生成评分字典功能
- 路径中包含空格时请使用双引号包围

**📝 路径设置示例**：
```json
// ✅ 正确的路径配置
{
  "data_root": "/Users/lzz/Documents/GitHub/IVF_AI/prediction/code",
  "image_folder": "image",
  "score_dict_path": "score_dict.json",
  "embryo_info_path": "BiLSTM/embryo_info.csv"
}

// ❌ 错误的路径配置  
{
  "data_root": "./data",           // ❌ 不要使用相对路径
  "image_folder": "../images",     // ❌ 不要使用相对路径
  "score_dict_path": ""            // ❌ 不能为空
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
  
  // 🆕 备份文件风格数据结构配置（推荐）
  "embryo_info_path": "/Users/your_username/data/embryo_info.csv",
  
  // 🔄 自动评分字典生成配置（传统方式）
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
  
  // 🆕 备份文件风格数据结构配置（推荐）
  "embryo_info_path": "/absolute/path/to/embryo_info.csv",  // 胚胎信息文件路径
  
  // 🔄 自动生成评分字典的源CSV路径（传统方式）
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
  "data_root": "/Users/lzz/Documents/GitHub/IVF_AI/prediction/code",
  "image_folder": "image",
  "score_dict_path": "score_dict.json", 
  "embryo_info_path": "BiLSTM/embryo_info.csv",
  
  "model_name": "vgg16",      // 使用最轻量的模型
  "epochs": 20,               // 快速训练
  "batch_size": 64,           // 大批量
  "n_folds": 2,              // 少量折数
  "patience": 5,             // 快速早停
  "use_cache": true,         // 启用缓存
  "verbose": 2               // 详细输出
}
```

**💡 配置文件使用技巧**：

1. **快速开始**：
```bash
# 第一次使用，创建默认配置
python main.py --create-config

# 编辑配置文件，只需修改关键路径
vim bilstm_config.json  # 修改 data_root 即可
```

2. **路径配置检查**：
```bash
# 使用配置前先验证路径
python -c "
import json, os
config = json.load(open('bilstm_config.json'))
data_root = config['data_root']
image_path = os.path.join(data_root, config['image_folder'])
score_path = os.path.join(data_root, config['score_dict_path'])
print(f'✅ 数据根目录: {data_root}' if os.path.exists(data_root) else f'❌ 数据根目录不存在: {data_root}')
print(f'✅ 图像目录: {image_path}' if os.path.exists(image_path) else f'❌ 图像目录不存在: {image_path}')  
print(f'✅ 评分文件: {score_path}' if os.path.exists(score_path) else f'❌ 评分文件不存在: {score_path}')
"
```

3. **模型选择建议**：
```json
// 💻 内存有限时使用单模型
{
  "model_name": "efficientnet",  // 推荐：平衡性能和效率
  "batch_size": 16
}

// 🚀 性能优先时使用多模型
{
  "model_name": "all",           // 最佳精度
  "batch_size": 32
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


#### 🔧 配置验证

**使用配置前验证**
```bash
# 验证JSON格式
python -c "import json; json.load(open('your_config.json'))"

# 验证配置完整性
python main.py --config your_config.json --mode train --help
```


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

### v2.4.0 (当前版本) - 完全匹配backup_final.py策略
- ✅ 🆕 **完全匹配backup_final.py数据加载策略**
  - 🔄 新增 `load_and_prepare_embryo_data()` 函数，完全按照原backup策略
  - 📊 使用 `filtered_clean_embryos.csv` 直接读取胚胎信息，而不是生成评分字典
  - 🎯 ICM/TE评分映射：`{'A': 1, 'B': 2, 'C': 4}` 和 `{'A': 1, 'B': 3, 'C': 4}`
  - 🧬 poor胚胎自动处理：分配6.0-6.5随机评分
  - 👴 年龄信息：通过`PatientID`前缀映射到`SlidePatientBirthDates.csv`
  - 📈 缺失值处理：使用**中位数填充**而不是跳过胚胎，避免使用默认值
- ✅ 🆕 **服务器环境兼容性**
  - 🖼️ 保留复杂目录结构图像读取功能（`/liaozizhuo/image/D2023.../images/F0/`）
  - 🔍 保留图像格式检测和临时文件处理
  - 💾 智能缓存系统与新数据策略完全集成
- ✅ 🆕 **配置文件更新**
  - 📋 更新 `bilstm_config.json` 匹配服务器路径
  - 🔧 新增 `maindb_csv_path`, `filtered_csv_path`, `slide_info_path` 配置
  - 🎯 `data_root` 更新为 `"/liaozizhuo/image"`
- ✅ 🆕 **向后兼容性保持**
  - 🔄 `stratified_cross_validation_with_cache()` 支持新旧两套参数
  - 📊 自动检测使用backup策略还是传统策略
  - 🛠️ 保持所有原有功能接口不变

### v2.3.0 (历史版本)
- ✅ 🆕 **自动胚胎信息文件生成系统**
  - 🔄 自动从 `score_dict.json` 生成 `embryo_info.csv`
  - 📊 BS分数到质量等级的智能映射 (Excellent/Good/Fair/Poor)
  - 🕒 基于文件时间戳的自动更新检测
  - 🎯 完全集成到训练流程的自动化生成
  - 🛠️ 新增 `generate_embryo_info_from_score_dict()` 工具函数
- ✅ 🆕 **配置文件详细指南**
  - 📋 新增配置文件快速设置指南
  - 🔧 详细的路径配置说明和验证方法
  - 💡 不同场景的配置示例和使用技巧
  - ⚠️ 常见配置错误和解决方案

### v2.2.0 (历史版本) 
- ✅ 🆕 **备份文件风格数据结构支持**
  - 🔄 基于 EmbryoID 和 FullPath 的结构化数据匹配
  - 📊 CSV 格式的胚胎信息管理 (embryo_info.csv)
  - 🎯 与备份系统完全一致的数据加载逻辑
  - 🔗 支持完整路径的图像定位和元数据集成
- ✅ 🆕 配置参数扩展
  - 新增 `embryo_info_path` 配置选项
  - 向后兼容传统评分字典格式
  - 灵活的数据结构选择支持

### v2.1.0 (历史版本)
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

## 🔄 完整训练流程详解

### 📋 训练流程概览

BiLSTM胚胎分类系统使用**双层机器学习管道**，完整训练流程如下：

```
                           BiLSTM胚胎分类系统 - 完整训练流程
                                        
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                🎯 一键启动                                          │
│                       python main.py --mode train                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         1️⃣ 数据加载与准备 (backup_final.py策略)                     │
│                                                                                     │
│  📋 MainDB.csv                🧬 filtered_clean_embryos.csv      👴 SlidePatientBirthDates.csv │
│  ├─ ICM评分映射               ├─ 胚胎信息(ID,质量,患者ID)          ├─ 年龄信息映射              │
│  ├─ TE评分映射                ├─ EmbryoQuality                    └─ 中位数填充缺失值         │
│  └─ BS_prime计算              └─ poor胚胎随机评分(6.0-6.5)                                  │
│                                                                                     │
│  输出: root_dir + score_dict + valid_embryo_df                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            2️⃣ 图像预处理 (STORK管道)                               │
│                                                                                     │
│  📁 复杂目录结构              🔍 智能图像搜索             ⭕ Hough圆检测            │
│  /embryo_id/                  ├─ images/F0/ (优先)        ├─ 培养皿边界检测         │
│  └─images/                    ├─ images/F15/              ├─ 96.2%成功率            │
│    ├─F0/                      └─ images/F30/              └─ 智能裁剪               │
│    ├─F15/ ← focal depth       🖼️ 格式检测                📐 标准化                │
│    └─图像文件                  └─ 支持无扩展名图像         └─ 224x224 + 数据增强    │
│                                                                                     │
│  输出: 预处理后的标准化图像 [batch_size, 224, 224, 3]                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            3️⃣ 多模型特征提取                                        │
│                                                                                     │
│  🧠 预训练模型融合            💾 智能缓存系统              🚀 性能优化               │
│  ├─ EfficientNetV2M (1280维) ├─ MD5哈希键                 ├─ 9次→1次重复优化         │
│  ├─ ResNet152V2 (2048维)     ├─ 原子文件操作              ├─ GPU内存管理             │
│  ├─ DenseNet201 (1920维)     ├─ 自动失效检测             └─ 多GPU并行               │
│  ├─ VGG16 (512维)            └─ 缓存复用                                           │
│  └─ 融合特征 (5760维)                                                              │
│                                                                                     │
│  输出: 高维深度特征 [batch_size, sequence_length, 5760]                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          4️⃣ BiLSTM模型训练 (第一层 - 回归)                         │
│                                                                                     │
│  🧠 神经网络架构                                                                   │
│  Input(5760) → Bidirectional(LSTM(128)) → TimeDistributed(Dense(64)) → Dense(1)   │
│                                                                                     │
│  🔄 特性优势                 🛡️ 正则化策略              ⏱️ 训练控制                │
│  ├─ 双向LSTM时序建模         ├─ Dropout(0.3)             ├─ EarlyStopping           │
│  ├─ 捕获前后关系             ├─ 防过拟合                 ├─ 学习率调度               │
│  └─ 连续BS评分预测           └─ L2正则化                 └─ 模型检查点               │
│                                                                                     │
│  输出: 训练好的BiLSTM回归模型 + 囊胚评分预测                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         5️⃣ 逻辑回归分类 (第二层 - 分类)                            │
│                                                                                     │
│  🔗 特征融合                  ⚖️ 类别平衡                📊 最终决策               │
│  ├─ BiLSTM预测评分 (连续)     ├─ balanced权重策略          ├─ 二分类输出             │
│  └─ 母体年龄 (临床特征)       ├─ 处理数据不平衡            ├─ Excellent/Good vs     │
│                               └─ 标准化预处理              └─ Fair/Poor             │
│  🎯 双层优势                                                                      │
│  ├─ 专门化训练: BiLSTM专注回归 + 逻辑回归专注分类                                   │
│  ├─ 特征互补: 深度学习特征 + 临床经验特征                                          │
│  └─ 性能提升: 两阶段优于端到端                                                    │
│                                                                                     │
│  输出: 最终分类结果 + 概率分数 + 性能指标                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            6️⃣ 交叉验证评估                                         │
│                                                                                     │
│  🔀 严格验证策略              📊 数据分离                  📈 多指标评估            │
│  ├─ 4折分层K折验证            ├─ 70%训练 / 30%测试         ├─ 准确率 (Accuracy)     │
│  ├─ 确保类别平衡              ├─ 避免数据泄漏              ├─ AUC分数               │
│  ├─ 随机种子控制              └─ 严格分离                 ├─ 精确率 (Precision)    │
│  └─ 可重现结果                🎨 训练策略                 ├─ 召回率 (Recall)       │
│                               ├─ 训练时数据增强            ├─ F1分数                │
│                               └─ 验证时禁用增强            └─ 混淆矩阵               │
│                                                                                     │
│  输出: 平均性能 ± 标准差 + 各折详细结果                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           7️⃣ 结果可视化与保存                                      │
│                                                                                     │
│  📈 综合可视化                📁 模型保存                  📊 结果输出              │
│  ├─ training_curves.png       ├─ fold_1_model.h5           ├─ cv_results.json      │
│  ├─ confusion_matrices.png    ├─ fold_2_model.h5           ├─ 详细数值结果         │
│  ├─ roc_curves.png           ├─ fold_3_model.h5           ├─ 配置参数记录         │
│  └─ performance_summary.png   └─ fold_4_model.h5           └─ 训练日志             │
│                                                                                     │
│  🎯 最终输出: 完整的性能报告 + 训练好的模型 + 可视化结果                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

                          ✨ 训练完成！系统准备用于胚胎质量预测 ✨
```

### 🎯 核心优势总结

```
💡 双层机器学习管道
├─ 第一层: BiLSTM回归 → 预测连续囊胚评分
└─ 第二层: 逻辑回归 → 最终质量分类

🔧 智能自动化
├─ backup_final.py完全匹配策略
├─ 端到端自动化流程  
├─ 智能缓存与优化
└─ 一键训练命令

✅ 严格验证框架
├─ 4折分层交叉验证
├─ 数据泄漏防护
├─ 多指标综合评估
└─ 可重现结果

🚀 性能优化
├─ GPU内存管理
├─ 特征缓存复用
├─ 多模型融合
└─ 参数自动调整
```

### 📋 快速开始

```bash
# 完整训练流程（一键启动）
python main.py --mode train --config bilstm_config.json

# 训练过程监控
nvidia-smi                              # GPU使用情况
tail -f bilstm_results/training.log     # 实时日志
du -sh bilstm_feature_cache/            # 缓存使用情况
```

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
