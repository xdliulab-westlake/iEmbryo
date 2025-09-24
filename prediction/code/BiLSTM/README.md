# BiLSTM 胚胎分类系统 🧬

基于深度学习的双层机器学习管道，用于胚胎质量分类：
1. **BiLSTM回归**：预测囊胚评分（BS Score）
2. **逻辑回归**：最终质量分类

## ⚠️ 最新修复

- ✅ **预处理冲突**：移除双重ImageNet标准化，每个模型使用自己的预处理
- ✅ **特征维度匹配**：统一配置文件与实际输出维度
- ⚠️ **配置路径**：需更新硬编码服务器路径为本地路径

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install tensorflow>=2.8.0 opencv-python scikit-learn pandas numpy matplotlib

# 2. 创建配置文件
python main.py --create-config

# 3. 编辑配置文件（修改数据路径）
vim bilstm_config.json

# 4. 开始训练
python main.py --mode train --config bilstm_config.json
```

## 🏗️ 项目结构

```
BiLSTM/
├── main.py                       # 🚀 主程序
├── bilstm_config.json            # ⚙️ 配置文件
├── feature_extractor.py          # 🔍 多模型特征提取器
├── model.py                      # 🧠 BiLSTM模型
├── preprocessing/                # 🖼️ 图像预处理
│   ├── bilstm_image_utils.py     # Hough圆检测
│   └── data_loader_bilstm.py     # 数据加载
├── bilstm_feature_cache/         # 💾 特征缓存
├── bilstm_results/               # 📊 输出结果
└── [其他模块...]
```

## 🔄 训练流程

```
                           BiLSTM胚胎分类系统 - 完整训练流程

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                🎯 一键启动                                          │
│                       python main.py --mode train                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         1️⃣ 数据加载与准备 - 完整流程                               │
│                                                                                     │
│  🔄 步骤1: 生成时间序列图像路径                                                    │
│  └─ generate_image_paths.py: 从原始目录提取33个标准时间点图像                      │
│                                                                                     │
│  🔄 步骤2: 过滤和清理数据                                                          │
│  └─ K-means聚类去除空白图像 + 时间完整性检查                                       │
│                                                                                     │
│  📋 MainDB.csv                🧬 filtered_clean_embryos.csv      👴 SlidePatientBirthDates.csv │
│  ├─ ICM评分映射               ├─ 33个时间点图像路径               ├─ 年龄信息映射              │
│  ├─ TE评分映射                ├─ EmbryoQuality标签                └─ 中位数填充缺失值         │
│  └─ BS_prime计算              └─ poor胚胎随机评分(6.0-6.5)                                  │
│                                                                                     │
│  输出: root_dir + score_dict + valid_embryo_df (包含FullPath列)                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       2️⃣ 图像预处理 (STORK管道) ✅已简化优化                        │
│                                                                                     │
│  📂 时间序列图像文件          🔍 直接路径加载              ⭕ Hough圆检测            │
│  embryo_96.0h.jpg            ├─ CSV中FullPath           ├─ 培养皿边界检测         │
│  embryo_96.5h.jpg            ├─ 标准OpenCV读取          ├─ 96.2%成功率            │
│  embryo_97.0h.jpg            └─ 无复杂目录搜索          └─ 智能中心裁剪           │
│  ... (33个时间点)             🖼️ 简化加载流程            📐 标准化                │
│  embryo_112.0h.jpg           └─ 移除冗余格式检测         └─ 224x224 + 数据增强    │
│                                                                                     │
│  输出: 预处理后图像 [batch_size, 224, 224, 3] (uint8格式，无重复标准化)             │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            3️⃣ 多模型特征提取 ✅已优化                               │
│                                                                                     │
│  🧠 预训练模型融合            💾 智能缓存系统              🚀 性能优化               │
│  ├─ EfficientNetV2M (1280维) ├─ MD5哈希键                 ├─ 9次→1次重复优化         │
│  ├─ ResNet152V2 (2048维)     ├─ 原子文件操作              ├─ GPU内存管理             │
│  ├─ DenseNet201 (1920维)     ├─ 自动失效检测             └─ 多GPU并行               │
│  ├─ VGG16 (512维)            └─ 缓存复用                                           │
│  └─ 融合特征 (5760维)                                                              │
│                                                                                     │
│  🔧 各模型独立ImageNet预处理: efficientnet_preprocess, resnet_preprocess...        │
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
│  ├─ 5折分层K折验证            ├─ 70%训练 / 30%测试         ├─ 准确率 (Accuracy)     │
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
│  └─ performance_summary.png   └─ fold_5_model.h5           └─ 训练日志             │
│                                                                                     │
│  🎯 最终输出: 完整的性能报告 + 训练好的模型 + 可视化结果                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

                          ✨ 训练完成！系统准备用于胚胎质量预测 ✨
```

## ⚙️ 配置文件

### 完整配置模板 (bilstm_config.json)

```json
{
  // 📁 数据路径
  "data_root": "/your/local/path/to/data",
  "maindb_csv_path": "/path/to/MainDB.csv",
  "filtered_csv_path": "/path/to/filtered_clean_embryos.csv",
  "slide_info_path": "/path/to/SlidePatientBirthDates.csv",

  // 🧠 模型设置
  "model_name": "all",        // all, efficientnet, resnet, densenet, vgg16
  "feature_dim": 5760,        // 自动调整: all(5760), efficientnet(1280), resnet(2048), densenet(1920), vgg16(512)
  "lstm_units": 128,
  "dropout_rate": 0.3,
  "learning_rate": 0.001,

  // 🏋️ 训练参数
  "epochs": 100,
  "batch_size": 32,
  "patience": 15,
  "n_folds": 5,
  "validation_split": 0.3,

  // 🖼️ 预处理
  "img_size": 224,
  "apply_petri_crop": true,   // Hough圆检测
  "use_augmentation": true,

  // 💾 缓存设置
  "use_cache": true,
  "results_save_dir": "bilstm_results"
}
```

### 快速配置示例

```json
// 🚀 快速测试
{
  "model_name": "vgg16",
  "epochs": 20,
  "n_folds": 2,
  "batch_size": 64
}

// 🎯 高精度训练
{
  "model_name": "all",
  "epochs": 200,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "dropout_rate": 0.4
}
```

## 📊 输入输出

### 系统输入
```
📥 必需文件:
├── MainDB.csv (ICM/TE评分)
├── filtered_clean_embryos.csv (胚胎信息，包含FullPath图像路径)
├── SlidePatientBirthDates.csv (年龄信息)
└── 时间序列图像文件 (embryo_96.0h.jpg, embryo_96.5h.jpg, ...)
```

### 系统输出
```
📤 训练结果:
├── bilstm_results/
│   ├── cv_results.json (详细性能指标)
│   ├── fold_X_model.h5 (BiLSTM模型)
│   ├── logistic_model_fold_X.pkl (逻辑回归模型)
│   └── *.png (可视化图表)
└── bilstm_feature_cache/ (特征缓存)
```

### 性能指标示例
```json
{
  "avg_performance": {
    "bilstm_metrics": {"mae": 0.85, "r2": 0.67},
    "logistic_metrics": {"accuracy": 0.84, "auc": 0.89, "f1": 0.84}
  },
  "stability_metrics": {"accuracy_std": 0.03, "stability_grade": "High"}
}
```

## 🔧 使用示例

```python
from BiLSTM import MultiModelFeatureExtractor, create_model, configure_gpu

# 配置GPU
configure_gpu()

# 特征提取
extractor = MultiModelFeatureExtractor(model_name='all')
features = extractor.extract_features(images)  # 输出5760维特征

# 模型训练
model = create_model(feature_dim=5760, lstm_units=128, dropout_rate=0.3)
trained_model, history = train_model(X_train, y_train, X_val, y_val)
```

## 🐛 常见问题

| 问题 | 解决方案 |
|------|----------|
| GPU内存不足 | 减小batch_size(16)，使用单模型(vgg16) |
| 训练速度慢 | 启用缓存，增大batch_size(64) |
| 精度不够 | 使用all模型，增加epochs(200) |
| 路径错误 | 检查配置文件中的绝对路径 |

## 📝 更新日志

### v2.5.0 (当前)
- 🔧 修复预处理冲突问题
- 🔧 修复特征维度匹配问题
- 📋 项目结构优化建议

---

**🎯 核心优势**：模块化设计 + 智能缓存 + Hough圆检测 + 严格验证