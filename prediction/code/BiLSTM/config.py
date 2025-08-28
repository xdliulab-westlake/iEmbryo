"""BiLSTM模型配置文件"""

# 模型架构配置
MODEL_CONFIG = {
    'feature_dim': 1024,          # 特征降维维度
    'lstm_units_1': 128,          # 第一层LSTM单元数
    'lstm_units_2': 64,           # 第二层LSTM单元数
    'dense_units': [128, 64],     # 全连接层单元数
    'dropout_rates': [0.3, 0.3, 0.4, 0.4],  # dropout率
    'l2_reg': 1e-3,              # L2正则化系数
}

# 训练配置
TRAINING_CONFIG = {
    'learning_rate': 1e-5,        # 学习率
    'batch_size': 4,              # 批量大小
    'epochs': 12,                 # 训练轮数
    'patience': 5,                # 早停耐心值
    'min_delta': 1e-4,           # 最小改善阈值
}

# 数据配置
DATA_CONFIG = {
    'validation_split': 0.2,      # 验证集比例
    'test_split': 0.15,          # 测试集比例
    'random_state': 42,           # 随机种子
    'n_folds': 5,                 # 交叉验证折数
    'n_repeats': 1,               # 交叉验证重复次数
}

# 路径配置
PATH_CONFIG = {
    'image_root': "/liaozizhuo/image",
    'score_csv': "/liaozizhuo/scripts/MainDB.csv",
    'embryo_csv': "/liaozizhuo/scripts/filtered_clean_embryos.csv",
    'age_csv': "/liaozizhuo/scripts/SlidePatientBirthDates.csv",
    'output_dir': "/liaozizhuo/scripts"
}
