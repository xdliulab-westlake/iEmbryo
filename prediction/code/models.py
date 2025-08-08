from config import *

def create_model(feature_dim, **hyperparams):
    """创建适应多模型特征的BiLSTM模型，支持超参数配置"""
    
    # 设置默认超参数
    default_params = {
        'lstm_units_1': 256,
        'lstm_units_2': 128,
        'dense_units_1': 128,
        'dense_units_2': 64,
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.2,
        'dense_dropout': 0.3,
        'l2_reg': 1e-4,
        'learning_rate': 1e-4,
        'feature_reduction': 2048
    }
    
    # 更新默认参数
    for key, value in hyperparams.items():
        if key in default_params:
            default_params[key] = value
    
    # 计算输入维度 (11个焦平面 * 特征维度)
    input_dim = 11 * feature_dim
    
    model = Sequential([
        Input(shape=(33, input_dim)),
        
        # 特征降维层
        TimeDistributed(Dense(default_params['feature_reduction'], activation='relu')),
        tf.keras.layers.Dropout(default_params['dropout_rate']),
        
        # BiLSTM层
        Bidirectional(LSTM(default_params['lstm_units_1'], 
                          return_sequences=True, 
                          dropout=default_params['dropout_rate'], 
                          recurrent_dropout=default_params['recurrent_dropout'])),
        Bidirectional(LSTM(default_params['lstm_units_2'], 
                          dropout=default_params['dropout_rate'], 
                          recurrent_dropout=default_params['recurrent_dropout'])),
        
        # 全连接层
        tf.keras.layers.Dropout(default_params['dense_dropout']),
        Dense(default_params['dense_units_1'], activation='relu', 
              kernel_regularizer=l2(default_params['l2_reg'])),
        tf.keras.layers.Dropout(default_params['dense_dropout']),
        Dense(default_params['dense_units_2'], activation='relu', 
              kernel_regularizer=l2(default_params['l2_reg'])),
        Dense(1)  # 回归预测 BS_prime 分数
    ])
    
    # 使用自适应学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=default_params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model 