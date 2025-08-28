"""模型架构模块 - 从BiLSTM.py中提取的模型定义函数"""

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import TimeDistributed, Bidirectional, LSTM, Dense
from tensorflow.keras.regularizers import l2
from config import MODEL_CONFIG

def create_model(feature_dim):
    """创建适应多模型特征的BiLSTM模型"""
    input_dim = 11 * feature_dim
    
    # 强制设置数据类型策略
    tf.keras.mixed_precision.set_global_policy('float32')
    
    model = Sequential([
        Input(shape=(33, input_dim), dtype=tf.float32),
        
        # 特征降维层
        TimeDistributed(Dense(
            MODEL_CONFIG['feature_dim'], 
            activation='relu', 
            kernel_regularizer=l2(MODEL_CONFIG['l2_reg']),
            dtype=tf.float32
        )),
        tf.keras.layers.Dropout(MODEL_CONFIG['dropout_rates'][0]),
        
        # BiLSTM层
        Bidirectional(LSTM(
            MODEL_CONFIG['lstm_units_1'], 
            return_sequences=True, 
            dropout=MODEL_CONFIG['dropout_rates'][1], 
            kernel_regularizer=l2(MODEL_CONFIG['l2_reg']),
            dtype=tf.float32
        )),
        tf.keras.layers.Dropout(MODEL_CONFIG['dropout_rates'][2]),
        
        Bidirectional(LSTM(
            MODEL_CONFIG['lstm_units_2'], 
            dropout=MODEL_CONFIG['dropout_rates'][3], 
            kernel_regularizer=l2(MODEL_CONFIG['l2_reg']),
            dtype=tf.float32
        )),
        
        # 全连接层
        tf.keras.layers.Dropout(0.4),
        Dense(
            MODEL_CONFIG['dense_units'][0], 
            activation='relu', 
            kernel_regularizer=l2(MODEL_CONFIG['l2_reg']), 
            dtype=tf.float32
        ),
        tf.keras.layers.Dropout(0.4),
        Dense(
            MODEL_CONFIG['dense_units'][1], 
            activation='relu', 
            kernel_regularizer=l2(MODEL_CONFIG['l2_reg']), 
            dtype=tf.float32
        ),
        Dense(1, dtype=tf.float32)
    ])
    
    # 使用自适应学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def create_model_simple(feature_dim):
    """创建简化版本的BiLSTM模型，减少过拟合"""
    input_dim = 11 * feature_dim
    
    model = Sequential([
        Input(shape=(33, input_dim), dtype=tf.float32),
        
        # 减少特征维度
        TimeDistributed(Dense(512, activation='relu', 
                            kernel_regularizer=l2(1e-3),
                            dtype=tf.float32)),
        tf.keras.layers.Dropout(0.4),
        
        # 减少LSTM单元数
        Bidirectional(LSTM(64, 
                          return_sequences=True, 
                          dropout=0.4,
                          kernel_regularizer=l2(1e-3),
                          dtype=tf.float32
                         )),
        tf.keras.layers.Dropout(0.4),
        
        Bidirectional(LSTM(32, 
                          dropout=0.4,
                          kernel_regularizer=l2(1e-3),
                          dtype=tf.float32
                         )),
        
        # 减少全连接层
        tf.keras.layers.Dropout(0.5),
        Dense(16, activation='relu', 
              kernel_regularizer=l2(1e-3), 
              dtype=tf.float32),
        tf.keras.layers.Dropout(0.5),
        Dense(1, dtype=tf.float32)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model
