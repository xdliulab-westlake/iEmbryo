"""
BiLSTM 模型模块
包含BiLSTM模型构建、训练、评估等核心功能
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
import time

def create_model(feature_dim, lstm_units=128, dropout_rate=0.3, learning_rate=0.001):
    """
    Create BiLSTM regression model for blastocyst score prediction.
    
    Args:
        feature_dim: Input feature dimension
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # 输入层
    inputs = Input(shape=(None, feature_dim), name='sequence_input')
    
    # 双向LSTM层
    lstm_out = Bidirectional(
        LSTM(
            lstm_units, 
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='bilstm_layer'
        )
    )(inputs)
    
    # 时间分布式全连接层
    dense_out = TimeDistributed(
        Dense(64, activation='relu', name='dense_layer')
    )(lstm_out)
    
    # Dropout层
    dropout_out = TimeDistributed(
        Dropout(dropout_rate, name='dropout_layer')
    )(dense_out)
    
    # 输出层 - 回归任务
    outputs = TimeDistributed(
        Dense(1, activation='linear', name='output_layer')
    )(dropout_out)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Regressor')
    
    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mae',  # 平均绝对误差
        metrics=['mae', 'mse']
    )
    
    return model

def batch_predict_bs_scores(model, X_data, batch_size=8):
    """
    Batch prediction of blastocyst scores to handle memory constraints.
    
    Args:
        model: Trained BiLSTM model
        X_data: Input feature data
        batch_size: Batch size for prediction
        
    Returns:
        Predicted blastocyst scores
    """
    n_samples = X_data.shape[0]
    all_predictions = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_X = X_data[start_idx:end_idx]
        
        # 预测
        batch_pred = model.predict(batch_X, verbose=0)
        all_predictions.append(batch_pred)
    
    # 合并所有预测结果
    predictions = np.concatenate(all_predictions, axis=0)
    
    # 如果是时间序列，取最后一个时间步的预测
    if len(predictions.shape) == 3:
        predictions = predictions[:, -1, :]  # (n_samples, 1)
    
    return predictions.flatten()

def train_model(X_train, y_train, X_val, y_val, feature_dim, save_path,
                lstm_units=128, dropout_rate=0.3, learning_rate=0.001,
                epochs=100, batch_size=32, patience=15, verbose=1):
    """
    Train BiLSTM model with early stopping and learning rate reduction.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_dim: Input feature dimension
        save_path: Path to save the best model
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        epochs: Maximum number of epochs
        batch_size: Batch size
        patience: Early stopping patience
        verbose: Verbosity level
        
    Returns:
        tuple: (trained_model, training_history)
    """
    from .utils import create_callbacks, cleanup_memory
    
    print(f"开始训练BiLSTM模型...")
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    print(f"模型参数: LSTM单元={lstm_units}, dropout={dropout_rate}, lr={learning_rate}")
    
    # 清理内存
    cleanup_memory()
    
    # 创建模型
    model = create_model(
        feature_dim=feature_dim,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # 打印模型结构
    if verbose > 0:
        model.summary()
    
    # 创建回调函数
    callbacks = create_callbacks(patience=patience)
    
    # 如果提供了保存路径，添加模型检查点
    if save_path:
        from tensorflow.keras.callbacks import ModelCheckpoint
        checkpoint = ModelCheckpoint(
            save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # 训练模型
    start_time = time.time()
    
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.1f}秒")
        
        return model, history
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        cleanup_memory()
        raise

def evaluate_model(model, X_test, y_test, batch_size=8):
    """
    Evaluate trained BiLSTM model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Evaluation metrics
    """
    print("评估模型性能...")
    
    # 预测
    y_pred = batch_predict_bs_scores(model, X_test, batch_size=batch_size)
    
    # 确保维度一致
    if len(y_test.shape) > 1 and y_test.shape[-1] == 1:
        y_test = y_test.flatten()
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算R²
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred,
        'true_values': y_test
    }
    
    print(f"测试集评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}") 
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return metrics

def load_trained_model(model_path):
    """
    Load a pre-trained BiLSTM model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"加载模型失败: {e}")

def create_model_ensemble(feature_dim, n_models=3, **model_kwargs):
    """
    Create ensemble of BiLSTM models with different initializations.
    
    Args:
        feature_dim: Input feature dimension
        n_models: Number of models in ensemble
        **model_kwargs: Additional model parameters
        
    Returns:
        List of compiled models
    """
    models = []
    
    for i in range(n_models):
        # 使用不同的随机种子创建多样性
        tf.random.set_seed(42 + i)
        
        model = create_model(feature_dim, **model_kwargs)
        models.append(model)
        
        print(f"创建集成模型 {i+1}/{n_models}")
    
    return models

def ensemble_predict(models, X_data, method='mean', batch_size=8):
    """
    Make predictions using model ensemble.
    
    Args:
        models: List of trained models
        X_data: Input data
        method: Ensemble method ('mean', 'median')
        batch_size: Batch size for prediction
        
    Returns:
        Ensemble predictions
    """
    all_predictions = []
    
    for i, model in enumerate(models):
        pred = batch_predict_bs_scores(model, X_data, batch_size=batch_size)
        all_predictions.append(pred)
        print(f"模型 {i+1} 预测完成")
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)  # (n_models, n_samples)
    
    # 集成预测
    if method == 'mean':
        ensemble_pred = np.mean(all_predictions, axis=0)
    elif method == 'median':
        ensemble_pred = np.median(all_predictions, axis=0)
    else:
        raise ValueError(f"不支持的集成方法: {method}")
    
    return ensemble_pred

def get_model_config_summary(model):
    """
    Get a summary of model configuration.
    
    Args:
        model: Keras model
        
    Returns:
        dict: Model configuration summary
    """
    return {
        'total_params': model.count_params(),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'optimizer': type(model.optimizer).__name__,
        'loss': model.loss,
        'metrics': model.metrics_names
    }