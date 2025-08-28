"""训练工具模块 - 从BiLSTM.py中提取的训练相关函数"""

import tensorflow as tf
import numpy as np
import gc
import time
from config import TRAINING_CONFIG

def configure_gpu_for_lstm():
    """配置GPU环境，优化LSTM训练"""
    print("=== 配置GPU环境 ===")
    
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)

    print(f"检测到 {num_gpus} 个GPU: {[gpu.name for gpu in gpus]}")

    if num_gpus > 0:
        gpu_ids = ','.join([str(i) for i in range(num_gpus)])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"设置CUDA_VISIBLE_DEVICES = {gpu_ids}")
        
        os.environ['TF_GPU_THREAD_COUNT'] = str(num_gpus)
        print(f"设置GPU线程数 = {num_gpus}")
    else:
        print("未检测到GPU，将使用CPU")

    # 通用GPU优化设置
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"已为 {num_gpus} 个GPU启用内存增长")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    
    return num_gpus

def train_main_model(X_train, y_train, X_val, y_val, feature_dim, save_model_path):
    """训练主模型"""
    print("=== 训练主模型 ===")
    
    # 强制转换数据类型
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    # 检测GPU
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    # 选择分布式策略
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"使用多GPU训练，GPU数量: {strategy.num_replicas_in_sync}")
    elif num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        print("使用单GPU策略")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        print("使用CPU策略")
    
    # 动态调整批量大小
    feature_size = X_train.shape[2]
    if feature_size > 50000:
        base_batch_size = 2 if num_gpus <= 1 else 4
    elif feature_size > 30000:
        base_batch_size = 4 if num_gpus <= 1 else 6
    else:
        base_batch_size = 8 if num_gpus <= 1 else 12
    
    global_batch_size = base_batch_size * max(1, strategy.num_replicas_in_sync)
    
    # 清理GPU内存
    tf.keras.backend.clear_session()
    gc.collect()
    
    with strategy.scope():
        from model_architecture import create_model
        model = create_model(feature_dim)
        
        # 构建回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=TRAINING_CONFIG['patience'], 
                restore_best_weights=True, 
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7, 
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                save_model_path, 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=1
            )
        ]
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=global_batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    print(f"\n=== 训练完成 ===")
    print(f"使用了 {len(gpus)} 个GPU")
    print(f"最佳验证损失: {min(history.history['val_loss']):.4f}")
    
    return model, history
