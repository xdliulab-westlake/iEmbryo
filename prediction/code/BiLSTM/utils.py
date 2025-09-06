"""
BiLSTM 工具函数模块
包含GPU配置、数据类型转换、路径检查等通用功能
"""

import os
import gc
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def ensure_float32(*arrays):
    """
    Convert data types to float32.
    
    Args:
        *arrays: Variable number of arrays to convert
        
    Returns:
        List of arrays converted to float32
    """
    return [arr.astype(np.float32) if hasattr(arr, 'astype') else np.array(arr, dtype=np.float32) for arr in arrays]

def configure_gpu():
    """
    Configure GPU settings for optimal performance.
    """
    try:
        # 设置GPU内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # 设置GPU内存限制
            tf.config.experimental.set_limit(gpus[0], 10240)  # 10GB limit
            print(f"GPU配置成功: {len(gpus)} GPU(s)")
        else:
            print("未检测到GPU，使用CPU训练")
            
    except RuntimeError as e:
        print(f"GPU配置失败: {e}")

def load_score_dict(csv_path):
    """
    Load blastocyst score dictionary from CSV file.
    
    Args:
        csv_path: Path to CSV file containing embryo scores
        
    Returns:
        dict: Dictionary mapping embryo IDs to blastocyst scores
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"评分文件不存在: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # 假设CSV包含EmbryoID和BlastocystScore列
        if 'EmbryoID' not in df.columns or 'BlastocystScore' not in df.columns:
            print("Warning: CSV文件格式可能不正确")
            print(f"可用列: {df.columns.tolist()}")
        
        score_dict = dict(zip(df['EmbryoID'], df['BlastocystScore']))
        print(f"成功加载 {len(score_dict)} 个胚胎评分")
        
        return score_dict
        
    except Exception as e:
        raise ValueError(f"加载评分文件失败: {e}")

def check_required_paths():
    """
    Check if all required paths exist and are accessible.
    
    Returns:
        bool: True if all paths are valid
    """
    required_paths = [
        "/liaozizhuo/selected_images",
        "/liaozizhuo/scripts"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print(f"缺失路径: {missing_paths}")
        return False
    
    print("所有必需路径检查通过")
    return True

def check_cudnn_compatibility():
    """
    Check CUDNN compatibility for LSTM training.
    """
    try:
        # 检查CUDNN版本
        if tf.test.is_built_with_cuda():
            print("✅ TensorFlow CUDA 支持已启用")
            
            # 测试CUDNN LSTM
            if tf.config.list_physical_devices('GPU'):
                print("✅ GPU设备可用")
                # 简单的CUDNN兼容性测试
                lstm = tf.keras.layers.LSTM(32, return_sequences=True)
                test_input = tf.random.normal((1, 10, 64))
                try:
                    _ = lstm(test_input)
                    print("✅ CUDNN LSTM 兼容性测试通过")
                except Exception as e:
                    print(f"❌ CUDNN LSTM 测试失败: {e}")
            else:
                print("⚠️ 未检测到GPU设备")
        else:
            print("⚠️ TensorFlow 未构建CUDA支持")
    except Exception as e:
        print(f"❌ CUDNN兼容性检查失败: {e}")

def create_image_augmentation_fn():
    """
    Create image augmentation function for training.
    
    Returns:
        Function that applies random augmentations to images
    """
    def augment_image(image):
        """Apply random augmentations to an image."""
        # 随机水平翻转
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_flip_left_right(image)
        
        # 随机亮度调整
        if tf.random.uniform([]) > 0.7:
            image = tf.image.random_brightness(image, max_delta=0.1)
        
        # 随机对比度调整
        if tf.random.uniform([]) > 0.7:
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            
        # 随机饱和度调整  
        if tf.random.uniform([]) > 0.8:
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        
        # 随机色调调整
        if tf.random.uniform([]) > 0.9:
            image = tf.image.random_hue(image, max_delta=0.05)
            
        return image
    
    return augment_image

def get_unified_training_config():
    """
    Get unified training configuration.
    
    Returns:
        dict: Training configuration parameters
    """
    return {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'patience': 15,
        'min_lr': 1e-7,
        'reduce_lr_patience': 5,
        'reduce_lr_factor': 0.5,
        'validation_split': 0.2,
        'verbose': 1
    }

def create_callbacks(patience=15, min_lr=1e-7, reduce_lr_patience=5, reduce_lr_factor=0.5):
    """
    Create standard training callbacks.
    
    Args:
        patience: EarlyStopping patience
        min_lr: Minimum learning rate
        reduce_lr_patience: ReduceLROnPlateau patience
        reduce_lr_factor: Learning rate reduction factor
        
    Returns:
        list: List of callback objects
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        )
    ]
    
    return callbacks

def cleanup_memory():
    """
    Clean up GPU and system memory.
    """
    # 清理Keras会话
    tf.keras.backend.clear_session()
    
    # 强制垃圾回收
    gc.collect()
    
    # 等待GPU内存释放
    if tf.config.list_physical_devices('GPU'):
        time.sleep(2)

def print_system_info():
    """
    Print system and TensorFlow information.
    """
    print("=== 系统信息 ===")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"Python版本: {tf.config.list_physical_devices()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU设备: {len(gpus)} 个")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("GPU设备: 无")
    
    print(f"CUDA构建支持: {tf.test.is_built_with_cuda()}")
    print("================")

def validate_data_shapes(*data_arrays):
    """
    Validate that data arrays have consistent shapes.
    
    Args:
        *data_arrays: Variable number of data arrays to validate
        
    Returns:
        bool: True if shapes are consistent
    """
    if len(data_arrays) < 2:
        return True
    
    first_len = len(data_arrays[0])
    for i, arr in enumerate(data_arrays[1:], 1):
        if len(arr) != first_len:
            print(f"形状不一致: 数组0长度={first_len}, 数组{i}长度={len(arr)}")
            return False
    
    print(f"数据形状验证通过: {first_len} 个样本")
    return True

def safe_divide(a, b, default=0.0):
    """
    Safe division with default value for division by zero.
    
    Args:
        a: Numerator
        b: Denominator  
        default: Default value when b is zero
        
    Returns:
        Result of a/b or default if b is zero
    """
    return default if b == 0 else a / b