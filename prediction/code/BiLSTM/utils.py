"""
BiLSTM 工具函数模块
包含GPU配置、数据类型转换、路径检查等通用功能
"""

import os
import gc
import time
import random
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
    Compatible with TensorFlow 2.x versions.
    """
    try:
        # 设置GPU内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 设置GPU内存限制 (兼容新旧版本TensorFlow)
            try:
                # TensorFlow 2.2+版本使用这种方式
                if hasattr(tf.config.experimental, 'set_virtual_device_configuration'):
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
                    )
                # 旧版本TensorFlow
                elif hasattr(tf.config.experimental, 'set_limit'):
                    tf.config.experimental.set_limit(gpus[0], 10240)
                    
            except RuntimeError as limit_error:
                # 如果内存限制设置失败，只使用内存增长
                print(f"⚠️ GPU内存限制设置失败: {limit_error}")
                print("继续使用GPU内存增长模式...")
                
            print(f"✅ GPU配置成功: {len(gpus)} GPU(s)")
            
        else:
            print("⚠️ 未检测到GPU，使用CPU训练")
            # 为CPU优化设置
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
    except Exception as e:
        print(f"❌ GPU配置失败: {e}")
        print("继续使用默认配置...")

def load_score_dict(csv_path):
    """
    Load blastocyst score dictionary from CSV file.
    Enhanced version with ICM/TE scoring and poor embryo handling.
    
    Args:
        csv_path: Path to CSV file containing embryo scores
        
    Returns:
        dict: Dictionary mapping embryo IDs to blastocyst scores  
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"评分文件不存在: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # ICM和TE评分映射
        icm_map = {'A': 1, 'B': 2, 'C': 4}
        te_map = {'A': 1, 'B': 3, 'C': 4}
        
        # 计算BS_prime分数
        if 'ICM' in df.columns and 'TE' in df.columns:
            df["ICM_score"] = df["ICM"].map(icm_map)
            df["TE_score"] = df["TE"].map(te_map)
            df["BS_prime"] = df["ICM_score"] + df["TE_score"]
            score_dict = dict(zip(df["EmbryoID"], df["BS_prime"]))
        elif 'BlastocystScore' in df.columns:
            score_dict = dict(zip(df['EmbryoID'], df['BlastocystScore']))
        else:
            raise ValueError("CSV文件必须包含 ICM/TE 列或 BlastocystScore 列")
        
        # 处理低质量胚胎数据（如果存在）
        filtered_path = "/liaozizhuo/scripts/filtered_clean_embryos.csv"
        if os.path.exists(filtered_path):
            filtered_df = pd.read_csv(filtered_path)
            poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
            for emb_id in poor_embryos["EmbryoID"].unique():
                if emb_id not in score_dict:
                    score_dict[emb_id] = 6.0 + random.uniform(0, 0.5)
        
        print(f"成功加载 {len(score_dict)} 个胚胎评分")
        return score_dict
        
    except Exception as e:
        raise ValueError(f"加载评分文件失败: {e}")

def check_required_paths(data_root=None, image_folder="image", score_dict_path="score_dict.json"):
    """
    Check if all required paths exist and are accessible.
    Enhanced version with configurable paths.
    
    Args:
        data_root: Root data directory
        image_folder: Image folder name
        score_dict_path: Score dictionary file path
        
    Returns:
        bool: True if all paths are valid
    """
    print("=== 检查必需的路径和文件 ===")
    
    # 默认路径（向后兼容）
    if data_root is None:
        required_paths = [
            "/liaozizhuo/image",
            "/liaozizhuo/scripts"
        ]
        required_files = [
            "/liaozizhuo/scripts/MainDB.csv",
            "/liaozizhuo/scripts/filtered_clean_embryos.csv"
        ]
    else:
        # 使用配置的路径
        required_paths = [os.path.join(data_root, image_folder)]
        required_files = [os.path.join(data_root, score_dict_path)]
    
    all_valid = True
    
    # 检查目录
    for path in required_paths:
        if not os.path.exists(path):
            print(f"警告: 目录不存在: {path}")
            try:
                os.makedirs(path, exist_ok=True)
                print(f"已创建目录: {path}")
            except Exception as e:
                print(f"创建目录失败: {e}")
                all_valid = False
        else:
            print(f"✓ 目录存在: {path}")
    
    # 检查文件
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"✗ 文件不存在: {file_path}")
            all_valid = False
        else:
            print(f"✓ 文件存在: {file_path}")
    
    print("路径检查完成")
    return all_valid

def check_cudnn_compatibility():
    """
    Check CUDNN compatibility and set environment variables.
    Enhanced version with comprehensive GPU setup.
    """
    print("=== 检查CUDNN兼容性 ===")
    
    try:
        # 设置CUDNN环境变量以提高兼容性
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # 禁用自动调优
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # 启用确定性模式
        
        # 检查TensorFlow版本
        tf_version = tf.__version__
        print(f"TensorFlow版本: {tf_version}")
        
        # 检查GPU设备
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"检测到 {len(gpus)} 个GPU")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                
            # 配置GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("已启用GPU内存增长")
            
            # 简单的CUDNN兼容性测试
            if tf.test.is_built_with_cuda():
                print("✅ TensorFlow CUDA 支持已启用")
                lstm = tf.keras.layers.LSTM(32, return_sequences=True)
                test_input = tf.random.normal((1, 10, 64))
                try:
                    _ = lstm(test_input)
                    print("✅ CUDNN LSTM 兼容性测试通过")
                except Exception as e:
                    print(f"❌ CUDNN LSTM 测试失败: {e}")
            else:
                print("⚠️ TensorFlow 未构建CUDA支持")
        else:
            print("未检测到GPU")
            
    except Exception as e:
        print(f"CUDNN兼容性检查失败: {e}")
    
    return True

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
    Enhanced version with more comprehensive parameters for cross-validation consistency.
    
    Returns:
        dict: Training configuration parameters
    """
    return {
        'epochs': 300,
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'use_data_augmentation': True,
        'min_lr': 1e-6,
        'lr_factor': 0.5,
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

def generate_score_dict_from_csv(maindb_path, filtered_path=None, output_path="score_dict.json"):
    """
    从CSV文件生成评分字典并保存为JSON
    
    Args:
        maindb_path (str): MainDB.csv 文件路径
        filtered_path (str, optional): filtered_clean_embryos.csv 文件路径
        output_path (str): 输出的 score_dict.json 文件路径
        
    Returns:
        dict: 生成的评分字典
    """
    import json
    
    print("=== 开始生成评分字典 ===")
    
    # 检查主数据文件
    if not os.path.exists(maindb_path):
        raise FileNotFoundError(f"主数据文件不存在: {maindb_path}")
    
    print(f"📂 读取主数据文件: {maindb_path}")
    df = pd.read_csv(maindb_path)
    print(f"✅ 成功读取 {len(df)} 行数据")
    print(f"📋 数据列: {list(df.columns)}")
    
    # ICM和TE评分映射
    icm_map = {'A': 1, 'B': 2, 'C': 4}
    te_map = {'A': 1, 'B': 3, 'C': 4}
    
    score_dict = {}
    
    # 计算BS_prime分数
    if 'ICM' in df.columns and 'TE' in df.columns:
        print("📊 使用 ICM/TE 评分计算 BS_prime")
        df["ICM_score"] = df["ICM"].map(icm_map)
        df["TE_score"] = df["TE"].map(te_map)
        df["BS_prime"] = df["ICM_score"] + df["TE_score"]
        
        # 创建评分字典
        for _, row in df.iterrows():
            embryo_id = str(row["EmbryoID"])
            bs_score = float(row["BS_prime"])
            age = float(row.get("Age", 30))  # 默认年龄30
            quality = int(row.get("Quality", 1)) if "Quality" in df.columns else (1 if bs_score <= 4 else 0)
            
            score_dict[embryo_id] = {
                "bs_score": bs_score,
                "age": age,
                "quality": quality
            }
            
    elif 'BlastocystScore' in df.columns:
        print("📊 使用 BlastocystScore 直接评分")
        for _, row in df.iterrows():
            embryo_id = str(row["EmbryoID"])
            bs_score = float(row["BlastocystScore"])
            age = float(row.get("Age", 30))
            quality = int(row.get("Quality", 1)) if "Quality" in df.columns else (1 if bs_score <= 4 else 0)
            
            score_dict[embryo_id] = {
                "bs_score": bs_score,
                "age": age,
                "quality": quality
            }
    else:
        raise ValueError("CSV文件必须包含 ICM/TE 列或 BlastocystScore 列")
    
    print(f"✅ 从主数据文件生成 {len(score_dict)} 个胚胎评分")
    
    # 处理低质量胚胎数据（如果提供）
    if filtered_path and os.path.exists(filtered_path):
        print(f"📂 读取低质量胚胎文件: {filtered_path}")
        filtered_df = pd.read_csv(filtered_path)
        print(f"✅ 成功读取 {len(filtered_df)} 行低质量胚胎数据")
        
        # 查找低质量胚胎
        if "EmbryoQuality" in filtered_df.columns:
            poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
            print(f"🔍 找到 {len(poor_embryos)} 个低质量胚胎")
            
            for emb_id in poor_embryos["EmbryoID"].unique():
                emb_id_str = str(emb_id)
                if emb_id_str not in score_dict:
                    # 为低质量胚胎分配较高的BS分数 (6.0-6.5)
                    bs_score = 6.0 + random.uniform(0, 0.5)
                    age = 30.0  # 默认年龄
                    quality = 0  # 低质量
                    
                    score_dict[emb_id_str] = {
                        "bs_score": bs_score,
                        "age": age,
                        "quality": quality
                    }
                    
            print(f"✅ 为低质量胚胎添加了额外的评分数据")
        else:
            print("⚠️ 低质量胚胎文件中没有 EmbryoQuality 列")
    elif filtered_path:
        print(f"⚠️ 低质量胚胎文件不存在: {filtered_path}")
    
    # 保存为JSON文件
    print(f"💾 保存评分字典到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    bs_scores = [data["bs_score"] for data in score_dict.values()]
    qualities = [data["quality"] for data in score_dict.values()]
    
    print(f"\n📊 生成的评分字典统计:")
    print(f"  总胚胎数: {len(score_dict)}")
    print(f"  BS分数范围: {min(bs_scores):.2f} - {max(bs_scores):.2f}")
    print(f"  平均BS分数: {sum(bs_scores)/len(bs_scores):.2f}")
    print(f"  高质量胚胎: {sum(qualities)} ({sum(qualities)/len(qualities)*100:.1f}%)")
    print(f"  低质量胚胎: {len(qualities)-sum(qualities)} ({(len(qualities)-sum(qualities))/len(qualities)*100:.1f}%)")
    
    print(f"\n🎉 评分字典生成完成! 保存在: {output_path}")
    return score_dict