"""
BiLSTM 工具函数模块
包含GPU配置、数据类型转换、路径检查等通用功能
"""

import os
import gc
import time
import random
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======================================================================
# 匹配backup_final.py策略的数据加载函数
# ======================================================================

def load_and_prepare_embryo_data(config):
    """
    加载和准备胚胎数据，完全匹配原backup_final.py的策略
    
    Args:
        config: BiLSTMConfig配置对象
        
    Returns:
        tuple: (root_dir, score_dict, valid_embryo_df)
    """
    print("=== 数据加载和准备 ===")
    
    # 获取配置参数
    root_dir = "/liaozizhuo/image"
    maindb_csv_path = config.get('maindb_csv_path') or "/liaozizhuo/scripts/MainDB.csv"
    filtered_csv_path = config.get('filtered_csv_path') or "/liaozizhuo/scripts/filtered_clean_embryos.csv"
    slide_info_path = config.get('slide_info_path') or "/liaozizhuo/scripts/SlidePatientBirthDates.csv"
    
    # 检查必需文件
    required_files = [maindb_csv_path, filtered_csv_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 必需文件不存在: {file_path}")
    
    # 1. 先加载年龄信息
    print("👴 从SlidePatientBirthDates.csv加载年龄信息...")
    print(f"   文件路径: {slide_info_path}")
    age_map = {}
    if os.path.exists(slide_info_path):
        try:
            slide_info_df = pd.read_csv(slide_info_path)
            print(f"✅ 成功加载年龄信息: {slide_info_df.shape[0]} 条患者记录")

            # 提取年龄映射（完全按照原backup策略）
            slide_info_df["EmbryoPatientID"] = slide_info_df["SlideID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
            age_map = dict(zip(slide_info_df["EmbryoPatientID"], slide_info_df["Age"]))
            print(f"✅ 建立年龄映射: {len(age_map)} 个患者")
        except Exception as e:
            print(f"❌ 加载年龄信息失败: {e}")
            age_map = {}
    else:
        print(f"⚠️ 年龄信息文件不存在: {slide_info_path}")

    # 2. 加载评分信息（传入年龄映射）
    print("📋 加载评分信息...")
    score_dict = load_score_dict_backup_style(maindb_csv_path, filtered_csv_path, age_map)

    # 3. 读取胚胎信息
    print("🧬 从filtered_clean_embryos.csv读取胚胎信息...")
    print(f"   文件路径: {filtered_csv_path}")
    try:
        valid_embryo_df = pd.read_csv(filtered_csv_path)
        print(f"✅ 成功加载胚胎信息: {valid_embryo_df.shape[0]} 条图像记录，{len(valid_embryo_df['EmbryoID'].unique())} 个胚胎")
    except Exception as e:
        print(f"❌ 加载胚胎信息失败: {e}")
        raise

    # 4. 将年龄信息从score_dict映射到valid_embryo_df
    print("🔗 将年龄信息映射到胚胎数据...")
    valid_embryo_df["Age"] = valid_embryo_df["EmbryoID"].map(
        lambda eid: score_dict.get(eid, {}).get("age") if isinstance(score_dict.get(eid), dict) else None
    )

    # 🔥 删除缺少年龄信息的胚胎（而不是填充）
    age_missing_images = valid_embryo_df["Age"].isna().sum()
    if age_missing_images > 0:
        print(f"⚠️ {age_missing_images} 张图像缺少年龄信息")
        print(f"🗑️  删除缺少年龄信息的胚胎...")

        # 记录删除前的胚胎和图像数量
        before_images = len(valid_embryo_df)
        before_embryos = len(valid_embryo_df['EmbryoID'].unique()) if 'EmbryoID' in valid_embryo_df.columns else before_images // 33

        # 删除年龄缺失的胚胎
        valid_embryo_df = valid_embryo_df[valid_embryo_df["Age"].notna()].copy()

        # 记录删除后的胚胎和图像数量
        after_images = len(valid_embryo_df)
        after_embryos = len(valid_embryo_df['EmbryoID'].unique()) if 'EmbryoID' in valid_embryo_df.columns else after_images // 33

        removed_images = before_images - after_images
        removed_embryos = before_embryos - after_embryos

        print(f"✅ 删除了 {removed_embryos} 个胚胎（{removed_images} 张图像）")
        print(f"✅ 剩余 {after_embryos} 个胚胎（{after_images} 张图像）")
    else:
        print(f"✅ 所有胚胎都有年龄信息")
    
    # 4. 数据验证
    total_images = len(valid_embryo_df)
    estimated_embryos = total_images // 33  # 估算胚胎数量

    # 🔥 过滤掉fair胚胎，只保留good和poor
    if 'EmbryoQuality' in valid_embryo_df.columns:
        before_filter = len(valid_embryo_df)
        before_embryos = len(valid_embryo_df['EmbryoID'].unique())

        # 过滤fair胚胎
        valid_embryo_df = valid_embryo_df[valid_embryo_df['EmbryoQuality'].isin(['good', 'poor'])].copy()

        after_filter = len(valid_embryo_df)
        after_embryos = len(valid_embryo_df['EmbryoID'].unique())

        removed_images = before_filter - after_filter
        removed_embryos = before_embryos - after_embryos

        if removed_embryos > 0:
            print(f"🗑️ 过滤掉fair胚胎: {removed_embryos} 个胚胎（{removed_images} 张图像）")

    # 重新计算统计信息
    total_images = len(valid_embryo_df)
    estimated_embryos = total_images // 33

    print(f"📊 最终数据量: {estimated_embryos} 个胚胎（{total_images} 张图像）")
    if 'EmbryoQuality' in valid_embryo_df.columns:
        # 按胚胎计数统计质量分布
        embryo_quality_counts = valid_embryo_df.groupby('EmbryoID')['EmbryoQuality'].first().value_counts()
        print(f"📊 质量分布（按胚胎计数）:\n{embryo_quality_counts}")
    print(f"📊 年龄范围: {valid_embryo_df['Age'].min():.1f} - {valid_embryo_df['Age'].max():.1f}")
    
    # 检查数据质量
    if 'EmbryoQuality' in valid_embryo_df.columns:
        quality_counts = valid_embryo_df['EmbryoQuality'].value_counts()
        if len(quality_counts) < 2:
            print("⚠️ 胚胎质量类别不足，可能影响训练效果")
    
    return root_dir, score_dict, valid_embryo_df


def load_score_dict_backup_style(maindb_csv_path, filtered_csv_path, age_map=None):
    """
    按照backup_final.py的策略加载评分字典

    Args:
        age_map: 患者ID前缀 -> 年龄的映射字典
    """
    print(f"📋 从MainDB.csv加载ICM/TE评分数据")
    print(f"   文件路径: {maindb_csv_path}")

    df = pd.read_csv(maindb_csv_path)
    print(f"   原始数据量: {len(df)} 条记录")

    # 🔍 详细统计信息 - 修正列名
    # MainDB.csv使用RowName作为胚胎ID
    embryo_id_col = 'RowName' if 'RowName' in df.columns else 'EmbryoID'
    unique_embryos = df[embryo_id_col].nunique()
    print(f"   去重后胚胎数: {unique_embryos} 个胚胎")

    # 检查是否有重复的胚胎ID
    duplicates = df[df.duplicated(subset=[embryo_id_col], keep=False)]
    if len(duplicates) > 0:
        print(f"   ⚠️ 发现重复胚胎ID: {len(duplicates)} 条记录")
        print(f"   重复胚胎示例: {duplicates[embryo_id_col].head().tolist()}")

    # 检查质量分布（MainDB.csv包含good/fair，poor来自filtered_clean_embryos.csv）
    if 'Quality' in df.columns:
        quality_counts = df['Quality'].value_counts()
        print(f"   MainDB.csv质量分布: {dict(quality_counts)}")

    # ICM和TE映射（完全按照backup策略）
    icm_map = {'A': 1, 'B': 2, 'C': 4}
    te_map = {'A': 1, 'B': 3, 'C': 4}

    df["ICM_score"] = df["ICM"].map(icm_map)
    df["TE_score"] = df["TE"].map(te_map)
    df["BS_prime"] = df["ICM_score"] + df["TE_score"]

    # 🔧 创建包含年龄信息的评分字典
    score_dict = {}
    for _, row in df.iterrows():
        embryo_id = str(row[embryo_id_col])
        bs_score = float(row["BS_prime"])

        # 🔥 通过患者ID前缀匹配年龄
        age = None
        if age_map and embryo_id:
            patient_id_prefix = extract_patient_id_from_embryo_id(embryo_id)
            if patient_id_prefix and patient_id_prefix in age_map:
                age = float(age_map[patient_id_prefix])

        score_dict[embryo_id] = {
            "bs_score": bs_score,
            "age": age,
            "quality": 1 if row.get("Quality") == "good" else 0  # good=1, fair=0
        }

    print(f"✅ MainDB评分加载完成: {len(score_dict)} 个胚胎")

    # 处理filtered_clean_embryos.csv中的poor胚胎（完全按照backup策略）
    if os.path.exists(filtered_csv_path):
        print(f"📋 从filtered_clean_embryos.csv补充poor胚胎数据")
        print(f"   文件路径: {filtered_csv_path}")
        filtered_df = pd.read_csv(filtered_csv_path)
        print(f"   数据量: {len(filtered_df)} 条图像记录")
        if 'EmbryoQuality' in filtered_df.columns:
            poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
            poor_count = 0
            for emb_id in poor_embryos["EmbryoID"].unique():
                if emb_id not in score_dict:
                    # 🔥 为poor胚胎也匹配年龄
                    age = None
                    if age_map and emb_id:
                        patient_id_prefix = extract_patient_id_from_embryo_id(emb_id)
                        if patient_id_prefix and patient_id_prefix in age_map:
                            age = float(age_map[patient_id_prefix])

                    score_dict[emb_id] = {
                        "bs_score": 6.0 + random.uniform(0, 0.5),
                        "age": age,
                        "quality": 0  # poor=0
                    }
                    poor_count += 1

            if poor_count > 0:
                print(f"   补充了 {poor_count} 个poor胚胎的评分")

    print(f"✅ 总计加载了 {len(score_dict)} 个胚胎的评分信息")
    return score_dict

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
        
        # 处理低质量胚胎数据（如果提供）
        if filtered_path and os.path.exists(filtered_path):
            print(f"📂 读取低质量胚胎文件: {filtered_path}")
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
        print("⚠️ 警告：未提供data_root参数，请在配置中指定正确的数据根目录")
        return False
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

def generate_embryo_info_from_score_dict(score_dict_path, output_path, data_root, image_folder="image", require_age=True):
    """
    从评分字典生成胚胎信息CSV文件
    
    Args:
        score_dict_path (str): score_dict.json 文件路径
        output_path (str): 输出的 embryo_info.csv 文件路径
        data_root (str): 数据根目录路径
        image_folder (str): 图像文件夹名称
        require_age (bool): 是否要求必须有年龄信息，True则跳过无年龄信息的胚胎
        
    Returns:
        str: 生成的文件路径
    """
    print("=== 开始生成胚胎信息文件 ===")
    
    # 检查评分字典文件
    if not os.path.exists(score_dict_path):
        raise FileNotFoundError(f"评分字典文件不存在: {score_dict_path}")
    
    print(f"📂 读取评分字典: {score_dict_path}")
    with open(score_dict_path, 'r', encoding='utf-8') as f:
        score_dict = json.load(f)
    print(f"✅ 成功读取 {len(score_dict)} 个胚胎评分")
    
    # ICM/TE映射用于质量分类
    icm_map = {'A': 1, 'B': 2, 'C': 4}
    te_map = {'A': 1, 'B': 3, 'C': 4}
    
    # BS分数到质量的映射
    def bs_score_to_quality(bs_score):
        """将BS分数转换为质量等级"""
        if bs_score <= 2:
            return "Excellent"
        elif bs_score <= 4:
            return "Good" 
        elif bs_score <= 6:
            return "Fair"
        else:
            return "Poor"
    
    # 创建胚胎信息列表
    embryo_info_list = []
    
    for embryo_id, info in score_dict.items():
        # 获取基本信息
        bs_score = info.get("bs_score", 5.0)
        age = info.get("age")
        
        # 如果要求必须有年龄信息但没有找到，跳过这个胚胎
        if require_age and age is None:
            print(f"❌ 跳过胚胎 {embryo_id}：score_dict中缺少年龄信息")
            continue
        
        # 生成质量等级
        quality = bs_score_to_quality(bs_score)
        
        # 生成完整的图像路径
        full_path = os.path.join(data_root, image_folder, embryo_id)
        
        embryo_info_list.append({
            "EmbryoID": embryo_id,
            "FullPath": full_path,
            "EmbryoQuality": quality,
            "Age": int(age)
        })
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(embryo_info_list)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存到CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ 胚胎信息文件已生成: {output_path}")
    print(f"📊 包含 {len(df)} 条记录")
    print("📋 质量等级分布:")
    quality_counts = df['EmbryoQuality'].value_counts()
    for quality, count in quality_counts.items():
        print(f"   {quality}: {count} 个")
    
    return output_path

def load_age_information(slide_info_path):
    """
    从SlidePatientBirthDates.csv加载年龄信息
    
    Args:
        slide_info_path (str): SlidePatientBirthDates.csv 文件路径
        
    Returns:
        dict: 胚胎ID到年龄的映射字典，如果文件不存在返回None
    """
    if not slide_info_path or not os.path.exists(slide_info_path):
        print(f"⚠️ 年龄信息文件不存在: {slide_info_path}")
        return {}

    try:
        print(f"📂 读取年龄信息文件: {slide_info_path}")
        slide_info_df = pd.read_csv(slide_info_path)
        print(f"✅ 成功加载年龄信息: {slide_info_df.shape}")
        print(f"   数据列: {list(slide_info_df.columns)}")

        # 检查必要列
        if 'SlideID' not in slide_info_df.columns:
            print("❌ SlideID列不存在于年龄信息文件中")
            return {}

        if 'Age' not in slide_info_df.columns:
            print("❌ Age列不存在于年龄信息文件中")
            return {}

        # 显示数据示例
        print(f"   前3行数据:")
        print(slide_info_df[['SlideID', 'Age']].head(3).to_string(index=False))

        # 直接使用SlideID作为患者ID，不需要正则提取
        # 因为年龄文件中的SlideID格式就是: D2024.03.14_S00001_I5068_P

        # 过滤掉年龄为0的记录（无效年龄）
        valid_ages = slide_info_df[slide_info_df["Age"] > 0]
        print(f"   有效年龄记录: {len(valid_ages)} / {len(slide_info_df)}")

        # 创建年龄映射：直接用SlideID
        age_map = dict(zip(valid_ages["SlideID"], valid_ages["Age"]))

        # 移除空键值
        age_map = {k: v for k, v in age_map.items() if k is not None and not pd.isna(k)}

        print(f"✅ 提取到 {len(age_map)} 个患者的年龄信息")
        if len(age_map) > 0:
            sample_keys = list(age_map.keys())[:3]
            print(f"   年龄映射示例: {[(k, age_map[k]) for k in sample_keys]}")

        return age_map
            
    except Exception as e:
        print(f"❌ 加载年龄信息失败: {e}")
        return None


def extract_patient_id_from_embryo_id(embryo_id):
    """
    从胚胎ID中提取患者ID前缀
    
    Args:
        embryo_id (str): 胚胎ID，如 'D2023.09.04_S00918_I4425_P_10'
        
    Returns:
        str: 患者ID前缀，如 'D2023.09.04_S00918_I4425_P'
    """
    import re
    match = re.match(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)", embryo_id)
    if match:
        return match.group(1)
    return None


def generate_score_dict_from_csv(maindb_path, filtered_path=None, slide_info_path=None, output_path="score_dict.json"):
    """
    胚胎评分字典生成逻辑:
    1. MainDB.csv → 提取ICM/TE评分 → 分类为good/fair
    2. filtered_clean_embryos.csv → 补充poor胚胎信息
    3. 根据患者ID前缀赋予年龄(同一患者所有胚胎年龄相同)
    4. 删除无法匹配年龄的胚胎

    Args:
        maindb_path (str): MainDB.csv 文件路径
        filtered_path (str, optional): filtered_clean_embryos.csv 文件路径
        slide_info_path (str, optional): SlidePatientBirthDates.csv 年龄信息文件路径
        output_path (str): 输出的 score_dict.json 文件路径

    Returns:
        dict: 生成的评分字典
    """
    import json

    print("=== 开始生成评分字典 ===")

    # 加载年龄信息
    age_map = load_age_information(slide_info_path) if slide_info_path else {}
    if age_map and len(age_map) > 0:
        print(f"✅ 成功加载年龄映射: {len(age_map)} 个患者")
    else:
        print("⚠️ 年龄映射为空")
        age_map = {}
    
    # 检查主数据文件
    if not os.path.exists(maindb_path):
        raise FileNotFoundError(f"主数据文件不存在: {maindb_path}")

    print(f"📂 读取主数据文件: {maindb_path}")
    df = pd.read_csv(maindb_path)
    print(f"✅ 成功读取 {len(df)} 行数据")
    
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
        embryo_sample_count = 0
        for _, row in df.iterrows():
            embryo_id = str(row["EmbryoID"])
            bs_score = float(row["BS_prime"])

            
            # 尝试从年龄映射中获取年龄
            age = None
            age_source = None
            
            if age_map and embryo_id:
                # 从胚胎ID提取患者ID前缀
                patient_id_prefix = extract_patient_id_from_embryo_id(embryo_id)
                if patient_id_prefix and patient_id_prefix in age_map:
                    age = float(age_map[patient_id_prefix])
                    age_source = "SlidePatientBirthDates.csv"
            
            if age is None and "Age" in df.columns and pd.notna(row.get("Age")):
                # 如果CSV文件中直接有Age列
                age = float(row["Age"])
                age_source = "MainDB.csv"
            
            if age is None:
                # 🔧 给默认年龄0，后续根据患者ID补充，最后删除年龄为0的胚胎
                age = 0.0
                # print(f"⚠️ 胚胎 {embryo_id} 暂时使用默认年龄: {age}")  # 减少日志输出
            
            # 处理Quality字段，支持文本和数字格式
            if "Quality" in df.columns:
                quality_val = row.get("Quality", 1)
                if isinstance(quality_val, str):
                    # 文本质量映射
                    # 🔥 包含所有质量类型，fair在训练时过滤
                    quality_map = {
                        'good': 1, 'high': 1, 'excellent': 1,
                        'fair': 0,  # fair胚胎标记为0，训练时会过滤
                        'poor': 0, 'bad': 0, 'low': 0
                    }
                    quality = quality_map.get(quality_val.lower(), 1 if bs_score <= 4 else 0)
                else:
                    quality = int(quality_val)
            else:
                quality = 1 if bs_score <= 4 else 0
            
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
            
            # 尝试从年龄映射中获取年龄
            age = None
            age_source = None
            
            if age_map and embryo_id:
                # 从胚胎ID提取患者ID前缀
                patient_id_prefix = extract_patient_id_from_embryo_id(embryo_id)
                if patient_id_prefix and patient_id_prefix in age_map:
                    age = float(age_map[patient_id_prefix])
                    age_source = "SlidePatientBirthDates.csv"
            
            if age is None and "Age" in df.columns and pd.notna(row.get("Age")):
                # 如果CSV文件中直接有Age列
                age = float(row["Age"])
                age_source = "MainDB.csv"
            
            if age is None:
                # 🔧 给默认年龄0，后续根据患者ID补充，最后删除年龄为0的胚胎
                age = 0.0
                # print(f"⚠️ 胚胎 {embryo_id} 暂时使用默认年龄: {age}")  # 减少日志输出
            
            # 处理Quality字段，支持文本和数字格式
            if "Quality" in df.columns:
                quality_val = row.get("Quality", 1)
                if isinstance(quality_val, str):
                    # 文本质量映射
                    # 🔥 包含所有质量类型，fair在训练时过滤
                    quality_map = {
                        'good': 1, 'high': 1, 'excellent': 1,
                        'fair': 0,  # fair胚胎标记为0，训练时会过滤
                        'poor': 0, 'bad': 0, 'low': 0
                    }
                    quality = quality_map.get(quality_val.lower(), 1 if bs_score <= 4 else 0)
                else:
                    quality = int(quality_val)
            else:
                quality = 1 if bs_score <= 4 else 0
            
            score_dict[embryo_id] = {
                "bs_score": bs_score,
                "age": age,
                "quality": quality
            }
    else:
        raise ValueError("CSV文件必须包含 ICM/TE 列或 BlastocystScore 列")
    
    print(f"✅ 从主数据文件生成 {len(score_dict)} 个胚胎评分")
    
    # 处理低质量胚胎数据
    if filtered_path and os.path.exists(filtered_path):
        print(f"📂 读取low quality胚胎文件: {filtered_path}")
        filtered_df = pd.read_csv(filtered_path)
        
        # 查找低质量胚胎
        if "EmbryoQuality" in filtered_df.columns:
            poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
            print(f"🔍 找到 {len(poor_embryos)} 个低质量胚胎")
            
            for emb_id in poor_embryos["EmbryoID"].unique():
                emb_id_str = str(emb_id)
                if emb_id_str not in score_dict:
                    # 为低质量胚胎分配较高的BS分数 (6.0-6.5)
                    bs_score = 6.0 + random.uniform(0, 0.5)
                    age = 0.0  # 默认年龄0，后续补充
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

    # 根据患者ID补充年龄信息
    print("🔗 根据患者ID补充年龄信息...")
    if age_map:
        age_补充_count = 0
        for embryo_id, data in score_dict.items():
            if data["age"] == 0.0:
                patient_id_prefix = extract_patient_id_from_embryo_id(embryo_id)
                if patient_id_prefix and patient_id_prefix in age_map:
                    data["age"] = float(age_map[patient_id_prefix])
                    age_补充_count += 1

        print(f"✅ 为 {age_补充_count} 个胚胎补充了年龄信息")
    else:
        print("❌ 年龄映射为空，无法补充年龄")

    # 🔥 删除所有年龄为0的胚胎
    print("🗑️ 删除年龄为0的胚胎...")
    before_count = len(score_dict)
    score_dict = {k: v for k, v in score_dict.items() if v["age"] != 0.0}
    after_count = len(score_dict)
    removed_count = before_count - after_count

    if removed_count > 0:
        print(f"✅ 删除了 {removed_count} 个缺少年龄信息的胚胎")
        print(f"✅ 剩余 {after_count} 个胚胎有完整信息")
    else:
        print(f"✅ 所有胚胎都有年龄信息")

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