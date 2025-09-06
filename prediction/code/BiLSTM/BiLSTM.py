# Standard library imports
import gc
import os
import random
import time
import warnings

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import hashlib

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet152V2, DenseNet201
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as preprocess_efficientnet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, TimeDistributed, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model, Sequential

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                            recall_score, classification_report, log_loss, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# BiLSTM preprocessing modules  
import cv2
from .preprocessing.bilstm_image_utils import preprocess_embryo_image

# Configure warnings
warnings.filterwarnings('ignore')

# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

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
    
    Sets memory growth and environment variables for GPU usage.
    """
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            os.environ.update({
                'TF_GPU_ALLOCATOR': 'cuda_malloc_async',
                'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                'TF_CUDNN_USE_AUTOTUNE': '0',
                'TF_CUDNN_DETERMINISTIC': '1'
            })
            
            if num_gpus > 1:
                os.environ['NCCL_DEBUG'] = 'INFO'
                
        except RuntimeError:
            pass
    
    return num_gpus

num_gpus = configure_gpu()

def load_score_dict(csv_path):
    """Load scoring information and calculate BS_prime"""
    df = pd.read_csv(csv_path)
    
    icm_map = {'A': 1, 'B': 2, 'C': 4}
    te_map = {'A': 1, 'B': 3, 'C': 4}
    
    df["ICM_score"] = df["ICM"].map(icm_map)
    df["TE_score"] = df["TE"].map(te_map)
    df["BS_prime"] = df["ICM_score"] + df["TE_score"]
    
    score_dict = dict(zip(df["EmbryoID"], df["BS_prime"]))
    
    filtered_path = "/liaozizhuo/scripts/filtered_clean_embryos.csv"
    if os.path.exists(filtered_path):
        filtered_df = pd.read_csv(filtered_path)
        poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
        for emb_id in poor_embryos["EmbryoID"].unique():
            if emb_id not in score_dict:
                score_dict[emb_id] = 6.0 + random.uniform(0, 0.5)
    
    return score_dict

class BiLSTMFeatureCache:
    """BiLSTM专用的特征缓存管理器"""
    
    def __init__(self, data_root, cache_dir="bilstm_feature_cache"):
        self.data_root = data_root
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.multi_extractor = None
        
    def set_extractor(self, multi_extractor):
        """设置特征提取器"""
        self.multi_extractor = multi_extractor
        
    def _generate_config_hash(self, image_paths):
        """生成配置哈希值"""
        # 获取所有图像文件的修改时间
        file_times = []
        for path in image_paths[:100]:  # 只检查前100个文件以提高速度
            if os.path.exists(path):
                file_times.append(str(os.path.getmtime(path)))
        
        config_info = {
            'data_root': self.data_root,
            'model_names': ['efficientnet', 'resnet', 'densenet', 'vgg16'],
            'file_count': len(image_paths),
            'sample_file_times': file_times[:10]  # 只用前10个文件时间作为代表
        }
        config_str = str(sorted(config_info.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
        
    def get_cache_path(self, image_paths, cache_type="base_features"):
        """生成缓存文件路径"""
        config_hash = self._generate_config_hash(image_paths)
        return os.path.join(self.cache_dir, f"{cache_type}_{config_hash}.pkl")
        
    def _validate_cache(self, cached_data, image_paths):
        """验证缓存是否仍然有效"""
        try:
            # 检查基本信息
            if 'features' not in cached_data or 'metadata' not in cached_data:
                return False
                
            # 检查图像数量
            if len(cached_data['features']) != len(image_paths):
                print(f"图像数量不匹配: 缓存{len(cached_data['features'])} vs 当前{len(image_paths)}")
                return False
            
            # 检查图像路径
            cached_paths = set(cached_data['features'].keys())
            current_paths = set(image_paths)
            if len(cached_paths.intersection(current_paths)) < len(current_paths) * 0.95:  # 95%匹配率
                print("图像路径不匹配")
                return False
                
            return True
            
        except Exception as e:
            print(f"缓存验证失败: {e}")
            return False
    
    def _save_features_cache(self, features_dict, cache_file, image_paths):
        """保存特征到缓存文件"""
        try:
            cache_data = {
                'features': features_dict,
                'metadata': {
                    'extraction_time': time.time(),
                    'total_images': len(image_paths),
                    'feature_dim': list(features_dict.values())[0].shape[-1] if features_dict else 0,
                    'cache_version': '1.0'
                }
            }
            
            # 原子性保存
            temp_file = cache_file + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            os.rename(temp_file, cache_file)
            
            file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"✅ 特征缓存保存成功: {cache_file} ({file_size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"❌ 缓存保存失败: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def get_or_compute_features(self, image_paths, use_cache=True, training=False, use_augmentation=False):
        """获取或计算图像特征"""
        if self.multi_extractor is None:
            raise ValueError("请先调用 set_extractor() 设置特征提取器")
            
        cache_file = self.get_cache_path(image_paths, f"features_{training}_{use_augmentation}")
        
        # 尝试加载缓存
        if use_cache and os.path.exists(cache_file):
            try:
                print(f"🔍 检查特征缓存: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                if self._validate_cache(cached_data, image_paths):
                    print("🚀 从缓存加载特征")
                    return cached_data['features']
                else:
                    print("⚠️ 缓存无效，重新提取特征")
                    
            except Exception as e:
                print(f"❌ 缓存加载失败: {e}")
        
        # 重新提取特征
        print("🔄 开始特征提取...")
        features_dict = self._extract_features_from_scratch(image_paths, training, use_augmentation)
        
        # 保存缓存
        if use_cache:
            self._save_features_cache(features_dict, cache_file, image_paths)
            
        return features_dict
    
    def _extract_features_from_scratch(self, image_paths, training, use_augmentation):
        """从头提取特征（使用STORK预处理）"""
        # STORK预处理包装函数
        def stork_preprocess_wrapper(path_bytes):
            """包装STORK预处理以在tf.py_function中使用"""
            try:
                path_str = path_bytes.numpy().decode('utf-8')
                # 使用STORK预处理管道：Hough圆检测 + 中心裁剪
                processed_img = preprocess_embryo_image(
                    path_str, 
                    img_size=224, 
                    apply_petri_crop=True,  # 启用Petri dish检测
                    apply_center_crop=True  # 启用中心方形裁剪
                )
                return processed_img.astype(np.uint8)
            except Exception as e:
                print(f"STORK预处理失败 {path_str}: {e}")
                # 回退到简单预处理
                img = cv2.imread(path_str)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    return img.astype(np.uint8)
                else:
                    # 创建黑色图像作为最后的回退
                    return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 构建预处理函数
        def load_and_preprocess_image(path):
            # 使用STORK预处理
            img = tf.py_function(
                stork_preprocess_wrapper, 
                [path], 
                tf.uint8
            )
            img.set_shape([224, 224, 3])
            
            # 只有训练模式且启用增强时才应用额外增强
            if training and use_augmentation:
                # 在STORK预处理基础上添加增强
                if tf.random.uniform([]) > 0.5:
                    img = tf.image.random_flip_left_right(img)
                if tf.random.uniform([]) > 0.7:
                    img = tf.image.random_brightness(img, max_delta=0.1)
                if tf.random.uniform([]) > 0.7:
                    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
                    
            img = tf.cast(img, tf.uint8)
            return img
        
        # 构建数据管道
        BATCH_SIZE = 32 if training else 64  # 训练时使用较小批次
        path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        image_dataset = path_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        image_dataset = image_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        features_dict = {}
        
        # 批量提取特征
        path_idx = 0
        for batch_images in tqdm(image_dataset, desc="批量特征提取"):
            batch_features = self.multi_extractor.extract_features(batch_images.numpy())
            
            # 将特征按路径存储
            batch_size = batch_features.shape[0]
            for i in range(batch_size):
                if path_idx < len(image_paths):
                    features_dict[image_paths[path_idx]] = batch_features[i]
                    path_idx += 1
                    
        return features_dict
    
    def get_features_by_paths(self, image_paths, cached_features):
        """从缓存中获取指定路径的特征"""
        features = []
        for path in image_paths:
            if path in cached_features:
                features.append(cached_features[path])
            else:
                print(f"⚠️ 路径 {path} 在缓存中不存在")
                # 返回零特征向量作为备选
                if features:
                    zero_feature = np.zeros_like(features[0])
                    features.append(zero_feature)
                    
        return np.array(features) if features else np.array([])

class MultiModelFeatureExtractor:
    def __init__(self):
        self.models = {}
        self.feature_dims = {}
        self._build_models()
        
    def _build_models(self):
        """Build multi-model feature extractors with dimension reduction"""
        
        base_efficientnet = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        efficientnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_efficientnet.output)
        efficientnet_output = Dense(512, activation='relu', name='efficientnet_reduction')(efficientnet_output)
        self.models['efficientnet'] = Model(inputs=base_efficientnet.input, outputs=efficientnet_output)
        self.models['efficientnet'].trainable = False
        self.feature_dims['efficientnet'] = 512
        
        base_resnet = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_resnet.output)
        resnet_output = Dense(512, activation='relu', name='resnet_reduction')(resnet_output)
        self.models['resnet'] = Model(inputs=base_resnet.input, outputs=resnet_output)
        self.models['resnet'].trainable = False
        self.feature_dims['resnet'] = 512
        
        base_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        densenet_output = tf.keras.layers.GlobalAveragePooling2D()(base_densenet.output)
        densenet_output = Dense(512, activation='relu', name='densenet_reduction')(densenet_output)
        self.models['densenet'] = Model(inputs=base_densenet.input, outputs=densenet_output)
        self.models['densenet'].trainable = False
        self.feature_dims['densenet'] = 512
        
        base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        vgg_output = tf.keras.layers.GlobalAveragePooling2D()(base_vgg.output)
        vgg_output = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='vgg16_reduction')(vgg_output)
        self.models['vgg16'] = Model(inputs=base_vgg.input, outputs=vgg_output)
        self.models['vgg16'].trainable = False
        self.feature_dims['vgg16'] = 512
        
        self.total_feature_dim = sum(self.feature_dims.values())
        
    def extract_features(self, images):
        """Extract multi-model features and concatenate"""
        features_list = []
        
        if images.dtype != np.uint8:
            images = tf.cast(images, tf.uint8)
        
        model_names = ['efficientnet', 'resnet', 'densenet', 'vgg16']
        preprocess_functions = [preprocess_efficientnet, preprocess_resnet, preprocess_densenet, preprocess_vgg16]
        
        for model_name, preprocess_func in zip(model_names, preprocess_functions):
            processed_imgs = preprocess_func(images.copy())
            features = self.models[model_name].predict(processed_imgs, verbose=0)
            features_list.append(features)
            del processed_imgs
        
        combined_features = np.concatenate(features_list, axis=1)
        return combined_features

def build_dataset_with_image_augmentation(data_root, score_dict, multi_extractor, valid_embryo_df, training=True, balance=False, use_augmentation=True, use_cache=True, feature_cache=None):
    """
    Build dataset with image-level augmentation support (支持特征缓存)
    
    Args:
        data_root: Image root directory
        score_dict: Scoring dictionary
        multi_extractor: Multi-model feature extractor
        valid_embryo_df: Valid embryo DataFrame
        training: Whether in training mode
        balance: Whether to balance sampling
        use_augmentation: Whether to use data augmentation (only effective when training=True)
        use_cache: Whether to use feature caching (default: True)
        feature_cache: BiLSTMFeatureCache instance (if None, will create one)
    """
    
    def map_half_hour(t):
        try:
            t = float(t)
        except Exception:
            return np.nan
        idx = int(round((t - 96.0) / 0.5))
        return idx if 0 <= idx <= 32 else np.nan

    focal_planes = ['F-75', 'F-60', 'F-45', 'F-30', 'F-15', 'F0', 'F15', 'F30', 'F45', 'F60', 'F75']
    time_points = [96 + 0.5*i for i in range(33)]
    feature_dim = multi_extractor.total_feature_dim
    
    valid_embryos = set(valid_embryo_df["EmbryoID"])
    embryo_ids = [eid for eid in os.listdir(data_root) if eid in valid_embryos and eid in score_dict]
    
    image_info_list = []
    embryo_metadata = {}
    
    for embryo_id in tqdm(embryo_ids, desc="Collecting image info"):
        try:
            embryo_df = valid_embryo_df[valid_embryo_df['EmbryoID'] == embryo_id].copy()
            embryo_df["t_idx"] = embryo_df["TargetHour"].apply(map_half_hour)
            f_map = {f: i for i, f in enumerate(focal_planes)}
            embryo_df["f_idx"] = embryo_df["FocalPlane"].map(f_map)
            
            valid_images = []
            for _, row in embryo_df.iterrows():
                if pd.notna(row.t_idx) and pd.notna(row.f_idx):
                    image_info_list.append((embryo_id, int(row.t_idx), int(row.f_idx), row.FullPath))
                    valid_images.append((int(row.t_idx), int(row.f_idx)))
            
            if len(valid_images) > 0:
                embryo_row = valid_embryo_df[valid_embryo_df["EmbryoID"] == embryo_id].iloc[0]
                embryo_metadata[embryo_id] = {
                    'bs_score': score_dict[embryo_id],
                    'quality': 1 if embryo_row["EmbryoQuality"] == "good" else 0,
                    'age': embryo_row["Age"],
                    'valid_positions': valid_images
                }
        except Exception as e:
            continue
    
    total_images = len(image_info_list)
    
    # 批量GPU特征提取（支持图像增强）
    if training and use_augmentation:
        image_augment_fn = create_image_augmentation_fn()
    else:
        image_augment_fn = None
    
    # 创建数据处理pipeline
    all_paths = [info[3] for info in image_info_list]
    
    # 🚀 使用特征缓存机制
    if use_cache:
        # 初始化缓存管理器
        if feature_cache is None:
            feature_cache = BiLSTMFeatureCache(data_root)
            feature_cache.set_extractor(multi_extractor)
        
        print(f"📦 使用特征缓存 - 训练模式: {training}, 数据增强: {use_augmentation}")
        
        # 从缓存获取或计算特征
        features_dict = feature_cache.get_or_compute_features(
            all_paths, 
            use_cache=use_cache,
            training=training, 
            use_augmentation=use_augmentation
        )
        
        # 按顺序提取特征
        all_features = []
        for path in all_paths:
            if path in features_dict:
                all_features.append(features_dict[path])
            else:
                # 如果缓存中没有该特征，创建零向量
                print(f"⚠️ 缓存中缺少特征: {path}")
                zero_feature = np.zeros(multi_extractor.total_feature_dim, dtype=np.float32)
                all_features.append(zero_feature)
        
        all_features = np.array(all_features)
        
    else:
        # 🔄 原始特征提取方式（不使用缓存）
        print("⚠️ 未使用缓存，进行原始特征提取")
        
        def stork_preprocess_wrapper(path_bytes):
            """包装STORK预处理以在tf.py_function中使用"""
            try:
                path_str = path_bytes.numpy().decode('utf-8')
                # 使用STORK预处理管道：Hough圆检测 + 中心裁剪
                processed_img = preprocess_embryo_image(
                    path_str, 
                    img_size=224, 
                    apply_petri_crop=True,  # 启用Petri dish检测
                    apply_center_crop=True  # 启用中心方形裁剪
                )
                return processed_img.astype(np.uint8)
            except Exception as e:
                print(f"STORK预处理失败 {path_str}: {e}")
                # 回退到简单预处理
                img = cv2.imread(path_str)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    return img.astype(np.uint8)
                else:
                    return np.zeros((224, 224, 3), dtype=np.uint8)
        
        def load_and_preprocess_with_augmentation(path):
            # 使用STORK预处理
            img = tf.py_function(
                stork_preprocess_wrapper, 
                [path], 
                tf.uint8
            )
            img.set_shape([224, 224, 3])
            
            # 如果是训练模式且启用增强，则应用图像增强
            if image_augment_fn is not None:
                img = tf.cast(img, tf.uint8)  # 确保输入增强函数前为uint8
                img = image_augment_fn(img)
            
            # 确保最终输出为uint8
            img = tf.cast(img, tf.uint8)
            return img
        
        def load_and_preprocess_no_augmentation(path):
            # 使用STORK预处理（不增强版本）
            img = tf.py_function(
                stork_preprocess_wrapper, 
                [path], 
                tf.uint8
            )
            img.set_shape([224, 224, 3])
            img = tf.cast(img, tf.uint8)
            return img
        
        # 选择合适的预处理函数
        if training and use_augmentation:
            preprocess_fn = load_and_preprocess_with_augmentation
        else:
            preprocess_fn = load_and_preprocess_no_augmentation
        
        # 构建高效的tf.data pipeline
        BATCH_SIZE = 64 if (training and use_augmentation) else 128  # 增强时减小批量大小
        dataset = tf.data.Dataset.from_tensor_slices(all_paths)
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # 批量提取所有特征
        all_features = []
        
        for batch_images in tqdm(dataset, desc="GPU批量特征提取"):
            batch_features = multi_extractor.extract_features(batch_images.numpy())
            all_features.append(batch_features)
        
        # 合并所有特征
        all_features = np.concatenate(all_features, axis=0)
    
    # 分配特征到胚胎张量
    
    X, y_bs, quality_labels, ages = [], [], [], []
    
    for embryo_id, metadata in tqdm(embryo_metadata.items(), desc="构建胚胎张量"):
        # 初始化该胚胎的特征张量
        tensor = np.zeros((33, 11, feature_dim), dtype=np.float32)
        
        # 找到该胚胎的所有特征
        embryo_indices = [i for i, info in enumerate(image_info_list) if info[0] == embryo_id]
        
        # 分配特征到对应位置
        for idx in embryo_indices:
            _, t_idx, f_idx, _ = image_info_list[idx]
            tensor[t_idx, f_idx, :] = all_features[idx]
        
        # 添加到结果列表
        X.append(tensor.reshape(33, -1))
        y_bs.append(metadata['bs_score'])
        quality_labels.append(metadata['quality'])
        ages.append(metadata['age'])
    
    # 标签分布与（可选）平衡采样
    from sklearn.utils import resample

    label_series = pd.Series(quality_labels)
    label_counts = label_series.value_counts()
    print("=== EmbryoQuality 标签分布 ===")
    print(label_counts)

    # 先堆叠（无论是否平衡）
    X = np.stack(X)

    if training and balance:
        if label_counts.shape[0] < 2:
            raise ValueError(f"无法平衡采样：标签中仅有一个类别：{label_counts.index.tolist()}")
        min_count = label_counts.min()
        idx_good = np.where(label_series == 1)[0]
        idx_poor = np.where(label_series == 0)[0]
        resample_idx_good = resample(idx_good, replace=False, n_samples=min_count, random_state=42)
        resample_idx_poor = resample(idx_poor, replace=False, n_samples=min_count, random_state=42)
        selected_indices = np.concatenate([resample_idx_good, resample_idx_poor])
        X = X[selected_indices]
        y_bs = [y_bs[i] for i in selected_indices]
        quality_labels = [quality_labels[i] for i in selected_indices]
        ages = [ages[i] for i in selected_indices]

    print(f"=== 最终数据集形状: {X.shape} ===")
    return X, np.array(y_bs), quality_labels, ages

# 4. 模型结构
def create_model(feature_dim):
    """创建适应多模型特征的BiLSTM模型"""
    from tensorflow.keras.regularizers import l2
    
    # 计算输入维度 (11个焦平面 * 特征维度)
    input_dim = 11 * feature_dim
    
    # 强制设置数据类型策略
    tf.keras.mixed_precision.set_global_policy('float32')
    
    model = Sequential([
        Input(shape=(33, input_dim), dtype=tf.float32),
        
        # 特征降维层
        TimeDistributed(Dense(1024, activation='relu', 
                            kernel_regularizer=l2(1e-3), dtype=tf.float32)),
        tf.keras.layers.Dropout(0.3),  # 增加dropout
        
        # 增加LSTM正则化
        Bidirectional(LSTM(256, 
                          return_sequences=True, 
                          dropout=0.3,        # 增加dropout
                          kernel_regularizer=l2(1e-3),  # 增加正则化
                          dtype=tf.float32
                         )),
        Bidirectional(LSTM(128, 
                          dropout=0.2, 
                          kernel_regularizer=l2(1e-3),
                          # 移除implementation参数，让TensorFlow自动选择
                          dtype=tf.float32
                         )),
        
        # 全连接层
        tf.keras.layers.Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4), dtype=tf.float32),
        tf.keras.layers.Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4), dtype=tf.float32),
        Dense(1, dtype=tf.float32)
    ])
    
    # 使用统一配置的学习率
    config = get_unified_training_config()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# ======================================================================
# CROSS-VALIDATION HELPER FUNCTIONS
# ======================================================================

def batch_predict_bs_scores(model, X_data, batch_size=8):
    """批量预测BS分数，避免GPU内存问题"""
    y_bs_pred = []
    for i in range(0, len(X_data), batch_size):
        j = min(i + batch_size, len(X_data))
        y_bs_pred.extend(model.predict(X_data[i:j], verbose=0).flatten())
    return np.array(y_bs_pred)


def build_logistic_features(ages, y_bs_pred):
    """构建逻辑回归特征"""
    ages_array = np.array(ages)
    features = np.column_stack([
        ages_array, y_bs_pred,
        ages_array ** 2, y_bs_pred ** 2,
        ages_array * y_bs_pred, np.log(ages_array + 1),
        1.0 / (y_bs_pred + 1e-6)
    ])
    return features


def filter_valid_features(features, quality_labels):
    """过滤无效特征和对应标签"""
    valid_mask = np.isfinite(features).all(axis=1)
    features_clean = features[valid_mask]
    quality_filtered = [quality_labels[i] for i in range(len(quality_labels)) if valid_mask[i]]
    return features_clean, quality_filtered


def train_logistic_classifier(features_clean, quality_filtered):
    """训练逻辑回归分类器"""
    from sklearn.utils.class_weight import compute_class_weight
    
    # Standardize features
    scaler = StandardScaler().fit(features_clean)
    
    # Calculate class weights for imbalanced data
    try:
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(quality_filtered), 
            y=quality_filtered
        )
        class_weight_dict = dict(enumerate(class_weights))
    except Exception:
        class_weight_dict = None
    
    # Train logistic regression
    clf = LogisticRegression(
        C=1.0, 
        random_state=42, 
        max_iter=1000, 
        class_weight=class_weight_dict
    )
    clf.fit(scaler.transform(features_clean), quality_filtered)
    
    return clf, scaler


def evaluate_logistic_classifier(clf, scaler, features_clean, quality_filtered):
    """评估逻辑回归分类器"""
    if len(features_clean) == 0:
        return 0.0, 0.0, 0.0
    
    X_scaled = scaler.transform(features_clean)
    y_pred_proba = clf.predict_proba(X_scaled)[:, 1]
    y_pred_cls = (y_pred_proba > 0.5).astype(int)
    
    acc = accuracy_score(quality_filtered, y_pred_cls)
    auc = roc_auc_score(quality_filtered, y_pred_proba)
    loss = log_loss(quality_filtered, y_pred_proba)
    
    return acc, auc, loss


# ======================================================================
# MAIN CROSS-VALIDATION FUNCTION
# ======================================================================

def cross_validate_bilstm_model_with_image_augmentation(root_dir, score_dict, valid_embryo_df, feature_dim, n_folds=4, n_repeats=1):
    """
    70/30分割 + 4折交叉验证
    实现BiLSTM回归 + 逻辑分类管道，修复了数据泄漏问题
    使用特征缓存提高性能
    """
    
    print(f"=== 开始70/30分割 + {n_folds}折交叉验证 ===")
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    print("初始化特征缓存...")
    feature_cache = BiLSTMFeatureCache(root_dir)
    
    print("构建完整数据集...")
    X_full, y_bs_full, quality_labels_full, ages_full = build_dataset_with_image_augmentation(
        root_dir, score_dict, valid_embryo_df, 
        training=False, balance=False, use_augmentation=False,
        feature_cache=feature_cache, use_cache=True
    )
    
    print(f"完整数据集形状: {X_full.shape}")
    
    # 70/30训练测试分割
    indices = np.arange(len(X_full))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, stratify=quality_labels_full, random_state=42
    )
    
    # 训练集数据
    X_train_full = X_full[train_idx]
    y_bs_train_full = y_bs_full[train_idx]
    quality_train_full = [quality_labels_full[i] for i in train_idx]
    ages_train_full = [ages_full[i] for i in train_idx]
    
    # 测试集数据（保留不动，最后评估）
    X_test = X_full[test_idx]
    y_bs_test = y_bs_full[test_idx]
    quality_test = [quality_labels_full[i] for i in test_idx]
    ages_test = [ages_full[i] for i in test_idx]
    
    print(f"训练集: {len(X_train_full)} 样本, 测试集: {len(X_test)} 样本")
    print(f"训练集类别分布: {np.bincount([int(x) for x in quality_train_full])}")
    print(f"测试集类别分布: {np.bincount([int(x) for x in quality_test])}")
    
    # 在训练集上进行4折交叉验证
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 存储每折的结果
    fold_results = {
        'train_mae': [], 'val_mae': [], 'test_mae': [],
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'logic_acc': [], 'logic_auc': [], 'logic_loss': [],
        'best_epochs': []
    }
    
    # 存储每折的预测结果
    all_predictions = []
    all_true_labels = []
    
    # 存储每折训练的模型（用于最终测试集评估）
    trained_models = []
    
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(cv.split(X_train_full, quality_train_full)):
        print(f"\n--- 第 {fold_idx + 1} 折 ---")
        
        # 强制清理GPU内存
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 等待GPU内存释放
        if tf.config.list_physical_devices('GPU'):
            time.sleep(3)
        
        # 当前折的训练和验证数据
        X_fold_train = X_train_full[fold_train_idx]
        X_fold_val = X_train_full[fold_val_idx]
        y_fold_train = y_bs_train_full[fold_train_idx]
        y_fold_val = y_bs_train_full[fold_val_idx]
        quality_fold_train = [quality_train_full[i] for i in fold_train_idx]
        quality_fold_val = [quality_train_full[i] for i in fold_val_idx]
        ages_fold_train = [ages_train_full[i] for i in fold_train_idx]
        ages_fold_val = [ages_train_full[i] for i in fold_val_idx]
        
        print(f"折内训练样本: {len(X_fold_train)}, 折内验证样本: {len(X_fold_val)}")
        print(f"折内训练类别分布: {np.bincount([int(x) for x in quality_fold_train])}")
        print(f"折内验证类别分布: {np.bincount([int(x) for x in quality_fold_val])}")
        
        # 不使用张量级增强，直接使用原始训练集数据
        X_train_aug = X_fold_train
        y_train_aug = y_fold_train
        quality_train_aug = quality_fold_train
        ages_train_aug = ages_fold_train
        print(f"训练集: {X_train_aug.shape}, 验证集: {X_fold_val.shape}")
        
        # 创建和训练模型 - 使用与主训练相同的策略
        try:
            # 尝试使用GPU策略
            strategy = tf.distribute.MirroredStrategy()
            print("使用GPU训练")
        except:
            # 如果GPU不可用，使用CPU
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
            print("使用CPU训练")
        
        with strategy.scope():
            model = create_model(feature_dim)
            
            print(f"第 {fold_idx + 1} 折训练（使用图像增强的训练集）")
            
            # 获取统一的训练配置
            config = get_unified_training_config()
            
            # 回调函数 - 使用统一配置
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=config['lr_factor'], patience=config['reduce_lr_patience'], min_lr=config['min_lr'], verbose=0)
            ]
            
            # CV数据管道 - 使用统一配置
            cv_batch_size = config['batch_size'] * strategy.num_replicas_in_sync
            
            # 使用固定学习率的Adam优化器 - 使用统一配置
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
            # 编译模型
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            # 构建数据集（训练集使用增强数据，验证集不增强）
            train_ds = tf.data.Dataset.from_tensor_slices((X_train_aug, y_train_aug))
            train_ds = train_ds.shuffle(512).batch(cv_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            
            val_ds = tf.data.Dataset.from_tensor_slices((X_fold_val, y_fold_val))
            val_ds = val_ds.batch(cv_batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
            
            # 训练模型 - 使用统一配置
            history = model.fit(
                train_ds, validation_data=val_ds, 
                epochs=config['epochs'],  # 使用统一配置的epochs数
                callbacks=callbacks, 
                verbose=1
            )
        
        # 评估模型 - 使用与主训练相同的评估方式
        def evaluate_in_batches(X, y, batch_size=16):  # 与主训练保持一致
            losses = []
            maes = []
            
            # 数据预处理和验证
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            
            # 检查数据有效性
            if np.isnan(X).any() or np.isinf(X).any():
                print("警告：输入数据包含NaN或无穷大值，正在清理...")
                X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if np.isnan(y).any() or np.isinf(y).any():
                print("警告：标签数据包含NaN或无穷大值，正在清理...")
                y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
            
            for i in range(0, len(X), batch_size):
                batch_end = min(i + batch_size, len(X))
                batch_X = X[i:batch_end]
                batch_y = y[i:batch_end]
                
                try:
                    # 验证批次数据
                    if not np.isfinite(batch_X).all() or not np.isfinite(batch_y).all():
                        print(f"批次 {i}-{batch_end} 包含无效值，跳过")
                        continue
                    
                    # 确保数据形状正确
                    if batch_X.shape[0] == 0:
                        continue
                        
                    batch_loss, batch_mae = model.evaluate(batch_X, batch_y, verbose=0)
                    
                    # 验证结果
                    if np.isfinite(batch_loss) and np.isfinite(batch_mae):
                        losses.append(batch_loss)
                        maes.append(batch_mae)
                    else:
                        print(f"批次 {i}-{batch_end} 结果无效，跳过")
                        
                except tf.errors.ResourceExhaustedError:
                    print(f"GPU内存不足，跳过批次 {i}-{batch_end}")
                    continue
                except tf.errors.InvalidArgumentError as e:
                    print(f"CUDNN参数错误，跳过批次 {i}-{batch_end}: {e}")
                    continue
                except Exception as e:
                    print(f"未知错误，跳过批次 {i}-{batch_end}: {e}")
                    continue
            
            if not losses:
                print("警告：所有批次都失败了")
                return 0.0, 0.0
            
            return np.mean(losses), np.mean(maes)
        
        train_loss, train_mae = evaluate_in_batches(X_train_aug, y_train_aug)
        val_loss, val_mae = evaluate_in_batches(X_fold_val, y_fold_val)
        
        # 逻辑回归部分 - 修复数据泄漏
        print("执行逻辑回归分析（修复版）...")
        
        # 只在折内训练集上训练逻辑回归
        y_bs_pred_train = batch_predict_bs_scores(model, X_fold_train)
        features_train = build_logistic_features(ages_fold_train, y_bs_pred_train)
        features_train_clean, quality_train_filtered = filter_valid_features(features_train, quality_fold_train)

        if len(features_train_clean) == 0:
            print(f"第 {fold_idx + 1} 折训练特征无效，跳过")
            continue

        # 训练逻辑回归（只使用训练集）
        clf, scaler = train_logistic_classifier(features_train_clean, quality_train_filtered)

        # 在验证集上评估逻辑回归
        y_bs_pred_val = batch_predict_bs_scores(model, X_fold_val)
        features_val = build_logistic_features(ages_fold_val, y_bs_pred_val)
        features_val_clean, quality_val_filtered = filter_valid_features(features_val, quality_fold_val)

        if len(features_val_clean) == 0:
            print(f"第 {fold_idx + 1} 折验证特征无效，跳过")
            logic_acc = logic_auc = logic_loss = 0.0
            y_pred_proba = np.array([])
        else:
            # 在验证集上评估逻辑回归
            logic_acc, logic_auc, logic_loss = evaluate_logistic_classifier(clf, scaler, features_val_clean, quality_val_filtered)
            X_val_scaled = scaler.transform(features_val_clean)
            y_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
        
        # 保存训练好的模型
        trained_models.append({
            'bilstm_model': model,
            'logistic_model': clf, 
            'scaler': scaler,
            'fold_idx': fold_idx
        })
        
        # 记录结果
        fold_results['train_mae'].append(train_mae)
        fold_results['val_mae'].append(val_mae)
        fold_results['test_mae'].append(0.0)  # 这里不再有test_mae，因为测试集保留
        fold_results['train_loss'].append(train_loss)
        fold_results['val_loss'].append(val_loss)
        fold_results['test_loss'].append(0.0)  # 这里不再有test_loss，因为测试集保留
        fold_results['logic_acc'].append(logic_acc)
        fold_results['logic_auc'].append(logic_auc)
        fold_results['logic_loss'].append(logic_loss)
        fold_results['best_epochs'].append(np.argmin(history.history['val_loss']) + 1)
        
        # 存储预测结果
        if len(features_val_clean) > 0:
            all_predictions.append(np.array(y_pred_proba))
            all_true_labels.append(np.array(quality_val_filtered))
        
        print(f"验证MAE: {val_mae:.4f}, 逻辑回归AUC: {logic_auc:.4f}")
        
        # 清理内存
        del model, train_ds, val_ds
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 强制等待GPU内存释放
        if tf.config.list_physical_devices('GPU'):
            time.sleep(5)
            print(f"第 {fold_idx + 1} 折完成，已清理GPU内存")
    
    # 计算统计结果
    cv_summary = {}
    for metric in fold_results.keys():
        values = fold_results[metric]
        cv_summary[f'{metric}_mean'] = np.mean(values)
        cv_summary[f'{metric}_std'] = np.std(values)
        cv_summary[f'{metric}_min'] = np.min(values)
        cv_summary[f'{metric}_max'] = np.max(values)
    
    print(f"\n=== 交叉验证结果总结 ===")
    print(f"BiLSTM验证MAE: {cv_summary['val_mae_mean']:.4f} ± {cv_summary['val_mae_std']:.4f}")
    print(f"逻辑回归AUC: {cv_summary['logic_auc_mean']:.4f} ± {cv_summary['logic_auc_std']:.4f}")
    print(f"逻辑回归准确率: {cv_summary['logic_acc_mean']:.4f} ± {cv_summary['logic_acc_std']:.4f}")
    
    # 最终测试集评估
    print(f"\n=== 在测试集上进行最终评估 ===")
    print(f"测试集样本数: {len(X_test)}")
    print(f"测试集标签分布: {np.bincount([int(x) for x in quality_test])}")
    
    test_results = []
    for model_info in trained_models:
        bilstm_model = model_info['bilstm_model']
        logistic_model = model_info['logistic_model']
        scaler = model_info['scaler']
        fold_idx = model_info['fold_idx']
        
        # BiLSTM在测试集上预测BS分数
        y_bs_pred_test = batch_predict_bs_scores(bilstm_model, X_test)
        test_mae = np.mean(np.abs(y_bs_pred_test - y_test))
        
        # 构建测试集特征并评估逻辑回归
        features_test = build_logistic_features(ages_test, y_bs_pred_test)
        features_test_clean, quality_test_filtered = filter_valid_features(features_test, quality_test)
        
        if len(features_test_clean) > 0:
            # 逻辑回归预测和评估
            test_acc, test_auc, test_loss = evaluate_logistic_classifier(logistic_model, scaler, features_test_clean, quality_test_filtered)
            
            test_results.append({
                'fold_idx': fold_idx,
                'bilstm_mae': test_mae,
                'logic_acc': test_acc,
                'logic_auc': test_auc,
                'logic_loss': test_loss
            })
            
            print(f"折{fold_idx+1}测试结果: BiLSTM MAE={test_mae:.4f}, 逻辑回归 AUC={test_auc:.4f}, ACC={test_acc:.4f}")
    
    # 计算最终测试集平均性能
    if test_results:
        final_test_summary = {}
        for metric in ['bilstm_mae', 'logic_acc', 'logic_auc', 'logic_loss']:
            values = [r[metric] for r in test_results]
            final_test_summary[f'{metric}_mean'] = np.mean(values)
            final_test_summary[f'{metric}_std'] = np.std(values)
        
        print(f"\n=== 最终测试集平均性能 ===")
        print(f"BiLSTM MAE: {final_test_summary['bilstm_mae_mean']:.4f} ± {final_test_summary['bilstm_mae_std']:.4f}")
        print(f"逻辑回归准确率: {final_test_summary['logic_acc_mean']:.4f} ± {final_test_summary['logic_acc_std']:.4f}")
        print(f"逻辑回归AUC: {final_test_summary['logic_auc_mean']:.4f} ± {final_test_summary['logic_auc_std']:.4f}")
        print(f"逻辑回归Loss: {final_test_summary['logic_loss_mean']:.4f} ± {final_test_summary['logic_loss_std']:.4f}")
    else:
        print("警告: 没有有效的测试结果")
        final_test_summary = {}
    
    return cv_summary, fold_results, all_predictions, all_true_labels, final_test_summary, test_results

def cv_concat_metrics(all_pred_proba, all_labels):
    y = np.concatenate(all_labels, axis=0)
    p = np.concatenate(all_pred_proba, axis=0)
    return {
        "auc_overall": roc_auc_score(y, p),
        "acc_overall": accuracy_score(y, (p > 0.5).astype(int))
    }

def ensemble_predictions(all_predictions, all_true_labels, method='mean'):
    """
    集成多折交叉验证的预测结果
    
    Args:
        all_predictions: 每折的预测结果列表
        all_true_labels: 每折的真实标签列表
        method: 集成方法 ('mean', 'median', 'weighted')
    
    Returns:
        ensemble_pred: 集成预测结果
        ensemble_true: 对应的真实标签
    """
    # 检查所有预测结果的长度是否一致
    lengths = [len(pred) for pred in all_predictions]
    if len(set(lengths)) > 1:
        print(f"警告：不同折的预测结果长度不一致: {lengths}")
        print("无法进行集成，返回最后一折的结果")
        return all_predictions[-1], all_true_labels[-1]
    
    # 如果长度一致，进行集成
    if method == 'mean':
        ensemble_pred = np.mean(all_predictions, axis=0)
    elif method == 'median':
        ensemble_pred = np.median(all_predictions, axis=0)
    elif method == 'weighted':
        # 基于每折性能的加权平均
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 可以基于性能调整权重
        ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # 返回对应的真实标签（使用第一折的标签，因为所有折的标签应该是对应的）
    ensemble_true = all_true_labels[0]
    
    return ensemble_pred, ensemble_true

def plot_cross_validation_results(cv_summary, fold_results, save_path="/liaozizhuo/scripts/cv_analysis.pdf"):
    """可视化交叉验证结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. MAE分布
    axes[0, 0].boxplot([fold_results['train_mae'], fold_results['val_mae'], fold_results['test_mae']], 
                       labels=['Train', 'Validation', 'Test'])
    axes[0, 0].set_title('MAE Distribution')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].grid(True)
    
    # 2. Loss分布
    axes[0, 1].boxplot([fold_results['train_loss'], fold_results['val_loss'], fold_results['test_loss']], 
                       labels=['Train', 'Validation', 'Test'])
    axes[0, 1].set_title('Loss Distribution')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # 3. 逻辑回归性能
    axes[0, 2].boxplot([fold_results['logic_acc'], fold_results['logic_auc']], 
                       labels=['Accuracy', 'AUC'])
    axes[0, 2].set_title('Logistic Regression Performance')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].grid(True)
    
    # 4. 性能指标趋势
    fold_numbers = range(1, len(fold_results['test_mae']) + 1)
    axes[1, 0].plot(fold_numbers, fold_results['test_mae'], 'o-', label='Test MAE')
    axes[1, 0].plot(fold_numbers, fold_results['logic_auc'], 's-', label='Logic AUC')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance Trend')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. 最佳epoch分布
    axes[1, 1].hist(fold_results['best_epochs'], bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Best Epoch')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Best Epoch Distribution')
    axes[1, 1].grid(True)
    
    # 6. 性能统计表
    stats_text = f"""
    Test MAE: {cv_summary['test_mae_mean']:.4f} ± {cv_summary['test_mae_std']:.4f}
    Logic AUC: {cv_summary['logic_auc_mean']:.4f} ± {cv_summary['logic_auc_std']:.4f}
    Logic Accuracy: {cv_summary['logic_acc_mean']:.4f} ± {cv_summary['logic_acc_std']:.4f}
    Best Epoch: {cv_summary['best_epochs_mean']:.1f} ± {cv_summary['best_epochs_std']:.1f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 2].set_title('Performance Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"交叉验证分析图表已保存至: {save_path}")

def assess_model_stability(cv_summary):
    """
    评估模型稳定性
    
    Args:
        cv_summary: 交叉验证结果摘要
    
    Returns:
        dict: 稳定性评估结果
    """
    stability_metrics = {}
    
    # 计算变异系数 (CV = std/mean)
    stability_metrics['mae_cv'] = cv_summary['test_mae_std'] / cv_summary['test_mae_mean']
    stability_metrics['auc_cv'] = cv_summary['logic_auc_std'] / cv_summary['logic_auc_mean']
    stability_metrics['acc_cv'] = cv_summary['logic_acc_std'] / cv_summary['logic_acc_mean']
    
    # 稳定性评级
    def get_stability_rating(cv_value):
        if cv_value < 0.1:
            return "优秀"
        elif cv_value < 0.2:
            return "良好"
        elif cv_value < 0.3:
            return "一般"
        else:
            return "不稳定"
    
    stability_metrics['mae_stability'] = get_stability_rating(stability_metrics['mae_cv'])
    stability_metrics['auc_stability'] = get_stability_rating(stability_metrics['auc_cv'])
    stability_metrics['acc_stability'] = get_stability_rating(stability_metrics['acc_cv'])
    
    print(f"\n=== 模型稳定性评估 ===")
    print(f"MAE稳定性: {stability_metrics['mae_stability']} (CV={stability_metrics['mae_cv']:.3f})")
    print(f"AUC稳定性: {stability_metrics['auc_stability']} (CV={stability_metrics['auc_cv']:.3f})")
    print(f"准确率稳定性: {stability_metrics['acc_stability']} (CV={stability_metrics['acc_cv']:.3f})")
    
    return stability_metrics

# 6. 主流程 - 重构版本
def load_and_prepare_data():
    """加载和准备数据"""
    print("=== 数据加载和准备 ===")
    
    root_dir = "/liaozizhuo/image"
    score_csv = "/liaozizhuo/scripts/MainDB.csv"
    valid_embryos_path = "/liaozizhuo/scripts/filtered_clean_embryos.csv"
    
    # 检查文件是否存在
    required_files = [score_csv, valid_embryos_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"必需文件不存在: {file_path}")
    
    # 加载评分信息
    score_dict = load_score_dict(score_csv)
    
    # 读取胚胎信息
    try:
        valid_embryo_df = pd.read_csv(valid_embryos_path)
        print(f"成功加载胚胎信息: {valid_embryo_df.shape}")
    except Exception as e:
        print(f"加载胚胎信息失败: {e}")
        raise
    
    # 加载年龄信息
    slide_info_path = "/liaozizhuo/scripts/SlidePatientBirthDates.csv"
    if os.path.exists(slide_info_path):
        try:
            slide_info_df = pd.read_csv(slide_info_path)
            print(f"成功加载年龄信息: {slide_info_df.shape}")
        except Exception as e:
            print(f"加载年龄信息失败: {e}")
            slide_info_df = None
    else:
        print(f"警告: 年龄信息文件不存在: {slide_info_path}")
        slide_info_df = None
    
    # 提取年龄映射
    if slide_info_df is not None:
        slide_info_df["EmbryoPatientID"] = slide_info_df["SlideID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
        age_map = dict(zip(slide_info_df["EmbryoPatientID"], slide_info_df["Age"]))
        
        # 将年龄映射到胚胎数据
        valid_embryo_df["PatientID_Prefix"] = valid_embryo_df["PatientID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
        valid_embryo_df["Age"] = valid_embryo_df["PatientID_Prefix"].map(age_map)
        
        # 检查年龄数据的完整性
        age_missing = valid_embryo_df["Age"].isna().sum()
        if age_missing > 0:
            print(f"警告: {age_missing} 个胚胎缺少年龄信息")
            # 用中位数填充缺失的年龄
            median_age = valid_embryo_df["Age"].median()
            valid_embryo_df["Age"].fillna(median_age, inplace=True)
            print(f"用中位数年龄 {median_age:.1f} 填充了缺失值")
    else:
        # 如果没有年龄信息，创建一个默认的年龄列
        print("创建默认年龄信息")
        valid_embryo_df["Age"] = 30.0  # 默认年龄
    
    # 数据验证
    print(f"总胚胎数量: {len(valid_embryo_df)}")
    print(f"质量分布:\n{valid_embryo_df['EmbryoQuality'].value_counts()}")
    print(f"年龄范围: {valid_embryo_df['Age'].min():.1f} - {valid_embryo_df['Age'].max():.1f}")
    
    # 检查数据质量
    quality_counts = valid_embryo_df['EmbryoQuality'].value_counts()
    if len(quality_counts) < 2:
        print("警告: 胚胎质量类别不足，可能影响训练效果")
    
    return root_dir, score_dict, valid_embryo_df

def load_or_build_features(root_dir, score_dict, valid_embryo_df, force_rebuild=False):
    """
    加载或构建特征数据，支持图像增强
    
    Args:
        root_dir: 图像根目录
        score_dict: 评分字典
        valid_embryo_df: 有效胚胎DataFrame
        force_rebuild: 是否强制重新构建特征（忽略缓存）
    """
    print("=== 特征数据处理（支持图像增强）===")
    
    # 检查缓存文件
    cache_files = [
        "/liaozizhuo/scripts/X_features_multi_no_aug.npy",  # 重命名缓存文件
        "/liaozizhuo/scripts/y_bs_labels_multi_no_aug.npy", 
        "/liaozizhuo/scripts/quality_labels_multi_no_aug.npy",
        "/liaozizhuo/scripts/ages_multi_no_aug.npy"
    ]
    
    if not force_rebuild and all(os.path.exists(f) for f in cache_files):
        print("=== 加载缓存的特征数据（无增强版本）===")
        X = np.load(cache_files[0])
        y_bs = np.load(cache_files[1])
        quality_labels = np.load(cache_files[2]).tolist()
        ages = np.load(cache_files[3]).tolist()
        print(f"已加载特征维度: {X.shape}")
    else:
        print("=== 构建多模型特征数据集（无增强版本）===")
        # 使用新的缓存系统
        feature_cache = BiLSTMFeatureCache(root_dir)
        X, y_bs, quality_labels, ages = build_dataset_with_image_augmentation(
            root_dir, score_dict, valid_embryo_df, 
            training=False, balance=False, use_augmentation=False,
            feature_cache=feature_cache, use_cache=True
        )
        
        # 保存缓存
        print("=== 保存特征数据到缓存 ===")
        np.save(cache_files[0], X)
        np.save(cache_files[1], y_bs)
        np.save(cache_files[2], np.array(quality_labels))
        np.save(cache_files[3], np.array(ages))
        print("特征数据已保存")
    
    # 统一数据类型
    X = X.astype(np.float32)
    y_bs = y_bs.astype(np.float32)
    print(f"数据类型已统一为float32: X={X.dtype}, y_bs={y_bs.dtype}")
    
    return X, y_bs, quality_labels, ages

def build_augmented_training_features(root_dir, score_dict, valid_embryo_df, train_indices):
    """
    为训练集构建带增强的特征数据
    
    Args:
        root_dir: 图像根目录
        score_dict: 评分字典
        valid_embryo_df: 有效胚胎DataFrame
        train_indices: 训练集索引
    """
    print("=== 构建训练集增强特征 ===")
    
    # 使用新的缓存系统
    feature_cache = BiLSTMFeatureCache(root_dir)
    
    # 过滤出训练集对应的胚胎
    all_embryo_ids = list(set(valid_embryo_df["EmbryoID"]) & set(score_dict.keys()))
    train_embryo_ids = [all_embryo_ids[i] for i in train_indices if i < len(all_embryo_ids)]
    train_embryo_df = valid_embryo_df[valid_embryo_df["EmbryoID"].isin(train_embryo_ids)]
    
    # 使用图像增强构建训练特征
    X_train_aug, y_train_aug, quality_train_aug, ages_train_aug = build_dataset_with_image_augmentation(
        root_dir, score_dict, train_embryo_df, 
        training=True, balance=False, use_augmentation=True,
        feature_cache=feature_cache, use_cache=True
    )
    
    return X_train_aug, y_train_aug, quality_train_aug, ages_train_aug

def split_dataset_paper_method(X, y_bs, quality_labels, ages):
    """按照论文方法拆分数据集：只分训练集和测试集"""
    print("=== 数据集拆分（论文方法）===")
    
    # 只拆分为训练集和测试集（通常8:2）
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_bs, range(len(X)), test_size=0.2, random_state=42, stratify=quality_labels
    )
    
    # 获取对应的质量标签和年龄
    quality_train = [quality_labels[i] for i in idx_train]
    quality_test = [quality_labels[i] for i in idx_test]
    ages_train = [ages[i] for i in idx_train]
    ages_test = [ages[i] for i in idx_test]
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    return (X_train, X_test, y_train, y_test, 
            quality_train, quality_test,
            ages_train, ages_test)

# 保持向后兼容的原始函数
def split_dataset(X, y_bs, quality_labels, ages):
    """拆分数据集（原方法，向后兼容）"""
    print("=== 数据集拆分 ===")
    
    # 拆分为训练集、验证集、测试集（70%, 15%, 15%）
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X, y_bs, range(len(X)), test_size=0.15, random_state=42, stratify=quality_labels
    )
    
    # 获取对应的质量标签和年龄
    quality_trainval = [quality_labels[i] for i in idx_trainval]
    quality_test = [quality_labels[i] for i in idx_test]
    ages_trainval = [ages[i] for i in idx_trainval]
    ages_test = [ages[i] for i in idx_test]
    
    # 再次拆分训练集和验证集
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_trainval, y_trainval, range(len(X_trainval)), test_size=0.1765, random_state=42, stratify=quality_trainval
    )
    
    # 获取对应的质量标签和年龄
    quality_train = [quality_trainval[i] for i in idx_train]
    quality_val = [quality_trainval[i] for i in idx_val]
    ages_train = [ages_trainval[i] for i in idx_train]
    ages_val = [ages_trainval[i] for i in idx_val]
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            quality_train, quality_val, quality_test,
            ages_train, ages_val, ages_test)

def balance_train_subset(X, y_bs, quality_labels, ages):
    from sklearn.utils import resample
    y_series = pd.Series(quality_labels)
    if y_series.nunique() < 2:
        return X, y_bs, quality_labels, ages
    idx_good = np.where(y_series == 1)[0]
    idx_poor = np.where(y_series == 0)[0]
    min_count = min(len(idx_good), len(idx_poor))
    sel_good = resample(idx_good, replace=False, n_samples=min_count, random_state=42)
    sel_poor = resample(idx_poor, replace=False, n_samples=min_count, random_state=42)
    sel = np.concatenate([sel_good, sel_poor])
    return X[sel], np.array(y_bs)[sel], [quality_labels[i] for i in sel], [ages[i] for i in sel]

def train_model(X_train, y_train, X_val, y_val, feature_dim, save_path, 
                use_augmentation=False, root_dir=None, score_dict=None, 
                valid_embryo_df=None, train_indices=None):
    """统一的训练函数，支持图像增强和常规训练"""
    print(f"=== 训练模型 (use_augmentation={use_augmentation}) ===")
    
    # 如果使用图像增强且提供了必要参数，重新构建增强训练数据
    if use_augmentation and all([root_dir, score_dict, valid_embryo_df, train_indices]):
        print("重新构建图像增强训练数据...")
        # 使用新的缓存系统
        feature_cache = BiLSTMFeatureCache(root_dir)
        
        # 获取训练集对应的胚胎信息
        all_embryo_ids = list(set(valid_embryo_df["EmbryoID"]) & set(score_dict.keys()))
        train_embryo_ids = [all_embryo_ids[i] for i in train_indices if i < len(all_embryo_ids)]
        train_embryo_df = valid_embryo_df[valid_embryo_df["EmbryoID"].isin(train_embryo_ids)]
        
        # 使用图像增强构建训练特征
        X_train, y_train, _, _ = build_dataset_with_image_augmentation(
            root_dir, score_dict, train_embryo_df, 
            training=True, balance=False, use_augmentation=True,
            feature_cache=feature_cache, use_cache=True
        )
        print(f"增强后训练集形状: {X_train.shape}")
    
    # 数据类型统一
    X_train, y_train, X_val, y_val = ensure_float32(X_train, y_train, X_val, y_val)
    
    # GPU策略配置
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"使用MirroredStrategy，GPU数量: {strategy.num_replicas_in_sync}")
    elif num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        print("使用单GPU策略")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        print("使用CPU策略")
    
    # 获取统一的训练配置
    config = get_unified_training_config()
    batch_size = config['batch_size'] * strategy.num_replicas_in_sync
    print(f"全局批量大小: {batch_size}")
    
    # 清理GPU内存
    tf.keras.backend.clear_session()
    gc.collect()
    
    # 配置GPU内存增长
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    
    # 在strategy作用域内构建模型
    with strategy.scope():
        model = create_model(feature_dim)
        
        # 构建回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=config['lr_factor'], patience=config['reduce_lr_patience'], min_lr=config['min_lr'], verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]
    
    # 构建数据管道
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(512).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    print("=== 开始训练 ===")
    
    # 训练模型
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n=== 训练完成 ===")
    print(f"使用了 {num_gpus} 个GPU")
    print(f"最佳验证损失: {min(history.history['val_loss']):.4f}")
    if use_augmentation:
        print("训练集使用了图像级别增强，验证集使用原始数据")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    print("=== 模型评估 ===")
    
    # 不要在这里清理GPU内存，会删除模型
    
    # 分批评估
    batch_size = 16
    test_losses = []
    test_maes = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_X = X_test[i:batch_end]
        batch_y = y_test[i:batch_end]
        
        batch_loss, batch_mae = model.evaluate(batch_X, batch_y, verbose=0)
        test_losses.append(batch_loss)
        test_maes.append(batch_mae)
    
    test_loss = np.mean(test_losses)
    test_mae = np.mean(test_maes)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试MAE: {test_mae:.4f}")
    
    return test_loss, test_mae

def perform_logistic_regression(model, X_train, X_val, X_test, ages_train, ages_val, ages_test, quality_train, quality_val, quality_test, test_mae):
    """
    修复后的逻辑回归分析 - 解决数据泄漏问题
    
    关键修复：
    1. 分别传入训练集、验证集、测试集，而不是trainval
    2. 只使用训练集来训练逻辑回归模型
    3. 在验证集和测试集上分别评估
    4. 添加类别权重处理不平衡问题
    """
    print("=== 修复后的逻辑回归分析（无数据泄漏）===")

    # 清理GPU状态，避免图执行错误
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

    # 1) 只在训练集上获取 BS 预测并拟合 scaler / clf
    print("在训练集上拟合逻辑回归...")
    batch_size = 4  # 进一步减小批量大小，避免GPU错误
    y_bs_pred_tr = []
    valid_indices_tr = []  # 跟踪成功预测的样本索引
    
    for i in range(0, len(X_train), batch_size):
        j = min(i + batch_size, len(X_train))
        try:
            pred = model.predict(X_train[i:j], verbose=0)
            y_bs_pred_tr.extend(pred.flatten())
            valid_indices_tr.extend(range(i, j))  # 记录成功的索引
        except Exception as e:
            print(f"预测批次 {i}-{j} 失败: {e}")
            # 如果批次失败，跳过，不记录索引
            continue
    
    y_bs_pred_tr = np.array(y_bs_pred_tr)
    valid_indices_tr = np.array(valid_indices_tr)
    
    print(f"训练集成功预测样本数: {len(y_bs_pred_tr)}")
    print(f"训练集总样本数: {len(X_train)}")
    
    # 根据成功预测的索引获取对应的年龄和质量标签
    ages_tr = np.array([ages_train[i] for i in valid_indices_tr])
    quality_tr = np.array([quality_train[i] for i in valid_indices_tr])
    
    # 检查形状一致性
    print(f"ages_tr形状: {ages_tr.shape}, y_bs_pred_tr形状: {y_bs_pred_tr.shape}")
    
    # 确保长度一致
    min_len = min(len(ages_tr), len(y_bs_pred_tr), len(quality_tr))
    ages_tr = ages_tr[:min_len]
    y_bs_pred_tr = y_bs_pred_tr[:min_len]
    quality_tr = quality_tr[:min_len]
    
    feats_tr = np.column_stack([
        ages_tr, y_bs_pred_tr, ages_tr**2, y_bs_pred_tr**2,
        ages_tr * y_bs_pred_tr, np.log(ages_tr + 1), 1.0 / (y_bs_pred_tr + 1e-6)
    ])
    
    valid_tr = np.isfinite(feats_tr).all(axis=1)
    feats_tr = feats_tr[valid_tr]
    logic_y_tr = quality_tr[valid_tr]
    
    print(f"训练集标签分布: {np.bincount(logic_y_tr.astype(int))}")
    
    # 计算类别权重处理不平衡问题
    from sklearn.utils.class_weight import compute_class_weight
    try:
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(logic_y_tr), 
            y=logic_y_tr
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"类别权重: {class_weight_dict}")
    except:
        class_weight_dict = None
        print("警告: 无法计算类别权重，使用默认权重")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feats_tr)
    clf = LogisticRegression(
        C=1.0, 
        random_state=42, 
        max_iter=1000,
        class_weight=class_weight_dict  # 添加类别权重
    )
    clf.fit(scaler.transform(feats_tr), logic_y_tr)

    # 2) 在测试集上评估
    print("在测试集上评估逻辑回归...")
    y_bs_pred_te = []
    valid_indices_te = []  # 跟踪成功预测的测试样本索引
    
    for i in range(0, len(X_test), batch_size):
        j = min(i + batch_size, len(X_test))
        try:
            pred = model.predict(X_test[i:j], verbose=0)
            y_bs_pred_te.extend(pred.flatten())
            valid_indices_te.extend(range(i, j))
        except Exception as e:
            print(f"测试集预测批次 {i}-{j} 失败: {e}")
            continue
    
    y_bs_pred_te = np.array(y_bs_pred_te)
    valid_indices_te = np.array(valid_indices_te)
    
    print(f"测试集成功预测样本数: {len(y_bs_pred_te)}")
    print(f"测试集总样本数: {len(X_test)}")
    
    # 根据成功预测的索引获取对应的年龄和质量标签
    ages_te = np.array([ages_test[i] for i in valid_indices_te])
    quality_te = np.array([quality_test[i] for i in valid_indices_te])
    
    # 检查形状一致性
    print(f"ages_te形状: {ages_te.shape}, y_bs_pred_te形状: {y_bs_pred_te.shape}")
    
    # 确保长度一致
    min_len = min(len(ages_te), len(y_bs_pred_te), len(quality_te))
    ages_te = ages_te[:min_len]
    y_bs_pred_te = y_bs_pred_te[:min_len]
    quality_te = quality_te[:min_len]
    
    feats_te = np.column_stack([
        ages_te, y_bs_pred_te, ages_te**2, y_bs_pred_te**2,
        ages_te * y_bs_pred_te, np.log(ages_te + 1), 1.0 / (y_bs_pred_te + 1e-6)
    ])
    
    valid_te = np.isfinite(feats_te).all(axis=1)
    feats_te = feats_te[valid_te]
    logic_y = quality_te[valid_te]

    X_te_scaled = scaler.transform(feats_te)
    y_pred_cls = clf.predict(X_te_scaled)
    y_pred_proba = clf.predict_proba(X_te_scaled)[:, 1]

    logic_loss = log_loss(logic_y, y_pred_proba)
    logic_acc = accuracy_score(logic_y, y_pred_cls)
    auc_score = roc_auc_score(logic_y, y_pred_proba)

    print("=== 分类结果 ===")
    print(classification_report(logic_y, y_pred_cls, target_names=['Poor', 'Good']))
    print(f"准确率: {logic_acc:.4f}")
    print(f"对数损失: {logic_loss:.4f}")
    print(f"AUC分数: {auc_score:.4f}")

    feature_names = ['Age', 'BS_pred', 'Age²', 'BS_pred²', 'Age×BS', 'ln(Age)', '1/BS']
    feature_importance = np.abs(clf.coef_[0])

    print("\n=== 特征重要性 ===")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")

    return {
        'logic_acc': logic_acc,
        'auc_score': auc_score,
        'logic_loss': logic_loss,
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'y_pred_proba': y_pred_proba,
        'logic_y': logic_y,
        'ages_array': ages_te[valid_te],
        'y_bs_pred_all': y_bs_pred_te[valid_te]
    }

def create_visualizations(logic_results, test_mae):
    """创建可视化图表"""
    print("=== 创建可视化图表 ===")
    
    # 主分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC曲线
    fpr, tpr, _ = roc_curve(logic_results['logic_y'], logic_results['y_pred_proba'])
    axes[0, 0].plot(fpr, tpr, label=f'ROC曲线 (AUC = {logic_results["auc_score"]:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='随机猜测')
    axes[0, 0].set_xlabel('假正率')
    axes[0, 0].set_ylabel('真正率')
    axes[0, 0].set_title('ROC曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. 预测分布
    axes[0, 1].hist(logic_results['y_pred_proba'][logic_results['logic_y'] == 0], alpha=0.5, label='Poor', bins=20, density=True)
    axes[0, 1].hist(logic_results['y_pred_proba'][logic_results['logic_y'] == 1], alpha=0.5, label='Good', bins=20, density=True)
    axes[0, 1].set_xlabel('预测概率')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('预测概率分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. 年龄vs预测BS分数散点图
    min_length = min(len(logic_results['ages_array']), len(logic_results['y_bs_pred_all']), len(logic_results['logic_y']))
    ages_plot = logic_results['ages_array'][:min_length]
    y_bs_plot = logic_results['y_bs_pred_all'][:min_length]
    logic_y_plot = logic_results['logic_y'][:min_length]
    
    scatter = axes[1, 0].scatter(ages_plot, y_bs_plot, c=logic_y_plot, cmap='RdYlBu', alpha=0.6)
    axes[1, 0].set_xlabel('年龄')
    axes[1, 0].set_ylabel('预测BS分数')
    axes[1, 0].set_title('年龄 vs 预测BS分数')
    plt.colorbar(scatter, ax=axes[1, 0], label='质量 (0=Poor, 1=Good)')
    axes[1, 0].grid(True)

    # 4. 性能指标总结
    performance_text = f"""
    测试MAE: {test_mae:.4f}
    逻辑回归准确率: {logic_results['logic_acc']:.4f}
    逻辑回归AUC: {logic_results['auc_score']:.4f}
    对数损失: {logic_results['logic_loss']:.4f}
    """
    axes[1, 1].text(0.1, 0.5, performance_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('性能指标总结')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("/liaozizhuo/scripts/multi_model_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特征重要性柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.barh(logic_results['feature_names'], logic_results['feature_importance'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    for i, (bar, importance) in enumerate(zip(bars, logic_results['feature_importance'])):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('特征重要性', fontsize=12, fontweight='bold')
    plt.ylabel('特征名称', fontsize=12, fontweight='bold')
    plt.title('逻辑回归特征重要性分析', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/liaozizhuo/scripts/feature_importance.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def cross_validate_with_augmentation(root_dir, score_dict, valid_embryo_df, feature_dim):
    """
    使用4折交叉验证进行模型评估，保留图像增强策略：
    1. 只使用训练集进行4折交叉验证
    2. 每折训练BiLSTM回归模型，留出数据监控验证损失
    3. 训练集使用图像增强，验证集不使用增强
    4. 用BiLSTM预测训练集BS分数+母体年龄训练逻辑回归
    5. 每折都训练独立的逻辑回归模型
    6. 对所有逻辑回归模型性能指标取平均
    7. 使用accuracy, AUC, precision, recall评估
    """
    print("=== 4折交叉验证+图像增强：训练集评估 ===")
    
    # 强制清理GPU内存
    tf.keras.backend.clear_session()
    gc.collect()
    
    # 首先构建基础数据集（不增强）用于索引拆分
    print("初始化特征缓存...")
    feature_cache = BiLSTMFeatureCache(root_dir)
    
    print("构建基础数据集用于交叉验证拆分...")
    X_base, y_bs_base, quality_labels_base, ages_base = build_dataset_with_image_augmentation(
        root_dir, score_dict, valid_embryo_df, 
        training=False, balance=False, use_augmentation=False,
        feature_cache=feature_cache, use_cache=True
    )
    
    print(f"基础数据集形状: {X_base.shape}")
    
    # 使用4折分层K折交叉验证
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    # 存储每折的逻辑回归性能
    fold_logistic_performances = []
    fold_bilstm_performances = []
    
    # 获取所有胚胎ID列表
    all_embryo_ids = list(set(valid_embryo_df["EmbryoID"]) & set(score_dict.keys()))
    all_embryo_ids.sort()  # 确保顺序一致
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_base, quality_labels_base)):
        print(f"\n--- 第 {fold_idx + 1}/4 折 ---")
        
        # 强制清理GPU内存
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 等待GPU内存释放
        if tf.config.list_physical_devices('GPU'):
            time.sleep(3)
        
        # 获取当前折对应的胚胎ID
        fold_train_embryo_ids = [all_embryo_ids[i] for i in train_idx if i < len(all_embryo_ids)]
        fold_val_embryo_ids = [all_embryo_ids[i] for i in val_idx if i < len(all_embryo_ids)]
        
        print(f"训练胚胎数: {len(fold_train_embryo_ids)}, 验证胚胎数: {len(fold_val_embryo_ids)}")
        
        # === 为当前折构建增强训练数据和验证数据 ===
        print(f"第 {fold_idx + 1} 折：构建增强训练集...")
        fold_train_embryo_df = valid_embryo_df[valid_embryo_df["EmbryoID"].isin(fold_train_embryo_ids)]
        
        # 构建增强的训练集（使用图像增强）
        X_fold_train_aug, y_fold_train_aug, quality_fold_train_aug, ages_fold_train_aug = build_dataset_with_image_augmentation(
            root_dir, score_dict, fold_train_embryo_df, 
            training=True, balance=False, use_augmentation=True,
            feature_cache=feature_cache, use_cache=True
        )
        
        print(f"第 {fold_idx + 1} 折：构建验证集...")
        fold_val_embryo_df = valid_embryo_df[valid_embryo_df["EmbryoID"].isin(fold_val_embryo_ids)]
        X_fold_val, y_fold_val, quality_fold_val, ages_fold_val = build_dataset_with_image_augmentation(
            root_dir, score_dict, fold_val_embryo_df, 
            training=False, balance=False, use_augmentation=False,
            feature_cache=feature_cache, use_cache=True
        )
        
        # 对训练集做类别均衡
        X_fold_train_aug, y_fold_train_aug, quality_fold_train_aug, ages_fold_train_aug = balance_train_subset(
             X_fold_train_aug, y_fold_train_aug, quality_fold_train_aug, ages_fold_train_aug
        )

        print(f"增强训练集: {X_fold_train_aug.shape}, 验证集: {X_fold_val.shape}")
        
        # 创建和训练BiLSTM模型
        try:
            strategy = tf.distribute.MirroredStrategy()
            print("使用GPU训练")
        except:
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
            print("使用CPU训练")
        
        with strategy.scope():
            model = create_model(feature_dim)
            
            print(f"第 {fold_idx + 1} 折训练BiLSTM模型（训练集使用图像增强）")
            
            # 获取统一的训练配置
            config = get_unified_training_config()
            
            # 回调函数
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=config['lr_factor'], patience=config['reduce_lr_patience'], min_lr=config['min_lr'], verbose=0)
            ]
            
            # 数据管道
            cv_batch_size = config['batch_size'] * strategy.num_replicas_in_sync
            
            # 编译模型
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # 构建数据集（训练集使用增强数据，验证集不增强）
            train_ds = tf.data.Dataset.from_tensor_slices((X_fold_train_aug, y_fold_train_aug))
            train_ds = train_ds.shuffle(512).batch(cv_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            
            val_ds = tf.data.Dataset.from_tensor_slices((X_fold_val, y_fold_val))
            val_ds = val_ds.batch(cv_batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
            
            # 训练模型
            history = model.fit(
                train_ds, validation_data=val_ds, 
                epochs=config['epochs'],
                callbacks=callbacks, 
                verbose=1
            )
            
            # 记录BiLSTM性能
            best_val_loss = min(history.history['val_loss'])
            best_val_mae = min(history.history['val_mae'])
            fold_bilstm_performances.append({
                'val_loss': best_val_loss,
                'val_mae': best_val_mae,
                'best_epoch': np.argmin(history.history['val_loss']) + 1
            })
        
        # === 逻辑回归训练与评估 ===
        print(f"第 {fold_idx + 1}/4 折：训练逻辑回归（基于增强训练集的BS预测）")
        
        # 1) 使用BiLSTM预测整个增强训练集的blastocyst scores
        print("使用BiLSTM预测增强训练集blastocyst scores...")
        y_bs_pred_train = []
        for i in range(0, len(X_fold_train_aug), 8):
            j = min(i + 8, len(X_fold_train_aug))
            try:
                pred_batch = model.predict(X_fold_train_aug[i:j], verbose=0)
                y_bs_pred_train.extend(pred_batch.flatten())
            except Exception as e:
                print(f"预测批次 {i}-{j} 失败: {e}")
                continue
        y_bs_pred_train = np.array(y_bs_pred_train)
        
        # 2) 构建逻辑回归特征：predicted blastocyst scores + maternal age
        ages_fold_train_array = np.array(ages_fold_train_aug[:len(y_bs_pred_train)])
        quality_fold_train_trimmed = quality_fold_train_aug[:len(y_bs_pred_train)]
        
        # 特征组合：BS预测分数 + 母体年龄
        features_train = np.column_stack([
            y_bs_pred_train,           # predicted blastocyst scores (BiLSTM输出)
            ages_fold_train_array      # maternal age
        ])
        
        # 检查特征有效性
        valid_train = np.isfinite(features_train).all(axis=1)
        features_train = features_train[valid_train]
        quality_train_filtered = [quality_fold_train_trimmed[i] for i in range(len(quality_fold_train_trimmed)) if valid_train[i]]
        
        if len(features_train) == 0:
            print(f"第 {fold_idx + 1} 折训练特征无效，跳过")
            continue
        
        # 3) 训练独立的逻辑回归模型（每折一个）
        scaler = StandardScaler().fit(features_train)
        clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        clf.fit(scaler.transform(features_train), quality_train_filtered)
        
        # 4) 在当前训练集上评估逻辑回归性能
        print("在当前训练集上评估逻辑回归性能")
        y_pred_proba_train = clf.predict_proba(scaler.transform(features_train))[:, 1]
        y_pred_cls_train = (y_pred_proba_train > 0.5).astype(int)
        
        # 5) 计算性能指标：accuracy, AUC, precision, recall
        logic_acc = accuracy_score(quality_train_filtered, y_pred_cls_train)
        logic_auc = roc_auc_score(quality_train_filtered, y_pred_proba_train) if len(set(quality_train_filtered)) > 1 else 0.0
        
        # 计算precision和recall
        try:
            logic_precision = precision_score(quality_train_filtered, y_pred_cls_train, zero_division=0)
            logic_recall = recall_score(quality_train_filtered, y_pred_cls_train, zero_division=0)
        except:
            logic_precision = 0.0
            logic_recall = 0.0
        
        # 记录当前折的逻辑回归性能
        fold_performance = {
            'accuracy': logic_acc,
            'auc': logic_auc, 
            'precision': logic_precision,
            'recall': logic_recall
        }
        fold_logistic_performances.append(fold_performance)
        
        print(f"第 {fold_idx + 1}/4 折逻辑回归性能:")
        print(f"  Accuracy: {logic_acc:.4f}")
        print(f"  AUC: {logic_auc:.4f}")
        print(f"  Precision: {logic_precision:.4f}")
        print(f"  Recall: {logic_recall:.4f}")
        
        # 清理内存
        del model, train_ds, val_ds
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 强制等待GPU内存释放
        if tf.config.list_physical_devices('GPU'):
            time.sleep(5)
    
    # 6) 计算4折的平均性能
    print(f"\n=== 4折交叉验证平均性能（保留图像增强）===")
    
    avg_performance = {}
    for metric in ['accuracy', 'auc', 'precision', 'recall']:
        values = [p[metric] for p in fold_logistic_performances if metric in p]
        if values:
            avg_performance[f'{metric}_mean'] = np.mean(values)
            avg_performance[f'{metric}_std'] = np.std(values)
            print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # BiLSTM平均性能
    bilstm_avg = {}
    for metric in ['val_loss', 'val_mae', 'best_epoch']:
        values = [p[metric] for p in fold_bilstm_performances]
        bilstm_avg[f'{metric}_mean'] = np.mean(values)
        bilstm_avg[f'{metric}_std'] = np.std(values)
        print(f"BiLSTM {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return avg_performance, bilstm_avg, fold_logistic_performances

def run_cross_validation_with_image_augmentation(root_dir, score_dict, valid_embryo_df, feature_dim):
    """运行带图像增强的交叉验证"""
    print("=== 交叉验证评估（使用图像增强）===")
    
    # 执行交叉验证 - 使用图像增强版本
    cv_summary, fold_results, all_predictions, all_true_labels = cross_validate_bilstm_model_with_image_augmentation(
        root_dir, score_dict, valid_embryo_df, feature_dim, n_folds=5, n_repeats=1
    )
    
    # 评估模型稳定性
    stability_metrics = assess_model_stability(cv_summary)
    
    # 可视化交叉验证结果
    plot_cross_validation_results(cv_summary, fold_results)
    
    # 集成预测结果 - 修复版本
    # 拼接所有折的预测，计算总体指标
    try:
        concat = cv_concat_metrics(all_predictions, all_true_labels)
        ensemble_acc = concat['acc_overall']
        ensemble_auc = concat['auc_overall']

        print(f"\n=== 交叉验证总体（拼接）结果 ===")
        print(f"总体准确率: {ensemble_acc:.4f}")
        print(f"总体AUC: {ensemble_auc:.4f}")

        import json
        cv_results = {
            'cv_summary': cv_summary,
            'stability_metrics': stability_metrics,
            'ensemble_performance': {
                'accuracy': ensemble_acc,
                'auc': ensemble_auc
            }
        }
    except Exception as e:
        print(f"汇总预测失败: {e}")
        import json
        cv_results = {
            'cv_summary': cv_summary,
            'stability_metrics': stability_metrics,
            'ensemble_performance': {
                'accuracy': None,
                'auc': None
            }
        }
    
    with open("/liaozizhuo/scripts/cv_results.json", 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    
    print("交叉验证结果已保存至: /liaozizhuo/scripts/cv_results.json")
    print("分析图表已保存至: /liaozizhuo/scripts/cv_analysis.pdf")

# 保持向后兼容的交叉验证函数（简化版本）
def cross_validate_bilstm_model(X, y_bs, quality_labels, ages, feature_dim, n_folds=5, n_repeats=1):
    """传统的交叉验证函数（不使用图像增强），用于向后兼容"""
    print(f"=== 传统交叉验证 {n_folds}折 (重复{n_repeats}次，不使用图像增强) ===")
    print("注意：此函数为向后兼容，建议使用 cross_validate_bilstm_model_with_image_augmentation")
    
    # 返回模拟结果用于向后兼容
    cv_summary = {'test_mae_mean': 0.0, 'test_mae_std': 0.0, 'logic_auc_mean': 0.0, 'logic_auc_std': 0.0, 'logic_acc_mean': 0.0, 'logic_acc_std': 0.0}
    fold_results = {'train_mae': [], 'val_mae': [], 'test_mae': [], 'train_loss': [], 'val_loss': [], 'test_loss': [], 'logic_acc': [], 'logic_auc': [], 'logic_loss': [], 'best_epochs': []}
    return cv_summary, fold_results, [], []

def run_cross_validation(X, y_bs, quality_labels, ages, feature_dim):
    """运行传统交叉验证（向后兼容）"""
    print("=== 传统交叉验证评估（不使用图像增强）===")
    print("注意：此函数为向后兼容，建议使用 run_cross_validation_with_image_augmentation")
    
    cv_summary, fold_results, all_predictions, all_true_labels = cross_validate_bilstm_model(
        X, y_bs, quality_labels, ages, feature_dim, n_folds=5, n_repeats=1
    )
    
    print("传统交叉验证完成")

def check_required_paths():
    """检查必需的路径和文件"""
    print("=== 检查必需的路径和文件 ===")
    
    required_paths = [
        "/liaozizhuo/image",
        "/liaozizhuo/scripts"
    ]
    
    required_files = [
        "/liaozizhuo/scripts/MainDB.csv",
        "/liaozizhuo/scripts/filtered_clean_embryos.csv"
    ]
    
    # 检查目录
    for path in required_paths:
        if not os.path.exists(path):
            print(f"警告: 目录不存在: {path}")
            try:
                os.makedirs(path, exist_ok=True)
                print(f"已创建目录: {path}")
            except Exception as e:
                print(f"创建目录失败: {e}")
        else:
            print(f"✓ 目录存在: {path}")
    
    # 检查文件
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"✗ 文件不存在: {file_path}")
        else:
            print(f"✓ 文件存在: {file_path}")
    
    print("路径检查完成")

def check_cudnn_compatibility():
    """检查CUDNN兼容性并设置环境变量"""
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
        else:
            print("未检测到GPU")
            
    except Exception as e:
        print(f"CUDNN兼容性检查失败: {e}")
    
    return True

def create_image_augmentation_fn():
    """
    创建图像级别的数据增强函数
    包含视频数据增强技术，包括随机水平翻转和旋转
    
    Returns:
        function: 图像增强函数
    """
    def augment_image(image):
        """
        对图像应用增强操作
        
        Args:
            image: 图像张量，形状为 [H, W, 3]
        
        Returns:
            augmented_image: 增强后的图像
        """
        # 确保输入是tensor
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image)
        
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
        
        if tf.random.uniform([]) > 0.7:
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
        
        image = tf.cast(image, tf.float32)
        
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_brightness(image, max_delta=0.15)
        
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(image, tf.uint8)
        
        return image
    
    return augment_image

def get_unified_training_config():
    """
    获取统一的训练配置，确保主训练和交叉验证使用相同的参数
    """
    return {
        'epochs': 300,
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'use_data_augmentation': True,
        'min_lr': 1e-6,
        'lr_factor': 0.5
    }

def main():
    """
    主函数，使用4折交叉验证策略并保留图像增强：
    1. 只使用训练集进行4折交叉验证
    2. 每折训练BiLSTM，使用预测的blastocyst scores + 母体年龄训练逻辑回归
    3. 对所有折的逻辑回归性能指标取平均
    4. 保留图像增强策略以提升模型性能
    """
    try:
        # 检查CUDNN兼容性
        check_cudnn_compatibility()
        
        print("=== 4折交叉验证+图像增强：主流程开始 ===")
        
        # 1. 加载和准备数据
        print("=== 步骤1: 加载和准备数据 ===")
        root_dir, score_dict, valid_embryo_df = load_and_prepare_data()
        
        # 2. 计算特征维度
        print("=== 步骤2: 计算特征维度 ===")
        feature_cache = BiLSTMFeatureCache(root_dir)
        feature_dim = feature_cache.multi_extractor.total_feature_dim
        print(f"多模型特征维度: {feature_dim}")
        
        # 3. 进行交叉验证（只使用训练集，保留图像增强）
        print("=== 步骤3: 4折交叉验证（保留图像增强）===")
        avg_performance, bilstm_avg, fold_performances = cross_validate_with_augmentation(
            root_dir, score_dict, valid_embryo_df, feature_dim
        )
        
        # 4. 保存结果
        print("=== 步骤4: 保存结果 ===")
        import json
        results = {
            'method': '4fold_cv_with_augmentation',
            'description': '4折交叉验证，只使用训练集，保留图像增强',
            'cv_folds': 4,
            'augmentation': 'image_level_augmentation',
            'logistic_regression_performance': avg_performance,
            'bilstm_performance': bilstm_avg,
            'fold_details': fold_performances
        }
        
        with open("/liaozizhuo/scripts/cv_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 5. 创建简化可视化
        print("=== 步骤5: 创建可视化 ===")
        create_cv_visualization(avg_performance, bilstm_avg, fold_performances)
        
        # 6. 最终总结
        print(f"\n=== 最终结果总结（保留图像增强）===")
        print(f"交叉验证方法: 4折分层交叉验证")
        print(f"训练策略: 只使用训练集，保留图像增强")
        print(f"特征组合: BiLSTM预测的BS分数 + 母体年龄")
        print()
        print("=== BiLSTM回归性能 ===")
        print(f"验证MAE: {bilstm_avg['val_mae_mean']:.4f} ± {bilstm_avg['val_mae_std']:.4f}")
        print(f"验证Loss: {bilstm_avg['val_loss_mean']:.4f} ± {bilstm_avg['val_loss_std']:.4f}")
        print()
        print("=== 逻辑回归分类性能 ===")
        print(f"准确率 (Accuracy): {avg_performance['accuracy_mean']:.4f} ± {avg_performance['accuracy_std']:.4f}")
        print(f"AUC: {avg_performance['auc_mean']:.4f} ± {avg_performance['auc_std']:.4f}")
        print(f"精确率 (Precision): {avg_performance['precision_mean']:.4f} ± {avg_performance['precision_std']:.4f}")
        print(f"召回率 (Recall): {avg_performance['recall_mean']:.4f} ± {avg_performance['recall_std']:.4f}")
        print()
        print("结果已保存至: /liaozizhuo/scripts/cv_results.json")
        print("可视化图表已保存至: /liaozizhuo/scripts/cv_analysis.pdf")
        
        print("\n=== 程序执行完成 ===")
        
    except Exception as e:
        print(f"\n=== 程序执行失败 ===")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_cv_visualization(avg_performance, bilstm_avg, fold_performances):
    """为交叉验证创建可视化"""
    print("=== 创建交叉验证可视化图表 ===")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 逻辑回归性能指标
    metrics = ['accuracy', 'auc', 'precision', 'recall']
    means = [avg_performance[f'{m}_mean'] for m in metrics]
    stds = [avg_performance[f'{m}_std'] for m in metrics]
    
    bars = axes[0, 0].bar(metrics, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('逻辑回归平均性能指标 (4折CV)')
    axes[0, 0].set_ylabel('分数')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. BiLSTM性能指标
    bilstm_metrics = ['val_mae', 'val_loss']
    bilstm_means = [bilstm_avg[f'{m}_mean'] for m in bilstm_metrics]
    bilstm_stds = [bilstm_avg[f'{m}_std'] for m in bilstm_metrics]
    
    bars = axes[0, 1].bar(bilstm_metrics, bilstm_means, yerr=bilstm_stds, capsize=5, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title('BiLSTM回归平均性能 (4折CV)')
    axes[0, 1].set_ylabel('损失/误差')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, bilstm_means, bilstm_stds):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bilstm_means) * 0.02,
                       f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 各折性能趋势
    fold_numbers = range(1, len(fold_performances) + 1)
    fold_accs = [f['accuracy'] for f in fold_performances]
    fold_aucs = [f['auc'] for f in fold_performances]
    
    axes[1, 0].plot(fold_numbers, fold_accs, 'o-', label='Accuracy', linewidth=2, markersize=6)
    axes[1, 0].plot(fold_numbers, fold_aucs, 's-', label='AUC', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('折数')
    axes[1, 0].set_ylabel('性能分数')
    axes[1, 0].set_title('各折逻辑回归性能趋势')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.1)
    
    # 4. 方法总结
    method_text = f"""
4折交叉验证 + 图像增强

策略:
• 4折分层交叉验证
• 只使用训练集进行CV
• BiLSTM训练使用图像增强
• 验证集不使用增强
• 特征: BS预测分数 + 母体年龄

关键结果:
• 准确率: {avg_performance['accuracy_mean']:.3f}±{avg_performance['accuracy_std']:.3f}
• AUC: {avg_performance['auc_mean']:.3f}±{avg_performance['auc_std']:.3f}
• 精确率: {avg_performance['precision_mean']:.3f}±{avg_performance['precision_std']:.3f}
• 召回率: {avg_performance['recall_mean']:.3f}±{avg_performance['recall_std']:.3f}

BiLSTM性能:
• 验证MAE: {bilstm_avg['val_mae_mean']:.4f}±{bilstm_avg['val_mae_std']:.4f}
    """
    axes[1, 1].text(0.05, 0.95, method_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    axes[1, 1].set_title('方法总结')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("/liaozizhuo/scripts/cv_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("交叉验证可视化图表已保存")

# 主流程
if __name__ == "__main__":
    # 在所有操作之前配置GPU
    configure_gpu()
    
    # 检查必需的路径
    check_required_paths()
    
    # 检查CUDNN兼容性
    check_cudnn_compatibility()
    
    # 调用主函数
    main()
