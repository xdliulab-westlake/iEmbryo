"""
BiLSTM特征缓存模块
包含智能特征缓存系统，支持MD5哈希缓存键和原子文件操作
"""

import os
import time
import hashlib
import pickle
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

from .preprocessing.bilstm_image_utils import preprocess_embryo_image


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
        
        # 获取当前使用的模型信息
        model_info = self.multi_extractor.get_model_info() if self.multi_extractor else {'selected_models': ['unknown']}
        
        config_info = {
            'data_root': self.data_root,
            'selected_models': model_info.get('selected_models', ['unknown']),
            'model_name': model_info.get('model_name', 'unknown'),
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
                
                # 检查文件是否存在
                if not os.path.exists(path_str):
                    print(f"❌ 图像文件不存在: {path_str}")
                    print(f"   跳过此文件，使用默认黑色图像")
                    return np.zeros((224, 224, 3), dtype=np.uint8)
                
                # 使用STORK预处理管道：Hough圆检测 + 中心裁剪
                processed_img = preprocess_embryo_image(
                    path_str, 
                    img_size=224, 
                    apply_petri_crop=True,  # 启用Petri dish检测
                    apply_center_crop=True  # 启用中心方形裁剪
                )
                return processed_img.astype(np.uint8)
            except FileNotFoundError as e:
                print(f"❌ 文件不存在错误 {path_str}: {e}")
                print(f"   跳过此文件，使用默认黑色图像")
                return np.zeros((224, 224, 3), dtype=np.uint8)
            except Exception as e:
                print(f"⚠️ STORK预处理失败 {path_str}: {e}")
                print(f"   尝试简单预处理...")
                # 回退到简单预处理
                try:
                    img = cv2.imread(path_str)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        return img.astype(np.uint8)
                    else:
                        print(f"❌ 无法读取图像文件: {path_str}")
                        return np.zeros((224, 224, 3), dtype=np.uint8)
                except Exception as e2:
                    print(f"❌ 简单预处理也失败 {path_str}: {e2}")
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