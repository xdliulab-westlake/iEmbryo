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
import pandas as pd

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
        
        # 重新提取特征 - 智能选择方法
        print("🔄 开始特征提取...")

        # 检查GPU数量和图像数量，选择最优策略
        gpu_count = len(self.multi_extractor.gpu_devices) if hasattr(self.multi_extractor, 'gpu_devices') else 1
        use_multi_gpu = getattr(self.multi_extractor, 'use_multi_gpu', False)

        print(f"📊 GPU状态: {gpu_count} 个GPU, 多GPU模式: {use_multi_gpu}")
        print(f"📊 图像数量: {len(image_paths)}")

        # 根据情况选择最优方法
        if len(image_paths) > 500 or (use_multi_gpu and gpu_count > 1):
            print("🚀 使用超级优化方法")
            features_dict = self._extract_features_optimized(image_paths, training, use_augmentation)
        else:
            print("🔧 使用标准方法")
            features_dict = self._extract_features_from_scratch(image_paths, training, use_augmentation)
        
        # 保存缓存
        if use_cache:
            self._save_features_cache(features_dict, cache_file, image_paths)
            
        return features_dict
    
    def _extract_features_from_scratch(self, image_paths, training, use_augmentation):
        """从头提取特征（使用STORK预处理） - 优化版本"""
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
        
        # 构建数据管道 - 超保守的批次大小避免OOM
        BATCH_SIZE = 32  # 减小批次大小避免显存不足
        path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        image_dataset = path_dataset.map(
            load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
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

    def _extract_features_optimized(self, image_paths, training, use_augmentation):
        """优化的特征提取方法 - 使用更大批次和内存管理"""
        import cv2
        from concurrent.futures import ThreadPoolExecutor

        def load_image_optimized(path):
            """优化的图像加载函数"""
            try:
                if not os.path.exists(path):
                    print(f"❌ 图像文件不存在: {path}")
                    return np.zeros((224, 224, 3), dtype=np.uint8)

                # 使用STORK预处理管道
                processed_img = preprocess_embryo_image(
                    path,
                    img_size=224,
                    apply_petri_crop=True,
                    apply_center_crop=True
                )
                return processed_img.astype(np.uint8)

            except Exception as e:
                print(f"⚠️ 图像加载失败 {path}: {e}")
                # 回退到OpenCV加载
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        return img.astype(np.uint8)
                    else:
                        return np.zeros((224, 224, 3), dtype=np.uint8)
                except:
                    return np.zeros((224, 224, 3), dtype=np.uint8)

        print("🚀 使用原版高效的批次处理...")

        # 使用保守的批次大小
        features_dict = {}
        BATCH_SIZE = 32  # 减小批次大小避免显存不足

        print(f"🎯 批次大小: {BATCH_SIZE}")

        # 分批处理
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="批量处理"):
            batch_paths = image_paths[i:i+BATCH_SIZE]

            # 并行加载当前批次的图像 - 增加线程数
            with ThreadPoolExecutor(max_workers=16) as executor:
                batch_images = list(executor.map(load_image_optimized, batch_paths))

            # 转换为numpy数组
            batch_images = np.array(batch_images)

            # 应用数据增强（仅训练时）- 向量化操作
            if training and use_augmentation:
                # 向量化数据增强，更高效
                flip_mask = np.random.random(len(batch_images)) > 0.5
                for j in range(len(batch_images)):
                    if flip_mask[j]:
                        batch_images[j] = np.fliplr(batch_images[j])

                # 批量亮度调整
                brightness_mask = np.random.random(len(batch_images)) > 0.7
                brightness_factors = np.random.uniform(0.9, 1.1, len(batch_images))
                for j in range(len(batch_images)):
                    if brightness_mask[j]:
                        batch_images[j] = np.clip(batch_images[j] * brightness_factors[j], 0, 255).astype(np.uint8)

            # 提取特征 - 使用优化的特征提取器
            try:
                batch_features = self.multi_extractor.extract_features(batch_images)

                # 存储特征
                for j, path in enumerate(batch_paths):
                    if j < len(batch_features):  # 防止索引越界
                        features_dict[path] = batch_features[j]

            except Exception as e:
                print(f"⚠️ 批次特征提取失败: {e}")
                # 如果大批次失败，尝试小批次
                print("   尝试拆分为小批次...")
                SMALL_BATCH = 64
                for k in range(0, len(batch_images), SMALL_BATCH):
                    small_batch = batch_images[k:k+SMALL_BATCH]
                    small_paths = batch_paths[k:k+SMALL_BATCH]
                    try:
                        small_features = self.multi_extractor.extract_features(small_batch)
                        for j, path in enumerate(small_paths):
                            if j < len(small_features):
                                features_dict[path] = small_features[j]
                    except Exception as e2:
                        print(f"   小批次也失败: {e2}")

            # 清理内存并强制垃圾回收
            del batch_images
            if 'batch_features' in locals():
                del batch_features

            # 每处理一定数量后清理GPU内存
            if i % (BATCH_SIZE * 4) == 0:
                import gc
                gc.collect()
                # 不清理整个session，只是建议垃圾回收

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

    def process_multi_focal_data(self, embryo_df, use_cache=True, training=False, use_augmentation=False, require_complete=True):
        """
        处理多焦距深度数据 - 简单平均融合

        Args:
            embryo_df: 胚胎图像数据框
            use_cache: 是否使用缓存
            training: 是否为训练模式
            use_augmentation: 是否使用数据增强
            require_complete: 是否要求完整的33个时间点
        """
        if self.multi_extractor is None:
            raise ValueError("请先调用 set_extractor() 设置特征提取器")

        print(f"🔄 处理多焦距深度数据: {len(embryo_df)} 张图像")

        # 获取所有图像的特征
        all_image_paths = embryo_df['FullPath'].tolist()
        cached_features = self.get_or_compute_features(
            all_image_paths,
            use_cache=use_cache,
            training=training,
            use_augmentation=use_augmentation
        )

        # 按时间点分组并融合
        time_groups = embryo_df.groupby('TargetHour')
        sequence_features = []
        expected_timepoints = 33  # 预期的时间点数量

        # 检查时间点完整性
        if require_complete:
            unique_timepoints = len(time_groups)
            if unique_timepoints != expected_timepoints:
                embryo_id = embryo_df['EmbryoID'].iloc[0] if 'EmbryoID' in embryo_df.columns else 'Unknown'
                print(f"❌ 胚胎 {embryo_id} 时间点不完整: {unique_timepoints}/{expected_timepoints}")
                return None  # 返回None表示该胚胎被拒绝

        failed_timepoints = []  # 记录失败的时间点

        for target_hour, time_df in sorted(time_groups, key=lambda x: x[0]):
            focal_features = []

            for _, row in time_df.iterrows():
                path = row['FullPath']
                if path in cached_features:
                    focal_features.append(cached_features[path])

            if focal_features:
                # 简单平均融合
                if len(focal_features) > 1:
                    fused_feature = np.mean(focal_features, axis=0)
                else:
                    fused_feature = focal_features[0]
                sequence_features.append(fused_feature)
            else:
                # 记录失败的时间点
                failed_timepoints.append(target_hour)
                if require_complete:
                    embryo_id = embryo_df['EmbryoID'].iloc[0] if 'EmbryoID' in embryo_df.columns else 'Unknown'
                    print(f"❌ 胚胎 {embryo_id} 时间点 {target_hour} 特征提取失败")
                    return None  # 特征提取失败，拒绝该胚胎

        # 最终完整性检查
        if require_complete and len(sequence_features) != expected_timepoints:
            embryo_id = embryo_df['EmbryoID'].iloc[0] if 'EmbryoID' in embryo_df.columns else 'Unknown'
            print(f"❌ 胚胎 {embryo_id} 最终时间点不足: {len(sequence_features)}/{expected_timepoints}")
            if failed_timepoints:
                print(f"   失败的时间点: {failed_timepoints}")
            return None

        return np.array(sequence_features)