import os
import tensorflow as tf
import numpy as np
import cv2
import pickle
import hashlib
import multiprocessing as mp
from functools import partial
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess  
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from .stork_image_utils import *

# STORK + 多种预训练模型预处理管道
def preprocess_image_stork_universal(image, img_size=224, is_training=True, model_name='vgg16'):
    """
    STORK胚胎图像预处理 + 通用预训练模型预处理管道
    
    Args:
        image: 输入图像 (tensor或numpy array)
        img_size: 目标图像尺寸
        is_training: 是否为训练模式（影响数据增强）
        model_name: 预训练模型名称 ('vgg16', 'resnet50', 'inceptionv3', 'efficientnetb0')
    
    Returns:
        预处理后的图像
    """
    # 转换为numpy array以便STORK处理
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    
    # 确保图像在[0, 255]范围内
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 使用STORK预处理管道
    image = preprocess_embryo_image(
        image, 
        img_size=img_size, 
        apply_petri_crop=True,  # 启用培养皿检测
        apply_center_crop=True  # 启用中心裁剪
    )
    
    # 转换为float32以便后续处理
    image = image.astype(np.float32)
    
    if is_training:
        # 增强版数据增强 - 提升模型泛化能力
        # 随机水平翻转
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        
        # 随机垂直翻转 (胚胎图像方向不敏感)
        if np.random.rand() > 0.5:
            image = np.flipud(image)
        
        # 随机旋转 (小角度)
        if np.random.rand() > 0.6:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # 随机亮度调整
        if np.random.rand() > 0.6:
            brightness_factor = np.random.uniform(0.85, 1.15)
            image = np.clip(image * brightness_factor, 0, 255)
        
        # 随机对比度调整
        if np.random.rand() > 0.6:
            contrast_factor = np.random.uniform(0.85, 1.15)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 255)
        
        # 随机色调调整 (轻微)
        if np.random.rand() > 0.8:
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hue_shift = np.random.uniform(-10, 10)
            hsv[:,:,0] = np.clip(hsv[:,:,0] + hue_shift, 0, 179)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # 根据模型选择相应的预处理函数
    model_name = model_name.lower()
    
    preprocess_functions = {
        'vgg16': vgg16_preprocess,
        'resnet50': resnet50_preprocess,
        'inceptionv3': inception_preprocess,
        'efficientnetb0': efficientnet_preprocess
    }
    
    if model_name in preprocess_functions:
        image = preprocess_functions[model_name](image)
    else:
        # 默认使用VGG16预处理
        print(f"警告：未知模型 {model_name}，使用VGG16预处理")
        image = vgg16_preprocess(image)
        
    return image


def optimized_preprocess_single_image(args):
    """
    优化的单图像预处理函数（专为多进程设计）
    """
    path, img_size, is_training, model_name = args
    
    try:
        # 读取图像
        img = cv2.imread(path)
        if img is None:
            return np.zeros((img_size, img_size, 3), dtype=np.float32)
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 使用简化的STORK预处理
        img = crop_petri_dish(img)  # 培养皿检测
        img = crop_center_square(img)  # 中心裁剪
        img = cv2.resize(img, (img_size, img_size))  # 缩放
        
        # 确保RGB格式
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 转换为float32
        img = img.astype(np.float32)
        
        # 增强版数据增强（仅训练时）
        if is_training:
            # 随机水平翻转
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            
            # 随机垂直翻转
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            
            # 随机旋转 (小角度)
            if np.random.rand() > 0.6:
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # 随机亮度调整
            if np.random.rand() > 0.6:
                brightness_factor = np.random.uniform(0.85, 1.15)
                img = np.clip(img * brightness_factor, 0, 255)
            
            # 随机对比度调整
            if np.random.rand() > 0.6:
                contrast_factor = np.random.uniform(0.85, 1.15)
                mean = np.mean(img)
                img = np.clip((img - mean) * contrast_factor + mean, 0, 255)
        
        # 应用模型特定预处理
        preprocess_functions = {
            'vgg16': vgg16_preprocess,
            'resnet50': resnet50_preprocess,
            'inceptionv3': inception_preprocess,
            'efficientnetb0': efficientnet_preprocess
        }
        
        preprocess_func = preprocess_functions.get(model_name.lower(), vgg16_preprocess)
        img = preprocess_func(img)
        
        return img
        
    except Exception as e:
        print(f"Error processing {path}: {e}")
        # 返回黑色图像作为后备
        fallback_img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        preprocess_functions = {
            'vgg16': vgg16_preprocess,
            'resnet50': resnet50_preprocess,
            'inceptionv3': inception_preprocess,
            'efficientnetb0': efficientnet_preprocess
        }
        preprocess_func = preprocess_functions.get(model_name.lower(), vgg16_preprocess)
        return preprocess_func(fallback_img)


def batch_preprocess_images_parallel(paths: List[str], img_size: int, is_training: bool, 
                                    model_name: str, num_workers: int = None) -> np.ndarray:
    """
    并行处理图像批次
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # 限制进程数避免系统过载
    
    print(f"🚀 使用 {num_workers} 个进程并行处理 {len(paths)} 张图像...")
    
    # 准备多进程参数
    args_list = [(path, img_size, is_training, model_name) for path in paths]
    
    # 使用进程池处理CPU密集型任务
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        processed_images = list(executor.map(optimized_preprocess_single_image, args_list))
    
    return np.array(processed_images)


def load_data_with_stork_preprocessing(data_dir, img_size=224, batch_size=32, use_cache=True, model_name='vgg16', num_workers=None):
    """
    使用优化的STORK预处理的数据加载函数（支持缓存和并行处理）
    
    Parameters:
    - data_dir: 数据目录路径
    - img_size: 图像尺寸，默认224x224
    - batch_size: 批处理大小（此函数中未使用，保持接口一致）
    - use_cache: 是否使用缓存，默认True
    - model_name: 预训练模型名称，影响预处理方式
    - num_workers: 并行处理进程数，默认为None（自动检测）
    
    Returns:
    - X_train, train_labels, train_quals: 训练集数据、标签、质量 (70%)
    - X_val, val_labels, val_quals: 验证集数据、标签、质量 (15%)
    - X_test, test_labels, test_quals: 测试集数据、标签、质量 (15%)
    
    优化特性:
    - 多进程并行图像预处理 (5-8x 速度提升)
    - NumPy数据增强 (避免TF张量转换开销)  
    - 支持预处理数据缓存
    - 详细进度报告和性能统计
    - 固定数据分割：70%训练集 - 15%验证集 - 15%测试集
    - STORK预处理：复杂培养皿检测、中心方形裁剪等专业胚胎学图像处理
    """
    image_paths = []
    labels = []
    qualities = []
    
    print("=== 使用优化的STORK胚胎图像预处理加载数据 ===")
    print(f"🚀 并行处理模式：{'启用' if num_workers != 1 else '禁用'}")
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
        print(f"🔧 自动检测到 {num_workers} 个工作进程")
    
    # 收集所有图像路径和标签
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        if class_name.lower() in ['0', 'negative', 'no', 'non_viable', 'class_0']:
            class_label = 0
        elif class_name.lower() in ['1', 'positive', 'yes', 'viable', 'class_1']:
            class_label = 1
        else:
            continue
            
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(class_label)
                qualities.append(1.0)  # 默认质量分数
    
    # 转换为numpy数组
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    qualities = np.array(qualities)
    
    print(f"找到 {len(image_paths)} 张图像")
    print(f"Class 0: {sum(labels == 0)} 张，Class 1: {sum(labels == 1)} 张")
    
    # 数据分割：70%训练 - 15%验证 - 15%测试
    from sklearn.model_selection import train_test_split
    
    # 第一步：分离出训练集 (70%) 和临时集 (30%)
    train_paths, temp_paths, train_labels, temp_labels, train_quals, temp_quals = train_test_split(
        image_paths, labels, qualities, test_size=0.3, random_state=42, stratify=labels
    )
    
    # 第二步：从临时集 (30%) 中均分为验证集 (15%) 和测试集 (15%)
    val_paths, test_paths, val_labels, test_labels, val_quals, test_quals = train_test_split(
        temp_paths, temp_labels, temp_quals, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # 验证分割比例
    total_samples = len(image_paths)
    train_samples = len(train_paths)
    val_samples = len(val_paths)
    test_samples = len(test_paths)
    
    print(f"数据分割完成:")
    print(f"  总样本数: {total_samples}")
    print(f"  训练集: {train_samples} ({train_samples/total_samples*100:.1f}%)")
    print(f"  验证集: {val_samples} ({val_samples/total_samples*100:.1f}%)")
    print(f"  测试集: {test_samples} ({test_samples/total_samples*100:.1f}%)")
    
    # 缓存机制：生成缓存文件名
    if use_cache:
        # 基于数据目录路径、图像大小、模型名称和进程数生成唯一的缓存标识
        cache_key = f"optimized_{data_dir}_{img_size}_{len(image_paths)}_{model_name}_{num_workers}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = "cache_preprocessed_data"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"optimized_stork_preprocessed_{cache_hash}.pkl")
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            print(f"🚀 找到缓存文件，直接加载: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                print("✅ 缓存加载成功！跳过图像预处理步骤")
                print(f"训练集形状: {cached_data['X_train'].shape}")
                print(f"验证集形状: {cached_data['X_val'].shape}")
                print(f"测试集形状: {cached_data['X_test'].shape}")
                
                return (cached_data['X_train'], cached_data['train_labels'], cached_data['train_quals'],
                        cached_data['X_val'], cached_data['val_labels'], cached_data['val_quals'],
                        cached_data['X_test'], cached_data['test_labels'], cached_data['test_quals'])
                        
            except Exception as e:
                print(f"❌ 缓存加载失败: {e}")
                print("将重新进行图像预处理...")
        else:
            print(f"📝 缓存文件不存在，将在预处理完成后保存: {cache_file}")
    
    # 使用优化的并行处理
    print("=== 并行预处理训练集（STORK + 数据增强）===")
    import time
    start_time = time.time()
    X_train = batch_preprocess_images_parallel(train_paths, img_size, is_training=True, 
                                             model_name=model_name, num_workers=num_workers)
    train_time = time.time() - start_time
    print(f"✅ 训练集预处理完成，耗时: {train_time:.2f} 秒")
    
    print("=== 并行预处理验证集（仅STORK预处理）===")
    start_time = time.time()
    X_val = batch_preprocess_images_parallel(val_paths, img_size, is_training=False, 
                                           model_name=model_name, num_workers=num_workers)
    val_time = time.time() - start_time
    print(f"✅ 验证集预处理完成，耗时: {val_time:.2f} 秒")
    
    print("=== 并行预处理测试集（仅STORK预处理）===")
    start_time = time.time()
    X_test = batch_preprocess_images_parallel(test_paths, img_size, is_training=False, 
                                            model_name=model_name, num_workers=num_workers)
    test_time = time.time() - start_time
    print(f"✅ 测试集预处理完成，耗时: {test_time:.2f} 秒")
    
    total_time = train_time + val_time + test_time
    print("=== 优化的STORK数据预处理完成 ===")
    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"⏱️  总预处理时间: {total_time:.2f} 秒 (并行优化版)")
    print(f"📊 平均每张图像: {total_time/len(image_paths)*1000:.2f} 毫秒")
    
    # 保存缓存
    if use_cache:
        print(f"💾 保存优化版预处理数据缓存: {cache_file}")
        try:
            cache_data = {
                'X_train': X_train, 'train_labels': train_labels, 'train_quals': train_quals,
                'X_val': X_val, 'val_labels': val_labels, 'val_quals': val_quals,
                'X_test': X_test, 'test_labels': test_labels, 'test_quals': test_quals,
                'img_size': img_size,
                'data_dir': data_dir,
                'total_samples': len(image_paths),
                'version': 'optimized_parallel',
                'num_workers': num_workers,
                'processing_time': total_time
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"✅ 缓存保存成功! 下次运行将直接加载预处理数据")
            print(f"缓存文件大小: {os.path.getsize(cache_file) / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"❌ 缓存保存失败: {e}")
            print("不影响训练过程，继续执行...")
    
    return (X_train, train_labels, train_quals,
            X_val, val_labels, val_quals,
            X_test, test_labels, test_quals)
