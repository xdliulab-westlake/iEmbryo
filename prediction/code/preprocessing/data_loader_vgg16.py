import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from .stork_image_utils import *

# STORK + VGG16 预处理管道
def preprocess_image_stork_vgg16(image, img_size=224, is_training=True):
    """
    STORK胚胎图像预处理 + VGG16预处理管道
    
    Args:
        image: 输入图像 (tensor或numpy array)
        img_size: 目标图像尺寸
        is_training: 是否为训练模式（影响数据增强）
    
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
        # 训练时的轻量级数据增强 (在STORK预处理基础上)
        image = tf.convert_to_tensor(image)
        
        # 轻量级增强，保护胚胎诊断特征
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([]) > 0.7:
            image = tf.image.random_brightness(image, max_delta=0.1)
        if tf.random.uniform([]) > 0.7:
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            
        image = image.numpy()
    
    # 使用官方VGG16预处理
    image = preprocess_input(image)
    return image


def load_data_with_stork_preprocessing(data_dir, img_size=224, batch_size=32):
    """
    使用STORK预处理的数据加载函数
    
    Parameters:
    - data_dir: 数据目录路径
    - img_size: 图像尺寸，默认224x224
    - batch_size: 批处理大小（此函数中未使用，保持接口一致）
    
    Returns:
    - X_train, train_labels, train_quals: 训练集数据、标签、质量 (70%)
    - X_val, val_labels, val_quals: 验证集数据、标签、质量 (15%)
    - X_test, test_labels, test_quals: 测试集数据、标签、质量 (15%)
    
    Note:
    - 固定数据分割：70%训练集 - 15%验证集 - 15%测试集
    - 使用STORK预处理流程进行胚胎图像处理
    - 包括培养皿检测、中心方形裁剪等专业胚胎学图像处理
    - 训练集：应用数据增强 + STORK预处理
    - 验证/测试集：仅应用STORK预处理，不做数据增强
    """
    image_paths = []
    labels = []
    qualities = []
    
    print("=== 使用STORK胚胎图像预处理加载数据 ===")
    
    # 收集所有图像路径和标签
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        if class_name.lower() in ['0', 'negative', 'no', 'non_viable']:
            class_label = 0
        elif class_name.lower() in ['1', 'positive', 'yes', 'viable']:
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
    
    # 分别处理三个数据集
    def preprocess_images_stork(paths, is_training):
        """使用STORK预处理图像列表"""
        processed_images = []
        for i, path in enumerate(paths):
            if i % 1000 == 0:
                print(f"处理进度: {i+1}/{len(paths)}")
            
            try:
                # 使用OpenCV读取图像（更稳定）
                img = cv2.imread(path)
                if img is None:
                    print(f"无法读取图像: {path}")
                    # 创建黑色图像作为后备
                    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                else:
                    # OpenCV默认BGR，转换为RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 使用STORK整合的预处理
                img_processed = preprocess_image_stork_vgg16(
                    img, img_size=img_size, is_training=is_training
                )
                processed_images.append(img_processed)
                
            except Exception as e:
                print(f"处理图像 {path} 时出错: {e}")
                # 创建黑色图像作为后备
                fallback_img = np.zeros((img_size, img_size, 3), dtype=np.float32)
                fallback_img = preprocess_input(fallback_img)
                processed_images.append(fallback_img)
        
        return np.array(processed_images)
    
    print("=== 预处理训练集（STORK + 数据增强）===")
    X_train = preprocess_images_stork(train_paths, is_training=True)
    
    print("=== 预处理验证集（仅STORK预处理）===")
    X_val = preprocess_images_stork(val_paths, is_training=False)
    
    print("=== 预处理测试集（仅STORK预处理）===")
    X_test = preprocess_images_stork(test_paths, is_training=False)
    
    print("=== STORK数据预处理完成 ===")
    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    return (X_train, train_labels, train_quals,
            X_val, val_labels, val_quals,
            X_test, test_labels, test_quals)
