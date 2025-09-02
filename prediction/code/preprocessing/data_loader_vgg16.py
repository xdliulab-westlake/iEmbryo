import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# VGG16兼容的预处理管道
def preprocess_image_vgg16(image, img_size=224, is_training=True):
    """
    VGG16兼容的预处理函数
    """
    # 确保图像在[0, 255]范围内
    if image.dtype != tf.uint8:
        # 如果是[0,1]范围的float，转换为[0,255]的uint8
        if tf.reduce_max(image) <= 1.0:
            image = tf.cast(image * 255.0, tf.uint8)
        else:
            image = tf.cast(image, tf.uint8)
    
    # 转换为float32以便后续处理
    image = tf.cast(image, tf.float32)
    
    if is_training:
        # 训练时的数据增强
        scale_size = tf.random.uniform([], minval=256, maxval=480, dtype=tf.int32)
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        ratio = tf.cast(scale_size, tf.float32) / tf.cast(tf.minimum(height, width), tf.float32)
        new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * ratio), tf.int32) 
        new_width = tf.cast(tf.round(tf.cast(width, tf.float32) * ratio), tf.int32)
        
        image = tf.image.resize(image, [new_height, new_width])
        image = tf.image.random_crop(image, size=[img_size, img_size, 3])
        
        # 医学图像友好的增强（减少可能破坏诊断特征的变换）
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)  # 减小变化范围
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)  # 减小变化范围
        # 移除色相和饱和度变化（可能影响细胞特征）
    else:
        # 推理时的预处理
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        ratio = tf.cast(img_size + 32, tf.float32) / tf.cast(tf.minimum(height, width), tf.float32)
        new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * ratio), tf.int32)
        new_width = tf.cast(tf.round(tf.cast(width, tf.float32) * ratio), tf.int32)
        image = tf.image.resize(image, [new_height, new_width])
        image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)

    # 使用官方VGG16预处理
    image = preprocess_input(image)
    return image

def load_data_with_split(data_dir, img_size=224, batch_size=32, test_size=0.3, val_size=0.15):
    """
    一次性加载数据，然后分别处理训练集和验证/测试集
    
    Parameters:
    - data_dir: 数据目录路径
    - img_size: 图像尺寸，默认224x224
    - batch_size: 批处理大小（此函数中未使用，保持接口一致）
    - test_size: 测试集比例，默认0.3（30%）
    - val_size: 验证集比例，默认0.15（15%）
    
    Returns:
    - X_train, train_labels, train_quals: 训练集数据、标签、质量
    - X_val, val_labels, val_quals: 验证集数据、标签、质量  
    - X_test, test_labels, test_quals: 测试集数据、标签、质量
    
    Note:
    - 训练集：应用数据增强 (is_training=True)
    - 验证集：不应用数据增强 (is_training=False)
    - 测试集：不应用数据增强 (is_training=False)
    - 最终分配比例：训练集70%，验证集15%，测试集15%
    """
    image_paths = []
    labels = []
    qualities = []
    
    # 收集图像路径和标签
    for label, class_name in enumerate(['class_0', 'class_1']):
        class_dir = os.path.join(data_dir, 'train', class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            continue
            
        for img_file in sorted(os.listdir(class_dir)):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            image_paths.append(img_path)
            labels.append(label)
            qualities.append(1 if class_name == 'class_1' else 0)
    
    print(f"找到 {len(image_paths)} 张图像")
    print(f"Class 0: {labels.count(0)} 张，Class 1: {labels.count(1)} 张")
    
    # 先分割数据（基于路径，不处理图像）
    from sklearn.model_selection import train_test_split
    
    # 第一次分割：训练集 vs 临时集 (70% vs 30%)
    train_paths, temp_paths, train_labels, temp_labels, train_quals, temp_quals = train_test_split(
        image_paths, labels, qualities, test_size=0.3, random_state=42, stratify=labels
    )
    
    # 第二次分割：验证集 vs 测试集 (从30%中分出15%和15%)
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
    def preprocess_images(paths, is_training):
        """预处理图像列表"""
        processed_images = []
        for path in paths:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, expand_animations=False)
            img = preprocess_image_vgg16(img, img_size, is_training)
            processed_images.append(img.numpy())
        return np.array(processed_images)
    
    print("=== 预处理训练集（启用增强）===")
    X_train = preprocess_images(train_paths, is_training=True)
    
    print("=== 预处理验证集（不增强）===")
    X_val = preprocess_images(val_paths, is_training=False)
    
    print("=== 预处理测试集（不增强）===")
    X_test = preprocess_images(test_paths, is_training=False)
    
    return (X_train, train_labels, train_quals,
            X_val, val_labels, val_quals,
            X_test, test_labels, test_quals)
