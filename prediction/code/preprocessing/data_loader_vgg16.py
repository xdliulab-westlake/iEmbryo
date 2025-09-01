import os
import tensorflow as tf
import numpy as np
import re
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

# Read and preprocess image file
def parse_image_vgg16(img_path, label, img_size=224, is_training=True):
    """VGG16兼容的图像解析函数"""
    img_path = tf.convert_to_tensor(img_path, dtype=tf.string)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, expand_animations=False)

    # 处理通道数
    num_channels = tf.shape(img)[-1]
    img = tf.cond(tf.equal(num_channels, 1), lambda: tf.image.grayscale_to_rgb(img), lambda: img)
    img = tf.cond(tf.greater(num_channels, 3), lambda: img[:, :, :3], lambda: img)

    img.set_shape([None, None, 3])
    img = preprocess_image_vgg16(img, img_size=img_size, is_training=is_training)
    return img, label

# 兼容性包装函数
def load_data_vgg16(data_dir, img_size=224, batch_size=32, is_training=True):
    """
    VGG16兼容的数据加载函数
    """
    image_paths = []
    labels = []
    qualities = []

    # Traverse class folders and collect image paths
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

    # Convert paths and labels to tensors
    image_paths_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int64)

    # Build dataset with VGG16 preprocessing
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, labels_tensor))
    dataset = dataset.map(
        lambda x, y: parse_image_vgg16(x, y, img_size, is_training), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Load all preprocessed images into memory
    images_list = []
    labels_list = []
    for img_batch, label_batch in dataset:
        images_list.append(img_batch.numpy())
        labels_list.append(label_batch.numpy())

    # Return final arrays
    return np.concatenate(images_list), np.concatenate(labels_list), np.array(qualities), image_paths

# 向后兼容的包装
def load_data(data_dir, img_size=224, batch_size=32, is_training=True):
    """向后兼容的函数名"""
    return load_data_vgg16(data_dir, img_size, batch_size, is_training)

def load_data_with_split(data_dir, img_size=224, batch_size=32, test_size=0.3, val_size=0.15):
    """
    一次性加载数据，然后分别处理训练集和验证/测试集
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
    
    # 第一次分割：训练集 vs 临时集
    train_paths, temp_paths, train_labels, temp_labels, train_quals, temp_quals = train_test_split(
        image_paths, labels, qualities, test_size=test_size, random_state=42, stratify=labels
    )
    
    # 第二次分割：验证集 vs 测试集
    val_paths, test_paths, val_labels, test_labels, val_quals, test_quals = train_test_split(
        temp_paths, temp_labels, temp_quals, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
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
