import os
import tensorflow as tf
import numpy as np
import cv2
import pickle
import hashlib
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess  
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from .bilstm_image_utils import preprocess_embryo_image
from tensorflow import keras

def build_feature_extractor(img_size: int, model_name: str = 'efficientnet'):
    """
    Build feature extractor for BiLSTM with preprocessing
    
    Args:
        img_size: Input image size
        model_name: Model name ('efficientnet', 'resnet', 'densenet', 'vgg16')
    
    Returns:
        Keras model for feature extraction
    """
    if model_name == 'efficientnet':
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M
        feature_extractor = EfficientNetV2M(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(img_size, img_size, 3)
        )
        preprocess_fn = efficientnet_preprocess
    elif model_name == 'resnet':
        from tensorflow.keras.applications import ResNet152V2
        feature_extractor = ResNet152V2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(img_size, img_size, 3)
        )
        preprocess_fn = resnet_preprocess
    elif model_name == 'densenet':
        from tensorflow.keras.applications import DenseNet201
        feature_extractor = DenseNet201(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(img_size, img_size, 3)
        )
        preprocess_fn = densenet_preprocess
    elif model_name == 'vgg16':
        from tensorflow.keras.applications import VGG16
        feature_extractor = VGG16(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(img_size, img_size, 3)
        )
        preprocess_fn = vgg16_preprocess
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    inputs = keras.Input((img_size, img_size, 3))
    preprocessed = preprocess_fn(inputs)
    outputs = feature_extractor(preprocessed)
    
    return keras.Model(inputs, outputs, name=f"{model_name}_feature_extractor")

def preprocess_image_bilstm_universal(image, img_size=224, is_training=True):
    """
    BiLSTM胚胎图像预处理 + 统一ImageNet标准化

    Args:
        image: 输入图像 (tensor或numpy array)
        img_size: 目标图像尺寸
        is_training: 是否为训练模式（影响数据增强）

    Returns:
        预处理后的图像 (统一ImageNet标准化)
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
    
    # 使用BiLSTM预处理管道
    image = preprocess_embryo_image(
        image, 
        img_size=img_size, 
        apply_petri_crop=True,  # 启用培养皿检测
        apply_center_crop=True  # 启用中心裁剪
    )
    
    # 转换为float32以便后续处理
    image = image.astype(np.float32)
    
    if is_training:
        # 训练时的轻量级数据增强 (在Hough预处理基础上)
        if np.random.random() > 0.5:
            image = tf.image.random_flip_left_right(image)
        if np.random.random() > 0.7:
            image = tf.image.random_brightness(image, max_delta=0.1)
        if np.random.random() > 0.7:
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        if np.random.random() > 0.8:
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    
    # 🔥 不在这里应用ImageNet标准化，留给各模型特定的预处理函数
    # 确保图像在 [0, 255] 范围内，以uint8格式返回
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def load_and_preprocess_images_for_bilstm(image_paths, img_size=224, is_training=True,
                                         batch_size=32):
    """
    为BiLSTM加载和预处理批量图像

    Args:
        image_paths: 图像路径列表
        img_size: 目标图像尺寸
        is_training: 是否为训练模式
        batch_size: 批量大小

    Returns:
        预处理后的图像数据集 (统一ImageNet标准化)
    """
    def load_and_preprocess_fn(path):
        # 读取图像
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.uint8)
        
        # 应用BiLSTM预处理
        img = tf.py_function(
            lambda x: preprocess_image_bilstm_universal(x, img_size, is_training),
            [img],
            tf.float32
        )
        img.set_shape([img_size, img_size, 3])
        
        return img
    
    # 构建数据集
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if batch_size:
        dataset = dataset.batch(batch_size)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_embryo_images_by_class(data_root, valid_embryo_ids=None):
    """
    根据类别获取胚胎图像路径
    
    Args:
        data_root: 数据根目录
        valid_embryo_ids: 有效胚胎ID列表
    
    Returns:
        dict: {class_label: [image_paths]}
    """
    class_images = {'0': [], '1': []}
    
    # 支持的图像格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # 遍历数据目录
    for root, dirs, files in os.walk(data_root):
        # 检查是否是类别目录
        class_folder = os.path.basename(root)
        
        if class_folder in ['0', 'class_0', 'negative', 'no', 'non_viable']:
            target_class = '0'
        elif class_folder in ['1', 'class_1', 'positive', 'yes', 'viable']:  
            target_class = '1'
        else:
            continue
            
        # 收集该类别的所有图像
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # 提取胚胎ID（如果需要）
                if valid_embryo_ids is not None:
                    # 尝试从文件路径中提取胚胎ID
                    embryo_id = extract_embryo_id_from_path(os.path.join(root, file))
                    if embryo_id and embryo_id not in valid_embryo_ids:
                        continue
                
                class_images[target_class].append(os.path.join(root, file))
    
    return class_images

def extract_embryo_id_from_path(file_path):
    """
    从文件路径中提取胚胎ID
    
    Args:
        file_path: 文件路径
    
    Returns:
        str: 胚胎ID或None
    """
    # 这里需要根据实际的文件命名规则来实现
    # 示例实现
    parts = file_path.split('/')
    for part in parts:
        if part.startswith('embryo_') or part.startswith('sample_'):
            return part
    
    return None

def create_bilstm_cache_key(image_paths, params):
    """
    为BiLSTM创建缓存键
    
    Args:
        image_paths: 图像路径列表
        params: 预处理参数字典
    
    Returns:
        str: 缓存键
    """
    # 创建基于路径和参数的哈希
    paths_str = ''.join(sorted(image_paths))
    params_str = str(sorted(params.items()))
    combined_str = paths_str + params_str
    
    return hashlib.md5(combined_str.encode()).hexdigest()[:16]

def load_cached_features(cache_file):
    """
    加载缓存的特征
    
    Args:
        cache_file: 缓存文件路径
    
    Returns:
        dict: 缓存的特征数据
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"加载缓存失败: {e}")
    
    return None

def save_features_cache(features_data, cache_file):
    """
    保存特征缓存
    
    Args:
        features_data: 特征数据
        cache_file: 缓存文件路径
    """
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(features_data, f)
        print(f"特征缓存已保存: {cache_file}")
    except Exception as e:
        print(f"保存缓存失败: {e}")