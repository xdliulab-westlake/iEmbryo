"""数据增强模块 - 从BiLSTM.py中提取的数据增强相关函数"""

import tensorflow as tf
import numpy as np

def create_augmentation_fn(training=True):
    """创建数据增强函数"""
    if not training:
        return lambda x: x
        
    def augment(image):
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)  # 左右翻转
        if tf.random.uniform([]) > 0.7:
            image = tf.image.flip_up_down(image)     # 上下翻转
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_brightness(image, max_delta=0.1)  # 亮度调整
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)  # 对比度调整
        return image
    return augment

def improved_cv_augmentation(features, training=True):
    """改进的CV数据增强"""
    if not training:
        return features
    
    # 添加高斯噪声
    if tf.random.uniform([]) > 0.7:
        noise_std = tf.random.uniform([], 0.001, 0.005)
        noise = tf.random.normal(tf.shape(features), mean=0.0, stddev=noise_std)
        features = features + noise
    
    # 随机缩放
    if tf.random.uniform([]) > 0.8:
        scale_factor = tf.random.uniform([], 0.95, 1.05)
        features = features * scale_factor
    
    # 随机掩码（模拟dropout）
    if tf.random.uniform([]) > 0.9:
        mask = tf.random.uniform(tf.shape(features)) > 0.1
        features = features * tf.cast(mask, tf.float32)
    
    return features

def simple_cv_augmentation(features, training=True):
    """简化的CV数据增强"""
    if not training:
        return features
    if tf.random.uniform([]) > 0.8:
        noise = tf.random.normal(tf.shape(features), mean=0.0, stddev=0.003)
        features = features + noise
    return features
