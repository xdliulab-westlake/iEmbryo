"""
统一数据增强模块
为STORK胚胎分类系统提供标准化的数据增强功能
"""

import cv2
import numpy as np
from typing import Union, List


class EmbryoAugmentationConfig:
    """胚胎图像数据增强配置类"""
    
    def __init__(self):
        # 几何变换概率
        self.horizontal_flip_prob = 0.5
        self.vertical_flip_prob = 0.5 
        self.rotation_prob = 0.4
        
        # 光度变换概率
        self.brightness_prob = 0.4
        self.contrast_prob = 0.4
        
        # 变换参数范围
        self.rotation_angle_range = (-15, 15)  # 度
        self.brightness_factor_range = (0.85, 1.15)
        self.contrast_factor_range = (0.85, 1.15)
        
        # 边界处理模式
        self.border_mode = cv2.BORDER_REFLECT


class EmbryoDataAugmenter:
    """胚胎图像数据增强器"""
    
    def __init__(self, config: EmbryoAugmentationConfig = None):
        self.config = config or EmbryoAugmentationConfig()
    
    def _ensure_uint8(self, image: np.ndarray) -> tuple:
        """确保图像为uint8格式，并记录原始格式"""
        original_max = np.max(image)
        is_normalized = original_max <= 1.0
        
        if image.dtype != np.uint8:
            if is_normalized:
                processed_image = (image * 255.0).astype(np.uint8)
            else:
                processed_image = image.astype(np.uint8)
        else:
            processed_image = image.copy()
            
        return processed_image, is_normalized
    
    def _restore_format(self, image: np.ndarray, is_normalized: bool) -> np.ndarray:
        """恢复图像到原始格式"""
        if is_normalized:
            return image.astype(np.float32) / 255.0
        else:
            return image.astype(np.float32)
    
    def _apply_horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """应用随机水平翻转"""
        if np.random.rand() < self.config.horizontal_flip_prob:
            return np.fliplr(image)
        return image
    
    def _apply_vertical_flip(self, image: np.ndarray) -> np.ndarray:
        """应用随机垂直翻转"""
        if np.random.rand() < self.config.vertical_flip_prob:
            return np.flipud(image)
        return image
    
    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """应用随机小角度旋转"""
        if np.random.rand() < self.config.rotation_prob:
            angle = np.random.uniform(*self.config.rotation_angle_range)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h), borderMode=self.config.border_mode)
        return image
    
    def _apply_brightness_adjustment(self, image: np.ndarray) -> np.ndarray:
        """应用随机亮度调整"""
        if np.random.rand() < self.config.brightness_prob:
            brightness_factor = np.random.uniform(*self.config.brightness_factor_range)
            return np.clip(image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        return image
    
    def _apply_contrast_adjustment(self, image: np.ndarray) -> np.ndarray:
        """应用随机对比度调整"""
        if np.random.rand() < self.config.contrast_prob:
            contrast_factor = np.random.uniform(*self.config.contrast_factor_range)
            mean = np.mean(image.astype(np.float32))
            return np.clip((image.astype(np.float32) - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        return image
    
    def augment_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        对单张图像应用数据增强
        
        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            
        Returns:
            增强后的图像，保持原始数据类型和格式
        """
        # 确保为uint8格式进行处理
        aug_image, is_normalized = self._ensure_uint8(image)
        
        # 应用几何变换
        aug_image = self._apply_horizontal_flip(aug_image)
        aug_image = self._apply_vertical_flip(aug_image)
        aug_image = self._apply_rotation(aug_image)
        
        # 应用光度变换
        aug_image = self._apply_brightness_adjustment(aug_image)
        aug_image = self._apply_contrast_adjustment(aug_image)
        
        # 恢复原始格式
        return self._restore_format(aug_image, is_normalized)
    
    def augment_batch(self, images: np.ndarray) -> np.ndarray:
        """
        对批量图像应用数据增强
        
        Args:
            images: 输入图像批次 (N, H, W, C) 或 (N, H, W)
            
        Returns:
            增强后的图像批次
        """
        augmented_images = []
        
        for image in images:
            aug_image = self.augment_single_image(image)
            augmented_images.append(aug_image)
        
        return np.array(augmented_images)
    
    def print_config(self):
        """打印当前增强配置"""
        print("🎨 数据增强配置:")
        print(f"   水平翻转概率: {self.config.horizontal_flip_prob}")
        print(f"   垂直翻转概率: {self.config.vertical_flip_prob}")
        print(f"   旋转概率: {self.config.rotation_prob} (角度: {self.config.rotation_angle_range}°)")
        print(f"   亮度调整概率: {self.config.brightness_prob} (范围: {self.config.brightness_factor_range})")
        print(f"   对比度调整概率: {self.config.contrast_prob} (范围: {self.config.contrast_factor_range})")


# 全局默认增强器实例
_default_augmenter = None

def get_default_augmenter() -> EmbryoDataAugmenter:
    """获取默认的数据增强器单例"""
    global _default_augmenter
    if _default_augmenter is None:
        _default_augmenter = EmbryoDataAugmenter()
    return _default_augmenter

def augment_training_data(images: np.ndarray, 
                         augmenter: EmbryoDataAugmenter = None,
                         verbose: bool = False) -> np.ndarray:
    """
    为训练数据应用标准增强
    
    Args:
        images: 输入图像 (N, H, W, C)
        augmenter: 自定义增强器，None时使用默认配置
        verbose: 是否打印增强配置
        
    Returns:
        增强后的图像
    """
    if augmenter is None:
        augmenter = get_default_augmenter()
    
    if verbose:
        augmenter.print_config()
    
    return augmenter.augment_batch(images)

def create_custom_augmenter(horizontal_flip_prob: float = 0.5,
                          vertical_flip_prob: float = 0.5,
                          rotation_prob: float = 0.4,
                          brightness_prob: float = 0.4,
                          contrast_prob: float = 0.4,
                          rotation_range: tuple = (-15, 15),
                          brightness_range: tuple = (0.85, 1.15),
                          contrast_range: tuple = (0.85, 1.15)) -> EmbryoDataAugmenter:
    """
    创建自定义配置的数据增强器
    
    Args:
        各种增强参数的概率和范围
        
    Returns:
        配置好的数据增强器
    """
    config = EmbryoAugmentationConfig()
    config.horizontal_flip_prob = horizontal_flip_prob
    config.vertical_flip_prob = vertical_flip_prob
    config.rotation_prob = rotation_prob
    config.brightness_prob = brightness_prob
    config.contrast_prob = contrast_prob
    config.rotation_angle_range = rotation_range
    config.brightness_factor_range = brightness_range
    config.contrast_factor_range = contrast_range
    
    return EmbryoDataAugmenter(config)


