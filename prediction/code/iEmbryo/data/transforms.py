"""
数据变换和增强
支持训练和验证的不同变换策略
"""

import torch
import torchvision.transforms as transforms
from typing import Dict, Optional, Callable


def create_train_transforms(config: Dict) -> Callable:
    """
    创建训练数据变换

    Args:
        config: 配置字典

    Returns:
        训练数据变换组合
    """
    # 根据模型选择合适的输入尺寸
    model_name = config.get('model', {}).get('name', 'vgg16')
    if model_name == 'inceptionv3':
        img_size = 299
    else:
        img_size = config['data']['img_size']

    aug_config = config.get('augmentation', {})

    if not aug_config.get('enabled', True):
        # 禁用数据增强
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    # 启用数据增强
    # 注意：STORK预处理已经做了圆形培养皿裁剪，这里直接resize保持圆形
    transform_list = [
        transforms.Resize((img_size, img_size)),  # 直接resize，不做裁剪
    ]

    # 旋转
    if aug_config.get('rotation_range', 0) > 0:
        transform_list.append(
            transforms.RandomRotation(aug_config['rotation_range'])
        )

    # 水平翻转
    if aug_config.get('horizontal_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # 垂直翻转（对胚胎图像可能有用）
    if aug_config.get('vertical_flip', False):
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    # 颜色抖动（亮度、对比度、饱和度、色调）
    brightness_range = aug_config.get('brightness_range', [0.9, 1.1])
    contrast_range = aug_config.get('contrast_range', [0.9, 1.1])
    saturation_range = aug_config.get('saturation_range', [0.9, 1.1])
    hue_range = aug_config.get('hue_range', [-0.1, 0.1])

    transform_list.append(
        transforms.ColorJitter(
            brightness=(brightness_range[0], brightness_range[1]),
            contrast=(contrast_range[0], contrast_range[1]),
            saturation=(saturation_range[0], saturation_range[1]),
            hue=(hue_range[0], hue_range[1])
        )
    )

    # 随机擦除
    if aug_config.get('random_erasing', False):
        transform_list.extend([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            transforms.ToPILImage()
        ])

    # 最终变换
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(transform_list)


def create_val_transforms(config: Dict) -> Callable:
    """
    创建验证/测试数据变换

    Args:
        config: 配置字典

    Returns:
        验证数据变换组合
    """
    # 根据模型选择合适的输入尺寸
    model_name = config.get('model', {}).get('name', 'vgg16')
    if model_name == 'inceptionv3':
        img_size = 299
    else:
        img_size = config['data']['img_size']

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def create_gradcam_transforms(config: Dict) -> Callable:
    """
    创建Grad-CAM可视化用的变换（不归一化）

    Args:
        config: 配置字典

    Returns:
        Grad-CAM变换组合
    """
    img_size = config['data']['img_size']

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])


class MixUp:
    """MixUp数据增强"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, batch):
        """
        应用MixUp增强

        Args:
            batch: (images, labels) 批次

        Returns:
            混合后的批次
        """
        images, labels = batch
        batch_size = images.size(0)

        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1

        # 随机排列索引
        index = torch.randperm(batch_size)

        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[index]

        # 混合标签
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """CutMix数据增强"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, batch):
        """
        应用CutMix增强

        Args:
            batch: (images, labels) 批次

        Returns:
            混合后的批次
        """
        images, labels = batch
        batch_size = images.size(0)

        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1

        # 随机排列索引
        index = torch.randperm(batch_size)

        # 计算边界框
        W, H = images.size(-1), images.size(-2)
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机选择中心点
        cx = torch.randint(W, (1,))
        cy = torch.randint(H, (1,))

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        # 应用CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


def create_advanced_transforms(config: Dict) -> Dict[str, Callable]:
    """
    创建高级数据增强变换

    Args:
        config: 配置字典

    Returns:
        包含各种变换的字典
    """
    transforms_dict = {
        'train': create_train_transforms(config),
        'val': create_val_transforms(config),
        'gradcam': create_gradcam_transforms(config)
    }

    # 添加MixUp和CutMix
    aug_config = config.get('augmentation', {})
    if aug_config.get('mixup_alpha', 0) > 0:
        transforms_dict['mixup'] = MixUp(alpha=aug_config['mixup_alpha'])

    if aug_config.get('cutmix_alpha', 0) > 0:
        transforms_dict['cutmix'] = CutMix(alpha=aug_config['cutmix_alpha'])

    return transforms_dict


# 预定义的变换策略
TRANSFORM_PRESETS = {
    'basic': {
        'augmentation': {
            'enabled': True,
            'rotation_range': 10,
            'horizontal_flip': True,
            'brightness_range': [0.9, 1.1]
        }
    },
    'aggressive': {
        'augmentation': {
            'enabled': True,
            'rotation_range': 20,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': [0.8, 1.2],
            'random_erasing': True,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0
        }
    },
    'minimal': {
        'augmentation': {
            'enabled': True,
            'rotation_range': 5,
            'horizontal_flip': True
        }
    },
    'none': {
        'augmentation': {
            'enabled': False
        }
    }
}