"""
GPU加速数据增强模块
使用Kornia库实现GPU端数据增强，大幅提升训练速度
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import kornia as K
import kornia.augmentation as KA


class GPUAugmentation(nn.Module):
    """
    GPU加速数据增强模块

    在模型内部执行数据增强，避免CPU-GPU数据传输开销
    """

    def __init__(self, config: Dict, training: bool = True):
        super().__init__()
        self.training_mode = training
        aug_config = config.get('augmentation', {})

        if not aug_config.get('enabled', True) or not training:
            self.augmentations = nn.Identity()
            return

        # 构建增强序列
        augment_list = []

        # 几何变换
        if aug_config.get('rotation_range', 0) > 0:
            rotation_range = aug_config['rotation_range']
            augment_list.append(
                KA.RandomRotation(
                    degrees=rotation_range,
                    p=0.5
                )
            )

        # 翻转
        if aug_config.get('horizontal_flip', False):
            augment_list.append(KA.RandomHorizontalFlip(p=0.5))

        if aug_config.get('vertical_flip', False):
            augment_list.append(KA.RandomVerticalFlip(p=0.5))

        # 颜色增强
        if aug_config.get('brightness_range'):
            brightness_range = aug_config['brightness_range']
            # Kornia使用加法，而不是乘法
            brightness_factor = (brightness_range[1] - brightness_range[0]) / 2
            augment_list.append(
                KA.ColorJitter(
                    brightness=brightness_factor,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=0.5
                )
            )

        # 随机擦除
        if aug_config.get('random_erasing', False):
            augment_list.append(
                KA.RandomErasing(
                    scale=(0.02, 0.1),
                    ratio=(0.3, 3.3),
                    p=0.1
                )
            )

        # 高斯噪声（GPU特有增强）
        if aug_config.get('gaussian_noise', False):
            augment_list.append(
                KA.RandomGaussianNoise(
                    mean=0.0,
                    std=0.01,
                    p=0.2
                )
            )

        # 高斯模糊
        if aug_config.get('gaussian_blur', False):
            augment_list.append(
                KA.RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=(0.1, 2.0),
                    p=0.2
                )
            )

        # 弹性变换（医学图像特有）
        if aug_config.get('elastic_transform', False):
            augment_list.append(
                KA.RandomElasticTransform(
                    kernel_size=(63, 63),
                    sigma=(32.0, 32.0),
                    alpha=(1.0, 1.0),
                    p=0.2
                )
            )

        # 创建增强容器
        if augment_list:
            self.augmentations = KA.AugmentationSequential(
                *augment_list,
                data_keys=["input"],
                random_apply=1  # 随机应用部分增强
            )
        else:
            self.augmentations = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [B, C, H, W]

        Returns:
            增强后的张量
        """
        if not self.training or not self.training_mode:
            return x

        # 确保输入在正确设备上
        if not x.is_cuda:
            x = x.cuda()

        # 应用增强
        return self.augmentations(x)


class MixUpGPU(nn.Module):
    """GPU版本的MixUp增强"""

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        应用MixUp增强

        Args:
            x: 图像张量 [B, C, H, W]
            y: 标签张量 [B]

        Returns:
            (mixed_x, y_a, y_b, lam)
        """
        if not self.training or self.alpha <= 0:
            return x, y, y, 1.0

        batch_size = x.size(0)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(x.device)

        # 随机排列
        index = torch.randperm(batch_size, device=x.device)

        # 混合
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


class CutMixGPU(nn.Module):
    """GPU版本的CutMix增强"""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        应用CutMix增强

        Args:
            x: 图像张量 [B, C, H, W]
            y: 标签张量 [B]

        Returns:
            (mixed_x, y_a, y_b, lam)
        """
        if not self.training or self.alpha <= 0:
            return x, y, y, 1.0

        batch_size = x.size(0)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(x.device)

        # 随机排列
        index = torch.randperm(batch_size, device=x.device)

        # 计算边界框
        W, H = x.size(-1), x.size(-2)
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机选择中心点
        cx = torch.randint(W, (1,), device=x.device)
        cy = torch.randint(H, (1,), device=x.device)

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        # 应用CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


class STORKGPUAugmentation(nn.Module):
    """
    STORK专用GPU数据增强

    针对胚胎图像的特殊增强策略
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        aug_config = config.get('augmentation', {})

        # 基础增强
        self.basic_aug = GPUAugmentation(config, training=True)

        # 高级增强
        self.mixup = None
        self.cutmix = None

        if aug_config.get('mixup_alpha', 0) > 0:
            self.mixup = MixUpGPU(alpha=aug_config['mixup_alpha'])

        if aug_config.get('cutmix_alpha', 0) > 0:
            self.cutmix = CutMixGPU(alpha=aug_config['cutmix_alpha'])

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> tuple:
        """
        前向传播

        Args:
            x: 输入图像 [B, C, H, W]
            y: 标签 [B] (可选)

        Returns:
            增强后的数据
        """
        # 基础增强
        x = self.basic_aug(x)

        # 如果没有标签，只返回图像
        if y is None:
            return x

        # 随机选择高级增强
        if self.training and (self.mixup is not None or self.cutmix is not None):
            if torch.rand(1) < 0.5 and self.mixup is not None:
                return self.mixup(x, y)
            elif self.cutmix is not None:
                return self.cutmix(x, y)

        return x, y, y, 1.0


def create_gpu_transforms(config: Dict) -> STORKGPUAugmentation:
    """
    创建GPU数据增强

    Args:
        config: 配置字典

    Returns:
        GPU增强模块
    """
    return STORKGPUAugmentation(config)


def create_simple_gpu_transforms(config: Dict) -> GPUAugmentation:
    """
    创建简单GPU数据增强（只有基础增强）

    Args:
        config: 配置字典

    Returns:
        简单GPU增强模块
    """
    return GPUAugmentation(config, training=True)


def check_kornia_available() -> bool:
    """检查Kornia是否可用"""
    try:
        import kornia
        import kornia.augmentation
        return True
    except ImportError:
        return False


# 增强预设配置
GPU_TRANSFORM_PRESETS = {
    'basic_gpu': {
        'augmentation': {
            'enabled': True,
            'rotation_range': 15,
            'horizontal_flip': True,
            'brightness_range': [0.9, 1.1],
            'gaussian_noise': False,
            'gaussian_blur': False
        }
    },
    'aggressive_gpu': {
        'augmentation': {
            'enabled': True,
            'rotation_range': 20,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': [0.8, 1.2],
            'random_erasing': True,
            'gaussian_noise': True,
            'gaussian_blur': True,
            'elastic_transform': True,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0
        }
    },
    'medical_gpu': {
        'augmentation': {
            'enabled': True,
            'rotation_range': 10,
            'horizontal_flip': True,
            'vertical_flip': True,
            'brightness_range': [0.9, 1.1],
            'gaussian_noise': True,
            'elastic_transform': True
        }
    }
}