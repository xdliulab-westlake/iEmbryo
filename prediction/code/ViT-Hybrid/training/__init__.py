"""
Training package for Multi-Task TMFI-Net

完全使用TMFI-Net的原始架构和损失函数进行多任务学习
"""

from .train_tmfi import (
    train_one_epoch,
    validate,
    main
)

__all__ = [
    'train_one_epoch',
    'validate',
    'main'
]
