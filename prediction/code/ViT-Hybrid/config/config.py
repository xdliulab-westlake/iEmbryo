"""
Configuration for TMFI-Net Model (PyTorch)
Based on "Transformer-Based Multi-Scale Feature Integration Network for Video Saliency Prediction"
Adapted for Embryo Quality Assessment

Author: Liao Zizhuo
Date: 2025-12-22
"""

from pathlib import Path


class ModelConfig:
    """TMFI-Net 模型配置"""
    # 基本参数
    TIME_STEPS = 33
    FOCAL_PLANES = 11
    IMAGE_SIZE = 224

    # Transformer参数
    NUM_TRANSFORMER_LAYERS = 4  # Total number of transformer layers
    NUM_HEADS = 8               # Number of attention heads
    D_MODEL = 512               # Transformer feature dimension (平衡性能和内存)
    DFF = 1024                  # Feed-forward network dimension
    DROPOUT_RATE = 0.1

    # 焦平面融合
    FUSION_TYPE = 'average'     # Focal plane fusion method: 'average', 'max', or 'weighted'

    # TMFI-Net特定参数
    NUM_HIERARCHICAL_LEVELS = 4  # Number of hierarchical levels in decoder (推荐4)
    SEMANTIC_CHANNELS = 192      # Output channels for semantic-guided encoder
    NUM_CLASSES = 2              # Number of output classes (2 for binary classification)


class TrainingConfig:
    """训练配置"""
    BATCH_SIZE = 8  # 增大batch size提高稳定性（4→8）
    EPOCHS = 100
    LEARNING_RATE = 1e-4  # Pre-Norm允许更高的学习率（5e-5→1e-4）
                          # Pre-Norm梯度流更好，不需要极低的LR
                          # 同时梯度裁剪(norm=1.0)防止爆炸
    WEIGHT_DECAY = 1e-5

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7

    MONITOR_METRIC = 'val_auc'
    MONITOR_MODE = 'max'

    NUM_WORKERS = 1  # 改为 1 以避免内存溢出（使用 LRU 缓存时不需要多 workers）
    PIN_MEMORY = True

    VGG_BATCH_SIZE_PER_GPU = 64


class DataConfig:
    """数据配置"""
    DATA_ROOT = '/path/to/embryo/data'
    SCORE_DICT_PATH = 'score_dict.json'

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    APPLY_PETRI_CROP = True
    APPLY_CENTER_CROP = True

    # 平衡采样：True=使用加权随机采样平衡两个类别；False=保持原始分布
    BALANCED_SAMPLING = True


class PathConfig:
    """路径配置（使用绝对路径）"""
    # 项目固定根目录
    PROJECT_ROOT = Path('/liaozizhuo/scripts/ViT-Hybrid')

    # 所有路径均为绝对路径
    MODEL_SAVE_DIR = str(PROJECT_ROOT / 'checkpoints')
    LOG_DIR = str(PROJECT_ROOT / 'logs')
    RESULT_DIR = str(PROJECT_ROOT / 'results')
    CACHE_DIR = str(PROJECT_ROOT / 'cache')  # Validation cache directory


def get_model_config():
    """获取TMFI-Net模型配置字典"""
    return {
        # 基本参数
        'time_steps': ModelConfig.TIME_STEPS,
        'focal_planes': ModelConfig.FOCAL_PLANES,

        # Transformer参数
        'num_transformer_layers': ModelConfig.NUM_TRANSFORMER_LAYERS,
        'num_heads': ModelConfig.NUM_HEADS,
        'd_model': ModelConfig.D_MODEL,
        'dff': ModelConfig.DFF,
        'dropout_rate': ModelConfig.DROPOUT_RATE,

        # 焦平面融合
        'fusion_type': ModelConfig.FUSION_TYPE,

        # TMFI-Net特定参数
        'num_hierarchical_levels': ModelConfig.NUM_HIERARCHICAL_LEVELS,
        'semantic_channels': ModelConfig.SEMANTIC_CHANNELS,
        'num_classes': ModelConfig.NUM_CLASSES
    }


def get_training_config():
    """获取训练配置字典"""
    return {
        'batch_size': TrainingConfig.BATCH_SIZE,
        'epochs': TrainingConfig.EPOCHS,
        'learning_rate': TrainingConfig.LEARNING_RATE,
        'weight_decay': TrainingConfig.WEIGHT_DECAY,
        'early_stopping_patience': TrainingConfig.EARLY_STOPPING_PATIENCE,
        'reduce_lr_patience': TrainingConfig.REDUCE_LR_PATIENCE,
        'reduce_lr_factor': TrainingConfig.REDUCE_LR_FACTOR,
        'min_lr': TrainingConfig.MIN_LR,
        'monitor_metric': TrainingConfig.MONITOR_METRIC,
        'monitor_mode': TrainingConfig.MONITOR_MODE,
        'num_workers': TrainingConfig.NUM_WORKERS,
        'pin_memory': TrainingConfig.PIN_MEMORY,
        'vgg_batch_size_per_gpu': TrainingConfig.VGG_BATCH_SIZE_PER_GPU
    }


def get_data_config():
    """获取数据配置字典"""
    return {
        'data_root': DataConfig.DATA_ROOT,
        'score_dict_path': DataConfig.SCORE_DICT_PATH,
        'time_steps': ModelConfig.TIME_STEPS,
        'focal_planes': ModelConfig.FOCAL_PLANES,
        'image_size': ModelConfig.IMAGE_SIZE,
        'batch_size': TrainingConfig.BATCH_SIZE,
        'train_ratio': DataConfig.TRAIN_RATIO,
        'val_ratio': DataConfig.VAL_RATIO,
        'num_workers': TrainingConfig.NUM_WORKERS,
        'pin_memory': TrainingConfig.PIN_MEMORY,
        'apply_petri_crop': DataConfig.APPLY_PETRI_CROP,
        'apply_center_crop': DataConfig.APPLY_CENTER_CROP,
        'balanced_sampling': DataConfig.BALANCED_SAMPLING
    }
