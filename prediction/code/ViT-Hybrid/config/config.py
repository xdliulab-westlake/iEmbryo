"""
Configuration for ViT-Hybrid Model (PyTorch)
Author: Liao Zizhuo
Date: 2025-10-27
"""

from pathlib import Path


class ModelConfig:
    """模型配置"""
    TIME_STEPS = 33
    FOCAL_PLANES = 11
    IMAGE_SIZE = 224

    NUM_TRANSFORMER_LAYERS = 4
    NUM_HEADS = 8
    D_MODEL = 4096  # 与VGG16输出维度保持一致（STORK风格）
    DFF = 2048      # 前馈网络维度
    DROPOUT_RATE = 0.1

    FUSION_TYPE = 'average'


class TrainingConfig:
    """训练配置"""
    BATCH_SIZE = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7

    MONITOR_METRIC = 'val_auc'
    MONITOR_MODE = 'max'

    NUM_WORKERS = 4
    PIN_MEMORY = True


class DataConfig:
    """数据配置"""
    DATA_ROOT = '/path/to/embryo/data'
    SCORE_DICT_PATH = 'score_dict.json'

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    APPLY_PETRI_CROP = True
    APPLY_CENTER_CROP = True


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
    """获取模型配置字典"""
    return {
        'time_steps': ModelConfig.TIME_STEPS,
        'focal_planes': ModelConfig.FOCAL_PLANES,
        'num_transformer_layers': ModelConfig.NUM_TRANSFORMER_LAYERS,
        'num_heads': ModelConfig.NUM_HEADS,
        'd_model': ModelConfig.D_MODEL,
        'dff': ModelConfig.DFF,
        'dropout_rate': ModelConfig.DROPOUT_RATE,
        'fusion_type': ModelConfig.FUSION_TYPE
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
        'pin_memory': TrainingConfig.PIN_MEMORY
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
        'apply_center_crop': DataConfig.APPLY_CENTER_CROP
    }
