"""
Model package for Multi-Task TMFI-Net

Main components:
- TMFI-Embryo: Multi-Task TMFI-Net for embryo quality assessment
- Focal Fusion: Multi-focal plane fusion for embryo data

完全使用TMFI-Net的原始架构和损失函数
"""

# TMFI-Embryo model (100% TMFI-Net architecture)
from .tmfi_embryo import (
    EmbryoQualityModel,
    DecoderConvUp,
    Gate,
    create_tmfi_embryo_model
)

# Video Swin Transformer (TMFI-Net backbone)
from .swin_transformer import SwinTransformer3D

# Focal plane fusion (from dataset package)
from dataset.focal_fusion import (
    FocalPlaneFusion as FocalPlaneDataFusion,
    FocalFusionWrapper,
    create_focal_fusion_dataloader_wrapper
)

__all__ = [
    # Multi-Task TMFI-Net
    'EmbryoQualityModel',
    'DecoderConvUp',
    'Gate',
    'SwinTransformer3D',
    'create_tmfi_embryo_model',

    # Focal Fusion
    'FocalPlaneDataFusion',
    'FocalFusionWrapper',
    'create_focal_fusion_dataloader_wrapper',
]
