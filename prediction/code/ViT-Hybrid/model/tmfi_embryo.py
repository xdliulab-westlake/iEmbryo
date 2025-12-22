"""
Multi-Task TMFI-Net for Embryo Quality Assessment
完全使用TMFI-Net的原始架构，进行多任务学习：
1. 主任务：胚胎质量分类 (Good/Poor)
2. 辅助任务：Saliency prediction (使用TMFI-Net原始decoder和loss)

保留TMFI-Net的所有原始组件：
- Video Swin Transformer backbone
- Gate modules (Multi-Dimensional Attention)
- DecoderConvUp (FPN + Hierarchical Decoder)
- Saliency output (B, H, W)

新增分类头，从decoder特征中提取

Author: Liao Zizhuo
Date: 2025-12-22
"""

import torch
from torch import nn
from .swin_transformer import SwinTransformer3D


class Gate(nn.Module):
    """
    Multi-Dimensional Attention Module (from TMFI-Net)

    Applies sigmoid gating across channel, temporal, and spatial dimensions
    """
    def __init__(self, in_plane):
        super(Gate, self).__init__()
        self.gate = nn.Conv3d(in_plane, in_plane, kernel_size=(1, 3, 3),
                             stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, rgb_fea):
        gate = torch.sigmoid(self.gate(rgb_fea))
        gate_fea = rgb_fea * gate + rgb_fea
        return gate_fea


class EmbryoQualityModel(nn.Module):
    """
    Multi-Task TMFI-Net for Embryo Quality Assessment

    完全使用TMFI-Net的原始架构，输出：
    1. Saliency map (B, H, W) - 用于TMFI-Net的saliency loss (KLDiv, CC, Similarity)
    2. Classification logits (B, num_classes) - 用于分类loss (CrossEntropyLoss)

    Args:
        pretrain: Path to pretrained Video Swin Transformer weights
        num_classes: Number of classification classes (default: 2)
    """
    def __init__(self, pretrain=None, num_classes=2):
        super(EmbryoQualityModel, self).__init__()

        self.num_classes = num_classes

        # TMFI-Net原始组件
        self.backbone = SwinTransformer3D(pretrained=pretrain)
        self.decoder = DecoderConvUp()

        # 新增：分类头（从decoder特征提取）
        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(192 * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (B, C, T, H, W) - Input video

        Returns:
            saliency_map: (B, H, W) - For TMFI-Net saliency loss
            logits: (B, num_classes) - For classification loss
        """
        # Backbone
        x, [y1, y2, y3, y4] = self.backbone(x)

        # Decoder: saliency map + intermediate features
        saliency_map, (feat1, feat2, feat3, feat4) = self.decoder(x, y3, y2, y1)

        # Classification from decoder features
        pool1 = self.global_pool(feat1).squeeze(-1).squeeze(-1)
        pool2 = self.global_pool(feat2).squeeze(-1).squeeze(-1)
        pool3 = self.global_pool(feat3).squeeze(-1).squeeze(-1)
        pool4 = self.global_pool(feat4).squeeze(-1).squeeze(-1)

        concat_feat = torch.cat([pool1, pool2, pool3, pool4], dim=1)
        pooled_feat = concat_feat.mean(dim=2)
        logits = self.classifier(pooled_feat)

        return saliency_map, logits


class DecoderConvUp(nn.Module):
    """
    TMFI-Net的原始Decoder
    完全保留原始实现，输出saliency map和中间特征

    组件（完全来自TMFI-Net）：
    1. Semantic-Guided Encoder (FPN structure)
    2. Multi-Dimensional Attention (Gate modules)
    3. Hierarchical Decoder
    4. Saliency Map Generation
    """
    def __init__(self):
        super(DecoderConvUp, self).__init__()

        # Upsampling layers
        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.upsampling4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear')
        self.upsampling8 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear')

        # Channel alignment (Semantic-Guided Encoder)
        self.conv1 = nn.Conv3d(96, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(192, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv3 = nn.Conv3d(384, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv4 = nn.Conv3d(768, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))

        # FPN structure
        self.convs1 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), bias=False)
        self.convs2 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), bias=False)
        self.convs3 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), bias=False)

        # Multi-Dimensional Attention (Gate modules)
        self.gate1 = Gate(192)
        self.gate2 = Gate(192)
        self.gate3 = Gate(192)
        self.gate4 = Gate(192)

        # Hierarchical Decoder branches (for attention map)
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            self.upsampling2,
            nn.Sigmoid()
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            self.upsampling4,
            nn.Sigmoid()
        )

        # Saliency map fusion (TMFI-Net原始)
        self.convout = nn.Sequential(
            nn.Conv3d(4, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, y4, y3, y2, y1):
        """
        TMFI-Net原始forward

        Args:
            y4, y3, y2, y1: Multi-level features from backbone

        Returns:
            saliency_map: (B, H, W) - 与TMFI-Net loss兼容的saliency map
            features: (y1, y2, y3, y4) - 中间特征，用于分类
        """
        # Channel alignment
        y1 = self.conv1(y1)
        y2 = self.conv2(y2)
        y3 = self.conv3(y3)
        y4 = self.conv4(y4)

        # Semantic-Guided Encoder (FPN structure)
        t3 = self.upsampling2(y4) + y3
        y3 = self.convs3(t3)
        t2 = self.upsampling2(t3) + y2 + self.upsampling4(y4)
        y2 = self.convs2(t2)
        t1 = self.upsampling2(t2) + y1 + self.upsampling8(y4)
        y1 = self.convs1(t1)

        # Multi-Dimensional Attention
        y1 = self.gate1(y1)
        y2 = self.gate2(y2)
        y3 = self.gate3(y3)
        y4 = self.gate4(y4)

        # Hierarchical decoding
        z1 = self.convtsp1(y1)
        z2 = self.convtsp2(y2)
        z3 = self.convtsp3(y3)
        z4 = self.convtsp4(y4)

        # Fuse and output saliency map
        z0 = self.convout(torch.cat((z1, z2, z3, z4), 1))  # (B, 1, T', H, W)

        # Squeeze to (B, H, W) - TMFI-Net原始格式
        saliency_map = z0.view(z0.size(0), z0.size(3), z0.size(4))

        # Return saliency map and intermediate features
        return saliency_map, (y1, y2, y3, y4)


def create_tmfi_embryo_model(config=None):
    """
    Create Multi-Task TMFI-Net model for embryo quality assessment

    Args:
        config: Configuration dictionary with keys:
            - pretrained_path: Path to pretrained Video Swin Transformer
            - num_classes: Number of output classes (default: 2)

    Returns:
        EmbryoQualityModel instance (outputs saliency_map and logits)
    """
    if config is None:
        config = {}

    model = EmbryoQualityModel(
        pretrain=config.get('pretrained_path', None),
        num_classes=config.get('num_classes', 2)
    )

    return model
