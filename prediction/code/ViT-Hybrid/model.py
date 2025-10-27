"""
ViT-Hybrid Model Architecture (PyTorch)
Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import torch.nn as nn
from torchvision import models
from layers import FocalPlaneFusion, PositionalEncoding, TransformerBlock


class VGG16FeatureExtractor(nn.Module):
    """VGG16特征提取器"""
    def __init__(self, pretrained=True):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(True)
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def unfreeze_layers(self, num_layers=-4):
        """解冻部分VGG16层"""
        layers = list(self.features.children())
        for layer in layers[num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


class ViTHybridModel(nn.Module):
    """
    ViT-Hybrid模型: VGG16 + Transformer

    Args:
        time_steps: 时间点数量
        focal_planes: 焦平面数量
        num_transformer_layers: Transformer层数
        num_heads: 注意力头数
        d_model: 特征维度
        dff: 前馈网络维度
        dropout_rate: Dropout率
        fusion_type: 融合方式
    """
    def __init__(
        self,
        time_steps=33,
        focal_planes=11,
        num_transformer_layers=4,
        num_heads=8,
        d_model=512,
        dff=1024,
        dropout_rate=0.1,
        fusion_type='average'
    ):
        super(ViTHybridModel, self).__init__()

        self.time_steps = time_steps
        self.focal_planes = focal_planes
        self.d_model = d_model

        self.vgg16_extractor = VGG16FeatureExtractor(pretrained=True)
        self.focal_fusion = FocalPlaneFusion(fusion_type=fusion_type)
        self.pos_encoding = PositionalEncoding(time_steps, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_transformer_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(-1, 3, 224, 224)
        vgg_features = self.vgg16_extractor(x)
        vgg_features = vgg_features.view(batch_size, self.time_steps, self.focal_planes, self.d_model)

        time_features = []
        for t in range(self.time_steps):
            focal_features = vgg_features[:, t, :, :]
            fused_feature = self.focal_fusion(focal_features)
            time_features.append(fused_feature)

        temporal_features = torch.stack(time_features, dim=1)
        temporal_features = self.pos_encoding(temporal_features)

        x = temporal_features
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = torch.mean(x, dim=1)
        output = self.classifier(x)

        return output

    def unfreeze_vgg16(self, num_layers=-4):
        """解冻VGG16层"""
        self.vgg16_extractor.unfreeze_layers(num_layers)


def create_model(config=None):
    """
    创建模型

    Args:
        config: 配置字典

    Returns:
        model
    """
    if config is None:
        config = {}

    model = ViTHybridModel(
        time_steps=config.get('time_steps', 33),
        focal_planes=config.get('focal_planes', 11),
        num_transformer_layers=config.get('num_transformer_layers', 4),
        num_heads=config.get('num_heads', 8),
        d_model=config.get('d_model', 512),
        dff=config.get('dff', 1024),
        dropout_rate=config.get('dropout_rate', 0.1),
        fusion_type=config.get('fusion_type', 'average')
    )

    return model
