"""
ViT-Hybrid Model Architecture (PyTorch)
Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from torchvision import models
from .layers import FocalPlaneFusion, PositionalEncoding, TransformerBlock

# 本地预训练模型路径（与STORK保持一致）
PRETRAINED_PATHS = {
    'vgg16': "/liaozizhuo/scripts/STORK_lzz_torch/models/best_models/vgg16_best_val_accuracy.pth",
}


def _safe_load_state_dict(filepath):
    """
    安全地加载模型权重，处理模块导入冲突
    防止local models目录与torchvision.models的冲突
    """
    # 确保torchvision.models在sys.modules中
    from torchvision import models as tv_models
    if 'models' not in sys.modules:
        sys.modules['models'] = tv_models

    try:
        # 使用weights_only=False来支持旧的pickle格式，但注意安全性
        state_dict = torch.load(filepath, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading weights from {filepath}: {e}")
        print("Will initialize with random weights instead")
        return None

    return state_dict


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16特征提取器 - 模仿STORK结构

    输出：4096维特征向量（与STORK保持一致）
    结构：features + avgpool + flatten + classifier[:4]
    """
    def __init__(self, pretrained=True, pretrained_path=None):
        super(VGG16FeatureExtractor, self).__init__()

        # 先创建模型架构（不加载预训练权重）
        vgg16 = models.vgg16(pretrained=False)

        # 加载预训练权重
        if pretrained:
            # 优先使用指定的路径
            if pretrained_path and os.path.exists(pretrained_path):
                print(f"[VGG16] 加载本地预训练模型: {pretrained_path}")
                state_dict = _safe_load_state_dict(pretrained_path)
                if state_dict is not None:
                    vgg16.load_state_dict(state_dict, strict=False)
            # 其次使用默认路径
            elif 'vgg16' in PRETRAINED_PATHS:
                local_path = PRETRAINED_PATHS['vgg16']
                if os.path.exists(local_path):
                    print(f"[VGG16] 加载本地预训练模型: {local_path}")
                    state_dict = _safe_load_state_dict(local_path)
                    if state_dict is not None:
                        vgg16.load_state_dict(state_dict, strict=False)
                else:
                    print(f"[VGG16] 本地预训练模型不存在: {local_path}，使用在线模型")
                    vgg16 = models.vgg16(pretrained=True)
            else:
                print("[VGG16] 使用在线模型")
                vgg16 = models.vgg16(pretrained=True)

        # 提取卷积特征部分
        self.features = vgg16.features

        # 自适应平均池化到7x7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 使用原始分类器的前4层（fc1 + relu + dropout + fc2），输出4096维特征
        # 原始分类器: Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, 1000)
        self.classifier = nn.Sequential(
            vgg16.classifier[0],  # Linear(512*7*7=25088, 4096)
            vgg16.classifier[1],  # ReLU
            vgg16.classifier[2],  # Dropout
            vgg16.classifier[3]   # Linear(4096, 4096)
        )

        # 初始状态冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def unfreeze_layers(self, num_layers=None):
        """
        解冻部分VGG16层

        Args:
            num_layers: 解冻的层数，如果为None则使用STORK的最优策略（冻结前1/3，解冻后2/3）
        """
        if num_layers is None:
            # STORK的最优策略：冻结前1/3，解冻后2/3
            trainable_layers = self._collect_trainable_layers()
            total_layers = len(trainable_layers)
            freeze_ratio = 1/3
            optimal_fine_tune_at = int(total_layers * freeze_ratio)

            # 冻结前1/3的层
            for i, (name, layer) in enumerate(trainable_layers[:optimal_fine_tune_at]):
                for param in layer.parameters():
                    param.requires_grad = False

            # 解冻后2/3的层
            for i, (name, layer) in enumerate(trainable_layers[optimal_fine_tune_at:]):
                for param in layer.parameters():
                    param.requires_grad = True

            print(f"VGG16微调策略: 冻结前{optimal_fine_tune_at}/{total_layers}层，解冻后{total_layers-optimal_fine_tune_at}层")
        else:
            # 解冻最后num_layers层
            layers = list(self.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def _collect_trainable_layers(self):
        """收集所有可训练的层"""
        trainable_layers = []

        def collect_layers(module, prefix=""):
            for name, child in module.named_children():
                current_name = f"{prefix}.{name}" if prefix else name
                if len(list(child.parameters(recurse=False))) > 0:
                    trainable_layers.append((current_name, child))
                else:
                    collect_layers(child, current_name)

        collect_layers(self.features)
        return trainable_layers


class ViTHybridModel(nn.Module):
    """
    ViT-Hybrid模型: VGG16 + Transformer

    特征流：
    1. VGG16提取 4096维特征（STORK风格）
    2. 对焦平面进行融合
    3. Transformer建模时间序列
    4. 分类头输出预测

    Args:
        time_steps: 时间点数量（默认33）
        focal_planes: 焦平面数量（默认11）
        num_transformer_layers: Transformer层数（默认4）
        num_heads: 注意力头数（默认8）
        d_model: Transformer特征维度（默认4096，与VGG16输出保持一致）
        dff: 前馈网络维度（默认2048）
        dropout_rate: Dropout率（默认0.1）
        fusion_type: 焦平面融合方式（默认'average'）
    """
    def __init__(
        self,
        time_steps=33,
        focal_planes=11,
        num_transformer_layers=4,
        num_heads=8,
        d_model=4096,
        dff=2048,
        dropout_rate=0.1,
        fusion_type='average',
        pretrained_path=None,
        vgg_batch_size_per_gpu=64
    ):
        super(ViTHybridModel, self).__init__()

        self.time_steps = time_steps
        self.focal_planes = focal_planes
        self.d_model = d_model
        self.vgg16_feature_dim = 4096  # VGG16输出维度
        self.vgg_batch_size_per_gpu = vgg_batch_size_per_gpu

        # 特征提取器：VGG16（输出4096维）
        # 支持从本地加载预训练模型
        self.vgg16_extractor = VGG16FeatureExtractor(pretrained=True, pretrained_path=pretrained_path)

        # 焦平面融合层：融合11个焦平面，输出4096维特征
        self.focal_fusion = FocalPlaneFusion(
            input_dim=self.vgg16_feature_dim,
            output_dim=d_model,
            fusion_type=fusion_type
        )

        # 位置编码
        self.pos_encoding = PositionalEncoding(time_steps, d_model)

        # Transformer编码器
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_transformer_layers)
        ])

        # 分类头：输入d_model维特征，输出二分类概率
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention_weights=False):
        """
        前向传播

        输入: (batch_size, 33, 11, 3, 224, 224) - 批次 × 时间点 × 焦平面 × RGB × 高 × 宽
        返回:
            - 如果return_attention_weights=False: (batch_size, 1) 胚胎质量概率
            - 如果return_attention_weights=True: (output, attention_weights) 元组
        """
        batch_size = x.size(0)

        # 将所有图像展平为4D张量: (batch*33*11, 3, 224, 224)
        x = x.view(-1, 3, 224, 224)

        # VGG16特征提取: 分批处理避免整数溢出
        # 根据 GPU 数量和配置动态计算 vgg_batch_size
        # 原始 x shape: (batch*33*11, 3, 224, 224)
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # vgg_batch_size = 每 GPU 处理数 * GPU 数量
        vgg_batch_size = self.vgg_batch_size_per_gpu * num_gpus

        vgg_features_list = []

        for i in range(0, x.shape[0], vgg_batch_size):
            batch_x = x[i:i+vgg_batch_size]
            batch_features = self.vgg16_extractor(batch_x)
            vgg_features_list.append(batch_features)

        vgg_features = torch.cat(vgg_features_list, dim=0)

        # 重新整形为 (batch, 33, 11, 4096) - 保持原始的 batch × time × focal 维度
        vgg_features = vgg_features.view(batch_size, self.time_steps, self.focal_planes, self.vgg16_feature_dim)

        # 焦平面融合：对每个时间点的11个焦平面进行融合
        time_features = []
        for t in range(self.time_steps):
            focal_features = vgg_features[:, t, :, :]  # (batch, 11, 4096)
            fused_feature = self.focal_fusion(focal_features)  # (batch, d_model)
            time_features.append(fused_feature)

        # 堆叠为时间序列: (batch, 33, d_model)
        temporal_features = torch.stack(time_features, dim=1)

        # 添加位置编码
        temporal_features = self.pos_encoding(temporal_features)

        # Transformer编码
        x = temporal_features
        attention_weights = [] if return_attention_weights else None

        for transformer_block in self.transformer_blocks:
            if return_attention_weights:
                x, attn = transformer_block(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = transformer_block(x)

        # 全局平均池化: (batch, 33, d_model) -> (batch, d_model)
        x = torch.mean(x, dim=1)

        # 分类
        output = self.classifier(x)

        if return_attention_weights:
            return output, attention_weights
        return output

    def unfreeze_vgg16(self, num_layers=None):
        """
        解冻VGG16层进行微调

        Args:
            num_layers: 解冻的层数，None表示使用STORK最优策略（冻结前1/3，解冻后2/3）
        """
        self.vgg16_extractor.unfreeze_layers(num_layers)


def create_model(config=None):
    """
    创建模型

    Args:
        config: 配置字典，可以包含 'pretrained_path' 用于加载本地预训练模型

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
        fusion_type=config.get('fusion_type', 'average'),
        pretrained_path=config.get('pretrained_path', None),
        vgg_batch_size_per_gpu=config.get('vgg_batch_size_per_gpu', 64)
    )

    return model
