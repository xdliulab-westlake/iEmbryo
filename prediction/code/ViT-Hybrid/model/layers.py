"""
Custom Layers for ViT-Hybrid Model (PyTorch)
Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import torch.nn as nn
import math


class FocalPlaneFusion(nn.Module):
    """
    多焦平面融合模块 - 融合不同焦距的图像特征

    支持多种融合策略：平均、最大值、加权和

    Args:
        input_dim: 输入特征维度（来自VGG16，默认4096）
        output_dim: 输出特征维度（可选择投影维度，默认None表示保持原维度）
        fusion_type: 融合方式（'average', 'max', 或 'weighted'）

    输入: (batch, focal_planes, input_dim)
    输出: (batch, output_dim)
    """
    def __init__(self, input_dim=4096, output_dim=None, fusion_type='average'):
        super(FocalPlaneFusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.fusion_type = fusion_type

        # 如果需要维度投影
        if self.output_dim != self.input_dim:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, self.output_dim),
                nn.LayerNorm(self.output_dim)
            )
        else:
            self.projection = None

        # 加权融合的学习权重
        if fusion_type == 'weighted':
            self.weights = nn.Parameter(torch.ones(1, 1, 1) / 11.0)  # 11个焦平面

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, focal_planes, input_dim) 焦平面特征

        Returns:
            (batch, output_dim) 融合后的特征
        """
        if self.fusion_type == 'average':
            # 平均融合：简单直接，适合各焦平面均等重要的场景
            fused = torch.mean(x, dim=1)

        elif self.fusion_type == 'max':
            # 最大融合：选择最显著的焦平面特征
            fused = torch.max(x, dim=1)[0]

        elif self.fusion_type == 'weighted':
            # 加权融合：自适应学习各焦平面的重要性
            weights = self.weights.expand_as(x)
            fused = torch.sum(x * weights, dim=1)

        else:
            raise ValueError(f"不支持的融合方式: {self.fusion_type}")

        # 维度投影（如果需要）
        if self.projection is not None:
            fused = self.projection(fused)

        return fused


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, seq_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))

    def forward(self, x):
        return x + self.pos_encoding


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    包含: 多头自注意力 + 前馈网络 + 残差连接 + Layer Normalization
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, return_attention=False):
        """
        前向传播

        Args:
            x: (batch, seq_len, d_model)
            return_attention: 是否返回注意力权重

        Returns:
            如果return_attention=False: x (batch, seq_len, d_model)
            如果return_attention=True: (x, attn_weights) 其中attn_weights (batch, num_heads, seq_len, seq_len)
        """
        attn_output, attn_weights = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = self.layernorm2(x + ffn_output)

        if return_attention:
            return x, attn_weights
        return x
