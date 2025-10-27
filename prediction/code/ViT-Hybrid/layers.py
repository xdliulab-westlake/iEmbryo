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
    多焦平面融合模块
    Input: (batch, focal_planes, features)
    Output: (batch, features)
    """
    def __init__(self, fusion_type='average'):
        super(FocalPlaneFusion, self).__init__()
        self.fusion_type = fusion_type

    def forward(self, x):
        if self.fusion_type == 'average':
            return torch.mean(x, dim=1)
        elif self.fusion_type == 'max':
            return torch.max(x, dim=1)[0]
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")


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

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = self.layernorm2(x + ffn_output)

        return x
