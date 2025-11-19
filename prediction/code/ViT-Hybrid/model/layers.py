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


class TemporalAttentionPooling(nn.Module):
    """
    时间池化 - 学习每个时间步的重要性权重

    用可学习的权重对时间序列进行加权平均，让模型自动学习
    哪些时间步对最终的分类结果更重要

    Args:
        d_model: 输入特征维度

    输入: (batch, seq_len, d_model)
    输出: (batch, d_model)
    """
    def __init__(self, d_model, num_heads=8):
        super(TemporalAttentionPooling, self).__init__()
        self.d_model = d_model

        # 学习每个时间步的重要性权重
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.gelu = nn.GELU()

        # 初始化
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.constant_(self.fc2.bias, 0.0)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_weights=False):
        """
        Args:
            x: (batch, seq_len, d_model) 时间序列特征
            return_weights: 是否返回权重分布

        Returns:
            pooled: (batch, d_model) 池化后的特征
            weights: (batch, seq_len) 时间步权重分布（仅当 return_weights=True 时返回）
        """
        # 学习每个时间步的重要性权重
        # (batch, seq_len, d_model) -> (batch, seq_len, 1) -> (batch, seq_len)
        temporal_weights = self.fc1(x)  # (batch, seq_len, d_model // 2)
        temporal_weights = self.gelu(temporal_weights)
        temporal_weights = self.fc2(temporal_weights)  # (batch, seq_len, 1)
        temporal_weights = temporal_weights.squeeze(-1)  # (batch, seq_len)
        temporal_weights = torch.softmax(temporal_weights, dim=1)  # (batch, seq_len)

        # 加权求和
        pooled = torch.sum(x * temporal_weights.unsqueeze(-1), dim=1)  # (batch, d_model)
        pooled = self.norm(pooled)

        if return_weights:
            return pooled, temporal_weights
        return pooled


class TemporalMultiPooling(nn.Module):
    """
    多种时间池化策略的组合

    结合以下策略：
    - 平均池化: 全局信息
    - 最大池化: 峰值特征
    - 注意力池化: 学习的重要性
    - 最后时间步: 最新状态

    目的：让模型自适应选择不同的时间特征

    Args:
        d_model: 输入特征维度
        pooling_types: 要使用的池化策略列表
                      可选: ['mean', 'max', 'attention', 'last']

    输入: (batch, seq_len, d_model)
    输出: (batch, d_model * num_strategies) 拼接后的特征
    """
    def __init__(self, d_model, pooling_types=['mean', 'max', 'attention']):
        super(TemporalMultiPooling, self).__init__()
        self.d_model = d_model
        self.pooling_types = pooling_types

        # 初始化注意力池化（如果需要）
        if 'attention' in pooling_types:
            self.attention_pooling = TemporalAttentionPooling(d_model)

        # 初始化投影层用于融合多种策略
        # 输出维度 = d_model（无论使用多少种策略都融合回原维度）
        num_strategies = len(pooling_types)
        self.fusion_fc = nn.Sequential(
            nn.Linear(d_model * num_strategies, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) 时间序列特征

        Returns:
            fused: (batch, d_model) 融合后的特征
        """
        pooled_features = []

        for pool_type in self.pooling_types:
            if pool_type == 'mean':
                # 平均池化
                pooled = torch.mean(x, dim=1)  # (batch, d_model)

            elif pool_type == 'max':
                # 最大池化
                pooled = torch.max(x, dim=1)[0]  # (batch, d_model)

            elif pool_type == 'attention':
                # 注意力池化
                pooled = self.attention_pooling(x)  # (batch, d_model)

            elif pool_type == 'last':
                # 最后时间步
                pooled = x[:, -1, :]  # (batch, d_model)

            elif pool_type == 'first':
                # 第一个时间步
                pooled = x[:, 0, :]  # (batch, d_model)

            else:
                raise ValueError(f"不支持的池化类型: {pool_type}")

            pooled_features.append(pooled)

        # 拼接所有池化特征
        # (batch, d_model * num_strategies)
        concatenated = torch.cat(pooled_features, dim=1)

        # 通过融合层融合多种策略
        fused = self.fusion_fc(concatenated)  # (batch, d_model)

        return fused
