"""
Attention权重提取与可视化

包含：
1. 注意力权重提取
2. 注意力热力图可视化
3. 注意力流分析

Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import warnings


def extract_attention_weights(
    model,
    images: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> List[torch.Tensor]:
    """
    提取Transformer各层的注意力权重

    Args:
        model: 训练好的ViTHybridModel
        images: (batch, 33, 11, 224, 224, 3) 胚胎图像张量
        device: 计算设备

    Returns:
        attention_weights: 列表，长度为Transformer层数
            每个元素形状为 (batch, num_heads, seq_len=33, seq_len=33)

    Examples:
        >>> model.eval()
        >>> attn_weights = extract_attention_weights(model, images)
        >>> print(len(attn_weights))  # 4 (4层Transformer)
        >>> print(attn_weights[0].shape)  # (batch, 8, 33, 33)
    """
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        output, attention_weights = model(images, return_attention_weights=True)

    return attention_weights


def visualize_attention_heatmap(
    attention_weights: List[torch.Tensor],
    save_path: str = None,
    head_indices: List[int] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    可视化注意力权重热力图

    显示每层Transformer的注意力矩阵，帮助理解模型如何关注不同的时间点

    Args:
        attention_weights: 从extract_attention_weights返回的注意力权重列表
        save_path: 保存图像的路径，如果为None则显示
        head_indices: 可视化的注意力头索引列表，默认为[0]（第一个头）
        figsize: 图像大小

    Examples:
        >>> attn_weights = extract_attention_weights(model, images)
        >>> visualize_attention_heatmap(attn_weights, save_path='attention.png')
    """
    if head_indices is None:
        head_indices = [0]

    num_layers = len(attention_weights)
    num_heads = len(head_indices)

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=figsize,
        squeeze=False
    )

    hours = np.arange(96, 112.5, 0.5)  # 96-112小时

    for layer_idx, attn in enumerate(attention_weights):
        # attn shape: (batch, num_heads, seq_len, seq_len)
        # 取第一个batch样本
        attn_batch0 = attn[0].cpu().numpy()  # (num_heads, 33, 33)

        for head_pos, head_idx in enumerate(head_indices):
            ax = axes[layer_idx, head_pos]

            attn_matrix = attn_batch0[head_idx]  # (33, 33)

            # 绘制热力图
            sns.heatmap(
                attn_matrix,
                ax=ax,
                cmap='viridis',
                cbar=True if head_pos == num_heads - 1 else False,
                square=True,
                xticklabels=False,
                yticklabels=False,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Attention Weight'}
            )

            if head_pos == 0:
                ax.set_ylabel(f'Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
            if layer_idx == 0:
                ax.set_title(f'Head {head_idx}', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 注意力热力图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def compute_attention_flow(
    attention_weights: List[torch.Tensor],
    source_timepoint: int = 0,
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    计算从某个时间点出发的注意力流

    分析模型如何从一个时间点的信息流向其他时间点

    Args:
        attention_weights: 注意力权重列表
        source_timepoint: 源时间点索引 (0-32)
        aggregation: 聚合方式 ('mean' - 平均所有头, 'max' - 最大值)

    Returns:
        attention_flow: (num_layers, 33) 每层中源时间点对其他时间点的注意力
    """
    num_layers = len(attention_weights)
    attention_flow = np.zeros((num_layers, 33))

    for layer_idx, attn in enumerate(attention_weights):
        # attn: (batch, num_heads, 33, 33)
        attn_batch0 = attn[0].cpu().numpy()  # (num_heads, 33, 33)

        # 获取源时间点的注意力分布
        source_attention = attn_batch0[:, source_timepoint, :]  # (num_heads, 33)

        if aggregation == 'mean':
            attention_flow[layer_idx] = source_attention.mean(axis=0)
        elif aggregation == 'max':
            attention_flow[layer_idx] = source_attention.max(axis=0)
        else:
            raise ValueError(f"未知的聚合方式: {aggregation}")

    return attention_flow


def visualize_attention_flow(
    attention_weights: List[torch.Tensor],
    source_timepoint: int = 0,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    可视化注意力流

    显示源时间点的信息如何在各Transformer层中流向其他时间点

    Args:
        attention_weights: 注意力权重列表
        source_timepoint: 源时间点索引
        save_path: 保存路径
        figsize: 图像大小

    Examples:
        >>> attn_weights = extract_attention_weights(model, images)
        >>> # 可视化96小时(第0个时间点)的注意力流
        >>> visualize_attention_flow(attn_weights, source_timepoint=0)
    """
    attention_flow = compute_attention_flow(
        attention_weights,
        source_timepoint=source_timepoint,
        aggregation='mean'
    )

    hours = np.arange(96, 112.5, 0.5)
    source_hour = hours[source_timepoint]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for layer_idx in range(len(attention_weights)):
        ax = axes[layer_idx]

        flow = attention_flow[layer_idx]

        # 颜色编码：源时间点为红色，其他为蓝色
        colors = ['red' if i == source_timepoint else 'steelblue' for i in range(33)]

        ax.bar(hours, flow, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'Layer {layer_idx + 1} Attention Flow from {source_hour:.1f}h', fontsize=11)
        ax.set_xlabel('Target Time Point (hours)', fontsize=10)
        ax.set_ylabel('Attention Weight', fontsize=10)
        ax.set_ylim(0, max(flow) * 1.2)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 注意力流图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def get_most_attended_timepoints(
    attention_weights: List[torch.Tensor],
    layer_index: int = -1,
    aggregation: str = 'mean',
    topk: int = 5
) -> dict:
    """
    找到最被关注的时间点

    Args:
        attention_weights: 注意力权重列表
        layer_index: 分析的层索引 (-1表示最后一层)
        aggregation: 聚合方式
        topk: 返回前k个

    Returns:
        字典包含最被关注的时间点信息
    """
    if layer_index < 0:
        layer_index = len(attention_weights) + layer_index

    attn = attention_weights[layer_index]
    attn_batch0 = attn[0].cpu().numpy()  # (num_heads, 33, 33)

    if aggregation == 'mean':
        # 计算每个时间点被其他时间点关注的平均权重
        attention_received = attn_batch0.mean(axis=0).mean(axis=0)  # (33,)
    elif aggregation == 'max':
        attention_received = attn_batch0.max(axis=0).max(axis=0)
    else:
        raise ValueError(f"未知的聚合方式: {aggregation}")

    # 排序
    top_indices = np.argsort(attention_received)[-topk:][::-1]
    hours = np.arange(96, 112.5, 0.5)

    return {
        'timepoints': top_indices.tolist(),
        'hours': hours[top_indices].tolist(),
        'attention_scores': attention_received[top_indices].tolist()
    }


def print_attention_analysis_report(
    attention_weights: List[torch.Tensor],
    topk: int = 5
) -> str:
    """
    生成注意力分析报告

    Args:
        attention_weights: 注意力权重列表
        topk: 报告中显示的top-k时间点

    Returns:
        格式化的报告字符串
    """
    report = "=" * 70 + "\n"
    report += "注意力权重分析报告\n"
    report += "=" * 70 + "\n\n"

    report += f"Transformer层数: {len(attention_weights)}\n\n"

    for layer_idx, attn in enumerate(attention_weights):
        report += f"第 {layer_idx + 1} 层:\n"
        report += "-" * 70 + "\n"

        attn_batch0 = attn[0].cpu().numpy()  # (num_heads, 33, 33)
        num_heads = attn_batch0.shape[0]

        report += f"  注意力头数: {num_heads}\n"

        # 每个位置被关注的平均权重
        attention_received = attn_batch0.mean(axis=0).mean(axis=0)
        top_indices = np.argsort(attention_received)[-topk:][::-1]
        hours = np.arange(96, 112.5, 0.5)

        report += f"  最受关注的{topk}个时间点:\n"
        for i, idx in enumerate(top_indices, 1):
            report += f"    {i}. {hours[idx]:.1f}h (注意力得分: {attention_received[idx]:.4f})\n"

        report += "\n"

    report += "=" * 70 + "\n"

    return report
