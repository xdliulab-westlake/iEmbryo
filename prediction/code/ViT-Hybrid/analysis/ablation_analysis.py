"""
消融分析 (Ablation Analysis)

通过移除或遮挡每个时间点，观察对预测的影响，
衡量每个时间点对模型预测的必要性

Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings


def temporal_ablation_analysis(
    model,
    images: torch.Tensor,
    device: torch.device = torch.device('cpu'),
    ablation_mode: str = 'zero',
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    时间点消融分析

    通过逐个移除（遮挡）每个时间点，计算模型预测的变化程度，
    衡量该时间点的重要性。

    Args:
        model: 训练好的ViTHybridModel
        images: (batch, 33, 11, 224, 224, 3) 胚胎图像张量
        device: 计算设备
        ablation_mode: 遮挡方式
            - 'zero': 用0替换
            - 'noise': 用高斯噪声替换
            - 'mean': 用整个数据集的平均值替换
        threshold: 二分类阈值

    Returns:
        ablation_scores: (33,) 每个时间点的消融分数（删除时的预测变化）
        predictions: (33,) 删除各时间点后的预测值

    Examples:
        >>> model.eval()
        >>> ablation_scores, _ = temporal_ablation_analysis(model, images)
        >>> # ablation_scores[i] 表示删除第i个时间点时的预测变化幅度
    """
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        # 原始预测
        original_output = model(images)  # (batch, 1)

    ablation_scores = []
    ablated_predictions = []

    # 对每个时间点进行消融
    for t in range(33):
        images_ablated = images.clone()

        if ablation_mode == 'zero':
            # 用0替换
            images_ablated[:, t, :, :, :, :] = 0

        elif ablation_mode == 'noise':
            # 用高斯噪声替换
            noise = torch.randn_like(images[:, t, :, :, :, :]) * 0.1
            images_ablated[:, t, :, :, :, :] = images[:, t, :, :, :, :] + noise

        elif ablation_mode == 'mean':
            # 用平均值替换（这里用0作为代理）
            images_ablated[:, t, :, :, :, :] = torch.mean(
                images_ablated[:, :, :, :, :, :], dim=1, keepdim=True
            )

        else:
            raise ValueError(f"未知的消融模式: {ablation_mode}")

        with torch.no_grad():
            ablated_output = model(images_ablated)  # (batch, 1)

        # 预测变化幅度
        change = (original_output - ablated_output).abs().mean()
        ablation_scores.append(change.item())
        ablated_predictions.append(ablated_output.mean().item())

    return np.array(ablation_scores), np.array(ablated_predictions)


def focal_plane_ablation_analysis(
    model,
    images: torch.Tensor,
    device: torch.device = torch.device('cpu'),
    target_timepoint: int = 16  # 中点，约104小时
) -> Tuple[np.ndarray, np.ndarray]:
    """
    焦平面消融分析

    对某个特定时间点的不同焦平面进行消融分析，
    了解空间上（焦平面维度）的重要性差异

    Args:
        model: 训练好的ViTHybridModel
        images: (batch, 33, 11, 224, 224, 3)
        device: 计算设备
        target_timepoint: 目标时间点索引 (0-32)

    Returns:
        ablation_scores: (11,) 每个焦平面的消融分数
        predictions: (11,) 删除各焦平面后的预测值
    """
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        original_output = model(images)

    ablation_scores = []
    ablated_predictions = []

    # 对该时间点的每个焦平面进行消融
    for f in range(11):
        images_ablated = images.clone()
        images_ablated[:, target_timepoint, f, :, :, :] = 0

        with torch.no_grad():
            ablated_output = model(images_ablated)

        change = (original_output - ablated_output).abs().mean()
        ablation_scores.append(change.item())
        ablated_predictions.append(ablated_output.mean().item())

    return np.array(ablation_scores), np.array(ablated_predictions)


def visualize_temporal_ablation(
    ablation_scores: np.ndarray,
    ablated_predictions: np.ndarray = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    可视化时间点消融分析结果

    Args:
        ablation_scores: (33,) 消融分数
        ablated_predictions: (33,) 消融后的预测值（可选）
        save_path: 保存路径
        figsize: 图像大小

    Examples:
        >>> ablation_scores, _ = temporal_ablation_analysis(model, images)
        >>> visualize_temporal_ablation(ablation_scores, save_path='ablation.png')
    """
    hours = np.arange(96, 112.5, 0.5)

    fig, axes = plt.subplots(1, 2 if ablated_predictions is not None else 1, figsize=figsize)

    if ablated_predictions is None:
        axes = [axes]

    # 绘制消融分数
    ax = axes[0]
    colors = plt.cm.RdYlGn_r(ablation_scores / ablation_scores.max())
    ax.bar(hours, ablation_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Developmental Time (hours)', fontsize=12)
    ax.set_ylabel('Ablation Score (Prediction Change)', fontsize=12)
    ax.set_title('Temporal Ablation Analysis: Impact of Removing Each Timepoint', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    # 标注最重要的时间点
    top_3_idx = np.argsort(ablation_scores)[-3:]
    for idx in top_3_idx:
        ax.axvline(x=hours[idx], color='red', linestyle='--', alpha=0.5, linewidth=1)

    # 绘制消融后的预测值（可选）
    if ablated_predictions is not None and len(axes) > 1:
        ax = axes[1]
        ax.plot(hours, ablated_predictions, marker='o', linestyle='-', color='steelblue',
                markersize=5, alpha=0.7, linewidth=2, label='Prediction after ablation')
        ax.set_xlabel('Ablated Time Point (hours)', fontsize=12)
        ax.set_ylabel('Model Prediction', fontsize=12)
        ax.set_title('Model Output After Ablating Each Timepoint', fontsize=13)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 消融分析图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_focal_plane_ablation(
    ablation_scores: np.ndarray,
    target_timepoint: int = 16,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    可视化焦平面消融分析结果

    Args:
        ablation_scores: (11,) 消融分数
        target_timepoint: 目标时间点（用于标题）
        save_path: 保存路径
        figsize: 图像大小
    """
    hours = np.arange(96, 112.5, 0.5)
    target_hour = hours[target_timepoint]

    fig, ax = plt.subplots(figsize=figsize)

    focal_indices = np.arange(11)
    colors = plt.cm.viridis(ablation_scores / ablation_scores.max())

    ax.bar(focal_indices, ablation_scores, color=colors, alpha=0.8,
           edgecolor='black', linewidth=1)

    ax.set_xlabel('Focal Plane Index', fontsize=12)
    ax.set_ylabel('Ablation Score', fontsize=12)
    ax.set_title(f'Focal Plane Ablation at {target_hour:.1f}h (Timepoint {target_timepoint})', fontsize=13)
    ax.set_xticks(focal_indices)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 焦平面消融分析图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def print_ablation_analysis_report(
    ablation_scores: np.ndarray,
    topk: int = 5
) -> str:
    """
    生成消融分析报告

    Args:
        ablation_scores: (33,) 消融分数
        topk: 报告中显示的top-k时间点

    Returns:
        格式化的报告字符串
    """
    top_indices = np.argsort(ablation_scores)[-topk:][::-1]
    hours = np.arange(96, 112.5, 0.5)

    report = "=" * 70 + "\n"
    report += "时间点消融分析报告\n"
    report += "=" * 70 + "\n\n"

    report += f"总时间点数: {len(ablation_scores)}\n\n"

    report += "TOP-{} 最关键的时间点 (删除时预测变化最大):\n".format(topk)
    report += "-" * 70 + "\n"
    report += "{:<5} {:<15} {:<15}".format("排名", "发育阶段(小时)", "消融分数")
    report += "\n" + "-" * 70 + "\n"

    for i, idx in enumerate(top_indices, 1):
        report += "{:<5} {:<15.1f} {:<15.4f}\n".format(i, hours[idx], ablation_scores[idx])

    report += "-" * 70 + "\n\n"

    # 统计信息
    report += "统计信息:\n"
    report += "-" * 70 + "\n"
    report += f"平均消融分数: {ablation_scores.mean():.4f}\n"
    report += f"最大消融分数: {ablation_scores.max():.4f}\n"
    report += f"最小消融分数: {ablation_scores.min():.4f}\n"
    report += f"标准差: {ablation_scores.std():.4f}\n"
    report += "-" * 70 + "\n"

    report += "=" * 70 + "\n"

    return report
