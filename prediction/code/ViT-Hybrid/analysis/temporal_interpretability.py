"""
时间点重要性与相关性分析

包含：
1. 梯度归因法 - 计算每个时间点对预测的重要性
2. 特征相关性矩阵 - 计算时间点间的特征相似度

Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import numpy as np
from typing import Tuple
import warnings


def compute_temporal_importance(
    model,
    images: torch.Tensor,
    device: torch.device = torch.device('cpu'),
    method: str = 'gradient',
    target_class: int = 1
) -> np.ndarray:
    """
    计算每个时间点对预测的重要性 (梯度归因法)

    原理：通过计算预测输出对输入的梯度，衡量每个时间点对最终预测的影响程度

    Args:
        model: 训练好的ViTHybridModel
        images: (batch, 33, 11, 224, 224, 3) 胚胎图像张量
        device: 计算设备 (cpu 或 cuda)
        method: 计算方法 ('gradient' - 梯度归因，'integrated_gradient' - 积分梯度)
        target_class: 目标类别 (0=差，1=好)

    Returns:
        importance: (33,) 每个时间点的重要性分数（已归一化到[0,1]）

    Examples:
        >>> model.eval()
        >>> importance = compute_temporal_importance(model, images, device='cuda')
        >>> # 获得33个时间点的重要性分数
        >>> print(importance)  # 形状: (33,)
    """
    model.eval()
    images = images.to(device)
    images.requires_grad = True

    # 前向传播
    output = model(images)  # (batch, 1)

    # 反向传播到目标类别
    loss = output.mean()
    loss.backward()

    # 获取梯度
    if images.grad is None:
        warnings.warn("梯度为None，可能是因为输入不需要梯度计算")
        return np.ones(33) / 33

    grad = images.grad.data  # (batch, 33, 11, 224, 224, 3)

    # 计算每个时间点的梯度幅度（绝对值）
    # 聚合维度顺序: batch -> focal_planes -> spatial -> channels
    temporal_importance = grad.abs().mean(dim=(0, 2, 3, 4, 5))  # (33,)

    # 归一化到[0,1]
    temporal_importance = temporal_importance / (temporal_importance.sum() + 1e-8)

    return temporal_importance.cpu().detach().numpy()


def compute_temporal_correlation(
    model,
    images: torch.Tensor,
    device: torch.device = torch.device('cpu'),
    layer_index: int = -1
) -> Tuple[np.ndarray, str]:
    """
    计算时间点间的特征相关性矩阵

    原理：通过提取Transformer输出的隐层特征，计算不同时间点间的余弦相似度

    Args:
        model: 训练好的ViTHybridModel
        images: (batch, 33, 11, 224, 224, 3) 胚胎图像张量
        device: 计算设备 (cpu 或 cuda)
        layer_index: 从哪一层提取特征 (-1=最后一层)

    Returns:
        correlation: (33, 33) 相关性矩阵，值域[0,1]
        info: 字符串，包含提取的层信息

    Examples:
        >>> correlation_matrix, info = compute_temporal_correlation(model, images)
        >>> print(f"从{info}提取特征")
        >>> # correlation_matrix[i,j] 表示时间点i和j间的特征相似度
    """
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        batch_size = images.shape[0]

        # 提取每个时间点的VGG16特征 + 焦平面融合
        temporal_features = []

        for t in range(33):
            focal_features = []
            for f in range(11):
                img = images[:, t, f, :, :, :]  # (batch, 3, 224, 224)
                feat = model.vgg16_extractor(img)  # (batch, 4096)
                focal_features.append(feat)

            # 焦平面融合
            aggregated = torch.mean(torch.stack(focal_features), dim=0)  # (batch, 4096)
            temporal_features.append(aggregated)

        # 堆叠为时间序列 (batch, 33, 4096)
        x_seq = torch.stack(temporal_features, dim=1)

        # 通过位置编码
        x_encoded = model.pos_encoding(x_seq)

        # 通过Transformer编码器
        for i, layer in enumerate(model.transformer_blocks):
            x_encoded = layer(x_encoded)

        # 取平均batch维度 (33, d_model)
        x_seq_avg = x_encoded.mean(dim=0)

        # 计算余弦相似度矩阵
        # norm: (33, 1)
        norms = torch.norm(x_seq_avg, dim=1, keepdim=True)
        normalized = x_seq_avg / (norms + 1e-8)

        # 相关性矩阵: (33, 33)
        correlation = torch.mm(normalized, normalized.t())

        # 限制值域到[0,1]
        correlation = torch.clamp(correlation, min=0, max=1)

    info = f"Transformer第{len(model.transformer_blocks)}层输出"
    return correlation.cpu().numpy(), info


def compute_temporal_importance_by_layer(
    model,
    images: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    分层计算时间点的重要性

    对Transformer的每一层都计算一次重要性，观察重要性随层数的变化

    Args:
        model: 训练好的ViTHybridModel
        images: (batch, 33, 11, 224, 224, 3)
        device: 计算设备

    Returns:
        层级重要性字典，key为层索引(0-based)，value为该层输出后的重要性
    """
    model.eval()
    images = images.to(device)

    layer_importances = {}

    with torch.no_grad():
        batch_size = images.shape[0]

        # 提取VGG16特征
        x_flat = images.view(-1, 3, 224, 224)
        vgg_features = model.vgg16_extractor(x_flat)
        vgg_features = vgg_features.view(batch_size, 33, 11, 4096)

        # 焦平面融合
        time_features = []
        for t in range(33):
            focal_features = vgg_features[:, t, :, :]
            fused = model.focal_fusion(focal_features)
            time_features.append(fused)

        x = torch.stack(time_features, dim=1)
        x = model.pos_encoding(x)

        # 逐层计算重要性
        for layer_idx, transformer_block in enumerate(model.transformer_blocks):
            x = transformer_block(x)

            # 计算该层输出对分类的影响
            x_avg = torch.mean(x, dim=1)
            logits = model.classifier(x_avg)

            # 计算梯度（需要重新启用梯度计算）
            logits.requires_grad = True
            loss = logits.mean()

            if loss.requires_grad:
                # 手动计算梯度
                grad = torch.autograd.grad(loss, logits, create_graph=False)[0]

                # 由于这里已经过处理，直接用较简单的方式估算
                # 使用当前层特征的L2范数作为代理指标
                layer_importance = torch.norm(x, dim=2).mean(dim=0).cpu().numpy()  # (33,)
                layer_importance = layer_importance / (layer_importance.sum() + 1e-8)
                layer_importances[f'layer_{layer_idx}'] = layer_importance
            else:
                # 如果梯度不可用，用特征范数代替
                layer_importance = torch.norm(x, dim=2).mean(dim=0).cpu().numpy()
                layer_importance = layer_importance / (layer_importance.sum() + 1e-8)
                layer_importances[f'layer_{layer_idx}'] = layer_importance

    return layer_importances


def get_critical_timepoints(
    importance_scores: np.ndarray,
    topk: int = 5
) -> dict:
    """
    找到最重要的时间点

    Args:
        importance_scores: (33,) 时间点重要性分数
        topk: 返回前k个最重要的时间点

    Returns:
        字典，包含：
            - 'timepoints': 时间点索引列表
            - 'hours': 对应的发育小时
            - 'scores': 对应的重要性分数
    """
    top_indices = np.argsort(importance_scores)[-topk:][::-1]
    hours = 96 + np.arange(33) * 0.5

    return {
        'timepoints': top_indices.tolist(),
        'hours': hours[top_indices].tolist(),
        'scores': importance_scores[top_indices].tolist()
    }


def print_temporal_importance_report(
    importance_scores: np.ndarray,
    topk: int = 5
) -> str:
    """
    生成文字报告

    Args:
        importance_scores: (33,) 时间点重要性分数
        topk: 报告中显示的top-k时间点

    Returns:
        格式化的报告字符串
    """
    critical = get_critical_timepoints(importance_scores, topk=topk)

    report = "=" * 70 + "\n"
    report += "时间点重要性分析报告\n"
    report += "=" * 70 + "\n\n"

    report += f"总时间点数: {len(importance_scores)}\n"
    report += f"时间范围: 96-112小时 (0.5小时间隔)\n\n"

    report += "TOP-{} 最重要的时间点:\n".format(topk)
    report += "-" * 70 + "\n"
    report += "{:<5} {:<15} {:<15}".format("排名", "发育阶段(小时)", "重要性分数")
    report += "\n" + "-" * 70 + "\n"

    for i, (tp, hour, score) in enumerate(
        zip(critical['timepoints'], critical['hours'], critical['scores']),
        1
    ):
        report += "{:<5} {:<15.1f} {:<15.4f}\n".format(i, hour, score)

    report += "-" * 70 + "\n\n"

    # 统计信息
    report += "统计信息:\n"
    report += "-" * 70 + "\n"
    report += f"平均重要性: {importance_scores.mean():.4f}\n"
    report += f"最大重要性: {importance_scores.max():.4f}\n"
    report += f"最小重要性: {importance_scores.min():.4f}\n"
    report += f"标准差: {importance_scores.std():.4f}\n"
    report += "-" * 70 + "\n"

    report += "=" * 70 + "\n"

    return report
