"""
时间点可解释性分析示例脚本

演示如何使用分析工具来理解模型的时间点重要性和注意力机制

使用方式：
python analyze_temporal_interpretability.py --model checkpoints/final_model.pth \\
                                           --data_root /path/to/data \\
                                           --output results/temporal_analysis

Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import argparse
import os
from pathlib import Path
import json
import numpy as np

from model.model import create_model
from config.config import get_model_config
from analysis.temporal_interpretability import (
    compute_temporal_importance,
    compute_temporal_correlation,
    get_critical_timepoints,
    print_temporal_importance_report,
)
from analysis.attention_visualization import (
    extract_attention_weights,
    visualize_attention_heatmap,
    visualize_attention_flow,
    get_most_attended_timepoints,
    print_attention_analysis_report,
)
from analysis.ablation_analysis import (
    temporal_ablation_analysis,
    visualize_temporal_ablation,
    print_ablation_analysis_report,
)


def load_sample_data(data_root, num_samples=1):
    """
    加载样本数据用于分析

    这是一个示例函数，实际使用时需要根据您的数据格式修改

    Args:
        data_root: 数据根目录
        num_samples: 加载的样本数

    Returns:
        images: (batch, 33, 11, 3, 224, 224) 张量
    """
    # 示例：使用随机数据进行演示
    print(f"⚠️  使用随机数据进行演示（实际使用时请加载真实数据）")
    images = torch.randn(num_samples, 33, 11, 3, 224, 224)
    return images


def analyze_temporal_importance(
    model,
    images,
    device,
    output_dir
):
    """
    分析时间点重要性

    Args:
        model: 训练好的模型
        images: 输入图像
        device: 计算设备
        output_dir: 输出目录
    """
    print("\n" + "=" * 70)
    print("1️⃣  时间点重要性分析（梯度归因法）")
    print("=" * 70)

    # 计算重要性
    importance_scores = compute_temporal_importance(
        model,
        images,
        device=device,
        target_class=1
    )

    # 生成报告
    report = print_temporal_importance_report(importance_scores, topk=5)
    print(report)

    # 保存结果
    results = {
        'method': 'gradient_attribution',
        'importance_scores': importance_scores.tolist(),
        'critical_timepoints': get_critical_timepoints(importance_scores, topk=5)
    }

    output_file = os.path.join(output_dir, 'temporal_importance.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存: {output_file}")

    return importance_scores


def analyze_attention_weights(
    model,
    images,
    device,
    output_dir
):
    """
    分析注意力权重

    Args:
        model: 训练好的模型
        images: 输入图像
        device: 计算设备
        output_dir: 输出目录
    """
    print("\n" + "=" * 70)
    print("2️⃣  注意力权重分析")
    print("=" * 70)

    # 提取注意力权重
    attention_weights = extract_attention_weights(model, images, device=device)

    print(f"✓ 注意力权重已提取")
    print(f"  Transformer层数: {len(attention_weights)}")
    print(f"  每层权重形状: {attention_weights[0].shape}")

    # 可视化注意力热力图
    heatmap_file = os.path.join(output_dir, 'attention_heatmap.png')
    visualize_attention_heatmap(attention_weights, save_path=heatmap_file, head_indices=[0, 1])

    # 可视化注意力流（第一个时间点）
    flow_file = os.path.join(output_dir, 'attention_flow_t0.png')
    visualize_attention_flow(attention_weights, source_timepoint=0, save_path=flow_file)

    # 可视化注意力流（中间时间点）
    flow_file_mid = os.path.join(output_dir, 'attention_flow_t16.png')
    visualize_attention_flow(attention_weights, source_timepoint=16, save_path=flow_file_mid)

    # 生成报告
    report = print_attention_analysis_report(attention_weights, topk=5)
    print(report)

    # 保存最被关注的时间点
    most_attended = get_most_attended_timepoints(attention_weights, layer_index=-1, topk=5)
    results = {
        'most_attended_timepoints': most_attended,
        'layers': len(attention_weights),
        'heads': attention_weights[0].shape[1]
    }

    output_file = os.path.join(output_dir, 'attention_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 注意力分析结果已保存: {output_file}")

    return attention_weights


def analyze_temporal_correlation(
    model,
    images,
    device,
    output_dir
):
    """
    分析时间点间的特征相关性

    Args:
        model: 训练好的模型
        images: 输入图像
        device: 计算设备
        output_dir: 输出目录
    """
    print("\n" + "=" * 70)
    print("3️⃣  时间点特征相关性分析")
    print("=" * 70)

    # 计算相关性矩阵
    correlation_matrix, info = compute_temporal_correlation(
        model,
        images,
        device=device,
        layer_index=-1
    )

    print(f"✓ 相关性矩阵已计算 ({info})")
    print(f"  矩阵形状: {correlation_matrix.shape}")
    print(f"  平均相关性: {correlation_matrix.mean():.4f}")

    # 可视化
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        ax=ax,
        cmap='coolwarm',
        center=0.5,
        square=True,
        cbar_kws={'label': 'Feature Correlation'},
        xticklabels=range(0, 33, 4),
        yticklabels=range(0, 33, 4)
    )
    ax.set_xlabel('Time Point 2 (Index)', fontsize=12)
    ax.set_ylabel('Time Point 1 (Index)', fontsize=12)
    ax.set_title('Temporal Feature Correlation Matrix (Transformer Output)', fontsize=13)

    plt.tight_layout()
    corr_file = os.path.join(output_dir, 'temporal_correlation.png')
    plt.savefig(corr_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 相关性矩阵已保存: {corr_file}")
    plt.close()

    # 保存数值
    np.save(os.path.join(output_dir, 'correlation_matrix.npy'), correlation_matrix)


def analyze_ablation(
    model,
    images,
    device,
    output_dir
):
    """
    执行消融分析

    Args:
        model: 训练好的模型
        images: 输入图像
        device: 计算设备
        output_dir: 输出目录
    """
    print("\n" + "=" * 70)
    print("4️⃣  时间点消融分析")
    print("=" * 70)

    # 执行消融
    ablation_scores, ablated_preds = temporal_ablation_analysis(
        model,
        images,
        device=device,
        ablation_mode='zero'
    )

    print(f"✓ 消融分析完成")
    print(f"  消融分数形状: {ablation_scores.shape}")
    print(f"  平均消融分数: {ablation_scores.mean():.4f}")

    # 可视化
    ablation_file = os.path.join(output_dir, 'temporal_ablation.png')
    visualize_temporal_ablation(ablation_scores, ablated_preds, save_path=ablation_file)

    # 生成报告
    report = print_ablation_analysis_report(ablation_scores, topk=5)
    print(report)

    # 保存结果
    results = {
        'method': 'temporal_ablation',
        'ablation_mode': 'zero',
        'ablation_scores': ablation_scores.tolist(),
        'ablated_predictions': ablated_preds.tolist()
    }

    output_file = os.path.join(output_dir, 'ablation_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 消融分析结果已保存: {output_file}")

    return ablation_scores, ablated_preds


def main():
    parser = argparse.ArgumentParser(
        description='时间点可解释性分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python analyze_temporal_interpretability.py \\
    --model checkpoints/final_model.pth \\
    --output results/temporal_analysis
    """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/final_model.pth',
        help='模型权重路径'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='数据根目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/temporal_analysis',
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='计算设备'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='样本数量'
    )

    args = parser.parse_args()

    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"使用设备: {device}")

    # 创建输出目录
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n加载模型: {args.model}")
    config = get_model_config()
    model = create_model(config)

    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print("✓ 模型加载成功")
    else:
        print(f"⚠️  模型文件不存在，使用未训练的模型进行演示")

    model = model.to(device)
    model.eval()

    # 加载数据
    print(f"\n加载数据...")
    images = load_sample_data(args.data_root, num_samples=args.batch_size)
    images = images.to(device)

    # 执行分析
    print("\n" + "=" * 70)
    print("开始时间点可解释性分析")
    print("=" * 70)

    # 1. 时间点重要性分析
    importance_scores = analyze_temporal_importance(model, images, device, args.output)

    # 2. 注意力权重分析
    attention_weights = analyze_attention_weights(model, images, device, args.output)

    # 3. 时间点相关性分析
    analyze_temporal_correlation(model, images, device, args.output)

    # 4. 消融分析
    ablation_scores, _ = analyze_ablation(model, images, device, args.output)

    # 生成总结
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n输出文件已保存至: {args.output}")
    print("\n生成的文件：")
    print("  📊 temporal_importance.json - 时间点重要性分析结果")
    print("  📊 attention_analysis.json - 注意力权重分析结果")
    print("  📊 ablation_analysis.json - 消融分析结果")
    print("  🖼️  attention_heatmap.png - 注意力热力图")
    print("  🖼️  attention_flow_t0.png - 第0个时间点的注意力流")
    print("  🖼️  attention_flow_t16.png - 第16个时间点的注意力流")
    print("  🖼️  temporal_correlation.png - 时间点相关性矩阵")
    print("  🖼️  temporal_ablation.png - 时间点消融分析图")
    print("  📈 correlation_matrix.npy - 相关性矩阵数值")


if __name__ == '__main__':
    main()
