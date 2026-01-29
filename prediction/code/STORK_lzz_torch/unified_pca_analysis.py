#!/usr/bin/env python3
"""
统一的PCA分析脚本 - 确保散点图和图片序列使用相同的数据
"""

import torch
import numpy as np
from pathlib import Path
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import create_model
from config.config_manager import load_config
from visualize_three_classes import ThreeClassEmbryoDataset
from visualization.feature_space import visualize_feature_space
from analyze_pc_variations import (
    FeatureExtractorWithPaths,
    visualize_pc_sorted_sequence,
    visualize_image_pairs
)
from utils.pca_cache import PCACache, create_pca_config
from scipy.stats import gaussian_kde
from scipy.integrate import quad


def compute_transition_shape_metrics(features_pca, labels, n_components=5):
    """
    计算前n个PC中的过渡形状指标

    根据论文方法B：计算Good-Fair-Poor的连续过渡关系

    Args:
        features_pca: PCA特征 (N, n_components)
        labels: 标签 (N,) - 0: Poor, 1: Fair, 2: Good
        n_components: 要分析的主成分数量

    Returns:
        dict: 包含每个PC的过渡形状指标
    """
    print("\n" + "="*80)
    print("计算过渡形状指标 (Transition Shape Metrics)")
    print("="*80)

    class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}
    class_labels = {0: 'P', 1: 'F', 2: 'G'}

    results = {
        'pc_metrics': [],
        'best_pc': None,
        'best_b_value': -np.inf
    }

    # 对每个PC计算指标
    for pc_idx in range(min(n_components, features_pca.shape[1])):
        pc_values = features_pca[:, pc_idx]

        print(f"\n{'─'*80}")
        print(f"PC{pc_idx+1} 分析:")
        print(f"{'─'*80}")

        # 为每个类别提取PC值
        poor_pc = pc_values[labels == 0]
        fair_pc = pc_values[labels == 1]
        good_pc = pc_values[labels == 2]

        print(f"  样本数: Poor={len(poor_pc)}, Fair={len(fair_pc)}, Good={len(good_pc)}")
        print(f"  PC{pc_idx+1}范围: [{pc_values.min():.4f}, {pc_values.max():.4f}]")
        print(f"  均值: Poor={poor_pc.mean():.4f}, Fair={fair_pc.mean():.4f}, Good={good_pc.mean():.4f}")

        # 创建KDE估计器
        try:
            kde_poor = gaussian_kde(poor_pc)
            kde_fair = gaussian_kde(fair_pc)
            kde_good = gaussian_kde(good_pc)
        except Exception as e:
            print(f"  ⚠️  KDE创建失败: {e}")
            continue

        # 定义积分范围
        all_values = np.concatenate([poor_pc, fair_pc, good_pc])
        x_min, x_max = all_values.min(), all_values.max()
        x_range = x_max - x_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range

        # 计算密度重叠面积 (OVL)
        def compute_ovl(kde1, kde2, x_min, x_max):
            """计算两个KDE的重叠面积"""
            def integrand(x):
                return np.minimum(kde1(x), kde2(x))

            try:
                ovl, _ = quad(integrand, x_min, x_max, limit=100)
                return max(0, min(1, ovl))  # 限制在[0,1]范围
            except:
                return 0.0

        print(f"\n  计算密度重叠面积...")
        ovl_gf = compute_ovl(kde_good, kde_fair, x_min, x_max)
        ovl_fp = compute_ovl(kde_fair, kde_poor, x_min, x_max)
        ovl_gp = compute_ovl(kde_good, kde_poor, x_min, x_max)

        print(f"  OVL_GF (Good-Fair): {ovl_gf:.6f}")
        print(f"  OVL_FP (Fair-Poor): {ovl_fp:.6f}")
        print(f"  OVL_GP (Good-Poor): {ovl_gp:.6f}")

        # 检查过渡形状约束
        constraint_gf = ovl_gf > ovl_gp
        constraint_fp = ovl_fp > ovl_gp
        satisfies_constraint = constraint_gf and constraint_fp

        print(f"\n  过渡形状约束:")
        print(f"    OVL_GF > OVL_GP: {ovl_gf:.6f} > {ovl_gp:.6f} = {constraint_gf}")
        print(f"    OVL_FP > OVL_GP: {ovl_fp:.6f} > {ovl_gp:.6f} = {constraint_fp}")
        print(f"    满足约束: {satisfies_constraint}")

        # 计算B_k值（Fair作为Good和Poor的桥接程度）
        b_k = min(ovl_gf, ovl_fp) - ovl_gp
        print(f"\n  B_k = min(OVL_GF, OVL_FP) - OVL_GP")
        print(f"  B_k = min({ovl_gf:.6f}, {ovl_fp:.6f}) - {ovl_gp:.6f}")
        print(f"  B_k = {b_k:.6f}")

        # 保存结果
        pc_result = {
            'pc_index': pc_idx + 1,
            'ovl_gf': ovl_gf,
            'ovl_fp': ovl_fp,
            'ovl_gp': ovl_gp,
            'constraint_satisfied': satisfies_constraint,
            'b_k': b_k,
            'poor_mean': poor_pc.mean(),
            'fair_mean': fair_pc.mean(),
            'good_mean': good_pc.mean(),
            'poor_std': poor_pc.std(),
            'fair_std': fair_pc.std(),
            'good_std': good_pc.std()
        }

        results['pc_metrics'].append(pc_result)

        # 更新最佳PC
        if satisfies_constraint and b_k > results['best_b_value']:
            results['best_b_value'] = b_k
            results['best_pc'] = pc_idx + 1

    # 打印总结
    print(f"\n{'='*80}")
    print("过渡形状指标总结")
    print(f"{'='*80}")
    print(f"\n{'PC':<6} {'OVL_GF':<12} {'OVL_FP':<12} {'OVL_GP':<12} {'B_k':<12} {'约束':<8}")
    print(f"{'-'*70}")

    for metric in results['pc_metrics']:
        constraint_str = "✓" if metric['constraint_satisfied'] else "✗"
        print(f"PC{metric['pc_index']:<4} {metric['ovl_gf']:<12.6f} {metric['ovl_fp']:<12.6f} "
              f"{metric['ovl_gp']:<12.6f} {metric['b_k']:<12.6f} {constraint_str:<8}")

    print(f"\n最佳过渡形状PC: PC{results['best_pc']} (B_k = {results['best_b_value']:.6f})")
    print(f"{'='*80}\n")

    return results


def visualize_transition_shape_results(transition_results, output_dir):
    """
    可视化过渡形状指标结果

    Args:
        transition_results: compute_transition_shape_metrics的返回结果
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt
    import numpy as np

    print("\n📊 生成过渡形状指标可视化...")

    metrics = transition_results['pc_metrics']
    if not metrics:
        print("⚠️  没有可用的指标数据")
        return

    n_pcs = len(metrics)
    pc_indices = [m['pc_index'] for m in metrics]

    # 提取指标
    ovl_gf = [m['ovl_gf'] for m in metrics]
    ovl_fp = [m['ovl_fp'] for m in metrics]
    ovl_gp = [m['ovl_gp'] for m in metrics]
    b_k = [m['b_k'] for m in metrics]
    constraints = [m['constraint_satisfied'] for m in metrics]

    # ============================================================
    # 图1: OVL值对比 - 优化版
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))

    ax1 = axes[0]
    x = np.arange(n_pcs)
    width = 0.26

    # 使用更美观的配色
    color_gf = '#2ECC71'  # 绿色
    color_fp = '#F39C12'  # 橙色
    color_gp = '#E74C3C'  # 红色

    bars1 = ax1.bar(x - width, ovl_gf, width,
                    label='OVL_GF (Good-Fair)',
                    color=color_gf, alpha=0.85,
                    edgecolor='#1E8449', linewidth=1.5)
    bars2 = ax1.bar(x, ovl_fp, width,
                    label='OVL_FP (Fair-Poor)',
                    color=color_fp, alpha=0.85,
                    edgecolor='#CA6F1E', linewidth=1.5)
    bars3 = ax1.bar(x + width, ovl_gp, width,
                    label='OVL_GP (Good-Poor)',
                    color=color_gp, alpha=0.85,
                    edgecolor='#943126', linewidth=1.5)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Overlap (OVL)', fontsize=14, fontweight='bold')
    ax1.set_title('Density Overlap Between Quality Classes',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'PC{i}' for i in pc_indices], fontsize=12, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right', frameon=True,
              edgecolor='black', fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax1.set_ylim([0, max(max(ovl_gf), max(ovl_fp), max(ovl_gp)) * 1.15])
    ax1.set_axisbelow(True)

    for spine in ax1.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#2C3E50')

    # ============================================================
    # 图2: B_k值 - 优化版
    # ============================================================
    ax2 = axes[1]

    # 使用渐变色和更好的视觉效果
    colors = ['#27AE60' if c else '#E74C3C' for c in constraints]
    edge_colors = ['#1E8449' if c else '#943126' for c in constraints]

    bars = ax2.bar(x, b_k, width=0.65, color=colors, alpha=0.85,
                   edgecolor=edge_colors, linewidth=2)

    # 添加数值标签和符号
    for i, (bar, val, constraint) in enumerate(zip(bars, b_k, constraints)):
        height = bar.get_height()
        symbol = "✓" if constraint else "✗"

        # 根据正负值调整标签位置
        if height >= 0:
            va = 'bottom'
            y_pos = height + 0.005
        else:
            va = 'top'
            y_pos = height - 0.005

        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.4f}\n{symbol}',
                ha='center', va=va, fontsize=10, fontweight='bold')

    ax2.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
    ax2.set_ylabel('$B_k$ (Transition Quality Axis)', fontsize=14, fontweight='bold')
    ax2.set_title('Transition Shape Metric (Higher = Better Continuum Quality Axis)',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'PC{i}' for i in pc_indices], fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='#34495E', linestyle='-', linewidth=2, alpha=0.7)
    ax2.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)

    # 添加图例 - 英文版
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27AE60', edgecolor='#1E8449', linewidth=2,
              label='Satisfies Constraint (✓)'),
        Patch(facecolor='#E74C3C', edgecolor='#943126', linewidth=2,
              label='Violates Constraint (✗)')
    ]
    ax2.legend(handles=legend_elements, fontsize=12, loc='upper right',
              frameon=True, edgecolor='black', fancybox=True, shadow=True)

    for spine in ax2.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#2C3E50')

    plt.tight_layout()

    # 保存图表
    output_path = output_dir / 'transition_shape_metrics.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 保存: {output_path}")

    # ============================================================
    # 图3: 创建综合表格图 - 全英文版
    # ============================================================
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据 - 全英文
    table_data = [['PC', 'OVL_GF', 'OVL_FP', 'OVL_GP', 'B_k', 'Constraint', 'Evaluation']]

    for metric in metrics:
        constraint_str = "✓ Pass" if metric['constraint_satisfied'] else "✗ Fail"
        is_best = (metric['pc_index'] == transition_results['best_pc'])
        eval_str = "★ Best" if is_best else ""

        row = [
            f"PC{metric['pc_index']}",
            f"{metric['ovl_gf']:.4f}",
            f"{metric['ovl_fp']:.4f}",
            f"{metric['ovl_gp']:.4f}",
            f"{metric['b_k']:.4f}",
            constraint_str,
            eval_str
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.12, 0.12, 0.12, 0.12, 0.14, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.8)

    # 设置表头样式
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white', fontsize=13)
        cell.set_height(0.08)

    # 设置行颜色和样式
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]

            # 高亮最佳PC
            if transition_results['best_pc'] and metrics[i-1]['pc_index'] == transition_results['best_pc']:
                cell.set_facecolor('#D5F4E6')
                if j == 0:  # PC列
                    cell.set_text_props(weight='bold', color='#27AE60', fontsize=12)
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#ECF0F1')
                else:
                    cell.set_facecolor('#FFFFFF')

            # Constraint列特殊颜色
            if j == 5:
                if "Pass" in table_data[i][j]:
                    cell.set_text_props(color='#27AE60', weight='bold')
                else:
                    cell.set_text_props(color='#E74C3C', weight='bold')

            # Evaluation列特殊颜色
            if j == 6 and table_data[i][j]:
                cell.set_text_props(color='#F39C12', weight='bold', fontsize=13)

    plt.title('Transition Shape Metrics Summary Table',
              fontsize=16, fontweight='bold', pad=25)

    table_path = output_dir / 'transition_shape_table.pdf'
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 保存: {table_path}")


def visualize_quality_continuum(features_pca, labels, output_dir):
    """
    可视化胚胎质量的连续变化（质量流）

    概念: We conceptualize embryo quality as a continuous latent variable
          rather than discrete categories.

    方法: 按PC1排序所有胚胎，展示Good/Fair/Poor的密度变化

    Args:
        features_pca: PCA特征 (N, n_components)
        labels: 标签 (N,) - 0: Poor, 1: Fair, 2: Good
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    print("\n🌊 生成质量流式变化密度图...")

    class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}
    colors = {0: '#E74C3C', 1: '#F39C12', 2: '#27AE60'}

    # 按PC1排序
    pc1_values = features_pca[:, 0]
    sorted_indices = np.argsort(pc1_values)
    sorted_pc1 = pc1_values[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    # ============================================================
    # 子图1: 密度分布图 (Density Plot)
    # ============================================================
    ax1 = axes[0]

    # 为每个类别创建密度分布
    n_bins = 100
    pc1_range = (sorted_pc1.min(), sorted_pc1.max())

    for label_idx in [0, 1, 2]:
        # 提取该类别的PC1值
        class_pc1 = pc1_values[labels == label_idx]

        # 计算密度 (使用核密度估计)
        hist, bin_edges = np.histogram(class_pc1, bins=n_bins, range=pc1_range, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 平滑处理
        smoothed_hist = gaussian_filter1d(hist, sigma=2)

        # 绘制填充区域
        ax1.fill_between(bin_centers, smoothed_hist,
                         alpha=0.5,
                         color=colors[label_idx],
                         label=class_names[label_idx])

        # 绘制轮廓线
        ax1.plot(bin_centers, smoothed_hist,
                color=colors[label_idx],
                linewidth=2,
                alpha=0.8)

    ax1.set_xlabel('PC1 (Quality Continuum: Low → High)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax1.set_title('Embryo Quality as Continuous Latent Variable\nDensity Distribution along PC1',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='upper right', frameon=True, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')

    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # ============================================================
    # 子图2: 质量流图 (Quality Flow)
    # ============================================================
    ax2 = axes[1]

    # 创建颜色序列（按PC1排序后的标签）
    colors_array = np.array([colors[label] for label in sorted_labels])

    # 绘制质量流 (每个点是一个胚胎)
    positions = np.arange(len(sorted_pc1))
    ax2.scatter(sorted_pc1, np.zeros_like(sorted_pc1),
               c=colors_array,
               alpha=0.6,
               s=20,
               edgecolors='none')

    # 添加背景色带，展示质量变化趋势
    window_size = max(len(sorted_labels) // 50, 10)
    quality_trend = np.zeros(len(sorted_labels))
    for i in range(len(sorted_labels)):
        start = max(0, i - window_size // 2)
        end = min(len(sorted_labels), i + window_size // 2)
        quality_trend[i] = np.mean(sorted_labels[start:end])

    # 归一化到[-0.5, 0.5]范围
    trend_normalized = quality_trend / 2 - 0.5

    ax2.fill_between(sorted_pc1, -0.3, 0.3,
                     color='lightgray',
                     alpha=0.3)
    ax2.plot(sorted_pc1, trend_normalized * 0.5,
            color='black',
            linewidth=2,
            alpha=0.7,
            label='Quality Trend')

    ax2.set_xlabel('PC1 (Sorted Embryos: Low Quality → High Quality)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Quality\nTrend', fontsize=12, fontweight='bold', rotation=0, labelpad=40)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([])
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # 隐藏上下边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存图表
    output_path = output_dir / 'quality_continuum_density.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 保存: {output_path}")

    # ============================================================
    # 额外生成：累积分布函数图 (CDF)
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    for label_idx in [0, 1, 2]:
        class_pc1 = sorted(pc1_values[labels == label_idx])
        cdf = np.arange(1, len(class_pc1) + 1) / len(class_pc1)

        ax.plot(class_pc1, cdf,
               color=colors[label_idx],
               linewidth=3,
               label=class_names[label_idx],
               alpha=0.8)

    ax.set_xlabel('PC1 (Quality Score)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('Cumulative Distribution Function (CDF) of Embryo Quality\nShowing Quality as Continuous Variable',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout()

    output_path_cdf = output_dir / 'quality_continuum_cdf.pdf'
    plt.savefig(output_path_cdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 保存: {output_path_cdf}")

    # ============================================================
    # 统计信息
    # ============================================================
    print("\n📊 质量连续性统计:")
    print(f"  PC1范围: [{sorted_pc1.min():.4f}, {sorted_pc1.max():.4f}]")
    print(f"  PC1均值: Poor={pc1_values[labels==0].mean():.4f}, "
          f"Fair={pc1_values[labels==1].mean():.4f}, "
          f"Good={pc1_values[labels==2].mean():.4f}")
    print(f"  PC1标准差: Poor={pc1_values[labels==0].std():.4f}, "
          f"Fair={pc1_values[labels==1].std():.4f}, "
          f"Good={pc1_values[labels==2].std():.4f}")

    # 计算重叠度
    poor_pc1 = pc1_values[labels == 0]
    fair_pc1 = pc1_values[labels == 1]
    good_pc1 = pc1_values[labels == 2]

    poor_fair_overlap = (poor_pc1.max() - fair_pc1.min()) / (poor_pc1.max() - poor_pc1.min())
    fair_good_overlap = (fair_pc1.max() - good_pc1.min()) / (fair_pc1.max() - fair_pc1.min())

    print(f"  类别重叠度: Poor-Fair={poor_fair_overlap:.2%}, Fair-Good={fair_good_overlap:.2%}")
    print("  ✅ 这表明质量确实是连续变化的，而非离散分类！")


def run_unified_analysis(
    model_path: str,
    csv_path: str,
    base_dir: str,
    fair_dir: str,
    poor_csv_path: str,
    poor_base_dir: str,
    output_dir: str,
    config_path: str = 'config/default.yaml',
    model_name: str = 'vgg16',
    max_samples_per_class: int = 1000,
    max_fair_samples: int = 1000,
    random_seed: int = 42,
    use_cache: bool = True,
    cache_only: bool = False,
    n_components: int = 3,
    batch_size: int = 64,
    num_workers: int = 4
):
    """
    统一的PCA分析流程

    Args:
        model_path: 模型路径
        csv_path: Good胚胎CSV路径
        base_dir: Good胚胎基础目录
        fair_dir: Fair胚胎目录
        poor_csv_path: Poor胚胎CSV路径
        poor_base_dir: Poor胚胎基础目录
        output_dir: 输出目录
        config_path: 配置文件路径
        model_name: 模型名称 (vgg16, inceptionv3, densenet121等)
        max_samples_per_class: 每类最大样本数
        max_fair_samples: Fair类最大样本数
        random_seed: 随机种子
        use_cache: 是否使用缓存
        cache_only: 仅生成/加载PCA缓存，不生成可视化
        n_components: PCA主成分数量
        batch_size: 批次大小
        num_workers: DataLoader工作进程数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print("统一PCA分析流程")
    print(f"{'='*80}")
    print(f"🖥️  设备: {device}")
    print(f"🎲 随机种子: {random_seed}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建配置字典（用于缓存管理）
    pca_config = create_pca_config(
        model_path=model_path,
        csv_path=csv_path,
        base_dir=base_dir,
        fair_dir=fair_dir,
        poor_csv_path=poor_csv_path,
        poor_base_dir=poor_base_dir,
        max_samples_per_class=max_samples_per_class,
        max_fair_samples=max_fair_samples,
        random_seed=random_seed
    )

    # 初始化缓存管理器
    cache = PCACache(cache_dir=output_dir / 'pca_cache')

    # 尝试加载缓存
    cache_data = None
    if use_cache:
        print("\n🔍 检查PCA缓存...")
        cache_data = cache.load(config_dict=pca_config)
        if cache_only and cache_data is not None:
            print("\n✅ 已加载PCA缓存，按需退出 (cache_only=True)")
            print(f"  💾 PCA缓存: {output_dir / 'pca_cache'}")
            return

    # 如果没有缓存，提取特征并训练PCA
    if cache_data is None:
        print("\n" + "="*80)
        print("步骤1: 提取特征并训练PCA")
        print("="*80)

        # 加载配置
        config_manager = load_config(config_path)
        config = config_manager.config
        config['model']['name'] = model_name

        # 创建数据集（使用随机种子确保可重复性）
        print("\n📂 加载数据集...")
        img_size = 299 if model_name == 'inceptionv3' else 224
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        dataset = ThreeClassEmbryoDataset(
            csv_path=csv_path,
            base_dir=base_dir,
            fair_dir=fair_dir,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
            max_fair_samples=max_fair_samples,
            poor_csv_path=poor_csv_path,
            poor_base_dir=poor_base_dir,
            random_seed=random_seed  # 关键：使用固定随机种子
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 不打乱顺序
            num_workers=num_workers,
            pin_memory=True
        )

        # 加载模型
        print("\n🔧 加载模型...")
        checkpoint = torch.load(model_path, map_location=device)
        model = create_model(config['model'])

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("✅ 模型加载完成")

        # 提取特征
        print("\n🔬 提取特征...")
        extractor = FeatureExtractorWithPaths(model)
        features, labels, paths = extractor.extract_features_with_paths(dataloader, device)

        print(f"\n特征统计:")
        print(f"  特征形状: {features.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  路径数量: {len(paths)}")

        # 训练PCA
        print(f"\n📊 训练PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features)
        print(f"PCA方差解释率: {pca.explained_variance_ratio_}")
        print(f"累计方差解释率: {np.sum(pca.explained_variance_ratio_):.4f}")

        # 保存缓存
        if use_cache:
            cache.save(features, labels, paths, pca, pca_config)

        if cache_only:
            print("\n✅ PCA缓存已生成，按需退出 (cache_only=True)")
            print(f"  💾 PCA缓存: {output_dir / 'pca_cache'}")
            return

    else:
        print("\n✅ 使用缓存的PCA数据")
        features = cache_data['features']
        labels = cache_data['labels']
        paths = cache_data['paths']
        pca = cache_data['pca_model']
        features_pca = cache_data['features_pca']

    # ========================================================================
    # 步骤2: 生成散点图可视化
    # ========================================================================
    print("\n" + "="*80)
    print("步骤2: 生成PCA/t-SNE/UMAP散点图")
    print("="*80)

    scatter_output = output_dir / 'scatter_plots'
    scatter_output.mkdir(exist_ok=True)

    # 绘制散点图的辅助函数
    def plot_scatter(features_2d, labels, method, output_path, explained_variance=None):
        """绘制2D散点图"""
        import matplotlib.pyplot as plt

        class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}
        colors = {0: '#E74C3C', 1: '#F39C12', 2: '#27AE60'}

        fig, ax = plt.subplots(figsize=(10, 8))

        for label in [0, 1, 2]:
            mask = labels == label
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=colors[label],
                label=class_names[label],
                alpha=0.6,
                s=30,
                edgecolors='none'
            )

        if method.upper() == 'PCA' and explained_variance is not None:
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
        else:
            ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
            ax.set_ylabel(f'{method.upper()} 2', fontsize=12)

        ax.set_title(f'{method.upper()} Visualization of Embryo Features', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 保存: {output_path}")

    def plot_scatter_3d(features_3d, labels, method, output_path, explained_variance=None):
        """绘制3D散点图"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}
        colors = {0: '#E74C3C', 1: '#F39C12', 2: '#27AE60'}

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for label in [0, 1, 2]:
            mask = labels == label
            ax.scatter(
                features_3d[mask, 0],
                features_3d[mask, 1],
                features_3d[mask, 2],
                c=colors[label],
                label=class_names[label],
                alpha=0.6,
                s=30,
                edgecolors='none'
            )

        if method.upper() == 'PCA' and explained_variance is not None:
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
            ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)', fontsize=12)
        else:
            ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
            ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
            ax.set_zlabel(f'{method.upper()} 3', fontsize=12)

        ax.set_title(f'{method.upper()} 3D Visualization of Embryo Features', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 保存: {output_path}")

    # PCA散点图（使用已经计算好的PCA结果）
    print("\n🎨 生成PCA散点图...")
    plot_scatter(
        features_2d=features_pca[:, :2],
        labels=labels,
        method='pca',
        output_path=scatter_output / 'pca_scatter.pdf',
        explained_variance=pca.explained_variance_ratio_[:2]
    )

    # PCA 3D散点图
    if n_components >= 3:
        print("🎨 生成PCA 3D散点图...")
        plot_scatter_3d(
            features_3d=features_pca[:, :3],
            labels=labels,
            method='pca',
            output_path=scatter_output / 'pca_scatter_3d.pdf',
            explained_variance=pca.explained_variance_ratio_[:3]
        )

    # t-SNE散点图
    print("\n🎨 生成t-SNE散点图...")
    try:
        from sklearn.manifold import TSNE
        # 2D t-SNE
        tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30)
        features_tsne = tsne.fit_transform(features)
        plot_scatter(
            features_2d=features_tsne,
            labels=labels,
            method='tsne',
            output_path=scatter_output / 'tsne_scatter.pdf'
        )

        # 3D t-SNE
        print("🎨 生成t-SNE 3D散点图...")
        tsne_3d = TSNE(n_components=3, random_state=random_seed, perplexity=30)
        features_tsne_3d = tsne_3d.fit_transform(features)
        plot_scatter_3d(
            features_3d=features_tsne_3d,
            labels=labels,
            method='tsne',
            output_path=scatter_output / 'tsne_scatter_3d.pdf'
        )
    except Exception as e:
        print(f"⚠️  t-SNE失败，跳过: {e}")

    # UMAP散点图（如果安装）
    print("\n🎨 生成UMAP散点图...")
    try:
        import umap
        # 2D UMAP
        reducer = umap.UMAP(n_components=2, random_state=random_seed)
        features_umap = reducer.fit_transform(features)
        plot_scatter(
            features_2d=features_umap,
            labels=labels,
            method='umap',
            output_path=scatter_output / 'umap_scatter.pdf'
        )

        # 3D UMAP
        print("🎨 生成UMAP 3D散点图...")
        reducer_3d = umap.UMAP(n_components=3, random_state=random_seed)
        features_umap_3d = reducer_3d.fit_transform(features)
        plot_scatter_3d(
            features_3d=features_umap_3d,
            labels=labels,
            method='umap',
            output_path=scatter_output / 'umap_scatter_3d.pdf'
        )
    except Exception as e:
        print(f"⚠️  UMAP失败，跳过: {e}")

    # ========================================================================
    # 步骤3: 生成PC排序的图片序列
    # ========================================================================
    print("\n" + "="*80)
    print("步骤3: 生成PC值排序的图片序列")
    print("="*80)

    sequence_output = output_dir / 'pc_sequences'
    sequence_output.mkdir(exist_ok=True)

    # 按PC1, PC2, PC3排序
    for pc_idx in range(1, min(4, n_components + 1)):
        visualize_pc_sorted_sequence(
            features_pca=features_pca,
            labels=labels,
            paths=paths,
            output_dir=sequence_output,
            pc_component=pc_idx,
            n_images=50
        )

    # ========================================================================
    # 步骤4: 生成PC对称的图片对
    # ========================================================================
    print("\n" + "="*80)
    print("步骤4: 分析PC对称的图片对")
    print("="*80)

    pairs_output = output_dir / 'pc_pairs'
    pairs_output.mkdir(exist_ok=True)

    # 找出PC对称的图片对
    all_results = {
        'pc1_similar_pc2_opposite': [],
        'pc2_similar_pc1_opposite': []
    }

    # PC1接近，PC2极端相反（关于x轴对称）
    print("\n🔍 寻找PC1接近，PC2极端相反的图片对...")
    all_pairs = []
    for i in range(len(features_pca)):
        for j in range(i+1, len(features_pca)):
            feat1 = features_pca[i]
            feat2 = features_pca[j]

            # PC2必须符号相反
            if feat1[1] * feat2[1] < 0:
                pc1_diff = abs(feat1[0] - feat2[0])
                score = abs(feat1[1]) + abs(feat2[1]) - pc1_diff * 0.5
                all_pairs.append({
                    'idx1': i,
                    'idx2': j,
                    'score': score,
                    'pc1_1': feat1[0],
                    'pc2_1': feat1[1],
                    'pc1_2': feat2[0],
                    'pc2_2': feat2[1],
                    'pc1_diff': pc1_diff,
                    'pc2_diff': abs(feat1[1] - feat2[1])
                })

    all_pairs.sort(key=lambda x: x['score'], reverse=True)
    used_indices = set()
    class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}

    for pair in all_pairs[:100]:
        if len(all_results['pc1_similar_pc2_opposite']) >= 20:
            break
        if pair['idx1'] not in used_indices and pair['idx2'] not in used_indices:
            all_results['pc1_similar_pc2_opposite'].append({
                'class1_path': paths[pair['idx1']],
                'class2_path': paths[pair['idx2']],
                'class1_pc1': pair['pc1_1'],
                'class1_pc2': pair['pc2_1'],
                'class2_pc1': pair['pc1_2'],
                'class2_pc2': pair['pc2_2'],
                'pc1_diff': pair['pc1_diff'],
                'pc2_diff': pair['pc2_diff'],
                'score': pair['score'],
                'class1_name': class_names.get(labels[pair['idx1']], 'Unknown'),
                'class2_name': class_names.get(labels[pair['idx2']], 'Unknown')
            })
            used_indices.add(pair['idx1'])
            used_indices.add(pair['idx2'])

    # PC2接近，PC1极端相反（关于y轴对称）
    print("🔍 寻找PC2接近，PC1极端相反的图片对...")
    all_pairs = []
    for i in range(len(features_pca)):
        for j in range(i+1, len(features_pca)):
            feat1 = features_pca[i]
            feat2 = features_pca[j]

            if feat1[0] * feat2[0] < 0:
                pc2_diff = abs(feat1[1] - feat2[1])
                score = abs(feat1[0]) + abs(feat2[0]) - pc2_diff * 0.5
                all_pairs.append({
                    'idx1': i,
                    'idx2': j,
                    'score': score,
                    'pc1_1': feat1[0],
                    'pc2_1': feat1[1],
                    'pc1_2': feat2[0],
                    'pc2_2': feat2[1],
                    'pc1_diff': abs(feat1[0] - feat2[0]),
                    'pc2_diff': pc2_diff
                })

    all_pairs.sort(key=lambda x: x['score'], reverse=True)
    used_indices = set()

    for pair in all_pairs[:100]:
        if len(all_results['pc2_similar_pc1_opposite']) >= 20:
            break
        if pair['idx1'] not in used_indices and pair['idx2'] not in used_indices:
            all_results['pc2_similar_pc1_opposite'].append({
                'class1_path': paths[pair['idx1']],
                'class2_path': paths[pair['idx2']],
                'class1_pc1': pair['pc1_1'],
                'class1_pc2': pair['pc2_1'],
                'class2_pc1': pair['pc1_2'],
                'class2_pc2': pair['pc2_2'],
                'pc1_diff': pair['pc1_diff'],
                'pc2_diff': pair['pc2_diff'],
                'score': pair['score'],
                'class1_name': class_names.get(labels[pair['idx1']], 'Unknown'),
                'class2_name': class_names.get(labels[pair['idx2']], 'Unknown')
            })
            used_indices.add(pair['idx1'])
            used_indices.add(pair['idx2'])

    # 可视化图片对
    visualize_image_pairs(all_results, pairs_output, comparison_name="PC_Symmetry")

    # ========================================================================
    # 步骤5: 计算过渡形状指标
    # ========================================================================
    print("\n" + "="*80)
    print("步骤5: 计算前5个PC的过渡形状指标")
    print("="*80)

    transition_output = output_dir / 'transition_shape'
    transition_output.mkdir(exist_ok=True)

    # 计算过渡形状指标
    transition_results = compute_transition_shape_metrics(
        features_pca=features_pca,
        labels=labels,
        n_components=min(5, n_components)
    )

    # 保存结果到JSON文件
    import json
    results_file = transition_output / 'transition_metrics.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        json_results = {
            'pc_metrics': [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in metric.items()}
                for metric in transition_results['pc_metrics']
            ],
            'best_pc': transition_results['best_pc'],
            'best_b_value': float(transition_results['best_b_value'])
        }
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 过渡形状指标已保存到: {results_file}")

    # 可视化过渡形状结果
    visualize_transition_shape_results(transition_results, transition_output)

    # ========================================================================
    # 步骤6: 生成质量流式变化密度图
    # ========================================================================
    print("\n" + "="*80)
    print("步骤6: 生成质量流式变化密度图 (Quality Continuum)")
    print("="*80)

    continuum_output = output_dir / 'quality_continuum'
    continuum_output.mkdir(exist_ok=True)

    visualize_quality_continuum(
        features_pca=features_pca,
        labels=labels,
        output_dir=continuum_output
    )

    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "="*80)
    print("✅ 所有分析完成！")
    print("="*80)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  📊 散点图: {scatter_output}")
    print(f"  🖼️  图片序列: {sequence_output}")
    print(f"  🔍 对称图片对: {pairs_output}")
    print(f"  📈 过渡形状指标: {transition_output}")
    print(f"  🌊 质量流式变化: {continuum_output}")
    print(f"  💾 PCA缓存: {output_dir / 'pca_cache'}")
    print("\n💡 提示: 下次运行会自动使用缓存，确保数据一致性")
    print("="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='统一的PCA分析流程')
    parser.add_argument('--model_path', type=str,
                       default='../vgg16_best_val_accuracy.pth',
                       help='模型路径')
    parser.add_argument('--csv_path', type=str,
                       default='/Users/lzz/Desktop/IVF/good_embryos/embryos_all_focal_planes.csv',
                       help='Good胚胎CSV路径')
    parser.add_argument('--base_dir', type=str,
                       default='/Users/lzz/Desktop/IVF/good_embryos',
                       help='Good胚胎基础目录')
    parser.add_argument('--fair_dir', type=str,
                       default='/Users/lzz/Desktop/IVF/fair_embryos',
                       help='Fair胚胎目录')
    parser.add_argument('--poor_csv_path', type=str,
                       default='/Users/lzz/Desktop/IVF/poor_embryos/embryos_all_focal_planes.csv',
                       help='Poor胚胎CSV路径')
    parser.add_argument('--poor_base_dir', type=str,
                       default='/Users/lzz/Desktop/IVF/poor_embryos',
                       help='Poor胚胎基础目录')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/unified_pca_analysis',
                       help='输出目录')
    parser.add_argument('--max_samples_per_class', type=int, default=1000,
                       help='每类最大样本数')
    parser.add_argument('--max_fair_samples', type=int, default=1000,
                       help='Fair类最大样本数')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--no_cache', action='store_true',
                       help='禁用缓存，强制重新提取特征')
    parser.add_argument('--cache_only', action='store_true',
                       help='仅生成/加载PCA缓存，不生成可视化')
    parser.add_argument('--n_components', type=int, default=5,
                       help='PCA主成分数量')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader工作进程数')
    parser.add_argument('--model_name', type=str, default='vgg16',
                       help='模型名称: vgg16, inceptionv3, densenet121, etc.')

    args = parser.parse_args()

    run_unified_analysis(
        model_path=args.model_path,
        csv_path=args.csv_path,
        base_dir=args.base_dir,
        fair_dir=args.fair_dir,
        poor_csv_path=args.poor_csv_path,
        poor_base_dir=args.poor_base_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_samples_per_class=args.max_samples_per_class,
        max_fair_samples=args.max_fair_samples,
        random_seed=args.random_seed,
        use_cache=not args.no_cache,
        cache_only=args.cache_only,
        n_components=args.n_components,
        num_workers=args.num_workers
    )
