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
    n_components: int = 3,
    batch_size: int = 64
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
        n_components: PCA主成分数量
        batch_size: 批次大小
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
            num_workers=4,
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
    # 完成
    # ========================================================================
    print("\n" + "="*80)
    print("✅ 所有分析完成！")
    print("="*80)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  📊 散点图: {scatter_output}")
    print(f"  🖼️  图片序列: {sequence_output}")
    print(f"  🔍 对称图片对: {pairs_output}")
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
    parser.add_argument('--n_components', type=int, default=3,
                       help='PCA主成分数量')
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
        n_components=args.n_components
    )
