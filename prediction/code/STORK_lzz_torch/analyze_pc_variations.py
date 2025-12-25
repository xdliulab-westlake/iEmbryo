#!/usr/bin/env python3
"""
分析主成分变化对应的图片特征差异
找出PC1接近PC2相差大，或PC2接近PC1相差大的Good和Poor胚胎图片
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import create_model
from config.config_manager import load_config
from visualize_three_classes import ThreeClassEmbryoDataset


class FeatureExtractorWithPaths:
    """带路径记录的特征提取器"""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.features = []
        self._register_hook()

    def _register_hook(self):
        """注册hook"""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                feat = output.detach()
                if len(feat.shape) > 2:
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                    feat = feat.view(feat.size(0), -1)
                self.features.append(feat.cpu())

        # VGG16: 使用avgpool层
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'avgpool'):
            self.model.model.avgpool.register_forward_hook(hook_fn)
            print("Hook注册在: model.avgpool (VGG16)")

    def extract_features_with_paths(self, dataloader, device):
        """提取特征并记录路径"""
        all_features = []
        all_labels = []
        all_paths = []

        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="提取特征")):
                images = images.to(device)

                # 清空特征
                self.features = []

                # 前向传播
                _ = self.model(images)

                # 收集特征
                if len(self.features) > 0:
                    batch_features = self.features[0].numpy()
                    all_features.append(batch_features)
                    all_labels.append(labels.cpu().numpy())

                    # 记录图片路径
                    batch_start = batch_idx * dataloader.batch_size
                    batch_end = batch_start + len(images)
                    batch_paths = [dataloader.dataset.image_paths[i]
                                  for i in range(batch_start, min(batch_end, len(dataloader.dataset)))]
                    all_paths.extend(batch_paths)

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        return features, labels, all_paths


def find_pc_variation_pairs(features, labels, paths, n_pairs=10):
    """
    找出PC变化对应的图片对 - 寻找极端对称的情况
    使用贪心匹配算法，确保每个胚胎只配对一次

    Args:
        features: PCA降维后的特征 (N, 2)
        labels: 标签 (N,)
        paths: 图片路径列表
        n_pairs: 每种情况找多少对

    Returns:
        dict: 包含不同情况的图片对
    """
    # 分离Good和Poor
    good_mask = labels == 2
    poor_mask = labels == 0

    good_features = features[good_mask]
    poor_features = features[poor_mask]
    good_paths = [paths[i] for i in range(len(paths)) if good_mask[i]]
    poor_paths = [paths[i] for i in range(len(paths)) if poor_mask[i]]

    print(f"\nGood胚胎数量: {len(good_paths)}")
    print(f"Poor胚胎数量: {len(poor_paths)}")
    print(f"Good PC1范围: [{good_features[:, 0].min():.2f}, {good_features[:, 0].max():.2f}]")
    print(f"Good PC2范围: [{good_features[:, 1].min():.2f}, {good_features[:, 1].max():.2f}]")
    print(f"Poor PC1范围: [{poor_features[:, 0].min():.2f}, {poor_features[:, 0].max():.2f}]")
    print(f"Poor PC2范围: [{poor_features[:, 1].min():.2f}, {poor_features[:, 1].max():.2f}]")

    results = {
        'pc1_similar_pc2_opposite': [],  # PC1接近，PC2一正一负且绝对值都大（关于x轴对称）
        'pc2_similar_pc1_opposite': []   # PC2接近，PC1一正一负且绝对值都大（关于y轴对称）
    }

    # 1. 找PC1接近，PC2极端相反的图片对（关于x轴对称）
    print("\n寻找PC1接近，PC2极端相反的图片对（关于x轴对称）...")

    # 计算所有可能的配对得分
    all_pairs = []
    for i, g_feat in enumerate(good_features):
        pc1_diff = np.abs(poor_features[:, 0] - g_feat[0])

        for j, p_feat in enumerate(poor_features):
            # PC2必须符号相反
            if g_feat[1] * p_feat[1] < 0:
                # 得分 = 两者绝对值之和（都大更好） - PC1差异（差异小更好）
                score = abs(g_feat[1]) + abs(p_feat[1]) - pc1_diff[j] * 0.5
                all_pairs.append({
                    'good_idx': i,
                    'poor_idx': j,
                    'score': score,
                    'good_pc1': g_feat[0],
                    'good_pc2': g_feat[1],
                    'poor_pc1': p_feat[0],
                    'poor_pc2': p_feat[1],
                    'pc1_diff': pc1_diff[j],
                    'pc2_diff': abs(g_feat[1] - p_feat[1])
                })

    # 按得分排序
    all_pairs.sort(key=lambda x: x['score'], reverse=True)

    # 贪心选择：每次选择得分最高且未使用的配对
    used_good = set()
    used_poor = set()
    for pair in all_pairs:
        if len(results['pc1_similar_pc2_opposite']) >= n_pairs:
            break

        if pair['good_idx'] not in used_good and pair['poor_idx'] not in used_poor:
            results['pc1_similar_pc2_opposite'].append({
                'good_path': good_paths[pair['good_idx']],
                'poor_path': poor_paths[pair['poor_idx']],
                'good_pc1': pair['good_pc1'],
                'good_pc2': pair['good_pc2'],
                'poor_pc1': pair['poor_pc1'],
                'poor_pc2': pair['poor_pc2'],
                'pc1_diff': pair['pc1_diff'],
                'pc2_diff': pair['pc2_diff'],
                'score': pair['score']
            })
            used_good.add(pair['good_idx'])
            used_poor.add(pair['poor_idx'])

    # 2. 找PC2接近，PC1极端相反的图片对（关于y轴对称）
    print("寻找PC2接近，PC1极端相反的图片对（关于y轴对称）...")

    all_pairs = []
    for i, g_feat in enumerate(good_features):
        pc2_diff = np.abs(poor_features[:, 1] - g_feat[1])

        for j, p_feat in enumerate(poor_features):
            # PC1必须符号相反
            if g_feat[0] * p_feat[0] < 0:
                # 得分 = 两者绝对值之和 - PC2差异
                score = abs(g_feat[0]) + abs(p_feat[0]) - pc2_diff[j] * 0.5
                all_pairs.append({
                    'good_idx': i,
                    'poor_idx': j,
                    'score': score,
                    'good_pc1': g_feat[0],
                    'good_pc2': g_feat[1],
                    'poor_pc1': p_feat[0],
                    'poor_pc2': p_feat[1],
                    'pc1_diff': abs(g_feat[0] - p_feat[0]),
                    'pc2_diff': pc2_diff[j]
                })

    # 按得分排序
    all_pairs.sort(key=lambda x: x['score'], reverse=True)

    # 贪心选择
    used_good = set()
    used_poor = set()
    for pair in all_pairs:
        if len(results['pc2_similar_pc1_opposite']) >= n_pairs:
            break

        if pair['good_idx'] not in used_good and pair['poor_idx'] not in used_poor:
            results['pc2_similar_pc1_opposite'].append({
                'good_path': good_paths[pair['good_idx']],
                'poor_path': poor_paths[pair['poor_idx']],
                'good_pc1': pair['good_pc1'],
                'good_pc2': pair['good_pc2'],
                'poor_pc1': pair['poor_pc1'],
                'poor_pc2': pair['poor_pc2'],
                'pc1_diff': pair['pc1_diff'],
                'pc2_diff': pair['pc2_diff'],
                'score': pair['score']
            })
            used_good.add(pair['good_idx'])
            used_poor.add(pair['poor_idx'])

    return results


def visualize_image_pairs(results, output_dir):
    """可视化图片对"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. PC1接近，PC2极端相反（关于x轴对称）
    print("\n生成PC1接近PC2极端相反的图片对（关于x轴对称）...")
    n_pairs = len(results['pc1_similar_pc2_opposite'])
    fig, axes = plt.subplots(n_pairs, 2, figsize=(8, n_pairs * 4))
    fig.suptitle('PC1 Similar, PC2 Opposite (Symmetric about X-axis)\nGood vs Poor', fontsize=16, y=0.998)

    for i, pair in enumerate(results['pc1_similar_pc2_opposite']):
        # Good图片
        good_img = Image.open(pair['good_path']).convert('RGB')
        axes[i, 0].imshow(good_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(
            f"#{i+1} Good\nPC1:{pair['good_pc1']:.2f}, PC2:{pair['good_pc2']:.2f}",
            fontsize=9, color='green'
        )

        # Poor图片
        poor_img = Image.open(pair['poor_path']).convert('RGB')
        axes[i, 1].imshow(poor_img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(
            f"#{i+1} Poor\nPC1:{pair['poor_pc1']:.2f}, PC2:{pair['poor_pc2']:.2f}\n"
            f"ΔPC1:{pair['pc1_diff']:.3f}, ΔPC2:{pair['pc2_diff']:.3f}",
            fontsize=9, color='red'
        )

    plt.tight_layout()
    save_path = output_dir / 'pc1_similar_pc2_opposite_pairs.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {save_path} ({n_pairs}对)")

    # 2. PC2接近，PC1极端相反（关于y轴对称）
    print("生成PC2接近PC1极端相反的图片对（关于y轴对称）...")
    n_pairs = len(results['pc2_similar_pc1_opposite'])
    fig, axes = plt.subplots(n_pairs, 2, figsize=(8, n_pairs * 4))
    fig.suptitle('PC2 Similar, PC1 Opposite (Symmetric about Y-axis)\nGood vs Poor', fontsize=16, y=0.998)

    for i, pair in enumerate(results['pc2_similar_pc1_opposite']):
        # Good图片
        good_img = Image.open(pair['good_path']).convert('RGB')
        axes[i, 0].imshow(good_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(
            f"#{i+1} Good\nPC1:{pair['good_pc1']:.2f}, PC2:{pair['good_pc2']:.2f}",
            fontsize=9, color='green'
        )

        # Poor图片
        poor_img = Image.open(pair['poor_path']).convert('RGB')
        axes[i, 1].imshow(poor_img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(
            f"#{i+1} Poor\nPC1:{pair['poor_pc1']:.2f}, PC2:{pair['poor_pc2']:.2f}\n"
            f"ΔPC1:{pair['pc1_diff']:.3f}, ΔPC2:{pair['pc2_diff']:.3f}",
            fontsize=9, color='red'
        )

    plt.tight_layout()
    save_path = output_dir / 'pc2_similar_pc1_opposite_pairs.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {save_path} ({n_pairs}对)")


def main():
    """主函数"""
    print("="*80)
    print("分析VGG16主成分变化对应的图片特征")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 配置
    model_path = '../vgg16_best_val_accuracy.pth'
    csv_path = '/Users/lzz/Documents/GitHub/IVF_AI/extracted_samples/embryos_all_focal_planes.csv'
    base_dir = '/Users/lzz/Documents/GitHub/IVF_AI'
    fair_dir = '/Users/lzz/Downloads/fair_embryos'
    output_dir = 'outputs/pc_variation_analysis'
    config_path = 'config/default.yaml'

    # 加载配置
    config_manager = load_config(config_path)
    config = config_manager.config
    config['model']['name'] = 'vgg16'

    # 创建数据集
    print("\n加载数据集...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = ThreeClassEmbryoDataset(
        csv_path=csv_path,
        base_dir=base_dir,
        fair_dir=fair_dir,
        transform=transform,
        max_samples_per_class=1000,  # 每类1000个样本
        max_fair_samples=1000  # Fair胚胎1000个
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 加载模型
    print("\n加载VGG16模型...")
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
    print("\n提取特征...")
    extractor = FeatureExtractorWithPaths(model)
    features, labels, paths = extractor.extract_features_with_paths(dataloader, device)

    print(f"\n特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"路径数量: {len(paths)}")

    # PCA降维
    print("\n执行PCA降维...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    print(f"PCA方差解释率: {pca.explained_variance_ratio_}")
    print(f"累计方差解释率: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 找图片对
    print("\n寻找主成分变化对应的图片对...")
    results = find_pc_variation_pairs(features_pca, labels, paths, n_pairs=50)  # 增加到50对

    # 打印结果
    print("\n" + "="*80)
    print(f"PC1接近，PC2极端相反的前20对（共{len(results['pc1_similar_pc2_opposite'])}对）（关于x轴对称）:")
    print("="*80)
    for i, pair in enumerate(results['pc1_similar_pc2_opposite'][:20], 1):
        print(f"\n#{i}:")
        print(f"  Good: PC1={pair['good_pc1']:.3f}, PC2={pair['good_pc2']:.3f}")
        print(f"  Poor: PC1={pair['poor_pc1']:.3f}, PC2={pair['poor_pc2']:.3f}")
        print(f"  差异: ΔPC1={pair['pc1_diff']:.3f}, ΔPC2={pair['pc2_diff']:.3f}")
        print(f"  |PC2|之和: {abs(pair['good_pc2']) + abs(pair['poor_pc2']):.3f} (越大越极端)")

    print("\n" + "="*80)
    print(f"PC2接近，PC1极端相反的前20对（共{len(results['pc2_similar_pc1_opposite'])}对）（关于y轴对称）:")
    print("="*80)
    for i, pair in enumerate(results['pc2_similar_pc1_opposite'][:20], 1):
        print(f"\n#{i}:")
        print(f"  Good: PC1={pair['good_pc1']:.3f}, PC2={pair['good_pc2']:.3f}")
        print(f"  Poor: PC1={pair['poor_pc1']:.3f}, PC2={pair['poor_pc2']:.3f}")
        print(f"  差异: ΔPC1={pair['pc1_diff']:.3f}, ΔPC2={pair['pc2_diff']:.3f}")
        print(f"  |PC1|之和: {abs(pair['good_pc1']) + abs(pair['poor_pc1']):.3f} (越大越极端)")

    # 可视化
    print("\n可视化图片对...")
    visualize_image_pairs(results, output_dir)

    print("\n" + "="*80)
    print("✅ 完成！")
    print(f"📁 结果保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
