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

        # 尝试不同的hook位置
        hook_registered = False

        # VGG16/ResNet/InceptionV3: 使用avgpool层
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'avgpool'):
            self.model.model.avgpool.register_forward_hook(hook_fn)
            print("Hook注册在: model.avgpool (ResNet/VGG16/InceptionV3)")
            hook_registered = True
        # DenseNet: 使用倒数第二个模块（在classifier之前）
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'features'):
            # 对于DenseNet，我们在features之后注册hook
            # 获取model的所有子模块
            modules = list(self.model.model.children())
            if len(modules) > 0:
                # 通常DenseNet的结构是: features -> classifier
                # 我们在features的最后注册hook
                self.model.model.features.register_forward_hook(hook_fn)
                print("Hook注册在: model.features (DenseNet)")
                hook_registered = True

        if not hook_registered:
            # 默认：在模型的倒数第二个模块注册
            modules = list(self.model.children())
            if len(modules) >= 2:
                modules[-2].register_forward_hook(hook_fn)
                print("Hook注册在倒数第二个模块")
            else:
                print("⚠️ 警告：无法找到合适的hook位置")

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


def find_pc_variation_pairs(features, labels, paths, n_pairs=10, class1=2, class2=0, class1_name="Good", class2_name="Poor"):
    """
    找出PC变化对应的图片对 - 寻找极端对称的情况
    使用贪心匹配算法，确保每个胚胎只配对一次

    Args:
        features: PCA降维后的特征 (N, 2)
        labels: 标签 (N,)
        paths: 图片路径列表
        n_pairs: 每种情况找多少对
        class1: 第一个类别的标签值
        class2: 第二个类别的标签值
        class1_name: 第一个类别的名称
        class2_name: 第二个类别的名称

    Returns:
        dict: 包含不同情况的图片对
    """
    # 分离两个类别
    class1_mask = labels == class1
    class2_mask = labels == class2

    class1_features = features[class1_mask]
    class2_features = features[class2_mask]
    class1_paths = [paths[i] for i in range(len(paths)) if class1_mask[i]]
    class2_paths = [paths[i] for i in range(len(paths)) if class2_mask[i]]

    print(f"\n{class1_name}胚胎数量: {len(class1_paths)}")
    print(f"{class2_name}胚胎数量: {len(class2_paths)}")
    print(f"{class1_name} PC1范围: [{class1_features[:, 0].min():.2f}, {class1_features[:, 0].max():.2f}]")
    print(f"{class1_name} PC2范围: [{class1_features[:, 1].min():.2f}, {class1_features[:, 1].max():.2f}]")
    print(f"{class2_name} PC1范围: [{class2_features[:, 0].min():.2f}, {class2_features[:, 0].max():.2f}]")
    print(f"{class2_name} PC2范围: [{class2_features[:, 1].min():.2f}, {class2_features[:, 1].max():.2f}]")

    results = {
        'pc1_similar_pc2_opposite': [],  # PC1接近，PC2一正一负且绝对值都大（关于x轴对称）
        'pc2_similar_pc1_opposite': []   # PC2接近，PC1一正一负且绝对值都大（关于y轴对称）
    }

    # 1. 找PC1接近，PC2极端相反的图片对（关于x轴对称）
    print("\n寻找PC1接近，PC2极端相反的图片对（关于x轴对称）...")

    # 计算所有可能的配对得分
    all_pairs = []
    for i, feat1 in enumerate(class1_features):
        pc1_diff = np.abs(class2_features[:, 0] - feat1[0])

        for j, feat2 in enumerate(class2_features):
            # PC2必须符号相反
            if feat1[1] * feat2[1] < 0:
                # 得分 = 两者绝对值之和（都大更好） - PC1差异（差异小更好）
                score = abs(feat1[1]) + abs(feat2[1]) - pc1_diff[j] * 0.5
                all_pairs.append({
                    'class1_idx': i,
                    'class2_idx': j,
                    'score': score,
                    'class1_pc1': feat1[0],
                    'class1_pc2': feat1[1],
                    'class2_pc1': feat2[0],
                    'class2_pc2': feat2[1],
                    'pc1_diff': pc1_diff[j],
                    'pc2_diff': abs(feat1[1] - feat2[1])
                })

    # 按得分排序
    all_pairs.sort(key=lambda x: x['score'], reverse=True)

    # 贪心选择：每次选择得分最高且未使用的配对
    used_class1 = set()
    used_class2 = set()
    for pair in all_pairs:
        if len(results['pc1_similar_pc2_opposite']) >= n_pairs:
            break

        if pair['class1_idx'] not in used_class1 and pair['class2_idx'] not in used_class2:
            results['pc1_similar_pc2_opposite'].append({
                'class1_path': class1_paths[pair['class1_idx']],
                'class2_path': class2_paths[pair['class2_idx']],
                'class1_pc1': pair['class1_pc1'],
                'class1_pc2': pair['class1_pc2'],
                'class2_pc1': pair['class2_pc1'],
                'class2_pc2': pair['class2_pc2'],
                'pc1_diff': pair['pc1_diff'],
                'pc2_diff': pair['pc2_diff'],
                'score': pair['score'],
                'class1_name': class1_name,
                'class2_name': class2_name
            })
            used_class1.add(pair['class1_idx'])
            used_class2.add(pair['class2_idx'])

    # 2. 找PC2接近，PC1极端相反的图片对（关于y轴对称）
    print("寻找PC2接近，PC1极端相反的图片对（关于y轴对称）...")

    all_pairs = []
    for i, feat1 in enumerate(class1_features):
        pc2_diff = np.abs(class2_features[:, 1] - feat1[1])

        for j, feat2 in enumerate(class2_features):
            # PC1必须符号相反
            if feat1[0] * feat2[0] < 0:
                # 得分 = 两者绝对值之和 - PC2差异
                score = abs(feat1[0]) + abs(feat2[0]) - pc2_diff[j] * 0.5
                all_pairs.append({
                    'class1_idx': i,
                    'class2_idx': j,
                    'score': score,
                    'class1_pc1': feat1[0],
                    'class1_pc2': feat1[1],
                    'class2_pc1': feat2[0],
                    'class2_pc2': feat2[1],
                    'pc1_diff': abs(feat1[0] - feat2[0]),
                    'pc2_diff': pc2_diff[j]
                })

    # 按得分排序
    all_pairs.sort(key=lambda x: x['score'], reverse=True)

    # 贪心选择
    used_class1 = set()
    used_class2 = set()
    for pair in all_pairs:
        if len(results['pc2_similar_pc1_opposite']) >= n_pairs:
            break

        if pair['class1_idx'] not in used_class1 and pair['class2_idx'] not in used_class2:
            results['pc2_similar_pc1_opposite'].append({
                'class1_path': class1_paths[pair['class1_idx']],
                'class2_path': class2_paths[pair['class2_idx']],
                'class1_pc1': pair['class1_pc1'],
                'class1_pc2': pair['class1_pc2'],
                'class2_pc1': pair['class2_pc1'],
                'class2_pc2': pair['class2_pc2'],
                'pc1_diff': pair['pc1_diff'],
                'pc2_diff': pair['pc2_diff'],
                'score': pair['score'],
                'class1_name': class1_name,
                'class2_name': class2_name
            })
            used_class1.add(pair['class1_idx'])
            used_class2.add(pair['class2_idx'])

    return results


def visualize_image_pairs(results, output_dir, comparison_name="Good_vs_Poor"):
    """可视化图片对"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取类别名称
    if len(results['pc1_similar_pc2_opposite']) > 0:
        class1_name = results['pc1_similar_pc2_opposite'][0]['class1_name']
        class2_name = results['pc1_similar_pc2_opposite'][0]['class2_name']
    else:
        class1_name = "Class1"
        class2_name = "Class2"

    # 颜色映射
    color_map = {'Good': 'green', 'Fair': 'orange', 'Poor': 'red'}
    class1_color = color_map.get(class1_name, 'blue')
    class2_color = color_map.get(class2_name, 'purple')

    # 1. PC1接近，PC2极端相反（关于x轴对称）
    print(f"\n生成PC1接近PC2极端相反的图片对（关于x轴对称）- {class1_name} vs {class2_name}...")
    n_pairs = len(results['pc1_similar_pc2_opposite'])
    if n_pairs == 0:
        print("  ⚠️ 没有找到符合条件的图片对")
    else:
        fig, axes = plt.subplots(n_pairs, 2, figsize=(8, n_pairs * 4))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'PC1 Similar, PC2 Opposite (Symmetric about X-axis)\n{class1_name} vs {class2_name}', fontsize=16, y=0.998)

        for i, pair in enumerate(results['pc1_similar_pc2_opposite']):
            # Class1图片
            class1_img = Image.open(pair['class1_path']).convert('RGB')
            axes[i, 0].imshow(class1_img)
            axes[i, 0].axis('off')
            axes[i, 0].set_title(
                f"#{i+1} {class1_name}\nPC1:{pair['class1_pc1']:.2f}, PC2:{pair['class1_pc2']:.2f}",
                fontsize=9, color=class1_color
            )

            # Class2图片
            class2_img = Image.open(pair['class2_path']).convert('RGB')
            axes[i, 1].imshow(class2_img)
            axes[i, 1].axis('off')
            axes[i, 1].set_title(
                f"#{i+1} {class2_name}\nPC1:{pair['class2_pc1']:.2f}, PC2:{pair['class2_pc2']:.2f}\n"
                f"ΔPC1:{pair['pc1_diff']:.3f}, ΔPC2:{pair['pc2_diff']:.3f}",
                fontsize=9, color=class2_color
            )

        plt.tight_layout()
        save_path = output_dir / f'{comparison_name}_pc1_similar_pc2_opposite.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_path} ({n_pairs}对)")

    # 2. PC2接近，PC1极端相反（关于y轴对称）
    print(f"\n生成PC2接近PC1极端相反的图片对（关于y轴对称）- {class1_name} vs {class2_name}...")
    n_pairs = len(results['pc2_similar_pc1_opposite'])
    if n_pairs == 0:
        print("  ⚠️ 没有找到符合条件的图片对")
    else:
        fig, axes = plt.subplots(n_pairs, 2, figsize=(8, n_pairs * 4))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'PC2 Similar, PC1 Opposite (Symmetric about Y-axis)\n{class1_name} vs {class2_name}', fontsize=16, y=0.998)

        for i, pair in enumerate(results['pc2_similar_pc1_opposite']):
            # Class1图片
            class1_img = Image.open(pair['class1_path']).convert('RGB')
            axes[i, 0].imshow(class1_img)
            axes[i, 0].axis('off')
            axes[i, 0].set_title(
                f"#{i+1} {class1_name}\nPC1:{pair['class1_pc1']:.2f}, PC2:{pair['class1_pc2']:.2f}",
                fontsize=9, color=class1_color
            )

            # Class2图片
            class2_img = Image.open(pair['class2_path']).convert('RGB')
            axes[i, 1].imshow(class2_img)
            axes[i, 1].axis('off')
            axes[i, 1].set_title(
                f"#{i+1} {class2_name}\nPC1:{pair['class2_pc1']:.2f}, PC2:{pair['class2_pc2']:.2f}\n"
                f"ΔPC1:{pair['pc1_diff']:.3f}, ΔPC2:{pair['pc2_diff']:.3f}",
                fontsize=9, color=class2_color
            )

        plt.tight_layout()
        save_path = output_dir / f'{comparison_name}_pc2_similar_pc1_opposite.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_path} ({n_pairs}对)")


def visualize_pc_sorted_sequence(features_pca, labels, paths, output_dir, pc_component=1, n_images=50):
    """
    按PC值排序生成连续的图片序列

    Args:
        features_pca: PCA降维后的特征 (N, 2或3)
        labels: 标签 (N,)
        paths: 图片路径列表
        output_dir: 输出目录
        pc_component: 使用哪个主成分排序 (1, 2 或 3)
        n_images: 要显示的图片数量
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 颜色映射
    class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}
    color_map = {'Good': 'green', 'Fair': 'orange', 'Poor': 'red'}

    # 按指定PC值排序
    pc_idx = pc_component - 1  # 转换为索引 (0 或 1)
    pc_values = features_pca[:, pc_idx]
    sorted_indices = np.argsort(pc_values)

    # 均匀采样 n_images 张图片
    total_samples = len(sorted_indices)
    if n_images > total_samples:
        n_images = total_samples
        selected_indices = sorted_indices
    else:
        # 均匀采样
        step = total_samples / n_images
        selected_indices = [sorted_indices[int(i * step)] for i in range(n_images)]

    print(f"\n生成按PC{pc_component}排序的连续图片序列...")
    print(f"  总样本数: {total_samples}")
    print(f"  选择图片数: {len(selected_indices)}")
    print(f"  PC{pc_component}范围: [{pc_values.min():.3f}, {pc_values.max():.3f}]")

    # 计算网格布局 - 每行显示多少张图片
    n_cols = 5  # 每行5张图片
    n_rows = int(np.ceil(len(selected_indices) / n_cols))

    # 创建图表
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(f'Embryo Images Sorted by PC{pc_component} (Low to High)',
                 fontsize=16, fontweight='bold', y=0.998)

    # 展平axes数组以便于索引
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # 绘制每张图片
    for i, idx in enumerate(selected_indices):
        img = Image.open(paths[idx]).convert('RGB')
        pc1_val = features_pca[idx, 0]
        pc2_val = features_pca[idx, 1]
        label = labels[idx]
        class_name = class_names.get(label, 'Unknown')
        color = color_map.get(class_name, 'blue')

        # 构建标题 - 根据维度决定显示哪些PC值
        if features_pca.shape[1] >= 3:
            pc3_val = features_pca[idx, 2]
            title_text = f"#{i+1} {class_name}\nPC1:{pc1_val:.2f}, PC2:{pc2_val:.2f}, PC3:{pc3_val:.2f}"
        else:
            title_text = f"#{i+1} {class_name}\nPC1:{pc1_val:.2f}, PC2:{pc2_val:.2f}"

        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(title_text, fontsize=8, color=color)

    # 隐藏多余的子图
    for i in range(len(selected_indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path = output_dir / f'pc{pc_component}_sorted_sequence.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 保存: {save_path} ({len(selected_indices)}张图片)")


def main():
    """主函数"""
    print("="*80)
    print("分析VGG16主成分变化对应的图片特征")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 配置
    model_path = '../vgg16_best_val_accuracy.pth'
    csv_path = '/Users/lzz/Desktop/IVF/good_embryos/embryos_all_focal_planes.csv'
    base_dir = '/Users/lzz/Desktop/IVF/good_embryos'
    poor_csv_path = '/Users/lzz/Desktop/IVF/poor_embryos/embryos_all_focal_planes.csv'
    poor_base_dir = '/Users/lzz/Desktop/IVF/poor_embryos'
    fair_dir = '/Users/lzz/Desktop/IVF/fair_embryos'
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
        max_fair_samples=1000,  # 加载Fair胚胎
        poor_csv_path=poor_csv_path,
        poor_base_dir=poor_base_dir
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
    pca = PCA(n_components=3)  # 改为3个主成分以包含PC3
    features_pca = pca.fit_transform(features)
    print(f"PCA方差解释率: {pca.explained_variance_ratio_}")
    print(f"累计方差解释率: {np.sum(pca.explained_variance_ratio_):.4f}")

    # PC 变化分析 - 不分组，整体分析所有样本
    print("\n" + "="*80)
    print("分析主成分变化对应的图片特征")
    print("="*80)

    # 找图片对 - 使用所有样本（不限制类别）
    # 修改函数调用，传入所有类别的数据
    all_results = {
        'pc1_similar_pc2_opposite': [],
        'pc2_similar_pc1_opposite': []
    }

    # 1. 找PC1接近，PC2极端相反的图片对（关于x轴对称）
    print("\n寻找PC1接近，PC2极端相反的图片对（关于x轴对称）...")
    all_pairs = []
    for i in range(len(features_pca)):
        for j in range(i+1, len(features_pca)):
            feat1 = features_pca[i]
            feat2 = features_pca[j]

            # PC2必须符号相反
            if feat1[1] * feat2[1] < 0:
                pc1_diff = abs(feat1[0] - feat2[0])
                # 得分 = 两者绝对值之和（都大更好） - PC1差异（差异小更好）
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

    # 按得分排序并选择前20对
    all_pairs.sort(key=lambda x: x['score'], reverse=True)
    used_indices = set()
    for pair in all_pairs[:100]:  # 检查前100对，选择20对
        if len(all_results['pc1_similar_pc2_opposite']) >= 20:
            break
        if pair['idx1'] not in used_indices and pair['idx2'] not in used_indices:
            class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}
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

    # 2. 找PC2接近，PC1极端相反的图片对（关于y轴对称）
    print("寻找PC2接近，PC1极端相反的图片对（关于y轴对称）...")
    all_pairs = []
    for i in range(len(features_pca)):
        for j in range(i+1, len(features_pca)):
            feat1 = features_pca[i]
            feat2 = features_pca[j]

            # PC1必须符号相反
            if feat1[0] * feat2[0] < 0:
                pc2_diff = abs(feat1[1] - feat2[1])
                # 得分 = 两者绝对值之和 - PC2差异
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

    # 按得分排序并选择前20对
    all_pairs.sort(key=lambda x: x['score'], reverse=True)
    used_indices = set()
    for pair in all_pairs[:100]:
        if len(all_results['pc2_similar_pc1_opposite']) >= 20:
            break
        if pair['idx1'] not in used_indices and pair['idx2'] not in used_indices:
            class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}
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

    # 打印结果
    print(f"\nPC1接近，PC2极端相反的前10对（共{len(all_results['pc1_similar_pc2_opposite'])}对）（关于x轴对称）:")
    for i, pair in enumerate(all_results['pc1_similar_pc2_opposite'][:10], 1):
        print(f"#{i}: {pair['class1_name']} PC1={pair['class1_pc1']:.3f}, PC2={pair['class1_pc2']:.3f} | "
              f"{pair['class2_name']} PC1={pair['class2_pc1']:.3f}, PC2={pair['class2_pc2']:.3f} | "
              f"ΔPC1={pair['pc1_diff']:.3f}, ΔPC2={pair['pc2_diff']:.3f}")

    print(f"\nPC2接近，PC1极端相反的前10对（共{len(all_results['pc2_similar_pc1_opposite'])}对）（关于y轴对称）:")
    for i, pair in enumerate(all_results['pc2_similar_pc1_opposite'][:10], 1):
        print(f"#{i}: {pair['class1_name']} PC1={pair['class1_pc1']:.3f}, PC2={pair['class1_pc2']:.3f} | "
              f"{pair['class2_name']} PC1={pair['class2_pc1']:.3f}, PC2={pair['class2_pc2']:.3f} | "
              f"ΔPC1={pair['pc1_diff']:.3f}, ΔPC2={pair['pc2_diff']:.3f}")

    # 可视化
    print(f"\n可视化图片对...")
    visualize_image_pairs(all_results, output_dir, comparison_name="PC_Variation_Analysis")

    # 生成按PC1和PC2排序的连续图片序列
    print("\n" + "="*80)
    print("生成PC值排序的连续图片序列")
    print("="*80)

    # 按PC1排序
    visualize_pc_sorted_sequence(features_pca, labels, paths, output_dir,
                                pc_component=1, n_images=50)

    # 按PC2排序
    visualize_pc_sorted_sequence(features_pca, labels, paths, output_dir,
                                pc_component=2, n_images=50)

    # 按PC3排序
    visualize_pc_sorted_sequence(features_pca, labels, paths, output_dir,
                                pc_component=3, n_images=50)

    print("\n" + "="*80)
    print("✅ 完成！")
    print(f"📁 结果保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
