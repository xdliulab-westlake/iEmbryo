"""
胚胎特征空间可视化
使用模型提取的特征进行降维可视化（PCA、UMAP）
分析good、fair、poor胚胎在特征空间中的分布
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. Install with: pip install umap-learn")

from torch.utils.data import DataLoader
from tqdm import tqdm


class FeatureExtractor:
    """模型特征提取器"""

    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        """
        初始化特征提取器

        Args:
            model: 训练好的模型
            layer_name: 要提取特征的层名称，None表示使用倒数第二层
        """
        self.model = model
        self.model.eval()
        self.features = []
        self.layer_name = layer_name

        # 注册hook
        self._register_hook()

    def _register_hook(self):
        """注册前向hook以提取特征"""
        def hook_fn(module, input, output):
            # 将特征展平为1D向量
            if isinstance(output, torch.Tensor):
                feat = output.detach()
                if len(feat.shape) > 2:  # 如果是卷积层输出 (B, C, H, W)
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                    feat = feat.view(feat.size(0), -1)
                self.features.append(feat.cpu())

        # 自动查找特征层
        if self.layer_name:
            # 用户指定层
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    module.register_forward_hook(hook_fn)
                    print(f"Hook注册在层: {name}")
                    return
        else:
            # 自动选择：查找最后一个全连接层之前的层
            modules = list(self.model.modules())

            # 对于常见模型结构的处理
            if hasattr(self.model, 'model'):
                # BackboneModel包装
                base_model = self.model.model

                # ResNet系列：使用avgpool层
                if hasattr(base_model, 'avgpool'):
                    base_model.avgpool.register_forward_hook(hook_fn)
                    print(f"Hook注册在层: model.avgpool (ResNet)")

                # VGG系列：使用features后的avgpool
                elif hasattr(base_model, 'features') and hasattr(base_model, 'avgpool'):
                    base_model.avgpool.register_forward_hook(hook_fn)
                    print(f"Hook注册在层: model.avgpool (VGG)")

                # InceptionV3：使用最后的avgpool
                elif 'inception' in str(type(base_model)).lower():
                    # InceptionV3没有avgpool属性，需要找到最后的adaptive_avg_pool2d
                    for name, module in base_model.named_modules():
                        if 'avgpool' in name.lower():
                            module.register_forward_hook(hook_fn)
                            print(f"Hook注册在层: {name}")
                            break
                else:
                    # 降级方案：注册倒数第二个模块
                    if len(modules) > 2:
                        modules[-2].register_forward_hook(hook_fn)
                        print(f"Hook注册在倒数第二个模块")

    def extract_features(self,
                        dataloader: DataLoader,
                        device: torch.device,
                        max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        提取特征

        Args:
            dataloader: 数据加载器
            device: 设备
            max_samples: 最大样本数（None表示全部）

        Returns:
            (features, labels, image_paths): 特征数组、标签数组、图像路径列表
        """
        all_features = []
        all_labels = []
        all_paths = []

        self.model.to(device)
        self.model.eval()

        sample_count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取特征"):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    images = batch
                    labels = None

                images = images.to(device)

                # 清空之前的特征
                self.features = []

                # 前向传播（触发hook）
                _ = self.model(images)

                # 收集特征
                if len(self.features) > 0:
                    batch_features = self.features[0]
                    all_features.append(batch_features.numpy())

                    if labels is not None:
                        all_labels.append(labels.cpu().numpy())

                    sample_count += len(images)

                    if max_samples and sample_count >= max_samples:
                        break

        # 合并所有批次
        features = np.vstack(all_features) if all_features else np.array([])
        labels = np.concatenate(all_labels) if all_labels else np.array([])

        print(f"\n提取的特征形状: {features.shape}")
        print(f"标签形状: {labels.shape}")

        return features, labels, all_paths


class FeatureSpaceVisualizer:
    """特征空间可视化器"""

    # 类别名称映射
    CLASS_NAMES = {
        0: 'Poor',
        1: 'Fair',
        2: 'Good'
    }

    # 颜色映射
    CLASS_COLORS = {
        0: '#e74c3c',  # 红色 - Poor
        1: '#f39c12',  # 橙色 - Fair
        2: '#2ecc71'   # 绿色 - Good
    }

    def __init__(self, output_dir: Union[str, Path]):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def visualize_pca(self,
                     features: np.ndarray,
                     labels: np.ndarray,
                     n_components: int = 2,
                     title: str = "PCA Visualization") -> np.ndarray:
        """
        PCA降维可视化

        Args:
            features: 特征矩阵 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
            n_components: 降维后的维度（2或3）
            title: 图表标题

        Returns:
            降维后的特征
        """
        print(f"\n执行PCA降维到{n_components}维...")

        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(features)

        # 打印方差解释率
        explained_var = pca.explained_variance_ratio_
        print(f"方差解释率: {explained_var}")
        print(f"累计方差解释率: {np.sum(explained_var):.4f}")

        # 可视化
        if n_components == 2:
            self._plot_2d(features_reduced, labels, title,
                         f"pca_2d.pdf",
                         xlabel=f"PC1 ({explained_var[0]:.2%})",
                         ylabel=f"PC2 ({explained_var[1]:.2%})")
        elif n_components == 3:
            self._plot_3d(features_reduced, labels, title,
                         f"pca_3d.pdf",
                         xlabel=f"PC1 ({explained_var[0]:.2%})",
                         ylabel=f"PC2 ({explained_var[1]:.2%})",
                         zlabel=f"PC3 ({explained_var[2]:.2%})")

        return features_reduced

    def visualize_tsne(self,
                      features: np.ndarray,
                      labels: np.ndarray,
                      n_components: int = 2,
                      perplexity: float = 30.0,
                      title: str = "t-SNE Visualization") -> np.ndarray:
        """
        t-SNE降维可视化

        Args:
            features: 特征矩阵
            labels: 标签数组
            n_components: 降维后的维度（2或3）
            perplexity: t-SNE参数
            title: 图表标题

        Returns:
            降维后的特征
        """
        print(f"\n执行t-SNE降维到{n_components}维...")
        print(f"  perplexity={perplexity}")

        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                   random_state=42, n_iter=1000)
        features_reduced = tsne.fit_transform(features)

        # 可视化
        if n_components == 2:
            self._plot_2d(features_reduced, labels, title, "tsne_2d.pdf")
        elif n_components == 3:
            self._plot_3d(features_reduced, labels, title, "tsne_3d.pdf")

        return features_reduced

    def visualize_umap(self,
                      features: np.ndarray,
                      labels: np.ndarray,
                      n_components: int = 2,
                      n_neighbors: int = 15,
                      min_dist: float = 0.1,
                      title: str = "UMAP Visualization") -> np.ndarray:
        """
        UMAP降维可视化

        Args:
            features: 特征矩阵
            labels: 标签数组
            n_components: 降维后的维度（2或3）
            n_neighbors: UMAP邻居数
            min_dist: UMAP最小距离
            title: 图表标题

        Returns:
            降维后的特征
        """
        if not UMAP_AVAILABLE:
            print("⚠️  UMAP未安装，跳过UMAP可视化")
            return None

        print(f"\n执行UMAP降维到{n_components}维...")
        print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}")

        reducer = umap.UMAP(n_components=n_components,
                           n_neighbors=n_neighbors,
                           min_dist=min_dist,
                           random_state=42)
        features_reduced = reducer.fit_transform(features)

        # 可视化
        if n_components == 2:
            self._plot_2d(features_reduced, labels, title, "umap_2d.pdf")
        elif n_components == 3:
            self._plot_3d(features_reduced, labels, title, "umap_3d.pdf")

        return features_reduced

    def _plot_2d(self,
                features: np.ndarray,
                labels: np.ndarray,
                title: str,
                filename: str,
                xlabel: str = "Component 1",
                ylabel: str = "Component 2"):
        """绘制2D散点图"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 获取唯一类别
        unique_labels = np.unique(labels)

        # 为每个类别绘制点
        for label in unique_labels:
            mask = labels == label
            class_name = self.CLASS_NAMES.get(label, f"Class {label}")
            color = self.CLASS_COLORS.get(label, None)

            ax.scatter(features[mask, 0],
                      features[mask, 1],
                      c=color,
                      label=f"{class_name} (n={np.sum(mask)})",
                      alpha=0.6,
                      s=30,
                      edgecolors='white',
                      linewidth=0.5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 2D可视化已保存: {save_path}")

    def _plot_3d(self,
                features: np.ndarray,
                labels: np.ndarray,
                title: str,
                filename: str,
                xlabel: str = "Component 1",
                ylabel: str = "Component 2",
                zlabel: str = "Component 3"):
        """绘制3D散点图"""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 获取唯一类别
        unique_labels = np.unique(labels)

        # 为每个类别绘制点
        for label in unique_labels:
            mask = labels == label
            class_name = self.CLASS_NAMES.get(label, f"Class {label}")
            color = self.CLASS_COLORS.get(label, None)

            ax.scatter(features[mask, 0],
                      features[mask, 1],
                      features[mask, 2],
                      c=color,
                      label=f"{class_name} (n={np.sum(mask)})",
                      alpha=0.6,
                      s=30,
                      edgecolors='white',
                      linewidth=0.5)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_zlabel(zlabel, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 3D可视化已保存: {save_path}")

    def plot_variance_explained(self, features: np.ndarray, max_components: int = 50):
        """绘制PCA方差解释率"""
        print(f"\n计算前{max_components}个主成分的方差解释率...")

        n_components = min(max_components, features.shape[1], features.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(features)

        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 单个成分方差解释率
        ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
        ax1.set_xlabel('主成分编号', fontsize=12)
        ax1.set_ylabel('方差解释率', fontsize=12)
        ax1.set_title('各主成分方差解释率', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 累计方差解释率
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var,
                marker='o', markersize=4, linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax2.axhline(y=0.90, color='orange', linestyle='--', label='90%')
        ax2.set_xlabel('主成分数量', fontsize=12)
        ax2.set_ylabel('累计方差解释率', fontsize=12)
        ax2.set_title('累计方差解释率', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'pca_variance_explained.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 方差解释率图已保存: {save_path}")

        # 打印关键信息
        n_95 = np.argmax(cumulative_var >= 0.95) + 1
        n_90 = np.argmax(cumulative_var >= 0.90) + 1
        print(f"   达到90%方差需要: {n_90} 个主成分")
        print(f"   达到95%方差需要: {n_95} 个主成分")


def visualize_feature_space(model: nn.Module,
                            dataloader: DataLoader,
                            device: torch.device,
                            output_dir: Union[str, Path],
                            methods: List[str] = ['pca', 'tsne', 'umap'],
                            max_samples: Optional[int] = None):
    """
    完整的特征空间可视化流程

    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
        output_dir: 输出目录
        methods: 降维方法列表，支持 'pca', 'tsne', 'umap'
        max_samples: 最大样本数
    """
    print("="*80)
    print("开始特征空间可视化分析")
    print("="*80)

    # 1. 提取特征
    extractor = FeatureExtractor(model)
    features, labels, _ = extractor.extract_features(dataloader, device, max_samples)

    if len(features) == 0:
        print("❌ 未能提取到特征，请检查模型和数据")
        return

    # 2. 创建可视化器
    visualizer = FeatureSpaceVisualizer(output_dir)

    # 3. 绘制方差解释率（仅PCA）
    if 'pca' in methods:
        visualizer.plot_variance_explained(features)

    # 4. 执行各种降维方法
    if 'pca' in methods:
        visualizer.visualize_pca(features, labels, n_components=2,
                                title="PCA 2D - Embryo Quality Feature Space")
        visualizer.visualize_pca(features, labels, n_components=3,
                                title="PCA 3D - Embryo Quality Feature Space")

    if 'tsne' in methods:
        visualizer.visualize_tsne(features, labels, n_components=2,
                                 title="t-SNE 2D - Embryo Quality Feature Space")

    if 'umap' in methods:
        visualizer.visualize_umap(features, labels, n_components=2,
                                 title="UMAP 2D - Embryo Quality Feature Space")
        visualizer.visualize_umap(features, labels, n_components=3,
                                 title="UMAP 3D - Embryo Quality Feature Space")

    print("\n" + "="*80)
    print("特征空间可视化完成！")
    print(f"结果保存在: {output_dir}")
    print("="*80)
