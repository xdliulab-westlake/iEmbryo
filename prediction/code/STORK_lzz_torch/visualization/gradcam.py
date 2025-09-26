"""
Grad-CAM可视化实现
PyTorch版本的Grad-CAM，用于解释模型决策
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


class GradCAM:
    """Grad-CAM可视化类"""

    def __init__(self, model, target_layer_name: str = None):
        """
        初始化Grad-CAM

        Args:
            model: 要分析的模型
            target_layer_name: 目标层名称，如果为None则自动选择
        """
        self.model = model
        self.model.eval()

        # 自动选择目标层
        if target_layer_name is None:
            target_layer_name = self._find_target_layer()

        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # 注册钩子
        self._register_hooks()

    def _find_target_layer(self) -> str:
        """自动查找合适的目标层"""
        # 根据模型类型选择合适的层
        model_name = self.model.__class__.__name__.lower()

        # 预定义的层映射
        layer_mapping = {
            'vgg': 'backbone.features.28',      # VGG最后一个卷积层
            'resnet': 'backbone.layer4.2.conv2',  # ResNet最后一个卷积层
            'efficientnet': 'backbone.blocks.6',   # EfficientNet
            'inception': 'backbone.Mixed_7c',       # InceptionV3
        }

        # 查找匹配的层
        for key, layer_name in layer_mapping.items():
            if key in model_name or hasattr(self.model, 'model_name'):
                if hasattr(self.model, 'model_name'):
                    if key in self.model.model_name:
                        return layer_name

        # 如果没有找到，尝试查找backbone中的最后一个卷积层
        return self._find_last_conv_layer()

    def _find_last_conv_layer(self) -> str:
        """查找最后一个卷积层"""
        target_layer = None

        # 遍历模型的所有模块
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.modules.conv.Conv2d)):
                target_layer = name

        if target_layer is None:
            raise ValueError("无法找到合适的卷积层")

        print(f"🎯 自动选择目标层: {target_layer}")
        return target_layer

    def _register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # 获取目标层
        target_layer = self._get_layer_by_name(self.target_layer_name)
        if target_layer is None:
            raise ValueError(f"找不到层: {self.target_layer_name}")

        # 注册钩子
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _get_layer_by_name(self, layer_name: str):
        """根据名称获取层"""
        layer = self.model
        for name in layer_name.split('.'):
            if hasattr(layer, name):
                layer = getattr(layer, name)
            else:
                return None
        return layer

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        生成Grad-CAM热力图

        Args:
            input_tensor: 输入张量 [1, C, H, W]
            class_idx: 目标类别索引，如果为None则使用预测类别

        Returns:
            热力图数组
        """
        # 确保输入是正确的形状
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # 克隆张量并设置梯度
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # 设置模型为评估模式但启用梯度计算
        self.model.eval()

        # 前向传播
        with torch.enable_grad():
            output = self.model(input_tensor)

            # 如果没有指定类别，使用预测的类别
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            # 清零梯度
            self.model.zero_grad()

            # 反向传播
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)

        # 获取梯度和激活
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # 计算权重（全局平均池化）
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 生成CAM
        cam = torch.sum(weights * activations, dim=1).squeeze(0)  # [H, W]

        # ReLU（只保留正值）
        cam = F.relu(cam)

        # 归一化到0-1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()

    def visualize(self,
                  image: np.ndarray,
                  input_tensor: torch.Tensor,
                  class_idx: int = None,
                  alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        可视化Grad-CAM

        Args:
            image: 原始图像 [H, W, 3]
            input_tensor: 预处理后的输入张量
            class_idx: 目标类别
            alpha: 叠加透明度

        Returns:
            (heatmap, superimposed_image)
        """
        # 生成CAM
        cam = self.generate_cam(input_tensor, class_idx)

        # 调整CAM大小到原图尺寸
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # 创建热力图
        heatmap = cm.jet(cam_resized)[:, :, :3] * 255
        heatmap = heatmap.astype(np.uint8)

        # 叠加到原图
        superimposed = heatmap * alpha + image * (1 - alpha)
        superimposed = superimposed.astype(np.uint8)

        return heatmap, superimposed


class GradCAMVisualizer:
    """Grad-CAM可视化管理器"""

    def __init__(self, model, config: Dict[str, Any]):
        """
        初始化可视化器

        Args:
            model: 训练好的模型
            config: 配置字典
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # 创建Grad-CAM实例
        self.gradcam = GradCAM(model)

        # 输出目录
        self.output_dir = Path(config.get('paths', {}).get('output_dir', './outputs')) / 'gradcam'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_samples(self,
                         data_loader,
                         num_samples: int = 10,
                         save_individual: bool = True) -> Dict[str, Any]:
        """
        为多个样本生成Grad-CAM可视化

        Args:
            data_loader: 数据加载器
            num_samples: 可视化样本数量
            save_individual: 是否保存单个样本

        Returns:
            可视化结果字典
        """
        print(f"🎨 生成 {num_samples} 个样本的Grad-CAM可视化...")

        results = {
            'samples': [],
            'class_stats': {'correct': 0, 'total': 0},
            'attention_patterns': {'class_0': [], 'class_1': []}
        }

        sample_count = 0
        class_counts = {0: 0, 1: 0}

        for batch_idx, (images, labels) in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # 获取预测（使用no_grad进行预测）
            with torch.no_grad():
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)

            # 处理批次中的每个样本
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break

                # 平衡类别采样
                true_label = labels[i].item()
                if class_counts[true_label] >= num_samples // 2:
                    continue

                image_tensor = images[i:i+1]
                true_label = labels[i].item()
                pred_label = predictions[i].item()

                # 生成可视化
                sample_result = self._visualize_single_sample(
                    image_tensor, true_label, pred_label, sample_count
                )

                if sample_result:
                    results['samples'].append(sample_result)
                    results['attention_patterns'][f'class_{true_label}'].append(
                        sample_result['attention_map']
                    )

                    # 统计
                    if true_label == pred_label:
                        results['class_stats']['correct'] += 1
                    results['class_stats']['total'] += 1

                    class_counts[true_label] += 1
                    sample_count += 1

        # 创建汇总可视化
        self._create_summary_visualization(results)

        print(f"✅ Grad-CAM可视化完成")
        print(f"   准确率: {results['class_stats']['correct']}/{results['class_stats']['total']} "
              f"({100*results['class_stats']['correct']/max(1, results['class_stats']['total']):.1f}%)")

        return results

    def _visualize_single_sample(self,
                                image_tensor: torch.Tensor,
                                true_label: int,
                                pred_label: int,
                                sample_idx: int) -> Optional[Dict]:
        """可视化单个样本"""
        try:
            # 反归一化图像用于显示
            image_np = self._denormalize_image(image_tensor[0])

            # 生成Grad-CAM
            heatmap, superimposed = self.gradcam.visualize(
                image_np, image_tensor, class_idx=pred_label
            )

            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # 原图
            axes[0].imshow(image_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # 热力图
            axes[1].imshow(heatmap)
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')

            # 叠加图
            axes[2].imshow(superimposed)
            axes[2].set_title(f'Overlay\nTrue: {true_label}, Pred: {pred_label}')
            axes[2].axis('off')

            # 添加整体标题
            accuracy_marker = "✓" if true_label == pred_label else "✗"
            fig.suptitle(f'Sample {sample_idx} {accuracy_marker}', fontsize=14)

            plt.tight_layout()

            # 保存
            save_path = self.output_dir / f'sample_{sample_idx:03d}_class_{true_label}_pred_{pred_label}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            return {
                'sample_idx': sample_idx,
                'true_label': true_label,
                'pred_label': pred_label,
                'correct': true_label == pred_label,
                'save_path': str(save_path),
                'attention_map': self.gradcam.generate_cam(image_tensor, pred_label)
            }

        except Exception as e:
            print(f"⚠️  样本 {sample_idx} 可视化失败: {e}")
            return None

    def _denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """反归一化图像"""
        # ImageNet标准化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)

        # 反归一化
        denorm = tensor * std + mean
        denorm = torch.clamp(denorm, 0, 1)

        # 转换为numpy并调整维度
        image_np = denorm.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        return image_np

    def _create_summary_visualization(self, results: Dict):
        """创建汇总可视化"""
        if not results['samples']:
            return

        print("📊 创建Grad-CAM汇总分析...")

        # 分析注意力模式
        self._analyze_attention_patterns(results)

        # 创建准确率统计
        self._create_accuracy_analysis(results)

    def _analyze_attention_patterns(self, results: Dict):
        """分析注意力模式"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        for class_idx in [0, 1]:
            attention_maps = results['attention_patterns'][f'class_{class_idx}']
            if not attention_maps:
                continue

            # 计算平均注意力图
            avg_attention = np.mean(attention_maps, axis=0)

            # 显示平均注意力图
            im = axes[class_idx, 0].imshow(avg_attention, cmap='jet')
            axes[class_idx, 0].set_title(f'Class {class_idx} Average Attention')
            axes[class_idx, 0].axis('off')
            plt.colorbar(im, ax=axes[class_idx, 0])

            # 显示注意力强度分布
            intensities = [np.mean(attention) for attention in attention_maps]
            axes[class_idx, 1].hist(intensities, bins=10, alpha=0.7)
            axes[class_idx, 1].set_title(f'Class {class_idx} Attention Intensity')
            axes[class_idx, 1].set_xlabel('Average Attention')
            axes[class_idx, 1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_pattern_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_accuracy_analysis(self, results: Dict):
        """创建准确率分析"""
        # 按类别统计准确率
        class_stats = {0: {'correct': 0, 'total': 0}, 1: {'correct': 0, 'total': 0}}

        for sample in results['samples']:
            class_idx = sample['true_label']
            class_stats[class_idx]['total'] += 1
            if sample['correct']:
                class_stats[class_idx]['correct'] += 1

        # 创建条形图
        fig, ax = plt.subplots(figsize=(8, 6))

        classes = ['Class 0', 'Class 1']
        accuracies = []
        for class_idx in [0, 1]:
            if class_stats[class_idx]['total'] > 0:
                acc = class_stats[class_idx]['correct'] / class_stats[class_idx]['total']
            else:
                acc = 0
            accuracies.append(acc * 100)

        bars = ax.bar(classes, accuracies, color=['skyblue', 'lightcoral'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Per-Class Accuracy in Grad-CAM Samples')
        ax.set_ylim(0, 100)

        # 添加数值标签
        for bar, acc, class_idx in zip(bars, accuracies, [0, 1]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%\n({class_stats[class_idx]["correct"]}/{class_stats[class_idx]["total"]})',
                   ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'gradcam_accuracy_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()


def create_gradcam_visualizer(model, config: Dict[str, Any]) -> GradCAMVisualizer:
    """
    创建Grad-CAM可视化器的便捷函数

    Args:
        model: 训练好的模型
        config: 配置字典

    Returns:
        GradCAMVisualizer实例
    """
    return GradCAMVisualizer(model, config)