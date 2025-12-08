"""
训练过程可视化工具
绘制训练/验证的准确率和损失曲线
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TrainingPlotter:
    """训练过程绘图器"""

    def __init__(self, output_dir: str = 'outputs', model_name: str = 'model'):
        """
        初始化绘图器

        Args:
            output_dir: 输出目录
            model_name: 模型名称
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self,
                              train_history: Dict[str, List[float]],
                              val_history: Dict[str, List[float]],
                              save_name: Optional[str] = None) -> str:
        """
        绘制训练历史（准确率和损失）

        Args:
            train_history: 训练历史，包含 'loss' 和 'acc' 键
            val_history: 验证历史，包含 'loss' 和 'acc' 键
            save_name: 保存文件名（可选）

        Returns:
            保存的文件路径
        """
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 获取数据
        train_acc = train_history.get('acc', [])
        train_loss = train_history.get('loss', [])
        val_acc = val_history.get('acc', [])
        val_loss = val_history.get('loss', [])

        epochs = range(1, len(train_acc) + 1)

        # 绘制准确率
        ax1.plot(epochs, [acc/100 for acc in train_acc], 'b-', label='Training acc', linewidth=1.5, alpha=0.8)
        if len(val_acc) > 0:
            ax1.plot(epochs, [acc/100 for acc in val_acc], 'r-', label='Validation acc', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # 绘制损失
        ax2.plot(epochs, train_loss, 'b-', label='Training loss', linewidth=1.5, alpha=0.8)
        if len(val_loss) > 0:
            ax2.plot(epochs, val_loss, 'r-', label='Validation loss', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if save_name is None:
            save_name = f'{self.model_name}_training_history.png'

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"训练曲线已保存到: {save_path}")
        return str(save_path)

    def plot_metric_comparison(self,
                               metrics: Dict[str, List[float]],
                               metric_name: str = 'Metric',
                               save_name: Optional[str] = None) -> str:
        """
        绘制多个指标的比较图

        Args:
            metrics: 指标字典，键为指标名称，值为指标值列表
            metric_name: 指标名称
            save_name: 保存文件名（可选）

        Returns:
            保存的文件路径
        """
        plt.figure(figsize=(10, 6))

        for name, values in metrics.items():
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=name, linewidth=2, alpha=0.8)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_name is None:
            save_name = f'{self.model_name}_{metric_name.lower()}_comparison.png'

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"指标比较图已保存到: {save_path}")
        return str(save_path)

    def plot_learning_rate(self,
                           learning_rates: List[float],
                           save_name: Optional[str] = None) -> str:
        """
        绘制学习率变化曲线

        Args:
            learning_rates: 学习率列表
            save_name: 保存文件名（可选）

        Returns:
            保存的文件路径
        """
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(learning_rates) + 1)
        plt.plot(epochs, learning_rates, 'g-', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数刻度

        if save_name is None:
            save_name = f'{self.model_name}_learning_rate.png'

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"学习率曲线已保存到: {save_path}")
        return str(save_path)

    def plot_confusion_matrix_evolution(self,
                                        confusion_matrices: List[np.ndarray],
                                        class_names: List[str],
                                        epochs_to_plot: List[int],
                                        save_name: Optional[str] = None) -> str:
        """
        绘制混淆矩阵演化过程

        Args:
            confusion_matrices: 混淆矩阵列表
            class_names: 类别名称
            epochs_to_plot: 要绘制的epoch列表
            save_name: 保存文件名（可选）

        Returns:
            保存的文件路径
        """
        n_plots = len(epochs_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))

        if n_plots == 1:
            axes = [axes]

        for idx, (ax, epoch) in enumerate(zip(axes, epochs_to_plot)):
            cm = confusion_matrices[epoch]
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f'Epoch {epoch}')

            # 添加colorbar
            plt.colorbar(im, ax=ax)

            # 设置刻度
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

            # 添加数值标签
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')

        plt.tight_layout()

        if save_name is None:
            save_name = f'{self.model_name}_confusion_matrix_evolution.png'

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"混淆矩阵演化图已保存到: {save_path}")
        return str(save_path)


def create_training_plotter(output_dir: str = 'outputs',
                            model_name: str = 'model') -> TrainingPlotter:
    """
    创建训练绘图器的工厂函数

    Args:
        output_dir: 输出目录
        model_name: 模型名称

    Returns:
        TrainingPlotter实例
    """
    return TrainingPlotter(output_dir=output_dir, model_name=model_name)


def plot_training_comparison(histories: Dict[str, Dict[str, List[float]]],
                             output_dir: str = 'outputs',
                             save_name: str = 'model_comparison.png') -> str:
    """
    比较多个模型的训练历史

    Args:
        histories: 模型训练历史字典，格式: {model_name: {'train_acc': [...], 'val_acc': [...], ...}}
        output_dir: 输出目录
        save_name: 保存文件名

    Returns:
        保存的文件路径
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    for model_name, history in histories.items():
        epochs = range(1, len(history.get('train_acc', [])) + 1)

        # 训练准确率
        if 'train_acc' in history:
            ax1.plot(epochs, [acc/100 for acc in history['train_acc']],
                    label=model_name, linewidth=1.5, alpha=0.8)

        # 验证准确率
        if 'val_acc' in history:
            ax2.plot(epochs, [acc/100 for acc in history['val_acc']],
                    label=model_name, linewidth=1.5, alpha=0.8)

        # 训练损失
        if 'train_loss' in history:
            ax3.plot(epochs, history['train_loss'],
                    label=model_name, linewidth=1.5, alpha=0.8)

        # 验证损失
        if 'val_loss' in history:
            ax4.plot(epochs, history['val_loss'],
                    label=model_name, linewidth=1.5, alpha=0.8)

    # 设置子图
    ax1.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"模型比较图已保存到: {save_path}")
    return str(save_path)
