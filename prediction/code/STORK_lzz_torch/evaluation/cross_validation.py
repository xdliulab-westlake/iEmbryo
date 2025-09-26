"""
交叉验证实现
支持K折交叉验证和分层交叉验证
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import copy
import json

# 使用绝对导入而不是相对导入
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import STORKTrainer
from data.dataset import EmbryoDataset
from models.backbone import BackboneModel
from torch.utils.data import DataLoader, Subset


class CrossValidator:
    """交叉验证器"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化交叉验证器

        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device

        # 交叉验证参数
        cv_config = config.get('cross_validation', {})
        self.n_folds = cv_config.get('n_folds', 5)
        self.stratified = cv_config.get('stratified', True)
        self.shuffle = cv_config.get('shuffle', True)
        self.random_state = cv_config.get('random_state', 42)

        # 结果存储
        self.fold_results = []
        self.fold_histories = []
        self.fold_models = []

        # 输出目录
        self.output_dir = Path(config.get('paths', {}).get('output_dir', './outputs')) / 'cross_validation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_cross_validation(self,
                           dataset: EmbryoDataset,
                           create_model_fn) -> Dict[str, Any]:
        """
        运行交叉验证

        Args:
            dataset: 完整数据集
            create_model_fn: 创建模型的函数

        Returns:
            交叉验证结果字典
        """
        print(f"🔄 开始 {self.n_folds} 折交叉验证...")

        # 获取标签用于分层
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

        # 创建交叉验证分割器
        if self.stratified:
            kfold = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            kfold = KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )

        # 重置结果
        self.fold_results = []
        self.fold_histories = []
        self.fold_models = []

        # 执行交叉验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)), labels)):
            print(f"\n📊 处理第 {fold + 1}/{self.n_folds} 折...")

            # 运行单折验证
            fold_result = self._run_single_fold(
                dataset, train_idx, val_idx, fold, create_model_fn
            )

            if fold_result:
                self.fold_results.append(fold_result)
                print(f"   第 {fold + 1} 折完成 - 准确率: {fold_result['accuracy']:.4f}")
            else:
                print(f"   第 {fold + 1} 折失败")

        # 汇总结果
        summary = self._summarize_results()

        # 保存详细结果
        self._save_detailed_results(summary)

        # 创建可视化
        self._create_visualizations()

        print(f"\n✅ 交叉验证完成!")
        print(f"   平均准确率: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        print(f"   结果保存在: {self.output_dir}")

        return summary

    def _run_single_fold(self,
                        dataset: EmbryoDataset,
                        train_idx: np.ndarray,
                        val_idx: np.ndarray,
                        fold: int,
                        create_model_fn) -> Optional[Dict]:
        """运行单折验证"""
        try:
            # 创建数据子集
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # 创建数据加载器
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config.get('data', {}).get('num_workers', 4),
                pin_memory=True
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config.get('data', {}).get('num_workers', 4),
                pin_memory=True
            )

            # 创建模型
            model = create_model_fn()
            model = model.to(self.device)

            # 创建训练器
            trainer = STORKTrainer(model, self.config, train_loader, val_loader, self.device)

            # 训练模型 - 使用交叉验证专用轮数或主训练轮数
            cv_epochs = self.config.get('cross_validation', {}).get('epochs', None)
            if cv_epochs is None:
                # 如果没有设置专用轮数，使用主训练轮数
                num_epochs = self.config['training']['epochs']
            else:
                # 使用交叉验证专用轮数
                num_epochs = cv_epochs

            print(f"   训练 {num_epochs} 轮...")
            history = trainer.train(num_epochs)

            # 评估模型
            metrics = self._evaluate_fold(model, val_loader, fold)

            # 保存fold模型
            if self.config.get('cross_validation', {}).get('save_fold_models', False):
                model_path = self.output_dir / f'fold_{fold}_model.pth'
                torch.save(model.state_dict(), model_path)
                metrics['model_path'] = str(model_path)

            # 存储结果
            self.fold_histories.append(history)
            if self.config.get('cross_validation', {}).get('keep_models', False):
                self.fold_models.append(copy.deepcopy(model))

            return metrics

        except Exception as e:
            print(f"❌ 第 {fold + 1} 折训练失败: {e}")
            return None

    def _evaluate_fold(self, model: torch.nn.Module, val_loader: DataLoader, fold: int) -> Dict:
        """评估单折模型"""
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {
            'fold': fold,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'predictions': all_preds.tolist(),
            'true_labels': all_labels.tolist(),
            'probabilities': all_probs.tolist()
        }

        # 计算AUC（如果是二分类）
        if len(np.unique(all_labels)) == 2:
            metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])

        return metrics

    def _summarize_results(self) -> Dict[str, Any]:
        """汇总交叉验证结果"""
        if not self.fold_results:
            return {'error': 'No successful folds'}

        # 收集所有指标
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if 'auc' in self.fold_results[0]:
            metrics.append('auc')

        summary = {
            'n_folds': len(self.fold_results),
            'fold_results': self.fold_results
        }

        # 计算每个指标的统计信息
        for metric in metrics:
            values = [result[metric] for result in self.fold_results]
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
            summary[f'min_{metric}'] = np.min(values)
            summary[f'max_{metric}'] = np.max(values)
            summary[f'{metric}_values'] = values

        # 计算置信区间
        for metric in metrics:
            values = np.array([result[metric] for result in self.fold_results])
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            summary[f'{metric}_ci'] = (ci_lower, ci_upper)

        return summary

    def _save_detailed_results(self, summary: Dict[str, Any]):
        """保存详细结果"""
        # 保存JSON格式结果
        results_file = self.output_dir / 'cv_results.json'

        # 处理numpy数组和不可序列化的对象
        serializable_summary = self._make_serializable(summary)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)

        # 创建CSV格式的汇总表
        self._create_summary_table(summary)

        print(f"📄 详细结果已保存: {results_file}")

    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    def _create_summary_table(self, summary: Dict[str, Any]):
        """创建汇总表"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if 'auc' in self.fold_results[0]:
            metrics.append('auc')

        # 创建DataFrame
        data = []
        for i, result in enumerate(self.fold_results):
            row = {'Fold': i + 1}
            for metric in metrics:
                row[metric.capitalize()] = f"{result[metric]:.4f}"
            data.append(row)

        # 添加平均值行
        mean_row = {'Fold': 'Mean'}
        for metric in metrics:
            mean_val = summary[f'mean_{metric}']
            std_val = summary[f'std_{metric}']
            mean_row[metric.capitalize()] = f"{mean_val:.4f} ± {std_val:.4f}"
        data.append(mean_row)

        df = pd.DataFrame(data)

        # 保存CSV
        csv_file = self.output_dir / 'cv_summary.csv'
        df.to_csv(csv_file, index=False)

        print(f"📊 汇总表已保存: {csv_file}")

    def _create_visualizations(self):
        """创建可视化图表"""
        if not self.fold_results:
            return

        # 1. 指标分布箱线图
        self._plot_metrics_boxplot()

        # 2. 训练历史
        self._plot_training_history()

        # 3. 混淆矩阵
        self._plot_confusion_matrices()

        # 4. ROC曲线（如果是二分类）
        if 'auc' in self.fold_results[0]:
            self._plot_roc_curves()

    def _plot_metrics_boxplot(self):
        """绘制指标分布箱线图"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if 'auc' in self.fold_results[0]:
            metrics.append('auc')

        # 准备数据
        data = []
        for metric in metrics:
            for result in self.fold_results:
                data.append({
                    'Metric': metric.capitalize(),
                    'Value': result[metric]
                })

        df = pd.DataFrame(data)

        # 创建箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Metric', y='Value')
        plt.title('Cross-Validation Metrics Distribution')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        # 添加数值标签
        for i, metric in enumerate(metrics):
            values = [result[metric] for result in self.fold_results]
            mean_val = np.mean(values)
            plt.text(i, mean_val + 0.05, f'{mean_val:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_training_history(self):
        """绘制训练历史"""
        if not self.fold_histories:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 损失曲线
        for i, history in enumerate(self.fold_histories):
            if 'train_loss' in history and 'val_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                axes[0].plot(epochs, history['train_loss'], alpha=0.7, label=f'Fold {i+1} Train')
                axes[0].plot(epochs, history['val_loss'], alpha=0.7, label=f'Fold {i+1} Val')

        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        for i, history in enumerate(self.fold_histories):
            if 'train_accuracy' in history and 'val_accuracy' in history:
                epochs = range(1, len(history['train_accuracy']) + 1)
                axes[1].plot(epochs, history['train_accuracy'], alpha=0.7, label=f'Fold {i+1} Train')
                axes[1].plot(epochs, history['val_accuracy'], alpha=0.7, label=f'Fold {i+1} Val')

        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix

        n_folds = len(self.fold_results)
        cols = min(3, n_folds)
        rows = (n_folds + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_folds == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i, result in enumerate(self.fold_results):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            cm = confusion_matrix(result['true_labels'], result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Fold {i+1} (Acc: {result["accuracy"]:.3f})')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        # 隐藏多余的子图
        for i in range(n_folds, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_roc_curves(self):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve

        plt.figure(figsize=(8, 8))

        aucs = []
        for i, result in enumerate(self.fold_results):
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(result['true_labels'],
                                  np.array(result['probabilities'])[:, 1])
            auc = result['auc']
            aucs.append(auc)

            plt.plot(fpr, tpr, alpha=0.7,
                    label=f'Fold {i+1} (AUC = {auc:.3f})')

        # 绘制随机分类线
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random')

        # 计算平均AUC
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves (Mean AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def create_cross_validator(config: Dict[str, Any], device: torch.device) -> CrossValidator:
    """
    创建交叉验证器的便捷函数

    Args:
        config: 配置字典
        device: 计算设备

    Returns:
        CrossValidator实例
    """
    return CrossValidator(config, device)