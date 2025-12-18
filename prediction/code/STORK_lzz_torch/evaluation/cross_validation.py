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
        运行交叉验证（按胚胎ID分组，避免数据泄漏）

        Args:
            dataset: 完整数据集
            create_model_fn: 创建模型的函数

        Returns:
            交叉验证结果字典
        """
        print(f"开始 {self.n_folds} 折交叉验证（胚胎级别分组）...")

        # ===== Step 1: 提取胚胎ID和分组 =====
        from collections import defaultdict
        try:
            from utils.data_stats import extract_embryo_id
        except ImportError:
            from ..utils.data_stats import extract_embryo_id

        # 收集所有图像路径和标签
        all_image_paths = []
        all_labels = []

        if hasattr(dataset, 'datasets'):
            # ConcatDataset
            for sub_dataset in dataset.datasets:
                if hasattr(sub_dataset, 'cached_paths') and hasattr(sub_dataset, 'cached_labels'):
                    all_image_paths.extend(sub_dataset.cached_paths)
                    all_labels.extend(sub_dataset.cached_labels)
                elif hasattr(sub_dataset, 'image_paths') and hasattr(sub_dataset, 'labels'):
                    all_image_paths.extend(sub_dataset.image_paths)
                    all_labels.extend(sub_dataset.labels)
        elif hasattr(dataset, 'cached_paths') and hasattr(dataset, 'cached_labels'):
            all_image_paths = dataset.cached_paths
            all_labels = dataset.cached_labels
        elif hasattr(dataset, 'image_paths') and hasattr(dataset, 'labels'):
            all_image_paths = dataset.image_paths
            all_labels = dataset.labels
        else:
            # 降级方案：逐个读取
            print("⚠️  警告：无法获取图像路径，降级为按图像分割（可能存在数据泄漏）")
            all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
            return self._run_cross_validation_by_images(dataset, all_labels, create_model_fn)

        all_labels = np.array(all_labels)

        # 按胚胎ID分组
        embryo_groups = defaultdict(list)  # {embryo_id: [image_indices]}
        embryo_labels = {}  # {embryo_id: label}

        for idx, (img_path, label) in enumerate(zip(all_image_paths, all_labels)):
            embryo_id = extract_embryo_id(img_path)
            embryo_groups[embryo_id].append(idx)

            # 使用多数投票确定胚胎标签
            if embryo_id not in embryo_labels:
                embryo_labels[embryo_id] = label

        print(f"\n📊 数据集统计:")
        print(f"  总图像数: {len(all_image_paths)}")
        print(f"  总胚胎数: {len(embryo_groups)}")
        print(f"  平均每胚胎: {len(all_image_paths)/len(embryo_groups):.1f} 张图像")

        # 统计每个类别的胚胎数
        embryo_ids = list(embryo_groups.keys())
        embryo_label_array = np.array([embryo_labels[eid] for eid in embryo_ids])

        print(f"\n胚胎级别类别分布:")
        for class_label in [0, 1]:
            count = np.sum(embryo_label_array == class_label)
            print(f"  Class {class_label}: {count} 个胚胎")

        # ===== Step 2: 创建交叉验证分割器（基于胚胎ID）=====
        if self.stratified:
            kfold = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            splits = kfold.split(embryo_ids, embryo_label_array)
        else:
            kfold = KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            splits = kfold.split(embryo_ids)

        # 重置结果
        self.fold_results = []
        self.fold_histories = []
        self.fold_models = []

        # ===== Step 3: 执行交叉验证 =====
        for fold, (train_embryo_idx, val_embryo_idx) in enumerate(splits):
            print(f"\n{'='*80}")
            print(f"处理第 {fold + 1}/{self.n_folds} 折...")
            print(f"{'='*80}")

            # 获取该折的训练和验证胚胎ID
            train_embryo_ids = [embryo_ids[i] for i in train_embryo_idx]
            val_embryo_ids = [embryo_ids[i] for i in val_embryo_idx]

            # 将胚胎映射回图像索引
            train_img_indices = []
            val_img_indices = []

            for eid in train_embryo_ids:
                train_img_indices.extend(embryo_groups[eid])

            for eid in val_embryo_ids:
                val_img_indices.extend(embryo_groups[eid])

            train_img_indices = np.array(train_img_indices)
            val_img_indices = np.array(val_img_indices)

            # 验证无数据泄漏
            overlap = set(train_embryo_ids) & set(val_embryo_ids)
            if overlap:
                print(f"⚠️  警告：发现 {len(overlap)} 个胚胎重叠！")
            else:
                print(f"✅ 验证通过：训练集和验证集无胚胎重叠")

            # 统计该折的分布
            train_fold_labels = all_labels[train_img_indices]
            val_fold_labels = all_labels[val_img_indices]
            train_counts = np.bincount(train_fold_labels.astype(int))
            val_counts = np.bincount(val_fold_labels.astype(int))

            print(f"\nFold {fold + 1} 统计:")
            print(f"  训练集:")
            print(f"    胚胎数: {len(train_embryo_ids)}")
            print(f"    图像数: {len(train_img_indices)}")
            print(f"    Class 0: {train_counts[0]:5d} 张 ({train_counts[0]/(train_counts[0]+train_counts[1])*100:5.1f}%)")
            print(f"    Class 1: {train_counts[1]:5d} 张 ({train_counts[1]/(train_counts[0]+train_counts[1])*100:5.1f}%)")
            print(f"  验证集:")
            print(f"    胚胎数: {len(val_embryo_ids)}")
            print(f"    图像数: {len(val_img_indices)}")
            print(f"    Class 0: {val_counts[0]:5d} 张 ({val_counts[0]/(val_counts[0]+val_counts[1])*100:5.1f}%)")
            print(f"    Class 1: {val_counts[1]:5d} 张 ({val_counts[1]/(val_counts[0]+val_counts[1])*100:5.1f}%)")

            # 运行单折验证
            fold_result = self._run_single_fold(
                dataset, train_img_indices, val_img_indices, fold, create_model_fn
            )

            if fold_result:
                self.fold_results.append(fold_result)
                print(f"\n✅ 第 {fold + 1} 折完成 - 准确率: {fold_result['accuracy']:.4f}")
            else:
                print(f"\n❌ 第 {fold + 1} 折失败")

        # 汇总结果
        summary = self._summarize_results()

        # 保存详细结果
        self._save_detailed_results(summary)

        # 创建可视化
        self._create_visualizations()

        print(f"\n{'='*80}")
        print(f"交叉验证完成!")
        print(f"{'='*80}")
        print(f"\n【图像级别性能】")
        print(f"   平均准确率: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        print(f"   平均F1分数: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

        # 显示胚胎级别性能（如果存在）
        if 'mean_embryo_accuracy' in summary:
            print(f"\n【胚胎级别性能】")
            print(f"   平均准确率: {summary['mean_embryo_accuracy']:.4f} ± {summary['std_embryo_accuracy']:.4f}")
            print(f"   平均F1分数: {summary['mean_embryo_f1']:.4f} ± {summary['std_embryo_f1']:.4f}")

            # 计算改进百分比
            acc_improvement = (summary['mean_embryo_accuracy'] - summary['mean_accuracy']) * 100
            f1_improvement = (summary['mean_embryo_f1'] - summary['mean_f1']) * 100

            print(f"\n【性能对比】")
            print(f"   准确率改进: {acc_improvement:+.2f}%")
            print(f"   F1分数改进: {f1_improvement:+.2f}%")

            if 'total_embryos' in summary and 'total_images' in summary:
                print(f"\n【数据统计】")
                print(f"   总胚胎数: {summary['total_embryos']}")
                print(f"   总图像数: {summary['total_images']}")
                if 'avg_images_per_embryo' in summary:
                    print(f"   每胚胎平均图像数: {summary['avg_images_per_embryo']:.1f}")

        print(f"\n   结果保存在: {self.output_dir}")

        return summary

    def _run_cross_validation_by_images(self, dataset, labels, create_model_fn):
        """降级方案：按图像分割的交叉验证（存在数据泄漏风险）"""
        print("⚠️  使用图像级别分割，可能存在数据泄漏！")

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

        self.fold_results = []
        self.fold_histories = []
        self.fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)), labels)):
            print(f"\n处理第 {fold + 1}/{self.n_folds} 折...")
            fold_result = self._run_single_fold(
                dataset, train_idx, val_idx, fold, create_model_fn
            )
            if fold_result:
                self.fold_results.append(fold_result)

        summary = self._summarize_results()
        self._save_detailed_results(summary)
        self._create_visualizations()
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
            metrics = self._evaluate_fold(model, val_loader, fold, dataset, val_idx)

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
            print(f"第 {fold + 1} 折训练失败: {e}")
            return None

    def _get_image_paths_from_dataset(self, dataset: EmbryoDataset, indices: np.ndarray) -> List[str]:
        """从数据集中提取指定索引的图像路径"""
        image_paths = []

        # 尝试从数据集中获取图像路径
        if hasattr(dataset, 'datasets'):
            # ConcatDataset
            all_paths = []
            for sub_dataset in dataset.datasets:
                if hasattr(sub_dataset, 'cached_paths'):
                    all_paths.extend(sub_dataset.cached_paths)
                elif hasattr(sub_dataset, 'image_paths'):
                    all_paths.extend(sub_dataset.image_paths)
            image_paths = [all_paths[i] for i in indices]
        elif hasattr(dataset, 'cached_paths'):
            image_paths = [dataset.cached_paths[i] for i in indices]
        elif hasattr(dataset, 'image_paths'):
            image_paths = [dataset.image_paths[i] for i in indices]

        return image_paths

    def _evaluate_fold(self, model: torch.nn.Module, val_loader: DataLoader, fold: int,
                      dataset: EmbryoDataset, val_idx: np.ndarray) -> Dict:
        """评估单折模型（包含图像级别和胚胎级别指标）"""
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

        # 计算图像级别指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 图像级别指标
        image_metrics = {
            'fold': fold,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'predictions': all_preds.tolist(),
            'true_labels': all_labels.tolist(),
            'probabilities': all_probs.tolist()
        }

        # 计算AUC（如果是二分类）
        if len(np.unique(all_labels)) == 2:
            image_metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])

        # 计算胚胎级别指标
        try:
            # 获取图像路径
            image_paths = self._get_image_paths_from_dataset(dataset, val_idx)

            if image_paths:
                # 导入胚胎指标计算器
                try:
                    from evaluation.embryo_metrics import EmbryoMetricsCalculator
                except ImportError:
                    from embryo_metrics import EmbryoMetricsCalculator

                # 计算胚胎级别指标
                calculator = EmbryoMetricsCalculator(num_classes=2)
                embryo_results = calculator.calculate_metrics(
                    image_paths, all_labels, all_preds, all_probs, voting_method='majority'
                )

                # 添加胚胎级别指标
                embryo_level = embryo_results['embryo_level']
                image_metrics['embryo_accuracy'] = embryo_level['accuracy']
                image_metrics['embryo_precision'] = embryo_level['precision_macro']
                image_metrics['embryo_recall'] = embryo_level['recall_macro']
                image_metrics['embryo_f1'] = embryo_level['f1_macro']
                if 'auc_roc' in embryo_level:
                    image_metrics['embryo_auc'] = embryo_level['auc_roc']

                # 添加对比信息
                image_metrics['embryo_comparison'] = embryo_results['comparison']
                image_metrics['embryo_groups_info'] = embryo_results['embryo_groups_info']

                print(f"\n   胚胎级别性能:")
                print(f"     准确率: {embryo_level['accuracy']:.4f} (vs 图像级别: {image_metrics['accuracy']:.4f})")
                print(f"     F1分数: {embryo_level['f1_macro']:.4f} (vs 图像级别: {image_metrics['f1']:.4f})")
                print(f"     胚胎数: {embryo_results['embryo_groups_info']['num_embryos']}")
        except Exception as e:
            print(f"\n   ⚠️  无法计算胚胎级别指标: {e}")

        return image_metrics

    def _summarize_results(self) -> Dict[str, Any]:
        """汇总交叉验证结果"""
        if not self.fold_results:
            return {'error': 'No successful folds'}

        # 收集图像级别指标
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if 'auc' in self.fold_results[0]:
            metrics.append('auc')

        summary = {
            'n_folds': len(self.fold_results),
            'fold_results': self.fold_results
        }

        # 计算图像级别每个指标的统计信息
        for metric in metrics:
            values = [result[metric] for result in self.fold_results]
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
            summary[f'min_{metric}'] = np.min(values)
            summary[f'max_{metric}'] = np.max(values)
            summary[f'{metric}_values'] = values

        # 计算图像级别置信区间
        for metric in metrics:
            values = np.array([result[metric] for result in self.fold_results])
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            summary[f'{metric}_ci'] = (ci_lower, ci_upper)

        # 收集胚胎级别指标（如果存在）
        if 'embryo_accuracy' in self.fold_results[0]:
            embryo_metrics = ['embryo_accuracy', 'embryo_precision', 'embryo_recall', 'embryo_f1']
            if 'embryo_auc' in self.fold_results[0]:
                embryo_metrics.append('embryo_auc')

            # 计算胚胎级别每个指标的统计信息
            for metric in embryo_metrics:
                values = [result[metric] for result in self.fold_results if metric in result]
                if values:
                    summary[f'mean_{metric}'] = np.mean(values)
                    summary[f'std_{metric}'] = np.std(values)
                    summary[f'min_{metric}'] = np.min(values)
                    summary[f'max_{metric}'] = np.max(values)
                    summary[f'{metric}_values'] = values

            # 计算胚胎级别置信区间
            for metric in embryo_metrics:
                values = [result[metric] for result in self.fold_results if metric in result]
                if values:
                    values = np.array(values)
                    ci_lower = np.percentile(values, 2.5)
                    ci_upper = np.percentile(values, 97.5)
                    summary[f'{metric}_ci'] = (ci_lower, ci_upper)

            # 汇总胚胎组信息
            total_embryos = sum([result.get('embryo_groups_info', {}).get('num_embryos', 0)
                                for result in self.fold_results])
            total_images = sum([result.get('embryo_groups_info', {}).get('num_images', 0)
                               for result in self.fold_results])

            summary['total_embryos'] = total_embryos
            summary['total_images'] = total_images
            if total_embryos > 0:
                summary['avg_images_per_embryo'] = total_images / total_embryos

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

        print(f"详细结果已保存: {results_file}")

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
        # 图像级别指标
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if 'auc' in self.fold_results[0]:
            metrics.append('auc')

        # 胚胎级别指标
        has_embryo_metrics = 'embryo_accuracy' in self.fold_results[0]
        embryo_metrics = []
        if has_embryo_metrics:
            embryo_metrics = ['embryo_accuracy', 'embryo_precision', 'embryo_recall', 'embryo_f1']
            if 'embryo_auc' in self.fold_results[0]:
                embryo_metrics.append('embryo_auc')

        # 创建DataFrame
        data = []
        for i, result in enumerate(self.fold_results):
            row = {'Fold': i + 1}

            # 图像级别指标
            for metric in metrics:
                column_name = f"Image_{metric.capitalize()}"
                row[column_name] = f"{result[metric]:.4f}"

            # 胚胎级别指标
            if has_embryo_metrics:
                for metric in embryo_metrics:
                    if metric in result:
                        # 移除 'embryo_' 前缀并格式化列名
                        metric_name = metric.replace('embryo_', '').capitalize()
                        column_name = f"Embryo_{metric_name}"
                        row[column_name] = f"{result[metric]:.4f}"

                # 添加胚胎数量信息
                if 'embryo_groups_info' in result:
                    row['Num_Embryos'] = result['embryo_groups_info'].get('num_embryos', 0)

            data.append(row)

        # 添加平均值行
        mean_row = {'Fold': 'Mean'}

        # 图像级别平均值
        for metric in metrics:
            mean_val = summary[f'mean_{metric}']
            std_val = summary[f'std_{metric}']
            column_name = f"Image_{metric.capitalize()}"
            mean_row[column_name] = f"{mean_val:.4f} ± {std_val:.4f}"

        # 胚胎级别平均值
        if has_embryo_metrics:
            for metric in embryo_metrics:
                if f'mean_{metric}' in summary:
                    mean_val = summary[f'mean_{metric}']
                    std_val = summary[f'std_{metric}']
                    metric_name = metric.replace('embryo_', '').capitalize()
                    column_name = f"Embryo_{metric_name}"
                    mean_row[column_name] = f"{mean_val:.4f} ± {std_val:.4f}"

            if 'total_embryos' in summary:
                mean_row['Num_Embryos'] = summary['total_embryos'] // len(self.fold_results)

        data.append(mean_row)

        df = pd.DataFrame(data)

        # 保存CSV
        csv_file = self.output_dir / 'cv_summary.csv'
        df.to_csv(csv_file, index=False)

        print(f"汇总表已保存: {csv_file}")

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
        plt.savefig(self.output_dir / 'metrics_boxplot.pdf', dpi=150, bbox_inches='tight')
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
        plt.savefig(self.output_dir / 'training_history.pdf', dpi=150, bbox_inches='tight')
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
        plt.savefig(self.output_dir / 'confusion_matrices.pdf', dpi=150, bbox_inches='tight')
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
        plt.savefig(self.output_dir / 'roc_curves.pdf', dpi=150, bbox_inches='tight')
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