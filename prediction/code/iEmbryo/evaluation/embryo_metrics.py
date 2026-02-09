"""
胚胎级别评估指标计算
支持图像级别和胚胎级别的指标对比
使用多数投票机制聚合同一胚胎的多张图像预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from collections import defaultdict
import json
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.data_stats import extract_embryo_id
except ImportError:
    from ..utils.data_stats import extract_embryo_id


class EmbryoMetricsCalculator:
    """
    胚胎级别指标计算器
    支持图像级别和胚胎级别的指标计算和对比
    """

    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        """
        初始化胚胎级别指标计算器

        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names or ['Class_0', 'Class_1']

    def calculate_metrics(self,
                         image_paths: List[str],
                         y_true_image: np.ndarray,
                         y_pred_image: np.ndarray,
                         y_proba_image: Optional[np.ndarray] = None,
                         voting_method: str = 'majority') -> Dict[str, Any]:
        """
        计算图像级别和胚胎级别的指标

        Args:
            image_paths: 图像文件路径列表
            y_true_image: 图像级别的真实标签
            y_pred_image: 图像级别的预测标签
            y_proba_image: 图像级别的预测概率（可选）
            voting_method: 投票方法，支持 'majority'（多数投票）和 'soft'（概率平均）

        Returns:
            包含图像级别和胚胎级别指标的字典
        """
        if len(image_paths) != len(y_true_image) or len(image_paths) != len(y_pred_image):
            raise ValueError("image_paths, y_true_image, y_pred_image 长度必须相同")

        # 按胚胎ID分组
        embryo_groups = self._group_by_embryo(
            image_paths, y_true_image, y_pred_image, y_proba_image
        )

        # 计算图像级别指标
        image_metrics = self._calculate_image_metrics(y_true_image, y_pred_image, y_proba_image)

        # 进行胚胎级别投票
        y_true_embryo, y_pred_embryo, y_proba_embryo = self._embryo_voting(
            embryo_groups, voting_method
        )

        # 计算胚胎级别指标
        embryo_metrics = self._calculate_embryo_metrics(
            y_true_embryo, y_pred_embryo, y_proba_embryo
        )

        # 对比分析
        comparison = self._compare_metrics(image_metrics, embryo_metrics, embryo_groups)

        return {
            'image_level': image_metrics,
            'embryo_level': embryo_metrics,
            'comparison': comparison,
            'embryo_groups_info': self._get_embryo_groups_info(embryo_groups)
        }

    def _group_by_embryo(self,
                         image_paths: List[str],
                         y_true_image: np.ndarray,
                         y_pred_image: np.ndarray,
                         y_proba_image: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
        """
        按胚胎ID分组图像数据

        Returns:
            {embryo_id: {'paths': [...], 'y_true': [...], 'y_pred': [...], 'y_proba': [...]}}
        """
        embryo_groups = defaultdict(lambda: {
            'paths': [],
            'y_true': [],
            'y_pred': [],
            'y_proba': [] if y_proba_image is not None else None
        })

        for idx, img_path in enumerate(image_paths):
            embryo_id = extract_embryo_id(img_path)
            embryo_groups[embryo_id]['paths'].append(img_path)
            embryo_groups[embryo_id]['y_true'].append(int(y_true_image[idx]))
            embryo_groups[embryo_id]['y_pred'].append(int(y_pred_image[idx]))
            if y_proba_image is not None:
                embryo_groups[embryo_id]['y_proba'].append(y_proba_image[idx])

        # 转换为numpy数组（如果有的话）
        for embryo_id in embryo_groups:
            embryo_groups[embryo_id]['y_true'] = np.array(embryo_groups[embryo_id]['y_true'])
            embryo_groups[embryo_id]['y_pred'] = np.array(embryo_groups[embryo_id]['y_pred'])
            if embryo_groups[embryo_id]['y_proba'] is not None:
                embryo_groups[embryo_id]['y_proba'] = np.array(embryo_groups[embryo_id]['y_proba'])

        return dict(embryo_groups)

    def _embryo_voting(self,
                      embryo_groups: Dict[str, Dict[str, Any]],
                      voting_method: str = 'majority') -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        对胚胎进行投票，得出胚胎级别的预测

        Args:
            embryo_groups: 按胚胎分组的数据
            voting_method: 投票方法 ('majority' 或 'soft')

        Returns:
            (y_true_embryo, y_pred_embryo, y_proba_embryo)
        """
        y_true_embryo = []
        y_pred_embryo = []
        y_proba_embryo = [] if any(g['y_proba'] is not None for g in embryo_groups.values()) else None

        for embryo_id in sorted(embryo_groups.keys()):  # 排序以确保一致性
            group = embryo_groups[embryo_id]

            # 真实标签：多数投票（通常应该都相同）
            true_label = int(np.round(np.mean(group['y_true'])))
            y_true_embryo.append(true_label)

            # 预测标签：根据投票方法
            if voting_method == 'majority':
                # 多数投票
                mean_pred = np.mean(group['y_pred'])

                # 检查是否平票
                if mean_pred == 0.5:  # 平票情况：预测为0和1的图像数量相等
                    # 平票时比较平均的softmax概率，选择概率更高的类别
                    avg_proba = np.mean(group['y_proba'], axis=0)
                    pred_label = int(np.argmax(avg_proba))
                else:
                    # 非平票情况：正常多数投票
                    pred_label = int(np.round(mean_pred))

                y_pred_embryo.append(pred_label)

                if y_proba_embryo is not None and group['y_proba'] is not None:
                    # 概率取平均
                    avg_proba = np.mean(group['y_proba'], axis=0)
                    y_proba_embryo.append(avg_proba)

            elif voting_method == 'soft':
                # 软投票：基于概率的平均
                if group['y_proba'] is not None:
                    avg_proba = np.mean(group['y_proba'], axis=0)
                    pred_label = int(np.argmax(avg_proba))
                    y_proba_embryo.append(avg_proba)
                else:
                    # 如果没有概率，降级为多数投票
                    pred_label = int(np.round(np.mean(group['y_pred'])))

                y_pred_embryo.append(pred_label)
            else:
                raise ValueError(f"未知的投票方法: {voting_method}")

        return np.array(y_true_embryo), np.array(y_pred_embryo), \
               np.array(y_proba_embryo) if y_proba_embryo is not None else None

    def _calculate_image_metrics(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """计算图像级别的指标"""
        metrics = {
            'level': 'image',
            'num_samples': len(y_true),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        }

        # 按类别的指标
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            metrics[f'precision_{class_name}'] = float(
                precision_score(y_true, y_pred, labels=[class_idx], average='binary', zero_division=0)
            )
            metrics[f'recall_{class_name}'] = float(
                recall_score(y_true, y_pred, labels=[class_idx], average='binary', zero_division=0)
            )
            metrics[f'f1_{class_name}'] = float(
                f1_score(y_true, y_pred, labels=[class_idx], average='binary', zero_division=0)
            )

        # AUC
        if y_proba is not None and self.num_classes == 2:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except:
                pass

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        return metrics

    def _calculate_embryo_metrics(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """计算胚胎级别的指标"""
        metrics = {
            'level': 'embryo',
            'num_samples': len(y_true),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        }

        # 按类别的指标
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            metrics[f'precision_{class_name}'] = float(
                precision_score(y_true, y_pred, labels=[class_idx], average='binary', zero_division=0)
            )
            metrics[f'recall_{class_name}'] = float(
                recall_score(y_true, y_pred, labels=[class_idx], average='binary', zero_division=0)
            )
            metrics[f'f1_{class_name}'] = float(
                f1_score(y_true, y_pred, labels=[class_idx], average='binary', zero_division=0)
            )

        # AUC
        if y_proba is not None and self.num_classes == 2:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except:
                pass

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        return metrics

    def _compare_metrics(self,
                        image_metrics: Dict[str, Any],
                        embryo_metrics: Dict[str, Any],
                        embryo_groups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        对比图像级别和胚胎级别的指标

        Returns:
            对比分析结果
        """
        key_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        comparison = {
            'num_images': image_metrics['num_samples'],
            'num_embryos': embryo_metrics['num_samples'],
            'avg_images_per_embryo': image_metrics['num_samples'] / embryo_metrics['num_samples'],
        }

        # 逐指标对比
        for metric in key_metrics:
            image_val = image_metrics[metric]
            embryo_val = embryo_metrics[metric]
            diff = embryo_val - image_val
            pct_change = (diff / image_val * 100) if image_val != 0 else 0

            comparison[f'{metric}_image'] = image_val
            comparison[f'{metric}_embryo'] = embryo_val
            comparison[f'{metric}_difference'] = diff
            comparison[f'{metric}_pct_change'] = pct_change

        # 分析：胚胎级别是否显著改进
        accuracy_improvement = comparison['accuracy_pct_change']
        if accuracy_improvement > 2:
            comparison['improvement_note'] = f"胚胎级别投票显著改进准确率 {accuracy_improvement:.2f}%"
        elif accuracy_improvement > 0:
            comparison['improvement_note'] = f"胚胎级别投票略微改进准确率 {accuracy_improvement:.2f}%"
        elif accuracy_improvement < -2:
            comparison['improvement_note'] = f"胚胎级别投票降低准确率 {accuracy_improvement:.2f}%（需检查）"
        else:
            comparison['improvement_note'] = f"胚胎级别投票与图像级别性能接近 {accuracy_improvement:.2f}%"

        return comparison

    def _get_embryo_groups_info(self, embryo_groups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """获取胚胎分组统计信息"""
        images_per_embryo = [len(g['paths']) for g in embryo_groups.values()]

        return {
            'num_embryos': len(embryo_groups),
            'num_images': sum(images_per_embryo),
            'min_images_per_embryo': min(images_per_embryo),
            'max_images_per_embryo': max(images_per_embryo),
            'mean_images_per_embryo': float(np.mean(images_per_embryo)),
            'std_images_per_embryo': float(np.std(images_per_embryo)),
        }

    def print_comparison_report(self, metrics_result: Dict[str, Any]) -> None:
        """打印对比报告"""
        print("\n" + "="*80)
        print("胚胎级别评估对比报告")
        print("="*80)

        # 胚胎分组信息
        embryo_info = metrics_result['embryo_groups_info']
        print(f"\n【胚胎分组统计】")
        print(f"  总胚胎数: {embryo_info['num_embryos']}")
        print(f"  总图像数: {embryo_info['num_images']}")
        print(f"  每胚胎图像数 (平均): {embryo_info['mean_images_per_embryo']:.1f} ± {embryo_info['std_images_per_embryo']:.1f}")
        print(f"  每胚胎图像数 (范围): [{embryo_info['min_images_per_embryo']}, {embryo_info['max_images_per_embryo']}]")

        # 关键指标对比
        comparison = metrics_result['comparison']
        print(f"\n【关键指标对比】")
        print(f"{'指标':<25} {'图像级别':<15} {'胚胎级别':<15} {'改进':<10}")
        print("-" * 65)

        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            image_val = comparison[f'{metric}_image']
            embryo_val = comparison[f'{metric}_embryo']
            pct_change = comparison[f'{metric}_pct_change']

            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<25} {image_val:<15.4f} {embryo_val:<15.4f} {pct_change:>8.2f}%")

        print(f"\n【评价】")
        print(f"  {comparison['improvement_note']}")

        # 混淆矩阵
        print(f"\n【混淆矩阵】")
        image_cm = np.array(metrics_result['image_level']['confusion_matrix'])
        embryo_cm = np.array(metrics_result['embryo_level']['confusion_matrix'])

        print(f"  图像级别:")
        print(f"    TN={image_cm[0,0]}, FP={image_cm[0,1]}")
        print(f"    FN={image_cm[1,0]}, TP={image_cm[1,1]}")

        print(f"  胚胎级别:")
        print(f"    TN={embryo_cm[0,0]}, FP={embryo_cm[0,1]}")
        print(f"    FN={embryo_cm[1,0]}, TP={embryo_cm[1,1]}")

        print("="*80 + "\n")

    def save_metrics_report(self, metrics_result: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """保存指标报告到JSON文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics_result, f, indent=2)

        print(f"✓ 指标报告已保存: {output_path}")


def compare_predictions_with_embryo_voting(
    image_paths: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    voting_method: str = 'majority',
    output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    便捷函数：计算并对比图像级别和胚胎级别的指标

    Args:
        image_paths: 图像文件路径列表
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（可选）
        voting_method: 投票方法
        output_path: 输出路径（可选）

    Returns:
        对比结果字典
    """
    calculator = EmbryoMetricsCalculator(num_classes=2)
    metrics_result = calculator.calculate_metrics(
        image_paths, y_true, y_pred, y_proba, voting_method
    )

    # 打印报告
    calculator.print_comparison_report(metrics_result)

    # 保存报告
    if output_path:
        calculator.save_metrics_report(metrics_result, output_path)

    return metrics_result
