"""
评估指标和报告生成
提供全面的模型评估功能
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import warnings


class MetricsCalculator:
    """指标计算器"""

    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        """
        初始化指标计算器

        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]

    def calculate_all_metrics(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算所有评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率

        Returns:
            包含所有指标的字典
        """
        metrics = {}

        # 基础分类指标
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # 分类报告
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        # 如果提供了概率，计算基于概率的指标
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))

        # 按类别的详细指标
        metrics.update(self._calculate_per_class_metrics(y_true, y_pred))

        return metrics

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算基础指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        }

        return metrics

    def _calculate_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """计算基于概率的指标"""
        metrics = {}

        try:
            if self.num_classes == 2:
                # 二分类指标
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])

                # ROC曲线数据
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba[:, 1])
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }

                # PR曲线数据
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba[:, 1])
                metrics['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }

                # Average Precision
                from sklearn.metrics import average_precision_score
                metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])

            else:
                # 多分类指标
                try:
                    metrics['auc_roc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    metrics['auc_roc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
                except Exception as e:
                    warnings.warn(f"无法计算多分类AUC: {e}")

        except Exception as e:
            warnings.warn(f"计算概率指标时出错: {e}")

        return metrics

    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """计算每个类别的详细指标"""
        per_class = {}

        # 计算每个类别的精确率、召回率、F1
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        for i, class_name in enumerate(self.class_names):
            per_class[f'{class_name}_precision'] = float(precision_per_class[i])
            per_class[f'{class_name}_recall'] = float(recall_per_class[i])
            per_class[f'{class_name}_f1'] = float(f1_per_class[i])

        # 支持度（每个类别的样本数）
        unique, counts = np.unique(y_true, return_counts=True)
        for class_idx, count in zip(unique, counts):
            if class_idx < len(self.class_names):
                per_class[f'{self.class_names[class_idx]}_support'] = int(count)

        return per_class


class ReportGenerator:
    """评估报告生成器"""

    def __init__(self, output_dir: Union[str, Path], model_name: str = "model"):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
            model_name: 模型名称
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def generate_full_report(self,
                           metrics: Dict[str, Any],
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_proba: Optional[np.ndarray] = None,
                           class_names: Optional[List[str]] = None) -> str:
        """
        生成完整的评估报告

        Args:
            metrics: 计算得到的指标字典
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            class_names: 类别名称

        Returns:
            报告文件路径
        """
        print("📊 生成评估报告...")

        # 设置类别名称
        if class_names is None:
            num_classes = len(np.unique(y_true))
            class_names = [f'Class_{i}' for i in range(num_classes)]

        # 1. 生成JSON报告
        json_path = self._save_json_report(metrics)

        # 2. 生成HTML报告
        html_path = self._generate_html_report(metrics, y_true, y_pred, y_proba, class_names)

        # 3. 生成可视化
        self._generate_visualizations(metrics, y_true, y_pred, y_proba, class_names)

        # 4. 生成CSV摘要
        csv_path = self._generate_csv_summary(metrics)

        print(f"✅ 报告生成完成:")
        print(f"   HTML报告: {html_path}")
        print(f"   JSON报告: {json_path}")
        print(f"   CSV摘要: {csv_path}")

        return str(html_path)

    def _save_json_report(self, metrics: Dict[str, Any]) -> Path:
        """保存JSON格式报告"""
        json_path = self.output_dir / f'{self.model_name}_metrics.json'

        # 确保所有值都可序列化
        serializable_metrics = self._make_serializable(metrics)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

        return json_path

    def _generate_html_report(self,
                            metrics: Dict[str, Any],
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_proba: Optional[np.ndarray],
                            class_names: List[str]) -> Path:
        """生成HTML报告"""
        html_path = self.output_dir / f'{self.model_name}_report.html'

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.model_name} 评估报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .section {{ margin: 30px 0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #6c757d; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .confusion-matrix {{ text-align: center; }}
        .confusion-matrix td {{ text-align: center; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.model_name} 模型评估报告</h1>
        <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📊 核心指标</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('accuracy', 0):.4f}</div>
                <div class="metric-label">准确率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('precision_weighted', 0):.4f}</div>
                <div class="metric-label">精确率 (加权)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('recall_weighted', 0):.4f}</div>
                <div class="metric-label">召回率 (加权)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('f1_weighted', 0):.4f}</div>
                <div class="metric-label">F1分数 (加权)</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📈 详细指标</h2>
        {self._generate_metrics_table(metrics)}
    </div>

    <div class="section">
        <h2>🔍 混淆矩阵</h2>
        {self._generate_confusion_matrix_html(metrics.get('confusion_matrix', []), class_names)}
    </div>

    <div class="section">
        <h2>📋 分类报告</h2>
        {self._generate_classification_report_html(metrics.get('classification_report', {}))}
    </div>

    <div class="section">
        <h2>📊 可视化图表</h2>
        <p>以下图表已保存到输出目录:</p>
        <ul>
            <li>混淆矩阵热力图: confusion_matrix.png</li>
            <li>ROC曲线: roc_curve.png (如果适用)</li>
            <li>精确率-召回率曲线: pr_curve.png (如果适用)</li>
            <li>指标分布图: metrics_distribution.png</li>
        </ul>
    </div>
</body>
</html>
        """

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

    def _generate_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """生成指标表格HTML"""
        table_html = "<table><tr><th>指标</th><th>值</th></tr>"

        key_metrics = [
            ('accuracy', '准确率'),
            ('precision_macro', '精确率 (宏平均)'),
            ('precision_micro', '精确率 (微平均)'),
            ('precision_weighted', '精确率 (加权平均)'),
            ('recall_macro', '召回率 (宏平均)'),
            ('recall_micro', '召回率 (微平均)'),
            ('recall_weighted', '召回率 (加权平均)'),
            ('f1_macro', 'F1分数 (宏平均)'),
            ('f1_micro', 'F1分数 (微平均)'),
            ('f1_weighted', 'F1分数 (加权平均)'),
        ]

        if 'auc_roc' in metrics:
            key_metrics.append(('auc_roc', 'AUC-ROC'))
        if 'average_precision' in metrics:
            key_metrics.append(('average_precision', '平均精确率'))

        for key, label in key_metrics:
            value = metrics.get(key, 'N/A')
            if isinstance(value, (int, float)):
                value = f"{value:.4f}"
            table_html += f"<tr><td>{label}</td><td>{value}</td></tr>"

        table_html += "</table>"
        return table_html

    def _generate_confusion_matrix_html(self, cm: List[List[int]], class_names: List[str]) -> str:
        """生成混淆矩阵HTML表格"""
        if not cm:
            return "<p>混淆矩阵数据不可用</p>"

        html = '<table class="confusion-matrix"><tr><th></th>'
        for name in class_names:
            html += f'<th>预测 {name}</th>'
        html += '</tr>'

        for i, row in enumerate(cm):
            html += f'<tr><th>实际 {class_names[i]}</th>'
            for val in row:
                html += f'<td>{val}</td>'
            html += '</tr>'

        html += '</table>'
        return html

    def _generate_classification_report_html(self, report: Dict[str, Any]) -> str:
        """生成分类报告HTML表格"""
        if not report:
            return "<p>分类报告数据不可用</p>"

        html = "<table><tr><th>类别</th><th>精确率</th><th>召回率</th><th>F1分数</th><th>支持度</th></tr>"

        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                html += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{metrics.get('precision', 0):.4f}</td>
                    <td>{metrics.get('recall', 0):.4f}</td>
                    <td>{metrics.get('f1-score', 0):.4f}</td>
                    <td>{metrics.get('support', 0)}</td>
                </tr>
                """

        html += "</table>"
        return html

    def _generate_visualizations(self,
                               metrics: Dict[str, Any],
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray],
                               class_names: List[str]):
        """生成可视化图表"""
        # 1. 混淆矩阵热力图
        self._plot_confusion_matrix(metrics.get('confusion_matrix', []), class_names)

        # 2. ROC曲线（二分类）
        if 'roc_curve' in metrics:
            self._plot_roc_curve(metrics['roc_curve'], metrics.get('auc_roc', 0))

        # 3. 精确率-召回率曲线（二分类）
        if 'pr_curve' in metrics:
            self._plot_pr_curve(metrics['pr_curve'], metrics.get('average_precision', 0))

        # 4. 指标分布图
        self._plot_metrics_distribution(metrics)

    def _plot_confusion_matrix(self, cm: List[List[int]], class_names: List[str]):
        """绘制混淆矩阵"""
        if not cm:
            return

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_roc_curve(self, roc_data: Dict[str, List], auc: float):
        """绘制ROC曲线"""
        plt.figure(figsize=(8, 6))
        plt.plot(roc_data['fpr'], roc_data['tpr'],
                label=f'ROC曲线 (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='随机分类')
        plt.xlabel('假正率 (FPR)')
        plt.ylabel('真正率 (TPR)')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_pr_curve(self, pr_data: Dict[str, List], ap: float):
        """绘制精确率-召回率曲线"""
        plt.figure(figsize=(8, 6))
        plt.plot(pr_data['recall'], pr_data['precision'],
                label=f'PR曲线 (AP = {ap:.4f})', linewidth=2)
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_metrics_distribution(self, metrics: Dict[str, Any]):
        """绘制指标分布图"""
        # 提取主要指标
        main_metrics = {
            '准确率': metrics.get('accuracy', 0),
            '精确率': metrics.get('precision_weighted', 0),
            '召回率': metrics.get('recall_weighted', 0),
            'F1分数': metrics.get('f1_weighted', 0),
        }

        if 'auc_roc' in metrics:
            main_metrics['AUC'] = metrics['auc_roc']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(main_metrics.keys(), main_metrics.values(),
                      color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'][:len(main_metrics)])

        # 添加数值标签
        for bar, value in zip(bars, main_metrics.values()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.title('模型评估指标')
        plt.ylabel('分数')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_csv_summary(self, metrics: Dict[str, Any]) -> Path:
        """生成CSV格式的指标摘要"""
        csv_path = self.output_dir / f'{self.model_name}_summary.csv'

        # 提取主要指标
        summary_data = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not key.startswith('confusion') and 'curve' not in key:
                summary_data.append({'Metric': key, 'Value': value})

        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)

        return csv_path

    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
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


def evaluate_model(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  output_dir: Union[str, Path],
                  model_name: str = "model",
                  class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    评估模型的便捷函数

    Args:
        model: 要评估的模型
        data_loader: 测试数据加载器
        device: 计算设备
        output_dir: 输出目录
        model_name: 模型名称
        class_names: 类别名称

    Returns:
        评估指标字典
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("🔍 正在评估模型...")

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为numpy数组
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    # 计算指标
    num_classes = len(np.unique(y_true))
    calculator = MetricsCalculator(num_classes, class_names)
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_proba)

    # 生成报告
    generator = ReportGenerator(output_dir, model_name)
    generator.generate_full_report(metrics, y_true, y_pred, y_proba, class_names)

    return metrics