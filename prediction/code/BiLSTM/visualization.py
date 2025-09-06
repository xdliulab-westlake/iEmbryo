"""
BiLSTM 可视化模块
包含训练结果可视化、性能分析图表等功能
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history, save_path=None):
    """
    Plot training history including loss and metrics.
    
    Args:
        history: Keras training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲线
    axes[0, 0].plot(history.history['loss'], label='训练损失', color='blue')
    axes[0, 0].plot(history.history['val_loss'], label='验证损失', color='red')
    axes[0, 0].set_title('模型损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE曲线
    if 'mae' in history.history:
        axes[0, 1].plot(history.history['mae'], label='训练MAE', color='blue')
        axes[0, 1].plot(history.history['val_mae'], label='验证MAE', color='red')
        axes[0, 1].set_title('平均绝对误差')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # MSE曲线
    if 'mse' in history.history:
        axes[1, 0].plot(history.history['mse'], label='训练MSE', color='blue')
        axes[1, 0].plot(history.history['val_mse'], label='验证MSE', color='red')
        axes[1, 0].set_title('均方误差')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线（如果可用）
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='学习率', color='green')
        axes[1, 1].set_title('学习率变化')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存: {save_path}")
    
    plt.show()

def plot_prediction_scatter(y_true, y_pred, title="预测vs真实值", save_path=None):
    """
    Create scatter plot of predictions vs true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # 完美预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    
    # 计算R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 计算MAE和RMSE
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 添加统计信息
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"散点图已保存: {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_prob, title="ROC曲线", save_path=None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正例率 (FPR)')
    ax.set_ylabel('真正例率 (TPR)')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="混淆矩阵", save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # 添加文本注释
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['负例', '正例'])
    ax.set_yticklabels(['负例', '正例'])
    
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    
    plt.show()

def plot_cross_validation_results(cv_summary, fold_results, save_path=None):
    """
    Plot comprehensive cross-validation results.
    
    Args:
        cv_summary: Cross-validation summary statistics
        fold_results: Individual fold results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 准确率分布
    if 'logic_acc' in fold_results:
        axes[0, 0].boxplot(fold_results['logic_acc'], labels=['准确率'])
        axes[0, 0].set_title('逻辑回归准确率分布')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].grid(True, alpha=0.3)
    
    # AUC分布
    if 'logic_auc' in fold_results:
        axes[0, 1].boxplot(fold_results['logic_auc'], labels=['AUC'])
        axes[0, 1].set_title('逻辑回归AUC分布')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].grid(True, alpha=0.3)
    
    # MAE分布
    mae_data = []
    mae_labels = []
    if 'train_mae' in fold_results:
        mae_data.append(fold_results['train_mae'])
        mae_labels.append('训练')
    if 'val_mae' in fold_results:
        mae_data.append(fold_results['val_mae'])
        mae_labels.append('验证')
    if 'test_mae' in fold_results:
        mae_data.append(fold_results['test_mae'])
        mae_labels.append('测试')
    
    if mae_data:
        axes[0, 2].boxplot(mae_data, labels=mae_labels)
        axes[0, 2].set_title('BiLSTM MAE分布')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 训练epoch分布
    if 'best_epochs' in fold_results:
        axes[1, 0].hist(fold_results['best_epochs'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('最佳训练Epoch分布')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 性能指标对比
    metrics_names = []
    metrics_values = []
    
    if cv_summary:
        for metric, stats in cv_summary.items():
            if isinstance(stats, dict) and 'mean' in stats:
                metrics_names.append(metric)
                metrics_values.append(stats['mean'])
    
    if metrics_names:
        axes[1, 1].bar(metrics_names, metrics_values, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('平均性能指标')
        axes[1, 1].set_ylabel('数值')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    # 折间稳定性
    if 'logic_acc' in fold_results:
        fold_indices = range(1, len(fold_results['logic_acc']) + 1)
        axes[1, 2].plot(fold_indices, fold_results['logic_acc'], 'o-', label='准确率', linewidth=2, markersize=6)
        if 'logic_auc' in fold_results:
            axes[1, 2].plot(fold_indices, fold_results['logic_auc'], 's-', label='AUC', linewidth=2, markersize=6)
        axes[1, 2].set_title('各折性能稳定性')
        axes[1, 2].set_xlabel('折数')
        axes[1, 2].set_ylabel('性能指标')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"交叉验证结果图已保存: {save_path}")
    
    plt.show()

def create_cv_visualization(avg_performance, bilstm_avg, fold_performances):
    """
    Create simplified cross-validation visualization.
    
    Args:
        avg_performance: Average performance metrics
        bilstm_avg: BiLSTM average metrics
        fold_performances: Individual fold performances
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 逻辑回归性能
    if avg_performance:
        metrics = ['accuracy', 'auc', 'precision', 'recall']
        values = [avg_performance.get(m, 0) for m in metrics]
        
        bars = axes[0].bar(metrics, values, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0].set_title('逻辑回归平均性能')
        axes[0].set_ylabel('数值')
        axes[0].set_ylim(0, 1)
        
        # 在柱子上添加数值
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
    
    # BiLSTM性能
    if bilstm_avg:
        metrics = list(bilstm_avg.keys())
        values = list(bilstm_avg.values())
        
        bars = axes[1].bar(metrics, values, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].set_title('BiLSTM平均性能')
        axes[1].set_ylabel('MAE')
        
        # 在柱子上添加数值
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
    
    # 折间稳定性
    if fold_performances:
        fold_nums = list(range(1, len(fold_performances) + 1))
        
        if 'logic_acc' in fold_performances[0]:
            accuracies = [perf['logic_acc'] for perf in fold_performances]
            axes[2].plot(fold_nums, accuracies, 'o-', label='准确率', linewidth=2, markersize=6)
        
        if 'logic_auc' in fold_performances[0]:
            aucs = [perf['logic_auc'] for perf in fold_performances]
            axes[2].plot(fold_nums, aucs, 's-', label='AUC', linewidth=2, markersize=6)
        
        axes[2].set_title('各折性能稳定性')
        axes[2].set_xlabel('折数')
        axes[2].set_ylabel('性能指标')
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(coefficients, feature_names, title="特征重要性", save_path=None):
    """
    Plot feature importance from logistic regression coefficients.
    
    Args:
        coefficients: Model coefficients
        feature_names: List of feature names
        title: Plot title
        save_path: Path to save the plot
    """
    abs_coef = np.abs(coefficients)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(feature_names, abs_coef, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('绝对系数值')
    ax.set_title(title)
    
    # 在柱子末端添加数值
    for i, (bar, coef) in enumerate(zip(bars, coefficients)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{coef:.4f}', ha='left', va='center')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存: {save_path}")
    
    plt.show()

def create_comprehensive_report(results, save_path=None):
    """
    Create comprehensive visualization report.
    
    Args:
        results: Complete results dictionary
        save_path: Path to save the report
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 创建网格布局
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. ROC曲线
    if 'test_metrics' in results and 'probabilities' in results['test_metrics']:
        ax1 = fig.add_subplot(gs[0, 0])
        y_true = results['test_metrics']['true_labels']
        y_prob = results['test_metrics']['probabilities']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_title('ROC曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 混淆矩阵
    if 'test_metrics' in results:
        ax2 = fig.add_subplot(gs[0, 1])
        y_true = results['test_metrics']['true_labels']
        y_pred = results['test_metrics']['predictions']
        cm = confusion_matrix(y_true, y_pred)
        im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
        ax2.figure.colorbar(im, ax=ax2)
        
        # 添加文本注释
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['负例', '正例'])
        ax2.set_yticklabels(['负例', '正例'])
        ax2.set_title('混淆矩阵')
    
    # 3. 特征重要性
    if 'feature_importance' in results:
        ax3 = fig.add_subplot(gs[0, 2:])
        importance = results['feature_importance']
        features = ['BS分数', '母体年龄']
        coefs = [importance['bs_score_coeff'], importance['age_coeff']]
        bars = ax3.bar(features, np.abs(coefs), alpha=0.7)
        ax3.set_title('特征重要性')
        ax3.set_ylabel('绝对系数值')
        for bar, coef in zip(bars, coefs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{coef:.3f}', ha='center', va='bottom')
    
    # 4-6. 性能指标对比
    metrics_sections = ['train_metrics', 'val_metrics', 'test_metrics']
    titles = ['训练集性能', '验证集性能', '测试集性能']
    
    for i, (section, title) in enumerate(zip(metrics_sections, titles)):
        if section in results:
            ax = fig.add_subplot(gs[1, i])
            metrics = results[section]
            metric_names = ['accuracy', 'auc', 'precision', 'recall']
            values = [metrics.get(m, 0) for m in metric_names]
            bars = ax.bar(metric_names, values, alpha=0.7)
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"综合报告已保存: {save_path}")
    
    plt.show()