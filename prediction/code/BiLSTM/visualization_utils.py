"""可视化工具模块 - 从BiLSTM.py中提取的可视化相关函数"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

def create_visualizations(logic_results, test_mae):
    """创建可视化图表"""
    print("=== 创建可视化图表 ===")
    
    # 主分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC曲线
    fpr, tpr, _ = roc_curve(logic_results['logic_y'], logic_results['y_pred_proba'])
    axes[0, 0].plot(fpr, tpr, label=f'ROC曲线 (AUC = {logic_results["auc_score"]:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='随机猜测')
    axes[0, 0].set_xlabel('假正率')
    axes[0, 0].set_ylabel('真正率')
    axes[0, 0].set_title('ROC曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. 预测分布
    axes[0, 1].hist(logic_results['y_pred_proba'][logic_results['logic_y'] == 0], 
                    alpha=0.5, label='Poor', bins=20, density=True)
    axes[0, 1].hist(logic_results['y_pred_proba'][logic_results['logic_y'] == 1], 
                    alpha=0.5, label='Good', bins=20, density=True)
    axes[0, 1].set_xlabel('预测概率')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('预测概率分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. 年龄vs预测BS分数散点图
    min_length = min(len(logic_results['ages_array']), len(logic_results['y_bs_pred_all']), len(logic_results['logic_y']))
    ages_plot = logic_results['ages_array'][:min_length]
    y_bs_plot = logic_results['y_bs_pred_all'][:min_length]
    logic_y_plot = logic_results['logic_y'][:min_length]
    
    scatter = axes[1, 0].scatter(ages_plot, y_bs_plot, c=logic_y_plot, cmap='RdYlBu', alpha=0.6)
    axes[1, 0].set_xlabel('年龄')
    axes[1, 0].set_ylabel('预测BS分数')
    axes[1, 0].set_title('年龄 vs 预测BS分数')
    plt.colorbar(scatter, ax=axes[1, 0], label='质量 (0=Poor, 1=Good)')
    axes[1, 0].grid(True)

    # 4. 性能指标总结
    performance_text = f"""
    测试MAE: {test_mae:.4f}
    逻辑回归准确率: {logic_results['logic_acc']:.4f}
    逻辑回归AUC: {logic_results['auc_score']:.4f}
    对数损失: {logic_results['logic_loss']:.4f}
    """
    axes[1, 1].text(0.1, 0.5, performance_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('性能指标总结')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("/liaozizhuo/scripts/multi_model_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特征重要性柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.barh(logic_results['feature_names'], logic_results['feature_importance'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    for i, (bar, importance) in enumerate(zip(bars, logic_results['feature_importance'])):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('特征重要性', fontsize=12, fontweight='bold')
    plt.ylabel('特征名称', fontsize=12, fontweight='bold')
    plt.title('逻辑回归特征重要性分析', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/liaozizhuo/scripts/feature_importance.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_cross_validation_results(cv_summary, fold_results, save_path="/liaozizhuo/scripts/cv_analysis.pdf"):
    """可视化交叉验证结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. MAE分布
    axes[0, 0].boxplot([fold_results['train_mae'], fold_results['val_mae'], fold_results['test_mae']], 
                       labels=['Train', 'Validation', 'Test'])
    axes[0, 0].set_title('MAE Distribution')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].grid(True)
    
    # 2. Loss分布
    axes[0, 1].boxplot([fold_results['train_loss'], fold_results['val_loss'], fold_results['test_loss']], 
                       labels=['Train', 'Validation', 'Test'])
    axes[0, 1].set_title('Loss Distribution')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # 3. 逻辑回归性能
    axes[0, 2].boxplot([fold_results['logic_acc'], fold_results['logic_auc']], 
                       labels=['Accuracy', 'AUC'])
    axes[0, 2].set_title('Logistic Regression Performance')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].grid(True)
    
    # 4. 性能指标趋势
    fold_numbers = range(1, len(fold_results['test_mae']) + 1)
    axes[1, 0].plot(fold_numbers, fold_results['test_mae'], 'o-', label='Test MAE')
    axes[1, 0].plot(fold_numbers, fold_results['logic_auc'], 's-', label='Logic AUC')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance Trend')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. 最佳epoch分布
    axes[1, 1].hist(fold_results['best_epochs'], bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Best Epoch')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Best Epoch Distribution')
    axes[1, 1].grid(True)
    
    # 6. 性能统计表
    stats_text = f"""
    Test MAE: {cv_summary['test_mae_mean']:.4f} ± {cv_summary['test_mae_std']:.4f}
    Logic AUC: {cv_summary['logic_auc_mean']:.4f} ± {cv_summary['logic_auc_std']:.4f}
    Logic Accuracy: {cv_summary['logic_acc_mean']:.4f} ± {cv_summary['logic_acc_std']:.4f}
    Best Epoch: {cv_summary['best_epochs_mean']:.1f} ± {cv_summary['best_epochs_std']:.1f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 2].set_title('Performance Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"交叉验证分析图表已保存至: {save_path}")
