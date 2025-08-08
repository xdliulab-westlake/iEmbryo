from config import *
from models import create_model

def perform_logistic_regression(model, X, ages, quality_labels, test_mae):
    """执行逻辑回归分析"""
    print("=== 逻辑回归分析 ===")
    
    # 预测BS分数
    print("预测BS分数...")
    batch_size = 16
    y_bs_pred_all = []
    
    for i in range(0, len(X), batch_size):
        batch_end = min(i + batch_size, len(X))
        batch_X = X[i:batch_end]
        batch_pred = model.predict(batch_X, verbose=0).flatten()
        y_bs_pred_all.extend(batch_pred)
    
    y_bs_pred_all = np.array(y_bs_pred_all)

    # 构建增强特征
    ages_array = np.array(ages)
    enhanced_features = np.column_stack([
        ages_array, y_bs_pred_all, ages_array ** 2, y_bs_pred_all ** 2,
        ages_array * y_bs_pred_all, np.log(ages_array + 1), 1 / (y_bs_pred_all + 1e-6)
    ])
    
    logic_y = np.array(quality_labels)

    # 清理无效数据
    valid_idx = np.isfinite(enhanced_features).all(axis=1) & np.isfinite(logic_y)
    enhanced_features = enhanced_features[valid_idx]
    logic_y = logic_y[valid_idx]
    ages_array = ages_array[valid_idx]
    y_bs_pred_all = y_bs_pred_all[valid_idx]
    
    print(f"有效样本数: {len(logic_y)}")
    print(f"特征维度: {enhanced_features.shape[1]}")

    # 特征标准化和训练
    scaler = StandardScaler()
    enhanced_features_scaled = scaler.fit_transform(enhanced_features)

    clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    clf.fit(enhanced_features_scaled, logic_y)

    # 预测和评估
    y_pred_cls = clf.predict(enhanced_features_scaled)
    y_pred_proba = clf.predict_proba(enhanced_features_scaled)[:, 1]
    
    logic_loss = log_loss(logic_y, y_pred_proba)
    logic_acc = accuracy_score(logic_y, y_pred_cls)
    auc_score = roc_auc_score(logic_y, y_pred_proba)

    print("=== 分类结果 ===")
    print(classification_report(logic_y, y_pred_cls, target_names=['Poor', 'Good']))
    print(f"准确率: {logic_acc:.4f}")
    print(f"对数损失: {logic_loss:.4f}")
    print(f"AUC分数: {auc_score:.4f}")

    # 特征重要性
    feature_names = ['Age', 'BS_pred', 'Age²', 'BS_pred²', 'Age×BS', 'ln(Age)', '1/BS']
    feature_importance = np.abs(clf.coef_[0])
    
    print("\n=== 特征重要性 ===")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")
    
    return {
        'logic_acc': logic_acc,
        'auc_score': auc_score,
        'logic_loss': logic_loss,
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'y_pred_proba': y_pred_proba,
        'logic_y': logic_y,
        'ages_array': ages_array,
        'y_bs_pred_all': y_bs_pred_all
    }

def create_visualizations(logic_results, test_mae):
    """创建可视化图表"""
    print("=== 创建可视化图表 ===")
    
    # 主分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC曲线
    fpr, tpr, _ = roc_curve(logic_results['logic_y'], logic_results['y_pred_proba'])
    axes[0, 0].plot(fpr, tpr, label=f'ROC曲线 (AUC = {logic_results["auc_score"]:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='随机猜测')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. 预测分布
    axes[0, 1].hist(logic_results['y_pred_proba'][logic_results['logic_y'] == 0], alpha=0.5, label='Poor', bins=20, density=True)
    axes[0, 1].hist(logic_results['y_pred_proba'][logic_results['logic_y'] == 1], alpha=0.5, label='Good', bins=20, density=True)
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Predicted Probability Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. 年龄vs预测BS分数散点图
    min_length = min(len(logic_results['ages_array']), len(logic_results['y_bs_pred_all']), len(logic_results['logic_y']))
    ages_plot = logic_results['ages_array'][:min_length]
    y_bs_plot = logic_results['y_bs_pred_all'][:min_length]
    logic_y_plot = logic_results['logic_y'][:min_length]
    
    scatter = axes[1, 0].scatter(ages_plot, y_bs_plot, c=logic_y_plot, cmap='RdYlBu', alpha=0.6)
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Predicted BS Score')
    axes[1, 0].set_title('Age vs Predicted BS Score')
    plt.colorbar(scatter, ax=axes[1, 0], label='Quality (0=Poor, 1=Good)')
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
    axes[1, 1].set_title('Performance Metrics Summary')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATHS['multi_model_analysis'], dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特征重要性柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.barh(logic_results['feature_names'], logic_results['feature_importance'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    for i, (bar, importance) in enumerate(zip(bars, logic_results['feature_importance'])):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Names', fontsize=12, fontweight='bold')
    plt.title('Logistic Regression Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATHS['feature_importance'], dpi=300, bbox_inches='tight')
    plt.show() 