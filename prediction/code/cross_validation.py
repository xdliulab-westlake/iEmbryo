from config import *
from models import create_model

def cross_validate_bilstm_model(X, y_bs, quality_labels, ages, feature_dim, n_folds=5, n_repeats=3):
    """
    对BiLSTM模型进行交叉验证评估
    """
    print(f"=== 开始 {n_folds}折交叉验证 (重复{n_repeats}次) ===")
    
    # 强制清理GPU内存
    tf.keras.backend.clear_session()
    gc.collect()
    
    # 验证并修正特征维度
    expected_input_dim = 11 * feature_dim
    if X.shape[2] != expected_input_dim:
        print(f"特征维度不匹配，正在修正...")
        print(f"期望: {expected_input_dim}, 实际: {X.shape[2]}")
        # 如果维度不匹配，重新reshape数据
        if X.shape[2] == feature_dim:
            # 数据是 (n_samples, 33, feature_dim)，需要reshape为 (n_samples, 33, 11*feature_dim)
            X = X.reshape(X.shape[0], 33, 11, feature_dim).reshape(X.shape[0], 33, -1)
        elif X.shape[2] == 11 * feature_dim:
            # 维度已经正确
            pass
        else:
            raise ValueError(f"无法自动修正特征维度: {X.shape[2]}")
    
    print(f"修正后数据形状: {X.shape}")
    
    # 使用分层K折交叉验证（基于质量标签）
    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    
    # 存储每折的结果
    fold_results = {
        'train_mae': [], 'val_mae': [], 'test_mae': [],
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'logic_acc': [], 'logic_auc': [], 'logic_loss': [],
        'best_epochs': []
    }
    
    # 存储每折的预测结果（用于集成）
    all_predictions = []
    all_true_labels = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(cv.split(X, quality_labels)):
        print(f"\n--- 第 {fold_idx + 1} 折 ---")
        
        # 强制清理GPU内存
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 进一步拆分训练集和验证集
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y_bs[train_val_idx], y_bs[test_idx]
        quality_train_val, quality_test = [quality_labels[i] for i in train_val_idx], [quality_labels[i] for i in test_idx]
        ages_train_val, ages_test = [ages[i] for i in train_val_idx], [ages[i] for i in test_idx]
        
        # 再次分层拆分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, 
            stratify=quality_train_val
        )
        
        # 获取对应的质量标签和年龄
        quality_train = [quality_train_val[i] for i in range(len(quality_train_val)) if i < len(X_train)]
        quality_val = [quality_train_val[i] for i in range(len(quality_train_val)) if i >= len(X_train)]
        ages_train = [ages_train_val[i] for i in range(len(ages_train_val)) if i < len(X_train)]
        ages_val = [ages_train_val[i] for i in range(len(ages_train_val)) if i >= len(X_train)]
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        # 创建和训练模型 - 使用更保守的内存策略
        try:
            # 尝试使用GPU策略
            strategy = tf.distribute.MirroredStrategy()
            print("使用GPU训练")
        except:
            # 如果GPU不可用，使用CPU
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
            print("使用CPU训练")
        
        with strategy.scope():
            model = create_model(feature_dim)
            
            # 回调函数
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)
            ]
            
            # 更保守的数据管道 - 使用更小的批量大小
            batch_size = 4  # 进一步减小批量大小
            
            # 使用CPU加载数据，避免GPU内存问题
            with tf.device('/cpu:0'):
                train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_ds = train_ds.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # 训练模型
            history = model.fit(
                train_ds, validation_data=val_ds, 
                epochs=20, callbacks=callbacks, verbose=0
            )
        
        # 评估模型 - 使用更小的批量
        def evaluate_in_batches(X, y, batch_size=4):  # 进一步减小批量大小
            losses = []
            maes = []
            for i in range(0, len(X), batch_size):
                batch_end = min(i + batch_size, len(X))
                batch_X = X[i:batch_end]
                batch_y = y[i:batch_end]
                batch_loss, batch_mae = model.evaluate(batch_X, batch_y, verbose=0)
                losses.append(batch_loss)
                maes.append(batch_mae)
            return np.mean(losses), np.mean(maes)
        
        train_loss, train_mae = evaluate_in_batches(X_train, y_train)
        val_loss, val_mae = evaluate_in_batches(X_val, y_val)
        test_loss, test_mae = evaluate_in_batches(X_test, y_test)
        
        # 预测BS分数 - 使用更小的批量
        y_bs_pred = []
        for i in range(0, len(X_test), 4):  # 进一步减小批量大小
            batch_end = min(i + 4, len(X_test))
            batch_X = X_test[i:batch_end]
            batch_pred = model.predict(batch_X, verbose=0).flatten()
            y_bs_pred.extend(batch_pred)
        y_bs_pred = np.array(y_bs_pred)
        
        # 构建增强特征用于逻辑回归 - 添加数据清理
        ages_test_array = np.array(ages_test)
        
        # 检查并清理无效数据
        valid_idx = np.isfinite(ages_test_array) & np.isfinite(y_bs_pred) & (y_bs_pred > 0)
        if not np.all(valid_idx):
            print(f"警告：发现 {np.sum(~valid_idx)} 个无效样本，已过滤")
            ages_test_array = ages_test_array[valid_idx]
            y_bs_pred = y_bs_pred[valid_idx]
            quality_test = [quality_test[i] for i in range(len(quality_test)) if valid_idx[i]]
        
        # 构建增强特征
        enhanced_features = np.column_stack([
            ages_test_array, y_bs_pred, ages_test_array ** 2, y_bs_pred ** 2,
            ages_test_array * y_bs_pred, np.log(ages_test_array + 1), 1 / (y_bs_pred + 1e-6)
        ])
        
        # 再次检查特征的有效性
        valid_feature_idx = np.isfinite(enhanced_features).all(axis=1)
        if not np.all(valid_feature_idx):
            print(f"警告：发现 {np.sum(~valid_feature_idx)} 个无效特征，已过滤")
            enhanced_features = enhanced_features[valid_feature_idx]
            quality_test = [quality_test[i] for i in range(len(quality_test)) if valid_feature_idx[i]]
        
        # 标准化特征
        scaler = StandardScaler()
        enhanced_features_scaled = scaler.fit_transform(enhanced_features)
        
        # 训练逻辑回归
        clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        clf.fit(enhanced_features_scaled, quality_test)
        
        # 逻辑回归预测
        y_pred_proba = clf.predict_proba(enhanced_features_scaled)[:, 1]
        logic_acc = accuracy_score(quality_test, clf.predict(enhanced_features_scaled))
        logic_auc = roc_auc_score(quality_test, y_pred_proba)
        logic_loss = log_loss(quality_test, y_pred_proba)
        
        # 记录结果
        fold_results['train_mae'].append(train_mae)
        fold_results['val_mae'].append(val_mae)
        fold_results['test_mae'].append(test_mae)
        fold_results['train_loss'].append(train_loss)
        fold_results['val_loss'].append(val_loss)
        fold_results['test_loss'].append(test_loss)
        fold_results['logic_acc'].append(logic_acc)
        fold_results['logic_auc'].append(logic_auc)
        fold_results['logic_loss'].append(logic_loss)
        fold_results['best_epochs'].append(np.argmin(history.history['val_loss']) + 1)
        
        # 存储预测结果
        all_predictions.append(y_pred_proba)
        all_true_labels.extend(quality_test)
        
        print(f"测试MAE: {test_mae:.4f}, 逻辑回归AUC: {logic_auc:.4f}")
        
        # 清理内存
        del model, train_ds, val_ds
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 强制等待一小段时间让GPU释放内存
        time.sleep(2)
    
    # 计算统计结果
    cv_summary = {}
    for metric in fold_results.keys():
        values = fold_results[metric]
        cv_summary[f'{metric}_mean'] = np.mean(values)
        cv_summary[f'{metric}_std'] = np.std(values)
        cv_summary[f'{metric}_min'] = np.min(values)
        cv_summary[f'{metric}_max'] = np.max(values)
    
    print(f"\n=== 交叉验证结果总结 ===")
    print(f"BiLSTM测试MAE: {cv_summary['test_mae_mean']:.4f} ± {cv_summary['test_mae_std']:.4f}")
    print(f"逻辑回归AUC: {cv_summary['logic_auc_mean']:.4f} ± {cv_summary['logic_auc_std']:.4f}")
    print(f"逻辑回归准确率: {cv_summary['logic_acc_mean']:.4f} ± {cv_summary['logic_acc_std']:.4f}")
    
    return cv_summary, fold_results, all_predictions, all_true_labels

def ensemble_predictions(all_predictions, all_true_labels, method='mean'):
    """
    集成多折交叉验证的预测结果
    
    Args:
        all_predictions: 每折的预测结果列表
        all_true_labels: 每折的真实标签列表
        method: 集成方法 ('mean', 'median', 'weighted')
    
    Returns:
        ensemble_pred: 集成预测结果
        ensemble_true: 对应的真实标签
    """
    # 检查所有预测结果的长度是否一致
    lengths = [len(pred) for pred in all_predictions]
    if len(set(lengths)) > 1:
        print(f"警告：不同折的预测结果长度不一致: {lengths}")
        print("无法进行集成，返回最后一折的结果")
        return all_predictions[-1], all_true_labels[-1]
    
    # 如果长度一致，进行集成
    if method == 'mean':
        ensemble_pred = np.mean(all_predictions, axis=0)
    elif method == 'median':
        ensemble_pred = np.median(all_predictions, axis=0)
    elif method == 'weighted':
        # 基于每折性能的加权平均
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 可以基于性能调整权重
        ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # 返回对应的真实标签（使用第一折的标签，因为所有折的标签应该是对应的）
    ensemble_true = all_true_labels[0]
    
    return ensemble_pred, ensemble_true

def plot_cross_validation_results(cv_summary, fold_results, save_path=None):
    """可视化交叉验证结果"""
    if save_path is None:
        save_path = OUTPUT_PATHS['cv_analysis']
        
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

def assess_model_stability(cv_summary):
    """
    评估模型稳定性
    
    Args:
        cv_summary: 交叉验证结果摘要
    
    Returns:
        dict: 稳定性评估结果
    """
    stability_metrics = {}
    
    # 计算变异系数 (CV = std/mean)
    stability_metrics['mae_cv'] = cv_summary['test_mae_std'] / cv_summary['test_mae_mean']
    stability_metrics['auc_cv'] = cv_summary['logic_auc_std'] / cv_summary['logic_auc_mean']
    stability_metrics['acc_cv'] = cv_summary['logic_acc_std'] / cv_summary['logic_acc_mean']
    
    # 稳定性评级
    def get_stability_rating(cv_value):
        if cv_value < 0.1:
            return "优秀"
        elif cv_value < 0.2:
            return "良好"
        elif cv_value < 0.3:
            return "一般"
        else:
            return "不稳定"
    
    stability_metrics['mae_stability'] = get_stability_rating(stability_metrics['mae_cv'])
    stability_metrics['auc_stability'] = get_stability_rating(stability_metrics['auc_cv'])
    stability_metrics['acc_stability'] = get_stability_rating(stability_metrics['acc_cv'])
    
    print(f"\n=== 模型稳定性评估 ===")
    print(f"MAE稳定性: {stability_metrics['mae_stability']} (CV={stability_metrics['mae_cv']:.3f})")
    print(f"AUC稳定性: {stability_metrics['auc_stability']} (CV={stability_metrics['auc_cv']:.3f})")
    print(f"准确率稳定性: {stability_metrics['acc_stability']} (CV={stability_metrics['acc_cv']:.3f})")
    
    return stability_metrics

def run_cross_validation(X, y_bs, quality_labels, ages, feature_dim):
    """运行交叉验证"""
    print("=== 交叉验证评估 ===")
    
    # 执行交叉验证
    cv_summary, fold_results, all_predictions, all_true_labels = cross_validate_bilstm_model(
        X, y_bs, quality_labels, ages, feature_dim, n_folds=5, n_repeats=1
    )
    
    # 评估模型稳定性
    stability_metrics = assess_model_stability(cv_summary)
    
    # 可视化交叉验证结果
    plot_cross_validation_results(cv_summary, fold_results)
    
    # 集成预测结果 - 修复版本
    try:
        ensemble_pred, ensemble_true = ensemble_predictions(all_predictions, all_true_labels, method='mean')
        ensemble_acc = accuracy_score(ensemble_true, ensemble_pred > 0.5)
        ensemble_auc = roc_auc_score(ensemble_true, ensemble_pred)
        
        print(f"\n=== 集成学习结果 ===")
        print(f"集成准确率: {ensemble_acc:.4f}")
        print(f"集成AUC: {ensemble_auc:.4f}")
        
        # 保存结果
        cv_results = {
            'cv_summary': cv_summary,
            'stability_metrics': stability_metrics,
            'ensemble_performance': {
                'accuracy': ensemble_acc,
                'auc': ensemble_auc
            }
        }
    except Exception as e:
        print(f"集成预测失败: {e}")
        print("跳过集成学习，仅保存交叉验证结果")
        
        # 保存结果
        cv_results = {
            'cv_summary': cv_summary,
            'stability_metrics': stability_metrics,
            'ensemble_performance': {
                'accuracy': None,
                'auc': None
            }
        }
    
    with open(OUTPUT_PATHS['cv_results'], 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    
    print(f"交叉验证结果已保存至: {OUTPUT_PATHS['cv_results']}")
    print(f"分析图表已保存至: {OUTPUT_PATHS['cv_analysis']}") 