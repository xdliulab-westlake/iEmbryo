"""
BiLSTM 交叉验证模块
包含各种交叉验证策略和模型稳定性评估
"""

import numpy as np
import pandas as pd
import time
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf

def cv_concat_metrics(all_pred_proba, all_labels):
    """
    Concatenate cross-validation metrics from all folds.
    
    Args:
        all_pred_proba: List of prediction probabilities from each fold
        all_labels: List of true labels from each fold
        
    Returns:
        dict: Concatenated metrics
    """
    # 合并所有预测和标签
    combined_proba = np.concatenate(all_pred_proba)
    combined_labels = np.concatenate(all_labels)
    
    # 计算整体指标
    combined_pred = (combined_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(combined_labels, combined_pred),
        'auc': roc_auc_score(combined_labels, combined_proba),
        'total_samples': len(combined_labels),
        'positive_samples': np.sum(combined_labels),
        'negative_samples': len(combined_labels) - np.sum(combined_labels)
    }
    
    return metrics

def ensemble_predictions(all_predictions, all_true_labels, method='mean'):
    """
    Ensemble predictions from multiple models or folds.
    
    Args:
        all_predictions: List of prediction arrays
        all_true_labels: List of true label arrays
        method: Ensemble method ('mean', 'median', 'vote')
        
    Returns:
        dict: Ensemble results
    """
    # 确保所有预测都是numpy数组
    predictions = [np.array(pred) for pred in all_predictions]
    true_labels = [np.array(label) for label in all_true_labels]
    
    # 检查形状一致性
    if len(set(pred.shape for pred in predictions)) > 1:
        print("警告: 预测数组形状不一致")
        return None
    
    # 合并预测
    stacked_predictions = np.stack(predictions, axis=0)
    
    if method == 'mean':
        ensemble_pred = np.mean(stacked_predictions, axis=0)
    elif method == 'median':
        ensemble_pred = np.median(stacked_predictions, axis=0)
    elif method == 'vote':
        # 硬投票（需要二进制预测）
        ensemble_pred = np.round(np.mean(stacked_predictions, axis=0))
    else:
        raise ValueError(f"不支持的集成方法: {method}")
    
    # 合并真实标签
    combined_true = np.concatenate(true_labels)
    
    return {
        'ensemble_predictions': ensemble_pred,
        'combined_true_labels': combined_true,
        'individual_predictions': predictions,
        'method': method
    }

def assess_model_stability(cv_summary):
    """
    Assess model stability across cross-validation folds.
    
    Args:
        cv_summary: Cross-validation summary statistics
        
    Returns:
        dict: Stability assessment metrics
    """
    stability_metrics = {}
    
    for metric_name, stats in cv_summary.items():
        if isinstance(stats, dict) and 'std' in stats:
            mean_val = stats.get('mean', 0)
            std_val = stats.get('std', 0)
            
            # 计算变异系数 (CV = std/mean)
            cv_coeff = std_val / mean_val if mean_val != 0 else float('inf')
            
            stability_metrics[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'cv_coefficient': cv_coeff,
                'stability_grade': get_stability_grade(cv_coeff)
            }
    
    # 整体稳定性评估
    cv_coefficients = [m['cv_coefficient'] for m in stability_metrics.values() 
                       if m['cv_coefficient'] != float('inf')]
    
    if cv_coefficients:
        avg_cv_coeff = np.mean(cv_coefficients)
        overall_stability = get_stability_grade(avg_cv_coeff)
    else:
        avg_cv_coeff = float('inf')
        overall_stability = '未知'
    
    stability_metrics['overall'] = {
        'average_cv_coefficient': avg_cv_coeff,
        'stability_grade': overall_stability,
        'assessment': get_stability_assessment(overall_stability)
    }
    
    return stability_metrics

def get_stability_grade(cv_coefficient):
    """
    Get stability grade based on coefficient of variation.
    
    Args:
        cv_coefficient: Coefficient of variation
        
    Returns:
        str: Stability grade
    """
    if cv_coefficient <= 0.05:
        return '优秀'
    elif cv_coefficient <= 0.10:
        return '良好'
    elif cv_coefficient <= 0.15:
        return '一般'
    elif cv_coefficient <= 0.25:
        return '较差'
    else:
        return '很差'

def get_stability_assessment(stability_grade):
    """
    Get detailed stability assessment.
    
    Args:
        stability_grade: Stability grade
        
    Returns:
        str: Detailed assessment
    """
    assessments = {
        '优秀': '模型非常稳定，各折间性能差异很小',
        '良好': '模型稳定性良好，各折间性能基本一致',
        '一般': '模型稳定性一般，各折间有一定差异',
        '较差': '模型稳定性较差，各折间性能差异较大',
        '很差': '模型稳定性很差，各折间性能差异很大',
        '未知': '无法评估模型稳定性'
    }
    
    return assessments.get(stability_grade, '未知')

def stratified_cross_validation(X, y, quality_labels, n_folds=4, random_state=42):
    """
    Perform stratified cross-validation split.
    
    Args:
        X: Features
        y: Targets  
        quality_labels: Labels for stratification
        n_folds: Number of folds
        random_state: Random state for reproducibility
        
    Returns:
        Generator: Cross-validation fold indices
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, quality_labels)):
        yield fold_idx, train_idx, val_idx

def perform_nested_cross_validation(X, y, quality_labels, outer_folds=4, inner_folds=3):
    """
    Perform nested cross-validation for hyperparameter tuning.
    
    Args:
        X: Features
        y: Targets
        quality_labels: Labels for stratification
        outer_folds: Number of outer folds
        inner_folds: Number of inner folds
        
    Returns:
        dict: Nested CV results
    """
    from .model import create_model, train_model
    from .utils import cleanup_memory
    
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=43)
    
    outer_scores = []
    best_params_per_fold = []
    
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, quality_labels)):
        print(f"\n=== 外层第 {outer_fold + 1} 折 ===")
        
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        quality_outer_train = [quality_labels[i] for i in outer_train_idx]
        
        # 内层交叉验证进行超参数调优
        best_score = -float('inf')
        best_params = None
        
        # 简单的网格搜索
        param_grid = {
            'lstm_units': [64, 128, 256],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005, 0.002]
        }
        
        for lstm_units in param_grid['lstm_units']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    
                    inner_scores = []
                    
                    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train, quality_outer_train)):
                        # 清理内存
                        cleanup_memory()
                        
                        X_inner_train = X_outer_train[inner_train_idx]
                        X_inner_val = X_outer_train[inner_val_idx]
                        y_inner_train = y_outer_train[inner_train_idx]
                        y_inner_val = y_outer_train[inner_val_idx]
                        
                        # 训练模型
                        try:
                            model, _ = train_model(
                                X_inner_train, y_inner_train,
                                X_inner_val, y_inner_val,
                                feature_dim=X.shape[-1],
                                save_path=None,
                                lstm_units=lstm_units,
                                dropout_rate=dropout_rate,
                                learning_rate=learning_rate,
                                epochs=50,  # 减少epoch以加速搜索
                                patience=10,
                                verbose=0
                            )
                            
                            # 评估
                            val_loss = model.evaluate(X_inner_val, y_inner_val, verbose=0)[0]
                            inner_scores.append(-val_loss)  # 负损失作为分数
                            
                        except Exception as e:
                            print(f"内层训练失败: {e}")
                            inner_scores.append(-float('inf'))
                    
                    # 计算平均分数
                    avg_score = np.mean(inner_scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'lstm_units': lstm_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate
                        }
        
        print(f"最佳参数: {best_params}")
        best_params_per_fold.append(best_params)
        
        # 用最佳参数在外层测试集上评估
        cleanup_memory()
        
        # 分割外层训练集为训练和验证
        X_train_outer, X_val_outer, y_train_outer, y_val_outer = train_test_split(
            X_outer_train, y_outer_train, test_size=0.2, random_state=42
        )
        
        final_model, _ = train_model(
            X_train_outer, y_train_outer,
            X_val_outer, y_val_outer,
            feature_dim=X.shape[-1],
            save_path=None,
            **best_params,
            epochs=100,
            patience=15,
            verbose=0
        )
        
        # 在外层测试集上评估
        test_loss = final_model.evaluate(X_outer_test, y_outer_test, verbose=0)[0]
        outer_scores.append(test_loss)
        
        print(f"外层第 {outer_fold + 1} 折测试损失: {test_loss:.4f}")
    
    results = {
        'outer_scores': outer_scores,
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'best_params_per_fold': best_params_per_fold,
        'overall_best_params': get_most_common_params(best_params_per_fold)
    }
    
    return results

def get_most_common_params(params_list):
    """
    Get most common parameter combination.
    
    Args:
        params_list: List of parameter dictionaries
        
    Returns:
        dict: Most common parameters
    """
    # 简化实现：返回众数参数
    param_keys = params_list[0].keys()
    most_common = {}
    
    for key in param_keys:
        values = [params[key] for params in params_list]
        most_common[key] = max(set(values), key=values.count)
    
    return most_common

def cross_validate_with_time_tracking(cv_function, *args, **kwargs):
    """
    Wrapper function to track cross-validation execution time.
    
    Args:
        cv_function: Cross-validation function to execute
        *args, **kwargs: Arguments for the CV function
        
    Returns:
        tuple: (cv_results, timing_info)
    """
    start_time = time.time()
    
    # 执行交叉验证
    cv_results = cv_function(*args, **kwargs)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    timing_info = {
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'total_time_hours': total_time / 3600,
        'average_time_per_fold': total_time / kwargs.get('n_folds', 4)
    }
    
    print(f"\n=== 交叉验证时间统计 ===")
    print(f"总耗时: {timing_info['total_time_hours']:.2f}小时 ({timing_info['total_time_minutes']:.1f}分钟)")
    print(f"平均每折耗时: {timing_info['average_time_per_fold']/60:.1f}分钟")
    
    return cv_results, timing_info

def validate_cv_setup(X, y, quality_labels, n_folds=4):
    """
    Validate cross-validation setup and data consistency.
    
    Args:
        X: Features
        y: Targets
        quality_labels: Quality labels
        n_folds: Number of folds
        
    Returns:
        bool: True if setup is valid
    """
    print("=== 交叉验证设置验证 ===")
    
    # 检查数据形状一致性
    if len(X) != len(y) or len(X) != len(quality_labels):
        print(f"❌ 数据形状不一致: X={len(X)}, y={len(y)}, labels={len(quality_labels)}")
        return False
    
    # 检查类别分布
    unique_labels = np.unique(quality_labels)
    print(f"类别标签: {unique_labels}")
    
    for label in unique_labels:
        count = np.sum(np.array(quality_labels) == label)
        if count < n_folds:
            print(f"❌ 类别 {label} 样本数 ({count}) 少于折数 ({n_folds})")
            return False
        print(f"类别 {label}: {count} 个样本")
    
    # 检查数据中的NaN和Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("❌ X中包含NaN或Inf值")
        return False
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("❌ y中包含NaN或Inf值")
        return False
    
    print("✅ 交叉验证设置验证通过")
    return True

def create_cv_summary_report(cv_results, save_path=None):
    """
    Create comprehensive cross-validation summary report.
    
    Args:
        cv_results: Cross-validation results
        save_path: Path to save the report
        
    Returns:
        str: Summary report
    """
    report_lines = [
        "=" * 60,
        "交叉验证总结报告",
        "=" * 60,
        ""
    ]
    
    if 'fold_results' in cv_results:
        fold_results = cv_results['fold_results']
        n_folds = len(fold_results)
        
        report_lines.extend([
            f"交叉验证折数: {n_folds}",
            f"每折样本统计:",
            ""
        ])
        
        # 每折详细信息
        for i, fold in enumerate(fold_results):
            report_lines.append(f"第 {i+1} 折:")
            if 'logic_acc' in fold:
                report_lines.append(f"  逻辑回归准确率: {fold['logic_acc']:.4f}")
            if 'logic_auc' in fold:
                report_lines.append(f"  逻辑回归AUC: {fold['logic_auc']:.4f}")
            if 'test_mae' in fold:
                report_lines.append(f"  BiLSTM测试MAE: {fold['test_mae']:.4f}")
            report_lines.append("")
    
    if 'cv_summary' in cv_results:
        cv_summary = cv_results['cv_summary']
        report_lines.extend([
            "平均性能指标:",
            "-" * 30
        ])
        
        for metric, stats in cv_summary.items():
            if isinstance(stats, dict) and 'mean' in stats:
                report_lines.append(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        report_lines.append("")
    
    # 模型稳定性评估
    if 'cv_summary' in cv_results:
        stability = assess_model_stability(cv_results['cv_summary'])
        if 'overall' in stability:
            report_lines.extend([
                "模型稳定性评估:",
                "-" * 30,
                f"整体稳定性等级: {stability['overall']['stability_grade']}",
                f"评估结果: {stability['overall']['assessment']}",
                ""
            ])
    
    report_lines.extend([
        "=" * 60,
        "报告生成完成",
        "=" * 60
    ])
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"交叉验证报告已保存: {save_path}")
    
    return report_text

def stratified_cross_validation_with_cache(
    cache, n_folds=4, test_ratio=0.3, feature_dim=512, lstm_units=128, 
    dropout_rate=0.3, learning_rate=0.001, epochs=100, batch_size=32, 
    patience=15, use_cache=True, use_augmentation=True, verbose=1, random_state=42
):
    """
    执行完整的分层交叉验证训练流程，使用特征缓存系统
    
    Args:
        cache: BiLSTMFeatureCache实例
        n_folds: 交叉验证折数
        test_ratio: 测试集比例
        feature_dim: 特征维度
        lstm_units: LSTM单元数
        dropout_rate: Dropout比率
        learning_rate: 学习率
        epochs: 训练轮数
        batch_size: 批量大小
        patience: 早停耐心值
        use_cache: 是否使用缓存
        use_augmentation: 是否使用数据增强
        verbose: 详细输出级别
        random_state: 随机种子
        
    Returns:
        dict: 交叉验证结果
    """
    print(f"🔍 开始数据准备...")
    
    try:
        # 导入必要的模块
        from .utils import load_score_dict
        from .model import create_model, train_model
        from .logistic import perform_logistic_regression
        import os
        import json
        
        # 加载评分数据
        score_dict_path = os.path.join(cache.data_root, "score_dict.json")
        
        if not os.path.exists(score_dict_path):
            raise FileNotFoundError(f"评分文件不存在: {score_dict_path}")
            
        with open(score_dict_path, 'r') as f:
            score_data = json.load(f)
        
        print(f"✅ 加载评分数据: {len(score_data)} 个胚胎")
        
        # 获取图像路径列表
        image_folder = os.path.join(cache.data_root, "image")
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"图像文件夹不存在: {image_folder}")
            
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        available_images = [f for f in image_files if f in score_data]
        
        if not available_images:
            raise ValueError("没有找到匹配的图像和评分数据")
            
        print(f"✅ 找到匹配的图像: {len(available_images)} 个")
        
        # 构建完整路径
        image_paths = [os.path.join(image_folder, img) for img in available_images]
        
        # 提取特征
        print(f"🔄 开始特征提取...")
        features_dict = cache.get_or_compute_features(
            image_paths=image_paths,
            use_cache=use_cache,
            training=True,
            use_augmentation=use_augmentation
        )
        
        # 准备训练数据
        features_list = []
        bs_scores = []
        ages = []
        qualities = []
        
        for img_file in available_images:
            img_path = os.path.join(image_folder, img_file)
            
            if img_path in features_dict and img_file in score_data:
                features_list.append(features_dict[img_path])
                
                data = score_data[img_file]
                bs_scores.append(data['bs_score'])
                ages.append(data['age'])
                qualities.append(data['quality'])
        
        # 转换为numpy数组
        X = np.array(features_list)
        y_bs = np.array(bs_scores)
        y_quality = np.array(qualities)
        ages_array = np.array(ages)
        
        print(f"✅ 数据准备完成:")
        print(f"  特征矩阵: {X.shape}")
        print(f"  BS分数范围: {y_bs.min():.2f} - {y_bs.max():.2f}")
        print(f"  高质量胚胎: {np.sum(y_quality)} / {len(y_quality)} ({np.mean(y_quality)*100:.1f}%)")
        
        # 验证数据
        if len(X) < n_folds:
            raise ValueError(f"样本数({len(X)})少于折数({n_folds})")
            
        # 执行分层交叉验证
        from sklearn.model_selection import StratifiedKFold
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        fold_results = []
        all_predictions = []
        all_labels = []
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(cv.split(X, y_quality)):
            print(f"\n=== 第 {fold_idx + 1} 折交叉验证 ===")
            
            # 分割数据
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_bs_train_val, y_bs_test = y_bs[train_val_idx], y_bs[test_idx]
            y_quality_train_val, y_quality_test = y_quality[train_val_idx], y_quality[test_idx]
            ages_train_val, ages_test = ages_array[train_val_idx], ages_array[test_idx]
            
            # 再次分割训练和验证
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_bs_train, y_bs_val = train_test_split(
                X_train_val, y_bs_train_val, test_size=0.2, random_state=random_state
            )
            y_quality_train, y_quality_val = train_test_split(
                y_quality_train_val, test_size=0.2, random_state=random_state
            )[1], train_test_split(
                y_quality_train_val, test_size=0.2, random_state=random_state
            )[0]
            ages_train, ages_val = train_test_split(
                ages_train_val, test_size=0.2, random_state=random_state
            )[1], train_test_split(
                ages_train_val, test_size=0.2, random_state=random_state
            )[0]
            
            print(f"📊 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
            
            # 训练BiLSTM模型
            print(f"🧠 训练BiLSTM模型...")
            try:
                bilstm_model, history = train_model(
                    X_train.reshape(-1, 1, feature_dim), y_bs_train,
                    X_val.reshape(-1, 1, feature_dim), y_bs_val,
                    feature_dim=feature_dim,
                    lstm_units=lstm_units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    verbose=max(0, verbose-1),
                    save_path=None
                )
                
                # 预测BS分数
                bs_pred_train = bilstm_model.predict(X_train.reshape(-1, 1, feature_dim), verbose=0).flatten()
                bs_pred_val = bilstm_model.predict(X_val.reshape(-1, 1, feature_dim), verbose=0).flatten()
                bs_pred_test = bilstm_model.predict(X_test.reshape(-1, 1, feature_dim), verbose=0).flatten()
                
                # 计算BiLSTM指标
                from sklearn.metrics import mean_absolute_error
                train_mae = mean_absolute_error(y_bs_train, bs_pred_train)
                val_mae = mean_absolute_error(y_bs_val, bs_pred_val)
                test_mae = mean_absolute_error(y_bs_test, bs_pred_test)
                
                print(f"📊 BiLSTM MAE - 训练: {train_mae:.4f}, 验证: {val_mae:.4f}, 测试: {test_mae:.4f}")
                
                # 逻辑回归分类
                print(f"📊 训练逻辑回归分类器...")
                
                # 准备逻辑回归特征
                logistic_train_features = np.column_stack([bs_pred_train, ages_train])
                logistic_val_features = np.column_stack([bs_pred_val, ages_val])
                logistic_test_features = np.column_stack([bs_pred_test, ages_test])
                
                # 训练逻辑回归
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import accuracy_score, roc_auc_score
                
                scaler = StandardScaler()
                logistic_train_scaled = scaler.fit_transform(logistic_train_features)
                logistic_val_scaled = scaler.transform(logistic_val_features)
                logistic_test_scaled = scaler.transform(logistic_test_features)
                
                lr_model = LogisticRegression(class_weight='balanced', random_state=random_state)
                lr_model.fit(logistic_train_scaled, y_quality_train)
                
                # 预测和评估
                quality_pred_test = lr_model.predict(logistic_test_scaled)
                quality_prob_test = lr_model.predict_proba(logistic_test_scaled)[:, 1]
                
                test_acc = accuracy_score(y_quality_test, quality_pred_test)
                test_auc = roc_auc_score(y_quality_test, quality_prob_test) if len(np.unique(y_quality_test)) > 1 else 0.5
                
                print(f"📊 逻辑回归 - 准确率: {test_acc:.4f}, AUC: {test_auc:.4f}")
                
                # 保存结果
                fold_result = {
                    'fold': fold_idx + 1,
                    'bilstm_train_mae': train_mae,
                    'bilstm_val_mae': val_mae,
                    'bilstm_test_mae': test_mae,
                    'logic_accuracy': test_acc,
                    'logic_auc': test_auc,
                    'test_samples': len(X_test)
                }
                
                fold_results.append(fold_result)
                all_predictions.extend(quality_prob_test)
                all_labels.extend(y_quality_test)
                
            except Exception as e:
                print(f"❌ 第{fold_idx + 1}折训练失败: {e}")
                continue
        
        # 计算平均性能
        if fold_results:
            avg_performance = {
                'accuracy': np.mean([r['logic_accuracy'] for r in fold_results]),
                'accuracy_std': np.std([r['logic_accuracy'] for r in fold_results]),
                'auc': np.mean([r['logic_auc'] for r in fold_results]),
                'auc_std': np.std([r['logic_auc'] for r in fold_results])
            }
            
            bilstm_avg = {
                'mae': np.mean([r['bilstm_test_mae'] for r in fold_results]),
                'mae_std': np.std([r['bilstm_test_mae'] for r in fold_results])
            }
            
            print(f"\n🎉 交叉验证完成！")
            print(f"平均准确率: {avg_performance['accuracy']:.4f} ± {avg_performance['accuracy_std']:.4f}")
            print(f"平均AUC: {avg_performance['auc']:.4f} ± {avg_performance['auc_std']:.4f}")
            print(f"平均MAE: {bilstm_avg['mae']:.4f} ± {bilstm_avg['mae_std']:.4f}")
            
            return {
                'avg_performance': avg_performance,
                'bilstm_avg': bilstm_avg,
                'fold_performances': fold_results,
                'n_folds': len(fold_results),
                'total_samples': len(X)
            }
        else:
            raise RuntimeError("所有折都训练失败")
            
    except Exception as e:
        print(f"❌ 交叉验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise