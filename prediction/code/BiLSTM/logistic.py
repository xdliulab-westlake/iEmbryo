"""
BiLSTM 逻辑回归模块
包含逻辑回归特征构建、训练、评估等功能
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                            recall_score, classification_report, log_loss, roc_curve)
import warnings
warnings.filterwarnings('ignore')

def build_logistic_features(ages, y_bs_pred):
    """
    Build features for logistic regression classifier.
    
    Args:
        ages: Maternal ages
        y_bs_pred: Predicted blastocyst scores from BiLSTM
        
    Returns:
        numpy.ndarray: Combined features for logistic regression
    """
    # 确保输入是numpy数组
    ages = np.array(ages).reshape(-1, 1)
    y_bs_pred = np.array(y_bs_pred).reshape(-1, 1)
    
    # 组合特征：预测的BS分数 + 母体年龄
    features = np.hstack([y_bs_pred, ages])
    
    print(f"逻辑回归特征构建完成: {features.shape}")
    return features

def filter_valid_features(features, quality_labels):
    """
    Filter out invalid features and labels.
    
    Args:
        features: Feature array
        quality_labels: Quality labels
        
    Returns:
        tuple: (filtered_features, filtered_labels)
    """
    # 转换为numpy数组
    features = np.array(features)
    quality_labels = np.array(quality_labels)
    
    # 检查NaN值
    valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(quality_labels))
    
    # 检查无限值
    valid_mask &= ~(np.isinf(features).any(axis=1) | np.isinf(quality_labels))
    
    features_clean = features[valid_mask]
    quality_filtered = quality_labels[valid_mask]
    
    print(f"特征过滤: {len(features)} -> {len(features_clean)} (移除 {len(features) - len(features_clean)} 个无效样本)")
    
    return features_clean, quality_filtered

def train_logistic_classifier(features_clean, quality_filtered, random_state=42, 
                             max_iter=1000, class_weight='balanced'):
    """
    Train logistic regression classifier.
    
    Args:
        features_clean: Clean feature array
        quality_filtered: Filtered quality labels
        random_state: Random state for reproducibility
        max_iter: Maximum iterations
        class_weight: Class weight strategy
        
    Returns:
        tuple: (classifier, scaler, training_metrics)
    """
    print("训练逻辑回归分类器...")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    
    # 训练逻辑回归
    clf = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight=class_weight,
        solver='lbfgs'  # 适合小数据集
    )
    
    clf.fit(features_scaled, quality_filtered)
    
    # 计算训练集性能
    train_pred = clf.predict(features_scaled)
    train_pred_proba = clf.predict_proba(features_scaled)[:, 1]
    
    training_metrics = {
        'accuracy': accuracy_score(quality_filtered, train_pred),
        'auc': roc_auc_score(quality_filtered, train_pred_proba),
        'precision': precision_score(quality_filtered, train_pred),
        'recall': recall_score(quality_filtered, train_pred),
        'log_loss': log_loss(quality_filtered, train_pred_proba)
    }
    
    print(f"逻辑回归训练完成:")
    print(f"  训练准确率: {training_metrics['accuracy']:.4f}")
    print(f"  训练AUC: {training_metrics['auc']:.4f}")
    
    return clf, scaler, training_metrics

def evaluate_logistic_classifier(clf, scaler, features_clean, quality_filtered):
    """
    Evaluate logistic regression classifier.
    
    Args:
        clf: Trained classifier
        scaler: Fitted scaler
        features_clean: Clean features
        quality_filtered: Filtered labels
        
    Returns:
        dict: Evaluation metrics
    """
    # 标准化特征
    features_scaled = scaler.transform(features_clean)
    
    # 预测
    pred = clf.predict(features_scaled)
    pred_proba = clf.predict_proba(features_scaled)[:, 1]
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(quality_filtered, pred),
        'auc': roc_auc_score(quality_filtered, pred_proba),
        'precision': precision_score(quality_filtered, pred),
        'recall': recall_score(quality_filtered, pred),
        'log_loss': log_loss(quality_filtered, pred_proba),
        'predictions': pred,
        'probabilities': pred_proba,
        'true_labels': quality_filtered
    }
    
    return metrics

def perform_logistic_regression(model, X_train, X_val, X_test, ages_train, ages_val, ages_test, 
                               quality_train, quality_val, quality_test, test_mae):
    """
    Perform complete logistic regression pipeline.
    
    Args:
        model: Trained BiLSTM model
        X_train, X_val, X_test: Feature data splits
        ages_train, ages_val, ages_test: Age data splits  
        quality_train, quality_val, quality_test: Quality label splits
        test_mae: Test MAE from BiLSTM
        
    Returns:
        dict: Complete logistic regression results
    """
    from .model import batch_predict_bs_scores
    
    print("=== 逻辑回归分析 ===")
    
    # 使用BiLSTM预测BS分数
    print("使用BiLSTM预测BS分数...")
    y_bs_train_pred = batch_predict_bs_scores(model, X_train)
    y_bs_val_pred = batch_predict_bs_scores(model, X_val) 
    y_bs_test_pred = batch_predict_bs_scores(model, X_test)
    
    # 构建逻辑回归特征
    print("构建逻辑回归特征...")
    train_features = build_logistic_features(ages_train, y_bs_train_pred)
    val_features = build_logistic_features(ages_val, y_bs_val_pred)
    test_features = build_logistic_features(ages_test, y_bs_test_pred)
    
    # 过滤无效特征
    train_features_clean, quality_train_filtered = filter_valid_features(train_features, quality_train)
    val_features_clean, quality_val_filtered = filter_valid_features(val_features, quality_val)
    test_features_clean, quality_test_filtered = filter_valid_features(test_features, quality_test)
    
    # 训练逻辑回归
    clf, scaler, train_metrics = train_logistic_classifier(train_features_clean, quality_train_filtered)
    
    # 评估性能
    print("评估逻辑回归性能...")
    val_metrics = evaluate_logistic_classifier(clf, scaler, val_features_clean, quality_val_filtered)
    test_metrics = evaluate_logistic_classifier(clf, scaler, test_features_clean, quality_test_filtered)
    
    print(f"验证集性能: 准确率={val_metrics['accuracy']:.4f}, AUC={val_metrics['auc']:.4f}")
    print(f"测试集性能: 准确率={test_metrics['accuracy']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # 整合结果
    results = {
        'test_mae': test_mae,
        'classifier': clf,
        'scaler': scaler,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': {
            'bs_score_coeff': clf.coef_[0][0],
            'age_coeff': clf.coef_[0][1],
            'intercept': clf.intercept_[0]
        }
    }
    
    return results

def analyze_feature_importance(clf, feature_names=['BS_Score', 'Age']):
    """
    Analyze logistic regression feature importance.
    
    Args:
        clf: Trained classifier
        feature_names: List of feature names
        
    Returns:
        dict: Feature importance analysis
    """
    coefficients = clf.coef_[0]
    intercept = clf.intercept_[0]
    
    importance_analysis = {
        'coefficients': dict(zip(feature_names, coefficients)),
        'intercept': intercept,
        'feature_ranking': sorted(zip(feature_names, np.abs(coefficients)), 
                                key=lambda x: x[1], reverse=True)
    }
    
    print("特征重要性分析:")
    print(f"截距: {intercept:.4f}")
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.4f}")
    
    return importance_analysis

def get_classification_report_detailed(y_true, y_pred, y_pred_proba=None):
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        dict: Detailed classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'classification_report': classification_report(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        
        # ROC曲线数据
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr,
            'tpr': tpr, 
            'thresholds': thresholds
        }
    
    return metrics

def cross_validation_logistic(features_list, labels_list, n_folds=4):
    """
    Perform cross-validation for logistic regression across multiple folds.
    
    Args:
        features_list: List of feature arrays for each fold
        labels_list: List of label arrays for each fold  
        n_folds: Number of folds
        
    Returns:
        dict: Cross-validation results
    """
    fold_metrics = []
    
    for fold_idx in range(n_folds):
        print(f"\n--- 逻辑回归第 {fold_idx + 1} 折 ---")
        
        # 当前折的特征和标签
        fold_features = features_list[fold_idx]
        fold_labels = labels_list[fold_idx]
        
        # 过滤无效数据
        features_clean, labels_filtered = filter_valid_features(fold_features, fold_labels)
        
        if len(features_clean) == 0:
            print(f"第 {fold_idx + 1} 折无有效数据，跳过")
            continue
            
        # 训练和评估
        clf, scaler, train_metrics = train_logistic_classifier(features_clean, labels_filtered)
        eval_metrics = evaluate_logistic_classifier(clf, scaler, features_clean, labels_filtered)
        
        fold_metrics.append({
            'fold': fold_idx + 1,
            'n_samples': len(features_clean),
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'classifier': clf,
            'scaler': scaler
        })
    
    # 计算平均性能
    avg_metrics = {}
    if fold_metrics:
        metric_names = ['accuracy', 'auc', 'precision', 'recall', 'log_loss']
        for metric in metric_names:
            values = [fold['eval_metrics'][metric] for fold in fold_metrics]
            avg_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
    
    return {
        'fold_metrics': fold_metrics,
        'average_metrics': avg_metrics,
        'n_valid_folds': len(fold_metrics)
    }