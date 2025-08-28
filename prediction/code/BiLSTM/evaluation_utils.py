"""评估工具模块 - 从BiLSTM.py中提取的评估相关函数"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    print("=== 模型评估 ===")
    
    # 分批评估
    batch_size = 16
    test_losses = []
    test_maes = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_X = X_test[i:batch_end]
        batch_y = y_test[i:batch_end]
        
        batch_loss, batch_mae = model.evaluate(batch_X, batch_y, verbose=0)
        test_losses.append(batch_loss)
        test_maes.append(batch_mae)
    
    test_loss = np.mean(test_losses)
    test_mae = np.mean(test_maes)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试MAE: {test_mae:.4f}")
    
    return test_loss, test_mae

def perform_logistic_regression(model, X_trainval, ages_trainval, quality_trainval, 
                              X_test, ages_test, quality_test, test_mae):
    """执行逻辑回归分析"""
    print("=== 逻辑回归分析 ===")

    # 清理GPU状态
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

    # 在训练+验证集上获取BS预测
    print("在训练+验证集上拟合逻辑回归...")
    batch_size = 4
    y_bs_pred_tr = []
    valid_indices_tr = []
    
    for i in range(0, len(X_trainval), batch_size):
        j = min(i + batch_size, len(X_trainval))
        try:
            pred = model.predict(X_trainval[i:j], verbose=0)
            y_bs_pred_tr.extend(pred.flatten())
            valid_indices_tr.extend(range(i, j))
        except Exception as e:
            print(f"预测批次 {i}-{j} 失败: {e}")
            continue
    
    y_bs_pred_tr = np.array(y_bs_pred_tr)
    valid_indices_tr = np.array(valid_indices_tr)
    
    # 获取对应的年龄和质量标签
    ages_tr = np.array([ages_trainval[i] for i in valid_indices_tr])
    quality_tr = np.array([quality_trainval[i] for i in valid_indices_tr])
    
    # 构建特征
    feats_tr = np.column_stack([
        ages_tr, y_bs_pred_tr, ages_tr**2, y_bs_pred_tr**2,
        ages_tr * y_bs_pred_tr, np.log(ages_tr + 1), 1.0 / (y_bs_pred_tr + 1e-6)
    ])
    
    valid_tr = np.isfinite(feats_tr).all(axis=1)
    feats_tr = feats_tr[valid_tr]
    logic_y_tr = quality_tr[valid_tr]

    # 拟合逻辑回归
    scaler = StandardScaler().fit(feats_tr)
    clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    clf.fit(scaler.transform(feats_tr), logic_y_tr)

    # 在测试集上评估
    print("在测试集上评估逻辑回归...")
    y_bs_pred_te = []
    valid_indices_te = []
    
    for i in range(0, len(X_test), batch_size):
        j = min(i + batch_size, len(X_test))
        try:
            pred = model.predict(X_test[i:j], verbose=0)
            y_bs_pred_te.extend(pred.flatten())
            valid_indices_te.extend(range(i, j))
        except Exception as e:
            print(f"测试集预测批次 {i}-{j} 失败: {e}")
            continue
    
    y_bs_pred_te = np.array(y_bs_pred_te)
    valid_indices_te = np.array(valid_indices_te)
    
    # 获取对应的年龄和质量标签
    ages_te = np.array([ages_test[i] for i in valid_indices_te])
    quality_te = np.array([quality_test[i] for i in valid_indices_te])
    
    # 构建测试集特征
    feats_te = np.column_stack([
        ages_te, y_bs_pred_te, ages_te**2, y_bs_pred_te**2,
        ages_te * y_bs_pred_te, np.log(ages_te + 1), 1.0 / (y_bs_pred_te + 1e-6)
    ])
    
    valid_te = np.isfinite(feats_te).all(axis=1)
    feats_te = feats_te[valid_te]
    logic_y = quality_te[valid_te]

    # 预测和评估
    X_te_scaled = scaler.transform(feats_te)
    y_pred_cls = clf.predict(X_te_scaled)
    y_pred_proba = clf.predict_proba(X_te_scaled)[:, 1]

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
        'ages_array': ages_te[valid_te],
        'y_bs_pred_all': y_bs_pred_te[valid_te]
    }
