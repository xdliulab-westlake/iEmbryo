"""交叉验证模块 - 从BiLSTM.py中提取的交叉验证相关函数"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gc
import time
from config import DATA_CONFIG

def cross_validate_bilstm_model(X, y_bs, quality_labels, ages, feature_dim, n_folds=5, n_repeats=1):
    """对BiLSTM模型进行交叉验证评估"""
    print(f"=== 开始 {n_folds}折交叉验证 (重复{n_repeats}次) ===")
    
    # 强制清理GPU内存
    tf.keras.backend.clear_session()
    gc.collect()
    
    # 验证并修正特征维度
    expected_input_dim = 11 * feature_dim
    if X.shape[2] != expected_input_dim:
        print(f"特征维度不匹配，正在修正...")
        if X.shape[2] == feature_dim:
            X = X.reshape(X.shape[0], 33, 11, feature_dim).reshape(X.shape[0], 33, -1)
        elif X.shape[2] != 11 * feature_dim:
            raise ValueError(f"无法自动修正特征维度: {X.shape[2]}")
    
    print(f"修正后数据形状: {X.shape}")
    
    # 使用分层K折交叉验证
    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    
    # 存储每折的结果
    fold_results = {
        'train_mae': [], 'val_mae': [], 'test_mae': [],
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'logic_acc': [], 'logic_auc': [], 'logic_loss': [],
        'best_epochs': []
    }
    
    # 存储每折的预测结果
    all_predictions = []
    all_true_labels = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(cv.split(X, quality_labels)):
        print(f"\n--- 第 {fold_idx + 1} 折 ---")
        
        # 强制清理GPU内存
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 等待GPU内存释放
        if tf.config.list_physical_devices('GPU'):
            time.sleep(3)
        
        # 数据分割
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y_bs[train_val_idx], y_bs[test_idx]
        quality_train_val, quality_test = [quality_labels[i] for i in train_val_idx], [quality_labels[i] for i in test_idx]
        ages_train_val, ages_test = [ages[i] for i in train_val_idx], [ages[i] for i in test_idx]
        
        # 进一步拆分训练集和验证集
        idx_tv = np.arange(len(X_train_val))
        idx_tr, idx_va = train_test_split(
            idx_tv, test_size=0.2, random_state=42, stratify=np.array(quality_train_val)
        )
        X_train, X_val = X_train_val[idx_tr], X_train_val[idx_va]
        y_train, y_val = y_train_val[idx_tr], y_train_val[idx_va]
        quality_train = [quality_train_val[i] for i in idx_tr]
        quality_val = [quality_train_val[i] for i in idx_va]
        ages_train = [ages_train_val[i] for i in idx_tr]
        ages_val = [ages_train_val[i] for i in idx_va]
        
        # 平衡训练集
        X_train, y_train, quality_train, ages_train = balance_train_subset(
            X_train, y_train, quality_train, ages_train
        )

        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        # 训练模型
        model, history = train_fold_model(X_train, y_train, X_val, y_val, feature_dim)
        
        # 评估模型
        train_loss, train_mae = evaluate_in_batches(model, X_train, y_train)
        val_loss, val_mae = evaluate_in_batches(model, X_val, y_val)
        test_loss, test_mae = evaluate_in_batches(model, X_test, y_test)
        
        # 逻辑回归分析
        logic_acc, logic_auc, logic_loss = perform_fold_logistic_regression(
            model, X_train_val, y_train_val, ages_train_val, quality_train_val,
            X_test, y_test, ages_test, quality_test
        )
        
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
        
        print(f"测试MAE: {test_mae:.4f}, 逻辑回归AUC: {logic_auc:.4f}")
        
        # 清理内存
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
        if tf.config.list_physical_devices('GPU'):
            time.sleep(5)
            print(f"第 {fold_idx + 1} 折完成，已清理GPU内存")
    
    # 计算统计结果
    cv_summary = calculate_cv_summary(fold_results)
    
    print(f"\n=== 交叉验证结果总结 ===")
    print(f"BiLSTM测试MAE: {cv_summary['test_mae_mean']:.4f} ± {cv_summary['test_mae_std']:.4f}")
    print(f"逻辑回归AUC: {cv_summary['logic_auc_mean']:.4f} ± {cv_summary['logic_auc_std']:.4f}")
    
    return cv_summary, fold_results, all_predictions, all_true_labels

def train_fold_model(X_train, y_train, X_val, y_val, feature_dim):
    """训练单折模型"""
    from model_architecture import create_model
    
    # 创建模型
    model = create_model(feature_dim)
    
    # 回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0
        )
    ]
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=12,
        callbacks=callbacks,
        verbose=0
    )
    
    return model, history

def evaluate_in_batches(model, X, y, batch_size=2):
    """分批评估模型"""
    losses = []
    maes = []
    
    # 数据预处理和验证
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # 检查数据有效性
    if np.isnan(X).any() or np.isinf(X).any():
        print("警告：输入数据包含NaN或无穷大值，正在清理...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if np.isnan(y).any() or np.isinf(y).any():
        print("警告：标签数据包含NaN或无穷大值，正在清理...")
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    for i in range(0, len(X), batch_size):
        batch_end = min(i + batch_size, len(X))
        batch_X = X[i:batch_end]
        batch_y = y[i:batch_end]
        
        try:
            if not np.isfinite(batch_X).all() or not np.isfinite(batch_y).all():
                continue
            
            if batch_X.shape[0] == 0:
                continue
                
            batch_loss, batch_mae = model.evaluate(batch_X, batch_y, verbose=0)
            
            if np.isfinite(batch_loss) and np.isfinite(batch_mae):
                losses.append(batch_loss)
                maes.append(batch_mae)
                
        except Exception as e:
            print(f"批次 {i}-{batch_end} 失败: {e}")
            continue
    
    if not losses:
        return 0.0, 0.0
    
    return np.mean(losses), np.mean(maes)

def perform_fold_logistic_regression(model, X_trainval, y_trainval, ages_trainval, quality_trainval,
                                   X_test, y_test, ages_test, quality_test):
    """执行单折逻辑回归分析"""
    # 获取训练+验证集的预测
    y_bs_pred_trainval = model.predict(X_trainval, verbose=0).flatten()
    
    # 构建特征
    ages_trainval_array = np.array(ages_trainval)
    features_trainval = np.column_stack([
        ages_trainval_array, y_bs_pred_trainval,
        ages_trainval_array ** 2, y_bs_pred_trainval ** 2,
        ages_trainval_array * y_bs_pred_trainval, np.log(ages_trainval_array + 1),
        1.0 / (y_bs_pred_trainval + 1e-6)
    ])
    
    # 拟合逻辑回归
    valid_trainval = np.isfinite(features_trainval).all(axis=1)
    features_trainval = features_trainval[valid_trainval]
    quality_train_val_filtered = [quality_trainval[i] for i in range(len(quality_trainval)) if valid_trainval[i]]
    
    scaler = StandardScaler().fit(features_trainval)
    clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    clf.fit(scaler.transform(features_trainval), quality_train_val_filtered)
    
    # 在测试集上评估
    y_bs_pred = model.predict(X_test, verbose=0).flatten()
    
    ages_test_array = np.array(ages_test)
    features_test = np.column_stack([
        ages_test_array, y_bs_pred,
        ages_test_array ** 2, y_bs_pred ** 2,
        ages_test_array * y_bs_pred, np.log(ages_test_array + 1),
        1.0 / (y_bs_pred + 1e-6)
    ])
    
    valid_test = np.isfinite(features_test).all(axis=1)
    features_test = features_test[valid_test]
    quality_test_filtered = [quality_test[i] for i in range(len(quality_test)) if valid_test[i]]
    
    X_test_scaled = scaler.transform(features_test)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    logic_acc = accuracy_score(quality_test_filtered, (y_pred_proba > 0.5).astype(int))
    logic_auc = roc_auc_score(quality_test_filtered, y_pred_proba)
    logic_loss = log_loss(quality_test_filtered, y_pred_proba)
    
    return logic_acc, logic_auc, logic_loss

def balance_train_subset(X, y_bs, quality_labels, ages):
    """平衡训练子集"""
    from sklearn.utils import resample
    
    y_series = pd.Series(quality_labels)
    if y_series.nunique() < 2:
        return X, y_bs, quality_labels, ages
    
    idx_good = np.where(y_series == 1)[0]
    idx_poor = np.where(y_series == 0)[0]
    min_count = min(len(idx_good), len(idx_poor))
    
    sel_good = resample(idx_good, replace=False, n_samples=min_count, random_state=42)
    sel_poor = resample(idx_poor, replace=False, n_samples=min_count, random_state=42)
    sel = np.concatenate([sel_good, sel_poor])
    
    return X[sel], np.array(y_bs)[sel], [quality_labels[i] for i in sel], [ages[i] for i in sel]

def calculate_cv_summary(fold_results):
    """计算交叉验证结果摘要"""
    cv_summary = {}
    for metric in fold_results.keys():
        values = fold_results[metric]
        cv_summary[f'{metric}_mean'] = np.mean(values)
        cv_summary[f'{metric}_std'] = np.std(values)
        cv_summary[f'{metric}_min'] = np.min(values)
        cv_summary[f'{metric}_max'] = np.max(values)
    
    return cv_summary
