"""主运行脚本 - 整合所有模块"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_processing import load_score_dict, build_dataset_fast
from feature_extraction import MultiModelFeatureExtractor
from model_architecture import create_model
from training_utils import configure_gpu_for_lstm, train_main_model
from evaluation_utils import evaluate_model, perform_logistic_regression
from visualization_utils import create_visualizations
from cross_validation import cross_validate_bilstm_model
import numpy as np
import pandas as pd

def main():
    """主函数"""
    try:
        print("=== IVF_AI BiLSTM模型训练 ===")
        
        # 1. 配置GPU
        configure_gpu_for_lstm()
        
        # 2. 加载数据
        print("=== 步骤1: 加载数据 ===")
        score_dict = load_score_dict(PATH_CONFIG['score_csv'])
        
        valid_embryo_df = pd.read_csv(PATH_CONFIG['embryo_csv'])
        print(f"成功加载胚胎信息: {valid_embryo_df.shape}")
        
        # 3. 构建特征
        print("=== 步骤2: 构建特征 ===")
        multi_extractor = MultiModelFeatureExtractor(target_dim=512)
        X, y_bs, quality_labels, ages = build_dataset_fast(
            PATH_CONFIG['image_root'], 
            score_dict, 
            multi_extractor, 
            valid_embryo_df, 
            training=False
        )
        
        # 4. 数据分割
        print("=== 步骤3: 数据分割 ===")
        from sklearn.model_selection import train_test_split
        
        X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
            X, y_bs, range(len(X)), 
            test_size=DATA_CONFIG['test_split'], 
            random_state=DATA_CONFIG['random_state'], 
            stratify=quality_labels
        )
        
        quality_trainval = [quality_labels[i] for i in idx_trainval]
        quality_test = [quality_labels[i] for i in idx_test]
        ages_trainval = [ages[i] for i in idx_trainval]
        ages_test = [ages[i] for i in idx_test]
        
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_trainval, y_trainval, range(len(X_trainval)), 
            test_size=DATA_CONFIG['validation_split']/(1-DATA_CONFIG['test_split']), 
            random_state=DATA_CONFIG['random_state'], 
            stratify=quality_trainval
        )
        
        quality_train = [quality_trainval[i] for i in idx_train]
        quality_val = [quality_trainval[i] for i in idx_val]
        ages_train = [ages_trainval[i] for i in idx_train]
        ages_val = [ages_trainval[i] for i in idx_val]
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        # 5. 训练模型
        print("=== 步骤4: 训练模型 ===")
        feature_dim = X.shape[2] // 11
        save_model_path = os.path.join(PATH_CONFIG['output_dir'], "bilstm_quality_model.h5")
        
        model, history = train_main_model(X_train, y_train, X_val, y_val, feature_dim, save_model_path)
        
        # 6. 运行交叉验证（包含完整评估）
        print("=== 步骤5: 运行交叉验证评估 ===")
        cv_summary, fold_results, all_predictions, all_true_labels = cross_validate_bilstm_model(
            X, y_bs, quality_labels, ages, feature_dim
        )
        
        # 7. 创建可视化（基于交叉验证结果）
        print("=== 步骤6: 创建可视化 ===")
        from visualization_utils import plot_cross_validation_results
        plot_cross_validation_results(cv_summary, fold_results)
        
        # 8. 总结结果
        print("=== 步骤7: 结果总结 ===")
        print(f"交叉验证完成！")
        print(f"BiLSTM测试MAE: {cv_summary['test_mae_mean']:.4f} ± {cv_summary['test_mae_std']:.4f}")
        print(f"逻辑回归AUC: {cv_summary['logic_auc_mean']:.4f} ± {cv_summary['logic_auc_std']:.4f}")
        print(f"最佳模型已保存至: {save_model_path}")
        
        print("\n=== 程序执行完成 ===")
        
    except Exception as e:
        print(f"\n=== 程序执行失败 ===")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
