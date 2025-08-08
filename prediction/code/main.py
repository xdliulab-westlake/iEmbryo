from config import *
from data_loader import load_and_prepare_data, split_dataset, load_or_build_features
from feature_extractor import MultiModelFeatureExtractor
from models import create_model
from hyperparameter_tuning import grid_search_hyperparameters, plot_grid_search_results
from training import train_main_model, evaluate_model
from evaluation import perform_logistic_regression, create_visualizations
from cross_validation import run_cross_validation

def main():
    """主程序入口"""
    # 配置参数
    save_model_path = OUTPUT_PATHS['model']
    
    # 1. 加载和准备数据
    root_dir, score_dict, valid_embryo_df = load_and_prepare_data()
    
    # 2. 加载或构建特征
    X, y_bs, quality_labels, ages = load_or_build_features(root_dir, score_dict, valid_embryo_df)
    
    # 3. 网格搜索超参数优化
    print("\n=== 开始超参数优化 ===")
    best_params, best_score, grid_results = grid_search_hyperparameters(
        X, y_bs, quality_labels, ages, X.shape[2] // 11,
        n_folds=3, max_epochs=30, patience=10
    )
    
    # 可视化网格搜索结果
    plot_grid_search_results(grid_results)
    
    # 4. 使用最佳参数拆分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y_bs, quality_labels)
    
    # 5. 使用最佳参数训练主模型
    print(f"\n=== 使用最佳参数训练模型 ===")
    print(f"最佳参数: {best_params}")
    model, history = train_main_model(X_train, y_train, X_val, y_val, X.shape[2] // 11, save_model_path)
    
    # 6. 评估模型
    test_loss, test_mae = evaluate_model(model, X_test, y_test)
    
    # 7. 逻辑回归分析
    logic_results = perform_logistic_regression(model, X, ages, quality_labels, test_mae)
    
    # 8. 创建可视化
    create_visualizations(logic_results, test_mae)
    
    # 9. 运行交叉验证
    run_cross_validation(X, y_bs, quality_labels, ages, X.shape[2] // 11)
    
    # 10. 最终总结
    print(f"\n=== 最终结果总结 ===")
    print(f"多模型特征维度: {X.shape[2]}")
    print(f"最佳超参数: {best_params}")
    print(f"网格搜索最佳验证MAE: {best_score:.4f}")
    print(f"BiLSTM测试MAE: {test_mae:.4f}")
    print(f"逻辑回归准确率: {logic_results['logic_acc']:.4f}")
    print(f"逻辑回归AUC: {logic_results['auc_score']:.4f}")
    print(f"分析图表已保存至: {OUTPUT_PATHS['multi_model_analysis']}")
    print(f"特征重要性图表已保存至: {OUTPUT_PATHS['feature_importance']}")
    print(f"网格搜索分析图表已保存至: {OUTPUT_PATHS['grid_search_analysis']}")

if __name__ == "__main__":
    main()