"""
BiLSTM 胚胎分类包

一个完整的双层机器学习系统，用于胚胎质量分类：
1. BiLSTM 回归预测囊胚评分
2. 逻辑回归进行最终质量分类

模块结构:
- feature_cache: 智能特征缓存系统
- feature_extractor: 多模型特征提取器
- preprocessing: 图像预处理和数据加载
- model: BiLSTM模型构建和训练
- logistic: 逻辑回归分类器
- cross_validation: 交叉验证和模型评估
- visualization: 结果可视化
- utils: 通用工具函数
"""

# 特征缓存模块
from .feature_cache import BiLSTMFeatureCache

# 特征提取模块
from .feature_extractor import MultiModelFeatureExtractor

# 预处理模块
from .preprocessing import (
    preprocess_embryo_image,
    crop_center_square,
    crop_petri_dish,
    build_feature_extractor
)

# 模型模块
from .model import (
    create_model,
    train_model,
    evaluate_model,
    batch_predict_bs_scores,
    load_trained_model
)

# 逻辑回归模块
from .logistic import (
    build_logistic_features,
    train_logistic_classifier,
    evaluate_logistic_classifier,
    perform_logistic_regression
)

# 交叉验证模块
from .cross_validation import (
    assess_model_stability,
    ensemble_predictions,
    stratified_cross_validation,
    validate_cv_setup
)

# 可视化模块
from .visualization import (
    plot_training_history,
    plot_prediction_scatter,
    plot_roc_curve,
    plot_confusion_matrix,
    create_cv_visualization
)

# 工具模块
from .utils import (
    ensure_float32,
    configure_gpu,
    load_score_dict,
    check_required_paths,
    create_image_augmentation_fn,
    cleanup_memory,
    check_cudnn_compatibility,
    get_unified_training_config,
    create_callbacks,
    print_system_info,
    validate_data_shapes,
    safe_divide
)

__version__ = "2.0.0"
__author__ = "IVF AI Team"
__description__ = "BiLSTM-based embryo classification system with Hough circle detection (modular version)"

__all__ = [
    # 特征缓存
    'BiLSTMFeatureCache',
    
    # 特征提取
    'MultiModelFeatureExtractor',
    
    # 预处理
    'preprocess_embryo_image',
    'crop_center_square',
    'crop_petri_dish',
    'build_feature_extractor',
    
    # 模型
    'create_model',
    'train_model',
    'evaluate_model',
    'batch_predict_bs_scores',
    'load_trained_model',
    
    # 逻辑回归
    'build_logistic_features',
    'train_logistic_classifier',
    'evaluate_logistic_classifier',
    'perform_logistic_regression',
    
    # 交叉验证
    'assess_model_stability',
    'ensemble_predictions',
    'stratified_cross_validation',
    'validate_cv_setup',
    
    # 可视化
    'plot_training_history',
    'plot_prediction_scatter', 
    'plot_roc_curve',
    'plot_confusion_matrix',
    'create_cv_visualization',
    
    # 工具
    'ensure_float32',
    'configure_gpu',
    'load_score_dict',
    'check_required_paths',
    'create_image_augmentation_fn',
    'cleanup_memory',
    'check_cudnn_compatibility',
    'get_unified_training_config',
    'create_callbacks',
    'print_system_info',
    'validate_data_shapes',
    'safe_divide'
]