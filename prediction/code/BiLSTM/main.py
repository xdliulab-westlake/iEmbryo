#!/usr/bin/env python
"""
BiLSTM 胚胎分类系统主程序
独立的训练和评估入口点，提供完整的参数配置和训练流程
"""

import os
import sys
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到系统路径，这样可以找到BiLSTM包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from BiLSTM import (
    configure_gpu,
    check_required_paths,
    BiLSTMFeatureCache,
    create_model,
    train_model,
    perform_logistic_regression,
    stratified_cross_validation_with_cache,
    create_cv_visualization,
    cleanup_memory,
    generate_score_dict_from_csv
)

class BiLSTMConfig:
    """BiLSTM配置管理类"""
    
    def __init__(self, config_file=None):
        # 默认配置
        self.config = {
            # 数据路径
            "data_root": "/Users/lzz/Documents/GitHub/IVF_AI/prediction/code",
            "image_folder": "image",
            "score_dict_path": "score_dict.json",
            "cache_dir": "bilstm_feature_cache",
            
            # 评分字典生成路径
            "maindb_csv_path": "",  # MainDB.csv 文件路径
            "filtered_csv_path": "",  # filtered_clean_embryos.csv 文件路径（可选）
            "slide_info_path": "",  # SlidePatientBirthDates.csv 年龄信息文件路径
            
            # 🆕 备份文件风格的数据结构支持
            "embryo_info_path": "",  # 胚胎图像信息文件路径 (包含EmbryoID, FullPath等列)
            
            # 模型参数
            "model_name": "all",        # 特征提取模型: all, efficientnet, resnet, densenet, vgg16
            "feature_dim": 5760,        # 自动根据model_name调整
            "lstm_units": 128,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            
            # 训练参数
            "epochs": 100,
            "batch_size": 32,
            "patience": 15,
            "validation_split": 0.3,
            
            # 交叉验证参数
            "n_folds": 4,
            "test_ratio": 0.3,
            "random_state": 42,
            
            # 预处理参数
            "img_size": 224,
            "apply_petri_crop": True,
            "apply_center_crop": True,
            "use_augmentation": True,
            
            # 缓存参数
            "use_cache": True,
            "force_recompute": False,
            
            # 输出参数
            "model_save_path": "bilstm_model.h5",
            "results_save_dir": "bilstm_results",
            "verbose": 1
        }
        
        # 如果提供了配置文件，则加载
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
            
        # 自动调整特征维度
        self._update_feature_dim()
    
    def load_config(self, config_file):
        """从JSON文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            self.config.update(user_config)
            print(f"✅ 配置文件已加载: {config_file}")
        except Exception as e:
            print(f"⚠️ 配置文件加载失败: {e}")
            print("使用默认配置...")
    
    def save_config(self, config_file):
        """保存当前配置到JSON文件"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✅ 配置已保存到: {config_file}")
        except Exception as e:
            print(f"❌ 配置保存失败: {e}")
    
    def get(self, key, default=None):
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """设置配置值"""
        self.config[key] = value
        # 如果改变了模型选择，自动调整特征维度
        if key == "model_name":
            self._update_feature_dim()
            
    def _update_feature_dim(self):
        """根据模型选择自动更新特征维度"""
        model_name = self.config.get("model_name", "all")
        
        # 特征维度映射（使用原始维度，不再压缩）
        feature_dims = {
            "all": 5760,          # 1280 + 2048 + 1920 + 512 = 5760维
            "efficientnet": 1280, # EfficientNetV2M原始维度
            "resnet": 2048,       # ResNet152V2原始维度
            "densenet": 1920,     # DenseNet201原始维度
            "vgg16": 512          # VGG16原始维度
        }
        
        if model_name in feature_dims:
            old_dim = self.config.get("feature_dim", 5760)
            new_dim = feature_dims[model_name]
            self.config["feature_dim"] = new_dim
            
            if old_dim != new_dim:
                print(f"🔄 特征维度已调整: {old_dim} -> {new_dim} (模型: {model_name})")
        else:
            print(f"⚠️ 未知的模型名称: {model_name}，使用默认配置")
    
    def print_config(self):
        """打印当前配置"""
        print("\n=== BiLSTM 配置参数 ===")
        for section in ["数据路径", "模型参数", "训练参数", "交叉验证参数", "预处理参数"]:
            print(f"\n{section}:")
            if section == "数据路径":
                keys = ["data_root", "image_folder", "score_dict_path", "cache_dir", "embryo_info_path"]
            elif section == "模型参数":
                keys = ["model_name", "feature_dim", "lstm_units", "dropout_rate", "learning_rate"]
            elif section == "训练参数":
                keys = ["epochs", "batch_size", "patience", "validation_split"]
            elif section == "交叉验证参数":
                keys = ["n_folds", "test_ratio", "random_state"]
            elif section == "预处理参数":
                keys = ["img_size", "apply_petri_crop", "apply_center_crop", "use_augmentation"]
            
            for key in keys:
                if key in self.config:
                    value = self.config[key]
                    if key == "model_name":
                        available = "[all, efficientnet, resnet, densenet, vgg16]"
                        print(f"  {key}: {value} {available}")
                    else:
                        print(f"  {key}: {value}")

def setup_directories(config):
    """创建必要的目录"""
    dirs_to_create = [
        config.get("cache_dir"),
        config.get("results_save_dir"),
        os.path.dirname(config.get("model_save_path", ""))
    ]
    
    for dir_path in dirs_to_create:
        if dir_path and dir_path != "":
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"📁 目录已创建: {dir_path}")

def validate_paths(config):
    """验证必要路径是否存在"""
    required_paths = {
        "数据根目录": config.get("data_root"),
        "图像文件夹": os.path.join(config.get("data_root"), config.get("image_folder")),
        "分数字典": os.path.join(config.get("data_root"), config.get("score_dict_path"))
    }
    
    print("\n=== 路径验证 ===")
    all_valid = True
    for name, path in required_paths.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (不存在)")
            all_valid = False
    
    return all_valid

def run_training(config):
    """运行完整的训练流程"""
    print("\n" + "="*50)
    print("🚀 开始 BiLSTM 胚胎分类系统训练")
    print("="*50)
    
    # 配置GPU
    print("\n--- 配置GPU环境 ---")
    configure_gpu()
    
    # 验证路径
    if not validate_paths(config):
        print("❌ 路径验证失败，请检查配置")
        return False
    
    # 创建目录
    setup_directories(config)
    
    try:
        # 初始化特征缓存
        print("\n--- 初始化特征缓存系统 ---")
        cache = BiLSTMFeatureCache(
            data_root=config.get("data_root"),
            cache_dir=config.get("cache_dir")
        )
        
        # 根据配置初始化特征提取器
        model_name = config.get("model_name", "all")
        print(f"\n--- 初始化特征提取器 (模型: {model_name}) ---")
        from BiLSTM import MultiModelFeatureExtractor
        extractor = MultiModelFeatureExtractor(model_name=model_name)
        cache.set_extractor(extractor)
        
        # 检查是否使用多焦距数据模式
        use_multi_focal = config.get("use_multi_focal", False)
        multi_focal_csv = config.get("multi_focal_csv", "/liaozizhuo/selected_image_paths.csv")

        if use_multi_focal and os.path.exists(multi_focal_csv):
            print("\n--- 加载多焦距深度数据 ---")
            from BiLSTM.utils import load_and_prepare_multi_focal_data
            root_dir, score_dict, multi_focal_df = load_and_prepare_multi_focal_data(config, multi_focal_csv)

            print("\n--- 开始多焦距深度交叉验证训练 ---")
            from BiLSTM.cross_validation import stratified_cross_validation_multi_focal
            cv_results = stratified_cross_validation_multi_focal(
                cache=cache,
                multi_focal_df=multi_focal_df,
                score_dict=score_dict,
                n_folds=config.get("n_folds"),
                test_ratio=config.get("test_ratio"),
                feature_dim=config.get("feature_dim"),
                lstm_units=config.get("lstm_units"),
                dropout_rate=config.get("dropout_rate"),
                learning_rate=config.get("learning_rate"),
                epochs=config.get("epochs"),
                batch_size=config.get("batch_size"),
                patience=config.get("patience"),
                use_cache=config.get("use_cache"),
                use_augmentation=config.get("use_augmentation"),
                verbose=config.get("verbose"),
                random_state=config.get("random_state")
            )
        else:
            # 加载数据（使用backup策略）
            print("\n--- 加载数据（匹配backup策略）---")
            from BiLSTM.utils import load_and_prepare_embryo_data
            root_dir, score_dict, valid_embryo_df = load_and_prepare_embryo_data(config)

            # 执行交叉验证训练
            print("\n--- 开始交叉验证训练 ---")
            cv_results = stratified_cross_validation_with_cache(
                cache=cache,
                n_folds=config.get("n_folds"),
                test_ratio=config.get("test_ratio"),
                feature_dim=config.get("feature_dim"),
                lstm_units=config.get("lstm_units"),
                dropout_rate=config.get("dropout_rate"),
                learning_rate=config.get("learning_rate"),
                epochs=config.get("epochs"),
                batch_size=config.get("batch_size"),
                patience=config.get("patience"),
                use_cache=config.get("use_cache"),
                use_augmentation=config.get("use_augmentation"),
                verbose=config.get("verbose"),
                random_state=config.get("random_state"),
                # 使用新的数据加载策略参数
                root_dir=root_dir,
                score_dict=score_dict,
                valid_embryo_df=valid_embryo_df
            )
        
        # 创建可视化结果
        print("\n--- 生成可视化结果 ---")
        create_cv_visualization(
            cv_results['avg_performance'],
            cv_results['bilstm_avg'],
            cv_results['fold_performances']
        )
        
        # 保存结果
        results_dir = config.get("results_save_dir")
        results_file = os.path.join(results_dir, "cv_results.json")
        
        # 准备可序列化的结果
        serializable_results = {
            "avg_performance": cv_results['avg_performance'],
            "bilstm_avg": cv_results['bilstm_avg'],
            "config": config.config,
            "n_folds": len(cv_results['fold_performances'])
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存到: {results_file}")
        
        # 显示最终结果
        print("\n" + "="*50)
        print("🎉 训练完成！最终结果:")
        print("="*50)
        print(f"逻辑回归平均准确率: {cv_results['avg_performance']['accuracy']:.4f} ± {cv_results['avg_performance']['accuracy_std']:.4f}")
        print(f"逻辑回归平均AUC: {cv_results['avg_performance']['auc']:.4f} ± {cv_results['avg_performance']['auc_std']:.4f}")
        print(f"BiLSTM平均MAE: {cv_results['bilstm_avg']['mae']:.4f} ± {cv_results['bilstm_avg']['mae_std']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理GPU内存
        cleanup_memory()

def run_inference(config, model_path, image_paths):
    """运行推理模式"""
    print(f"\n🔮 开始推理模式")
    print(f"模型路径: {model_path}")
    print(f"图像数量: {len(image_paths)}")
    
    # 配置GPU
    configure_gpu()
    
    try:
        from BiLSTM import load_trained_model, batch_predict_bs_scores
        
        # 加载模型
        model = load_trained_model(model_path)
        
        # 初始化特征缓存
        cache = BiLSTMFeatureCache(
            data_root=config.get("data_root"),
            cache_dir=config.get("cache_dir")
        )
        
        # 提取特征
        print("提取图像特征...")
        features = cache.get_or_compute_features(
            image_paths=image_paths,
            use_cache=config.get("use_cache"),
            training=False,
            use_augmentation=False
        )
        
        # 预测
        print("执行预测...")
        predictions = batch_predict_bs_scores(model, features)
        
        # 显示结果
        print("\n=== 预测结果 ===")
        for i, (img_path, pred) in enumerate(zip(image_paths, predictions)):
            print(f"{i+1}. {os.path.basename(img_path)}: BS分数 = {pred:.4f}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ 推理过程中发生错误: {e}")
        return None
    
    finally:
        cleanup_memory()

def create_default_config_file(config_path="bilstm_config.json"):
    """创建默认配置文件"""
    config = BiLSTMConfig()
    config.save_config(config_path)
    print(f"✅ 默认配置文件已创建: {config_path}")
    return config_path

def auto_generate_embryo_info_if_needed(config):
    """
    自动检查并生成胚胎信息文件（如果需要）
    
    Args:
        config: BiLSTMConfig 配置对象
    """
    from BiLSTM.utils import generate_embryo_info_from_score_dict
    
    data_root = config.get("data_root")
    score_dict_path = os.path.join(data_root, config.get("score_dict_path"))
    embryo_info_path = config.get("embryo_info_path")
    
    # 如果没有配置 embryo_info_path，跳过自动生成
    if not embryo_info_path or not embryo_info_path.strip():
        print("📋 未配置 embryo_info_path，跳过自动生成胚胎信息文件")
        return
    
    # 转换为绝对路径
    if not os.path.isabs(embryo_info_path):
        embryo_info_path = os.path.join(data_root, embryo_info_path)
    
    # 检查评分字典是否存在
    if not os.path.exists(score_dict_path):
        print(f"⚠️ 评分字典文件不存在: {score_dict_path}")
        print("   请先确保评分字典已生成")
        return
    
    # 检查是否需要生成胚胎信息文件
    need_generate = False
    
    if not os.path.exists(embryo_info_path):
        print(f"📝 胚胎信息文件不存在: {embryo_info_path}")
        need_generate = True
    else:
        # 检查文件修改时间，如果评分字典更新，重新生成
        score_dict_mtime = os.path.getmtime(score_dict_path)
        embryo_info_mtime = os.path.getmtime(embryo_info_path)
        
        if score_dict_mtime > embryo_info_mtime:
            print(f"📝 评分字典已更新，需要重新生成胚胎信息文件")
            need_generate = True
        else:
            print(f"✅ 胚胎信息文件已存在且是最新的: {embryo_info_path}")
    
    # 如果需要生成，自动生成胚胎信息文件
    if need_generate:
        print(f"\n🔄 自动生成胚胎信息文件...")
        print(f"📂 评分字典文件: {score_dict_path}")
        print(f"📂 目标文件: {embryo_info_path}")
        
        try:
            generate_embryo_info_from_score_dict(
                score_dict_path=score_dict_path,
                output_path=embryo_info_path,
                data_root=data_root,
                image_folder=config.get("image_folder")
            )
            print(f"✅ 自动生成胚胎信息文件完成: {embryo_info_path}")
            
        except Exception as e:
            print(f"❌ 自动生成胚胎信息文件失败: {e}")
            print("   请检查评分字典文件格式是否正确")
            return
    
    print()  # 添加空行分隔

def auto_generate_score_dict_if_needed(config):
    """
    自动检查并生成评分字典（如果需要）
    
    Args:
        config: BiLSTMConfig 配置对象
    """
    maindb_path = config.get("maindb_csv_path")
    filtered_path = config.get("filtered_csv_path")
    score_dict_path = config.get("score_dict_path")
    
    # 如果没有配置 maindb_csv_path，跳过自动生成
    if not maindb_path or not maindb_path.strip():
        print("📋 未配置 maindb_csv_path，跳过自动生成评分字典")
        return
    
    # 检查主数据文件是否存在
    if not os.path.exists(maindb_path):
        print(f"⚠️ 主数据文件不存在: {maindb_path}")
        print("   请检查配置文件中的 maindb_csv_path 路径是否正确")
        return
    
    # 检查是否需要生成评分字典
    need_generate = False
    
    if not os.path.exists(score_dict_path):
        print(f"📝 评分字典文件不存在: {score_dict_path}")
        need_generate = True
    else:
        # 检查文件修改时间，如果主数据文件更新，重新生成
        maindb_mtime = os.path.getmtime(maindb_path)
        score_dict_mtime = os.path.getmtime(score_dict_path)
        
        if maindb_mtime > score_dict_mtime:
            print(f"📝 主数据文件已更新，需要重新生成评分字典")
            need_generate = True
        else:
            print(f"✅ 评分字典文件已存在且是最新的: {score_dict_path}")
    
    # 如果需要生成，自动生成评分字典
    if need_generate:
        print(f"\n🔄 自动生成评分字典...")
        print(f"📂 主数据文件: {maindb_path}")
        if filtered_path and filtered_path.strip():
            print(f"📂 过滤文件: {filtered_path}")
        
        try:
            generate_score_dict_from_csv(
                maindb_path=maindb_path,
                filtered_path=filtered_path if filtered_path and filtered_path.strip() else None,
                slide_info_path="/liaozizhuo/scripts/SlidePatientBirthDates.csv",
                output_path=score_dict_path
            )
            print(f"✅ 自动生成评分字典完成: {score_dict_path}")
            
        except Exception as e:
            print(f"❌ 自动生成评分字典失败: {e}")
            print("   请检查数据文件格式是否正确，或手动运行:")
            print(f"   python main.py --generate-score-dict --maindb {maindb_path}")
            sys.exit(1)
    
    print()  # 添加空行分隔

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="BiLSTM胚胎分类系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认配置训练
  python main.py --mode train
  
  # 使用自定义配置文件训练
  python main.py --mode train --config my_config.json
  
  # 创建默认配置文件
  python main.py --create-config
  
  # 推理模式
  python main.py --mode inference --model model.h5 --images img1.jpg img2.jpg
  
  # 使用单个模型训练
  python main.py --mode train --feature-model efficientnet
  
  # 使用ResNet训练
  python main.py --mode train --feature-model resnet --epochs 50
  
  # 生成评分字典 - 使用命令行参数
  python main.py --generate-score-dict --maindb MainDB.csv --filtered filtered.csv
  
  # 生成评分字典 - 使用配置文件
  python main.py --generate-score-dict --config my_config.json
  
  # 只使用主数据库生成评分字典
  python main.py --generate-score-dict --maindb MainDB.csv --output-score-dict my_scores.json
        """
    )
    
    parser.add_argument('--mode', choices=['train', 'inference'], 
                       help='运行模式: train(训练) 或 inference(推理)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (JSON格式)')
    parser.add_argument('--create-config', action='store_true',
                       help='创建默认配置文件')
    parser.add_argument('--model', type=str,
                       help='推理模式下的模型文件路径')
    parser.add_argument('--images', nargs='+',
                       help='推理模式下的图像文件路径列表')
    parser.add_argument('--epochs', type=int,
                       help='训练轮数 (覆盖配置文件)')
    parser.add_argument('--batch-size', type=int,
                       help='批量大小 (覆盖配置文件)')
    parser.add_argument('--learning-rate', type=float,
                       help='学习率 (覆盖配置文件)')
    parser.add_argument('--feature-model', type=str, dest='model_name_arg', 
                       help='特征提取模型选择: all, efficientnet, resnet, densenet, vgg16')
    
    # 评分字典生成选项
    parser.add_argument('--generate-score-dict', action='store_true',
                       help='生成评分字典模式')
    parser.add_argument('--maindb', type=str,
                       help='主数据库CSV文件路径 (MainDB.csv)')
    parser.add_argument('--filtered', type=str,
                       help='低质量胚胎CSV文件路径 (filtered_clean_embryos.csv, 可选)')
    parser.add_argument('--output-score-dict', type=str, default='score_dict.json',
                       help='输出评分字典文件路径 (默认: score_dict.json)')
    
    args = parser.parse_args()
    
    # 创建配置文件
    if args.create_config:
        create_default_config_file()
        return
    
    # 生成评分字典模式
    if args.generate_score_dict:
        # 先加载配置以获取可能的默认路径
        config = BiLSTMConfig(args.config)
        
        # 确定使用的路径（命令行参数优先级高于配置文件）
        maindb_path = args.maindb or config.get("maindb_csv_path")
        filtered_path = args.filtered or config.get("filtered_csv_path") or None
        output_path = args.output_score_dict
        
        # 检查是否有有效的maindb路径
        if not maindb_path:
            print("❌ 错误: 需要提供MainDB.csv文件路径")
            print("方法1: 使用命令行参数")
            print("  python main.py --generate-score-dict --maindb /path/to/MainDB.csv")
            print("方法2: 在配置文件中设置 'maindb_csv_path'")
            print("  python main.py --generate-score-dict --config my_config.json")
            return
        
        try:
            print(f"📂 使用MainDB文件: {maindb_path}")
            if filtered_path:
                print(f"📂 使用过滤文件: {filtered_path}")
            else:
                print("📂 未提供过滤文件，只使用MainDB")
                
            generate_score_dict_from_csv(
                maindb_path=maindb_path,
                filtered_path=filtered_path,
                slide_info_path="/liaozizhuo/scripts/SlidePatientBirthDates.csv",
                output_path=output_path
            )
            print(f"✅ 评分字典生成完成: {output_path}")
        except Exception as e:
            print(f"❌ 生成评分字典失败: {e}")
        return
    
    # 加载配置
    config = BiLSTMConfig(args.config)
    
    # 命令行参数覆盖配置文件
    if args.epochs:
        config.set('epochs', args.epochs)
    if args.batch_size:
        config.set('batch_size', args.batch_size)
    if args.learning_rate:
        config.set('learning_rate', args.learning_rate)
    if args.model_name_arg:
        config.set('model_name', args.model_name_arg)
    
    # 显示配置
    config.print_config()
    
    
    # 自动检查并生成评分字典（如果需要）
    auto_generate_score_dict_if_needed(config)
    
    # 自动检查并生成胚胎信息文件（如果需要）
    auto_generate_embryo_info_if_needed(config)
    
    # 根据模式执行
    if args.mode == 'train':
        success = run_training(config)
        if success:
            print("\n🎉 训练成功完成！")
        else:
            print("\n❌ 训练失败！")
            sys.exit(1)
            
    elif args.mode == 'inference':
        if not args.model:
            print("❌ 推理模式需要指定模型文件 (--model)")
            sys.exit(1)
        if not args.images:
            print("❌ 推理模式需要指定图像文件 (--images)")
            sys.exit(1)
            
        predictions = run_inference(config, args.model, args.images)
        if predictions is None:
            sys.exit(1)
            
    else:
        print("请指定运行模式: --mode train 或 --mode inference")
        print("使用 --help 查看详细帮助")

if __name__ == "__main__":
    main()