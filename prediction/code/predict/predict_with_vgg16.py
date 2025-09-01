# predict_with_vgg16_voting.py
"""
STORK/VGG16 胚胎质量预测脚本（带投票机制）
基于训练好的VGG16模型进行胚胎质量二分类预测
支持胚胎级别的投票机制
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from collections import defaultdict
import re

def set_gpu_memory_growth():
    """设置GPU内存增长"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"已为 {len(gpus)} 个GPU启用内存增长")
        except Exception as e:
            print(f"GPU配置错误: {e}")

def extract_embryo_info(image_path):
    """
    从图像路径中提取胚胎ID和焦面信息
    例如：/path/to/D2024_10_04_S01672_I4425_P_1/images/F-15/image.jpeg
    返回：(embryo_id, focal_plane)
    """
    try:
        # 提取胚胎ID（目录名）
        path_parts = image_path.split('/')
        for i, part in enumerate(path_parts):
            if re.match(r'D\d{4}_\d{2}_\d{2}_S\d+_I\d+_P_\d+', part):
                embryo_id = part
                break
        else:
            # 如果没有找到标准格式，使用倒数第三个目录作为胚胎ID
            embryo_id = path_parts[-3]
        
        # 提取焦面信息
        focal_plane = None
        for part in path_parts:
            if part.startswith('F') and (part.startswith('F-') or part.startswith('F0') or part.startswith('F1') or part.startswith('F2')):
                focal_plane = part
                break
        
        return embryo_id, focal_plane
    except Exception as e:
        print(f"解析路径失败 {image_path}: {e}")
        return None, None

def load_model_smart(model_path):
    """
    智能加载模型，支持多种格式
    """
    print(f"正在加载模型: {model_path}")
    
    # 检查模型格式
    if os.path.isdir(model_path):
        # 检查是否是SavedModel格式
        if os.path.exists(os.path.join(model_path, 'saved_model.pb')):
            print("检测到SavedModel格式，使用tf.saved_model.load()加载")
            try:
                # 尝试用TensorFlow SavedModel加载
                model = tf.saved_model.load(model_path)
                
                # 包装为可调用的推理函数
                if hasattr(model, 'signatures'):
                    # 获取默认签名
                    infer = model.signatures['serving_default']
                    print(f"SavedModel签名: {list(model.signatures.keys())}")
                    
                    # 创建包装类
                    class SavedModelWrapper:
                        def __init__(self, infer_func):
                            self.infer_func = infer_func
                            self.input_shape = None
                            self.output_shape = None
                            
                        def predict(self, x, verbose=0):
                            if isinstance(x, tf.data.Dataset):
                                predictions = []
                                for batch in x:
                                    if verbose:
                                        print(f"处理批次: {batch.shape}")
                                    input_key = list(self.infer_func.structured_input_signature[1].keys())[0]
                                    batch_pred = self.infer_func(**{input_key: batch})
                                    output_key = list(batch_pred.keys())[0]
                                    predictions.append(batch_pred[output_key].numpy())
                                return np.concatenate(predictions, axis=0)
                            else:
                                input_key = list(self.infer_func.structured_input_signature[1].keys())[0]
                                result = self.infer_func(**{input_key: x})
                                output_key = list(result.keys())[0]
                                return result[output_key].numpy()
                        
                        def get_layer(self, name):
                            return None
                    
                    return SavedModelWrapper(infer)
                else:
                    raise ValueError("SavedModel没有可用的signatures")
                    
            except Exception as e:
                print(f"SavedModel加载失败: {e}")
                print("尝试使用keras.layers.TFSMLayer")
                try:
                    from tensorflow.keras.layers import TFSMLayer
                    tfsm_layer = TFSMLayer(model_path, call_endpoint='serving_default')
                    input_shape = (224, 224, 3)
                    inputs = tf.keras.Input(shape=input_shape)
                    outputs = tfsm_layer(inputs)
                    model = tf.keras.Model(inputs, outputs)
                    print("使用TFSMLayer成功加载模型")
                    return model
                except Exception as e2:
                    print(f"TFSMLayer加载也失败: {e2}")
                    raise
        else:
            print("目录格式模型，尝试作为Keras模型加载")
            try:
                model = tf.keras.models.load_model(model_path)
                return model
            except Exception as e:
                print(f"Keras模型加载失败: {e}")
                raise
    else:
        # 文件格式
        if model_path.endswith('.h5'):
            print("检测到H5格式")
            model = tf.keras.models.load_model(model_path)
            return model
        elif model_path.endswith('.keras'):
            print("检测到Keras格式")
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            print("未知文件格式，尝试直接加载")
            model = tf.keras.models.load_model(model_path)
            return model

def preprocess_image_vgg16(image_path, img_size=224):
    """VGG16标准预处理函数"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img

def load_and_preprocess_images(image_paths, img_size=224, batch_size=64):
    """批量加载和预处理图像"""
    def preprocess_fn(path):
        return preprocess_image_vgg16(path, img_size)
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def collect_image_paths(images_dir, supported_formats=('*.jpg', '*.jpeg', '*.png', '*.bmp')):
    """递归收集指定目录下的所有图像文件"""
    image_paths = []
    
    for fmt in supported_formats:
        pattern = os.path.join(images_dir, "**", fmt)
        paths = glob.glob(pattern, recursive=True)
        image_paths.extend(paths)
    
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        raise ValueError(f"在目录 {images_dir} 中未找到任何图像文件")
    
    print(f"找到 {len(image_paths)} 张图像")
    return image_paths

def predict_focal_planes(model, dataset, image_paths):
    """对所有焦面图像进行预测"""
    print("开始预测焦面图像...")
    
    predictions = model.predict(dataset, verbose=1)
    print(f"预测输出形状: {predictions.shape}")
    
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        prob_poor = predictions[:, 0]
        prob_good = predictions[:, 1]
        pred_labels = np.argmax(predictions, axis=1)
    elif predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
        predictions = predictions.flatten()
        prob_good = predictions
        prob_poor = 1 - predictions
        pred_labels = (predictions >= 0.5).astype(int)
    else:
        if predictions.shape[1] > 2:
            print(f"警告: 输出有{predictions.shape[1]}个类别，只使用前两个")
            prob_poor = predictions[:, 0]
            prob_good = predictions[:, 1]
            pred_labels = np.argmax(predictions[:, :2], axis=1)
        else:
            raise ValueError(f"不支持的预测输出格式: {predictions.shape}")
    
    # 创建焦面级别的结果
    focal_results = []
    for i, image_path in enumerate(image_paths):
        embryo_id, focal_plane = extract_embryo_info(image_path)
        focal_results.append({
            'image_path': image_path,
            'embryo_id': embryo_id,
            'focal_plane': focal_plane,
            'prob_poor': prob_poor[i],
            'prob_good': prob_good[i],
            'pred_label': pred_labels[i],
            'pred_class': 'Poor' if pred_labels[i] == 0 else 'Good',
            'confidence': max(prob_poor[i], prob_good[i])
        })
    
    return pd.DataFrame(focal_results)

def embryo_level_voting(focal_results, voting_method='majority'):
    """
    胚胎级别投票机制
    
    参数:
    - focal_results: 焦面级别的预测结果DataFrame
    - voting_method: 投票方法
        - 'majority': 多数投票
        - 'probability': 概率平均
        - 'confidence': 基于置信度的加权投票
        - 'unanimous_conservative': 保守投票（所有焦面都是Good才预测Good）
    """
    print(f"\n=== 开始胚胎级别投票 (方法: {voting_method}) ===")
    
    embryo_results = []
    embryo_groups = focal_results.groupby('embryo_id')
    
    for embryo_id, group in embryo_groups:
        if group.empty or embryo_id is None:
            continue
            
        focal_count = len(group)
        good_votes = (group['pred_label'] == 1).sum()
        poor_votes = (group['pred_label'] == 0).sum()
        
        # 计算平均概率
        avg_prob_good = group['prob_good'].mean()
        avg_prob_poor = group['prob_poor'].mean()
        avg_confidence = group['confidence'].mean()
        
        # 根据投票方法决定最终预测
        if voting_method == 'majority':
            # 多数投票
            final_pred = 1 if good_votes > poor_votes else 0
            final_confidence = max(avg_prob_good, avg_prob_poor)
            
        elif voting_method == 'probability':
            # 概率平均
            final_pred = 1 if avg_prob_good > avg_prob_poor else 0
            final_confidence = max(avg_prob_good, avg_prob_poor)
            
        elif voting_method == 'confidence':
            # 基于置信度的加权投票
            weights = group['confidence'].values
            weighted_good = np.average(group['prob_good'].values, weights=weights)
            weighted_poor = np.average(group['prob_poor'].values, weights=weights)
            final_pred = 1 if weighted_good > weighted_poor else 0
            final_confidence = max(weighted_good, weighted_poor)
            avg_prob_good = weighted_good
            avg_prob_poor = weighted_poor
            
        elif voting_method == 'unanimous_conservative':
            # 保守投票：所有焦面都是Good才预测Good
            final_pred = 1 if good_votes == focal_count else 0
            final_confidence = avg_prob_good if final_pred == 1 else avg_prob_poor
            
        else:
            raise ValueError(f"不支持的投票方法: {voting_method}")
        
        # 收集详细信息
        focal_details = []
        for _, row in group.iterrows():
            focal_details.append({
                'focal_plane': row['focal_plane'],
                'pred_class': row['pred_class'],
                'confidence': row['confidence'],
                'prob_good': row['prob_good']
            })
        
        embryo_results.append({
            'embryo_id': embryo_id,
            'focal_count': focal_count,
            'good_votes': good_votes,
            'poor_votes': poor_votes,
            'vote_ratio': good_votes / focal_count,
            'avg_prob_good': avg_prob_good,
            'avg_prob_poor': avg_prob_poor,
            'final_pred': final_pred,
            'final_class': 'Good' if final_pred == 1 else 'Poor',
            'final_confidence': final_confidence,
            'voting_method': voting_method,
            'focal_details': focal_details
        })
        
        print(f"胚胎 {embryo_id}: {focal_count}个焦面, {good_votes}票Good, {poor_votes}票Poor → {['Poor', 'Good'][final_pred]} (置信度: {final_confidence:.4f})")
    
    return pd.DataFrame(embryo_results)

def save_detailed_results(focal_results, embryo_results, output_dir):
    """保存详细的预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存焦面级别结果
    focal_csv = os.path.join(output_dir, "focal_plane_predictions.csv")
    focal_results.to_csv(focal_csv, index=False)
    print(f"焦面级别结果已保存: {focal_csv}")
    
    # 保存胚胎级别结果
    embryo_csv = os.path.join(output_dir, "embryo_level_predictions.csv")
    embryo_simple = embryo_results[['embryo_id', 'focal_count', 'good_votes', 'poor_votes', 
                                   'vote_ratio', 'avg_prob_good', 'avg_prob_poor', 
                                   'final_pred', 'final_class', 'final_confidence', 'voting_method']].copy()
    embryo_simple.to_csv(embryo_csv, index=False)
    print(f"胚胎级别结果已保存: {embryo_csv}")
    
    # 生成汇总报告
    report_path = os.path.join(output_dir, "prediction_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 胚胎质量预测汇总报告 ===\n\n")
        f.write(f"总胚胎数: {len(embryo_results)}\n")
        f.write(f"预测为Good: {(embryo_results['final_pred'] == 1).sum()}\n")
        f.write(f"预测为Poor: {(embryo_results['final_pred'] == 0).sum()}\n")
        f.write(f"平均置信度: {embryo_results['final_confidence'].mean():.4f}\n")
        f.write(f"投票方法: {embryo_results['voting_method'].iloc[0]}\n\n")
        
        f.write("=== 各胚胎详细结果 ===\n")
        for _, row in embryo_results.iterrows():
            f.write(f"\n胚胎ID: {row['embryo_id']}\n")
            f.write(f"  焦面数量: {row['focal_count']}\n")
            f.write(f"  投票结果: {row['good_votes']}票Good, {row['poor_votes']}票Poor\n")
            f.write(f"  最终预测: {row['final_class']} (置信度: {row['final_confidence']:.4f})\n")
            f.write(f"  平均概率: Good={row['avg_prob_good']:.4f}, Poor={row['avg_prob_poor']:.4f}\n")
    
    print(f"汇总报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="使用训练好的STORK/VGG16模型预测胚胎质量（带投票机制）")
    
    # 必需参数
    parser.add_argument("--images-dir", type=str, required=True, 
                        help="包含图像的目录路径（递归搜索）")
    parser.add_argument("--model-path", type=str, required=True,
                        help="训练好的VGG16模型路径（支持SavedModel、H5、Keras格式）")
    
    # 可选参数
    parser.add_argument("--output-dir", type=str, default="prediction_results",
                        help="输出目录（默认: prediction_results）")
    parser.add_argument("--voting-method", type=str, default="majority",
                        choices=['majority', 'probability', 'confidence', 'unanimous_conservative'],
                        help="投票方法（默认: majority）")
    parser.add_argument("--img-size", type=int, default=224,
                        help="输入图像尺寸（默认: 224）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批处理大小（默认: 64）")
    
    args = parser.parse_args()
    
    # 设置GPU
    set_gpu_memory_growth()
    
    print("=== STORK/VGG16 胚胎质量预测（投票机制）===")
    print(f"图像目录: {args.images_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"投票方法: {args.voting_method}")
    print(f"输出目录: {args.output_dir}")
    
    # 检查路径
    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"图像目录不存在: {args.images_dir}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 收集图像路径
    print("\n=== 收集图像文件 ===")
    image_paths = collect_image_paths(args.images_dir)
    
    # 加载模型
    print(f"\n=== 加载模型 ===")
    model = load_model_smart(args.model_path)
    print("模型加载成功")
    
    # 预处理图像
    print(f"\n=== 预处理图像 ===")
    dataset = load_and_preprocess_images(image_paths, args.img_size, args.batch_size)
    
    # 进行焦面级别预测
    print(f"\n=== 焦面级别预测 ===")
    focal_results = predict_focal_planes(model, dataset, image_paths)
    
    # 胚胎级别投票
    embryo_results = embryo_level_voting(focal_results, args.voting_method)
    
    # 保存结果
    print(f"\n=== 保存结果 ===")
    save_detailed_results(focal_results, embryo_results, args.output_dir)
    
    # 打印汇总统计
    print(f"\n=== 最终预测统计 ===")
    print(f"总胚胎数: {len(embryo_results)}")
    print(f"预测为Good: {(embryo_results['final_pred'] == 1).sum()}")
    print(f"预测为Poor: {(embryo_results['final_pred'] == 0).sum()}")
    print(f"平均置信度: {embryo_results['final_confidence'].mean():.4f}")
    
    # 显示投票详情统计
    print(f"\n=== 投票详情统计 ===")
    focal_count_stats = embryo_results['focal_count'].describe()
    print(f"每个胚胎的焦面数量统计:")
    print(f"  平均: {focal_count_stats['mean']:.1f}")
    print(f"  最少: {int(focal_count_stats['min'])}")
    print(f"  最多: {int(focal_count_stats['max'])}")
    
    # 显示一些示例
    print(f"\n=== 示例结果 ===")
    for i, row in embryo_results.head(5).iterrows():
        print(f"胚胎 {row['embryo_id']}: {row['focal_count']}个焦面 → {row['final_class']} (置信度: {row['final_confidence']:.4f})")
    
    print(f"\n=== 预测完成 ===")
    return focal_results, embryo_results

if __name__ == "__main__":
    focal_results, embryo_results = main()