# predict_with_vgg16_voting.py
"""
STORK/VGG16 胚胎质量预测脚本（带投票机制和Grad-CAM）
基于训练好的VGG16模型进行胚胎质量二分类预测
支持胚胎级别的投票机制和Grad-CAM可视化
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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
         /path/to/D2024_10_04_S01672_I4425_P_1/images/F30/image.jpeg
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
        
        # 提取焦面信息 - 更灵活的匹配
        focal_plane = None
        for part in path_parts:
            # 匹配各种焦面格式：F-15, F30, F45, F0, F15, F-30等
            if re.match(r'F-?\d+$', part):  # F后面可选连字符，然后是数字
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
            print("检测到SavedModel格式")
            try:
                # 首先尝试使用TFSMLayer创建Keras模型（支持Grad-CAM）
                from tensorflow.keras.layers import TFSMLayer
                print("尝试使用TFSMLayer创建Keras模型以支持Grad-CAM")
                
                tfsm_layer = TFSMLayer(model_path, call_endpoint='serving_default')
                input_shape = (224, 224, 3)
                inputs = tf.keras.Input(shape=input_shape)
                outputs = tfsm_layer(inputs)
                model = tf.keras.Model(inputs, outputs)
                model._gradcam_supported = True
                print("使用TFSMLayer成功加载模型，支持Grad-CAM")
                return model
                
            except Exception as e1:
                print(f"TFSMLayer加载失败: {e1}")
                print("回退到SavedModel加载（不支持Grad-CAM）")
                try:
                    # 回退到原来的SavedModel加载方式
                    model = tf.saved_model.load(model_path)
                    
                    if hasattr(model, 'signatures'):
                        infer = model.signatures['serving_default']
                        print(f"SavedModel签名: {list(model.signatures.keys())}")
                        
                        class SavedModelWrapper:
                            def __init__(self, infer_func):
                                self.infer_func = infer_func
                                self.input_shape = None
                                self.output_shape = None
                                self._gradcam_supported = False
                                
                            def predict(self, x, verbose=0):
                                if isinstance(x, tf.data.Dataset):
                                    predictions = []
                                    for batch in x:
                                        if verbose:
                                            print(f"处理批次: {batch.shape}")
                                        input_key = list(self.infer_func.structured_input_signature[1].keys())[0]
                                        batch_pred = self.infer_func(**{input_key: batch})
                                        
                                        # 处理字典输出
                                        if isinstance(batch_pred, dict):
                                            output_key = list(batch_pred.keys())[0]
                                            batch_result = batch_pred[output_key].numpy()
                                        else:
                                            batch_result = batch_pred.numpy()
                                        
                                        predictions.append(batch_result)
                                    return np.concatenate(predictions, axis=0)
                                else:
                                    input_key = list(self.infer_func.structured_input_signature[1].keys())[0]
                                    result = self.infer_func(**{input_key: x})
                                    
                                    # 处理字典输出
                                    if isinstance(result, dict):
                                        output_key = list(result.keys())[0]
                                        return result[output_key].numpy()
                                    else:
                                        return result.numpy()
                            
                            def get_layer(self, name):
                                return None
                        
                        return SavedModelWrapper(infer)
                    else:
                        raise ValueError("SavedModel没有可用的signatures")
                        
                except Exception as e2:
                    print(f"SavedModel加载也失败: {e2}")
                    raise
        else:
            print("目录格式模型，尝试作为Keras模型加载")
            try:
                model = tf.keras.models.load_model(model_path)
                model._gradcam_supported = True
                return model
            except Exception as e:
                print(f"Keras模型加载失败: {e}")
                raise
    else:
        # 文件格式
        if model_path.endswith('.h5'):
            print("检测到H5格式")
            model = tf.keras.models.load_model(model_path)
            model._gradcam_supported = True
            return model
        elif model_path.endswith('.keras'):
            print("检测到Keras格式")
            model = tf.keras.models.load_model(model_path)
            model._gradcam_supported = True
            return model
        else:
            print("未知文件格式，尝试直接加载")
            model = tf.keras.models.load_model(model_path)
            model._gradcam_supported = True
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
    
    # 处理SavedModel的字典输出
    if isinstance(predictions, dict):
        print(f"检测到字典格式输出，键: {list(predictions.keys())}")
        # 取第一个输出（通常是预测结果）
        output_key = list(predictions.keys())[0]
        predictions = predictions[output_key]
        print(f"使用输出键: {output_key}")
    
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

def generate_gradcam_for_image(model, image_path, class_index, layer_names=['block5_conv3', 'block4_conv3', 'block3_conv3']):
    """
    为单张图像生成Grad-CAM或替代的注意力可视化
    自动检测模型类型并选择合适的方法
    """
    if not hasattr(model, '_gradcam_supported') or not model._gradcam_supported:
        print("模型不支持Grad-CAM")
        return None, None, None
    
    # 检查是否是TFSMLayer包装的模型
    if hasattr(model, 'layers') and len(model.layers) >= 1:
        layer_names_available = [layer.name for layer in model.layers]
        print(f"检测到模型层: {layer_names_available}")
        
        # 如果检测到TFSMLayer或者只有input_layer和tfsm_layer
        if ('tfsm_layer' in layer_names_available or 
            (len(layer_names_available) == 2 and 'input_layer' in layer_names_available)):
            print("检测到TFSMLayer包装模型，使用输入梯度方法")
            return generate_gradcam_for_savedmodel(model, image_path, class_index)
    
    # 检查是否能找到VGG16的传统层
    available_layers = []
    if hasattr(model, 'layers'):
        available_layers = [layer.name for layer in model.layers]
    
    # 如果找不到任何VGG16层，直接使用输入梯度方法
    vgg_layers_found = any(layer_name in available_layers for layer_name in layer_names)
    if not vgg_layers_found:
        print(f"未找到VGG16传统层，可用层: {available_layers[:5]}...")
        print("使用输入梯度方法作为替代")
        return generate_gradcam_for_savedmodel(model, image_path, class_index)
    
    # 原始的Grad-CAM方法（适用于常规Keras模型）
    try:
        # 加载并预处理图像
        img_tensor = preprocess_image_vgg16(image_path)
        img_tensor = tf.expand_dims(img_tensor, 0)
        
        # 加载原始图像用于显示
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (224, 224))
        
        best_heatmap = None
        best_layer = None
        
        # 尝试不同的层
        for layer_name in layer_names:
            try:
                # 检查层是否存在
                layer = model.get_layer(layer_name)
                if layer is None:
                    continue
        
                # 创建Grad-CAM模型
                grad_model = tf.keras.Model(
                    inputs=[model.inputs],
                    outputs=[model.get_layer(layer_name).output, model.output]
                )
        
                # 计算梯度
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_tensor)
                    if predictions.shape[1] == 2:
                        loss = predictions[:, class_index]
                    else:
                        loss = predictions[:, 0] if class_index == 0 else (1 - predictions[:, 0])
        
                # 获取梯度
                grads = tape.gradient(loss, conv_outputs)
                if grads is None:
                    continue
                    
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
                # 生成热力图
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                
                # 归一化
                heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
                heatmap_np = heatmap.numpy()
                
                # 调整大小
                heatmap_resized = cv2.resize(heatmap_np, (224, 224))
                
                # 如果这是第一个成功的热力图，或者质量更好，则保存
                if best_heatmap is None or np.max(heatmap_resized) > np.max(best_heatmap):
                    best_heatmap = heatmap_resized
                    best_layer = layer_name
                    
            except Exception as e:
                print(f"层 {layer_name} Grad-CAM生成失败: {e}")
                continue
        
        if best_heatmap is not None:
            return best_heatmap, original_img, best_layer
        else:
            # 如果所有层都失败，尝试输入梯度方法作为后备
            print("传统Grad-CAM失败，尝试输入梯度方法")
            return generate_gradcam_for_savedmodel(model, image_path, class_index)
        
    except Exception as e:
        print(f"Grad-CAM生成失败，尝试输入梯度方法: {e}")
        return generate_gradcam_for_savedmodel(model, image_path, class_index)

def generate_gradcam_for_savedmodel(model, image_path, class_index=0):
    """
    为SavedModel格式生成替代的注意力可视化
    使用输入梯度方法代替Grad-CAM
    """
    try:
        # 加载并预处理图像
        img_tensor = preprocess_image_vgg16(image_path)
        img_tensor = tf.expand_dims(img_tensor, 0)
        
        # 加载原始图像用于显示
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (224, 224))
        
        # 使用输入梯度方法
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            
            # 直接调用模型进行预测
            if hasattr(model, 'predict'):
                predictions = model(img_tensor, training=False)  # 使用__call__而不是predict
            else:
                predictions = model(img_tensor)
            
            # 处理字典输出
            if isinstance(predictions, dict):
                output_key = list(predictions.keys())[0]
                predictions = predictions[output_key]
            
            # 选择目标类别的得分
            if len(predictions.shape) > 1 and predictions.shape[1] == 2:
                target_score = predictions[0, class_index]
            elif len(predictions.shape) > 1 and predictions.shape[1] == 1:
                target_score = predictions[0, 0] if class_index == 0 else (1 - predictions[0, 0])
            else:
                # 处理单一输出
                predictions_flat = tf.reshape(predictions, [-1])
                target_score = predictions_flat[0] if class_index == 0 else (1 - predictions_flat[0])
        
        # 计算梯度
        grads = tape.gradient(target_score, img_tensor)
        
        if grads is None:
            print("梯度计算失败")
            return None, original_img, "input_gradient"
        
        # 计算梯度的绝对值并取平均
        grads_abs = tf.abs(grads)
        attention_map = tf.reduce_mean(grads_abs, axis=-1)[0]  # 取第一个样本，平均RGB通道
        
        # 归一化到0-1
        attention_min = tf.reduce_min(attention_map)
        attention_max = tf.reduce_max(attention_map)
        attention_map = (attention_map - attention_min) / (attention_max - attention_min + 1e-8)
        
        return attention_map.numpy(), original_img, "input_gradient"
        
    except Exception as e:
        print(f"输入梯度方法失败: {e}")
        return None, None, None

def create_embryo_gradcam_visualization(model, embryo_group, embryo_id, save_dir):
    """
    为单个胚胎创建Grad-CAM可视化
    包括所有焦面的热力图和汇总分析
    """
    if not hasattr(model, '_gradcam_supported') or not model._gradcam_supported:
        print(f"模型不支持Grad-CAM，跳过胚胎 {embryo_id}")
        return
    
    print(f"生成胚胎 {embryo_id} 的Grad-CAM可视化...")
    
    embryo_dir = os.path.join(save_dir, f"embryo_{embryo_id}")
    os.makedirs(embryo_dir, exist_ok=True)
    
    focal_data = []
    
    # 为每个焦面生成Grad-CAM
    for _, row in embryo_group.iterrows():
        image_path = row['image_path']
        focal_plane = row['focal_plane']
        pred_class = row['pred_class']
        confidence = row['confidence']
        pred_label = row['pred_label']
        
        # 生成Grad-CAM
        heatmap, original_img, layer_used = generate_gradcam_for_image(
            model, image_path, pred_label
        )
        
        if heatmap is not None and original_img is not None:
            focal_data.append({
                'focal_plane': focal_plane,
                'original_img': original_img,
                'heatmap': heatmap,
                'pred_class': pred_class,
                'confidence': confidence,
                'layer_used': layer_used
            })
    
    if not focal_data:
        print(f"胚胎 {embryo_id} 无法生成Grad-CAM")
        return
    
    # 创建综合可视化
    n_focals = len(focal_data)
    if n_focals <= 6:
        cols = min(3, n_focals)
        rows = (n_focals + cols - 1) // cols
    else:
        cols = 4
        rows = (n_focals + cols - 1) // cols
    
    fig = plt.figure(figsize=(4*cols, 3*rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    for i, data in enumerate(focal_data):
        row = i // cols
        col = i % cols
        
        # 原始图像
        ax = fig.add_subplot(gs[row, col])
        
        # 创建叠加图像
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * data['heatmap']), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加
        superimposed = heatmap_colored * 0.4 + data['original_img'] * 0.6
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        ax.imshow(superimposed)
        ax.set_title(f"{data['focal_plane']}\n{data['pred_class']} ({data['confidence']:.3f})")
        ax.axis('off')
        
        # 保存单独的焦面结果
        focal_save_path = os.path.join(embryo_dir, f"{data['focal_plane']}_gradcam.png")
        cv2.imwrite(focal_save_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    
    plt.suptitle(f"Embryo {embryo_id} - Grad-CAM Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存综合图
    overview_path = os.path.join(embryo_dir, f"{embryo_id}_gradcam_overview.png")
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成热力图统计
    heatmaps = [data['heatmap'] for data in focal_data]
    avg_heatmap = np.mean(heatmaps, axis=0)
    max_heatmap = np.max(heatmaps, axis=0)
    
    # 保存统计热力图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(avg_heatmap, cmap='jet')
    axes[0].set_title('Average Attention Heatmap')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(max_heatmap, cmap='jet')
    axes[1].set_title('Maximum Attention Heatmap')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(f"Embryo {embryo_id} - Attention Statistical Analysis")
    plt.tight_layout()
    
    stats_path = os.path.join(embryo_dir, f"{embryo_id}_attention_stats.png")
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"胚胎 {embryo_id} Grad-CAM结果已保存到: {embryo_dir}")

def embryo_level_voting(focal_results, voting_method='majority'):
    """
    胚胎级别投票机制
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
            final_pred = 1 if good_votes > poor_votes else 0
            final_confidence = max(avg_prob_good, avg_prob_poor)
            
        elif voting_method == 'probability':
            final_pred = 1 if avg_prob_good > avg_prob_poor else 0
            final_confidence = max(avg_prob_good, avg_prob_poor)
            
        elif voting_method == 'confidence':
            weights = group['confidence'].values
            weighted_good = np.average(group['prob_good'].values, weights=weights)
            weighted_poor = np.average(group['prob_poor'].values, weights=weights)
            final_pred = 1 if weighted_good > weighted_poor else 0
            final_confidence = max(weighted_good, weighted_poor)
            avg_prob_good = weighted_good
            avg_prob_poor = weighted_poor
            
        elif voting_method == 'unanimous_conservative':
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

def generate_embryo_gradcam_analysis(model, focal_results, embryo_results, gradcam_dir, max_embryos=None):
    """
    为选定的胚胎生成Grad-CAM分析
    """
    if not hasattr(model, '_gradcam_supported') or not model._gradcam_supported:
        print("模型不支持Grad-CAM，跳过可视化")
        return
    
    print(f"\n=== 生成Grad-CAM可视化 ===")
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # 选择要分析的胚胎
    if max_embryos is not None:
        selected_embryos = embryo_results.head(max_embryos)
    else:
        selected_embryos = embryo_results
    
    embryo_groups = focal_results.groupby('embryo_id')
    
    for _, embryo_row in selected_embryos.iterrows():
        embryo_id = embryo_row['embryo_id']
        if embryo_id in embryo_groups.groups:
            embryo_group = embryo_groups.get_group(embryo_id)
            create_embryo_gradcam_visualization(model, embryo_group, embryo_id, gradcam_dir)
    
    print(f"Grad-CAM分析完成，结果保存在: {gradcam_dir}")

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
    parser = argparse.ArgumentParser(description="使用训练好的STORK/VGG16模型预测胚胎质量（带投票机制和Grad-CAM）")
    
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
    
    # Grad-CAM相关参数
    parser.add_argument("--generate-gradcam", action="store_true",
                        help="是否生成Grad-CAM可视化")
    parser.add_argument("--gradcam-dir", type=str, default="gradcam_analysis",
                        help="Grad-CAM结果保存目录（默认: gradcam_analysis）")
    parser.add_argument("--max-embryos-gradcam", type=int, default=None,
                        help="生成Grad-CAM的最大胚胎数量（默认: 全部）")
    
    args = parser.parse_args()
    
    # 设置GPU
    set_gpu_memory_growth()
    
    print("=== STORK/VGG16 胚胎质量预测（投票机制+Grad-CAM）===")
    print(f"图像目录: {args.images_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"投票方法: {args.voting_method}")
    print(f"输出目录: {args.output_dir}")
    print(f"生成Grad-CAM: {args.generate_gradcam}")
    
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
    
    if hasattr(model, '_gradcam_supported'):
        print(f"Grad-CAM支持: {'是' if model._gradcam_supported else '否'}")
    
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
    
    # 生成Grad-CAM可视化
    if args.generate_gradcam:
        gradcam_path = os.path.join(args.output_dir, args.gradcam_dir)
        generate_embryo_gradcam_analysis(
            model, focal_results, embryo_results, 
            gradcam_path, args.max_embryos_gradcam
        )
    
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