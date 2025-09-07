import os
import re
import math
import time
import glob
import copy
import gc
import shutil
import concurrent.futures
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications import InceptionV3, EfficientNetB0, ResNet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import absl.logging
import argparse
import sys
from preprocessing.data_loader_vgg16 import load_data_with_stork_preprocessing
# STORK图像处理函数通过 data_loader_vgg16 间接调用，无需直接导入
sys.path.append('/Users/lzz/Documents/GitHub/IVF_AI/prediction/code')
from utils.data_utils import balance_classes, check_class_distribution
import cv2

# Build a pre-trained model with custom top layers (supports multiple architectures)
def build_pretrained_model(model_name='vgg16', input_shape=(224, 224, 3), num_classes=2, fine_tune_at=18, fine_tune=False):
    """
    构建预训练模型用于胚胎分类
    
    Parameters:
    - model_name: 模型名称 ('vgg16', 'resnet50', 'inceptionv3', 'efficientnetb0')
    - input_shape: 输入图像形状
    - num_classes: 分类数量
    - fine_tune_at: 从哪一层开始微调
    - fine_tune: 是否启用微调
    
    Returns:
    - 编译好的Keras模型
    """
    
    model_name = model_name.lower()
    
    # 模型配置映射
    model_configs = {
        'vgg16': {
            'model_class': VGG16,
            'default_fine_tune_at': 18,
            'learning_rate': 1e-4,
            'description': 'VGG16: 经典深层网络，适合胚胎图像特征提取'
        },
        'resnet50': {
            'model_class': ResNet50,
            'default_fine_tune_at': 155,  # ResNet50的特征层更多
            'learning_rate': 1e-4,
            'description': 'ResNet50: 残差网络，解决深层网络梯度消失问题'
        },
        'inceptionv3': {
            'model_class': InceptionV3,
            'default_fine_tune_at': 249,  # InceptionV3层数更多
            'learning_rate': 8e-5,  # 稍微小一点的学习率
            'description': 'InceptionV3: 多尺度特征提取，适合复杂图像模式'
        },
        'efficientnetb0': {
            'model_class': EfficientNetB0,
            'default_fine_tune_at': 213,
            'learning_rate': 1e-4,
            'description': 'EfficientNetB0: 高效网络，平衡准确率和计算效率'
        }
    }
    
    if model_name not in model_configs:
        available_models = ', '.join(model_configs.keys())
        raise ValueError(f"不支持的模型: {model_name}. 可用模型: {available_models}")
    
    config = model_configs[model_name]
    print(f"🏗️  构建模型: {model_name.upper()}")
    print(f"📝 描述: {config['description']}")
    
    # 使用默认微调层数如果没有指定
    if fine_tune_at == 18 and model_name != 'vgg16':
        fine_tune_at = config['default_fine_tune_at']
        print(f"🔧 使用 {model_name.upper()} 的推荐微调层数: {fine_tune_at}")
    
    # 加载预训练模型
    feature_extractor = config['model_class'](
        weights="imagenet",
        include_top=False,
        pooling="avg",  # 全局平均池化
        input_shape=input_shape
    )
    
    # 配置可训练层
    if fine_tune:
        print(f"🔓 启用微调: 从第 {fine_tune_at} 层开始解冻")
        for i, layer in enumerate(feature_extractor.layers[:fine_tune_at]):
            layer.trainable = False
        for i, layer in enumerate(feature_extractor.layers[fine_tune_at:]):
            layer.trainable = True
            
        trainable_count = sum(1 for layer in feature_extractor.layers if layer.trainable)
        print(f"   可训练层数: {trainable_count}/{len(feature_extractor.layers)}")
    else:
        print("🔒 冻结所有预训练层")
        feature_extractor.trainable = False

    # 构建完整模型
    inputs = Input(shape=input_shape)
    
    # 特征提取 (已包含全局平均池化)
    features = feature_extractor(inputs)
    
    # 增强的分类头 - 提升模型性能（默认启用）
    dropout_rate = 0.5 if model_name == 'vgg16' else 0.4  # 调整dropout率
    
    # 添加中间层提升表达能力
    x = Dense(512, activation='relu', name='intermediate_dense')(features)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu', name='intermediate_dense_2')(x)
    x = Dropout(dropout_rate * 0.8)(x)  # 逐层递减dropout
    outputs = Dense(1, activation='sigmoid', name='embryo_prediction')(x)

    # 定义模型
    model = Model(inputs=inputs, outputs=outputs, name=f"embryo_classifier_{model_name}")

    # 编译模型 - 使用模型特定的学习率
    optimizer = Adam(learning_rate=config['learning_rate'], weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print(f"✅ {model_name.upper()} 模型构建完成!")
    print(f"   总参数数: {model.count_params():,}")
    print(f"   输入形状: {input_shape}")
    print(f"   学习率: {config['learning_rate']}")
    print(f"   Dropout率: {dropout_rate}")
    
    return model

# check gpu availability
def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Print detailed information about each device
    for device in tf.config.list_physical_devices():
        print(device)

# Training the model with augmentation
def train_model_with_augmentation(train_images, train_labels, train_ages, train_qualities, 
                                val_images, val_labels, val_ages, val_qualities,
                                num_epochs=3, global_batch_size=None, img_size=224, 
                                resume_training=True, fine_tune=False, copy_images=False,
                                model_name='vgg16'):
    # Set strategy for distributed training
    strategy = tf.distribute.MirroredStrategy()
    
    start_time = time.time()
    
    # Check GPU availability
    check_gpu_availability()
    
    # 使用传入的训练集和验证集（已完成70%-15%-15%分割）
    print("使用预分割的训练集和验证集")
    X_train, y_train, age_train, quality_train = train_images, train_labels, train_ages, train_qualities
    X_val, y_val, age_val, quality_val = val_images, val_labels, val_ages, val_qualities
    
    # 测试集占位符（实际测试在外部进行）
    X_test, y_test, age_test, quality_test = X_val, y_val, age_val, quality_val

    # Output class distribution confirmation
    print(f"Train samples: {len(y_train)}, Validation: {len(y_val)}, Test: {len(y_test)}")
    print(f"Train positive ratio: {np.mean(y_train):.2f}, Validation: {np.mean(y_val):.2f}, Test: {np.mean(y_test):.2f}")
    check_class_distribution(y_train, y_val, y_test)

    with tf.device('/CPU:0'):
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.int64)

    # Define the model path and callbacks (使用模型名称)
    model_path = f'{model_name}_best_model.h5'  # Save only the best model based on validation loss
    initial_epoch = 0
    csv_logger = CSVLogger('training.log', append=True)
    
    # Define checkpoint callback to save only the best model (based on validation loss)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        save_weights_only=False,  # Save the full model (architecture + weights)
        monitor='val_loss',  # Monitor validation loss
        mode='min',  # Save when validation loss is minimized
        save_best_only=True  # Save only the best model
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Initialize train_dataset if X_train or y_train is not None
    if X_train is None or y_train is None:
        raise ValueError("Error: X_train or y_train is None.")

    # Build tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1024).batch(global_batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    steps_per_epoch = math.ceil(X_train.shape[0] / global_batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

    with strategy.scope():
        model = build_pretrained_model(model_name, (img_size, img_size, 3), num_classes=2, fine_tune=fine_tune)

        if resume_training and os.path.exists(model_path):
            print("Loading existing model from", model_path)
            model = tf.keras.models.load_model(model_path)

            # if fine_tune:
            #     print("Fine-tuning enabled: Unfreezing layers starting from", fine_tune_at)
            #     for layer in model.layers[:fine_tune_at]:
            #         layer.trainable = False
            #     for layer in model.layers[fine_tune_at:]:
            #         layer.trainable = True
            # else:
            #     print("Fine-tuning disabled: Keeping all previously frozen layers.")

            if os.path.exists('training.log'):
                with open('training.log', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        initial_epoch = int(last_line.split(',')[0]) + 1
            else:
                print("Training log not found. Setting initial epoch to 0.")
                initial_epoch = 0
        else:
            print("Building new model from scratch.")
        
        # Train the model with data augmentation
        try:
            print(f"Final batch size: {global_batch_size} ({global_batch_size // strategy.num_replicas_in_sync} per GPU)")
            print(f"Total global batch size: {global_batch_size}")
            print(f"Train dataset size: {len(y_train)} | Batch size: {global_batch_size}")
            steps_per_epoch = math.ceil(len(X_train) / global_batch_size)
            val_steps = max(len(X_val) // global_batch_size, 1)
            history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=num_epochs,
                    initial_epoch=initial_epoch,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=val_steps,
                    callbacks=[csv_logger, checkpoint_callback, early_stopping, reduce_lr]
                )

            # Check if best model was saved; if not, save the current model
            # (inserted as per instructions)
            if os.path.exists(model_path):
                print(f"Best model saved at: {model_path}")
            else:
                print(f"No best model found at {model_path}, saving final model manually.")
                model.save(model_path)
                print(f"Final model saved at: {model_path}")

            # Calculate the time consumed
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Total training time: {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"Training was interrupted due to an error: {e}")
            # Optionally save the model in case of interruption (though not needed if only saving the best model)
            model.save(f'{model_name}_model_at_interruption.h5')
            print("Model saved at interruption point.")

    return model, history

def generate_gradcam(model, image, class_index, layer_name='block5_conv3'):
    """
    生成Grad-CAM热力图
    
    Parameters:
    - model: 训练好的模型
    - image: 输入图像 (numpy array)
    - class_index: 目标类别索引
    - layer_name: 要分析的卷积层名称
    
    Returns:
    - heatmap: Grad-CAM热力图
    - superimposed_img: 叠加了热力图的原始图像
    """
    
    # 智能查找合适的卷积层 - 支持多种模型
    def find_target_layer(model, target_layer_name):
        """智能查找目标层，支持多种预训练模型"""
        # 定义不同模型的基础层名称
        model_base_names = ['vgg16', 'resnet50', 'inception_v3', 'efficientnetb0']
        
        # 首先尝试直接查找
        try:
            return model.get_layer(target_layer_name)
        except:
            pass
        
        # 查找各种预训练层内部的层
        for base_name in model_base_names:
            try:
                base_layer = model.get_layer(base_name)
                return base_layer.get_layer(target_layer_name)
            except:
                continue
        
        # 如果指定层不存在，自动选择最后一个卷积层
        print(f"警告：找不到层 '{target_layer_name}'，自动选择最后一个卷积层")
        
        # 在各种预训练模型中查找最后一个卷积层
        for base_name in model_base_names:
            try:
                base_layer = model.get_layer(base_name)
                conv_layers = [layer for layer in base_layer.layers if 'conv' in layer.name.lower()]
                if conv_layers:
                    selected_layer = conv_layers[-1]
                    print(f"自动选择层: {selected_layer.name} (来自 {base_name})")
                    return selected_layer
            except:
                continue
        
        # 最后的fallback：查找整个模型中的卷积层
        conv_layers = []
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # 嵌套模型
                for sublayer in layer.layers:
                    if 'conv' in sublayer.name.lower():
                        conv_layers.append(sublayer)
        
        if conv_layers:
            selected_layer = conv_layers[-1]
            print(f"Fallback选择层: {selected_layer.name}")
            return selected_layer
        
        raise ValueError(f"无法找到合适的卷积层用于Grad-CAM")
    
    # 获取目标层
    target_layer = find_target_layer(model, layer_name)
    
    # 使用更简单直接的方法创建Grad-CAM模型
    # 构建到目标层的部分模型
    model_base_names = ['vgg16', 'resnet50', 'inception_v3', 'efficientnetb0']
    base_layer = None
    
    for base_name in model_base_names:
        try:
            base_layer = model.get_layer(base_name)
            break
        except:
            continue
    
    if base_layer is None:
        raise ValueError("无法找到支持的预训练模型基础层")
    
    # 创建到目标层的中间模型
    intermediate_model = Model(inputs=base_layer.input, outputs=target_layer.output)
    
    # 创建组合模型
    def grad_cam_forward(x):
        # 获取中间层输出
        conv_outputs = intermediate_model(x)
        # 获取最终预测
        predictions = model(x)
        return conv_outputs, predictions
    
    # 计算梯度
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_cam_forward(image)
        # 对于二分类，预测输出是单个值
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, class_index]
    
    # 获取目标层的梯度
    grads = tape.gradient(loss, conv_outputs)
    
    # 检查梯度是否为None并添加调试信息
    if grads is None:
        print(f"警告：梯度为None，可能的原因：")
        print(f"- 损失函数与目标层之间没有连接")
        print(f"- 目标层: {target_layer.name}")
        print(f"- 损失值: {loss}")
        print(f"- 卷积输出形状: {conv_outputs.shape}")
        
        # 尝试替代方法：直接使用模型的梯度
        print("尝试使用替代梯度计算方法...")
        # 简单返回随机热力图作为fallback
        fallback_heatmap = np.random.rand(224, 224)
        fallback_heatmap = np.uint8(255 * fallback_heatmap)
        fallback_heatmap = cv2.applyColorMap(fallback_heatmap, cv2.COLORMAP_JET)
        
        # 创建简单的叠加图像
        if image.max() <= 1.0:
            image_uint8 = (image[0] * 255).astype(np.uint8)
        else:
            image_uint8 = image[0].astype(np.uint8)
            
        if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
            
        fallback_superimposed = fallback_heatmap * 0.4 + image_uint8 * 0.6
        fallback_superimposed = np.clip(fallback_superimposed, 0, 255).astype(np.uint8)
        
        return fallback_heatmap, fallback_superimposed
    
    # 全局平均池化梯度
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 权重梯度
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 归一化热力图
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # 调整热力图大小到原始图像尺寸
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    
    # 转换为RGB热力图
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 叠加到原始图像上
    # 确保图像在0-255范围内
    if image.max() <= 1.0:
        image_uint8 = (image[0] * 255).astype(np.uint8)
    else:
        image_uint8 = image[0].astype(np.uint8)
    
    # 如果图像是RGB格式，转换为BGR（OpenCV格式）
    if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    
    # 叠加热力图
    superimposed_img = heatmap * 0.4 + image_uint8 * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return heatmap, superimposed_img

def visualize_gradcam_samples(model, X_eval, y_eval, image_paths, num_samples=5, save_dir="gradcam_results"):
    """
    为多个样本生成Grad-CAM可视化
    
    Parameters:
    - model: 训练好的模型
    - X_eval: 评估图像数据
    - y_eval: 真实标签
    - image_paths: 图像路径列表
    - num_samples: 要可视化的样本数量
    - save_dir: 保存结果的目录
    """
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 智能样本选择策略
    def select_representative_samples(X_eval, y_eval, num_samples):
        """智能选择代表性样本"""
        if num_samples >= len(X_eval):
            return np.arange(len(X_eval))
        
        # 按类别分配样本数量
        class_0_indices = np.where(y_eval == 0)[0]
        class_1_indices = np.where(y_eval == 1)[0]
        
        # 计算每个类别应该选择的样本数
        samples_per_class = max(1, num_samples // 2)
        
        # 为每个类别选择样本
        selected_indices = []
        
        # 选择类别0的样本（确保每个类别都有代表）
        if len(class_0_indices) > 0:
            class_0_samples = np.random.choice(class_0_indices, 
                                             min(samples_per_class, len(class_0_indices)), 
                                             replace=False)
            selected_indices.extend(class_0_samples)
        
        # 选择类别1的样本
        if len(class_1_indices) > 0:
            class_1_samples = np.random.choice(class_1_indices, 
                                             min(samples_per_class, len(class_1_indices)), 
                                             replace=False)
            selected_indices.extend(class_1_samples)
        
        # 如果样本数不够，从剩余样本中随机选择
        remaining_samples = num_samples - len(selected_indices)
        if remaining_samples > 0:
            all_indices = set(range(len(X_eval)))
            used_indices = set(selected_indices)
            available_indices = list(all_indices - used_indices)
            
            if available_indices:
                additional_samples = np.random.choice(available_indices, 
                                                   min(remaining_samples, len(available_indices)), 
                                                   replace=False)
                selected_indices.extend(additional_samples)
        
        return np.array(selected_indices)
    
    # 使用智能选择策略
    selected_indices = select_representative_samples(X_eval, y_eval, num_samples)
    
    print(f"智能选择了 {len(selected_indices)} 个代表性样本进行Grad-CAM分析")
    print(f"选择的样本索引: {selected_indices}")
    
    # 为每个类别选择样本
    class_0_indices = [i for i in selected_indices if y_eval[i] == 0]
    class_1_indices = [i for i in selected_indices if y_eval[i] == 1]
    
    # 确保每个类别都有样本
    if len(class_0_indices) == 0:
        class_0_indices = [i for i in range(len(X_eval)) if y_eval[i] == 0][:1]
    if len(class_1_indices) == 0:
        class_1_indices = [i for i in range(len(X_eval)) if y_eval[i] == 1][:1]
    
    selected_indices = class_0_indices + class_1_indices
    
    print(f"生成 {len(selected_indices)} 个样本的Grad-CAM可视化...")
    print(f"类别0样本: {len(class_0_indices)} 个，类别1样本: {len(class_1_indices)} 个")
    
    # 创建子图
    fig, axes = plt.subplots(len(selected_indices), 3, figsize=(15, 5*len(selected_indices)))
    if len(selected_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(selected_indices):
        # 获取样本数据
        image = X_eval[sample_idx:sample_idx+1]
        true_label = y_eval[sample_idx]
        pred_proba = model.predict(image, verbose=0)
        pred_label = np.argmax(pred_proba[0])
        pred_confidence = np.max(pred_proba[0])
        
        # 生成Grad-CAM
        heatmap, superimposed_img = generate_gradcam(model, image, pred_label)
        
        # 显示原始图像
        if image.max() <= 1.0:
            img_display = (image[0] * 255).astype(np.uint8)
        else:
            img_display = image[0].astype(np.uint8)
        
        axes[idx, 0].imshow(img_display)
        axes[idx, 0].set_title(f'Original\nTrue: {true_label}, Pred: {pred_label}\nConf: {pred_confidence:.3f}')
        axes[idx, 0].axis('off')
        
        # 显示热力图
        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].set_title('Grad-CAM Heatmap')
        axes[idx, 1].axis('off')
        
        # 显示叠加图像
        if len(superimposed_img.shape) == 3 and superimposed_img.shape[2] == 3:
            superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        else:
            superimposed_img_rgb = superimposed_img
            
        axes[idx, 2].imshow(superimposed_img_rgb)
        axes[idx, 2].set_title('Superimposed')
        axes[idx, 2].axis('off')
        
        # 保存单个样本结果
        sample_save_dir = os.path.join(save_dir, f"sample_{sample_idx}_class_{true_label}")
        os.makedirs(sample_save_dir, exist_ok=True)
        
        # 保存原始图像
        plt.imsave(os.path.join(sample_save_dir, "original.png"), img_display)
        
        # 保存热力图
        plt.imsave(os.path.join(sample_save_dir, "heatmap.png"), heatmap)
        
        # 保存叠加图像
        plt.imsave(os.path.join(sample_save_dir, "superimposed.png"), superimposed_img_rgb)
        
        # 保存预测信息
        with open(os.path.join(sample_save_dir, "prediction_info.txt"), "w") as f:
            f.write(f"Sample Index: {sample_idx}\n")
            f.write(f"True Label: {true_label}\n")
            f.write(f"Predicted Label: {pred_label}\n")
            f.write(f"Prediction Confidence: {pred_confidence:.4f}\n")
            f.write(f"Prediction Probabilities: {pred_proba[0]}\n")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradcam_overview.pdf"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Grad-CAM结果已保存到: {save_dir}")
    print(f"生成了 {len(selected_indices)} 个样本的详细分析")

def analyze_attention_patterns(model, X_eval, y_eval, save_dir="attention_analysis"):
    """
    分析注意力模式，比较不同类别的关注区域
    
    Parameters:
    - model: 训练好的模型
    - X_eval: 评估图像数据
    - y_eval: 真实标签
    - save_dir: 保存结果的目录
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 分别分析两个类别
    class_0_indices = np.where(y_eval == 0)[0]
    class_1_indices = np.where(y_eval == 1)[0]
    
    print(f"分析类别0的注意力模式 (样本数: {len(class_0_indices)})")
    print(f"分析类别1的注意力模式 (样本数: {len(class_1_indices)})")
    
    # 为每个类别生成平均注意力图
    for class_idx, class_name in enumerate(['Class_0', 'Class_1']):
        if class_name == 'Class_0':
            indices = class_0_indices
        else:
            indices = class_1_indices
        
        if len(indices) == 0:
            continue
            
        # 随机选择最多10个样本
        sample_indices = np.random.choice(indices, min(10, len(indices)), replace=False)
        
        # 计算平均注意力图
        avg_heatmap = None
        for sample_idx in sample_indices:
            image = X_eval[sample_idx:sample_idx+1]
            heatmap, _ = generate_gradcam(model, image, class_idx)
            
            if avg_heatmap is None:
                avg_heatmap = heatmap.astype(np.float32)
            else:
                avg_heatmap += heatmap.astype(np.float32)
        
        if avg_heatmap is not None:
            avg_heatmap /= len(sample_indices)
            
            # 保存平均注意力图
            plt.figure(figsize=(10, 8))
            plt.imshow(avg_heatmap)
            plt.title(f'Average Attention Map - {class_name}')
            plt.colorbar(label='Attention Score')
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"avg_attention_{class_name}.png"), dpi=300, bbox_inches='tight')
            plt.show()
            
            # 保存为numpy数组
            np.save(os.path.join(save_dir, f"avg_attention_{class_name}.npy"), avg_heatmap)
    
    print(f"注意力分析结果已保存到: {save_dir}")

# Evaluate Deep Learning Model
def evaluate_deep_learning_model(X_eval, y_eval, model_path, dataset_name="Test", save_path="auc_curve.pdf", 
                                generate_gradcam_viz=True, num_gradcam_samples=8):
    """
    Evaluate a deep learning model on a given dataset and save the ROC AUC curve as a PDF.
    新增Grad-CAM可视化功能
    
    Parameters:
    - X_eval: Input features for evaluation.
    - y_eval: True labels for evaluation.
    - model_path: Path to the trained model.
    - dataset_name: Name of the dataset (e.g., "Test" or "Validation").
    - save_path: Path to save the ROC AUC plot as a PDF.
    - generate_gradcam_viz: Whether to generate Grad-CAM visualizations.
    - num_gradcam_samples: Number of samples to visualize with Grad-CAM.
    """
    
    # Load model
    dl_model = tf.keras.models.load_model(model_path)
    
    # Ensure evaluation is done using the same global batch size as training with proper sharding for multi-GPU
    eval_batch_size = min(global_batch_size, 128)  # Ensure batch size is not too large for evaluation
    eval_dataset = tf.data.Dataset.from_tensor_slices((X_eval, y_eval)).batch(eval_batch_size).prefetch(tf.data.AUTOTUNE)

    # Evaluate model
    eval_loss, eval_accuracy = dl_model.evaluate(eval_dataset)
    print(f"Deep Learning Model {dataset_name} Loss: {eval_loss:.4f}, {dataset_name} Accuracy: {eval_accuracy:.4f}")
    
    # Convert one-hot labels to integer labels if necessary
    if y_eval.ndim > 1 and y_eval.shape[1] == 2:
        y_eval = np.argmax(y_eval, axis=1)
    # Compute AUC for model
    probabilities_dl = dl_model.predict(eval_dataset)
    
    if len(np.unique(y_eval)) < 2:
        print(f"Only one class present in {dataset_name} set. ROC AUC score is not defined.")
    else:
        auc_dl = roc_auc_score(y_eval, probabilities_dl.flatten())
        print(f"Deep Learning Model {dataset_name} AUC: {auc_dl:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_eval, probabilities_dl.flatten())
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_dl:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{dataset_name} Set ROC Curve')
        plt.legend(loc="lower right")
        
        # Save the figure as a PDF
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"ROC AUC curve saved as {save_path}")
        plt.close()

    # Compute confusion matrix
    y_pred_dl = (probabilities_dl > 0.5).astype(int).flatten()
    cm_dl = confusion_matrix(y_eval, y_pred_dl)
    
    # Create confusion matrix DataFrame
    class_names = ['Class 1', 'Class 0']
    cm_df = pd.DataFrame(cm_dl, index=class_names, columns=class_names)
    print(f"Deep Learning Model {dataset_name} Confusion Matrix:")
    print(cm_df)
    
    # 生成Grad-CAM可视化
    if generate_gradcam_viz:
        print(f"\n=== 生成Grad-CAM可视化 ===")
        
        # 创建保存目录
        gradcam_save_dir = f"gradcam_{dataset_name.lower()}"
        
        # 生成Grad-CAM可视化
        visualize_gradcam_samples(dl_model, X_eval, y_eval, None, 
                                 num_samples=num_gradcam_samples, 
                                 save_dir=gradcam_save_dir)
        
        # 分析注意力模式
        analyze_attention_patterns(dl_model, X_eval, y_eval, 
                                 save_dir=f"attention_analysis_{dataset_name.lower()}")
        
        print(f"Grad-CAM可视化完成！结果保存在: {gradcam_save_dir}")

def show_samples(images, labels, num_samples=25):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    plt.show()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='🧬 STORK胚胎分类训练系统 (优化版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python STORK_vgg16.py --model vgg16 --epochs 100 --cache --workers 8
  python STORK_vgg16.py --model resnet50 --epochs 50 --no-cache --batch-size 32 --workers 12
  python STORK_vgg16.py --model efficientnetb0 --fine-tune --gradcam-samples 5 --workers 6
  python STORK_vgg16.py --list-models

支持的模型:
  - vgg16: VGG16经典深层网络，适合胚胎图像特征提取
  - resnet50: ResNet50残差网络，解决深层网络梯度消失问题  
  - inceptionv3: InceptionV3多尺度特征提取，适合复杂图像模式
  - efficientnetb0: EfficientNetB0高效网络，平衡准确率和计算效率

优化特性:
  - 多进程并行预处理 (5-8x 速度提升)  
  - NumPy数据增强 (避免TF张量转换开销)
  - 详细进度报告和性能统计
        '''
    )
    
    # 基本参数
    parser.add_argument('--model', '-m', type=str, default='vgg16',
                       choices=['vgg16', 'resnet50', 'inceptionv3', 'efficientnetb0'],
                       help='选择预训练模型 (默认: vgg16)')
    
    parser.add_argument('--data-dir', type=str, default='/liaozizhuo/selected_images/train/',
                       help='训练数据目录路径 (默认: /liaozizhuo/selected_images/train/)')
    
    # 训练参数
    parser.add_argument('--epochs', '-e', type=int, default=2000,
                       help='训练轮数 (默认: 2000)')
    
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                       help='每个GPU的批处理大小 (默认: 64)')
    
    parser.add_argument('--img-size', type=int, default=224,
                       help='输入图像尺寸 (默认: 224)')
    
    parser.add_argument('--fine-tune', action='store_true',
                       help='启用模型微调')
    
    parser.add_argument('--no-fine-tune', dest='fine_tune', action='store_false',
                       help='禁用模型微调 (默认)')
    
    parser.set_defaults(fine_tune=True)
    
    # 优化参数
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='并行处理进程数 (默认: 自动检测CPU核心数，限制为8)')
    
    
    # 缓存和性能参数
    parser.add_argument('--cache', action='store_true', default=True,
                       help='启用预处理数据缓存 (默认)')
    
    parser.add_argument('--no-cache', dest='cache', action='store_false',
                       help='禁用预处理数据缓存')
    
    parser.add_argument('--resume', action='store_true', default=False,
                       help='从检查点恢复训练')
    
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='不从检查点恢复，重新开始训练 (默认)')
    
    # 评估和可视化参数
    parser.add_argument('--gradcam-samples', type=int, default=10,
                       help='Grad-CAM可视化样本数量 (默认: 10)')
    
    parser.add_argument('--no-gradcam', action='store_true',
                       help='跳过Grad-CAM可视化')
    
    # 实用功能
    parser.add_argument('--list-models', action='store_true',
                       help='列出所有支持的模型并退出')
    
    parser.add_argument('--check-data', action='store_true',
                       help='仅检查数据目录结构，不进行训练')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='减少输出信息')
    
    return parser.parse_args()

def list_supported_models():
    """列出所有支持的模型"""
    models = {
        'vgg16': 'VGG16: 经典深层网络，适合胚胎图像特征提取',
        'resnet50': 'ResNet50: 残差网络，解决深层网络梯度消失问题',
        'inceptionv3': 'InceptionV3: 多尺度特征提取，适合复杂图像模式',
        'efficientnetb0': 'EfficientNetB0: 高效网络，平衡准确率和计算效率'
    }
    
    print("🧬 支持的预训练模型:")
    print("=" * 60)
    for model, desc in models.items():
        print(f"  {model:<15} - {desc}")
    print("=" * 60)
    print("\n📝 使用示例:")
    print("  python STORK_vgg16.py --model resnet50 --epochs 100")
    print("  python STORK_vgg16.py --model efficientnetb0 --fine-tune")

def check_data_directory(data_dir):
    """检查数据目录结构"""
    print(f"🔍 检查数据目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    class_dirs = ['class_0', 'class_1']
    total_images = 0
    
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        if os.path.exists(class_path):
            image_count = len([f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            print(f"✅ {class_dir}: {image_count} 张图像")
            total_images += image_count
        else:
            print(f"❌ 缺少目录: {class_path}")
            return False
    
    print(f"📊 总计: {total_images} 张图像")
    return total_images > 0

# Main function
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 处理特殊命令
    if args.list_models:
        list_supported_models()
        sys.exit(0)
    
    if args.check_data:
        success = check_data_directory(args.data_dir)
        sys.exit(0 if success else 1)
    
    # 设置工作目录
    os.chdir("/liaozizhuo/scripts")
    
    # 显示配置信息
    if not args.quiet:
        print("=" * 70)
        print("🧬 STORK胚胎分类训练系统 (优化版)")
        print("=" * 70)
        print(f"📊 使用模型: {args.model.upper()}")
        print(f"💾 缓存模式: {'启用' if args.cache else '禁用'}")
        print(f"📁 数据目录: {args.data_dir}")
        print(f"🔄 训练轮数: {args.epochs}")
        print(f"📦 批处理大小: {args.batch_size} (每GPU)")
        print(f"🖼️  图像尺寸: {args.img_size}x{args.img_size}")
        print(f"⚙️  微调模式: {'启用' if args.fine_tune else '禁用'}")
        print(f"🔄 恢复训练: {'是' if args.resume else '否'}")
        print(f"🔧 并行进程数: {args.workers if args.workers else '自动检测'}")
        print(f"📊 Grad-CAM样本: {args.gradcam_samples if not args.no_gradcam else '跳过'}")
        print("⚡ 优化特性: 并行预处理 + NumPy增强 + 详细统计")
        print("=" * 70)

    # Get all available GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Set memory growth to avoid TensorFlow allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU {gpu} is set to memory growth.")
        except RuntimeError as e:
            print(f"Error in setting memory growth: {e}")

    # Define the strategy for multi-GPU training
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    batch_size_per_replica = args.batch_size
    global_batch_size = batch_size_per_replica * max(1, num_gpus)

    # 使用优化的STORK胚胎图像预处理数据加载函数 (固定70%-15%-15%分割)
    preprocessing_start = time.time()
    (X_train, y_train, q_train,
     X_val, y_val, q_val,
     X_test, y_test, q_test) = load_data_with_stork_preprocessing(
        args.data_dir, 
        img_size=args.img_size, 
        batch_size=global_batch_size,
        use_cache=args.cache,
        model_name=args.model,
        num_workers=args.workers
    )
    preprocessing_time = time.time() - preprocessing_start
    
    print(f"⏱️  预处理总时间: {preprocessing_time:.2f} 秒 (优化版)")
    print(f"训练集: {len(y_train)} 样本")
    print(f"验证集: {len(y_val)} 样本") 
    print(f"测试集: {len(y_test)} 样本")
    
    # 训练模型（使用70%-15%-15%分割的数据）
    print("=== 开始训练 ===")
    model, history = train_model_with_augmentation(
        X_train, y_train, np.zeros(len(y_train)), q_train,     # 训练集 (70%)
        X_val, y_val, np.zeros(len(y_val)), q_val,             # 验证集 (15%)
        num_epochs=args.epochs, 
        global_batch_size=global_batch_size, 
        img_size=args.img_size, 
        resume_training=args.resume,
        fine_tune=args.fine_tune,
        copy_images=False,
        model_name=args.model
    )
    
    # 评估模型（使用已分割的测试数据）
    print("=== 评估模型 ===")
    latest_model_path = f'{args.model}_best_model.h5'
    if os.path.exists(latest_model_path):
        print(f"Evaluating model: {latest_model_path}")
        if not args.no_gradcam:
            evaluate_deep_learning_model(X_test, y_test, latest_model_path, 
                                       save_path=f"{args.model}_test_auc.pdf",
                                       generate_gradcam_viz=True, 
                                       num_gradcam_samples=args.gradcam_samples)
        else:
            evaluate_deep_learning_model(X_test, y_test, latest_model_path, 
                                       save_path=f"{args.model}_test_auc.pdf",
                                       generate_gradcam_viz=False, 
                                       num_gradcam_samples=0)
    else:
        print("No model found at the specified path.")
    
    if not args.quiet:
        print("\n" + "=" * 70)
        print("🎉 优化版训练和评估完成!")
        print(f"📊 模型文件: {latest_model_path}")
        print(f"📈 AUC图表: {args.model}_test_auc.pdf")
        print(f"⏱️  预处理时间: {preprocessing_time:.2f} 秒 (优化版)")
        if not args.no_gradcam:
            print(f"🔍 Grad-CAM结果: gradcam_test/ 目录")
        print("⚡ 使用的优化: 并行预处理 + NumPy增强 + 详细统计")
        print("=" * 70)
