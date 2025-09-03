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
    Input, GlobalAveragePooling2D, Dense, Dropout, Flatten
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
from preprocessing.data_loader import *
from preprocessing.image_utils import copy_images_by_split
from utils.data_utils import balance_classes, check_class_distribution
import cv2

# Build a pre-trained VGG16 model with custom top layers
def build_vgg16_model(input_shape=(224, 224, 3), num_classes=2, fine_tune_at=18, fine_tune=False):
    # Load the VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    if fine_tune:
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
        # Freeze the base model layers up to the specified layer
        # for layer in base_model.layers[fine_tune_at:]:
        #     layer.trainable = True
    else:
        base_model.trainable = False

    # Create the new model on top
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
  
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a learning rate schedule
    optimizer = Adam(learning_rate=1e-4, weight_decay=1e-5)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# check gpu availability
def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Print detailed information about each device
    for device in tf.config.list_physical_devices():
        print(device)

# Training the model with augmentation
def train_model_with_augmentation(images, labels, ages, qualities, num_epochs=3, global_batch_size=None, img_size=224, resume_training=True, fine_tune=False, copy_images=False):
    # Set strategy for distributed training
    strategy = tf.distribute.MirroredStrategy()
    
    start_time = time.time()
    
    # Check GPU availability
    check_gpu_availability()
    
    # 检查输入数据是否已经分割
    if len(images.shape) == 4 and images.shape[1] == 224 and images.shape[2] == 224:
        # 数据已经是预处理好的图像数组，直接使用
        print("检测到已预处理的数据，跳过数据分割步骤")
        X_train, y_train, age_train, quality_train = images, labels, ages, qualities
        
        # 创建虚拟的验证集（用于训练时的验证）
        # 这里可以从训练集中分出一部分作为验证集
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val, age_train, age_val, quality_train, quality_val = train_test_split(
            X_train, y_train, age_train, quality_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 创建虚拟的测试集（保持与原来接口一致）
        X_test, y_test, age_test, quality_test = X_val, y_val, age_val, quality_val
        
    else:
        # 原来的逻辑：平衡类别并分割数据
        print("使用原始数据分割逻辑")
        # Balance classes
        balanced_images, balanced_labels, balanced_ages, balanced_qualities = balance_classes(images, labels, ages, qualities)

        # Split the dataset with stratification
        X_train, X_temp, y_train, y_temp, age_train, age_temp, quality_train, quality_temp = train_test_split(
            balanced_images, balanced_labels, balanced_ages, balanced_qualities, test_size=0.3, random_state=42, stratify=balanced_labels
        )
        X_val, X_test, y_val, y_test, age_val, age_test, quality_val, quality_test = train_test_split(
            X_temp, y_temp, age_temp, quality_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

    # Output class distribution confirmation
    print(f"Train samples: {len(y_train)}, Validation: {len(y_val)}, Test: {len(y_test)}")
    print(f"Train positive ratio: {np.mean(y_train):.2f}, Validation: {np.mean(y_val):.2f}, Test: {np.mean(y_test):.2f}")
    check_class_distribution(y_train, y_val, y_test)

    with tf.device('/CPU:0'):
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.int64)

    # Define the model path and callbacks
    model_path = 'vgg16_best_model.h5'  # Save only the best model based on validation loss
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
        model = build_vgg16_model((img_size, img_size, 3), num_classes=2)

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
            model.save('vgg16_model_at_interruption.h5')
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
    
    # 创建模型，输出指定层的激活和目标类别的预测
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # 计算梯度
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]
    
    # 获取目标层的梯度
    grads = tape.gradient(loss, conv_outputs)
    
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
        auc_dl = roc_auc_score(y_eval, probabilities_dl[:, 1])
        print(f"Deep Learning Model {dataset_name} AUC: {auc_dl:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_eval, probabilities_dl[:, 1])
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

# Main function
if __name__ == "__main__":
    os.chdir("/liaozizhuo/scripts")
    data_dir = '/liaozizhuo/selected_images/'

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
    batch_size_per_replica = 64
    global_batch_size = batch_size_per_replica * max(1, num_gpus)

    # 使用新的数据加载函数
    print("=== 加载和分割数据 ===")
    (X_train, y_train, q_train,
     X_val, y_val, q_val,
     X_test, y_test, q_test) = load_data_with_split(
        data_dir, img_size=224, batch_size=global_batch_size, test_size=0.3, val_size=0.15
    )
    
    print(f"训练集: {len(y_train)} 样本")
    print(f"验证集: {len(y_val)} 样本") 
    print(f"测试集: {len(y_test)} 样本")
    
    # 训练模型（传入已分割的数据）
    print("=== 开始训练 ===")
    # 注意：这里需要修改train_model_with_augmentation函数，让它接收已分割的数据
    model, history = train_model_with_augmentation(
        X_train, y_train, np.zeros(len(y_train)), q_train, 
        num_epochs=2000, 
        global_batch_size=global_batch_size, 
        img_size=224, 
        resume_training=False,
        fine_tune=True,
        copy_images=False
    )
    
    # 评估模型（使用已分割的测试数据）
    print("=== 评估模型 ===")
    latest_model_path = 'vgg16_best_model.h5'
    if os.path.exists(latest_model_path):
        print(f"Evaluating model: {latest_model_path}")
        evaluate_deep_learning_model(X_test, y_test, latest_model_path, 
                                   save_path="test_auc.pdf",
                                   generate_gradcam_viz=True, 
                                   num_gradcam_samples=10)  # 减少到3个样本
    else:
        print("No model found at the specified path.")
