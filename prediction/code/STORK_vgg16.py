import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from preprocessing.data_loader_vgg16 import load_data_vgg16, preprocess_image_vgg16  # 修正导入
from utils.data_utils import balance_classes, check_class_distribution
import seaborn as sns
import glob

def build_vgg16_model(input_shape=(224, 224, 3), num_classes=2):
    """
    构建VGG16模型
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # 添加分类头
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model, learning_rate=1e-4):
    """编译模型"""
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]  # 添加AUC指标
    )
    return model

def setup_callbacks(model_path='vgg16_embryo_quality_model'):
    """设置训练回调函数"""
    callbacks = [
        CSVLogger('embryo_training.log', append=True),
        ModelCheckpoint(
            filepath=model_path,
            save_weights_only=False,
            monitor='val_auc',  # 改为监控AUC
            mode='max',  # AUC越大越好
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',  # 改为监控AUC
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_auc',  # 改为监控AUC
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks

def create_balanced_dataset(data_dir, img_size=224, apply_balance=True):
    """
    使用VGG16兼容的数据加载，简化版本
    """
    print("=== 数据加载和平衡（VGG16兼容）===")
    
    # 使用data_loader_vgg16中的函数
    images, labels, qualities, image_paths = load_data_vgg16(
        data_dir, img_size=img_size, batch_size=1000, is_training=True
    )
    
    print(f"原始数据分布:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} 张图像")
    
    # 不再在这里平衡，直接返回原始数据
    return images, labels, qualities

def create_datasets(images, labels, batch_size=32, validation_split=0.2):
    """
    创建数据集（图像已经预处理过了）
    """
    train_indices, val_indices = train_test_split(
        range(len(images)), 
        test_size=validation_split, 
        random_state=42, 
        stratify=labels
    )
    
    # 直接创建数据集，图像已经预处理过
    train_dataset = tf.data.Dataset.from_tensor_slices((images[train_indices], labels[train_indices]))
    train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((images[val_indices], labels[val_indices]))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, train_indices, val_indices

# 将数据增强函数移到模块级别
def augment_image(image, label):
    """数据增强函数"""
    # 随机水平翻转
    image = tf.image.random_flip_left_right(image)
    # 随机亮度变化
    image = tf.image.random_brightness(image, max_delta=0.1)
    # 随机对比度变化
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # 随机旋转（小角度）- 使用兼容的旋转方法
    angle = tf.random.uniform([], -0.1, 0.1)
    # 使用 tf.image.rot90 的替代方案
    image = tf.image.rot90(image, k=tf.cast(angle * 10, tf.int32) % 4)
    return image, label

def create_datasets_with_augmentation(images, labels, batch_size=32, validation_split=0.2):
    """创建带数据增强的训练数据集"""
    train_indices, val_indices = train_test_split(
        range(len(images)), 
        test_size=validation_split, 
        random_state=42, 
        stratify=labels
    )
    
    # 训练集带增强
    train_dataset = tf.data.Dataset.from_tensor_slices((images[train_indices], labels[train_indices]))
    train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 验证集不增强
    val_dataset = tf.data.Dataset.from_tensor_slices((images[val_indices], labels[val_indices]))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, train_indices, val_indices

def find_vgg16_base_model(model):
    """
    更鲁棒的VGG16基础模型查找
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            # 检查是否是VGG16模型
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                # 检查是否有VGG16特征层
                layer_names = [l.name for l in layer.layers]
                if any('block' in name and 'conv' in name for name in layer_names):
                    return layer
    return None

def train_embryo_quality_model(data_dir, 
                             num_epochs=50, 
                             batch_size=32, 
                             img_size=224, 
                             learning_rate=1e-4,
                             fine_tune=True,
                             apply_balance=True):
    """训练胚胎质量预测模型"""
    start_time = time.time()
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # 1. 先加载原始数据（不平衡）
    try:
        images, labels, qualities = create_balanced_dataset(
            data_dir, img_size, apply_balance=False
        )
        if len(images) == 0:
            raise ValueError("没有加载到任何图像数据")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None
    
    # 2. 先进行数据划分（保持原始分布）
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images, labels,
        test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"\n=== 数据划分 ===")
    print(f"总数据: {len(images)}")
    print(f"训练+验证: {len(train_val_images)}, 测试: {len(test_images)}")
    
    # 3. 仅对训练集进行平衡
    if apply_balance:
        print("对训练集应用样本平衡...")
        dummy_ages = np.zeros(len(train_val_labels))
        
        balanced_images, balanced_labels, _, balanced_qualities = balance_classes(
            train_val_images, train_val_labels, dummy_ages, qualities[:len(train_val_labels)]
        )
        
        print(f"平衡后训练集分布:")
        unique, counts = np.unique(balanced_labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} 张图像")
        
        train_val_images = balanced_images
        train_val_labels = balanced_labels
    
    # 4. 创建带增强的训练数据集
    train_dataset, val_dataset, train_idx, val_idx = create_datasets_with_augmentation(
        train_val_images, train_val_labels, batch_size, validation_split=0.22
    )
    
    print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}")
    
    # 5. 检查最终的类别分布
    print(f"\n=== 最终类别分布 ===")
    check_class_distribution(
        train_val_labels[train_idx], 
        train_val_labels[val_idx], 
        test_labels
    )
    
    # 6. 构建并编译模型
    try:
        model = build_vgg16_model((img_size, img_size, 3), num_classes=2)
        model = compile_model(model, learning_rate=learning_rate)
        print(f"模型构建和编译成功")
    except Exception as e:
        print(f"模型构建或编译失败: {e}")
        return None, None, None
    
    print(f"\n=== 模型信息 ===")
    print(f"模型参数总数: {model.count_params():,}")
    print(f"可训练参数: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    # 7. 设置回调函数
    callbacks = setup_callbacks('vgg16_embryo_quality_model')
    
    print(f"\n=== 开始训练 ===")
    
    try:
        # 第一阶段：训练分类头
        print("\n=== 第一阶段：训练分类头 ===")
        history1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=min(20, num_epochs//2),
            callbacks=callbacks,
            verbose=1
        )
        
        # 第二阶段：微调
        if fine_tune and num_epochs > 20:
            print("\n=== 第二阶段：微调 ===")
            
            # 使用改进的VGG16查找方法
            vgg16_base = find_vgg16_base_model(model)
            if vgg16_base is not None:
                vgg16_base.trainable = True
                # 只训练最后4层
                for layer in vgg16_base.layers[:-4]:
                    layer.trainable = False
                
                print(f"找到VGG16基础模型，共{len(vgg16_base.layers)}层，解冻最后4层")
            else:
                print("警告：未找到VGG16基础模型，跳过微调")
                fine_tune = False
            
            if fine_tune:
                # 重新编译
                model = compile_model(model, learning_rate=learning_rate/10)
                print(f"微调阶段可训练参数: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
                
                # 继续训练
                history2 = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=num_epochs,
                    initial_epoch=len(history1.history['loss']),
                    callbacks=callbacks,
                    verbose=1
                )
        
        # 计算训练时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n总训练时间: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        try:
            model.save('vgg16_embryo_interrupted_model')
            print("已保存中断时的模型")
        except Exception as save_error:
            print(f"模型保存失败: {save_error}")
    
    return model, test_images, test_labels

def preprocess_single_image_for_prediction(image_path, img_size=224):
    """
    单张图像预测的预处理（与训练保持一致）
    """
    # 加载图像
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, expand_animations=False)
    
    # 处理通道数
    if len(image.shape) == 2:
        image = tf.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[-1] == 1:
            image = tf.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[:, :, :3]
    
    image.set_shape([None, None, 3])
    
    # 使用与训练相同的预处理
    image = preprocess_image_vgg16(image, img_size=img_size, is_training=False)
    
    return image

def predict_single_image(model_path, image_path, img_size=224):
    """
    修正的单张图像预测函数
    """
    model = tf.keras.models.load_model(model_path)
    
    # 使用与训练一致的预处理
    image = preprocess_single_image_for_prediction(image_path, img_size)
    image = tf.expand_dims(image, 0)  # 添加批次维度
    
    # 预测
    prediction = model.predict(image, verbose=0)
    quality_score = prediction[0, 1]  # 高质量概率
    
    print(f"图像: {image_path}")
    print(f"高质量胚胎概率: {quality_score:.4f}")
    print(f"预测结果: {'高质量' if quality_score > 0.5 else '低质量'}")
    
    return quality_score

def predict_batch_images(model_path, image_paths, img_size=224, batch_size=32):
    """
    修正的批量预测函数
    """
    print(f"加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    def process_path_for_prediction(path):
        return preprocess_single_image_for_prediction(path, img_size)
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(
        process_path_for_prediction,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 预测
    predictions = model.predict(dataset, verbose=1)
    quality_scores = predictions[:, 1]
    
    # 输出结果
    results = []
    for i, (path, score) in enumerate(zip(image_paths, quality_scores)):
        result = {
            'image_path': path,
            'quality_score': float(score),
            'prediction': '高质量' if score > 0.5 else '低质量'
        }
        results.append(result)
        print(f"{i+1:3d}. {os.path.basename(path):30s} | 质量评分: {score:.4f} | {result['prediction']}")
    
    return results

def predict_folder(model_path, folder_path, img_size=224, batch_size=32, save_results=True):
    """
    预测整个文件夹中的图像
    """
    import glob
    
    # 获取所有图像文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')
    image_paths = []
    
    print(f"正在扫描文件夹: {folder_path}")
    
    # 扫描所有支持的图像格式（包括大小写）
    for ext in image_extensions:
        # 小写扩展名
        pattern_lower = os.path.join(folder_path, f"*{ext}")
        image_paths.extend(glob.glob(pattern_lower))
        
        # 大写扩展名
        pattern_upper = os.path.join(folder_path, f"*{ext.upper()}")
        image_paths.extend(glob.glob(pattern_upper))
    
    # 递归搜索子文件夹（可选）
    for ext in image_extensions:
        # 递归搜索小写
        pattern_recursive_lower = os.path.join(folder_path, "**", f"*{ext}")
        image_paths.extend(glob.glob(pattern_recursive_lower, recursive=True))
        
        # 递归搜索大写
        pattern_recursive_upper = os.path.join(folder_path, "**", f"*{ext.upper()}")
        image_paths.extend(glob.glob(pattern_recursive_upper, recursive=True))
    
    # 去重并排序
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"在 {folder_path} 中未找到图像文件")
        print(f"支持的格式: {', '.join(image_extensions)}")
        return []
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 显示前几个文件作为示例
    if len(image_paths) <= 5:
        print("发现的图像文件:")
        for path in image_paths:
            print(f"  - {path}")
    else:
        print("发现的图像文件（前5个）:")
        for path in image_paths[:5]:
            print(f"  - {path}")
        print(f"  ... 还有 {len(image_paths)-5} 个文件")
    
    print(f"\n开始批量预测...")
    
    # 批量预测
    try:
        results = predict_batch_images(model_path, image_paths, img_size, batch_size)
        
        # 保存结果
        if save_results and results:
            # 创建结果DataFrame
            results_df = pd.DataFrame(results)
            
            # 添加更多统计信息
            results_df['filename'] = results_df['image_path'].apply(os.path.basename)
            results_df['directory'] = results_df['image_path'].apply(os.path.dirname)
            
            # 生成带时间戳的文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f'prediction_results_{timestamp}.csv'
            
            # 保存CSV文件
            results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
            print(f"\n预测结果已保存到: {results_file}")
            
            # 生成统计报告
            generate_prediction_report(results, results_file.replace('.csv', '_report.txt'))
            
        return results
        
    except Exception as e:
        print(f"批量预测过程中出现错误: {e}")
        return []

def generate_prediction_report(results, report_file):
    """
    生成详细的预测报告
    """
    if not results:
        return
    
    # 统计结果
    total_images = len(results)
    high_quality_count = sum(1 for r in results if r['prediction'] == '高质量')
    low_quality_count = total_images - high_quality_count
    
    # 计算概率分布
    quality_scores = [r['quality_score'] for r in results]
    avg_score = np.mean(quality_scores)
    min_score = np.min(quality_scores)
    max_score = np.max(quality_scores)
    std_score = np.std(quality_scores)
    
    # 置信度分析
    high_confidence_high = sum(1 for r in results if r['prediction'] == '高质量' and r['quality_score'] > 0.8)
    high_confidence_low = sum(1 for r in results if r['prediction'] == '低质量' and r['quality_score'] < 0.2)
    uncertain = sum(1 for r in results if 0.3 <= r['quality_score'] <= 0.7)
    
    # 生成报告
    report_content = f"""
胚胎质量预测报告
==================

预测概述:
---------
总图像数: {total_images}
高质量胚胎: {high_quality_count} ({high_quality_count/total_images*100:.1f}%)
低质量胚胎: {low_quality_count} ({low_quality_count/total_images*100:.1f}%)

质量评分统计:
-----------
平均评分: {avg_score:.4f}
最高评分: {max_score:.4f}
最低评分: {min_score:.4f}
标准差: {std_score:.4f}

置信度分析:
-----------
高置信度高质量 (>0.8): {high_confidence_high} ({high_confidence_high/total_images*100:.1f}%)
高置信度低质量 (<0.2): {high_confidence_low} ({high_confidence_low/total_images*100:.1f}%)
不确定区间 (0.3-0.7): {uncertain} ({uncertain/total_images*100:.1f}%)

详细结果:
---------
"""
    
    # 添加按评分排序的详细结果
    sorted_results = sorted(results, key=lambda x: x['quality_score'], reverse=True)
    
    report_content += "\n高质量胚胎 (按评分降序):\n"
    for i, result in enumerate([r for r in sorted_results if r['prediction'] == '高质量'], 1):
        filename = os.path.basename(result['image_path'])
        report_content += f"{i:3d}. {filename:40s} | 评分: {result['quality_score']:.4f}\n"
    
    report_content += "\n低质量胚胎 (按评分升序):\n"
    low_quality_sorted = sorted([r for r in sorted_results if r['prediction'] == '低质量'], 
                               key=lambda x: x['quality_score'])
    for i, result in enumerate(low_quality_sorted, 1):
        filename = os.path.basename(result['image_path'])
        report_content += f"{i:3d}. {filename:40s} | 评分: {result['quality_score']:.4f}\n"
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"详细报告已保存到: {report_file}")
    
    # 在控制台显示摘要
    print(f"\n=== 预测摘要 ===")
    print(f"总图像数: {total_images}")
    print(f"高质量: {high_quality_count} ({high_quality_count/total_images*100:.1f}%)")
    print(f"低质量: {low_quality_count} ({low_quality_count/total_images*100:.1f}%)")
    print(f"平均质量评分: {avg_score:.4f}")
    print(f"高置信度预测: {high_confidence_high + high_confidence_low} ({(high_confidence_high + high_confidence_low)/total_images*100:.1f}%)")

def predict_folder_with_visualization(model_path, folder_path, img_size=224, batch_size=32, 
                                    save_results=True, show_samples=True, num_samples=10):
    """
    带可视化的文件夹预测功能
    """
    # 执行预测
    results = predict_folder(model_path, folder_path, img_size, batch_size, save_results)
    
    if not results or not show_samples:
        return results
    
    # 可视化预测结果
    try:
        import matplotlib.pyplot as plt
        
        # 选择样本进行展示
        high_quality_samples = [r for r in results if r['prediction'] == '高质量']
        low_quality_samples = [r for r in results if r['prediction'] == '低质量']
        
        # 每类选择几个样本
        samples_per_class = min(num_samples // 2, 5)
        
        # 按评分排序选择最有代表性的样本
        high_samples = sorted(high_quality_samples, key=lambda x: x['quality_score'], reverse=True)[:samples_per_class]
        low_samples = sorted(low_quality_samples, key=lambda x: x['quality_score'])[:samples_per_class]
        
        all_samples = high_samples + low_samples
        
        if all_samples:
            # 创建可视化图表
            fig, axes = plt.subplots(2, len(all_samples)//2 + len(all_samples)%2, figsize=(15, 8))
            if len(all_samples) == 1:
                axes = [axes]
            elif len(all_samples) <= 2:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            else:
                axes = axes.flatten()
            
            for i, sample in enumerate(all_samples):
                if i >= len(axes):
                    break
                    
                try:
                    # 读取并显示图像
                    img = plt.imread(sample['image_path'])
                    axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                    
                    # 设置标题
                    filename = os.path.basename(sample['image_path'])
                    title = f"{filename[:20]}...\n{sample['prediction']}\n评分: {sample['quality_score']:.3f}"
                    axes[i].set_title(title, fontsize=10)
                    axes[i].axis('off')
                    
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"无法显示\n{filename}", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            
            # 隐藏多余的子图
            for i in range(len(all_samples), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.suptitle('胚胎质量预测结果示例', fontsize=14, y=1.02)
            
            # 保存可视化结果
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_file = f'prediction_samples_{timestamp}.png'
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"预测样本可视化已保存到: {viz_file}")
            
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
    
    return results

# 评估函数保持不变，但需要修正数据集创建
def evaluate_model(model_path, X_test, y_test, save_plots=True):
    """评估模型"""
    print(f"\n=== 模型评估 ===")
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"已加载模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return
    
    # 创建测试数据集（图像已经预处理过了）
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # 评估模型
    test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
    print(f"测试结果:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # 预测
    predictions = model.predict(test_dataset)
    y_pred_proba = predictions[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 详细分析（保持原有逻辑）
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['低质量', '高质量']))
    
    return test_results

# 主函数
if __name__ == "__main__":
    # 设置随机种子确保可复现
    import random
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # GPU配置
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU {gpu} 设置为按需增长")
        except RuntimeError as e:
            print(f"GPU设置错误: {e}")
    
    # 参数设置
    data_dir = '/liaozizhuo/selected_images/'
    img_size = 224
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    
    print("=== 胚胎质量预测模型训练（修正版）===")
    
    # 训练模型
    model, X_test, y_test = train_embryo_quality_model(
        data_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        img_size=img_size,
        learning_rate=learning_rate,
        fine_tune=True,
        apply_balance=True
    )
    
    # 评估模型
    model_path = 'vgg16_embryo_quality_model'
    if os.path.exists(model_path):
        evaluate_model(model_path, X_test, y_test)
    else:
        print("未找到保存的最佳模型")
    
    print("\n训练和评估完成！")
    
    # 示例：使用文件夹预测功能
    print("\n=== 文件夹预测示例 ===")
    if os.path.exists(model_path):
        # 让用户输入实际路径
        test_folder = input("请输入要预测的图像文件夹路径: ").strip()
        if test_folder and os.path.exists(test_folder):
            print(f"开始预测文件夹: {test_folder}")
            results = predict_folder_with_visualization(
                model_path, 
                test_folder, 
                img_size=img_size, 
                batch_size=batch_size,
                save_results=True,
                show_samples=True
            )
            print(f"文件夹预测完成，共处理 {len(results)} 张图像")
        else:
            print(f"路径无效或文件夹不存在: {test_folder}")
