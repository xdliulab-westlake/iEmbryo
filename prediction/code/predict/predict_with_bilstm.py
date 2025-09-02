# predict_with_bilstm.py
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet152V2, DenseNet201
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
try:
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as preprocess_efficientnet
except ImportError:
    from tensorflow.keras.applications.efficientnet import EfficientNetB3 as EfficientNetV2M
    from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import cv2
from tqdm import tqdm
import glob
import re

def set_gpu_memory_growth():
    """配置GPU内存增长"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"已为 {len(gpus)} 个GPU启用内存增长")
        except Exception as e:
            print(f"GPU配置错误: {e}")

class MultiModelFeatureExtractor:
    """多模型特征提取器"""
    def __init__(self):
        self.models = {}
        self.feature_dims = {}
        self._build_models()
        
    def _build_models(self):
        """构建多个预训练模型，降维到512维"""
        print("构建多模型特征提取器...")
        
        # 1. EfficientNetV2-M (降维到512)
        base_efficientnet = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        efficientnet_output = GlobalAveragePooling2D()(base_efficientnet.output)
        efficientnet_output = Dense(512, activation='relu', name='efficientnet_reduction')(efficientnet_output)
        self.models['efficientnet'] = Model(inputs=base_efficientnet.input, outputs=efficientnet_output)
        self.models['efficientnet'].trainable = False
        self.feature_dims['efficientnet'] = 512
        
        # 2. ResNet152V2 (降维到512)
        base_resnet = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_output = GlobalAveragePooling2D()(base_resnet.output)
        resnet_output = Dense(512, activation='relu', name='resnet_reduction')(resnet_output)
        self.models['resnet'] = Model(inputs=base_resnet.input, outputs=resnet_output)
        self.models['resnet'].trainable = False
        self.feature_dims['resnet'] = 512
        
        # 3. DenseNet201 (降维到512)
        base_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        densenet_output = GlobalAveragePooling2D()(base_densenet.output)
        densenet_output = Dense(512, activation='relu', name='densenet_reduction')(densenet_output)
        self.models['densenet'] = Model(inputs=base_densenet.input, outputs=densenet_output)
        self.models['densenet'].trainable = False
        self.feature_dims['densenet'] = 512
        
        # 4. VGG16 (保持512)
        base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        vgg_output = GlobalAveragePooling2D()(base_vgg.output)
        vgg_output = Dense(512, activation='relu', name='vgg16_reduction')(vgg_output)
        self.models['vgg16'] = Model(inputs=base_vgg.input, outputs=vgg_output)
        self.models['vgg16'].trainable = False
        self.feature_dims['vgg16'] = 512
        
        self.total_feature_dim = sum(self.feature_dims.values())
        print(f"总特征维度: {self.total_feature_dim}")
        
    def extract_features(self, images):
        """提取多模型特征并拼接"""
        features_list = []
        
        # 确保图像是uint8格式
        if images.dtype != np.uint8:
            images = tf.cast(images, tf.uint8)
        
        # 逐个处理模型
        model_names = ['efficientnet', 'resnet', 'densenet', 'vgg16']
        preprocess_functions = [preprocess_efficientnet, preprocess_resnet, preprocess_densenet, preprocess_vgg16]
        
        for model_name, preprocess_func in zip(model_names, preprocess_functions):
            processed_imgs = preprocess_func(images.copy())
            features = self.models[model_name].predict(processed_imgs, verbose=0)
            features_list.append(features)
            
        # 拼接所有特征
        combined_features = np.concatenate(features_list, axis=1)
        return combined_features

def load_and_preprocess_image(image_path, img_size=224):
    """加载和预处理单张图像"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        img = cv2.resize(img, (img_size, img_size))
        
        return img
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None

def parse_image_filename(filename):
    """
    解析图像文件名，提取焦平面和时间点信息
    
    支持的文件名格式：
    - F-15_100.0_D2023.08.05_S00886_I4424_P_WELL6_RUN487.jpeg
    - F-15_106.5_D2024.08.23_S00314_I5069_P_WELL8_RUN540.jpeg
    
    返回: (focal_plane, time_point)
    """
    # 移除文件扩展名
    name = os.path.splitext(filename)[0]
    
    # 匹配焦平面和时间点
    pattern = r'^(F-?\d+)_(\d+\.?\d*)_.*'
    match = re.match(pattern, name)
    
    if match:
        focal_plane = match.group(1)
        time_point = float(match.group(2))
        return focal_plane, time_point
    else:
        return None, None

def collect_embryo_images(embryo_dir):
    """
    从胚胎目录收集所有图像路径
    
    Args:
        embryo_dir: 胚胎目录路径
    
    Returns:
        dict: {focal_plane_time_point: image_path}
    """
    image_paths = {}
    
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # 收集所有图像文件
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(embryo_dir, ext)))
        all_images.extend(glob.glob(os.path.join(embryo_dir, ext.upper())))
    
    print(f"在 {embryo_dir} 中找到 {len(all_images)} 张图像")
    
    # 解析每个图像文件名
    for image_path in all_images:
        filename = os.path.basename(image_path)
        focal_plane, time_point = parse_image_filename(filename)
        
        if focal_plane is not None and time_point is not None:
            # 构建键名
            key = f"{focal_plane}_{time_point}"
            image_paths[key] = image_path
        else:
            print(f"警告: 无法解析文件名 {filename}")
    
    return image_paths

def build_embryo_features_from_directory(embryo_data, multi_extractor, img_size=224):
    """
    从胚胎目录构建特征
    
    Args:
        embryo_data: 包含胚胎信息的字典列表
        multi_extractor: 多模型特征提取器
        img_size: 图像尺寸
    
    Returns:
        features: 特征数组 [N, 33, 11*feature_dim]
        valid_embryos: 有效的胚胎索引
    """
    print("=== 从胚胎目录构建特征 ===")
    
    # 定义标准的焦平面和时间点
    focal_planes = ['F-75', 'F-60', 'F-45', 'F-30', 'F-15', 'F0', 'F15', 'F30', 'F45', 'F60', 'F75']
    time_points = [96 + 0.5*i for i in range(33)]  # 96.0, 96.5, 97.0, ..., 112.0
    feature_dim = multi_extractor.total_feature_dim
    
    features = []
    valid_embryos = []
    
    for embryo_idx, embryo_info in enumerate(tqdm(embryo_data, desc="处理胚胎")):
        try:
            print(f"\n处理胚胎 {embryo_info['embryo_id']}...")
            
            # 初始化该胚胎的特征张量
            tensor = np.zeros((33, 11, feature_dim), dtype=np.float32)
            valid_images_count = 0
            
            # 处理每个时间点和焦平面的图像
            for t_idx, time_point in enumerate(time_points):
                for f_idx, focal_plane in enumerate(focal_planes):
                    # 构建键名
                    key = f"{focal_plane}_{time_point}"
                    
                    # 查找对应的图像路径
                    image_path = embryo_info['image_paths'].get(key)
                    
                    if image_path and os.path.exists(image_path):
                        # 加载和预处理图像
                        img = load_and_preprocess_image(image_path, img_size)
                        if img is not None:
                            # 扩展维度用于批处理
                            img_batch = np.expand_dims(img, axis=0)
                            
                            # 提取特征
                            img_features = multi_extractor.extract_features(img_batch)
                            
                            # 存储特征
                            tensor[t_idx, f_idx, :] = img_features[0]
                            valid_images_count += 1
                            
                            print(f"  {key}: ✓")
                        else:
                            print(f"  {key}: ✗ (预处理失败)")
                    else:
                        if image_path:
                            print(f"  {key}: ✗ (文件不存在)")
                        else:
                            print(f"  {key}: ✗ (未找到)")
            
            # 只有当有足够图像时才添加该胚胎
            min_required_images = 50  # 至少需要50张有效图像
            if valid_images_count >= min_required_images:
                features.append(tensor.reshape(33, -1))
                valid_embryos.append(embryo_idx)
                print(f"胚胎 {embryo_info['embryo_id']}: 成功处理 {valid_images_count} 张图像 ✓")
            else:
                print(f"胚胎 {embryo_info['embryo_id']}: 图像不足 ({valid_images_count}/{33*11} 张)，跳过 ✗")
                
        except Exception as e:
            print(f"处理胚胎 {embryo_info['embryo_id']} 时出错: {e}")
            continue
    
    if not features:
        raise ValueError("没有成功处理任何胚胎")
    
    features = np.stack(features)
    print(f"\n成功构建特征，形状: {features.shape}")
    print(f"有效胚胎数量: {len(valid_embryos)}")
    
    return features, valid_embryos

def batched_predict(model, X, batch_size=16):
    """批量预测"""
    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    y = model.predict(ds, verbose=1).flatten()
    return y

def build_lr_features(ages, y_bs_pred):
    """构建逻辑回归特征"""
    ages = np.asarray(ages, dtype=np.float32)
    y_bs_pred = np.asarray(y_bs_pred, dtype=np.float32)
    feats = np.column_stack([
        ages,
        y_bs_pred,
        ages ** 2,
        y_bs_pred ** 2,
        ages * y_bs_pred,
        np.log(ages + 1.0),
        1.0 / (y_bs_pred + 1e-6),
    ])
    return feats

def main():
    parser = argparse.ArgumentParser(description="Predict embryo quality using BiLSTM model from CSV input.")
    parser.add_argument("--input-csv", type=str, required=True,
                       help="Path to CSV file with embryo directory paths and ages (2 columns: path, age).")
    parser.add_argument("--model-path", type=str, default="bilstm_quality_model.h5", 
                       help="Path to trained BiLSTM model (.h5).")
    parser.add_argument("--output-csv", type=str, default="bilstm_predictions.csv", 
                       help="Output CSV path.")
    parser.add_argument("--scaler-path", type=str, required=True, 
                       help="Path to sklearn scaler (joblib) for logistic regression.")
    parser.add_argument("--logreg-path", type=str, required=True, 
                       help="Path to sklearn logistic regression (joblib) for quality classification.")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for prediction.")
    parser.add_argument("--img-size", type=int, default=224, 
                       help="Image size for preprocessing.")
    args = parser.parse_args()

    set_gpu_memory_growth()

    # 加载输入CSV
    print("=== 加载输入CSV ===")
    try:
        input_df = pd.read_csv(args.input_csv)
        print(f"加载了 {len(input_df)} 行数据")
        
        # 检查列数
        if len(input_df.columns) < 2:
            raise ValueError("CSV必须至少包含2列：胚胎目录路径和年龄")
        
        # 重命名列以便使用
        input_df.columns = ['embryo_path', 'age'] + list(input_df.columns[2:])
        
        print("CSV列名:", input_df.columns.tolist())
        print(f"胚胎目录路径列: {input_df['embryo_path'].iloc[0]}")
        print(f"年龄列: {input_df['age'].iloc[0]}")
        
    except Exception as e:
        print(f"加载输入CSV失败: {e}")
        raise
    
    # 验证胚胎目录路径
    print("\n=== 验证胚胎目录路径 ===")
    valid_embryos = []
    
    for idx, row in input_df.iterrows():
        embryo_path = row['embryo_path']
        age = row['age']
        
        if not os.path.exists(embryo_path):
            print(f"警告: 胚胎目录不存在，跳过: {embryo_path}")
            continue
        
        if not os.path.isdir(embryo_path):
            print(f"警告: 路径不是目录，跳过: {embryo_path}")
            continue
        
        # 提取胚胎ID（从路径的最后一部分）
        embryo_id = os.path.basename(embryo_path)
        
        # 收集图像路径
        image_paths = collect_embryo_images(embryo_path)
        
        embryo_info = {
            'embryo_id': embryo_id,
            'age': age,
            'image_paths': image_paths,
            'embryo_path': embryo_path
        }
        
        valid_embryos.append(embryo_info)
        print(f"胚胎 {embryo_id} (年龄: {age}): 找到 {len(image_paths)} 张图像")
    
    if not valid_embryos:
        raise ValueError("没有找到有效的胚胎数据")
    
    print(f"\n总共找到 {len(valid_embryos)} 个有效胚胎")
    
    # 初始化特征提取器
    print("\n=== 初始化特征提取器 ===")
    multi_extractor = MultiModelFeatureExtractor()
    
    # 从胚胎目录构建特征
    try:
        features, valid_indices = build_embryo_features_from_directory(
            valid_embryos, multi_extractor, args.img_size
        )
        print(f"成功构建特征，形状: {features.shape}")
        print(f"有效胚胎数量: {len(valid_indices)}")
    except Exception as e:
        print(f"构建特征失败: {e}")
        raise
    
    # 加载BiLSTM模型
    print("\n=== 加载BiLSTM模型 ===")
    try:
        model = tf.keras.models.load_model(args.model_path)
        print(f"成功加载模型: {args.model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise
    
    # 预测BS分数
    print("\n=== 预测BS分数 ===")
    y_bs_pred = batched_predict(model, features, batch_size=args.batch_size)
    
    # 准备输出数据
    valid_embryo_data = [valid_embryos[i] for i in valid_indices]
    
    data = {
        "EmbryoID": [embryo['embryo_id'] for embryo in valid_embryo_data],
        "Age": [embryo['age'] for embryo in valid_embryo_data],
        "bs_pred": y_bs_pred,
        "ValidImages": [len(embryo['image_paths']) for embryo in valid_embryo_data]
    }
    
    # 执行逻辑回归分类
    print("\n=== 执行逻辑回归分类 ===")
    try:
        import joblib
        
        # 构建特征
        ages_array = np.array([embryo['age'] for embryo in valid_embryo_data])
        feats = build_lr_features(ages_array, y_bs_pred)
        
        # 加载预训练的scaler和分类器
        scaler = joblib.load(args.scaler_path)
        clf = joblib.load(args.logreg_path)
        
        # 预测
        feats_scaled = scaler.transform(feats)
        proba = clf.predict_proba(feats_scaled)[:, 1]  # 好胚胎的概率
        
        data["prob_good"] = proba
        data["prob_poor"] = 1.0 - proba
        data["pred_label"] = (proba >= 0.5).astype(int)  # 1 = Good, 0 = Poor
        data["predicted_quality"] = ["Good" if label == 1 else "Poor" for label in data["pred_label"]]
        
        print("逻辑回归分类完成")
        
    except Exception as e:
        print(f"逻辑回归分类失败: {e}")
        raise
    
    # 保存结果
    df = pd.DataFrame(data)
    df.to_csv(args.output_csv, index=False)
    print(f"\n预测结果已保存到: {args.output_csv}")
    
    # 打印摘要
    print("\n=== 预测结果摘要 ===")
    print(f"总胚胎数: {len(df)}")
    print(f"BS分数范围: {df['bs_pred'].min():.2f} - {df['bs_pred'].max():.2f}")
    print(f"平均BS分数: {df['bs_pred'].mean():.2f}")
    
    # 按预测质量分组统计
    print(f"\n质量分类结果:")
    quality_counts = df['predicted_quality'].value_counts()
    print(f"  预测质量分布:\n{quality_counts}")
    
    # 计算年龄和BS分数的相关性
    age_bs_corr = df['Age'].corr(df['bs_pred'])
    print(f"  年龄与BS分数相关性: {age_bs_corr:.3f}")
    
    # 按年龄分组统计
    print(f"\n按年龄分组统计:")
    age_bins = [0, 30, 35, 40, 100]
    age_labels = ['<30', '30-35', '35-40', '>40']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    for age_group in age_labels:
        group_data = df[df['AgeGroup'] == age_group]
        if len(group_data) > 0:
            print(f"  {age_group}岁 ({len(group_data)} 个):")
            print(f"    平均BS分数: {group_data['bs_pred'].mean():.2f}")
            print(f"    好胚胎比例: {(group_data['predicted_quality'] == 'Good').mean():.1%}")

if __name__ == "__main__":
    main()