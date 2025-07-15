import os
import numpy as np
import cv2
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications import VGG16, ResNet152V2, DenseNet201
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
# EfficientNetV2 需要单独导入
try:
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as preprocess_efficientnet
except ImportError:
    # 如果EfficientNetV2不可用，使用EfficientNetB7作为替代
    from tensorflow.keras.applications.efficientnet import EfficientNetB7 as EfficientNetV2M
    from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, TimeDistributed, Bidirectional, LSTM, Dense, Concatenate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import random

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. 加载评分信息，计算BS_prime
def load_score_dict(csv_path):
    df = pd.read_csv(csv_path)

    # 定义打分映射
    icm_map = {'A': 1, 'B': 2, 'C': 4}
    te_map = {'A': 1, 'B': 3, 'C': 4}

    # 计算 BS_prime = ICM_score + TE_score
    df["ICM_score"] = df["ICM"].map(icm_map)
    df["TE_score"] = df["TE"].map(te_map)
    df["BS_prime"] = df["ICM_score"] + df["TE_score"]

    # 首先构建已有的 EmbryoID: BS_prime
    score_dict = dict(zip(df["EmbryoID"], df["BS_prime"]))

    # === 新增: 为缺失 BS_prime 的 poor 胚胎赋伪分数 ===
    # 读取 filtered_clean_embryos.csv，查找 poor 且不在 score_dict 的 EmbryoID
    filtered_path = "/liaozizhuo/scripts/filtered_clean_embryos.csv"
    if os.path.exists(filtered_path):
        filtered_df = pd.read_csv(filtered_path)
        poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
        for emb_id in poor_embryos["EmbryoID"].unique():
            if emb_id not in score_dict:
                # 伪分数: 6.0 + random.uniform(0, 0.5)
                score_dict[emb_id] = 6.0 + random.uniform(0, 0.5)
    else:
        print(f"Warning: {filtered_path} not found, skip pseudo BS_prime for poor embryos.")

    return score_dict

# 2. 多模型特征提取器
class MultiModelFeatureExtractor:
    def __init__(self):
        self.models = {}
        self.feature_dims = {}
        
        # 构建多个预训练模型
        self._build_models()
        
    def _build_models(self):
        """构建多个预训练模型"""
        print("Building multi-model feature extractors...")
        
        # 1. EfficientNetV2-M (最新且高效)
        base_efficientnet = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        efficientnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_efficientnet.output)
        self.models['efficientnet'] = Model(inputs=base_efficientnet.input, outputs=efficientnet_output)
        self.models['efficientnet'].trainable = False
        self.feature_dims['efficientnet'] = 1280
        
        # 2. ResNet152V2 (深度残差网络)
        base_resnet = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_resnet.output)
        self.models['resnet'] = Model(inputs=base_resnet.input, outputs=resnet_output)
        self.models['resnet'].trainable = False
        self.feature_dims['resnet'] = 2048
        
        # 3. DenseNet201 (特征重用)
        base_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        densenet_output = tf.keras.layers.GlobalAveragePooling2D()(base_densenet.output)
        self.models['densenet'] = Model(inputs=base_densenet.input, outputs=densenet_output)
        self.models['densenet'].trainable = False
        self.feature_dims['densenet'] = 1920
        
        # 4. VGG16 (保留原有的，作为baseline)
        base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        vgg_output = tf.keras.layers.GlobalAveragePooling2D()(base_vgg.output)
        self.models['vgg16'] = Model(inputs=base_vgg.input, outputs=vgg_output)
        self.models['vgg16'].trainable = False
        self.feature_dims['vgg16'] = 512
        
        # 总特征维度
        self.total_feature_dim = sum(self.feature_dims.values())
        print(f"Total feature dimension: {self.total_feature_dim}")
        print(f"Individual dimensions: {self.feature_dims}")
        
    def extract_features(self, images):
        """提取多模型特征并拼接"""
        features_list = []
        
        # EfficientNet 预处理和特征提取
        efficientnet_imgs = preprocess_efficientnet(images.copy())
        efficientnet_features = self.models['efficientnet'].predict(efficientnet_imgs, verbose=0)
        features_list.append(efficientnet_features)
        
        # ResNet 预处理和特征提取
        resnet_imgs = preprocess_resnet(images.copy())
        resnet_features = self.models['resnet'].predict(resnet_imgs, verbose=0)
        features_list.append(resnet_features)
        
        # DenseNet 预处理和特征提取
        densenet_imgs = preprocess_densenet(images.copy())
        densenet_features = self.models['densenet'].predict(densenet_imgs, verbose=0)
        features_list.append(densenet_features)
        
        # VGG16 预处理和特征提取
        vgg_imgs = preprocess_vgg16(images.copy())
        vgg_features = self.models['vgg16'].predict(vgg_imgs, verbose=0)
        features_list.append(vgg_features)
        
        # 拼接所有特征
        combined_features = np.concatenate(features_list, axis=1)
        return combined_features

# 3. 数据增强函数
class MedicalImageAugmentation:
    """专门为医学图像设计的数据增强"""
    
    @staticmethod
    def get_augmentation_pipeline(training=True):
        """获取数据增强管道"""
        if not training:
            # 验证/测试时不使用增强
            return lambda x: x
            
        def augment_image(image):
            """应用数据增强"""
            # 转换为float32
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            
            # 1. 随机水平翻转 (50%概率)
            if tf.random.uniform([]) > 0.5:
                image = tf.image.flip_left_right(image)
            
            # 2. 随机垂直翻转 (30%概率，医学图像需谨慎)
            if tf.random.uniform([]) > 0.7:
                image = tf.image.flip_up_down(image)
            
            # 3. 随机旋转 (-15度到15度)
            angle = tf.random.uniform([], -15, 15) * np.pi / 180
            image = tf.image.rot90(image, k=tf.cast(angle / (np.pi/2), tf.int32))
            
            # 4. 亮度调整 (±0.1)
            image = tf.image.random_brightness(image, max_delta=0.1)
            
            # 5. 对比度调整 (0.9-1.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            
            # 6. 随机高斯噪声 (轻微)
            noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01)
            image = tf.clip_by_value(image + noise, 0.0, 1.0)
            
            # 7. 随机色调调整 (轻微)
            image = tf.image.random_hue(image, max_delta=0.02)
            
            # 8. 随机饱和度调整
            image = tf.image.random_saturation(image, lower=0.95, upper=1.05)
            
            return image * 255.0  # 转回原来的范围
            
        return augment_image

# 4. 批量提取图片特征（升级版）
def extract_features_batch(image_paths, multi_extractor, training=True):
    """使用多模型提取器和数据增强"""
    augmentation = MedicalImageAugmentation.get_augmentation_pipeline(training)
    
    def _load_and_augment_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        
        # 应用数据增强
        if training:
            img = augmentation(img)
        
        return img

    # 构建数据pipeline
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(_load_and_augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    img_ds = img_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    # 转换为numpy数组以便多模型处理
    images = []
    for batch in img_ds:
        images.append(batch.numpy())
    images = np.concatenate(images, axis=0)
    
    # 使用多模型提取特征
    features = multi_extractor.extract_features(images)
    return features

# 5. 对一个胚胎构建特征张量（升级版）
def build_tensor_for_embryo(embryo_id, valid_embryo_df, multi_extractor, training=True):
    # 获取总特征维度
    feature_dim = multi_extractor.total_feature_dim
    tensor = np.zeros((33, 11, feature_dim), dtype=np.float32)
    
    focal_planes = ['F-75', 'F-60', 'F-45', 'F-30', 'F-15', 'F0', 'F15', 'F30', 'F45', 'F60', 'F75']
    time_points = [96 + 0.5*i for i in range(33)]

    embryo_df = valid_embryo_df[valid_embryo_df['EmbryoID'] == embryo_id].copy()
    embryo_df["t_idx"] = embryo_df["TargetHour"].map({t: i for i, t in enumerate(time_points)})
    embryo_df["f_idx"] = embryo_df["FocalPlane"].map({f: i for i, f in enumerate(focal_planes)})

    image_paths = embryo_df["FullPath"].tolist()
    features = extract_features_batch(image_paths, multi_extractor, training)  # shape: (N, feature_dim)

    for i, row in enumerate(embryo_df.itertuples(index=False)):
        if pd.isna(row.t_idx) or pd.isna(row.f_idx):
            continue
        t_idx = int(row.t_idx)
        f_idx = int(row.f_idx)
        tensor[t_idx, f_idx, :] = features[i]

    return tensor.reshape(33, -1)  # (33, 11 * feature_dim)

# 6. 构建训练数据（升级版）
def build_dataset_fast(data_root, score_dict, multi_extractor, valid_embryo_df, training=True):
    """高效GPU批量处理版本：一次性处理所有图像"""
    print(f"=== 高效GPU批量构建数据集 (training={training}) ===")
    
    # Step 1: 收集所有图像路径和元信息
    print("Step 1: 收集图像信息...")
    focal_planes = ['F-75', 'F-60', 'F-45', 'F-30', 'F-15', 'F0', 'F15', 'F30', 'F45', 'F60', 'F75']
    time_points = [96 + 0.5*i for i in range(33)]
    feature_dim = multi_extractor.total_feature_dim
    
    valid_embryos = set(valid_embryo_df["EmbryoID"])
    embryo_ids = [eid for eid in os.listdir(data_root) if eid in valid_embryos and eid in score_dict]
    
    # 收集所有图像信息：[embryo_id, t_idx, f_idx, path]
    image_info_list = []
    embryo_metadata = {}  # 存储每个胚胎的标签信息
    
    print(f"处理 {len(embryo_ids)} 个胚胎的图像信息...")
    for embryo_id in tqdm(embryo_ids, desc="收集图像信息"):
        try:
            embryo_df = valid_embryo_df[valid_embryo_df['EmbryoID'] == embryo_id].copy()
            embryo_df["t_idx"] = embryo_df["TargetHour"].map({t: i for i, t in enumerate(time_points)})
            embryo_df["f_idx"] = embryo_df["FocalPlane"].map({f: i for i, f in enumerate(focal_planes)})
            
            # 收集该胚胎的所有有效图像
            valid_images = []
            for _, row in embryo_df.iterrows():
                if pd.notna(row.t_idx) and pd.notna(row.f_idx):
                    image_info_list.append((embryo_id, int(row.t_idx), int(row.f_idx), row.FullPath))
                    valid_images.append((int(row.t_idx), int(row.f_idx)))
            
            # 存储胚胎元数据
            if len(valid_images) > 0:
                embryo_row = valid_embryo_df[valid_embryo_df["EmbryoID"] == embryo_id].iloc[0]
                embryo_metadata[embryo_id] = {
                    'bs_score': score_dict[embryo_id],
                    'quality': 1 if embryo_row["EmbryoQuality"] == "good" else 0,
                    'age': embryo_row["Age"],
                    'valid_positions': valid_images
                }
        except Exception as e:
            print(f"[SKIP] {embryo_id}: {e}")
    
    total_images = len(image_info_list)
    print(f"总共收集到 {total_images} 张图像")
    
    # Step 2: 高效批量GPU特征提取
    print("Step 2: 批量GPU特征提取...")
    
    def create_augmentation_fn():
        """创建数据增强函数"""
        if not training:
            return lambda x: x
            
        def augment(image):
            if tf.random.uniform([]) > 0.5:
                image = tf.image.flip_left_right(image)
            if tf.random.uniform([]) > 0.7:
                image = tf.image.flip_up_down(image)
            if tf.random.uniform([]) > 0.5:
                image = tf.image.random_brightness(image, max_delta=0.1)
            if tf.random.uniform([]) > 0.5:
                image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            return image
        return augment
    
    augmentation_fn = create_augmentation_fn()
    
    # 创建高效的数据pipeline
    all_paths = [info[3] for info in image_info_list]
    
    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        
        # 应用数据增强
        if training:
            img = augmentation_fn(img)
        
        return img
    
    # 构建高效的tf.data pipeline
    BATCH_SIZE = 128  # 增大批量大小
    dataset = tf.data.Dataset.from_tensor_slices(all_paths)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 批量提取所有特征
    print("开始批量特征提取...")
    all_features = []
    
    for batch_images in tqdm(dataset, desc="GPU批量特征提取"):
        batch_features = multi_extractor.extract_features(batch_images.numpy())
        all_features.append(batch_features)
    
    # 合并所有特征
    all_features = np.concatenate(all_features, axis=0)
    print(f"提取完成！特征维度: {all_features.shape}")
    
    # Step 3: 高效分配特征到胚胎张量
    print("Step 3: 分配特征到胚胎张量...")
    
    X, y_bs, quality_labels, ages = [], [], [], []
    
    for embryo_id, metadata in tqdm(embryo_metadata.items(), desc="构建胚胎张量"):
        # 初始化该胚胎的特征张量
        tensor = np.zeros((33, 11, feature_dim), dtype=np.float32)
        
        # 找到该胚胎的所有特征
        embryo_indices = [i for i, info in enumerate(image_info_list) if info[0] == embryo_id]
        
        # 分配特征到对应位置
        for idx in embryo_indices:
            _, t_idx, f_idx, _ = image_info_list[idx]
            tensor[t_idx, f_idx, :] = all_features[idx]
        
        # 添加到结果列表
        X.append(tensor.reshape(33, -1))
        y_bs.append(metadata['bs_score'])
        quality_labels.append(metadata['quality'])
        ages.append(metadata['age'])
    
    print(f"构建完成！共处理 {len(X)} 个胚胎")
    
    # Step 4: 平衡采样
    print("Step 4: 平衡采样...")
    from sklearn.utils import resample
    
    label_series = pd.Series(quality_labels)
    label_counts = label_series.value_counts()
    print("=== EmbryoQuality 标签分布 ===")
    print(label_counts)

    if label_counts.shape[0] < 2:
        raise ValueError(f"无法平衡采样：标签中仅有一个类别：{label_counts.index.tolist()}")

    min_count = label_counts.min()
    idx_good = np.where(label_series == 1)[0]
    idx_poor = np.where(label_series == 0)[0]
    
    resample_idx_good = resample(idx_good, replace=False, n_samples=min_count, random_state=42)
    resample_idx_poor = resample(idx_poor, replace=False, n_samples=min_count, random_state=42)
    selected_indices = np.concatenate([resample_idx_good, resample_idx_poor])
    
    X = [X[i] for i in selected_indices]
    y_bs = [y_bs[i] for i in selected_indices]
    quality_labels = [quality_labels[i] for i in selected_indices]
    ages = [ages[i] for i in selected_indices]
    
    X = np.stack(X)
    print(f"=== 最终数据集形状: {X.shape} ===")
    
    return X, np.array(y_bs), quality_labels, ages

# 7. 模型结构（升级版）
def create_model(feature_dim):
    """创建适应多模型特征的BiLSTM模型"""
    from tensorflow.keras.regularizers import l2
    
    # 计算输入维度 (11个焦平面 * 特征维度)
    input_dim = 11 * feature_dim
    
    model = Sequential([
        Input(shape=(33, input_dim)),
        
        # 特征降维层
        TimeDistributed(Dense(2048, activation='relu')),
        tf.keras.layers.Dropout(0.2),
        
        # BiLSTM层
        Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
        
        # 全连接层
        tf.keras.layers.Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        tf.keras.layers.Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(1)  # 回归预测 BS_prime 分数
    ])
    
    # 使用自适应学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# 7. 主流程
if __name__ == "__main__":
    import os

    root_dir = "/liaozizhuo/image"
    score_csv = "/liaozizhuo/scripts/MainDB.csv"
    valid_embryos_path = "/liaozizhuo/scripts/filtered_clean_embryos.csv"
    save_model_path = "/liaozizhuo/scripts/bilstm_quality_model.h5"

    print("=== 初始化多模型特征提取器 ===")
    score_dict = load_score_dict(score_csv)
    multi_extractor = MultiModelFeatureExtractor()  # 使用多模型特征提取器

    # 读取 valid_embryos_path 以便后续提取标签
    valid_embryo_df = pd.read_csv(valid_embryos_path)

    # 加载 SlideID 与年龄的映射
    slide_info_path = "/liaozizhuo/scripts/SlidePatientBirthDates.csv"
    slide_info_df = pd.read_csv(slide_info_path)

    # 提取 PatientID 与 Age 映射（PatientID 是去掉下划线后缀的）
    slide_info_df["EmbryoPatientID"] = slide_info_df["SlideID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
    age_map = dict(zip(slide_info_df["EmbryoPatientID"], slide_info_df["Age"]))

    # 将 age 映射到 valid_embryo_df
    valid_embryo_df["PatientID_Prefix"] = valid_embryo_df["PatientID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
    valid_embryo_df["Age"] = valid_embryo_df["PatientID_Prefix"].map(age_map)

    print("=== 数据概览 ===")
    print(f"总胚胎数量: {len(valid_embryo_df)}")
    print(f"质量分布:\n{valid_embryo_df['EmbryoQuality'].value_counts()}")
    print(f"年龄范围: {valid_embryo_df['Age'].min():.1f} - {valid_embryo_df['Age'].max():.1f}")

    # 检查是否有缓存的特征文件
    cache_files = [
        "/liaozizhuo/scripts/X_features_multi.npy",
        "/liaozizhuo/scripts/y_bs_labels_multi.npy", 
        "/liaozizhuo/scripts/quality_labels_multi.npy",
        "/liaozizhuo/scripts/ages_multi.npy"
    ]
    
    if all(os.path.exists(f) for f in cache_files):
        print("=== 加载缓存的特征数据 ===")
        X = np.load(cache_files[0])
        y_bs = np.load(cache_files[1])
        quality_labels = np.load(cache_files[2]).tolist()
        ages = np.load(cache_files[3]).tolist()
        print(f"已加载特征维度: {X.shape}")
    else:
        print("=== 构建多模型特征数据集 ===")
        X, y_bs, quality_labels, ages = build_dataset_fast(root_dir, score_dict, multi_extractor, valid_embryo_df, training=True)

    # === 保存多模型特征和标签 ===
    if not all(os.path.exists(f) for f in cache_files):
        print("=== 保存特征数据到缓存 ===")
        np.save("/liaozizhuo/scripts/X_features_multi.npy", X)
        np.save("/liaozizhuo/scripts/y_bs_labels_multi.npy", y_bs)
        np.save("/liaozizhuo/scripts/quality_labels_multi.npy", np.array(quality_labels))
        np.save("/liaozizhuo/scripts/ages_multi.npy", np.array(ages))
        print("特征数据已保存")

    # === 数据集拆分：训练集、验证集、测试集 ===
    print("=== 数据集拆分 ===")
    # 拆分为训练集、验证集、测试集（70%, 15%, 15%）
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_bs, test_size=0.15, random_state=42, stratify=quality_labels)
    
    # 获取对应的质量标签和年龄
    quality_trainval = [quality_labels[i] for i in range(len(quality_labels)) if i < len(X_trainval)]
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1765, random_state=42)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")

    # === 模型创建和训练 ===
    print("=== 创建多模型BiLSTM ===")
    feature_dim = multi_extractor.total_feature_dim
    print(f"使用特征维度: {feature_dim}")
    
    # 多卡支持
    strategy = tf.distribute.MirroredStrategy()
    print(f"可用GPU数量: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = create_model(feature_dim)
        print("模型结构:")
        model.summary()

    print("=== 开始训练 ===")
    
    # 回调函数
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1),
        ModelCheckpoint(save_model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # 使用 tf.data 加速数据加载
    batch_size = 32  # 降低batch size以适应更大的特征维度
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 训练模型
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=200,  # 减少epochs，依赖early stopping
        callbacks=callbacks,
        verbose=1
    )

    # 打印训练历史
    print("=== 训练完成 ===")
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    best_val_mae = history.history['val_mae'][best_epoch-1]
    
    print(f"最佳epoch: {best_epoch}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳验证MAE: {best_val_mae:.4f}")

    # === 在测试集上评估模型 ===
    print("=== 测试集评估 ===")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试MAE: {test_mae:.4f}")

    # 保存模型
    if os.path.exists(save_model_path):
        print(f"模型已保存至: {save_model_path}")
    else:
        print("模型保存失败!")

    # ===== 多模型BiLSTM预测 + 增强逻辑回归分类 =====
    print("=== 逻辑回归分类 ===")
    
    # 使用 BiLSTM 模型预测 BS 分数
    print("预测BS分数...")
    y_bs_pred_all = model.predict(X, verbose=0).flatten()

    # 构建增强特征：[年龄, 预测的BS_prime, 年龄^2, BS_prime^2, 年龄*BS_prime]
    ages_array = np.array(ages)
    enhanced_features = np.column_stack([
        ages_array,                           # 年龄
        y_bs_pred_all,                       # 预测的BS分数  
        ages_array ** 2,                     # 年龄平方
        y_bs_pred_all ** 2,                  # BS分数平方
        ages_array * y_bs_pred_all,          # 交互项
        np.log(ages_array + 1),              # 对数年龄
        1 / (y_bs_pred_all + 1e-6)          # 倒数BS分数
    ])
    
    logic_y = np.array(quality_labels)

    # 检查并移除含 NaN 或 inf 的样本
    valid_idx = np.isfinite(enhanced_features).all(axis=1) & np.isfinite(logic_y)
    enhanced_features = enhanced_features[valid_idx]
    logic_y = logic_y[valid_idx]
    
    print(f"有效样本数: {len(logic_y)}")
    print(f"特征维度: {enhanced_features.shape[1]}")

    # 特征标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    enhanced_features_scaled = scaler.fit_transform(enhanced_features)

    # 训练逻辑回归模型（使用正则化）
    clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    clf.fit(enhanced_features_scaled, logic_y)

    # 预测和评估
    y_pred_cls = clf.predict(enhanced_features_scaled)
    y_pred_proba = clf.predict_proba(enhanced_features_scaled)[:, 1]
    
    # 详细分类报告
    print("=== 分类结果 ===")
    print(classification_report(logic_y, y_pred_cls, target_names=['Poor', 'Good']))

    # 性能指标
    logic_loss = log_loss(logic_y, y_pred_proba)
    logic_acc = accuracy_score(logic_y, y_pred_cls)
    auc_score = roc_auc_score(logic_y, y_pred_proba)

    print(f"准确率: {logic_acc:.4f}")
    print(f"对数损失: {logic_loss:.4f}")
    print(f"AUC分数: {auc_score:.4f}")

    # 特征重要性分析
    feature_names = ['Age', 'BS_pred', 'Age²', 'BS_pred²', 'Age×BS', 'ln(Age)', '1/BS']
    feature_importance = np.abs(clf.coef_[0])
    
    print("\n=== 特征重要性 ===")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")

    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC曲线
    fpr, tpr, _ = roc_curve(logic_y, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_score:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='随机猜测')
    axes[0, 0].set_xlabel('假正率')
    axes[0, 0].set_ylabel('真正率')
    axes[0, 0].set_title('ROC曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. 特征重要性
    axes[0, 1].barh(feature_names, feature_importance)
    axes[0, 1].set_xlabel('重要性')
    axes[0, 1].set_title('特征重要性')
    axes[0, 1].grid(True)

    # 3. 预测分布
    axes[1, 0].hist(y_pred_proba[logic_y == 0], alpha=0.5, label='Poor', bins=20, density=True)
    axes[1, 0].hist(y_pred_proba[logic_y == 1], alpha=0.5, label='Good', bins=20, density=True)
    axes[1, 0].set_xlabel('预测概率')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].set_title('预测概率分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. 年龄vs预测BS分数散点图
    scatter = axes[1, 1].scatter(ages_array, y_bs_pred_all, c=logic_y, cmap='RdYlBu', alpha=0.6)
    axes[1, 1].set_xlabel('年龄')
    axes[1, 1].set_ylabel('预测BS分数')
    axes[1, 1].set_title('年龄 vs 预测BS分数')
    plt.colorbar(scatter, ax=axes[1, 1], label='质量 (0=Poor, 1=Good)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("/liaozizhuo/scripts/multi_model_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("/liaozizhuo/scripts/multi_model_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== 结果总结 ===")
    print(f"多模型特征维度: {multi_extractor.total_feature_dim}")
    print(f"BiLSTM测试MAE: {test_mae:.4f}")
    print(f"逻辑回归准确率: {logic_acc:.4f}")
    print(f"逻辑回归AUC: {auc_score:.4f}")
    print("分析图表已保存至: /liaozizhuo/scripts/multi_model_analysis.pdf")
