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
    # 如果EfficientNetV2不可用，使用更轻量的EfficientNetB3
    from tensorflow.keras.applications.efficientnet import EfficientNetB3 as EfficientNetV2M
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

# 交叉验证相关导入
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 多GPU环境变量（在导入tensorflow之前）
# 智能GPU配置（自动检测）
import tensorflow as tf

# 先检测有多少GPU
gpus = tf.config.list_physical_devices('GPU')
num_gpus = len(gpus)

print(f"检测到 {num_gpus} 个GPU: {[gpu.name for gpu in gpus]}")

# 根据GPU数量动态设置环境变量
if num_gpus > 0:
    # 自动设置可见GPU
    gpu_ids = ','.join([str(i) for i in range(num_gpus)])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    print(f"设置CUDA_VISIBLE_DEVICES = {gpu_ids}")
    
    # 动态设置线程数
    os.environ['TF_GPU_THREAD_COUNT'] = str(num_gpus)
    print(f"设置GPU线程数 = {num_gpus}")
else:
    print("未检测到GPU，将使用CPU")

# 通用GPU优化设置
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# 多GPU时的NCCL优化
if num_gpus > 1:
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_TREE_THRESHOLD'] = '0'
    print("启用多GPU NCCL通信优化")

# GPU内存配置
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"已为 {num_gpus} 个GPU启用内存增长")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")

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

    # 为缺失 BS_prime 的 poor 胚胎赋伪分数
    filtered_path = "/liaozizhuo/scripts/filtered_clean_embryos.csv"
    if os.path.exists(filtered_path):
        filtered_df = pd.read_csv(filtered_path)
        poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
        for emb_id in poor_embryos["EmbryoID"].unique():
            if emb_id not in score_dict:
                score_dict[emb_id] = 6.0 + random.uniform(0, 0.5)
    else:
        print(f"Warning: {filtered_path} not found, skip pseudo BS_prime for poor embryos.")

    return score_dict

# 2. 多模型特征提取器
class MultiModelFeatureExtractor:
    def __init__(self):
        self.models = {}
        self.feature_dims = {}
        self._build_models()
        
    def _build_models(self):
        """构建多个预训练模型"""
        print("Building multi-model feature extractors...")
        
        # 1. EfficientNetV2-M
        base_efficientnet = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        efficientnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_efficientnet.output)
        self.models['efficientnet'] = Model(inputs=base_efficientnet.input, outputs=efficientnet_output)
        self.models['efficientnet'].trainable = False
        self.feature_dims['efficientnet'] = 1280
        
        # 2. ResNet152V2
        base_resnet = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_resnet.output)
        self.models['resnet'] = Model(inputs=base_resnet.input, outputs=resnet_output)
        self.models['resnet'].trainable = False
        self.feature_dims['resnet'] = 2048
        
        # 3. DenseNet201
        base_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        densenet_output = tf.keras.layers.GlobalAveragePooling2D()(base_densenet.output)
        self.models['densenet'] = Model(inputs=base_densenet.input, outputs=densenet_output)
        self.models['densenet'].trainable = False
        self.feature_dims['densenet'] = 1920
        
        # 4. VGG16
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
        """提取多模型特征并拼接 - 内存优化版本"""
        features_list = []
        
        # 确保图像是uint8格式
        if images.dtype != np.uint8:
            images = tf.cast(images, tf.uint8)
        
        # 逐个处理模型，减少内存使用
        model_names = ['efficientnet', 'resnet', 'densenet', 'vgg16']
        preprocess_functions = [preprocess_efficientnet, preprocess_resnet, preprocess_densenet, preprocess_vgg16]
        
        for model_name, preprocess_func in zip(model_names, preprocess_functions):
            # 只复制一次图像
            processed_imgs = preprocess_func(images.copy())
            features = self.models[model_name].predict(processed_imgs, verbose=0)
            features_list.append(features)
            
            # 清理内存
            del processed_imgs
        
        # 拼接所有特征
        combined_features = np.concatenate(features_list, axis=1)
        return combined_features

# 3. 构建训练数据（保留这个，因为它是核心功能）
def build_dataset_fast(data_root, score_dict, multi_extractor, valid_embryo_df, training=True, balance=False):
    """高效GPU批量处理版本：一次性处理所有图像"""
    print(f"=== 高效GPU批量构建数据集 (training={training}) ===")

    def map_half_hour(t):
        try:
            t = float(t)
        except Exception:
            return np.nan
        idx = int(round((t - 96.0) / 0.5))
        return idx if 0 <= idx <= 32 else np.nan

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
            embryo_df["t_idx"] = embryo_df["TargetHour"].apply(map_half_hour)
            f_map = {f: i for i, f in enumerate(focal_planes)}
            embryo_df["f_idx"] = embryo_df["FocalPlane"].map(f_map)
            
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
        
        # 应用数据增强（在预处理之前）
        if training:
            img = augmentation_fn(img)
        
        # 保持uint8格式，让模型特定预处理函数处理
        img = tf.cast(img, tf.uint8)  # 改为uint8
        
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
    
    # Step 4: 标签分布与（可选）平衡采样
    print("Step 4: 标签分布与（可选）平衡采样...")
    from sklearn.utils import resample

    label_series = pd.Series(quality_labels)
    label_counts = label_series.value_counts()
    print("=== EmbryoQuality 标签分布 ===")
    print(label_counts)

    # 先堆叠（无论是否平衡）
    X = np.stack(X)

    if training and balance:
        if label_counts.shape[0] < 2:
            raise ValueError(f"无法平衡采样：标签中仅有一个类别：{label_counts.index.tolist()}")
        min_count = label_counts.min()
        idx_good = np.where(label_series == 1)[0]
        idx_poor = np.where(label_series == 0)[0]
        resample_idx_good = resample(idx_good, replace=False, n_samples=min_count, random_state=42)
        resample_idx_poor = resample(idx_poor, replace=False, n_samples=min_count, random_state=42)
        selected_indices = np.concatenate([resample_idx_good, resample_idx_poor])
        X = X[selected_indices]
        y_bs = [y_bs[i] for i in selected_indices]
        quality_labels = [quality_labels[i] for i in selected_indices]
        ages = [ages[i] for i in selected_indices]

    print(f"=== 最终数据集形状: {X.shape} ===")
    return X, np.array(y_bs), quality_labels, ages

# 4. 模型结构
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
        Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.0)),
        
        # 全连接层
        tf.keras.layers.Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        tf.keras.layers.Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(1)  # 回归预测 BS_prime 分数
    ])
    
    # 使用自适应学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# 5. 交叉验证相关函数（保持这些，因为它们是核心功能）
def cross_validate_bilstm_model(X, y_bs, quality_labels, ages, feature_dim, n_folds=5, n_repeats=1):
    """
    对BiLSTM模型进行交叉验证评估
    """
    # 添加缺失的导入
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler
    import gc
    
    print(f"=== 开始 {n_folds}折交叉验证 (重复{n_repeats}次) ===")
    
    # 强制清理GPU内存
    tf.keras.backend.clear_session()
    gc.collect()
    
    # 验证并修正特征维度
    expected_input_dim = 11 * feature_dim
    if X.shape[2] != expected_input_dim:
        print(f"特征维度不匹配，正在修正...")
        print(f"期望: {expected_input_dim}, 实际: {X.shape[2]}")
        # 如果维度不匹配，重新reshape数据
        if X.shape[2] == feature_dim:
            # 数据是 (n_samples, 33, feature_dim)，需要reshape为 (n_samples, 33, 11*feature_dim)
            X = X.reshape(X.shape[0], 33, 11, feature_dim).reshape(X.shape[0], 33, -1)
        elif X.shape[2] == 11 * feature_dim:
            # 维度已经正确
            pass
        else:
            raise ValueError(f"无法自动修正特征维度: {X.shape[2]}")
    
    print(f"修正后数据形状: {X.shape}")
    
    # 使用分层K折交叉验证（基于质量标签）
    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    
    # 存储每折的结果
    fold_results = {
        'train_mae': [], 'val_mae': [], 'test_mae': [],
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'logic_acc': [], 'logic_auc': [], 'logic_loss': [],
        'best_epochs': []
    }
    
    # 存储每折的预测结果（用于集成）
    all_predictions = []
    all_true_labels = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(cv.split(X, quality_labels)):
        print(f"\n--- 第 {fold_idx + 1} 折 ---")
        
        # 强制清理GPU内存
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 进一步拆分训练集和验证集
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y_bs[train_val_idx], y_bs[test_idx]
        quality_train_val, quality_test = [quality_labels[i] for i in train_val_idx], [quality_labels[i] for i in test_idx]
        ages_train_val, ages_test = [ages[i] for i in train_val_idx], [ages[i] for i in test_idx]
        
        # 基于索引对 train_val 做二次切分，保证对齐
        idx_tv = np.arange(len(X_train_val))
        idx_tr, idx_va = train_test_split(
            idx_tv, test_size=0.2, random_state=42, stratify=np.array(quality_train_val)
        )
        X_train, X_val = X_train_val[idx_tr], X_train_val[idx_va]
        y_train, y_val = y_train_val[idx_tr], y_train_val[idx_va]
        quality_train = [quality_train_val[i] for i in idx_tr]
        quality_val   = [quality_train_val[i] for i in idx_va]
        ages_train    = [ages_train_val[i]    for i in idx_tr]
        ages_val      = [ages_train_val[i]    for i in idx_va]
        
        X_train, y_train, quality_train, ages_train = balance_train_subset(
             X_train, y_train, quality_train, ages_train
        )

        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        # 创建和训练模型 - 使用更保守的内存策略
        try:
            # 尝试使用GPU策略
            strategy = tf.distribute.MirroredStrategy()
            print("使用GPU训练")
        except:
            # 如果GPU不可用，使用CPU
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
            print("使用CPU训练")
        
        with strategy.scope():
            model = create_model(feature_dim)
            
            # 简化CV增强
            def simple_cv_augmentation(features, training=True):
                if not training:
                    return features
                if tf.random.uniform([]) > 0.8:
                    noise = tf.random.normal(tf.shape(features), mean=0.0, stddev=0.003)
                    features = features + noise
                return features
            
            # 回调函数
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0)
            ]
            
            # CV数据管道 - 更小的批量大小
            cv_batch_size = 4 * strategy.num_replicas_in_sync  # 进一步减小
            
            # 简化的训练类
            class SimpleCVModel(tf.keras.Model):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                    
                def call(self, inputs, training=None):
                    if training:
                        inputs = simple_cv_augmentation(inputs, training=True)
                    return self.base_model(inputs, training=training)
            
            cv_model = SimpleCVModel(model)
            
            # 必须编译包装模型
            cv_model.compile(
                optimizer=model.optimizer,    # 使用原模型的优化器
                loss=model.loss,             # 使用原模型的损失函数
                metrics=model.metrics         # 使用原模型的评估指标
            )
            
            # 数据管道
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_ds = train_ds.shuffle(512).batch(cv_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_ds = val_ds.batch(cv_batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
            
            # 训练模型
            history = cv_model.fit(
                train_ds, validation_data=val_ds, 
                epochs=10, callbacks=callbacks, verbose=0
            )
        
        # 评估模型 - 使用更小的批量
        def evaluate_in_batches(X, y, batch_size=8):  # 增大批量大小
            losses = []
            maes = []
            for i in range(0, len(X), batch_size):
                batch_end = min(i + batch_size, len(X))
                batch_X = X[i:batch_end]
                batch_y = y[i:batch_end]
                batch_loss, batch_mae = model.evaluate(batch_X, batch_y, verbose=0)
                losses.append(batch_loss)
                maes.append(batch_mae)
            return np.mean(losses), np.mean(maes)
        
        train_loss, train_mae = evaluate_in_batches(X_train, y_train)
        val_loss, val_mae = evaluate_in_batches(X_val, y_val)
        test_loss, test_mae = evaluate_in_batches(X_test, y_test)
        
        # 1) 用 BiLSTM 计算 train_val 的 BS 预测
        y_bs_pred_trainval = []
        for i in range(0, len(X_train_val), 8):
            j = min(i + 8, len(X_train_val))
            y_bs_pred_trainval.extend(model.predict(X_train_val[i:j], verbose=0).flatten())
        y_bs_pred_trainval = np.array(y_bs_pred_trainval)

        ages_trainval_array = np.array(ages_train_val)
        features_trainval = np.column_stack([
            ages_trainval_array, y_bs_pred_trainval,
            ages_trainval_array ** 2, y_bs_pred_trainval ** 2,
            ages_trainval_array * y_bs_pred_trainval, np.log(ages_trainval_array + 1),
            1.0 / (y_bs_pred_trainval + 1e-6)
        ])

        valid_trainval = np.isfinite(features_trainval).all(axis=1)
        features_trainval = features_trainval[valid_trainval]
        quality_train_val_filtered = [quality_train_val[i] for i in range(len(quality_train_val)) if valid_trainval[i]]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(features_trainval)

        clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        clf.fit(scaler.transform(features_trainval), quality_train_val_filtered)

        # 2) 在 test 上评估
        y_bs_pred = []
        for i in range(0, len(X_test), 8):
            j = min(i + 8, len(X_test))
            y_bs_pred.extend(model.predict(X_test[i:j], verbose=0).flatten())
        y_bs_pred = np.array(y_bs_pred)

        ages_test_array = np.array(ages_test)
        features_test = np.column_stack([
            ages_test_array, y_bs_pred,
            ages_test_array ** 2, y_bs_pred ** 2,
            ages_test_array * y_bs_pred, np.log(ages_test_array + 1),
            1.0 / (y_bs_pred + 1e-6)
        ])
        valid_test = np.isfinite(features_test).all(axis=1)
        features_test = features_test[valid_test]
        quality_test_filtered = [quality_test[i] for i in range(len(quality_test)) if valid_test[i]]

        X_test_scaled = scaler.transform(features_test)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        logic_acc = accuracy_score(quality_test_filtered, (y_pred_proba > 0.5).astype(int))
        logic_auc = roc_auc_score(quality_test_filtered, y_pred_proba)
        logic_loss = log_loss(quality_test_filtered, y_pred_proba)
        
        # 记录结果
        fold_results['train_mae'].append(train_mae)
        fold_results['val_mae'].append(val_mae)
        fold_results['test_mae'].append(test_mae)
        fold_results['train_loss'].append(train_loss)
        fold_results['val_loss'].append(val_loss)
        fold_results['test_loss'].append(test_loss)
        fold_results['logic_acc'].append(logic_acc)
        fold_results['logic_auc'].append(logic_auc)
        fold_results['logic_loss'].append(logic_loss)
        fold_results['best_epochs'].append(np.argmin(history.history['val_loss']) + 1)
        
        # 存储预测结果
        all_predictions.append(np.array(y_pred_proba))
        all_true_labels.append(np.array(quality_test_filtered))
        
        print(f"测试MAE: {test_mae:.4f}, 逻辑回归AUC: {logic_auc:.4f}")
        
        # 清理内存
        del model, train_ds, val_ds
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 强制等待一小段时间让GPU释放内存
        import time
        time.sleep(1)  # 减少等待时间
    
    # 计算统计结果
    cv_summary = {}
    for metric in fold_results.keys():
        values = fold_results[metric]
        cv_summary[f'{metric}_mean'] = np.mean(values)
        cv_summary[f'{metric}_std'] = np.std(values)
        cv_summary[f'{metric}_min'] = np.min(values)
        cv_summary[f'{metric}_max'] = np.max(values)
    
    print(f"\n=== 交叉验证结果总结 ===")
    print(f"BiLSTM测试MAE: {cv_summary['test_mae_mean']:.4f} ± {cv_summary['test_mae_std']:.4f}")
    print(f"逻辑回归AUC: {cv_summary['logic_auc_mean']:.4f} ± {cv_summary['logic_auc_std']:.4f}")
    print(f"逻辑回归准确率: {cv_summary['logic_acc_mean']:.4f} ± {cv_summary['logic_acc_std']:.4f}")
    
    return cv_summary, fold_results, all_predictions, all_true_labels

def cv_concat_metrics(all_pred_proba, all_labels):
    y = np.concatenate(all_labels, axis=0)
    p = np.concatenate(all_pred_proba, axis=0)
    return {
        "auc_overall": roc_auc_score(y, p),
        "acc_overall": accuracy_score(y, (p > 0.5).astype(int))
    }

def ensemble_predictions(all_predictions, all_true_labels, method='mean'):
    """
    集成多折交叉验证的预测结果
    
    Args:
        all_predictions: 每折的预测结果列表
        all_true_labels: 每折的真实标签列表
        method: 集成方法 ('mean', 'median', 'weighted')
    
    Returns:
        ensemble_pred: 集成预测结果
        ensemble_true: 对应的真实标签
    """
    # 检查所有预测结果的长度是否一致
    lengths = [len(pred) for pred in all_predictions]
    if len(set(lengths)) > 1:
        print(f"警告：不同折的预测结果长度不一致: {lengths}")
        print("无法进行集成，返回最后一折的结果")
        return all_predictions[-1], all_true_labels[-1]
    
    # 如果长度一致，进行集成
    if method == 'mean':
        ensemble_pred = np.mean(all_predictions, axis=0)
    elif method == 'median':
        ensemble_pred = np.median(all_predictions, axis=0)
    elif method == 'weighted':
        # 基于每折性能的加权平均
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 可以基于性能调整权重
        ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # 返回对应的真实标签（使用第一折的标签，因为所有折的标签应该是对应的）
    ensemble_true = all_true_labels[0]
    
    return ensemble_pred, ensemble_true

def plot_cross_validation_results(cv_summary, fold_results, save_path="/liaozizhuo/scripts/cv_analysis.pdf"):
    """可视化交叉验证结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. MAE分布
    axes[0, 0].boxplot([fold_results['train_mae'], fold_results['val_mae'], fold_results['test_mae']], 
                       labels=['Train', 'Validation', 'Test'])
    axes[0, 0].set_title('MAE Distribution')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].grid(True)
    
    # 2. Loss分布
    axes[0, 1].boxplot([fold_results['train_loss'], fold_results['val_loss'], fold_results['test_loss']], 
                       labels=['Train', 'Validation', 'Test'])
    axes[0, 1].set_title('Loss Distribution')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # 3. 逻辑回归性能
    axes[0, 2].boxplot([fold_results['logic_acc'], fold_results['logic_auc']], 
                       labels=['Accuracy', 'AUC'])
    axes[0, 2].set_title('Logistic Regression Performance')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].grid(True)
    
    # 4. 性能指标趋势
    fold_numbers = range(1, len(fold_results['test_mae']) + 1)
    axes[1, 0].plot(fold_numbers, fold_results['test_mae'], 'o-', label='Test MAE')
    axes[1, 0].plot(fold_numbers, fold_results['logic_auc'], 's-', label='Logic AUC')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance Trend')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. 最佳epoch分布
    axes[1, 1].hist(fold_results['best_epochs'], bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Best Epoch')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Best Epoch Distribution')
    axes[1, 1].grid(True)
    
    # 6. 性能统计表
    stats_text = f"""
    Test MAE: {cv_summary['test_mae_mean']:.4f} ± {cv_summary['test_mae_std']:.4f}
    Logic AUC: {cv_summary['logic_auc_mean']:.4f} ± {cv_summary['logic_auc_std']:.4f}
    Logic Accuracy: {cv_summary['logic_acc_mean']:.4f} ± {cv_summary['logic_acc_std']:.4f}
    Best Epoch: {cv_summary['best_epochs_mean']:.1f} ± {cv_summary['best_epochs_std']:.1f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 2].set_title('Performance Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"交叉验证分析图表已保存至: {save_path}")

def assess_model_stability(cv_summary):
    """
    评估模型稳定性
    
    Args:
        cv_summary: 交叉验证结果摘要
    
    Returns:
        dict: 稳定性评估结果
    """
    stability_metrics = {}
    
    # 计算变异系数 (CV = std/mean)
    stability_metrics['mae_cv'] = cv_summary['test_mae_std'] / cv_summary['test_mae_mean']
    stability_metrics['auc_cv'] = cv_summary['logic_auc_std'] / cv_summary['logic_auc_mean']
    stability_metrics['acc_cv'] = cv_summary['logic_acc_std'] / cv_summary['logic_acc_mean']
    
    # 稳定性评级
    def get_stability_rating(cv_value):
        if cv_value < 0.1:
            return "优秀"
        elif cv_value < 0.2:
            return "良好"
        elif cv_value < 0.3:
            return "一般"
        else:
            return "不稳定"
    
    stability_metrics['mae_stability'] = get_stability_rating(stability_metrics['mae_cv'])
    stability_metrics['auc_stability'] = get_stability_rating(stability_metrics['auc_cv'])
    stability_metrics['acc_stability'] = get_stability_rating(stability_metrics['acc_cv'])
    
    print(f"\n=== 模型稳定性评估 ===")
    print(f"MAE稳定性: {stability_metrics['mae_stability']} (CV={stability_metrics['mae_cv']:.3f})")
    print(f"AUC稳定性: {stability_metrics['auc_stability']} (CV={stability_metrics['auc_cv']:.3f})")
    print(f"准确率稳定性: {stability_metrics['acc_stability']} (CV={stability_metrics['acc_cv']:.3f})")
    
    return stability_metrics

# 6. 主流程 - 重构版本
def load_and_prepare_data():
    """加载和准备数据"""
    print("=== 数据加载和准备 ===")
    
    root_dir = "/liaozizhuo/image"
    score_csv = "/liaozizhuo/scripts/MainDB.csv"
    valid_embryos_path = "/liaozizhuo/scripts/filtered_clean_embryos.csv"
    
    # 加载评分信息
    score_dict = load_score_dict(score_csv)
    
    # 读取胚胎信息
    valid_embryo_df = pd.read_csv(valid_embryos_path)
    
    # 加载年龄信息
    slide_info_path = "/liaozizhuo/scripts/SlidePatientBirthDates.csv"
    slide_info_df = pd.read_csv(slide_info_path)
    
    # 提取年龄映射
    slide_info_df["EmbryoPatientID"] = slide_info_df["SlideID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
    age_map = dict(zip(slide_info_df["EmbryoPatientID"], slide_info_df["Age"]))
    
    # 将年龄映射到胚胎数据
    valid_embryo_df["PatientID_Prefix"] = valid_embryo_df["PatientID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
    valid_embryo_df["Age"] = valid_embryo_df["PatientID_Prefix"].map(age_map)
    
    print(f"总胚胎数量: {len(valid_embryo_df)}")
    print(f"质量分布:\n{valid_embryo_df['EmbryoQuality'].value_counts()}")
    print(f"年龄范围: {valid_embryo_df['Age'].min():.1f} - {valid_embryo_df['Age'].max():.1f}")
    
    return root_dir, score_dict, valid_embryo_df

def load_or_build_features(root_dir, score_dict, valid_embryo_df):
    """加载或构建特征数据"""
    print("=== 特征数据处理 ===")
    
    # 检查缓存文件
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
        multi_extractor = MultiModelFeatureExtractor()
        X, y_bs, quality_labels, ages = build_dataset_fast(root_dir, score_dict, multi_extractor, valid_embryo_df, training=False)
        
        # 保存缓存
        print("=== 保存特征数据到缓存 ===")
        np.save("/liaozizhuo/scripts/X_features_multi.npy", X)
        np.save("/liaozizhuo/scripts/y_bs_labels_multi.npy", y_bs)
        np.save("/liaozizhuo/scripts/quality_labels_multi.npy", np.array(quality_labels))
        np.save("/liaozizhuo/scripts/ages_multi.npy", np.array(ages))
        print("特征数据已保存")
    
    return X, y_bs, quality_labels, ages

def split_dataset(X, y_bs, quality_labels, ages):
    """拆分数据集"""
    print("=== 数据集拆分 ===")
    
    # 拆分为训练集、验证集、测试集（70%, 15%, 15%）
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X, y_bs, range(len(X)), test_size=0.15, random_state=42, stratify=quality_labels
    )
    
    # 获取对应的质量标签和年龄
    quality_trainval = [quality_labels[i] for i in idx_trainval]
    quality_test = [quality_labels[i] for i in idx_test]
    ages_trainval = [ages[i] for i in idx_trainval]
    ages_test = [ages[i] for i in idx_test]
    
    # 再次拆分训练集和验证集
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_trainval, y_trainval, range(len(X_trainval)), test_size=0.1765, random_state=42, stratify=quality_trainval
    )
    
    # 获取对应的质量标签和年龄
    quality_train = [quality_trainval[i] for i in idx_train]
    quality_val = [quality_trainval[i] for i in idx_val]
    ages_train = [ages_trainval[i] for i in idx_train]
    ages_val = [ages_trainval[i] for i in idx_val]
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            quality_train, quality_val, quality_test,
            ages_train, ages_val, ages_test)

def balance_train_subset(X, y_bs, quality_labels, ages):
    from sklearn.utils import resample
    y_series = pd.Series(quality_labels)
    if y_series.nunique() < 2:
        return X, y_bs, quality_labels, ages
    idx_good = np.where(y_series == 1)[0]
    idx_poor = np.where(y_series == 0)[0]
    min_count = min(len(idx_good), len(idx_poor))
    sel_good = resample(idx_good, replace=False, n_samples=min_count, random_state=42)
    sel_poor = resample(idx_poor, replace=False, n_samples=min_count, random_state=42)
    sel = np.concatenate([sel_good, sel_poor])
    return X[sel], np.array(y_bs)[sel], [quality_labels[i] for i in sel], [ages[i] for i in sel]

def train_main_model(X_train, y_train, X_val, y_val, feature_dim, save_model_path):
    print("=== 训练主模型（智能GPU策略）===")
    
    # 重新检测GPU（防止配置变化）
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    print(f"当前可用GPU数量: {num_gpus}")
    
    # 智能选择分布式策略
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"使用MirroredStrategy，GPU数量: {strategy.num_replicas_in_sync}")
    elif num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        print("使用单GPU策略")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        print("使用CPU策略")
    
    # 动态调整批量大小
    feature_size = X_train.shape[2]
    
    # 根据GPU数量和特征大小智能调整
    if feature_size > 50000:
        base_batch_size = 2 if num_gpus <= 1 else 4
    elif feature_size > 30000:
        base_batch_size = 4 if num_gpus <= 1 else 6
    else:
        base_batch_size = 8 if num_gpus <= 1 else 12
    
    global_batch_size = base_batch_size * max(1, strategy.num_replicas_in_sync)
    
    print(f"智能批量大小配置:")
    print(f"  - 特征维度: {feature_size}")
    print(f"  - 基础批量: {base_batch_size}")
    print(f"  - 全局批量: {global_batch_size}")
    print(f"  - GPU副本数: {strategy.num_replicas_in_sync}")
    
    # 清理GPU内存
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    
    # 配置GPU内存增长
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    
    print(f"检测到 {len(gpus)} 个GPU")
    
    # 多GPU策略配置
    if len(gpus) > 1:
        # 明确指定GPU设备
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        print(f"使用多GPU训练，GPU数量: {strategy.num_replicas_in_sync}")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0" if gpus else "/cpu:0")
        print("使用单GPU/CPU训练")

    # 合理的批量大小设置
    feature_size = X_train.shape[2]
    print(f"特征维度: {feature_size}")
    
    # 更合理的批量大小计算
    if feature_size > 50000:
        base_batch_size = 4  # 适中的批量
    elif feature_size > 30000:
        base_batch_size = 8
    else:
        base_batch_size = 16
    
    global_batch_size = base_batch_size * strategy.num_replicas_in_sync
    print(f"全局批量大小: {global_batch_size} (基础: {base_batch_size} × GPU数: {strategy.num_replicas_in_sync})")
    
    # 轻量级增强函数
    def lightweight_augmentation(features, training=True):
        if not training:
            return features
        
        # 使用固定种子确保各GPU一致性
        if tf.random.uniform([]) > 0.7:
            noise = tf.random.normal(tf.shape(features), mean=0.0, stddev=0.002)
            features = features + noise
        
        return features

    # 在CPU上创建数据集（避免GPU内存问题）
    print("=== 在CPU上构建数据管道 ===")
    
    # 数据增强在数据管道中进行，确保一致性
    def augment_batch(x, y):
        x_aug = lightweight_augmentation(x, training=True)
        return x_aug, y
    
    # 训练集 - 使用生成器避免大内存分配
    def train_generator():
        while True:  # 添加无限循环
            # 创建索引并shuffle
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            for i in indices:
                x = X_train[i].copy()
                # 应用轻量增强
                if np.random.random() > 0.7:
                    noise = np.random.normal(0, 0.002, x.shape).astype(np.float32)
                    x = x + noise
                yield x, y_train[i]
    
    train_ds = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(33, 63360), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    train_ds = train_ds.batch(global_batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # 验证集 - 使用生成器
    def val_generator():
        for i in range(len(X_val)):
            yield X_val[i], y_val[i]
    
    val_ds = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(33, 63360), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    val_ds = val_ds.batch(global_batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # 在strategy作用域内构建
    with strategy.scope():
        # ✅ 只在GPU上创建模型
        model = create_model(feature_dim)
        print("模型参数统计:")
        model.summary()

        # 构建回调函数
        print("=== 构建回调函数 ===")
        
        # 简化：直接使用Keras的fit方法，自动处理分布式训练
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(save_model_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

    print("=== 开始多GPU训练（使用model.fit）===")
    
    # 使用Keras的fit方法，自动处理分布式训练
    history = model.fit(
        train_ds,
        steps_per_epoch=len(X_train) // global_batch_size,  # 指定步数
        validation_data=val_ds,
        # 移除validation_steps，让验证集自然结束
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n=== 训练完成 ===")
    print(f"使用了 {len(gpus)} 个GPU")
    print(f"最佳验证损失: {min(history.history['val_loss']):.4f}")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    print("=== 模型评估 ===")
    
    # 不要在这里清理GPU内存，会删除模型
    
    # 分批评估
    batch_size = 16
    test_losses = []
    test_maes = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_X = X_test[i:batch_end]
        batch_y = y_test[i:batch_end]
        
        batch_loss, batch_mae = model.evaluate(batch_X, batch_y, verbose=0)
        test_losses.append(batch_loss)
        test_maes.append(batch_mae)
    
    test_loss = np.mean(test_losses)
    test_mae = np.mean(test_maes)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试MAE: {test_mae:.4f}")
    
    return test_loss, test_mae

def perform_logistic_regression(model, X_trainval, ages_trainval, quality_trainval, X_test, ages_test, quality_test, test_mae):
    print("=== 逻辑回归分析 ===")

    # 清理GPU状态，避免图执行错误
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

    # 1) 在训练+验证集上获取 BS 预测并拟合 scaler / clf
    print("在训练+验证集上拟合逻辑回归...")
    batch_size = 4  # 进一步减小批量大小，避免GPU错误
    y_bs_pred_tr = []
    valid_indices_tr = []  # 跟踪成功预测的样本索引
    
    for i in range(0, len(X_trainval), batch_size):
        j = min(i + batch_size, len(X_trainval))
        try:
            pred = model.predict(X_trainval[i:j], verbose=0)
            y_bs_pred_tr.extend(pred.flatten())
            valid_indices_tr.extend(range(i, j))  # 记录成功的索引
        except Exception as e:
            print(f"预测批次 {i}-{j} 失败: {e}")
            # 如果批次失败，跳过，不记录索引
            continue
    
    y_bs_pred_tr = np.array(y_bs_pred_tr)
    valid_indices_tr = np.array(valid_indices_tr)
    
    print(f"训练集成功预测样本数: {len(y_bs_pred_tr)}")
    print(f"训练集总样本数: {len(X_trainval)}")
    
    # 根据成功预测的索引获取对应的年龄和质量标签
    ages_tr = np.array([ages_trainval[i] for i in valid_indices_tr])
    quality_tr = np.array([quality_trainval[i] for i in valid_indices_tr])
    
    # 检查形状一致性
    print(f"ages_tr形状: {ages_tr.shape}, y_bs_pred_tr形状: {y_bs_pred_tr.shape}")
    
    # 确保长度一致
    min_len = min(len(ages_tr), len(y_bs_pred_tr), len(quality_tr))
    ages_tr = ages_tr[:min_len]
    y_bs_pred_tr = y_bs_pred_tr[:min_len]
    quality_tr = quality_tr[:min_len]
    
    feats_tr = np.column_stack([
        ages_tr, y_bs_pred_tr, ages_tr**2, y_bs_pred_tr**2,
        ages_tr * y_bs_pred_tr, np.log(ages_tr + 1), 1.0 / (y_bs_pred_tr + 1e-6)
    ])
    
    valid_tr = np.isfinite(feats_tr).all(axis=1)
    feats_tr = feats_tr[valid_tr]
    logic_y_tr = quality_tr[valid_tr]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feats_tr)
    clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    clf.fit(scaler.transform(feats_tr), logic_y_tr)

    # 2) 在测试集上评估
    print("在测试集上评估逻辑回归...")
    y_bs_pred_te = []
    valid_indices_te = []  # 跟踪成功预测的测试样本索引
    
    for i in range(0, len(X_test), batch_size):
        j = min(i + batch_size, len(X_test))
        try:
            pred = model.predict(X_test[i:j], verbose=0)
            y_bs_pred_te.extend(pred.flatten())
            valid_indices_te.extend(range(i, j))
        except Exception as e:
            print(f"测试集预测批次 {i}-{j} 失败: {e}")
            continue
    
    y_bs_pred_te = np.array(y_bs_pred_te)
    valid_indices_te = np.array(valid_indices_te)
    
    print(f"测试集成功预测样本数: {len(y_bs_pred_te)}")
    print(f"测试集总样本数: {len(X_test)}")
    
    # 根据成功预测的索引获取对应的年龄和质量标签
    ages_te = np.array([ages_test[i] for i in valid_indices_te])
    quality_te = np.array([quality_test[i] for i in valid_indices_te])
    
    # 检查形状一致性
    print(f"ages_te形状: {ages_te.shape}, y_bs_pred_te形状: {y_bs_pred_te.shape}")
    
    # 确保长度一致
    min_len = min(len(ages_te), len(y_bs_pred_te), len(quality_te))
    ages_te = ages_te[:min_len]
    y_bs_pred_te = y_bs_pred_te[:min_len]
    quality_te = quality_te[:min_len]
    
    feats_te = np.column_stack([
        ages_te, y_bs_pred_te, ages_te**2, y_bs_pred_te**2,
        ages_te * y_bs_pred_te, np.log(ages_te + 1), 1.0 / (y_bs_pred_te + 1e-6)
    ])
    
    valid_te = np.isfinite(feats_te).all(axis=1)
    feats_te = feats_te[valid_te]
    logic_y = quality_te[valid_te]

    X_te_scaled = scaler.transform(feats_te)
    y_pred_cls = clf.predict(X_te_scaled)
    y_pred_proba = clf.predict_proba(X_te_scaled)[:, 1]

    logic_loss = log_loss(logic_y, y_pred_proba)
    logic_acc = accuracy_score(logic_y, y_pred_cls)
    auc_score = roc_auc_score(logic_y, y_pred_proba)

    print("=== 分类结果 ===")
    print(classification_report(logic_y, y_pred_cls, target_names=['Poor', 'Good']))
    print(f"准确率: {logic_acc:.4f}")
    print(f"对数损失: {logic_loss:.4f}")
    print(f"AUC分数: {auc_score:.4f}")

    feature_names = ['Age', 'BS_pred', 'Age²', 'BS_pred²', 'Age×BS', 'ln(Age)', '1/BS']
    feature_importance = np.abs(clf.coef_[0])

    print("\n=== 特征重要性 ===")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")

    return {
        'logic_acc': logic_acc,
        'auc_score': auc_score,
        'logic_loss': logic_loss,
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'y_pred_proba': y_pred_proba,
        'logic_y': logic_y,
        'ages_array': ages_te[valid_te],
        'y_bs_pred_all': y_bs_pred_te[valid_te]
    }

def create_visualizations(logic_results, test_mae):
    """创建可视化图表"""
    print("=== 创建可视化图表 ===")
    
    # 主分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC曲线
    fpr, tpr, _ = roc_curve(logic_results['logic_y'], logic_results['y_pred_proba'])
    axes[0, 0].plot(fpr, tpr, label=f'ROC曲线 (AUC = {logic_results["auc_score"]:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='随机猜测')
    axes[0, 0].set_xlabel('假正率')
    axes[0, 0].set_ylabel('真正率')
    axes[0, 0].set_title('ROC曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. 预测分布
    axes[0, 1].hist(logic_results['y_pred_proba'][logic_results['logic_y'] == 0], alpha=0.5, label='Poor', bins=20, density=True)
    axes[0, 1].hist(logic_results['y_pred_proba'][logic_results['logic_y'] == 1], alpha=0.5, label='Good', bins=20, density=True)
    axes[0, 1].set_xlabel('预测概率')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('预测概率分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. 年龄vs预测BS分数散点图
    min_length = min(len(logic_results['ages_array']), len(logic_results['y_bs_pred_all']), len(logic_results['logic_y']))
    ages_plot = logic_results['ages_array'][:min_length]
    y_bs_plot = logic_results['y_bs_pred_all'][:min_length]
    logic_y_plot = logic_results['logic_y'][:min_length]
    
    scatter = axes[1, 0].scatter(ages_plot, y_bs_plot, c=logic_y_plot, cmap='RdYlBu', alpha=0.6)
    axes[1, 0].set_xlabel('年龄')
    axes[1, 0].set_ylabel('预测BS分数')
    axes[1, 0].set_title('年龄 vs 预测BS分数')
    plt.colorbar(scatter, ax=axes[1, 0], label='质量 (0=Poor, 1=Good)')
    axes[1, 0].grid(True)

    # 4. 性能指标总结
    performance_text = f"""
    测试MAE: {test_mae:.4f}
    逻辑回归准确率: {logic_results['logic_acc']:.4f}
    逻辑回归AUC: {logic_results['auc_score']:.4f}
    对数损失: {logic_results['logic_loss']:.4f}
    """
    axes[1, 1].text(0.1, 0.5, performance_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('性能指标总结')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("/liaozizhuo/scripts/multi_model_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特征重要性柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.barh(logic_results['feature_names'], logic_results['feature_importance'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    for i, (bar, importance) in enumerate(zip(bars, logic_results['feature_importance'])):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('特征重要性', fontsize=12, fontweight='bold')
    plt.ylabel('特征名称', fontsize=12, fontweight='bold')
    plt.title('逻辑回归特征重要性分析', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/liaozizhuo/scripts/feature_importance.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def run_cross_validation(X, y_bs, quality_labels, ages, feature_dim):
    """运行交叉验证"""
    print("=== 交叉验证评估 ===")
    
    # 执行交叉验证 - 修改为1次重复
    cv_summary, fold_results, all_predictions, all_true_labels = cross_validate_bilstm_model(
        X, y_bs, quality_labels, ages, feature_dim, n_folds=5, n_repeats=1  # 改为1次重复
    )
    
    # 评估模型稳定性
    stability_metrics = assess_model_stability(cv_summary)
    
    # 可视化交叉验证结果
    plot_cross_validation_results(cv_summary, fold_results)
    
    # 集成预测结果 - 修复版本
    # 拼接所有折的预测，计算总体指标
    try:
        concat = cv_concat_metrics(all_predictions, all_true_labels)
        ensemble_acc = concat['acc_overall']
        ensemble_auc = concat['auc_overall']

        print(f"\n=== 交叉验证总体（拼接）结果 ===")
        print(f"总体准确率: {ensemble_acc:.4f}")
        print(f"总体AUC: {ensemble_auc:.4f}")

        import json
        cv_results = {
            'cv_summary': cv_summary,
            'stability_metrics': stability_metrics,
            'ensemble_performance': {
                'accuracy': ensemble_acc,
                'auc': ensemble_auc
            }
        }
    except Exception as e:
        print(f"汇总预测失败: {e}")
        import json
        cv_results = {
            'cv_summary': cv_summary,
            'stability_metrics': stability_metrics,
            'ensemble_performance': {
                'accuracy': None,
                'auc': None
            }
        }
    
    with open("/liaozizhuo/scripts/cv_results.json", 'w') as f:
        json.dump(cv_results, f, indent=2, default=str)
    
    print("交叉验证结果已保存至: /liaozizhuo/scripts/cv_results.json")
    print("分析图表已保存至: /liaozizhuo/scripts/cv_analysis.pdf")

# 主流程
if __name__ == "__main__":
    # 配置参数
    save_model_path = "/liaozizhuo/scripts/bilstm_quality_model.h5"
    
    # 1. 加载和准备数据
    root_dir, score_dict, valid_embryo_df = load_and_prepare_data()
    
    # 2. 加载或构建特征
    X, y_bs, quality_labels, ages = load_or_build_features(root_dir, score_dict, valid_embryo_df)
    
    # 3. 拆分数据集 - 修改这里
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     quality_train, quality_val, quality_test,
     ages_train, ages_val, ages_test) = split_dataset(X, y_bs, quality_labels, ages)
    # 仅对训练集做类别均衡
    X_train, y_train, quality_train, ages_train = balance_train_subset(X_train, y_train, quality_train, ages_train)
    
    # 4. 训练主模型
    model, history = train_main_model(X_train, y_train, X_val, y_val, X.shape[2] // 11, save_model_path)
    
    # 5. 评估模型
    test_loss, test_mae = evaluate_model(model, X_test, y_test)
    
    # 6. 逻辑回归分析 - 修改这里，传递测试集的年龄
    X_trainval_concat = np.concatenate([X_train, X_val], axis=0)
    ages_trainval = ages_train + ages_val
    quality_trainval = quality_train + quality_val
    logic_results = perform_logistic_regression(model, X_trainval_concat, ages_trainval, quality_trainval, X_test, ages_test, quality_test, test_mae)
        
    # 7. 创建可视化
    create_visualizations(logic_results, test_mae)
    
    # 8. 运行交叉验证
    run_cross_validation(X, y_bs, quality_labels, ages, X.shape[2] // 11)
    
    # 9. 最终总结
    print(f"\n=== 最终结果总结 ===")
    print(f"多模型特征维度: {X.shape[2]}")
    print(f"BiLSTM测试MAE: {test_mae:.4f}")
    print(f"逻辑回归准确率: {logic_results['logic_acc']:.4f}")
    print(f"逻辑回归AUC: {logic_results['auc_score']:.4f}")
    print("分析图表已保存至: /liaozizhuo/scripts/multi_model_analysis.pdf")
    print("特征重要性图表已保存至: /liaozizhuo/scripts/feature_importance.pdf")

