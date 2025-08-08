from config import *

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

def load_or_build_features(root_dir, score_dict, valid_embryo_df):
    """加载或构建特征数据"""
    print("=== 特征数据处理 ===")
    
    # 检查缓存文件
    cache_files = list(CACHE_PATHS.values())
    
    if all(os.path.exists(f) for f in cache_files):
        print("=== 加载缓存的特征数据 ===")
        X = np.load(CACHE_PATHS['X_features'])
        y_bs = np.load(CACHE_PATHS['y_bs_labels'])
        quality_labels = np.load(CACHE_PATHS['quality_labels']).tolist()
        ages = np.load(CACHE_PATHS['ages']).tolist()
        print(f"已加载特征维度: {X.shape}")
    else:
        print("=== 构建多模型特征数据集 ===")
        multi_extractor = MultiModelFeatureExtractor()
        X, y_bs, quality_labels, ages = build_dataset_fast(root_dir, score_dict, multi_extractor, valid_embryo_df, training=True)
        
        # 保存缓存
        print("=== 保存特征数据到缓存 ===")
        np.save(CACHE_PATHS['X_features'], X)
        np.save(CACHE_PATHS['y_bs_labels'], y_bs)
        np.save(CACHE_PATHS['quality_labels'], np.array(quality_labels))
        np.save(CACHE_PATHS['ages'], np.array(ages))
        print("特征数据已保存")
    
    return X, y_bs, quality_labels, ages 