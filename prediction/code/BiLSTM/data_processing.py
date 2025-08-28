"""数据处理模块 - 从BiLSTM.py中提取的数据处理函数"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm
from config import PATH_CONFIG, DATA_CONFIG

def load_score_dict(csv_path):
    """加载评分信息并计算BS_prime"""
    try:
        df = pd.read_csv(csv_path)
        print(f"成功加载评分文件: {csv_path}")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"加载评分文件失败: {e}")
        raise

    # 定义打分映射
    icm_map = {'A': 1, 'B': 2, 'C': 4}
    te_map = {'A': 1, 'B': 3, 'C': 4}

    # 计算 BS_prime = ICM_score + TE_score
    df["ICM_score"] = df["ICM"].map(icm_map)
    df["TE_score"] = df["TE"].map(te_map)
    df["BS_prime"] = df["ICM_score"] + df["TE_score"]

    # 构建评分字典
    score_dict = dict(zip(df["EmbryoID"], df["BS_prime"]))
    print(f"成功构建评分字典，包含 {len(score_dict)} 个胚胎")

    return score_dict

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
    
    # 收集所有图像信息
    image_info_list = []
    embryo_metadata = {}
    
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
    
    # Step 2: 批量特征提取
    print("Step 2: 批量GPU特征提取...")
    all_paths = [info[3] for info in image_info_list]
    
    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.uint8)
        return img
    
    # 构建tf.data pipeline
    BATCH_SIZE = 128
    dataset = tf.data.Dataset.from_tensor_slices(all_paths)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 批量提取特征
    print("开始批量特征提取...")
    all_features = []
    for batch_images in tqdm(dataset, desc="GPU批量特征提取"):
        batch_features = multi_extractor.extract_features(batch_images.numpy())
        all_features.append(batch_features)
    
    all_features = np.concatenate(all_features, axis=0)
    print(f"提取完成！特征维度: {all_features.shape}")
    
    # Step 3: 构建最终数据集
    print("Step 3: 构建最终数据集...")
    X, y_bs, quality_labels, ages = [], [], [], []
    
    for embryo_id, metadata in tqdm(embryo_metadata.items(), desc="构建胚胎张量"):
        tensor = np.zeros((33, 11, feature_dim), dtype=np.float32)
        
        embryo_indices = [i for i, info in enumerate(image_info_list) if info[0] == embryo_id]
        
        for idx in embryo_indices:
            _, t_idx, f_idx, _ = image_info_list[idx]
            tensor[t_idx, f_idx, :] = all_features[idx]
        
        X.append(tensor.reshape(33, -1))
        y_bs.append(metadata['bs_score'])
        quality_labels.append(metadata['quality'])
        ages.append(metadata['age'])
    
    # 数据平衡（可选）
    if training and balance:
        X, y_bs, quality_labels, ages = balance_dataset(X, y_bs, quality_labels, ages)
    
    X = np.stack(X)
    print(f"=== 最终数据集形状: {X.shape} ===")
    
    return X, np.array(y_bs), quality_labels, ages

def balance_dataset(X, y_bs, quality_labels, ages):
    """平衡数据集"""
    from sklearn.utils import resample
    
    label_series = pd.Series(quality_labels)
    label_counts = label_series.value_counts()
    
    if label_counts.shape[0] < 2:
        return X, y_bs, quality_labels, ages
    
    min_count = label_counts.min()
    idx_good = np.where(label_series == 1)[0]
    idx_poor = np.where(label_series == 0)[0]
    
    resample_idx_good = resample(idx_good, replace=False, n_samples=min_count, random_state=42)
    resample_idx_poor = resample(idx_poor, replace=False, n_samples=min_count, random_state=42)
    
    selected_indices = np.concatenate([resample_idx_good, resample_idx_poor])
    
    return X[selected_indices], np.array(y_bs)[selected_indices], \
           [quality_labels[i] for i in selected_indices], \
           [ages[i] for i in selected_indices]
