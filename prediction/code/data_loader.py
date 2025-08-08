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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import random
import warnings
import gc
import itertools
import json
import time

warnings.filterwarnings('ignore')

# GPU配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 路径配置
DATA_PATHS = {
    'root_dir': "/liaozizhuo/image",
    'score_csv': "/liaozizhuo/scripts/MainDB.csv",
    'valid_embryos_path': "/liaozizhuo/scripts/filtered_clean_embryos.csv",
    'slide_info_path': "/liaozizhuo/scripts/SlidePatientBirthDates.csv",
    'filtered_path': "/liaozizhuo/scripts/filtered_clean_embryos.csv"
}

CACHE_PATHS = {
    'X_features': "/liaozizhuo/scripts/X_features_multi.npy",
    'y_bs_labels': "/liaozizhuo/scripts/y_bs_labels_multi.npy",
    'quality_labels': "/liaozizhuo/scripts/quality_labels_multi.npy",
    'ages': "/liaozizhuo/scripts/ages_multi.npy"
}

OUTPUT_PATHS = {
    'model': "/liaozizhuo/scripts/bilstm_quality_model.h5",
    'grid_search_results': "/liaozizhuo/scripts/grid_search_results.json",
    'cv_results': "/liaozizhuo/scripts/cv_results.json",
    'multi_model_analysis': "/liaozizhuo/scripts/multi_model_analysis.pdf",
    'feature_importance': "/liaozizhuo/scripts/feature_importance.pdf",
    'grid_search_analysis': "/liaozizhuo/scripts/grid_search_analysis.pdf",
    'cv_analysis': "/liaozizhuo/scripts/cv_analysis.pdf"
}

## 2. `data_loader.py` - 数据加载和预处理
```python:data_loader.py
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_score_dict(csv_path):
    """加载评分信息，计算BS_prime"""
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
    if os.path.exists(DATA_PATHS['filtered_path']):
        filtered_df = pd.read_csv(DATA_PATHS['filtered_path'])
        poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
        for emb_id in poor_embryos["EmbryoID"].unique():
            if emb_id not in score_dict:
                score_dict[emb_id] = 6.0 + random.uniform(0, 0.5)
    else:
        print(f"Warning: {DATA_PATHS['filtered_path']} not found, skip pseudo BS_prime for poor embryos.")

    return score_dict

def load_and_prepare_data():
    """加载和准备数据"""
    print("=== 数据加载和准备 ===")
    
    # 加载评分信息
    score_dict = load_score_dict(DATA_PATHS['score_csv'])
    
    # 读取胚胎信息
    valid_embryo_df = pd.read_csv(DATA_PATHS['valid_embryos_path'])
    
    # 加载年龄信息
    slide_info_df = pd.read_csv(DATA_PATHS['slide_info_path'])
    
    # 提取年龄映射
    slide_info_df["EmbryoPatientID"] = slide_info_df["SlideID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
    age_map = dict(zip(slide_info_df["EmbryoPatientID"], slide_info_df["Age"]))
    
    # 将年龄映射到胚胎数据
    valid_embryo_df["PatientID_Prefix"] = valid_embryo_df["PatientID"].str.extract(r"^(D\d{4}\.\d{2}\.\d{2}_S\d{5}_I\d{4}_P)")
    valid_embryo_df["Age"] = valid_embryo_df["PatientID_Prefix"].map(age_map)
    
    print(f"总胚胎数量: {len(valid_embryo_df)}")
    print(f"质量分布:\n{valid_embryo_df['EmbryoQuality'].value_counts()}")
    print(f"年龄范围: {valid_embryo_df['Age'].min():.1f} - {valid_embryo_df['Age'].max():.1f}")
    
    return DATA_PATHS['root_dir'], score_dict, valid_embryo_df

def split_dataset(X, y_bs, quality_labels):
    """拆分数据集"""
    print("=== 数据集拆分 ===")
    
    # 拆分为训练集、验证集、测试集（70%, 15%, 15%）
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_bs, test_size=0.15, random_state=42, stratify=quality_labels)
    
    # 获取对应的质量标签
    quality_trainval = [quality_labels[i] for i in range(len(quality_labels)) if i < len(X_trainval)]
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1765, random_state=42)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test 