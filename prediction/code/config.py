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