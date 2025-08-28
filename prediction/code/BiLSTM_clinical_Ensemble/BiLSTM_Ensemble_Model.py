import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet152V2, DenseNet201
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as preprocess_efficientnet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Bidirectional, LSTM, Dense
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, log_loss, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 临床信息特征工程 ====================
class ClinicalFeatureExtractor:
    """临床信息特征提取器 - 替换简单的age特征"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def extract_clinical_features(self, embryo_df, patient_df, treatment_df):
        """提取丰富的临床特征"""
        print("=== 提取临床特征 ===")
        
        # 1. 基础患者信息
        clinical_features = []
        
        # 年龄相关特征
        clinical_features.append(embryo_df['Age'].values)
        clinical_features.append(embryo_df['Age'].values ** 2)  # 年龄平方
        clinical_features.append(np.log(embryo_df['Age'].values + 1))  # 年龄对数
        
        # 2. 胚胎发育特征
        if 'EmbryoStage' in embryo_df.columns:
            stage_encoder = LabelEncoder()
            embryo_stage = stage_encoder.fit_transform(embryo_df['EmbryoStage'].fillna('Unknown'))
            clinical_features.append(embryo_stage)
            self.label_encoders['EmbryoStage'] = stage_encoder
            
        # 3. 治疗相关特征
        if 'TreatmentType' in treatment_df.columns:
            treatment_encoder = LabelEncoder()
            treatment_type = treatment_encoder.fit_transform(treatment_df['TreatmentType'].fillna('Unknown'))
            clinical_features.append(treatment_type)
            self.label_encoders['TreatmentType'] = treatment_encoder
            
        # 4. 时间相关特征
        if 'FertilizationTime' in embryo_df.columns:
            fertilization_time = pd.to_datetime(embryo_df['FertilizationTime'])
            clinical_features.append(fertilization_time.dt.dayofweek.values)  # 星期几
            clinical_features.append(fertilization_time.dt.month.values)      # 月份
            clinical_features.append(fertilization_time.dt.quarter.values)    # 季度
            
        # 5. 历史特征（如果有多次治疗记录）
        if 'PreviousAttempts' in patient_df.columns:
            clinical_features.append(patient_df['PreviousAttempts'].fillna(0).values)
            
        # 6. 实验室指标（如果有）
        lab_features = ['FSH', 'LH', 'AMH', 'E2', 'P4']
        for feature in lab_features:
            if feature in patient_df.columns:
                clinical_features.append(patient_df[feature].fillna(patient_df[feature].median()).values)
                
        # 7. 组合特征
        age = embryo_df['Age'].values
        if 'EmbryoStage' in embryo_df.columns:
            age_stage_interaction = age * embryo_stage
            clinical_features.append(age_stage_interaction)
            
        # 转换为numpy数组
        clinical_array = np.column_stack(clinical_features)
        
        # 标准化
        clinical_array_scaled = self.scaler.fit_transform(clinical_array)
        
        # 记录特征名称
        self.feature_names = [
            'Age', 'Age²', 'ln(Age+1)', 'EmbryoStage', 'TreatmentType',
            'DayOfWeek', 'Month', 'Quarter', 'PreviousAttempts',
            'FSH', 'LH', 'AMH', 'E2', 'P4', 'Age×Stage'
        ][:clinical_array_scaled.shape[1]]
        
        print(f"临床特征维度: {clinical_array_scaled.shape}")
        print(f"特征名称: {self.feature_names}")
        
        return clinical_array_scaled

# ==================== 2. BiLSTM模型（预测BS分数） ====================
class BiLSTMModel:
    """BiLSTM模型 - 预测胚胎质量分数"""
    
    def __init__(self, feature_dim=2048):
        self.feature_dim = feature_dim
        self.model = None
        
    def build_model(self):
        """构建BiLSTM模型"""
        input_dim = 11 * self.feature_dim
        
        model = tf.keras.Sequential([
            Input(shape=(33, input_dim), dtype=tf.float32),
            
            # 特征降维层
            TimeDistributed(Dense(2048, activation='relu', dtype=tf.float32)),
            tf.keras.layers.Dropout(0.2),
            
            # BiLSTM层
            Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, dtype=tf.float32)),
            Bidirectional(LSTM(128, dropout=0.2, dtype=tf.float32)),
            
            # 全连接层
            tf.keras.layers.Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dense(1, dtype=tf.float32)  # 预测BS分数
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        """训练模型"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """预测BS分数"""
        return self.model.predict(X, verbose=0).flatten()

# ==================== 3. 多模型特征提取器 ====================
class MultiModelFeatureExtractor:
    """多模型特征提取器 - 类似Kaggle获胜者的图像模型"""
    
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
        efficientnet_output = Dense(512, activation='relu', name='efficientnet_reduction')(efficientnet_output)
        self.models['efficientnet'] = Model(inputs=base_efficientnet.input, outputs=efficientnet_output)
        self.models['efficientnet'].trainable = False
        self.feature_dims['efficientnet'] = 512
        
        # 2. ResNet152V2
        base_resnet = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_resnet.output)
        resnet_output = Dense(512, activation='relu', name='resnet_reduction')(resnet_output)
        self.models['resnet'] = Model(inputs=base_resnet.input, outputs=resnet_output)
        self.models['resnet'].trainable = False
        self.feature_dims['resnet'] = 512
        
        # 3. DenseNet201
        base_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        densenet_output = tf.keras.layers.GlobalAveragePooling2D()(base_densenet.output)
        densenet_output = Dense(512, activation='relu', name='densenet_reduction')(densenet_output)
        self.models['densenet'] = Model(inputs=base_densenet.input, outputs=densenet_output)
        self.models['densenet'].trainable = False
        self.feature_dims['densenet'] = 512
        
        # 4. VGG16
        base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        vgg_output = tf.keras.layers.GlobalAveragePooling2D()(base_vgg.output)
        vgg_output = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='vgg16_reduction')(vgg_output)
        self.models['vgg16'] = Model(inputs=base_vgg.input, outputs=vgg_output)
        self.models['vgg16'].trainable = False
        self.feature_dims['vgg16'] = 512
        
        self.total_feature_dim = sum(self.feature_dims.values())
        print(f"Total feature dimension: {self.total_feature_dim}")
        
    def extract_features(self, images):
        """提取多模型特征"""
        features_list = []
        
        model_names = ['efficientnet', 'resnet', 'densenet', 'vgg16']
        preprocess_functions = [preprocess_efficientnet, preprocess_resnet, preprocess_densenet, preprocess_vgg16]
        
        for model_name, preprocess_func in zip(model_names, preprocess_functions):
            processed_imgs = preprocess_func(images.copy())
            features = self.models[model_name].predict(processed_imgs, verbose=0)
            features_list.append(features)
            
        combined_features = np.concatenate(features_list, axis=1)
        return combined_features

# ==================== 4. GBDT集成模型 ====================
class GBDTEnsemble:
    """GBDT集成模型 - 类似Kaggle获胜者的策略"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
    def train_models(self, X_train, y_train, X_val, y_val, patient_ids):
        """训练多个GBDT模型"""
        print("=== 训练GBDT集成模型 ===")
        
        # 1. LightGBM (无图像特征)
        print("训练LightGBM (无图像特征)...")
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # 只使用临床特征
        clinical_features = X_train[:, :X_train.shape[1]//2]  # 假设前半部分是临床特征
        
        lgb_train = lgb.Dataset(clinical_features, label=y_train)
        lgb_val = lgb.Dataset(X_val[:, :X_val.shape[1]//2], label=y_val)
        
        self.models['lgb_clinical'] = lgb.train(
            lgb_params, lgb_train, valid_sets=[lgb_val], 
            num_boost_round=1000, early_stopping_rounds=50, verbose_eval=100
        )
        
        # 2. LightGBM (有图像特征)
        print("训练LightGBM (有图像特征)...")
        lgb_train_full = lgb.Dataset(X_train, label=y_train)
        lgb_val_full = lgb.Dataset(X_val, label=y_val)
        
        self.models['lgb_full'] = lgb.train(
            lgb_params, lgb_train_full, valid_sets=[lgb_val_full], 
            num_boost_round=1000, early_stopping_rounds=50, verbose_eval=100
        )
        
        # 3. CatBoost (有图像特征)
        print("训练CatBoost (有图像特征)...")
        cat_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_seed': 42,
            'verbose': 100
        }
        
        self.models['catboost'] = CatBoostClassifier(**cat_params)
        self.models['catboost'].fit(
            X_train, y_train, eval_set=(X_val, y_val), 
            early_stopping_rounds=50, verbose=False
        )
        
        # 4. XGBoost (有图像特征)
        print("训练XGBoost (有图像特征)...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': 42
        }
        
        self.models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        self.models['xgboost'].fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            early_stopping_rounds=50, verbose=False
        )
        
        # 记录特征重要性
        self._record_feature_importance()
        
    def _record_feature_importance(self):
        """记录各模型的特征重要性"""
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            elif hasattr(model, 'feature_importance'):
                self.feature_importance[name] = model.feature_importance
        
    def predict_ensemble(self, X):
        """集成预测"""
        predictions = {}
        
        # 获取各模型预测
        for name, model in self.models.items():
            if name == 'lgb_clinical':
                # 只使用临床特征
                X_clinical = X[:, :X.shape[1]//2]
                predictions[name] = model.predict_proba(X_clinical)[:, 1]
            else:
                predictions[name] = model.predict_proba(X)[:, 1]
        
        # 加权平均（可以基于验证集性能调整权重）
        weights = {
            'lgb_clinical': 0.2,
            'lgb_full': 0.3,
            'catboost': 0.25,
            'xgboost': 0.25
        }
        
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
            
        return ensemble_pred, predictions

# ==================== 5. GroupStratified交叉验证 ====================
def group_stratified_cv(X, y, patient_ids, n_splits=5):
    """GroupStratified交叉验证 - 按患者ID分组"""
    print("=== GroupStratified交叉验证 ===")
    
    # 创建患者级别的标签
    patient_labels = []
    unique_patients = list(set(patient_ids))
    
    for patient_id in unique_patients:
        patient_indices = [i for i, pid in enumerate(patient_ids) if pid == patient_id]
        patient_label = 1 if any(y[i] == 1 for i in patient_indices) else 0
        patient_labels.append(patient_label)
    
    # 使用GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits)
    
    cv_results = {
        'train_scores': [], 'val_scores': [], 'test_scores': [],
        'feature_importance': [], 'predictions': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, patient_labels, groups=unique_patients)):
        print(f"\n--- 第 {fold + 1} 折 ---")
        
        # 获取对应的样本索引
        train_patients = [unique_patients[i] for i in train_idx]
        test_patients = [unique_patients[i] for i in test_idx]
        
        train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
        test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
        
        # 进一步拆分训练集和验证集
        train_val_split = train_test_split(
            train_indices, test_size=0.2, random_state=42, 
            stratify=[y[i] for i in train_indices]
        )
        train_indices, val_indices = train_val_split
        
        # 准备数据
        X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
        patient_train = [patient_ids[i] for i in train_indices]
        patient_val = [patient_ids[i] for i in val_indices]
        patient_test = [patient_ids[i] for i in test_indices]
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        # 训练BiLSTM模型
        bilstm = BiLSTMModel()
        bilstm.build_model()
        bilstm.train(X_train, y_train, X_val, y_val, epochs=10)
        
        # 获取BiLSTM预测的BS分数
        bs_scores_train = bilstm.predict(X_train)
        bs_scores_val = bilstm.predict(X_val)
        bs_scores_test = bilstm.predict(X_test)
        
        # 训练GBDT集成模型
        gbdt_ensemble = GBDTEnsemble()
        gbdt_ensemble.train_models(X_train, y_train, X_val, y_val, patient_train)
        
        # 预测
        ensemble_pred, individual_preds = gbdt_ensemble.predict_ensemble(X_test)
        
        # 评估
        test_auc = roc_auc_score(y_test, ensemble_pred)
        test_acc = accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))
        
        cv_results['test_scores'].append(test_auc)
        cv_results['predictions'].append(ensemble_pred)
        
        print(f"测试集AUC: {test_auc:.4f}, 准确率: {test_acc:.4f}")
        
    # 计算平均性能
    mean_auc = np.mean(cv_results['test_scores'])
    std_auc = np.std(cv_results['test_scores'])
    
    print(f"\n=== 交叉验证结果 ===")
    print(f"平均AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return cv_results

# ==================== 6. 主流程 ====================
def main():
    """主函数"""
    try:
        print("=== BiLSTM + GBDT集成模型 ===")
        print("基于Kaggle获胜者策略的改进版本")
        
        # 1. 加载数据
        print("\n=== 步骤1: 加载数据 ===")
        # 这里需要根据你的实际数据路径进行调整
        data_paths = {
            'embryo_data': '/liaozizhuo/scripts/filtered_clean_embryos.csv',
            'patient_data': '/liaozizhuo/scripts/patient_info.csv',  # 需要创建
            'treatment_data': '/liaozizhuo/scripts/treatment_info.csv',  # 需要创建
            'image_features': '/liaozizhuo/scripts/X_features_multi.npy'
        }
        
        # 2. 特征工程
        print("\n=== 步骤2: 特征工程 ===")
        clinical_extractor = ClinicalFeatureExtractor()
        
        # 3. 训练BiLSTM模型
        print("\n=== 步骤3: 训练BiLSTM模型 ===")
        bilstm = BiLSTMModel()
        bilstm.build_model()
        
        # 4. 训练GBDT集成模型
        print("\n=== 步骤4: 训练GBDT集成模型 ===")
        gbdt_ensemble = GBDTEnsemble()
        
        # 5. 交叉验证
        print("\n=== 步骤5: 交叉验证 ===")
        # cv_results = group_stratified_cv(X, y, patient_ids)
        
        print("\n=== 程序执行完成 ===")
        
    except Exception as e:
        print(f"\n=== 程序执行失败 ===")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
