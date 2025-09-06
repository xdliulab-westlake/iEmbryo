"""
BiLSTM多模型特征提取器模块
支持单模型或多模型融合的特征提取器
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet152V2, DenseNet201
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as preprocess_efficientnet
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class MultiModelFeatureExtractor:
    """多模型特征提取器，支持单模型或多模型融合"""
    
    def __init__(self, model_name='all'):
        """
        初始化特征提取器
        
        Args:
            model_name (str): 模型选择
                - 'all': 使用所有模型融合特征 (1536维)
                - 'efficientnet': 仅使用EfficientNetV2M (512维)
                - 'resnet': 仅使用ResNet152V2 (512维)
                - 'densenet': 仅使用DenseNet201 (512维)
                - 'vgg16': 仅使用VGG16 (512维)
        """
        self.model_name = model_name
        self.models = {}
        self.feature_dims = {}
        self.selected_models = self._get_selected_models(model_name)
        self._build_models()
        
    def _get_selected_models(self, model_name):
        """获取选择的模型列表"""
        available_models = ['efficientnet', 'resnet', 'densenet', 'vgg16']
        
        if model_name == 'all':
            return available_models
        elif model_name in available_models:
            return [model_name]
        else:
            raise ValueError(f"不支持的模型名称: {model_name}。可选: {available_models + ['all']}")
    
    def _build_models(self):
        """构建选择的模型特征提取器"""
        
        model_builders = {
            'efficientnet': self._build_efficientnet,
            'resnet': self._build_resnet,
            'densenet': self._build_densenet,
            'vgg16': self._build_vgg16
        }
        
        # 只构建选择的模型
        for model_name in self.selected_models:
            model_builders[model_name]()
            
        # 计算总特征维度
        self.total_feature_dim = sum(self.feature_dims.values())
        
        print(f"✅ 特征提取器初始化完成 (模式: {self.model_name})")
        for name in self.selected_models:
            print(f"   - {self._get_model_display_name(name)}: {self.feature_dims[name]}维")
        print(f"   - 总特征维度: {self.total_feature_dim}维")
        
    def _get_model_display_name(self, model_name):
        """获取模型显示名称"""
        display_names = {
            'efficientnet': 'EfficientNetV2M',
            'resnet': 'ResNet152V2', 
            'densenet': 'DenseNet201',
            'vgg16': 'VGG16'
        }
        return display_names.get(model_name, model_name)
        
    def _build_efficientnet(self):
        """构建EfficientNetV2M模型"""
        base_model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        self.models['efficientnet'] = Model(inputs=base_model.input, outputs=output)
        self.models['efficientnet'].trainable = False
        self.feature_dims['efficientnet'] = 1280
        
    def _build_resnet(self):
        """构建ResNet152V2模型"""
        base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        self.models['resnet'] = Model(inputs=base_model.input, outputs=output)
        self.models['resnet'].trainable = False
        self.feature_dims['resnet'] = 2048
        
    def _build_densenet(self):
        """构建DenseNet201模型"""
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        self.models['densenet'] = Model(inputs=base_model.input, outputs=output)
        self.models['densenet'].trainable = False
        self.feature_dims['densenet'] = 1920
        
    def _build_vgg16(self):
        """构建VGG16模型"""
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        self.models['vgg16'] = Model(inputs=base_model.input, outputs=output)
        self.models['vgg16'].trainable = False
        self.feature_dims['vgg16'] = 512
        
    def extract_features(self, images):
        """
        Extract features using selected model(s)
        
        Args:
            images: Batch of images (numpy array or tensor)
            
        Returns:
            numpy.ndarray: Features from selected model(s)
        """
        features_list = []
        
        # 确保图像数据类型正确
        if images.dtype != np.uint8:
            images = tf.cast(images, tf.uint8)
        
        # 预处理函数映射
        preprocess_map = {
            'efficientnet': preprocess_efficientnet,
            'resnet': preprocess_resnet,
            'densenet': preprocess_densenet,
            'vgg16': preprocess_vgg16
        }
        
        # 为选择的模型提取特征
        for model_name in self.selected_models:
            if model_name in self.models:
                # 使用对应的预处理函数
                preprocess_func = preprocess_map[model_name]
                processed_imgs = preprocess_func(images.copy())
                features = self.models[model_name].predict(processed_imgs, verbose=0)
                features_list.append(features)
                
                # 释放内存
                del processed_imgs
        
        # 如果只有一个模型，直接返回；否则拼接
        if len(features_list) == 1:
            return features_list[0]
        else:
            return np.concatenate(features_list, axis=1)
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'selected_models': self.selected_models.copy(),
            'feature_dims': self.feature_dims.copy(),
            'total_feature_dim': self.total_feature_dim,
            'available_models': ['efficientnet', 'resnet', 'densenet', 'vgg16', 'all']
        }
    
    def get_single_model_features(self, images, model_name):
        """
        使用指定单个模型提取特征（调试用）
        
        Args:
            images: 图像批次
            model_name: 模型名称 ('efficientnet', 'resnet', 'densenet', 'vgg16')
            
        Returns:
            numpy.ndarray: 单个模型的特征
        """
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"模型 '{model_name}' 不可用。当前可用模型: {available}")
        
        # 获取对应的预处理函数
        preprocess_map = {
            'efficientnet': preprocess_efficientnet,
            'resnet': preprocess_resnet,
            'densenet': preprocess_densenet,
            'vgg16': preprocess_vgg16
        }
        
        if images.dtype != np.uint8:
            images = tf.cast(images, tf.uint8)
            
        preprocess_func = preprocess_map[model_name]
        processed_imgs = preprocess_func(images.copy())
        features = self.models[model_name].predict(processed_imgs, verbose=0)
        
        return features
        
    def switch_model(self, model_name):
        """
        切换到不同的模型配置（重新初始化）
        
        Args:
            model_name (str): 新的模型名称
        """
        print(f"切换模型配置: {self.model_name} -> {model_name}")
        self.__init__(model_name)
        
    @property
    def feature_dim(self):
        """获取特征维度（兼容性属性）"""
        return self.total_feature_dim
        
    def is_single_model(self):
        """检查是否为单模型模式"""
        return len(self.selected_models) == 1
        
    def get_available_models(self):
        """获取所有可用模型名称"""
        return ['efficientnet', 'resnet', 'densenet', 'vgg16', 'all']
        
    def print_model_summary(self):
        """打印模型配置摘要"""
        print(f"\n=== 特征提取器配置摘要 ===")
        print(f"模式: {self.model_name}")
        print(f"选择的模型: {self.selected_models}")
        print(f"总特征维度: {self.total_feature_dim}")
        print(f"单模型模式: {'是' if self.is_single_model() else '否'}")
        print("="*30)