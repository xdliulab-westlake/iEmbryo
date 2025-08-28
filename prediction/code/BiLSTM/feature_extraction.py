"""特征提取模块 - 从BiLSTM.py中提取的特征提取相关函数"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet152V2, DenseNet201
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

try:
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as preprocess_efficientnet
except ImportError:
    from tensorflow.keras.applications.efficientnet import EfficientNetB3 as EfficientNetV2M
    from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet

class MultiModelFeatureExtractor:
    """多模型特征提取器"""
    
    def __init__(self, target_dim=512):
        self.target_dim = target_dim
        self.models = {}
        self.feature_dims = {}
        self._build_models()
        
    def _build_models(self):
        """构建多个预训练模型，使用降维减少过拟合"""
        print("Building multi-model feature extractors with dimensionality reduction...")
        
        # 1. EfficientNetV2-M
        base_efficientnet = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        efficientnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_efficientnet.output)
        efficientnet_output = Dense(self.target_dim, activation='relu', name='efficientnet_reduction')(efficientnet_output)
        self.models['efficientnet'] = Model(inputs=base_efficientnet.input, outputs=efficientnet_output)
        self.models['efficientnet'].trainable = False
        self.feature_dims['efficientnet'] = self.target_dim
        
        # 2. ResNet152V2
        base_resnet = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_output = tf.keras.layers.GlobalAveragePooling2D()(base_resnet.output)
        resnet_output = Dense(self.target_dim, activation='relu', name='resnet_reduction')(resnet_output)
        self.models['resnet'] = Model(inputs=base_resnet.input, outputs=resnet_output)
        self.models['resnet'].trainable = False
        self.feature_dims['resnet'] = self.target_dim
        
        # 3. DenseNet201
        base_densenet = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        densenet_output = tf.keras.layers.GlobalAveragePooling2D()(base_densenet.output)
        densenet_output = Dense(self.target_dim, activation='relu', name='densenet_reduction')(densenet_output)
        self.models['densenet'] = Model(inputs=base_densenet.input, outputs=densenet_output)
        self.models['densenet'].trainable = False
        self.feature_dims['densenet'] = self.target_dim
        
        # 4. VGG16
        base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        vgg_output = tf.keras.layers.GlobalAveragePooling2D()(base_vgg.output)
        vgg_output = Dense(self.target_dim, activation='relu', 
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4), 
                          name='vgg16_reduction')(vgg_output)
        self.models['vgg16'] = Model(inputs=base_vgg.input, outputs=vgg_output)
        self.models['vgg16'].trainable = False
        self.feature_dims['vgg16'] = self.target_dim
        
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
