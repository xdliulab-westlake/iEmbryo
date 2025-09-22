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
    
    def __init__(self, model_name='all', use_multi_gpu=True, batch_size=256, force_cpu=False):
        """
        初始化特征提取器

        Args:
            model_name (str): 模型选择
                - 'all': 使用所有模型融合特征 (5760维: 1280+2048+1920+512)
                - 'efficientnet': 仅使用EfficientNetV2M (1280维)
                - 'resnet': 仅使用ResNet152V2 (2048维)
                - 'densenet': 仅使用DenseNet201 (1920维)
                - 'vgg16': 仅使用VGG16 (512维)
            use_multi_gpu (bool): 是否使用多GPU加速
            batch_size (int): 批次大小，建议根据GPU内存调整
            force_cpu (bool): 强制使用CPU模式（GPU显存不足时）
        """
        self.model_name = model_name
        self.use_multi_gpu = use_multi_gpu
        self.batch_size = batch_size
        self.force_cpu = force_cpu
        self.models = {}
        self.feature_dims = {}

        # 如果强制CPU模式，跳过GPU检测
        if force_cpu:
            print("🔧 强制CPU模式")
            self.gpu_devices = ['/CPU:0']
            self.use_multi_gpu = False
        else:
            self.gpu_devices = self._get_available_gpus()

        self.selected_models = self._get_selected_models(model_name)

        try:
            self._build_models()
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("🔄 GPU显存不足，自动切换到CPU模式...")
                self.gpu_devices = ['/CPU:0']
                self.use_multi_gpu = False
                self.force_cpu = True
                # 清理已创建的模型
                self.models = {}
                self.feature_dims = {}
                # 重新构建
                self._build_models()
            else:
                raise e
        
    def _get_available_gpus(self):
        """获取可用的GPU设备列表 - 安全的GPU配置"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"🔍 检测到 {len(gpus)} 个物理GPU设备")

                # 安全的GPU配置策略
                try:
                    logical_gpus = tf.config.list_logical_devices('GPU')

                    # 如果已经有虚拟设备，直接使用
                    if logical_gpus:
                        print(f"✅ 使用已配置的 {len(logical_gpus)} 个逻辑GPU设备")
                        return [f'/GPU:{i}' for i in range(len(logical_gpus))]

                    # 尝试安全的GPU配置
                    available_devices = []
                    for i, gpu in enumerate(gpus):
                        try:
                            # 限制显存使用，避免OOM
                            tf.config.experimental.set_memory_growth(gpu, True)

                            # 可选：设置显存限制（如果需要）
                            # tf.config.experimental.set_virtual_device_configuration(
                            #     gpu,
                            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
                            # )

                            available_devices.append(f'/GPU:{i}')
                            print(f"✅ GPU {i} 配置成功")

                        except Exception as gpu_error:
                            print(f"⚠️ GPU {i} 配置失败: {gpu_error}")
                            continue

                    if available_devices:
                        print(f"✅ 成功配置 {len(available_devices)} 个GPU设备")
                        return available_devices
                    else:
                        print("⚠️ 所有GPU配置失败，使用CPU")
                        return ['/CPU:0']

                except Exception as config_error:
                    print(f"⚠️ GPU配置过程失败: {config_error}")
                    print("🔄 尝试直接使用第一个GPU...")
                    try:
                        # 最后尝试：只使用第一个GPU
                        return ['/GPU:0']
                    except:
                        print("⚠️ 无法使用GPU，回退到CPU")
                        return ['/CPU:0']
            else:
                print("⚠️ 未检测到GPU设备，使用CPU")
                return ['/CPU:0']

        except Exception as e:
            print(f"❌ GPU检测失败: {e}")
            print("🔄 强制使用CPU模式")
            return ['/CPU:0']

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

        # 简化模型构建，避免显存OOM
        print(f"🔧 构建模型 (可用设备: {len(self.gpu_devices)}个)")

        # 只构建选择的模型 - 让TensorFlow自动分配设备
        for model_name in self.selected_models:
            try:
                print(f"   📍 构建 {self._get_model_display_name(model_name)}")
                model_builders[model_name]()
                print(f"   ✅ {self._get_model_display_name(model_name)} 构建成功")
            except Exception as e:
                print(f"   ❌ {self._get_model_display_name(model_name)} 构建失败: {e}")
                # 如果是显存不足，建议解决方案
                if "out of memory" in str(e).lower():
                    print(f"   💡 建议：释放其他GPU进程或使用CPU模式")
                raise e

        # 计算总特征维度
        self.total_feature_dim = sum(self.feature_dims.values())

        print(f"✅ 特征提取器初始化完成 (模式: {self.model_name})")
        print(f"   🎯 多GPU模式: {'启用' if self.use_multi_gpu else '禁用'}")
        print(f"   📦 批次大小: {self.batch_size}")
        print(f"   🔧 计算设备: {self.gpu_devices}")
        for name in self.selected_models:
            print(f"   - {self._get_model_display_name(name)}: {self.feature_dims[name]}维")
        print(f"   - 总特征维度: {self.total_feature_dim}维")

        # GPU状态检查
        self._check_gpu_availability()
        
    def _get_model_display_name(self, model_name):
        """获取模型显示名称"""
        display_names = {
            'efficientnet': 'EfficientNetV2M',
            'resnet': 'ResNet152V2', 
            'densenet': 'DenseNet201',
            'vgg16': 'VGG16'
        }
        return display_names.get(model_name, model_name)

    def _check_gpu_availability(self):
        """检查GPU可用性和加速状态"""
        try:
            print(f"\n🔍 GPU加速检查:")

            # 检查TensorFlow是否构建了GPU支持
            print(f"   - TensorFlow GPU支持: {tf.test.is_built_with_cuda()}")
            print(f"   - 可用GPU数量: {len(tf.config.list_physical_devices('GPU'))}")

            # 测试GPU计算
            if '/GPU:0' in str(self.gpu_devices):
                with tf.device('/GPU:0'):
                    # 创建测试张量
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    result = tf.matmul(a, b)

                print(f"   ✅ GPU计算测试通过")
                print(f"   🚀 将使用GPU加速推理")
            else:
                print(f"   ⚠️ 将使用CPU计算")

        except Exception as e:
            print(f"   ❌ GPU检查失败: {e}")
            print(f"   🔄 将回退到CPU模式")

        print("=" * 40)
        
    def _build_efficientnet(self):
        """构建EfficientNetV2M模型"""
        base_model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        self.models['efficientnet'] = Model(inputs=base_model.input, outputs=output)
        # 🔥 启用微调：冻结前1/3的层，微调后2/3的层
        self.models['efficientnet'].trainable = True
        total_layers = len(self.models['efficientnet'].layers)
        freeze_layers = int(total_layers / 3)
        for layer in self.models['efficientnet'].layers[:freeze_layers]:
            layer.trainable = False
        self.feature_dims['efficientnet'] = 1280
        
    def _build_resnet(self):
        """构建ResNet152V2模型"""
        base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        self.models['resnet'] = Model(inputs=base_model.input, outputs=output)
        # 🔥 启用微调：冻结前1/3的层，微调后2/3的层
        self.models['resnet'].trainable = True
        total_layers = len(self.models['resnet'].layers)
        freeze_layers = int(total_layers / 3)
        for layer in self.models['resnet'].layers[:freeze_layers]:
            layer.trainable = False
        self.feature_dims['resnet'] = 2048
        
    def _build_densenet(self):
        """构建DenseNet201模型"""
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        self.models['densenet'] = Model(inputs=base_model.input, outputs=output)
        # 🔥 启用微调：冻结前1/3的层，微调后2/3的层
        self.models['densenet'].trainable = True
        total_layers = len(self.models['densenet'].layers)
        freeze_layers = int(total_layers / 3)
        for layer in self.models['densenet'].layers[:freeze_layers]:
            layer.trainable = False
        self.feature_dims['densenet'] = 1920
        
    def _build_vgg16(self):
        """构建VGG16模型"""
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        self.models['vgg16'] = Model(inputs=base_model.input, outputs=output)
        # 🔥 启用微调：冻结前1/3的层，微调后2/3的层
        self.models['vgg16'].trainable = True
        total_layers = len(self.models['vgg16'].layers)
        freeze_layers = int(total_layers / 3)
        for layer in self.models['vgg16'].layers[:freeze_layers]:
            layer.trainable = False
        self.feature_dims['vgg16'] = 512
        
    def extract_features(self, images):
        """简单高效的特征提取 - 强制GPU加速"""
        features_list = []

        if images.dtype != np.uint8:
            images = tf.cast(images, tf.uint8)

        # 预处理函数映射
        preprocess_map = {
            'efficientnet': preprocess_efficientnet,
            'resnet': preprocess_resnet,
            'densenet': preprocess_densenet,
            'vgg16': preprocess_vgg16
        }

        # 确保使用GPU进行推理
        gpu_device = '/GPU:0' if '/GPU:' in str(self.gpu_devices) else '/CPU:0'

        # 显示当前使用的设备
        print(f"🔧 使用设备: {gpu_device} (批次大小: {len(images)})")

        with tf.device(gpu_device):
            for model_name in self.selected_models:
                if model_name in self.models:
                    preprocess_func = preprocess_map[model_name]
                    processed_imgs = preprocess_func(images.copy())

                    # 强制使用GPU并显示进度
                    features = self.models[model_name].predict(
                        processed_imgs,
                        verbose=1,  # 显示推理进度
                        batch_size=min(32, len(processed_imgs))  # 小批次确保稳定
                    )
                    features_list.append(features)
                    print(f"   ✅ {model_name} 完成，形状: {features.shape}")
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
        
    def configure_gpu_settings(self, use_multi_gpu=True, batch_size=256):
        """
        动态配置GPU设置

        Args:
            use_multi_gpu (bool): 是否启用多GPU
            batch_size (int): 新的批次大小
        """
        self.use_multi_gpu = use_multi_gpu
        self.batch_size = batch_size
        print(f"🔧 GPU设置已更新:")
        print(f"   - 多GPU模式: {'启用' if use_multi_gpu else '禁用'}")
        print(f"   - 批次大小: {batch_size}")
        print(f"   - 可用GPU数量: {len(self.gpu_devices)}")

    def print_gpu_status(self):
        """打印GPU状态信息"""
        print(f"\n=== GPU状态检查 ===")
        print(f"物理GPU设备: {tf.config.list_physical_devices('GPU')}")
        print(f"逻辑GPU设备: {tf.config.list_logical_devices('GPU')}")
        print(f"可用GPU设备: {self.gpu_devices}")
        print(f"多GPU模式: {'启用' if self.use_multi_gpu else '禁用'}")

        # 检查每个GPU的内存使用情况
        try:
            for i, device in enumerate(self.gpu_devices):
                if 'GPU' in device:
                    print(f"GPU {i}: {device}")
                    # 尝试在每个GPU上创建一个小张量来验证可用性
                    with tf.device(device):
                        test_tensor = tf.constant([1.0, 2.0, 3.0])
                        print(f"  ✅ GPU {i} 可用，测试张量创建成功")
        except Exception as e:
            print(f"  ⚠️ GPU状态检查失败: {e}")

        print("="*30)

    def print_model_summary(self):
        """打印模型配置摘要"""
        print(f"\n=== 特征提取器配置摘要 ===")
        print(f"模式: {self.model_name}")
        print(f"选择的模型: {self.selected_models}")
        print(f"总特征维度: {self.total_feature_dim}")
        print(f"单模型模式: {'是' if self.is_single_model() else '否'}")
        print(f"多GPU模式: {'启用' if self.use_multi_gpu else '禁用'}")
        print(f"批次大小: {self.batch_size}")
        print(f"可用GPU: {len(self.gpu_devices)}个")

        # 显示详细的GPU状态
        self.print_gpu_status()
        print("="*30)