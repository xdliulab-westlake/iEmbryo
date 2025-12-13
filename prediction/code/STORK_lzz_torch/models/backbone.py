"""
简化版预训练模型加载
直接使用原始预训练权重，仅修改分类头
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict


class BackboneModel(nn.Module):
    """简化版模型封装 - 直接使用原始预训练权重"""

    # 本地预训练模型路径（优先使用）
    PRETRAINED_PATHS = {
        'vgg16': "/liaozizhuo/scripts/pretrain_models/vgg16-397923af.pth",
        'vgg19': "/liaozizhuo/scripts/pretrain_models/vgg19-dcbb9e26.pth",
        'resnet50': "/liaozizhuo/scripts/pretrain_models/resnet50-11ad3fa6.pth",
        'resnet152': "/liaozizhuo/scripts/pretrain_models/resnet152-394f9c45.pth",
        'inceptionv3': "/liaozizhuo/scripts/pretrain_models/inception_v3_google-0cc3c7bd.pth",
        'efficientnetb0': "/liaozizhuo/scripts/pretrain_models/efficientnet-b0-355c32eb.pth",
        'densenet121': "/liaozizhuo/scripts/pretrain_models/densenet121-a639ec97.pth",
        'densenet169': "/liaozizhuo/scripts/pretrain_models/densenet169-b2777c0a.pth",
        'densenet201': "/liaozizhuo/scripts/pretrain_models/densenet201-c1103571.pth"
    }

    # 模型配置（包含最后一层的输入维度和属性名）
    MODEL_CONFIGS = {
        'vgg16': {
            'loader': lambda pretrained=False: models.vgg16(pretrained=pretrained),
            'classifier_attr': 'classifier',
            'last_layer_in_features': 4096,
            'description': 'VGG16: 经典深层网络'
        },
        'resnet50': {
            'loader': lambda pretrained=False: models.resnet50(pretrained=pretrained),
            'classifier_attr': 'fc',
            'last_layer_in_features': 2048,
            'description': 'ResNet50: 残差网络'
        },
        'resnet152': {
            'loader': lambda pretrained=False: models.resnet152(pretrained=pretrained),
            'classifier_attr': 'fc',
            'last_layer_in_features': 2048,
            'description': 'ResNet152: 更深的残差网络'
        },
        'inceptionv3': {
            'loader': lambda pretrained=False: models.inception_v3(pretrained=pretrained, aux_logits=False),
            'classifier_attr': 'fc',
            'last_layer_in_features': 2048,
            'description': 'InceptionV3: 多尺度特征提取'
        },
        'efficientnetb0': {
            'loader': lambda pretrained=False: timm.create_model('efficientnet_b0', pretrained=pretrained),
            'classifier_attr': 'classifier',
            'last_layer_in_features': 1280,
            'description': 'EfficientNetB0: 高效网络'
        },
        'densenet121': {
            'loader': lambda pretrained=False: models.densenet121(pretrained=pretrained),
            'classifier_attr': 'classifier',
            'last_layer_in_features': 1024,
            'description': 'DenseNet121: 密集连接网络'
        },
        'densenet169': {
            'loader': lambda pretrained=False: models.densenet169(pretrained=pretrained),
            'classifier_attr': 'classifier',
            'last_layer_in_features': 1664,
            'description': 'DenseNet169: 更深的密集连接网络'
        },
        'densenet201': {
            'loader': lambda pretrained=False: models.densenet201(pretrained=pretrained),
            'classifier_attr': 'classifier',
            'last_layer_in_features': 1920,
            'description': 'DenseNet201: 最深的密集连接网络'
        },
        'vgg19': {
            'loader': lambda pretrained=False: models.vgg19(pretrained=pretrained),
            'classifier_attr': 'classifier',
            'last_layer_in_features': 4096,
            'description': 'VGG19: VGG系列最深版本'
        },
        'convnext_tiny': {
            'loader': lambda pretrained=False: timm.create_model('convnext_tiny', pretrained=pretrained),
            'classifier_attr': 'head.fc',
            'last_layer_in_features': 768,
            'description': 'ConvNeXt-Tiny: 现代纯卷积网络(小)'
        }
    }

    def __init__(self,
                 model_name: str = 'vgg16',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 fine_tune: bool = True,
                 fine_tune_layers: int = None):
        """
        初始化预训练模型，仅修改分类头

        Args:
            model_name: 模型名称（vgg16, resnet50, resnet152等）
            num_classes: 分类类别数（默认2：好/坏胚胎）
            pretrained: 是否使用预训练权重（默认True）
            fine_tune: 是否启用微调（默认True）
            fine_tune_layers: 微调层数（None=自动选择，冻结前1/3层，微调后2/3层）
        """
        super().__init__()
        import os

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(self.MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.num_classes = num_classes
        self.config = self.MODEL_CONFIGS[model_name]
        self.fine_tune = fine_tune

        # 加载预训练模型
        print(f"加载预训练模型: {model_name}")

        # 优先使用本地预训练权重
        if model_name in self.PRETRAINED_PATHS:
            local_path = self.PRETRAINED_PATHS[model_name]
            if os.path.exists(local_path):
                print(f"  使用本地预训练权重: {local_path}")
                # 创建未预训练的模型，然后加载本地权重
                self.model = self.config['loader'](pretrained=False)
                state_dict = torch.load(local_path, map_location='cpu')
                self.model.load_state_dict(state_dict, strict=False)
            else:
                print(f"  本地预训练权重不存在: {local_path}")
                # 本地不存在，检查是否需要从网络下载
                if pretrained:
                    print(f"  从网络下载预训练权重...")
                    self.model = self.config['loader'](pretrained=True)
                else:
                    self.model = self.config['loader'](pretrained=False)
        else:
            # 不在本地路径列表中，根据pretrained标志决定
            self.model = self.config['loader'](pretrained=pretrained)

        # 修改分类头
        classifier_attr = self.config['classifier_attr']
        in_features = self.config['last_layer_in_features']

        # VGG系列（VGG16/19）的classifier是Sequential，需要替换整个部分
        if model_name in ['vgg16', 'vgg19']:
            self.model.classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, num_classes)
            )
        # ConvNeXt特殊处理（分类层名称是head.fc）
        elif model_name.startswith('convnext'):
            # ConvNeXt的head.fc是一个Linear层，可以直接替换
            setattr(self.model.head, 'fc', nn.Linear(in_features, num_classes))
        else:
            # 其他模型直接替换最后一层
            setattr(self.model, classifier_attr, nn.Linear(in_features, num_classes))

        print(f"✅ 已加载 {model_name} 预训练模型")
        print(f"   输出类别数: {num_classes}")

        # 设置微调策略
        if fine_tune:
            self._setup_fine_tuning(fine_tune_layers)
        else:
            # 冻结所有参数，仅训练分类头
            self._freeze_all_but_classifier()

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def print_model_info(self):
        """打印模型信息"""
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"✅ 已加载 {self.model_name} 预训练模型")
        print(f"   输出类别数: {self.num_classes}")
        print(f"   总参数数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   冻结参数: {total_params - trainable_params:,}")

    def __getstate__(self):
        """自定义pickle支持，排除不可序列化的lambda函数"""
        state = self.__dict__.copy()
        # 完全删除config引用（它包含lambda函数）
        state.pop('config', None)
        # 保存必要的配置信息
        state['_model_config'] = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'fine_tune': self.fine_tune,
            'classifier_attr': self.config['classifier_attr'],
            'last_layer_in_features': self.config['last_layer_in_features'],
        }
        return state

    def __setstate__(self, state):
        """自定义unpickle支持，恢复模型配置"""
        model_config = state.pop('_model_config', {})
        self.__dict__.update(state)
        # 恢复config引用（指向类级别的MODEL_CONFIGS）
        if model_config:
            self.config = self.MODEL_CONFIGS.get(
                model_config.get('model_name'),
                self.MODEL_CONFIGS['vgg16']
            )
        else:
            # 如果没有保存的config，使用默认的
            self.config = self.MODEL_CONFIGS.get('vgg16')

    def _setup_fine_tuning(self, fine_tune_layers: int = None):
        """
        设置微调策略

        Args:
            fine_tune_layers: 指定微调的层数
                            None: 自动选择，冻结前1/3层，微调后2/3层
                            >0: 指定微调后面几层
        """
        # 获取所有参数
        all_params = list(self.model.parameters())
        total_layers = len(all_params)

        if fine_tune_layers is None:
            # 自动策略：冻结前1/3，微调后2/3
            freeze_layers = total_layers // 3
            trainable_layers = total_layers - freeze_layers
        else:
            # 指定策略
            freeze_layers = total_layers - fine_tune_layers
            trainable_layers = fine_tune_layers

        # 冻结前面的层
        for param in all_params[:freeze_layers]:
            param.requires_grad = False

        # 解冻后面的层
        for param in all_params[freeze_layers:]:
            param.requires_grad = True

        print(f"微调配置:")
        print(f"  总层数: {total_layers}")
        print(f"  冻结层: {freeze_layers} (前 {freeze_layers/total_layers*100:.1f}%)")
        print(f"  训练层: {trainable_layers} (后 {trainable_layers/total_layers*100:.1f}%)")

    def _freeze_all_but_classifier(self):
        """冻结所有参数，仅保留分类头可训练"""
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻分类头参数
        classifier_attr = self.config['classifier_attr']

        # 处理嵌套属性（如 head.fc）
        if '.' in classifier_attr:
            attr_parts = classifier_attr.split('.')
            classifier = self.model
            for part in attr_parts:
                classifier = getattr(classifier, part)
        else:
            classifier = getattr(self.model, classifier_attr)

        for param in classifier.parameters():
            param.requires_grad = True

        print(f"微调配置:")
        print(f"  冻结所有 backbone 参数")
        print(f"  仅训练分类头 ({classifier_attr})")


def create_model(config: Dict) -> BackboneModel:
    """
    从配置字典创建模型

    Args:
        config: 配置字典，包含：
            - name: 模型名称
            - num_classes: 类别数
            - pretrained: 是否使用预训练权重
            - fine_tune: 是否启用微调
            - fine_tune_layers: 微调的层数

    Returns:
        BackboneModel 实例
    """
    return BackboneModel(
        model_name=config.get('name', 'vgg16'),
        num_classes=config.get('num_classes', 2),
        pretrained=config.get('pretrained', True),
        fine_tune=config.get('fine_tune', True),
        fine_tune_layers=config.get('fine_tune_layers', None)
    )


def list_supported_models() -> Dict[str, str]:
    """列出所有支持的模型"""
    return {name: config['description']
            for name, config in BackboneModel.MODEL_CONFIGS.items()}
