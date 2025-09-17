"""
预训练模型封装
支持多种主流架构的统一接口
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Tuple


class BackboneModel(nn.Module):
    """预训练模型骨干网络封装"""

    # 模型配置字典
    MODEL_CONFIGS = {
        'vgg16': {
            'torchvision_name': 'vgg16',
            'timm_name': 'vgg16',
            'feature_dim': 512,
            'fine_tune_layer': 'features.24',  # 从第24层开始微调
            'description': 'VGG16: 经典深层网络，适合胚胎图像特征提取'
        },
        'resnet50': {
            'torchvision_name': 'resnet50',
            'timm_name': 'resnet50',
            'feature_dim': 2048,
            'fine_tune_layer': 'layer3',  # 从layer3开始微调
            'description': 'ResNet50: 残差网络，解决深层网络梯度消失问题'
        },
        'resnet152': {
            'torchvision_name': 'resnet152',
            'timm_name': 'resnet152',
            'feature_dim': 2048,
            'fine_tune_layer': 'layer3',
            'description': 'ResNet152: 更深的残差网络，更强的表征能力'
        },
        'inceptionv3': {
            'torchvision_name': 'inception_v3',
            'timm_name': 'inception_v3',
            'feature_dim': 2048,
            'fine_tune_layer': 'Mixed_6a',
            'description': 'InceptionV3: 多尺度特征提取，适合复杂图像模式'
        },
        'efficientnetb0': {
            'torchvision_name': None,  # 使用timm
            'timm_name': 'efficientnet_b0',
            'feature_dim': 1280,
            'fine_tune_layer': 'blocks.5',
            'description': 'EfficientNetB0: 高效网络，平衡准确率和计算效率'
        }
    }

    def __init__(self,
                 model_name: str = 'vgg16',
                 pretrained: bool = True,
                 num_classes: int = 2,
                 fine_tune: bool = True,
                 fine_tune_layers: int = None):
        """
        初始化预训练模型

        Args:
            model_name: 模型名称
            pretrained: 是否使用预训练权重
            num_classes: 分类类别数
            fine_tune: 是否启用微调
            fine_tune_layers: 微调层数（可选）
        """
        super().__init__()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(self.MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.model_config = self.MODEL_CONFIGS[model_name]
        self.num_classes = num_classes
        self.fine_tune = fine_tune

        # 创建骨干网络
        self.backbone = self._create_backbone(pretrained)

        # 获取特征维度
        self.feature_dim = self.model_config['feature_dim']

        # 创建分类头
        self.classifier = self._create_classifier()

        # 设置微调策略
        if fine_tune:
            self._setup_fine_tuning(fine_tune_layers)
        else:
            self._freeze_backbone()

    def _create_backbone(self, pretrained: bool) -> nn.Module:
        """创建骨干网络"""
        # 优先使用timm，回退到torchvision
        try:
            if self.model_config['timm_name']:
                backbone = timm.create_model(
                    self.model_config['timm_name'],
                    pretrained=pretrained,
                    num_classes=0,  # 移除分类头
                    global_pool=''  # 移除全局池化
                )
                print(f"✅ 使用timm加载模型: {self.model_config['timm_name']}")
                return backbone
        except Exception as e:
            print(f"⚠️  timm加载失败: {e}")

        # 回退到torchvision
        if self.model_config['torchvision_name']:
            if self.model_name == 'vgg16':
                backbone = models.vgg16(pretrained=pretrained)
                # 移除分类器，只保留特征提取器
                backbone = backbone.features
            elif self.model_name.startswith('resnet'):
                backbone = getattr(models, self.model_config['torchvision_name'])(pretrained=pretrained)
                # 移除最后的全连接层
                backbone = nn.Sequential(*list(backbone.children())[:-1])
            elif self.model_name == 'inceptionv3':
                backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
                backbone = nn.Sequential(*list(backbone.children())[:-1])

            print(f"✅ 使用torchvision加载模型: {self.model_config['torchvision_name']}")
            return backbone

        raise RuntimeError(f"无法创建模型: {self.model_name}")

    def _create_classifier(self) -> nn.Module:
        """创建分类头"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )

    def _setup_fine_tuning(self, fine_tune_layers: int = None):
        """设置微调策略"""
        # 先冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 根据模型类型解冻指定层
        if fine_tune_layers is not None:
            self._unfreeze_top_layers(fine_tune_layers)
        else:
            self._unfreeze_default_layers()

        print(f"🔧 微调模式: 解冻部分层用于训练")

    def _unfreeze_top_layers(self, num_layers: int):
        """解冻顶部指定数量的层"""
        modules = list(self.backbone.modules())
        total_modules = len(modules)

        # 从后往前解冻指定数量的层
        for i in range(max(0, total_modules - num_layers), total_modules):
            for param in modules[i].parameters():
                param.requires_grad = True

    def _unfreeze_default_layers(self):
        """根据模型类型解冻默认层"""
        fine_tune_layer = self.model_config['fine_tune_layer']

        if self.model_name == 'vgg16':
            # VGG16: 解冻最后几个卷积层
            for i, layer in enumerate(self.backbone):
                if i >= 24:  # 从第24层开始
                    for param in layer.parameters():
                        param.requires_grad = True

        elif self.model_name.startswith('resnet'):
            # ResNet: 解冻最后几个残差块
            if hasattr(self.backbone, 'layer4'):
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, 'layer3'):
                for param in self.backbone.layer3.parameters():
                    param.requires_grad = True

        # 其他模型可以添加相应的解冻策略

    def _freeze_backbone(self):
        """冻结骨干网络"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 骨干网络已冻结，仅训练分类头")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 特征提取
        features = self.backbone(x)

        # 分类
        output = self.classifier(features)

        return output

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征图（用于Grad-CAM等可视化）"""
        return self.backbone(x)

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """获取可训练参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params, total_params

    def print_model_info(self):
        """打印模型信息"""
        trainable, total = self.get_trainable_parameters()
        print(f"\n🤖 模型信息:")
        print(f"   名称: {self.model_name}")
        print(f"   描述: {self.model_config['description']}")
        print(f"   特征维度: {self.feature_dim}")
        print(f"   分类数: {self.num_classes}")
        print(f"   总参数: {total:,}")
        print(f"   可训练参数: {trainable:,} ({100*trainable/total:.1f}%)")
        print(f"   微调模式: {'启用' if self.fine_tune else '禁用'}")


def create_model(config: Dict) -> BackboneModel:
    """
    根据配置创建模型的便捷函数

    Args:
        config: 模型配置字典

    Returns:
        BackboneModel实例
    """
    return BackboneModel(
        model_name=config.get('name', 'vgg16'),
        pretrained=config.get('pretrained', True),
        num_classes=config.get('num_classes', 2),
        fine_tune=config.get('fine_tune', True),
        fine_tune_layers=config.get('fine_tune_layers', None)
    )


def list_supported_models() -> Dict[str, str]:
    """列出所有支持的模型"""
    return {name: config['description']
            for name, config in BackboneModel.MODEL_CONFIGS.items()}