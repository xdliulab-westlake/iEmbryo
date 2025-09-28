"""
预训练模型封装
支持多种主流架构的统一接口
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Tuple
try:
    from ..data.gpu_transforms import create_gpu_transforms, check_kornia_available
except ImportError:
    # 处理相对导入问题
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from data.gpu_transforms import create_gpu_transforms, check_kornia_available


class BackboneModel(nn.Module):
    """预训练模型骨干网络封装"""

    # 预训练模型路径映射
    PRETRAINED_PATHS = {
        'vgg16': "/liaozizhuo/scripts/pretrain_models/vgg16-397923af.pth",
        'resnet50': "/liaozizhuo/scripts/pretrain_models/resnet50-11ad3fa6.pth",
        'resnet152': "/liaozizhuo/scripts/pretrain_models/resnet152-394f9c45.pth",
        'inceptionv3': "/liaozizhuo/scripts/pretrain_models/inception_v3_google-0cc3c7bd.pth",
        'efficientnetb0': "/liaozizhuo/scripts/pretrain_models/efficientnet-b0-355c32eb.pth"
    }

    # 模型配置字典
    MODEL_CONFIGS = {
        'vgg16': {
            'torchvision_name': 'vgg16',
            'timm_name': 'vgg16',
            'feature_dim': 4096,
            'fine_tune_layer': 'features.24',
            'description': 'VGG16: 经典深层网络，适合胚胎图像特征提取'
        },
        'resnet50': {
            'torchvision_name': 'resnet50',
            'timm_name': 'resnet50',
            'feature_dim': 2048,
            'fine_tune_layer': 'layer3',
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
            'torchvision_name': None,
            'timm_name': 'efficientnet_b0',
            'feature_dim': 1280,
            'fine_tune_layer': 'blocks.5',
            'description': 'EfficientNetB0: 高效网络，平衡准确率和计算效率'
        }
    }

    def __init__(self,
                 model_name: str = 'vgg16',
                 pretrained: bool = True,
                 pretrained_path: str = None,
                 num_classes: int = 2,
                 fine_tune: bool = True,
                 fine_tune_layers: int = None,
                 config: Dict = None):
        """
        初始化预训练模型

        Args:
            model_name: 模型名称
            pretrained: 是否使用预训练权重
            pretrained_path: 本地预训练模型路径
            num_classes: 分类类别数
            fine_tune: 是否启用微调
            fine_tune_layers: 微调层数（可选）
            config: 配置字典（用于GPU数据增强）
        """
        super().__init__()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(self.MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.model_config = self.MODEL_CONFIGS[model_name]
        self.num_classes = num_classes
        self.fine_tune = fine_tune
        self.pretrained_path = pretrained_path
        self.config = config or {}

        # 初始化GPU数据增强
        self.gpu_aug = self._setup_gpu_augmentation()

        # 创建骨干网络
        self.backbone = self._create_backbone(pretrained, pretrained_path)
        self.feature_dim = self.model_config['feature_dim']

        # 创建分类头
        self.classifier = self._create_classifier()

        # 设置微调策略
        if fine_tune:
            self._setup_fine_tuning(fine_tune_layers)
        else:
            self._freeze_backbone()

    def _setup_gpu_augmentation(self):
        """设置GPU数据增强"""
        if not self.config or not check_kornia_available():
            if self.config:
                print("Kornia未安装，无法使用GPU数据增强")
            return None

        try:
            gpu_aug = create_gpu_transforms(self.config)
            print("GPU数据增强模块已启用")
            return gpu_aug
        except Exception as e:
            print(f"GPU数据增强初始化失败: {e}")
            print("回退到CPU数据增强")
            return None

    def _create_backbone(self, pretrained: bool, pretrained_path: str = None) -> nn.Module:
        """创建骨干网络，直接使用预训练模型"""
        print(f"直接使用预训练模型: {self.model_name}")
        return self._create_online_model(pretrained)

    def _create_online_model(self, pretrained: bool) -> nn.Module:
        """创建在线模型（统一timm和torchvision逻辑）"""
        # 优先使用timm
        if self.model_config['timm_name']:
            try:
                backbone = timm.create_model(
                    self.model_config['timm_name'],
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool=''
                )
                print(f"使用timm加载模型: {self.model_config['timm_name']}")
                return backbone
            except Exception as e:
                print(f"timm加载失败: {e}，回退到torchvision")

        # 回退到torchvision
        return self._create_torchvision_model(pretrained)

    def _create_torchvision_model(self, pretrained: bool) -> nn.Module:
        """使用torchvision创建模型"""
        model_creators = {
            'vgg16': self._create_vgg16,
            'inceptionv3': self._create_inception_v3
        }

        # ResNet系列
        if self.model_name.startswith('resnet'):
            return self._create_resnet(pretrained)

        if self.model_name in model_creators:
            return model_creators[self.model_name](pretrained)

        raise ValueError(f"不支持的模型: {self.model_name}")

    def _create_vgg16(self, pretrained: bool) -> nn.Module:
        """创建VGG16模型"""
        backbone = models.vgg16(pretrained=pretrained)
        features = backbone.features
        classifier_features = nn.Sequential(*list(backbone.classifier.children())[:4])
        return nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            classifier_features
        )

    def _create_resnet(self, pretrained: bool) -> nn.Module:
        """创建ResNet系列模型"""
        backbone = getattr(models, self.model_config['torchvision_name'])(pretrained=pretrained)
        return nn.Sequential(*list(backbone.children())[:-1])

    def _create_inception_v3(self, pretrained: bool) -> nn.Module:
        """创建InceptionV3模型"""
        backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
        return nn.Sequential(*list(backbone.children())[:-1])

    def _create_classifier(self) -> nn.Module:
        """创建分类头"""
        # VGG16已输出展平特征
        if self.model_name == 'vgg16':
            return nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes)
            )

        # 其他模型需要pooling和flatten
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

        # 应用微调策略
        if fine_tune_layers is not None:
            self._unfreeze_top_layers(fine_tune_layers)
        else:
            self._apply_optimal_fine_tuning()

        print(f"微调模式: 解冻部分层用于训练")

    def _apply_optimal_fine_tuning(self):
        """应用最优微调策略：固定前1/3层，微调后2/3层"""
        trainable_layers = self._collect_trainable_layers()
        total_layers = len(trainable_layers)
        optimal_fine_tune_at = int(total_layers * 1/3)

        print(f"启用最优微调策略: 固定前1/3层")
        print(f"{self.model_name.upper()} 层数分析:")
        print(f"   总层数: {total_layers}")
        print(f"   固定层数: {optimal_fine_tune_at} (前 {optimal_fine_tune_at/total_layers:.1%})")
        print(f"   微调层数: {total_layers - optimal_fine_tune_at} (后 {(total_layers - optimal_fine_tune_at)/total_layers:.1%})")

        # 冻结前1/3的层
        for i, (layer_name, layer) in enumerate(trainable_layers[:optimal_fine_tune_at]):
            for param in layer.parameters():
                param.requires_grad = False

        # 解冻后2/3的层
        for i, (layer_name, layer) in enumerate(trainable_layers[optimal_fine_tune_at:]):
            for param in layer.parameters():
                param.requires_grad = True

        frozen_count = optimal_fine_tune_at
        trainable_count = total_layers - optimal_fine_tune_at

        print(f"微调配置完成: {trainable_count}/{total_layers} 层可训练")
        print(f"   冻结层: {frozen_count} 层 (前1/3)")
        print(f"   训练层: {trainable_count} 层 (后2/3)")

    def _collect_trainable_layers(self):
        """收集所有可训练的层"""
        trainable_layers = []

        def collect_layers(module, prefix=""):
            for name, child in module.named_children():
                current_name = f"{prefix}.{name}" if prefix else name
                if len(list(child.parameters(recurse=False))) > 0:
                    trainable_layers.append((current_name, child))
                else:
                    collect_layers(child, current_name)

        collect_layers(self.backbone)
        return trainable_layers

    def _unfreeze_top_layers(self, num_layers: int):
        """解冻顶部指定数量的层"""
        modules = list(self.backbone.modules())
        total_modules = len(modules)

        for i in range(max(0, total_modules - num_layers), total_modules):
            for param in modules[i].parameters():
                param.requires_grad = True

    def _freeze_backbone(self):
        """冻结骨干网络"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("骨干网络已冻结，仅训练分类头")

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """前向传播"""
        # GPU数据增强（如果启用且在训练模式）
        if self.gpu_aug is not None and self.training:
            if labels is not None:
                # 有标签时，可能会应用MixUp/CutMix
                x, labels_a, labels_b, lam = self.gpu_aug(x, labels)
                features = self.backbone(x)
                output = self.classifier(features)
                return output, labels_a, labels_b, lam
            else:
                x = self.gpu_aug(x)

        # 特征提取和分类
        features = self.backbone(x)
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
        print(f"\n模型信息:")
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
        pretrained_path=config.get('pretrained_path', None),
        num_classes=config.get('num_classes', 2),
        fine_tune=config.get('fine_tune', True),
        fine_tune_layers=config.get('fine_tune_layers', None),
        config={'model': config}
    )


def list_supported_models() -> Dict[str, str]:
    """列出所有支持的模型"""
    return {name: config['description']
            for name, config in BackboneModel.MODEL_CONFIGS.items()}