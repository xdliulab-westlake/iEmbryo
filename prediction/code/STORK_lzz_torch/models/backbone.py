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

    # 预训练模型路径映射 - 根据您的实际路径修改
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
            'feature_dim': 4096,  # VGG16完整模型的特征维度
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
            pretrained_path: 本地预训练模型路径，如果提供则优先使用本地模型
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

        # 创建GPU数据增强模块
        self.gpu_aug = None
        if config and check_kornia_available():
            try:
                self.gpu_aug = create_gpu_transforms(config)
                print("✅ GPU数据增强模块已启用")
            except Exception as e:
                print(f"⚠️ GPU数据增强初始化失败: {e}")
                print("🔄 回退到CPU数据增强")
        elif config:
            print("⚠️ Kornia未安装，无法使用GPU数据增强")

        # 创建骨干网络
        self.backbone = self._create_backbone(pretrained, pretrained_path)

        # 获取特征维度
        self.feature_dim = self.model_config['feature_dim']

        # 创建分类头
        self.classifier = self._create_classifier()

        # 设置微调策略
        if fine_tune:
            self._setup_fine_tuning(fine_tune_layers)
        else:
            self._freeze_backbone()

    def _create_backbone(self, pretrained: bool, pretrained_path: str = None) -> nn.Module:
        """创建骨干网络，支持本地预训练模型"""

        # 如果没有指定路径，尝试使用默认路径
        if not pretrained_path and self.model_name in self.PRETRAINED_PATHS:
            pretrained_path = self.PRETRAINED_PATHS[self.model_name]
            print(f"🔧 使用默认预训练模型路径: {pretrained_path}")

        # 如果有本地路径且文件存在，优先使用本地模型
        if pretrained_path and os.path.exists(pretrained_path):
            return self._load_local_pretrained(pretrained_path)
        elif pretrained_path and not os.path.exists(pretrained_path):
            print(f"⚠️ 预训练模型文件不存在: {pretrained_path}")
            print(f"🔄 回退到在线预训练模型 {self.model_name}...")
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

    def _load_local_pretrained(self, pretrained_path: str) -> nn.Module:
        """加载本地预训练模型"""
        try:
            print(f"🔄 正在加载本地预训练模型: {pretrained_path}")

            # 根据文件扩展名判断加载方式
            if pretrained_path.endswith('.pth') or pretrained_path.endswith('.pt'):
                # PyTorch格式
                return self._load_pytorch_model(pretrained_path)
            else:
                raise ValueError(f"不支持的模型格式: {pretrained_path}")

        except Exception as e:
            print(f"❌ 本地模型加载失败: {e}")
            print("🔄 回退到在线预训练模型...")
            # 回退到在线模型
            return self._create_online_backbone(True)

    def _load_pytorch_model(self, model_path: str) -> nn.Module:
        """加载PyTorch格式的预训练模型"""
        # 先创建模型架构（不加载预训练权重）
        if self.model_config['timm_name'] and self.model_name != 'vgg16':
            # 对于非VGG16模型，使用timm
            try:
                backbone = timm.create_model(
                    self.model_config['timm_name'],
                    pretrained=False,  # 不从网络下载
                    num_classes=0,
                    global_pool=''
                )
            except Exception as e:
                print(f"⚠️ timm创建模型失败: {e}，使用torchvision")
                backbone = self._create_torchvision_backbone(pretrained=False)
        else:
            # VGG16或其他情况使用torchvision创建空模型
            backbone = self._create_torchvision_backbone(pretrained=False)

        # 加载本地权重
        checkpoint = torch.load(model_path, map_location='cpu')

        # 处理不同的checkpoint格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 清理state_dict中的键名（去除可能的前缀）
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # 去除可能的 'backbone.', 'model.', 'module.' 等前缀
            clean_key = key
            for prefix in ['backbone.', 'model.', 'module.']:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break

            # 对于VGG16，需要重新映射键名到我们的Sequential结构
            if self.model_name == 'vgg16':
                if clean_key.startswith('features.'):
                    # features部分映射到0.xxx
                    clean_key = '0.' + clean_key[len('features.'):]
                elif clean_key.startswith('classifier.'):
                    # classifier的前4层映射到3.xxx
                    classifier_idx = int(clean_key.split('.')[1])
                    if classifier_idx < 4:  # 只保留前4层
                        clean_key = '3.' + clean_key[len('classifier.'):]
                    else:
                        continue  # 跳过后面的分类层

            cleaned_state_dict[clean_key] = value

        # 加载权重（允许部分匹配）
        missing_keys, unexpected_keys = backbone.load_state_dict(cleaned_state_dict, strict=False)

        if missing_keys:
            print(f"⚠️ 缺少的键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"⚠️ 多余的键: {len(unexpected_keys)} 个")

        print(f"✅ 成功加载本地PyTorch模型: {model_path}")

        return backbone

    def _create_torchvision_backbone(self, pretrained: bool) -> nn.Module:
        """使用torchvision创建骨干网络"""
        if self.model_name == 'vgg16':
            backbone = models.vgg16(pretrained=pretrained)
            # 对于VGG16，保留features + 部分classifier作为特征提取器
            # VGG16的classifier: [0-2是前两个Linear+ReLU, 3是dropout, 4-6是最后的分类层]
            # 我们保留features + classifier[:4] 作为特征提取器，输出4096维
            features = backbone.features
            classifier_features = nn.Sequential(*list(backbone.classifier.children())[:4])  # 只要前4层
            backbone = nn.Sequential(
                features,                    # 0: 卷积特征提取
                nn.AdaptiveAvgPool2d((7, 7)), # 1: 自适应池化到7x7
                nn.Flatten(),                # 2: 展平
                classifier_features          # 3: 分类器前4层
            )
        elif self.model_name.startswith('resnet'):
            backbone = getattr(models, self.model_config['torchvision_name'])(pretrained=pretrained)
            # 移除最后的全连接层
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif self.model_name == 'inceptionv3':
            backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Torchvision不支持模型: {self.model_name}")

        return backbone

    def _create_online_backbone(self, pretrained: bool) -> nn.Module:
        """创建在线预训练模型（原有逻辑）"""
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
        return self._create_torchvision_backbone(pretrained)

    def _create_classifier(self) -> nn.Module:
        """创建分类头"""
        # 对于VGG16，backbone已经输出展平的特征，不需要再pooling和flatten
        if self.model_name == 'vgg16':
            return nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes)
            )
        else:
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
        """设置最优微调策略：固定前1/3层，微调后2/3层"""
        # 先冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 使用最优微调策略（与STORK_lzz保持一致）
        if fine_tune_layers is not None:
            self._unfreeze_top_layers(fine_tune_layers)
        else:
            self._apply_optimal_fine_tuning()

        print(f"🔧 微调模式: 解冻部分层用于训练")

    def _apply_optimal_fine_tuning(self):
        """应用最优微调策略：固定前1/3层，微调后2/3层（与STORK_lzz一致）"""
        # 获取所有可训练的层（不包括容器层）
        trainable_layers = []

        def collect_trainable_layers(module, prefix=""):
            """递归收集所有有参数的层"""
            for name, child in module.named_children():
                current_name = f"{prefix}.{name}" if prefix else name
                if len(list(child.parameters(recurse=False))) > 0:
                    # 这是一个有参数的层
                    trainable_layers.append((current_name, child))
                else:
                    # 继续递归
                    collect_trainable_layers(child, current_name)

        collect_trainable_layers(self.backbone)

        total_layers = len(trainable_layers)
        optimal_fine_tune_at = int(total_layers * 1/3)

        print(f"🔓 启用最优微调策略: 固定前1/3层")
        print(f"📊 {self.model_name.upper()} 层数分析:")
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

        # 统计可训练参数
        frozen_count = optimal_fine_tune_at
        trainable_count = total_layers - optimal_fine_tune_at

        print(f"✅ 微调配置完成: {trainable_count}/{total_layers} 层可训练")
        print(f"   🔒 冻结层: {frozen_count} 层 (前1/3)")
        print(f"   🔓 训练层: {trainable_count} 层 (后2/3)")

    def _unfreeze_top_layers(self, num_layers: int):
        """解冻顶部指定数量的层"""
        modules = list(self.backbone.modules())
        total_modules = len(modules)

        # 从后往前解冻指定数量的层
        for i in range(max(0, total_modules - num_layers), total_modules):
            for param in modules[i].parameters():
                param.requires_grad = True

    def _unfreeze_default_layers(self):
        """根据模型类型解冻默认层（已弃用，使用最优微调策略）"""
        # 这个方法已被 _apply_optimal_fine_tuning 替代
        # 保留只是为了向后兼容
        print("⚠️ 使用已弃用的默认微调策略，建议使用最优策略")
        self._apply_optimal_fine_tuning()

    def _freeze_backbone(self):
        """冻结骨干网络"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 骨干网络已冻结，仅训练分类头")

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """前向传播"""
        # GPU数据增强（如果启用且在训练模式）
        if self.gpu_aug is not None and self.training:
            if labels is not None:
                # 有标签时，可能会应用MixUp/CutMix
                x, labels_a, labels_b, lam = self.gpu_aug(x, labels)

                # 特征提取
                features = self.backbone(x)

                # 分类
                output = self.classifier(features)

                # 返回混合后的结果和混合参数
                return output, labels_a, labels_b, lam
            else:
                # 无标签时，只应用基础增强
                x = self.gpu_aug(x)

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
    # config参数直接就是model配置，不需要再嵌套取model
    return BackboneModel(
        model_name=config.get('name', 'vgg16'),
        pretrained=config.get('pretrained', True),
        pretrained_path=config.get('pretrained_path', None),
        num_classes=config.get('num_classes', 2),
        fine_tune=config.get('fine_tune', True),
        fine_tune_layers=config.get('fine_tune_layers', None),
        config={'model': config}  # 重新构建完整配置结构用于GPU增强
    )


def list_supported_models() -> Dict[str, str]:
    """列出所有支持的模型"""
    return {name: config['description']
            for name, config in BackboneModel.MODEL_CONFIGS.items()}