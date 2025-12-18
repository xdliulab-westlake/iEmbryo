#!/usr/bin/env python3
"""
可视化三类胚胎特征空间：Good, Fair, Poor
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import create_model
from visualization.feature_space import visualize_feature_space
from config.config_manager import load_config


def convert_old_state_dict(state_dict, model_name):
    """
    转换旧版本模型的state_dict键名到新版本

    旧版本: backbone.0.x..., classifier.x...
    新版本: model.x..., model.fc...
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # 转换backbone键名
        if key.startswith('backbone.0.'):
            # backbone.0.0.weight -> model.conv1.weight
            # backbone.0.1.weight -> model.bn1.weight
            # backbone.0.4.x -> model.layer1.x (ResNet)
            # 等等
            remainder = key[len('backbone.0.'):]
            new_key = f'model.{remainder}'

        # 转换classifier键名
        elif key.startswith('classifier.'):
            # classifier.x.weight -> model.fc.weight (或其他分类器结构)
            if model_name in ['resnet50', 'resnet152']:
                # ResNet的分类器是单层fc
                # classifier.9.weight -> model.fc.weight (假设最后一层是第9层)
                if 'classifier.9.' in key or 'classifier.6.' in key:
                    new_key = key.replace('classifier.9.', 'model.fc.').replace('classifier.6.', 'model.fc.')
                else:
                    # 其他classifier层跳过
                    continue
            elif model_name == 'vgg16':
                # VGG16的分类器是多层
                remainder = key[len('classifier.'):]
                new_key = f'model.classifier.{remainder}'
            elif model_name.startswith('densenet'):
                # DenseNet的分类器
                if 'classifier.9.' in key or 'classifier.6.' in key:
                    new_key = key.replace('classifier.9.', 'model.classifier.').replace('classifier.6.', 'model.classifier.')
                else:
                    continue

        new_state_dict[new_key] = value

    return new_state_dict


def patch_legacy_model(legacy_model):
    """
    修补旧版本模型对象，添加缺失的属性使其兼容

    旧模型结构:
    - ResNet/VGG: backbone: Sequential([Sequential(...), ...])
    - DenseNet: backbone: Sequential([Conv, BN, ReLU, MaxPool, DenseBlock, ...])
    - InceptionV3: backbone: Sequential([Sequential(...), ...])

    新模型期望:
    - model: 包含backbone的模型
    """
    # 找到实际的backbone模型
    if not hasattr(legacy_model, 'backbone'):
        return legacy_model

    if not hasattr(legacy_model.backbone, '__getitem__') or len(legacy_model.backbone) == 0:
        legacy_model.model = legacy_model.backbone
        print(f"   修补模型: 添加 model 属性指向 backbone")
        return legacy_model

    # backbone[0]是实际的网络模型或第一层
    actual_backbone = legacy_model.backbone[0]

    # 情况1: backbone[0]直接是ResNet/VGG模型（有avgpool或features属性）
    if hasattr(actual_backbone, 'avgpool'):
        legacy_model.model = actual_backbone
        print(f"   修补模型: {type(actual_backbone).__name__} (有avgpool)")
        return legacy_model

    if hasattr(actual_backbone, 'features'):
        legacy_model.model = actual_backbone
        print(f"   修补模型: {type(actual_backbone).__name__} (有features)")
        return legacy_model

    # 情况2: backbone[0]是Sequential（展开的ResNet/InceptionV3）
    if hasattr(actual_backbone, '__getitem__') and len(actual_backbone) > 0:
        # 查找AdaptiveAvgPool层
        has_avgpool = False
        for idx, item in enumerate(actual_backbone):
            if 'AdaptiveAvgPool' in type(item).__name__:
                # 找到avgpool，添加属性引用
                actual_backbone.avgpool = actual_backbone[idx]
                has_avgpool = True
                break

        if has_avgpool:
            legacy_model.model = actual_backbone
            print(f"   修补模型: 展开的ResNet/InceptionV3 (添加avgpool引用)")
            return legacy_model

    # 情况3: DenseNet结构 - backbone直接包含所有层
    # 检查backbone本身是否包含adaptive_avg_pool
    if len(legacy_model.backbone) >= 3:
        # DenseNet: backbone = Sequential([features, ReLU, AdaptiveAvgPool2d, ...])
        for idx, layer in enumerate(legacy_model.backbone):
            if 'AdaptiveAvgPool' in type(layer).__name__:
                # 创建一个包含features的简化结构
                # DenseNet的features通常在backbone[0]
                if hasattr(legacy_model.backbone[0], '__len__'):
                    # backbone[0]是features Sequential
                    features_seq = legacy_model.backbone[0]
                    # 添加avgpool引用
                    features_seq.avgpool = legacy_model.backbone[idx]
                    legacy_model.model = features_seq
                    print(f"   修补模型: DenseNet (features + avgpool引用)")
                    return legacy_model

    # 默认情况：使用actual_backbone
    legacy_model.model = actual_backbone
    print(f"   修补模型: 默认使用 {type(actual_backbone).__name__}")
    return legacy_model


class ThreeClassEmbryoDataset(Dataset):
    """三类胚胎数据集：Good, Fair, Poor"""

    def __init__(self, csv_path: str, base_dir: str, fair_dir: str, transform=None, max_samples_per_class=None, max_fair_samples=None):
        """
        初始化数据集

        Args:
            csv_path: CSV文件路径（包含Good和Poor）
            base_dir: CSV中图片的基础目录
            fair_dir: Fair胚胎的目录
            transform: 图像转换
            max_samples_per_class: 每个类别的最大样本数
            max_fair_samples: Fair类别的最大样本数（优先于max_samples_per_class）
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.embryo_ids = []

        # 1. 从CSV加载Good和Poor
        print("\n📂 从CSV加载Good和Poor胚胎...")
        df = pd.read_csv(csv_path)
        base_path = Path(base_dir)

        good_count = 0
        poor_count = 0

        for _, row in df.iterrows():
            img_path = base_path / row['Saved_Path']
            if img_path.exists():
                quality = int(row['Label'])
                if quality == 1:  # Viable → Good
                    self.image_paths.append(str(img_path))
                    self.labels.append(2)  # Good = 2
                    self.embryo_ids.append(row['Embryo_ID'])
                    good_count += 1
                elif quality == 0:  # Non-viable → Poor
                    self.image_paths.append(str(img_path))
                    self.labels.append(0)  # Poor = 0
                    self.embryo_ids.append(row['Embryo_ID'])
                    poor_count += 1

        print(f"  ✓ Good胚胎: {good_count} 张图片")
        print(f"  ✓ Poor胚胎: {poor_count} 张图片")

        # 2. 从目录加载Fair胚胎
        print("\n📂 从目录加载Fair胚胎...")
        fair_path = Path(fair_dir)
        fair_images = []
        fair_embryo_ids = []

        if fair_path.exists():
            # 递归查找所有图片文件
            for img_file in fair_path.rglob("*.jpeg"):
                embryo_id = img_file.parent.parent.name
                fair_images.append(str(img_file))
                fair_embryo_ids.append(embryo_id)

            # 也查找.jpg文件
            for img_file in fair_path.rglob("*.jpg"):
                embryo_id = img_file.parent.parent.name
                fair_images.append(str(img_file))
                fair_embryo_ids.append(embryo_id)

        # 如果指定了max_fair_samples，随机采样
        if max_fair_samples and len(fair_images) > max_fair_samples:
            print(f"  Fair胚胎总数: {len(fair_images)} 张图片")
            print(f"  随机采样: {max_fair_samples} 张图片")
            indices = np.random.choice(len(fair_images), max_fair_samples, replace=False)
            fair_images = [fair_images[i] for i in indices]
            fair_embryo_ids = [fair_embryo_ids[i] for i in indices]

        # 添加到数据集
        self.image_paths.extend(fair_images)
        self.labels.extend([1] * len(fair_images))  # Fair = 1
        self.embryo_ids.extend(fair_embryo_ids)
        fair_count = len(fair_images)

        print(f"  ✓ Fair胚胎: {fair_count} 张图片")

        # 3. 类别平衡采样（可选）
        if max_samples_per_class:
            self._balance_classes(max_samples_per_class)

        # 4. 统计信息
        print(f"\n📊 最终数据集统计:")
        print(f"  总图像数: {len(self.image_paths)}")
        print(f"  总胚胎数: {len(set(self.embryo_ids))}")

        unique_labels, counts = np.unique(self.labels, return_counts=True)
        class_names = {0: 'Poor', 1: 'Fair', 2: 'Good'}
        print(f"\n  类别分布:")
        for label, count in zip(unique_labels, counts):
            print(f"    {class_names[label]} ({label}): {count} 张图片 ({count/len(self.labels)*100:.1f}%)")

    def _balance_classes(self, max_samples_per_class: int):
        """平衡各类别样本数"""
        print(f"\n⚖️  平衡类别采样 (每类最多 {max_samples_per_class} 个样本)...")

        # 按类别分组
        class_indices = {0: [], 1: [], 2: []}
        for i, label in enumerate(self.labels):
            class_indices[label].append(i)

        # 对每个类别进行采样
        balanced_indices = []
        for class_id, indices in class_indices.items():
            if len(indices) > max_samples_per_class:
                sampled = np.random.choice(indices, max_samples_per_class, replace=False)
                balanced_indices.extend(sampled)
            else:
                balanced_indices.extend(indices)

        # 重新构建数据
        self.image_paths = [self.image_paths[i] for i in balanced_indices]
        self.labels = [self.labels[i] for i in balanced_indices]
        self.embryo_ids = [self.embryo_ids[i] for i in balanced_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  加载图像失败: {img_path}, 错误: {e}")
            # 返回黑色图像作为占位符
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def visualize_three_classes(model_path: str,
                            csv_path: str,
                            base_dir: str,
                            fair_dir: str,
                            output_dir: str,
                            config_path: str = 'config/default.yaml',
                            model_name: str = None,
                            max_samples_per_class: int = None,
                            max_fair_samples: int = None,
                            batch_size: int = 64,
                            methods: list = ['pca', 'tsne', 'umap']):
    """
    可视化三类胚胎特征空间

    Args:
        model_path: 模型路径
        csv_path: CSV文件路径（Good和Poor）
        base_dir: CSV图片基础目录
        fair_dir: Fair胚胎目录
        output_dir: 输出目录
        config_path: 配置文件路径
        model_name: 模型名称（覆盖config中的设置）
        max_samples_per_class: 每类最大样本数
        max_fair_samples: Fair类别的最大样本数
        batch_size: 批次大小
        methods: 降维方法
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用设备: {device}")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    print(f"\n⚙️  加载配置: {config_path}")
    config_manager = load_config(config_path)
    config = config_manager.config

    # 如果指定了模型名称，覆盖配置
    if model_name:
        print(f"⚙️  覆盖模型名称: {model_name}")
        config['model']['name'] = model_name

    # 创建图像转换
    img_size = config['data'].get('img_size', 224)
    if config['model']['name'] == 'inceptionv3':
        img_size = 299

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    print("\n" + "="*80)
    print("📂 加载三类胚胎数据集...")
    print("="*80)
    dataset = ThreeClassEmbryoDataset(
        csv_path=csv_path,
        base_dir=base_dir,
        fair_dir=fair_dir,
        transform=transform,
        max_samples_per_class=max_samples_per_class,
        max_fair_samples=max_fair_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 加载模型
    print("\n" + "="*80)
    print("🔧 加载模型...")
    print("="*80)

    checkpoint = torch.load(model_path, map_location=device)

    # 检查是否是旧版本的完整模型对象
    if hasattr(checkpoint, 'state_dict') and hasattr(checkpoint, 'eval'):
        # checkpoint是完整的旧版本模型对象
        print(f"📦 检测到旧版本模型对象: {type(checkpoint).__name__}")

        # 修补旧模型使其兼容
        model = patch_legacy_model(checkpoint)
        print("✅ 已修补旧版本模型")

    elif isinstance(checkpoint, dict):
        # checkpoint是字典，可能包含state_dict
        print("📦 检测到字典格式checkpoint")
        model = create_model(config['model'])

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 检查是否需要转换键名
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('backbone.') or first_key.startswith('classifier.'):
            print("🔄 检测到旧版本键名，进行转换...")
            state_dict = convert_old_state_dict(state_dict, model_name or config['model']['name'])

        model.load_state_dict(state_dict, strict=False)
        print("✅ 模型加载成功")

    else:
        # 其他情况
        print("📦 未知格式，尝试直接加载")
        model = create_model(config['model'])
        model.load_state_dict(checkpoint, strict=False)
        print("✅ 模型加载成功")

    model.to(device)
    model.eval()

    print(f"✅ 模型已移动到 {device} 并设置为评估模式")

    # 可视化特征空间
    print("\n" + "="*80)
    print("🎨 开始三类胚胎特征空间可视化...")
    print("="*80)

    visualize_feature_space(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        methods=methods,
        max_samples=None  # 已经在数据集中限制了
    )

    print("\n" + "="*80)
    print("✅ 完成！")
    print(f"📁 结果保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='可视化三类胚胎特征空间：Good, Fair, Poor')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--csv_path', type=str,
                       default='/Users/lzz/Documents/GitHub/IVF_AI/extracted_samples/embryos_all_focal_planes.csv',
                       help='CSV文件路径（Good和Poor）')
    parser.add_argument('--base_dir', type=str,
                       default='/Users/lzz/Documents/GitHub/IVF_AI',
                       help='CSV图片基础目录')
    parser.add_argument('--fair_dir', type=str,
                       default='/Users/lzz/Downloads/fair_embryos',
                       help='Fair胚胎目录')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/three_class_visualization',
                       help='输出目录')
    parser.add_argument('--config', type=str,
                       default='config/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--model_name', type=str, default=None,
                       help='模型名称（覆盖config中的设置）: vgg16, resnet50, resnet152, inceptionv3, densenet121, densenet169, densenet201')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='每类最大样本数（None表示使用全部）')
    parser.add_argument('--max_fair_samples', type=int, default=None,
                       help='Fair类别的最大样本数（None表示使用全部）')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['pca', 'tsne', 'umap'],
                       help='降维方法: pca, tsne, umap')

    args = parser.parse_args()

    visualize_three_classes(
        model_path=args.model_path,
        csv_path=args.csv_path,
        base_dir=args.base_dir,
        fair_dir=args.fair_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        model_name=args.model_name,
        max_samples_per_class=args.max_samples_per_class,
        max_fair_samples=args.max_fair_samples,
        batch_size=args.batch_size,
        methods=args.methods
    )
