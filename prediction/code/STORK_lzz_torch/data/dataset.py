"""
数据集类定义
支持胚胎图像分类和特征缓存机制
"""

import os
import pickle
import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class EmbryoDataset(Dataset):
    """胚胎图像数据集"""

    def __init__(self,
                 image_paths: List[str],
                 labels: List[int],
                 qualities: Optional[List[float]] = None,
                 transform: Optional[Callable] = None,
                 use_cache: bool = True,
                 cache_dir: str = "./cache"):
        """
        初始化数据集

        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            qualities: 质量分数列表（可选）
            transform: 数据变换
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
        """
        self.image_paths = image_paths
        self.labels = labels
        self.qualities = qualities or [0.0] * len(image_paths)
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

        # 创建缓存目录
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 验证数据
        assert len(self.image_paths) == len(self.labels), "图像路径和标签数量不匹配"
        assert len(self.image_paths) == len(self.qualities), "图像路径和质量分数数量不匹配"

        print(f"📊 数据集初始化完成:")
        print(f"   样本数量: {len(self.image_paths)}")
        print(f"   类别分布: {self._get_class_distribution()}")
        print(f"   缓存: {'启用' if self.use_cache else '禁用'}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        # 获取缓存的图像数据
        if self.use_cache:
            image = self._load_cached_image(idx)
        else:
            image = self._load_image(idx)

        label = self.labels[idx]

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_image(self, idx: int) -> Image.Image:
        """加载原始图像"""
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"❌ 加载图像失败: {image_path}, 错误: {e}")
            # 返回一个默认的黑色图像
            return Image.new('RGB', (224, 224), color='black')

    def _load_cached_image(self, idx: int) -> Image.Image:
        """加载缓存的图像数据"""
        image_path = self.image_paths[idx]
        cache_filename = self._get_cache_filename(image_path)
        cache_filepath = self.cache_dir / cache_filename

        # 如果缓存存在且文件未修改，加载缓存
        if cache_filepath.exists():
            try:
                # 检查原文件修改时间
                original_mtime = os.path.getmtime(image_path)
                cache_mtime = os.path.getmtime(cache_filepath)

                if cache_mtime >= original_mtime:
                    with open(cache_filepath, 'rb') as f:
                        return pickle.load(f)
            except:
                pass  # 缓存损坏，重新加载

        # 加载原始图像并缓存
        image = self._load_image(idx)
        try:
            with open(cache_filepath, 'wb') as f:
                pickle.dump(image, f)
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")

        return image

    def _get_cache_filename(self, image_path: str) -> str:
        """生成缓存文件名"""
        # 使用文件路径的哈希作为缓存文件名
        path_hash = hashlib.md5(image_path.encode()).hexdigest()
        return f"image_{path_hash}.pkl"

    def _get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_sample_weights(self) -> torch.Tensor:
        """计算样本权重（用于处理类别不平衡）"""
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)

        # 计算类别权重（逆频率）
        class_weights = total_samples / (len(class_counts) * class_counts)

        # 为每个样本分配权重
        sample_weights = [class_weights[label] for label in self.labels]

        return torch.tensor(sample_weights, dtype=torch.float32)


class STORKDataLoader:
    """STORK数据加载器管理类"""

    def __init__(self, config: Dict):
        """
        初始化数据加载器

        Args:
            config: 数据配置
        """
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.img_size = config['data']['img_size']
        self.train_split = config['data']['train_split']
        self.use_cache = config['data']['use_cache']
        self.num_workers = config['data']['num_workers']

        # 加载数据路径和标签
        self.image_paths, self.labels, self.qualities = self._load_data_info()

        print(f"📁 数据加载器初始化:")
        print(f"   数据目录: {self.data_dir}")
        print(f"   总样本数: {len(self.image_paths)}")
        print(f"   训练比例: {self.train_split}")

    def _load_data_info(self) -> Tuple[List[str], List[int], List[float]]:
        """加载数据路径和标签信息"""
        image_paths = []
        labels = []
        qualities = []

        data_dir = Path(self.data_dir)

        # 假设数据结构为: data_dir/class_name/images
        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_label = self._get_class_label(class_dir.name)
            if class_label is None:
                continue

            # 遍历类别目录下的图像
            for image_file in class_dir.glob('*'):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(str(image_file))
                    labels.append(class_label)

                    # 尝试从文件名解析质量分数
                    quality = self._parse_quality_from_filename(image_file.name)
                    qualities.append(quality)

        return image_paths, labels, qualities

    def _get_class_label(self, class_name: str) -> Optional[int]:
        """从类别名称获取标签"""
        # 根据你的数据集结构调整
        class_mapping = {
            'good': 1,      # 好质量胚胎
            'poor': 0,      # 差质量胚胎
            'high': 1,      # 高质量
            'low': 0,       # 低质量
            '1': 1,
            '0': 0
        }

        class_name_lower = class_name.lower()
        return class_mapping.get(class_name_lower, None)

    def _parse_quality_from_filename(self, filename: str) -> float:
        """从文件名解析质量分数"""
        # 这里可以根据实际的文件命名规则来解析质量分数
        # 示例: "embryo_001_quality_0.85.jpg"
        try:
            # 简单的解析逻辑，可以根据需要调整
            if 'quality' in filename.lower():
                parts = filename.split('_')
                for part in parts:
                    try:
                        if 0 <= float(part) <= 1:
                            return float(part)
                    except ValueError:
                        continue
        except:
            pass

        return 0.5  # 默认质量分数

    def create_data_loaders(self,
                           train_transform: Optional[Callable] = None,
                           val_transform: Optional[Callable] = None,
                           test_size: Optional[float] = None) -> Tuple[torch.utils.data.DataLoader,
                                                                     Optional[torch.utils.data.DataLoader],
                                                                     Optional[torch.utils.data.DataLoader]]:
        """
        创建训练、验证和测试数据加载器

        Args:
            train_transform: 训练数据变换
            val_transform: 验证数据变换
            test_size: 测试集比例（如果为None，则按照配置的train_split划分）

        Returns:
            (train_loader, val_loader, test_loader)
        """
        # 数据分割
        if test_size is None:
            # 使用论文级70/30分割
            test_size = 1.0 - self.train_split

        # 分割数据
        train_paths, test_paths, train_labels, test_labels, train_qualities, test_qualities = train_test_split(
            self.image_paths, self.labels, self.qualities,
            test_size=test_size,
            stratify=self.labels,
            random_state=42
        )

        # 进一步分割训练集为训练和验证
        val_config = self.config.get('validation', {})
        if val_config.get('cross_validation', False):
            # 交叉验证模式：不分割验证集
            val_paths, val_labels, val_qualities = None, None, None
        else:
            val_split = val_config.get('validation_split', 0.2)
            if val_split > 0:
                train_paths, val_paths, train_labels, val_labels, train_qualities, val_qualities = train_test_split(
                    train_paths, train_labels, train_qualities,
                    test_size=val_split,
                    stratify=train_labels,
                    random_state=42
                )
            else:
                val_paths, val_labels, val_qualities = None, None, None

        # 创建数据集
        train_dataset = EmbryoDataset(
            train_paths, train_labels, train_qualities,
            transform=train_transform,
            use_cache=self.use_cache,
            cache_dir=self.config['paths']['cache_dir']
        )

        val_dataset = None
        if val_paths is not None:
            val_dataset = EmbryoDataset(
                val_paths, val_labels, val_qualities,
                transform=val_transform,
                use_cache=self.use_cache,
                cache_dir=self.config['paths']['cache_dir']
            )

        test_dataset = EmbryoDataset(
            test_paths, test_labels, test_qualities,
            transform=val_transform,  # 测试集使用验证变换
            use_cache=self.use_cache,
            cache_dir=self.config['paths']['cache_dir']
        )

        # 创建数据加载器
        batch_size = self.config['training']['batch_size']

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        print(f"📊 数据加载器创建完成:")
        print(f"   训练集: {len(train_dataset)} 样本")
        if val_dataset:
            print(f"   验证集: {len(val_dataset)} 样本")
        print(f"   测试集: {len(test_dataset)} 样本")

        return train_loader, val_loader, test_loader