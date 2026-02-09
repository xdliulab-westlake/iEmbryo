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

        print(f"数据集初始化完成:")
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
            print(f"加载图像失败: {image_path}, 错误: {e}")
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
            print(f"缓存保存失败: {e}")

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


# NOTE: STORKDataLoader has been moved to parallel_processing.py for better integration
# with preprocessing and caching mechanisms. This module focuses on the EmbryoDataset class
# for PyTorch data loading. Use:
#   from data import STORKDataLoader  # Imports from parallel_processing
# See data/parallel_processing.py:338 for the active implementation
