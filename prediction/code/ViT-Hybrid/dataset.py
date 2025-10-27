"""
Dataset and DataLoader for ViT-Hybrid Model (PyTorch)
Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class EmbryoDataset(Dataset):
    """
    胚胎数据集

    数据结构:
    data_root/
        good/
            embryo_001/
                time_000/
                    focal_00.jpg
                    ...
                    focal_10.jpg
                ...
        poor/
            embryo_002/
                ...
    """
    def __init__(
        self,
        data_root,
        time_steps=33,
        focal_planes=11,
        image_size=224,
        transform=None
    ):
        self.data_root = Path(data_root)
        self.time_steps = time_steps
        self.focal_planes = focal_planes
        self.image_size = image_size
        self.transform = transform

        self.class_to_label = {'poor': 0, 'good': 1}
        self.embryo_paths, self.labels = self._load_dataset()

    def _load_dataset(self):
        """加载数据集路径和标签"""
        embryo_paths = []
        labels = []

        for class_name in ['poor', 'good']:
            class_dir = self.data_root / class_name
            if not class_dir.exists():
                continue

            label = self.class_to_label[class_name]
            for embryo_dir in class_dir.iterdir():
                if embryo_dir.is_dir():
                    embryo_paths.append(embryo_dir)
                    labels.append(label)

        return embryo_paths, labels

    def _load_image(self, image_path):
        """加载单张图像"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            return img
        except:
            return torch.zeros(3, self.image_size, self.image_size)

    def __len__(self):
        return len(self.embryo_paths)

    def __getitem__(self, idx):
        """
        Returns:
            sequence: (time_steps, focal_planes, 3, H, W)
            label: 0 or 1
        """
        embryo_path = self.embryo_paths[idx]
        label = self.labels[idx]

        sequence = torch.zeros(
            self.time_steps,
            self.focal_planes,
            3,
            self.image_size,
            self.image_size
        )

        for t in range(self.time_steps):
            time_dir = embryo_path / f'time_{t:03d}'
            if not time_dir.exists():
                continue

            for f in range(self.focal_planes):
                focal_path = time_dir / f'focal_{f:02d}.jpg'
                if focal_path.exists():
                    sequence[t, f] = self._load_image(focal_path)

        return sequence, torch.tensor(label, dtype=torch.float32)


def get_transforms(augment=False):
    """获取数据变换"""
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


def create_data_loaders(data_root, config):
    """
    创建数据加载器

    Args:
        data_root: 数据根目录
        config: 配置字典

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    full_dataset = EmbryoDataset(
        data_root=data_root,
        time_steps=config.get('time_steps', 33),
        focal_planes=config.get('focal_planes', 11),
        image_size=config.get('image_size', 224),
        transform=None
    )

    total_size = len(full_dataset)
    train_size = int(total_size * config.get('train_ratio', 0.8))
    val_size = int(total_size * config.get('val_ratio', 0.1))
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
