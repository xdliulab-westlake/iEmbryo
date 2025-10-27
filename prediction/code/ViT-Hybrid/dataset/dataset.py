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
import cv2
import copy
from pathlib import Path


class ImagePreprocessor:
    """图像预处理工具"""

    @staticmethod
    def crop_center_square(frame):
        """裁剪中心正方形区域"""
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    @staticmethod
    def crop_petri_dish(frame):
        """使用Hough圆检测培养皿并裁剪"""
        if len(frame.shape) == 3:
            gimg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gimg = frame.copy()

        cimg = copy.deepcopy(frame)
        img_height, img_width = gimg.shape
        img_center_x, img_center_y = img_width // 2, img_height // 2

        gimg_blurred = cv2.GaussianBlur(gimg, (9, 9), 2)

        img_diagonal = np.sqrt(img_height ** 2 + img_width ** 2)
        min_radius = max(50, int(img_diagonal * 0.15))
        max_radius = min(int(img_diagonal * 0.45), max(img_height, img_width) // 2)

        param_combinations = [(60, 40), (70, 45), (50, 35), (80, 50), (90, 55)]
        all_circles = []

        for param1, param2 in param_combinations:
            circles = cv2.HoughCircles(
                gimg_blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(80, int(min(img_height, img_width) * 0.2)),
                param1=param1,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )

            if circles is not None and len(circles[0]) > 0:
                for circle in circles[0]:
                    x, y, r = circle
                    distance_to_center = np.sqrt((x - img_center_x) ** 2 + (y - img_center_y) ** 2)
                    all_circles.append({
                        'x': x, 'y': y, 'r': r,
                        'dist': distance_to_center
                    })

        if not all_circles:
            return frame

        best_circle = max(all_circles, key=lambda c: c['r'])
        x, y, r = int(best_circle['x']), int(best_circle['y']), int(best_circle['r'])

        mask = np.zeros_like(cimg)
        if len(cimg.shape) == 3:
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        else:
            cv2.circle(mask, (x, y), r, 255, -1)

        result = cv2.bitwise_and(cimg, mask)
        return result if result.any() else frame


class EmbryoDataset(Dataset):
    """
    胚胎数据集

    支持两种数据结构:

    方式1 (BiLSTM风格):
    data_root/
        image/
            embryo_001.jpg
            embryo_002.jpg
            ...
        score_dict.json (包含quality标签)

    方式2 (时间序列):
    data_root/
        good/
            embryo_001/
                time_000/
                    focal_00.jpg
                    ...
                time_032/
        poor/
            embryo_002/
                ...
    """
    def __init__(
        self,
        data_root,
        score_dict_path=None,
        time_steps=33,
        focal_planes=11,
        image_size=224,
        apply_petri_crop=True,
        apply_center_crop=True,
        transform=None
    ):
        self.data_root = Path(data_root)
        self.time_steps = time_steps
        self.focal_planes = focal_planes
        self.image_size = image_size
        self.apply_petri_crop = apply_petri_crop
        self.apply_center_crop = apply_center_crop
        self.transform = transform
        self.preprocessor = ImagePreprocessor()

        self.embryo_paths, self.labels = self._load_dataset(score_dict_path)

    def _load_dataset(self, score_dict_path=None):
        """加载数据集路径和标签"""
        embryo_paths = []
        labels = []

        # 方式1: BiLSTM风格 (单张图像+评分字典)
        image_dir = self.data_root / 'image'
        if image_dir.exists() and score_dict_path:
            import json
            score_dict_path = Path(score_dict_path)
            if not score_dict_path.is_absolute():
                score_dict_path = self.data_root / score_dict_path

            if score_dict_path.exists():
                with open(score_dict_path, 'r') as f:
                    score_dict = json.load(f)

                for image_file in image_dir.glob('*.jpg'):
                    image_name = image_file.name
                    if image_name in score_dict:
                        quality = score_dict[image_name].get('quality', 0)
                        embryo_paths.append(image_file)
                        labels.append(quality)
                return embryo_paths, labels

        # 方式2: 时间序列风格 (good/poor子目录)
        self.class_to_label = {'poor': 0, 'good': 1}
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
        """
        加载单张图像并进行预处理

        Args:
            image_path: 图像路径

        Returns:
            预处理后的图像张量
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                img = Image.open(image_path).convert('RGB')
                img = np.array(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.apply_petri_crop:
                img = self.preprocessor.crop_petri_dish(img)

            if self.apply_center_crop:
                img = self.preprocessor.crop_center_square(img)

            img = cv2.resize(img, (self.image_size, self.image_size))

            img = Image.fromarray(img.astype(np.uint8))

            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

            return img

        except Exception as e:
            return torch.zeros(3, self.image_size, self.image_size)

    def __len__(self):
        return len(self.embryo_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) - BiLSTM风格单张图像
            或
            sequence: (time_steps, focal_planes, 3, H, W) - 时间序列
            label: 0 or 1
        """
        embryo_path = self.embryo_paths[idx]
        label = self.labels[idx]

        # 方式1: BiLSTM风格 (单张图像)
        if embryo_path.is_file():
            image = self._load_image(embryo_path)
            return image, torch.tensor(label, dtype=torch.float32)

        # 方式2: 时间序列风格 (多时间点多焦平面)
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
        config: 配置字典 (包含score_dict_path, 若为BiLSTM风格数据)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    full_dataset = EmbryoDataset(
        data_root=data_root,
        score_dict_path=config.get('score_dict_path'),
        time_steps=config.get('time_steps', 33),
        focal_planes=config.get('focal_planes', 11),
        image_size=config.get('image_size', 224),
        apply_petri_crop=config.get('apply_petri_crop', True),
        apply_center_crop=config.get('apply_center_crop', True),
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
        pin_memory=config.get('pin_memory', True)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )

    return train_loader, val_loader, test_loader
