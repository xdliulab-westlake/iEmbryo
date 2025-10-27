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

    支持三种数据结构:

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

    方式3 (多焦距多时间点):
    通过 selected_image_paths.csv 加载
    image/
        embryo_id/
            images/
                F-15/
                    embryo_id_RUN*.jpeg
                F0/
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

                # 检查是否为嵌套目录结构 (image/embryo_id/images/...)
                # 还是平面结构 (image/embryo_001.jpg)
                first_item = next(image_dir.iterdir(), None)
                if first_item and first_item.is_dir():
                    # 嵌套结构: image/embryo_id/images/F.../xxx.jpg
                    return self._load_nested_dataset(image_dir, score_dict)
                else:
                    # 平面结构: image/embryo_001.jpg
                    return self._load_flat_dataset(image_dir, score_dict)

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

    def _load_flat_dataset(self, image_dir, score_dict):
        """加载平面结构 (image/embryo_001.jpg)"""
        embryo_paths = []
        labels = []

        # 支持所有图像扩展名，并匹配不含扩展名的 score_dict key
        for image_file in image_dir.glob('*'):
            if not image_file.is_file():
                continue

            # 使用 stem 获取不含扩展名的文件名
            image_name_without_ext = image_file.stem

            if image_name_without_ext in score_dict:
                quality = score_dict[image_name_without_ext].get('quality', 0)
                embryo_paths.append(image_file)
                labels.append(quality)

        return embryo_paths, labels

    def _load_nested_dataset(self, image_dir, score_dict):
        """加载嵌套结构 (image/embryo_id/images/focal_plane/xxx.jpg)"""
        embryo_paths = []
        labels = []

        # 遍历胚胎ID目录
        for embryo_dir in image_dir.iterdir():
            if not embryo_dir.is_dir():
                continue

            # 胚胎ID就是目录名
            embryo_id = embryo_dir.name

            # 检查评分字典中是否有这个胚胎ID
            if embryo_id in score_dict:
                quality = score_dict[embryo_id].get('quality', 0)
                embryo_paths.append(embryo_dir)
                labels.append(quality)

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
            image: (3, H, W) - BiLSTM风格单张图像或嵌套结构聚合图像
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

        # 检查是否为嵌套结构 (image/embryo_id/images/F-xx/)
        images_subdir = embryo_path / 'images'
        if images_subdir.exists():
            return self._load_nested_structure(images_subdir, label)

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

    def _load_nested_structure(self, images_dir, label):
        """
        Load images from nested structure (images/F-xx/ with multiple RUN images)
        Aggregates focal planes into a single representative image.

        Args:
            images_dir: Path to 'images' directory containing F-xx subdirectories
            label: Quality label

        Returns:
            image: (3, H, W) - Aggregated representative image
            label: Quality label tensor
        """
        focal_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])

        if not focal_dirs:
            return torch.zeros(3, self.image_size, self.image_size), torch.tensor(label, dtype=torch.float32)

        # Collect images from all focal planes
        all_images = []
        for focal_dir in focal_dirs:
            # Find image files (support different extensions)
            image_files = (list(focal_dir.glob('*.jpg')) +
                          list(focal_dir.glob('*.jpeg')) +
                          list(focal_dir.glob('*.png')) +
                          list(focal_dir.glob('*.JPG')) +
                          list(focal_dir.glob('*.JPEG')) +
                          list(focal_dir.glob('*.PNG')))

            if image_files:
                # Use the first RUN image from this focal plane
                image = self._load_image(image_files[0])
                all_images.append(image)

        if not all_images:
            return torch.zeros(3, self.image_size, self.image_size), torch.tensor(label, dtype=torch.float32)

        # Aggregate: Average across all focal planes
        stacked = torch.stack(all_images)  # (num_focal_planes, 3, H, W)
        aggregated_image = torch.mean(stacked, dim=0)  # (3, H, W)

        return aggregated_image, torch.tensor(label, dtype=torch.float32)


class MultiFocalEmbryoDataset(Dataset):
    """
    多焦距多时间点胚胎数据集
    从selected_image_paths.csv加载，支持完整性检查
    """
    def __init__(
        self,
        multi_focal_df,
        image_size=224,
        apply_petri_crop=True,
        apply_center_crop=True,
        transform=None,
        target_timepoints=33
    ):
        """
        Args:
            multi_focal_df: DataFrame from selected_image_paths.csv (already validated)
            image_size: Target image size
            apply_petri_crop: Whether to apply Petri dish detection
            apply_center_crop: Whether to apply center square crop
            transform: Image transforms
            target_timepoints: Expected number of timepoints
        """
        self.multi_focal_df = multi_focal_df
        self.image_size = image_size
        self.apply_petri_crop = apply_petri_crop
        self.apply_center_crop = apply_center_crop
        self.transform = transform
        self.target_timepoints = target_timepoints
        self.preprocessor = ImagePreprocessor()

        # Create embryo index
        self.embryo_ids = sorted(set(multi_focal_df['EmbryoID']))
        self.embryo_index = {eid: i for i, eid in enumerate(self.embryo_ids)}

    def __len__(self):
        return len(self.embryo_ids)

    def __getitem__(self, idx):
        """
        Returns aggregated image from all focal planes for a single embryo

        Returns:
            image: (3, H, W) - Aggregated from all focal planes and timepoints
            embryo_id: str - Embryo identifier
        """
        embryo_id = self.embryo_ids[idx]
        embryo_df = self.multi_focal_df[self.multi_focal_df['EmbryoID'] == embryo_id]

        # Aggregate all images across timepoints and focal planes
        all_images = []

        for _, row in embryo_df.iterrows():
            full_path = row['FullPath']
            try:
                image = self._load_image(full_path)
                all_images.append(image)
            except Exception as e:
                print(f"Failed to load {full_path}: {e}")
                continue

        if not all_images:
            # Return black image if all loading failed
            return torch.zeros(3, self.image_size, self.image_size), embryo_id

        # Average across all loaded images
        stacked = torch.stack(all_images)  # (num_images, 3, H, W)
        aggregated_image = torch.mean(stacked, dim=0)  # (3, H, W)

        return aggregated_image, embryo_id

    def _load_image(self, image_path):
        """Load and preprocess a single image"""
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


def create_multi_focal_data_loaders(multi_focal_df, embryo_to_label, config):
    """
    Create data loaders for multi-focal multi-timepoint data

    Args:
        multi_focal_df: DataFrame from selected_image_paths.csv (already validated)
        embryo_to_label: Dictionary mapping embryo_id to quality label
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_tensor)
    """
    import torch
    from sklearn.model_selection import train_test_split

    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    # Create dataset
    full_dataset = MultiFocalEmbryoDataset(
        multi_focal_df=multi_focal_df,
        image_size=config.get('image_size', 224),
        apply_petri_crop=config.get('apply_petri_crop', True),
        apply_center_crop=config.get('apply_center_crop', True),
        transform=None,
        target_timepoints=config.get('time_steps', 33)
    )

    embryo_ids = full_dataset.embryo_ids
    labels = torch.tensor([embryo_to_label[eid] for eid in embryo_ids], dtype=torch.float32)

    # Train-val-test split
    total_size = len(full_dataset)
    train_ratio = config.get('train_ratio', 0.8)
    val_ratio = config.get('val_ratio', 0.1)
    test_ratio = 1.0 - train_ratio - val_ratio

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_indices, temp_indices = train_test_split(
        range(total_size),
        test_size=(val_size + test_size) / total_size,
        stratify=labels.numpy(),
        random_state=42
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_size / (val_size + test_size),
        stratify=labels[temp_indices].numpy(),
        random_state=42
    )

    from torch.utils.data import Subset

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )

    print(f"\nData split summary:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")

    return train_loader, val_loader, test_loader, labels
