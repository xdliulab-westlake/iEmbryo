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
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict


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


class OfflineImagePreprocessor:
    """
    离线预处理所有图像以加速训练
    将确定性预处理步骤提前完成，保存为 pickle 格式
    在 __getitem__ 中只需加载预处理好的图像，显著减少 CPU 开销
    """

    def __init__(self, image_size=224, verbose=True):
        """
        Args:
            image_size: 目标图像大小
            verbose: 是否打印进度信息
        """
        self.image_size = image_size
        self.verbose = verbose
        self.preprocessor = ImagePreprocessor()
        self.processed_count = 0
        self.failed_count = 0

    def preprocess_image(self, image_path, apply_petri_crop=True, apply_center_crop=True):
        """
        预处理单张图像（确定性操作）

        Args:
            image_path: 图像路径
            apply_petri_crop: 是否应用 Petri 培养皿检测
            apply_center_crop: 是否应用中心裁剪

        Returns:
            PIL Image 对象（保存为 pickle 后可快速加载）
        """
        try:
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                img = Image.open(image_path).convert('RGB')
                img = np.array(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 应用确定性预处理
            if apply_petri_crop:
                img = self.preprocessor.crop_petri_dish(img)

            if apply_center_crop:
                img = self.preprocessor.crop_center_square(img)

            # 调整大小
            img = cv2.resize(img, (self.image_size, self.image_size))

            # 转为 PIL Image（便于加载时应用数据增强）
            img_pil = Image.fromarray(img.astype(np.uint8))

            return img_pil

        except Exception as e:
            if self.verbose:
                print(f"  Failed to preprocess {image_path}: {e}")
            return None

    def preprocess_multi_focal_dataset(
        self,
        multi_focal_df,
        output_dir='preprocessed_data',
        apply_petri_crop=True,
        apply_center_crop=True,
        num_processes=None,
        train_ratio=0.7,
        val_ratio=0.15,
        split_by_embryo=True
    ):
        """
        Offline preprocess multi-focal dataset images and split into train/val/test sets
        预处理多焦距胚胎图像数据，并按比例分开保存为训练/验证/测试集

        Args:
            multi_focal_df: DataFrame with 'EmbryoID', 'TargetHour', 'FullPath' columns
            output_dir: Output directory
            apply_petri_crop: Whether to apply Petri dish detection (enabled by default)
            apply_center_crop: Whether to apply center square crop
            num_processes: Number of processes for parallel processing (default: use all available cores)
            train_ratio: Training set ratio (default 0.7)
            val_ratio: Validation set ratio (default 0.15)
            split_by_embryo: If True, split by embryo ID; if False, split by image (default True)

        Returns:
            (output_dir, processed_count, failed_count)
        """
        from multiprocessing import Pool
        from sklearn.model_selection import train_test_split
        import os

        # 如果未指定进程数，使用所有可用的 CPU 核心
        if num_processes is None:
            num_processes = os.cpu_count()

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        test_ratio = 1.0 - train_ratio - val_ratio

        # Step 1: 按比例分割胚胎ID (确保同一胚胎的所有图像都在同一集合中)
        if split_by_embryo:
            embryo_ids = sorted(set(multi_focal_df['EmbryoID']))
            train_embryos, temp_embryos = train_test_split(
                embryo_ids,
                test_size=(val_ratio + test_ratio),
                random_state=42
            )
            val_embryos, test_embryos = train_test_split(
                temp_embryos,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=42
            )

            # 根据胚胎ID给行分类
            train_df = multi_focal_df[multi_focal_df['EmbryoID'].isin(train_embryos)]
            val_df = multi_focal_df[multi_focal_df['EmbryoID'].isin(val_embryos)]
            test_df = multi_focal_df[multi_focal_df['EmbryoID'].isin(test_embryos)]
        else:
            # 按图像分割
            indices = list(range(len(multi_focal_df)))
            train_indices, temp_indices = train_test_split(
                indices,
                test_size=(val_ratio + test_ratio),
                random_state=42
            )
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=42
            )
            train_df = multi_focal_df.iloc[train_indices]
            val_df = multi_focal_df.iloc[val_indices]
            test_df = multi_focal_df.iloc[test_indices]

        total_images = len(multi_focal_df)
        self.processed_count = 0
        self.failed_count = 0

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"OFFLINE IMAGE PREPROCESSING - Multi-Focal Dataset (Split)")
            print(f"{'='*70}")
            print(f"Total images to process: {total_images}")
            print(f"Output directory: {output_dir}")
            print(f"Processes: {num_processes} (using all available CPU cores)")
            print(f"Petri crop: {apply_petri_crop}")
            print(f"Center crop: {apply_center_crop}")
            print(f"Storage: Three pkl files (train/val/test)")
            print(f"Split method: By {'Embryo ID' if split_by_embryo else 'Image'}")
            print(f"Train: {len(train_df)} images ({100*len(train_df)/total_images:.1f}%)")
            print(f"Val:   {len(val_df)} images ({100*len(val_df)/total_images:.1f}%)")
            print(f"Test:  {len(test_df)} images ({100*len(test_df)/total_images:.1f}%)\n")

        # Step 2: 并行处理所有数据
        all_dfs = [
            ('train', train_df),
            ('val', val_df),
            ('test', test_df)
        ]

        for split_name, split_df in all_dfs:
            if len(split_df) == 0:
                if self.verbose:
                    print(f"Skipping {split_name} split (empty)")
                continue

            if self.verbose:
                print(f"Processing {split_name} set ({len(split_df)} images)...")

            # Prepare data for parallel processing
            rows_data = [
                (row['EmbryoID'], int(row['TargetHour']), row['FullPath'],
                 apply_petri_crop, apply_center_crop)
                for _, row in split_df.iterrows()
            ]

            # Parallel processing
            with Pool(num_processes) as pool:
                # 使用较小的 chunksize 以获得更频繁的进度条更新
                # 128 个进程 * 50 chunk = 6400 任务/批次更新
                results = tqdm(
                    pool.imap_unordered(
                        self._preprocess_single_image_static,
                        rows_data,
                        chunksize=50  # 减小到 50 以获得更频繁的进度更新
                    ),
                    total=len(rows_data),
                    disable=not self.verbose,
                    desc=f"Preprocessing {split_name}",
                    ncols=100,
                    mininterval=1.0  # 最少每 1 秒更新一次
                )

                # 收集预处理后的图像
                preprocessed_data = {}
                split_processed = 0
                split_failed = 0
                for result in results:
                    if result is not None:
                        embryo_id, target_hour, img_pil = result
                        key = f"{embryo_id}_{target_hour:03d}"
                        preprocessed_data[key] = img_pil
                        split_processed += 1
                    else:
                        split_failed += 1

                self.processed_count += split_processed
                self.failed_count += split_failed

                # 保存该分割的数据到单个 pkl 文件
                pkl_path = output_dir / f'preprocessed_images_{split_name}.pkl'
                if self.verbose:
                    print(f"Saving {len(preprocessed_data)} {split_name} images...")

                with open(pkl_path, 'wb') as f:
                    pickle.dump(preprocessed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                if self.verbose:
                    print(f"{split_name.upper()} split: {split_processed} successful, {split_failed} failed")
                    print(f"File size: {pkl_path.stat().st_size / (1024**3):.2f} GB\n")

        if self.verbose:
            print(f"{'='*70}")
            print(f"PREPROCESSING COMPLETED")
            print(f"{'='*70}")
            print(f"Total successfully processed: {self.processed_count}/{total_images}")
            print(f"Total failed: {self.failed_count}/{total_images}")
            print(f"Success rate: {100*self.processed_count/total_images:.1f}%")
            print(f"Output location: {output_dir}")
            print(f"Files: preprocessed_images_train.pkl, preprocessed_images_val.pkl, preprocessed_images_test.pkl")
            print(f"{'='*70}\n")

        return output_dir, self.processed_count, self.failed_count

    @staticmethod
    def _preprocess_single_image_static(args):
        """
        Static method for parallel processing (required for multiprocessing)
        返回预处理后的图像而不是直接保存，所有图像将在主进程中一次性保存到单个pkl文件

        Args:
            args: (embryo_id, target_hour, full_path, apply_petri_crop, apply_center_crop)

        Returns:
            (embryo_id, target_hour, img_pil) tuple if successful, None otherwise
        """
        embryo_id, target_hour, full_path, apply_petri_crop, apply_center_crop = args
        preprocessor = ImagePreprocessor()

        try:
            # Read image
            img = cv2.imread(str(full_path))
            if img is None:
                img = Image.open(full_path).convert('RGB')
                img = np.array(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply preprocessing
            if apply_petri_crop:
                img = preprocessor.crop_petri_dish(img)

            if apply_center_crop:
                img = preprocessor.crop_center_square(img)

            # Resize to standard size
            img = cv2.resize(img, (224, 224))

            # Convert to PIL Image
            img_pil = Image.fromarray(img.astype(np.uint8))

            # 返回处理后的图像，而不是直接保存
            return (embryo_id, target_hour, img_pil)

        except Exception as e:
            return None

    def check_preprocessed_data(self, output_dir, multi_focal_df):
        """
        Check integrity of preprocessed data
        检查三个 pkl 文件中是否包含所有必需的图像

        Args:
            output_dir: Preprocessed data output directory
            multi_focal_df: Original DataFrame

        Returns:
            (is_complete, missing_count) tuple
        """
        output_dir = Path(output_dir)
        splits = ['train', 'val', 'test']
        all_preprocessed_data = {}

        # 尝试加载所有三个 pkl 文件
        for split in splits:
            pkl_path = output_dir / f'preprocessed_images_{split}.pkl'

            # 如果 pkl 文件不存在，尝试加载旧的单一 pkl 格式（向后兼容）
            if not pkl_path.exists():
                if split == 'train':
                    # 尝试加载旧格式的单个 pkl 文件
                    old_pkl_path = output_dir / 'preprocessed_images.pkl'
                    if old_pkl_path.exists():
                        try:
                            with open(old_pkl_path, 'rb') as f:
                                all_preprocessed_data.update(pickle.load(f))
                        except Exception as e:
                            return False, len(multi_focal_df)
                        continue
                return False, len(multi_focal_df)

            # 加载当前分割的 pkl 文件
            try:
                with open(pkl_path, 'rb') as f:
                    all_preprocessed_data.update(pickle.load(f))
            except Exception as e:
                return False, len(multi_focal_df)

        # 检查是否包含所有必需的图像
        missing_count = 0
        for _, row in multi_focal_df.iterrows():
            embryo_id = row['EmbryoID']
            target_hour = int(row['TargetHour'])
            key = f"{embryo_id}_{target_hour:03d}"
            if key not in all_preprocessed_data:
                missing_count += 1

        is_complete = missing_count == 0
        return is_complete, missing_count


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
    从selected_image_paths.csv加载，支持完整性检查和离线预处理加载

    使用模式：
    1. 从原始图像加载：设置 preprocessed_dir=None
    2. 从预处理图像加载：设置 preprocessed_dir='preprocessed_data'
    """
    def __init__(
        self,
        multi_focal_df,
        image_size=224,
        apply_petri_crop=True,
        apply_center_crop=True,
        transform=None,
        target_timepoints=33,
        preprocessed_dir=None,
        cache_size=1000
    ):
        """
        Args:
            multi_focal_df: DataFrame from selected_image_paths.csv (already validated)
            image_size: Target image size
            apply_petri_crop: Whether to apply Petri dish detection
            apply_center_crop: Whether to apply center square crop
            transform: Image transforms
            target_timepoints: Expected number of timepoints
            preprocessed_dir: 预处理数据目录。如果为 None，从原始图像加载；否则从预处理图像加载
            cache_size: LRU 缓存大小（防止内存溢出，默认 1000 张图像）
        """
        self.multi_focal_df = multi_focal_df
        self.image_size = image_size
        self.apply_petri_crop = apply_petri_crop
        self.apply_center_crop = apply_center_crop
        self.transform = transform
        self.target_timepoints = target_timepoints
        self.preprocessor = ImagePreprocessor()
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        # 预加载 pkl 文件到内存（在 DataLoader worker 中）
        # 这样每次加载图像都是 O(1) 字典查找，而不是每次都打开文件
        self._preprocessed_data_full = {}
        if self.preprocessed_dir:
            self._preload_preprocessed_data()

        # LRU 缓存：作为后备方案（如果预加载失败）
        self._image_cache = OrderedDict()
        self.cache_size = cache_size

        # Performance optimization: Pre-load embryo index using Dict instead of querying DataFrame each time
        self.embryo_ids = sorted(set(multi_focal_df['EmbryoID']))
        self.embryo_index = self._build_embryo_index()

    def _preload_preprocessed_data(self):
        """
        预加载所有预处理的 pkl 文件到内存
        这样 __getitem__ 中加载图像就是 O(1) 字典查找，而不是每次都打开文件
        """
        try:
            # 尝试加载分割式 pkl 文件 (train/val/test)
            splits = ['train', 'val', 'test']
            for split in splits:
                pkl_path = self.preprocessed_dir / f'preprocessed_images_{split}.pkl'
                if pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        split_data = pickle.load(f)
                        self._preprocessed_data_full.update(split_data)

            # 如果没找到分割式文件，尝试加载向后兼容的单一 pkl 文件
            if not self._preprocessed_data_full:
                pkl_path = self.preprocessed_dir / 'preprocessed_images.pkl'
                if pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        self._preprocessed_data_full = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to preload preprocessed data: {e}")
            self._preprocessed_data_full = {}

    def _build_embryo_index(self):
        """
        构建胚胎ID到其所有行的映射（Dict 而不是 DataFrame）
        这样 __getitem__ 中就不用每次都进行 DataFrame 过滤了
        """
        index = {}
        for embryo_id in self.embryo_ids:
            # 一次性获取该胚胎的所有行，转为列表（而不是保持为 DataFrame）
            embryo_rows = self.multi_focal_df[
                self.multi_focal_df['EmbryoID'] == embryo_id
            ][['TargetHour', 'FullPath']].values.tolist()
            index[embryo_id] = embryo_rows
        return index

    def __len__(self):
        return len(self.embryo_ids)

    def __getitem__(self, idx):
        """
        Returns time series with focal planes preserved for a single embryo

        Processing:
        1. Group images by TargetHour (timepoint)
        2. For each timepoint, keep all focal planes (up to 11)
        3. Return: (33, 11, 3, H, W) - 33 timepoints × 11 focal planes
           embryo_id: str - Embryo identifier

        Returns:
            time_series: (33, 11, 3, 224, 224) - Time series preserving focal plane dimension
            embryo_id: str - Embryo identifier
        """
        embryo_id = self.embryo_ids[idx]
        # Use Dict lookup instead of DataFrame filtering (O(1) vs O(n))
        embryo_records = self.embryo_index[embryo_id]

        # Group by TargetHour to get all focal planes for each timepoint
        timepoint_images = {}  # {TargetHour: [image_tensors for different focal planes]}

        for target_hour, full_path in embryo_records:
            try:
                # Load from preprocessed data or original images
                if self.preprocessed_dir is not None:
                    image = self._load_preprocessed_image(embryo_id, target_hour)
                else:
                    image = self._load_image(full_path)

                if target_hour not in timepoint_images:
                    timepoint_images[target_hour] = []

                timepoint_images[target_hour].append(image)

            except Exception as e:
                print(f"Failed to load image for embryo {embryo_id} (t={target_hour}): {e}")
                continue

        # Sort by timepoint and preserve focal planes
        sorted_timepoints = sorted(timepoint_images.keys())
        time_series = []

        for target_hour in sorted_timepoints:
            focal_images_pil = timepoint_images[target_hour]

            if not focal_images_pil:
                # Add black focal plane stack if no images for this timepoint
                time_series.append(torch.zeros(11, 3, self.image_size, self.image_size))
            else:
                # Apply transform and convert PIL images to tensors
                focal_images_tensor = []
                for img_pil in focal_images_pil:
                    if self.transform:
                        img_tensor = self.transform(img_pil)
                    else:
                        img_tensor = transforms.ToTensor()(img_pil)
                    focal_images_tensor.append(img_tensor)

                # Pad or truncate focal planes to 11
                num_focal_planes = len(focal_images_tensor)
                if num_focal_planes < 11:
                    # Pad with black images
                    padding = [torch.zeros(3, self.image_size, self.image_size)
                              for _ in range(11 - num_focal_planes)]
                    focal_images_tensor = focal_images_tensor + padding
                elif num_focal_planes > 11:
                    # Truncate to 11
                    focal_images_tensor = focal_images_tensor[:11]

                # Stack focal planes: (11, 3, H, W)
                stacked = torch.stack(focal_images_tensor)
                time_series.append(stacked)

        if not time_series:
            # Return black time series if all loading failed
            time_series = [torch.zeros(11, 3, self.image_size, self.image_size)
                          for _ in range(self.target_timepoints)]

        # Stack to (T, 11, 3, H, W)
        time_series_tensor = torch.stack(time_series)  # (num_timepoints, 11, 3, H, W)

        # Pad or truncate to target_timepoints if needed
        if time_series_tensor.shape[0] < self.target_timepoints:
            # Pad with black focal plane stacks
            padding = torch.zeros(
                self.target_timepoints - time_series_tensor.shape[0],
                11, 3, self.image_size, self.image_size
            )
            time_series_tensor = torch.cat([time_series_tensor, padding], dim=0)
        elif time_series_tensor.shape[0] > self.target_timepoints:
            # Truncate (shouldn't happen if validation works correctly)
            time_series_tensor = time_series_tensor[:self.target_timepoints]

        return time_series_tensor, embryo_id

    def _load_preprocessed_image(self, embryo_id, target_hour):
        """
        Load image from preprocessed pickle files (preloaded into memory)
        从预加载的预处理图像中加载（O(1) 字典查找，极快）

        Args:
            embryo_id: Embryo ID
            target_hour: Time point

        Returns:
            PIL Image (transform will be applied in __getitem__)
        """
        try:
            key = f"{embryo_id}_{int(target_hour):03d}"

            # 优先从预加载的数据中查找（O(1) 操作，极快）
            if self._preprocessed_data_full and key in self._preprocessed_data_full:
                return self._preprocessed_data_full[key]

            # 如果预加载失败或图像不在预加载的数据中，返回黑色图像
            return Image.new('RGB', (self.image_size, self.image_size))

        except Exception as e:
            print(f"Failed to load preprocessed image {embryo_id}_{int(target_hour):03d}: {e}")
            return Image.new('RGB', (self.image_size, self.image_size))

    def _load_image(self, image_path):
        """
        Load and preprocess image from file (for cases without preprocessed data)

        Returns:
            PIL Image (transform will be applied in __getitem__)
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

            # Convert to PIL Image (transform will be applied in __getitem__)
            img_pil = Image.fromarray(img.astype(np.uint8))

            return img_pil

        except Exception as e:
            return Image.new('RGB', (self.image_size, self.image_size))


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
    创建数据加载器（训练集使用下采样平衡）

    Args:
        data_root: 数据根目录
        config: 配置字典 (包含score_dict_path, 若为BiLSTM风格数据)

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import Subset
    import random

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

    # 不进行下采样平衡，保留原始数据分布以使用加权损失处理类别不平衡
    train_indices = train_dataset.indices
    train_labels = [full_dataset.labels[i] for i in train_indices]

    poor_count = sum(1 for label in train_labels if label == 0)
    good_count = sum(1 for label in train_labels if label == 1)

    print(f"\nTraining set: {poor_count} Poor + {good_count} Good (原始分布，保留所有样本)")
    print(f"  注：将使用加权CrossEntropyLoss处理类别不平衡")

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


def multi_focal_collate_fn(batch, embryo_to_label):
    """
    Custom collate function for multi-focal data
    Converts embryo_id to label tensor

    Args:
        batch: List of (time_series, embryo_id) tuples
        embryo_to_label: Dict mapping embryo_id to label

    Returns:
        (images_tensor, labels_tensor)
    """
    images = []
    labels = []

    for image_tensor, embryo_id in batch:
        images.append(image_tensor)
        # Get label from embryo_id
        label = embryo_to_label.get(embryo_id, 0)
        labels.append(label)

    return torch.stack(images), torch.tensor(labels, dtype=torch.float32)


def create_multi_focal_data_loaders(multi_focal_df, embryo_to_label, config, enable_preprocessing=True):
    """
    Create data loaders for multi-focal multi-timepoint data
    支持自动离线预处理所有图像以加速数据加载

    Args:
        multi_focal_df: DataFrame from selected_image_paths.csv (already validated)
        embryo_to_label: Dictionary mapping embryo_id to quality label
        config: Configuration dictionary
        enable_preprocessing: 是否启用自动离线预处理（默认 True）

    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_tensor)
    """
    import torch
    from sklearn.model_selection import train_test_split
    import os

    # Step 1: Auto preprocessing (if needed)
    preprocessed_dir = None
    if enable_preprocessing:
        # Use absolute path: project_root / preprocessed_data
        # project_root is the directory where main.py is located
        project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
        preprocessed_dir = project_root / 'preprocessed_data'

        # 检查预处理数据是否完整
        preprocessor = OfflineImagePreprocessor(
            image_size=config.get('image_size', 224),
            verbose=True
        )

        is_complete, missing_count = preprocessor.check_preprocessed_data(
            preprocessed_dir, multi_focal_df
        )

        if not is_complete:
            print(f"\nPreprocessed data incomplete ({missing_count} images missing)")
            print(f"Starting offline preprocessing...\n")

            # Execute offline preprocessing
            preprocessed_dir, processed_count, failed_count = preprocessor.preprocess_multi_focal_dataset(
                multi_focal_df,
                output_dir=str(preprocessed_dir),
                apply_petri_crop=config.get('apply_petri_crop', True),
                apply_center_crop=config.get('apply_center_crop', True)
            )
        else:
            print(f"\nPreprocessed data is complete, using cached preprocessed images\n")

    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    # Step 2: Create datasets (specify preprocessed directory)
    # Create dataset for training (with augmentation)
    # 创建数据集（自动从所有预处理文件加载）
    train_dataset_base = MultiFocalEmbryoDataset(
        multi_focal_df=multi_focal_df,
        image_size=config.get('image_size', 224),
        apply_petri_crop=config.get('apply_petri_crop', True),
        apply_center_crop=config.get('apply_center_crop', True),
        transform=train_transform,
        target_timepoints=config.get('time_steps', 33),
        preprocessed_dir=str(preprocessed_dir) if preprocessed_dir else None
    )

    # 使用同一数据集对象处理 val 和 test（它们共享缓存）
    val_test_dataset_base = MultiFocalEmbryoDataset(
        multi_focal_df=multi_focal_df,
        image_size=config.get('image_size', 224),
        apply_petri_crop=config.get('apply_petri_crop', True),
        apply_center_crop=config.get('apply_center_crop', True),
        transform=val_transform,
        target_timepoints=config.get('time_steps', 33),
        preprocessed_dir=str(preprocessed_dir) if preprocessed_dir else None
    )

    # Use base dataset for getting embryo IDs and labels
    full_dataset = train_dataset_base

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

    # 不进行下采样平衡，保留原始数据分布以使用加权损失处理类别不平衡
    import random
    train_labels = labels[train_indices].numpy()
    train_indices_list = list(train_indices)

    poor_count = sum(1 for label in train_labels if label == 0)
    good_count = sum(1 for label in train_labels if label == 1)

    print(f"Training set: {poor_count} Poor + {good_count} Good (原始分布，保留所有样本)")
    print(f"  注：将使用加权CrossEntropyLoss处理类别不平衡")

    from torch.utils.data import Subset

    # Create subsets using appropriate datasets (训练集保留所有样本)
    train_dataset = Subset(train_dataset_base, train_indices_list)
    val_dataset = Subset(val_test_dataset_base, val_indices)
    test_dataset = Subset(val_test_dataset_base, test_indices)

    # Create custom collate function with embryo_to_label
    from functools import partial
    collate_fn = partial(multi_focal_collate_fn, embryo_to_label=embryo_to_label)

    # Step 3: Create DataLoaders (with optimized parameters)
    # Use num_workers and pin_memory from config, with better defaults
    num_workers = config.get('num_workers', 4)  # Default 4 workers
    pin_memory = config.get('pin_memory', True)  # Default True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    print(f"\nData split summary:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")

    return train_loader, val_loader, test_loader, labels
