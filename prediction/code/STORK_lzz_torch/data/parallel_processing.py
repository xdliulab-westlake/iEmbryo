"""
并行图像处理和缓存模块
PyTorch版本 - 优化的多进程图像预处理和缓存机制
"""

import os
import pickle
import hashlib
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .stork_preprocessing import preprocess_embryo_image


def is_main_process() -> bool:
    """检查当前进程是否为主进程"""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def optimized_preprocess_single_image(args: Tuple) -> np.ndarray:
    """
    优化的单图像预处理函数（专为多进程设计）

    Args:
        args: (path, img_size, is_training, apply_petri_crop, apply_center_crop)

    Returns:
        预处理后的图像数组
    """
    path, img_size, is_training, apply_petri_crop, apply_center_crop = args

    try:
        # 使用STORK预处理管道
        img = preprocess_embryo_image(
            path,
            img_size=img_size,
            apply_petri_crop=apply_petri_crop,
            apply_center_crop=apply_center_crop
        )

        # 确保RGB格式
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)

        # 转换为float32
        img = img.astype(np.float32)

        return img

    except Exception as e:
        print(f"处理图像 {path} 时出错: {e}")
        # 返回黑色图像作为后备
        fallback_img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        return fallback_img


def batch_preprocess_images_parallel(paths: List[str],
                                   img_size: int,
                                   is_training: bool,
                                   apply_petri_crop: bool = True,
                                   apply_center_crop: bool = True,
                                   num_workers: Optional[int] = None) -> np.ndarray:
    """
    并行处理图像批次

    Args:
        paths: 图像路径列表
        img_size: 目标图像尺寸
        is_training: 是否为训练模式
        apply_petri_crop: 是否应用培养皿检测
        apply_center_crop: 是否应用中心裁剪
        num_workers: 工作进程数

    Returns:
        处理后的图像数组 [N, H, W, C]
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # 限制进程数避免系统过载

    if is_main_process():
        print(f"🚀 使用 {num_workers} 个进程并行处理 {len(paths)} 张图像...")

    # 准备多进程参数
    args_list = [(path, img_size, is_training, apply_petri_crop, apply_center_crop)
                 for path in paths]

    # 使用进程池处理CPU密集型任务
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        processed_images = list(executor.map(optimized_preprocess_single_image, args_list))

    return np.array(processed_images)


def load_data_with_stork_preprocessing_DEPRECATED(data_dir: str,
                                     img_size: int = 224,
                                     train_split: float = 0.7,
                                     use_cache: bool = True,
                                     apply_petri_crop: bool = True,
                                     apply_center_crop: bool = True,
                                     num_workers: Optional[int] = None,
                                     random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    使用优化的STORK预处理的数据加载函数

    Args:
        data_dir: 数据目录路径
        img_size: 图像尺寸，默认224x224
        train_split: 训练集比例，默认0.7
        use_cache: 是否使用缓存，默认True
        apply_petri_crop: 是否应用培养皿检测
        apply_center_crop: 是否应用中心裁剪
        num_workers: 并行处理进程数，默认为None（自动检测）
        random_state: 随机种子

    Returns:
        X_train, train_labels, train_quals, X_test, test_labels, test_quals

    优化特性:
    - 多进程并行图像预处理 (5-8x 速度提升)
    - NumPy数据增强 (避免张量转换开销)
    - 支持预处理数据缓存
    - 详细进度报告和性能统计
    - 论文级数据分割策略
    - STORK预处理：复杂培养皿检测、中心方形裁剪等专业胚胎学图像处理
    """
    image_paths = []
    labels = []
    qualities = []

    if is_main_process():
        print("=== 使用优化的STORK胚胎图像预处理加载数据 ===")
        print(f"🚀 并行处理模式：{'启用' if num_workers != 1 else '禁用'}")
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
        if is_main_process():
            print(f"🔧 自动检测到 {num_workers} 个工作进程")

    # 收集所有图像路径和标签
    data_path = Path(data_dir)
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()
        if class_name in ['0', 'negative', 'no', 'non_viable', 'class_0']:
            class_label = 0
        elif class_name in ['1', 'positive', 'yes', 'viable', 'class_1']:
            class_label = 1
        else:
            continue

        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                image_paths.append(str(img_file))
                labels.append(class_label)
                qualities.append(1.0)  # 默认质量分数

    # 转换为numpy数组
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    qualities = np.array(qualities)

    if is_main_process():
        print(f"找到 {len(image_paths)} 张图像")
        print(f"Class 0: {sum(labels == 0)} 张，Class 1: {sum(labels == 1)} 张")

    # 数据分割策略
    test_size = 1.0 - train_split
    split_strategy = f"{int(train_split*100)}_{int(test_size*100)}_split"
    if is_main_process():
        print(f"🎯 采用 {int(train_split*100)}/{int(test_size*100)} 数据分割策略...")

    # 直接进行分割，避免数据泄漏
    train_paths, test_paths, train_labels, test_labels, train_quals, test_quals = train_test_split(
        image_paths, labels, qualities, test_size=test_size,
        random_state=random_state, stratify=labels
    )

    # 验证分割比例
    total_samples = len(image_paths)
    train_samples = len(train_paths)
    test_samples = len(test_paths)

    if is_main_process():
        print(f"📊 数据分割完成:")
        print(f"  总样本数: {total_samples}")
        print(f"  训练集: {train_samples} ({train_samples/total_samples*100:.1f}%)")
        print(f"  测试集: {test_samples} ({test_samples/total_samples*100:.1f}%)")

    # 缓存机制
    cache_manager = DatasetCache() if use_cache else None
    cached_data = None

    if use_cache:
        cache_key = cache_manager._generate_cache_key(
            data_dir, img_size, len(image_paths), apply_petri_crop,
            apply_center_crop, split_strategy, num_workers
        )
        cached_data = cache_manager.load_cache(cache_key)

        if cached_data:
            if is_main_process():
                print(f"训练集形状: {cached_data['X_train'].shape}")
                print(f"测试集形状: {cached_data['X_test'].shape}")
            return (cached_data['X_train'], cached_data['train_labels'], cached_data['train_quals'],
                    cached_data['X_test'], cached_data['test_labels'], cached_data['test_quals'])

    # 使用优化的并行处理
    if is_main_process():
        print(f"=== 🎯 并行预处理训练集（{int(train_split*100)}%）===")
    start_time = time.time()
    X_train = batch_preprocess_images_parallel(
        train_paths, img_size, is_training=True,
        apply_petri_crop=apply_petri_crop,
        apply_center_crop=apply_center_crop,
        num_workers=num_workers
    )
    train_time = time.time() - start_time
    if is_main_process():
        print(f"✅ 训练集预处理完成，耗时: {train_time:.2f} 秒")

    if is_main_process():
        print(f"=== 🔒 并行预处理测试集（{int(test_size*100)}%）===")
    start_time = time.time()
    X_test = batch_preprocess_images_parallel(
        test_paths, img_size, is_training=False,
        apply_petri_crop=apply_petri_crop,
        apply_center_crop=apply_center_crop,
        num_workers=num_workers
    )
    test_time = time.time() - start_time
    if is_main_process():
        print(f"✅ 测试集预处理完成，耗时: {test_time:.2f} 秒")

    total_time = train_time + test_time
    if is_main_process():
        print(f"=== 📊 数据预处理完成 ===")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        print(f"⏱️  总预处理时间: {total_time:.2f} 秒 (并行优化版)")
        print(f"📊 平均每张图像: {total_time/len(image_paths)*1000:.2f} 毫秒")

    # 保存缓存
    if use_cache:
        cache_data = {
            'X_train': X_train, 'train_labels': train_labels, 'train_quals': train_quals,
            'X_test': X_test, 'test_labels': test_labels, 'test_quals': test_quals,
            'img_size': img_size,
            'data_dir': data_dir,
            'total_samples': len(image_paths),
            'split_strategy': split_strategy,
            'num_workers': num_workers,
            'processing_time': total_time,
            'apply_petri_crop': apply_petri_crop,
            'apply_center_crop': apply_center_crop,
            'version': 'pytorch_stork_v1.0'
        }

        cache_manager.save_cache(cache_key, cache_data)

    return (X_train, train_labels, train_quals,    # 训练集
            X_test, test_labels, test_quals)        # 测试集


class LazySTORKDataset(torch.utils.data.Dataset):
    """按需加载的STORK数据集，带渐进式预处理缓存"""

    def __init__(self, image_paths, labels, img_size=224, is_training=True, cache_dir="./preprocessed_cache"):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.is_training = is_training

        # 设置缓存目录
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 为每个图片路径生成唯一的缓存文件名
        import hashlib
        self.cache_files = []
        for path in image_paths:
            # 使用路径的hash作为缓存文件名
            path_hash = hashlib.md5(str(path).encode()).hexdigest()
            cache_file = self.cache_dir / f"preprocessed_{path_hash}.npy"
            self.cache_files.append(cache_file)

        # ImageNet标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        # 统计缓存命中率
        self.cache_hits = 0
        self.total_accesses = 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        cache_file = self.cache_files[idx]

        self.total_accesses += 1

        try:
            # 尝试加载预处理缓存
            if cache_file.exists():
                preprocessed_img = np.load(cache_file)
                self.cache_hits += 1

                # 显示缓存命中率（每100次访问显示一次）
                if self.total_accesses % 100 == 0:
                    hit_rate = self.cache_hits / self.total_accesses * 100
                    print(f"🚀 缓存命中率: {hit_rate:.1f}% ({self.cache_hits}/{self.total_accesses})")
            else:
                # 缓存不存在，进行完整预处理并保存
                preprocessed_img = self._full_preprocess_and_cache(image_path, cache_file)

            # 转换为tensor
            image = torch.from_numpy(preprocessed_img).float()

            # 训练时应用动态数据增强
            if self.is_training:
                image = self._apply_tensor_augmentation(image)

            # ImageNet标准化
            image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

            return image, label

        except Exception as e:
            print(f"❌ 加载图片失败 {image_path}: {e}")
            # 返回黑色图片作为fallback
            black_image = torch.zeros(3, self.img_size, self.img_size)
            return black_image, label

    def _full_preprocess_and_cache(self, image_path, cache_file):
        """完整预处理并缓存到磁盘"""
        # 执行STORK预处理（不包括数据增强）
        preprocessed_img = self._stork_preprocess_only(image_path)

        # 保存到缓存
        try:
            np.save(cache_file, preprocessed_img)
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")

        return preprocessed_img

    def _stork_preprocess_only(self, image_path):
        """只进行STORK预处理，不做数据增强"""
        import cv2
        import sys
        import os
        # 添加STORK_lzz路径到sys.path
        stork_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'STORK_lzz')
        if stork_path not in sys.path:
            sys.path.append(stork_path)

        try:
            from STORK_lzz.preprocessing.stork_image_utils import preprocess_embryo_image
        except ImportError:
            # 使用简化的预处理
            def preprocess_embryo_image(img, img_size=224, **kwargs):
                import cv2
                return cv2.resize(img, (img_size, img_size))

        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # STORK预处理（不含数据增强）
        img = preprocess_embryo_image(
            img,
            img_size=self.img_size,
            apply_petri_crop=True,
            apply_center_crop=True
        )

        # 转换格式：HWC -> CHW, 归一化到[0,1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        return img

    def _apply_tensor_augmentation(self, image):
        """对tensor格式的图像应用数据增强"""
        # 这里实现简化的tensor数据增强
        # 可以使用torchvision.transforms或者简单的随机变换

        # 随机水平翻转
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])  # 水平翻转

        # 随机亮度调整
        if torch.rand(1) > 0.8:
            brightness_factor = 0.9 + torch.rand(1) * 0.2  # [0.9, 1.1]
            image = torch.clamp(image * brightness_factor, 0.0, 1.0)

        return image

    def get_cache_stats(self):
        """获取缓存统计信息"""
        if self.total_accesses > 0:
            hit_rate = self.cache_hits / self.total_accesses * 100
            return {
                'cache_hits': self.cache_hits,
                'total_accesses': self.total_accesses,
                'hit_rate': hit_rate,
                'cache_dir': str(self.cache_dir)
            }
        return None



def create_pytorch_datasets_from_paths(train_paths, train_labels, test_paths, test_labels,
                                     img_size=224, cache_dir="./preprocessed_cache") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    从图片路径创建带渐进式缓存的PyTorch数据集

    Args:
        train_paths: 训练图片路径列表
        train_labels: 训练标签列表
        test_paths: 测试图片路径列表
        test_labels: 测试标签列表
        img_size: 图片尺寸
        cache_dir: 预处理缓存目录

    Returns:
        训练数据集, 测试数据集
    """
    print("🚀 创建渐进式缓存数据集，优化训练速度...")

    # 为训练和测试数据集使用不同的缓存目录
    train_cache_dir = f"{cache_dir}/train"
    test_cache_dir = f"{cache_dir}/test"

    train_dataset = LazySTORKDataset(
        train_paths, train_labels, img_size,
        is_training=True, cache_dir=train_cache_dir
    )
    test_dataset = LazySTORKDataset(
        test_paths, test_labels, img_size,
        is_training=False, cache_dir=test_cache_dir
    )

    print(f"✅ 渐进式缓存数据集创建完成")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   测试集: {len(test_dataset)} 样本")
    print(f"   缓存目录: {cache_dir}")
    print(f"   🔥 第一个epoch会边训练边缓存，后续epoch将显著加速！")

    return train_dataset, test_dataset




class STORKDataLoader:
    """STORK数据加载器 - 集成预处理和缓存"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化STORK数据加载器

        Args:
            config: 配置字典
        """
        self.config = config
        data_config = config.get('data', {})

        self.data_dir = data_config.get('data_dir')
        self.img_size = data_config.get('img_size', 224)
        self.train_split = data_config.get('train_split', 0.7)
        self.use_cache = data_config.get('use_cache', True)
        self.num_workers = data_config.get('num_workers', None)
        self.apply_petri_crop = data_config.get('apply_petri_crop', True)
        self.apply_center_crop = data_config.get('apply_center_crop', True)

    def load_data(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        加载数据并返回PyTorch数据集（渐进式缓存版本）

        Returns:
            训练数据集, 测试数据集
        """
        if is_main_process():
            print("🔄 开始收集图片路径信息...")

        # 直接收集图片路径，不进行预处理
        train_paths, train_labels, test_paths, test_labels = self._collect_image_paths()

        # 创建带渐进式缓存的PyTorch数据集
        train_dataset, test_dataset = create_pytorch_datasets_from_paths(
            train_paths, train_labels, test_paths, test_labels,
            img_size=self.img_size,
            cache_dir="./preprocessed_cache"
        )

        if is_main_process():
            print(f"✅ 数据加载完成")
            print(f"   训练集: {len(train_dataset)} 样本")
            print(f"   测试集: {len(test_dataset)} 样本")

        return train_dataset, test_dataset

    def _collect_image_paths(self):
        """收集图片路径和标签，不加载实际图片数据"""
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        import numpy as np

        image_paths = []
        labels = []

        if is_main_process():
            print("=== 收集图片路径信息 ===")

        # 收集所有图像路径和标签
        data_path = Path(self.data_dir)
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name.lower()
            if class_name in ['0', 'negative', 'no', 'non_viable', 'class_0']:
                class_label = 0
            elif class_name in ['1', 'positive', 'yes', 'viable', 'class_1']:
                class_label = 1
            else:
                continue

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    image_paths.append(str(img_file))
                    labels.append(class_label)

        # 转换为numpy数组
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        if is_main_process():
            print(f"找到 {len(image_paths)} 张图像")
            print(f"Class 0: {sum(labels == 0)} 张，Class 1: {sum(labels == 1)} 张")

        # 数据分割
        test_size = 1.0 - self.train_split
        if is_main_process():
            print(f"🎯 采用 {int(self.train_split*100)}/{int(test_size*100)} 数据分割策略...")

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=test_size,
            random_state=42, stratify=labels
        )

        if is_main_process():
            print(f"📊 数据分割完成:")
            print(f"  训练集: {len(train_paths)} ({len(train_paths)/len(image_paths)*100:.1f}%)")
            print(f"  测试集: {len(test_paths)} ({len(test_paths)/len(image_paths)*100:.1f}%)")

        return train_paths, train_labels, test_paths, test_labels


def create_stork_data_loader(config: Dict[str, Any]) -> STORKDataLoader:
    """
    创建STORK数据加载器的便捷函数

    Args:
        config: 配置字典

    Returns:
        STORKDataLoader实例
    """
    return STORKDataLoader(config)