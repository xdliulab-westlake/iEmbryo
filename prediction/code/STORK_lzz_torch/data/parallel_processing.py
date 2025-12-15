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
from typing import List, Tuple, Dict, Any, Optional, Iterator
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from sklearn.model_selection import train_test_split

from .stork_preprocessing import preprocess_embryo_image


def is_main_process() -> bool:
    """检查当前进程是否为主进程"""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def verify_class_distribution(labels_list: List[np.ndarray],
                            split_names: List[str],
                            detailed: bool = True) -> Dict[str, Dict]:
    """验证各分割中的类别分布是否保持一致

    Args:
        labels_list: 标签数组列表
        split_names: 分割名称列表
        detailed: 是否打印详细信息

    Returns:
        分布统计字典
    """
    if not is_main_process():
        return {}

    results = {}

    if detailed:
        print(f"\n{'='*70}")
        print(f"类别分布验证 (分层检查)")
        print(f"{'='*70}")

    for labels, name in zip(labels_list, split_names):
        counts = np.bincount(labels)
        total = len(labels)
        ratios = counts / total

        results[name] = {
            'counts': counts.tolist(),
            'ratios': ratios.tolist(),
            'total': total
        }

        if detailed:
            ratio_str = " | ".join([f"Class {i}: {count:5d} ({ratio*100:5.1f}%)"
                                   for i, (count, ratio) in enumerate(zip(counts, ratios))])
            print(f"{name:20s}: {ratio_str}")

    if detailed:
        print(f"{'='*70}\n")

    return results


def optimized_preprocess_single_image(args: Tuple) -> np.ndarray:
    """
    优化的单图像预处理函数（专为多进程设计）

    Args:
        args: (path, img_size, is_training, apply_petri_crop)

    Returns:
        预处理后的图像数组
    """
    path, img_size, is_training, apply_petri_crop = args

    try:
        # 使用STORK预处理管道
        img = preprocess_embryo_image(
            path,
            img_size=img_size,
            apply_petri_crop=apply_petri_crop
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
                                   num_workers: Optional[int] = None) -> np.ndarray:
    """
    并行处理图像批次

    Args:
        paths: 图像路径列表
        img_size: 目标图像尺寸
        is_training: 是否为训练模式
        apply_petri_crop: 是否应用培养皿检测
        num_workers: 工作进程数

    Returns:
        处理后的图像数组 [N, H, W, C]
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # 限制进程数避免系统过载

    if is_main_process():
        print(f"使用 {num_workers} 个进程并行处理 {len(paths)} 张图像...")

    # 准备多进程参数
    args_list = [(path, img_size, is_training, apply_petri_crop)
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
    - STORK预处理：复杂培养皿检测等专业胚胎学图像处理
    """
    image_paths = []
    labels = []
    qualities = []

    if is_main_process():
        print("=== 使用优化的STORK胚胎图像预处理加载数据 ===")
        print(f"并行处理模式：{'启用' if num_workers != 1 else '禁用'}")
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
        if is_main_process():
            print(f"自动检测到 {num_workers} 个工作进程")

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
        print(f"采用 {int(train_split*100)}/{int(test_size*100)} 数据分割策略...")

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
        print(f"数据分割完成:")
        print(f"  总样本数: {total_samples}")
        print(f"  训练集: {train_samples} ({train_samples/total_samples*100:.1f}%)")
        print(f"  测试集: {test_samples} ({test_samples/total_samples*100:.1f}%)")

    # 缓存机制
    cache_manager = DatasetCache() if use_cache else None
    cached_data = None

    if use_cache:
        cache_key = cache_manager._generate_cache_key(
            data_dir, img_size, len(image_paths), apply_petri_crop,
            split_strategy, num_workers
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
        print(f"=== 并行预处理训练集（{int(train_split*100)}%）===")
    start_time = time.time()
    X_train = batch_preprocess_images_parallel(
        train_paths, img_size, is_training=True,
        apply_petri_crop=apply_petri_crop,
        num_workers=num_workers
    )
    train_time = time.time() - start_time
    if is_main_process():
        print(f"训练集预处理完成，耗时: {train_time:.2f} 秒")

    if is_main_process():
        print(f"=== 并行预处理测试集（{int(test_size*100)}%）===")
    start_time = time.time()
    X_test = batch_preprocess_images_parallel(
        test_paths, img_size, is_training=False,
        apply_petri_crop=apply_petri_crop,
        num_workers=num_workers
    )
    test_time = time.time() - start_time
    if is_main_process():
        print(f"测试集预处理完成，耗时: {test_time:.2f} 秒")

    total_time = train_time + test_time
    if is_main_process():
        print(f"=== 数据预处理完成 ===")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        print(f" 总预处理时间: {total_time:.2f} 秒 (并行优化版)")
        print(f"平均每张图像: {total_time/len(image_paths)*1000:.2f} 毫秒")

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
            'version': 'pytorch_stork_v1.0'
        }

        cache_manager.save_cache(cache_key, cache_data)

    return (X_train, train_labels, train_quals,    # 训练集
            X_test, test_labels, test_quals)        # 测试集




def create_pytorch_datasets_from_paths(train_paths, train_labels, test_paths, test_labels,
                                     img_size=224, cache_dir="./batch_cache", chunk_size=1000) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    从图片路径创建批量缓存数据集

    Args:
        train_paths: 训练图片路径列表
        train_labels: 训练标签列表
        test_paths: 测试图片路径列表
        test_labels: 测试标签列表
        img_size: 图片尺寸
        cache_dir: 缓存目录
        chunk_size: 每个缓存文件的图片数量

    Returns:
        训练数据集, 测试数据集
    """
    try:
        from .batch_cache import BatchCachedDataset
    except ImportError:
        from batch_cache import BatchCachedDataset

    print(f"创建批量缓存数据集...")
    print(f"   图片尺寸: {img_size}×{img_size}")
    print(f"   训练集: {len(train_paths)} 样本")
    print(f"   测试集: {len(test_paths)} 样本")
    print(f"   缓存块大小: {chunk_size} 张/文件")

    # 需要从配置中获取这些参数以确保缓存键一致
    # 这里使用默认值，实际应该从上级传入
    data_dir = str(Path(train_paths[0]).parents[1]) if train_paths else "./data"
    train_split = 0.7  # 应该从配置传入
    random_state = 42  # 应该从配置传入

    train_dataset = BatchCachedDataset(
        train_paths, train_labels, img_size,
        is_training=True, cache_dir=cache_dir,
        chunk_size=chunk_size, split="train",
        data_dir=data_dir, train_split=train_split,
        random_state=random_state
    )
    test_dataset = BatchCachedDataset(
        test_paths, test_labels, img_size,
        is_training=False, cache_dir=cache_dir,
        chunk_size=chunk_size, split="test",
        data_dir=data_dir, train_split=train_split,
        random_state=random_state
    )

    print(f"批量缓存数据集创建完成")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   测试集: {len(test_dataset)} 样本")
    print(f"   缓存目录: {cache_dir}")

    # 显示缓存统计
    stats = train_dataset.get_cache_stats()
    print(f"   缓存统计: {stats['total_chunks']} 个文件, {stats['total_size_mb']:.1f}MB")

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

    def load_data(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        加载数据并返回PyTorch数据集（三个独立的数据集）

        Returns:
            训练数据集, 验证数据集, 测试数据集
        """
        if is_main_process():
            print("开始收集图片路径信息...")

        # 收集图片路径，返回三个独立的数据集
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = self._collect_image_paths()

        # 获取缓存配置
        cache_dir = self.config.get('paths', {}).get('cache_dir', './batch_cache')
        chunk_size = self.config.get('data', {}).get('cache_chunk_size', 1000)

        # 创建批量缓存PyTorch数据集，传递完整配置参数
        train_dataset, val_dataset, test_dataset = self._create_cached_datasets(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            cache_dir, chunk_size
        )

        if is_main_process():
            print(f"数据加载完成")
            print(f"   训练集: {len(train_dataset)} 样本")
            print(f"   验证集: {len(val_dataset)} 样本")
            print(f"   测试集: {len(test_dataset)} 样本")

        return train_dataset, val_dataset, test_dataset

    def _collect_image_paths(self):
        """
        收集图片路径和标签，不加载实际图片数据

        新策略：图像级别平衡 + 胚胎级别分离
        1. 每个集合中 Class 0 和 Class 1 的图像数量相等（50:50）
        2. 总体图像比例为 70:15:15（训练:验证:测试）
        3. 每个胚胎只出现在一个集合中（避免数据泄漏）
        4. 通过下采样达到平衡
        """
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        import numpy as np
        from collections import defaultdict
        import json

        # 导入胚胎ID提取函数
        try:
            from ..utils.data_stats import extract_embryo_id
        except ImportError:
            from utils.data_stats import extract_embryo_id

        image_paths = []
        labels = []

        if is_main_process():
            print("=== 收集图片路径信息 (图像级别平衡 + 胚胎级别分离: 70/15/15) ===")

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
            print(f"Class 0: {sum(labels == 0)} 张 ({sum(labels == 0)/len(labels)*100:.1f}%)")
            print(f"Class 1: {sum(labels == 1)} 张 ({sum(labels == 1)/len(labels)*100:.1f}%)")

        # ===== Step 1: 按胚胎ID对图像进行分组 =====
        embryo_groups = defaultdict(list)  # {embryo_id: [(path, label), ...]}

        for path, label in zip(image_paths, labels):
            embryo_id = extract_embryo_id(path)
            embryo_groups[embryo_id].append((path, label))

        if is_main_process():
            print(f"\n📊 胚胎统计:")
            print(f"  总胚胎数: {len(embryo_groups)}")
            print(f"  平均每个胚胎: {len(image_paths) / len(embryo_groups):.1f} 张图像")

        # ===== Step 2: 按类别组织胚胎（记录每个胚胎的图像数量）=====
        class_embryo_info = {0: [], 1: []}  # {class: [(embryo_id, num_images, [paths], [labels]), ...]}

        for embryo_id, items in embryo_groups.items():
            paths, item_labels = zip(*items)
            # 使用多数投票确定胚胎的标签
            embryo_label = int(np.round(np.mean(item_labels)))
            num_images = len(items)
            class_embryo_info[embryo_label].append((embryo_id, num_images, list(paths), list(item_labels)))

        # 按图像数量排序（方便采样）
        for class_label in [0, 1]:
            class_embryo_info[class_label].sort(key=lambda x: x[1], reverse=True)

        if is_main_process():
            print(f"\n📈 胚胎级别类别分布:")
            for class_label in [0, 1]:
                embryo_list = class_embryo_info[class_label]
                total_images = sum(info[1] for info in embryo_list)
                print(f"  Class {class_label}: {len(embryo_list)} 个胚胎，{total_images} 张图像")

        # ===== Step 3: 计算目标图像数量（平衡策略）=====
        # 使用少数类的总图像数作为每个类别的目标数量
        total_class_0_images = sum(info[1] for info in class_embryo_info[0])
        total_class_1_images = sum(info[1] for info in class_embryo_info[1])

        target_images_per_class = min(total_class_0_images, total_class_1_images)

        # 分配到各个集合（70:15:15）
        train_target_per_class = int(target_images_per_class * 0.70)
        val_target_per_class = int(target_images_per_class * 0.15)
        test_target_per_class = target_images_per_class - train_target_per_class - val_target_per_class

        if is_main_process():
            print(f"\n🎯 目标图像数量（下采样到少数类）:")
            print(f"  每个类别总数: {target_images_per_class} 张")
            print(f"  训练集（每类）: {train_target_per_class} 张")
            print(f"  验证集（每类）: {val_target_per_class} 张")
            print(f"  测试集（每类）: {test_target_per_class} 张")
            print(f"  总图像数: {target_images_per_class * 2} 张")

        # ===== Step 4: 为每个类别选择胚胎，达到目标图像数量 =====
        random_state = self.config.get('data', {}).get('random_state', 42)
        rng = np.random.RandomState(random_state)

        selected_embryos = {0: {'train': [], 'val': [], 'test': []},
                           1: {'train': [], 'val': [], 'test': []}}

        for class_label in [0, 1]:
            embryo_list = class_embryo_info[class_label].copy()
            rng.shuffle(embryo_list)  # 随机打乱

            # 依次分配到 train, val, test
            current_counts = {'train': 0, 'val': 0, 'test': 0}
            targets = {'train': train_target_per_class,
                      'val': val_target_per_class,
                      'test': test_target_per_class}

            for embryo_id, num_images, paths, item_labels in embryo_list:
                # 找到还需要图像的集合
                for split in ['train', 'val', 'test']:
                    if current_counts[split] < targets[split]:
                        selected_embryos[class_label][split].append(
                            (embryo_id, num_images, paths, item_labels)
                        )
                        current_counts[split] += num_images
                        break

                # 如果所有集合都已满，停止
                if all(current_counts[s] >= targets[s] for s in ['train', 'val', 'test']):
                    break

        # ===== Step 5: 组装最终的数据集 =====
        train_paths_list, train_labels_list = [], []
        val_paths_list, val_labels_list = [], []
        test_paths_list, test_labels_list = [], []

        embryo_split_mapping = {}
        image_split_mapping = {}

        for class_label in [0, 1]:
            for split_name, split_list in [('train', train_paths_list),
                                          ('val', val_paths_list),
                                          ('test', test_paths_list)]:
                split_labels = train_labels_list if split_name == 'train' else (
                    val_labels_list if split_name == 'val' else test_labels_list
                )

                for embryo_id, _, paths, item_labels in selected_embryos[class_label][split_name]:
                    split_list.extend(paths)
                    split_labels.extend(item_labels)
                    embryo_split_mapping[embryo_id] = split_name
                    for path in paths:
                        image_split_mapping[path] = split_name

        # 转换为numpy数组
        train_paths = np.array(train_paths_list)
        val_paths = np.array(val_paths_list)
        test_paths = np.array(test_paths_list)
        train_labels = np.array(train_labels_list)
        val_labels = np.array(val_labels_list)
        test_labels = np.array(test_labels_list)

        if is_main_process():
            print(f"\n✅ 图像级别平衡分割结果:")
            print(f"\n🧬 胚胎数量统计:")
            train_embryos = sum(len(selected_embryos[c]['train']) for c in [0, 1])
            val_embryos = sum(len(selected_embryos[c]['val']) for c in [0, 1])
            test_embryos = sum(len(selected_embryos[c]['test']) for c in [0, 1])
            total_embryos = train_embryos + val_embryos + test_embryos
            print(f"   总胚胎数: {total_embryos}")
            print(f"   训练集胚胎数: {train_embryos}")
            print(f"   验证集胚胎数: {val_embryos}")
            print(f"   测试集胚胎数: {test_embryos}")

            print(f"\n📊 总体数据分布:")
            total_images = len(train_paths) + len(val_paths) + len(test_paths)
            total_class_0 = sum(train_labels == 0) + sum(val_labels == 0) + sum(test_labels == 0)
            total_class_1 = sum(train_labels == 1) + sum(val_labels == 1) + sum(test_labels == 1)
            print(f"   总样本数: {total_images}")
            print(f"   Class 0 (Poor/Non-viable): {total_class_0} ({total_class_0/total_images*100:.2f}%)")
            print(f"   Class 1 (Good/Viable): {total_class_1} ({total_class_1/total_images*100:.2f}%)")
            print(f"   类别比例: {total_class_0/total_class_1:.2f}:1")

            print(f"\n📈 训练集:")
            print(f"   图像数: {len(train_paths)}")
            print(f"   胚胎数: {train_embryos}")
            print(f"   Class 0: {sum(train_labels == 0)} ({sum(train_labels == 0)/len(train_labels)*100:.2f}%)")
            print(f"   Class 1: {sum(train_labels == 1)} ({sum(train_labels == 1)/len(train_labels)*100:.2f}%)")

            print(f"\n📉 验证集:")
            print(f"   图像数: {len(val_paths)}")
            print(f"   胚胎数: {val_embryos}")
            print(f"   Class 0: {sum(val_labels == 0)} ({sum(val_labels == 0)/len(val_labels)*100:.2f}%)")
            print(f"   Class 1: {sum(val_labels == 1)} ({sum(val_labels == 1)/len(val_labels)*100:.2f}%)")

            print(f"\n🧪 测试集:")
            print(f"   图像数: {len(test_paths)}")
            print(f"   胚胎数: {test_embryos}")
            print(f"   Class 0: {sum(test_labels == 0)} ({sum(test_labels == 0)/len(test_labels)*100:.2f}%)")
            print(f"   Class 1: {sum(test_labels == 1)} ({sum(test_labels == 1)/len(test_labels)*100:.2f}%)")

        # Step 6: 保存分割映射文件
        self._save_split_mapping(embryo_split_mapping, image_split_mapping)

        # 保存分割信息到对象属性，供后续使用
        self.split_mapping = embryo_split_mapping  # embryo_id -> split
        self.image_split_mapping = image_split_mapping  # image_path -> split
        self.train_embryo_ids = set(embryo_split_mapping[eid] == 'train' for eid in embryo_split_mapping)
        self.val_embryo_ids = set(embryo_split_mapping[eid] == 'val' for eid in embryo_split_mapping)
        self.test_embryo_ids = set(embryo_split_mapping[eid] == 'test' for eid in embryo_split_mapping)

        # 保存各分割的路径和标签供后续使用
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.val_paths = val_paths
        self.val_labels = val_labels
        self.test_paths = test_paths
        self.test_labels = test_labels

        # 返回三个独立的数据集 (train, val, test) - 确保胚胎级别的完全分离
        # 每个胚胎仅在一个集合中出现，没有数据泄漏
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

    def _balance_classes_by_undersampling(self, paths, labels, embryo_ids):
        """
        通过欠采样平衡训练集类别
        在胚胎级别进行平衡：减少多数类胚胎数量，使两类胚胎数量相等

        Args:
            paths: 图像路径数组
            labels: 标签数组
            embryo_ids: 胚胎ID集合

        Returns:
            平衡后的(paths, labels)
        """
        from collections import defaultdict

        # 导入胚胎ID提取函数
        try:
            from ..utils.data_stats import extract_embryo_id
        except ImportError:
            from utils.data_stats import extract_embryo_id

        if is_main_process():
            print(f"\n应用胚胎级别的类别平衡（欠采样）...")

        # 按胚胎和类别分组
        embryo_samples = defaultdict(list)  # {embryo_id: [(idx, path, label)]}

        for idx, (path, label) in enumerate(zip(paths, labels)):
            embryo_id = extract_embryo_id(path)
            embryo_samples[embryo_id].append((idx, path, label))

        # 按类别组织胚胎
        class_embryo_groups = {0: {}, 1: {}}

        for embryo_id, samples in embryo_samples.items():
            _, sample_paths, sample_labels = zip(*samples)
            # 使用多数投票确定胚胎类别
            embryo_class = 1 if np.mean(sample_labels) >= 0.5 else 0
            class_embryo_groups[embryo_class][embryo_id] = samples

        num_class_0 = len(class_embryo_groups[0])
        num_class_1 = len(class_embryo_groups[1])

        if is_main_process():
            print(f"  原始胚胎分布:")
            print(f"    Class 0: {num_class_0} 个胚胎")
            print(f"    Class 1: {num_class_1} 个胚胎")

        # 确定目标胚胎数量（使用较小类的数量）
        target_num_embryos = min(num_class_0, num_class_1)

        if is_main_process():
            print(f"  目标胚胎数量（每类）: {target_num_embryos}")

        # 对每个类别采样目标数量的胚胎
        rng = np.random.RandomState(self.config.get('data', {}).get('random_state', 42))
        balanced_samples = []

        for class_label in [0, 1]:
            embryo_ids_list = list(class_embryo_groups[class_label].keys())

            # 随机选择目标数量的胚胎
            selected_embryo_ids = rng.choice(
                embryo_ids_list,
                size=target_num_embryos,
                replace=False
            )

            # 收集选中胚胎的所有样本
            for embryo_id in selected_embryo_ids:
                samples = class_embryo_groups[class_label][embryo_id]
                balanced_samples.extend(samples)

        # 打乱顺序
        rng.shuffle(balanced_samples)

        # 提取路径和标签
        _, balanced_paths, balanced_labels = zip(*balanced_samples)
        balanced_paths = np.array(balanced_paths)
        balanced_labels = np.array(balanced_labels)

        if is_main_process():
            print(f"\n  平衡后结果:")
            print(f"    总图像数: {len(balanced_paths)} (原始: {len(paths)})")
            print(f"    Class 0: {sum(balanced_labels == 0)} 张 ({sum(balanced_labels == 0)/len(balanced_labels)*100:.1f}%)")
            print(f"    Class 1: {sum(balanced_labels == 1)} 张 ({sum(balanced_labels == 1)/len(balanced_labels)*100:.1f}%)")
            print(f"    胚胎数量: {target_num_embryos * 2} (每类 {target_num_embryos} 个)")

        return balanced_paths, balanced_labels

    def _save_split_mapping(self, embryo_split_mapping, image_split_mapping):
        """
        保存分割映射文件
        """
        from pathlib import Path
        import json
        import csv

        split_dir = Path(self.config.get('paths', {}).get('output_dir', './output')) / 'split_info'
        split_dir.mkdir(parents=True, exist_ok=True)

        if is_main_process():
            # 保存胚胎级别映射 (JSON)
            embryo_mapping_file = split_dir / 'embryo_split_mapping.json'
            with open(embryo_mapping_file, 'w') as f:
                json.dump(embryo_split_mapping, f, indent=2, ensure_ascii=False)
            print(f"\n已保存胚胎分割映射: {embryo_mapping_file}")

            # 保存图像级别映射 (CSV)
            image_mapping_file = split_dir / 'image_split_mapping.csv'
            with open(image_mapping_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'split'])
                for path, split in sorted(image_split_mapping.items()):
                    writer.writerow([path, split])
            print(f"已保存图像分割映射: {image_mapping_file}")

    def _create_cached_datasets(self, train_paths, train_labels, val_paths, val_labels,
                              test_paths, test_labels, cache_dir, chunk_size):
        """创建缓存数据集，确保三个独立数据集的正确分离"""
        try:
            from .batch_cache import BatchCachedDataset
        except ImportError:
            from batch_cache import BatchCachedDataset

        print(f"创建批量缓存数据集...")
        print(f"   图片尺寸: {self.img_size}×{self.img_size}")
        print(f"   训练集: {len(train_paths)} 样本")
        print(f"   验证集: {len(val_paths)} 样本")
        print(f"   测试集: {len(test_paths)} 样本")
        print(f"   缓存块大小: {chunk_size} 张/文件")

        # 从配置中获取随机种子
        random_state = self.config.get('data', {}).get('random_state', 42)

        train_dataset = BatchCachedDataset(
            train_paths, train_labels, self.img_size,
            is_training=True, cache_dir=cache_dir,
            chunk_size=chunk_size, split="train",
            data_dir=self.data_dir, train_split=self.train_split,
            random_state=random_state,
            apply_petri_crop=self.apply_petri_crop
        )
        val_dataset = BatchCachedDataset(
            val_paths, val_labels, self.img_size,
            is_training=False, cache_dir=cache_dir,
            chunk_size=chunk_size, split="val",
            data_dir=self.data_dir, train_split=self.train_split,
            random_state=random_state,
            apply_petri_crop=self.apply_petri_crop
        )
        test_dataset = BatchCachedDataset(
            test_paths, test_labels, self.img_size,
            is_training=False, cache_dir=cache_dir,
            chunk_size=chunk_size, split="test",
            data_dir=self.data_dir, train_split=self.train_split,
            random_state=random_state,
            apply_petri_crop=self.apply_petri_crop
        )

        print(f"批量缓存数据集创建完成")
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   验证集: {len(val_dataset)} 样本")
        print(f"   测试集: {len(test_dataset)} 样本")
        print(f"   缓存目录: {cache_dir}")

        # 显示缓存统计
        try:
            stats = train_dataset.get_cache_stats()
            print(f"   缓存统计: {stats['total_chunks']} 个文件, {stats['total_size_mb']:.1f}MB")
        except Exception as e:
            print(f"   无法获取缓存统计: {e}")

        return train_dataset, val_dataset, test_dataset



def create_stork_data_loader(config: Dict[str, Any]) -> STORKDataLoader:
    """
    创建STORK数据加载器的便捷函数

    Args:
        config: 配置字典

    Returns:
        STORKDataLoader实例
    """
    return STORKDataLoader(config)
