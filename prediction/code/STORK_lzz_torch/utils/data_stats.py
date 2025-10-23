"""
数据集统计工具
提供数据集类别分布、胚胎数量统计等功能
"""

import numpy as np
import re
from pathlib import Path
from typing import Union, Dict, Any, Tuple, List
from torch.utils.data import Dataset, Subset

try:
    import torch
except ImportError:  # 避免在缺少torch时失败
    torch = None


def extract_embryo_id(image_path: str) -> str:
    """
    从图像路径中提取胚胎ID

    支持以下常见命名：
    - D2022.04.15_S00003_I4425_P_8.jpg           -> D2022.04.15_S00003_I4425_P_8
    - D2023.05.31_S00744_I4424_P_WELL12_RUN595   -> D2023.05.31_S00744_I4424_P_WELL12
    - 其它以 Dyyyy.mm.dd_Sxxxx_Ixxxx_P... 开头的变体
    """
    filename = Path(image_path).stem

    # 先去掉运行编号 / 焦面等后缀，如 _RUNxxx, _F-xx, _FRAMExx 等
    filename = re.split(r'_(?:RUN\d+.*|F-?\d+|FOCUS-?\d+|FRAME\d+|IMG\d+)$', filename)[0]

    # 匹配标准的 Dxxxx_Sxxxx_Ixxxx_P... 结构
    match = re.match(
        r'(D\d{4}\.\d{2}\.\d{2}_S\d+_I\d+_P(?:_[A-Za-z0-9]+)*)',
        filename,
        flags=re.IGNORECASE
    )
    if match:
        return match.group(1)

    return filename


def _labels_paths_from_cached_dataset(dataset: Dataset) -> Tuple[np.ndarray, List[str]]:
    """
    尝试直接从批量缓存数据集中获取标签与路径，避免逐样本遍历
    """
    if not hasattr(dataset, 'cached_labels'):
        return None

    labels_tensor = dataset.cached_labels
    if torch is not None and isinstance(labels_tensor, torch.Tensor):
        labels = labels_tensor.cpu().numpy()
    else:
        labels = np.array(labels_tensor)

    if hasattr(dataset, 'cached_paths') and dataset.cached_paths:
        paths = list(dataset.cached_paths)
    elif hasattr(dataset, 'image_paths') and dataset.image_paths is not None:
        paths = list(dataset.image_paths)
    else:
        # 回退：生成占位符路径
        paths = [str(i) for i in range(len(labels))]

    return labels, paths


def get_labels_and_paths_from_subset(subset: Subset) -> Tuple[np.ndarray, list]:
    """从Subset中提取所有标签和图像路径"""
    cached = _labels_paths_from_cached_dataset(subset.dataset)
    if cached is not None:
        labels, paths = cached
        indices = np.array(subset.indices)
        return labels[indices], [paths[i] for i in indices]

    labels = []
    paths = []
    for idx in subset.indices:
        if hasattr(subset.dataset, 'image_paths'):
            path = subset.dataset.image_paths[idx]
        elif hasattr(subset.dataset, 'data'):
            path = str(idx)
        else:
            path = str(idx)

        _, label = subset.dataset[idx]
        labels.append(int(label))
        paths.append(path)
    return np.array(labels), paths


def get_labels_from_subset(subset: Subset) -> np.ndarray:
    """从Subset中提取所有标签"""
    labels, _ = get_labels_and_paths_from_subset(subset)
    return labels


def get_labels_and_paths_from_dataset(dataset: Dataset) -> Tuple[np.ndarray, list]:
    """从Dataset中提取所有标签和图像路径"""
    cached = _labels_paths_from_cached_dataset(dataset)
    if cached is not None:
        return cached

    labels = []
    paths = []
    for i in range(len(dataset)):
        if hasattr(dataset, 'image_paths'):
            path = dataset.image_paths[i]
        else:
            path = str(i)

        _, label = dataset[i]
        labels.append(int(label))
        paths.append(path)
    return np.array(labels), paths


def get_labels_from_dataset(dataset: Dataset) -> np.ndarray:
    """从Dataset中提取所有标签"""
    labels, _ = get_labels_and_paths_from_dataset(dataset)
    return labels


def analyze_class_distribution(labels: np.ndarray) -> Dict[str, Any]:
    """分析类别分布

    Args:
        labels: 标签数组

    Returns:
        包含类别分布统计的字典
    """
    class_counts = np.bincount(labels)
    total = len(labels)

    stats = {
        'total': total,
        'class_counts': class_counts,
        'class_0_count': int(class_counts[0]),
        'class_1_count': int(class_counts[1]),
        'class_0_ratio': class_counts[0] / total,
        'class_1_ratio': class_counts[1] / total,
        'imbalance_ratio': class_counts[0] / class_counts[1] if class_counts[1] > 0 else 0
    }

    return stats


def print_dataset_statistics(train_data: Union[Dataset, Subset],
                            val_data: Union[Dataset, Subset],
                            test_data: Union[Dataset, Subset],
                            class_names: list = None) -> Dict[str, Any]:
    """打印训练集、验证集、测试集的详细统计信息

    Args:
        train_data: 训练集
        val_data: 验证集
        test_data: 测试集
        class_names: 类别名称列表，默认为['Poor/Non-viable', 'Good/Viable']

    Returns:
        包含所有统计信息的字典
    """
    if class_names is None:
        class_names = ['Poor/Non-viable', 'Good/Viable']

    # 提取标签和路径
    if isinstance(train_data, Subset):
        train_labels, train_paths = get_labels_and_paths_from_subset(train_data)
        val_labels, val_paths = get_labels_and_paths_from_subset(val_data)
    else:
        train_labels, train_paths = get_labels_and_paths_from_dataset(train_data)
        val_labels, val_paths = get_labels_and_paths_from_dataset(val_data)

    test_labels, test_paths = get_labels_and_paths_from_dataset(test_data)

    # 统计胚胎数量
    train_embryos = set(extract_embryo_id(p) for p in train_paths)
    val_embryos = set(extract_embryo_id(p) for p in val_paths)
    test_embryos = set(extract_embryo_id(p) for p in test_paths)
    all_embryos = train_embryos | val_embryos | test_embryos

    # 分析各数据集分布
    train_stats = analyze_class_distribution(train_labels)
    val_stats = analyze_class_distribution(val_labels)
    test_stats = analyze_class_distribution(test_labels)

    # 计算总体统计
    total_samples = len(train_labels) + len(val_labels) + len(test_labels)
    total_class_0 = train_stats['class_0_count'] + val_stats['class_0_count'] + test_stats['class_0_count']
    total_class_1 = train_stats['class_1_count'] + val_stats['class_1_count'] + test_stats['class_1_count']

    # 打印统计信息
    print(f"\n{'='*70}")
    print(f"数据集详细统计")
    print(f"{'='*70}")

    print(f"\n🧬 胚胎数量统计:")
    print(f"   总胚胎数: {len(all_embryos)}")
    print(f"   训练集胚胎数: {len(train_embryos)}")
    print(f"   验证集胚胎数: {len(val_embryos)}")
    print(f"   测试集胚胎数: {len(test_embryos)}")

    print(f"\n📊 总体数据分布:")
    print(f"   总样本数: {total_samples}")
    print(f"   Class 0 ({class_names[0]}): {total_class_0} ({total_class_0/total_samples*100:.2f}%)")
    print(f"   Class 1 ({class_names[1]}): {total_class_1} ({total_class_1/total_samples*100:.2f}%)")
    print(f"   类别比例: {total_class_0/total_class_1:.2f}:1")

    print(f"\n📈 训练集:")
    print(f"   图像数: {train_stats['total']}")
    print(f"   胚胎数: {len(train_embryos)}")
    print(f"   Class 0: {train_stats['class_0_count']} ({train_stats['class_0_ratio']*100:.2f}%)")
    print(f"   Class 1: {train_stats['class_1_count']} ({train_stats['class_1_ratio']*100:.2f}%)")

    print(f"\n📉 验证集:")
    print(f"   图像数: {val_stats['total']}")
    print(f"   胚胎数: {len(val_embryos)}")
    print(f"   Class 0: {val_stats['class_0_count']} ({val_stats['class_0_ratio']*100:.2f}%)")
    print(f"   Class 1: {val_stats['class_1_count']} ({val_stats['class_1_ratio']*100:.2f}%)")

    print(f"\n🧪 测试集:")
    print(f"   图像数: {test_stats['total']}")
    print(f"   胚胎数: {len(test_embryos)}")
    print(f"   Class 0: {test_stats['class_0_count']} ({test_stats['class_0_ratio']*100:.2f}%)")
    print(f"   Class 1: {test_stats['class_1_count']} ({test_stats['class_1_ratio']*100:.2f}%)")

    print(f"\n{'='*70}\n")

    # 返回统计结果
    return {
        'train': train_stats,
        'val': val_stats,
        'test': test_stats,
        'total': {
            'samples': total_samples,
            'class_0': total_class_0,
            'class_1': total_class_1
        },
        'embryos': {
            'total': len(all_embryos),
            'train': len(train_embryos),
            'val': len(val_embryos),
            'test': len(test_embryos)
        }
    }
