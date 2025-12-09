#!/usr/bin/env python3
"""
验证数据分割的完整性
检查胚胎在不同数据集（train/val/test）之间是否有重叠
"""

import sys
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from data import STORKDataLoader
from config import load_config
import numpy as np


def extract_embryo_id(path):
    """提取胚胎ID"""
    try:
        from utils.data_stats import extract_embryo_id as _extract
        return _extract(str(path))
    except:
        # 备用提取逻辑
        filename = Path(path).stem
        parts = filename.split('_')
        if len(parts) >= 7:
            return '_'.join(parts[:7])
        return filename


def verify_split_integrity():
    """验证数据分割的完整性"""
    print("\n" + "="*80)
    print("验证数据分割的完整性")
    print("="*80)

    # 创建一个最小的配置
    config = {
        'data': {
            'data_dir': '/liaozizhuo/selected_images/train',
            'img_size': 224,
            'train_split': 0.7,
            'use_cache': False,  # 暂时关闭缓存以测试分割逻辑
            'num_workers': 4,
            'apply_petri_crop': True,
            'apply_center_crop': True,
            'random_state': 42,
            'embryo_balanced_sampling': False
        },
        'paths': {
            'output_dir': './output',
            'cache_dir': './batch_cache',
            'log_dir': './logs',
            'model_dir': './checkpoints'
        }
    }

    # 创建数据加载器
    data_loader = STORKDataLoader(config)

    # 收集图片路径
    print("\n收集图片路径...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = data_loader._collect_image_paths()

    # 提取胚胎IDs
    print("\n提取胚胎IDs...")
    train_embryos = set()
    val_embryos = set()
    test_embryos = set()

    for path in train_paths:
        embryo_id = extract_embryo_id(path)
        train_embryos.add(embryo_id)

    for path in val_paths:
        embryo_id = extract_embryo_id(path)
        val_embryos.add(embryo_id)

    for path in test_paths:
        embryo_id = extract_embryo_id(path)
        test_embryos.add(embryo_id)

    # 计算重叠
    print("\n" + "="*80)
    print("胚胎分布统计")
    print("="*80)
    print(f"训练集胚胎数: {len(train_embryos)}")
    print(f"验证集胚胎数: {len(val_embryos)}")
    print(f"测试集胚胎数: {len(test_embryos)}")
    print(f"总胚胎数（无重复）: {len(train_embryos | val_embryos | test_embryos)}")

    train_val_overlap = len(train_embryos & val_embryos)
    train_test_overlap = len(train_embryos & test_embryos)
    val_test_overlap = len(val_embryos & test_embryos)
    three_way_overlap = len(train_embryos & val_embryos & test_embryos)

    print("\n" + "="*80)
    print("胚胎重叠检查")
    print("="*80)
    print(f"train ∩ val: {train_val_overlap} (应该为 0)")
    print(f"train ∩ test: {train_test_overlap} (应该为 0)")
    print(f"val ∩ test: {val_test_overlap} (应该为 0)")
    print(f"train ∩ val ∩ test: {three_way_overlap} (应该为 0)")

    # 验证数据集大小一致性
    print("\n" + "="*80)
    print("数据集大小验证")
    print("="*80)
    print(f"训练集: {len(train_paths)} 张图像，{len(train_embryos)} 个胚胎")
    print(f"验证集: {len(val_paths)} 张图像，{len(val_embryos)} 个胚胎")
    print(f"测试集: {len(test_paths)} 张图像，{len(test_embryos)} 个胚胎")
    print(f"总计: {len(train_paths) + len(val_paths) + len(test_paths)} 张图像")

    # 验证标签分布
    print("\n" + "="*80)
    print("标签分布验证")
    print("="*80)
    train_labels_arr = np.array(train_labels)
    val_labels_arr = np.array(val_labels)
    test_labels_arr = np.array(test_labels)

    print(f"训练集: Class 0: {sum(train_labels_arr == 0)}, Class 1: {sum(train_labels_arr == 1)}")
    print(f"验证集: Class 0: {sum(val_labels_arr == 0)}, Class 1: {sum(val_labels_arr == 1)}")
    print(f"测试集: Class 0: {sum(test_labels_arr == 0)}, Class 1: {sum(test_labels_arr == 1)}")

    # 最终检查
    print("\n" + "="*80)
    if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
        print("✓ 数据分割验证通过！胚胎完全分离，无数据泄漏")
        print("="*80 + "\n")
        return True
    else:
        print("✗ 数据分割验证失败！存在胚胎重叠：")
        print(f"  - train ∩ val: {train_val_overlap}")
        print(f"  - train ∩ test: {train_test_overlap}")
        print(f"  - val ∩ test: {val_test_overlap}")
        print("="*80 + "\n")
        return False


if __name__ == '__main__':
    success = verify_split_integrity()
    sys.exit(0 if success else 1)
