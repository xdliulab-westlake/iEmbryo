#!/usr/bin/env python3
"""
从缓存中随机提取胚胎样本（包含所有焦面）
按胚胎为单位提取，每个胚胎包含其所有焦面的图片
只选择含有完整11个焦面的胚胎
"""

import os
import sys
import json
import random
import shutil
import re
import torch
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import argparse


def load_fair_embryo_ids(csv_path="../MainDB.csv"):
    """
    从MainDB.csv加载Fair胚胎的ID列表

    Args:
        csv_path: MainDB.csv文件路径

    Returns:
        set of fair embryo IDs
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        # 第一列可能是EmbryoID或RowName，格式: D2024.03.18_S00006_I5068_P_5
        # 兼容两种格式
        if 'EmbryoID' in df.columns:
            id_column = 'EmbryoID'
        elif 'RowName' in df.columns:
            id_column = 'RowName'
        else:
            raise ValueError(f"CSV文件中找不到'EmbryoID'或'RowName'列。可用列: {df.columns.tolist()}")

        fair_embryos = set(df[df['Quality'] == 'fair'][id_column].values)
        print(f"从 {csv_path} 加载了 {len(fair_embryos)} 个Fair胚胎ID (使用列: {id_column})")
        if len(fair_embryos) > 0:
            sample_ids = sorted(fair_embryos)[:3]
            print(f"  示例Fair胚胎ID: {sample_ids}")
        return fair_embryos
    except Exception as e:
        print(f"警告: 无法加载Fair胚胎列表: {e}")
        print("将不进行Fair胚胎筛选")
        import traceback
        traceback.print_exc()
        return None


def load_images_from_data_dir(data_dir, fair_embryo_ids=None):
    """
    直接从原始数据目录加载图片路径（用于Fair胚胎）

    Args:
        data_dir: 原始数据目录路径
        fair_embryo_ids: Fair胚胎ID集合

    Returns:
        (图片路径列表, 标签列表)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_paths = []
    all_labels = []

    print(f"从原始数据目录加载Fair胚胎图片: {data_dir}")

    # Debug: 显示前几个目录名和Fair胚胎ID
    all_dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"\n🔍 调试信息:")
    print(f"  数据目录中总共有 {len(all_dirs)} 个子目录")
    print(f"  前5个目录名示例: {all_dirs[:5]}")
    if fair_embryo_ids:
        fair_list = list(fair_embryo_ids)
        print(f"  Fair胚胎ID示例 (前5个): {fair_list[:5]}")

    matched_count = 0

    # 遍历数据目录中的所有胚胎文件夹
    for embryo_dir in data_path.iterdir():
        if not embryo_dir.is_dir():
            continue

        embryo_id = embryo_dir.name

        # 如果提供了Fair胚胎列表，只处理Fair胚胎
        if fair_embryo_ids:
            # 直接比较（格式应该一致）
            if embryo_id in fair_embryo_ids:
                matched_count += 1
                if matched_count <= 3:
                    print(f"  ✓ 匹配到Fair胚胎目录: {embryo_id}")
            else:
                # 如果直接匹配失败，尝试格式转换（兼容不同格式）
                # D2024_10_04_S01672_I4425_P_5 <-> D2024.10.04_S01672_I4425_P_5
                embryo_id_alt = embryo_id.replace('_', '.', 2) if '_' in embryo_id[:10] else embryo_id.replace('.', '_', 2)
                if embryo_id_alt in fair_embryo_ids:
                    matched_count += 1
                    if matched_count <= 3:
                        print(f"  ✓ 匹配到Fair胚胎目录: {embryo_id} (转换后匹配)")
                else:
                    continue
        else:
            matched_count += 1

        # 检查是否有images子目录
        images_subdir = embryo_dir / "images"
        if images_subdir.exists() and images_subdir.is_dir():
            focal_parent_dir = images_subdir
        else:
            focal_parent_dir = embryo_dir

        # 遍历胚胎目录下的所有焦面目录（F开头的目录）
        for focal_dir in focal_parent_dir.iterdir():
            if not focal_dir.is_dir():
                continue

            focal_dir_name = focal_dir.name
            if not focal_dir_name.startswith('F'):
                continue

            # 在焦面目录中查找图片
            # 对于Fair胚胎，文件名不包含焦面信息，焦面信息在目录名中
            # 例如: images/F0/D2024.10.04_S01672_I4425_P_WELL3_RUN500.jpeg
            for img_file in focal_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    # 检查是否有RUN号码（可选）
                    # Fair胚胎的图片路径会被记录，焦面信息需要从目录名提取
                    all_paths.append(str(img_file))
                    # Fair胚胎的标签设为-1（表示未知），后续可以根据需要调整
                    all_labels.append(-1)

    print(f"\n📊 匹配统计:")
    print(f"  匹配到的Fair胚胎目录数: {matched_count}")
    print(f"✅ 找到 {len(all_paths)} 张Fair胚胎图片")

    return all_paths, all_labels


def load_all_cache_data(cache_dir="./batch_cache", fair_embryo_ids=None):
    """
    加载所有缓存数据（训练集+测试集），可选筛选Fair胚胎

    Args:
        cache_dir: 缓存目录路径
        fair_embryo_ids: Fair胚胎ID集合，如果提供则只加载Fair胚胎

    Returns:
        (图片路径列表, 标签列表)
    """
    cache_path = Path(cache_dir)
    index_file = cache_path / "index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"缓存索引文件不存在: {index_file}")

    with open(index_file, 'r') as f:
        index = json.load(f)

    if not index:
        raise ValueError("缓存索引为空")

    all_paths = []
    all_labels = []
    total_caches = len(index)
    fair_filtered_count = 0
    total_processed = 0

    print(f"找到 {total_caches} 个缓存集")
    if fair_embryo_ids:
        print(f"将只加载Fair胚胎的图片 (共 {len(fair_embryo_ids)} 个Fair胚胎)")

    # 加载所有缓存集的数据
    for cache_idx, (cache_key, cache_info) in enumerate(index.items(), 1):
        split_type = cache_info.get('split', 'unknown')
        num_images = cache_info.get('total_images', 0)
        print(f"\n[{cache_idx}/{total_caches}] 加载缓存: {split_type} ({num_images} 张图片)")

        chunks = cache_info.get('chunks', [])
        for chunk_info in chunks:
            chunk_file = cache_path / chunk_info['filename']

            if not chunk_file.exists():
                print(f"  警告: 缓存文件不存在: {chunk_file}")
                continue

            try:
                chunk_data = torch.load(chunk_file, map_location='cpu')
                chunk_paths = chunk_data['paths']
                chunk_labels = chunk_data['labels'].tolist()

                # 如果提供了Fair胚胎列表，进行筛选
                if fair_embryo_ids:
                    filtered_paths = []
                    filtered_labels = []
                    for path, label in zip(chunk_paths, chunk_labels):
                        total_processed += 1
                        # 从路径中提取胚胎ID
                        embryo_id = extract_embryo_id(Path(path).name)
                        if embryo_id in fair_embryo_ids:
                            filtered_paths.append(path)
                            filtered_labels.append(label)
                            fair_filtered_count += 1

                    all_paths.extend(filtered_paths)
                    all_labels.extend(filtered_labels)
                    print(f"  ✓ 加载块: {chunk_info['filename']} (筛选出 {len(filtered_paths)}/{len(chunk_paths)} 张Fair胚胎图片)")
                else:
                    all_paths.extend(chunk_paths)
                    all_labels.extend(chunk_labels)
                    print(f"  ✓ 加载块: {chunk_info['filename']} ({len(chunk_paths)} 张)")

            except Exception as e:
                print(f"  ✗ 加载失败: {chunk_info['filename']}, 错误: {e}")
                continue

    if fair_embryo_ids:
        print(f"\n✅ 从 {total_processed} 张图片中筛选出 {len(all_paths)} 张Fair胚胎图片")
    else:
        print(f"\n✅ 总共加载 {len(all_paths)} 张图片（来自所有缓存集）")

    return all_paths, all_labels


def extract_focal_plane_from_filename(filename_or_path):
    """
    从文件名或路径中提取焦面信息

    支持两种格式:
    1. 文件名中包含焦面: D2023.05.31_S00744_I4424_P_WELL12_RUN595_F-15.jpeg -> -15
    2. 目录路径中包含焦面: /path/images/F-15/filename.jpeg -> -15

    Args:
        filename_or_path: 文件名或完整路径字符串

    Returns:
        焦面数值（字符串形式，保留负号）
    """
    # 先尝试从文件名中提取 _F{数字} 格式
    match = re.search(r'_F(-?\d+)', str(filename_or_path))
    if match:
        return match.group(1)

    # 如果文件名中没有，尝试从路径中提取 /F{数字}/ 格式
    match = re.search(r'/F(-?\d+)/', str(filename_or_path))
    if match:
        return match.group(1)

    # Windows路径格式
    match = re.search(r'\\F(-?\d+)\\', str(filename_or_path))
    if match:
        return match.group(1)

    return "Unknown"


def extract_embryo_id(filename):
    """
    从文件名中提取胚胎ID
    例如: D2023.05.31_S00744_I4424_P_WELL12_RUN595_F-15.jpeg
          -> D2023.05.31_S00744_I4424_P_WELL12

    Args:
        filename: 文件名字符串

    Returns:
        胚胎ID字符串
    """
    # 提取从 D 开始到 WELL 后面数字的部分
    match = re.search(r'(D\d{4}\.\d{2}\.\d{2}_S\d+_I\d+_P_WELL\d+)', filename)
    if match:
        return match.group(1)
    # 如果没有匹配到，返回完整文件名用于排序
    return filename


def group_images_by_embryo(image_paths, labels, required_focal_planes=11, filter_label=None):
    """
    将图片按胚胎分组，并筛选出含有完整焦面的胚胎

    Args:
        image_paths: 图片路径列表
        labels: 标签列表
        required_focal_planes: 要求的焦面数量（默认11）
        filter_label: 筛选特定标签的胚胎 (None=全部, 0=Non-viable, 1=Viable, "fair"=Fair)

    Returns:
        embryo_dict: {embryo_id: {'paths': [...], 'label': 0/1, 'focal_planes': [...]}}
    """
    embryo_dict_all = defaultdict(lambda: {'paths': [], 'labels': [], 'focal_planes': []})

    print("\n按胚胎分组图片...")
    for path, label in zip(image_paths, labels):
        filename = Path(path).name
        embryo_id = extract_embryo_id(filename)
        # 传递完整路径以支持从目录结构中提取焦面信息
        focal_plane = extract_focal_plane_from_filename(path)

        embryo_dict_all[embryo_id]['paths'].append(path)
        embryo_dict_all[embryo_id]['labels'].append(label)
        embryo_dict_all[embryo_id]['focal_planes'].append(focal_plane)

    print(f"共找到 {len(embryo_dict_all)} 个不同的胚胎")

    # 显示所有胚胎的焦面数量分布
    focal_counts_all = [len(data['paths']) for data in embryo_dict_all.values()]
    print(f"所有胚胎的焦面数统计:")
    print(f"  平均: {sum(focal_counts_all)/len(focal_counts_all):.1f} 个焦面/胚胎")
    print(f"  最少: {min(focal_counts_all)} 个焦面")
    print(f"  最多: {max(focal_counts_all)} 个焦面")

    # 筛选出含有完整焦面的胚胎
    filter_text = f"标签为 {filter_label} 且" if filter_label is not None else ""
    print(f"\n筛选{filter_text}含有完整 {required_focal_planes} 个焦面的胚胎...")
    embryo_dict = {}

    for embryo_id, data in embryo_dict_all.items():
        num_focal_planes = len(data['paths'])

        if num_focal_planes == required_focal_planes:
            # 确定胚胎的统一标签（大多数投票）
            labels_list = data['labels']
            data['label'] = max(set(labels_list), key=labels_list.count)

            # 如果指定了标签筛选
            if filter_label is not None:
                if filter_label == "fair":
                    # Fair胚胎的特殊处理 - 检查路径中是否包含fair标记
                    path_lower = str(data['paths'][0]).lower()
                    if 'fair' not in path_lower:
                        continue
                elif isinstance(filter_label, int) and data['label'] != filter_label:
                    continue

            embryo_dict[embryo_id] = data

    print(f"✅ 找到 {len(embryo_dict)} 个{filter_text}含有完整 {required_focal_planes} 个焦面的胚胎")

    if len(embryo_dict) == 0:
        print("❌ 警告: 没有找到符合条件的胚胎！")
        print("\n焦面数量分布:")
        focal_count_dist = Counter(focal_counts_all)
        for count, num_embryos in sorted(focal_count_dist.items()):
            print(f"  {count} 个焦面: {num_embryos} 个胚胎")

        # 显示标签分布
        if filter_label is not None:
            print("\n提示: 请检查数据中是否存在指定标签的胚胎")
    else:
        # 统计完整胚胎中的质量分布
        viable_count = sum(1 for data in embryo_dict.values() if data['label'] == 1)
        non_viable_count = len(embryo_dict) - viable_count
        print(f"完整焦面胚胎的质量分布:")
        print(f"  Viable (可行): {viable_count} ({viable_count/len(embryo_dict)*100:.1f}%)")
        print(f"  Non-viable (非可行): {non_viable_count} ({non_viable_count/len(embryo_dict)*100:.1f}%)")

    return embryo_dict


def extract_random_embryos(embryo_dict, num_embryos=100, output_dir="./extracted_samples", random_seed=42):
    """
    随机提取胚胎及其所有焦面

    Args:
        embryo_dict: 胚胎字典
        num_embryos: 要提取的胚胎数量
        output_dir: 输出目录
        random_seed: 随机种子

    Returns:
        CSV数据的DataFrame
    """
    # 设置随机种子
    random.seed(random_seed)

    # 检查胚胎数量
    total_embryos = len(embryo_dict)
    if num_embryos > total_embryos:
        print(f"警告: 请求的胚胎数 ({num_embryos}) 超过可用胚胎数 ({total_embryos})")
        num_embryos = total_embryos

    # 随机选择胚胎
    all_embryo_ids = list(embryo_dict.keys())
    random.shuffle(all_embryo_ids)
    selected_embryo_ids = all_embryo_ids[:num_embryos]

    # 按胚胎ID排序
    selected_embryo_ids.sort()

    print(f"\n已随机选择 {num_embryos} 个胚胎")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 准备CSV数据
    csv_data = []
    total_images = 0
    successful_copies = 0

    print(f"\n开始提取胚胎及其所有焦面到目录: {output_dir}")
    print("=" * 80)

    for embryo_idx, embryo_id in enumerate(selected_embryo_ids, 1):
        embryo_data = embryo_dict[embryo_id]
        embryo_label = embryo_data['label']
        quality_label = "Viable" if embryo_label == 1 else "Non-viable"

        paths = embryo_data['paths']
        focal_planes = embryo_data['focal_planes']

        if embryo_idx % 20 == 0 or embryo_idx == 1:
            print(f"胚胎 {embryo_idx}/{num_embryos}: {embryo_id} ({quality_label}, {len(paths)} 个焦面)")

        # 为每个胚胎创建单独的目录
        embryo_output_dir = output_path / embryo_id
        embryo_output_dir.mkdir(parents=True, exist_ok=True)

        # 复制该胚胎的所有焦面图片
        for focal_idx, (src_path, focal_plane) in enumerate(zip(paths, focal_planes), 1):
            src_path = Path(src_path)

            if not src_path.exists():
                print(f"  警告: 原始图片不存在: {src_path}")
                continue

            # 为每个焦面创建子目录
            focal_output_dir = embryo_output_dir / f"F{focal_plane}"
            focal_output_dir.mkdir(parents=True, exist_ok=True)

            # 使用原始文件名保存到焦面目录
            original_filename = src_path.name
            dst_path = focal_output_dir / original_filename

            try:
                shutil.copy2(src_path, dst_path)
                successful_copies += 1
                total_images += 1

                # 记录到CSV
                csv_data.append({
                    'Embryo_Index': embryo_idx,
                    'Embryo_ID': embryo_id,
                    'Focal_Index': focal_idx,
                    'Focal_Plane': focal_plane,
                    'Filename': original_filename,
                    'Quality': quality_label,
                    'Label': embryo_label,
                    'Original_Path': str(src_path),
                    'Saved_Path': str(dst_path)
                })

            except Exception as e:
                print(f"  错误: 无法复制 {src_path.name}: {e}")
                continue

    # 创建DataFrame
    df = pd.DataFrame(csv_data)

    print("\n" + "=" * 80)
    print("提取完成!")
    print("=" * 80)
    print(f"✅ 成功提取: {len(selected_embryo_ids)} 个胚胎")
    print(f"✅ 成功复制: {successful_copies}/{total_images} 张图片")

    # 统计信息
    if not df.empty:
        # 按胚胎统计
        unique_embryos = df['Embryo_ID'].unique()
        viable_embryos = len(df[df['Label'] == 1]['Embryo_ID'].unique())
        non_viable_embryos = len(unique_embryos) - viable_embryos

        print(f"\n📊 胚胎质量分布:")
        print(f"  Viable (可行): {viable_embryos} ({viable_embryos/len(unique_embryos)*100:.1f}%)")
        print(f"  Non-viable (非可行): {non_viable_embryos} ({non_viable_embryos/len(unique_embryos)*100:.1f}%)")

        # 焦面统计
        focal_planes_list = df[df['Focal_Plane'] != 'Unknown']['Focal_Plane'].tolist()
        if focal_planes_list:
            focal_distribution = Counter(focal_planes_list)
            print(f"\n📊 焦面分布 (共 {len(df)} 张图片):")
            sorted_planes = sorted(focal_distribution.items(),
                                 key=lambda x: int(x[0]) if x[0].lstrip('-').isdigit() else 0)
            for plane, count in sorted_planes:
                print(f"  焦面 F{plane}: {count} ({count/len(df)*100:.1f}%)")

        # 每个胚胎的焦面数统计
        focal_counts_per_embryo = df.groupby('Embryo_ID')['Focal_Plane'].count()
        print(f"\n📊 每个胚胎的焦面数统计:")
        print(f"  平均: {focal_counts_per_embryo.mean():.1f} 个焦面/胚胎")
        print(f"  最少: {focal_counts_per_embryo.min()} 个焦面")
        print(f"  最多: {focal_counts_per_embryo.max()} 个焦面")

    return df


def save_csv_report(df, output_dir="./extracted_samples", filename="embryos_all_focal_planes.csv"):
    """
    保存CSV报告

    Args:
        df: DataFrame
        output_dir: 输出目录
        filename: CSV文件名
    """
    output_path = Path(output_dir)
    csv_path = output_path / filename

    # 保存CSV
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 CSV报告已保存到: {csv_path}")

    # 显示前几行
    print("\n📋 CSV内容预览 (前10行):")
    preview_df = df.head(10)[['Embryo_Index', 'Embryo_ID', 'Focal_Plane', 'Filename', 'Quality']]
    print(preview_df.to_string(index=False))

    print(f"\n💾 CSV文件包含 {len(df)} 行记录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从所有缓存中随机提取胚胎样本（仅含完整焦面的胚胎）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 提取100个含有完整11个焦面的胚胎（所有质量等级）
  python extract_samples.py

  # 提取100个Fair胚胎（需要MainDB.csv）
  python extract_samples.py --fair-only --maindb-path ./MainDB.csv

  # 提取200个Fair胚胎到自定义目录
  python extract_samples.py --fair-only --num-embryos 200 --output-dir ./fair_embryos

  # 要求胚胎必须有15个焦面
  python extract_samples.py --required-focal-planes 15

  # 使用不同的随机种子
  python extract_samples.py --random-seed 123
        """)

    parser.add_argument('--cache-dir', type=str, default='./batch_cache',
                        help='缓存目录路径 (默认: ./batch_cache)')
    parser.add_argument('--data-dir', type=str, default='/liaozizhuo/selected_images',
                        help='原始数据目录路径（用于Fair胚胎）(默认: /liaozizhuo/selected_images)')
    parser.add_argument('--output-dir', type=str, default='./extracted_samples',
                        help='输出目录路径 (默认: ./extracted_samples)')
    parser.add_argument('--num-embryos', type=int, default=100,
                        help='要提取的胚胎数量 (默认: 100)')
    parser.add_argument('--required-focal-planes', type=int, default=11,
                        help='要求的焦面数量 (默认: 11)')
    parser.add_argument('--fair-only', action='store_true',
                        help='只提取Fair胚胎（需要配合--maindb-path和--data-dir使用）')
    parser.add_argument('--maindb-path', type=str, default='/Users/lzz/Documents/GitHub/IVF_AI/MainDB.csv',
                        help='MainDB.csv文件路径 (默认: /Users/lzz/Documents/GitHub/IVF_AI/MainDB.csv)')
    parser.add_argument('--csv-filename', type=str, default='embryos_all_focal_planes.csv',
                        help='CSV文件名 (默认: embryos_all_focal_planes.csv)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    try:
        embryo_type_text = "Fair胚胎 " if args.fair_only else ""
        print("=" * 80)
        print(f"胚胎样本提取工具 (仅含完整 {args.required_focal_planes} 个焦面的{embryo_type_text}胚胎)")
        print("=" * 80)

        # 如果指定了fair-only，加载Fair胚胎列表
        fair_embryo_ids = None
        if args.fair_only:
            print(f"\n📋 加载Fair胚胎列表: {args.maindb_path}")
            fair_embryo_ids = load_fair_embryo_ids(args.maindb_path)
            if not fair_embryo_ids:
                print("❌ 错误: 无法加载Fair胚胎列表")
                return 1

        # 加载数据：Fair胚胎从原始数据目录，其他从缓存
        if args.fair_only:
            print(f"\n📂 原始数据目录: {args.data_dir}")
            image_paths, labels = load_images_from_data_dir(args.data_dir, fair_embryo_ids)
        else:
            print(f"\n📂 缓存目录: {args.cache_dir}")
            image_paths, labels = load_all_cache_data(args.cache_dir, None)

        if not image_paths:
            print("错误: 未能加载任何数据")
            return 1

        # 按胚胎分组（只保留含有完整焦面的胚胎）
        embryo_dict = group_images_by_embryo(image_paths, labels, args.required_focal_planes, None)

        if not embryo_dict:
            print(f"\n❌ 错误: 没有找到含有完整 {args.required_focal_planes} 个焦面的{embryo_type_text}胚胎")
            print("请尝试调整 --required-focal-planes 参数")
            return 1

        # 检查是否有足够的胚胎
        available_embryos = len(embryo_dict)
        if args.num_embryos > available_embryos:
            print(f"\n⚠️  警告: 请求 {args.num_embryos} 个胚胎，但只有 {available_embryos} 个符合条件")
            print(f"将提取所有 {available_embryos} 个胚胎")

        # 提取随机胚胎及其所有焦面
        df = extract_random_embryos(
            embryo_dict,
            num_embryos=args.num_embryos,
            output_dir=args.output_dir,
            random_seed=args.random_seed
        )

        if df.empty:
            print("错误: 未能提取任何样本")
            return 1

        # 保存CSV报告
        save_csv_report(df, args.output_dir, args.csv_filename)

        print("\n" + "=" * 80)
        print("✅ 提取完成!")
        print(f"📁 图片保存在: {args.output_dir}")
        print(f"📄 CSV文件: {Path(args.output_dir) / args.csv_filename}")
        if args.fair_only:
            print(f"🏷️  胚胎类型: Fair胚胎（未训练）")
            print(f"📂 数据来源: 原始数据目录")
        else:
            print(f"📂 数据来源: 缓存")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
