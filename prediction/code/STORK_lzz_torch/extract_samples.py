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


def load_embryo_quality_from_maindb(csv_path="../MainDB.csv"):
    """
    从MainDB.csv加载所有胚胎的质量标签

    Args:
        csv_path: MainDB.csv文件路径

    Returns:
        dict: {embryo_id: quality_label} 其中 quality_label 为 'good', 'poor', 或 'fair'
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

        # 创建胚胎ID到质量标签的映射
        embryo_quality = {}
        for _, row in df.iterrows():
            embryo_id = row[id_column]
            quality = row['Quality'].lower()  # 转换为小写
            embryo_quality[embryo_id] = quality

        # 统计各质量等级的数量
        quality_counts = Counter(embryo_quality.values())
        print(f"从 {csv_path} 加载了 {len(embryo_quality)} 个胚胎的质量标签 (使用列: {id_column})")
        print(f"  质量分布:")
        for quality, count in sorted(quality_counts.items()):
            print(f"    {quality.capitalize()}: {count} 个胚胎")

        # Debug: 显示一些示例ID
        sample_ids = list(embryo_quality.keys())[:3]
        print(f"  🔍 调试: MainDB中的示例ID:")
        for sid in sample_ids:
            print(f"    {sid} -> {embryo_quality[sid]}")

        return embryo_quality
    except Exception as e:
        print(f"警告: 无法加载胚胎质量标签: {e}")
        print("将使用默认标签")
        import traceback
        traceback.print_exc()
        return None


def load_images_from_data_dir(data_dir, embryo_quality=None, filter_quality=None):
    """
    直接从原始数据目录加载图片路径

    Args:
        data_dir: 原始数据目录路径
        embryo_quality: 胚胎ID到质量标签的映射字典
        filter_quality: 只加载特定质量的胚胎（None=全部, 'good', 'poor', 'fair'）

    Returns:
        (图片路径列表, 标签列表, 胚胎ID列表)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_paths = []
    all_labels = []
    all_embryo_ids = []

    filter_text = f"{filter_quality} " if filter_quality else ""
    print(f"从原始数据目录加载{filter_text}胚胎图片: {data_dir}")

    # Debug: 显示前几个目录名
    all_dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"\n🔍 调试信息:")
    print(f"  数据目录中总共有 {len(all_dirs)} 个子目录")
    print(f"  前5个目录名示例: {all_dirs[:5]}")
    if embryo_quality and filter_quality:
        filtered_ids = [eid for eid, q in embryo_quality.items() if q == filter_quality]
        print(f"  {filter_quality.capitalize()}胚胎ID示例 (前5个): {filtered_ids[:5]}")

    matched_count = 0

    # 遍历数据目录中的所有胚胎文件夹
    for embryo_dir in data_path.iterdir():
        if not embryo_dir.is_dir():
            continue

        embryo_id = embryo_dir.name

        # 如果提供了胚胎质量字典，进行筛选
        if embryo_quality:
            # 直接比较（格式应该一致）
            if embryo_id in embryo_quality:
                if filter_quality and embryo_quality[embryo_id] != filter_quality:
                    continue
                matched_count += 1
                if matched_count <= 3:
                    quality = embryo_quality[embryo_id]
                    print(f"  ✓ 匹配到{quality.capitalize()}胚胎目录: {embryo_id}")
            else:
                # 如果直接匹配失败，尝试格式转换（兼容不同格式）
                # D2024_10_04_S01672_I4425_P_5 <-> D2024.10.04_S01672_I4425_P_5
                embryo_id_alt = embryo_id.replace('_', '.', 2) if '_' in embryo_id[:10] else embryo_id.replace('.', '_', 2)
                if embryo_id_alt in embryo_quality:
                    if filter_quality and embryo_quality[embryo_id_alt] != filter_quality:
                        continue
                    matched_count += 1
                    if matched_count <= 3:
                        quality = embryo_quality[embryo_id_alt]
                        print(f"  ✓ 匹配到{quality.capitalize()}胚胎目录: {embryo_id} (转换后匹配)")
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
            # 文件名不包含焦面信息，焦面信息在目录名中
            # 例如: images/F0/D2024.10.04_S01672_I4425_P_WELL3_RUN500.jpeg
            for img_file in focal_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    all_paths.append(str(img_file))
                    # 标签设为-1（表示从MainDB获取），后续会根据embryo_quality设置
                    all_labels.append(-1)
                    all_embryo_ids.append(embryo_id)

    print(f"\n📊 匹配统计:")
    print(f"  匹配到的{filter_text}胚胎目录数: {matched_count}")
    print(f"✅ 找到 {len(all_paths)} 张{filter_text}胚胎图片")

    return all_paths, all_labels, all_embryo_ids


def load_all_cache_data(cache_dir="./batch_cache", embryo_quality=None, filter_quality=None):
    """
    加载所有缓存数据（训练集+测试集），可选筛选特定质量的胚胎

    Args:
        cache_dir: 缓存目录路径
        embryo_quality: 胚胎ID到质量标签的映射字典
        filter_quality: 只加载特定质量的胚胎（None=全部, 'good', 'poor', 'fair'）

    Returns:
        (图片路径列表, 标签列表, 胚胎ID列表)
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
    all_embryo_ids = []
    total_caches = len(index)
    filtered_count = 0
    total_processed = 0

    filter_text = f"{filter_quality} " if filter_quality else ""
    print(f"找到 {total_caches} 个缓存集")
    if filter_quality and embryo_quality:
        quality_count = sum(1 for q in embryo_quality.values() if q == filter_quality)
        print(f"将只加载{filter_quality.capitalize()}胚胎的图片 (共 {quality_count} 个{filter_quality.capitalize()}胚胎)")

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

                # 如果提供了胚胎质量字典，进行筛选
                if embryo_quality and filter_quality:
                    filtered_paths = []
                    filtered_labels = []
                    filtered_embryo_ids = []

                    # Debug: 显示第一个文件名和提取的ID（仅第一个chunk）
                    if total_processed == 0 and len(chunk_paths) > 0:
                        sample_filename = Path(chunk_paths[0]).name
                        sample_id = extract_embryo_id(sample_filename)
                        print(f"  🔍 调试: 文件名示例: {sample_filename}")
                        print(f"  🔍 调试: 提取的ID: {sample_id}")
                        print(f"  🔍 调试: ID在质量字典中? {sample_id in embryo_quality}")
                        if sample_id in embryo_quality:
                            print(f"  🔍 调试: 该ID的质量: {embryo_quality[sample_id]}")

                    for path, label in zip(chunk_paths, chunk_labels):
                        total_processed += 1
                        # 从路径中提取胚胎ID
                        embryo_id = extract_embryo_id(Path(path).name)
                        if embryo_id in embryo_quality and embryo_quality[embryo_id] == filter_quality:
                            filtered_paths.append(path)
                            filtered_labels.append(label)
                            filtered_embryo_ids.append(embryo_id)
                            filtered_count += 1

                    all_paths.extend(filtered_paths)
                    all_labels.extend(filtered_labels)
                    all_embryo_ids.extend(filtered_embryo_ids)
                    print(f"  ✓ 加载块: {chunk_info['filename']} (筛选出 {len(filtered_paths)}/{len(chunk_paths)} 张{filter_quality.capitalize()}胚胎图片)")
                else:
                    all_paths.extend(chunk_paths)
                    all_labels.extend(chunk_labels)
                    # 提取胚胎ID
                    chunk_embryo_ids = [extract_embryo_id(Path(p).name) for p in chunk_paths]
                    all_embryo_ids.extend(chunk_embryo_ids)
                    print(f"  ✓ 加载块: {chunk_info['filename']} ({len(chunk_paths)} 张)")

            except Exception as e:
                print(f"  ✗ 加载失败: {chunk_info['filename']}, 错误: {e}")
                continue

    if filter_quality and embryo_quality:
        print(f"\n✅ 从 {total_processed} 张图片中筛选出 {len(all_paths)} 张{filter_quality.capitalize()}胚胎图片")
    else:
        print(f"\n✅ 总共加载 {len(all_paths)} 张图片（来自所有缓存集）")

    return all_paths, all_labels, all_embryo_ids


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


def normalize_embryo_id(embryo_id):
    """
    规范化胚胎ID，将WELL格式转换为数字格式以匹配MainDB.csv
    D2023.05.31_S00744_I4424_P_WELL12 -> D2023.05.31_S00744_I4424_P_12
    D2024.03.18_S00006_I5068_P_5 -> D2024.03.18_S00006_I5068_P_5 (保持不变)

    Args:
        embryo_id: 胚胎ID字符串

    Returns:
        规范化后的胚胎ID
    """
    # 将 P_WELL12 格式转换为 P_12 格式
    normalized = re.sub(r'_P_WELL(\d+)', r'_P_\1', embryo_id)
    return normalized


def extract_embryo_id(filename):
    """
    从文件名中提取胚胎ID并规范化
    支持两种格式:
    1. D2023.05.31_S00744_I4424_P_WELL12_RUN595_F-15.jpeg -> D2023.05.31_S00744_I4424_P_12
    2. D2024.03.18_S00006_I5068_P_5 -> D2024.03.18_S00006_I5068_P_5

    Args:
        filename: 文件名字符串

    Returns:
        规范化后的胚胎ID字符串
    """
    # 先尝试提取包含 WELL 的格式
    match = re.search(r'(D\d{4}\.\d{2}\.\d{2}_S\d+_I\d+_P_WELL\d+)', filename)
    if match:
        return normalize_embryo_id(match.group(1))

    # 尝试提取不包含 WELL 的格式 (MainDB.csv中的格式)
    match = re.search(r'(D\d{4}\.\d{2}\.\d{2}_S\d+_I\d+_P_\d+)', filename)
    if match:
        return match.group(1)

    # 如果都没有匹配到，返回完整文件名用于排序
    return filename


def group_images_by_embryo(image_paths, labels, embryo_ids, embryo_quality=None, required_focal_planes=11, filter_quality=None):
    """
    将图片按胚胎分组，并筛选出含有完整焦面的胚胎

    Args:
        image_paths: 图片路径列表
        labels: 标签列表
        embryo_ids: 胚胎ID列表
        embryo_quality: 胚胎ID到质量标签的映射字典
        required_focal_planes: 要求的焦面数量（默认11）
        filter_quality: 筛选特定质量的胚胎 (None=全部, 'good', 'poor', 'fair')

    Returns:
        embryo_dict: {embryo_id: {'paths': [...], 'label': 0/1, 'quality': 'good'/'poor'/'fair', 'focal_planes': [...]}}
    """
    embryo_dict_all = defaultdict(lambda: {'paths': [], 'labels': [], 'focal_planes': []})

    print("\n按胚胎分组图片...")
    for path, label, embryo_id in zip(image_paths, labels, embryo_ids):
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
    filter_text = f"质量为 {filter_quality} 且" if filter_quality is not None else ""
    print(f"\n筛选{filter_text}含有完整 {required_focal_planes} 个焦面的胚胎...")
    embryo_dict = {}

    for embryo_id, data in embryo_dict_all.items():
        num_focal_planes = len(data['paths'])

        if num_focal_planes == required_focal_planes:
            # 确定胚胎的统一标签（大多数投票）
            labels_list = data['labels']
            data['label'] = max(set(labels_list), key=labels_list.count)

            # 从 embryo_quality 字典获取质量标签
            if embryo_quality and embryo_id in embryo_quality:
                data['quality'] = embryo_quality[embryo_id]
            else:
                # 如果没有质量标签，根据 label 推断
                if data['label'] == 1:
                    data['quality'] = 'good'
                elif data['label'] == 0:
                    data['quality'] = 'poor'
                else:
                    data['quality'] = 'unknown'

            # 如果指定了质量筛选
            if filter_quality is not None:
                if data['quality'] != filter_quality:
                    continue

            embryo_dict[embryo_id] = data

    print(f"✅ 找到 {len(embryo_dict)} 个{filter_text}含有完整 {required_focal_planes} 个焦面的胚胎")

    if len(embryo_dict) == 0:
        print("❌ 警告: 没有找到符合条件的胚胎！")
        print("\n焦面数量分布:")
        focal_count_dist = Counter(focal_counts_all)
        for count, num_embryos in sorted(focal_count_dist.items()):
            print(f"  {count} 个焦面: {num_embryos} 个胚胎")

        # 显示质量分布
        if filter_quality is not None:
            print("\n提示: 请检查数据中是否存在指定质量的胚胎")
    else:
        # 统计完整胚胎中的质量分布
        quality_counts = Counter(data['quality'] for data in embryo_dict.values())
        print(f"完整焦面胚胎的质量分布:")
        for quality in ['good', 'poor', 'fair', 'unknown']:
            count = quality_counts.get(quality, 0)
            if count > 0:
                print(f"  {quality.capitalize()}: {count} ({count/len(embryo_dict)*100:.1f}%)")

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
        quality_label = embryo_data.get('quality', 'unknown')

        paths = embryo_data['paths']
        focal_planes = embryo_data['focal_planes']

        if embryo_idx % 20 == 0 or embryo_idx == 1:
            print(f"胚胎 {embryo_idx}/{num_embryos}: {embryo_id} ({quality_label.capitalize()}, {len(paths)} 个焦面)")

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

        # 统计每个质量等级的胚胎数
        quality_counts = {}
        for quality in ['good', 'poor', 'fair', 'unknown']:
            count = len(df[df['Quality'] == quality]['Embryo_ID'].unique())
            if count > 0:
                quality_counts[quality] = count

        print(f"\n📊 胚胎质量分布:")
        for quality, count in quality_counts.items():
            print(f"  {quality.capitalize()}: {count} ({count/len(unique_embryos)*100:.1f}%)")

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
  python extract_samples.py --quality fair --maindb-path ./MainDB.csv

  # 提取200个Good胚胎到自定义目录
  python extract_samples.py --quality good --num-embryos 200 --output-dir ./good_embryos

  # 提取Poor胚胎
  python extract_samples.py --quality poor --maindb-path ./MainDB.csv

  # 每种质量各提取200个胚胎（good/poor/fair各200个）
  python extract_samples.py --num-per-quality 200 --maindb-path ./MainDB.csv

  # 要求胚胎必须有15个焦面
  python extract_samples.py --required-focal-planes 15

  # 使用不同的随机种子
  python extract_samples.py --random-seed 123
        """)

    parser.add_argument('--cache-dir', type=str, default='./batch_cache',
                        help='缓存目录路径 (默认: ./batch_cache)')
    parser.add_argument('--data-dir', type=str, default='/liaozizhuo/selected_images',
                        help='原始数据目录路径（用于从原始数据加载）(默认: /liaozizhuo/selected_images)')
    parser.add_argument('--output-dir', type=str, default='./extracted_samples',
                        help='输出目录路径 (默认: ./extracted_samples)')
    parser.add_argument('--num-embryos', type=int, default=100,
                        help='要提取的胚胎数量 (默认: 100)')
    parser.add_argument('--num-per-quality', type=int, default=None,
                        help='每种质量各提取的胚胎数量（自动提取good/poor/fair各N个），与--quality互斥')
    parser.add_argument('--required-focal-planes', type=int, default=11,
                        help='要求的焦面数量 (默认: 11)')
    parser.add_argument('--quality', type=str, choices=['good', 'poor', 'fair'], default=None,
                        help='只提取特定质量的胚胎 (good/poor/fair)，需要配合--maindb-path使用')
    parser.add_argument('--use-data-dir', action='store_true',
                        help='从原始数据目录加载而不是从缓存加载')
    parser.add_argument('--maindb-path', type=str, default='../MainDB.csv',
                        help='MainDB.csv文件路径 (默认: ../MainDB.csv)')
    parser.add_argument('--csv-filename', type=str, default='embryos_all_focal_planes.csv',
                        help='CSV文件名 (默认: embryos_all_focal_planes.csv)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    try:
        # 检查参数冲突
        if args.num_per_quality and args.quality:
            print("❌ 错误: --num-per-quality 和 --quality 参数不能同时使用")
            return 1

        embryo_type_text = f"{args.quality.capitalize()} " if args.quality else ""
        if args.num_per_quality:
            embryo_type_text = "各质量等级 "

        print("=" * 80)
        print(f"胚胎样本提取工具 (仅含完整 {args.required_focal_planes} 个焦面的{embryo_type_text}胚胎)")
        print("=" * 80)

        # 加载胚胎质量标签（如果提供了MainDB.csv）
        embryo_quality = None
        if args.quality or args.num_per_quality or Path(args.maindb_path).exists():
            print(f"\n📋 加载胚胎质量标签: {args.maindb_path}")
            embryo_quality = load_embryo_quality_from_maindb(args.maindb_path)
            if not embryo_quality:
                print("⚠️  警告: 无法加载胚胎质量标签，将使用默认标签")

        # 如果使用 num_per_quality，需要分别处理每个质量等级
        if args.num_per_quality:
            all_dfs = []
            for quality in ['good', 'poor', 'fair']:
                print(f"\n{'='*80}")
                print(f"处理 {quality.capitalize()} 质量胚胎...")
                print(f"{'='*80}")

                # 加载数据：从原始数据目录或缓存
                if args.use_data_dir:
                    print(f"\n📂 原始数据目录: {args.data_dir}")
                    image_paths, labels, embryo_ids = load_images_from_data_dir(
                        args.data_dir,
                        embryo_quality=embryo_quality,
                        filter_quality=quality
                    )
                else:
                    print(f"\n📂 缓存目录: {args.cache_dir}")
                    image_paths, labels, embryo_ids = load_all_cache_data(
                        args.cache_dir,
                        embryo_quality=embryo_quality,
                        filter_quality=quality
                    )

                if not image_paths:
                    print(f"⚠️  警告: 没有找到 {quality.capitalize()} 质量的图片，跳过...")
                    continue

                # 按胚胎分组
                embryo_dict = group_images_by_embryo(
                    image_paths,
                    labels,
                    embryo_ids,
                    embryo_quality=embryo_quality,
                    required_focal_planes=args.required_focal_planes,
                    filter_quality=quality
                )

                if not embryo_dict:
                    print(f"⚠️  警告: 没有找到含有完整 {args.required_focal_planes} 个焦面的 {quality.capitalize()} 胚胎，跳过...")
                    continue

                # 检查是否有足够的胚胎
                available_embryos = len(embryo_dict)
                num_to_extract = min(args.num_per_quality, available_embryos)
                if args.num_per_quality > available_embryos:
                    print(f"\n⚠️  警告: 请求 {args.num_per_quality} 个胚胎，但只有 {available_embryos} 个符合条件")
                    print(f"将提取所有 {available_embryos} 个 {quality.capitalize()} 胚胎")

                # 提取随机胚胎
                df = extract_random_embryos(
                    embryo_dict,
                    num_embryos=num_to_extract,
                    output_dir=args.output_dir,
                    random_seed=args.random_seed
                )

                if not df.empty:
                    all_dfs.append(df)

            # 合并所有DataFrame
            if not all_dfs:
                print("\n❌ 错误: 未能提取任何样本")
                return 1

            df = pd.concat(all_dfs, ignore_index=True)

            # 重新计算 Embryo_Index，确保索引连续
            embryo_id_to_index = {}
            current_index = 1
            for idx, row in df.iterrows():
                embryo_id = row['Embryo_ID']
                if embryo_id not in embryo_id_to_index:
                    embryo_id_to_index[embryo_id] = current_index
                    current_index += 1
                df.at[idx, 'Embryo_Index'] = embryo_id_to_index[embryo_id]

            print("\n" + "=" * 80)
            print("汇总统计:")
            print("=" * 80)
            unique_embryos = df['Embryo_ID'].unique()
            print(f"✅ 成功提取: {len(unique_embryos)} 个胚胎")
            print(f"✅ 成功复制: {len(df)} 张图片")

            # 按质量统计
            quality_counts = {}
            for quality in ['good', 'poor', 'fair']:
                count = len(df[df['Quality'] == quality]['Embryo_ID'].unique())
                if count > 0:
                    quality_counts[quality] = count

            print(f"\n📊 胚胎质量分布:")
            for quality, count in quality_counts.items():
                print(f"  {quality.capitalize()}: {count} 个胚胎 ({count*args.required_focal_planes} 张图片)")

        else:
            # 单一质量或所有质量的处理
            # 加载数据：从原始数据目录或缓存
            if args.use_data_dir:
                print(f"\n📂 原始数据目录: {args.data_dir}")
                image_paths, labels, embryo_ids = load_images_from_data_dir(
                    args.data_dir,
                    embryo_quality=embryo_quality,
                    filter_quality=args.quality
                )
            else:
                print(f"\n📂 缓存目录: {args.cache_dir}")
                image_paths, labels, embryo_ids = load_all_cache_data(
                    args.cache_dir,
                    embryo_quality=embryo_quality,
                    filter_quality=args.quality
                )

            if not image_paths:
                print("错误: 未能加载任何数据")
                return 1

            # 按胚胎分组（只保留含有完整焦面的胚胎）
            embryo_dict = group_images_by_embryo(
                image_paths,
                labels,
                embryo_ids,
                embryo_quality=embryo_quality,
                required_focal_planes=args.required_focal_planes,
                filter_quality=args.quality
            )

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
        if args.quality:
            print(f"🏷️  胚胎质量: {args.quality.capitalize()}")
        elif args.num_per_quality:
            print(f"🏷️  胚胎质量: 每种质量各 {args.num_per_quality} 个 (Good/Poor/Fair)")
        if args.use_data_dir:
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
