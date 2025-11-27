"""
胚胎质量统计脚本
统计所有来源中的胚胎质量分布: poor, fair, good

Author: Claude Code
Date: 2025-11-21
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import json


def get_embryo_quality_from_maindb(maindb_path='MainDB.csv'):
    """从 MainDB.csv 加载胚胎质量信息"""
    if not os.path.exists(maindb_path):
        print(f"❌ 文件不存在: {maindb_path}")
        return {}

    try:
        df = pd.read_csv(maindb_path)
        quality_dict = {}

        # 自动检测胚胎 ID 列（可能是 RowName, EmbryoID 等）
        embryo_col = None
        for col in df.columns:
            if col.lower() in ['rowname', 'embryoid', 'embryo_id']:
                embryo_col = col
                break

        if not embryo_col:
            print(f"❌ 未找到胚胎 ID 列，可用列: {list(df.columns)}")
            return {}

        for idx, row in df.iterrows():
            # 读取胚胎 ID
            embryo_id_raw = str(row[embryo_col]).strip()
            # 规范化胚胎 ID（统一日期分隔符）
            embryo_id = normalize_embryo_id(embryo_id_raw)
            quality = str(row['Quality']).lower().strip()
            quality_dict[embryo_id] = quality

        return quality_dict
    except Exception as e:
        print(f"❌ 读取 MainDB.csv 出错: {e}")
        return {}


def normalize_embryo_id(embryo_id_str):
    """
    规范化胚胎 ID 格式
    处理两种格式:
    - D2022_04_15_S00003_I4425_P_8 (selected_images 中的格式，下划线分隔)
    - D2022.04.15_S00003_I4425_P_8 (MainDB 中的格式，点分隔日期)
    都转换为统一格式: D2022_04_15_S00003_I4425_P_8
    """
    embryo_id_str = str(embryo_id_str).strip()

    # 统一日期分隔符：将点替换为下划线
    normalized = embryo_id_str.replace('.', '_')

    return normalized


def count_embryos_with_images(selected_images_path='../selected_images', maindb_embryos=None):
    """
    统计 selected_images 中有实际图片的胚胎，以及它们在 MainDB 中的质量分类
    逻辑：
    - good_with_images: selected_images 中存在且在 MainDB 中质量为 'good' 的胚胎
    - fair_with_images: selected_images 中存在且在 MainDB 中质量为 'fair' 的胚胎
    - poor_with_images: selected_images 中存在但不在 MainDB 中的胚胎（没有质量标签）
    """
    if maindb_embryos is None:
        maindb_embryos = {}

    embryos_with_images = {
        'good': [],
        'fair': [],
        'poor': []
    }

    if os.path.exists(selected_images_path):
        # 遍历 selected_images 目录，找出所有有图片的胚胎
        for item in os.listdir(selected_images_path):
            item_path = os.path.join(selected_images_path, item)
            # 只处理目录 (每个目录代表一个胚胎)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # 规范化胚胎 ID
                embryo_id = normalize_embryo_id(item)

                # 检查这个胚胎在 MainDB 中的质量
                if embryo_id in maindb_embryos:
                    quality = maindb_embryos[embryo_id]
                    embryos_with_images[quality].append(embryo_id)
                else:
                    # 不在 MainDB 中的胚胎 = poor
                    embryos_with_images['poor'].append(embryo_id)

    return {
        'good_with_images': embryos_with_images['good'],
        'fair_with_images': embryos_with_images['fair'],
        'poor_with_images': embryos_with_images['poor']
    }


def main(maindb_path='./MainDB.csv'):
    print("\n" + "="*80)
    print("胚胎质量统计 (Embryo Quality Statistics)")
    print("="*80)

    # 1. 从 MainDB.csv 统计
    print("\n【步骤 1】加载 MainDB.csv...")
    print("-" * 80)
    print(f"  使用路径: {maindb_path}")

    quality_dict = get_embryo_quality_from_maindb(maindb_path)

    if quality_dict:
        maindb_stats = defaultdict(int)
        maindb_embryo_ids = set()

        for embryo_id, quality in quality_dict.items():
            maindb_stats[quality] += 1
            maindb_embryo_ids.add(embryo_id)

        print(f"✓ 成功加载 {len(quality_dict)} 条记录\n")
        print("MainDB.csv 质量分布:")
        total_maindb = sum(maindb_stats.values())
        for quality in sorted(maindb_stats.keys()):
            count = maindb_stats[quality]
            pct = (count / total_maindb) * 100
            quality_display = quality.capitalize() if quality else 'Unknown'
            print(f"  {quality_display:12s}: {count:6,d} ({pct:6.2f}%)")
        print(f"  {'Total':12s}: {total_maindb:6,d} (100.00%)")
    else:
        print("❌ 无法加载 MainDB.csv")
        maindb_stats = defaultdict(int)
        maindb_embryo_ids = set()

    # 2. 统计 selected_images 中有实际图片的胚胎
    print("\n【步骤 2】统计 selected_images 中有实际图片的胚胎...")
    print("-" * 80)

    images_stats = count_embryos_with_images(
        selected_images_path='../selected_images',
        maindb_embryos=quality_dict
    )

    good_count = len(images_stats['good_with_images'])
    fair_count = len(images_stats['fair_with_images'])
    poor_count = len(images_stats['poor_with_images'])

    print(f"  ✓ 有图片的 Good 胚胎: {good_count:,d}")
    print(f"  ✓ 有图片的 Fair 胚胎: {fair_count:,d}")
    print(f"  ✓ 有图片的 Poor 胚胎: {poor_count:,d}")

    # 3. 生成综合统计
    print("\n【步骤 3】综合统计结果...")
    print("-" * 80)

    combined_stats = {
        'good': good_count,
        'fair': fair_count,
        'poor': poor_count
    }

    print("\n最终胚胎质量分布:")
    total = sum(combined_stats.values())

    if total > 0:
        for quality in ['poor', 'fair', 'good']:
            count = combined_stats[quality]
            pct = (count / total) * 100
            quality_display = quality.capitalize()
            print(f"  {quality_display:12s}: {count:7,d} ({pct:6.2f}%)")
        print(f"  {'Total':12s}: {total:7,d} (100.00%)")
    else:
        print("  ❌ 未找到任何胚胎数据")

    # 4. 详细说明
    print("\n【数据来源说明】")
    print("-" * 80)
    print(f"""
统计的是 selected_images 中有实际图片的胚胎：

Good (优良) 胚胎:
  - selected_images 中有图片，且在 MainDB 中质量为 'good': {good_count:,d}

Fair (一般) 胚胎:
  - selected_images 中有图片，且在 MainDB 中质量为 'fair': {fair_count:,d}

Poor (差) 胚胎:
  - selected_images 中有图片，但不在 MainDB 中: {poor_count:,d}
""")

    # 5. 保存统计结果
    print("\n【步骤 5】保存结果...")
    print("-" * 80)

    output_file = 'embryo_quality_stats.json'
    stats_to_save = {
        'total_embryos_with_images': total,
        'quality_distribution': {k: int(v) for k, v in combined_stats.items()},
        'percentages': {
            k: round((v / total) * 100, 2) if total > 0 else 0
            for k, v in combined_stats.items()
        },
        'maindb_total': {
            'good': int(maindb_stats.get('good', 0)),
            'fair': int(maindb_stats.get('fair', 0)),
            'total': len(quality_dict)
        },
        'images_statistics': {
            'good_with_images': good_count,
            'fair_with_images': fair_count,
            'poor_with_images': poor_count
        }
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存结果出错: {e}")

    print("\n" + "="*80 + "\n")

    return combined_stats


if __name__ == '__main__':
    import sys

    # 支持命令行参数指定 MainDB.csv 路径
    maindb_path = sys.argv[1] if len(sys.argv) > 1 else './MainDB.csv'

    stats = main(maindb_path=maindb_path)
