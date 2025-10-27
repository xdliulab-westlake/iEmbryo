#!/usr/bin/env python3
"""
生成 score_dict.json 文件的工具 (ViT-Hybrid 版)
从 MainDB.csv 和可选的 filtered_clean_embryos.csv 生成评分字典
支持与 BiLSTM 相同的图像读取逻辑，但不需要年龄标签
"""

import os
import json
import random
import argparse
import pandas as pd
from pathlib import Path


def generate_score_dict(maindb_path, filtered_path=None, output_path="score_dict.json",
                       image_dir=None, verbose=True):
    """
    从CSV文件生成评分字典并保存为JSON

    Args:
        maindb_path (str): MainDB.csv 文件路径
        filtered_path (str, optional): filtered_clean_embryos.csv 文件路径
        output_path (str): 输出的 score_dict.json 文件路径
        image_dir (str, optional): 图像目录路径，用于验证图像存在
        verbose (bool): 是否打印详细信息

    Returns:
        dict: 生成的评分字典
    """
    if verbose:
        print("=== ViT-Hybrid 评分字典生成工具 ===")

    # 检查主数据文件
    if not os.path.exists(maindb_path):
        raise FileNotFoundError(f"主数据文件不存在: {maindb_path}")

    if verbose:
        print(f"📂 读取主数据文件: {maindb_path}")
    df = pd.read_csv(maindb_path)
    if verbose:
        print(f"✅ 成功读取 {len(df)} 行数据")
        print(f"📋 数据列: {list(df.columns)}")

    # ICM和TE评分映射
    icm_map = {'A': 1, 'B': 2, 'C': 4}
    te_map = {'A': 1, 'B': 3, 'C': 4}

    # 质量标签映射
    quality_map = {'good': 1, 'poor': 0, 'Good': 1, 'Poor': 0}

    score_dict = {}

    # 使用ICM/TE评分计算BS_prime
    if 'ICM' in df.columns and 'TE' in df.columns:
        if verbose:
            print("📊 使用 ICM/TE 评分计算 BS_prime")
        df["ICM_score"] = df["ICM"].map(icm_map)
        df["TE_score"] = df["TE"].map(te_map)
        df["BS_prime"] = df["ICM_score"] + df["TE_score"]

        # 统计质量标签分布
        if 'Quality' in df.columns and verbose:
            quality_counts = df['Quality'].value_counts()
            print(f"📋 原始质量标签分布: {dict(quality_counts)}")

        # 统计去除情况
        removed_missing_quality = 0
        removed_unknown_quality = 0
        removed_no_image = 0

        # 检查图像目录是否存在
        image_dir_path = None
        if image_dir:
            image_dir_path = Path(image_dir)
            if not image_dir_path.exists():
                image_dir_path = None
                if verbose:
                    print(f"⚠️ 图像目录不存在: {image_dir}")

        # 创建评分字典
        for _, row in df.iterrows():
            embryo_id = str(row["EmbryoID"])

            # 检查Quality列是否存在且有效
            if "Quality" not in df.columns or pd.isna(row["Quality"]):
                if verbose:
                    print(f"❌ 缺失质量标签，去除胚胎: {embryo_id}")
                removed_missing_quality += 1
                continue

            quality_raw = str(row["Quality"]).strip()
            if quality_raw not in quality_map:
                if verbose:
                    print(f"❌ 未知质量标签 '{quality_raw}'，去除胚胎: {embryo_id}")
                removed_unknown_quality += 1
                continue

            # 如果提供了图像目录，检查图像是否存在
            if image_dir_path:
                # 支持多种图像格式
                image_found = False
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    image_path = image_dir_path / f"{embryo_id}{ext}"
                    if image_path.exists():
                        image_found = True
                        break

                if not image_found:
                    if verbose:
                        print(f"❌ 找不到胚胎图像，去除胚胎: {embryo_id}")
                    removed_no_image += 1
                    continue

            # 只有有效的质量标签才会被保留
            bs_score = float(row["BS_prime"])
            quality = quality_map[quality_raw]

            score_dict[embryo_id] = {
                "bs_score": bs_score,
                "quality": quality
            }
    else:
        raise ValueError("MainDB.csv必须包含 ICM 和 TE 列")

    if verbose:
        print(f"\n✅ 从主数据文件生成 {len(score_dict)} 个胚胎评分")
        print(f"📊 数据筛选统计:")
        print(f"   原始胚胎数: {len(df)}")
        print(f"   缺失质量标签: {removed_missing_quality}")
        print(f"   未知质量标签: {removed_unknown_quality}")
        if image_dir_path:
            print(f"   找不到对应图像: {removed_no_image}")
        print(f"   最终保留: {len(score_dict)}")
        if len(df) > 0:
            print(f"   保留率: {len(score_dict)/len(df)*100:.1f}%")

    # 统计最终的标签分布
    quality_stats = {'good': 0, 'poor': 0}
    bs_scores = []
    for embryo_data in score_dict.values():
        if embryo_data['quality'] == 1:
            quality_stats['good'] += 1
        else:
            quality_stats['poor'] += 1
        bs_scores.append(embryo_data['bs_score'])

    if verbose:
        print(f"\n📊 最终质量标签统计:")
        print(f"   Good (优质): {quality_stats['good']} ({quality_stats['good']/len(score_dict)*100:.1f}%)")
        print(f"   Poor (低质): {quality_stats['poor']} ({quality_stats['poor']/len(score_dict)*100:.1f}%)")
        print(f"📊 BS分数统计:")
        if bs_scores:
            print(f"   范围: {min(bs_scores):.1f} - {max(bs_scores):.1f}")
            print(f"   平均: {sum(bs_scores)/len(bs_scores):.2f}")

    # 处理低质量胚胎数据（如果提供）
    if filtered_path and os.path.exists(filtered_path):
        if verbose:
            print(f"\n📂 读取低质量胚胎文件: {filtered_path}")
        filtered_df = pd.read_csv(filtered_path)
        if verbose:
            print(f"✅ 成功读取 {len(filtered_df)} 行低质量胚胎数据")
            print(f"📋 低质量数据列: {list(filtered_df.columns)}")

        # 查找低质量胚胎
        if "EmbryoQuality" in filtered_df.columns:
            poor_embryos = filtered_df[filtered_df["EmbryoQuality"] == "poor"]
            if verbose:
                print(f"🔍 找到 {len(poor_embryos)} 个低质量胚胎")

            added_count = 0
            for emb_id in poor_embryos["EmbryoID"].unique():
                emb_id_str = str(emb_id)
                if emb_id_str not in score_dict:
                    # 检查是否有对应的图像
                    if image_dir_path:
                        image_found = False
                        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                            image_path = image_dir_path / f"{emb_id_str}{ext}"
                            if image_path.exists():
                                image_found = True
                                break
                        if not image_found:
                            continue

                    # 为低质量胚胎分配较高的BS分数 (6.0-6.5)
                    bs_score = 6.0 + random.uniform(0, 0.5)
                    quality = 0  # 低质量

                    score_dict[emb_id_str] = {
                        "bs_score": bs_score,
                        "quality": quality
                    }
                    added_count += 1

            if verbose:
                print(f"✅ 为 {added_count} 个低质量胚胎添加了评分数据")
        else:
            if verbose:
                print("⚠️ 低质量胚胎文件中没有 EmbryoQuality 列")
    elif filtered_path and verbose:
        print(f"⚠️ 低质量胚胎文件不存在: {filtered_path}")

    # 保存为JSON文件
    if verbose:
        print(f"\n💾 保存评分字典到: {output_path}")

    # 创建输出目录
    output_dir = Path(output_path).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n🎉 评分字典生成完成!")
        print(f"📁 保存在: {Path(output_path).absolute()}")
        print(f"📊 总胚胎数: {len(score_dict)}")
        if quality_stats['good'] + quality_stats['poor'] > 0:
            print(f"   Good: {quality_stats['good']} ({quality_stats['good']/(quality_stats['good']+quality_stats['poor'])*100:.1f}%)")
            print(f"   Poor: {quality_stats['poor']} ({quality_stats['poor']/(quality_stats['good']+quality_stats['poor'])*100:.1f}%)")

    return score_dict


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="ViT-Hybrid: 生成 score_dict.json 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基础用法 - 只使用 MainDB.csv
  python generate_score_dict.py --maindb MainDB.csv

  # 包含低质量胚胎数据
  python generate_score_dict.py --maindb MainDB.csv --filtered filtered_clean_embryos.csv

  # 指定图像目录并验证图像存在
  python generate_score_dict.py --maindb MainDB.csv --image_dir ./image

  # 完整用法
  python generate_score_dict.py -m MainDB.csv -f filtered.csv -i ./image -o my_score_dict.json
        """
    )
    parser.add_argument("--maindb", "-m", required=True,
                       help="MainDB.csv 文件路径")
    parser.add_argument("--filtered", "-f", default=None,
                       help="filtered_clean_embryos.csv 文件路径（可选）")
    parser.add_argument("--image_dir", "-i", default=None,
                       help="图像目录路径，用于验证图像存在（可选）")
    parser.add_argument("--output", "-o", default="score_dict.json",
                       help="输出文件路径（默认: score_dict.json）")

    args = parser.parse_args()

    try:
        generate_score_dict(
            args.maindb,
            args.filtered,
            args.output,
            args.image_dir,
            verbose=True
        )
        return 0
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exit(main())
    else:
        print("=== ViT-Hybrid 评分字典生成工具 ===")
        print("用法示例:")
        print("  python generate_score_dict.py --maindb MainDB.csv")
        print("  python generate_score_dict.py -m MainDB.csv -f filtered.csv -i ./image -o my_score_dict.json")
        print()
        print("请提供参数运行脚本，或者直接调用 generate_score_dict() 函数")
