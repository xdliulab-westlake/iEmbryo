#!/usr/bin/env python3
"""
ViT-Hybrid 数据诊断工具
检查数据加载问题
"""

import os
import json
from pathlib import Path

def diagnose_data(data_root):
    """诊断数据目录结构和问题"""

    print("\n" + "=" * 70)
    print("ViT-Hybrid 数据诊断")
    print("=" * 70)

    data_path = Path(data_root)

    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        return False

    print(f"✓ 数据目录存在: {data_root}")
    print(f"✓ 数据目录内容:")

    # 列出目录内容
    try:
        contents = list(data_path.iterdir())
        if not contents:
            print(f"   ⚠️  目录为空！")
            return False

        for item in sorted(contents):
            if item.is_dir():
                print(f"   📁 {item.name}/ (目录)")
            else:
                print(f"   📄 {item.name} (文件)")
    except Exception as e:
        print(f"❌ 无法列出目录: {e}")
        return False

    print("\n" + "-" * 70)
    print("检查数据格式")
    print("-" * 70)

    # 检查 BiLSTM 格式 (image/ + score_dict.json)
    image_dir = data_path / "image"
    score_dict_path = data_path / "score_dict.json"

    bilstm_format = False
    if image_dir.exists():
        image_files = list(image_dir.glob("*.[jJ][pP][gG]")) + \
                      list(image_dir.glob("*.[pP][nN][gG]")) + \
                      list(image_dir.glob("*.[jJ][pP][eE][gG]"))
        print(f"📁 image/ 目录存在")
        print(f"   图像文件数: {len(image_files)}")

        if len(image_files) > 0:
            print(f"   样本:")
            for img in image_files[:5]:
                print(f"     - {img.name}")
            if len(image_files) > 5:
                print(f"     ... 以及 {len(image_files) - 5} 个其他文件")

        if score_dict_path.exists():
            print(f"📄 score_dict.json 存在")
            try:
                with open(score_dict_path, 'r') as f:
                    score_dict = json.load(f)
                print(f"   胚胎数量: {len(score_dict)}")
                if len(score_dict) > 0:
                    first_key = list(score_dict.keys())[0]
                    print(f"   样本数据: {first_key} → {score_dict[first_key]}")
                bilstm_format = True
            except Exception as e:
                print(f"❌ score_dict.json 格式错误: {e}")
        else:
            print(f"❌ score_dict.json 不存在")
    else:
        print(f"❌ image/ 目录不存在")

    # 检查时间序列格式 (good/ + poor/)
    temporal_format = False
    good_dir = data_path / "good"
    poor_dir = data_path / "poor"

    if good_dir.exists() or poor_dir.exists():
        print(f"\n📁 检查时间序列格式 (good/poor):")

        if good_dir.exists():
            good_embryos = [d for d in good_dir.iterdir() if d.is_dir()]
            print(f"   good/ 目录存在")
            print(f"   胚胎数: {len(good_embryos)}")
            if len(good_embryos) > 0:
                first_embryo = good_embryos[0]
                time_dirs = [d for d in first_embryo.iterdir() if d.is_dir() and d.name.startswith('time_')]
                print(f"   样本: {first_embryo.name} 有 {len(time_dirs)} 个时间点")
                temporal_format = True
        else:
            print(f"   ❌ good/ 目录不存在")

        if poor_dir.exists():
            poor_embryos = [d for d in poor_dir.iterdir() if d.is_dir()]
            print(f"   poor/ 目录存在")
            print(f"   胚胎数: {len(poor_embryos)}")
        else:
            print(f"   ❌ poor/ 目录不存在")

    # 总结
    print("\n" + "-" * 70)
    print("诊断总结")
    print("-" * 70)

    if bilstm_format:
        print("✓ 检测到 BiLSTM 格式 (image/ + score_dict.json)")
        print("  使用命令: python main.py train --data_root /liaozizhuo/")

        # 检查是否有匹配的图像和评分
        with open(score_dict_path, 'r') as f:
            score_dict = json.load(f)

        image_names = set(img.name for img in image_dir.glob("*.*"))
        score_keys = set(score_dict.keys())

        print(f"\n  图像数: {len(image_names)}")
        print(f"  评分数: {len(score_keys)}")

        # 检查匹配
        matches = 0
        for key in score_keys:
            # 尝试多种匹配方式
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                if f"{key}{ext}" in image_names:
                    matches += 1
                    break

        print(f"  匹配数: {matches}")

        if matches == 0:
            print(f"  ⚠️  警告：没有找到匹配的图像和评分！")
            print(f"     检查:")
            print(f"     1. 图像名是否与评分字典中的 key 对应")
            print(f"     2. 文件扩展名是否一致")

        return matches > 0

    elif temporal_format:
        print("✓ 检测到时间序列格式 (good/poor)")
        print("  使用命令: python main.py train --data_root /liaozizhuo/ --score_dict_path None")
        return True

    else:
        print("❌ 未检测到任何有效的数据格式")
        print("  需要以下之一:")
        print("  1. BiLSTM 格式: /liaozizhuo/image/*.jpg + /liaozizhuo/score_dict.json")
        print("  2. 时间序列格式: /liaozizhuo/good/embryo_*/time_*/focal_*.jpg")
        return False

if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/liaozizhuo/"
    diagnose_data(data_root)
