# extract_test_embryos.py
"""
从filtered_clean_embryos.csv中提取测试用的胚胎图片
提取20个好胚胎和20个坏胚胎的图片，分别放入两个文件夹
"""

import os
import pandas as pd
import shutil
import random
from pathlib import Path

def extract_test_embryos(csv_path, output_dir, n_good=20, n_poor=20, random_seed=42):
    """
    提取测试用的胚胎图片
    
    Args:
        csv_path: filtered_clean_embryos.csv的路径
        output_dir: 输出目录
        n_good: 好胚胎数量
        n_poor: 坏胚胎数量
        random_seed: 随机种子
    """
    
    # 设置随机种子
    random.seed(random_seed)
    
    print(f"=== 开始提取测试胚胎图片 ===")
    print(f"CSV文件: {csv_path}")
    print(f"输出目录: {output_dir}")
    print(f"好胚胎数量: {n_good}")
    print(f"坏胚胎数量: {n_poor}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件，共 {len(df)} 条记录")
        print(f"胚胎质量分布:\n{df['EmbryoQuality'].value_counts()}")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 按胚胎ID分组
    embryo_groups = df.groupby('EmbryoID')
    print(f"总胚胎数量: {len(embryo_groups)}")
    
    # 按质量分类胚胎
    good_embryos = []
    poor_embryos = []
    
    for embryo_id, group in embryo_groups:
        quality = group['EmbryoQuality'].iloc[0]  # 取第一个记录的质量标签
        if quality == 'good':
            good_embryos.append(embryo_id)
        elif quality == 'poor':
            poor_embryos.append(embryo_id)
    
    print(f"好胚胎数量: {len(good_embryos)}")
    print(f"坏胚胎数量: {len(poor_embryos)}")
    
    # 随机选择指定数量的胚胎
    if len(good_embryos) < n_good:
        print(f"警告: 好胚胎数量不足，只有 {len(good_embryos)} 个")
        n_good = len(good_embryos)
    
    if len(poor_embryos) < n_poor:
        print(f"警告: 坏胚胎数量不足，只有 {len(poor_embryos)} 个")
        n_poor = len(poor_embryos)
    
    selected_good = random.sample(good_embryos, n_good)
    selected_poor = random.sample(poor_embryos, n_poor)
    
    print(f"已选择 {len(selected_good)} 个好胚胎和 {len(selected_poor)} 个坏胚胎")
    
    # 创建输出目录
    good_dir = os.path.join(output_dir, "good_embryos")
    poor_dir = os.path.join(output_dir, "poor_embryos")
    
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(poor_dir, exist_ok=True)
    
    # 提取好胚胎图片
    print(f"\n=== 提取好胚胎图片到: {good_dir} ===")
    good_count = extract_embryo_images(df, selected_good, good_dir, "good")
    
    # 提取坏胚胎图片
    print(f"\n=== 提取坏胚胎图片到: {poor_dir} ===")
    poor_count = extract_embryo_images(df, selected_poor, poor_dir, "poor")
    
    # 生成统计报告
    print(f"\n=== 提取完成 ===")
    print(f"好胚胎图片数量: {good_count}")
    print(f"坏胚胎图片数量: {poor_count}")
    print(f"总图片数量: {good_count + poor_count}")
    
    # 生成测试胚胎列表
    generate_test_list(output_dir, selected_good, selected_poor, good_count, poor_count)
    
    return good_count, poor_count

def extract_embryo_images(df, embryo_ids, output_dir, quality_label):
    """
    提取指定胚胎的图片
    
    Args:
        df: 数据框
        embryo_ids: 胚胎ID列表
        output_dir: 输出目录
        quality_label: 质量标签
    
    Returns:
        提取的图片数量
    """
    
    total_images = 0
    
    for i, embryo_id in enumerate(embryo_ids):
        print(f"处理胚胎 {i+1}/{len(embryo_ids)}: {embryo_id}")
        
        # 获取该胚胎的所有记录
        embryo_data = df[df['EmbryoID'] == embryo_id]
        
        # 创建胚胎子目录
        embryo_dir = os.path.join(output_dir, embryo_id)
        os.makedirs(embryo_dir, exist_ok=True)
        
        # 复制图片
        for _, row in embryo_data.iterrows():
            source_path = row['FullPath']
            
            if pd.isna(source_path) or not os.path.exists(source_path):
                print(f"  警告: 图片路径不存在: {source_path}")
                continue
            
            # 构建目标文件名
            focal_plane = row['FocalPlane']
            target_hour = row['TargetHour']
            
            # 从源路径提取文件名
            source_filename = os.path.basename(source_path)
            name_without_ext = os.path.splitext(source_filename)[0]
            
            # 目标文件名格式: FocalPlane_TargetHour_原文件名
            target_filename = f"{focal_plane}_{target_hour}_{name_without_ext}.jpeg"
            target_path = os.path.join(embryo_dir, target_filename)
            
            try:
                # 复制图片
                shutil.copy2(source_path, target_path)
                total_images += 1
                print(f"  复制: {focal_plane}_{target_hour} -> {target_filename}")
            except Exception as e:
                print(f"  复制失败: {source_path} -> {target_path}, 错误: {e}")
    
    print(f"  {quality_label}胚胎图片提取完成，共 {total_images} 张")
    return total_images

def generate_test_list(output_dir, good_embryos, poor_embryos, good_count, poor_count):
    """生成测试胚胎列表文件"""
    
    test_list_path = os.path.join(output_dir, "test_embryos_list.txt")
    
    with open(test_list_path, 'w', encoding='utf-8') as f:
        f.write("=== 测试胚胎列表 ===\n\n")
        
        f.write(f"好胚胎数量: {len(good_embryos)}\n")
        f.write(f"好胚胎图片总数: {good_count}\n")
        f.write("好胚胎ID列表:\n")
        for embryo_id in good_embryos:
            f.write(f"  - {embryo_id}\n")
        
        f.write(f"\n坏胚胎数量: {len(poor_embryos)}\n")
        f.write(f"坏胚胎图片总数: {poor_count}\n")
        f.write("坏胚胎ID列表:\n")
        for embryo_id in poor_embryos:
            f.write(f"  - {embryo_id}\n")
        
        f.write(f"\n总计:\n")
        f.write(f"  胚胎数量: {len(good_embryos) + len(poor_embryos)}\n")
        f.write(f"  图片数量: {good_count + poor_count}\n")
    
    print(f"测试胚胎列表已保存到: {test_list_path}")

def main():
    """主函数"""
    
    # 配置参数
    csv_path = "/liaozizhuo/scripts/filtered_clean_embryos.csv"  # 根据你的实际路径修改
    output_dir = "/liaozizhuo/scripts/test_embryos_dataset"  # 输出目录
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        print("请检查文件路径是否正确")
        return
    
    # 提取测试胚胎
    try:
        good_count, poor_count = extract_test_embryos(
            csv_path=csv_path,
            output_dir=output_dir,
            n_good=20,
            n_poor=20,
            random_seed=42
        )
        
        print(f"\n=== 脚本执行完成 ===")
        print(f"输出目录: {output_dir}")
        print(f"好胚胎图片: {good_count} 张")
        print(f"坏胚胎图片: {poor_count} 张")
        print(f"总图片: {good_count + poor_count} 张")
        
    except Exception as e:
        print(f"脚本执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()