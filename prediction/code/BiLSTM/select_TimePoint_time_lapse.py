import os
import pandas as pd
import glob
import numpy as np
import re
from collections import defaultdict

# 配置路径
csv_file = "/liaozizhuo/scripts/Embryo_Times_PostFertilization_Hours.csv"
image_root = "/liaozizhuo/image"
output_csv = "/liaozizhuo/selected_image_paths.csv"


# 标准时间点
standard_hours = np.arange(96.0, 112.5, 0.5)

# 读取时间表
df = pd.read_csv(csv_file)
df = df[(df['Time_PostFertilization_Hours'] >= 96.0) & (df['Time_PostFertilization_Hours'] <= 112.0)]

results = []
focal_depth_stats = defaultdict(int)

# 遍历所有子目录：/liaozizhuo/image/Dxxx_P_xx
embryo_dirs = glob.glob(os.path.join(image_root, "D*_P_*"))
print(f"[INFO] Found {len(embryo_dirs)} embryo directories")

for i, embryo_dir in enumerate(embryo_dirs):
    if i % 100 == 0:
        print(f"[INFO] Processing {i}/{len(embryo_dirs)} embryos...")

    embryo_folder = os.path.basename(embryo_dir)
    embryo_parts = embryo_folder.split("_")
    if len(embryo_parts) < 2:
        continue

    patient_id = "_".join(embryo_parts[:-1])  # 去掉最后一个元素，如 _1
    embryo_number = embryo_parts[-1]

    embryo_df = df[df['PatientID'] == patient_id].copy()
    if embryo_df.empty:
        print(f"[WARNING] No time data found for {patient_id}")
        continue

    for target_time in standard_hours:
        embryo_df.loc[:, 'abs_diff'] = (embryo_df['Time_PostFertilization_Hours'] - target_time).abs()
        min_diff = embryo_df['abs_diff'].min()
        closest_row = embryo_df[embryo_df['abs_diff'] == min_diff].nsmallest(1, 'Time_PostFertilization_Hours').iloc[0]

        run = int(closest_row['Run'])
        run_pattern = re.compile(rf"RUN{run}\D")  # e.g., match "RUN486_" or "RUN486."

        # 🔥 收集所有焦距深度的匹配图像
        all_focal_images = []
        for root, _, files in os.walk(embryo_dir):
            for fname in files:
                if fname.endswith('.jpeg') and re.search(run_pattern, fname):
                    full_path = os.path.join(root, fname)
                    focal_plane = os.path.basename(os.path.dirname(full_path))

                    # 确保是有效的焦距深度目录
                    if focal_plane.startswith('F'):
                        all_focal_images.append({
                            "focal_plane": focal_plane,
                            "full_path": full_path,
                            "filename": fname
                        })

        # 为每个找到的焦距深度创建记录
        time_error = abs(closest_row["Time_PostFertilization_Hours"] - target_time)

        for focal_img in all_focal_images:
            results.append({
                "PatientID": patient_id,
                "EmbryoNumber": embryo_number,
                "FocalPlane": focal_img["focal_plane"],
                "Run": run,
                "TargetHour": target_time,
                "MatchedHour": closest_row["Time_PostFertilization_Hours"],
                "TimeError": time_error,
                "Filename": focal_img["filename"],
                "FullPath": focal_img["full_path"]
            })
            focal_depth_stats[focal_img["focal_plane"]] += 1


# 保存结果为 CSV
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by=["PatientID", "EmbryoNumber", "TargetHour", "FocalPlane"])
result_df.to_csv(output_csv, index=False)

# 统计信息
total_images = len(result_df)
unique_embryos = len(result_df.groupby(['PatientID', 'EmbryoNumber']))
unique_timepoints = len(result_df.groupby(['PatientID', 'EmbryoNumber', 'TargetHour']))

print(f"[INFO] Multi-focal image path collection completed!")
print(f"[INFO] Statistics:")
print(f"  Total images: {total_images}")
print(f"  Total embryos: {unique_embryos}")
print(f"  Total timepoints: {unique_timepoints}")
if unique_timepoints > 0:
    print(f"  Avg focal depths per timepoint: {total_images / unique_timepoints:.1f}")
print(f"  Saved to: {output_csv}")

print(f"[INFO] Focal depth distribution:")
for focal_plane, count in sorted(focal_depth_stats.items()):
    print(f"  {focal_plane}: {count} images")
