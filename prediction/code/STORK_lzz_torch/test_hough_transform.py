"""
测试 Hough 圆变换的脚本
生成原始图片和变换后图片的对比 PDF
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import sys
from glob import glob

# 导入 STORK 预处理模块
from data.stork_preprocessing import crop_petri_dish


def load_and_visualize_hough_transform(image_paths, output_pdf='hough_transform_results.pdf'):
    """
    加载图片，应用 Hough 圆检测，并生成对比 PDF

    Args:
        image_paths: 图片路径列表
        output_pdf: 输出 PDF 文件名
    """
    # 只取前10张图片
    image_paths = image_paths[:10]

    with PdfPages(output_pdf) as pdf:
        for idx, img_path in enumerate(image_paths, 1):
            print(f"处理图片 {idx}/10: {Path(img_path).name}")

            # 读取原始图片
            original = cv2.imread(img_path)
            if original is None:
                print(f"  警告: 无法加载图片 {img_path}")
                continue

            # 转换为 RGB（OpenCV 默认是 BGR）
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            # 应用 Hough 圆检测和培养皿裁剪
            transformed = crop_petri_dish(original_rgb)

            # 创建对比图
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # 显示原始图片
            axes[0].imshow(original_rgb)
            axes[0].set_title(f'原始图片 #{idx}', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # 显示变换后的图片
            axes[1].imshow(transformed)
            axes[1].set_title(f'Hough 圆检测后 #{idx}', fontsize=12, fontweight='bold')
            axes[1].axis('off')

            # 添加图片信息
            img_name = Path(img_path).name
            plt.suptitle(f'{img_name}\n原始尺寸: {original.shape[1]}x{original.shape[0]}',
                        fontsize=10, y=0.98)

            plt.tight_layout()

            # 保存到 PDF
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"  ✓ 完成")

    print(f"\n✅ PDF 已保存: {output_pdf}")


def visualize_hough_detection_process(image_path, output_pdf='hough_detection_process.pdf'):
    """
    详细可视化 Hough 圆检测过程（可选的调试版本）

    Args:
        image_path: 单张图片路径
        output_pdf: 输出 PDF 文件名
    """
    import copy
    import pandas as pd

    # 读取图片
    frame = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 转换为灰度图
    gimg = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # 应用高斯模糊
    gimg_blurred = cv2.GaussianBlur(gimg, (9, 9), 2)

    # 计算参数
    img_height, img_width = gimg.shape
    img_diagonal = np.sqrt(img_height**2 + img_width**2)
    min_radius = max(50, int(img_diagonal * 0.15))
    max_radius = min(int(img_diagonal * 0.45), max(img_height, img_width) // 2)

    # 检测圆形
    param_combinations = [(60, 40), (70, 45), (50, 35)]

    with PdfPages(output_pdf) as pdf:
        # 第1页：原始图片和预处理步骤
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(frame_rgb)
        axes[0, 0].set_title('1. 原始图片', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gimg, cmap='gray')
        axes[0, 1].set_title('2. 灰度图', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(gimg_blurred, cmap='gray')
        axes[1, 0].set_title('3. 高斯模糊后', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        # 显示参数信息
        info_text = f"图像尺寸: {img_width} x {img_height}\n"
        info_text += f"图像对角线: {img_diagonal:.0f}\n"
        info_text += f"半径范围: {min_radius} - {max_radius}\n"
        info_text += f"参数组合: {param_combinations}"

        axes[1, 1].text(0.1, 0.5, info_text, fontsize=10,
                       verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('4. 检测参数', fontsize=12, fontweight='bold')

        plt.suptitle('Hough 圆检测 - 预处理步骤', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # 第2页：检测到的圆形
        all_circles = []
        img_center_x, img_center_y = img_width // 2, img_height // 2

        for param1, param2 in param_combinations:
            circles = cv2.HoughCircles(
                gimg_blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(80, int(min(img_height, img_width) * 0.2)),
                param1=param1,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )

            if circles is not None and len(circles[0]) > 0:
                for circle in circles[0]:
                    x, y, r = circle
                    distance_to_center = np.sqrt((x - img_center_x)**2 + (y - img_center_y)**2)
                    all_circles.append({
                        'x': x, 'y': y, 'radius': r,
                        'distance_to_center': distance_to_center,
                        'params': f"({param1}, {param2})"
                    })

        # 可视化所有检测到的圆
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 显示所有检测到的圆
        img_with_circles = frame_rgb.copy()
        for circle_info in all_circles:
            cv2.circle(img_with_circles,
                      (int(circle_info['x']), int(circle_info['y'])),
                      int(circle_info['radius']),
                      (255, 0, 0), 3)

        axes[0].imshow(img_with_circles)
        axes[0].set_title(f'检测到 {len(all_circles)} 个圆', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 选择最佳圆形
        if all_circles:
            df_circles = pd.DataFrame(all_circles)
            df_circles['radius_norm'] = (df_circles['radius'] - df_circles['radius'].min()) / (df_circles['radius'].max() - df_circles['radius'].min() + 1e-6)
            df_circles['center_norm'] = 1 - (df_circles['distance_to_center'] / df_circles['distance_to_center'].max())
            ideal_radius = (min_radius + max_radius) * 0.7
            df_circles['size_reasonableness'] = 1 - abs(df_circles['radius'] - ideal_radius) / (max_radius - min_radius)
            df_circles['score'] = (
                df_circles['radius_norm'] * 0.4 +
                df_circles['center_norm'] * 0.35 +
                df_circles['size_reasonableness'] * 0.25
            )

            best_circle = df_circles.loc[df_circles['score'].idxmax()]

            # 显示最佳圆形
            img_with_best = frame_rgb.copy()
            cv2.circle(img_with_best,
                      (int(best_circle['x']), int(best_circle['y'])),
                      int(best_circle['radius']),
                      (0, 255, 0), 5)

            axes[1].imshow(img_with_best)
            axes[1].set_title(f'最佳圆形 (得分: {best_circle["score"]:.3f})',
                            fontsize=12, fontweight='bold')
            axes[1].axis('off')

        plt.suptitle('Hough 圆检测 - 圆形检测结果', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # 第3页：最终结果
        transformed = crop_petri_dish(frame_rgb)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(frame_rgb)
        axes[0].set_title('原始图片', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(transformed)
        axes[1].set_title('裁剪后的培养皿图像', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        plt.suptitle('Hough 圆检测 - 最终结果', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"✅ 详细检测过程已保存: {output_pdf}")


def main():
    """主函数"""
    print("=" * 60)
    print("Hough 圆变换测试脚本")
    print("=" * 60)

    # 查找图片
    # 使用绝对路径
    image_dir = Path("/Users/lzz/Documents/GitHub/IVF_AI/extracted_samples")

    if not image_dir.exists():
        print(f"❌ 图片目录不存在: {image_dir}")
        print("请在命令行指定图片路径")
        return

    # 递归查找所有图片
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob(str(image_dir / "**" / ext), recursive=True))

    if not image_paths:
        print(f"❌ 在 {image_dir} 中未找到图片")
        return

    print(f"\n找到 {len(image_paths)} 张图片")
    print(f"将处理前 10 张图片\n")

    # 生成对比 PDF
    output_path = Path(__file__).parent / "hough_transform_results.pdf"
    load_and_visualize_hough_transform(image_paths, str(output_path))

    # 可选：生成详细检测过程（第一张图片）
    if len(image_paths) > 0:
        print(f"\n生成详细检测过程（第一张图片）...")
        detail_output = Path(__file__).parent / "hough_detection_process.pdf"
        visualize_hough_detection_process(image_paths[0], str(detail_output))

    print("\n" + "=" * 60)
    print("✅ 所有任务完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - {output_path}")
    if len(image_paths) > 0:
        print(f"  - {detail_output}")


if __name__ == "__main__":
    main()
