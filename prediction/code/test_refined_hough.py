#!/usr/bin/env python3
"""
改进版Hough检测 - 解决检测圆圈过小的问题
添加圆形大小和位置的智能筛选
"""

import os
import random
import numpy as np
import cv2
import copy
import pandas as pd

def refined_crop_petri_dish(frame):
    """
    改进的Petri dish检测 - 优先选择大圆圈和居中圆圈
    """
    if len(frame.shape) == 3:
        gimg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gimg = frame.copy()
        
    cimg = copy.deepcopy(frame)
    img_height, img_width = gimg.shape
    img_center_x, img_center_y = img_width // 2, img_height // 2
    
    # Apply Gaussian blur
    gimg_blurred = cv2.GaussianBlur(gimg, (9, 9), 2)
    
    # 更严格的半径范围 - 培养皿应该是图像中的主要对象
    img_diagonal = np.sqrt(img_height**2 + img_width**2)
    min_radius = max(50, int(img_diagonal * 0.15))  # 提高最小半径到15%
    max_radius = min(int(img_diagonal * 0.45), max(img_height, img_width) // 2)
    
    print(f"图像尺寸: {img_width}x{img_height}, 半径范围: {min_radius}-{max_radius}")
    
    # 尝试不同参数组合
    param_combinations = [
        (60, 40),   # 稍微不那么敏感，避免小圆圈
        (70, 45),   # 中等敏感度
        (50, 35),   # 较敏感
        (80, 50),   # 不敏感
        (90, 55),   # 最不敏感
    ]
    
    all_circles = []
    
    # 收集所有可能的圆圈
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
                # 计算到图像中心的距离
                distance_to_center = np.sqrt((x - img_center_x)**2 + (y - img_center_y)**2)
                all_circles.append({
                    'x': x, 'y': y, 'radius': r,
                    'distance_to_center': distance_to_center,
                    'params': (param1, param2)
                })
    
    if not all_circles:
        return frame, None, False
    
    # 转换为DataFrame进行筛选
    df_circles = pd.DataFrame(all_circles)
    
    # 智能筛选策略：
    # 1. 半径权重 (40%): 更大的圆圈得分更高
    # 2. 位置权重 (35%): 更接近中心的圆圈得分更高  
    # 3. 大小合理性权重 (25%): 半径在合理范围内的得分更高
    
    # 归一化各项指标到0-1
    df_circles['radius_norm'] = (df_circles['radius'] - df_circles['radius'].min()) / (df_circles['radius'].max() - df_circles['radius'].min() + 1e-6)
    df_circles['center_norm'] = 1 - (df_circles['distance_to_center'] / df_circles['distance_to_center'].max())
    
    # 大小合理性：偏好中等偏大的圆圈
    ideal_radius = (min_radius + max_radius) * 0.7  # 偏向较大圆圈
    df_circles['size_reasonableness'] = 1 - abs(df_circles['radius'] - ideal_radius) / (max_radius - min_radius)
    
    # 综合评分
    df_circles['score'] = (
        df_circles['radius_norm'] * 0.4 +
        df_circles['center_norm'] * 0.35 + 
        df_circles['size_reasonableness'] * 0.25
    )
    
    # 选择得分最高的圆圈
    best_circle = df_circles.loc[df_circles['score'].idxmax()]
    
    # 创建掩模
    mask = np.zeros_like(cimg)
    if len(cimg.shape) == 3:
        mask = cv2.circle(mask.copy(), (int(best_circle['x']), int(best_circle['y'])), 
                        int(best_circle['radius']), (255, 255, 255), -1)
    else:
        mask = cv2.circle(mask.copy(), (int(best_circle['x']), int(best_circle['y'])), 
                        int(best_circle['radius']), 255, -1)
    
    # 应用掩模
    result = cv2.bitwise_and(cimg, mask)
    
    used_params = {
        'params': best_circle['params'],
        'radius': int(best_circle['radius']),
        'center': (int(best_circle['x']), int(best_circle['y'])),
        'score': round(best_circle['score'], 3),
        'candidates': len(df_circles)
    }
    
    return result, used_params, True

def test_refined_hough_detection(data_dir, output_dir="refined_hough_results", num_samples=20):
    """测试改进的Hough检测"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/original', exist_ok=True)
    os.makedirs(f'{output_dir}/optimized', exist_ok=True) 
    os.makedirs(f'{output_dir}/refined', exist_ok=True)
    
    # 收集图像路径 (使用相同的随机种子)
    image_paths = []
    for sample_name in os.listdir(data_dir):
        sample_path = os.path.join(data_dir, sample_name)
        if not os.path.isdir(sample_path) or sample_name.startswith('.'):
            continue
        
        images_dir = os.path.join(sample_path, 'images')
        if not os.path.exists(images_dir):
            continue
        
        for time_point in os.listdir(images_dir):
            time_path = os.path.join(images_dir, time_point)
            if not os.path.isdir(time_path):
                continue
                
            for img_name in os.listdir(time_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(time_path, img_name)
                    image_paths.append((img_path, sample_name, time_point))
    
    random.seed(42)
    selected_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    print(f'=== 改进版Hough检测测试 ===')
    print(f'解决小圆圈问题，优化大圆圈选择策略')
    print(f'测试样本: {len(selected_paths)} 张\n')
    
    # 导入之前的优化方法作为对比
    def optimized_crop_petri_dish(frame):
        if len(frame.shape) == 3:
            gimg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gimg = frame.copy()
        cimg = copy.deepcopy(frame)
        gimg_blurred = cv2.GaussianBlur(gimg, (9, 9), 2)
        img_height, img_width = gimg.shape
        img_diagonal = np.sqrt(img_height**2 + img_width**2)
        min_radius = max(10, int(img_diagonal * 0.1))
        max_radius = min(int(img_diagonal * 0.4), max(img_height, img_width) // 2)
        
        circles = cv2.HoughCircles(
            gimg_blurred, cv2.HOUGH_GRADIENT, dp=1,
            minDist=max(100, int(min(img_height, img_width) * 0.3)),
            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles is not None and len(circles) > 0:
            df_circle = pd.DataFrame(data=circles[0], columns=['x', 'y', 'radius'])
            df_circle = df_circle.sort_values(by=['radius'], ascending=False)
            circle = df_circle.iloc[0].values
            mask = np.zeros_like(cimg)
            if len(cimg.shape) == 3:
                mask = cv2.circle(mask.copy(), (int(circle[0]), int(circle[1])), 
                                int(circle[2]), (255, 255, 255), -1)
            else:
                mask = cv2.circle(mask.copy(), (int(circle[0]), int(circle[1])), 
                                int(circle[2]), 255, -1)
            result = cv2.bitwise_and(cimg, mask)
            return result, {'radius': int(circle[2])}, True
        return frame, None, False
    
    refined_successful = 0
    optimized_successful = 0
    
    for idx, (img_path, sample_name, time_point) in enumerate(selected_paths):
        try:
            original_img = cv2.imread(img_path)
            if original_img is None:
                continue
            
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # 优化方法
            opt_processed, opt_params, opt_success = optimized_crop_petri_dish(original_img_rgb)
            if opt_success:
                optimized_successful += 1
            
            print(f'\\n{idx+1:2d}. {sample_name[:15]:15s} {time_point:8s}')
            
            # 改进方法  
            ref_processed, ref_params, ref_success = refined_crop_petri_dish(original_img_rgb)
            if ref_success:
                refined_successful += 1
            
            # 输出详细信息
            if opt_success and ref_success:
                opt_radius = opt_params['radius']
                ref_radius = ref_params['radius']
                radius_change = ref_radius - opt_radius
                
                print(f'    优化方法: 半径 {opt_radius}px')
                print(f'    改进方法: 半径 {ref_radius}px (变化: {radius_change:+d}px)')
                print(f'    改进详情: score={ref_params["score"]}, 候选数={ref_params["candidates"]}')
                
                if radius_change > 20:
                    print(f'    ✅ 检测到更大圆圈 (+{radius_change}px)')
                elif radius_change < -20:
                    print(f'    ⚠️  选择了较小圆圈 ({radius_change}px)')
                else:
                    print(f'    ➡️  半径变化不大 ({radius_change:+d}px)')
            
            # 保存结果
            img_name = os.path.basename(img_path)
            filename_base = f'{idx+1:02d}_{sample_name}_{time_point}_{os.path.splitext(img_name)[0]}'
            
            cv2.imwrite(f'{output_dir}/original/{filename_base}.png', 
                       cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{output_dir}/optimized/{filename_base}.png', 
                       cv2.cvtColor(opt_processed, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{output_dir}/refined/{filename_base}.png', 
                       cv2.cvtColor(ref_processed, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f'错误 {idx+1}: {e}')
    
    print(f'\\n=== 对比结果 ===')
    print(f'优化方法成功率: {optimized_successful}/{len(selected_paths)} ({optimized_successful/len(selected_paths)*100:.1f}%)')
    print(f'改进方法成功率: {refined_successful}/{len(selected_paths)} ({refined_successful/len(selected_paths)*100:.1f}%)')
    print(f'结果保存在: {output_dir}/')
    
    return optimized_successful, refined_successful, len(selected_paths)

if __name__ == "__main__":
    data_dir = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/selected_images'
    opt_success, ref_success, total = test_refined_hough_detection(data_dir, num_samples=10)  # 减少样本便于查看详情
    
    print(f'\\n📊 改进效果统计:')
    print(f'优化方法: {opt_success}/{total} ({opt_success/total*100:.1f}%)')  
    print(f'改进方法: {ref_success}/{total} ({ref_success/total*100:.1f}%)')
    print(f'变化: {ref_success-opt_success:+d} ({(ref_success-opt_success)/total*100:+.1f}%)')