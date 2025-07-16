#!/usr/bin/env python3
"""
快速检查缓存文件状态的调试脚本
"""

import os
import numpy as np

def check_cache_files():
    """检查缓存文件的详细状态"""
    
    cache_files = [
        "/liaozizhuo/scripts/X_features_multi.npy",
        "/liaozizhuo/scripts/y_bs_labels_multi.npy", 
        "/liaozizhuo/scripts/quality_labels_multi.npy",
        "/liaozizhuo/scripts/ages_multi.npy"
    ]
    
    print("=== 缓存文件状态检查 ===")
    
    all_exist = True
    for i, file_path in enumerate(cache_files):
        print(f"\n文件 {i+1}: {os.path.basename(file_path)}")
        print(f"路径: {file_path}")
        
        if os.path.exists(file_path):
            try:
                # 检查文件大小
                file_size = os.path.getsize(file_path)
                print(f"✅ 存在 - 大小: {file_size / (1024**2):.2f} MB")
                
                # 尝试加载文件检查完整性
                try:
                    data = np.load(file_path)
                    print(f"✅ 可读取 - 形状: {data.shape}")
                    print(f"✅ 数据类型: {data.dtype}")
                    
                    # 检查是否有NaN或inf
                    if np.issubdtype(data.dtype, np.floating):
                        has_nan = np.isnan(data).any()
                        has_inf = np.isinf(data).any()
                        print(f"✅ 数据质量 - NaN: {has_nan}, Inf: {has_inf}")
                    
                except Exception as e:
                    print(f"❌ 加载失败: {e}")
                    all_exist = False
                    
            except Exception as e:
                print(f"❌ 文件损坏: {e}")
                all_exist = False
        else:
            print("❌ 不存在")
            all_exist = False
    
    print(f"\n=== 总结 ===")
    if all_exist:
        print("✅ 所有缓存文件都存在且完整，应该能正常加载")
        
        # 检查逻辑
        print("\n=== 加载逻辑检查 ===")
        condition_result = all(os.path.exists(f) for f in cache_files)
        print(f"all(os.path.exists(f) for f in cache_files): {condition_result}")
        
        if condition_result:
            print("✅ 缓存条件满足，应该会加载缓存而不是重新提取特征")
        else:
            print("❌ 缓存条件不满足，会重新提取特征")
            
    else:
        print("❌ 有缓存文件缺失或损坏，需要重新生成")
        
        # 列出缺失的文件
        missing_files = [f for f in cache_files if not os.path.exists(f)]
        if missing_files:
            print(f"\n缺失文件:")
            for f in missing_files:
                print(f"  - {f}")

    # 检查目录权限
    cache_dir = os.path.dirname(cache_files[0])
    print(f"\n=== 目录权限检查 ===")
    print(f"缓存目录: {cache_dir}")
    
    if os.path.exists(cache_dir):
        print(f"✅ 目录存在")
        if os.access(cache_dir, os.R_OK):
            print(f"✅ 可读")
        else:
            print(f"❌ 不可读")
            
        if os.access(cache_dir, os.W_OK):
            print(f"✅ 可写")
        else:
            print(f"❌ 不可写")
    else:
        print(f"❌ 目录不存在")

if __name__ == "__main__":
    check_cache_files() 