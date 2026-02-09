#!/usr/bin/env python3
"""
批量运行所有模型的三类胚胎特征空间可视化
"""

import subprocess
from pathlib import Path

# 模型配置
MODELS = [
    {'name': 'vgg16', 'path': '../vgg16_best_val_accuracy.pth'},
    {'name': 'resnet50', 'path': '../resnet50_best_val_accuracy.pth'},
    {'name': 'resnet152', 'path': '../resnet152_best_val_accuracy.pth'},
    {'name': 'densenet121', 'path': '../densenet121_best_val_accuracy.pth'},
    {'name': 'densenet169', 'path': '../densenet169_best_val_accuracy.pth'},
    {'name': 'densenet201', 'path': '../densenet201_best_val_accuracy.pth'},
    {'name': 'inceptionv3', 'path': '../inceptionv3_best_val_accuracy.pth'},
]

# 数据路径配置
CSV_PATH = '/Users/lzz/Desktop/IVF/good_embryos/embryos_all_focal_planes.csv'
BASE_DIR = '/Users/lzz/Documents/GitHub/IVF_AI'
FAIR_DIR = '/Users/lzz/Desktop/IVF/fair_embryos'

# 可视化参数
OUTPUT_ROOT = 'outputs'
MAX_SAMPLES_PER_CLASS = 1000
MAX_FAIR_SAMPLES = 1000
BATCH_SIZE = 64
METHODS = ['pca']  # 只使用PCA，跳过t-SNE和UMAP (环境问题)

def run_visualization(model_config):
    """运行单个模型的可视化"""
    model_name = model_config['name']
    model_path = model_config['path']

    print("\n" + "="*80)
    print(f"🚀 开始处理模型: {model_name}")
    print("="*80)

    # 检查模型文件是否存在
    if not Path(model_path).exists():
        print(f"⚠️  模型文件不存在: {model_path}")
        print("跳过该模型...")
        return False

    # 构建输出目录
    output_dir = f"{OUTPUT_ROOT}/three_class_{model_name}"

    # 构建命令
    cmd = [
        'python', 'visualize_three_classes.py',
        '--model_path', model_path,
        '--model_name', model_name,
        '--csv_path', CSV_PATH,
        '--base_dir', BASE_DIR,
        '--fair_dir', FAIR_DIR,
        '--output_dir', output_dir,
        '--max_samples_per_class', str(MAX_SAMPLES_PER_CLASS),
        '--max_fair_samples', str(MAX_FAIR_SAMPLES),
        '--batch_size', str(BATCH_SIZE),
        '--methods', *METHODS
    ]

    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {model_name} 可视化完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} 可视化失败: {e}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("📊 批量运行所有模型的三类胚胎特征空间可视化")
    print("="*80)

    success_count = 0
    fail_count = 0

    for model_config in MODELS:
        if run_visualization(model_config):
            success_count += 1
        else:
            fail_count += 1

    # 总结
    print("\n" + "="*80)
    print("📈 处理完成总结")
    print("="*80)
    print(f"✅ 成功: {success_count} 个模型")
    print(f"❌ 失败: {fail_count} 个模型")
    print(f"📁 结果保存在: {OUTPUT_ROOT}/")
    print("="*80)

if __name__ == '__main__':
    main()
