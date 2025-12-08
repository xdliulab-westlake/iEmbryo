#!/usr/bin/env python3
"""
从检查点文件绘制训练历史曲线

用法:
    python plot_training_history.py --checkpoint checkpoints/model_best_model.pth
    python plot_training_history.py --checkpoint checkpoints/model_best_model.pth --output custom_plot.png
"""

import argparse
import torch
from pathlib import Path
from visualization import create_training_plotter


def main():
    parser = argparse.ArgumentParser(description='从检查点绘制训练历史')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='检查点文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片文件名 (可选)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录 (默认: outputs)')
    args = parser.parse_args()

    # 检查检查点文件是否存在
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return

    # 加载检查点
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 提取训练历史
    train_history = checkpoint.get('train_history', {'loss': [], 'acc': []})
    val_history = checkpoint.get('val_history', {'loss': [], 'acc': []})

    # 打印统计信息
    print(f"\n训练历史统计:")
    print(f"  训练轮数: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  最佳验证准确率: {checkpoint.get('best_val_acc', 0):.2f}%")
    print(f"  训练数据点数: {len(train_history.get('loss', []))}")
    print(f"  验证数据点数: {len(val_history.get('loss', []))}")

    # 检查是否有数据
    if len(train_history.get('loss', [])) == 0:
        print("\n错误: 检查点中没有训练历史数据")
        return

    # 创建绘图器
    model_name = checkpoint_path.stem.replace('_best_model', '').replace('checkpoint_', '')
    plotter = create_training_plotter(output_dir=args.output_dir, model_name=model_name)

    # 绘制训练历史
    print("\n生成训练曲线图...")
    save_path = plotter.plot_training_history(
        train_history=train_history,
        val_history=val_history,
        save_name=args.output
    )

    print(f"\n完成! 训练曲线已保存到: {save_path}")


if __name__ == "__main__":
    main()
