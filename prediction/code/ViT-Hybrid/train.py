#!/usr/bin/env python
"""
Multi-Task TMFI-Net Training Entry Point
直接运行此脚本进行训练

用法:
    python train.py --maindb /path/to/MainDB.csv --selected_images /path/to/selected_image_paths.csv

Author: Liao Zizhuo
Date: 2025-12-22
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入训练主函数
from training.train_tmfi import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multi-Task TMFI-Net for Embryo Quality Assessment')

    # Data paths
    parser.add_argument('--maindb', type=str, required=True,
                        help='Path to MainDB.csv (required)')
    parser.add_argument('--selected_images', type=str, required=True,
                        help='Path to selected_image_paths.csv (required)')
    parser.add_argument('--filtered', type=str, default=None,
                        help='Path to filtered_clean_embryos.csv (optional)')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip validation and use cached results (if cache exists)')
    parser.add_argument('--force_no_validation', action='store_true',
                        help='Force skip validation completely (use all data without completeness check - FAST but risky)')

    # Model configuration
    parser.add_argument('--pretrained_swin', type=str, default=None,
                        help='Path to pretrained Video Swin Transformer weights (optional)')
    parser.add_argument('--fusion_type', type=str, default='average',
                        choices=['average', 'max', 'middle', 'learned'],
                        help='Focal plane fusion strategy (default: average)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    # Saliency loss (optional, requires ground truth saliency maps)
    parser.add_argument('--use_saliency_loss', action='store_true',
                        help='Use saliency loss (requires GT saliency maps)')
    parser.add_argument('--saliency_weight', type=float, default=0.1,
                        help='Weight for saliency loss (default: 0.1)')

    args = parser.parse_args()

    # Display configuration
    print("\n" + "="*70)
    print("Multi-Task TMFI-Net Training")
    print("="*70)
    print("\nConfiguration:")
    print(f"  MainDB: {args.maindb}")
    print(f"  Selected Images: {args.selected_images}")
    if args.filtered:
        print(f"  Filtered: {args.filtered}")
    print(f"  Fusion Type: {args.fusion_type}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Num Workers: {args.num_workers}")
    if args.use_saliency_loss:
        print(f"  Saliency Loss: Enabled (weight={args.saliency_weight})")
    print("="*70 + "\n")

    # Run training
    main(args)
