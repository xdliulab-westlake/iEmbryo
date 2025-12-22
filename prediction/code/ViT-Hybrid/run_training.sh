#!/bin/bash
# Multi-Task TMFI-Net Training Script
#
# 使用方法:
#   1. 修改下面的数据路径
#   2. 运行: bash run_training.sh
#
# Author: Liao Zizhuo
# Date: 2025-12-22

# ========================================
# 配置参数 - 请根据实际情况修改
# ========================================

# 数据路径
MAINDB="/path/to/MainDB.csv"
SELECTED_IMAGES="/path/to/selected_image_paths.csv"
FILTERED=""  # 可选，留空表示不使用

# 训练参数
FUSION_TYPE="average"  # 焦平面融合策略: average, max, middle, learned
EPOCHS=100
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_WORKERS=4

# 可选参数
PRETRAINED_SWIN=""  # Video Swin Transformer预训练权重路径，留空表示不使用
USE_SALIENCY_LOSS=false  # 是否使用saliency loss（需要GT saliency map）
SALIENCY_WEIGHT=0.1
FORCE_NO_VALIDATION=true  # 跳过数据验证（快速但不检查数据完整性）⚠️

# ========================================
# 运行训练
# ========================================

echo "========================================"
echo "Multi-Task TMFI-Net Training"
echo "========================================"
echo ""
echo "Configuration:"
echo "  MainDB: $MAINDB"
echo "  Selected Images: $SELECTED_IMAGES"
echo "  Fusion Type: $FUSION_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# 构建命令
CMD="python train.py \
    --maindb $MAINDB \
    --selected_images $SELECTED_IMAGES \
    --fusion_type $FUSION_TYPE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS"

# 添加可选参数
if [ -n "$FILTERED" ]; then
    CMD="$CMD --filtered $FILTERED"
fi

if [ -n "$PRETRAINED_SWIN" ]; then
    CMD="$CMD --pretrained_swin $PRETRAINED_SWIN"
fi

if [ "$USE_SALIENCY_LOSS" = true ]; then
    CMD="$CMD --use_saliency_loss --saliency_weight $SALIENCY_WEIGHT"
fi

if [ "$FORCE_NO_VALIDATION" = true ]; then
    CMD="$CMD --force_no_validation"
fi

# 运行训练
echo "Running command:"
echo "$CMD"
echo ""
echo "========================================"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
