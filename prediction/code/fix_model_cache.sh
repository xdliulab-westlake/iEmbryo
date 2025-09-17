#!/bin/bash

echo "🔧 修复PyTorch模型缓存问题..."

# 清理损坏的torch hub缓存
echo "🧹 清理torch hub缓存..."
rm -rf /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# 清理pip缓存
echo "🧹 清理pip缓存..."
pip cache purge

echo "✅ 缓存清理完成，可以重新训练了"