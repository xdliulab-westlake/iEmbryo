#!/usr/bin/env python
"""
验证VGG16修复 - 确保没有重复的全连接层
"""

import torch
import sys
sys.path.insert(0, '.')

from models.backbone import BackboneModel

print("\n" + "=" * 80)
print("VGG16 全连接层重复问题验证")
print("=" * 80)

# 创建VGG16模型（include_top=True）
print("\n📝 创建 VGG16 分类模型 (include_top=True)")
print("-" * 80)
model_clf = BackboneModel(
    model_name='vgg16',
    num_classes=2,
    include_top=True,
    pretrained=False
)

# 设置为eval模式以禁用dropout
model_clf.eval()

# 统计参数数量
total_params = sum(p.numel() for p in model_clf.parameters())
print(f"总参数数: {total_params:,}")

# 分别统计backbone和classifier的参数
backbone_params = sum(p.numel() for p in model_clf.backbone.parameters())
classifier_params = sum(p.numel() for p in model_clf.classifier.parameters())

print(f"  - Backbone 参数: {backbone_params:,}")
print(f"  - Classifier 参数: {classifier_params:,}")
print(f"  - 总计: {backbone_params + classifier_params:,}")

# 测试前向传播
print("\n🔍 前向传播测试:")
print("-" * 80)
x = torch.randn(2, 3, 224, 224)

# 获取backbone输出
backbone_output = model_clf.backbone(x)
print(f"Backbone 输出形状: {backbone_output.shape}")
print(f"  预期: [2, 4096] (VGG16特征维度)")

# 获取classifier输出
classifier_output = model_clf.classifier(backbone_output)
print(f"Classifier 输出形状: {classifier_output.shape}")
print(f"  预期: [2, 2] (2个类别)")

# 完整前向传播
final_output = model_clf(x)
print(f"模型最终输出形状: {final_output.shape}")
print(f"  预期: [2, 2] (2个类别)")

# 验证输出一致
assert torch.allclose(classifier_output, final_output), "输出不一致！"
print("\n✅ 输出一致，无重复全连接层问题\n")

# 对比ResNet
print("\n" + "=" * 80)
print("对比 ResNet50 (确保ResNet也是正确的)")
print("=" * 80)

model_resnet = BackboneModel(
    model_name='resnet50',
    num_classes=2,
    include_top=True,
    pretrained=False
)

model_resnet.eval()  # 设置为eval模式

x = torch.randn(2, 3, 224, 224)
backbone_output = model_resnet.backbone(x)
classifier_output = model_resnet.classifier(backbone_output)
final_output = model_resnet(x)

print(f"Backbone 输出形状: {backbone_output.shape} (预期: [2, 2048])")
print(f"Classifier 输出形状: {classifier_output.shape} (预期: [2, 2])")
print(f"最终输出形状: {final_output.shape} (预期: [2, 2])")

assert torch.allclose(classifier_output, final_output), "ResNet输出不一致！"
print("✅ ResNet50 也是正确的\n")

print("=" * 80)
print("✅ 验证完成：VGG16 和 ResNet50 都没有重复的全连接层")
print("=" * 80 + "\n")
