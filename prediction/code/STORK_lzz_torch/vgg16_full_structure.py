#!/usr/bin/env python
"""
VGG16 完整结构展示和分析
"""

import torch
import torchvision.models as models
import sys
sys.path.insert(0, '.')

from models.backbone import BackboneModel

print("\n" + "=" * 100)
print("VGG16 完整结构分析")
print("=" * 100)

# 加载原始VGG16
print("\n1️⃣ 原始 VGG16 的完整结构（torchvision）")
print("-" * 100)
vgg16_original = models.vgg16(pretrained=False)

print("\n📦 VGG16 包含两个主要部分：")
print("\na) features (卷积部分):")
for i, layer in enumerate(vgg16_original.features):
    print(f"   [{i:2d}] {layer}")

print("\n\nb) classifier (分类部分 - 全连接):")
for i, layer in enumerate(vgg16_original.classifier):
    print(f"   [{i:2d}] {layer}")

# 现在展示STORK模型的结构
print("\n\n" + "=" * 100)
print("2️⃣ STORK BackboneModel(VGG16, include_top=True) 的结构")
print("=" * 100)

model = BackboneModel(
    model_name='vgg16',
    num_classes=2,
    include_top=True,
    pretrained=False
)

print("\n📦 STORK VGG16 backbone (返回特征):")
print("\nself.backbone = nn.Sequential(")
for i, layer in enumerate(model.backbone):
    print(f"   [{i}] {layer}")
print(")")

print("\n\n📦 STORK VGG16 classifier (分类头):")
print("\nself.classifier = nn.Sequential(")
for i, layer in enumerate(model.classifier):
    print(f"   [{i}] {layer}")
print(")")

# 现在展示数据流
print("\n\n" + "=" * 100)
print("3️⃣ 数据流和维度变化")
print("=" * 100)

x = torch.randn(1, 3, 224, 224)
print(f"\n输入图像: {x.shape}")
print("           [batch_size, channels, height, width]")

print("\n┌─ 经过 Backbone ──────────────────────────────────────")
print("│")

# 经过features
features_output = model.backbone[0](x)  # features
print(f"│ 1. features (13层卷积) → {features_output.shape}")
print(f"│    [B, 512, 7, 7]")

# 经过pool
pool_output = model.backbone[1](features_output)  # AdaptiveAvgPool2d
print(f"│")
print(f"│ 2. AdaptiveAvgPool2d((7, 7)) → {pool_output.shape}")
print(f"│    [B, 512, 7, 7]")

# 经过flatten
flatten_output = model.backbone[2](pool_output)  # Flatten
print(f"│")
print(f"│ 3. Flatten() → {flatten_output.shape}")
print(f"│    [B, 512*7*7] = [B, 25088]")

print("│")
print("└─ Backbone 输出: [B, 25088] 特征向量")

print("\n┌─ 经过 Classifier (分类头) ───────────────────────────")
print("│")

classifier_output = model.classifier(flatten_output)
print(f"│ 输入: {flatten_output.shape}")
print(f"│")

c1 = model.classifier[0](flatten_output)  # Dropout
print(f"│ 1. Dropout(0.5) → {c1.shape}")

c2 = model.classifier[1](c1)  # Linear
print(f"│ 2. Linear(25088, 1024) → {c2.shape}")

c3 = model.classifier[2](c2)  # ReLU
print(f"│ 3. ReLU() → {c3.shape}")

c4 = model.classifier[3](c3)  # Dropout
print(f"│ 4. Dropout(0.5) → {c4.shape}")

c5 = model.classifier[4](c4)  # Linear
print(f"│ 5. Linear(1024, 2) → {c5.shape}")

print("│")
print("└─ Classifier 输出: [B, 2] 分类结果（logits）")

# 展示为什么需要AdaptiveAvgPool2d
print("\n\n" + "=" * 100)
print("4️⃣ 为什么需要 AdaptiveAvgPool2d？")
print("=" * 100)

print("""
❓ 问题：为什么VGG16的卷积部分输出是 [B, 512, 7, 7] 而不是 [B, 25088]？

💡 答案：

1. 卷积层只做特征提取，不做维度变换
   - VGG16 features (13层卷积) 输出: [B, 512, 7, 7]
   - 这是一个 4 维张量（批次, 通道, 高, 宽）
   - 不能直接送入全连接层（全连接层只能接 2D 张量）

2. 池化层的作用：
   - 将 [B, 512, 7, 7] 池化为 [B, 512, 7, 7]（保持大小）
   - ❓ 但为什么还要池化，大小没变啊？

   原因：
   - 不同的输入图像大小，池化后都变成 [B, 512, 7, 7]
   - 比如输入 256×256，卷积后可能是 [B, 512, 8, 8]
   - AdaptiveAvgPool2d((7, 7)) 会自动调整到 [B, 512, 7, 7]
   - 这样全连接层能处理各种大小的输入！

3. Flatten 层的作用：
   - 把 4D 张量转换为 2D 张量
   - [B, 512, 7, 7] → [B, 512*7*7] = [B, 25088]
   - 现在才能送入全连接层

4. 全连接层才做分类：
   - Linear(25088, 1024) - 第一层降维
   - Linear(1024, 2) - 第二层输出类别数
""")

# 展示为什么include_top对backbone没有影响
print("\n\n" + "=" * 100)
print("5️⃣ include_top=True/False 时的区别")
print("=" * 100)

print("\n┌─ include_top=True (分类模式) ──────────────────────")
print("│ Input → Backbone → Classifier → Output Logits [B, 2]")
print("│ 用于：胚胎分类任务")
print("└─")

print("\n┌─ include_top=False (特征提取模式) ────────────────")
print("│ Input → Backbone → Output Features [B, 25088]")
print("│ 用于：提取特征用于其他任务（迁移学习）")
print("└─")

print("\n💡 关键点：")
print("   • Backbone 的结构在两种模式下是完全相同的！")
print("   • AdaptiveAvgPool2d 和 Flatten 都在 Backbone 中")
print("   • 区别只在于是否添加 Classifier")
print("   • include_top 参数只控制是否创建 self.classifier")

# 对比ResNet
print("\n\n" + "=" * 100)
print("6️⃣ 为什么 ResNet 没有 AdaptiveAvgPool2d？")
print("=" * 100)

resnet = BackboneModel(
    model_name='resnet50',
    num_classes=2,
    include_top=True,
    pretrained=False
)

print("\n📦 ResNet50 backbone 结构:")
print("\nself.backbone = nn.Sequential(")
for i, layer in enumerate(resnet.backbone):
    if i < 2:  # 只显示前两个，避免太长
        print(f"   [{i}] {layer}")
    elif i == 2:
        print(f"   ...")
print(")")

print("\n💡 区别：")
print("   • ResNet 的最后一层已经包含了 AdaptiveAvgPool + Flatten")
print("   • 原始 ResNet: avgpool + flatten 在模型内部")
print("   • 所以 backbone 直接输出 [B, 2048] 特征")
print("   • VGG16 没有，所以要自己加")

print("\n\n" + "=" * 100)
print("总结")
print("=" * 100)

print("""
✅ VGG16 Backbone 的完整过程：

输入: [B, 3, 224, 224]
  ↓
卷积层 (13层) → [B, 512, 7, 7]
  ↓
AdaptiveAvgPool2d((7, 7)) → [B, 512, 7, 7]  ← 自适应池化（统一维度）
  ↓
Flatten() → [B, 25088]  ← 展平为1D特征
  ↓
(如果 include_top=True)
Classifier (全连接层) → [B, 2]  ← 分类输出
  ↓
(如果 include_top=False)
直接返回 [B, 25088] → 特征提取

✨ 核心原因：
   1. 卷积层不改变空间维度，只改变通道数
   2. AdaptiveAvgPool 用于统一不同大小输入的输出维度
   3. Flatten 把 4D 张量转为 2D（卷积→FC的必要步骤）
   4. include_top 参数不影响 Backbone，只影响是否加 Classifier
""")
