#!/usr/bin/env python
"""
测试：AdaptiveAvgPool2d 是否必须自己操作？
"""

import torch
import torchvision.models as models

print("\n" + "=" * 100)
print("测试：VGG16 的池化是否需要自己操作")
print("=" * 100)

# 加载原始VGG16
vgg16 = models.vgg16(pretrained=False)

print("\n📊 VGG16 features 部分的最后几层：")
print("-" * 100)
for i in range(24, 31):
    print(f"  [{i:2d}] {list(vgg16.features.children())[i]}")

print("\n\n💡 关键观察：VGG16 features 最后已经包含了 MaxPool2d")

# 测试不同输入大小
print("\n\n" + "=" * 100)
print("1️⃣ 测试固定输入大小（224×224）")
print("=" * 100)

vgg16.eval()
x_224 = torch.randn(1, 3, 224, 224)

features_output = vgg16.features(x_224)
print(f"\n输入:  {x_224.shape}")
print(f"输出:  {features_output.shape}")
print(f"❓ 是否始终是 [1, 512, 7, 7]？")

# 展平后
flattened = torch.flatten(features_output, 1)
print(f"Flatten 后: {flattened.shape}")
print(f"特征维度: {flattened.shape[1]}")

print("\n\n" + "=" * 100)
print("2️⃣ 测试不同输入大小（256×256）")
print("=" * 100)

x_256 = torch.randn(1, 3, 256, 256)
features_output_256 = vgg16.features(x_256)
print(f"\n输入:  {x_256.shape}")
print(f"输出:  {features_output_256.shape}")
print(f"❌ 大小改变了！Flatten 会失败或产生不同的特征维度")

# 展平后会有问题
flattened_256 = torch.flatten(features_output_256, 1)
print(f"Flatten 后: {flattened_256.shape}")
print(f"特征维度: {flattened_256.shape[1]}")
print(f"\n⚠️  特征维度不同：{flattened.shape[1]} vs {flattened_256.shape[1]}")

print("\n\n" + "=" * 100)
print("3️⃣ 使用 AdaptiveAvgPool2d 统一维度")
print("=" * 100)

pool = torch.nn.AdaptiveAvgPool2d((7, 7))

# 224×224 输入
pooled_224 = pool(features_output)
flattened_pooled_224 = torch.flatten(pooled_224, 1)
print(f"\n224×224 输入:")
print(f"  特征输出: {features_output.shape}")
print(f"  池化后:   {pooled_224.shape}")
print(f"  Flatten: {flattened_pooled_224.shape}")

# 256×256 输入
pooled_256 = pool(features_output_256)
flattened_pooled_256 = torch.flatten(pooled_256, 1)
print(f"\n256×256 输入:")
print(f"  特征输出: {features_output_256.shape}")
print(f"  池化后:   {pooled_256.shape}")
print(f"  Flatten: {flattened_pooled_256.shape}")

print(f"\n✅ 统一了！都是 {flattened_pooled_224.shape}")

print("\n\n" + "=" * 100)
print("总结")
print("=" * 100)

print("""
❓ 问题：AdaptiveAvgPool2d 需要自己操作吗？

✅ 答案：不需要！这个池化层已经在我们的代码中自动处理了

📍 代码位置：models/backbone.py 的 _create_vgg16() 方法

return nn.Sequential(
    features,                         # ← 自动处理卷积
    nn.AdaptiveAvgPool2d((7, 7)),    # ← 自动处理池化（我们已经加了）
    nn.Flatten()                      # ← 自动处理展平
)

🔄 完整流程：
1. 输入图像 → features (卷积)
2. 输出可能不同大小 (e.g., [B,512,7,7] 或 [B,512,8,8])
3. AdaptiveAvgPool2d 自动统一到 [B,512,7,7]
4. Flatten 转为 [B, 25088]
5. 送入 Classifier

💡 要点：
• 所有这些操作都在 Backbone 内部自动完成
• 你不需要在训练时做任何特殊操作
• forward() 时只需要 model(images) 就行
• 其他所有的层变换都自动处理

用法：
  model = BackboneModel(model_name='vgg16', include_top=True)
  output = model(images)  # ← 自动处理所有的池化、展平、分类

一行代码，搞定所有！
""")

print("\n" + "=" * 100)
print("✨ 结论：不需要自己操作，代码已经帮你处理好了")
print("=" * 100 + "\n")
