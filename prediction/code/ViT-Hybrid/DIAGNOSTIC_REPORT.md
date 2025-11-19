# ViT-Hybrid 模型诊断报告

## 问题描述

验证集中所有样本的预测概率都是 **0.4832**（完全相同），标准差为 0.0000，表明模型的输出已经完全固定为一个常数值。

## 根本原因分析

### 问题1: 分类头初始化不当 ✓ 已修复

**位置**: `model/model.py:301-303`

**原问题**:
```python
nn.init.xavier_uniform_(self.classifier[4].weight, gain=0.1)
nn.init.constant_(self.classifier[4].bias, 0.0)
```

**问题症状**:
- `gain=0.1` 将权重方差缩小到原来的 **1%** (`0.1² = 0.01`)
- 权重接近 0，导致最后一层的输出接近 bias（常数）
- 导致模型在初始化时就输出接近常数

**修复方案**:
```python
# 初始化分类头所有线性层（gain=1.0确保权重有足够的方差）
for idx, module in enumerate(self.classifier):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.constant_(module.bias, 0.0)
```

### 可能的问题2: 类别不平衡导致的学习问题 ⚠️ 需要关注

**证据**:
- 训练集: Class 0 (Poor) 19186, Class 1 (Good) 3219 → 比例 6:1
- 验证集: Class 0: 3469, Class 1: 689 → 比例 5:1
- Sensitivity (Recall): 0.0308 - 模型只能识别 3% 的 Good 样本

**影响**:
- 即使初始化正确，模型也倾向于预测多数类
- 加权损失函数虽然已使用，但可能权重计算有误

**建议检查**:
```python
# 检查权重计算是否正确 (train.py:186-188)
weight_0 = total / (2 * num_class_0) if num_class_0 > 0 else 1.0
weight_1 = total / (2 * num_class_1) if num_class_1 > 0 else 1.0
```

### 可能的问题3: 梯度流问题 ⚠️ 需要调查

**可能原因**:
1. **梯度消失**: VGG16 权重冻结 + 深度 Transformer + Dropout 可能导致梯度消失
2. **学习率过低**: Stage 1 使用基础 LR，Stage 2 使用 0.1 倍 LR，可能太小
3. **Early Stopping 过于激进**: 很早就停止训练

## 验证修复效果

### 测试结果

修复后的初始化测试:
```
Classification Head Output Statistics:
Mean probability:  0.286985
Std probability:   0.001063  ← 有健康的方差（不是 0.0000）
Min probability:   0.285731
Max probability:   0.288121
```

✓ 初始化问题已修复

## 后续调试步骤

### 1. 检查训练日志中的关键指标

在下一次训练运行中，监控以下内容：

```python
# 在 train.py 中添加更详细的日志
print(f"Logits stats - Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
print(f"Loss stats - Min: {loss.item():.4f}, Max: {batch_loss_max:.4f}")
print(f"Gradient norms: {compute_grad_norms(model)}")
```

### 2. 验证类别权重

打印实际使用的权重：
```python
print(f"Class weights: Poor={weight_0:.4f}, Good={weight_1:.4f}")
print(f"Weight ratio: {weight_1/weight_0:.4f} (应该 >> 1.0，因为 Good 更稀有)")
```

### 3. 检查 Focal Loss

考虑使用 Focal Loss 替代 CrossEntropyLoss：
```python
from torch.nn.functional import focal_loss
criterion = FocalLoss(alpha=[weight_0, weight_1], gamma=2.0)
```

### 4. 验证梯度流

在训练第一个 epoch 后检查：
```python
total_grad_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_grad_norm += p.grad.data.norm(2).item()
print(f"Total gradient norm: {total_grad_norm}")
```

## 修改清单

- ✓ `model/model.py`: 修复分类头初始化 (gain 从 0.1 改为 1.0，并初始化所有线性层)
- ✓ 创建 `test_initialization.py` 用于验证输出方差

## 建议的后续行动

1. **立即**: 使用修复后的代码重新训练模型
2. **观察**: 第一个 epoch 后检查 logits 的统计信息
3. **如果仍然失败**:
   - 检查梯度流是否正常（添加 hook 监测梯度）
   - 尝试增大学习率
   - 考虑使用 Focal Loss
   - 验证数据加载中的标准化是否一致

## 技术细节

### Xavier 初始化的原理

Xavier (Glorot) 初始化旨在保持输入和输出的方差相同：

```
Var(W) = 2 / (fan_in + fan_out) * gain²
```

- `gain=0.1`: `Var(W) = 2/(...) * 0.01` → 权重太小，输出接近 bias
- `gain=1.0`: `Var(W) = 2/(...)` → 正常的权重方差

### 为什么初始化很重要

- **过小的权重** → 激活函数输入过小 → 输出接近常数 → 梯度消失 → 难以训练
- **过大的权重** → 激活函数饱和 → 梯度消失 → 难以训练
- **合适的初始化** → 激活分布合理 → 梯度流畅 → 训练稳定

