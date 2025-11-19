# Vision Transformer 初始化策略分析 - 可借鉴点

## Vision Transformer 的初始化方式

你找到的代码展示了标准 Vision Transformer 的初始化：

```python
def _init_vit_weights(m):
    """ViT weight initialization"""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)  # ← 关键！
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# 应用到整个模型
self.apply(_init_vit_weights)
```

### 关键特点

1. **Truncated Normal 初始化**（而不是 Xavier 或 Kaiming）
   - 使用 `trunc_normal_` 而不是 `xavier_uniform_` 或 `kaiming_uniform_`
   - std=0.01（非常小！但这是 ViT 的特点）

2. **Bias 初始化为 0**
   - 这是标准做法

3. **LayerNorm 的初始化**
   - Weight 初始化为 1（保持恒等）
   - Bias 初始化为 0

4. **位置编码初始化**
   ```python
   nn.init.trunc_normal_(self.pos_embed, std=0.02)
   nn.init.trunc_normal_(self.cls_token, std=0.02)
   ```

## 为什么 ViT 用 Truncated Normal？

### 对比：Xavier vs Kaiming vs Truncated Normal

```python
# Xavier (我们当前用的)
nn.init.xavier_uniform_(weight, gain=1.0)
# Var(W) = 2 / (fan_in + fan_out)
# std ~ 0.02-0.06

# Kaiming (PyTorch 默认)
nn.init.kaiming_uniform_(weight, a=sqrt(5))
# Var(W) = 2 / fan_in
# std ~ 0.01-0.02

# Truncated Normal (ViT 用的)
nn.init.trunc_normal_(weight, std=0.01)
# 完全由 std 参数控制
# std = 0.01 (固定值，小于 Xavier 和 Kaiming)
```

### ViT 为什么选择更小的 std？

1. **ViT 有大量的 Layer Normalization**
   - 每个 Block 都有 LayerNorm
   - LayerNorm 稳定了激活值
   - 所以不需要大的初始化权重

2. **ViT 使用 Transformer Blocks**
   - 有残差连接（skip connection）
   - 残差连接让梯度流动更稳定
   - 所以权重可以更小（类似 ResNet）

3. **ViT 初始化哲学**
   - "小而稳定" 比 "大而激进" 更好
   - 权重小 + LayerNorm + 残差连接 = 稳定训练

## 我们的 ViT-Hybrid 能否借鉴？

### 当前架构

ViT-Hybrid 有：
- ✓ LayerNorm（在 Transformer Block 中）
- ✓ 残差连接（在 Transformer Block 中）
- ✗ **分类头没有 LayerNorm**（问题！）
- ✗ **分类头没有残差连接**（问题！）

### 问题诊断

我们遇到的低方差问题：
```
Transformer 输出特征方差: 0.0001 (很小，但正常，因为有 LayerNorm)
      ↓
分类头没有 LayerNorm 来稳定这个小方差
      ↓
输出方差: 0.00006 (更小)
```

## 改进方案

### 方案 A：添加 LayerNorm（推荐）

```python
self.classifier = nn.Sequential(
    nn.LayerNorm(d_model),           # ← 来自 Vision Transformer 的启发
    nn.Dropout(0.5),
    nn.Linear(d_model, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

# 现在可以用更小的初始化了
for module in self.classifier:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.01)  # 像 ViT 一样
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

**优点**：
- LayerNorm 稳定输入特征的方差
- 可以使用更小的权重初始化
- 与 ViT 保持一致的哲学
- 更稳定的训练

### 方案 B：使用 Truncated Normal（更激进的改进）

```python
# 导入 truncated normal
from timm.layers import trunc_normal_

self.classifier = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Dropout(0.5),
    nn.Linear(d_model, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

# 用 ViT 的初始化方式
for module in self.classifier:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

**优点**：
- 完全采用 Vision Transformer 的标准做法
- Truncated Normal 避免极端值
- 更稳定的初始化

### 方案 C：保持当前方案但理解原因

```python
# 当前的 Xavier (gain=1.0) 也可以，因为：
# 1. 分类头是简单的 FC 层
# 2. 比 ViT 的方法更激进，但这样也OK
# 3. 关键是添加 LayerNorm 来稳定输入特征方差
```

## Vision Transformer 代码的其他启发

### 1. Stochastic Depth（随机深度）

```python
dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
# drop_path_ratio 从 0 逐渐增加到最大值
# 深层网络的 dropout 率更高
```

**启发**：可以在深层 Transformer 中使用更高的 dropout

### 2. PatchEmbed 初始化

```python
# Conv2d 用 Kaiming
nn.init.kaiming_normal_(m.weight, mode="fan_out")

# Linear 用 Truncated Normal
nn.init.trunc_normal_(m.weight, std=.01)
```

**启发**：卷积层和全连接层用不同的初始化方法

### 3. Position Embedding 初始化

```python
nn.init.trunc_normal_(self.pos_embed, std=0.02)
nn.init.trunc_normal_(self.cls_token, std=0.02)
```

**启发**：特殊的 token 和 embedding 用较小的初始化（std=0.02）

## 推荐的改进步骤

### 立即可做（下次训练）

添加 LayerNorm 到分类头：

```python
self.classifier = nn.Sequential(
    nn.LayerNorm(d_model),           # 新增
    nn.Dropout(0.5),
    nn.Linear(d_model, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

# 保持当前的 Xavier (gain=1.0) 初始化
for module in self.classifier:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.constant_(module.bias, 0.0)
```

### 进阶改进（如果 LayerNorm 不够）

尝试 ViT 的 Truncated Normal 初始化：

```python
from torch.nn.init import trunc_normal_

for module in self.classifier:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

## 总结对比表

| 方面 | Vision Transformer | ViT-Hybrid 当前 | ViT-Hybrid 建议 |
|-----|------------------|------------|------------|
| **分类头结构** | Linear | Linear + Dropout | LayerNorm + Linear + Dropout |
| **初始化方法** | Truncated Normal (std=0.01) | Xavier (gain=1.0) | Xavier (gain=1.0) 或 Trunc Normal |
| **位置编码** | 有 + 初始化 std=0.02 | 无（时间编码用参数） | 可考虑 |
| **LayerNorm** | 大量（每个 Block） | 有（Transformer 中） | **建议添加到分类头** |
| **残差连接** | 有（每个 Block） | 有（Transformer 中） | 分类头无（不需要） |
| **输出稳定性** | ✓ 很稳定 | ⚠ 方差小 | ✓ 改进 |

## 最终建议

**立即采取**（下次训练立即可做）：

```python
# model/model.py 中的分类头改为：
self.classifier = nn.Sequential(
    nn.LayerNorm(d_model),           # ← 新增，参考 Vision Transformer
    nn.Dropout(0.5),
    nn.Linear(d_model, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

# 初始化保持不变（或改为 Truncated Normal）
for module in self.classifier:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

**优点**：
- ✓ 稳定输入特征的方差（像 Vision Transformer 一样）
- ✓ 使用业界标准的做法
- ✓ 可以改善训练稳定性
- ✓ 代码改动最小

