# ViT-Hybrid 时间点可解释性分析指南

## 概述

本指南详细说明如何使用 ViT-Hybrid 模型的**时间点重要性分析**功能，理解模型如何做出预测决策。

## 问题设定

### 核心问题

在胚胎发育过程中（96-112小时），不同时间点对质量评估的重要性不同。我们想知道：

1. **时间点重要性**：哪些时间点最影响模型的判断？
2. **时间点关联**：不同时间点间如何相互影响？
3. **发育过程理解**：模型是否抓住了真实的生物学特征？

## 快速开始

### 一键分析

```bash
cd /Users/lzz/Documents/GitHub/IVF_AI/prediction/code/ViT-Hybrid

python analyze_temporal_interpretability.py \
    --model checkpoints/final_model.pth \
    --output results/temporal_analysis
```

### 使用GPU加速

```bash
python analyze_temporal_interpretability.py \
    --model checkpoints/final_model.pth \
    --device cuda \
    --output results/temporal_analysis
```

## 四种分析方法

### 方法1：梯度归因法（最快）

```python
from analysis.temporal_interpretability import compute_temporal_importance

importance_scores = compute_temporal_importance(model, images, device='cuda')
# importance_scores: (33,) - 每个时间点的重要性分数
```

### 方法2：注意力权重可视化（最直观）

```python
from analysis.attention_visualization import (
    extract_attention_weights,
    visualize_attention_heatmap
)

attention_weights = extract_attention_weights(model, images)
visualize_attention_heatmap(attention_weights, save_path='attention.png')
```

### 方法3：特征相关性分析（理解连续性）

```python
from analysis.temporal_interpretability import compute_temporal_correlation

correlation_matrix, info = compute_temporal_correlation(model, images)
# correlation_matrix: (33, 33) - 时间点间的特征相似度
```

### 方法4：消融分析（最严谨但最慢）

```python
from analysis.ablation_analysis import temporal_ablation_analysis

ablation_scores, _ = temporal_ablation_analysis(model, images)
# ablation_scores: (33,) - 删除该时间点后的预测变化
```

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `temporal_importance.json` | 梯度归因法结果 |
| `attention_analysis.json` | 注意力权重统计 |
| `ablation_analysis.json` | 消融分析结果 |
| `attention_heatmap.png` | 注意力热力图 |
| `temporal_correlation.png` | 特征相关性矩阵 |
| `temporal_ablation.png` | 消融分析图 |

## 核心概念

### 时间点重要性
- **高分值**：模型高度依赖该时间点的信息
- **低分值**：模型较少使用该时间点的信息
- 通常关键发育时期（约102-106小时）重要性最高

### 注意力权重
- **热力图行**：查询时间点
- **热力图列**：被关注的时间点
- **颜色亮度**：注意力权重大小
- **对角线强**：时间点重视自身信息
- **非对角线强**：时间点间信息交互

### 特征相关性
- **接近1**：特征相似，相邻发育阶段
- **接近0**：特征不同，差异明显
- **块状结构**：不同发育阶段的聚类

### 消融分数
- **高分值**：删除该时间点大幅改变预测
- **低分值**：删除该时间点预测变化不大
- 最可信但计算量大（33次前向传播）

## 编程接口详解

### 时间点重要性

```python
from analysis.temporal_interpretability import (
    compute_temporal_importance,
    get_critical_timepoints,
    print_temporal_importance_report
)

# 计算重要性
importance = compute_temporal_importance(model, images, device='cuda')

# 找出最重要的5个时间点
critical = get_critical_timepoints(importance, topk=5)
print(f"Top-5 hours: {critical['hours']}")
print(f"Top-5 scores: {critical['scores']}")

# 生成报告
report = print_temporal_importance_report(importance, topk=10)
print(report)
```

### 注意力权重

```python
from analysis.attention_visualization import (
    extract_attention_weights,
    visualize_attention_heatmap,
    visualize_attention_flow,
    get_most_attended_timepoints
)

# 提取权重
attn_weights = extract_attention_weights(model, images)

# 可视化热力图
visualize_attention_heatmap(attn_weights, save_path='attention.png')

# 可视化注意力流
visualize_attention_flow(attn_weights, source_timepoint=0, save_path='flow.png')

# 找最被关注的时间点
most_attended = get_most_attended_timepoints(attn_weights, topk=5)
```

### 焦平面分析

```python
from analysis.ablation_analysis import (
    focal_plane_ablation_analysis,
    visualize_focal_plane_ablation
)

# 分析特定时间点的焦平面重要性
scores, _ = focal_plane_ablation_analysis(model, images, target_timepoint=16)

# 可视化
visualize_focal_plane_ablation(scores, target_timepoint=16)
```

## 最佳实践

1. **同时使用多种方法**进行对比验证
2. **对多个样本进行分析**，观察一致性
3. **生成图表**便于理解和交流
4. **与领域专家讨论**结果的生物学合理性
5. **定期更新分析**跟踪模型演进

## 常见问题

**Q: 梯度为什么是零？**
A: 确保 `images.requires_grad = True` 或在函数内自动计算

**Q: 为什么不同方法结果不一致？**
A: 正常现象。梯度、注意力、消融各测量不同方面。建议同时参考。

**Q: 热力图全是一样的颜色？**
A: 可能模型尚未充分训练，或数据不足以区分时间点

---

更详细的使用说明请参考 README.md 第九章 "模型可解释性分析"
