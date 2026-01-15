# PCA 分析工具使用指南

## 问题诊断

**原问题**：`analyze_pc_variations.py` 生成的图片在散点图上找不到

**根本原因**：
1. 🎲 **随机采样**：每次运行时 `np.random.choice()` 选择不同的样本子集
2. 📐 **PCA 坐标系不同**：不同样本子集训练出的 PCA 主成分方向不同
3. ❌ **数据不一致**：两个脚本使用的数据集实际上不同

## 解决方案

创建了 **统一的 PCA 分析系统**，确保：
- ✅ 固定随机种子（`random_seed=42`）
- ✅ 使用相同的数据子集
- ✅ 使用相同的 PCA 模型
- ✅ 缓存机制确保可重复性

---

## 新工具介绍

### 1. **utils/pca_cache.py** - PCA 缓存管理器

**功能**：保存和加载特征、PCA 模型、标签、路径

```python
from utils.pca_cache import PCACache, create_pca_config

# 创建缓存管理器
cache = PCACache(cache_dir='outputs/pca_cache')

# 保存
config = create_pca_config(
    model_path='model.pth',
    csv_path='data.csv',
    ...
    random_seed=42
)
cache.save(features, labels, paths, pca_model, config)

# 加载
cache_data = cache.load(config_dict=config)
features_pca = cache_data['features_pca']
```

### 2. **unified_pca_analysis.py** - 统一分析脚本 ⭐

**一键生成所有可视化**：
- 📊 PCA/t-SNE/UMAP 散点图
- 🖼️ PC1/PC2/PC3 排序的图片序列
- 🔍 PC 对称的图片对

**使用方法**：

```bash
cd /Users/lzz/Documents/GitHub/IVF_AI/prediction/code/STORK_lzz_torch

# 基础用法（使用默认参数）
python unified_pca_analysis.py

# 自定义参数
python unified_pca_analysis.py \
    --model_path ../vgg16_best_val_accuracy.pth \
    --output_dir outputs/my_analysis \
    --max_samples_per_class 1000 \
    --random_seed 42

# 强制重新计算（不使用缓存）
python unified_pca_analysis.py --no_cache

# 调整 PCA 主成分数量
python unified_pca_analysis.py --n_components 5
```

**输出结构**：

```
outputs/unified_pca_analysis/
├── pca_cache/               # PCA 缓存（确保一致性）
│   └── pca_cache_a3b7c2d1.pkl
├── scatter_plots/           # 散点图
│   ├── pca_scatter.pdf
│   ├── tsne_scatter.pdf
│   └── umap_scatter.pdf
├── pc_sequences/            # 按 PC 排序的图片序列
│   ├── pc1_sorted_sequence.pdf
│   ├── pc2_sorted_sequence.pdf
│   └── pc3_sorted_sequence.pdf
└── pc_pairs/                # PC 对称的图片对
    ├── PC_Symmetry_pc1_similar_pc2_opposite.pdf
    └── PC_Symmetry_pc2_similar_pc1_opposite.pdf
```

### 3. **修改后的原脚本**

#### visualize_three_classes.py
- ✅ 新增 `random_seed` 参数
- ✅ 确保采样可重复

```python
dataset = ThreeClassEmbryoDataset(
    csv_path=csv_path,
    base_dir=base_dir,
    fair_dir=fair_dir,
    transform=transform,
    random_seed=42  # 新增：固定随机种子
)
```

#### analyze_pc_variations.py
- ✅ 支持 PC3 分析
- ✅ PCA 降维改为 3 个主成分

---

## 使用建议

### 方案 A：使用统一脚本（推荐）⭐

```bash
# 一次生成所有可视化，数据完全一致
python unified_pca_analysis.py --random_seed 42
```

**优点**：
- 数据 100% 一致
- 一键生成所有结果
- 自动缓存，下次运行更快

### 方案 B：分别运行（需确保参数一致）

```bash
# 1. 先运行散点图生成
python visualize_three_classes.py \
    --model_path ../vgg16_best_val_accuracy.pth \
    --max_samples_per_class 1000 \
    --random_seed 42  # 必须指定相同种子

# 2. 再运行图片序列（需修改代码设置相同的 random_seed）
python analyze_pc_variations.py
```

**注意**：需要手动修改 `analyze_pc_variations.py` 的代码，添加：

```python
dataset = ThreeClassEmbryoDataset(
    ...
    random_seed=42  # 必须与上面一致
)
```

---

## 验证数据一致性

运行统一脚本后，检查输出：

```
✅ PCA缓存已保存: outputs/unified_pca_analysis/pca_cache/pca_cache_a3b7c2d1.pkl
   缓存ID: a3b7c2d1
   样本数: 3000
   特征维度: 512
   PCA维度: 3
   方差解释率: [0.342 0.198 0.087]
```

**如何验证**：
1. 在散点图上找一个特殊点（比如 PC1=3.5, PC2=-2.1）
2. 在 PC 排序序列中找到相同坐标的图片
3. 确认是同一张图片 ✅

---

## 常见问题

### Q1: 缓存文件在哪里？
**A**: `outputs/unified_pca_analysis/pca_cache/pca_cache_*.pkl`

### Q2: 如何清除缓存重新计算？
**A**:
```bash
rm -rf outputs/unified_pca_analysis/pca_cache
# 或使用 --no_cache 参数
python unified_pca_analysis.py --no_cache
```

### Q3: 不同参数会生成不同缓存吗？
**A**: 是的。缓存 ID 基于配置哈希，不同参数会生成不同缓存。

### Q4: 如何查看所有可用缓存？
**A**:
```bash
python -c "from utils.pca_cache import PCACache; PCACache().list_caches()"
```

### Q5: 为什么需要固定 random_seed？
**A**:
- `max_samples_per_class=1000` 时会随机采样
- 不同的 seed 会选择不同的样本子集
- 固定 seed 确保每次选择相同的样本

---

## 工作流程示例

### 场景 1: 第一次分析

```bash
# 运行统一分析
python unified_pca_analysis.py --random_seed 42

# 输出：
# ✅ 特征提取完成
# ✅ PCA 训练完成
# ✅ 缓存已保存
# ✅ 生成散点图
# ✅ 生成图片序列
# ✅ 生成对称图片对
```

### 场景 2: 修改参数重新分析

```bash
# 更改样本数
python unified_pca_analysis.py \
    --random_seed 42 \
    --max_samples_per_class 500  # 改为 500

# 会生成新的缓存（因为参数不同）
```

### 场景 3: 使用缓存快速重新生成可视化

```bash
# 相同参数，使用缓存
python unified_pca_analysis.py --random_seed 42

# 输出：
# ✅ 已加载 PCA 缓存
# ⚡ 跳过特征提取（使用缓存）
# ✅ 生成所有可视化
```

---

## 技术细节

### 随机种子的作用范围

```python
# 设置种子后影响：
np.random.seed(42)

# 1. Fair 胚胎采样
np.random.choice(fair_images, max_fair_samples, replace=False)

# 2. 类别平衡采样
np.random.choice(indices, max_samples_per_class, replace=False)

# 3. t-SNE 降维
TSNE(random_state=42)

# 4. UMAP 降维
UMAP(random_state=42)
```

### 缓存 ID 计算

```python
# 基于这些参数生成唯一 ID：
{
    'model_path': '/path/to/model.pth',
    'csv_path': '/path/to/data.csv',
    'max_samples_per_class': 1000,
    'max_fair_samples': 1000,
    'random_seed': 42,
    ...
}
# → MD5 哈希 → 'a3b7c2d1'
```

---

## 总结

| 问题 | 旧方案 | 新方案 |
|------|--------|--------|
| 数据一致性 | ❌ 每次不同 | ✅ 完全一致 |
| 可重复性 | ❌ 无法重现 | ✅ 100% 重现 |
| 使用便利性 | ❌ 需分别运行 | ✅ 一键生成 |
| 性能 | ❌ 重复计算 | ✅ 缓存加速 |

**推荐做法**：
1. ✅ 使用 `unified_pca_analysis.py` 一键生成所有分析
2. ✅ 固定 `random_seed=42` 确保可重复性
3. ✅ 首次运行后会自动缓存，后续分析更快
4. ✅ 所有可视化使用完全相同的数据和 PCA 模型

现在散点图上的每个点都能在图片序列中找到对应的图片了！🎉
