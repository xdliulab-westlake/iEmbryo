# 故障排除指南

常见问题和解决方案

---

## 🔧 导入错误

### 1. ModuleNotFoundError: No module named 'training'

**错误信息**:
```
ModuleNotFoundError: No module named 'training'
```

**原因**: Python找不到training模块

**解决方案**:

✅ **方案1: 使用train.py（推荐）**
```bash
python train.py --maindb /path/to/MainDB.csv --selected_images /path/to/selected_image_paths.csv
```

✅ **方案2: 确保在项目根目录运行**
```bash
cd /path/to/ViT-Hybrid
python -m training.train_tmfi --maindb /path/to/MainDB.csv --selected_images /path/to/selected_image_paths.csv
```

---

### 2. ImportError: cannot import name 'parse_args'

**错误信息**:
```
ImportError: cannot import name 'parse_args' from 'training.train_tmfi'
```

**原因**: train.py文件版本过旧

**解决方案**:
已修复，重新运行即可。如果仍有问题，请使用最新的train.py文件。

---

## ⚠️ 警告信息

### 1. timm FutureWarning

**警告信息**:
```
FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
```

**说明**:
这是timm库的版本兼容性警告，**不影响训练**，可以安全忽略。

**解决方案（可选）**:
如果想消除警告，可以升级timm版本：
```bash
pip install --upgrade timm
```

或者在代码开头添加：
```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
```

---

## 💾 数据错误

### 1. FileNotFoundError: MainDB.csv not found

**错误信息**:
```
FileNotFoundError: MainDB.csv not found: /path/to/MainDB.csv
```

**原因**: 数据文件路径不正确

**解决方案**:
1. 检查文件是否存在:
   ```bash
   ls -la /path/to/MainDB.csv
   ```

2. 使用绝对路径:
   ```bash
   python train.py --maindb /absolute/path/to/MainDB.csv
   ```

3. 或使用相对路径（相对于当前目录）:
   ```bash
   python train.py --maindb ./data/MainDB.csv
   ```

---

### 2. No quality label found for embryo

**错误信息**:
```
Warning: Embryo E001 rejected - No quality label
```

**原因**: 胚胎在MainDB.csv中缺少Quality字段

**解决方案**:
检查MainDB.csv，确保包含Quality列：
```csv
EmbryoID,ICM,TE,Quality
E001,A,A,Good
E002,B,C,Poor
```

Quality字段必须是以下之一：
- `Good`, `good` → 好胚胎
- `Poor`, `poor`, `Fair`, `fair` → 差胚胎

---

### 3. Incomplete timepoints

**错误信息**:
```
Warning: Embryo E002 rejected - Incomplete timepoints (found 30/33)
```

**原因**: 胚胎数据不完整，缺少时间点

**解决方案**:
检查selected_image_paths.csv，确保每个胚胎有33个完整时间点（0-32）：
```bash
# 检查某个胚胎的时间点数量
grep "E002" selected_image_paths.csv | cut -d',' -f2 | sort -u | wc -l
# 应该输出: 33
```

---

### 4. Incomplete focal planes

**错误信息**:
```
Warning: Embryo E003 rejected - Incomplete focal planes at timepoint 5 (found 9/11)
```

**原因**: 某个时间点缺少焦平面数据

**解决方案**:
检查selected_image_paths.csv，确保每个时间点有11个焦平面（0-10）：
```bash
# 检查某个胚胎某个时间点的焦平面数量
grep "E003,5," selected_image_paths.csv | wc -l
# 应该输出: 11
```

---

## 🖥️ CUDA/GPU错误

### 1. CUDA out of memory

**错误信息**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**解决方案**:

✅ **方案1: 减小batch size**
```bash
python train.py --batch_size 2  # 从4减到2
```

✅ **方案2: 减少num_workers**
```bash
python train.py --num_workers 2  # 从4减到2
```

✅ **方案3: 使用CPU训练（慢但稳定）**
```bash
CUDA_VISIBLE_DEVICES="" python train.py ...
```

✅ **方案4: 清空GPU缓存**
```python
import torch
torch.cuda.empty_cache()
```

---

### 2. CUDA driver version mismatch

**错误信息**:
```
CUDA driver version is insufficient for CUDA runtime version
```

**解决方案**:
检查CUDA版本并安装匹配的PyTorch：
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装匹配的PyTorch
# CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

---

## 🏋️ 训练问题

### 1. Loss不下降

**现象**: 训练多个epoch后loss仍然很高

**可能原因和解决方案**:

✅ **检查数据加载**:
```python
# 在训练循环前添加
for inputs, labels in train_loader:
    print(f"Batch shape: {inputs.shape}")
    print(f"Labels: {labels}")
    print(f"Labels range: {labels.min()}-{labels.max()}")
    break
```

✅ **调整学习率**:
```bash
# 尝试更大的学习率
python train.py --lr 1e-3  # 从1e-4增加到1e-3
```

✅ **检查类别平衡**:
训练时会打印类别分布，确保数据平衡：
```
Class distribution:
  Class 0 (Poor): 50 samples (50.0%)
  Class 1 (Good): 50 samples (50.0%)
```

---

### 2. 训练很慢

**解决方案**:

✅ **使用离线预处理**:
训练脚本会自动使用离线预处理和缓存。

✅ **增加num_workers**:
```bash
python train.py --num_workers 8  # 根据CPU核心数调整
```

✅ **使用更快的存储**:
将数据移动到SSD而不是HDD。

---

### 3. 验证集AUC很低

**现象**: 训练集loss下降但验证集AUC很低

**可能原因**: 过拟合或数据分布不一致

**解决方案**:

✅ **检查数据划分**:
```python
# 训练时会打印数据划分信息
Training set: 105 embryos (70.0%)
Validation set: 22 embryos (15.0%)
Test set: 23 embryos (15.0%)
```

✅ **使用早停**:
已自动启用，patience=15个epoch

✅ **查看混淆矩阵**:
训练时会定期打印混淆矩阵，检查模型是否偏向某一类。

---

## 📊 评估问题

### 1. 所有预测都是同一类

**现象**: 模型总是预测Good或Poor

**解决方案**:

✅ **检查类别权重**:
代码已使用平衡采样（WeightedRandomSampler）。

✅ **查看概率分布**:
训练时会打印概率统计：
```
Probability statistics (class 1 - Good):
  Mean: 0.5234, Std: 0.2156
  Min: 0.0123, Max: 0.9876
  > 0.5: 65 / 100
```

✅ **尝试不同的学习率**:
```bash
python train.py --lr 5e-4
```

---

## 🔍 调试技巧

### 1. 启用详细日志

在train.py开头添加：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 检查模型输出

在训练循环中添加：
```python
print(f"Saliency map shape: {saliency_map.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
```

### 3. 可视化注意力图

训练会自动保存注意力图到`results/attention_maps/`，可以使用：
```python
import torch
import matplotlib.pyplot as plt

attention = torch.load('results/attention_maps/attention_epoch_099.pt')
plt.imshow(attention[0, 0].cpu(), cmap='hot')
plt.colorbar()
plt.savefig('attention.png')
```

---

## 📞 获取帮助

如果以上方案都无法解决问题：

1. **检查日志**: 查看完整的错误堆栈信息
2. **查看文档**: 阅读 [README.md](README.md) 和 [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)
3. **检查数据**: 使用 [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) 中的检查清单
4. **报告问题**: 提供完整的错误信息和运行命令

---

**最后更新**: 2025-12-22
**版本**: 1.0.0
