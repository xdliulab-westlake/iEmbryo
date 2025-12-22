# 数据标签要求说明

Multi-Task TMFI-Net需要以下数据文件和标签：

---

## 📋 必需文件

### 1. MainDB.csv（必需）

**用途**: 提供胚胎质量标签和形态学评分

**必需字段**:

| 字段名 | 类型 | 必需 | 说明 | 示例 |
|--------|------|------|------|------|
| `EmbryoID` | string | ✅ | 胚胎唯一标识符 | `"E001"`, `"P123_E02"` |
| `Quality` | string | ✅ | 胚胎质量标签 | `"Good"`, `"Poor"`, `"good"`, `"poor"` |
| `ICM` | string | ⚠️ | 内细胞团评分 | `"A"`, `"B"`, `"C"` |
| `TE` | string | ⚠️ | 滋养层评分 | `"A"`, `"B"`, `"C"` |

⚠️ `ICM`和`TE`用于计算BS (Blastocyst Score)，如果缺失会使用默认值。

**示例**:
```csv
EmbryoID,ICM,TE,Quality
E001,A,A,Good
E002,B,B,Good
E003,C,C,Poor
E004,B,C,Poor
```

**质量标签映射**:
- `"Good"`, `"good"` → 1 (好胚胎)
- `"Poor"`, `"poor"`, `"Fair"`, `"fair"` → 0 (差胚胎)

**BS评分计算**:
```python
ICM评分: {'A': 1, 'B': 2, 'C': 4}
TE评分:  {'A': 1, 'B': 3, 'C': 4}
BS分数 = ICM评分 + TE评分  # 范围: 2-8
```

---

### 2. selected_image_paths.csv（必需）

**用途**: 提供多焦距多时间点图像路径

**必需字段**:

| 字段名 | 类型 | 必需 | 说明 | 示例 |
|--------|------|------|------|------|
| `EmbryoID` | string | ✅ | 胚胎ID（需与MainDB.csv匹配） | `"E001"` |
| `Timepoint` | int | ✅ | 时间点索引 (0-32) | `0`, `1`, ..., `32` |
| `FocalPlane` | int | ✅ | 焦平面索引 (0-10) | `0`, `1`, ..., `10` |
| `ImagePath` | string | ✅ | 图像文件绝对路径或相对路径 | `"/data/images/E001_t0_f0.jpg"` |

或者如果没有`EmbryoID`字段，需要：

| 字段名 | 类型 | 必需 | 说明 | 示例 |
|--------|------|------|------|------|
| `PatientID` | string | ✅ | 患者ID | `"P123"` |
| `EmbryoNumber` | int | ✅ | 胚胎编号 | `1`, `2`, `3` |

系统会自动创建`EmbryoID = PatientID + "_" + EmbryoNumber`

**示例**:
```csv
EmbryoID,Timepoint,FocalPlane,ImagePath
E001,0,0,/data/images/E001_t00_f00.jpg
E001,0,1,/data/images/E001_t00_f01.jpg
E001,0,2,/data/images/E001_t00_f02.jpg
...
E001,0,10,/data/images/E001_t00_f10.jpg
E001,1,0,/data/images/E001_t01_f00.jpg
...
E001,32,10,/data/images/E001_t32_f10.jpg
```

**数据完整性要求**:
- ✅ 每个胚胎必须有 **33个时间点** (Timepoint 0-32)
- ✅ 每个时间点必须有 **11个焦平面** (FocalPlane 0-10)
- ✅ 总计每个胚胎需要 **33 × 11 = 363张图像**
- ✅ 图像文件必须存在且可访问

**时间点说明**:
- 时间范围: 96-112小时 (受精后第4-5天)
- 时间间隔: 0.5小时
- 时间点数量: 33个
- Timepoint 0 = 96h, Timepoint 32 = 112h

**焦平面说明**:
- 焦平面数量: 11个
- FocalPlane 0-10，其中5为中间焦平面

---

### 3. filtered_clean_embryos.csv（可选）

**用途**: 提供额外的胚胎质量标签（补充MainDB.csv）

**必需字段**:

| 字段名 | 类型 | 必需 | 说明 | 示例 |
|--------|------|------|------|------|
| `EmbryoID` | string | ✅ | 胚胎ID | `"E005"` |
| `EmbryoQuality` | string | ✅ | 质量标签 | `"Good"`, `"Poor"`, `"viable"`, `"non_viable"` |

**示例**:
```csv
EmbryoID,EmbryoQuality
E005,Good
E006,Poor
E007,viable
E008,non_viable
```

**质量标签映射**:
- `"good"`, `"yes"`, `"viable"` → 1 (好胚胎)
- `"poor"`, `"no"`, `"non_viable"` → 0 (差胚胎)

**合并策略**:
- 如果胚胎在MainDB.csv中已有标签，使用MainDB的标签
- 如果胚胎仅在filtered_clean_embryos.csv中，使用该标签
- BS评分默认为6.0（差胚胎的默认值）

---

## 🔍 数据验证

训练脚本会自动验证以下内容：

### 1. 标签验证
- ✅ 检查MainDB.csv中的`Quality`字段是否有效
- ✅ 验证所有胚胎都有质量标签
- ⚠️ 跳过没有有效标签的胚胎

### 2. 完整性验证
- ✅ 检查每个胚胎是否有完整的33个时间点
- ✅ 检查每个时间点是否有完整的11个焦平面
- ✅ 验证图像文件是否存在
- ⚠️ 拒绝不完整的胚胎数据

### 3. 数据统计
训练时会打印：
```
Valid embryos: 150 / 200 (75.0%)
Class distribution:
  Class 0 (Poor): 65 samples (43.3%)
  Class 1 (Good): 85 samples (56.7%)
```

### 4. 缓存机制
- 首次验证结果会缓存到 `cache/validation_cache_*.json`
- 下次训练时自动加载缓存，跳过验证
- 如需重新验证，删除缓存文件或设置 `use_cache=False`

---

## 📝 完整示例

### MainDB.csv
```csv
EmbryoID,ICM,TE,Quality,PatientAge,TransferResult
E001,A,A,Good,32,Pregnant
E002,B,B,Good,28,Pregnant
E003,C,C,Poor,35,Failed
E004,B,C,Poor,30,Failed
E005,A,B,Good,29,Pregnant
```

### selected_image_paths.csv（部分）
```csv
EmbryoID,Timepoint,FocalPlane,ImagePath
E001,0,0,/mnt/data/embryos/E001/t00_f00.jpg
E001,0,1,/mnt/data/embryos/E001/t00_f01.jpg
E001,0,2,/mnt/data/embryos/E001/t00_f02.jpg
E001,0,3,/mnt/data/embryos/E001/t00_f03.jpg
E001,0,4,/mnt/data/embryos/E001/t00_f04.jpg
E001,0,5,/mnt/data/embryos/E001/t00_f05.jpg
E001,0,6,/mnt/data/embryos/E001/t00_f06.jpg
E001,0,7,/mnt/data/embryos/E001/t00_f07.jpg
E001,0,8,/mnt/data/embryos/E001/t00_f08.jpg
E001,0,9,/mnt/data/embryos/E001/t00_f09.jpg
E001,0,10,/mnt/data/embryos/E001/t00_f10.jpg
E001,1,0,/mnt/data/embryos/E001/t01_f00.jpg
...
E001,32,10,/mnt/data/embryos/E001/t32_f10.jpg
E002,0,0,/mnt/data/embryos/E002/t00_f00.jpg
...
```

### filtered_clean_embryos.csv（可选）
```csv
EmbryoID,EmbryoQuality,Notes
E006,Good,Additional validated embryo
E007,Poor,Failed to implant
E008,viable,Successful pregnancy
```

---

## 🚀 使用示例

### 最小配置（只需MainDB + selected_image_paths）

```bash
python -m training.train_tmfi \
    --maindb /path/to/MainDB.csv \
    --selected_images /path/to/selected_image_paths.csv \
    --epochs 100 \
    --batch_size 4
```

### 完整配置（包含filtered数据）

```bash
python -m training.train_tmfi \
    --maindb /path/to/MainDB.csv \
    --selected_images /path/to/selected_image_paths.csv \
    --filtered /path/to/filtered_clean_embryos.csv \
    --fusion_type average \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4
```

---

## ⚠️ 常见问题

### 1. 缺少标签
```
Error: No quality label found for embryo E001
```
**解决**: 确保该胚胎在MainDB.csv或filtered_clean_embryos.csv中有Quality/EmbryoQuality字段

### 2. 时间点不完整
```
Warning: Embryo E002 rejected - Incomplete timepoints (found 30/33)
```
**解决**: 确保该胚胎有完整的33个时间点数据

### 3. 焦平面不完整
```
Warning: Embryo E003 rejected - Incomplete focal planes at timepoint 5 (found 9/11)
```
**解决**: 确保每个时间点都有11个焦平面图像

### 4. 图像文件不存在
```
Warning: Image file not found: /path/to/E004_t10_f05.jpg
```
**解决**: 检查selected_image_paths.csv中的路径是否正确

---

## 📊 数据统计

训练前会显示详细统计：

```
========================================================================
LOADING QUALITY LABELS
========================================================================
Loaded MainDB.csv: 200 embryos
Processed MainDB: 180 embryos with valid quality labels
Loaded filtered_clean_embryos.csv: 30 embryos
After merging: 200 total embryos

Quality distribution:
  Good (1): 110 embryos (55.0%)
  Poor (0): 90 embryos (45.0%)

========================================================================
LOADING MULTI-FOCAL DATA
========================================================================
Loaded 72600 image records
🔍 验证胚胎质量标签和时间点完整性...
✓ Validated 150/200 embryos (75.0%)

Validation statistics:
  Total embryos: 200
  Valid embryos: 150
  Rejected embryos: 50

Rejection reasons:
  Incomplete timepoints: 30
  Incomplete focal planes: 15
  Missing quality label: 5

Final dataset: 54450 images from 150 embryos

========================================================================
DATA SPLIT
========================================================================
Training set: 105 embryos (70.0%)
Validation set: 22 embryos (15.0%)
Test set: 23 embryos (15.0%)
```

---

## 📌 总结

### 最小要求
1. ✅ **MainDB.csv** - 包含 `EmbryoID` 和 `Quality` 字段
2. ✅ **selected_image_paths.csv** - 包含完整的33×11图像路径

### 推荐配置
1. ✅ MainDB.csv - 包含 `EmbryoID`, `ICM`, `TE`, `Quality`
2. ✅ selected_image_paths.csv - 完整的多焦距数据
3. ⭐ filtered_clean_embryos.csv - 额外的质量标签（可选）

### 数据要求
- 每个胚胎: **33个时间点 × 11个焦平面 = 363张图像**
- 质量标签: **Good (1) 或 Poor (0)**
- 图像格式: **JPG, PNG等常见格式，224×224或更大**

---

**作者**: Liao Zizhuo
**日期**: 2025-12-22
**版本**: 1.0.0
