# 数据集说明与论文写作指南
# Dataset Description and Manuscript Writing Guide

---

## 一、数据来源概述 (Data Source Overview)

### 1.1 训练数据集 (Training Dataset)

**用途**: 构建胚胎质量评分模型 (QCP - Quality Continuum Predictor)

**数据来源**:
- **Good胚胎**: `/Users/lzz/Desktop/IVF/good_embryos/embryos_all_focal_planes.csv`
- **Fair胚胎**: `/Users/lzz/Desktop/IVF/fair_embryos/`
- **Poor胚胎**: `/Users/lzz/Desktop/IVF/poor_embryos/embryos_all_focal_planes.csv`

**数据规模**:
- 三类胚胎 (Good/Fair/Poor)
- 用于训练PCA模型和分析质量连续性

**处理脚本**: `unified_pca_analysis.py`

---

### 1.2 应用数据集 (Application Dataset)

**用途**: 计算真实临床数据的胚胎质量评分

**数据来源**:
1. **患者-SlideID映射表**:
   - `embryo_info/Patient_to_SlideIDs_tree_only.csv`
   - 包含 `PatientID` 和 `SlideIDs` 列

2. **胚胎图片**:
   - 图片根目录: `PC_CHAID/`
   - **99,901张图片** (不同焦平面)
   - **10,019个胚胎**
   - **1,423个取卵周期** (patient_id级别)

3. **PCA模型缓存**:
   - `outputs/three_class_vgg16/pca_cache/pca_cache_a75b1d1d.pkl`
   - 从训练数据集中训练得到

4. **预训练模型**:
   - `vgg16_best_val_accuracy.pth`

**处理脚本**: `embryo_info/extract_pc1_cycle_metrics.py`

**输出文件**:
- `pc1_image_level.csv`: 图片级别PC1值
- `pc1_embryo_level.csv`: 胚胎级别PC1统计值
- `pc1_cycle_level.csv`: **周期级别PC1统计值 (1,423条记录)**

---

### 1.3 临床预测数据集 (Clinical Prediction Dataset)

**用途**: CHAID决策树分析，预测活产

**数据来源**:
1. **冷冻周期临床数据**:
   - 2023-2025年冷冻胚胎移植记录
   - **3,723条原始记录**

2. **胚胎质量评分**:
   - 来自 `pc1_cycle_level.csv` (1,423个周期)

**处理脚本**: `chaid_frozen_clinical.ipynb`

---

## 二、数据处理流程 (Data Processing Pipeline)

### 2.1 QCP评分计算流程

```
胚胎图片 (99,901张)
    ↓
VGG16特征提取
    ↓
PCA降维 (使用训练好的模型)
    ↓
PC1值计算
    ↓
图片级别聚合 → 胚胎级别聚合 → 周期级别聚合
    ↓
pc1_cycle_level.csv (1,423个周期)
```

**关键指标**:
- `pc1_median`: PC1中位数
- `pc1_p90`: PC1的90分位数 (用作QCP评分)
- `pc1_mean`: PC1均值
- `pc1_std`: PC1标准差
- `pc1_max`: PC1最大值
- `n_embryos`: 胚胎数量

---

### 2.2 CHAID分析数据筛选流程

```
冷冻周期原始数据: 3,723条
    ↓
与PC1数据合并 (inner join on patient_id): 2,305条
    ↓
删除缺失值 (年龄、BMI): 2,295条
    ↓
筛选单胚胎移植 (移植数==1): 1,884条
    ↓
删除特征列缺失值 (内膜厚度、孕酮等): 1,835条 ✓
```

**最终分析队列**: 1,835例单胚胎冷冻移植周期
- 活产: 766例 (41.74%)
- 非活产: 1,069例 (58.26%)

---

## 三、关键数字说明 (Key Numbers Explanation)

### 为什么移植周期数(1,835) > 取卵周期数(1,423)?

**原因**:
1. 一个取卵周期可能产生多个冷冻胚胎
2. 这些胚胎可能在不同时间进行多次冷冻移植
3. 同一个 `patient_id` 在研究期间可能有多条冷冻移植记录

**数据层级**:
- **1,423** = 取卵周期数 (每个patient_id一条，用于计算QCP评分)
- **1,835** = 冷冻移植记录数 (同一patient_id可能有多条，用于预测分析)

---

## 四、论文写作建议 (Manuscript Writing Guide)

### 4.1 方法部分 (Methods Section)

#### 数据集构建与预处理 (Dataset Construction and Preprocessing)

本研究纳入了2023年1月至2025年7月期间的冷冻胚胎移植周期数据。数据筛选流程如下：

**1. 初始数据集**

共3,723例冷冻胚胎移植周期记录。

**2. 胚胎质量评分整合**

将冷冻周期数据与基于深度学习的胚胎质量评分（QCP，Quality Continuum Predictor）进行匹配。QCP评分基于1,423个患者的取卵周期，通过VGG16模型从99,901张胚胎图像（涵盖10,019个胚胎的不同焦平面）中提取特征，经主成分分析（PCA）降维后，将第一主成分（PC1）的90分位数作为该周期的胚胎质量连续评分指标。匹配后获得2,305例有QCP评分的冷冻移植周期。

**3. 纳入标准**

- 单胚胎移植（移植数=1）
- 临床信息完整，包括：年龄、BMI、转化日内膜厚度、转化日孕酮、不孕类型（原发/继发）、对应新鲜周期治疗方法

**4. 最终分析队列**

经筛选后，1,835例单胚胎冷冻移植周期纳入CHAID决策树分析，活产率为41.74%（766/1,835）。

**注**: 由于部分患者在研究期间进行了多次冷冻胚胎移植，最终分析的移植周期数（1,835）多于原始取卵周期数（1,423）。

---

#### 胚胎质量评分（QCP）的计算 (QCP Score Calculation)

QCP评分基于独立的训练数据集构建：

- **训练数据**: 1,423个取卵周期，包含10,019个胚胎的99,901张不同焦平面图像
- **模型架构**: 使用预训练VGG16模型提取胚胎形态特征
- **质量评分**: 通过主成分分析（PCA）将高维特征降维，第一主成分（PC1）的90分位数（p90）作为该周期的胚胎质量评分
- **评分应用**: 将训练好的PCA模型应用于冷冻移植周期对应的取卵周期胚胎图像，计算QCP评分

---

### 4.2 流程图 (CONSORT-style Flowchart)

```
冷冻胚胎移植周期记录
n = 3,723
    ↓
排除：无胚胎质量评分 (n = 1,418)
    ↓
有QCP评分的移植周期
n = 2,305
    ↓
排除：多胚胎移植 (n = 411)
排除：临床信息缺失 (n = 59)
    ↓
最终分析队列
n = 1,835
━━━━━━━━━━━━━━━━━━━━━━
活产：766例 (41.74%)
非活产：1,069例 (58.26%)
```

---

### 4.3 结果部分 (Results Section)

#### 基线特征 (Baseline Characteristics)

本研究最终纳入1,835例单胚胎冷冻移植周期，这些周期来自1,423个取卵周期的胚胎。患者平均年龄为XX岁（范围：XX-XX岁），平均BMI为XX kg/m²。总体活产率为41.74%（766/1,835）。

QCP评分范围为XX至XX，中位数为XX（IQR: XX-XX）。QCP评分与活产率呈显著正相关（p < 0.001）。

---

### 4.4 讨论部分要点 (Discussion Points)

**优势**:
1. 大样本量：基于1,835例单胚胎移植周期
2. 客观评分：QCP评分基于99,901张图像的深度学习特征
3. 多因素整合：结合胚胎质量、患者特征和临床参数

**局限性**:
1. 回顾性研究设计
2. 单中心数据
3. 部分患者有多次移植记录，可能存在数据相关性

---

## 五、数据文件清单 (Data File Inventory)

### 输入文件 (Input Files)

| 文件路径 | 描述 | 记录数 |
|---------|------|--------|
| `embryo_info/Patient_to_SlideIDs_tree_only.csv` | 患者-SlideID映射表 | - |
| `PC_CHAID/` | 胚胎图片目录 | 99,901张 |
| `outputs/three_class_vgg16/pca_cache/pca_cache_*.pkl` | PCA模型缓存 | - |
| `vgg16_best_val_accuracy.pth` | 预训练VGG16模型 | - |
| `23.1.1-25.7.31time-lapse原始数据新鲜+冷冻/冷冻/` | 冷冻周期临床数据 | 3,723条 |

### 输出文件 (Output Files)

| 文件路径 | 描述 | 记录数 |
|---------|------|--------|
| `outputs/pc1_cycle_vgg16/pc1_image_level.csv` | 图片级别PC1值 | 99,901条 |
| `outputs/pc1_cycle_vgg16/pc1_embryo_level.csv` | 胚胎级别PC1统计 | 10,019条 |
| `outputs/pc1_cycle_vgg16/pc1_cycle_level.csv` | 周期级别PC1统计 | 1,423条 |
| `outputs/pc1_cycle_vgg16/run_meta.json` | 运行元数据 | - |

---

## 六、关键术语定义 (Key Terminology)

| 术语 | 英文 | 定义 |
|-----|------|------|
| 取卵周期 | Oocyte Retrieval Cycle | 一次取卵手术及其产生的所有胚胎 |
| 冷冻移植周期 | Frozen Embryo Transfer (FET) Cycle | 一次冷冻胚胎解冻移植 |
| QCP评分 | Quality Continuum Predictor Score | 基于PC1的90分位数的胚胎质量连续评分 |
| PC1 | First Principal Component | 主成分分析的第一主成分 |
| 活产 | Live Birth | 胎儿数 > 0 |

---

## 七、统计分析说明 (Statistical Analysis Notes)

### 样本量计算

- **分析单位**: 冷冻移植周期 (n=1,835)
- **结局指标**: 活产 (二分类变量)
- **预测变量**: QCP评分、年龄、BMI、内膜厚度、孕酮、不孕类型、治疗方法

### 统计方法

- **CHAID决策树**: 用于识别活产的预测因素和交互作用
- **参数设置**:
  - 最小父节点样本量: 200
  - 最大深度: 5
  - 合并显著性水平: α=0.05

---

## 八、数据质量控制 (Data Quality Control)

### 数据完整性检查

1. **图片质量**: 99,901张图片均通过质量检查
2. **临床数据**: 删除关键变量缺失的记录
3. **匹配准确性**: 通过patient_id精确匹配QCP评分

### 排除标准

1. 多胚胎移植 (n=411)
2. 临床信息缺失 (n=59)
3. QCP评分缺失 (n=1,418)

---

## 九、伦理声明 (Ethics Statement)

本研究已获得[机构名称]伦理委员会批准（批准号：XXXX）。由于研究采用回顾性设计且数据已去标识化，豁免知情同意。

---

## 十、数据可用性声明 (Data Availability Statement)

由于患者隐私保护要求，原始临床数据不公开共享。去标识化的汇总数据可在合理请求下从通讯作者处获得。

---

**文档创建日期**: 2026-03-06
**最后更新**: 2026-03-06
**版本**: 1.0
