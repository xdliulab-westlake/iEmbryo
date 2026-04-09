#!/usr/bin/env python3
"""
QCP与妊娠率关系分析（仅移植数=1的周期）
Analysis of QCP (Quality Continuum Predictor) and Pregnancy Rate Relationship (Single Embryo Transfer Cycles Only)
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

# 输出目录
OUTPUT_DIR = Path("/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/iEmbryo/outputs/pc1_cycle_vgg16/pregnancy_rate_set1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("QCP与妊娠率关系分析（仅移植数=1的周期）")
print("="*80)

# ============================================================================
# 1. 数据加载
# ============================================================================
print("\n📂 加载数据...")

# 加载胚胎级别PC1数据
PARENT_DIR = Path("/Users/lzz/Downloads/fair_embryos_pc1_results/")
df_embryo = pd.read_csv(PARENT_DIR / "pc1_embryo_level.csv")
print(f"  胚胎级别数据: {len(df_embryo)} 条记录")

# 加载临床数据
df_clinical = pd.read_csv("/Users/lzz/Downloads/df_fair_no_nan.csv")
print(f"  临床数据: {len(df_clinical)} 条记录")

# 加载 MainDB 用于囊胚等级匹配
df_maindb = pd.read_csv("/Users/lzz/Documents/GitHub/IVF_AI/MainDB.csv")
df_maindb['embryo_id'] = df_maindb['RowName'].str.replace(r'_(\d+)$', r'_WELL\1', regex=True)
df_maindb_unique = df_maindb.drop_duplicates(subset=['embryo_id'])

# 囊胚等级匹配：只保留临床等级与 MainDB ICM/TE 一致的 SlideID
df_clinical['SlideID'] = df_clinical['SlideID'].str.strip()
df_pred_ids = pd.read_csv("/Users/lzz/Downloads/embryo_results_2025-12-10.csv")
df_pred_ids['SlideID'] = df_pred_ids['embryo_id'].str.extract(r'^(D\d{4}\.\d{2}\.\d{2}_S\d+_I\d+_P)')
merged_grade = df_clinical.merge(df_pred_ids, on='SlideID', how='inner')
merged_grade = merged_grade.merge(df_maindb_unique[['embryo_id', 'ICM', 'TE']], on='embryo_id', how='left')
merged_grade['囊胚等级_clean'] = merged_grade['囊胚等级'].astype(str).str.upper().str.strip()
grade_extracted = merged_grade['囊胚等级_clean'].str.extract(r'([ABC])\s*([ABC])\s*$')
merged_grade['Grade_ICM'] = grade_extracted[0]
merged_grade['Grade_TE']  = grade_extracted[1]
merged_grade['ICM_clean'] = merged_grade['ICM'].astype(str).str.upper().str.strip()
merged_grade['TE_clean']  = merged_grade['TE'].astype(str).str.upper().str.strip()
valid_slides = set(merged_grade[
    (merged_grade['Grade_ICM'] == merged_grade['ICM_clean']) &
    (merged_grade['Grade_TE']  == merged_grade['TE_clean'])
]['SlideID'])
print(f"  囊胚等级匹配后有效SlideID数: {len(valid_slides)}")

# 只保留匹配的 SlideID 且移植数=1
df_clinical = df_clinical[
    (df_clinical['SlideID'].isin(valid_slides)) &
    (df_clinical['移植数'] == 1)
].copy()
print(f"  移植数=1的周期: {len(df_clinical)} 条记录")

# ============================================================================
# 2. 数据预处理
# ============================================================================
print("\n🔧 数据预处理...")

# 2.1 按 ARTID 顺序为每个移植周期分配对应排名的胚胎
# 同一 SlideID 可能对应多个 ARTID（最多3次），依次分配 QCP 第1、2、3高的胚胎
print("  按ARTID顺序分配胚胎QCP...")
df_embryo['QCP'] = df_embryo['embryo_pc1_max']
df_embryo['slide_id'] = df_embryo['slide_id'].str.strip()

# 为每个 SlideID 预先排好胚胎 QCP 排名
slide_embryo_ranked = {}
for slide_id, grp in df_embryo.groupby('slide_id'):
    slide_embryo_ranked[slide_id] = grp.sort_values('QCP', ascending=False).reset_index(drop=True)

# 按 ARTID 升序排列，依次取第1、2、3高QCP胚胎
df_clinical_sorted = df_clinical.sort_values(['SlideID', 'ARTID']).copy()
df_clinical_sorted['_rank'] = df_clinical_sorted.groupby('SlideID').cumcount()  # 0-based rank

rows = []
for _, row in df_clinical_sorted.iterrows():
    sid = row['SlideID']
    rank = row['_rank']
    if sid not in slide_embryo_ranked:
        continue
    ranked = slide_embryo_ranked[sid]
    if rank >= len(ranked):
        rank = len(ranked) - 1  # 胚胎数不足时取最后一个
    embryo_row = ranked.iloc[rank]
    rows.append({**row.to_dict(),
                 'QCP': embryo_row['QCP'],
                 'embryo_pc1_median': embryo_row['embryo_pc1_median'],
                 'embryo_pc1_std': embryo_row['embryo_pc1_std'],
                 'n_focal_planes': embryo_row['n_focal_planes'],
                 'embryo_id': embryo_row['embryo_id'],
                 'qcp_rank': rank + 1})

df_merged = pd.DataFrame(rows).drop(columns=['_rank'])
print(f"  分配后记录数: {len(df_merged)}")

# 2.3 计算妊娠率（基于妊娠结局：只有临床妊娠算成功）
df_merged['pregnancy'] = (df_merged['妊娠结局'] == '临床妊娠').astype(int)

print(f"\n📊 数据统计:")
print(f"  总样本数: {len(df_merged)}")
print(f"  妊娠样本（临床妊娠）: {df_merged['pregnancy'].sum()} ({df_merged['pregnancy'].mean()*100:.2f}%)")
print(f"  未妊娠样本: {(df_merged['pregnancy']==0).sum()} ({(df_merged['pregnancy']==0).mean()*100:.2f}%)")
print(f"  妊娠结局分布:")
for outcome, count in df_merged['妊娠结局'].value_counts().items():
    print(f"    {outcome}: {count} ({count/len(df_merged)*100:.2f}%)")
print(f"  QCP范围: [{df_merged['QCP'].min():.4f}, {df_merged['QCP'].max():.4f}]")
print(f"  QCP均值: {df_merged['QCP'].mean():.4f} ± {df_merged['QCP'].std():.4f}")

# 保存合并后的数据
df_merged.to_csv(OUTPUT_DIR / "qcp_pregnancy_merged.csv", index=False)
print(f"\n✅ 合并数据已保存: qcp_pregnancy_merged.csv")

# ============================================================================
# 3. 统计分析
# ============================================================================
print("\n" + "="*80)
print("统计分析")
print("="*80)

# 3.1 按妊娠结果分组统计
print("\n📊 按妊娠结果分组的QCP统计:")
grouped = df_merged.groupby('pregnancy')['QCP'].describe()
print(grouped)

# 3.2 T检验
pregnant_qcp = df_merged[df_merged['pregnancy'] == 1]['QCP']
not_pregnant_qcp = df_merged[df_merged['pregnancy'] == 0]['QCP']
t_stat, p_value = stats.ttest_ind(pregnant_qcp, not_pregnant_qcp)
print(f"\n📈 独立样本T检验:")
print(f"  妊娠组QCP均值: {pregnant_qcp.mean():.4f} ± {pregnant_qcp.std():.4f}")
print(f"  未妊娠组QCP均值: {not_pregnant_qcp.mean():.4f} ± {not_pregnant_qcp.std():.4f}")
print(f"  t统计量: {t_stat:.4f}")
print(f"  p值: {p_value:.4e}")
print(f"  显著性: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# 3.3 Mann-Whitney U检验（非参数检验）
u_stat, u_p_value = stats.mannwhitneyu(pregnant_qcp, not_pregnant_qcp, alternative='two-sided')
print(f"\n📈 Mann-Whitney U检验:")
print(f"  U统计量: {u_stat:.4f}")
print(f"  p值: {u_p_value:.4e}")

# 3.4 相关性分析
corr_pearson, p_pearson = stats.pearsonr(df_merged['QCP'], df_merged['pregnancy'])
corr_spearman, p_spearman = stats.spearmanr(df_merged['QCP'], df_merged['pregnancy'])
print(f"\n📈 相关性分析:")
print(f"  Pearson相关系数: {corr_pearson:.4f} (p={p_pearson:.4e})")
print(f"  Spearman相关系数: {corr_spearman:.4f} (p={p_spearman:.4e})")

# ============================================================================
# 4. 可视化分析
# ============================================================================
print("\n" + "="*80)
print("可视化分析")
print("="*80)

# 4.1 QCP分布对比图（密度图 + 直方图）
print("\n📊 生成QCP分布对比图...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 密度图
ax1 = axes[0, 0]
sns.kdeplot(data=df_merged, x='QCP', hue='pregnancy', fill=True, alpha=0.5, ax=ax1,
            palette={0: "#E74C3C", 1: "#27AE60"})
ax1.set_xlabel('QCP Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
ax1.set_title('QCP Distribution by Pregnancy Outcome (Density Plot)', fontsize=14, fontweight='bold')
ax1.legend(labels=['Not Pregnant', 'Pregnant'], fontsize=11)
ax1.grid(True, alpha=0.3)

# 子图2: 直方图
ax2 = axes[0, 1]
bins = np.linspace(df_merged['QCP'].min(), df_merged['QCP'].max(), 30)
ax2.hist(not_pregnant_qcp, bins=bins, alpha=0.6, label='Not Pregnant', color='#E74C3C', edgecolor='black')
ax2.hist(pregnant_qcp, bins=bins, alpha=0.6, label='Pregnant', color='#27AE60', edgecolor='black')
ax2.set_xlabel('QCP Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('QCP Distribution by Pregnancy Outcome (Histogram)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# 子图3: 箱线图
ax3 = axes[1, 0]
box_data = [not_pregnant_qcp, pregnant_qcp]
bp = ax3.boxplot(box_data, labels=['Not Pregnant', 'Pregnant'],
                 patch_artist=True, widths=0.6,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
bp['boxes'][0].set_facecolor('#E74C3C')
bp['boxes'][1].set_facecolor('#27AE60')
ax3.set_ylabel('QCP Score', fontsize=12, fontweight='bold')
ax3.set_title('QCP Distribution by Pregnancy Outcome (Box Plot)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 添加统计显著性标记
y_max = df_merged['QCP'].max()
y_min = df_merged['QCP'].min()
y_range = y_max - y_min
ax3.plot([1, 2], [y_max + 0.1*y_range, y_max + 0.1*y_range], 'k-', linewidth=1.5)
sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
ax3.text(1.5, y_max + 0.12*y_range, sig_text, ha='center', fontsize=14, fontweight='bold')

# 子图4: 小提琴图
ax4 = axes[1, 1]
sns.violinplot(data=df_merged, x='pregnancy', y='QCP', ax=ax4, inner='box')
ax4.set_xticklabels(['Not Pregnant', 'Pregnant'])
ax4.set_xlabel('Pregnancy Outcome', fontsize=12, fontweight='bold')
ax4.set_ylabel('QCP Score', fontsize=12, fontweight='bold')
ax4.set_title('QCP Distribution by Pregnancy Outcome (Violin Plot)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'qcp_pregnancy_distribution.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 保存: qcp_pregnancy_distribution.pdf")

# 4.2 QCP分位数与妊娠率关系
print("\n📊 生成QCP分位数与妊娠率关系图...")
df_merged['QCP_decile'] = pd.qcut(df_merged['QCP'], q=10, labels=False, duplicates='drop') + 1
pregnancy_by_decile = df_merged.groupby('QCP_decile').agg({
    'pregnancy': ['mean', 'sum', 'count'],
    'QCP': ['mean', 'min', 'max']
}).reset_index()
pregnancy_by_decile.columns = ['decile', 'pregnancy_rate', 'n_pregnant', 'n_total',
                                'qcp_mean', 'qcp_min', 'qcp_max']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 子图1: 柱状图
ax1 = axes[0]
bars = ax1.bar(pregnancy_by_decile['decile'], pregnancy_by_decile['pregnancy_rate']*100,
               color=plt.cm.RdYlGn(pregnancy_by_decile['pregnancy_rate']),
               edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('QCP Decile', fontsize=12, fontweight='bold')
ax1.set_ylabel('Pregnancy Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Pregnancy Rate by QCP Decile', fontsize=14, fontweight='bold')
ax1.set_xticks(pregnancy_by_decile['decile'])
ax1.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (bar, rate, n) in enumerate(zip(bars, pregnancy_by_decile['pregnancy_rate'],
                                        pregnancy_by_decile['n_total'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{rate*100:.1f}%\n(n={int(n)})',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 子图2: 折线图
ax2 = axes[1]
ax2.plot(pregnancy_by_decile['decile'], pregnancy_by_decile['pregnancy_rate']*100,
         marker='o', linewidth=2.5, markersize=8, color='#2E86AB', label='Pregnancy Rate')
ax2.fill_between(pregnancy_by_decile['decile'], 0, pregnancy_by_decile['pregnancy_rate']*100,
                 alpha=0.3, color='#2E86AB')
ax2.set_xlabel('QCP Decile', fontsize=12, fontweight='bold')
ax2.set_ylabel('Pregnancy Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Pregnancy Rate Trend by QCP Decile', fontsize=14, fontweight='bold')
ax2.set_xticks(pregnancy_by_decile['decile'])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'qcp_decile_pregnancy_rate.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 保存: qcp_decile_pregnancy_rate.pdf")

# 4.3 ROC曲线
print("\n📊 生成ROC曲线...")
fpr, tpr, thresholds = roc_curve(df_merged['pregnancy'], df_merged['QCP'])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color='#2E86AB', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve: QCP Predicting Pregnancy', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# 添加最佳阈值点
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
        label=f'Optimal Threshold = {optimal_threshold:.3f}')
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'qcp_pregnancy_roc.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 保存: qcp_pregnancy_roc.pdf")
print(f"  AUC: {roc_auc:.4f}")
print(f"  最佳阈值: {optimal_threshold:.4f}")

# 4.4 散点图（QCP vs 妊娠率，带抖动）
print("\n📊 生成散点图...")
fig, ax = plt.subplots(figsize=(12, 8))

# 添加抖动以避免重叠
jitter_strength = 0.02
df_merged['pregnancy_jitter'] = df_merged['pregnancy'] + np.random.normal(0, jitter_strength, len(df_merged))

scatter = ax.scatter(df_merged['QCP'], df_merged['pregnancy_jitter'],
                    c=df_merged['pregnancy'], cmap='RdYlGn',
                    alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel('QCP Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Pregnancy Outcome (0=No, 1=Yes)', fontsize=12, fontweight='bold')
ax.set_title('QCP Score vs Pregnancy Outcome (with Jitter)', fontsize=14, fontweight='bold')
ax.set_ylim([-0.2, 1.2])
ax.grid(True, alpha=0.3)

# 添加逻辑回归拟合曲线
from sklearn.linear_model import LogisticRegression
X = df_merged['QCP'].values.reshape(-1, 1)
y = df_merged['pregnancy'].values
log_reg = LogisticRegression()
log_reg.fit(X, y)
X_test = np.linspace(df_merged['QCP'].min(), df_merged['QCP'].max(), 300).reshape(-1, 1)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
ax.plot(X_test, y_pred_proba, 'r-', linewidth=3, label='Logistic Regression Fit')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'qcp_pregnancy_scatter.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 保存: qcp_pregnancy_scatter.pdf")

# ============================================================================
# 5. 生成统计报告
# ============================================================================
print("\n" + "="*80)
print("生成统计报告")
print("="*80)

report = f"""
# QCP与妊娠率关系分析报告
# Analysis Report: QCP and Pregnancy Rate Relationship

生成日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 数据概况

- 总样本数: {len(df_merged)}
- 妊娠样本: {df_merged['pregnancy'].sum()} ({df_merged['pregnancy'].mean()*100:.2f}%)
- 未妊娠样本: {(df_merged['pregnancy']==0).sum()} ({(df_merged['pregnancy']==0).mean()*100:.2f}%)
- QCP范围: [{df_merged['QCP'].min():.4f}, {df_merged['QCP'].max():.4f}]
- QCP均值: {df_merged['QCP'].mean():.4f} ± {df_merged['QCP'].std():.4f}

## 2. 统计检验结果

### 2.1 按妊娠结果分组的QCP统计

妊娠组:
- 样本数: {len(pregnant_qcp)}
- QCP均值: {pregnant_qcp.mean():.4f}
- QCP标准差: {pregnant_qcp.std():.4f}
- QCP中位数: {pregnant_qcp.median():.4f}
- QCP范围: [{pregnant_qcp.min():.4f}, {pregnant_qcp.max():.4f}]

未妊娠组:
- 样本数: {len(not_pregnant_qcp)}
- QCP均值: {not_pregnant_qcp.mean():.4f}
- QCP标准差: {not_pregnant_qcp.std():.4f}
- QCP中位数: {not_pregnant_qcp.median():.4f}
- QCP范围: [{not_pregnant_qcp.min():.4f}, {not_pregnant_qcp.max():.4f}]

### 2.2 独立样本T检验

- t统计量: {t_stat:.4f}
- p值: {p_value:.4e}
- 显著性: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}
- 结论: {'妊娠组QCP显著高于未妊娠组' if t_stat > 0 and p_value < 0.05 else '两组QCP无显著差异'}

### 2.3 Mann-Whitney U检验（非参数检验）

- U统计量: {u_stat:.4f}
- p值: {u_p_value:.4e}
- 显著性: {'***' if u_p_value < 0.001 else '**' if u_p_value < 0.01 else '*' if u_p_value < 0.05 else 'ns'}

### 2.4 相关性分析

- Pearson相关系数: {corr_pearson:.4f} (p={p_pearson:.4e})
- Spearman相关系数: {corr_spearman:.4f} (p={p_spearman:.4e})
- 结论: {'QCP与妊娠率呈显著正相关' if corr_pearson > 0 and p_pearson < 0.05 else 'QCP与妊娠率无显著相关'}

## 3. ROC曲线分析

- AUC: {roc_auc:.4f}
- 最佳阈值: {optimal_threshold:.4f}
- 预测能力: {'优秀' if roc_auc >= 0.9 else '良好' if roc_auc >= 0.8 else '中等' if roc_auc >= 0.7 else '一般'}

## 4. QCP分位数与妊娠率

{pregnancy_by_decile.to_string(index=False)}

## 5. 结论

1. {'妊娠组的QCP评分显著高于未妊娠组 (p<0.05)' if p_value < 0.05 else '两组QCP评分无显著差异'}
2. {'QCP评分与妊娠率呈显著正相关' if p_pearson < 0.05 else 'QCP评分与妊娠率无显著相关'}
3. QCP评分的AUC为{roc_auc:.3f}，{'表明其具有良好的妊娠预测能力' if roc_auc >= 0.7 else '预测能力有限'}
4. 随着QCP分位数的增加，妊娠率呈{'上升' if pregnancy_by_decile['pregnancy_rate'].iloc[-1] > pregnancy_by_decile['pregnancy_rate'].iloc[0] else '下降'}趋势

## 6. 生成的图表

1. qcp_pregnancy_distribution.pdf - QCP分布对比图（密度图、直方图、箱线图、小提琴图）
2. qcp_decile_pregnancy_rate.pdf - QCP分位数与妊娠率关系图
3. qcp_pregnancy_roc.pdf - ROC曲线
4. qcp_pregnancy_scatter.pdf - 散点图（带逻辑回归拟合）

## 7. 数据文件

- qcp_pregnancy_merged.csv - 合并后的完整数据
- qcp_pregnancy_analysis_report.txt - 本报告
"""

# 保存报告
with open(OUTPUT_DIR / 'qcp_pregnancy_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n✅ 统计报告已保存: qcp_pregnancy_analysis_report.txt")

# ============================================================================
# 6. 保存分位数统计表
# ============================================================================
pregnancy_by_decile.to_csv(OUTPUT_DIR / 'qcp_decile_statistics.csv', index=False)
print("✅ 分位数统计表已保存: qcp_decile_statistics.csv")

# ============================================================================
# 完成
# ============================================================================
print("\n" + "="*80)
print("✅ 分析完成！")
print("="*80)
print(f"\n📁 所有结果已保存到: {OUTPUT_DIR}")
print("\n生成的文件:")
print("  1. qcp_pregnancy_merged.csv - 合并数据")
print("  2. qcp_pregnancy_distribution.pdf - 分布对比图")
print("  3. qcp_decile_pregnancy_rate.pdf - 分位数妊娠率图")
print("  4. qcp_pregnancy_roc.pdf - ROC曲线")
print("  5. qcp_pregnancy_scatter.pdf - 散点图")
print("  6. qcp_pregnancy_analysis_report.txt - 统计报告")
print("  7. qcp_decile_statistics.csv - 分位数统计表")
print("\n" + "="*80)
