"""
Step 5: 可视化
  - 图 A: 4×4 Spearman 相关矩阵热图
  - 图 B: 所有时期两两 QCP 散点图（6 对）
  - 图 C: 四时期 QCP 分布小提琴图
  - 图 D: 平均 QCP 排名分布直方图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr, linregress
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE_DIR   = Path(__file__).parent
RESULT_DIR = BASE_DIR / "results"
FIG_DIR    = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

QCP_CSV       = RESULT_DIR / "fair_embryo_temporal_ranking.csv"
RHO_CSV       = RESULT_DIR / "spearman_rho.csv"
PVAL_CORR_CSV = RESULT_DIR / "spearman_pval_bonferroni.csv"
LIVE_BIRTH_CSV = Path("/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/iEmbryo/outputs/pc1_cycle_vgg16/live_birth_rate_set1/qcp_live_birth_merged.csv")

STAGES       = ["tPNf", "t5", "tM", "tB"]
STAGE_LABELS = ["tPNf", "t5", "tM", "tB"]
STAGE_COLORS = {
    "tPNf": "#4E79A7",
    "t5":   "#F28E2B",
    "tM":   "#59A14F",
    "tB":   "#E15759",
}

OUTCOME_COLORS = {
    0: "#8E8E8E",   # no live birth
    1: "#2CA02C",   # live birth
}


def sig_label(p):
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"


# ── 图 A: Spearman 热图 ───────────────────────────────────────────────────────
def plot_heatmap(rho_df, pval_df, out_path, n_embryos):
    slate_colors = ["#F4F6F9", "#D4E2EF", "#A8C4DC", "#6A9DBF", "#3A6E96", "#1C3F5E"]
    cmap = LinearSegmentedColormap.from_list("slate_blue", slate_colors, N=512)

    def sig_label(p):
        if np.isnan(p): return ""
        if p < 0.001:   return "***"
        if p < 0.01:    return "**"
        if p < 0.05:    return "*"
        return "ns"

    n = len(STAGES)
    fig = plt.figure(figsize=(5.4, 4.8), facecolor="white")
    ax  = fig.add_axes([0.13, 0.13, 0.67, 0.70])
    cax = fig.add_axes([0.82, 0.13, 0.03, 0.70])

    data = rho_df.values.copy().astype(float)
    data[np.eye(n, dtype=bool)] = np.nan
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")

    for i in range(n):
        rect = mpatches.FancyBboxPatch((i-0.5, i-0.5), 1, 1,
            boxstyle="square,pad=0", linewidth=0, facecolor="#EFEFEF", zorder=2)
        ax.add_patch(rect)
        ax.text(i, i, "—", ha="center", va="center",
                fontsize=14, color="#BBBBBB", fontweight="light", zorder=3)

    for k in range(n + 1):
        ax.axhline(k-0.5, color="white", linewidth=2.0, zorder=4)
        ax.axvline(k-0.5, color="white", linewidth=2.0, zorder=4)

    for i in range(n):
        for j in range(n):
            if i == j: continue
            rho = rho_df.iloc[i, j]
            p   = pval_df.iloc[i, j]
            sig = sig_label(p)
            tc  = "white" if rho > 0.56 else "#1C2B3A"
            ax.text(j, i-0.12, f"{rho:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=tc, zorder=5)
            ax.text(j, i+0.30, sig, ha="center", va="center",
                    fontsize=8.5, color=tc,
                    fontstyle="italic" if sig=="ns" else "normal", zorder=5)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(STAGE_LABELS, fontsize=12, fontweight="bold", color="#1C2B3A")
    ax.set_yticklabels(STAGE_LABELS, fontsize=12, fontweight="bold",
                       color="#1C2B3A", rotation=0)
    ax.tick_params(length=0, pad=7)
    for sp in ax.spines.values(): sp.set_visible(False)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman ρ", fontsize=10, labelpad=9, color="#1C2B3A")
    cb.ax.tick_params(labelsize=9, length=3, width=0.7)
    cb.outline.set_linewidth(0)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.set_title(
        "Cross-stage QCP Spearman correlation\n"
        rf"(Fair embryos, $n$ = {n_embryos})",
        fontsize=11, pad=13, loc="left",
        color="#1C2B3A", fontweight="normal", linespacing=1.5)

    fig.text(0.13, 0.035,
        "Significance: * p < 0.05,  ** p < 0.01,  *** p < 0.001  (Bonferroni-corrected)",
        fontsize=8, color="#777777", ha="left", va="bottom", style="italic")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ── 图 B: 所有时期两两散点图（6 对）────────────────────────────────────────────
def plot_pairwise_scatter(df, out_path):
    pairs = list(combinations(STAGES, 2))  # 6 对
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes = axes.flatten()

    # Rank (percentile) per stage to make stages comparable
    rank_cols = {}
    for s in STAGES:
        col = f"qcp_{s}"
        # Higher QCP = better -> rank 1 is best
        ranks = df[col].rank(ascending=False, method="average")
        rank_pct = 1.0 - (ranks - 1) / (ranks.max() - 1)
        rank_cols[s] = rank_pct

    for ax, (s1, s2) in zip(axes, pairs):
        col1, col2 = f"qcp_{s1}", f"qcp_{s2}"
        mask = df[col1].notna() & df[col2].notna()
        sub  = df[mask].copy()
        sub["r1"] = rank_cols[s1][mask]
        sub["r2"] = rank_cols[s2][mask]

        if "live_birth" in sub.columns:
            for outcome in [0, 1]:
                sub_o = sub[sub["live_birth"] == outcome]
                if len(sub_o) == 0:
                    continue
                ax.scatter(sub_o["r1"], sub_o["r2"],
                           alpha=0.35, s=12,
                           color=OUTCOME_COLORS[outcome], edgecolors="none",
                           label="Live Birth" if outcome == 1 else "No Live Birth")
        else:
            ax.scatter(sub["r1"], sub["r2"],
                       alpha=0.35, s=12, color="#555555", edgecolors="none")

        # 回归线
        slope, intercept, *_ = linregress(sub["r1"], sub["r2"])
        x_line = np.linspace(sub["r1"].min(), sub["r1"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color="#E15759", linewidth=1.5, linestyle="--")

        rho, p = spearmanr(sub["r1"], sub["r2"])
        sig = sig_label(p)
        ax.text(0.05, 0.95, f"ρ = {rho:.3f}  {sig}",
                transform=ax.transAxes, fontsize=9.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

        ax.set_xlabel(f"{s1} Rank (percentile)", fontsize=10)
        ax.set_ylabel(f"{s2} Rank (percentile)", fontsize=10)
        ax.set_title(f"{s1} vs {s2} Rank  (n={len(sub)})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

        if "live_birth" in sub.columns:
            ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.8)

    plt.suptitle("Pairwise QCP Rank Scatter Plots Across Developmental Stages",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── 图 C: 四时期 QCP 分布小提琴图 ────────────────────────────────────────────
def plot_violin(df, out_path):
    # 转为 long format
    records = []
    for stage in STAGES:
        col = f"qcp_{stage}"
        for v in df[col].dropna():
            records.append({"Stage": stage, "QCP": v})
    long_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = [STAGE_COLORS[s] for s in STAGES]

    sns.violinplot(data=long_df, x="Stage", y="QCP",
                   palette=palette, inner="box",
                   order=STAGES, ax=ax, linewidth=1.2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Developmental Stage", fontsize=12)
    ax.set_ylabel("QCP (PC1 Score)", fontsize=12)
    ax.set_title("QCP Distribution Across Developmental Stages\n(Fair embryos, n=2448)",
                 fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    # 标注均值
    for i, stage in enumerate(STAGES):
        col = f"qcp_{stage}"
        mean_val = df[col].mean()
        ax.text(i, ax.get_ylim()[1] * 0.92,
                f"μ={mean_val:.2f}", ha="center", fontsize=8.5, color="black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── 图 D: 平均 QCP 排名分布直方图 ────────────────────────────────────────────
def plot_rank_distribution(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左：平均 QCP 直方图
    ax = axes[0]
    ax.hist(df["qcp_mean"], bins=60, color="#4E79A7", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.axvline(df["qcp_mean"].mean(), color="#E15759", linewidth=1.5,
               linestyle="--", label=f"Mean = {df['qcp_mean'].mean():.3f}")
    ax.set_xlabel("Mean QCP (across 4 stages)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Mean QCP\n(Fair embryos, n=2448)", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # 右：各时期 QCP 叠加 KDE
    ax2 = axes[1]
    for stage in STAGES:
        col = f"qcp_{stage}"
        vals = df[col].dropna()
        vals.plot.kde(ax=ax2, label=stage, color=STAGE_COLORS[stage], linewidth=2)

    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("QCP Score", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("QCP Density per Stage", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Fair Embryo QCP Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── 图 E: 按排名分位数的活产率 ────────────────────────────────────────────────
def plot_live_birth_by_rank(df, out_path, n_bins=5):
    if "live_birth" not in df.columns:
        print("  [E] live_birth not available. Skipping.")
        return

    df_lb = df[df["live_birth"].notna()].copy()
    if df_lb.empty:
        print("  [E] No live_birth labels. Skipping.")
        return

    # build rank percentiles
    for s in STAGES:
        col = f"qcp_{s}"
        ranks = df_lb[col].rank(ascending=False, method="average")
        df_lb[f"rank_{s}"] = 1.0 - (ranks - 1) / (ranks.max() - 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, s in zip(axes, STAGES):
        rcol = f"rank_{s}"
        df_lb["bin"] = pd.qcut(df_lb[rcol], q=n_bins, labels=False, duplicates="drop")
        grp = df_lb.groupby("bin")["live_birth"].agg(["mean", "count"]).reset_index()
        x = grp["bin"].astype(int)
        ax.plot(x, grp["mean"], marker="o", color=STAGE_COLORS[s], linewidth=1.8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Rank Quantile (low → high)", fontsize=10)
        ax.set_ylabel("Live Birth Rate", fontsize=10)
        ax.set_title(f"{s} Rank vs Live Birth", fontsize=10)
        for i, row in grp.iterrows():
            ax.text(row["bin"], row["mean"] + 0.03, f"n={int(row['count'])}",
                    ha="center", fontsize=8, color="gray")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Live Birth Rate by QCP Rank Quantile", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── 指标: 单期/多期排名 AUC ────────────────────────────────────────────────
def report_auc(df):
    if "live_birth" not in df.columns:
        print("  [AUC] live_birth not available. Skipping.")
        return

    df_lb = df[df["live_birth"].notna()].copy()
    if df_lb.empty:
        print("  [AUC] No live_birth labels. Skipping.")
        return

    # rank percentiles
    rank_cols = []
    for s in STAGES:
        col = f"qcp_{s}"
        ranks = df_lb[col].rank(ascending=False, method="average")
        rcol = f"rank_{s}"
        df_lb[rcol] = 1.0 - (ranks - 1) / (ranks.max() - 1)
        rank_cols.append(rcol)

    y = df_lb["live_birth"].astype(int).values

    # single-stage AUC
    print("\n  [AUC] Single-stage rank AUC:")
    for s, rcol in zip(STAGES, rank_cols):
        auc = roc_auc_score(y, df_lb[rcol].values)
        print(f"    {s}: AUC={auc:.3f}")

    # multi-stage AUC (CV logistic regression)
    X = df_lb[rank_cols].values
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, prob)
    print(f"  [AUC] 4-stage rank (5-fold CV LR): AUC={auc:.3f}")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Step 5: Visualization")
    print("=" * 60)

    df      = pd.read_csv(QCP_CSV)
    rho_df  = pd.read_csv(RHO_CSV, index_col=0).loc[STAGES, STAGES]
    pval_df = pd.read_csv(PVAL_CORR_CSV, index_col=0).loc[STAGES, STAGES]

    print(f"  Loaded {len(df)} embryos")

    # Join live birth outcome if available
    if LIVE_BIRTH_CSV.exists():
        lb = pd.read_csv(LIVE_BIRTH_CSV)
        if "embryo_id" in lb.columns:
            lb["embryo_id_norm"] = lb["embryo_id"].astype(str).str.strip().str.split("_RUN").str[0]
            # aggregate: any live birth for this embryo_id
            if "live_birth" in lb.columns:
                lb_agg = lb.groupby("embryo_id_norm", as_index=False)["live_birth"].max()
                df = df.merge(lb_agg, left_on="embryo_id", right_on="embryo_id_norm", how="left")
                df.drop(columns=["embryo_id_norm"], inplace=True)
            elif "pregnancy" in lb.columns:
                lb_agg = lb.groupby("embryo_id_norm", as_index=False)["pregnancy"].max()
                lb_agg = lb_agg.rename(columns={"pregnancy": "live_birth"})
                df = df.merge(lb_agg, left_on="embryo_id", right_on="embryo_id_norm", how="left")
                df.drop(columns=["embryo_id_norm"], inplace=True)

    print("\n  [A] Spearman heatmap...")
    plot_heatmap(rho_df, pval_df, FIG_DIR / "figA_spearman_heatmap.png", len(df))

    print("\n  [B] Pairwise scatter plots...")
    plot_pairwise_scatter(df, FIG_DIR / "figB_pairwise_scatter.png")

    print("\n  [C] Violin plots...")
    plot_violin(df, FIG_DIR / "figC_violin.png")

    print("\n  [D] Rank distribution...")
    plot_rank_distribution(df, FIG_DIR / "figD_rank_distribution.png")

    print("\n  [E] Live birth by rank quantile...")
    plot_live_birth_by_rank(df, FIG_DIR / "figE_live_birth_by_rank.png")

    report_auc(df)

    print(f"\n  All figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
