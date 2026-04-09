"""
Step 3: 四时期 QCP 跨时期 Spearman 相关矩阵
输入: results/fair_embryo_temporal_ranking.csv
输出: results/spearman_rho.csv
      results/spearman_pval.csv
      results/spearman_pval_bonferroni.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy import stats

BASE_DIR   = Path(__file__).parent
RESULT_DIR = BASE_DIR / "results"
QCP_CSV    = RESULT_DIR / "fair_embryo_temporal_ranking.csv"

STAGES = ["tPNf", "t5", "tM", "tB"]


def compute_spearman_matrix(df: pd.DataFrame):
    n = len(STAGES)
    rho_mat  = np.eye(n)
    pval_mat = np.zeros((n, n))
    n_pairs  = n * (n - 1) // 2  # 6

    for i, j in combinations(range(n), 2):
        s1, s2 = STAGES[i], STAGES[j]
        col1, col2 = f"qcp_{s1}", f"qcp_{s2}"
        mask = df[col1].notna() & df[col2].notna()
        sub  = df[mask]
        if len(sub) < 5:
            rho, p = np.nan, np.nan
        else:
            rho, p = stats.spearmanr(sub[col1], sub[col2])
        rho_mat[i, j] = rho_mat[j, i] = rho
        pval_mat[i, j] = pval_mat[j, i] = p

    corrected = np.minimum(pval_mat * n_pairs, 1.0)
    np.fill_diagonal(corrected, 0)
    return rho_mat, pval_mat, corrected


def main():
    print("=" * 60)
    print("Step 3: Cross-Stage Spearman Correlation")
    print("=" * 60)

    df = pd.read_csv(QCP_CSV)
    print(f"  Loaded {len(df)} embryos")

    rho_mat, pval_mat, corrected_mat = compute_spearman_matrix(df)

    rho_df  = pd.DataFrame(rho_mat,       index=STAGES, columns=STAGES)
    pval_df = pd.DataFrame(pval_mat,      index=STAGES, columns=STAGES)
    corr_df = pd.DataFrame(corrected_mat, index=STAGES, columns=STAGES)

    rho_df.to_csv(RESULT_DIR / "spearman_rho.csv")
    pval_df.to_csv(RESULT_DIR / "spearman_pval.csv")
    corr_df.to_csv(RESULT_DIR / "spearman_pval_bonferroni.csv")

    print("\n  Spearman ρ matrix:")
    print(rho_df.round(3).to_string())
    print("\n  Bonferroni-corrected p-values:")
    print(corr_df.round(6).to_string())

    # 逐对打印
    print("\n  Pairwise results:")
    print(f"  {'Pair':<12} {'ρ':>8} {'p':>12} {'p_bonf':>12} {'sig':>6}")
    print("  " + "-" * 52)
    for i, j in combinations(range(len(STAGES)), 2):
        s1, s2 = STAGES[i], STAGES[j]
        rho = rho_mat[i, j]
        p   = pval_mat[i, j]
        pb  = corrected_mat[i, j]
        sig = "***" if pb < 0.001 else ("**" if pb < 0.01 else ("*" if pb < 0.05 else "ns"))
        print(f"  {s1}-{s2:<8} {rho:>8.4f} {p:>12.2e} {pb:>12.2e} {sig:>6}")

    print(f"\n  Saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
