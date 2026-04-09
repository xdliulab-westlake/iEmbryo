#!/usr/bin/env python3
"""
四时期共有 Fair 胚胎 QCP 排序分析

PCA cache 建立方式（与 unified_pca_analysis.py 一致）：
  每个时期用该时期对应的 VGG16 模型提取特征，
  fit PCA 的数据 = Good(tB图片,≤1000张) + Fair(当前时期图片,≤1000张) + Poor(tB图片,≤1000张)
  → PC1 零点有实际含义：>0 偏 Good，<0 偏 Poor
  → 方向由 Good/Poor PC1 均值直接决定（Good mean > Poor mean）

时期 → 模型映射：
  tB   → vgg16_best_val_accuracy.pth        (code/)
  t5   → vgg16_best_val_accuracy_t5.pth     (prediction/)
  tM   → vgg16_best_val_accuracy_tM.pth     (prediction/)
  tPNf → vgg16_best_val_accuracy_tPNf.pth   (prediction/)
"""

import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

STAGE_DIRS = {
    "tB":   Path("/Users/lzz/Desktop/IVF/development_stage/all_fair_embryos"),
    "t5":   Path("/Users/lzz/Desktop/IVF/development_stage/all_fair_embryos_t5"),
    "tM":   Path("/Users/lzz/Desktop/IVF/development_stage/all_fair_embryos_tM"),
    "tPNf": Path("/Users/lzz/Desktop/IVF/development_stage/all_fair_embryos_tPNf"),
}

STAGE_MODELS = {
    "tB":   Path("/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/vgg16_best_val_accuracy.pth"),
    "t5":   Path("/Users/lzz/Documents/GitHub/IVF_AI/prediction/vgg16_best_val_accuracy_t5.pth"),
    "tM":   Path("/Users/lzz/Documents/GitHub/IVF_AI/prediction/vgg16_best_val_accuracy_tM.pth"),
    "tPNf": Path("/Users/lzz/Documents/GitHub/IVF_AI/prediction/vgg16_best_val_accuracy_tPNf.pth"),
}

STAGES = ["tPNf", "t5", "tM", "tB"]

# Good/Poor 胚胎目录（多焦面，按胚胎/焦面结构）
GOOD_DIR = Path("/Users/lzz/Desktop/IVF/PCA_analysis/good_1000_embryos")
POOR_DIR = Path("/Users/lzz/Desktop/IVF/PCA_analysis/poor_1000_embryos")

# 只排序指定 Fair 胚胎（来自临床单移植周期分析的 merged 结果）；None 则不过滤
FILTER_EMBRYO_CSV = Path("/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/iEmbryo/outputs/pc1_cycle_vgg16/live_birth_rate_set1/qcp_live_birth_merged.csv")

# Fair 图像来源（与 pc1_cycle_vgg16 一致）
FAIR_IMAGE_ROOT = Path("/Users/lzz/Downloads/fair_embryos")

MAX_PER_CLASS = 1000   # Good/Poor/Fair 各最多取1000张（所有焦面中随机采样）
N_COMPONENTS  = 5
RANDOM_SEED   = 42

CACHE_DIR  = BASE_DIR / "features" / "pca_cache"
RESULT_DIR = BASE_DIR / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── 找共有胚胎 ────────────────────────────────────────────────────────────────
def find_common_embryos():
    sets = {}
    for stage, d in STAGE_DIRS.items():
        sets[stage] = set(e for e in os.listdir(d) if (d / e).is_dir())
        print(f"  {stage}: {len(sets[stage])} embryos")
    common = sorted(sets["tB"] & sets["t5"] & sets["tM"] & sets["tPNf"])
    print(f"  Common across all 4 stages: {len(common)}")
    return common


def _normalize_embryo_id(eid: str) -> str:
    # Remove trailing _RUNxxx if present
    if "_RUN" in eid:
        eid = eid.split("_RUN")[0]
    return eid


def filter_common_by_csv(common_embryos, csv_path: Path):
    if csv_path is None or not csv_path.exists():
        print("  No filter CSV provided or file not found. Skipping filter.")
        return common_embryos

    df = pd.read_csv(csv_path)
    cols = set(df.columns)
    embryo_ids = set()
    slide_ids = set()

    if "embryo_id" in cols:
        embryo_ids = set(
            df["embryo_id"]
              .dropna()
              .astype(str)
              .str.strip()
              .map(_normalize_embryo_id)
        )
    if "slide_id" in cols:
        slide_ids = set(df["slide_id"].dropna().astype(str).str.strip())
    if "SlideID" in cols:
        slide_ids |= set(df["SlideID"].dropna().astype(str).str.strip())

    if not embryo_ids and not slide_ids:
        print("  Filter CSV has no embryo_id/slide_id columns. Skipping filter.")
        return common_embryos

    common_set = set(common_embryos)

    # 1) 直接用 embryo_id 过滤（优先）
    filtered = set()
    if embryo_ids:
        filtered |= (common_set & embryo_ids)

    # 2) 若没有 embryo_id，则用 SlideID 前缀匹配
    if not filtered and slide_ids:
        for eid in common_embryos:
            for sid in slide_ids:
                if eid.startswith(sid + "_WELL"):
                    filtered.add(eid)
                    break

    filtered = sorted(filtered)
    print(f"  Filter CSV: {csv_path}")
    print(f"  Filtered embryos: {len(filtered)} (from {len(common_embryos)})")
    return filtered


def filter_from_csv_against_root(csv_path: Path, root_dir: Path):
    """从CSV筛选胚胎ID，并限定在指定root目录下存在的胚胎"""
    if csv_path is None or not csv_path.exists():
        print("  No filter CSV provided or file not found. Skipping filter.")
        return []
    if root_dir is None or not root_dir.exists():
        print("  Fair image root not found. Skipping filter.")
        return []

    df = pd.read_csv(csv_path)
    cols = set(df.columns)
    embryo_ids = set()
    slide_ids = set()

    if "embryo_id" in cols:
        embryo_ids = set(
            df["embryo_id"]
              .dropna()
              .astype(str)
              .str.strip()
              .map(_normalize_embryo_id)
        )
    if "slide_id" in cols:
        slide_ids = set(df["slide_id"].dropna().astype(str).str.strip())
    if "SlideID" in cols:
        slide_ids |= set(df["SlideID"].dropna().astype(str).str.strip())

    if not embryo_ids and not slide_ids:
        print("  Filter CSV has no embryo_id/slide_id columns. Skipping filter.")
        return []

    root_set = set(e for e in os.listdir(root_dir) if (root_dir / e).is_dir())

    filtered = set()
    if embryo_ids:
        filtered |= (root_set & embryo_ids)
    if not filtered and slide_ids:
        for eid in root_set:
            for sid in slide_ids:
                if eid.startswith(sid + "_WELL"):
                    filtered.add(eid)
                    break

    filtered = sorted(filtered)
    print(f"  Filter CSV: {csv_path}")
    print(f"  Filtered embryos: {len(filtered)} (from {len(root_set)})")
    return filtered


# ── 加载 Good/Poor 图片路径（多焦面，图片级随机采样1000张）─────────────────
def load_good_poor_paths(max_per_class=MAX_PER_CLASS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    def _load(base_dir, class_name):
        paths = []
        for eid_dir in sorted(base_dir.iterdir()):
            if not eid_dir.is_dir():
                continue
            for focal_dir in sorted(eid_dir.iterdir()):
                if not focal_dir.is_dir():
                    continue
                for p in focal_dir.glob("*.jpeg"):
                    paths.append(str(p))
        if len(paths) > max_per_class:
            idx = rng.choice(len(paths), max_per_class, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        print(f"  {class_name}: {len(paths)} images (all focal planes)")
        return paths

    good_paths = _load(GOOD_DIR, "Good")
    poor_paths = _load(POOR_DIR, "Poor")
    return good_paths, poor_paths


# ── 加载 Fair 图片路径（所有焦面，图片级采样，用于 PCA fit）─────────────────
def load_fair_paths_for_pca(stage, common_embryos, max_samples=MAX_PER_CLASS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    stage_dir = STAGE_DIRS[stage]
    paths, eids = [], []
    for eid in common_embryos:
        eid_dir = stage_dir / eid
        if not eid_dir.exists():
            continue
        for focal_dir in sorted(eid_dir.iterdir()):
            if not focal_dir.is_dir():
                continue
            for p in focal_dir.glob("*.jpeg"):
                paths.append(str(p))
                eids.append(eid)
    if len(paths) > max_samples:
        idx = rng.choice(len(paths), max_samples, replace=False)
        idx = sorted(idx)
        paths = [paths[i] for i in idx]
        eids  = [eids[i]  for i in idx]
    print(f"  Fair ({stage}, PCA): {len(paths)} images (all focal planes)")
    return paths, eids


# ── 加载 Fair 图片路径（所有焦面，全量，用于排序）────────────────────────────
def load_fair_paths_all_focal(stage, common_embryos):
    stage_dir = STAGE_DIRS[stage]
    paths, eids = [], []
    for eid in common_embryos:
        eid_dir = stage_dir / eid
        if not eid_dir.exists():
            continue
        for focal_dir in sorted(eid_dir.iterdir()):
            if not focal_dir.is_dir():
                continue
            for p in focal_dir.glob("*.jpeg"):
                paths.append(str(p))
                eids.append(eid)
    print(f"  Fair ({stage}, all): {len(paths)} images from {len(set(eids))} embryos")
    return paths, eids


def load_fair_paths_from_root(root_dir: Path, embryo_ids: list):
    paths, eids = [], []
    for eid in embryo_ids:
        eid_dir = root_dir / eid
        if not eid_dir.exists():
            continue
        for focal_dir in sorted(eid_dir.iterdir()):
            if not focal_dir.is_dir():
                continue
            for p in focal_dir.glob("*.jpeg"):
                paths.append(str(p))
                eids.append(eid)
    print(f"  Fair (root): {len(paths)} images from {len(set(eids))} embryos")
    return paths, eids


# ── 构建特征提取器（512 维，与 unified_pca_analysis.py 一致）─────────────────
def build_extractor(model_path: Path):
    """
    VGG16 特征提取器（与 analyze_pc_variations.py 逻辑一致）:
      features -> avgpool -> adaptive_avg_pool2d(1,1) -> flatten
    仅加载 features/avgpool 权重，忽略 classifier 不匹配。
    """
    vgg = models.vgg16(weights=None)
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 只加载 features/avgpool 部分，兼容带有 model. 前缀的权重
    vgg_state = {
        k.replace("model.", ""): v
        for k, v in state.items()
        if k.startswith("model.features") or k.startswith("model.avgpool")
    }
    vgg.load_state_dict(vgg_state, strict=False)
    vgg.eval().to(DEVICE)

    class AvgPoolExtractor(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.features = base.features
            self.avgpool = base.avgpool

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)  # (B, 512, 7, 7)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            return x.view(x.size(0), -1)  # (B, 512)

    return AvgPoolExtractor(vgg).eval().to(DEVICE)


# ── 批量提取特征 ──────────────────────────────────────────────────────────────
def extract_features(extractor, image_paths: list, batch_size=64, desc="Extracting"):
    all_feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"  {desc}"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(TRANSFORM(Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"  [WARN] {p}: {e}")
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            all_feats.append(extractor(batch).cpu().numpy())
    return np.vstack(all_feats)


# ── PCA Cache ─────────────────────────────────────────────────────────────────
def get_cache_id(stage, model_path, n_good, n_fair, n_poor):
    cfg = json.dumps({
        "stage": stage,
        "model": str(model_path.resolve()),
        "n_good": n_good, "n_fair": n_fair, "n_poor": n_poor,
        "focal_plane": "all",
        "n_components": N_COMPONENTS,
        "seed": RANDOM_SEED,
        "version": "v5_new_good_poor_dirs",
    }, sort_keys=True)
    return hashlib.md5(cfg.encode()).hexdigest()[:8]


def save_cache(cache_id, stage, pca_model, fair_features, fair_embryo_ids,
               good_features, poor_features):
    cache_file = CACHE_DIR / f"pca_cache_{stage}_{cache_id}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump({
            "pca_model": pca_model,
            "fair_features": fair_features,
            "fair_embryo_ids": fair_embryo_ids,
            "fair_features_pca": pca_model.transform(fair_features),
            "good_features_pca": pca_model.transform(good_features),
            "poor_features_pca": pca_model.transform(poor_features),
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
        }, f)
    print(f"  Cache saved: {cache_file}")


def load_cache(cache_id, stage):
    cache_file = CACHE_DIR / f"pca_cache_{stage}_{cache_id}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    print(f"  Cache loaded: {cache_file}")
    return data


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Temporal Fair Embryo QCP Ranking  (v2: Good+Fair+Poor PCA)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Step 1: 共有胚胎
    print("\n[Step 1] Finding common embryos...")
    common_embryos = find_common_embryos()
    # 仅用于排序：使用 pc1_cycle_vgg16 的 fair_embryos 目录作为来源
    if FAIR_IMAGE_ROOT and FAIR_IMAGE_ROOT.exists():
        ranking_embryos = filter_from_csv_against_root(FILTER_EMBRYO_CSV, FAIR_IMAGE_ROOT)
    else:
        ranking_embryos = filter_common_by_csv(common_embryos, FILTER_EMBRYO_CSV)

    # Step 2: 加载 Good/Poor 路径（tB 图片，所有时期共用）
    print("\n[Step 2] Loading Good/Poor reference images (tB)...")
    good_paths, poor_paths = load_good_poor_paths()

    # Step 3: 逐时期建 PCA cache，提取 fair 胚胎 QCP
    qcp_per_stage = {eid: {} for eid in ranking_embryos}

    for stage in STAGES:
        print(f"\n{'─'*70}")
        print(f"[Stage: {stage}]")
        model_path = STAGE_MODELS[stage]

        # Fair 图片路径：所有焦面1000张用于 fit PCA，全量用于排序
        fair_paths_pca, fair_eids_pca = load_fair_paths_for_pca(stage, common_embryos)
        if FAIR_IMAGE_ROOT and FAIR_IMAGE_ROOT.exists():
            fair_paths_all, fair_eids_all = load_fair_paths_from_root(FAIR_IMAGE_ROOT, ranking_embryos)
        else:
            fair_paths_all, fair_eids_all = load_fair_paths_all_focal(stage, ranking_embryos)

        cache_id = get_cache_id(stage, model_path,
                                len(good_paths), len(fair_paths_pca), len(poor_paths))
        cache_data = load_cache(cache_id, stage)

        if cache_data is None:
            print(f"  Loading model: {model_path.name}")
            extractor = build_extractor(model_path)

            print(f"  Extracting Good features...")
            good_feats = extract_features(extractor, good_paths, desc="Good")
            print(f"  Extracting Poor features...")
            poor_feats = extract_features(extractor, poor_paths, desc="Poor")
            print(f"  Extracting Fair features for PCA ({stage}, {len(fair_paths_pca)})...")
            fair_feats_pca_fit = extract_features(extractor, fair_paths_pca, desc="Fair(PCA)")

            # fit PCA on Good(1000) + Fair(1000) + Poor(1000)
            all_feats = np.vstack([good_feats, fair_feats_pca_fit, poor_feats])
            print(f"  Fitting PCA on {all_feats.shape[0]} samples "
                  f"(Good={len(good_feats)}, Fair={len(fair_feats_pca_fit)}, Poor={len(poor_feats)})...")
            pca = PCA(n_components=N_COMPONENTS)
            pca.fit(all_feats)
            print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")

            # 统一方向：Good mean > Poor mean
            good_pc1 = pca.transform(good_feats)[:, 0]
            poor_pc1  = pca.transform(poor_feats)[:, 0]
            if good_pc1.mean() < poor_pc1.mean():
                pca.components_[0] *= -1
                print(f"  Flipped PC1 direction")
            print(f"  Good PC1 mean={pca.transform(good_feats)[:,0].mean():.4f}, "
                  f"Poor PC1 mean={pca.transform(poor_feats)[:,0].mean():.4f}")

            # 提取全部 fair 胚胎特征用于排序
            print(f"  Extracting ALL Fair features ({stage}, {len(fair_paths_all)})...")
            fair_feats_all = extract_features(extractor, fair_paths_all, desc="Fair(all)")
            fair_feats_pca_all = pca.transform(fair_feats_all)

            save_cache(cache_id, stage, pca, fair_feats_all, fair_eids_all,
                       good_feats, poor_feats)
        else:
            pca = cache_data["pca_model"]
            fair_feats_pca_all = cache_data["fair_features_pca"]
            fair_eids_all = cache_data["fair_embryo_ids"]
            good_pc1 = cache_data["good_features_pca"][:, 0]
            poor_pc1  = cache_data["poor_features_pca"][:, 0]
            print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")
            print(f"  Good PC1 mean={good_pc1.mean():.4f}, Poor PC1 mean={poor_pc1.mean():.4f}")

            # 若指定 fair 图像根目录，则使用该目录重新提取 fair 特征（仅复用PCA模型）
            if FAIR_IMAGE_ROOT and FAIR_IMAGE_ROOT.exists():
                print(f"  Loading model: {model_path.name}")
                extractor = build_extractor(model_path)
                print(f"  Extracting ALL Fair features from root ({len(fair_paths_all)})...")
                fair_feats_all = extract_features(extractor, fair_paths_all, desc="Fair(all)")
                fair_feats_pca_all = pca.transform(fair_feats_all)
                # fair_eids_all 已来自 root 的路径加载
            else:
                # 若仅排序特定胚胎，过滤缓存中的 fair 特征
                if ranking_embryos:
                    keep = set(ranking_embryos)
                    mask = np.array([eid in keep for eid in fair_eids_all])
                    fair_feats_pca_all = fair_feats_pca_all[mask]
                    fair_eids_all = [eid for eid, m in zip(fair_eids_all, mask) if m]

        # 按胚胎聚合：多焦面 PC1 取均值
        pc1_scores = fair_feats_pca_all[:, 0]
        eid_to_scores = {}
        for eid, score in zip(fair_eids_all, pc1_scores):
            eid_to_scores.setdefault(eid, []).append(float(score))
        for eid, scores in eid_to_scores.items():
            qcp_per_stage.setdefault(eid, {})[stage] = float(np.mean(scores))

        print(f"  Fair QCP ({len(eid_to_scores)} embryos, {len(pc1_scores)} images): "
              f"mean={pc1_scores.mean():.4f}, std={pc1_scores.std():.4f}, "
              f"min={pc1_scores.min():.4f}, max={pc1_scores.max():.4f}")

    # Step 4: 整理结果并排序
    print(f"\n{'─'*70}")
    print("[Step 4] Aggregating and ranking...")

    rows = []
    for eid in common_embryos:
        row = {"embryo_id": eid}
        scores = []
        for stage in STAGES:
            val = qcp_per_stage.get(eid, {}).get(stage, np.nan)
            row[f"qcp_{stage}"] = val
            if not np.isnan(val):
                scores.append(val)
        row["qcp_mean"] = float(np.mean(scores)) if scores else np.nan
        row["n_stages"] = len(scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    df_complete = df[df["n_stages"] == 4].copy()
    df_complete = df_complete.sort_values("qcp_mean", ascending=False).reset_index(drop=True)
    df_complete.insert(0, "rank", df_complete.index + 1)

    out_path = RESULT_DIR / "fair_embryo_temporal_ranking.csv"
    df_complete.to_csv(out_path, index=False)

    cols = ["rank", "embryo_id"] + [f"qcp_{s}" for s in STAGES] + ["qcp_mean"]
    print(f"\n  Embryos with all 4 stages: {len(df_complete)}")
    print("\n  Top 10:")
    print(df_complete[cols].head(10).to_string(index=False))
    print("\n  Bottom 10:")
    print(df_complete[cols].tail(10).to_string(index=False))
    print("\n  QCP statistics per stage:")
    for stage in STAGES:
        col = f"qcp_{stage}"
        print(f"    {stage}: mean={df_complete[col].mean():.4f}, "
              f"std={df_complete[col].std():.4f}, "
              f"min={df_complete[col].min():.4f}, max={df_complete[col].max():.4f}")

    print(f"\n{'='*70}")
    print(f"Done.  Output: {out_path}")
    print("=" * 70)
    return df_complete


if __name__ == "__main__":
    main()
