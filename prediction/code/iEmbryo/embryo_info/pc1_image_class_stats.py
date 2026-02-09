#!/usr/bin/env python3
"""
Compute PC1 per image using a cached PCA model, then aggregate by class.
Supports two input modes:
  1) folders: good/fair/poor directories
  2) xlsx: external dataset with embryo IDs + quality labels
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_VGG16_PATH = "/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/vgg16_best_val_accuracy.pth"


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, str(path)


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = []
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                feat = output.detach()
                if len(feat.shape) > 2:
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                    feat = feat.view(feat.size(0), -1)
                self.features.append(feat.cpu())

        hook_registered = False
        if hasattr(self.model, "model") and hasattr(self.model.model, "avgpool"):
            self.model.model.avgpool.register_forward_hook(hook_fn)
            hook_registered = True
        elif hasattr(self.model, "model") and hasattr(self.model.model, "features"):
            self.model.model.features.register_forward_hook(hook_fn)
            hook_registered = True

        if not hook_registered:
            modules = list(self.model.children())
            if len(modules) >= 2:
                modules[-2].register_forward_hook(hook_fn)

    def extract(self, dataloader, device):
        all_features = []
        all_paths = []
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for images, paths in dataloader:
                images = images.to(device)
                self.features = []
                _ = self.model(images)
                if self.features:
                    batch_features = self.features[0].numpy()
                    all_features.append(batch_features)
                    all_paths.extend(list(paths))

        if not all_features:
            return np.empty((0, 0)), []
        return np.vstack(all_features), all_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PC1 per image and aggregate by class"
    )
    parser.add_argument(
        "--mode",
        choices=["folders", "xlsx"],
        required=True,
        help="Input mode: folders or xlsx",
    )
    parser.add_argument(
        "--dataset-name",
        default="dataset",
        help="Name used in outputs (e.g., WMU, FDU)",
    )

    # folders mode
    parser.add_argument("--good-dir", default=None)
    parser.add_argument("--fair-dir", default=None)
    parser.add_argument("--poor-dir", default=None)

    # xlsx mode
    parser.add_argument("--xlsx-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--embryo-id-col", default="胚胎ID")
    parser.add_argument("--quality-col", default="Quality")

    # common
    parser.add_argument(
        "--pca-cache",
        required=True,
        help="Path to pca_cache_*.pkl",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model checkpoint path",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override model name (e.g., vgg16)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "default.yaml"),
        help="Config yaml used by create_model",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=2000,
        help="Random sample size per class",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (e.g., cpu, cuda, mps)",
    )
    return parser.parse_args()


def infer_model_name(model_path: str):
    lower = model_path.lower()
    if "inception" in lower:
        return "inceptionv3"
    if "densenet121" in lower:
        return "densenet121"
    if "densenet169" in lower:
        return "densenet169"
    if "densenet201" in lower:
        return "densenet201"
    if "resnet50" in lower:
        return "resnet50"
    if "resnet152" in lower:
        return "resnet152"
    if "vgg16" in lower:
        return "vgg16"
    return None


def load_pca_cache(pca_cache_path: Path):
    with open(pca_cache_path, "rb") as f:
        cache_data = torch.load(f) if str(pca_cache_path).endswith(".pt") else None
    if cache_data is None:
        import pickle
        with open(pca_cache_path, "rb") as f:
            cache_data = pickle.load(f)
    pca_model = cache_data.get("pca_model")
    if pca_model is None:
        raise ValueError("PCA model not found in cache")
    cache_config = cache_data.get("config", {})
    return pca_model, cache_config


def build_model(model_path, model_name, config_path):
    from models.backbone import create_model
    from config.config_manager import load_config

    config_manager = load_config(config_path)
    config = config_manager.config
    config["model"]["name"] = model_name
    # Force disable pretrained weights to avoid downloading ImageNet weights
    config["model"]["pretrained"] = False
    checkpoint = torch.load(model_path, map_location="cpu")
    model = create_model(config["model"])
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    return model


def build_transform(model_name):
    img_size = 299 if model_name == "inceptionv3" else 224
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collect_images_from_dir(root_dir: Path):
    images = []
    for path in root_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return images


def collect_from_class_dirs(good_dir, fair_dir, poor_dir):
    class_to_paths = {}
    class_to_paths["Good"] = collect_images_from_dir(Path(good_dir))
    class_to_paths["Fair"] = collect_images_from_dir(Path(fair_dir))
    class_to_paths["Poor"] = collect_images_from_dir(Path(poor_dir))

    class_to_mapping = {}
    for cls, paths in class_to_paths.items():
        mapping = {}
        for path in paths:
            try:
                embryo_id = path.parents[1].name
            except Exception:
                embryo_id = ""
            mapping[str(path)] = embryo_id
        class_to_mapping[cls] = mapping
    return class_to_mapping


def normalize_quality(value: str):
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    if v.lower() in {"good", "good-quality", "good_quality"}:
        return "Good"
    if v.lower() in {"fair", "fair-quality", "fair_quality"}:
        return "Fair"
    if v.lower() in {"poor", "poor-quality", "poor_quality"}:
        return "Poor"
    return None


def collect_from_xlsx(xlsx_path, image_root, embryo_id_col, quality_col):
    df = pd.read_excel(xlsx_path)
    if embryo_id_col not in df.columns or quality_col not in df.columns:
        raise ValueError(f"Missing columns: {embryo_id_col} / {quality_col}")

    class_to_mapping = {"Good": {}, "Fair": {}, "Poor": {}}
    missing_embryo_ids = []
    conflicting_quality = []
    embryo_quality = {}
    processed_embryos = set()

    for _, row in df.iterrows():
        quality = normalize_quality(row.get(quality_col))
        if quality is None:
            continue
        embryo_id = str(row.get(embryo_id_col)).strip()
        if not embryo_id or embryo_id.lower() == "nan":
            continue

        if embryo_id in embryo_quality and embryo_quality[embryo_id] != quality:
            conflicting_quality.append(
                {"embryo_id": embryo_id, "quality_a": embryo_quality[embryo_id], "quality_b": quality}
            )
            continue
        embryo_quality[embryo_id] = quality

        embryo_dir = Path(image_root) / Path(embryo_id)
        if not embryo_dir.exists():
            missing_embryo_ids.append(embryo_id)
            continue

        if embryo_id in processed_embryos:
            continue
        processed_embryos.add(embryo_id)

        for path in embryo_dir.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            class_to_mapping[quality][str(path)] = embryo_id

    return class_to_mapping, missing_embryo_ids, conflicting_quality


def sample_class_images(class_to_mapping, sample_per_class, seed):
    rng = random.Random(seed)
    sampled = []

    for cls, mapping in class_to_mapping.items():
        items = list(mapping.items())
        rng.shuffle(items)
        if sample_per_class is not None:
            items = items[: min(sample_per_class, len(items))]
        for path, embryo_id in items:
            sampled.append({
                "class": cls,
                "image_path": path,
                "embryo_id": embryo_id,
            })

    return sampled


def compute_pc1(sampled_items, pca_model, model, transform, batch_size, num_workers, device):
    image_paths = [item["image_path"] for item in sampled_items]
    dataset = ImagePathDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    extractor = FeatureExtractor(model)
    features, paths = extractor.extract(dataloader, device)
    if features.size == 0:
        raise ValueError("No features extracted; check input images.")

    features_pca = pca_model.transform(features)
    pc1_values = features_pca[:, 0]

    path_to_pc1 = {str(p): float(pc1) for p, pc1 in zip(paths, pc1_values)}
    return path_to_pc1


def spearman_rho(values, labels):
    if len(values) < 2:
        return float("nan")
    s1 = pd.Series(values).rank(method="average")
    s2 = pd.Series(labels).rank(method="average")
    corr = np.corrcoef(s1, s2)[0, 1]
    return float(corr)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_cache_path = Path(args.pca_cache)
    if not pca_cache_path.exists():
        raise FileNotFoundError(f"PCA cache not found: {pca_cache_path}")

    pca_model, cache_config = load_pca_cache(pca_cache_path)

    model_path = args.model_path or cache_config.get("model_path") or DEFAULT_VGG16_PATH
    if not model_path:
        raise ValueError("Model path not provided and not in cache config")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model_name = args.model_name or cache_config.get("model_name") or infer_model_name(str(model_path))
    if not model_name:
        raise ValueError("Unable to infer model name; pass --model-name")

    model = build_model(str(model_path), model_name, args.config)
    transform = build_transform(model_name)

    if args.mode == "folders":
        if not args.good_dir or not args.fair_dir or not args.poor_dir:
            raise ValueError("folders mode requires --good-dir, --fair-dir, --poor-dir")
        class_to_mapping = collect_from_class_dirs(args.good_dir, args.fair_dir, args.poor_dir)
        missing_embryo_ids = []
        conflicting_quality = []
    else:
        if not args.xlsx_path or not args.image_root:
            raise ValueError("xlsx mode requires --xlsx-path and --image-root")
        class_to_mapping, missing_embryo_ids, conflicting_quality = collect_from_xlsx(
            args.xlsx_path,
            args.image_root,
            args.embryo_id_col,
            args.quality_col,
        )

    sampled_items = sample_class_images(
        class_to_mapping,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )
    if not sampled_items:
        raise ValueError("No images sampled. Check inputs.")

    path_to_pc1 = compute_pc1(
        sampled_items,
        pca_model,
        model,
        transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=torch.device(args.device),
    )

    records = []
    for item in sampled_items:
        pc1 = path_to_pc1.get(item["image_path"])
        if pc1 is None:
            continue
        records.append({
            "dataset": args.dataset_name,
            "class": item["class"],
            "image_path": item["image_path"],
            "embryo_id": item["embryo_id"],
            "pc1": pc1,
        })

    if not records:
        raise ValueError("No PC1 values computed; check inputs.")

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "pc1_image_level.csv", index=False)

    summary = (
        df.groupby(["dataset", "class"], as_index=False)
        .agg(n_images=("pc1", "size"), pc1_mean=("pc1", "mean"), pc1_std=("pc1", "std"))
    )
    summary.to_csv(output_dir / "pc1_class_mean.csv", index=False)

    # Spearman correlation between PC1 and ordinal labels (Poor=0, Fair=1, Good=2)
    df_ord = df[df["class"].isin(["Good", "Fair", "Poor"])].copy()
    df_ord["label"] = df_ord["class"].map({"Good": 2, "Fair": 1, "Poor": 0})
    rho = spearman_rho(df_ord["pc1"].tolist(), df_ord["label"].tolist())
    spearman_df = pd.DataFrame([
        {
            "dataset": args.dataset_name,
            "n": len(df_ord),
            "n_good": int((df_ord["class"] == "Good").sum()),
            "n_fair": int((df_ord["class"] == "Fair").sum()),
            "n_poor": int((df_ord["class"] == "Poor").sum()),
            "rho": rho,
            "rho_positive": bool(rho > 0) if not np.isnan(rho) else False,
        }
    ])
    spearman_df.to_csv(output_dir / "spearman_correlation.csv", index=False)

    if missing_embryo_ids:
        pd.DataFrame({"embryo_id": sorted(set(missing_embryo_ids))}).to_csv(
            output_dir / "missing_embryo_ids.csv", index=False
        )

    if conflicting_quality:
        pd.DataFrame(conflicting_quality).to_csv(
            output_dir / "conflicting_quality.csv", index=False
        )

    metadata = {
        "dataset": args.dataset_name,
        "mode": args.mode,
        "pca_cache": str(pca_cache_path),
        "model_path": str(model_path),
        "model_name": model_name,
        "sample_per_class": args.sample_per_class,
        "seed": args.seed,
        "num_images": len(records),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
