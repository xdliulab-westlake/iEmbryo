#!/usr/bin/env python3
"""
Extract embryo images by slide ID, compute PC1 per image, and aggregate to cycle-level metrics.
"""

import argparse
import json
import re
import shutil
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract images by slideID and compute PC1 cycle metrics"
    )
    parser.add_argument(
        "--patient-slide-csv",
        default=str(Path(__file__).parent / "Patient_to_SlideIDs_tree_only.csv"),
        help="CSV with PatientID and SlideIDs columns",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Root directory that contains embryo images",
    )
    parser.add_argument(
        "--pca-cache",
        default=None,
        help="Path to pca_cache_*.pkl (from previous PCA run)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model checkpoint path (default uses cache config or vgg16_best_val_accuracy.pth)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override model name (e.g., inceptionv3, densenet121, vgg16)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "default.yaml"),
        help="Config yaml used by create_model",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "pc1_cycle"),
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract images, skip model/PCA computation",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["none", "copy", "symlink"],
        default="none",
        help="Whether to copy/symlink matched images to output",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Destination dir for extracted images (defaults to output-dir/extracted_images)",
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
        "--match-mode",
        choices=["token", "contains"],
        default="token",
        help="How to match slideID from filename",
    )
    return parser.parse_args()


def split_slide_ids(raw_value: str):
    if pd.isna(raw_value):
        return []
    parts = re.split(r"[;,|]", str(raw_value))
    return [p.strip() for p in parts if p.strip()]


def parse_slide_id_from_name(filename: str):
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return None


def parse_embryo_id_and_focal(filename: str):
    stem = Path(filename).stem
    m = re.match(r"^(.*)_F(-?\d+)$", stem)
    if m:
        embryo_id = m.group(1)
        focal = int(m.group(2))
        return embryo_id, focal
    return stem, None


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


def find_default_pca_cache(project_root: Path):
    candidates = sorted(project_root.glob("outputs/three_class_*/pca_cache/pca_cache_*.pkl"))
    return candidates[0] if candidates else None


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


def main():
    args = parse_args()
    patient_csv = Path(args.patient_slide_csv)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not patient_csv.exists():
        raise FileNotFoundError(f"CSV not found: {patient_csv}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    pca_cache_path = None
    pca_model = None
    model = None
    transform = None
    model_path = None
    model_name = None
    cache_config = {}

    if not args.extract_only:
        from models.backbone import create_model
        from config.config_manager import load_config

        pca_cache_path = Path(args.pca_cache) if args.pca_cache else None
        if pca_cache_path is None:
            pca_cache_path = find_default_pca_cache(Path(__file__).resolve().parents[1])
        if pca_cache_path is None or not pca_cache_path.exists():
            raise FileNotFoundError("No PCA cache found. Provide --pca-cache.")

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

        model_path = args.model_path or cache_config.get("model_path") or DEFAULT_VGG16_PATH
        if not model_path:
            raise ValueError("Model path not provided and not in cache config")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        model_name = args.model_name or infer_model_name(str(model_path))
        if not model_name:
            raise ValueError("Unable to infer model name; pass --model-name")

        # Load model
        config_manager = load_config(args.config)
        config = config_manager.config
        config["model"]["name"] = model_name
        checkpoint = torch.load(model_path, map_location="cpu")
        model = create_model(config["model"])
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)

        img_size = 299 if model_name == "inceptionv3" else 224
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Read patient-slide map
    patient_df = pd.read_csv(patient_csv)
    if "PatientID" not in patient_df.columns or "SlideIDs" not in patient_df.columns:
        raise ValueError("CSV must contain PatientID and SlideIDs columns")

    patient_to_slides = {}
    slide_ids = set()
    for _, row in patient_df.iterrows():
        slides = split_slide_ids(row["SlideIDs"])
        if not slides:
            continue
        patient_to_slides[row["PatientID"]] = slides
        slide_ids.update(slides)

    # Index images by slide ID
    slide_to_images = {sid: [] for sid in slide_ids}
    image_paths = []
    for path in image_root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        slide_id = parse_slide_id_from_name(path.name)
        if args.match_mode == "contains":
            for sid in slide_ids:
                if sid in path.name:
                    slide_to_images[sid].append(path)
        else:
            if slide_id in slide_to_images:
                slide_to_images[slide_id].append(path)

    for sid, paths in slide_to_images.items():
        image_paths.extend(paths)

    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise ValueError("No images matched. Check --image-root or --match-mode.")

    # Optional extraction
    if args.copy_mode != "none":
        extract_dir = Path(args.extract_dir) if args.extract_dir else output_dir / "extracted_images"
        extract_dir.mkdir(parents=True, exist_ok=True)
        for sid, paths in slide_to_images.items():
            for path in paths:
                dest_dir = extract_dir / sid
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / path.name
                if args.copy_mode == "copy":
                    if not dest.exists():
                        shutil.copy2(path, dest)
                else:
                    if not dest.exists():
                        dest.symlink_to(path)

    image_df = None
    embryo_df = None
    cycle_df = None

    if not args.extract_only:
        # Extract features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = ImagePathDataset(image_paths, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        extractor = FeatureExtractor(model)
        features, paths = extractor.extract(dataloader, device)
        if features.size == 0:
            raise ValueError("No features extracted. Check image paths and model.")

        # Compute PC1
        features_pca = pca_model.transform(features)
        pc1_values = features_pca[:, 0]

        # Build image-level table
        rows = []
        for path, pc1 in zip(paths, pc1_values):
            path = Path(path)
            slide_id = parse_slide_id_from_name(path.name)
            embryo_id, focal = parse_embryo_id_and_focal(path.name)
            rows.append({
                "image_path": str(path),
                "slide_id": slide_id,
                "embryo_id": embryo_id,
                "focal_plane": focal,
                "pc1": float(pc1),
            })

        image_df = pd.DataFrame(rows)
        image_df = image_df[image_df["slide_id"].isin(slide_ids)].copy()

    # Join patient ID
    slide_to_patient = {}
    for patient_id, slides in patient_to_slides.items():
        for sid in slides:
            slide_to_patient[sid] = patient_id
    image_df["patient_id"] = image_df["slide_id"].map(slide_to_patient)

    if not args.extract_only:
        # Join patient ID
        slide_to_patient = {}
        for patient_id, slides in patient_to_slides.items():
            for sid in slides:
                slide_to_patient[sid] = patient_id
        image_df["patient_id"] = image_df["slide_id"].map(slide_to_patient)

        # Embryo-level aggregation
        embryo_df = (
            image_df
            .groupby(["patient_id", "slide_id", "embryo_id"], as_index=False)
            .agg(
                embryo_pc1_median=("pc1", "median"),
                embryo_pc1_std=("pc1", "std"),
                embryo_pc1_max=("pc1", "max"),
                n_focal_planes=("pc1", "count"),
            )
        )

        # Cycle-level aggregation
        def p90(x):
            return float(np.percentile(x, 90)) if len(x) else np.nan

        cycle_df = (
            embryo_df
            .groupby(["patient_id"], as_index=False)
            .agg(
                pc1_median=("embryo_pc1_median", "median"),
                pc1_p90=("embryo_pc1_median", p90),
                pc1_mean=("embryo_pc1_median", "mean"),
                pc1_std=("embryo_pc1_median", "std"),
                pc1_max=("embryo_pc1_median", "max"),
                n_embryos=("embryo_pc1_median", "count"),
            )
        )

    # Save outputs
    if not args.extract_only:
        image_out = output_dir / "pc1_image_level.csv"
        embryo_out = output_dir / "pc1_embryo_level.csv"
        cycle_out = output_dir / "pc1_cycle_level.csv"

        image_df.to_csv(image_out, index=False)
        embryo_df.to_csv(embryo_out, index=False)
        cycle_df.to_csv(cycle_out, index=False)

    meta = {
        "patient_slide_csv": str(patient_csv),
        "image_root": str(image_root),
        "pca_cache": str(pca_cache_path) if pca_cache_path else None,
        "model_path": str(model_path) if model_path else None,
        "model_name": model_name,
        "match_mode": args.match_mode,
        "extract_only": bool(args.extract_only),
        "n_images": int(len(image_df)) if image_df is not None else 0,
        "n_embryos": int(len(embryo_df)) if embryo_df is not None else 0,
        "n_cycles": int(len(cycle_df)) if cycle_df is not None else 0,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if not args.extract_only:
        print("Saved:")
        print(f"  image-level: {image_out}")
        print(f"  embryo-level: {embryo_out}")
        print(f"  cycle-level: {cycle_out}")
    else:
        print("Extraction done. PC1 computation skipped (--extract-only).")


if __name__ == "__main__":
    main()
