"""
Multi-focal Multi-timepoint Data Loader for ViT-Hybrid
Loads and validates data from selected_image_paths.csv with completeness checks
Supports caching validation results to avoid redundant checks

Author: Liao Zizhuo
Date: 2025-10-27
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import hashlib
from datetime import datetime


class ValidationCache:
    """Caches validation results to avoid redundant checks"""

    def __init__(self, cache_dir='~/.vit_hybrid_cache'):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, csv_path: str) -> Path:
        """Generate cache file path based on CSV hash"""
        csv_hash = hashlib.md5(csv_path.encode()).hexdigest()[:8]
        return self.cache_dir / f"validation_cache_{csv_hash}.json"

    def load_cache(self, csv_path: str) -> Dict:
        """Load cached validation results"""
        cache_path = self.get_cache_path(csv_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  无法读取缓存: {e}")
                return None
        return None

    def save_cache(self, csv_path: str, data: Dict) -> None:
        """Save validation results to cache"""
        cache_path = self.get_cache_path(csv_path)
        data['cached_at'] = datetime.now().isoformat()
        data['csv_path'] = csv_path
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  无法保存缓存: {e}")

    def clear_cache(self) -> None:
        """Clear all cached validation results"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("✓ 缓存已清除")


class MultiFocalDataValidator:
    """Validates multi-focal time-lapse data for completeness"""

    EXPECTED_TIMEPOINTS = 33  # Expected number of timepoints (96-112 hours, 0.5 hour intervals)

    def __init__(self, verbose=True, use_cache=True, cache_dir=None):
        self.verbose = verbose
        self.use_cache = use_cache
        self.cache = ValidationCache(cache_dir=cache_dir) if use_cache else None
        self.validation_stats = {
            'total_embryos': 0,
            'valid_embryos': 0,
            'rejected_embryos': 0,
            'rejection_reasons': {}
        }

    def validate_embryo_completeness(self, embryo_df: pd.DataFrame, embryo_id: str) -> Tuple[bool, str]:
        """
        Validate if an embryo has complete 33 timepoints

        Args:
            embryo_df: DataFrame containing all images for one embryo
            embryo_id: Embryo ID for logging

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if embryo has data
        if embryo_df.empty:
            reason = "No image data"
            self._log_rejection(reason)
            return False, reason

        # Count unique timepoints
        unique_timepoints = embryo_df['TargetHour'].nunique()

        if unique_timepoints != self.EXPECTED_TIMEPOINTS:
            reason = f"Incomplete timepoints: {unique_timepoints}/{self.EXPECTED_TIMEPOINTS}"
            self._log_rejection(reason)
            return False, reason

        # Check if all images exist
        missing_files = []
        for _, row in embryo_df.iterrows():
            full_path = row['FullPath']
            if not Path(full_path).exists():
                missing_files.append(full_path)

        if missing_files:
            reason = f"Missing {len(missing_files)} image files"
            self._log_rejection(reason)
            if self.verbose and len(missing_files) <= 5:
                for f in missing_files[:5]:
                    print(f"  Missing: {f}")
            return False, reason

        return True, "Valid"

    def _log_rejection(self, reason: str):
        """Log rejection reason"""
        if reason not in self.validation_stats['rejection_reasons']:
            self.validation_stats['rejection_reasons'][reason] = 0
        self.validation_stats['rejection_reasons'][reason] += 1

    def print_summary(self):
        """Print validation summary"""
        if self.verbose:
            print("\n" + "="*70)
            print("MULTI-FOCAL DATA VALIDATION SUMMARY")
            print("="*70)
            print(f"Total embryos: {self.validation_stats['total_embryos']}")
            print(f"Valid embryos: {self.validation_stats['valid_embryos']}")
            print(f"Rejected embryos: {self.validation_stats['rejected_embryos']}")
            if self.validation_stats['valid_embryos'] > 0:
                success_rate = self.validation_stats['valid_embryos'] / self.validation_stats['total_embryos'] * 100
                print(f"Success rate: {success_rate:.1f}%")

            if self.validation_stats['rejection_reasons']:
                print("\nRejection reasons:")
                for reason, count in sorted(self.validation_stats['rejection_reasons'].items(),
                                          key=lambda x: x[1], reverse=True):
                    print(f"  {reason}: {count}")
            print("="*70)


class MultiFocalDataLoader:
    """Loads and prepares multi-focal time-lapse data from selected_image_paths.csv"""

    def __init__(self, verbose=True, use_cache=True, cache_dir=None):
        self.verbose = verbose
        self.use_cache = use_cache
        self.validator = MultiFocalDataValidator(verbose=verbose, use_cache=use_cache, cache_dir=cache_dir)

    def load_multi_focal_data(self,
                            selected_images_csv: str,
                            score_dict: Dict,
                            skip_validation=False) -> Tuple[pd.DataFrame, Dict]:
        """
        Load multi-focal data from selected_image_paths.csv with optional validation caching

        Args:
            selected_images_csv: Path to selected_image_paths.csv
            score_dict: Quality labels dictionary
            skip_validation: If True and cache exists, skip validation and use cached results

        Returns:
            Tuple of (validated_df, embryo_to_label)
        """
        print("\n" + "="*70)
        print("LOADING MULTI-FOCAL DATA")
        print("="*70)

        # Load CSV
        if not Path(selected_images_csv).exists():
            raise FileNotFoundError(f"selected_image_paths.csv not found: {selected_images_csv}")

        multi_focal_df = pd.read_csv(selected_images_csv)
        print(f"Loaded {len(multi_focal_df)} image records")

        # Create EmbryoID if not present
        if 'EmbryoID' not in multi_focal_df.columns:
            multi_focal_df['EmbryoID'] = (
                multi_focal_df['PatientID'].astype(str) + '_' +
                multi_focal_df['EmbryoNumber'].astype(str)
            )

        # Try to load cached validation results
        cached_results = None
        if self.use_cache and skip_validation:
            cached_results = self.validator.cache.load_cache(selected_images_csv)
            if cached_results:
                print("✓ 从缓存加载验证结果...")
                valid_embryo_ids = cached_results.get('valid_embryo_ids', [])
                embryo_to_label = cached_results.get('embryo_to_label', {})
                self.validator.validation_stats = cached_results.get('validation_stats', {})
                self.validator.print_summary()

                validated_df = multi_focal_df[multi_focal_df['EmbryoID'].isin(valid_embryo_ids)].copy()
                print(f"\nFinal dataset: {len(validated_df)} images from {len(valid_embryo_ids)} embryos")
                return validated_df, embryo_to_label

        # Otherwise, perform validation
        print("🔍 验证胚胎质量标签和时间点完整性...")
        valid_embryo_ids = []
        embryo_to_label = {}

        self.validator.validation_stats['total_embryos'] = len(set(multi_focal_df['EmbryoID']))

        for embryo_id, embryo_df in multi_focal_df.groupby('EmbryoID'):
            # Check if embryo has quality label
            if embryo_id not in score_dict:
                self.validator._log_rejection("No quality label")
                continue

            # Validate completeness
            is_valid, reason = self.validator.validate_embryo_completeness(embryo_df, embryo_id)

            if is_valid:
                valid_embryo_ids.append(embryo_id)
                embryo_to_label[embryo_id] = score_dict[embryo_id]['quality']
                self.validator.validation_stats['valid_embryos'] += 1
            else:
                self.validator.validation_stats['rejected_embryos'] += 1

        # Save validation results to cache
        if self.use_cache:
            cache_data = {
                'valid_embryo_ids': valid_embryo_ids,
                'embryo_to_label': embryo_to_label,
                'validation_stats': self.validator.validation_stats
            }
            self.validator.cache.save_cache(selected_images_csv, cache_data)
            print("✓ 验证结果已缓存")

        # Filter to valid embryos only
        validated_df = multi_focal_df[multi_focal_df['EmbryoID'].isin(valid_embryo_ids)].copy()

        self.validator.print_summary()

        if len(valid_embryo_ids) == 0:
            raise ValueError("No valid embryos found after validation")

        print(f"\nFinal dataset: {len(validated_df)} images from {len(valid_embryo_ids)} embryos")

        return validated_df, embryo_to_label

    def prepare_multi_focal_training_data(self,
                                        validated_df: pd.DataFrame,
                                        embryo_to_label: Dict) -> Tuple[List[str], np.ndarray]:
        """
        Prepare training data from validated multi-focal dataset

        Args:
            validated_df: Validated dataframe from load_multi_focal_data
            embryo_to_label: Embryo ID to quality label mapping

        Returns:
            Tuple of (embryo_ids, labels)
        """
        embryo_ids = sorted(set(validated_df['EmbryoID']))
        labels = np.array([embryo_to_label[eid] for eid in embryo_ids])

        print(f"\nTraining data prepared:")
        print(f"  Total embryos: {len(embryo_ids)}")
        print(f"  Good (1): {np.sum(labels == 1)}")
        print(f"  Poor (0): {np.sum(labels == 0)}")
        print(f"  Balance: {np.sum(labels == 1) / len(labels) * 100:.1f}% good embryos")

        return embryo_ids, labels


def load_score_dict_for_multi_focal(maindb_csv: str,
                                   filtered_csv: str = None) -> Dict:
    """
    Load and merge quality labels from MainDB.csv and filtered_clean_embryos.csv

    Args:
        maindb_csv: Path to MainDB.csv
        filtered_csv: Path to filtered_clean_embryos.csv (optional)

    Returns:
        Dictionary mapping embryo_id to {'quality': 0/1, 'bs_score': float}
    """
    print("\n" + "="*70)
    print("LOADING QUALITY LABELS")
    print("="*70)

    # Load MainDB.csv
    if not Path(maindb_csv).exists():
        raise FileNotFoundError(f"MainDB.csv not found: {maindb_csv}")

    df_main = pd.read_csv(maindb_csv)
    print(f"Loaded MainDB.csv: {len(df_main)} embryos")

    # Mapping for quality
    quality_map = {'good': 1, 'Good': 1, 'poor': 0, 'Poor': 0, 'fair': 0, 'Fair': 0}
    icm_map = {'A': 1, 'B': 2, 'C': 4}
    te_map = {'A': 1, 'B': 3, 'C': 4}

    score_dict = {}

    # Process MainDB
    for _, row in df_main.iterrows():
        embryo_id = str(row['EmbryoID']).strip()

        if pd.isna(row.get('Quality')) or row.get('Quality') not in quality_map:
            continue

        # Calculate BS score
        icm_score = icm_map.get(str(row.get('ICM', 'C')), 4)
        te_score = te_map.get(str(row.get('TE', 'C')), 4)
        bs_score = icm_score + te_score

        quality = quality_map[row['Quality']]

        score_dict[embryo_id] = {
            'quality': quality,
            'bs_score': float(bs_score),
            'source': 'MainDB'
        }

    print(f"Processed MainDB: {len(score_dict)} embryos with valid quality labels")

    # Load and merge filtered_clean_embryos.csv
    if filtered_csv and Path(filtered_csv).exists():
        df_filtered = pd.read_csv(filtered_csv)
        print(f"Loaded filtered_clean_embryos.csv: {len(df_filtered)} embryos")

        for _, row in df_filtered.iterrows():
            embryo_id = str(row['EmbryoID']).strip()

            if pd.isna(row.get('EmbryoQuality')):
                continue

            quality_raw = str(row['EmbryoQuality']).lower()
            if quality_raw in ['poor', 'no', 'non_viable']:
                quality = 0
            elif quality_raw in ['good', 'yes', 'viable']:
                quality = 1
            else:
                continue

            # Add or update
            if embryo_id not in score_dict:
                score_dict[embryo_id] = {
                    'quality': quality,
                    'bs_score': 6.0,  # Default BS score for poor embryos
                    'source': 'filtered'
                }

        print(f"After merging: {len(score_dict)} total embryos")

    # Print quality distribution
    good_count = sum(1 for v in score_dict.values() if v['quality'] == 1)
    poor_count = len(score_dict) - good_count
    print(f"\nQuality distribution:")
    print(f"  Good: {good_count} ({good_count/len(score_dict)*100:.1f}%)")
    print(f"  Poor: {poor_count} ({poor_count/len(score_dict)*100:.1f}%)")
    print("="*70)

    return score_dict
