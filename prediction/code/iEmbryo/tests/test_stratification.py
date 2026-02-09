"""
Unit tests for stratification verification
Tests that class distribution is maintained across data splits
"""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.parallel_processing import verify_class_distribution


class TestStratifiedSplits:
    """Test stratified data splitting"""

    def test_stratified_split_preserves_class_ratio(self):
        """Test that stratified split maintains class distribution ratio"""
        # Setup: STORK-like imbalance (83% Poor, 17% Good)
        n_poor = 3469
        n_good = 689
        labels = np.array([0] * n_poor + [1] * n_good)

        # Calculate expected ratio
        expected_ratio = n_good / (n_poor + n_good)

        # Perform stratified split (85/15)
        train_idx, test_idx = next(
            StratifiedKFold(n_splits=int(1/0.15)).split(range(len(labels)), labels)
        )

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Calculate actual ratios
        train_ratio = np.sum(train_labels == 1) / len(train_labels)
        test_ratio = np.sum(test_labels == 1) / len(test_labels)

        # Verify ratios are very close (stratification works)
        # Allow 2% deviation
        assert abs(train_ratio - expected_ratio) < 0.02, \
            f"Train ratio {train_ratio:.4f} too far from expected {expected_ratio:.4f}"
        assert abs(test_ratio - expected_ratio) < 0.02, \
            f"Test ratio {test_ratio:.4f} too far from expected {expected_ratio:.4f}"

        print(f"✓ Stratified split preserves ratio: {expected_ratio:.4f}")
        print(f"  Train ratio: {train_ratio:.4f}, Test ratio: {test_ratio:.4f}")

    def test_train_test_split_stratification(self):
        """Test sklearn train_test_split with stratify parameter"""
        labels = np.array([0] * 3469 + [1] * 689)
        indices = np.arange(len(labels))

        train_idx, test_idx = train_test_split(
            indices, test_size=0.15,
            random_state=42,
            stratify=labels
        )

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Verify split sizes (may not be exact due to rounding)
        expected_train = int(0.85 * len(labels))
        expected_test = len(labels) - expected_train
        assert abs(len(train_idx) - expected_train) <= 1
        assert abs(len(test_idx) - expected_test) <= 1

        # Verify class distribution maintained
        orig_ratio = np.mean(labels == 1)
        train_ratio = np.mean(train_labels == 1)
        test_ratio = np.mean(test_labels == 1)

        assert abs(train_ratio - orig_ratio) < 0.02
        assert abs(test_ratio - orig_ratio) < 0.02

        print("✓ train_test_split with stratify preserves distribution")

    def test_sequential_stratified_splits(self):
        """Test that sequential stratified splits maintain distribution"""
        # Simulate: train/test split, then train/val split
        labels = np.array([0] * 3469 + [1] * 689)

        # First split: 85/15
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)), test_size=0.15,
            random_state=42, stratify=labels
        )

        train_labels = labels[train_idx]

        # Second split: 70/30 of training
        train_idx2, val_idx = train_test_split(
            np.arange(len(train_labels)), test_size=15.0/85.0,
            random_state=42, stratify=train_labels
        )

        final_train_labels = train_labels[train_idx2]
        final_val_labels = train_labels[val_idx]

        # Calculate ratios at each stage
        orig_ratio = np.mean(labels == 1)
        train_ratio = np.mean(train_labels == 1)
        val_ratio = np.mean(final_val_labels == 1)

        # All should be similar (within 2%)
        assert abs(train_ratio - orig_ratio) < 0.02
        assert abs(val_ratio - orig_ratio) < 0.02

        print("✓ Sequential stratified splits maintain distribution")

    def test_stratified_kfold_consistency(self):
        """Test that StratifiedKFold creates consistent folds"""
        labels = np.array([0] * 3469 + [1] * 689)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_ratios = []
        for train_idx, val_idx in skf.split(np.arange(len(labels)), labels):
            fold_labels = labels[train_idx]
            fold_ratio = np.sum(fold_labels == 1) / len(fold_labels)
            fold_ratios.append(fold_ratio)

        # All fold ratios should be within 2% of each other
        fold_ratios = np.array(fold_ratios)
        mean_ratio = np.mean(fold_ratios)
        std_ratio = np.std(fold_ratios)

        assert std_ratio < 0.02, \
            f"Fold ratios too inconsistent: std={std_ratio:.4f}"

        print(f"✓ StratifiedKFold creates consistent folds")
        print(f"  Mean ratio: {mean_ratio:.4f} ± {std_ratio:.4f}")
        print(f"  All folds: {[f'{r:.4f}' for r in fold_ratios]}")

    def test_stratified_vs_non_stratified(self):
        """Compare stratified vs non-stratified splits"""
        labels = np.array([0] * 3469 + [1] * 689)
        orig_ratio = np.mean(labels == 1)

        # Stratified
        train_idx_s, test_idx_s = train_test_split(
            np.arange(len(labels)), test_size=0.15,
            random_state=42, stratify=labels
        )
        stratified_ratio = np.mean(labels[train_idx_s] == 1)

        # Non-stratified
        train_idx_ns, test_idx_ns = train_test_split(
            np.arange(len(labels)), test_size=0.15,
            random_state=42
        )
        non_stratified_ratio = np.mean(labels[train_idx_ns] == 1)

        # Stratified should be closer to original
        stratified_error = abs(stratified_ratio - orig_ratio)
        non_stratified_error = abs(non_stratified_ratio - orig_ratio)

        print(f"✓ Stratified vs Non-Stratified split comparison")
        print(f"  Original ratio: {orig_ratio:.4f}")
        print(f"  Stratified:     {stratified_ratio:.4f} (error: {stratified_error:.4f})")
        print(f"  Non-stratified: {non_stratified_ratio:.4f} (error: {non_stratified_error:.4f})")

        # Stratified should have lower error
        assert stratified_error < non_stratified_error, \
            "Stratified split should reduce class ratio error"


class TestVerificationFunction:
    """Test the verify_class_distribution function"""

    def test_verify_class_distribution_returns_dict(self):
        """Test that verification function returns correct structure"""
        labels_list = [
            np.array([0]*3469 + [1]*689),
            np.array([0]*2951 + [1]*586),
            np.array([0]*518 + [1]*103)
        ]
        split_names = ["Original", "Train+Val", "Test"]

        result = verify_class_distribution(
            labels_list, split_names, detailed=False
        )

        # Should return a dict with results
        assert isinstance(result, dict)
        assert "Original" in result
        assert "Train+Val" in result
        assert "Test" in result

        print("✓ verify_class_distribution returns correct structure")

    def test_verify_class_distribution_accuracy(self):
        """Test that verification function calculates correctly"""
        labels = np.array([0]*80 + [1]*20)  # 80/20 split
        split_names = ["All"]

        result = verify_class_distribution([labels], split_names, detailed=False)

        assert result["All"]["counts"] == [80, 20]
        assert abs(result["All"]["ratios"][0] - 0.8) < 0.01
        assert abs(result["All"]["ratios"][1] - 0.2) < 0.01
        assert result["All"]["total"] == 100

        print("✓ verify_class_distribution calculates correctly")

    def test_verify_class_distribution_with_perfect_split(self):
        """Test verification with perfectly stratified split"""
        orig_labels = np.array([0]*83 + [1]*17)  # ~83/17
        train_labels = np.array([0]*70 + [1]*15)  # Should be ~83/17
        test_labels = np.array([0]*13 + [1]*2)    # Should be ~87/13

        result = verify_class_distribution(
            [orig_labels, train_labels, test_labels],
            ["Original", "Train", "Test"],
            detailed=False
        )

        # All should have similar ratios
        orig_ratio = result["Original"]["ratios"][1]
        train_ratio = result["Train"]["ratios"][1]

        assert abs(train_ratio - orig_ratio) < 0.05

        print("✓ Verification detects stratification correctness")


class TestEdgeCases:
    """Test edge cases in stratification"""

    def test_stratification_with_single_minority_sample(self):
        """Test stratification with very imbalanced data"""
        # Note: sklearn.model_selection requires at least 2 samples per class
        # for stratification to work, so we use 2 as minimum
        labels = np.array([0]*999 + [1]*2)  # Minimum 2 minority samples

        train_idx, test_idx = train_test_split(
            np.arange(len(labels)), test_size=0.5,
            random_state=42, stratify=labels
        )

        # Should split without error
        assert len(train_idx) > 0
        assert len(test_idx) > 0

        print("✓ Stratification handles extreme imbalance (with 2+ samples per class)")

    def test_stratification_with_balanced_classes(self):
        """Test stratification with balanced classes"""
        labels = np.array([0]*500 + [1]*500)

        train_idx, test_idx = train_test_split(
            np.arange(len(labels)), test_size=0.2,
            random_state=42, stratify=labels
        )

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Both should have ~50/50 split
        train_ratio = np.mean(train_labels == 1)
        test_ratio = np.mean(test_labels == 1)

        assert abs(train_ratio - 0.5) < 0.01
        assert abs(test_ratio - 0.5) < 0.01

        print("✓ Stratification works with balanced data")

    def test_multiple_folds_coverage(self):
        """Test that each fold partitions data correctly"""
        labels = np.array([0]*3469 + [1]*689)
        skf = StratifiedKFold(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
            # Within each fold, train and val should not overlap
            assert len(set(train_idx) & set(val_idx)) == 0, \
                f"Fold {fold}: Train and val sets overlap"

            # Together they should cover all samples
            assert len(set(train_idx) | set(val_idx)) == len(labels), \
                f"Fold {fold}: Not all samples covered"

        print("✓ Cross-validation folds partition data correctly")


def run_all_tests():
    """Run all stratification tests"""
    test_classes = [
        TestStratifiedSplits,
        TestVerificationFunction,
        TestEdgeCases,
    ]

    print("\n" + "=" * 70)
    print("Running Stratification Unit Tests")
    print("=" * 70 + "\n")

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 70)

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name} FAILED: {str(e)}")

    print("\n" + "=" * 70)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print("=" * 70 + "\n")

    return passed_tests == total_tests


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
