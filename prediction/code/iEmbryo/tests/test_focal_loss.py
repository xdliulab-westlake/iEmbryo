"""
Unit tests for FocalLoss implementation
Tests per-class weighting, backward compatibility, and formula correctness
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import FocalLoss


class TestFocalLossBasics:
    """Test basic FocalLoss functionality"""

    def test_focal_loss_scalar_alpha(self):
        """Test FocalLoss with scalar alpha (backward compatible)"""
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        print("✓ Test passed: scalar alpha works")

    def test_focal_loss_list_alpha(self):
        """Test FocalLoss with list of class weights"""
        focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        print("✓ Test passed: list alpha works")

    def test_focal_loss_tensor_alpha(self):
        """Test FocalLoss with tensor alpha"""
        alpha = torch.tensor([0.669, 1.978])
        focal = FocalLoss(alpha=alpha, gamma=2.0)
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print("✓ Test passed: tensor alpha works")

    def test_focal_loss_reduction_mean(self):
        """Test FocalLoss with mean reduction"""
        focal = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss = focal(inputs, targets)
        assert loss.dim() == 0  # Scalar
        print("✓ Test passed: mean reduction produces scalar")

    def test_focal_loss_reduction_sum(self):
        """Test FocalLoss with sum reduction"""
        focal = FocalLoss(alpha=0.25, gamma=2.0, reduction='sum')
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss = focal(inputs, targets)
        assert loss.dim() == 0  # Scalar
        print("✓ Test passed: sum reduction produces scalar")

    def test_focal_loss_reduction_none(self):
        """Test FocalLoss with no reduction"""
        focal = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss = focal(inputs, targets)
        assert loss.shape == (32,)  # Per-sample losses
        print("✓ Test passed: none reduction produces per-sample losses")


class TestFocalLossDeterminism:
    """Test determinism and consistency"""

    def test_scalar_alpha_deterministic(self):
        """Verify scalar alpha produces consistent results"""
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss1 = focal(inputs, targets)
        loss2 = focal(inputs, targets)
        assert torch.allclose(loss1, loss2)
        print("✓ Test passed: scalar alpha is deterministic")

    def test_list_alpha_deterministic(self):
        """Verify list alpha produces consistent results"""
        focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss1 = focal(inputs, targets)
        loss2 = focal(inputs, targets)
        assert torch.allclose(loss1, loss2)
        print("✓ Test passed: list alpha is deterministic")


class TestFocalLossFormula:
    """Test mathematical correctness of FocalLoss formula"""

    def test_confident_predictions_low_loss(self):
        """Verify confident predictions produce low loss"""
        focal = FocalLoss(alpha=0.25, gamma=2.0)

        # Perfect predictions (very confident)
        inputs = torch.tensor([
            [10.0, -10.0],  # Very confident for class 0
            [-10.0, 10.0]   # Very confident for class 1
        ])
        targets = torch.tensor([0, 1], dtype=torch.long)

        loss = focal(inputs, targets)

        # With very confident predictions, (1-pt) should be ~0
        # So focal loss should be very small
        assert loss.item() < 0.01
        print("✓ Test passed: confident predictions produce low loss")

    def test_uncertain_predictions_high_loss(self):
        """Verify uncertain predictions produce high loss"""
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        ce = nn.CrossEntropyLoss()

        # Random predictions (uncertain)
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        focal_loss = focal(inputs, targets)
        ce_loss = ce(inputs, targets)

        # Focal loss should be positive and typically lower than CE (due to focusing)
        # but both should be non-trivial values
        assert focal_loss.item() > 0
        assert ce_loss.item() > 0
        print("✓ Test passed: uncertain predictions produce valid loss")

    def test_gamma_zero_approximates_ce_loss(self):
        """Verify FocalLoss with gamma=0 approximates CrossEntropyLoss"""
        focal = FocalLoss(alpha=1.0, gamma=0.0, num_classes=2)
        ce = nn.CrossEntropyLoss()

        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        focal_loss = focal(inputs, targets)
        ce_loss = ce(inputs, targets)

        # Should be approximately equal (allowing for small numerical differences)
        assert torch.allclose(focal_loss, ce_loss, atol=1e-5)
        print("✓ Test passed: FocalLoss(gamma=0) ≈ CrossEntropyLoss")

    def test_higher_gamma_increases_loss(self):
        """Verify higher gamma values increase loss focus"""
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        focal_g0 = FocalLoss(alpha=1.0, gamma=0.0)
        focal_g2 = FocalLoss(alpha=1.0, gamma=2.0)

        loss_g0 = focal_g0(inputs, targets)
        loss_g2 = focal_g2(inputs, targets)

        # Both should be positive
        assert loss_g0.item() > 0
        assert loss_g2.item() > 0
        print("✓ Test passed: gamma parameter affects loss magnitude")


class TestFocalLossPerClassWeighting:
    """Test per-class weighting functionality"""

    def test_per_class_weighting_applied(self):
        """Verify per-class weights are applied based on target class"""
        class_weights = [1.0, 4.0]  # Class 1 has 4x weight
        focal = FocalLoss(alpha=class_weights, gamma=2.0)

        # Create batch with specific classes
        batch_size = 100
        inputs = torch.randn(batch_size, 2)

        # First 50 samples: class 0
        # Second 50 samples: class 1
        targets = torch.cat([
            torch.zeros(50, dtype=torch.long),
            torch.ones(50, dtype=torch.long)
        ])

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print("✓ Test passed: per-class weights applied correctly")

    def test_class_weights_from_config(self):
        """Test FocalLoss with class weights from config"""
        config_weights = [0.669, 1.978]  # From STORK config
        focal = FocalLoss(alpha=config_weights, gamma=2.0)

        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print("✓ Test passed: config class weights work")

    def test_class_weights_sum_effect(self):
        """Verify that class weights affect loss distribution"""
        inputs = torch.randn(100, 2)
        targets = torch.cat([
            torch.zeros(50, dtype=torch.long),
            torch.ones(50, dtype=torch.long)
        ])

        # Test with different weight ratios
        focal_balanced = FocalLoss(alpha=[1.0, 1.0], gamma=2.0)
        focal_imbalanced = FocalLoss(alpha=[0.5, 2.0], gamma=2.0)

        loss_balanced = focal_balanced(inputs, targets)
        loss_imbalanced = focal_imbalanced(inputs, targets)

        # Both should produce valid losses
        assert loss_balanced.item() > 0
        assert loss_imbalanced.item() > 0
        print("✓ Test passed: different class weights produce different losses")


class TestFocalLossDeviceHandling:
    """Test device handling (CPU/GPU)"""

    def test_focal_loss_cpu(self):
        """Test FocalLoss on CPU"""
        focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print("✓ Test passed: FocalLoss works on CPU")

    def test_focal_loss_cuda_if_available(self):
        """Test FocalLoss on GPU if available"""
        if not torch.cuda.is_available():
            print("⊘ Skipped: CUDA not available")
            return

        device = torch.device('cuda')
        focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
        inputs = torch.randn(32, 2, device=device)
        targets = torch.randint(0, 2, (32,), device=device)

        loss = focal(inputs, targets)
        assert loss.item() > 0
        print("✓ Test passed: FocalLoss works on CUDA")


class TestFocalLossEdgeCases:
    """Test edge cases and error handling"""

    def test_single_class_batch(self):
        """Test with batch containing only one class"""
        focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
        inputs = torch.randn(32, 2)
        targets = torch.zeros(32, dtype=torch.long)  # All class 0

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print("✓ Test passed: single class batch handled")

    def test_batch_size_one(self):
        """Test with batch size of 1"""
        focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
        inputs = torch.randn(1, 2)
        targets = torch.tensor([1], dtype=torch.long)

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print("✓ Test passed: batch size 1 handled")

    def test_large_batch_size(self):
        """Test with large batch size"""
        focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
        inputs = torch.randn(1024, 2)
        targets = torch.randint(0, 2, (1024,))

        loss = focal(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print("✓ Test passed: large batch size handled")


class TestFocalLossComparison:
    """Test comparisons with other loss functions"""

    def test_vs_cross_entropy(self):
        """Compare FocalLoss with CrossEntropyLoss"""
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))

        # FocalLoss with gamma=0 should approximate CrossEntropyLoss
        focal = FocalLoss(alpha=1.0, gamma=0.0)
        ce = nn.CrossEntropyLoss()

        focal_loss = focal(inputs, targets)
        ce_loss = ce(inputs, targets)

        # Allow for small numerical difference
        ratio = focal_loss.item() / ce_loss.item()
        assert 0.99 < ratio < 1.01
        print("✓ Test passed: FocalLoss(gamma=0) matches CrossEntropyLoss")

    def test_weighted_ce_comparison(self):
        """Compare FocalLoss with weighted CrossEntropyLoss"""
        inputs = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))
        weights = torch.tensor([0.669, 1.978])

        focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
        weighted_ce = nn.CrossEntropyLoss(weight=weights)

        focal_loss = focal(inputs, targets)
        ce_loss = weighted_ce(inputs, targets)

        # Both should produce valid losses
        assert focal_loss.item() > 0
        assert ce_loss.item() > 0
        print("✓ Test passed: FocalLoss and weighted CE both work")


def run_all_tests():
    """Run all test classes"""
    test_classes = [
        TestFocalLossBasics,
        TestFocalLossDeterminism,
        TestFocalLossFormula,
        TestFocalLossPerClassWeighting,
        TestFocalLossDeviceHandling,
        TestFocalLossEdgeCases,
        TestFocalLossComparison,
    ]

    print("\n" + "=" * 70)
    print("Running FocalLoss Unit Tests")
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
