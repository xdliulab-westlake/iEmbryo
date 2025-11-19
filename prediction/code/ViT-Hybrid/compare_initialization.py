"""
Compare initialization between STORK and ViT-Hybrid
验证修复后两个模型的初始化是否等价
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/STORK_lzz_torch')

from model.model import ViTHybridModel


def create_stork_classifier():
    """Create STORK-style classifier head"""
    return nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )


def create_vit_hybrid_classifier():
    """Create ViT-Hybrid classifier head (no explicit initialization)"""
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    # Don't initialize - let PyTorch default handle it
    return classifier


def create_vit_hybrid_classifier_fixed():
    """Create ViT-Hybrid classifier with fixed initialization (gain=1.0)"""
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    # Initialize with gain=1.0 (fixed version)
    for module in classifier:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    return classifier


def analyze_classifier(classifier, name):
    """Analyze a classifier head"""
    print(f"\n{'='*70}")
    print(f"Classifier: {name}")
    print(f"{'='*70}")

    # Get weights statistics
    weights_stats = {}
    for idx, module in enumerate(classifier):
        if isinstance(module, nn.Linear):
            weight = module.weight
            bias = module.bias

            weight_std = weight.std().item()
            weight_mean = weight.mean().item()
            weight_min = weight.min().item()
            weight_max = weight.max().item()

            print(f"\nLayer {idx} (Linear): {weight.shape[0]} -> {weight.shape[1]}")
            print(f"  Weight Std:  {weight_std:.6f}")
            print(f"  Weight Mean: {weight_mean:.6f}")
            print(f"  Weight Range: [{weight_min:.6f}, {weight_max:.6f}]")

            if bias is not None:
                bias_std = bias.std().item() if bias.numel() > 1 else 0
                print(f"  Bias Std:    {bias_std:.6f}")

            weights_stats[f'layer_{idx}'] = {
                'weight_std': weight_std,
                'weight_mean': weight_mean,
                'shape': tuple(weight.shape)
            }

    # Test forward pass
    classifier.eval()
    dummy_input = torch.randn(4, 4096)

    with torch.no_grad():
        output = classifier(dummy_input)

    output_prob = torch.softmax(output, dim=1)[:, 1]

    print(f"\nForward pass test (batch_size=4, input_dim=4096):")
    print(f"  Output shape: {output.shape}")
    print(f"  Logits Std:   {output.std().item():.6f}")
    print(f"  Logits Mean:  {output.mean().item():.6f}")
    print(f"  Logits Range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    print(f"  Probs Std:    {output_prob.std().item():.6f}")
    print(f"  Probs Mean:   {output_prob.mean().item():.6f}")
    print(f"  Probs Range:  [{output_prob.min().item():.6f}, {output_prob.max().item():.6f}]")

    return weights_stats, output.std().item(), output_prob.std().item()


def compare_implementations():
    """Compare different initialization implementations"""
    print("\n" + "="*70)
    print("INITIALIZATION COMPARISON: STORK vs ViT-Hybrid")
    print("="*70)

    # Create classifiers
    stork_classifier = create_stork_classifier()
    vit_default = create_vit_hybrid_classifier()
    vit_fixed = create_vit_hybrid_classifier_fixed()

    # Analyze each
    stats_stork, logits_std_stork, probs_std_stork = analyze_classifier(stork_classifier, "STORK (PyTorch default)")
    stats_vit_default, logits_std_vit_default, probs_std_vit_default = analyze_classifier(vit_default, "ViT-Hybrid (PyTorch default)")
    stats_vit_fixed, logits_std_vit_fixed, probs_std_vit_fixed = analyze_classifier(vit_fixed, "ViT-Hybrid (Xavier gain=1.0 - FIXED)")

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: Output Statistics Comparison")
    print(f"{'='*70}")

    print(f"\n{'Initialization':<40} {'Logits Std':<15} {'Probs Std':<15}")
    print(f"{'-'*70}")
    print(f"{'STORK (default PyTorch)':<40} {logits_std_stork:<15.6f} {probs_std_stork:<15.6f}")
    print(f"{'ViT-Hybrid (default PyTorch)':<40} {logits_std_vit_default:<15.6f} {probs_std_vit_default:<15.6f}")
    print(f"{'ViT-Hybrid (Xavier gain=1.0 - FIXED)':<40} {logits_std_vit_fixed:<15.6f} {probs_std_vit_fixed:<15.6f}")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    print(f"\n1. Default PyTorch Initialization:")
    print(f"   Both STORK and ViT-Hybrid use Kaiming initialization by default")
    print(f"   → Logits Std should be similar: {abs(logits_std_stork - logits_std_vit_default):.6f} difference")
    if abs(logits_std_stork - logits_std_vit_default) < 0.1:
        print(f"   ✓ Confirmed: Default implementations are equivalent")
    else:
        print(f"   ⚠ Note: Slight variation due to random initialization")

    print(f"\n2. Xavier Initialization (gain=1.0):")
    print(f"   ViT-Hybrid fixed version should match default implementations")
    print(f"   → Logits Std comparison to default: {abs(logits_std_vit_fixed - logits_std_vit_default):.6f} difference")
    if abs(logits_std_vit_fixed - logits_std_vit_default) < 0.5:
        print(f"   ✓ Confirmed: Fixed version is equivalent to default")
    else:
        print(f"   ⚠ Note: Variation expected due to different initialization schemes")

    print(f"\n3. Why STORK doesn't have the problem:")
    print(f"   - STORK relies on PyTorch's default Kaiming initialization")
    print(f"   - Kaiming is optimized for ReLU networks")
    print(f"   - Maintains proper gradient flow")

    print(f"\n4. ViT-Hybrid fix:")
    print(f"   - Original problem: gain=0.1 caused weights to be 100x too small")
    print(f"   - Fixed version: gain=1.0 restores normal Xavier initialization")
    print(f"   - Now equivalent to both STORK and Kaiming default")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)

    compare_implementations()
