"""
Test script to verify classifier initialization produces varied output
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add model path
sys.path.insert(0, str(Path(__file__).parent))

from model.model import ViTHybridModel


def test_initialization():
    """Test that model outputs are not constant"""
    print("Testing ViT-Hybrid model initialization...")

    # Create model
    model = ViTHybridModel(
        time_steps=33,
        focal_planes=11,
        num_transformer_layers=4,
        num_heads=8,
        d_model=4096,
        dff=2048,
        dropout_rate=0.1,
        fusion_type='average'
    )

    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 33, 11, 3, 224, 224).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(dummy_input)  # (batch_size, 2)

    # Convert to probabilities
    probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probability of class 1 (Good)

    # Check statistics
    mean_prob = probs.mean().item()
    std_prob = probs.std().item()
    min_prob = probs.min().item()
    max_prob = probs.max().item()

    print(f"\n{'='*60}")
    print(f"Classification Head Output Statistics:")
    print(f"{'='*60}")
    print(f"Mean probability:  {mean_prob:.6f}")
    print(f"Std probability:   {std_prob:.6f}")
    print(f"Min probability:   {min_prob:.6f}")
    print(f"Max probability:   {max_prob:.6f}")
    print(f"All probabilities: {probs.tolist()}")
    print(f"{'='*60}\n")

    # Check for constant output issue
    if std_prob < 0.001:
        print("⚠️  WARNING: Output variance is too low! Model is producing nearly constant outputs.")
        print(f"   This indicates a potential initialization issue.")
        return False
    else:
        print("✓ Output variance is healthy.")
        print(f"  Standard deviation: {std_prob:.6f}")
        return True


if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)
