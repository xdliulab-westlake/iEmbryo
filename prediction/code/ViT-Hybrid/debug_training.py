"""
Debug script to monitor training dynamics
Tracks: logits variance, gradient norms, output statistics
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np

# Add model path
sys.path.insert(0, str(Path(__file__).parent))

from model.model import ViTHybridModel


def compute_model_grad_norm(model):
    """Compute total gradient norm across all parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    return (total_norm) ** 0.5


def compute_layer_grad_stats(model):
    """Compute gradient statistics per layer"""
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.mean().item()
            stats[name] = {'norm': grad_norm, 'mean': grad_mean}
    return stats


def test_gradient_flow():
    """Test that gradients flow through the entire model"""
    print("Testing gradient flow...")

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

    device = torch.device('cpu')
    model = model.to(device)
    model.train()

    # Create dummy data
    batch_size = 2
    dummy_input = torch.randn(batch_size, 33, 11, 3, 224, 224).to(device)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long).to(device)

    # Forward pass
    print("\n1. Forward pass...")
    outputs = model(dummy_input)  # (batch_size, 2)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Logits stats - Mean: {outputs.mean():.6f}, Std: {outputs.std():.6f}")
    print(f"   Logits range: [{outputs.min():.6f}, {outputs.max():.6f}]")

    # Loss computation
    print("\n2. Loss computation...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, dummy_labels)
    print(f"   Loss: {loss.item():.6f}")

    # Backward pass
    print("\n3. Backward pass...")
    loss.backward()
    print(f"   ✓ Backward pass completed")

    # Gradient statistics
    print("\n4. Gradient flow analysis...")
    total_grad_norm = compute_model_grad_norm(model)
    print(f"   Total gradient norm: {total_grad_norm:.6f}")

    # Check classifier gradients specifically
    print("\n5. Classifier layer gradients (output layers)...")
    classifier_grad_norm = 0
    for name, param in model.named_parameters():
        if 'classifier' in name and param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            classifier_grad_norm += grad_norm ** 2
            print(f"   {name}: norm={grad_norm:.6f}")
    classifier_grad_norm = classifier_grad_norm ** 0.5
    print(f"   Total classifier gradient norm: {classifier_grad_norm:.6f}")

    # Check VGG16 gradients
    print("\n6. VGG16 gradients (frozen during stage 1, unfrozen in stage 2)...")
    vgg_has_grad = False
    for name, param in model.named_parameters():
        if 'vgg16' in name and param.grad is not None:
            if param.grad.data.norm(2).item() > 0:
                vgg_has_grad = True
                break

    if vgg_has_grad:
        print(f"   ✓ VGG16 has gradients (layer unfrozen)")
    else:
        print(f"   ✓ VGG16 has NO gradients (layer frozen - expected in stage 1)")

    # Warnings
    print("\n7. Diagnostic checks...")
    if total_grad_norm < 1e-6:
        print(f"   ⚠️  WARNING: Very small total gradient norm ({total_grad_norm:.2e})")
        print(f"      This may indicate: vanishing gradients, frozen layers, or dead neurons")
    else:
        print(f"   ✓ Gradient norm is healthy: {total_grad_norm:.6f}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    test_gradient_flow()
