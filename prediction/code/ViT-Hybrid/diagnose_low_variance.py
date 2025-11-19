"""
Diagnose why output variance is low even with proper weight initialization
Check variance at each layer of the classifier
"""
import torch
import torch.nn as nn

# Test with simple data
def test_classifier_layers():
    """Test classifier layer by layer"""
    print("Testing classifier layers individually...")

    # Create the same classifier structure
    d_model = 4096
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(d_model, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    # Initialize like our model
    for module in classifier:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    classifier.eval()

    # Create input with different variances
    print(f"\n{'='*70}")
    print("Test 1: Input with small variance (like normalized features)")
    print(f"{'='*70}")

    # Small variance input (like normalized 4096-dim features)
    x_small = torch.randn(4, d_model) * 0.001  # Very small variance

    with torch.no_grad():
        out = classifier(x_small)

    probs = torch.softmax(out, dim=1)[:, 1]
    print(f"Input std: {x_small.std().item():.6f}")
    print(f"Output std: {out.std().item():.6f}")
    print(f"Prob std: {probs.std().item():.6f}")

    print(f"\n{'='*70}")
    print("Test 2: Input with normal variance")
    print(f"{'='*70}")

    x_normal = torch.randn(4, d_model) * 1.0  # Normal variance

    with torch.no_grad():
        out = classifier(x_normal)

    probs = torch.softmax(out, dim=1)[:, 1]
    print(f"Input std: {x_normal.std().item():.6f}")
    print(f"Output std: {out.std().item():.6f}")
    print(f"Prob std: {probs.std().item():.6f}")

    print(f"\n{'='*70}")
    print("Test 3: Trace through classifier layers")
    print(f"{'='*70}")

    x = torch.randn(4, d_model) * 0.1
    print(f"\nInput (4, {d_model}): std = {x.std().item():.6f}")

    # Layer 0: Dropout(0.5)
    x = classifier[0](x)
    print(f"After Dropout(0.5): std = {x.std().item():.6f}")

    # Layer 1: Linear(4096, 512)
    x = classifier[1](x)
    print(f"After Linear(4096→512): std = {x.std().item():.6f}")

    # Layer 2: ReLU
    x = classifier[2](x)
    print(f"After ReLU: std = {x.std().item():.6f}")

    # Layer 3: Dropout(0.3)
    x = classifier[3](x)
    print(f"After Dropout(0.3): std = {x.std().item():.6f}")

    # Layer 4: Linear(512, 2)
    x = classifier[4](x)
    print(f"After Linear(512→2): std = {x.std().item():.6f}")

    print(f"\n{'='*70}")
    print("Key Insight:")
    print(f"{'='*70}")
    print("""
If input has very small variance (e.g., 0.0001), then:
- Each layer multiplies by weights with std ~0.02-0.06
- But small × large = still might be small if multiplied multiple times

Possible causes:
1. Temporal pooling output has very small variance
2. Transformer output is normalized to near-zero
3. Dropout during inference might create constant-like outputs
4. Multiple layers are bottlenecking information flow
    """)

if __name__ == "__main__":
    test_classifier_layers()
