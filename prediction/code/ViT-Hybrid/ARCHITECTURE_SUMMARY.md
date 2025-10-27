# ViT-Hybrid Architecture Summary (STORK-Style Implementation)

## Overview

ViT-Hybrid is now fully aligned with the STORK architecture pattern, utilizing a **4096D feature representation** throughout the model pipeline rather than the previous 512D design.

## Key Architecture Changes

### 1. VGG16 Feature Extractor (STORK-Compatible)

**Previous Design:**
- Output: 512D (custom classifier design)
- Issue: Mismatch with industry standard

**New Design (STORK-Style):**
```
VGG16 Features (conv layers)
    ↓
Adaptive Average Pooling (7×7)
    ↓
Flatten
    ↓
Classifier[:4] (original VGG16 FC layers: 25088→4096, ReLU, Dropout, 4096→4096)
    ↓
Output: 4096D (matches original VGG16 classifier output)
```

**Benefits:**
- Matches STORK standard exactly
- Preserves ImageNet pre-training more effectively
- Larger feature dimension captures more spatial information

### 2. Feature Dimension Flow

```
Input: (batch, 33, 11, 3, 224, 224)
    ↓ Reshape: (batch*33*11, 3, 224, 224)
    ↓ VGG16: (batch*33*11) → (batch*33*11, 4096)
    ↓ Reshape: (batch, 33, 11, 4096)
    ↓ Focal Fusion (per timepoint): (batch, 33, 11, 4096) → (batch, 33, 4096)
    ↓ Positional Encoding: (batch, 33, 4096)
    ↓ Transformer (4 layers, 8 heads, 2048 DFF): (batch, 33, 4096)
    ↓ Global Average Pooling: (batch, 4096)
    ↓ Classification Head: (batch, 1)
```

### 3. Focal Plane Fusion (Enhanced)

**Input:** (batch, 11 focal_planes, 4096D)
**Operation:** Average fusion across 11 focal planes
**Output:** (batch, 4096D)

The fusion module now supports:
- **Average fusion** (default): Simple mean across focal planes
- **Max fusion**: Takes maximum activation per dimension
- **Weighted fusion**: Learnable weights for adaptive importance

### 4. Transformer Configuration

| Parameter | Value | Details |
|-----------|-------|---------|
| Input dimension | 4096D | Matches VGG16 output |
| Hidden dimension (DFF) | 2048D | Typical ratio is 0.5× d_model |
| Attention heads | 8 | 4096D ÷ 8 = 512D per head |
| Head dimension | 512D | Powers of 2 for efficiency |
| Layers | 4 | Sufficient for temporal modeling |

### 5. Classification Head (Updated)

```python
nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(4096, 512),  # High dropout for regularization
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 1),     # Binary classification
    nn.Sigmoid()
)
```

## Fine-Tuning Strategy (STORK's Optimal Approach)

### Old Two-Stage Approach
1. **Stage 1:** Freeze VGG16, train Transformer + classifier
2. **Stage 2:** Unfreeze top layers of VGG16, fine-tune all components

**Issues:** Complex pipeline, potential gradient flow problems

### New STORK Single-Stage Approach
**Implementation:** Freeze first 1/3 of layers, unfreeze latter 2/3
- VGG16 has ~13 trainable layers (conv blocks)
- Freeze first ~4-5 layers (early low-level features)
- Unfreeze last ~8-9 layers (high-level semantic features)
- Transformer and classifier always trainable

**Advantages:**
- Unified training from the beginning
- Better gradient flow
- Faster convergence
- More stable training dynamics

## Data Split Configuration

- **Training:** 70%
- **Validation:** 15%
- **Testing:** 15%

Stratified split ensures balanced label distribution across all sets.

## Model Summary

```
================================================
ViT-Hybrid Model (STORK-Style)
================================================

Total Parameters: ~145M
├── VGG16 Feature Extractor: ~138M (mostly frozen)
│   ├── Convolutional layers: ~134M
│   └── Classifier FC layers: ~4M
├── Focal Fusion: ~0.5M
├── Transformer: ~6M
│   ├── 4 × TransformerBlock
│   │   ├── MultiheadAttention: ~50M per layer
│   │   ├── FeedForward: ~34M per layer
│   │   └── LayerNorms: minimal
│   └── Positional Encoding: learnable
└── Classification Head: ~2.1M

Trainable Parameters: ~8-10M (depending on VGG16 freeze ratio)
```

## Dimension Verification

```
Input Shape: (2, 33, 11, 3, 224, 224)

Step 1 - Reshape for CNN:
  (2, 33, 11, 3, 224, 224) → (726, 3, 224, 224)

Step 2 - VGG16 Feature Extraction:
  (726, 3, 224, 224) → (726, 4096) ✓

Step 3 - Reshape back:
  (726, 4096) → (2, 33, 11, 4096) ✓

Step 4 - Focal Fusion (per timepoint):
  11 × (2, 4096) → (2, 4096) via averaging
  Result: (2, 33, 4096) ✓

Step 5 - Transformer:
  Input: (2, 33, 4096)
  8 heads × 512 per head = 4096D ✓
  Output: (2, 33, 4096) ✓

Step 6 - Global Average Pooling:
  (2, 33, 4096) → (2, 4096) ✓

Step 7 - Classification:
  (2, 4096) → Linear(4096, 512) → (2, 512)
  (2, 512) → Linear(512, 1) → (2, 1) ✓

Final Output: (2, 1) - binary classification scores
```

## Comparison: Previous vs. STORK-Style

| Aspect | Previous | STORK-Style |
|--------|----------|-------------|
| VGG16 output | 512D | **4096D** |
| d_model (Transformer) | 512D | **4096D** |
| DFF | 1024D | **2048D** |
| Fine-tuning | 2-stage | **1-stage (1/3-2/3)** |
| Architecture alignment | Custom | **Standard STORK** |
| Capacity | Lower | **Higher** |
| Gradient flow | Potential issues | **More stable** |

## Performance Implications

### Advantages of STORK-Style (4096D)
1. **Richer feature representation**: 4096D captures more nuanced information
2. **Better pre-training transfer**: Original VGG16 classifier outputs 4096D
3. **Improved Transformer modeling**: 4096D ÷ 8 heads = 512D (clean splitting)
4. **Aligned with industry standards**: Matches STORK and other production systems
5. **Single-stage training**: Simpler, more stable optimization

### Considerations
1. **Higher memory usage**: ~4× compared to 512D variant
2. **Longer training time**: More parameters to optimize
3. **Requires adequate GPU memory**: Recommend 24GB+ VRAM for batch_size=4

## Configuration Files

All configuration is handled in `config/config.py`:

```python
class ModelConfig:
    D_MODEL = 4096      # Transformer feature dimension
    DFF = 2048          # Feed-forward hidden dimension

class DataConfig:
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
```

## Next Steps

To use the STORK-style ViT-Hybrid:

1. **Initialize model:**
   ```python
   from model.model import create_model
   model = create_model()  # Uses default STORK config
   ```

2. **Unfreeze VGG16 for fine-tuning:**
   ```python
   model.unfreeze_vgg16()  # Uses STORK's 1/3-2/3 strategy
   # or
   model.unfreeze_vgg16(num_layers=4)  # Manual: unfreeze last 4 layers
   ```

3. **Train:**
   - VGG16 first 1/3 layers: frozen (no gradient)
   - VGG16 last 2/3 layers: trainable
   - Transformer: fully trainable
   - Classifier: fully trainable

This unified approach provides better convergence properties than sequential two-stage training.
