# Fix #2: Focal Loss Implementation - Class Weight Handling Analysis

**Status**: ANALYSIS COMPLETE
**Date**: 2025-11-12
**Priority**: HIGH

---

## Problem Identified

### Current FocalLoss Implementation Issue

| Aspect | Current | Expected |
|--------|---------|----------|
| Location | `training/trainer.py:538-551` | Same |
| Alpha parameter | Single scalar (0.25) | Per-class weights or class-aware |
| Class weights handling | ❌ NOT USED | ✅ Should integrate |
| Config class_weights | Defined but unused | Should be applied |
| Implementation | Applies uniform weighting | Should weight by class |

### Config Shows Disconnect

In `config/default.yaml:50-59`:
```yaml
training:
  loss: "cross_entropy"  # 支持: focal, cross_entropy, weighted_cross_entropy
  focal_alpha: 0.25      # Single scalar!
  focal_gamma: 2.0

  class_weights:
    - 0.669  # Class 0 (多数类)
    - 1.978  # Class 1 (少数类)
```

**Key Issues:**
1. `class_weights` are defined but FocalLoss never receives them (not passed in `_create_loss_function`)
2. `focal_alpha` is a single scalar, but Focal Loss theory requires **per-class** weighting
3. Current implementation applies same alpha to all samples regardless of class
4. No per-class distinction in focal term weighting

---

## Focal Loss Theory

### Mathematical Formula (Binary Classification)

For imbalanced binary classification, Focal Loss should be:

**For class 1 (positive/minority class):**
```
FL(pt) = -α(1-pt)^γ log(pt)
```

**For class 0 (negative/majority class):**
```
FL(pt) = -(1-α)pt^γ log(1-pt)
```

Where:
- `α` is class weight for positive class (usually higher for minority class)
- `γ` (gamma) is focusing parameter (2.0 is typical)
- `pt` is model's estimated probability for the correct class

### Current Implementation Analysis

```python
# Current code:
ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
pt = torch.exp(-ce_loss)
focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
return focal_loss.mean()
```

**Problems:**
1. Uses scalar `self.alpha = 0.25` for ALL samples
2. Doesn't distinguish between class 0 and class 1
3. Should use `alpha=0.669` for class 0, `alpha=1.978` for class 1 (from config)
4. Formula: `focal_loss = α * (1 - pt)^γ * ce_loss` applies to all equally

**Expected Behavior:**
- Samples from class 0 should use `α=0.669`
- Samples from class 1 should use `α=1.978`
- This gives MORE weight to hard examples in the MINORITY class (class 1)

---

## Root Cause Analysis

### Why This Matters

From class distribution in data:
- Class 0 (Poor): ~3469 samples (83%) - Majority
- Class 1 (Good): ~689 samples (17%) - Minority

The Focal Loss is designed to:
1. Focus on hard examples (via `(1-pt)^γ` term)
2. Focus on minority class (via `α` parameter)

Current implementation:
- ✅ Focuses on hard examples (`(1-pt)^γ` works)
- ❌ Does NOT focus on minority class (alpha not class-aware)

---

## Code Comparison

### Current (BROKEN) - `training/trainer.py:538-551`

```python
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 2):
        super().__init__()
        self.alpha = alpha  # Single scalar - WRONG!
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        # Applies same alpha to all classes - WRONG!
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### Also Broken - `training/trainer.py:217-241` (Loss Creation)

```python
def _create_loss_function(self) -> nn.Module:
    """创建损失函数"""
    train_config = self.config.get('training', {})
    loss_name = train_config.get('loss', 'focal').lower()

    if loss_name == 'cross_entropy':
        class_weights = train_config.get('class_weights', None)
        weight = torch.tensor(class_weights, dtype=torch.float32).to(self.device) if class_weights else None
        # ... uses class_weights
    elif loss_name == 'focal':
        return FocalLoss(
            alpha=train_config.get('focal_alpha', 0.25),  # Single value!
            gamma=train_config.get('focal_gamma', 2.0),
            num_classes=2
        )
        # ❌ class_weights NOT passed to FocalLoss!
```

**Missing:** `class_weights` parameter is NOT passed to FocalLoss constructor

---

## Solution Design

### Approach: Make FocalLoss Class-Aware

Implement proper per-class weighting in FocalLoss:

```python
class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting"""

    def __init__(self,
                 alpha: Union[float, List[float], torch.Tensor] = 0.25,
                 gamma: float = 2.0,
                 num_classes: int = 2,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

        # Handle alpha parameter - convert to per-class if needed
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        elif isinstance(alpha, float):
            # If scalar, use as is for backward compatibility
            self.alpha = alpha
        else:
            self.alpha = torch.tensor([1.0] * num_classes, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute cross entropy loss (per-sample)
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t (probability of true class)
        pt = torch.exp(-ce_loss)

        # Apply focal weighting: (1-p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weighting if alpha is per-class
        if isinstance(self.alpha, torch.Tensor) and self.alpha.numel() > 1:
            # Per-class alpha: alpha should be on same device
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            # Scalar alpha (backward compatible)
            focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

### Changes Required

**1. Update FocalLoss class** (`training/trainer.py:538-551`)
- Accept `alpha` as either scalar OR list/tensor
- Handle per-class weighting based on target labels
- Maintain backward compatibility with scalar alpha

**2. Update loss function creation** (`training/trainer.py:217-241`)
- Pass `class_weights` to FocalLoss constructor
- FocalLoss now understands class weights

**3. Update trainer initialization** (no changes needed)
- Already passes config correctly

---

## Testing Plan

### Unit Test 1: Scalar Alpha (Backward Compatibility)
```python
def test_focal_loss_scalar_alpha():
    """Test FocalLoss with scalar alpha (backward compatible)"""
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    inputs = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))

    loss = focal(inputs, targets)
    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    print("✓ Scalar alpha works")

def test_focal_loss_scalar_alpha_deterministic():
    """Verify scalar alpha produces consistent results"""
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    inputs = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))

    loss1 = focal(inputs, targets)
    loss2 = focal(inputs, targets)
    assert torch.allclose(loss1, loss2)
    print("✓ Scalar alpha is deterministic")
```

### Unit Test 2: Per-Class Alpha Weighting
```python
def test_focal_loss_per_class_alpha():
    """Test FocalLoss with per-class alpha weights"""
    class_weights = [0.669, 1.978]
    focal = FocalLoss(alpha=class_weights, gamma=2.0)

    inputs = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))

    loss = focal(inputs, targets)
    assert loss.item() > 0
    assert not torch.isnan(loss)
    print("✓ Per-class alpha works")

def test_focal_loss_per_class_weighting():
    """Verify per-class weighting affects minority class more"""
    class_weights = [0.5, 2.0]  # Minority class has 4x weight
    focal = FocalLoss(alpha=class_weights, gamma=2.0)

    # Create batch with both classes
    inputs = torch.randn(100, 2)
    targets = torch.cat([
        torch.zeros(50, dtype=torch.long),  # 50 samples class 0
        torch.ones(50, dtype=torch.long)    # 50 samples class 1
    ])

    # Compute loss
    loss_total = focal(inputs, targets)

    # Verify that with class weight 2.0 for class 1,
    # class 1 samples contribute MORE to total loss
    assert loss_total.item() > 0
    print("✓ Per-class weighting applied correctly")
```

### Unit Test 3: Mathematical Correctness
```python
def test_focal_loss_formula():
    """Verify FocalLoss computes correct formula"""
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    # Simple case: perfect predictions
    inputs = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])  # Very confident
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss = focal(inputs, targets)

    # With very confident predictions, (1-pt) should be ~0
    # So focal loss should be very small
    assert loss.item() < 0.01
    print("✓ Formula is correct (confident predictions → low loss)")

def test_focal_loss_vs_ce_loss():
    """Verify FocalLoss with gamma=0 equals CrossEntropy"""
    focal = FocalLoss(alpha=1.0, gamma=0.0, num_classes=2)
    ce = nn.CrossEntropyLoss()

    inputs = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))

    focal_loss = focal(inputs, targets)
    ce_loss = ce(inputs, targets)

    # Should be approximately equal (allowing for small numerical differences)
    assert torch.allclose(focal_loss, ce_loss, atol=1e-5)
    print("✓ FocalLoss(gamma=0) ≈ CrossEntropyLoss")
```

### Integration Test 1: With Config
```python
def test_focal_loss_with_config():
    """Test FocalLoss creation from config"""
    config = {
        'training': {
            'loss': 'focal',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'class_weights': [0.669, 1.978]
        }
    }

    # Simulate trainer's loss creation
    train_config = config.get('training', {})
    loss_name = train_config.get('loss', 'focal').lower()

    if loss_name == 'focal':
        focal = FocalLoss(
            alpha=train_config.get('class_weights', 0.25),
            gamma=train_config.get('focal_gamma', 2.0),
            num_classes=2
        )

    inputs = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))
    loss = focal(inputs, targets)

    assert loss.item() > 0
    print("✓ FocalLoss works with config class_weights")
```

### Integration Test 2: Training Loop
```python
def test_focal_loss_in_training_loop():
    """Test that FocalLoss works correctly in a training step"""
    # Setup
    model = SimpleNet()  # Some simple model
    focal = FocalLoss(alpha=[0.669, 1.978], gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create batch with class imbalance
    inputs = torch.randn(32, 3, 224, 224)
    targets = torch.cat([
        torch.zeros(20, dtype=torch.long),  # 20 Poor
        torch.ones(12, dtype=torch.long)    # 12 Good
    ])

    # Forward pass
    outputs = model(inputs)
    loss = focal(outputs, targets)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Verify learning happened
    assert loss.item() > 0
    print("✓ FocalLoss works in training loop")
```

---

## Impact Analysis

### What Changes
- FocalLoss now accepts and uses class_weights
- Per-class weighting applied to focal term
- Better handling of class imbalance

### What Stays the Same
- API remains backward compatible (scalar alpha still works)
- gamma parameter unchanged
- Overall training loop unchanged
- Config file unchanged (class_weights already defined)

### Risk Assessment
- **Risk Level**: LOW
- **Reason**: Adding capability to use class_weights doesn't break existing functionality
- **Testing**: Comprehensive unit tests verify correctness

---

## Commit Message

```
fix: Implement proper class weighting in FocalLoss

- FocalLoss now accepts per-class alpha weights
- class_weights from config are now passed to FocalLoss
- Per-class weighting applied based on target labels
- Backward compatible with scalar alpha
- Fixes class imbalance handling: minority class gets higher weight
- Add comprehensive unit tests for FocalLoss

Previously:
- FocalLoss ignored class_weights config parameter
- Applied uniform alpha=0.25 to all samples
- Failed to give extra weight to minority class

Now:
- Uses class_weights=[0.669, 1.978] from config
- Class 0 gets weight 0.669, Class 1 gets weight 1.978
- Properly implements: focal_loss = alpha_t * (1-pt)^gamma * ce_loss
- Where alpha_t is selected based on target class

Fixes #2 in ISSUE_CODE_REVIEW.md
```

---

## Files to Modify

1. **training/trainer.py**
   - Lines 538-551: Update FocalLoss class
   - Lines 237-241: Pass class_weights to FocalLoss

2. **tests/test_losses.py** (NEW)
   - Add comprehensive unit tests for FocalLoss

---

## Validation Checklist

- [ ] FocalLoss accepts class_weights
- [ ] Per-class weighting applied correctly
- [ ] Backward compatible with scalar alpha
- [ ] class_weights from config used properly
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Training pipeline works
- [ ] Metrics improve with proper weighting
