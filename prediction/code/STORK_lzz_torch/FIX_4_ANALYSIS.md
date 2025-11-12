# Fix #4: Validation Split Strategy - Stratification Verification

**Status**: ANALYSIS COMPLETE
**Date**: 2025-11-12
**Priority**: MEDIUM

---

## Problem Overview

### Current State: Stratification Already In Use ✅

The STORK codebase **ALREADY implements stratified splitting** in:

| Location | Method | Status |
|----------|--------|--------|
| `data/dataset.py:254-258` | train_test_split with `stratify=labels` | ✅ Correct |
| `data/dataset.py:270-275` | train_test_split with `stratify=train_labels` | ✅ Correct |
| `data/parallel_processing.py:182-185` | train_test_split with `stratify=labels` | ✅ Correct |
| `data/parallel_processing.py:433-436` | train_test_split with `stratify=labels` | ✅ Correct |
| `evaluation/cross_validation.py:80-90` | StratifiedKFold when `stratified=True` | ✅ Correct |

### What's Missing: Comprehensive Logging & Verification

While stratification is **correctly implemented**, there are areas for improvement:

1. **No Class Distribution Verification**: Code doesn't verify the stratification worked
2. **Missing Fold-Level Logging**: Each fold's class distribution not logged in cross-validation
3. **Inconsistent Logging Detail**: Some splits log class distribution, others don't
4. **No Unit Tests for Stratification**: No automated tests verify splits are stratified

---

## Detailed Analysis

### Current Implementation Review

#### ✅ **Binary Data Split** (data/dataset.py:254-258)

```python
train_paths, test_paths, train_labels, test_labels, ... = train_test_split(
    self.image_paths, self.labels, self.qualities,
    test_size=test_size,
    stratify=self.labels,  # ✅ Uses stratify
    random_state=42
)
```

**Status**: Good - stratified, has random_state
**Missing**: No verification logging

#### ✅ **Train/Val Split** (data/dataset.py:270-275)

```python
train_paths, val_paths, train_labels, val_labels, ... = train_test_split(
    train_paths, train_labels, train_qualities,
    test_size=val_split,
    stratify=train_labels,  # ✅ Uses stratify
    random_state=42
)
```

**Status**: Good - stratified
**Missing**: No logging of resulting class distribution

#### ✅ **Cross-Validation** (evaluation/cross_validation.py:79-90)

```python
if self.stratified:  # Config default: True
    kfold = StratifiedKFold(
        n_splits=self.n_folds,
        shuffle=self.shuffle,
        random_state=self.random_state
    )
else:
    kfold = KFold(...)  # Non-stratified fallback
```

**Status**: Good - StratifiedKFold used when configured
**Missing**: No per-fold class distribution logging

---

## Improvements Needed

### 1. Add Class Distribution Verification Logging

**For all train_test_split calls**, add verification:

```python
def _verify_stratification(original_labels, train_labels, test_labels, split_name=""):
    """Verify class distribution maintained after stratified split"""
    orig_dist = np.bincount(original_labels)
    train_dist = np.bincount(train_labels)
    test_dist = np.bincount(test_labels)

    print(f"\n{split_name} 分层验证:")
    print(f"  原始分布: {orig_dist} (比例: {orig_dist/len(original_labels)})")
    print(f"  训练集:  {train_dist} (比例: {train_dist/len(train_labels)})")
    print(f"  验证集:  {test_dist} (比例: {test_dist/len(test_labels)})")
```

### 2. Enhanced Cross-Validation Fold Logging

Add per-fold class distribution tracking:

```python
def _log_fold_distribution(fold_idx, train_idx, val_idx, all_labels):
    """Log class distribution for each fold"""
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]

    train_counts = np.bincount(train_labels)
    val_counts = np.bincount(val_labels)

    print(f"Fold {fold_idx}: 训练集 {train_counts} | 验证集 {val_counts}")
```

### 3. Add Unit Tests for Stratification

Create comprehensive tests:

```python
def test_stratified_split_maintains_distribution():
    """Verify stratified splits maintain class distribution"""
    labels = np.array([0]*500 + [1]*100)  # 5:1 imbalance

    train_idx, test_idx = next(StratifiedKFold(n_splits=5).split(X, labels))
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # Class distribution ratios should be similar
    train_ratio = np.sum(train_labels == 1) / len(train_labels)
    test_ratio = np.sum(test_labels == 1) / len(test_labels)
    orig_ratio = np.sum(labels == 1) / len(labels)

    # Allow 5% deviation
    assert abs(train_ratio - orig_ratio) < 0.05
    assert abs(test_ratio - orig_ratio) < 0.05
```

---

## Implementation Plan

### Step 1: Add Verification Helper Method

Create utility function in `data/parallel_processing.py`:

```python
def verify_class_distribution(labels_list: List[np.ndarray],
                            split_names: List[str]) -> None:
    """Verify class distribution across splits"""
    print(f"\n{'='*60}")
    print(f"Class Distribution Verification")
    print(f"{'='*60}")

    for labels, name in zip(labels_list, split_names):
        counts = np.bincount(labels)
        total = len(labels)
        ratios = counts / total
        print(f"{name:20s}: {counts} ({ratios*100:.1f}%)")
```

### Step 2: Add Logging to Data Loading

Update `_collect_image_paths()` in `data/parallel_processing.py`:

```python
# After first split (train/test)
verify_class_distribution(
    [labels, train_labels, test_labels],
    ["Original", "Train+Val", "Test"]
)

# After second split (train/val)
verify_class_distribution(
    [train_labels, val_labels],
    ["Train", "Validation"]
)
```

### Step 3: Enhance Cross-Validation Logging

Update `_run_single_fold()` in `evaluation/cross_validation.py`:

```python
# After creating train/val subsets
train_fold_labels = np.array([dataset[i][1] for i in train_idx])
val_fold_labels = np.array([dataset[i][1] for i in val_idx])

print(f"   Fold {fold + 1} 类别分布:")
print(f"     训练集: Class 0: {sum(train_fold_labels == 0)}, Class 1: {sum(train_fold_labels == 1)}")
print(f"     验证集: Class 0: {sum(val_fold_labels == 0)}, Class 1: {sum(val_fold_labels == 1)}")
```

### Step 4: Add Comprehensive Unit Tests

Create `tests/test_stratification.py`:

```python
def test_stratified_split_preserves_ratio():
    """Test that stratified splits preserve class ratio"""
    # Setup: 3469 Poor (83%) + 689 Good (17%)
    labels = np.array([0]*3469 + [1]*689)

    # Split: 70/30
    train_idx, test_idx = next(
        StratifiedKFold(n_splits=2).split(range(len(labels)), labels)
    )

    # Verify ratio preserved
    orig_ratio = np.mean(labels == 1)
    train_ratio = np.mean(labels[train_idx] == 1)
    test_ratio = np.mean(labels[test_idx] == 1)

    assert np.abs(train_ratio - orig_ratio) < 0.02
    assert np.abs(test_ratio - orig_ratio) < 0.02

def test_cross_validation_fold_stratification():
    """Test that cross-validation folds are stratified"""
    labels = np.array([0]*3469 + [1]*689)
    skf = StratifiedKFold(n_splits=5)

    ratios = []
    for train_idx, test_idx in skf.split(range(len(labels)), labels):
        fold_ratio = np.mean(labels[train_idx] == 1)
        ratios.append(fold_ratio)

    # All folds should have similar ratio (within 5%)
    mean_ratio = np.mean(ratios)
    for ratio in ratios:
        assert np.abs(ratio - mean_ratio) < 0.05
```

---

## Files to Modify

### 1. data/parallel_processing.py
- Add `verify_class_distribution()` helper function
- Add logging after each train_test_split
- Document stratification approach

### 2. evaluation/cross_validation.py
- Add per-fold class distribution logging in `_run_single_fold()`
- Verify stratification worked correctly
- Log fold statistics

### 3. tests/test_stratification.py (NEW)
- Add comprehensive stratification tests
- Verify ratio preservation
- Test edge cases

---

## Expected Outcomes

### Code Quality Improvements

1. **Transparency**: Users can see class distribution at each stage
2. **Verification**: Automatic checks that stratification worked
3. **Debugging**: Better logs if issues arise with data splitting
4. **Testing**: Automated verification of stratification correctness

### Logging Output Example

```
============================================================
Class Distribution Verification
============================================================
Original             : [3469  689] (83.4%, 16.6%)
Train+Val            : [2951  586] (83.4%, 16.6%)
Test                 : [ 518  103] (83.4%, 16.6%)

Fold 1 类别分布:
  训练集: Class 0: 2967, Class 1: 585
  验证集: Class 0: 522, Class 1: 105

Fold 2 类别分布:
  训练集: Class 0: 2951, Class 1: 587
  验证集: Class 0: 523, Class 1: 104

...
```

---

## Impact Analysis

### What Changes
- Added comprehensive logging of class distributions
- Added verification that stratification worked
- New unit tests for stratification

### What Stays the Same
- Data splitting methodology (already correct)
- Cross-validation approach (already correct)
- Model training (unchanged)

### Risk Assessment
- **Risk Level**: VERY LOW
- **Reason**: Only adding logging/verification, not changing logic
- **Breaking Changes**: None
- **Backward Compatibility**: Fully compatible

---

## Validation Checklist

- [ ] Class distribution logged at each split stage
- [ ] Cross-validation fold distributions logged
- [ ] All stratification unit tests pass
- [ ] No splitting logic changes (only logging)
- [ ] Verification functions work correctly
- [ ] Integration tests pass

---

## Status

✅ Analysis complete
⏳ Implementation ready (pending user approval)

