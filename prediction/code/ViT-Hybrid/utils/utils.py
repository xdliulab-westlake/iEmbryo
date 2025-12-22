"""
Utility Functions for ViT-Hybrid Model (PyTorch)
Author: Liao Zizhuo
Date: 2025-10-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix


def _remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys (for DataParallel models)"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value  # Remove 'module.' prefix
        else:
            new_state_dict[key] = value
    return new_state_dict


def _add_module_prefix(state_dict):
    """Add 'module.' prefix to state_dict keys (for DataParallel models)"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith('module.'):
            new_state_dict['module.' + key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, mode='max', delta=0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        import copy
        # Get the unwrapped model (remove DataParallel wrapper if present)
        base_model = model.module if hasattr(model, 'module') else model
        # Deep copy the entire state_dict to avoid shared references
        state_dict = copy.deepcopy(base_model.state_dict())

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = state_dict
        elif self._is_improved(score):
            self.best_score = score
            self.best_model_state = state_dict
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improved(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.delta
        else:
            return score < self.best_score - self.delta


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    计算评估指标

    Args:
        y_true: 真实标签 (numpy array)
        y_pred: 预测概率 (numpy array)
        threshold: 分类阈值

    Returns:
        metrics字典
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary', zero_division=0
    )

    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    accuracy = (y_true == y_pred_binary).mean()

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc)
    }

    return metrics


def save_checkpoint(state, save_path):
    """保存模型检查点"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(load_path, model, optimizer=None):
    """加载模型检查点"""
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0)


def save_history(history, save_path):
    """保存训练历史"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)


def load_history(load_path):
    """加载训练历史"""
    with open(load_path, 'r') as f:
        history = json.load(f)
    return history


def create_directories():
    """创建必要的目录"""
    from config.config import PathConfig

    dirs = [
        PathConfig.MODEL_SAVE_DIR,
        PathConfig.LOG_DIR,
        PathConfig.RESULT_DIR,
        PathConfig.CACHE_DIR
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def print_confusion_matrix(y_true, y_pred, threshold=0.5):
    """
    Print confusion matrix and related metrics

    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted probabilities (numpy array)
        threshold: Classification threshold (default: 0.5)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)

    # Extract confusion matrix components
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases where only one class is present
        if cm.shape == (1, 1):
            tn = cm[0, 0]
            fp = fn = tp = 0
        else:
            tn = fp = fn = tp = 0

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Print confusion matrix
    print("\n  Confusion Matrix:")
    print("                 Predicted")
    print("                Poor    Good")
    print(f"  Actual Poor   {tn:5d}   {fp:5d}")
    print(f"         Good   {fn:5d}   {tp:5d}")
    print(f"\n  Sensitivity (Recall):  {sensitivity:.4f}")
    print(f"  Specificity:           {specificity:.4f}")
    print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")


def print_dataset_distribution(train_loader, val_loader, test_loader):
    """
    Print sample distribution across train/val/test sets

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    def count_samples(loader):
        """Count samples and their labels in a data loader"""
        good_count = 0
        poor_count = 0

        for _, labels in loader:
            labels_np = labels.cpu().numpy()
            good_count += (labels_np == 1).sum()
            poor_count += (labels_np == 0).sum()

        return good_count, poor_count

    train_good, train_poor = count_samples(train_loader)
    val_good, val_poor = count_samples(val_loader)
    test_good, test_poor = count_samples(test_loader)

    print("\n" + "="*70)
    print("DATASET DISTRIBUTION")
    print("="*70)

    # Calculate percentages
    train_total = train_good + train_poor
    val_total = val_good + val_poor
    test_total = test_good + test_poor

    train_good_pct = 100 * train_good / train_total if train_total > 0 else 0
    val_good_pct = 100 * val_good / val_total if val_total > 0 else 0
    test_good_pct = 100 * test_good / test_total if test_total > 0 else 0

    print(f"\nTRAIN SET ({train_total} samples):")
    print(f"  Good: {train_good:4d} ({train_good_pct:6.2f}%) | Poor: {train_poor:4d} ({100-train_good_pct:6.2f}%)")

    print(f"\nVALIDATION SET ({val_total} samples):")
    print(f"  Good: {val_good:4d} ({val_good_pct:6.2f}%) | Poor: {val_poor:4d} ({100-val_good_pct:6.2f}%)")

    print(f"\nTEST SET ({test_total} samples):")
    print(f"  Good: {test_good:4d} ({test_good_pct:6.2f}%) | Poor: {test_poor:4d} ({100-test_good_pct:6.2f}%)")

    print("="*70)


class AverageMeter:
    """
    Computes and stores the average and current value
    Copied from TMFI-Net utils1.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
