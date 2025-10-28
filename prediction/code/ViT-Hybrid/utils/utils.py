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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


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
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif self._is_improved(score):
            self.best_score = score
            self.best_model_state = model.state_dict()
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
        PathConfig.RESULT_DIR
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
