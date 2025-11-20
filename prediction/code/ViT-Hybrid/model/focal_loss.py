"""
Focal Loss Implementation for Class Imbalance
Author: Liao Zizhuo
Date: 2025-11-20

Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
https://arxiv.org/abs/1708.02002

Focal Loss: FL(p_t) = -(1 - p_t)^γ * log(p_t)

Where:
- p_t is the model's predicted probability for the true class
- γ (gamma) is the focusing parameter (default: 2.0)
- The (1 - p_t)^γ term scales down easy examples (well-classified)
- It scales up hard examples (misclassified)

This makes the loss focus on hard, misclassified examples,
which is especially important for imbalanced datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    Particularly effective when:
    - Dataset has severe class imbalance (like 5:1)
    - Easy examples dominate the loss
    - Model needs to focus on hard/misclassified examples

    Args:
        alpha: Weight for class 0 (default: 0.25)
               High alpha = more weight on class 0
               Use alpha=0.25 when class 1 is minority (5:1 imbalance)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
               gamma=0: becomes CrossEntropyLoss
               gamma=2: standard for object detection
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Calculate Focal Loss

        Args:
            logits: (batch_size, num_classes) raw model outputs
            targets: (batch_size,) class labels (0 or 1)

        Returns:
            loss: scalar focal loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # (batch_size, 2)

        # Get log probabilities (more numerically stable)
        log_probs = F.log_softmax(logits, dim=1)  # (batch_size, 2)

        # Get probability for true class
        # For each sample, get the probability of its true class
        batch_size = targets.size(0)
        p_t = probs[torch.arange(batch_size), targets]  # (batch_size,)

        # Get log probability for true class
        log_p_t = log_probs[torch.arange(batch_size), targets]  # (batch_size,)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma  # (batch_size,)

        # Calculate base cross entropy loss: -log(p_t)
        ce_loss = -log_p_t  # (batch_size,)

        # Combine: focal loss = -(1 - p_t)^gamma * log(p_t)
        focal_loss = focal_weight * ce_loss  # (batch_size,)

        # Apply class weights (alpha for class 0, 1-alpha for class 1)
        # This further emphasizes minority class (class 1)
        class_weights = torch.where(
            targets == 0,
            torch.full_like(targets, self.alpha, dtype=torch.float32),
            torch.full_like(targets, 1 - self.alpha, dtype=torch.float32)
        )  # (batch_size,)

        focal_loss = focal_loss * class_weights

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted CrossEntropyLoss for comparison

    FL(p_t) = -α_t * log(p_t)
    where α_t is the weight for class t

    Args:
        weight: Tensor of shape (num_classes,) with class weights
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Calculate Weighted CrossEntropyLoss

        Args:
            logits: (batch_size, num_classes) raw model outputs
            targets: (batch_size,) class labels (0 or 1)

        Returns:
            loss: scalar loss value
        """
        return F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction=self.reduction
        )


# Comparison: Focal Loss vs Weighted CrossEntropyLoss
"""
Example with 5:1 imbalance (5 Poor : 1 Good):

Class 0 (Poor): 5000 samples, weight = 5000/(2*5000) = 0.5
Class 1 (Good):  1000 samples, weight = 5000/(2*1000) = 2.5

Scenario: Model incorrectly classifies a Good sample as Poor
True label: 1 (Good)
Predicted logit: [0.8, -0.2] → probs = [0.73, 0.27]
p_t = 0.27 (probability of true class)

Weighted CrossEntropyLoss (α=2.5):
    loss = -2.5 * log(0.27) ≈ 2.5 * 1.31 ≈ 3.275

Focal Loss (α=0.25, γ=2):
    focal_weight = (1 - 0.27)^2 = 0.73^2 ≈ 0.533
    base_loss = -log(0.27) ≈ 1.31
    class_weight = 1 - 0.25 = 0.75 (for class 1)
    focal_loss = 0.533 * 1.31 * 0.75 ≈ 0.526

Focal Loss is LOWER for easy examples (already classified correctly):
True label: 1 (Good)
Predicted: [0.1, 0.9] → p_t = 0.9
Weighted CE: -2.5 * log(0.9) ≈ 0.265
Focal Loss: (1-0.9)^2 * (-log(0.9)) * 0.75 = 0.01 * 0.105 * 0.75 ≈ 0.0008

Key insight:
- Focal Loss DOWN-weights easy examples
- Focal Loss UP-weights hard examples
- This makes training focus on hard misclassified samples
- Especially beneficial for imbalanced datasets
"""
