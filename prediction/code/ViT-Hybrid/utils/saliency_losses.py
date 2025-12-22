"""
Saliency Loss Functions from TMFI-Net
完全复制自TMFI-Net的loss.py，用于saliency prediction任务

用于多任务学习：
- 主任务：分类 (CrossEntropyLoss)
- 辅助任务：Saliency prediction (TMFI-Net losses)

Author: Original TMFI-Net
Date: 2025-12-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


def normalize_map(s_map):
    """
    Normalize the saliency/attention map

    Args:
        s_map: (B, H, W) saliency map

    Returns:
        Normalized map in [0, 1]
    """
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map + 1e-10)
    return norm_s_map


def kldiv_loss(pred_map, gt_map):
    """
    KL Divergence loss between predicted and ground truth saliency maps

    Args:
        pred_map: (B, H, W) predicted saliency/attention map
        gt_map: (B, H, W) ground truth saliency map

    Returns:
        KL divergence loss (scalar)
    """
    assert pred_map.size() == gt_map.size()
    batch_size = pred_map.size(0)
    w = pred_map.size(1)
    h = pred_map.size(2)

    # Normalize to probability distributions
    sum_pred = torch.sum(pred_map.view(batch_size, -1), 1)
    expand_pred = sum_pred.view(batch_size, 1, 1).expand(batch_size, w, h)
    pred_map = pred_map / (expand_pred + 1e-10)

    sum_gt = torch.sum(gt_map.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    gt_map = gt_map / (expand_gt + 1e-10)

    pred_map = pred_map.view(batch_size, -1)
    gt_map = gt_map.view(batch_size, -1)

    eps = 1e-10
    result = gt_map * torch.log(eps + gt_map / (pred_map + eps))
    return torch.mean(torch.sum(result, 1))


def cc_loss(pred_map, gt_map):
    """
    Correlation Coefficient loss (to maximize correlation)

    Args:
        pred_map: (B, H, W) predicted saliency/attention map
        gt_map: (B, H, W) ground truth saliency map

    Returns:
        Negative correlation coefficient (to minimize)
    """
    assert pred_map.size() == gt_map.size()
    batch_size = pred_map.size(0)
    w = pred_map.size(1)
    h = pred_map.size(2)

    mean_pred = torch.mean(pred_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_pred = torch.std(pred_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    pred_map_norm = (pred_map - mean_pred) / (std_pred + 1e-10)
    gt_map_norm = (gt_map - mean_gt) / (std_gt + 1e-10)

    ab = torch.sum((pred_map_norm * gt_map_norm).view(batch_size, -1), 1)
    aa = torch.sum((pred_map_norm * pred_map_norm).view(batch_size, -1), 1)
    bb = torch.sum((gt_map_norm * gt_map_norm).view(batch_size, -1), 1)

    cc = ab / (torch.sqrt(aa * bb) + 1e-10)

    # Return negative (to minimize)
    return -torch.mean(cc)


def similarity_loss(pred_map, gt_map):
    """
    Similarity loss between predicted and ground truth saliency maps

    Args:
        pred_map: (B, H, W) predicted saliency/attention map
        gt_map: (B, H, W) ground truth saliency map

    Returns:
        Negative similarity (to minimize)
    """
    batch_size = pred_map.size(0)
    w = pred_map.size(1)
    h = pred_map.size(2)

    pred_map = normalize_map(pred_map)
    gt_map = normalize_map(gt_map)

    # Normalize to probability distributions
    sum_pred = torch.sum(pred_map.view(batch_size, -1), 1)
    expand_pred = sum_pred.view(batch_size, 1, 1).expand(batch_size, w, h)
    pred_map = pred_map / (expand_pred + 1e-10)

    sum_gt = torch.sum(gt_map.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    gt_map = gt_map / (expand_gt + 1e-10)

    pred_map = pred_map.view(batch_size, -1)
    gt_map = gt_map.view(batch_size, -1)

    sim = torch.sum(torch.min(pred_map, gt_map), 1)

    # Return negative (to minimize)
    return -torch.mean(sim)


class CombinedSaliencyLoss(nn.Module):
    """
    Combined saliency loss for training attention maps

    This is optional and can be added to classification loss for better attention maps.

    Args:
        use_kldiv: Use KL divergence loss
        use_cc: Use correlation coefficient loss
        use_sim: Use similarity loss
        use_l1: Use L1 loss
        kldiv_weight: Weight for KL divergence
        cc_weight: Weight for CC
        sim_weight: Weight for similarity
        l1_weight: Weight for L1
    """
    def __init__(self,
                 use_kldiv=True,
                 use_cc=True,
                 use_sim=False,
                 use_l1=False,
                 kldiv_weight=1.0,
                 cc_weight=1.0,
                 sim_weight=1.0,
                 l1_weight=1.0):
        super(CombinedSaliencyLoss, self).__init__()

        self.use_kldiv = use_kldiv
        self.use_cc = use_cc
        self.use_sim = use_sim
        self.use_l1 = use_l1

        self.kldiv_weight = kldiv_weight
        self.cc_weight = cc_weight
        self.sim_weight = sim_weight
        self.l1_weight = l1_weight

        if use_l1:
            self.l1_criterion = nn.L1Loss()

    def forward(self, pred_map, gt_map):
        """
        Compute combined saliency loss

        Args:
            pred_map: (B, 1, H, W) or (B, H, W) predicted attention map
            gt_map: (B, 1, H, W) or (B, H, W) ground truth saliency map

        Returns:
            Combined loss (scalar)
        """
        # Squeeze if 4D
        if len(pred_map.size()) == 4:
            pred_map = pred_map.squeeze(1)
        if len(gt_map.size()) == 4:
            gt_map = gt_map.squeeze(1)

        assert len(pred_map.size()) == 3, "pred_map should be (B, H, W)"
        assert len(gt_map.size()) == 3, "gt_map should be (B, H, W)"

        loss = torch.FloatTensor([0.0]).to(pred_map.device)

        if self.use_kldiv:
            loss += self.kldiv_weight * kldiv_loss(pred_map, gt_map)

        if self.use_cc:
            loss += self.cc_weight * cc_loss(pred_map, gt_map)

        if self.use_sim:
            loss += self.sim_weight * similarity_loss(pred_map, gt_map)

        if self.use_l1:
            loss += self.l1_weight * self.l1_criterion(pred_map, gt_map)

        return loss


def get_tmfi_loss(pred_map, gt_map, use_kldiv=True, use_cc=True, use_sim=False, use_l1=False,
                  kldiv_coeff=1.0, cc_coeff=-1.0, sim_coeff=-1.0, l1_coeff=1.0):
    """
    TMFI-Net原始loss计算函数
    完全复制自TMFI-Net utils1.py的get_loss函数

    Args:
        pred_map: (B, H, W) predicted saliency map
        gt_map: (B, H, W) ground truth saliency map
        use_kldiv: Use KL divergence loss
        use_cc: Use correlation coefficient loss
        use_sim: Use similarity loss
        use_l1: Use L1 loss
        kldiv_coeff, cc_coeff, sim_coeff, l1_coeff: Loss coefficients

    Returns:
        Combined loss (scalar)
    """
    loss = torch.FloatTensor([0.0]).to(pred_map.device)
    criterion = nn.L1Loss()

    if use_kldiv:
        loss += kldiv_coeff * kldiv_loss(pred_map, gt_map)
    if use_cc:
        loss += cc_coeff * cc_loss(pred_map, gt_map)
    if use_l1:
        loss += l1_coeff * criterion(pred_map, gt_map)
    if use_sim:
        loss += sim_coeff * similarity_loss(pred_map, gt_map)

    return loss
