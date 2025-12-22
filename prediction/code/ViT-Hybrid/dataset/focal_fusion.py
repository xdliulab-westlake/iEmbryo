"""
Focal Plane Fusion for TMFI-Net

Converts multi-focal embryo data (T, F, C, H, W) to Video Swin Transformer format (C, T, H, W)

Author: Liao Zizhuo
Date: 2025-12-22
"""

import torch
import torch.nn as nn


class FocalPlaneFusion(nn.Module):
    """
    Fuses multiple focal planes into single frame per timepoint

    Supported fusion strategies:
    - average: Average across focal planes (default)
    - max: Max pooling across focal planes
    - middle: Select middle focal plane
    - learned: Learnable weighted fusion
    """

    def __init__(self, fusion_type='average', num_focal_planes=11):
        super(FocalPlaneFusion, self).__init__()

        self.fusion_type = fusion_type
        self.num_focal_planes = num_focal_planes

        if fusion_type == 'learned':
            # Learnable weights for each focal plane
            self.focal_weights = nn.Parameter(torch.ones(num_focal_planes) / num_focal_planes)

    def forward(self, x):
        """
        Fuse focal planes

        Args:
            x: (B, T, F, C, H, W) - Multi-focal time series

        Returns:
            (B, C, T, H, W) - Fused video tensor for Video Swin Transformer
        """
        B, T, F, C, H, W = x.shape

        if self.fusion_type == 'average':
            # Average across focal planes: (B, T, F, C, H, W) -> (B, T, C, H, W)
            x_fused = x.mean(dim=2)

        elif self.fusion_type == 'max':
            # Max pooling across focal planes
            x_fused = x.max(dim=2)[0]

        elif self.fusion_type == 'middle':
            # Select middle focal plane (F=5 for 11 planes)
            middle_idx = F // 2
            x_fused = x[:, :, middle_idx, :, :, :]

        elif self.fusion_type == 'learned':
            # Weighted sum with learnable weights
            # Normalize weights
            weights = torch.softmax(self.focal_weights, dim=0)  # (F,)

            # Reshape for broadcasting: (1, 1, F, 1, 1, 1)
            weights = weights.view(1, 1, F, 1, 1, 1)

            # Weighted sum: (B, T, F, C, H, W) * (1, 1, F, 1, 1, 1) -> (B, T, F, C, H, W)
            x_weighted = x * weights
            x_fused = x_weighted.sum(dim=2)  # (B, T, C, H, W)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Transpose to Video Swin Transformer format: (B, T, C, H, W) -> (B, C, T, H, W)
        x_fused = x_fused.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)

        return x_fused


def create_focal_fusion_dataloader_wrapper(base_dataloader, fusion_type='average'):
    """
    Create a wrapper around DataLoader that automatically fuses focal planes

    Args:
        base_dataloader: DataLoader returning (images, labels) where images is (B, T, F, C, H, W)
        fusion_type: Fusion strategy ('average', 'max', 'middle', 'learned')

    Returns:
        Iterator yielding (fused_images, labels) where fused_images is (B, C, T, H, W)
    """
    fusion_module = FocalPlaneFusion(fusion_type=fusion_type)

    class FusedDataLoaderIterator:
        def __init__(self, dataloader, fusion_module):
            self.dataloader = dataloader
            self.fusion_module = fusion_module
            self.iterator = None

        def __iter__(self):
            self.iterator = iter(self.dataloader)
            return self

        def __next__(self):
            images, labels = next(self.iterator)
            # Fuse focal planes
            fused_images = self.fusion_module(images)
            return fused_images, labels

        def __len__(self):
            return len(self.dataloader)

    return FusedDataLoaderIterator(base_dataloader, fusion_module)


class FocalFusionWrapper(nn.Module):
    """
    Wrapper that combines FocalPlaneFusion + TMFI-Embryo Model

    This module handles the full pipeline:
    1. Fuse focal planes: (B, T, F, C, H, W) -> (B, C, T, H, W)
    2. Run TMFI-Net: (B, C, T, H, W) -> (logits, attention_map)
    """

    def __init__(self, tmfi_model, fusion_type='average', num_focal_planes=11):
        super(FocalFusionWrapper, self).__init__()

        self.focal_fusion = FocalPlaneFusion(fusion_type=fusion_type, num_focal_planes=num_focal_planes)
        self.tmfi_model = tmfi_model

    def forward(self, x):
        """
        Forward pass with focal plane fusion

        Args:
            x: (B, T, F, C, H, W) - Multi-focal time series

        Returns:
            saliency_map: (B, H, W) - TMFI-Net saliency output
            logits: (B, num_classes) - Classification output
        """
        # Step 1: Fuse focal planes
        x_fused = self.focal_fusion(x)  # (B, C, T, H, W)

        # Step 2: Run Multi-Task TMFI-Net
        saliency_map, logits = self.tmfi_model(x_fused)

        return saliency_map, logits
