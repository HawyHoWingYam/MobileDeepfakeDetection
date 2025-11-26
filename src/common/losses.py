#!/usr/bin/env python3
"""
Shared Loss Functions
=====================

Common loss implementations used across Stage 2 (EfficientNet) and
optional GenConViT experts. This module is deliberately decoupled
from any specific architecture so that it can be reused without
pulling in GenConViT dependencies.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Standalone classification loss with optional focal / balanced variants."""

    def __init__(
        self,
        loss_type: str = "bce",
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            loss_type: Type of classification loss ('bce', 'focal', 'balanced')
            pos_weight: Positive class weight for imbalanced datasets
            label_smoothing: Label smoothing factor in [0, 1)
        """
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

        if loss_type == "bce":
            pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight else None
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        elif loss_type == "focal":
            self.loss_fn = self._focal_loss
            self.alpha = pos_weight or 1.0
            self.gamma = 2.0
        elif loss_type == "balanced":
            self.loss_fn = self._balanced_loss
            self.pos_weight = pos_weight or 1.0
        else:
            raise ValueError(f"Unknown classification loss type: {loss_type}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute classification loss."""
        if self.label_smoothing > 0:
            targets = self._apply_label_smoothing(targets)
        return self.loss_fn(logits, targets)

    def _apply_label_smoothing(self, targets: torch.Tensor) -> torch.Tensor:
        """Apply simple label smoothing around 0.5."""
        return targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    def _balanced_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Balanced loss with dynamic weighting between positive/negative examples."""
        pos_mask = targets == 1
        neg_mask = targets == 0

        pos_loss = (
            F.binary_cross_entropy_with_logits(
                logits[pos_mask], targets[pos_mask], reduction="mean"
            )
            if pos_mask.any()
            else 0
        )

        neg_loss = (
            F.binary_cross_entropy_with_logits(
                logits[neg_mask], targets[neg_mask], reduction="mean"
            )
            if neg_mask.any()
            else 0
        )

        return self.pos_weight * pos_loss + neg_loss

