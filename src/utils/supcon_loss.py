"""
AWARE-NET: Advanced Supervised Contrastive Loss Implementation (Research Module)

⚠️  ADVANCED RESEARCH MODULE - NOT NEEDED FOR STAGE 01:
This module implements supervised contrastive learning for research purposes.
For basic Stage 01 requirements (simple BCE classification), this is NOT needed.

Research Purpose:
- Implements authenticity modeling paradigm shift
- Creates "truthfulness fortress" in feature space
- Advanced research beyond standard fake detection

Mathematical Foundation (Research Level):
SupCon Loss = -1/N Σᵢ [1/|P(i)| Σₚ∈P(i) log(exp(zᵢ·zₚ/τ) / Σₐ∈A(i) exp(zᵢ·zₐ/τ))]

Where:
- zᵢ, zₚ, zₐ are L2-normalized feature vectors
- P(i) is the set of all samples with the same label as sample i
- A(i) is the set of all samples except sample i
- τ is the temperature parameter

Stage 01 Alternative:
For simple binary classification, use standard BCE loss instead (see train_mobile_simple.py)

Reference: Khosla et al. "Supervised Contrastive Learning" NeurIPS 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss implementation with numerical stability.

    This loss function creates a "truthfulness fortress" in the feature space
    by pulling together samples of the same class and pushing apart samples
    of different classes.

    Features:
    - Numerical stability to prevent overflow/underflow
    - Support for both "one" and "all" contrast modes
    - Temperature scaling for calibrated similarities
    - Comprehensive logging for debugging
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: Literal['one', 'all'] = 'all',
        base_temperature: float = 0.07,
        numerical_stability: bool = True,
        eps: float = 1e-8,
        max_exp: float = 88.0
    ):
        """
        Initialize SupCon loss function.

        Args:
            temperature: Temperature parameter τ for scaling similarities
            contrast_mode: 'one' - contrast against one positive, 'all' - against all positives
            base_temperature: Base temperature for normalization
            numerical_stability: Enable numerical stability measures
            eps: Small constant to prevent division by zero
            max_exp: Maximum value for exponential to prevent overflow
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.numerical_stability = numerical_stability
        self.eps = eps
        self.max_exp = max_exp

        logger.info(f"SupConLoss initialized: temp={temperature}, mode={contrast_mode}")

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SupCon loss.

        Args:
            features: Feature vectors of shape (batch_size, feature_dim) or
                     (batch_size, n_views, feature_dim)
            labels: Ground truth labels of shape (batch_size,)
            mask: Contrastive mask of shape (batch_size, batch_size)
                 If None, uses labels to create mask

        Returns:
            Scalar loss tensor
        """
        device = features.device

        # Handle multi-view features
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        if len(features.shape) > 3:
            raise ValueError('features can have at most 3 dimensions')

        batch_size = features.shape[0]
        n_views = features.shape[1]

        # Validate inputs
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and mask')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match batch size')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Reshape features: (batch_size * n_views, feature_dim)
        contrast_count = n_views
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_features = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_features = contrast_features
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown contrast_mode: {}'.format(self.contrast_mode))

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_features, contrast_features.T),
            self.temperature
        )

        # Apply numerical stability
        if self.numerical_stability:
            # Clamp to prevent overflow
            anchor_dot_contrast = torch.clamp(anchor_dot_contrast, max=self.max_exp)

            # For numerical stability, subtract max
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
        else:
            logits = anchor_dot_contrast

        # Tile mask for multi-view scenario
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask

        # Add small epsilon for numerical stability
        if self.numerical_stability:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        # Log statistics for debugging
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Invalid loss detected: {loss}")
            logger.error(f"Features stats: mean={features.mean()}, std={features.std()}")
            logger.error(f"Logits stats: mean={logits.mean()}, std={logits.std()}")

        return loss

    def get_feature_similarities(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Compute feature space statistics for analysis.

        Args:
            features: L2-normalized feature vectors (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)

        Returns:
            Dictionary containing similarity statistics
        """
        device = features.device

        # Ensure features are L2 normalized
        features = F.normalize(features, dim=1)

        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute pairwise similarities
        similarities = torch.matmul(features, features.T)

        # Separate intra-class and inter-class similarities
        mask_diag = torch.eye(mask.shape[0], device=device)
        intra_mask = mask * (1 - mask_diag)  # Same class, different samples
        inter_mask = (1 - mask) * (1 - mask_diag)  # Different class, different samples

        # Compute statistics
        intra_similarities = similarities[intra_mask.bool()]
        inter_similarities = similarities[inter_mask.bool()]

        stats = {
            'intra_class_mean': intra_similarities.mean().item() if len(intra_similarities) > 0 else 0.0,
            'intra_class_std': intra_similarities.std().item() if len(intra_similarities) > 0 else 0.0,
            'inter_class_mean': inter_similarities.mean().item() if len(inter_similarities) > 0 else 0.0,
            'inter_class_std': inter_similarities.std().item() if len(inter_similarities) > 0 else 0.0,
            'separation_margin': (intra_similarities.mean() - inter_similarities.mean()).item() if len(intra_similarities) > 0 and len(inter_similarities) > 0 else 0.0,
            'total_samples': features.shape[0],
            'feature_dim': features.shape[1]
        }

        return stats


class SupConLossWithLogging(SupConLoss):
    """
    SupCon loss with detailed logging for debugging and analysis.
    """

    def __init__(self, *args, log_frequency: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_frequency = log_frequency
        self.call_count = 0

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.call_count += 1

        # Compute loss
        loss = super().forward(features, labels, mask)

        # Log periodically
        if self.call_count % self.log_frequency == 0:
            if labels is not None:
                stats = self.get_feature_similarities(
                    F.normalize(features.squeeze(1) if len(features.shape) == 3 else features, dim=1),
                    labels
                )
                logger.info(f"Step {self.call_count}: Loss={loss:.4f}, "
                           f"Intra-class sim={stats['intra_class_mean']:.3f}±{stats['intra_class_std']:.3f}, "
                           f"Inter-class sim={stats['inter_class_mean']:.3f}±{stats['inter_class_std']:.3f}, "
                           f"Separation={stats['separation_margin']:.3f}")

        return loss


def test_supcon_loss():
    """
    Test function to validate SupCon loss implementation.
    """
    print("Testing SupCon Loss Implementation...")

    # Test parameters
    batch_size = 8
    feature_dim = 128
    n_classes = 2

    # Create test data
    features = torch.randn(batch_size, feature_dim)
    features = F.normalize(features, dim=1)  # L2 normalize
    labels = torch.randint(0, n_classes, (batch_size,))

    # Test basic loss computation
    criterion = SupConLoss(temperature=0.07)
    loss = criterion(features, labels)

    print(f"✓ Basic loss computation: {loss:.4f}")

    # Test multi-view scenario
    features_multiview = features.unsqueeze(1).repeat(1, 2, 1)  # (batch_size, 2, feature_dim)
    loss_multiview = criterion(features_multiview, labels)

    print(f"✓ Multi-view loss computation: {loss_multiview:.4f}")

    # Test feature space statistics
    stats = criterion.get_feature_similarities(features, labels)
    print(f"✓ Feature space analysis:")
    print(f"  Intra-class similarity: {stats['intra_class_mean']:.3f} ± {stats['intra_class_std']:.3f}")
    print(f"  Inter-class similarity: {stats['inter_class_mean']:.3f} ± {stats['inter_class_std']:.3f}")
    print(f"  Separation margin: {stats['separation_margin']:.3f}")

    # Test numerical stability
    extreme_features = torch.randn(batch_size, feature_dim) * 100  # Large values
    extreme_features = F.normalize(extreme_features, dim=1)

    criterion_stable = SupConLoss(temperature=0.01, numerical_stability=True)  # Small temp = large logits
    loss_stable = criterion_stable(extreme_features, labels)

    print(f"✓ Numerical stability test: {loss_stable:.4f}")

    # Test gradient flow
    features.requires_grad_(True)
    loss = criterion(features, labels)
    loss.backward()

    assert features.grad is not None, "Gradient should not be None"
    grad_norm = features.grad.norm().item()
    print(f"✓ Gradient flow test: grad_norm={grad_norm:.4f}")

    print("All tests passed! ✓")


if __name__ == "__main__":
    test_supcon_loss()