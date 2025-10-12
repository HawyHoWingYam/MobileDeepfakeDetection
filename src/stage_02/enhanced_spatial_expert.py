"""
Enhanced Spatial Expert with Focal Loss and Advanced Learning Rate Scheduling

This module extends the basic spatial expert with advanced training optimizations
including focal loss integration, graduated learning rates, and multi-resolution
inference pipeline optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import math

import timm


@dataclass
class FocalLossConfig:
    """Configuration for Focal Loss"""
    alpha: float = 0.25
    gamma: float = 2.0
    reduction: str = "mean"
    label_smoothing: float = 0.1

    # Adaptive focal loss settings
    adaptive_alpha: bool = True
    alpha_schedule: str = "constant"  # "constant", "linear", "cosine"
    initial_alpha: float = 0.25
    final_alpha: float = 0.5

    # Class balancing
    pos_weight: Optional[float] = None
    auto_balance: bool = True


@dataclass
class GraduatedLRConfig:
    """Configuration for graduated learning rates"""
    # Base learning rates
    backbone_lr_multiplier: float = 0.1
    classifier_lr_multiplier: float = 1.0
    spatial_modules_lr_multiplier: float = 1.0

    # Warmup settings
    warmup_epochs: int = 3
    warmup_method: str = "linear"  # "linear", "cosine"

    # Schedule settings
    schedule_type: str = "cosine"  # "cosine", "step", "exponential", "polynomial"
    step_size: int = 30
    gamma: float = 0.1
    min_lr_ratio: float = 0.01

    # Adaptive settings
    adaptive_lr: bool = True
    patience: int = 5
    factor: float = 0.5
    threshold: float = 1e-4


class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss with adaptive alpha and label smoothing

    This implementation includes:
    - Adaptive alpha scheduling
    - Label smoothing
    - Class balancing
    - Support for both binary and multi-class classification
    """

    def __init__(self, config: FocalLossConfig):
        super().__init__()
        self.config = config
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.reduction = config.reduction
        self.label_smoothing = config.label_smoothing

        # For adaptive alpha
        self.current_epoch = 0
        self.total_epochs = None

        # Class balancing
        if config.pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(config.pos_weight))
        else:
            self.pos_weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (N,) or (N, C)
            targets: Ground truth labels of shape (N,)
        """
        # Handle binary classification case
        if inputs.dim() == 1 or (inputs.dim() == 2 and inputs.size(1) == 1):
            return self._binary_focal_loss(inputs.squeeze(), targets.float())
        else:
            return self._multiclass_focal_loss(inputs, targets.long())

    def _binary_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary focal loss implementation"""
        # Apply sigmoid
        p = torch.sigmoid(inputs)

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Calculate base cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate alpha_t
        current_alpha = self._get_current_alpha()
        alpha_t = current_alpha * targets + (1 - current_alpha) * (1 - targets)

        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def _multiclass_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multi-class focal loss implementation"""
        # Apply softmax
        p = F.softmax(inputs, dim=1)

        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get p_t for each sample
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Calculate focal weight
        current_alpha = self._get_current_alpha()
        focal_weight = current_alpha * (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def _get_current_alpha(self) -> float:
        """Get current alpha value based on schedule"""
        if not self.config.adaptive_alpha or self.total_epochs is None:
            return self.alpha

        progress = self.current_epoch / self.total_epochs

        if self.config.alpha_schedule == "linear":
            alpha = self.config.initial_alpha + progress * (self.config.final_alpha - self.config.initial_alpha)
        elif self.config.alpha_schedule == "cosine":
            alpha = self.config.initial_alpha + 0.5 * (self.config.final_alpha - self.config.initial_alpha) * (1 + math.cos(math.pi * progress))
        else:  # constant
            alpha = self.alpha

        return alpha

    def update_epoch(self, epoch: int, total_epochs: Optional[int] = None):
        """Update current epoch for adaptive alpha"""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs


class GraduatedLRScheduler(_LRScheduler):
    """
    Graduated Learning Rate Scheduler with different rates for different components

    This scheduler allows different learning rate schedules for:
    - Backbone (pre-trained) layers
    - Classifier (new) layers
    - Spatial modules (custom) layers
    """

    def __init__(self, optimizer: optim.Optimizer, config: GraduatedLRConfig, last_epoch: int = -1):
        self.config = config
        self.warmup_epochs = config.warmup_epochs
        self.total_epochs = None
        self.base_lrs_dict = {}

        # Store base learning rates for each parameter group
        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            self.base_lrs_dict[group_name] = group['lr']

        super().__init__(optimizer, last_epoch)

    def set_total_epochs(self, total_epochs: int):
        """Set total number of epochs for cosine annealing"""
        self.total_epochs = total_epochs

    def get_lr(self) -> List[float]:
        """Calculate learning rates for current epoch"""
        if self.last_epoch < self.warmup_epochs:
            return self._get_warmup_lr()
        else:
            return self._get_scheduled_lr()

    def _get_warmup_lr(self) -> List[float]:
        """Calculate warmup learning rates"""
        warmup_factor = self._get_warmup_factor()

        lrs = []
        for group in self.optimizer.param_groups:
            group_name = group.get('name', f'group_{len(lrs)}')
            base_lr = self.base_lrs_dict[group_name]

            # Apply component-specific multiplier
            if 'backbone' in group_name.lower():
                multiplier = self.config.backbone_lr_multiplier
            elif 'classifier' in group_name.lower():
                multiplier = self.config.classifier_lr_multiplier
            elif 'spatial' in group_name.lower():
                multiplier = self.config.spatial_modules_lr_multiplier
            else:
                multiplier = 1.0

            lr = base_lr * multiplier * warmup_factor
            lrs.append(lr)

        return lrs

    def _get_scheduled_lr(self) -> List[float]:
        """Calculate scheduled learning rates"""
        epoch = self.last_epoch - self.warmup_epochs

        lrs = []
        for group in self.optimizer.param_groups:
            group_name = group.get('name', f'group_{len(lrs)}')
            base_lr = self.base_lrs_dict[group_name]

            # Apply component-specific multiplier
            if 'backbone' in group_name.lower():
                multiplier = self.config.backbone_lr_multiplier
            elif 'classifier' in group_name.lower():
                multiplier = self.config.classifier_lr_multiplier
            elif 'spatial' in group_name.lower():
                multiplier = self.config.spatial_modules_lr_multiplier
            else:
                multiplier = 1.0

            # Apply schedule
            schedule_factor = self._get_schedule_factor(epoch)
            lr = base_lr * multiplier * schedule_factor

            # Apply minimum learning rate
            min_lr = base_lr * multiplier * self.config.min_lr_ratio
            lr = max(lr, min_lr)

            lrs.append(lr)

        return lrs

    def _get_warmup_factor(self) -> float:
        """Calculate warmup factor"""
        if self.config.warmup_method == "linear":
            return self.last_epoch / self.warmup_epochs
        elif self.config.warmup_method == "cosine":
            return 0.5 * (1 + math.cos(math.pi * (self.warmup_epochs - self.last_epoch) / self.warmup_epochs))
        else:
            return 1.0

    def _get_schedule_factor(self, epoch: int) -> float:
        """Calculate schedule factor"""
        if self.config.schedule_type == "cosine":
            if self.total_epochs is None:
                return 1.0
            total_steps = self.total_epochs - self.warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * epoch / total_steps))

        elif self.config.schedule_type == "step":
            return self.config.gamma ** (epoch // self.config.step_size)

        elif self.config.schedule_type == "exponential":
            return self.config.gamma ** epoch

        elif self.config.schedule_type == "polynomial":
            if self.total_epochs is None:
                return 1.0
            total_steps = self.total_epochs - self.warmup_epochs
            return (1 - epoch / total_steps) ** 2

        else:
            return 1.0


class MultiResolutionInferencePipeline:
    """
    Optimized inference pipeline supporting multiple resolutions
    """

    def __init__(self, model: nn.Module, supported_resolutions: List[int] = [224, 256, 288, 320]):
        self.model = model
        self.supported_resolutions = supported_resolutions
        self.resolution_cache = {}
        self.current_resolution = None

    def set_resolution(self, resolution: int):
        """Set current resolution for inference"""
        if resolution not in self.supported_resolutions:
            raise ValueError(f"Resolution {resolution} not supported. Supported: {self.supported_resolutions}")

        self.current_resolution = resolution

        # Warm up model for this resolution if not cached
        if resolution not in self.resolution_cache:
            self._warmup_resolution(resolution)

    def _warmup_resolution(self, resolution: int):
        """Warm up model for specific resolution"""
        dummy_input = torch.randn(1, 3, resolution, resolution)
        if next(self.model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()

        self.model.eval()
        with torch.no_grad():
            _ = self.model(dummy_input)

        self.resolution_cache[resolution] = True

    def predict(self, images: torch.Tensor, resolution: Optional[int] = None) -> torch.Tensor:
        """
        Perform inference with automatic resolution handling

        Args:
            images: Input images tensor
            resolution: Target resolution (if None, uses current_resolution)

        Returns:
            Model predictions
        """
        if resolution is not None:
            self.set_resolution(resolution)

        target_resolution = self.current_resolution or self.supported_resolutions[1]  # Default to 256

        # Resize if necessary
        if images.size(-1) != target_resolution:
            images = F.interpolate(
                images,
                size=(target_resolution, target_resolution),
                mode='bilinear',
                align_corners=False
            )

        # Run inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)

        return predictions

    def predict_multi_resolution(
        self,
        images: torch.Tensor,
        resolutions: Optional[List[int]] = None,
        ensemble_method: str = "average"
    ) -> torch.Tensor:
        """
        Perform multi-resolution inference with ensembling

        Args:
            images: Input images tensor
            resolutions: List of resolutions to use
            ensemble_method: "average", "max", "weighted"

        Returns:
            Ensembled predictions
        """
        if resolutions is None:
            resolutions = self.supported_resolutions

        predictions = []

        for resolution in resolutions:
            pred = self.predict(images, resolution)
            predictions.append(pred)

        # Ensemble predictions
        if ensemble_method == "average":
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        elif ensemble_method == "max":
            ensemble_pred = torch.max(torch.stack(predictions), dim=0)[0]
        elif ensemble_method == "weighted":
            # Weight by resolution (higher resolution gets more weight)
            weights = torch.tensor([r / max(resolutions) for r in resolutions])
            weights = weights / weights.sum()

            weighted_preds = []
            for pred, weight in zip(predictions, weights):
                weighted_preds.append(pred * weight)
            ensemble_pred = torch.sum(torch.stack(weighted_preds), dim=0)
        else:
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)

        return ensemble_pred


class EfficientNetV2SpatialExpert(nn.Module):
    """
    Base Spatial Expert using EfficientNetV2 for spatial artifact detection
    """

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.model_name = "efficientnetv2_rw_s"  # Default model
        self.num_classes = 1

        # Load model
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )

        # Add classifier
        self.classifier = nn.Linear(self.backbone.num_features, self.num_classes)

        # Additional spatial modules can be added here
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class EnhancedSpatialExpert(EfficientNetV2SpatialExpert):
    """
    Enhanced Spatial Expert with advanced training optimizations
    """

    def __init__(
        self,
        config_path: str,
        focal_loss_config: Optional[FocalLossConfig] = None,
        lr_config: Optional[GraduatedLRConfig] = None
    ):
        super().__init__(config_path)

        # Enhanced components
        self.focal_loss_config = focal_loss_config or FocalLossConfig()
        self.lr_config = lr_config or GraduatedLRConfig()

        # Initialize focal loss
        self.focal_loss = FocalLoss(self.focal_loss_config)

        # Initialize multi-resolution pipeline
        self.inference_pipeline = MultiResolutionInferencePipeline(self)

        # Training state
        self.optimizer = None
        self.lr_scheduler = None

    def setup_optimizer(self, base_lr: float = 1e-3) -> optim.Optimizer:
        """Setup optimizer with graduated learning rates"""

        # Group parameters by component
        backbone_params = []
        classifier_params = []
        spatial_params = []

        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            elif any(module in name for module in ['spatial_attention', 'edge_enhancement', 'texture_analysis']):
                spatial_params.append(param)
            else:
                classifier_params.append(param)  # Default to classifier group

        # Create parameter groups
        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * self.lr_config.backbone_lr_multiplier,
                'name': 'backbone'
            })

        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': base_lr * self.lr_config.classifier_lr_multiplier,
                'name': 'classifier'
            })

        if spatial_params:
            param_groups.append({
                'params': spatial_params,
                'lr': base_lr * self.lr_config.spatial_modules_lr_multiplier,
                'name': 'spatial_modules'
            })

        # Create optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        return self.optimizer

    def setup_lr_scheduler(self, total_epochs: int) -> GraduatedLRScheduler:
        """Setup learning rate scheduler"""
        if self.optimizer is None:
            raise ValueError("Optimizer must be setup before learning rate scheduler")

        self.lr_scheduler = GraduatedLRScheduler(self.optimizer, self.lr_config)
        self.lr_scheduler.set_total_epochs(total_epochs)

        # Setup focal loss epochs
        self.focal_loss.update_epoch(0, total_epochs)

        return self.lr_scheduler

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        epoch: int
    ) -> Dict[str, float]:
        """
        Enhanced training step with focal loss and graduated learning rates

        Args:
            batch: (images, labels) tuple
            epoch: Current epoch number

        Returns:
            Dictionary with loss components
        """
        images, labels = batch

        # Update focal loss epoch
        self.focal_loss.update_epoch(epoch)

        # Forward pass
        output = self.forward(images)

        # Calculate focal loss
        if hasattr(output, 'confidence_scores'):
            logits = output.confidence_scores
        else:
            logits = output

        focal_loss = self.focal_loss(logits.squeeze(), labels.float())

        # Additional auxiliary losses if available
        auxiliary_losses = {}
        total_loss = focal_loss

        # Spatial attention regularization
        if hasattr(self, 'spatial_attention') and hasattr(self.spatial_attention, 'attention_weights'):
            attention_reg = torch.mean(self.spatial_attention.attention_weights ** 2)
            auxiliary_losses['attention_regularization'] = attention_reg * 0.01
            total_loss += auxiliary_losses['attention_regularization']

        # Edge preservation loss
        if hasattr(self, 'edge_enhancement') and hasattr(self.edge_enhancement, 'edge_weights'):
            edge_reg = torch.mean(self.edge_enhancement.edge_weights ** 2)
            auxiliary_losses['edge_regularization'] = edge_reg * 0.005
            total_loss += auxiliary_losses['edge_regularization']

        # Backward pass
        if self.training:
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self.optimizer.step()

        # Prepare loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'focal_loss': focal_loss.item(),
            **{k: v.item() for k, v in auxiliary_losses.items()}
        }

        return loss_dict

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Enhanced validation step"""
        images, labels = batch

        self.eval()
        with torch.no_grad():
            # Regular inference
            output = self.forward(images)

            if hasattr(output, 'confidence_scores'):
                logits = output.confidence_scores
            else:
                logits = output

            # Calculate validation loss
            val_loss = self.focal_loss(logits.squeeze(), labels.float())

            # Calculate metrics
            predictions = torch.sigmoid(logits.squeeze())
            binary_preds = (predictions > 0.5).float()

            accuracy = (binary_preds == labels.float()).float().mean()

            # Precision and recall
            tp = ((binary_preds == 1) & (labels == 1)).float().sum()
            fp = ((binary_preds == 1) & (labels == 0)).float().sum()
            fn = ((binary_preds == 0) & (labels == 1)).float().sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'val_loss': val_loss.item(),
            'val_accuracy': accuracy.item(),
            'val_precision': precision.item(),
            'val_recall': recall.item(),
            'val_f1': f1.item()
        }

    def step_lr_scheduler(self):
        """Step the learning rate scheduler"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_current_lrs(self) -> Dict[str, float]:
        """Get current learning rates for each parameter group"""
        if self.optimizer is None:
            return {}

        lrs = {}
        for group in self.optimizer.param_groups:
            group_name = group.get('name', 'default')
            lrs[group_name] = group['lr']

        return lrs

    def predict_optimized(
        self,
        images: torch.Tensor,
        resolution: Optional[int] = None,
        use_multi_resolution: bool = False
    ) -> torch.Tensor:
        """
        Optimized inference with resolution handling

        Args:
            images: Input images
            resolution: Target resolution
            use_multi_resolution: Whether to use multi-resolution ensembling

        Returns:
            Predictions
        """
        if use_multi_resolution:
            return self.inference_pipeline.predict_multi_resolution(images)
        else:
            return self.inference_pipeline.predict(images, resolution)


def test_enhanced_spatial_expert():
    """Test function for enhanced spatial expert"""
    print("Testing Enhanced Spatial Expert...")

    # Create dummy config files if they don't exist
    config_path = "test_spatial_config.json"
    if not Path(config_path).exists():
        config = {
            "model_architecture": {"model_name": "efficientnetv2_rw_s", "num_classes": 1},
            "multi_resolution_support": {"supported_resolutions": [224, 256]}
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

    try:
        # Create enhanced expert
        focal_config = FocalLossConfig(alpha=0.25, gamma=2.0)
        lr_config = GraduatedLRConfig(warmup_epochs=2)

        expert = EnhancedSpatialExpert(config_path, focal_config, lr_config)
        print("✓ Enhanced spatial expert created successfully")

        # Test optimizer setup
        optimizer = expert.setup_optimizer(base_lr=1e-3)
        print(f"✓ Optimizer setup with {len(optimizer.param_groups)} parameter groups")

        # Test scheduler setup
        scheduler = expert.setup_lr_scheduler(total_epochs=10)
        print("✓ Learning rate scheduler setup successfully")

        # Test inference pipeline
        dummy_input = torch.randn(2, 3, 256, 256)
        predictions = expert.predict_optimized(dummy_input, resolution=256)
        print(f"✓ Optimized inference: {predictions.shape}")

        # Test multi-resolution inference
        multi_res_pred = expert.predict_optimized(dummy_input, use_multi_resolution=True)
        print(f"✓ Multi-resolution inference: {multi_res_pred.shape}")

        # Test training step
        expert.train()
        loss_dict = expert.training_step((dummy_input, torch.randint(0, 2, (2,))), epoch=0)
        print(f"✓ Training step completed: {list(loss_dict.keys())}")

        # Test learning rates
        lrs = expert.get_current_lrs()
        print(f"✓ Current learning rates: {lrs}")

        print("✓ Enhanced Spatial Expert test completed successfully")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    finally:
        # Cleanup
        if Path(config_path).exists():
            Path(config_path).unlink()


if __name__ == "__main__":
    test_enhanced_spatial_expert()