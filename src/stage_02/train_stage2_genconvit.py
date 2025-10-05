"""
AWARE-NET Stage 2: GenConViT Expert Training Script

This script implements the complete training pipeline for the GenConViT
(Generative-Contrastive Vision Transformer) expert with dual-task optimization
(classification + reconstruction) for generative structure analysis.

Key Features:
- Dual-variant architecture: ED (Encoder-Decoder) and VAE variants
- Staged training: reconstruction pretraining → joint training → classification fine-tuning
- Advanced reconstruction quality analysis with SSIM, LPIPS, and perceptual metrics
- ConvNeXt-Swin hybrid backbone for multi-scale feature extraction
- Adaptive task weighting with reconstruction quality monitoring
- Professional domain validation for GAN/Diffusion detection

Usage:
    python train_stage2_genconvit.py --variant ed --mode staged_training --epochs 50
    python train_stage2_genconvit.py --variant vae --mode joint_training --epochs 30
    python train_stage2_genconvit.py --mode evaluation --checkpoint_path models/best_genconvit.pth
"""

import os
import sys
import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.cuda.amp import GradScaler, autocast

# AWARE-NET imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.experiment_utils import ExperimentManager, ExperimentConfig, ExperimentResult
from utils.metrics import AcademicMetrics, MetricResult, ComparisonResult
from utils.calibration_tools import CalibrationAnalyzer, TemperatureScalingResult
from utils.dataset_config import DatasetConfig

# Stage 2 specific imports
try:
    from .genconvit_expert import (
        GenConViTExpert, GenConViTConfig, GenConViTVariant, ReconstructionMetric
    )
    from .enhanced_genconvit import (
        EnhancedGenConViT, GenConViTConfig as EnhancedConfig,
        FusionStrategy, ReconstructionMode, DualVariantConfig
    )
    from .reconstruction_analysis import (
        ReconstructionQualityMetrics, ReconstructionConfig,
        ReconstructionAnalyzer
    )
    from .multi_resolution_dataloader import DataLoaderConfig
    from .data_augmentation import (
        AugmentationFactory, AugmentationConfig,
        CompressionSimulation
    )
    from .training_monitor import (
        TrainingMonitor, MonitoringConfig, SystemMetrics, TrainingMetrics
    )
except ImportError:  # pragma: no cover - script fallback
    from genconvit_expert import (
        GenConViTExpert, GenConViTConfig, GenConViTVariant, ReconstructionMetric
    )
    from enhanced_genconvit import (
        EnhancedGenConViT, GenConViTConfig as EnhancedConfig,
        FusionStrategy, ReconstructionMode, DualVariantConfig
    )
    from reconstruction_analysis import (
        ReconstructionQualityMetrics, ReconstructionConfig,
        ReconstructionAnalyzer
    )
    from multi_resolution_dataloader import DataLoaderConfig
    from data_augmentation import (
        AugmentationFactory, AugmentationConfig,
        CompressionSimulation
    )
    from training_monitor import (
        TrainingMonitor, MonitoringConfig, SystemMetrics, TrainingMetrics
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage2_genconvit_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODE_SUFFIXES = {
    'balanced': '_balanced',
    'anonymized': '_anonymized',
    'anonymized_balanced': '_anonymized_balanced'
}


def _apply_dataset_mode(manifest_path: Path, dataset_mode: str) -> Path:
    """Return manifest path adjusted for the requested dataset mode."""
    suffix = MODE_SUFFIXES.get(dataset_mode)
    if not suffix:
        return manifest_path
    return manifest_path.with_name(f"{manifest_path.stem}{suffix}{manifest_path.suffix}")


def resolve_manifest_paths(
    config_path: Union[str, Path],
    dataset_name: str,
    dataset_mode: str,
) -> Tuple[Dict[str, Path], Path]:
    """Resolve manifest paths for GenConViT datasets."""

    ds_config = DatasetConfig(config_path, dataset_name=dataset_name)
    manifests: Dict[str, Path] = {}

    for split in ('train', 'val', 'test'):
        base_manifest = ds_config.get_manifest_path(split)
        variant_manifest = _apply_dataset_mode(base_manifest, dataset_mode)
        selected_manifest = variant_manifest if variant_manifest.exists() else base_manifest

        if not selected_manifest.exists():
            raise FileNotFoundError(
                f"Manifest not found for {dataset_name}/{split} ({dataset_mode}): {selected_manifest}"
            )

        manifests[split] = selected_manifest

    return manifests, ds_config.root_path


def prepare_classification_targets(loss_module: nn.Module, targets: torch.Tensor) -> torch.Tensor:
    """Cast targets to the appropriate dtype for the given loss module."""
    if isinstance(loss_module, nn.CrossEntropyLoss):
        return targets.long()
    return targets.float()

@dataclass
class GenConViTTrainingConfig:
    """Comprehensive training configuration for GenConViT expert"""

    # Training mode
    mode: str = 'staged_training'  # 'staged_training', 'joint_training', 'evaluation'
    variant: str = 'ed'  # 'ed' (Encoder-Decoder) or 'vae' (Variational AutoEncoder)

    # Model configuration
    model_config: GenConViTConfig = None

    # Staged training configuration
    staged_training: Dict[str, Any] = None

    # Dual-task training parameters
    batch_size: int = 24
    learning_rate: float = 1e-3
    epochs: int = 50
    warmup_epochs: int = 5

    # Task weighting
    classification_weight: float = 0.7
    reconstruction_weight: float = 0.3
    perceptual_weight: float = 0.1
    kl_weight: float = 0.001  # For VAE variant

    # Adaptive weighting
    adaptive_weighting: bool = True
    reconstruction_quality_threshold: float = 0.75

    # Loss configuration
    classification_loss: str = 'focal_loss'
    reconstruction_loss: str = 'l1_perceptual'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Data configuration
    data_loader_config: DataLoaderConfig = None
    augmentation_config: AugmentationConfig = None
    reconstruction_config: ReconstructionConfig = None
    dataset_config: str = 'configs/datasets.json'
    manifest_dataset: str = 'celebdf_v2'
    manifest_mode: str = 'balanced'

    # Monitoring configuration
    monitoring_config: MonitoringConfig = None

    # Optimization
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    gradient_checkpointing: bool = True

    # Paths
    data_path: str = "data"
    output_dir: str = "experiments/stage_02/genconvit"
    weights_dir: str = "models/stage_02/genconvit"
    logs_dir: str = "logs/stage_02/genconvit"
    config_path: str = "configs/genconvit_expert_config.json"
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    test_manifest: Optional[str] = None
    dataset_root: str = "."

    # Performance targets
    target_auc: float = 0.90
    reconstruction_quality_auc: float = 0.93
    generative_artifact_auc: float = 0.91
    baseline_improvement: float = 0.02
    max_inference_time: float = 150.0  # ms

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        """Initialize default configurations"""
        if self.model_config is None:
            self.model_config = GenConViTConfig(
                variant=GenConViTVariant.ED if self.variant == 'ed' else GenConViTVariant.VAE
            )

        if self.data_loader_config is None:
            self.data_loader_config = DataLoaderConfig(
                batch_size=self.batch_size,
                expert_type="generative"
            )

        if self.augmentation_config is None:
            self.augmentation_config = AugmentationConfig(
                generative_expert_mode=True,
                spatial_expert_mode=False
            )

        if self.reconstruction_config is None:
            self.reconstruction_config = ReconstructionConfig(
                output_dir=os.path.join(self.output_dir, "reconstruction_analysis")
            )

        if self.monitoring_config is None:
            self.monitoring_config = MonitoringConfig(
                output_dir=os.path.join(self.logs_dir, "monitoring")
            )

        # Set staged training defaults
        if self.staged_training is None:
            self.staged_training = {
                'reconstruction_pretraining_epochs': 10,
                'joint_training_epochs': 30,
                'classification_finetuning_epochs': 10,
                'reconstruction_lr_multiplier': 2.0,
                'classification_lr_multiplier': 0.5
            }


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features for reconstruction quality assessment."""

    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()

        # Load VGG16 for perceptual loss
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.layer_name_to_index = {
            'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16, 'relu4_3': 23
        }

        # Extract features up to specified layers
        self.feature_extractor = nn.ModuleDict()
        for layer_name in layers:
            idx = self.layer_name_to_index[layer_name]
            self.feature_extractor[layer_name] = nn.Sequential(*list(vgg.children())[:idx+1])

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """Calculate perceptual loss between x and y."""
        loss = 0.0

        for layer_name in self.layers:
            x_features = self.feature_extractor[layer_name](x)
            y_features = self.feature_extractor[layer_name](y)
            loss += F.mse_loss(x_features, y_features)

        return loss / len(self.layers)


class ReconstructionLoss(nn.Module):
    """Combined reconstruction loss with multiple components."""

    def __init__(
        self,
        l1_weight: float = 0.5,
        perceptual_weight: float = 0.3,
        ssim_weight: float = 0.2,
        device: str = 'cuda'
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight

        # Initialize perceptual loss
        self.perceptual_loss = PerceptualLoss().to(device)

        # SSIM loss (simplified implementation)
        self.ssim_loss = self._ssim_loss

    def _ssim_loss(self, x, y):
        """Simplified SSIM loss calculation."""
        # Simplified SSIM implementation
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)

        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))

        return 1 - ssim_map.mean()

    def forward(self, reconstructed, target):
        """Calculate combined reconstruction loss."""
        l1_loss = F.l1_loss(reconstructed, target)
        perceptual_loss = self.perceptual_loss(reconstructed, target)
        ssim_loss = self.ssim_loss(reconstructed, target)

        total_loss = (
            self.l1_weight * l1_loss +
            self.perceptual_weight * perceptual_loss +
            self.ssim_weight * ssim_loss
        )

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'ssim_loss': ssim_loss.item()
        }


class GenConViTDataset(torch.utils.data.Dataset):
    """Dataset with manifest support for GenConViT training."""

    def __init__(
        self,
        split: str,
        resolution: int,
        augmentation_config: AugmentationConfig,
        return_reconstruction_target: bool,
        manifest_path: Optional[str] = None,
        dataset_root: str = ".",
        fallback_path: Optional[str] = None
    ):
        self.split = split
        self.resolution = resolution
        self.return_reconstruction_target = return_reconstruction_target
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.dataset_root = Path(dataset_root)
        self.augmentation_config = augmentation_config or AugmentationConfig(generative_expert_mode=True)

        if self.manifest_path and self.manifest_path.exists():
            if not self.dataset_root.exists():
                raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
            self.use_manifest = True
            self.data = pd.read_csv(self.manifest_path)
            if 'valid' in self.data.columns:
                self.data = self.data[self.data['valid'] == True]
            self.data = self.data.reset_index(drop=True)
            if self.data.empty:
                raise ValueError(f"Manifest {self.manifest_path} contains no valid samples")

            if self.split == 'train':
                self.augmentation = AugmentationFactory.create_augmentation(
                    "generative",
                    **self.augmentation_config.__dict__
                )
            else:
                self.augmentation = A.Compose([
                    A.Resize(self.resolution, self.resolution),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

            logger.info(
                "Loaded GenConViT manifest dataset (%s): samples=%d",
                self.manifest_path,
                len(self.data)
            )
        else:
            if not fallback_path:
                raise FileNotFoundError(
                    f"Manifest not provided for split={split}; specify fallback_path or manifest."
                )
            self.use_manifest = False
            from torchvision.datasets import ImageFolder
            fallback_root = Path(fallback_path)
            if not fallback_root.exists():
                raise FileNotFoundError(f"Fallback path not found: {fallback_root}")
            root = fallback_root / split
            if not root.exists():
                raise FileNotFoundError(f"Fallback split path not found: {root}")
            self.dataset = ImageFolder(root=str(root))
            if self.split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((self.resolution + 32, self.resolution + 32)),
                    transforms.RandomCrop((self.resolution, self.resolution)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.resolution, self.resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            logger.warning(
                "Using fallback ImageFolder dataset for split=%s at %s (samples=%d)",
                split,
                root,
                len(self.dataset)
            )

    def __len__(self):
        if self.use_manifest:
            return len(self.data)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.use_manifest:
            row = self.data.iloc[idx]
            label = float(row['label'])
            image_path = self.dataset_root / Path(row['image_path'])
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            if self.split == 'train':
                processed = self.augmentation(image_np)
            else:
                processed = self.augmentation(image=image_np)['image']
            if self.return_reconstruction_target:
                return processed, processed.clone(), torch.tensor(label, dtype=torch.float32)
            return processed, torch.tensor(label, dtype=torch.float32)

        image, label = self.dataset[idx]
        processed_image = self.transform(image)
        if self.return_reconstruction_target:
            return processed_image, processed_image.clone(), torch.tensor(float(label), dtype=torch.float32)
        return processed_image, torch.tensor(float(label), dtype=torch.float32)

    def get_labels(self) -> List[int]:
        if self.use_manifest:
            return self.data['label'].astype(int).tolist()
        return [self.dataset[i][1] for i in range(len(self.dataset))]


def setup_data_loaders(config: GenConViTTrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Setup data loaders for GenConViT training."""

    use_manifest = all([
        config.train_manifest,
        config.val_manifest,
        config.test_manifest,
    ])

    dataset_root = Path(config.dataset_root)
    if use_manifest and not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    train_dataset = GenConViTDataset(
        split='train',
        resolution=256,
        augmentation_config=config.augmentation_config,
        return_reconstruction_target=True,
        manifest_path=config.train_manifest if use_manifest else None,
        dataset_root=dataset_root if use_manifest else Path(config.data_path),
        fallback_path=config.data_path
    )

    val_dataset = GenConViTDataset(
        split='val',
        resolution=256,
        augmentation_config=config.augmentation_config,
        return_reconstruction_target=True,
        manifest_path=config.val_manifest if use_manifest else None,
        dataset_root=dataset_root if use_manifest else Path(config.data_path),
        fallback_path=config.data_path
    )

    test_dataset = GenConViTDataset(
        split='test',
        resolution=256,
        augmentation_config=config.augmentation_config,
        return_reconstruction_target=True,
        manifest_path=config.test_manifest if use_manifest else None,
        dataset_root=dataset_root if use_manifest else Path(config.data_path),
        fallback_path=config.data_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Data loaders created: train={len(train_loader)} batches, "
               f"val={len(val_loader)} batches, test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def create_model(config: GenConViTTrainingConfig) -> GenConViTExpert:
    """Create and initialize the GenConViT expert model."""

    model = GenConViTExpert(config.model_config)

    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"GenConViT expert model created: {config.variant} variant")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


def create_loss_functions(
    config: GenConViTTrainingConfig,
    device: torch.device
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
    """Create loss functions for dual-task training."""

    # Classification loss
    if config.classification_loss == 'focal_loss':
        from enhanced_spatial_expert import FocalLoss, FocalLossConfig
        focal_config = FocalLossConfig(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma
        )
        classification_loss = FocalLoss(focal_config)
    else:
        classification_loss = nn.BCEWithLogitsLoss()

    # Reconstruction loss
    reconstruction_loss = ReconstructionLoss(device=device)

    # KL divergence loss for VAE variant
    kl_loss = None
    if config.variant == 'vae':
        kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return classification_loss, reconstruction_loss, kl_loss


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: GenConViTTrainingConfig,
    steps_per_epoch: int,
    stage: str = 'joint'
) -> Tuple[optim.Optimizer, Any]:
    """Create optimizer and scheduler for different training stages."""

    # Separate parameter groups for different components
    convnext_params = []
    swin_params = []
    decoder_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if 'convnext' in name:
            convnext_params.append(param)
        elif 'swin' in name:
            swin_params.append(param)
        elif 'decoder' in name or 'reconstruction' in name:
            decoder_params.append(param)
        elif 'classifier' in name or 'classification' in name:
            classifier_params.append(param)

    # Create parameter groups with different learning rates
    if stage == 'reconstruction_pretraining':
        param_groups = [
            {'params': convnext_params, 'lr': config.learning_rate * 0.1, 'name': 'convnext'},
            {'params': swin_params, 'lr': config.learning_rate * 0.1, 'name': 'swin'},
            {'params': decoder_params, 'lr': config.learning_rate * 2.0, 'name': 'decoder'}
        ]
    elif stage == 'classification_finetuning':
        param_groups = [
            {'params': convnext_params, 'lr': config.learning_rate * 0.05, 'name': 'convnext'},
            {'params': swin_params, 'lr': config.learning_rate * 0.05, 'name': 'swin'},
            {'params': classifier_params, 'lr': config.learning_rate * 1.0, 'name': 'classifier'}
        ]
    else:  # joint training
        param_groups = [
            {'params': convnext_params, 'lr': config.learning_rate * 0.1, 'name': 'convnext'},
            {'params': swin_params, 'lr': config.learning_rate * 0.1, 'name': 'swin'},
            {'params': decoder_params, 'lr': config.learning_rate * 1.0, 'name': 'decoder'},
            {'params': classifier_params, 'lr': config.learning_rate * 1.0, 'name': 'classifier'}
        ]

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    # Create cosine annealing scheduler
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(f"Optimizer created for stage: {stage}")
    logger.info(f"Parameter groups: {[g['name'] for g in param_groups]}")

    return optimizer, scheduler


def train_epoch(
    model: GenConViTExpert,
    train_loader: DataLoader,
    classification_loss: nn.Module,
    reconstruction_loss: nn.Module,
    kl_loss: Optional[nn.Module],
    optimizer: optim.Optimizer,
    scheduler: Any,
    config: GenConViTTrainingConfig,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0,
    stage: str = 'joint'
) -> Dict[str, float]:
    """Train one epoch with dual-task optimization."""

    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    correct = 0
    total = 0

    # Adaptive task weighting
    if config.adaptive_weighting and epoch > 5:
        # Adjust weights based on reconstruction quality
        classification_weight = config.classification_weight
        reconstruction_weight = config.reconstruction_weight
    else:
        classification_weight = config.classification_weight
        reconstruction_weight = config.reconstruction_weight

    for batch_idx, (images, targets_recon, targets_class) in enumerate(train_loader):
        images = images.to(device)
        targets_recon = targets_recon.to(device)
        targets_class = targets_class.to(device)
        targets_class_prepared = prepare_classification_targets(classification_loss, targets_class)

        optimizer.zero_grad()

        if scaler and config.mixed_precision:
            with autocast():
                # Forward pass
                if config.variant == 'vae':
                    outputs = model(images, return_reconstruction=True, return_latent=True)
                    classification_logits = outputs['classification']
                    reconstructed = outputs['reconstruction']
                    mu = outputs.get('mu')
                    logvar = outputs.get('logvar')
                else:
                    outputs = model(images, return_reconstruction=True)
                    classification_logits = outputs['classification']
                    reconstructed = outputs['reconstruction']

                # Calculate losses
                if stage == 'reconstruction_pretraining':
                    # Focus on reconstruction
                    recon_loss, recon_details = reconstruction_loss(reconstructed, targets_recon)
                    loss = recon_loss
                    class_loss = torch.tensor(0.0, device=device)

                elif stage == 'classification_finetuning':
                    # Focus on classification
                    class_loss = classification_loss(classification_logits.squeeze(), targets_class_prepared)
                    loss = class_loss
                    recon_loss = torch.tensor(0.0, device=device)

                else:  # joint training
                    class_loss = classification_loss(classification_logits.squeeze(), targets_class_prepared)
                    recon_loss, recon_details = reconstruction_loss(reconstructed, targets_recon)

                    loss = (classification_weight * class_loss +
                           reconstruction_weight * recon_loss)

                # Add KL loss for VAE variant
                if config.variant == 'vae' and kl_loss is not None and mu is not None:
                    kl_divergence = kl_loss(mu, logvar)
                    loss += config.kl_weight * kl_divergence
                    total_kl_loss += kl_divergence.item()

            scaler.scale(loss).backward()
            if config.gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            scaler.step(optimizer)
            scaler.update()

        else:
            # Forward pass without mixed precision
            if config.variant == 'vae':
                outputs = model(images, return_reconstruction=True, return_latent=True)
                classification_logits = outputs['classification']
                reconstructed = outputs['reconstruction']
                mu = outputs.get('mu')
                logvar = outputs.get('logvar')
            else:
                outputs = model(images, return_reconstruction=True)
                classification_logits = outputs['classification']
                reconstructed = outputs['reconstruction']

            # Calculate losses (same logic as above)
            if stage == 'reconstruction_pretraining':
                recon_loss, recon_details = reconstruction_loss(reconstructed, targets_recon)
                loss = recon_loss
                class_loss = torch.tensor(0.0, device=device)
            elif stage == 'classification_finetuning':
                class_loss = classification_loss(classification_logits.squeeze(), targets_class_prepared)
                loss = class_loss
                recon_loss = torch.tensor(0.0, device=device)
            else:
                class_loss = classification_loss(classification_logits.squeeze(), targets_class_prepared)
                recon_loss, recon_details = reconstruction_loss(reconstructed, targets_recon)
                loss = (classification_weight * class_loss +
                       reconstruction_weight * recon_loss)

            if config.variant == 'vae' and kl_loss is not None and mu is not None:
                kl_divergence = kl_loss(mu, logvar)
                loss += config.kl_weight * kl_divergence
                total_kl_loss += kl_divergence.item()

            loss.backward()

            if config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)

            optimizer.step()

        scheduler.step()

        # Update statistics
        total_loss += loss.item()
        total_classification_loss += class_loss.item()
        total_reconstruction_loss += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0

        # Calculate accuracy
        if stage != 'reconstruction_pretraining':
            predicted = (torch.sigmoid(classification_logits.squeeze()) > 0.5).float()
            total += targets_class.size(0)
            correct += (predicted == targets_class).sum().item()

        # Log progress
        if batch_idx % 50 == 0:
            accuracy = 100. * correct / total if total > 0 else 0
            logger.info(f"Batch {batch_idx}/{len(train_loader)}: "
                       f"Loss={loss.item():.4f}, "
                       f"ClassLoss={class_loss.item():.4f}, "
                       f"ReconLoss={recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0:.4f}, "
                       f"Acc={accuracy:.2f}%")

    epoch_metrics = {
        'train_loss': total_loss / len(train_loader),
        'train_classification_loss': total_classification_loss / len(train_loader),
        'train_reconstruction_loss': total_reconstruction_loss / len(train_loader),
        'train_accuracy': 100. * correct / total if total > 0 else 0
    }

    if config.variant == 'vae':
        epoch_metrics['train_kl_loss'] = total_kl_loss / len(train_loader)

    return epoch_metrics


def validate_model(
    model: GenConViTExpert,
    val_loader: DataLoader,
    device: torch.device,
    metrics_calculator: AcademicMetrics,
    reconstruction_analyzer: Optional[Any] = None
) -> Dict[str, float]:
    """Validate model with comprehensive metrics including reconstruction quality."""

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    reconstruction_metrics = []

    with torch.no_grad():
        for batch_idx, (images, targets_recon, targets_class) in enumerate(val_loader):
            images = images.to(device)
            targets_recon = targets_recon.to(device)
            targets_class = targets_class.to(device)

            # Forward pass
            outputs = model(images, return_reconstruction=True)
            classification_logits = outputs['classification']
            reconstructed = outputs['reconstruction']

            probabilities = torch.sigmoid(classification_logits.squeeze())
            predictions = (probabilities > 0.5).float()

            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets_class.cpu().numpy())

            # Calculate reconstruction quality metrics for first batch
            if batch_idx == 0 and reconstruction_analyzer is not None:
                try:
                    recon_metrics = reconstruction_analyzer.analyze_batch(
                        original=targets_recon[:4],
                        reconstructed=reconstructed[:4],
                        save_visualizations=True
                    )
                    reconstruction_metrics.append(recon_metrics)
                    logger.info("Reconstruction analysis completed for validation batch")
                except Exception as e:
                    logger.warning(f"Reconstruction analysis failed: {e}")

    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # Calculate comprehensive metrics
    auc_result = metrics_calculator.calculate_auc_with_ci(y_true, y_prob)
    f1_result = metrics_calculator.calculate_f1_with_ci(y_true, y_pred)
    accuracy = np.mean(y_pred == y_true)

    metrics = {
        'val_auc': auc_result.value,
        'val_auc_ci_lower': auc_result.confidence_interval[0] if auc_result.confidence_interval else 0,
        'val_auc_ci_upper': auc_result.confidence_interval[1] if auc_result.confidence_interval else 0,
        'val_f1': f1_result.value,
        'val_f1_ci_lower': f1_result.confidence_interval[0] if f1_result.confidence_interval else 0,
        'val_f1_ci_upper': f1_result.confidence_interval[1] if f1_result.confidence_interval else 0,
        'val_accuracy': accuracy * 100
    }

    # Add reconstruction quality metrics
    if reconstruction_metrics:
        avg_metrics = reconstruction_metrics[0]  # For simplicity, use first batch
        metrics.update({
            'val_reconstruction_ssim': avg_metrics.get('mean_ssim', 0),
            'val_reconstruction_psnr': avg_metrics.get('mean_psnr', 0),
            'val_reconstruction_lpips': avg_metrics.get('mean_lpips', 1)
        })

    return metrics


def run_staged_training(config: GenConViTTrainingConfig) -> Dict[str, ExperimentResult]:
    """Run complete staged training: pretraining → joint → fine-tuning."""

    logger.info("Starting staged training")
    results = {}

    # Stage 1: Reconstruction pretraining
    logger.info("\n" + "="*60)
    logger.info("Stage 1: Reconstruction Pretraining")
    logger.info("="*60)

    pretraining_config = config
    pretraining_config.epochs = config.staged_training['reconstruction_pretraining_epochs']
    pretraining_result = train_genconvit_expert(pretraining_config, stage='reconstruction_pretraining')
    results['reconstruction_pretraining'] = pretraining_result

    # Stage 2: Joint training
    logger.info("\n" + "="*60)
    logger.info("Stage 2: Joint Training")
    logger.info("="*60)

    joint_config = config
    joint_config.epochs = config.staged_training['joint_training_epochs']
    # Load best model from pretraining
    joint_config.pretrained_weights = pretraining_result.model_path
    joint_result = train_genconvit_expert(joint_config, stage='joint_training')
    results['joint_training'] = joint_result

    # Stage 3: Classification fine-tuning
    logger.info("\n" + "="*60)
    logger.info("Stage 3: Classification Fine-tuning")
    logger.info("="*60)

    finetuning_config = config
    finetuning_config.epochs = config.staged_training['classification_finetuning_epochs']
    # Load best model from joint training
    finetuning_config.pretrained_weights = joint_result.model_path
    finetuning_result = train_genconvit_expert(finetuning_config, stage='classification_finetuning')
    results['classification_finetuning'] = finetuning_result

    # Final evaluation
    final_auc = finetuning_result.test_metrics['val_auc']
    logger.info(f"\nStaged training completed!")
    logger.info(f"Final AUC: {final_auc:.4f}")

    return results


def train_genconvit_expert(
    config: GenConViTTrainingConfig,
    stage: str = 'joint'
) -> ExperimentResult:
    """Main training function for GenConViT expert."""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Setup reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = config.deterministic
        torch.backends.cudnn.benchmark = not config.deterministic

    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.weights_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

    # Setup experiment management
    exp_config = ExperimentConfig(
        experiment_name=f"genconvit_{config.variant}_{stage}_{config.epochs}epochs",
        model_name=f"genconvit_{config.variant}",
        dataset_name="generative_artifacts_dataset",
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_epochs=config.epochs,
        model_params=asdict(config.model_config),
        seed=config.seed,
        output_path=config.output_dir
    )

    experiment_manager = ExperimentManager(base_path=config.output_dir)
    experiment_id = experiment_manager.create_experiment(exp_config)

    # Setup data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(config)

    # Create model
    model = create_model(config).to(device)

    # Load pretrained weights if specified
    if hasattr(config, 'pretrained_weights') and config.pretrained_weights:
        logger.info(f"Loading pretrained weights from {config.pretrained_weights}")
        checkpoint = torch.load(config.pretrained_weights)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Create loss functions
    classification_loss, reconstruction_loss, kl_loss = create_loss_functions(config, device)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, len(train_loader), stage
    )

    # Setup mixed precision
    scaler = GradScaler() if config.mixed_precision and device.type == 'cuda' else None

    # Setup monitoring and analysis
    training_monitor = TrainingMonitor(config.monitoring_config)
    training_monitor.start_monitoring()

    metrics_calculator = AcademicMetrics(confidence_level=0.95, n_bootstrap=1000)
    reconstruction_analyzer = ReconstructionAnalyzer(config.reconstruction_config)

    # Training loop
    best_val_auc = 0.0
    best_epoch = 0
    train_metrics_history = []
    val_metrics_history = []

    logger.info(f"Starting {stage} training for {config.epochs} epochs")

    start_time = time.time()

    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs} ({stage})")
        logger.info("-" * 50)

        # Training
        train_metrics = train_epoch(
            model, train_loader, classification_loss, reconstruction_loss, kl_loss,
            optimizer, scheduler, config, device, scaler, epoch, stage
        )
        train_metrics_history.append(train_metrics)

        # Validation
        val_metrics = validate_model(
            model, val_loader, device, metrics_calculator, reconstruction_analyzer
        )
        val_metrics_history.append(val_metrics)

        # Update monitoring
        training_monitor.log_epoch_metrics(epoch, train_metrics, val_metrics)

        logger.info(f"Validation AUC: {val_metrics['val_auc']:.4f}")
        logger.info(f"Validation F1: {val_metrics['val_f1']:.4f}")
        logger.info(f"Validation Accuracy: {val_metrics['val_accuracy']:.2f}%")

        if 'val_reconstruction_ssim' in val_metrics:
            logger.info(f"Reconstruction SSIM: {val_metrics['val_reconstruction_ssim']:.4f}")

        # Save best model
        if val_metrics['val_auc'] > best_val_auc:
            best_val_auc = val_metrics['val_auc']
            best_epoch = epoch

            best_model_path = os.path.join(config.weights_dir, f"best_genconvit_{config.variant}_{stage}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': asdict(config),
                'stage': stage
            }, best_model_path)

            logger.info(f"New best model saved: AUC={best_val_auc:.4f}")

    training_time = time.time() - start_time

    # Final test evaluation
    logger.info("\nFinal test evaluation...")
    test_metrics = validate_model(model, test_loader, device, metrics_calculator, reconstruction_analyzer)

    logger.info(f"Test AUC: {test_metrics['val_auc']:.4f}")
    logger.info(f"Test F1: {test_metrics['val_f1']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['val_accuracy']:.2f}%")

    # Stop monitoring
    training_monitor.stop_monitoring()

    # Check targets
    meets_targets = (
        test_metrics['val_auc'] >= config.target_auc and
        test_metrics['val_auc'] >= config.reconstruction_quality_auc - 0.02
    )

    logger.info(f"Meets targets: {meets_targets}")

    # Create experiment result
    result = ExperimentResult(
        experiment_id=experiment_id,
        config=exp_config,
        train_metrics={k: [m[k] for m in train_metrics_history if k in m]
                      for k in train_metrics_history[0].keys() if train_metrics_history},
        val_metrics={k: [m[k] for m in val_metrics_history if k in m]
                    for k in val_metrics_history[0].keys() if val_metrics_history},
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_val_score=best_val_auc,
        total_training_time=training_time,
        model_path=os.path.join(config.weights_dir, f"best_genconvit_{config.variant}_{stage}.pth"),
        success=meets_targets
    )

    # Save experiment result
    experiment_manager.save_experiment_result(result)

    logger.info(f"\n{stage} training completed in {training_time:.2f} seconds")
    logger.info(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")

    return result


def main():
    """Main entry point for GenConViT expert training."""

    parser = argparse.ArgumentParser(description='AWARE-NET Stage 2 GenConViT Expert Training')
    parser.add_argument('--mode', choices=['staged_training', 'joint_training', 'evaluation'],
                       default='staged_training', help='Training mode')
    parser.add_argument('--variant', choices=['ed', 'vae'], default='ed',
                       help='GenConViT variant (ED or VAE)')
    parser.add_argument('--config', type=str, help='Path to training config JSON')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data', help='Path to legacy ImageFolder dataset (fallback)')
    parser.add_argument('--dataset_config', type=str, default=None,
                       help='Dataset configuration JSON for manifest autoloading')
    parser.add_argument('--manifest_dataset', type=str, default=None,
                       help='Dataset key inside dataset configuration for manifest autoloading')
    parser.add_argument('--manifest_mode', type=str, default=None,
                       choices=['original', 'balanced', 'anonymized', 'anonymized_balanced'],
                       help='Manifest variant to use when autoloading')
    parser.add_argument('--train_manifest', type=str, help='Path to training manifest CSV')
    parser.add_argument('--val_manifest', type=str, help='Path to validation manifest CSV')
    parser.add_argument('--test_manifest', type=str, help='Path to test manifest CSV')
    parser.add_argument('--dataset_root', type=str, default='.', help='Root directory for manifest image paths')
    parser.add_argument('--output_dir', type=str, default='experiments/stage_02/genconvit',
                       help='Output directory')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create training config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)

        flattened_config = {
            k: v for k, v in config_dict.items()
            if k in GenConViTTrainingConfig.__dataclass_fields__
        }

        data_section = config_dict.get('data', {})
        for key in (
            'train_manifest',
            'val_manifest',
            'test_manifest',
            'dataset_root',
            'dataset_config',
            'manifest_dataset',
            'manifest_mode',
        ):
            if key in data_section:
                flattened_config[key] = data_section[key]

        config = GenConViTTrainingConfig(**flattened_config)
    else:
        config = GenConViTTrainingConfig()

    # Override with command line arguments
    config.mode = args.mode
    config.variant = args.variant
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.data_path = args.data_path
    config.dataset_config = args.dataset_config or config.dataset_config
    if args.manifest_dataset:
        config.manifest_dataset = args.manifest_dataset
    if args.manifest_mode:
        config.manifest_mode = args.manifest_mode
    config.train_manifest = args.train_manifest or config.train_manifest
    config.val_manifest = args.val_manifest or config.val_manifest
    config.test_manifest = args.test_manifest or config.test_manifest
    config.dataset_root = args.dataset_root or config.dataset_root
    config.output_dir = args.output_dir
    config.seed = args.seed

    if not all([config.train_manifest, config.val_manifest, config.test_manifest]):
        dataset_config_path = Path(config.dataset_config)
        try:
            manifests, dataset_root = resolve_manifest_paths(
                dataset_config_path,
                config.manifest_dataset,
                config.manifest_mode,
            )
        except FileNotFoundError as exc:
            logger.warning("Unable to auto-resolve manifests: %s", exc)
        else:
            config.train_manifest = config.train_manifest or str(manifests['train'])
            config.val_manifest = config.val_manifest or str(manifests['val'])
            config.test_manifest = config.test_manifest or str(manifests['test'])
            if not config.dataset_root or config.dataset_root == '.':
                config.dataset_root = str(dataset_root)
            logger.info(
                "Resolved manifests from %s for %s (%s mode)",
                dataset_config_path,
                config.manifest_dataset,
                config.manifest_mode,
            )

    if not all([config.train_manifest, config.val_manifest, config.test_manifest]):
        data_path = Path(config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                "No manifests provided and fallback data path does not exist: "
                f"{data_path}. Generate manifests or specify manifest paths explicitly."
            )

    if args.checkpoint_path:
        config.pretrained_weights = args.checkpoint_path

    logger.info(f"Starting Stage 2 GenConViT expert training")
    logger.info(f"Mode: {config.mode}, Variant: {config.variant}")

    try:
        if config.mode == 'staged_training':
            results = run_staged_training(config)
            logger.info("Staged training completed successfully")

        elif config.mode == 'joint_training':
            result = train_genconvit_expert(config, stage='joint_training')
            logger.info("Joint training completed successfully")

        elif config.mode == 'evaluation':
            # Evaluation mode - load model and evaluate
            logger.info("Evaluation mode - to be implemented")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    logger.info("Stage 2 GenConViT expert training script finished")


if __name__ == '__main__':
    main()
