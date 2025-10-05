"""
AWARE-NET Stage 2: Spatial Expert Training Script

This script implements the complete training pipeline for the EfficientNetV2-B0
spatial artifact detection expert with advanced optimizations and progressive
validation strategies.

Key Features:
- Progressive validation: concept → resolution comparison → model upgrade decision
- Advanced focal loss with graduated learning rates
- Multi-resolution adaptive training with spatial-preserving augmentations
- Comprehensive monitoring with Grad-CAM visualization
- Academic-grade evaluation with statistical significance testing
- Professional domain specialization validation

Usage:
    python train_stage2_spatial.py --mode concept_validation --epochs 10
    python train_stage2_spatial.py --mode resolution_comparison
    python train_stage2_spatial.py --mode full_training --epochs 50
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
    from .enhanced_spatial_expert import (
        FocalLoss, FocalLossConfig,
        GraduatedLRConfig, GraduatedLRScheduler
    )
    from .spatial_expert import EfficientNetV2SpatialExpert, SpatialExpertConfig, SpatialArtifactType
    from .multi_resolution_dataloader import DataLoaderConfig
    from .data_augmentation import (
        AugmentationFactory, AugmentationConfig,
        EdgePreservingAugmentation, CompressionSimulation
    )
    from .training_monitor import (
        TrainingMonitor, MonitoringConfig, SystemMetrics, TrainingMetrics
    )
    try:
        from .spatial_analysis_tools import GradCAMAnalyzer, SpatialArtifactAnalyzer
    except ImportError:
        GradCAMAnalyzer = SpatialArtifactAnalyzer = None
except ImportError:  # pragma: no cover - script fallback
    from enhanced_spatial_expert import (
        FocalLoss, FocalLossConfig,
        GraduatedLRConfig, GraduatedLRScheduler
    )
    from spatial_expert import EfficientNetV2SpatialExpert, SpatialExpertConfig, SpatialArtifactType
    from multi_resolution_dataloader import DataLoaderConfig
    from data_augmentation import (
        AugmentationFactory, AugmentationConfig,
        EdgePreservingAugmentation, CompressionSimulation
    )
    from training_monitor import (
        TrainingMonitor, MonitoringConfig, SystemMetrics, TrainingMetrics
    )
    try:
        from spatial_analysis_tools import GradCAMAnalyzer, SpatialArtifactAnalyzer
    except ImportError:
        GradCAMAnalyzer = SpatialArtifactAnalyzer = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage2_spatial_training.log'),
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
    """Resolve manifest paths for spatial expert datasets."""

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

@dataclass
class SpatialTrainingConfig:
    """Comprehensive training configuration for spatial expert"""

    # Training mode
    mode: str = 'concept_validation'  # 'concept_validation', 'resolution_comparison', 'full_training'

    # Model configuration
    model_config: SpatialExpertConfig = None

    # Progressive validation settings
    concept_validation: Dict[str, Any] = None
    resolution_comparison: Dict[str, Any] = None
    model_upgrade_decision: Dict[str, Any] = None

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 50
    warmup_epochs: int = 3

    # Loss configuration
    focal_loss_config: FocalLossConfig = None

    # Learning rate configuration
    lr_config: GraduatedLRConfig = None

    # Data configuration
    data_loader_config: DataLoaderConfig = None
    augmentation_config: AugmentationConfig = None
    dataset_config: str = 'configs/datasets.json'
    manifest_dataset: str = 'celebdf_v2'
    manifest_mode: str = 'balanced'

    # Monitoring configuration
    monitoring_config: MonitoringConfig = None

    # Optimization
    mixed_precision: bool = True
    gradient_clipping: float = 1.0

    # Paths
    data_path: str = "data"
    output_dir: str = "experiments/stage_02/spatial"
    weights_dir: str = "models/stage_02/spatial"
    logs_dir: str = "logs/stage_02/spatial"
    config_path: str = "configs/spatial_expert_config.json"
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    test_manifest: Optional[str] = None
    dataset_root: str = "."

    # Performance targets
    target_auc: float = 0.92
    spatial_artifact_auc: float = 0.95
    baseline_improvement: float = 0.03
    max_inference_time: float = 100.0  # ms

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        """Initialize default configurations"""
        if self.model_config is None:
            self.model_config = SpatialExpertConfig()

        if self.focal_loss_config is None:
            self.focal_loss_config = FocalLossConfig()

        if self.lr_config is None:
            self.lr_config = GraduatedLRConfig()

        if self.data_loader_config is None:
            self.data_loader_config = DataLoaderConfig(
                batch_size=self.batch_size,
                expert_type="spatial"
            )

        if self.augmentation_config is None:
            self.augmentation_config = AugmentationConfig(
                spatial_expert_mode=True
            )

        if self.monitoring_config is None:
            self.monitoring_config = MonitoringConfig(
                output_dir=os.path.join(self.logs_dir, "monitoring")
            )

        # Set progressive validation defaults
        if self.concept_validation is None:
            self.concept_validation = {
                'model_variant': 'efficientnetv2_rw_s',
                'resolution': 256,
                'sample_size': 1000,
                'epochs': 10,
                'batch_size': 32,
                'target_auc': 0.92
            }

        if self.resolution_comparison is None:
            self.resolution_comparison = {
                'resolutions': [224, 256, 288, 320],
                'samples_per_resolution': 500,
                'efficiency_weight': 0.3,
                'quality_weight': 0.7
            }

        if self.model_upgrade_decision is None:
            self.model_upgrade_decision = {
                'upgrade_candidates': ['efficientnetv2_rw_m', 'efficientnetv2_b1'],
                'upgrade_threshold_auc': 0.95,
                'cost_benefit_ratio': 2.0
            }


class SpatialExpertDataset(torch.utils.data.Dataset):
    """Dataset supporting manifest-driven loading with optional ImageFolder fallback."""

    def __init__(
        self,
        split: str,
        resolution: int,
        augmentation_config: AugmentationConfig,
        manifest_path: Optional[str] = None,
        dataset_root: str = ".",
        fallback_path: Optional[str] = None
    ):
        self.split = split
        self.resolution = resolution
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.dataset_root = Path(dataset_root)
        self.augmentation_config = augmentation_config or AugmentationConfig(spatial_expert_mode=True)

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
                    "spatial",
                    **self.augmentation_config.__dict__
                )
            else:
                self.augmentation = A.Compose([
                    A.Resize(self.resolution, self.resolution),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

            logger.info(
                "Loaded manifest dataset (%s): samples=%d", self.manifest_path, len(self.data)
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
                    transforms.RandomRotation(5),
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
                tensor = self.augmentation(image_np)
            else:
                tensor = self.augmentation(image=image_np)['image']
            return tensor, torch.tensor(label, dtype=torch.float32)

        image, label = self.dataset[idx]
        tensor = self.transform(image)
        return tensor, torch.tensor(float(label), dtype=torch.float32)

    def get_labels(self) -> List[int]:
        if self.use_manifest:
            return self.data['label'].astype(int).tolist()
        return [self.dataset[i][1] for i in range(len(self.dataset))]


def setup_data_loaders(config: SpatialTrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Setup data loaders with manifest support."""

    resolution = config.concept_validation['resolution'] if config.mode == 'concept_validation' else 256
    use_manifest = all([
        config.train_manifest,
        config.val_manifest,
        config.test_manifest,
    ])

    dataset_root = Path(config.dataset_root)
    if use_manifest and not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    train_dataset = SpatialExpertDataset(
        split='train',
        resolution=resolution,
        augmentation_config=config.augmentation_config,
        manifest_path=config.train_manifest if use_manifest else None,
        dataset_root=dataset_root if use_manifest else Path(config.data_path),
        fallback_path=config.data_path
    )

    val_dataset = SpatialExpertDataset(
        split='val',
        resolution=resolution,
        augmentation_config=config.augmentation_config,
        manifest_path=config.val_manifest if use_manifest else None,
        dataset_root=dataset_root if use_manifest else Path(config.data_path),
        fallback_path=config.data_path
    )

    test_dataset = SpatialExpertDataset(
        split='test',
        resolution=resolution,
        augmentation_config=config.augmentation_config,
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


def create_model(config: SpatialTrainingConfig) -> EfficientNetV2SpatialExpert:
    """Create and initialize the spatial expert model."""

    if config.mode == 'concept_validation':
        # Use smaller model for concept validation
        model_config = config.model_config
        model_config.model_name = config.concept_validation['model_variant']
    else:
        model_config = config.model_config

    model = EfficientNetV2SpatialExpert(model_config)

    # Log model information
    logger.info(f"Spatial expert model created: {model_config.model_name}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


def create_loss_function(config: SpatialTrainingConfig) -> FocalLoss:
    """Create focal loss function with advanced configuration."""

    return FocalLoss(config.focal_loss_config)


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: SpatialTrainingConfig,
    steps_per_epoch: int
) -> Tuple[optim.Optimizer, Any]:
    """Create graduated optimizer and scheduler."""

    # Separate parameter groups for graduated learning rates
    backbone_params = []
    classifier_params = []
    spatial_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'classifier' in name or 'fc' in name:
            classifier_params.append(param)
        else:
            spatial_params.append(param)

    # Create optimizer with graduated learning rates
    param_groups = [
        {
            'params': backbone_params,
            'lr': config.learning_rate * config.lr_config.backbone_lr_multiplier,
            'name': 'backbone'
        },
        {
            'params': classifier_params,
            'lr': config.learning_rate * config.lr_config.classifier_lr_multiplier,
            'name': 'classifier'
        },
        {
            'params': spatial_params,
            'lr': config.learning_rate * config.lr_config.spatial_modules_lr_multiplier,
            'name': 'spatial_modules'
        }
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=1e-5)

    # Create graduated scheduler
    scheduler = GraduatedLRScheduler(
        optimizer=optimizer,
        config=config.lr_config,
        steps_per_epoch=steps_per_epoch,
        total_epochs=config.epochs
    )

    logger.info("Graduated optimizer and scheduler created")
    logger.info(f"Learning rates: backbone={config.learning_rate * config.lr_config.backbone_lr_multiplier:.6f}, "
               f"classifier={config.learning_rate * config.lr_config.classifier_lr_multiplier:.6f}, "
               f"spatial={config.learning_rate * config.lr_config.spatial_modules_lr_multiplier:.6f}")

    return optimizer, scheduler


def train_epoch(
    model: EfficientNetV2SpatialExpert,
    train_loader: DataLoader,
    criterion: FocalLoss,
    optimizer: optim.Optimizer,
    scheduler: Any,
    config: SpatialTrainingConfig,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """Train one epoch with advanced monitoring."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Update focal loss epoch for adaptive alpha
    if hasattr(criterion, 'current_epoch'):
        criterion.current_epoch = epoch
        if criterion.total_epochs is None:
            criterion.total_epochs = config.epochs

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device).float()

        optimizer.zero_grad()

        if scaler and config.mixed_precision:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs.squeeze(), targets)

            scaler.scale(loss).backward()
            if config.gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()

            if config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)

            optimizer.step()

        scheduler.step()

        # Update statistics
        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Log progress
        if batch_idx % 50 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)}: "
                       f"Loss={loss.item():.4f}, "
                       f"Acc={100.*correct/total:.2f}%, "
                       f"LR={scheduler.get_last_lr()[0]:.6f}")

    epoch_metrics = {
        'train_loss': total_loss / len(train_loader),
        'train_accuracy': 100. * correct / total
    }

    return epoch_metrics


def validate_model(
    model: EfficientNetV2SpatialExpert,
    val_loader: DataLoader,
    device: torch.device,
    metrics_calculator: AcademicMetrics,
    grad_cam_analyzer: Optional[GradCAMAnalyzer] = None
) -> Dict[str, float]:
    """Validate model with comprehensive metrics and Grad-CAM analysis."""

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device).float()

            outputs = model(images)
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = (probabilities > 0.5).float()

            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Generate Grad-CAM visualizations for first batch
            if batch_idx == 0 and grad_cam_analyzer is not None:
                try:
                    grad_cam_results = grad_cam_analyzer.generate_spatial_analysis(
                        model=model,
                        images=images[:4],  # Analyze first 4 images
                        targets=targets[:4]
                    )
                    logger.info("Grad-CAM analysis completed for validation batch")
                except Exception as e:
                    logger.warning(f"Grad-CAM analysis failed: {e}")

    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # Calculate comprehensive metrics
    auc_result = metrics_calculator.calculate_auc_with_ci(y_true, y_prob)
    f1_result = metrics_calculator.calculate_f1_with_ci(y_true, y_pred)
    accuracy = np.mean(y_pred == y_true)
    precision = np.mean(y_pred[y_true == 1] == 1) if np.sum(y_true == 1) > 0 else 0
    recall = np.mean(y_pred[y_true == 1] == 1) if np.sum(y_true == 1) > 0 else 0

    metrics = {
        'val_auc': auc_result.value,
        'val_auc_ci_lower': auc_result.confidence_interval[0] if auc_result.confidence_interval else 0,
        'val_auc_ci_upper': auc_result.confidence_interval[1] if auc_result.confidence_interval else 0,
        'val_f1': f1_result.value,
        'val_f1_ci_lower': f1_result.confidence_interval[0] if f1_result.confidence_interval else 0,
        'val_f1_ci_upper': f1_result.confidence_interval[1] if f1_result.confidence_interval else 0,
        'val_accuracy': accuracy * 100,
        'val_precision': precision * 100,
        'val_recall': recall * 100
    }

    return metrics


def run_concept_validation(config: SpatialTrainingConfig) -> ExperimentResult:
    """Run 10-epoch concept validation for rapid decision making."""

    logger.info("Starting concept validation (10 epochs)")

    # Override epochs for concept validation
    concept_config = config
    concept_config.epochs = config.concept_validation['epochs']
    concept_config.batch_size = config.concept_validation['batch_size']

    return train_spatial_expert(concept_config)


def run_resolution_comparison(config: SpatialTrainingConfig) -> Dict[int, ExperimentResult]:
    """Run multi-resolution comparison experiment."""

    logger.info("Starting resolution comparison experiment")

    results = {}
    resolutions = config.resolution_comparison['resolutions']

    for resolution in resolutions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Resolution: {resolution}x{resolution}")
        logger.info(f"{'='*60}")

        # Create resolution-specific config
        res_config = config
        res_config.data_loader_config.default_resolution = resolution
        res_config.epochs = 15  # Shorter training for comparison

        result = train_spatial_expert(res_config)
        results[resolution] = result

        logger.info(f"Resolution {resolution}: AUC={result.test_metrics['val_auc']:.4f}")

    # Analyze results and make recommendation
    best_resolution = max(results.keys(), key=lambda r: results[r].test_metrics['val_auc'])
    logger.info(f"\nBest resolution: {best_resolution}x{best_resolution}")

    return results


def train_spatial_expert(config: SpatialTrainingConfig) -> ExperimentResult:
    """Main training function for spatial expert."""

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
        experiment_name=f"spatial_expert_{config.mode}_{config.epochs}epochs",
        model_name=config.model_config.model_name,
        dataset_name="spatial_artifacts_dataset",
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

    # Create loss function
    criterion = create_loss_function(config)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, len(train_loader)
    )

    # Setup mixed precision
    scaler = GradScaler() if config.mixed_precision and device.type == 'cuda' else None

    # Setup monitoring
    training_monitor = TrainingMonitor(config.monitoring_config)
    training_monitor.start_monitoring()

    # Setup metrics calculator and Grad-CAM analyzer
    metrics_calculator = AcademicMetrics(confidence_level=0.95, n_bootstrap=1000)
    grad_cam_analyzer = None
    if config.model_config.enable_grad_cam and GradCAMAnalyzer is not None:
        grad_cam_analyzer = GradCAMAnalyzer(
            save_dir=os.path.join(config.output_dir, "grad_cam_analysis")
        )

    # Training loop
    best_val_auc = 0.0
    best_epoch = 0
    train_metrics_history = []
    val_metrics_history = []

    logger.info(f"Starting training for {config.epochs} epochs (mode: {config.mode})")

    start_time = time.time()

    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
        logger.info("-" * 50)

        # Training
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            config, device, scaler, epoch
        )
        train_metrics_history.append(train_metrics)

        # Validation
        val_metrics = validate_model(
            model, val_loader, device, metrics_calculator, grad_cam_analyzer
        )
        val_metrics_history.append(val_metrics)

        # Update monitoring
        training_monitor.log_epoch_metrics(epoch, train_metrics, val_metrics)

        logger.info(f"Validation AUC: {val_metrics['val_auc']:.4f} "
                   f"[{val_metrics['val_auc_ci_lower']:.4f}, {val_metrics['val_auc_ci_upper']:.4f}]")
        logger.info(f"Validation F1: {val_metrics['val_f1']:.4f}")
        logger.info(f"Validation Accuracy: {val_metrics['val_accuracy']:.2f}%")

        # Save best model
        if val_metrics['val_auc'] > best_val_auc:
            best_val_auc = val_metrics['val_auc']
            best_epoch = epoch

            best_model_path = os.path.join(config.weights_dir, f"best_spatial_expert_{config.mode}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': asdict(config)
            }, best_model_path)

            logger.info(f"New best model saved: AUC={best_val_auc:.4f}")

        # Check concept validation targets
        if config.mode == 'concept_validation':
            target_auc = config.concept_validation['target_auc']
            if val_metrics['val_auc'] >= target_auc:
                logger.info(f"✅ Concept validation target reached: "
                           f"{val_metrics['val_auc']:.4f} >= {target_auc:.4f}")
                break
            elif epoch >= 5 and val_metrics['val_auc'] < target_auc - 0.05:
                logger.warning(f"⚠️ Concept validation falling behind target: "
                              f"{val_metrics['val_auc']:.4f} vs {target_auc:.4f}")

    training_time = time.time() - start_time

    # Final test evaluation
    logger.info("\nFinal test evaluation...")
    test_metrics = validate_model(model, test_loader, device, metrics_calculator, grad_cam_analyzer)

    logger.info(f"Test AUC: {test_metrics['val_auc']:.4f} "
               f"[{test_metrics['val_auc_ci_lower']:.4f}, {test_metrics['val_auc_ci_upper']:.4f}]")
    logger.info(f"Test F1: {test_metrics['val_f1']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['val_accuracy']:.2f}%")

    # Stop monitoring
    training_monitor.stop_monitoring()

    # Check targets
    meets_targets = (
        test_metrics['val_auc'] >= config.target_auc and
        test_metrics['val_auc'] >= config.spatial_artifact_auc - 0.02  # Allow some tolerance
    )

    logger.info(f"Meets targets: {meets_targets}")
    if meets_targets:
        logger.info("✅ Spatial expert training successful!")
    else:
        logger.warning("⚠️ Spatial expert did not meet all targets")

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
        model_path=os.path.join(config.weights_dir, f"best_spatial_expert_{config.mode}.pth"),
        success=meets_targets
    )

    # Save experiment result
    experiment_manager.save_experiment_result(result)

    logger.info(f"\nTraining completed in {training_time:.2f} seconds")
    logger.info(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")

    return result


def main():
    """Main entry point for spatial expert training."""

    parser = argparse.ArgumentParser(description='AWARE-NET Stage 2 Spatial Expert Training')
    parser.add_argument('--mode', choices=['concept_validation', 'resolution_comparison', 'full_training'],
                       default='concept_validation', help='Training mode')
    parser.add_argument('--config', type=str, help='Path to training config JSON')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
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
    parser.add_argument('--output_dir', type=str, default='experiments/stage_02/spatial',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create training config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)

        flattened_config = {
            k: v for k, v in config_dict.items()
            if k in SpatialTrainingConfig.__dataclass_fields__
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

        config = SpatialTrainingConfig(**flattened_config)
    else:
        config = SpatialTrainingConfig()

    # Override with command line arguments
    config.mode = args.mode
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
                f"{data_path}. Generate manifests via dataset tools or specify --train_manifest/--val_manifest/--test_manifest."
            )

    logger.info(f"Starting Stage 2 spatial expert training with mode: {config.mode}")

    try:
        if config.mode == 'concept_validation':
            result = run_concept_validation(config)
            logger.info("Concept validation completed successfully")

        elif config.mode == 'resolution_comparison':
            results = run_resolution_comparison(config)
            logger.info("Resolution comparison completed successfully")

        elif config.mode == 'full_training':
            result = train_spatial_expert(config)
            logger.info("Full training completed successfully")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    logger.info("Stage 2 spatial expert training script finished")


if __name__ == '__main__':
    main()
