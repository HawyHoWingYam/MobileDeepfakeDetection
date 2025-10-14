"""
AWARE-NET Stage 1: SupCon Training Script

This script implements the complete training pipeline for Stage 1 SupCon-based
rapid filtering system. It supports both SupCon and BCE training modes for
comprehensive comparison and validation.

Key Features:
- Dual training modes: SupCon contrastive learning vs BCE baseline
- Academic-grade evaluation with statistical significance testing
- Comprehensive experiment tracking and reproducibility
- Mobile-optimized MobileNetV4 architecture
- Advanced calibration and cascade strategy validation
- 10-epoch rapid validation for gate decision making

Usage:
    python train_stage1_supcon.py --mode supcon --epochs 50
    python train_stage1_supcon.py --mode comparison --quick_validation
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
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# AWARE-NET imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.experiment_utils import ExperimentManager, ExperimentConfig, ExperimentResult
from utils.metrics import AcademicMetrics, MetricResult, ComparisonResult
from utils.calibration_tools import CalibrationAnalyzer, TemperatureScalingResult
from utils.dataset_config import DatasetConfig

# Stage 1 specific imports
from supcon_loss import SupConLoss, SupConLossWithLogging
from mobilenetv4_model import MobileNetV4SupCon, create_mobilenetv4_supcon
from balanced_sampler import BalancedBatchSampler, ContrastivePairSampler
from temperature_scaling import TemperatureScaling
from evaluate_stage1 import evaluate_model_comprehensive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage1_training.log'),
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
    """Resolve manifest paths for the given dataset using DatasetConfig."""

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
class TrainingConfig:
    """Training configuration for Stage 1"""

    # Model configuration
    model_name: str = 'mobilenetv4_hybrid_medium'
    pretrained: bool = True
    projection_dim: int = 512
    num_classes: int = 2
    dropout_rate: float = 0.2

    # Training parameters
    mode: str = 'supcon'  # 'supcon', 'bce', 'comparison'
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    warmup_epochs: int = 5

    # SupCon specific
    temperature: float = 0.07
    min_samples_per_class: int = 4
    contrastive_weight: float = 1.0
    classification_weight: float = 0.1

    # Data parameters
    image_size: int = 256
    data_path: str = "data"
    dataset_config: str = 'configs/datasets.json'
    manifest_dataset: str = 'celebdf_v2'
    manifest_mode: str = 'balanced'
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    test_manifest: Optional[str] = None
    dataset_root: str = "."
    augmentation: bool = True

    # Optimization
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    gradient_clipping: float = 1.0
    mixed_precision: bool = True

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Validation
    quick_validation: bool = False  # 10-epoch mode
    val_frequency: int = 1
    save_frequency: int = 5

    # Performance targets
    target_auc: float = 0.90
    min_improvement: float = 0.03
    max_inference_time: float = 50.0  # ms
    max_memory_usage: float = 2.0  # GB

    # Paths
    output_dir: str = "experiments/stage_01"
    weights_dir: str = "models/stage_01"
    logs_dir: str = "logs/stage_01"


class Stage1ManifestDataset(Dataset):
    """Manifest-driven dataset supporting optional dual views for SupCon."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        root_dir: Union[str, Path],
        image_size: int = 256,
        augmentation: bool = True,
        contrastive_views: bool = False
    ):
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        self.contrastive_views = contrastive_views

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.data = pd.read_csv(self.manifest_path)
        if 'valid' in self.data.columns:
            self.data = self.data[self.data['valid'] == True]
        self.data = self.data.reset_index(drop=True)

        self.transform = self._create_transform(augmentation)

        logger.info(
            "Manifest dataset loaded: %s | samples=%d | contrastive_views=%s",
            self.manifest_path,
            len(self.data),
            contrastive_views,
        )

    def _create_transform(self, augmentation: bool) -> A.Compose:
        if augmentation:
            return A.Compose([
                A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
                A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        label = int(row['label'])
        image_rel_path = Path(row['image_path'])
        image_path = self.root_dir / image_rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        if self.contrastive_views:
            view1 = self.transform(image=image_np)['image']
            view2 = self.transform(image=image_np)['image']
            return (view1, view2), torch.tensor(label, dtype=torch.long)

        transformed = self.transform(image=image_np)['image']
        return transformed, torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> List[int]:
        return self.data['label'].astype(int).tolist()


class Stage1ImageFolderDataset(Dataset):
    """Legacy ImageFolder-based dataset (kept for backward compatibility)."""

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        contrastive_views: bool = False
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.contrastive_views = contrastive_views
        self.dataset = ImageFolder(root=str(self.data_path / split))
        self.classes = self.dataset.classes
        logger.warning(
            "Using legacy ImageFolder pipeline for Stage 1 (split=%s, samples=%d).",
            split,
            len(self.dataset)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            if self.contrastive_views:
                view1 = self.transform(image)
                view2 = self.transform(image)
                return (view1, view2), torch.tensor(label, dtype=torch.long)
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> List[int]:
        return [self.dataset[i][1] for i in range(len(self.dataset))]


def create_transforms(image_size: int, mode: str = 'train') -> transforms.Compose:
    """Create image transformations for training/validation."""

    if mode == 'train':
        # Training transformations with augmentation for contrastive learning
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transformations
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    return transform


def setup_data_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Setup train, validation, and test data loaders."""

    use_manifest = all([
        config.train_manifest,
        config.val_manifest,
        config.test_manifest,
    ])

    if use_manifest:
        logger.info("Using manifest-driven pipeline for Stage 1 datasets")
        train_dataset = Stage1ManifestDataset(
            manifest_path=config.train_manifest,
            root_dir=config.dataset_root,
            image_size=config.image_size,
            augmentation=True,
            contrastive_views=(config.mode == 'supcon')
        )

        val_dataset = Stage1ManifestDataset(
            manifest_path=config.val_manifest,
            root_dir=config.dataset_root,
            image_size=config.image_size,
            augmentation=False,
            contrastive_views=False
        )

        test_dataset = Stage1ManifestDataset(
            manifest_path=config.test_manifest,
            root_dir=config.dataset_root,
            image_size=config.image_size,
            augmentation=False,
            contrastive_views=False
        )
    else:
        logger.warning(
            "Manifest paths not provided; falling back to legacy ImageFolder pipeline (data_path=%s)",
            config.data_path
        )
        train_transform = create_transforms(config.image_size, mode='train')
        val_transform = create_transforms(config.image_size, mode='val')

        train_dataset = Stage1ImageFolderDataset(
            data_path=config.data_path,
            split='train',
            transform=train_transform,
            contrastive_views=(config.mode == 'supcon')
        )

        val_dataset = Stage1ImageFolderDataset(
            data_path=config.data_path,
            split='val',
            transform=val_transform,
            contrastive_views=False
        )

        test_dataset = Stage1ImageFolderDataset(
            data_path=config.data_path,
            split='test',
            transform=val_transform,
            contrastive_views=False
        )

    # Create balanced sampler for training
    train_labels = train_dataset.get_labels()
    train_sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=config.batch_size,
        min_samples_per_class=config.min_samples_per_class,
        strategy='balanced',
        seed=config.seed
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True
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


def create_model(config: TrainingConfig) -> MobileNetV4SupCon:
    """Create and initialize the model."""

    model = create_mobilenetv4_supcon(
        model_name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        projection_dim=config.projection_dim,
        dropout_rate=config.dropout_rate,
        use_projection_head=(config.mode == 'supcon')
    )

    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model created: {model_info}")

    return model


def create_loss_function(config: TrainingConfig) -> Tuple[nn.Module, Optional[nn.Module]]:
    """Create loss functions based on training mode."""

    if config.mode == 'supcon':
        contrastive_loss = SupConLossWithLogging(
            temperature=config.temperature,
            base_temperature=config.temperature,
            contrast_mode='all',
            numerical_stability=True,
            log_frequency=100
        )
        classification_loss = nn.CrossEntropyLoss()
        return contrastive_loss, classification_loss

    elif config.mode == 'bce':
        classification_loss = nn.CrossEntropyLoss()
        return classification_loss, None

    else:
        raise ValueError(f"Unknown training mode: {config.mode}")


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
    steps_per_epoch: int
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler."""

    # Create optimizer
    if config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Create scheduler
    if config.scheduler.lower() == 'cosine':
        total_steps = steps_per_epoch * config.epochs
        warmup_steps = steps_per_epoch * config.warmup_epochs

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.epochs // 3,
            gamma=0.1
        )
    else:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    return optimizer, scheduler


def train_epoch_supcon(
    model: MobileNetV4SupCon,
    train_loader: DataLoader,
    contrastive_loss: nn.Module,
    classification_loss: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """Train one epoch with SupCon loss."""

    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_classification_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Handle contrastive views
        if isinstance(data, tuple):
            view1, view2 = data
            batch_size = view1.size(0)
            images = torch.cat([view1, view2], dim=0)
            targets = targets.repeat(2)
        else:
            images = data
            batch_size = images.size(0)

        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        if scaler and config.mixed_precision:
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(images, return_projections=True, return_features=True)
                logits = outputs['logits']
                projections = outputs['projections']

                # Compute losses
                cont_loss = contrastive_loss(
                    projections.unsqueeze(1),
                    targets
                ) if projections is not None else 0.0

                class_loss = classification_loss(logits, targets)

                # Combined loss
                loss = (config.contrastive_weight * cont_loss +
                       config.classification_weight * class_loss)

            scaler.scale(loss).backward()
            if config.gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            scaler.step(optimizer)
            scaler.update()

        else:
            # Forward pass
            outputs = model(images, return_projections=True, return_features=True)
            logits = outputs['logits']
            projections = outputs['projections']

            # Compute losses
            cont_loss = contrastive_loss(
                projections.unsqueeze(1),
                targets
            ) if projections is not None else 0.0

            class_loss = classification_loss(logits, targets)

            # Combined loss
            loss = (config.contrastive_weight * cont_loss +
                   config.classification_weight * class_loss)

            loss.backward()

            if config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)

            optimizer.step()

        scheduler.step()

        # Update statistics
        total_loss += loss.item()
        if isinstance(cont_loss, torch.Tensor):
            total_contrastive_loss += cont_loss.item()
        total_classification_loss += class_loss.item()

        # Calculate accuracy
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log progress
        if batch_idx % 50 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)}: "
                       f"Loss={loss.item():.4f}, "
                       f"ContLoss={cont_loss.item() if isinstance(cont_loss, torch.Tensor) else 0:.4f}, "
                       f"ClassLoss={class_loss.item():.4f}, "
                       f"Acc={100.*correct/total:.2f}%")

    epoch_metrics = {
        'train_loss': total_loss / len(train_loader),
        'train_contrastive_loss': total_contrastive_loss / len(train_loader),
        'train_classification_loss': total_classification_loss / len(train_loader),
        'train_accuracy': 100. * correct / total
    }

    return epoch_metrics


def train_epoch_bce(
    model: MobileNetV4SupCon,
    train_loader: DataLoader,
    classification_loss: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """Train one epoch with BCE loss."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        if scaler and config.mixed_precision:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = classification_loss(logits, targets)

            scaler.scale(loss).backward()
            if config.gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(images)
            loss = classification_loss(logits, targets)
            loss.backward()

            if config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)

            optimizer.step()

        scheduler.step()

        # Update statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log progress
        if batch_idx % 50 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)}: "
                       f"Loss={loss.item():.4f}, "
                       f"Acc={100.*correct/total:.2f}%")

    epoch_metrics = {
        'train_loss': total_loss / len(train_loader),
        'train_accuracy': 100. * correct / total
    }

    return epoch_metrics


def validate_model(
    model: MobileNetV4SupCon,
    val_loader: DataLoader,
    device: torch.device,
    metrics_calculator: AcademicMetrics
) -> Dict[str, float]:
    """Validate model and compute comprehensive metrics."""

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_predictions.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

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

    return metrics


def save_checkpoint(
    model: MobileNetV4SupCon,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainingConfig,
    checkpoint_path: str
):
    """Save model checkpoint."""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': config.__dict__,
        'model_info': model.get_model_info()
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def train_model(config: TrainingConfig) -> ExperimentResult:
    """Main training function."""

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
        experiment_name=f"stage1_{config.mode}_{config.epochs}epochs",
        model_name=config.model_name,
        dataset_name="aware_net_dataset",
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_epochs=config.epochs if not config.quick_validation else 10,
        model_params={
            'projection_dim': config.projection_dim,
            'temperature': config.temperature,
            'mode': config.mode
        },
        seed=config.seed,
        output_path=config.output_dir
    )

    experiment_manager = ExperimentManager(base_path=config.output_dir)
    experiment_id = experiment_manager.create_experiment(exp_config)

    # Setup data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(config)

    # Create model
    model = create_model(config).to(device)

    # Create loss functions
    if config.mode == 'supcon':
        contrastive_loss, classification_loss = create_loss_function(config)
    else:
        classification_loss, _ = create_loss_function(config)
        contrastive_loss = None

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, len(train_loader)
    )

    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and device.type == 'cuda' else None

    # Setup metrics calculator
    metrics_calculator = AcademicMetrics(confidence_level=0.95, n_bootstrap=1000)

    # Training loop
    best_val_auc = 0.0
    best_epoch = 0
    train_metrics_history = []
    val_metrics_history = []

    num_epochs = 10 if config.quick_validation else config.epochs
    logger.info(f"Starting training for {num_epochs} epochs (mode: {config.mode})")

    start_time = time.time()

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 50)

        # Training
        if config.mode == 'supcon':
            train_metrics = train_epoch_supcon(
                model, train_loader, contrastive_loss, classification_loss,
                optimizer, scheduler, config, device, scaler
            )
        else:
            train_metrics = train_epoch_bce(
                model, train_loader, classification_loss,
                optimizer, scheduler, config, device, scaler
            )

        train_metrics_history.append(train_metrics)

        # Validation
        if (epoch + 1) % config.val_frequency == 0:
            val_metrics = validate_model(model, val_loader, device, metrics_calculator)
            val_metrics_history.append(val_metrics)

            logger.info(f"Validation AUC: {val_metrics['val_auc']:.4f} "
                       f"[{val_metrics['val_auc_ci_lower']:.4f}, {val_metrics['val_auc_ci_upper']:.4f}]")
            logger.info(f"Validation F1: {val_metrics['val_f1']:.4f} "
                       f"[{val_metrics['val_f1_ci_lower']:.4f}, {val_metrics['val_f1_ci_upper']:.4f}]")
            logger.info(f"Validation Accuracy: {val_metrics['val_accuracy']:.2f}%")

            # Save best model
            if val_metrics['val_auc'] > best_val_auc:
                best_val_auc = val_metrics['val_auc']
                best_epoch = epoch

                best_model_path = os.path.join(config.weights_dir, f"best_model_{config.mode}.pth")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    config, best_model_path
                )

        # Save periodic checkpoint
        if (epoch + 1) % config.save_frequency == 0:
            checkpoint_path = os.path.join(config.weights_dir, f"checkpoint_epoch_{epoch+1}.pth")
            current_val_metrics = val_metrics_history[-1] if val_metrics_history else {}
            save_checkpoint(
                model, optimizer, scheduler, epoch, current_val_metrics,
                config, checkpoint_path
            )

    training_time = time.time() - start_time

    # Final test evaluation
    logger.info("\nFinal test evaluation...")
    test_metrics = validate_model(model, test_loader, device, metrics_calculator)

    logger.info(f"Test AUC: {test_metrics['val_auc']:.4f} "
               f"[{test_metrics['val_auc_ci_lower']:.4f}, {test_metrics['val_auc_ci_upper']:.4f}]")
    logger.info(f"Test F1: {test_metrics['val_f1']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['val_accuracy']:.2f}%")

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
        model_path=os.path.join(config.weights_dir, f"best_model_{config.mode}.pth")
    )

    # Save experiment result
    experiment_manager.save_experiment_result(result)

    logger.info(f"\nTraining completed in {training_time:.2f} seconds")
    logger.info(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")

    return result


def run_comparison_experiment(config: TrainingConfig) -> Dict[str, ExperimentResult]:
    """Run SupCon vs BCE comparison experiment."""

    logger.info("Starting SupCon vs BCE comparison experiment")

    results = {}

    # Train SupCon model
    logger.info("\n" + "="*60)
    logger.info("Training SupCon Model")
    logger.info("="*60)

    supcon_config = config
    supcon_config.mode = 'supcon'
    supcon_result = train_model(supcon_config)
    results['supcon'] = supcon_result

    # Train BCE baseline
    logger.info("\n" + "="*60)
    logger.info("Training BCE Baseline")
    logger.info("="*60)

    bce_config = config
    bce_config.mode = 'bce'
    bce_result = train_model(bce_config)
    results['bce'] = bce_result

    # Compare results
    logger.info("\n" + "="*60)
    logger.info("Comparison Results")
    logger.info("="*60)

    supcon_auc = supcon_result.test_metrics['val_auc']
    bce_auc = bce_result.test_metrics['val_auc']
    improvement = supcon_auc - bce_auc

    logger.info(f"SupCon AUC: {supcon_auc:.4f}")
    logger.info(f"BCE AUC: {bce_auc:.4f}")
    logger.info(f"Improvement: {improvement:.4f} ({improvement/bce_auc*100:.1f}%)")

    # Check if improvement meets threshold
    if improvement >= config.min_improvement:
        logger.info(f"✅ SupCon shows significant improvement (>{config.min_improvement:.3f})")
        gate_decision = "PROCEED"
    else:
        logger.info(f"⚠️ SupCon improvement below threshold (<{config.min_improvement:.3f})")
        gate_decision = "CONDITIONAL"

    # Save comparison report
    comparison_report = {
        'supcon_metrics': supcon_result.test_metrics,
        'bce_metrics': bce_result.test_metrics,
        'improvement': improvement,
        'improvement_percentage': improvement/bce_auc*100,
        'meets_threshold': improvement >= config.min_improvement,
        'gate_decision': gate_decision,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    report_path = os.path.join(config.output_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)

    logger.info(f"Comparison report saved: {report_path}")

    return results


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description='AWARE-NET Stage 1 Training')
    parser.add_argument('--mode', choices=['supcon', 'bce', 'comparison'],
                       default='supcon', help='Training mode')
    parser.add_argument('--config', type=str, help='Path to training config JSON')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--quick_validation', action='store_true',
                       help='Quick 10-epoch validation mode')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
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
    parser.add_argument('--dataset_root', type=str, default='.',
                       help='Root directory for manifest-based datasets')
    parser.add_argument('--output_dir', type=str, default='experiments/stage_01',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create training config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()

    # Override with command line arguments
    config.mode = args.mode
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.quick_validation = args.quick_validation
    config.data_path = args.data_path
    config.train_manifest = args.train_manifest or config.train_manifest
    config.val_manifest = args.val_manifest or config.val_manifest
    config.test_manifest = args.test_manifest or config.test_manifest
    config.dataset_config = args.dataset_config or config.dataset_config
    if args.manifest_dataset:
        config.manifest_dataset = args.manifest_dataset
    if args.manifest_mode:
        config.manifest_mode = args.manifest_mode
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

    logger.info(f"Starting Stage 1 training with config: {config}")

    try:
        if config.mode == 'comparison':
            results = run_comparison_experiment(config)
            logger.info("Comparison experiment completed successfully")
        else:
            result = train_model(config)
            logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    logger.info("Stage 1 training script finished")


if __name__ == '__main__':
    main()
