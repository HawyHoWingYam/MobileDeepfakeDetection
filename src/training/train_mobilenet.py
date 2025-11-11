#!/usr/bin/env python3
"""
AWARE-NET Stage 01: Simple MobileNetV4 Training Script

This script implements a complete training pipeline for Stage 01 with:
- Simple MobileNetV4 architecture (BCE classification) - Stage 01 requirement
- Full multi-dataset support with balanced weighting - Stage 01 requirement
- Stage 00's excellent experiment tracking framework - integrated from utils
- Automatic TensorBoard logging and best model saving
- Compliant with stage_01.md requirements

Usage:
    python train_mobilenet.py --epochs 50 --lr 0.001 --batch_size 32
    python train_mobilenet.py --model_name mobilenetv4_hybrid_medium --freeze_backbone
"""

# Add src to path for imports
import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))
import json
import time
import datetime
import argparse
import random
import copy
from collections import deque
from typing import Dict, List, Any, Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    Dataset,
    ConcatDataset,
    WeightedRandomSampler,
    Sampler,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

# Stage 00: Import experiment framework
from utils.experiment_framework import (
    ExperimentFramework,
    setup_reproducible_environment,
    load_config,
    merge_configs,
)

# Phase 1: Import new experiment management system
from utils.experiment_manager import ExperimentManager
from utils.tracker import LocalTracker, WandbTracker

# Stage 01: Import model
from models.mobilenetv4_model import create_mobilenetv4_simple

# Stage 01: Import data processing
from training.dataset import CelebDFDataset, create_data_loaders

# Stage 00: Import evaluation
from utils.evaluation import ModelEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("train_mobilenet")


def create_multi_dataset_loader(
    config_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    override_image_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Stage 01: Create multi-dataset data loaders with equal weight balancing.

    This function implements 4-dataset equal-weight training without external dependencies.
    Uses PyTorch's ConcatDataset and WeightedRandomSampler for balanced multi-dataset learning.

    Args:
        config_path: Path to datasets.json configuration file
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader) with 4-dataset equal weighting
    """
    logger.info("Stage 01: Loading 4 datasets with equal weight balancing...")

    # Load configuration
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        datasets_config = json.load(f)

    # Define the 4 datasets to include
    dataset_names = ['celebdf_v2', 'faceforensics', 'deeperforensics', 'dfdc']
    datasets_config = datasets_config.get('datasets', {})

    # Create datasets for each split
    train_datasets = []
    val_datasets = []
    test_datasets = []
    dataset_info = {}

    for dataset_name in dataset_names:
        if dataset_name not in datasets_config:
            logger.warning(f"Dataset {dataset_name} not found in configuration, skipping")
            continue

        dataset_cfg = datasets_config[dataset_name]
        if not dataset_cfg.get('enabled', True):
            logger.info(f"Dataset {dataset_name} disabled, skipping")
            continue

        # Get paths
        root_path = Path(dataset_cfg.get('root_path', '.'))
        splits = dataset_cfg.get('splits', {})
        image_size = dataset_cfg.get('metadata', {}).get('image_size', [256])[0]
        if override_image_size is not None:
            image_size = int(override_image_size)

        # Create datasets
        try:
            train_manifest = root_path / splits.get('train', f'manifests/{dataset_name}_train_balanced.csv')
            val_manifest = root_path / splits.get('val', f'manifests/{dataset_name}_val_balanced.csv')
            test_manifest = root_path / splits.get('test', f'manifests/{dataset_name}_test_balanced.csv')

            logger.info(f"Loading {dataset_name}:")
            logger.info(f"  Train: {train_manifest}")
            logger.info(f"  Val: {val_manifest}")
            logger.info(f"  Test: {test_manifest}")

            # Create dataset instances
            train_ds = CelebDFDataset(
                manifest_path=train_manifest,
                root_path=root_path,
                image_size=image_size,
                augmentation=True,  # Enable augmentation for training
                normalize=True
            )

            val_ds = CelebDFDataset(
                manifest_path=val_manifest,
                root_path=root_path,
                image_size=image_size,
                augmentation=False,
                normalize=True
            )

            test_ds = CelebDFDataset(
                manifest_path=test_manifest,
                root_path=root_path,
                image_size=image_size,
                augmentation=False,
                normalize=True
            )

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

            # Store dataset info for weight calculation
            dataset_info[dataset_name] = {
                'train_size': len(train_ds),
                'val_size': len(val_ds),
                'test_size': len(test_ds)
            }

            logger.info(f"  ✓ {dataset_name}: Train={len(train_ds):,}, Val={len(val_ds):,}, Test={len(test_ds):,}")

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

    if not train_datasets:
        raise RuntimeError("No datasets could be loaded successfully")

    logger.info(f"Successfully loaded {len(train_datasets)} datasets for multi-dataset training")

    # Calculate equal weights for each dataset
    # Each dataset gets equal contribution regardless of size
    total_datasets = len(train_datasets)
    dataset_weights = []
    sample_weights = []

    for i, (dataset_name, info) in enumerate(dataset_info.items()):
        # Each dataset gets equal weight = 1/total_datasets
        dataset_weight = 1.0 / total_datasets
        # Each sample within the dataset gets equal share of the dataset weight
        sample_weight = dataset_weight / info['train_size']

        logger.info(f"Dataset {dataset_name}: weight={dataset_weight:.3f}, samples={info['train_size']:,}, per_sample_weight={sample_weight:.6f}")

        # Add weights for all samples in this dataset
        dataset_weights.extend([sample_weight] * info['train_size'])

    # Create weighted sampler for equal dataset contribution
    train_sampler = WeightedRandomSampler(
        weights=dataset_weights,
        num_samples=len(dataset_weights),
        replacement=True
    )

    # Combine datasets
    train_combined = ConcatDataset(train_datasets)
    val_combined = ConcatDataset(val_datasets)
    test_combined = ConcatDataset(test_datasets)

    logger.info(f"Combined dataset sizes:")
    logger.info(f"  Train: {len(train_combined):,} samples from {total_datasets} datasets")
    logger.info(f"  Val: {len(val_combined):,} samples")
    logger.info(f"  Test: {len(test_combined):,} samples")
    logger.info(f"Equal weight balancing: Each dataset contributes {1/total_datasets:.1%} of training batches")

    # Create data loaders
    train_loader = DataLoader(
        train_combined,
        batch_size=batch_size,
        sampler=train_sampler,  # Use weighted sampler for equal dataset contribution
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def create_ood_evaluation_loader(
    config_path: str,
    dataset_names: list[str],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    override_image_size: Optional[int] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Create evaluation loaders for Out-of-Distribution datasets (e.g., Deepfake-Eval-2024).

    This function loads only validation and test splits (no training data).
    Useful for final evaluation on unseen data sources.

    Args:
        config_path: Path to datasets.json configuration file
        dataset_names: List of dataset names to load (e.g., ['deepfake_eval_2024'])
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        override_image_size: Override image size from config

    Returns:
        Tuple of (val_loader, test_loader), either can be None if not configured
    """
    logger.info(f"Creating OOD evaluation loaders for datasets: {dataset_names}")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        datasets_config = json.load(f)

    datasets_config = datasets_config.get('datasets', {})

    val_datasets = []
    test_datasets = []

    for dataset_name in dataset_names:
        if dataset_name not in datasets_config:
            logger.warning(f"Dataset {dataset_name} not found in configuration, skipping")
            continue

        dataset_cfg = datasets_config[dataset_name]
        if not dataset_cfg.get('enabled', True):
            logger.info(f"Dataset {dataset_name} disabled, skipping")
            continue

        root_path = Path(dataset_cfg.get('root_path', '.'))
        splits = dataset_cfg.get('splits', {})
        image_size = dataset_cfg.get('metadata', {}).get('image_size', [256])[0]
        if override_image_size is not None:
            image_size = int(override_image_size)

        try:
            # Only load val and test splits (no training for OOD eval)
            val_manifest = root_path / splits.get('val')
            test_manifest = root_path / splits.get('test')

            logger.info(f"Loading OOD dataset {dataset_name}:")

            # Val split
            if val_manifest and val_manifest.exists():
                logger.info(f"  Val: {val_manifest}")
                val_ds = CelebDFDataset(
                    manifest_path=val_manifest,
                    root_path=root_path,
                    image_size=image_size,
                    augmentation=False,
                    normalize=True
                )
                val_datasets.append(val_ds)
                logger.info(f"  Val: {len(val_ds):,} samples")

            # Test split
            if test_manifest and test_manifest.exists():
                logger.info(f"  Test: {test_manifest}")
                test_ds = CelebDFDataset(
                    manifest_path=test_manifest,
                    root_path=root_path,
                    image_size=image_size,
                    augmentation=False,
                    normalize=True
                )
                test_datasets.append(test_ds)
                logger.info(f"  Test: {len(test_ds):,} samples")

        except Exception as e:
            logger.error(f"Failed to load OOD dataset {dataset_name}: {e}")
            continue

    # Create loaders
    val_loader = None
    test_loader = None

    if val_datasets:
        val_combined = ConcatDataset(val_datasets)
        val_loader = DataLoader(
            val_combined,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        logger.info(f"✓ Validation loader: {len(val_combined):,} samples")

    if test_datasets:
        test_combined = ConcatDataset(test_datasets)
        test_loader = DataLoader(
            test_combined,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        logger.info(f"✓ Test loader: {len(test_combined):,} samples")

    return val_loader, test_loader


def str2bool(value: Any) -> bool:
    """Return a boolean from common string representations."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


class SimpleMobileNetV4Trainer:
    """
    Simple MobileNetV4 trainer with integrated Stage 00 experiment framework.

    Combines Stage 00's excellent experiment tracking with Stage 01's
    simple MobileNetV4 architecture and multi-dataset balancing.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv4_hybrid_medium",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 50,
        device: Union[str, torch.device] = "auto",
        output_dir: str = "outputs/stage1",
        use_dataset_weights: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2,
        seed: int = 42,
        patience: int = 5,
        min_delta: float = 0.001,
        restore_best: bool = True,
        optimizer_name: str = 'adamw',
        weight_decay: float = 1e-4,
        num_workers: int = 12,
        image_size_override: Optional[int] = None,
        save_on_metric: str = 'auc',
        warmup_epochs: int = 0,
    ):
        """
        Initialize trainer using ExperimentFramework (Stage 00 integration).

        Args:
            model_name: MobileNetV4 variant to use
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Device to use ('auto' for automatic detection)
            output_dir: Output directory for results
            use_dataset_weights: Whether to balance dataset contributions (Stage 01 requirement)
            freeze_backbone: Whether to freeze backbone weights
            dropout_rate: Dropout rate for model
            seed: Random seed for reproducibility
            patience: Early stopping patience (epochs without improvement). Set to 0 to disable.
            min_delta: Minimum improvement in validation AUC to reset patience counter
            restore_best: Whether to restore best model weights after early stopping
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_dataset_weights = use_dataset_weights
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.optimizer_name = (optimizer_name or 'adamw').lower()
        self.weight_decay = float(weight_decay)
        self.num_workers = int(num_workers)
        self.image_size_override = image_size_override
        self.save_on_metric = (save_on_metric or 'auc').lower()
        self.warmup_epochs = int(warmup_epochs)

        # Stage 00: Initialize experiment framework with timestamped directories
        # and TensorBoard logging capabilities
        self.framework = ExperimentFramework(
            output_dir=output_dir, experiment_name="mobilenetv4_simple"
        )

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Stage 01: Initialize model (simple MobileNetV4 for BCE classification)
        self.model = create_mobilenetv4_simple(
            model_name=model_name,
            pretrained=True,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        
        # Optimization controls (safer defaults)
        self.learning_rate = learning_rate
        self.use_two_lrs = True
        self.backbone_lr_scale = 0.1
        self.grad_clip_norm = 1.0
        self.gradient_log_interval = 1

        # Stage 00: Initialize optimizer and loss
        # Create parameter groups so backbone uses a smaller LR (mitigates representation drift)
        named_params = list(self.model.named_parameters())
        backbone_params = [p for n, p in named_params if 'backbone' in n and p.requires_grad]
        head_params = [p for n, p in named_params if 'classifier' in n and p.requires_grad]

        def make_optimizer(param_groups):
            if self.optimizer_name == 'adamw':
                return optim.AdamW(param_groups, weight_decay=self.weight_decay)
            if self.optimizer_name == 'adam':
                return optim.Adam(param_groups, weight_decay=self.weight_decay)
            if self.optimizer_name == 'sgd':
                return optim.SGD(param_groups, momentum=0.9, weight_decay=self.weight_decay, nesterov=False)
            logger.warning("Unknown optimizer name, falling back to AdamW")
            return optim.AdamW(param_groups, weight_decay=self.weight_decay)

        if self.use_two_lrs and (len(backbone_params) > 0 and len(head_params) > 0):
            param_groups = [
                {'params': backbone_params, 'lr': self.learning_rate * self.backbone_lr_scale},
                {'params': head_params,    'lr': self.learning_rate},
            ]
            self.optimizer = make_optimizer(param_groups)
            logger.info("Optimizer param groups: backbone_lr=%g, head_lr=%g, wd=%g", self.learning_rate * self.backbone_lr_scale, self.learning_rate, self.weight_decay)
        else:
            # Fallback to single LR
            param_groups = [{'params': self.model.parameters(), 'lr': self.learning_rate}]
            self.optimizer = make_optimizer(param_groups)
            logger.info("Optimizer single LR: lr=%g, wd=%g", self.learning_rate, self.weight_decay)

        self.criterion = nn.BCEWithLogitsLoss()


        # Stage 01: CRITICAL - Verify optimizer parameters after model initialization
        logger.info("=== OPTIMIZER PARAMETER VALIDATION ===")

        # Count parameters by component
        total_params = 0
        trainable_params = 0
        backbone_params = 0
        classifier_params = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'backbone' in name:
                    backbone_params += param.numel()
                elif 'classifier' in name:
                    classifier_params += param.numel()

        logger.info(f"📊 Model Parameter Breakdown:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Backbone trainable: {backbone_params:,}")
        logger.info(f"   Classifier trainable: {classifier_params:,}")

        # CRITICAL FIX: Accurate optimizer parameter counting
        logger.info("🔍 ENHANCED OPTIMIZER PARAMETER ANALYSIS...")

        all_optimizer_params = []
        optimizer_param_details = []

        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            group_params = param_group['params']
            group_param_count = len(group_params)

            logger.info(f"   📦 Optimizer group {group_idx}: {group_param_count} parameters")

            # Count total elements in this group
            group_elements = sum(p.numel() for p in group_params)
            logger.info(f"      Elements in group {group_idx}: {group_elements:,}")

            # Add to our comprehensive lists
            all_optimizer_params.extend(group_params)

            # Store details for verification
            for param_idx, param in enumerate(group_params):
                optimizer_param_details.append({
                    'group': group_idx,
                    'index': param_idx,
                    'shape': param.shape,
                    'numel': param.numel(),
                    'requires_grad': param.requires_grad
                })

        total_optimizer_params = len(all_optimizer_params)
        total_optimizer_elements = sum(p.numel() for p in all_optimizer_params)

        logger.info(f"   📊 OPTIMIZER PARAMETER SUMMARY:")
        logger.info(f"      Total parameter tensors: {total_optimizer_params}")
        logger.info(f"      Total trainable elements: {total_optimizer_elements:,}")

        # Detailed breakdown by parameter sizes
        size_distribution = {}
        for detail in optimizer_param_details:
            size_key = str(detail['shape'])
            if size_key not in size_distribution:
                size_distribution[size_key] = {'count': 0, 'elements': 0}
            size_distribution[size_key]['count'] += 1
            size_distribution[size_key]['elements'] += detail['numel']

        logger.info(f"   📈 PARAMETER SIZE DISTRIBUTION:")
        for size, info in sorted(size_distribution.items()):
            logger.info(f"      Shape {size}: {info['count']} tensors, {info['elements']:,} elements")

        # Compare with model parameters
        model_params = list(self.model.parameters())
        model_param_count = len(model_params)
        model_elements = sum(p.numel() for p in model_params)

        logger.info(f"   🔍 MODEL VS OPTIMIZER COMPARISON:")
        logger.info(f"      Model: {model_param_count} tensors, {model_elements:,} elements")
        logger.info(f"      Optimizer: {total_optimizer_params} tensors, {total_optimizer_elements:,} elements")

        # Check for consistency
        if model_param_count == total_optimizer_params and model_elements == total_optimizer_elements:
            logger.info("✅ PERFECT MATCH: Model and optimizer parameters are identical!")
        else:
            logger.warning("⚠️  PARAMETER MISMATCH DETECTED:")
            if model_param_count != total_optimizer_params:
                logger.warning(f"   Tensor count mismatch: Model={model_param_count}, Optimizer={total_optimizer_params}")
            if model_elements != total_optimizer_elements:
                logger.warning(f"   Element count mismatch: Model={model_elements:,}, Optimizer={total_optimizer_elements:,}")
            logger.warning("   This may indicate optimizer setup issues!")

        logger.info(f"   Total optimizer parameters (corrected): {total_optimizer_params}")

        # CRITICAL: Verify classifier parameters are in optimizer using robust detection
        logger.info("🔍 ENHANCED CLASSIFIER PARAMETER DETECTION...")

        # Method 1: Get classifier parameters by name from model
        model_classifier_params = []
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                model_classifier_params.append((name, param.shape, param.numel()))
                logger.info(f"   🏷️  Model classifier param: {name} | Shape: {param.shape} | Count: {param.numel()}")

        logger.info(f"   📊 Total classifier params in model: {len(model_classifier_params)}")

        # Method 2: Check optimizer parameters by matching with model classifier params
        classifier_found_in_optimizer = 0
        optimizer_classifier_details = []

        for model_name, model_shape, model_count in model_classifier_params:
            found = False
            for opt_param in all_optimizer_params:
                if opt_param.shape == model_shape and opt_param.numel() == model_count:
                    classifier_found_in_optimizer += 1
                    optimizer_classifier_details.append((model_name, model_shape, "✅ FOUND"))
                    found = True
                    break

            if not found:
                optimizer_classifier_details.append((model_name, model_shape, "❌ MISSING"))

        # Method 3: Direct parameter object comparison (most reliable)
        model_param_objects = {name: param for name, param in self.model.named_parameters() if 'classifier' in name}
        optimizer_param_objects = set(all_optimizer_params)  # Use corrected parameter list

        direct_matches = 0
        for name, param in model_param_objects.items():
            if param in optimizer_param_objects:
                direct_matches += 1
                logger.info(f"   ✅ Direct match found: {name}")

        logger.info(f"   🔢 Detection results:")
        logger.info(f"      Model classifier params: {len(model_param_objects)}")
        logger.info(f"      Shape-matched in optimizer: {classifier_found_in_optimizer}")
        logger.info(f"      Direct object matches: {direct_matches}")

        # Detailed breakdown
        logger.info("   📋 Detailed parameter verification:")
        for name, shape, status in optimizer_classifier_details:
            logger.info(f"      {status} | {name} | {shape}")

        # Final determination using most reliable method (direct object matching)
        if direct_matches >= 2:  # Should find weight + bias
            logger.info("✅ SUCCESS: Classifier parameters properly tracked by optimizer!")
            logger.info(f"   📈 Direct object matches: {direct_matches}/{len(model_param_objects)}")
            logger.info("   🎯 Training will correctly update classifier weights!")
        else:
            logger.error("🚨 CRITICAL ERROR: Classifier parameters NOT properly tracked by optimizer!")
            logger.error(f"   Expected {len(model_param_objects)} classifier params, found {direct_matches}")
            logger.error("   🔧 Training will NOT update classifier weights properly!")
            logger.error("   🆘 This will prevent the model from learning!")

        # Verify parameter count consistency (using corrected counts)
        if total_optimizer_params == model_param_count:
            logger.info("✅ Optimizer tracks all model parameters correctly")
            logger.info(f"   📊 Verified: {total_optimizer_params} tensors in both model and optimizer")
        else:
            logger.error("🚨 MISMATCH: Optimizer parameter count doesn't match model parameters!")
            logger.error(f"   Model parameters: {model_param_count}")
            logger.error(f"   Optimizer parameters: {total_optimizer_params}")
            logger.error("   🔧 This could cause training issues!")

        logger.info("=== OPTIMIZER VALIDATION COMPLETE ===")

        # Training state
        self.best_auc = 0.0
        self.train_losses = deque(maxlen=100)
        self.val_losses = deque(maxlen=100)

        # Stage 01: Enhanced monitoring options
        self._log_gradients = True  # Enable gradient monitoring
        self._prev_train_auc = None
        self._training_start_time = None

        logger.info("🔧 Enhanced monitoring enabled:")
        logger.info("   ✅ Gradient flow tracking")
        logger.info("   ✅ Learning progress monitoring")
        logger.info("   ✅ Parameter update verification")

        # Stage 01: Enhanced early stopping setup
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.patience_counter = 0
        self.best_model_state = None
        self.best_epoch = 0

        # Full training history for plots (not bounded like deques)
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_auc_history = []
        self.val_auc_history = []

        # Per-epoch metrics consolidation (PR-2: Milestone 3)
        self.epoch_metrics_history = []

        if patience > 0:
            logger.info("🔧 Early stopping enabled:")
            logger.info(f"   ⏱️  Patience: {patience} epochs")
            logger.info(f"   📊 Min Delta: {min_delta}")
            logger.info(f"   🔄 Restore best weights: {restore_best}")
        else:
            logger.info("⏭️  Early stopping disabled (patience=0)")

    def setup_data_loaders(self):
        """
        Setup data loaders with multi-dataset support and balancing.

        Stage 00: Multi-dataset handling with weight balancing
        Stage 01: Use all 4 datasets with equal weight and true/fake balance
        """
        logger.info("Setting up multi-dataset data loaders with 4 datasets...")

        # Stage 01: Use our own multi-dataset training function
        # This provides equal weight balancing across all 4 datasets without external dependencies
        self.train_loader, self.val_loader, self.test_loader = create_multi_dataset_loader(
            config_path="configs/datasets.json",
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            seed=self.seed,
            override_image_size=self.image_size_override,
        )

        logger.info(f"✅ Effective batch size confirmed: {self.train_loader.batch_size}")
        logger.info(f"   Training batches per epoch: {len(self.train_loader)}")
        logger.info(f"   Validation batches per epoch: {len(self.val_loader)}")
        logger.info(f"Data loaders ready")

        # Stage 01: Debug - Verify 4-dataset loading and class balance
        logger.info("=== DEBUG: 4-Dataset Training Analysis ===")

        # Sample a few batches to verify balance
        sample_count = 0
        class_counts = torch.tensor([0, 0])  # [real, fake]

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if sample_count >= 100:  # Check first 100 samples
                break

            batch_class_counts = torch.bincount(targets.long(), minlength=2)
            class_counts += batch_class_counts
            sample_count += len(targets)

            if batch_idx < 3:  # Log first 3 batches details
                real_pct = (targets == 0).float().mean().item() * 100
                fake_pct = (targets == 1).float().mean().item() * 100
                logger.info(f"Batch {batch_idx}: {len(targets)} samples - Real: {real_pct:.1f}%, Fake: {fake_pct:.1f}%")

        real_pct = class_counts[0].item() / class_counts.sum().item() * 100
        fake_pct = class_counts[1].item() / class_counts.sum().item() * 100

        logger.info(f"First {sample_count} samples - Real: {real_pct:.1f}%, Fake: {fake_pct:.1f}%")
        logger.info(f"Expected balance: 50% Real, 50% Fake (within margin)")
        logger.info("=== END DEBUG ANALYSIS ===")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        all_predictions = []
        all_targets = []

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.float().to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            # Gradient clipping to avoid destabilizing updates
            if getattr(self, 'grad_clip_norm', 0) and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1

            # Store predictions for metrics
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                all_predictions.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            current_loss = epoch_loss / num_batches
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        # Calculate epoch metrics
        avg_loss = epoch_loss / max(num_batches, 1)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Stage 01: Enhanced training metrics logging
        logger.info(f"📈 Training Epoch {epoch+1} Results:")
        logger.info(f"   Total batches processed: {num_batches}")
        logger.info(f"   Average batch loss: {avg_loss:.6f}")
        logger.info(f"   Total samples processed: {len(all_predictions)}")
        logger.info(f"   Prediction range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
        pos_ratio = float(np.mean(all_targets))
        neg_ratio = 1.0 - pos_ratio
        logger.info(f"   Target distribution: Real={neg_ratio:.3f}, Fake={pos_ratio:.3f}")

        # Check gradient flow (critical for verifying classifier is learning)
        if hasattr(self, '_log_gradients') and (epoch % self.gradient_log_interval == 0):  # Every 5 epochs
            classifier_grad_norm = 0
            backbone_grad_norm = 0

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'classifier' in name:
                        classifier_grad_norm += grad_norm
                    elif 'backbone' in name:
                        backbone_grad_norm += grad_norm

            logger.info(f"📊 Gradient norms (Epoch {epoch+1}):")
            logger.info(f"   Classifier: {classifier_grad_norm:.6f}")
            logger.info(f"   Backbone: {backbone_grad_norm:.6f}")

            if classifier_grad_norm < 1e-8:
                logger.warning("⚠️  Very low classifier gradients - potential training issue!")

        # Compute training metrics from in-epoch collected data (no re-evaluation needed)
        logger.info("📊 Computing training metrics from in-epoch data (single pass - efficient)...")
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

        # Threshold predictions for binary classification
        pred_binary = (all_predictions > 0.5).astype(int)

        # Compute metrics directly from collected data
        try:
            train_auc = roc_auc_score(all_targets, all_predictions)
            train_f1 = f1_score(all_targets, pred_binary)
            train_acc = accuracy_score(all_targets, pred_binary)
            train_precision = precision_score(all_targets, pred_binary, zero_division=0)
            train_recall = recall_score(all_targets, pred_binary, zero_division=0)

            metrics = {
                'loss': avg_loss,
                'auc': train_auc,
                'f1': train_f1,
                'accuracy': train_acc,
                'precision': train_precision,
                'recall': train_recall
            }
        except Exception as e:
            logger.warning(f"⚠️  Failed to compute metrics from in-epoch data: {e}")
            metrics = {'loss': avg_loss, 'auc': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

        # Log key learning indicators
        logger.info(f"🎯 Training Performance Summary:")
        logger.info(f"   AUC: {metrics.get('auc', 0):.4f}")
        logger.info(f"   Loss: {metrics.get('loss', avg_loss):.4f}")
        logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"   F1-Score: {metrics.get('f1', 0):.4f}")

        # Check for learning progress
        if self._prev_train_auc is not None:
            auc_change = metrics.get('auc', 0) - self._prev_train_auc
            if auc_change > 0.001:
                logger.info(f"📈 Learning progress: AUC improved by {auc_change:+.4f}")
            elif auc_change < -0.001:
                logger.warning(f"📉 Learning degradation: AUC decreased by {auc_change:+.4f}")
            else:
                logger.info(f"➡️  Learning stable: AUC change {auc_change:+.4f}")

        self._prev_train_auc = metrics.get('auc', 0)

        return metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch and return metrics."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation {epoch+1}/{self.num_epochs}")

            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.float().to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # Track metrics
                epoch_loss += loss.item()
                num_batches += 1

                # Store predictions for metrics
                probs = torch.sigmoid(outputs)
                all_predictions.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Update progress bar
                current_loss = epoch_loss / num_batches
                pbar.set_postfix({"val_loss": f"{current_loss:.4f}"})

        # Calculate epoch metrics
        avg_loss = epoch_loss / max(num_batches, 1)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Compute validation metrics from in-epoch collected data (no re-evaluation needed)
        logger.info("🔍 Evaluating on validation set...")
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

        pred_binary = (all_predictions > 0.5).astype(int)

        try:
            val_auc = roc_auc_score(all_targets, all_predictions)
            val_f1 = f1_score(all_targets, pred_binary)
            val_acc = accuracy_score(all_targets, pred_binary)
            val_precision = precision_score(all_targets, pred_binary, zero_division=0)
            val_recall = recall_score(all_targets, pred_binary, zero_division=0)

            metrics = {
                'loss': avg_loss,
                'auc': val_auc,
                'f1': val_f1,
                'accuracy': val_acc,
                'precision': val_precision,
                'recall': val_recall
            }
        except Exception as e:
            logger.warning(f"⚠️  Failed to compute validation metrics: {e}")
            metrics = {'loss': avg_loss, 'auc': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

        return metrics

    def _create_val_loader_with_metadata(self):
        """
        Create a validation loader with metadata support for evaluation.

        Returns:
            DataLoader with return_meta=True, or None if creation fails
        """
        try:
            logger.info("Creating validation datasets with metadata enabled...")

            # Load configuration
            config_path = Path("configs/datasets.json")
            if not config_path.exists():
                logger.warning(f"Config not found: {config_path}")
                return None

            with open(config_path, 'r', encoding='utf-8') as f:
                datasets_config = json.load(f)

            # Define the 4 datasets to include
            dataset_names = ['celebdf_v2', 'faceforensics', 'deeperforensics', 'dfdc']
            datasets_config = datasets_config.get('datasets', {})

            # Create validation datasets with metadata
            val_datasets = []

            for dataset_name in dataset_names:
                if dataset_name not in datasets_config:
                    logger.debug(f"Dataset {dataset_name} not found in configuration")
                    continue

                dataset_cfg = datasets_config[dataset_name]
                if not dataset_cfg.get('enabled', True):
                    logger.debug(f"Dataset {dataset_name} disabled")
                    continue

                # Get paths
                root_path = Path(dataset_cfg.get('root_path', '.'))
                splits = dataset_cfg.get('splits', {})
                image_size = dataset_cfg.get('metadata', {}).get('image_size', [256])[0]

                # Create dataset instances with return_meta=True
                try:
                    val_manifest = root_path / splits.get('val', f'manifests/{dataset_name}_val_balanced.csv')

                    # Create dataset with metadata support
                    val_ds = CelebDFDataset(
                        manifest_path=val_manifest,
                        root_path=root_path,
                        image_size=image_size,
                        augmentation=False,
                        normalize=True,
                        return_meta=True  # Enable metadata collection
                    )

                    val_datasets.append(val_ds)
                    logger.debug(f"  ✓ {dataset_name}: Val={len(val_ds):,}")

                except Exception as e:
                    logger.warning(f"Failed to load {dataset_name} validation with metadata: {e}")
                    continue

            if not val_datasets:
                logger.warning("No validation datasets could be created with metadata")
                return None

            # Combine datasets
            val_combined = ConcatDataset(val_datasets)
            logger.info(f"Combined validation dataset: {len(val_combined):,} samples")

            # Create data loader
            val_loader_meta = DataLoader(
                val_combined,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )

            logger.info(f"✅ Validation loader with metadata created successfully")
            return val_loader_meta

        except Exception as e:
            logger.error(f"❌ Failed to create validation loader with metadata: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_learning_curves(self, final_epoch: int, early_stopped: bool):
        """Generate and save learning curve plots."""
        if len(self.train_loss_history) == 0:
            logger.warning("No training history to plot")
            return

        epochs = list(range(1, len(self.train_loss_history) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Learning Curves', fontsize=16, fontweight='bold')

        # Plot 1: Training Loss
        axes[0, 0].plot(epochs, self.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].axvline(self.best_epoch + 1, color='g', linestyle='--',
                           label=f'Best Epoch ({self.best_epoch+1})', alpha=0.7)
        if early_stopped:
            axes[0, 0].axvline(final_epoch + 1, color='r', linestyle='--',
                               label=f'Early Stop ({final_epoch+1})', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Validation Loss
        axes[0, 1].plot(epochs, self.val_loss_history, 'r-', label='Val Loss', linewidth=2)
        axes[0, 1].axvline(self.best_epoch + 1, color='g', linestyle='--',
                           label=f'Best Epoch ({self.best_epoch+1})', alpha=0.7)
        if early_stopped:
            axes[0, 1].axvline(final_epoch + 1, color='r', linestyle='--',
                               label=f'Early Stop ({final_epoch+1})', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Training AUC
        axes[1, 0].plot(epochs, self.train_auc_history, 'b-', label='Train AUC', linewidth=2)
        axes[1, 0].axvline(self.best_epoch + 1, color='g', linestyle='--',
                           label=f'Best Epoch ({self.best_epoch+1})', alpha=0.7)
        axes[1, 0].axhline(0.5, color='k', linestyle=':', label='Random (0.5)', alpha=0.5)
        if early_stopped:
            axes[1, 0].axvline(final_epoch + 1, color='r', linestyle='--',
                               label=f'Early Stop ({final_epoch+1})', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Training AUC')
        axes[1, 0].set_ylim([0.4, 1.0])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Validation AUC
        axes[1, 1].plot(epochs, self.val_auc_history, 'r-', label='Val AUC', linewidth=2)
        axes[1, 1].axvline(self.best_epoch + 1, color='g', linestyle='--',
                           label=f'Best Epoch ({self.best_epoch+1})', alpha=0.7)
        axes[1, 1].axhline(0.5, color='k', linestyle=':', label='Random (0.5)', alpha=0.5)
        if early_stopped:
            axes[1, 1].axvline(final_epoch + 1, color='r', linestyle='--',
                               label=f'Early Stop ({final_epoch+1})', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].set_ylim([0.4, 1.0])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.framework.experiment_dir / 'learning_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Learning curves saved to: {plot_path}")

    def _save_training_summary(self, final_epoch: int, early_stopped: bool):
        """Save training summary to JSON."""
        summary = {
            "training_completed": True,
            "early_stopped": early_stopped,
            "total_epochs_run": final_epoch + 1,
            "max_epochs_configured": self.num_epochs,
            "best_epoch": self.best_epoch + 1,
            "best_validation_auc": float(self.best_auc),
            "final_train_auc": float(self.train_auc_history[-1]) if self.train_auc_history else None,
            "final_val_auc": float(self.val_auc_history[-1]) if self.val_auc_history else None,
            "final_train_loss": float(self.train_loss_history[-1]) if self.train_loss_history else None,
            "final_val_loss": float(self.val_loss_history[-1]) if self.val_loss_history else None,
            "early_stopping_config": {
                "patience": self.patience,
                "min_delta": self.min_delta,
                "restore_best": self.restore_best
            },
            "training_history": {
                "train_loss": [float(x) for x in self.train_loss_history],
                "val_loss": [float(x) for x in self.val_loss_history],
                "train_auc": [float(x) for x in self.train_auc_history],
                "val_auc": [float(x) for x in self.val_auc_history]
            },
            # PR-2: Per-epoch metrics consolidation
            "epoch_metrics": self.epoch_metrics_history
        }

        summary_path = self.framework.experiment_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📝 Training summary saved to: {summary_path}")

    def train(self):
        """Main training loop with automatic best model saving (Stage 00 feature)."""
        import time

        logger.info("🚀 Starting MobileNetV4 training...")
        self._training_start_time = time.time()

        logger.info("📋 Training Configuration:")
        logger.info(f"   Model: {self.model.model_name}")
        logger.info(f"   Epochs: {self.num_epochs}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        logger.info(f"   Batch Size: {self.train_loader.batch_size}")
        logger.info(f"   Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"   Validation samples: {len(self.val_loader.dataset)}")

        # Log parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"   Total Parameters: {total_params:,}")
        logger.info(f"   Trainable Parameters: {trainable_params:,}")

        logger.info("🎯 Training Objectives:")
        logger.info("   Stage 01: Simple MobileNetV4 + BCE classification")
        logger.info("   Stage 00: Integrated experiment framework")
        logger.info("   📊 Enhanced monitoring and validation enabled")
        logger.info("=" * 60)

        # Validate model sanity before training with BatchNorm compatibility
        logger.info("🔍 Pre-training model validation...")
        logger.info("⚙️  Switching to eval mode for validation (BatchNorm compatibility)")

        try:
            # CRITICAL FIX: Handle BatchNorm like we did in initialization
            original_training_mode = self.model.training
            self.model.eval()
            logger.info(f"   ✅ Model mode: eval (was {original_training_mode})")

            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                logger.info(f"   📥 Validation input shape: {dummy_input.shape}")
                logger.info(f"   🚀 Executing validation forward pass...")

                dummy_output = self.model(dummy_input)

                logger.info(f"   ✅ Model forward pass successful")
                logger.info(f"   📤 Output shape: {dummy_output.shape}")
                logger.info(f"   📊 Output range: [{dummy_output.min().item():.4f}, {dummy_output.max().item():.4f}]")

                # Additional validation checks
                if dummy_output.numel() == 1:
                    logger.info(f"   ✅ Binary classification output confirmed: {dummy_output.item():.6f}")
                else:
                    logger.warning(f"   ⚠️  Unexpected output size: {dummy_output.shape}")

        except Exception as e:
            logger.error(f"   ❌ Model validation failed: {e}")
            logger.error(f"   🆘 This indicates a critical problem with the model setup")
            logger.error(f"   🔧 Common causes:")
            logger.error(f"      - BatchNorm compatibility issues")
            logger.error(f"      - Device mismatch (CPU vs GPU)")
            logger.error(f"      - Model architecture problems")
            raise

        finally:
            # Restore original training mode
            if original_training_mode:
                self.model.train()
                logger.info(f"   🔄 Restored training mode: {self.model.training}")
            else:
                logger.info(f"   ✅ Kept eval mode: {self.model.training}")

        logger.info("🎯 Pre-training validation completed successfully!")

        early_stopped = False
        final_epoch = 0

        for epoch in range(self.num_epochs):
            # Time this epoch
            epoch_start_time = time.time()

            # LR warmup (simple linear warmup across epochs)
            if getattr(self, 'warmup_epochs', 0) > 0 and epoch < self.warmup_epochs:
                scale = float(epoch + 1) / float(self.warmup_epochs)
                # param group 0 assumed backbone (if two groups), param group 1 head
                for idx, pg in enumerate(self.optimizer.param_groups):
                    if len(self.optimizer.param_groups) == 1:
                        base_lr = self.learning_rate
                    else:
                        base_lr = self.learning_rate if idx == 1 else self.learning_rate * self.backbone_lr_scale
                    pg['lr'] = base_lr * scale
                logger.info(f"   Warmup epoch {epoch+1}/{self.warmup_epochs} scale={scale:.3f}")

            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate epoch
            val_metrics = self.validate_epoch(epoch)

            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time

            # Track metrics history for plotting
            self.train_loss_history.append(train_metrics['loss'])
            self.val_loss_history.append(val_metrics['loss'])
            self.train_auc_history.append(train_metrics['auc'])
            self.val_auc_history.append(val_metrics['auc'])

            # PR-2: Consolidate per-epoch metrics
            epoch_data = {
                'epoch': epoch + 1,
                'lr': float(self.optimizer.param_groups[0]['lr']), 'lr_groups': [float(pg['lr']) for pg in self.optimizer.param_groups],
                'time_seconds': float(epoch_duration),
                'train': {k: float(v) for k, v in train_metrics.items()},
                'val': {k: float(v) for k, v in val_metrics.items()}
            }
            self.epoch_metrics_history.append(epoch_data)

            # Log current learning rates
            logger.info(f"   Current LRs: {[pg['lr'] for pg in self.optimizer.param_groups]}")

            # Log metrics to TensorBoard (Stage 00 feature)
            self.framework.log_metrics(epoch, train_metrics, mode="train")
            self.framework.log_metrics(epoch, val_metrics, mode="validation")

            # ===== EARLY STOPPING LOGIC (Stage 01 Enhancement) =====
            if val_metrics["auc"] > self.best_auc + self.min_delta:
                # Improvement detected
                improvement = val_metrics["auc"] - self.best_auc
                self.best_auc = val_metrics["auc"]
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best model state in memory for restoration
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                logger.info(f"✅ New best validation AUC: {self.best_auc:.4f} (+{improvement:.4f})")
                logger.info(f"📍 Best epoch: {epoch+1}, Patience counter reset")

                # Checkpoint saving handled by save policy (see below)
                self.framework.set_best_epoch(epoch)
            else:
                # No improvement
                if self.patience > 0:
                    self.patience_counter += 1
                    logger.info(f"⚠️  No improvement: Patience {self.patience_counter}/{self.patience}")

                    # Check if should stop
                    if self.patience_counter >= self.patience:
                        logger.info("=" * 60)
                        logger.info(f"🛑 EARLY STOPPING TRIGGERED")
                        logger.info(f"   Stopped at epoch: {epoch+1}/{self.num_epochs}")
                        logger.info(f"   Best epoch was: {self.best_epoch+1}")
                        logger.info(f"   Best validation AUC: {self.best_auc:.4f}")
                        logger.info(f"   No improvement for {self.patience} consecutive epochs")
                        logger.info("=" * 60)
                        early_stopped = True
                        final_epoch = epoch
                        break

            # Log epoch summary
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} [Train (in-epoch) vs Val] - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Best AUC: {self.best_auc:.4f}"
            )

            
            # Save-by-policy: save based on configured metric name ('auc' or 'f1', etc.)
            policy_metric_name = getattr(self, 'save_on_metric', 'auc')
            policy_metric_value = float(val_metrics.get(policy_metric_name, 0.0))
            self.framework.save_best_model(
                self.model, policy_metric_value, metric_name=policy_metric_name, optimizer=self.optimizer,
                additional_info={'hyperparameters': {'model_name': self.model_name, 'learning_rate': self.learning_rate, 'batch_size': self.batch_size, 'optimizer': self.optimizer_name, 'weight_decay': self.weight_decay}}
            )

            final_epoch = epoch

        logger.info("=" * 60)
        logger.info(f"Training completed! Best AUC: {self.best_auc:.4f}")

        # ===== BEST WEIGHT RESTORATION (Stage 01 Enhancement) =====
        if early_stopped and self.restore_best and self.best_model_state is not None:
            logger.info("🔄 Restoring best model weights...")
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"✅ Best model restored from epoch {self.best_epoch+1}")
            logger.info(f"   Best validation AUC: {self.best_auc:.4f}")
        elif early_stopped and not self.restore_best:
            logger.info("ℹ️  Best weights not restored (restore_best=False)")

        # ===== GENERATE LEARNING CURVES AND SUMMARY =====
        self._plot_learning_curves(final_epoch, early_stopped)
        self._save_training_summary(final_epoch, early_stopped)

        # ===== PR-3: POST-TRAINING EVALUATION AND PLOT GENERATION =====
        logger.info("=" * 60)
        logger.info("🎯 PR-3: Post-training evaluation and plot generation")
        logger.info("=" * 60)

        try:
            # Create val_loader with metadata for evaluation
            logger.info("📊 Creating validation loader with metadata...")
            val_loader_meta = self._create_val_loader_with_metadata()

            if val_loader_meta is not None:
                # Run full evaluation
                logger.info("🔍 Running full evaluation on validation set...")
                evaluator = ModelEvaluator(device=self.device)
                val_summary = evaluator.full_evaluation(
                    self.model,
                    val_loader_meta,
                    criterion=self.criterion,
                    mode='validation',
                    top_k_errors=32
                )

                logger.info(f"✅ Full evaluation completed")
                logger.info(f"   Validation AUC: {val_summary['metrics']['auc']:.4f}")
                logger.info(f"   Validation F1: {val_summary['metrics']['f1']:.4f}")
                logger.info(f"   Validation Accuracy: {val_summary['metrics']['accuracy']:.4f}")

                # Generate all plots with error handling (PR-3: 17-plot generation)
                logger.info("🎨 Generating comprehensive visualizations...")

                from utils.plotting import (
                    plot_roc_curve_precomputed, plot_precision_recall_precomputed,
                    plot_calibration_curve_precomputed, plot_confidence_histogram,
                    plot_train_val_gap, plot_lr_schedule, plot_image_grid,
                    plot_metrics_summary, plot_confusion_matrix, plot_class_distribution,
                    plot_threshold_analysis, plot_probability_distribution, create_comprehensive_report,
                    plot_error_analysis
                )

                plots_generated = []
                plots_failed = []

                # Plot 1: ROC Curve
                try:
                    logger.info("  [01/07] Generating ROC curve...")
                    fig = plot_roc_curve_precomputed(
                        val_summary['roc_curve']['fpr'],
                        val_summary['roc_curve']['tpr'],
                        val_summary['metrics']['auc'],
                        output_path=self.framework.experiment_dir / '01_roc_curve.png'
                    )
                    plots_generated.append('01_roc_curve.png')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"  ❌ Failed to generate ROC curve: {e}")
                    plots_failed.append(('01_roc_curve', str(e)))

                # Plot 2: Precision-Recall Curve
                try:
                    logger.info("  [02/07] Generating precision-recall curve...")
                    fig = plot_precision_recall_precomputed(
                        val_summary['pr_curve']['precision'],
                        val_summary['pr_curve']['recall'],
                        output_path=self.framework.experiment_dir / '02_pr_curve.png'
                    )
                    plots_generated.append('02_pr_curve.png')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"  ❌ Failed to generate PR curve: {e}")
                    plots_failed.append(('02_pr_curve', str(e)))

                # Plot 3: Calibration Curve
                try:
                    logger.info("  [03/07] Generating calibration plot...")
                    fig = plot_calibration_curve_precomputed(
                        val_summary['calibration_bins']['bin_centers'],
                        val_summary['calibration_bins']['avg_confidence'],
                        val_summary['calibration_bins']['accuracy'],
                        output_path=self.framework.experiment_dir / '03_calibration.png'
                    )
                    plots_generated.append('03_calibration.png')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"  ❌ Failed to generate calibration plot: {e}")
                    plots_failed.append(('03_calibration', str(e)))

                # Plot 4: Confidence Histogram
                try:
                    logger.info("  [04/07] Generating confidence histogram...")
                    fig = plot_confidence_histogram(
                        val_summary['confidence_hist']['bins'],
                        val_summary['confidence_hist']['real_hist'],
                        val_summary['confidence_hist']['fake_hist'],
                        output_path=self.framework.experiment_dir / '04_confidence_hist.png'
                    )
                    plots_generated.append('04_confidence_hist.png')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"  ❌ Failed to generate confidence histogram: {e}")
                    plots_failed.append(('04_confidence_hist', str(e)))

                # Plot 5: Train/Val Gap (Loss and AUC divergence)
                try:
                    logger.info("  [05/07] Generating train/val gap plot...")
                    fig = plot_train_val_gap(
                        self.epoch_metrics_history,
                        output_path=self.framework.experiment_dir / '05_train_val_gap.png'
                    )
                    plots_generated.append('05_train_val_gap.png')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"  ❌ Failed to generate train/val gap plot: {e}")
                    plots_failed.append(('05_train_val_gap', str(e)))

                # Plot 6: Learning Rate Schedule
                try:
                    logger.info("  [06/07] Generating learning rate schedule...")
                    fig = plot_lr_schedule(
                        self.epoch_metrics_history,
                        output_path=self.framework.experiment_dir / '06_lr_schedule.png'
                    )
                    plots_generated.append('06_lr_schedule.png')
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"  ❌ Failed to generate LR schedule: {e}")
                    plots_failed.append(('06_lr_schedule', str(e)))

                # Plot 7: Error Sample Images (top FP and FN)
                try:
                    logger.info("  [07/07] Generating error sample images...")
                    error_paths = val_summary['topk_fp_paths'][:16] + val_summary['topk_fn_paths'][:16]
                    if error_paths:
                        fig = plot_image_grid(
                            error_paths,
                            title='Top Error Samples (False Positives + False Negatives)',
                            output_path=self.framework.experiment_dir / '07_error_samples.png'
                        )
                        plots_generated.append('07_error_samples.png')
                        plt.close(fig)
                        # Also generate separate FP and FN grids if available (counts toward richer report)
                        try:
                            if val_summary['topk_fp_paths']:
                                fig = plot_image_grid(
                                    val_summary['topk_fp_paths'][:32],
                                    title='Top False Positives',
                                    output_path=self.framework.experiment_dir / '07a_top_fp_samples.png'
                                )
                                if fig is not None:
                                    plots_generated.append('07a_top_fp_samples.png')
                                    plt.close(fig)
                            if val_summary['topk_fn_paths']:
                                fig = plot_image_grid(
                                    val_summary['topk_fn_paths'][:32],
                                    title='Top False Negatives',
                                    output_path=self.framework.experiment_dir / '07b_top_fn_samples.png'
                                )
                                if fig is not None:
                                    plots_generated.append('07b_top_fn_samples.png')
                                    plt.close(fig)
                        except Exception as e:
                            logger.warning(f"  ❌ Failed to generate separate FP/FN grids: {e}")
                    else:
                        logger.info("  ℹ️  No error samples to visualize")
                except Exception as e:
                    logger.warning(f"  ❌ Failed to generate error samples: {e}")
                    plots_failed.append(('07_error_samples', str(e)))

                # Log plot generation summary
                logger.info(f"✅ Plot Generation Summary:")
                logger.info(f"   Generated: {len(plots_generated)}/7 plots")
                for plot_name in plots_generated:
                    logger.info(f"     ✓ {plot_name}")

                if plots_failed:
                    logger.warning(f"   Failed: {len(plots_failed)} plots")
                    for plot_name, error in plots_failed:
                        logger.warning(f"     ✗ {plot_name}: {error}")

                # Generate additional comprehensive report plots (metrics/confusion/class dist/threshold)
                try:
                    logger.info("  [+] Creating comprehensive report plots (metrics, confusion matrix, class dist, thresholds)...")
                    create_comprehensive_report(val_summary, output_dir=self.framework.experiment_dir)
                    # These filenames are deterministic in create_comprehensive_report
                    plots_generated.extend([
                        'metrics_summary.png',
                        'confusion_matrix.png',
                        'class_distribution.png',
                        'threshold_analysis.png'
                    ])
                except Exception as e:
                    logger.warning(f"  ❌ Failed to create comprehensive report plots: {e}")

                # Optional: Probability distribution and error analysis if raw arrays are available
                try:
                    if 'targets_array' in val_summary and 'probabilities_array' in val_summary:
                        import numpy as np
                        targets_np = np.asarray(val_summary['targets_array'])
                        probs_np = np.asarray(val_summary['probabilities_array']).reshape(-1)
                        preds_np = np.asarray(val_summary.get('predictions_array', (probs_np >= 0.5).astype(int))).reshape(-1)

                        # Probability distribution by class
                        real_probs = probs_np[targets_np == 0]
                        fake_probs = probs_np[targets_np == 1]
                        fig = plot_probability_distribution(
                            probabilities_real=real_probs,
                            probabilities_fake=fake_probs,
                            output_path=self.framework.experiment_dir / '08_probability_distribution.png'
                        )
                        if fig is not None:
                            plots_generated.append('08_probability_distribution.png')
                            plt.close(fig)

                        # Error analysis (FP/FN histograms)
                        fig = plot_error_analysis(
                            targets=targets_np,
                            predictions=preds_np,
                            probabilities=probs_np,
                            output_path=self.framework.experiment_dir / '09_error_analysis.png'
                        )
                        if fig is not None:
                            plots_generated.append('09_error_analysis.png')
                            plt.close(fig)
                except Exception as e:
                    logger.warning(f"  ❌ Failed to generate probability/error analysis plots: {e}")

                # Update training_summary.json with val_final and plots
                logger.info("💾 Updating training summary with final evaluation...")
                summary_path = self.framework.experiment_dir / 'training_summary.json'
                with open(summary_path, 'r') as f:
                    summary = json.load(f)

                # Add final validation evaluation
                summary['val_final'] = {
                    'metrics': val_summary['metrics'],
                    'loss': val_summary['loss'],
                    'confusion_matrix': val_summary['confusion_matrix'],
                    'total_samples': val_summary['total_samples'],
                    'class_distribution': val_summary['class_distribution'],
                    'probability_statistics': val_summary['probability_statistics'],
                }

                # Add plot artifacts list
                summary['artifacts'] = {
                    'plots': plots_generated,
                    'plots_failed': plots_failed,
                    'learning_curves': 'learning_curves.png',
                    'best_model': 'best_model.pth'
                }

                # Save updated summary
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

                logger.info(f"✅ Training summary updated with val_final and artifacts")
                logger.info(f"   File: {summary_path}")

            else:
                logger.warning("⚠️  Could not create validation loader with metadata, skipping full evaluation")

        except Exception as e:
            logger.error(f"❌ Error during post-training evaluation: {e}")
            logger.error(f"   Continuing with basic training summary...")
            import traceback
            traceback.print_exc()

        logger.info("=" * 60)
        logger.info(
            f"Training results saved to: {self.framework.experiment_dir}"
        )

        # Close experiment framework (Stage 00 feature)
        self.framework.close()

        return self.best_auc


def main():
    """Main training function with Phase 1 configuration system integration."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV4 for deepfake detection (Stage 01)"
    )

    # ===== Phase 1: Configuration Management =====
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_default.yaml",
        help="Path to configuration file (YAML or JSON)"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="mobilenetv4_hybrid_medium",
        help="MobileNetV4 model name",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--freeze_backbone",
        type=str2bool,
        default=False,
        help="Freeze backbone weights",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate", "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_epochs",
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    # Data arguments
    parser.add_argument(
        "--use_dataset_weights",
        type=str2bool,
        default=True,
        help="Balance dataset contributions (Stage 01 requirement)",
    )

    # System arguments
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stage1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Early stopping arguments (Stage 01 Enhancement)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without improvement). Set to 0 to disable."
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum improvement in validation AUC to reset patience counter"
    )
    parser.add_argument(
        "--restore_best",
        type=str2bool,
        default=True,
        help="Restore best model weights after early stopping"
    )
    

    args = parser.parse_args()

    # ===== Phase 1: Load configuration and merge with CLI (YAML first, CLI can override) =====
    config_dict = {}
    import sys as _sys
    argv = _sys.argv[1:]
    def _provided(*aliases: str) -> bool:
        for a in aliases:
            if a in argv:
                return True
            if any(x.startswith(a + '=') for x in argv):
                return True
        return False

    if args.config:
        try:
            config_dict = load_config(args.config)
            logger.info(f"✅ Configuration loaded from: {args.config}")

            # Model
            model_cfg = config_dict.get('model', {})
            if model_cfg:
                if not _provided('--model_name'):
                    args.model_name = model_cfg.get('name', args.model_name)
                if not _provided('--dropout_rate'):
                    args.dropout_rate = model_cfg.get('dropout', args.dropout_rate)
                if not _provided('--freeze_backbone'):
                    args.freeze_backbone = model_cfg.get('freeze_backbone', args.freeze_backbone)
                if not _provided('--seed'):
                    args.seed = model_cfg.get('seed', args.seed)

            # Training
            training_cfg = config_dict.get('training', {})
            if training_cfg:
                if not _provided('--learning_rate', '--lr'):
                    args.learning_rate = training_cfg.get('lr', args.learning_rate)
                if not _provided('--batch_size'):
                    args.batch_size = training_cfg.get('batch_size', args.batch_size)
                if not _provided('--num_epochs', '--epochs'):
                    args.num_epochs = training_cfg.get('epochs', args.num_epochs)
                # Additional training configs
                args.optimizer_name = training_cfg.get('optimizer', getattr(args, 'optimizer_name', 'adamw'))
                args.weight_decay = training_cfg.get('weight_decay', getattr(args, 'weight_decay', 1e-4))
                args.num_workers = training_cfg.get('num_workers', getattr(args, 'num_workers', 12))
                args.warmup_epochs = training_cfg.get('warmup_epochs', getattr(args, 'warmup_epochs', 0))

            # Early stopping
            es_cfg = config_dict.get('early_stopping', {})
            if es_cfg:
                if not _provided('--patience'):
                    args.patience = es_cfg.get('patience', args.patience)
                if not _provided('--min_delta'):
                    args.min_delta = es_cfg.get('min_delta', args.min_delta)
                if not _provided('--restore_best'):
                    args.restore_best = es_cfg.get('restore_best', getattr(args, 'restore_best', True))

            # Device
            device_cfg = config_dict.get('device', {})
            if device_cfg and not _provided('--device'):
                args.device = device_cfg.get('type', args.device)

            # Paths
            paths_cfg = config_dict.get('paths', {})
            if paths_cfg and not _provided('--output_dir'):
                args.output_dir = paths_cfg.get('output_dir', args.output_dir)

            # Data
            data_cfg = config_dict.get('data', {})
            if data_cfg:
                args.image_size = data_cfg.get('image_size', getattr(args, 'image_size', None))

            # Save policy
            save_cfg = config_dict.get('save_policy', {})
            if save_cfg:
                args.save_on_metric = save_cfg.get('on_best', getattr(args, 'save_on_metric', 'auc'))

            # Log merged configuration
            logger.info("📋 Configuration Summary (YAML merged, CLI overrides when provided):")
            logger.info(f"   Model: {args.model_name}, dropout={args.dropout_rate}, freeze_backbone={args.freeze_backbone}")
            logger.info(f"   Training: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.num_epochs}, optimizer={getattr(args, 'optimizer_name', 'adamw')}, weight_decay={getattr(args, 'weight_decay', 1e-4)}, num_workers={getattr(args, 'num_workers', 12)}, warmup_epochs={getattr(args, 'warmup_epochs', 0)}")
            logger.info(f"   EarlyStopping: patience={args.patience}, min_delta={args.min_delta}, restore_best={getattr(args, 'restore_best', True)}")
            logger.info(f"   Device: {args.device}")
            logger.info(f"   Paths: output_dir={args.output_dir}")
            logger.info(f"   Data: image_size_override={getattr(args, 'image_size', None)}")
            logger.info(f"   Save policy: on_best={getattr(args, 'save_on_metric', 'auc')}")

        except FileNotFoundError:
            logger.warning(f"⚠️  Config file not found: {args.config}, using CLI defaults")
        except Exception as e:
            logger.warning(f"⚠️  Error loading config: {e}, using CLI defaults")
# Setup reproducible environment (Stage 00 feature)
    setup_reproducible_environment(args.seed)

    # Create trainer and optionally load from checkpoint
    trainer = SimpleMobileNetV4Trainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        output_dir=args.output_dir,
        use_dataset_weights=args.use_dataset_weights,
        freeze_backbone=args.freeze_backbone,
        dropout_rate=args.dropout_rate,
        seed=args.seed,
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best=args.restore_best,
        optimizer_name=getattr(args, 'optimizer_name', 'adamw'),
        weight_decay=getattr(args, 'weight_decay', 1e-4),
        num_workers=getattr(args, 'num_workers', 12),
        image_size_override=getattr(args, 'image_size', None),
        save_on_metric=getattr(args, 'save_on_metric', 'auc'),
        warmup_epochs=getattr(args, 'warmup_epochs', 0),
    )

    

    # Setup data loaders (Stage 01 requirement)
    trainer.setup_data_loaders()

    # Start training
    best_auc = trainer.train()

    logger.info(f"Final best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    sys.exit(main())
