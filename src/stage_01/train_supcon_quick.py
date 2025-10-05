#!/usr/bin/env python3
"""
AWARE-NET Stage 01: SupCon Quick Validation Script

Simplified SupCon training for rapid validation (5-10 epochs).
Based on train_baseline.py's successful architecture.

Usage:
    python train_supcon_quick.py --epochs 10 --exclude-dataset deeperforensics_1_0
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from stage_00.baseline_model import EfficientNetV2B3Baseline
from stage_00.train_baseline import (
    UnifiedDeepfakeDataset,
    MultiDatasetWrapper,
    setup_device_with_fallback
)
# Direct import to avoid __init__.py issues
sys.path.insert(0, str(Path(__file__).parent))
from supcon_loss import SupConLoss
from train_stage1_supcon import Stage1ManifestDataset, resolve_manifest_paths
from balanced_sampler import BalancedBatchSampler


def create_experiment_directory(experiment_name: str) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{experiment_name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    return exp_dir


class SupConModel(nn.Module):
    """
    EfficientNetV2 + Projection Head for SupCon training
    """
    def __init__(
        self,
        model_name: str = 'tf_efficientnetv2_b0',
        projection_dim: int = 512,
        pretrained: bool = True
    ):
        super().__init__()

        # Backbone (reuse baseline model without classification head)
        self.backbone = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=pretrained,
            model_name=model_name
        )

        # Get feature dimension
        if 'b0' in model_name:
            feature_dim = 1280
        elif 'b3' in model_name:
            feature_dim = 1536
        else:
            feature_dim = 1024

        # Projection head for SupCon
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

        # Classification head (optional, for evaluation)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x, return_features=False):
        """
        Forward pass

        Args:
            x: Input images
            return_features: If True, return (projection, features, logits)
                           If False, return logits only
        """
        # Extract features using backbone (timm model returns features directly)
        features = self.backbone.backbone(x)

        if return_features:
            # For SupCon training
            projections = self.projection_head(features)
            # L2 normalize projections
            projections = F.normalize(projections, dim=1)

            # Also compute logits for monitoring
            logits = self.classifier(features)

            return projections, features, logits
        else:
            # For evaluation
            return self.classifier(features)


def train_epoch_supcon(
    model: nn.Module,
    train_loader: DataLoader,
    supcon_criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict:
    """Train one epoch with SupCon loss"""
    model.train()

    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]")

    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch (handle both 2-tuple and 3-tuple)
        if len(batch_data) == 3:
            data, targets, _ = batch_data  # With dataset_id
        else:
            data, targets = batch_data

        data, targets = data.to(device), targets.to(device)
        batch_size = data.size(0)

        # Forward pass
        projections, features, logits = model(data, return_features=True)

        # SupCon loss
        loss = supcon_criterion(projections.unsqueeze(1), targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/total_samples:.4f}'
        })

    avg_loss = total_loss / total_samples

    return {
        'loss': avg_loss,
        'samples': total_samples
    }


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict:
    """Validate using classification head"""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_probabilities = []
    all_targets = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]")

    with torch.no_grad():
        for batch_data in pbar:
            # Unpack batch
            if len(batch_data) == 3:
                data, targets, _ = batch_data
            else:
                data, targets = batch_data

            data, targets = data.to(device), targets.to(device)
            targets_bce = targets.float().unsqueeze(1)

            # Forward pass (classification only)
            logits = model(data, return_features=False)
            loss = criterion(logits, targets_bce)

            # Get predictions
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities > 0.5).float()

            # Statistics
            total_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets_bce).sum().item()

            all_predictions.extend(predicted.squeeze().cpu().numpy())
            all_probabilities.extend(probabilities.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

    # Calculate metrics
    from sklearn.metrics import roc_auc_score, f1_score

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)

    try:
        auc_score = roc_auc_score(all_targets, all_probabilities)
        f1 = f1_score(all_targets, all_predictions)
    except:
        auc_score = 0.0
        f1 = 0.0

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    # Threshold optimization: find best F1 score
    best_threshold = 0.5
    best_f1 = f1
    thresholds = np.arange(0.3, 0.8, 0.05)

    for threshold in thresholds:
        predictions_at_threshold = (all_probabilities > threshold).astype(int)
        try:
            f1_at_threshold = f1_score(all_targets, predictions_at_threshold)
            if f1_at_threshold > best_f1:
                best_f1 = f1_at_threshold
                best_threshold = threshold
        except:
            continue

    # Recalculate accuracy with best threshold
    best_predictions = (all_probabilities > best_threshold).astype(int)
    best_accuracy = np.mean(best_predictions == all_targets)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc_score,
        'f1': f1,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets
    }


def train_stage1_encoder_only(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    supcon_criterion: nn.Module,
    bce_criterion: nn.Module,
    device: torch.device,
    args,
    exp_dir: Path
) -> Dict:
    """
    Stage 1: Train encoder + projection head with SupCon loss only.
    Classifier is NOT trained in this stage.
    """
    print("\n" + "="*80)
    print(f"STAGE 1: Encoder Pretraining ({args.stage1_epochs} epochs)")
    print("="*80)
    print("Training: backbone + projection_head")
    print("Frozen: classifier (will be trained in Stage 2)")
    print()

    # Freeze classifier, unfreeze encoder
    for param in model.classifier.parameters():
        param.requires_grad = False
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.projection_head.parameters():
        param.requires_grad = True

    # Create optimizer for Stage 1 (only encoder parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.stage1_lr,
        weight_decay=1e-4
    )

    best_supcon_loss = float('inf')
    stage1_results = {
        'train_losses': [],
        'val_losses': [],
        'val_aucs': [],
        'val_accs': [],
        'val_f1s': []
    }

    for epoch in range(1, args.stage1_epochs + 1):
        # Train with SupCon loss only
        model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch:02d} [Train]")

        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch
            if len(batch_data) == 3:
                data, targets, _ = batch_data
            else:
                data, targets = batch_data

            data, targets = data.to(device), targets.to(device)
            batch_size = data.size(0)

            # Forward pass - get projections only
            projections, features, logits = model(data, return_features=True)

            # SupCon loss ONLY (no classification loss)
            loss = supcon_criterion(projections.unsqueeze(1), targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar
            pbar.set_postfix({
                'SupCon Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/total_samples:.4f}'
            })

        avg_train_loss = total_loss / total_samples

        # Validate (using classifier for monitoring, but it's not trained)
        val_metrics = validate_epoch(
            model, val_loader, bce_criterion, device, epoch
        )

        # Save results
        stage1_results['train_losses'].append(avg_train_loss)
        stage1_results['val_losses'].append(val_metrics['loss'])
        stage1_results['val_aucs'].append(val_metrics['auc'])
        stage1_results['val_accs'].append(val_metrics['accuracy'])
        stage1_results['val_f1s'].append(val_metrics['f1'])

        # Print summary
        print(f"\nStage1 Epoch {epoch:02d} Summary:")
        print(f"  Train SupCon Loss: {avg_train_loss:.4f}")
        print(f"  Val (untrained classifier): Loss={val_metrics['loss']:.4f}, "
              f"AUC={val_metrics['auc']:.4f} (not meaningful yet)")

        # Save best encoder based on SupCon loss
        if avg_train_loss < best_supcon_loss:
            best_supcon_loss = avg_train_loss
            stage1_checkpoint = exp_dir / 'checkpoints' / 'stage1_encoder.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'supcon_loss': avg_train_loss,
                'args': vars(args)
            }, stage1_checkpoint)
            print(f"  ✓ Saved Stage 1 encoder (SupCon loss: {best_supcon_loss:.4f})")

        print()

    print("="*80)
    print(f"Stage 1 completed! Best SupCon loss: {best_supcon_loss:.4f}")
    print(f"Encoder saved to: {stage1_checkpoint}")
    print("="*80)
    print()

    return stage1_results, stage1_checkpoint


def train_stage2_classifier_only(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bce_criterion: nn.Module,
    device: torch.device,
    args,
    exp_dir: Path,
    stage1_checkpoint: Path
) -> Dict:
    """
    Stage 2: Freeze encoder, train classifier only with BCE loss.
    This is where real classification performance comes from.
    """
    print("\n" + "="*80)
    print(f"STAGE 2: Classifier Fine-tuning ({args.stage2_epochs} epochs)")
    print("="*80)
    print(f"Loading encoder from: {stage1_checkpoint}")

    # Load Stage 1 checkpoint
    checkpoint = torch.load(stage1_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded encoder from epoch {checkpoint['epoch']} "
          f"(SupCon loss: {checkpoint['supcon_loss']:.4f})")
    print()

    print("Frozen: backbone + projection_head")
    print("Training: classifier ONLY")
    print()

    # Freeze encoder, unfreeze classifier
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.projection_head.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Create optimizer for Stage 2 (only classifier parameters)
    # Use smaller learning rate for fine-tuning
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=args.stage2_lr,
        weight_decay=1e-4
    )

    # Compute class weights for balanced training (improves F1)
    print("\nComputing class weights from training data...")
    train_labels = []
    sample_count = 0
    for batch_data in train_loader:
        if len(batch_data) == 3:
            _, targets, _ = batch_data
        else:
            _, targets = batch_data
        train_labels.extend(targets.cpu().numpy())
        sample_count += len(targets)
        if sample_count >= 10000:  # Sample enough for stable statistics
            break

    from collections import Counter
    label_counts = Counter(train_labels)
    total_samples = sum(label_counts.values())

    # Compute inverse frequency weights
    weight_for_0 = total_samples / (2.0 * label_counts[0]) if label_counts[0] > 0 else 1.0
    weight_for_1 = total_samples / (2.0 * label_counts[1]) if label_counts[1] > 0 else 1.0

    # pos_weight for BCEWithLogitsLoss (weight for positive class)
    pos_weight = torch.tensor([weight_for_1 / weight_for_0], device=device)

    print(f"Class distribution: Real={label_counts[0]}, Fake={label_counts[1]}")
    print(f"Class weights: Real={weight_for_0:.3f}, Fake={weight_for_1:.3f}")
    print(f"pos_weight for BCE: {pos_weight.item():.3f}")

    # Create weighted BCE criterion
    bce_criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("✓ Using weighted BCE loss for balanced training\n")

    best_val_auc = 0.0
    stage2_results = {
        'train_losses': [],
        'val_losses': [],
        'val_aucs': [],
        'val_accs': [],
        'val_f1s': []
    }

    for epoch in range(1, args.stage2_epochs + 1):
        # Train classifier only
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch:02d} [Train]")

        for batch_data in pbar:
            # Unpack batch
            if len(batch_data) == 3:
                data, targets, _ = batch_data
            else:
                data, targets = batch_data

            data, targets = data.to(device), targets.to(device)
            targets_bce = targets.float().unsqueeze(1)

            # Forward pass (classification only, encoder frozen)
            logits = model(data, return_features=False)
            loss = bce_criterion_weighted(logits, targets_bce)  # Use weighted loss

            # Get predictions
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities > 0.5).float()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets_bce).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, bce_criterion, device, epoch
        )

        # Save results
        stage2_results['train_losses'].append(avg_train_loss)
        stage2_results['val_losses'].append(val_metrics['loss'])
        stage2_results['val_aucs'].append(val_metrics['auc'])
        stage2_results['val_accs'].append(val_metrics['accuracy'])
        stage2_results['val_f1s'].append(val_metrics['f1'])

        # Print summary
        print(f"\nStage2 Epoch {epoch:02d} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"  Optimized: Best F1={val_metrics['best_f1']:.4f} @ threshold={val_metrics['best_threshold']:.2f}")

        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_path = exp_dir / 'checkpoints' / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'args': vars(args)
            }, best_model_path)
            print(f"  ✓ Saved best model (AUC: {best_val_auc:.4f})")

        print()

    print("="*80)
    print(f"Stage 2 completed! Best validation AUC: {best_val_auc:.4f}")
    print("="*80)
    print()

    return stage2_results


def main():
    parser = argparse.ArgumentParser(description='Stage 01 SupCon Quick Validation')

    # Model parameters
    parser.add_argument('--model', type=str, default='tf_efficientnetv2_b0',
                       choices=['tf_efficientnetv2_b0', 'tf_efficientnetv2_b3'],
                       help='Model architecture')
    parser.add_argument('--projection-dim', type=int, default=512,
                       help='Projection head dimension')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (quick validation: 5-10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='SupCon temperature parameter')

    # Dataset parameters
    parser.add_argument('--dataset-mode', type=str, default='balanced',
                       choices=['original', 'balanced'],
                       help='Dataset mode')
    parser.add_argument('--exclude-dataset', type=str, default=None,
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0', 'dfdc'],
                       help='Dataset to exclude (for LODO)')
    parser.add_argument('--test-dataset', type=str, default=None,
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0', 'dfdc'],
                       help='OOD test dataset')
    parser.add_argument('--train-manifest', type=str, help='Path to training manifest CSV')
    parser.add_argument('--val-manifest', type=str, help='Path to validation manifest CSV')
    parser.add_argument('--test-manifest', type=str, help='Path to test manifest CSV (used when manifest mode active)')
    parser.add_argument('--dataset-root', type=str, default='.', help='Root directory for manifest image paths')
    parser.add_argument('--image-size', type=int, default=256, help='Input image size for manifest loader')
    parser.add_argument('--dataset-config', type=str, default=None,
                       help='Dataset configuration JSON for manifest autoloading when explicit paths are not provided')

    # Experiment parameters
    parser.add_argument('--experiment-name', type=str, default='supcon_quick_validation',
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Two-stage training parameters
    parser.add_argument('--training-mode', type=str, default='two_stage',
                       choices=['joint', 'two_stage'],
                       help='Training mode: joint (single-stage) or two_stage')
    parser.add_argument('--stage1-epochs', type=int, default=10,
                       help='Stage 1 encoder training epochs')
    parser.add_argument('--stage2-epochs', type=int, default=5,
                       help='Stage 2 classifier training epochs')
    parser.add_argument('--stage1-lr', type=float, default=1e-3,
                       help='Stage 1 learning rate')
    parser.add_argument('--stage2-lr', type=float, default=1e-4,
                       help='Stage 2 learning rate (smaller for fine-tuning)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device, gpu_info = setup_device_with_fallback()
    print(f"Using device: {device}")
    if gpu_info:
        print(f"GPU: {gpu_info.get('name', 'Unknown')}\n")

    # Create experiment directory
    exp_dir = create_experiment_directory(args.experiment_name)
    print(f"Experiment directory: {exp_dir}\n")

    dataset_name_mapping = {
        'celebdf_v2': 'celebdf_v2',
        'faceforensics_plus_plus': 'faceforensics',
        'deeperforensics_1_0': 'deeperforensics',
        'dfdc': 'dfdc'
    }

    test_manifest = None

    dataset_config_path = Path(args.dataset_config) if args.dataset_config else Path('configs/datasets.json')

    using_manifest = all([
        args.train_manifest,
        args.val_manifest,
        args.test_manifest,
    ])

    if using_manifest:
        train_manifest = Path(args.train_manifest)
        val_manifest = Path(args.val_manifest)
        test_manifest = Path(args.test_manifest)
        for manifest in (train_manifest, val_manifest, test_manifest):
            if not manifest.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest}")

        print("Using manifest-driven datasets:\n"
              f"  Train: {train_manifest}\n"
              f"  Val:   {val_manifest}\n"
              f"  Test:  {test_manifest}\n")

        dataset_root = Path(args.dataset_root)
        train_dataset = Stage1ManifestDataset(
            manifest_path=train_manifest,
            root_dir=dataset_root,
            image_size=args.image_size,
            augmentation=True,
            contrastive_views=True
        )
        val_dataset = Stage1ManifestDataset(
            manifest_path=val_manifest,
            root_dir=dataset_root,
            image_size=args.image_size,
            augmentation=False,
            contrastive_views=False
        )

        train_sampler = BalancedBatchSampler(
            labels=train_dataset.get_labels(),
            batch_size=args.batch_size,
            min_samples_per_class=4,
            strategy='balanced',
            seed=args.seed
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        print(f"✓ Training samples: {len(train_dataset):,}")
        print(f"✓ Validation samples: {len(val_dataset):,}\n")

        train_dataset_list = None  # unused in manifest mode
        val_dataset_list = None
    else:
        # Determine datasets to use (legacy multi-dataset mode)
        all_datasets = ['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0', 'dfdc']
        if args.exclude_dataset:
            train_datasets = [d for d in all_datasets if d != args.exclude_dataset]
            print(f"LODO Training: Using {train_datasets}, excluding {args.exclude_dataset}\n")
        else:
            train_datasets = all_datasets
            print(f"Multi-dataset training: {train_datasets}\n")

        print("Loading datasets (legacy ImageFolder pipeline)...")
        train_dataset_list = []
        val_dataset_list = []

        dataset_config_available = dataset_config_path.exists()

        for dataset_name in train_datasets:
            manifests = {}
            if dataset_config_available:
                try:
                    manifest_paths, _ = resolve_manifest_paths(
                        dataset_config_path,
                        dataset_name,
                        args.dataset_mode,
                    )
                    manifests = {
                        'train': manifest_paths['train'],
                        'val': manifest_paths['val'],
                    }
                    print(f"  ✓ Using {dataset_config_path} manifests for {dataset_name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  ⚠️ Could not resolve manifests from {dataset_config_path} for {dataset_name}: {exc}")

            if not manifests:
                manifest_key = dataset_name_mapping.get(dataset_name)
                if manifest_key is None:
                    print(f"  ⚠️ Unknown dataset mapping for {dataset_name}, skipping")
                    continue

                train_manifest_path = Path(f"manifests/{manifest_key}_train_{args.dataset_mode}.csv")
                val_manifest_path = Path(f"manifests/{manifest_key}_val_{args.dataset_mode}.csv")
                if not train_manifest_path.exists() or not val_manifest_path.exists():
                    print(f"  ⚠️ Missing manifest for {dataset_name} ({args.dataset_mode}), skipping")
                    continue

                manifests = {
                    'train': train_manifest_path,
                    'val': val_manifest_path,
                }

            train_ds = UnifiedDeepfakeDataset(
                manifest_path=manifests['train'],
                dataset_name=f'{dataset_name}_train',
                transform=None,
                use_augmentation=True
            )
            train_dataset_list.append(train_ds)

            val_ds = UnifiedDeepfakeDataset(
                manifest_path=manifests['val'],
                dataset_name=f'{dataset_name}_val',
                transform=None,
                use_augmentation=False
            )
            val_dataset_list.append(val_ds)

        if not train_dataset_list:
            raise RuntimeError("No datasets available for training. Provide manifest paths or ensure manifests exist for the selected mode.")

        train_dataset = MultiDatasetWrapper(train_dataset_list)
        val_dataset = MultiDatasetWrapper(val_dataset_list)

        print(f"✓ Training samples: {len(train_dataset):,}")
        print(f"✓ Validation samples: {len(val_dataset):,}\n")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

    # Create model
    print("Creating SupCon model...")
    model = SupConModel(
        model_name=args.model,
        projection_dim=args.projection_dim,
        pretrained=True
    ).to(device)

    print(f"✓ Model created: {args.model}")
    print(f"  Projection dim: {args.projection_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Create loss functions
    supcon_criterion = SupConLoss(temperature=args.temperature)
    bce_criterion = nn.BCEWithLogitsLoss()

    # Training mode selection
    print("="*80)
    print(f"Training Mode: {args.training_mode.upper()}")
    print("="*80)
    print()

    if args.training_mode == 'two_stage':
        # ===================================================================
        # TWO-STAGE TRAINING (Recommended for academic rigor)
        # ===================================================================
        print("Using two-stage SupCon training:")
        print("  Stage 1: Train encoder + projection head (SupCon loss)")
        print("  Stage 2: Train classifier only (BCE loss)")
        print()

        # Stage 1: Encoder pretraining
        stage1_results, stage1_checkpoint = train_stage1_encoder_only(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            supcon_criterion=supcon_criterion,
            bce_criterion=bce_criterion,
            device=device,
            args=args,
            exp_dir=exp_dir
        )

        # Stage 2: Classifier fine-tuning
        stage2_results = train_stage2_classifier_only(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            bce_criterion=bce_criterion,
            device=device,
            args=args,
            exp_dir=exp_dir,
            stage1_checkpoint=stage1_checkpoint
        )

        # Combine results
        results = {
            'stage1': stage1_results,
            'stage2': stage2_results,
            'train_losses': stage2_results['train_losses'],
            'val_losses': stage2_results['val_losses'],
            'val_aucs': stage2_results['val_aucs'],
            'val_accs': stage2_results['val_accs'],
            'val_f1s': stage2_results['val_f1s']
        }
        best_val_auc = max(stage2_results['val_aucs'])
        checkpoint_path = exp_dir / 'checkpoints' / 'best_model.pth'

    else:  # 'joint' mode
        # ===================================================================
        # JOINT TRAINING (Original implementation for comparison)
        # ===================================================================
        print("Using joint SupCon training (single-stage):")
        print("  Training encoder + classifier together")
        print("  Loss = SupCon loss (projections) + BCE loss (classifier)")
        print()

        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4
        )

        print("="*80)
        print(f"Starting Joint SupCon Training ({args.epochs} epochs)")
        print("="*80)
        print()

        best_val_auc = 0.0
        results = {
            'train_losses': [],
            'val_losses': [],
            'val_aucs': [],
            'val_accs': [],
            'val_f1s': []
        }

        for epoch in range(1, args.epochs + 1):
            # Train
            train_metrics = train_epoch_supcon(
                model, train_loader, supcon_criterion,
                optimizer, device, epoch
            )

            # Validate
            val_metrics = validate_epoch(
                model, val_loader, bce_criterion, device, epoch
            )

            # Save results
            results['train_losses'].append(train_metrics['loss'])
            results['val_losses'].append(val_metrics['loss'])
            results['val_aucs'].append(val_metrics['auc'])
            results['val_accs'].append(val_metrics['accuracy'])
            results['val_f1s'].append(val_metrics['f1'])

            # Print summary
            print(f"\nEpoch {epoch:02d} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                checkpoint_path = exp_dir / 'checkpoints' / 'best_model.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_metrics['auc'],
                    'args': vars(args)
                }, checkpoint_path)
                print(f"  ✓ Saved best model (AUC: {best_val_auc:.4f})")

            print()

        print("="*80)
        print(f"Training completed! Best validation AUC: {best_val_auc:.4f}")
        print("="*80)
        print()

    # OOD evaluation if specified
    if using_manifest and args.test_manifest:
        print("\nEvaluating using provided test manifest")
        print("-" * 80)

        test_ds = Stage1ManifestDataset(
            manifest_path=test_manifest,
            root_dir=Path(args.dataset_root),
            image_size=args.image_size,
            augmentation=False,
            contrastive_views=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        test_dataset_name = test_manifest.name

    elif args.test_dataset:
        print(f"\nEvaluating on OOD test set: {args.test_dataset}")
        print("-"*80)

        try:
            manifest_paths, _ = resolve_manifest_paths(
                dataset_config_path,
                args.test_dataset,
                args.dataset_mode,
            )
            manifest_path = manifest_paths['test']
        except Exception:
            manifest_key = dataset_name_mapping.get(args.test_dataset)
            if manifest_key is None:
                raise ValueError(f"Unknown dataset mapping for test dataset {args.test_dataset}")
            manifest_path = Path(f'manifests/{manifest_key}_test_{args.dataset_mode}.csv')
            if not manifest_path.exists():
                raise FileNotFoundError(f"Test manifest not found: {manifest_path}")

        test_ds = UnifiedDeepfakeDataset(
            manifest_path=manifest_path,
            dataset_name=f'{args.test_dataset}_test',
            transform=None,
            use_augmentation=False
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        test_dataset_name = args.test_dataset

    else:
        test_loader = None
        test_dataset_name = None

    if test_loader is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = validate_epoch(
            model, test_loader, bce_criterion, device, -1
        )

        print("\n" + "="*80)
        print("OOD/Test Results")
        print("="*80)
        print(f"\nDataset: {test_dataset_name}")
        print(f"Samples: {len(test_ds):,}\n")
        print(f"Performance Metrics:")
        print(f"  AUC-ROC:  {test_metrics['auc']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Loss:     {test_metrics['loss']:.4f}")
        print("\n" + "="*80)

        results['test_auc'] = test_metrics['auc']
        results['test_acc'] = test_metrics['accuracy']
        results['test_f1'] = test_metrics['f1']

    # Save results
    results_path = exp_dir / 'results' / 'training_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Best model saved to: {checkpoint_path}")
    print(f"\nExperiment completed successfully!")


if __name__ == '__main__':
    main()
