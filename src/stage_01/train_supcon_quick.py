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

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc_score,
        'f1': f1,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets
    }


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
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0'],
                       help='Dataset to exclude (for LODO)')
    parser.add_argument('--test-dataset', type=str, default=None,
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0'],
                       help='OOD test dataset')

    # Experiment parameters
    parser.add_argument('--experiment-name', type=str, default='supcon_quick_validation',
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

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

    # Determine datasets to use
    all_datasets = ['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0']
    if args.exclude_dataset:
        train_datasets = [d for d in all_datasets if d != args.exclude_dataset]
        print(f"LODO Training: Using {train_datasets}, excluding {args.exclude_dataset}\n")
    else:
        train_datasets = all_datasets
        print(f"Multi-dataset training: {train_datasets}\n")

    # Create datasets
    print("Loading datasets...")
    train_dataset_list = []
    val_dataset_list = []

    # Map dataset names to manifest file names
    dataset_name_mapping = {
        'celebdf_v2': 'celebdf_v2',
        'faceforensics_plus_plus': 'faceforensics',
        'deeperforensics_1_0': 'deeperforensics'
    }

    for dataset_name in train_datasets:
        manifest_name = dataset_name_mapping[dataset_name]

        # Train dataset
        train_ds = UnifiedDeepfakeDataset(
            manifest_path=f'manifests/{manifest_name}_train_{args.dataset_mode}.csv',
            dataset_name=f'{dataset_name}_train',
            transform=None,
            use_augmentation=True
        )
        train_dataset_list.append(train_ds)

        # Val dataset
        val_ds = UnifiedDeepfakeDataset(
            manifest_path=f'manifests/{manifest_name}_val_{args.dataset_mode}.csv',
            dataset_name=f'{dataset_name}_val',
            transform=None,
            use_augmentation=False
        )
        val_dataset_list.append(val_ds)

    # Wrap datasets
    train_dataset = MultiDatasetWrapper(train_dataset_list)
    val_dataset = MultiDatasetWrapper(val_dataset_list)

    print(f"✓ Training samples: {len(train_dataset):,}")
    print(f"✓ Validation samples: {len(val_dataset):,}\n")

    # Create data loaders
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

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    # Training loop
    print("="*80)
    print(f"Starting SupCon Quick Validation ({args.epochs} epochs)")
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
    if args.test_dataset:
        print(f"\nEvaluating on OOD test set: {args.test_dataset}")
        print("-"*80)

        test_manifest_name = dataset_name_mapping[args.test_dataset]
        test_ds = UnifiedDeepfakeDataset(
            manifest_path=f'manifests/{test_manifest_name}_test_{args.dataset_mode}.csv',
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

        # Load best model
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        test_metrics = validate_epoch(
            model, test_loader, bce_criterion, device, -1
        )

        print("\n" + "="*80)
        print("OOD Test Results")
        print("="*80)
        print(f"\nDataset: {args.test_dataset}")
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
