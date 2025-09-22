#!/usr/bin/env python3
"""
AWARE-NET Stage 0: Baseline Model Training Script
Complete training pipeline with experiment tracking
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stage_00.baseline_model import EfficientNetV2B3Baseline, BaselineTrainer
from stage_00.dataset import create_data_loaders
from utils.experiment_utils import ExperimentManager, ExperimentConfig
from utils.metrics import AcademicMetrics
from utils.visualization import AcademicVisualizer

class MultiDatasetWrapper(Dataset):
    """
    Wrapper for ConcatDataset that supports get_class_counts() method
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.concat_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

    def get_class_counts(self) -> torch.Tensor:
        """
        Get total class counts across all datasets

        Returns:
            Tensor with [real_count, fake_count]
        """
        total_real = 0
        total_fake = 0

        for dataset in self.datasets:
            if hasattr(dataset, 'get_class_counts'):
                counts = dataset.get_class_counts()
                total_real += counts[0].item()
                total_fake += counts[1].item()
            else:
                # Fallback: count labels in dataset
                labels = [dataset[i][1].item() for i in range(len(dataset))]
                total_real += labels.count(0)
                total_fake += labels.count(1)

        return torch.tensor([total_real, total_fake], dtype=torch.float)

class UnifiedDeepfakeDataset(Dataset):
    """Dataset class for multi-dataset deepfake detection"""
    
    def __init__(self, manifest_path, dataset_name, transform=None, subset_ratio=1.0):
        self.manifest_path = Path(manifest_path)
        self.dataset_name = dataset_name
        self.transform = transform
        
        # Load manifest
        self.data = pd.read_csv(manifest_path)
        print(f"Loaded {len(self.data)} samples from {dataset_name}")
        
        # Filter valid samples
        if 'valid' in self.data.columns:
            valid_mask = self.data['valid'] == True
            self.data = self.data[valid_mask]
            print(f"After filtering: {len(self.data)} valid samples from {dataset_name}")
        
        # Apply subset ratio if less than 1.0
        if subset_ratio < 1.0:
            # Stratified sampling to maintain class balance
            real_samples = self.data[self.data['label'] == 0]
            fake_samples = self.data[self.data['label'] == 1]
            
            real_subset_size = int(len(real_samples) * subset_ratio)
            fake_subset_size = int(len(fake_samples) * subset_ratio)
            
            # Randomly sample while maintaining reproducibility
            real_subset = real_samples.sample(n=real_subset_size, random_state=42)
            fake_subset = fake_samples.sample(n=fake_subset_size, random_state=42)
            
            self.data = pd.concat([real_subset, fake_subset]).reset_index(drop=True)
            print(f"Subset to {subset_ratio*100:.1f}%: {len(self.data)} samples from {dataset_name}")
        
        # Print label distribution
        label_counts = self.data['label'].value_counts().sort_index()
        print(f"{dataset_name} label distribution: {dict(label_counts)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = sample['image_path']
        label = int(sample['label'])
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.float)
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_image = self.transform(Image.new('RGB', (256, 256), 'black'))
            else:
                black_image = torch.zeros(3, 256, 256)
            return black_image, torch.tensor(0.0, dtype=torch.float)

def create_multi_dataset_loaders(config_path, batch_size=32, num_workers=0, subset_ratio=1.0):
    """Create data loaders for multi-dataset training
    
    Args:
        config_path: Path to unified dataset configuration
        batch_size: Batch size for data loaders
        num_workers: Number of worker threads
        subset_ratio: Fraction of data to use (0.1 = 10%, 1.0 = 100%)
    """
    
    # Load unified configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Loading multi-dataset configuration...")
    print(f"Datasets: {config['metadata']['datasets_included']}")
    print(f"Total samples: {config['metadata']['total_samples']:,}")
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets for each split
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # Process each dataset
    for dataset_name, dataset_config in config['datasets'].items():
        if not dataset_config['enabled']:
            print(f"Skipping disabled dataset: {dataset_name}")
            continue
            
        print(f"\\nProcessing {dataset_name}...")
        manifests = dataset_config['manifests']
        
        # Create datasets for each split
        train_dataset = UnifiedDeepfakeDataset(
            manifest_path=manifests['train'],
            dataset_name=f"{dataset_name}_train",
            transform=transform,
            subset_ratio=subset_ratio
        )
        
        val_dataset = UnifiedDeepfakeDataset(
            manifest_path=manifests['val'], 
            dataset_name=f"{dataset_name}_val",
            transform=transform,
            subset_ratio=subset_ratio
        )
        
        test_dataset = UnifiedDeepfakeDataset(
            manifest_path=manifests['test'],
            dataset_name=f"{dataset_name}_test", 
            transform=transform,
            subset_ratio=subset_ratio
        )
        
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
    
    # Combine datasets using our custom wrapper
    combined_train = MultiDatasetWrapper(train_datasets)
    combined_val = MultiDatasetWrapper(val_datasets)
    combined_test = MultiDatasetWrapper(test_datasets)
    
    print(f"\\nCombined dataset sizes:")
    print(f"Train: {len(combined_train):,} samples")
    print(f"Val: {len(combined_val):,} samples") 
    print(f"Test: {len(combined_test):,} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        combined_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

def setup_device_with_fallback():
    """
    Intelligent GPU detection with fallback options
    
    Returns:
        Tuple of (device, gpu_info_dict or None)
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu'), None
    
    # Check all available GPUs and their capabilities
    compatible_gpus = []
    all_gpus = []
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        gpu_info = {
            'id': i,
            'name': gpu_name,
            'capability': f"{capability[0]}.{capability[1]}",
            'capability_int': capability[0] * 10 + capability[1],
            'memory_gb': memory_gb,
            'arch': f"sm_{capability[0]}{capability[1]}"
        }
        all_gpus.append(gpu_info)
        
        # Test GPU compatibility
        try:
            torch.cuda.set_device(i)
            test_tensor = torch.zeros(1, device=f'cuda:{i}')
            compatible_gpus.append(gpu_info)
            print(f"[OK] GPU {i}: {gpu_name} - Compatible")
        except Exception as e:
            print(f"[FAIL] GPU {i}: {gpu_name} - Incompatible ({str(e)[:50]})")
    
    if compatible_gpus:
        # Select the best compatible GPU (highest memory)
        best_gpu = max(compatible_gpus, key=lambda x: x['memory_gb'])
        torch.cuda.set_device(best_gpu['id'])
        device = torch.device(f"cuda:{best_gpu['id']}")
        print(f"[SELECTED] GPU {best_gpu['id']}: {best_gpu['name']}")
        return device, best_gpu
    
    # No compatible GPU found, provide upgrade suggestions
    print("[WARNING] No compatible GPU found!")
    
    for gpu in all_gpus:
        if gpu['arch'] in ['sm_120']:  # RTX 5060 Ti, 5090, etc.
            print(f"   {gpu['name']} requires PyTorch nightly build")
            print(f"   Suggested command: conda install pytorch-cuda=12.6 pytorch pytorch2-nightly -c pytorch-nightly -c nvidia")
        elif gpu['capability_int'] < 50:  # Very old GPUs
            print(f"   {gpu['name']} is too old (compute capability {gpu['capability']})")
    
    print("[INFO] Falling back to CPU training...")
    print("   For GPU acceleration, consider:")
    print("   1. Using a compatible GPU (RTX 3070Ti, RTX 4080, etc.)")
    print("   2. Installing PyTorch nightly for RTX 50-series support")
    
    return torch.device('cpu'), None

def setup_training(config: Dict[str, Any]) -> tuple:
    """
    Setup training components with intelligent GPU detection
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Tuple of (model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion)
    """
    # Intelligent GPU setup with fallback options
    device, gpu_info = setup_device_with_fallback()
    print(f"Selected device: {device}")
    
    if gpu_info:
        print(f"GPU: {gpu_info['name']}")
        print(f"Compute Capability: {gpu_info['capability']}")
        print(f"GPU Memory: {gpu_info['memory_gb']:.1f} GB")
        
        # Adjust batch size based on GPU memory
        if gpu_info['memory_gb'] < 8:
            suggested_batch = min(config['training']['batch_size'], 16)
            if suggested_batch < config['training']['batch_size']:
                print(f"[WARNING] Reducing batch size to {suggested_batch} due to GPU memory limitation")
                config['training']['batch_size'] = suggested_batch
    
    # Create data loaders - check if it's unified multi-dataset config
    print("\\nCreating data loaders...")
    
    # Check if this is a unified dataset configuration
    config_path = config['data']['config_path']
    if 'unified' in str(config_path).lower() or config.get('metadata', {}).get('dataset_type') == 'unified_multi_dataset':
        # Use multi-dataset loader
        # Check if subset ratio is specified in config or use command line argument
        subset_ratio = config.get('data', {}).get('subset_ratio', 1.0)
        
        train_loader, val_loader, test_loader = create_multi_dataset_loaders(
            config_path=config_path,
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4),
            subset_ratio=subset_ratio
        )
    else:
        # Use single dataset loader
        train_loader, val_loader, test_loader = create_data_loaders(
            config_path=config_path,
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4)
        )
    
    # Create model
    print("\\nCreating baseline model...")
    model = EfficientNetV2B3Baseline(
        num_classes=1,  # Changed to 1 for true BCE
        pretrained=True,
        dropout_rate=config['model']['dropout_rate']
    )
    model = model.to(device)
    
    print(f"Model info: {model.get_model_info()}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )
    
    # Setup loss function with class weighting for BCE
    if config['training'].get('use_class_weights', True):
        # Check if pos_weight is pre-calculated in config (for multi-dataset)
        if 'pos_weight' in config['training']:
            pos_weight = torch.tensor([config['training']['pos_weight']], device=device)
            print(f"Using pre-calculated pos_weight: {pos_weight.item():.4f}")
        else:
            # Calculate from dataset for single dataset case
            try:
                class_counts = train_loader.dataset.get_class_counts()
                real_count, fake_count = class_counts[0], class_counts[1]
                
                # For BCE, pos_weight = count_negative_samples / count_positive_samples
                # In our case: real_count (label 0) / fake_count (label 1)
                pos_weight = real_count / fake_count
                
                print(f"Dataset distribution:")
                print(f"  Real samples (label 0): {real_count:.0f} ({100*real_count/(real_count+fake_count):.1f}%)")
                print(f"  Fake samples (label 1): {fake_count:.0f} ({100*fake_count/(real_count+fake_count):.1f}%)")
                print(f"  Calculated pos_weight for BCE: {pos_weight.item():.4f}")
            except:
                # Fallback to default pos_weight for multi-dataset
                pos_weight = torch.tensor([2.33], device=device)  
                print(f"Using default pos_weight: {pos_weight.item():.4f}")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    return model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, device

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                epoch: int) -> Dict[str, float]:
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Training device
        epoch: Current epoch number
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d} [Train]')
    
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        # Convert targets to float for BCE and reshape to [batch_size, 1]
        targets = targets.float().unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        # For BCE, use sigmoid threshold of 0.5
        predicted = (torch.sigmoid(logits) > 0.5).float()
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }

def validate_epoch(model: nn.Module,
                   val_loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   epoch: int) -> Dict[str, float]:
    """
    Validate for one epoch
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Training device
        epoch: Current epoch number
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch:02d} [Val]')
    
    with torch.no_grad():
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            # Convert targets to float for BCE and reshape to [batch_size, 1]
            targets_bce = targets.float().unsqueeze(1)
            
            # Forward pass
            logits = model(data)
            loss = criterion(logits, targets_bce)
            
            # Get predictions and probabilities using sigmoid
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities > 0.5).float()
            
            # Statistics
            total_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets_bce).sum().item()
            
            # Store for metrics calculation (flatten for sklearn compatibility)
            all_predictions.extend(predicted.squeeze().cpu().numpy())
            all_probabilities.extend(probabilities.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())  # Keep original targets as integers
            
            # Update progress bar
            accuracy = 100. * correct / total
            avg_loss = total_loss / (len(all_targets) // val_loader.batch_size + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    # Calculate additional metrics
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Calculate AUC using probabilities for binary classification
    from sklearn.metrics import roc_auc_score, f1_score
    try:
        # For binary classification, use probabilities directly
        auc_score = roc_auc_score(all_targets, all_probabilities)
        f1 = f1_score(all_targets, all_predictions)
    except:
        auc_score = 0.0
        f1 = 0.0
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': correct / total,
        'auc': auc_score,
        'f1': f1,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets
    }

def main():
    parser = argparse.ArgumentParser(description='AWARE-NET Baseline Training')
    parser.add_argument('--config', type=str, default='configs/training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--experiment-name', type=str, default='baseline_efficientnet',
                       help='Name of the experiment')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--subset-ratio', type=float, default=1.0,
                       help='Fraction of dataset to use (0.1 = 10%, 1.0 = 100%)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Disable pretrained weights (start from random initialization)')
    parser.add_argument('--model', type=str, default='tf_efficientnetv2_b0',
                       choices=['tf_efficientnetv2_b0', 'efficientnetv2_rw_t'],
                       help='EfficientNetV2 model variant to use')
    
    args = parser.parse_args()
    
    # Create default configuration
    config = {
        'experiment': {
            'name': args.experiment_name,
            'description': 'EfficientNetV2-B3 baseline for CelebDF-v2',
            'tags': ['baseline', 'efficientnet', 'stage0']
        },
        'data': {
            'config_path': 'configs/unified_dataset_config.json',
            'image_size': 256,
            'subset_ratio': args.subset_ratio
        },
        'model': {
            'architecture': args.model,
            'num_classes': 1,  # Changed to 1 for true BCE
            'pretrained': not args.no_pretrained,
            'dropout_rate': 0.2
        },
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': 1e-4,
            'num_workers': 4,
            'use_class_weights': True,
            'early_stopping_patience': 10,
            'save_checkpoints': True
        }
    }
    
    # Load config file if it exists and merge with command line args
    if Path(args.config).exists():
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        
        # Store command line arguments that should override config file
        cmd_overrides = {
            'training': {
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'learning_rate': args.learning_rate
            },
            'experiment': {
                'name': args.experiment_name
            }
        }
        
        # Start with file config as base
        config.update(file_config)
        
        # Override with command line arguments
        if 'training' not in config:
            config['training'] = {}
        config['training'].update(cmd_overrides['training'])
        
        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment'].update(cmd_overrides['experiment'])
    
    print("Training Configuration:")
    print(json.dumps(config, indent=2))
    
    # Create experiment configuration
    exp_config = ExperimentConfig(
        experiment_name=config['experiment']['name'],
        model_name=config['model']['architecture'],
        dataset_name='celebdf_v2',
        description=config['experiment']['description'],
        tags=config['experiment']['tags'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['num_epochs'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup experiment manager
    exp_manager = ExperimentManager(base_path="experiments")
    
    # Run training in experiment context
    with exp_manager.experiment_context(exp_config) as experiment_id:
        print(f"\\nStarted experiment: {experiment_id}")
        
        # Setup training components
        model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, device = setup_training(config)
        
        # Training variables
        best_val_auc = 0.0
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': [],
            'val_f1': []
        }
        
        print(f"\\nStarting training for {config['training']['num_epochs']} epochs...")
        print("=" * 80)
        
        for epoch in range(1, config['training']['num_epochs'] + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            
            # Validation phase  
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch
            )
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics to experiment manager
            exp_manager.log_metrics(train_metrics, split='train', epoch=epoch)
            exp_manager.log_metrics({
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy'],
                'auc': val_metrics['auc'],
                'f1': val_metrics['f1']
            }, split='val', epoch=epoch)
            
            # Update history
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_auc'].append(val_metrics['auc'])
            training_history['val_f1'].append(val_metrics['f1'])
            
            # Save checkpoint if this is the best model
            is_best = val_metrics['auc'] > best_val_auc
            if is_best:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                
                # Save checkpoint
                exp_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=True
                )
                
                print(f"New best model saved! AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch:02d} Summary:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"  Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Best AUC: {best_val_auc:.4f}, Patience: {patience_counter}/{config['training']['early_stopping_patience']}")
            print("-" * 80)
            
            # Early stopping
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        print("\\nTraining completed!")
        print(f"Best validation AUC: {best_val_auc:.4f}")
        
        # Final evaluation on test set
        print("\\nEvaluating on test set...")
        test_metrics = validate_epoch(
            model, test_loader, criterion, device,
            epoch=-1  # Special epoch for test
        )
        
        # Log test metrics
        exp_manager.log_metrics({
            'loss': test_metrics['loss'],
            'accuracy': test_metrics['accuracy'],
            'auc': test_metrics['auc'],
            'f1': test_metrics['f1']
        }, split='test')
        
        print("Final Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        
        print(f"\\nExperiment {experiment_id} completed successfully!")
        return experiment_id

if __name__ == "__main__":
    experiment_id = main()
    print(f"\\nTraining finished. Experiment ID: {experiment_id}")