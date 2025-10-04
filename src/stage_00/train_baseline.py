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
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stage_00.baseline_model import EfficientNetV2B3Baseline, BaselineTrainer
from stage_00.dataset import create_data_loaders
from utils.experiment_utils import ExperimentManager, ExperimentConfig
from utils.metrics import AcademicMetrics
from utils.visualization import AcademicVisualizer

def calculate_dataset_weights_from_manifests(dataset_config):
    """Calculate dataset weights based on actual manifest sample counts"""
    dataset_manifests = {
        'celebdf_v2': 'manifests/celebdf_v2_train.csv',
        'faceforensics_plus_plus': 'manifests/faceforensics_train.csv',
        'deeperforensics_1_0': 'manifests/deeperforensics_train.csv'
    }

    dataset_counts = {}
    total_samples = 0

    for dataset_name, manifest_path in dataset_manifests.items():
        manifest_file = Path(manifest_path)
        if manifest_file.exists():
            df = pd.read_csv(manifest_file)
            count = len(df)
            dataset_counts[dataset_name] = count
            total_samples += count
        else:
            dataset_counts[dataset_name] = 0

    # Calculate proportional weights
    weights = {}
    if total_samples > 0:
        for dataset_name, count in dataset_counts.items():
            weight = round(count / total_samples, 3)
            weights[dataset_name] = weight

        print(f"📊 Dataset sample distribution (total: {total_samples:,}):")
        for dataset_name, count in dataset_counts.items():
            weight = weights.get(dataset_name, 0)
            print(f"  {dataset_name}: {count:,} samples ({weight:.1%})")

    return weights

class MultiDatasetWrapper(Dataset):
    """
    Wrapper for ConcatDataset that supports get_class_counts() method
    and tracks dataset_id for per-dataset metrics
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.concat_dataset = ConcatDataset(datasets)

        # Build mapping from sample index to dataset_id
        self._build_dataset_mapping()

    def _build_dataset_mapping(self):
        """Build mapping from sample index to (dataset_id, dataset_name)"""
        self.dataset_mapping = []
        self.dataset_names = []

        for dataset_id, dataset in enumerate(self.datasets):
            dataset_name = getattr(dataset, 'dataset_name', f'Dataset_{dataset_id}')
            # Store dataset name for reporting
            if dataset_name not in self.dataset_names:
                self.dataset_names.append(dataset_name)

            # Map each sample index to its dataset_id
            for _ in range(len(dataset)):
                self.dataset_mapping.append(dataset_id)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        image, label = self.concat_dataset[idx]
        dataset_id = self.dataset_mapping[idx]
        return image, label, dataset_id

    def get_dataset_name(self, dataset_id: int) -> str:
        """Get dataset name from dataset_id"""
        if dataset_id < len(self.datasets):
            return getattr(self.datasets[dataset_id], 'dataset_name', f'Dataset_{dataset_id}')
        return f'Unknown_{dataset_id}'

    def get_class_counts(self) -> torch.Tensor:
        """
        Get total class counts across all datasets

        Returns:
            Tensor with [real_count, fake_count]
        """
        total_real = 0
        total_fake = 0

        print(f"\n📊 Calculating class counts for {len(self.datasets)} datasets...")

        for idx, dataset in enumerate(self.datasets):
            dataset_name = getattr(dataset, 'dataset_name', f'Dataset {idx+1}')

            if hasattr(dataset, 'get_class_counts'):
                counts = dataset.get_class_counts()
                total_real += counts[0].item()
                total_fake += counts[1].item()
                print(f"  ✓ {dataset_name}: {counts[0]:.0f} real, {counts[1]:.0f} fake")
            else:
                # Fallback: count labels in dataset with progress bar
                print(f"  ⏳ Counting labels for {dataset_name} ({len(dataset):,} samples)...")
                labels = []

                # Use tqdm for progress bar
                from tqdm import tqdm
                for i in tqdm(range(len(dataset)), desc=f"    Scanning {dataset_name}", unit="samples"):
                    labels.append(dataset[i][1].item())

                real_count = labels.count(0)
                fake_count = labels.count(1)
                total_real += real_count
                total_fake += fake_count
                print(f"  ✓ {dataset_name}: {real_count:,} real, {fake_count:,} fake")

        print(f"\n📈 Total class counts:")
        print(f"  Real (label 0): {total_real:,}")
        print(f"  Fake (label 1): {total_fake:,}")
        print(f"  Balance ratio: {total_real/total_fake:.4f}")

        return torch.tensor([total_real, total_fake], dtype=torch.float)

class UnifiedDeepfakeDataset(Dataset):
    """Dataset class for multi-dataset deepfake detection with albumentations support"""

    def __init__(self, manifest_path, dataset_name, transform=None, subset_ratio=1.0, use_augmentation=False):
        self.manifest_path = Path(manifest_path)
        self.dataset_name = dataset_name
        self.transform = transform  # torchvision transforms (kept for backward compatibility)
        self.use_augmentation = use_augmentation

        # Build albumentations pipeline if augmentation is enabled
        if use_augmentation:
            # SupCon standard augmentation pipeline (enhanced for better generalization)
            self.albu_transform = A.Compose([
                # Spatial augmentations (invariance)
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3, border_mode=0),
                A.RandomResizedCrop(
                    size=(256, 256),
                    scale=(0.7, 1.0),  # More aggressive: 0.8→0.7
                    p=0.8  # Increased from 0.5
                ),

                # Color augmentations (SupCon standard: stronger)
                A.ColorJitter(
                    brightness=0.4,  # Doubled: 0.2→0.4
                    contrast=0.4,    # Doubled: 0.2→0.4
                    saturation=0.2,  # Doubled: 0.1→0.2
                    hue=0.1,         # Doubled: 0.05→0.1
                    p=0.8  # Increased from 0.5
                ),
                A.ToGray(p=0.2),  # SupCon standard (SimCLR)

                # Noise/blur augmentations
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Increased from 0.2
                A.GaussNoise(std_range=(0.02, 0.10), p=0.3),  # Normalized range: 0-1 (was 10-50/255)

                # Deepfake-specific: JPEG compression simulation
                A.ImageCompression(
                    quality_lower=70,
                    quality_upper=100,
                    p=0.3  # Simulate video compression artifacts
                ),

                # Normalization and conversion to tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            print(f"✓ SupCon augmentation enabled for {dataset_name}")
        else:
            # No augmentation mode - just normalize
            self.albu_transform = A.Compose([
                A.Resize(height=256, width=256),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

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

    def get_class_counts(self) -> torch.Tensor:
        """
        Get class counts directly from manifest (fast, no image loading)

        Returns:
            Tensor with [real_count, fake_count]
        """
        label_counts = self.data['label'].value_counts().sort_index()
        real_count = label_counts.get(0, 0)
        fake_count = label_counts.get(1, 0)
        return torch.tensor([real_count, fake_count], dtype=torch.float)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = sample['image_path']
        label = int(sample['label'])

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)  # Convert to numpy for albumentations

            # Apply albumentations transforms
            transformed = self.albu_transform(image=image_np)
            image_tensor = transformed['image']

            return image_tensor, torch.tensor(label, dtype=torch.float)

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            black_image_np = np.zeros((256, 256, 3), dtype=np.uint8)
            transformed = self.albu_transform(image=black_image_np)
            return transformed['image'], torch.tensor(0.0, dtype=torch.float)

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
    
    # Create data loaders using simplified configuration
    print("\\nCreating data loaders...")

    # Load dataset configuration to get manifest paths
    with open(config['data']['dataset_config'], 'r') as f:
        dataset_config = json.load(f)

    # Check if multi-dataset training is enabled
    multi_dataset = config['data'].get('multi_dataset', False)
    dataset_mode = config['data'].get('dataset_mode', 'original')

    if multi_dataset:
        # Multi-dataset training mode
        print("🔄 Multi-dataset training mode enabled")

        # Get list of datasets to exclude (for LODO evaluation)
        exclude_datasets = config['data'].get('exclude_datasets', [])
        if exclude_datasets:
            print(f"📌 LODO Mode: Excluding dataset(s): {', '.join(exclude_datasets)}")

        # Build manifest paths for all datasets
        dataset_short_names = {
            'celebdf_v2': 'celebdf_v2',
            'faceforensics_plus_plus': 'faceforensics',
            'deeperforensics_1_0': 'deeperforensics'
        }

        # Remove excluded datasets
        for excluded in exclude_datasets:
            if excluded in dataset_short_names:
                del dataset_short_names[excluded]
                print(f"  ✓ Removed {excluded} from training datasets")

        # Validate at least one dataset remains
        if len(dataset_short_names) == 0:
            raise ValueError("Cannot exclude all datasets! At least one dataset must be included for training.")

        print(f"📂 Training on {len(dataset_short_names)} dataset(s): {list(dataset_short_names.keys())}")

        # Create unified config for multi-dataset loader
        transform = transforms.Compose([
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        train_datasets = []
        val_datasets = []
        test_datasets = []

        for full_name, short_name in dataset_short_names.items():
            # Build manifest paths based on mode
            if dataset_mode == 'anonymized_balanced':
                manifests = {
                    'train': f'manifests/{short_name}_train_anonymized_balanced.csv',
                    'val': f'manifests/{short_name}_val_anonymized_balanced.csv',
                    'test': f'manifests/{short_name}_test_anonymized_balanced.csv'
                }
            elif dataset_mode == 'balanced':
                # NEW: Use balanced manifests with original paths
                manifests = {
                    'train': f'manifests/{short_name}_train_balanced.csv',
                    'val': f'manifests/{short_name}_val_balanced.csv',
                    'test': f'manifests/{short_name}_test_balanced.csv'
                }
            elif dataset_mode == 'anonymized':
                manifests = {
                    'train': f'manifests/{short_name}_train_anonymized.csv',
                    'val': f'manifests/{short_name}_val_anonymized.csv',
                    'test': f'manifests/{short_name}_test_anonymized.csv'
                }
            else:
                manifests = {
                    'train': f'manifests/{short_name}_train.csv',
                    'val': f'manifests/{short_name}_val.csv',
                    'test': f'manifests/{short_name}_test.csv'
                }

            # Check if manifest files exist
            if not Path(manifests['train']).exists():
                print(f"⚠️  Skipping {full_name}: manifest not found at {manifests['train']}")
                continue

            print(f"📂 Loading {full_name} ({dataset_mode} mode)...")

            # Get augmentation setting (only for training split)
            use_train_augmentation = config['data'].get('augmentation', False)

            # Create datasets
            train_ds = UnifiedDeepfakeDataset(
                manifest_path=manifests['train'],
                dataset_name=f"{full_name}_train",
                transform=None,  # Not used anymore (albumentations handles everything)
                subset_ratio=config['data'].get('subset_ratio', 1.0),
                use_augmentation=use_train_augmentation  # Enable augmentation for training
            )
            val_ds = UnifiedDeepfakeDataset(
                manifest_path=manifests['val'],
                dataset_name=f"{full_name}_val",
                transform=None,
                subset_ratio=config['data'].get('subset_ratio', 1.0),
                use_augmentation=False  # No augmentation for validation
            )
            test_ds = UnifiedDeepfakeDataset(
                manifest_path=manifests['test'],
                dataset_name=f"{full_name}_test",
                transform=None,
                subset_ratio=config['data'].get('subset_ratio', 1.0),
                use_augmentation=False  # No augmentation for test
            )

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

        # Combine datasets
        combined_train = MultiDatasetWrapper(train_datasets)
        combined_val = MultiDatasetWrapper(val_datasets)
        combined_test = MultiDatasetWrapper(test_datasets)

        print(f"\\n📊 Combined dataset sizes:")
        print(f"  Train: {len(combined_train):,} samples")
        print(f"  Val: {len(combined_val):,} samples")
        print(f"  Test: {len(combined_test):,} samples")

        # Calculate dataset weights for balanced sampling
        use_dataset_weights = config['data'].get('use_dataset_weights', True)
        sampler = None

        if use_dataset_weights:
            print(f"\\n⚖️  Calculating dataset-level weights for balanced sampling...")

            # Calculate weights: inverse of dataset size (so smaller datasets get more weight)
            dataset_sizes = [len(ds) for ds in train_datasets]
            total_samples = sum(dataset_sizes)

            # Create sample weights: each sample gets weight = 1 / (dataset_size)
            # This ensures each dataset contributes equally regardless of size
            sample_weights = []
            for ds_idx, ds_size in enumerate(dataset_sizes):
                weight_per_sample = 1.0 / ds_size
                sample_weights.extend([weight_per_sample] * ds_size)

            # Normalize weights
            sample_weights = torch.tensor(sample_weights, dtype=torch.float)
            sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

            # Print dataset contribution
            print(f"  Dataset contributions (with balanced sampling):")
            cumsum = 0
            for idx, (ds, size) in enumerate(zip(train_datasets, dataset_sizes)):
                ds_name = getattr(ds, 'dataset_name', f'Dataset {idx+1}')
                original_ratio = size / total_samples * 100
                # Each dataset will be sampled equally in expectation
                balanced_ratio = 100.0 / len(train_datasets)
                print(f"    {ds_name}: {size:,} samples ({original_ratio:.1f}% → {balanced_ratio:.1f}%)")
                cumsum += size

            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            print(f"  ✓ Weighted sampler created (balances dataset contributions)")

        # Create data loaders
        train_loader = DataLoader(
            combined_train,
            batch_size=config['training']['batch_size'],
            shuffle=(sampler is None),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )

        val_loader = DataLoader(
            combined_val,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )

        test_loader = DataLoader(
            combined_test,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )

    else:
        # Single dataset training mode
        dataset_name = config['data']['dataset_name']

        # Build manifest paths dynamically
        dataset_short_names = {
            'celebdf_v2': 'celebdf_v2',
            'faceforensics_plus_plus': 'faceforensics',
            'deeperforensics_1_0': 'deeperforensics'
        }

        short_name = dataset_short_names.get(dataset_name, dataset_name)

        if dataset_mode == 'anonymized_balanced':
            manifests = {
                'train': f'manifests/{short_name}_train_anonymized_balanced.csv',
                'val': f'manifests/{short_name}_val_anonymized_balanced.csv',
                'test': f'manifests/{short_name}_test_anonymized_balanced.csv'
            }
            print(f"Using anonymized + balanced {dataset_name} dataset")
        elif dataset_mode == 'balanced':
            manifests = {
                'train': f'manifests/{short_name}_train_balanced.csv',
                'val': f'manifests/{short_name}_val_balanced.csv',
                'test': f'manifests/{short_name}_test_balanced.csv'
            }
            print(f"Using balanced {dataset_name} dataset (original paths, 50/50 balanced)")
        elif dataset_mode == 'anonymized':
            manifests = {
                'train': f'manifests/{short_name}_train_anonymized.csv',
                'val': f'manifests/{short_name}_val_anonymized.csv',
                'test': f'manifests/{short_name}_test_anonymized.csv'
            }
            print(f"Using anonymized {dataset_name} dataset")
        else:
            if dataset_name in dataset_config['datasets']:
                manifests = dataset_config['datasets'][dataset_name]['manifests']
                print(f"Using original {dataset_name} dataset manifests")
            else:
                manifests = {
                    'train': f'manifests/{short_name}_train.csv',
                    'val': f'manifests/{short_name}_val.csv',
                    'test': f'manifests/{short_name}_test.csv'
                }
                print(f"Using original {dataset_name} dataset (auto-generated paths)")

        # Create datasets directly with manifest paths
        from stage_00.dataset import CelebDFDataset

        train_dataset = CelebDFDataset(
            manifest_path=manifests['train'],
            root_path=".",
            image_size=config['data']['image_size'],
            augmentation=True,
            normalize=True,
            path_leakage_check=True
        )

        val_dataset = CelebDFDataset(
            manifest_path=manifests['val'],
            root_path=".",
            image_size=config['data']['image_size'],
            augmentation=False,
            normalize=True,
            path_leakage_check=True
        )

        test_dataset = CelebDFDataset(
            manifest_path=manifests['test'],
            root_path=".",
            image_size=config['data']['image_size'],
            augmentation=False,
            normalize=True,
            path_leakage_check=True
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )
    
    # Create model
    print("\\nCreating baseline model...")
    model = EfficientNetV2B3Baseline(
        num_classes=1,  # Changed to 1 for true BCE
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate'],
        model_name=config['model']['name']
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

    for batch_idx, batch_data in enumerate(pbar):
        # Handle both 2-tuple (image, label) and 3-tuple (image, label, dataset_id)
        if len(batch_data) == 3:
            data, targets, _ = batch_data  # Ignore dataset_id during training
        else:
            data, targets = batch_data

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
    Validate for one epoch with per-dataset metrics breakdown

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Training device
        epoch: Current epoch number

    Returns:
        Dictionary of validation metrics (overall + per-dataset)
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_dataset_ids = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch:02d} [Val]')

    with torch.no_grad():
        for batch_data in pbar:
            # Handle both 2-tuple and 3-tuple
            if len(batch_data) == 3:
                data, targets, dataset_ids = batch_data
                has_dataset_id = True
            else:
                data, targets = batch_data
                dataset_ids = None
                has_dataset_id = False

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

            if has_dataset_id:
                all_dataset_ids.extend(dataset_ids.cpu().numpy())

            # Update progress bar
            accuracy = 100. * correct / total
            avg_loss = total_loss / (len(all_targets) // val_loader.batch_size + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)

    # Calculate overall metrics
    from sklearn.metrics import roc_auc_score, f1_score
    try:
        # For binary classification, use probabilities directly
        auc_score = roc_auc_score(all_targets, all_probabilities)
        f1 = f1_score(all_targets, all_predictions)
    except:
        auc_score = 0.0
        f1 = 0.0

    result = {
        'loss': total_loss / len(val_loader),
        'accuracy': correct / total,
        'auc': auc_score,
        'f1': f1,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets
    }

    # Calculate per-dataset metrics if dataset_ids available
    if len(all_dataset_ids) > 0:
        all_dataset_ids = np.array(all_dataset_ids)
        per_dataset_metrics = {}

        # Get dataset wrapper to retrieve dataset names
        dataset_wrapper = val_loader.dataset
        if hasattr(dataset_wrapper, 'get_dataset_name'):
            unique_dataset_ids = np.unique(all_dataset_ids)

            for dataset_id in unique_dataset_ids:
                # Get indices for this dataset
                dataset_mask = all_dataset_ids == dataset_id
                dataset_targets = all_targets[dataset_mask]
                dataset_predictions = all_predictions[dataset_mask]
                dataset_probabilities = all_probabilities[dataset_mask]

                # Calculate metrics for this dataset
                try:
                    dataset_auc = roc_auc_score(dataset_targets, dataset_probabilities)
                    dataset_f1 = f1_score(dataset_targets, dataset_predictions)
                    dataset_acc = (dataset_predictions == dataset_targets).mean()
                    dataset_name = dataset_wrapper.get_dataset_name(int(dataset_id))

                    per_dataset_metrics[dataset_name] = {
                        'auc': dataset_auc,
                        'f1': dataset_f1,
                        'accuracy': dataset_acc,
                        'num_samples': len(dataset_targets)
                    }
                except Exception as e:
                    print(f"Warning: Could not calculate metrics for dataset {dataset_id}: {e}")

            result['per_dataset_metrics'] = per_dataset_metrics

    return result

def run_evaluation(args):
    """
    Evaluation-only mode: Load checkpoint and test on specified dataset
    Used for LODO generalization testing
    """
    print(f"\n{'='*80}")
    print("LODO Generalization Evaluation Mode")
    print(f"{'='*80}\n")

    # Validate required arguments
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --eval-only mode")
    if not args.test_dataset:
        raise ValueError("--test-dataset is required for --eval-only mode")

    # Setup device
    device, gpu_info = setup_device_with_fallback()
    print(f"Using device: {device}")
    if gpu_info:
        print(f"GPU: {gpu_info.get('name', 'Unknown')}\n")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = EfficientNetV2B3Baseline(
        num_classes=1,
        pretrained=False,
        dropout_rate=0.2,
        model_name=args.model
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded successfully\n")

    # Create test dataset
    dataset_short_names = {
        'celebdf_v2': 'celebdf_v2',
        'faceforensics_plus_plus': 'faceforensics',
        'deeperforensics_1_0': 'deeperforensics'
    }

    short_name = dataset_short_names[args.test_dataset]
    dataset_mode = args.dataset_mode

    if dataset_mode == 'balanced':
        manifest_path = f'manifests/{short_name}_test_balanced.csv'
    else:
        manifest_path = f'manifests/{short_name}_test.csv'

    print(f"Loading test dataset: {args.test_dataset}")
    print(f"Manifest: {manifest_path}")

    # Create test dataset (no augmentation for evaluation)
    test_dataset = UnifiedDeepfakeDataset(
        manifest_path=manifest_path,
        dataset_name=f'{args.test_dataset}_test',
        transform=None,  # Not used (albumentations handles it)
        use_augmentation=False  # No augmentation for test evaluation
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"✓ Dataset loaded: {len(test_dataset):,} samples\n")

    # Run evaluation
    print("Running evaluation...")
    criterion = nn.BCEWithLogitsLoss()
    results = validate_epoch(model, test_loader, criterion, device, epoch=-1)

    # Print results
    print(f"\n{'='*80}")
    print("Out-of-Distribution Test Results")
    print(f"{'='*80}")
    print(f"\nDataset: {args.test_dataset}")
    print(f"Samples: {len(test_dataset):,}")
    print(f"\nPerformance Metrics:")
    print(f"  AUC-ROC:  {results['auc']:.4f}")
    print(f"  F1-Score: {results['f1']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Loss:     {results['loss']:.4f}")

    # Print per-dataset breakdown if available
    if 'per_dataset_metrics' in results and results['per_dataset_metrics']:
        print(f"\nPer-Dataset Breakdown:")
        for dataset_name, metrics in results['per_dataset_metrics'].items():
            print(f"  {dataset_name}:")
            print(f"    AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f} ({metrics['num_samples']:,} samples)")

    print(f"\n{'='*80}")
    print("Evaluation completed successfully!")
    print(f"{'='*80}\n")

    return results

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
                       help='Fraction of dataset to use (0.1 = 10%%, 1.0 = 100%%)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Disable pretrained weights (start from random initialization)')
    parser.add_argument('--model', type=str, default='tf_efficientnetv2_b0',
                       choices=['tf_efficientnetv2_b0', 'tf_efficientnetv2_b3', 'efficientnetv2_rw_t'],
                       help='EfficientNetV2 model variant to use')
    parser.add_argument('--dataset', type=str, default='celebdf_v2',
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0'],
                       help='Dataset to use for training')
    parser.add_argument('--dataset-mode', type=str, default='original',
                       choices=['original', 'anonymized', 'anonymized_balanced', 'balanced'],
                       help='Dataset mode: original, anonymized, anonymized_balanced, or balanced')
    parser.add_argument('--multi-dataset', action='store_true',
                       help='Enable multi-dataset training (uses all three datasets)')
    parser.add_argument('--exclude-dataset', type=str, action='append', default=[],
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0'],
                       help='Exclude specific dataset(s) from multi-dataset training (can be used multiple times for LODO evaluation)')

    # Evaluation-only mode parameters
    parser.add_argument('--eval-only', action='store_true',
                       help='Evaluation mode: load checkpoint and test on dataset (no training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation (required with --eval-only)')
    parser.add_argument('--test-dataset', type=str, default=None,
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0'],
                       help='Dataset to test on for evaluation (required with --eval-only)')

    args = parser.parse_args()

    # Check for evaluation-only mode
    if args.eval_only:
        print("\n==> Entering evaluation-only mode")
        run_evaluation(args)
        return  # Exit after evaluation

    # Create default configuration
    config = {
        'experiment': {
            'name': args.experiment_name,
            'description': 'EfficientNetV2 baseline training',
            'tags': ['baseline', 'efficientnet', 'stage0']
        },
        'data': {
            'dataset_config': 'configs/datasets.json',
            'dataset_name': args.dataset,
            'dataset_mode': args.dataset_mode,
            'multi_dataset': args.multi_dataset,
            'exclude_datasets': args.exclude_dataset,
            'image_size': 256,
            'subset_ratio': args.subset_ratio
        },
        'model': {
            'name': args.model,
            'architecture': args.model,
            'num_classes': 1,
            'pretrained': not args.no_pretrained,
            'dropout_rate': 0.2
        },
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'epochs': args.epochs,
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

        # Start with file config as base
        config.update(file_config)

        # Command line arguments override config file
        if 'training' not in config:
            config['training'] = {}
        config['training']['batch_size'] = args.batch_size
        config['training']['num_epochs'] = args.epochs
        config['training']['epochs'] = args.epochs
        config['training']['learning_rate'] = args.learning_rate

        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment']['name'] = args.experiment_name

        if 'model' not in config:
            config['model'] = {}
        config['model']['name'] = args.model
        config['model']['architecture'] = args.model
        config['model']['pretrained'] = not args.no_pretrained

        if 'data' not in config:
            config['data'] = {}
        config['data']['dataset_name'] = args.dataset
        config['data']['dataset_mode'] = args.dataset_mode
        config['data']['multi_dataset'] = args.multi_dataset
        config['data']['exclude_datasets'] = args.exclude_dataset
        config['data']['subset_ratio'] = args.subset_ratio
    
    print("Training Configuration:")
    print(json.dumps(config, indent=2))
    
    # Create experiment configuration - handle both old and new config formats
    exp_config = ExperimentConfig(
        experiment_name=config['experiment']['name'],
        model_name=config['model'].get('name', config['model'].get('architecture', args.model)),
        dataset_name=config['data'].get('dataset_name', 'celebdf_v2'),
        description=config['experiment']['description'],
        tags=config['experiment']['tags'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training'].get('epochs', config['training'].get('num_epochs', args.epochs)),
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

            # Print per-dataset breakdown if available
            if 'per_dataset_metrics' in val_metrics:
                print(f"\n  Per-Dataset Validation Breakdown:")
                for dataset_name, metrics in val_metrics['per_dataset_metrics'].items():
                    print(f"    {dataset_name}:")
                    print(f"      AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, "
                          f"Acc: {metrics['accuracy']:.4f} ({metrics['num_samples']:,} samples)")

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

        # Print per-dataset test breakdown if available
        if 'per_dataset_metrics' in test_metrics:
            print(f"\n  Per-Dataset Test Breakdown:")
            for dataset_name, metrics in test_metrics['per_dataset_metrics'].items():
                print(f"    {dataset_name}:")
                print(f"      AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, "
                      f"Acc: {metrics['accuracy']:.4f} ({metrics['num_samples']:,} samples)")
        
        print(f"\\nExperiment {experiment_id} completed successfully!")
        return experiment_id

if __name__ == "__main__":
    experiment_id = main()
    print(f"\\nTraining finished. Experiment ID: {experiment_id}")