#!/usr/bin/env python3
"""
AWARE-NET Stage 0: Baseline Model Training Script
Complete training pipeline with experiment tracking
"""

import os
import sys
import json
import time
import datetime
import argparse
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler, Sampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("stage_00.train_baseline")


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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stage_00.baseline_model import EfficientNetV2B3Baseline, BaselineTrainer
from stage_00.dataset import create_data_loaders
from utils.dataset_config import DatasetConfig
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

        logger.info("Dataset sample distribution (total: %s)", f"{total_samples:,}")
        for dataset_name, count in dataset_counts.items():
            weight = weights.get(dataset_name, 0)
            logger.info(
                "  %s: %s samples (%.1f%%)",
                dataset_name,
                f"{count:,}",
                weight * 100
            )

    return weights


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


def resolve_dataset_manifests(
    config_path: Union[str, Path],
    dataset_name: str,
    dataset_mode: str
) -> tuple[Dict[str, Path], Path]:
    """Resolve manifest paths for a dataset using DatasetConfig."""

    ds_config = DatasetConfig(config_path, dataset_name=dataset_name)
    manifests: Dict[str, Path] = {}
    missing_splits: List[str] = []

    for split in ('train', 'val', 'test'):
        if split not in ds_config.splits:
            missing_splits.append(split)
            continue

        base_manifest = ds_config.get_manifest_path(split)
        variant_manifest = _apply_dataset_mode(base_manifest, dataset_mode)
        selected_manifest = variant_manifest

        if variant_manifest == base_manifest:
            selected_manifest = base_manifest
        elif not variant_manifest.exists():
            logger.debug(
                "Manifest variant %s not found for %s/%s; falling back to %s",
                variant_manifest,
                dataset_name,
                split,
                base_manifest
            )
            selected_manifest = base_manifest

        if not selected_manifest.exists():
            logger.warning(
                "Manifest missing for %s/%s: %s",
                dataset_name,
                split,
                selected_manifest
            )
            return {}, ds_config.root_path

        manifests[split] = selected_manifest

    if missing_splits:
        logger.warning(
            "Dataset %s missing configured splits: %s",
            dataset_name,
            missing_splits
        )
        return {}, ds_config.root_path

    return manifests, ds_config.root_path


def load_multi_dataset_names(config_path: Union[str, Path], exclude: List[str]) -> List[str]:
    """Load enabled dataset names for unified multi-dataset training."""

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)

    dataset_names = raw_config.get('multi_dataset_configs', {}).get(
        'unified_training',
        {}
    ).get('datasets_included', [])

    if not dataset_names:
        dataset_names = [
            name for name, info in raw_config.get('datasets', {}).items()
            if info.get('enabled', True)
        ]

    exclude_set = set(exclude)
    dataset_names = [name for name in dataset_names if name not in exclude_set]

    return dataset_names

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
        self.dataset_offsets = []

        offset = 0

        for dataset_id, dataset in enumerate(self.datasets):
            dataset_name = getattr(dataset, 'dataset_name', f'Dataset_{dataset_id}')
            # Store dataset name for reporting
            if dataset_name not in self.dataset_names:
                self.dataset_names.append(dataset_name)

            self.dataset_offsets.append(offset)

            # Map each sample index to its dataset_id
            for _ in range(len(dataset)):
                self.dataset_mapping.append(dataset_id)
            offset += len(dataset)

        self.total_length = offset
        self._build_label_indices()

    def _build_label_indices(self):
        """Pre-compute indices for each dataset/label for stratified sampling."""
        self.label_indices: Dict[int, Dict[int, List[int]]] = {}

        logger.info("🔧 Building label indices for %d datasets...", len(self.datasets))

        for dataset_id, dataset in enumerate(self.datasets):
            offset = self.dataset_offsets[dataset_id]
            per_label: Dict[int, List[int]] = {}
            dataset_name = getattr(dataset, 'dataset_name', f'Dataset_{dataset_id}')

            logger.info("  📊 Processing %s (%s samples)...", dataset_name, f"{len(dataset):,}")

            if hasattr(dataset, 'get_indices_by_label'):
                # Fast path: use dataset's built-in method
                local_map = dataset.get_indices_by_label()
                for label, local_indices in local_map.items():
                    per_label[int(label)] = [offset + idx for idx in local_indices]
                logger.info("    ✅ Fast indexing completed for %s", dataset_name)
            else:
                # Slow path: iterate and gather labels (with progress bar)
                logger.warning("    ⚠️  Slow indexing path for %s - consider implementing get_indices_by_label()", dataset_name)

                per_label = {}
                from tqdm import tqdm

                for local_idx in tqdm(range(len(dataset)),
                                    desc=f"    Indexing {dataset_name}",
                                    unit="samples"):
                    try:
                        _, label = dataset[local_idx]
                        label_value = int(label.item() if hasattr(label, 'item') else label)
                        per_label.setdefault(label_value, []).append(offset + local_idx)
                    except Exception as e:
                        logger.error("Failed to process sample %d in %s: %s", local_idx, dataset_name, e)
                        continue

            self.label_indices[dataset_id] = per_label

            # Log summary for this dataset
            total_indices = sum(len(indices) for indices in per_label.values())
            logger.info("    📈 %s: %d labels, %s total indices",
                       dataset_name, len(per_label), f"{total_indices:,}")

        logger.info("✅ Label indices building completed")

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

        logger.info("Calculating class counts for %d datasets...", len(self.datasets))

        for idx, dataset in enumerate(self.datasets):
            dataset_name = getattr(dataset, 'dataset_name', f'Dataset {idx+1}')

            if hasattr(dataset, 'get_class_counts'):
                counts = dataset.get_class_counts()
                total_real += counts[0].item()
                total_fake += counts[1].item()
                logger.info("  ✓ %s: %.0f real, %.0f fake", dataset_name, counts[0], counts[1])
            else:
                # Fallback: count labels in dataset with progress bar
                logger.info("  ⏳ Counting labels for %s (%s samples)...", dataset_name, f"{len(dataset):,}")
                labels = []

                # Use tqdm for progress bar
                from tqdm import tqdm
                for i in tqdm(range(len(dataset)), desc=f"    Scanning {dataset_name}", unit="samples"):
                    labels.append(dataset[i][1].item())

                real_count = labels.count(0)
                fake_count = labels.count(1)
                total_real += real_count
                total_fake += fake_count
                logger.info("  ✓ %s: %s real, %s fake", dataset_name, f"{real_count:,}", f"{fake_count:,}")

        ratio = total_real / max(total_fake, 1)
        logger.info("Total class counts → real=%s, fake=%s, ratio=%.4f",
                    f"{total_real:,}", f"{total_fake:,}", ratio)

        return torch.tensor([total_real, total_fake], dtype=torch.float)


class StrictBalancedBatchSampler(Sampler[List[int]]):
    """Sampler enforcing equal dataset and class contributions per batch."""

    def __init__(
        self,
        dataset: MultiDatasetWrapper,
        batch_size: int,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset_wrapper = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)

        self.dataset_ids = list(range(len(self.dataset_wrapper.datasets)))
        if not self.dataset_ids:
            raise ValueError("StrictBalancedBatchSampler requires at least one dataset")

        if batch_size % len(self.dataset_ids) != 0:
            raise ValueError(
                "Batch size must be divisible by the number of datasets for strict balancing"
            )

        # Collect available labels across all datasets
        label_set = set()
        for per_label in self.dataset_wrapper.label_indices.values():
            label_set.update(per_label.keys())
        if not label_set:
            raise ValueError("No class labels found across datasets")

        self.labels = sorted(label_set)
        self.num_labels = len(self.labels)

        if batch_size % (len(self.dataset_ids) * self.num_labels) != 0:
            raise ValueError(
                "Batch size must be divisible by (datasets × classes) to enforce strict balance"
            )

        self.samples_per_dataset = batch_size // len(self.dataset_ids)
        self.samples_per_label = self.samples_per_dataset // self.num_labels

        if self.samples_per_label == 0:
            raise ValueError("Batch size too small to allocate samples per class")

        # Prepare pools for deterministic yet reshufflable sampling
        self.original_pools: Dict[int, Dict[int, List[int]]] = {}
        self.state: Dict[int, Dict[int, Dict[str, Any]]] = {}

        for ds_id in self.dataset_ids:
            per_label = self.dataset_wrapper.label_indices.get(ds_id, {})
            # Ensure every dataset has entries for each label
            missing = [label for label in self.labels if not per_label.get(label)]
            if missing:
                dataset_name = self.dataset_wrapper.get_dataset_name(ds_id)
                missing_str = ", ".join(str(m) for m in missing)
                raise ValueError(
                    f"Dataset '{dataset_name}' lacks samples for label(s): {missing_str}. "
                    "Provide balanced manifests or disable strict batch balancing."
                )

            self.original_pools[ds_id] = {
                label: per_label[label][:] for label in self.labels
            }
            self.state[ds_id] = {}
            for label in self.labels:
                pool_copy = self.original_pools[ds_id][label][:]
                self.rng.shuffle(pool_copy)
                self.state[ds_id][label] = {'pool': pool_copy, 'idx': 0}

        self.num_batches = max(1, len(self.dataset_wrapper) // batch_size)

    def _draw_indices(self, ds_id: int, label: int, count: int) -> List[int]:
        state = self.state[ds_id][label]
        pool = state['pool']
        idx = state['idx']

        if idx + count > len(pool):
            pool = self.original_pools[ds_id][label][:]
            self.rng.shuffle(pool)
            state['pool'] = pool
            idx = 0

        selected = pool[idx:idx + count]
        if len(selected) < count:
            # Fallback to sampling with replacement if pool smaller than quota
            needed = count - len(selected)
            selected.extend(self.rng.choices(self.original_pools[ds_id][label], k=needed))
            idx = 0
        else:
            idx += count

        state['idx'] = idx
        return selected

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices: List[int] = []
            for ds_id in self.dataset_ids:
                for label in self.labels:
                    batch_indices.extend(
                        self._draw_indices(ds_id, label, self.samples_per_label)
                    )
            self.rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches

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
                    quality_range=(70, 100),
                    p=0.3  # Simulate video compression artifacts
                ),

                # Normalization and conversion to tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            logger.info("SupCon augmentation enabled for %s", dataset_name)
        else:
            # No augmentation mode - just normalize
            self.albu_transform = A.Compose([
                A.Resize(height=256, width=256),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

        # Load manifest
        self.data = pd.read_csv(manifest_path)
        logger.info("Loaded %d samples from %s", len(self.data), dataset_name)

        # Filter valid samples
        if 'valid' in self.data.columns:
            valid_mask = self.data['valid'] == True
            self.data = self.data[valid_mask]
            logger.info("After filtering: %d valid samples from %s", len(self.data), dataset_name)

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
            logger.info("Subset to %.1f%%: %d samples from %s", subset_ratio * 100, len(self.data), dataset_name)

        # Print label distribution
        label_counts = self.data['label'].value_counts().sort_index()
        logger.info("%s label distribution: %s", dataset_name, dict(label_counts))

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

    def get_indices_by_label(self) -> Dict[int, List[int]]:
        """Return local indices grouped by label."""
        grouped = self.data.groupby('label').groups
        return {int(label): list(indexes) for label, indexes in grouped.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = sample['image_path']
        label = int(sample['label'])

        # Load image with enhanced error handling
        try:
            # Check if file exists before attempting to load
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)  # Convert to numpy for albumentations

            # Validate image dimensions
            if image_np.size == 0:
                raise ValueError(f"Empty image: {image_path}")

            # Apply albumentations transforms
            transformed = self.albu_transform(image=image_np)
            image_tensor = transformed['image']

            return image_tensor, torch.tensor(label, dtype=torch.float)

        except FileNotFoundError as e:
            logger.error("File not found: %s", str(e))
            # Return a dummy black image as fallback
            dummy_image = torch.zeros(3, 256, 256)
            return dummy_image, torch.tensor(label, dtype=torch.float)
        except Exception as e:
            logger.error("Failed to load %s: %s", image_path, e)
            # Return a dummy black image as fallback instead of raising
            dummy_image = torch.zeros(3, 256, 256)
            return dummy_image, torch.tensor(label, dtype=torch.float)

def create_multi_dataset_loaders(
    config_path,
    dataset_mode='original',
    batch_size=32,
    num_workers=0,
    subset_ratio=1.0,
    use_augmentation=False,
    use_dataset_weights=True,
    exclude=None,
    strict_batch_balance=False,
    seed=42,
):
    """Create data loaders for multi-dataset training using DatasetConfig manifests."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")

    exclude = exclude or []
    dataset_names = load_multi_dataset_names(config_path, exclude)
    if not dataset_names:
        raise RuntimeError(
            "No datasets available for multi-dataset training after applying exclusions"
        )

    logger.info(
        "Multi-dataset mode → config=%s | mode=%s | datasets=%s",
        config_path,
        dataset_mode,
        dataset_names,
    )

    train_datasets = []
    val_datasets = []
    test_datasets = []
    dataset_sizes = []

    # Enhanced logging for debugging data loading instability
    logger.info("="*60)
    logger.info("MULTI-DATASET LOADING ANALYSIS")
    logger.info("="*60)

    total_real_samples = 0
    total_fake_samples = 0
    dataset_details = []

    for dataset_name in dataset_names:
        manifests, _ = resolve_dataset_manifests(config_path, dataset_name, dataset_mode)
        if not manifests:
            logger.warning(
                "Skipping %s: manifests unavailable for mode '%s'",
                dataset_name,
                dataset_mode,
            )
            continue

        required_splits = {'train', 'val', 'test'}
        missing = [split for split in required_splits if split not in manifests]
        if missing:
            logger.warning(
                "Skipping %s: missing manifest splits %s",
                dataset_name,
                missing,
            )
            continue

        logger.info(
            "Loading %s manifests → train=%s | val=%s | test=%s",
            dataset_name,
            manifests['train'],
            manifests['val'],
            manifests['test'],
        )

        train_ds = UnifiedDeepfakeDataset(
            manifest_path=manifests['train'],
            dataset_name=f"{dataset_name}_train",
            transform=None,
            subset_ratio=subset_ratio,
            use_augmentation=use_augmentation,
        )

        # Get dataset statistics without loading individual samples
        try:
            # Use the dataset's internal statistics if available
            if hasattr(train_ds, 'get_class_counts'):
                class_counts = train_ds.get_class_counts()
                # Handle both dict and tensor formats
                if isinstance(class_counts, dict):
                    real_count = class_counts.get(0, 0)
                    fake_count = class_counts.get(1, 0)
                else:
                    # Handle tensor format - convert to list if needed
                    if hasattr(class_counts, 'cpu'):
                        class_counts = class_counts.cpu().numpy()
                    if len(class_counts) >= 2:
                        real_count = int(class_counts[0])
                        fake_count = int(class_counts[1])
                    else:
                        real_count = fake_count = 0
            else:
                # Fast estimation: read from manifest directly
                import pandas as pd
                df = pd.read_csv(manifests['train'], usecols=['label'])
                real_count = (df['label'] == 0).sum()
                fake_count = (df['label'] == 1).sum()

            total_real_samples += real_count
            total_fake_samples += fake_count

            dataset_details.append({
                'name': dataset_name,
                'total_samples': len(train_ds),
                'real_samples': real_count,
                'fake_samples': fake_count,
                'manifest_path': str(manifests['train'])
            })

            logger.info(f"  {dataset_name}: {len(train_ds)} samples (Real: {real_count}, Fake: {fake_count})")
        except Exception as e:
            # Fallback: just show total samples without class breakdown
            logger.warning(f"Could not get detailed statistics for {dataset_name}: {e}")
            dataset_details.append({
                'name': dataset_name,
                'total_samples': len(train_ds),
                'real_samples': -1,
                'fake_samples': -1,
                'manifest_path': str(manifests['train'])
            })
            logger.info(f"  {dataset_name}: {len(train_ds)} samples (class stats unavailable)")

        val_ds = UnifiedDeepfakeDataset(
            manifest_path=manifests['val'],
            dataset_name=f"{dataset_name}_val",
            transform=None,
            subset_ratio=subset_ratio,
            use_augmentation=False,
        )
        test_ds = UnifiedDeepfakeDataset(
            manifest_path=manifests['test'],
            dataset_name=f"{dataset_name}_test",
            transform=None,
            subset_ratio=subset_ratio,
            use_augmentation=False,
        )

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)
        dataset_sizes.append(len(train_ds))

    # Log comprehensive analysis
    logger.info("="*60)
    logger.info("DATASET LOADING SUMMARY")
    logger.info("="*60)
    for detail in dataset_details:
        logger.info(f"{detail['name']:20} | Total: {detail['total_samples']:6,} | Real: {detail['real_samples']:6,} | Fake: {detail['fake_samples']:6,}")

    logger.info("-"*60)
    logger.info(f"{'TOTAL':20} | Total: {total_real_samples + total_fake_samples:6,} | Real: {total_real_samples:6,} | Fake: {total_fake_samples:6,}")
    logger.info("="*60)

    if not train_datasets:
        raise RuntimeError(
            "No datasets could be loaded for multi-dataset training. "
            "Check manifest availability or dataset exclusions."
        )

    combined_train = MultiDatasetWrapper(train_datasets)
    combined_val = MultiDatasetWrapper(val_datasets)
    combined_test = MultiDatasetWrapper(test_datasets)

    logger.info(
        "Combined dataset sizes → train=%s, val=%s, test=%s",
        f"{len(combined_train):,}",
        f"{len(combined_val):,}",
        f"{len(combined_test):,}",
    )

    sampler = None
    batch_sampler = None
    if strict_batch_balance:
        batch_sampler = StrictBalancedBatchSampler(
            combined_train,
            batch_size=batch_size,
            seed=seed,
        )
        logger.info("Strict batch balancing enabled (equal dataset/class contributions per batch)")
    elif use_dataset_weights and dataset_sizes:
        total_samples = sum(dataset_sizes)
        logger.info(
            "Balancing dataset contributions across %d datasets",
            len(dataset_sizes),
        )
        sample_weights = []
        for ds, size in zip(train_datasets, dataset_sizes):
            weight_per_sample = 1.0 / max(size, 1)
            sample_weights.extend([weight_per_sample] * size)
            original_ratio = (size / total_samples) * 100 if total_samples else 0
            balanced_ratio = 100 / len(dataset_sizes)
            logger.info(
                "  %s: original %.1f%% → balanced %.1f%% (%s samples)",
                getattr(ds, 'dataset_name', 'dataset'),
                original_ratio,
                balanced_ratio,
                f"{size:,}",
            )

        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
        sample_weights_tensor = (
            sample_weights_tensor / sample_weights_tensor.sum() * len(sample_weights_tensor)
        )

        sampler = WeightedRandomSampler(
            sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=False,
        )
        logger.info("WeightedRandomSampler configured for balanced dataset sampling (replacement=False)")
        logger.info("Sampler will use %d samples total (same as combined dataset)", len(sample_weights_tensor))

    pin_memory = torch.cuda.is_available()
    shuffle_train = sampler is None and batch_sampler is None

    # Enhanced logging for DataLoader configuration
    if batch_sampler is not None:
        logger.info("Using StrictBalancedBatchSampler for deterministic batch construction")
    elif sampler is not None:
        logger.info("Using WeightedRandomSampler for balanced dataset sampling (no shuffle)")
    else:
        logger.info("Using standard DataLoader with shuffle=True")

    if batch_sampler is not None:
        train_loader = DataLoader(
            combined_train,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            combined_train,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        combined_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def setup_device_with_fallback():
    """
    Intelligent GPU detection with fallback options
    
    Returns:
        Tuple of (device, gpu_info_dict or None)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
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
            logger.info("[OK] GPU %d: %s - Compatible", i, gpu_name)
        except Exception as e:
            logger.warning("[FAIL] GPU %d: %s - Incompatible (%s)", i, gpu_name, str(e)[:50])
    
    if compatible_gpus:
        # Select the best compatible GPU (highest memory)
        best_gpu = max(compatible_gpus, key=lambda x: x['memory_gb'])
        torch.cuda.set_device(best_gpu['id'])
        device = torch.device(f"cuda:{best_gpu['id']}")
        logger.info("[SELECTED] GPU %d: %s", best_gpu['id'], best_gpu['name'])
        return device, best_gpu

    # No compatible GPU found, provide upgrade suggestions
    logger.warning("No compatible GPU found. Falling back to CPU.")

    for gpu in all_gpus:
        if gpu['arch'] in ['sm_120']:  # RTX 5060 Ti, 5090, etc.
            logger.info(
                "%s requires PyTorch nightly build. Suggested command: conda install pytorch-cuda=12.6 pytorch pytorch2-nightly -c pytorch-nightly -c nvidia",
                gpu['name']
            )
        elif gpu['capability_int'] < 50:  # Very old GPUs
            logger.info("%s is too old (compute capability %s)", gpu['name'], gpu['capability'])

    logger.info("Falling back to CPU training. For GPU acceleration consider: 1) compatible GPU (RTX 3070Ti, 4080, etc.) 2) installing PyTorch nightly for RTX 50-series support")
    
    return torch.device('cpu'), None

class TrainingLogger:
    """
    Enhanced training logger with CSV, TXT logging and visualization support

    Automatically generates:
    - CSV logs for easy analysis
    - TXT logs for human reading
    - Training curve plots
    - Learning rate schedules
    - Per-dataset breakdowns
    """

    def __init__(self, experiment_manager, visualizer=None, experiment_id=None):
        """
        Initialize training logger

        Args:
            experiment_manager: ExperimentManager instance
            visualizer: AcademicVisualizer instance (optional)
            experiment_id: Experiment ID (required if called outside context manager)
        """
        self.exp_manager = experiment_manager
        self.visualizer = visualizer or AcademicVisualizer()

        # Get experiment directory - use provided experiment_id or current_experiment
        if experiment_id is not None:
            self.exp_dir = Path(self.exp_manager.base_path) / experiment_id
        else:
            self.exp_dir = Path(self.exp_manager.base_path) / self.exp_manager.current_experiment
        self.logs_dir = self.exp_dir / "logs"
        self.plots_dir = self.exp_dir / "plots"

        # Create log files
        self.csv_log_path = self.logs_dir / "training_log.csv"
        self.txt_log_path = self.logs_dir / "training_log.txt"
        self.lr_log_path = self.logs_dir / "lr_schedule.csv"
        self.dataset_log_path = self.logs_dir / "per_dataset_metrics.csv"

        # In-memory storage
        self.epoch_logs = []
        self.lr_logs = []
        self.dataset_logs = []

        # Initialize TXT log file
        with open(self.txt_log_path, 'w') as f:
            f.write(f"Training Log - {datetime.datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def log_epoch(self, epoch, train_metrics, val_metrics, lr, per_dataset_metrics=None):
        """
        Log metrics for one epoch

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            lr: Current learning rate
            per_dataset_metrics: Optional per-dataset breakdown
        """
        # Prepare epoch log entry
        log_entry = {
            'epoch': epoch,
            'lr': lr,
            'train_loss': train_metrics.get('loss', 0),
            'train_accuracy': train_metrics.get('accuracy', 0),
            'val_loss': val_metrics.get('loss', 0),
            'val_accuracy': val_metrics.get('accuracy', 0),
            'val_auc': val_metrics.get('auc', 0),
            'val_f1': val_metrics.get('f1', 0),
            'val_precision': val_metrics.get('precision', 0),
            'val_recall': val_metrics.get('recall', 0),
            'timestamp': datetime.datetime.now().isoformat()
        }

        self.epoch_logs.append(log_entry)

        # Log learning rate
        self.lr_logs.append({'epoch': epoch, 'lr': lr, 'timestamp': datetime.datetime.now().isoformat()})

        # Log per-dataset metrics if available
        if per_dataset_metrics:
            for dataset_name, metrics in per_dataset_metrics.items():
                dataset_entry = {
                    'epoch': epoch,
                    'dataset': dataset_name,
                    'auc': metrics.get('auc', 0),
                    'f1': metrics.get('f1', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'num_samples': metrics.get('num_samples', 0)
                }
                self.dataset_logs.append(dataset_entry)

        # Append to TXT log
        self._write_txt_log(epoch, train_metrics, val_metrics, lr, per_dataset_metrics)

        # Save CSV incrementally (in case of interruption)
        self._save_csv_incremental()

    def _write_txt_log(self, epoch, train_metrics, val_metrics, lr, per_dataset_metrics):
        """Write formatted text log entry"""
        with open(self.txt_log_path, 'a') as f:
            f.write(f"Epoch {epoch:03d} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Learning Rate: {lr:.6e}\n")
            f.write(f"  Train - Loss: {train_metrics.get('loss', 0):.4f}, Acc: {train_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  Val   - Loss: {val_metrics.get('loss', 0):.4f}, Acc: {val_metrics.get('accuracy', 0):.4f}, "
                   f"AUC: {val_metrics.get('auc', 0):.4f}, F1: {val_metrics.get('f1', 0):.4f}\n")

            if per_dataset_metrics:
                f.write(f"  Per-Dataset Breakdown:\n")
                for dataset_name, metrics in per_dataset_metrics.items():
                    f.write(f"    {dataset_name}: AUC={metrics.get('auc', 0):.4f}, "
                           f"F1={metrics.get('f1', 0):.4f}, Acc={metrics.get('accuracy', 0):.4f} "
                           f"({metrics.get('num_samples', 0):,} samples)\n")

            f.write("-" * 80 + "\n\n")

    def _save_csv_incremental(self):
        """Save logs to CSV files incrementally"""
        # Save epoch logs
        if self.epoch_logs:
            df_epochs = pd.DataFrame(self.epoch_logs)
            df_epochs.to_csv(self.csv_log_path, index=False)

        # Save LR schedule
        if self.lr_logs:
            df_lr = pd.DataFrame(self.lr_logs)
            df_lr.to_csv(self.lr_log_path, index=False)

        # Save per-dataset metrics
        if self.dataset_logs:
            df_datasets = pd.DataFrame(self.dataset_logs)
            df_datasets.to_csv(self.dataset_log_path, index=False)

    def generate_training_curves(self, training_history):
        """
        Generate training curves plot

        Args:
            training_history: Dict with 'train_loss', 'val_loss', etc.
        """
        try:
            fig = self.visualizer.plot_performance_over_time(training_history)
            plot_path = self.plots_dir / "training_curves"

            # Save as both PNG and PDF
            fig.savefig(f"{plot_path}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{plot_path}.pdf", bbox_inches='tight')
            plt.close(fig)

            logger.info("  ✓ Saved training curves: %s.png", plot_path)
        except Exception as e:
            logger.warning("  ⚠ Failed to generate training curves: %s", e)

    def generate_lr_schedule_plot(self):
        """Generate learning rate schedule plot"""
        if not self.lr_logs:
            return

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            epochs = [log['epoch'] for log in self.lr_logs]
            lrs = [log['lr'] for log in self.lr_logs]

            ax.plot(epochs, lrs, 'o-', linewidth=2, markersize=4, color='#2E86AB')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

            plot_path = self.plots_dir / "learning_rate"
            fig.savefig(f"{plot_path}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{plot_path}.pdf", bbox_inches='tight')
            plt.close(fig)

            logger.info("  ✓ Saved learning rate plot: %s.png", plot_path)
        except Exception as e:
            logger.warning("  ⚠ Failed to generate LR plot: %s", e)

    def generate_per_dataset_comparison(self):
        """Generate per-dataset performance comparison plot"""
        if not self.dataset_logs:
            return

        try:
            df = pd.DataFrame(self.dataset_logs)

            # Get unique datasets
            datasets = df['dataset'].unique()

            if len(datasets) < 2:
                return  # No point plotting single dataset

            # Create subplots for each metric
            metrics = ['auc', 'f1', 'accuracy']
            fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

            if len(metrics) == 1:
                axes = [axes]

            colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

            for idx, metric in enumerate(metrics):
                ax = axes[idx]

                for i, dataset in enumerate(datasets):
                    dataset_data = df[df['dataset'] == dataset]
                    ax.plot(dataset_data['epoch'], dataset_data[metric],
                           'o-', label=dataset, color=colors[i],
                           linewidth=2, markersize=4)

                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} by Dataset')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_path = self.plots_dir / "per_dataset_comparison"
            fig.savefig(f"{plot_path}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{plot_path}.pdf", bbox_inches='tight')
            plt.close(fig)

            logger.info("  ✓ Saved per-dataset comparison: %s.png", plot_path)
        except Exception as e:
            logger.warning("  ⚠ Failed to generate per-dataset plot: %s", e)

    def generate_final_report(self, training_history, test_metrics=None):
        """
        Generate complete visualization report at end of training

        Args:
            training_history: Complete training history dict
            test_metrics: Optional test set metrics
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("Generating Training Report and Visualizations")
        logger.info("=" * 80)

        # 1. Training curves
        self.generate_training_curves(training_history)

        # 2. Learning rate schedule
        self.generate_lr_schedule_plot()

        # 3. Per-dataset comparison
        self.generate_per_dataset_comparison()

        # 4. Save final CSVs
        self._save_csv_incremental()

        # 5. Generate summary markdown
        self._generate_summary_markdown(training_history, test_metrics)

        logger.info("")
        logger.info("Training report generated successfully!")
        logger.info("  Logs: %s", self.logs_dir)
        logger.info("  Plots: %s", self.plots_dir)
        logger.info("%s", "=" * 80)

    def _generate_summary_markdown(self, training_history, test_metrics):
        """Generate a markdown summary of training"""
        summary_path = self.logs_dir / "training_summary.md"

        with open(summary_path, 'w') as f:
            f.write("# Training Summary\n\n")
            f.write(f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Training info
            f.write("## Training Progress\n\n")
            f.write(f"- Total Epochs: {len(self.epoch_logs)}\n")

            if self.epoch_logs:
                last_epoch = self.epoch_logs[-1]
                f.write(f"- Final Train Loss: {last_epoch['train_loss']:.4f}\n")
                f.write(f"- Final Train Accuracy: {last_epoch['train_accuracy']:.4f}\n")
                f.write(f"- Final Val Loss: {last_epoch['val_loss']:.4f}\n")
                f.write(f"- Final Val AUC: {last_epoch['val_auc']:.4f}\n")
                f.write(f"- Final Val F1: {last_epoch['val_f1']:.4f}\n\n")

                # Best validation performance
                best_auc_epoch = max(self.epoch_logs, key=lambda x: x['val_auc'])
                f.write(f"- Best Val AUC: {best_auc_epoch['val_auc']:.4f} (Epoch {best_auc_epoch['epoch']})\n\n")

            # Test metrics
            if test_metrics:
                f.write("## Test Set Performance\n\n")
                f.write(f"- Test Loss: {test_metrics.get('loss', 0):.4f}\n")
                f.write(f"- Test Accuracy: {test_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"- Test AUC: {test_metrics.get('auc', 0):.4f}\n")
                f.write(f"- Test F1: {test_metrics.get('f1', 0):.4f}\n\n")

            # Artifacts
            f.write("## Generated Artifacts\n\n")
            f.write("### Logs\n")
            f.write("- `training_log.csv` - Detailed epoch-by-epoch metrics\n")
            f.write("- `training_log.txt` - Human-readable training log\n")
            f.write("- `lr_schedule.csv` - Learning rate schedule\n")
            if self.dataset_logs:
                f.write("- `per_dataset_metrics.csv` - Per-dataset performance breakdown\n")
            f.write("\n### Plots\n")
            f.write("- `training_curves.png/pdf` - Training and validation curves\n")
            f.write("- `learning_rate.png/pdf` - Learning rate schedule\n")
            if self.dataset_logs:
                f.write("- `per_dataset_comparison.png/pdf` - Performance by dataset\n")

        logger.info("  ✓ Saved training summary: %s", summary_path)

def setup_training(config: Dict[str, Any]) -> tuple:
    """Setup training components with manifest-aware configuration."""
    device, gpu_info = setup_device_with_fallback()
    logger.info("Selected device: %s", device)

    if gpu_info:
        logger.info(
            "GPU: %s (compute %s, %.1f GB)",
            gpu_info["name"],
            gpu_info["capability"],
            gpu_info["memory_gb"],
        )
        if gpu_info["memory_gb"] < 8:
            suggested_batch = min(config["training"]["batch_size"], 16)
            if suggested_batch < config["training"]["batch_size"]:
                logger.warning(
                    "Reducing batch size to %d due to limited GPU memory (%.1f GB)",
                    suggested_batch,
                    gpu_info["memory_gb"],
                )
                config["training"]["batch_size"] = suggested_batch
    else:
        logger.info("GPU not detected; continuing with CPU pipeline")

    data_section = config.get("data", {})
    dataset_config_path = Path(data_section.get("dataset_config", "configs/datasets.json"))
    if not dataset_config_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found: {dataset_config_path}"
        )

    dataset_mode = data_section.get("dataset_mode", "original")
    subset_ratio = float(data_section.get("subset_ratio", 1.0))
    multi_dataset = data_section.get("multi_dataset", False)
    exclude_datasets = data_section.get("exclude_datasets", []) or []
    use_dataset_weights = data_section.get("use_dataset_weights", True)
    use_augmentation = data_section.get("augmentation", False)
    strict_batch_balance = data_section.get("strict_batch_balance", False)
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    pin_memory = torch.cuda.is_available()
    sampler_seed = config.get("training", {}).get("seed", 42)

    logger.info(
        "Data configuration → dataset_config=%s | mode=%s | multi_dataset=%s | subset_ratio=%.2f",
        dataset_config_path,
        dataset_mode,
        multi_dataset,
        subset_ratio,
    )
    if exclude_datasets:
        logger.info("Excluding datasets: %s", exclude_datasets)
    if strict_batch_balance:
        logger.info("Strict batch balancing enabled for training batches")

    if multi_dataset:
        train_loader, val_loader, test_loader = create_multi_dataset_loaders(
            config_path=dataset_config_path,
            dataset_mode=dataset_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_ratio=subset_ratio,
            use_augmentation=use_augmentation,
            use_dataset_weights=use_dataset_weights,
            exclude=exclude_datasets,
            strict_batch_balance=strict_batch_balance,
            seed=sampler_seed,
        )
    else:
        if strict_batch_balance:
            logger.warning(
                "Strict batch balancing requested but multi-dataset mode is disabled; "
                "fallback to standard shuffling."
            )
        dataset_name = data_section.get("dataset_name", "celebdf_v2")
        manifests, dataset_root = resolve_dataset_manifests(
            dataset_config_path,
            dataset_name,
            dataset_mode,
        )
        if not manifests:
            raise FileNotFoundError(
                f"Manifests for dataset '{dataset_name}' (mode={dataset_mode}) were not found. "
                "Run the manifest generation utilities or adjust dataset configuration."
            )

        logger.info(
            "Using dataset %s with manifests → train=%s | val=%s | test=%s",
            dataset_name,
            manifests["train"],
            manifests["val"],
            manifests["test"],
        )

        from stage_00.dataset import CelebDFDataset

        train_dataset = CelebDFDataset(
            manifest_path=manifests["train"],
            root_path=str(dataset_root),
            image_size=data_section.get("image_size", 256),
            augmentation=use_augmentation,
            normalize=True,
            path_leakage_check=data_section.get("path_leakage_check", True),
        )
        val_dataset = CelebDFDataset(
            manifest_path=manifests["val"],
            root_path=str(dataset_root),
            image_size=data_section.get("image_size", 256),
            augmentation=False,
            normalize=True,
            path_leakage_check=data_section.get("path_leakage_check", True),
        )
        test_dataset = CelebDFDataset(
            manifest_path=manifests["test"],
            root_path=str(dataset_root),
            image_size=data_section.get("image_size", 256),
            augmentation=False,
            normalize=True,
            path_leakage_check=data_section.get("path_leakage_check", True),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
            info = dataset.get_dataset_info()
            logger.info(
                "%s dataset → total=%d real=%d fake=%d balance=%.3f",
                split_name,
                info["total_samples"],
                info["real_samples"],
                info["fake_samples"],
                info["balance_ratio"],
            )

    logger.info("Creating baseline model (%s)", config["model"]["name"])
    model = EfficientNetV2B3Baseline(
        num_classes=1,
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout_rate"],
        model_name=config["model"]["name"],
    ).to(device)
    logger.info("Model info: %s", model.get_model_info())

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=1e-6,
    )

    if config["training"].get("use_class_weights", True):
        if "pos_weight" in config["training"]:
            pos_weight = torch.tensor([config["training"]["pos_weight"]], device=device)
            logger.info("Using pre-configured pos_weight: %.4f", pos_weight.item())
        else:
            try:
                class_counts = train_loader.dataset.get_class_counts()
                real_count = float(class_counts[0].item())
                fake_count = float(class_counts[1].item())
                total = real_count + fake_count
                if fake_count == 0:
                    pos_weight_value = 1.0
                    logger.warning("Detected zero fake samples; defaulting pos_weight to 1.0")
                else:
                    pos_weight_value = real_count / fake_count
                pos_weight = torch.tensor([pos_weight_value], device=device)
                real_pct = (real_count / total * 100) if total else 0.0
                fake_pct = (fake_count / total * 100) if total else 0.0
                logger.info(
                    "Dataset distribution → real=%d (%.1f%%) | fake=%d (%.1f%%)",
                    int(real_count),
                    real_pct,
                    int(fake_count),
                    fake_pct,
                )
                logger.info("Calculated pos_weight for BCE: %.4f", pos_weight.item())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Unable to derive class weights from dataset (%s); falling back to default pos_weight=2.33",
                    exc,
                )
                pos_weight = torch.tensor([2.33], device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Class weighting disabled; using standard BCEWithLogitsLoss")

    logger.debug("Effective training configuration:\n%s", json.dumps(config, indent=2))

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
                    logger.warning(
                        "Could not calculate metrics for dataset %s: %s",
                        dataset_id,
                        e,
                    )

            result['per_dataset_metrics'] = per_dataset_metrics

    return result

def run_evaluation(args):
    """
    Evaluation-only mode: Load checkpoint and test on specified dataset.
    Used for LODO generalization testing.
    """
    banner = "=" * 80
    logger.info("")
    logger.info(banner)
    logger.info("LODO Generalization Evaluation Mode")
    logger.info(banner)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --eval-only mode")
    if not args.test_dataset:
        raise ValueError("--test-dataset is required for --eval-only mode")

    device, gpu_info = setup_device_with_fallback()
    logger.info("Using device: %s", device)
    if gpu_info:
        logger.info("GPU: %s", gpu_info.get('name', 'Unknown'))

    logger.info("Loading checkpoint: %s", args.checkpoint)
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
    logger.info("Model checkpoint loaded successfully")

    dataset_short_names = {
        'celebdf_v2': 'celebdf_v2',
        'faceforensics_plus_plus': 'faceforensics',
        'deeperforensics_1_0': 'deeperforensics'
    }

    short_name = dataset_short_names.get(args.test_dataset, args.test_dataset)
    dataset_mode = args.dataset_mode

    if dataset_mode == 'balanced':
        manifest_path = Path(f'manifests/{short_name}_test_balanced.csv')
    else:
        manifest_path = Path(f'manifests/{short_name}_test.csv')

    logger.info(
        "Loading test dataset %s (manifest=%s)",
        args.test_dataset,
        manifest_path,
    )

    test_dataset = UnifiedDeepfakeDataset(
        manifest_path=manifest_path,
        dataset_name=f'{args.test_dataset}_test',
        transform=None,
        use_augmentation=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == 'cuda'
    )

    logger.info("Test dataset loaded: %s samples", f"{len(test_dataset):,}")
    logger.info("Running evaluation...")
    criterion = nn.BCEWithLogitsLoss()
    results = validate_epoch(model, test_loader, criterion, device, epoch=-1)

    logger.info("")
    logger.info(banner)
    logger.info("Out-of-Distribution Test Results")
    logger.info(banner)
    logger.info("Dataset: %s", args.test_dataset)
    logger.info("Samples: %s", f"{len(test_dataset):,}")
    logger.info("  AUC-ROC:  %.4f", results['auc'])
    logger.info("  F1-Score: %.4f", results['f1'])
    logger.info("  Accuracy: %.4f", results['accuracy'])
    logger.info("  Loss:     %.4f", results['loss'])

    if 'per_dataset_metrics' in results and results['per_dataset_metrics']:
        logger.info("Per-Dataset Breakdown:")
        for dataset_name, metrics in results['per_dataset_metrics'].items():
            logger.info(
                "  %s → AUC: %.4f, F1: %.4f, Acc: %.4f (%s samples)",
                dataset_name,
                metrics['auc'],
                metrics['f1'],
                metrics['accuracy'],
                f"{metrics['num_samples']:,}",
            )

    logger.info(banner)
    logger.info("Evaluation completed successfully!")

    return results



def main():
    parser = argparse.ArgumentParser(description='AWARE-NET Baseline Training')
    parser.add_argument('--config', type=str, default='configs/training.json',
                       help='Path to training configuration file')
    parser.add_argument('--dataset-config', type=str, default=None,
                       help='Override dataset configuration manifest path (defaults to config file)')
    parser.add_argument('--experiment-name', type=str, default='baseline_efficientnet',
                       help='Name of the experiment')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size (overrides config if specified)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config if specified)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config if specified)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--subset-ratio', type=float, default=None,
                       help='Fraction of dataset to use (0.1 = 10%%, 1.0 = 100%%, overrides config if specified)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Disable pretrained weights (start from random initialization)')
    parser.add_argument('--model', type=str, default=None,
                       choices=['tf_efficientnetv2_b0', 'tf_efficientnetv2_b3', 'efficientnetv2_rw_t'],
                       help='EfficientNetV2 model variant to use (overrides config if specified)')
    parser.add_argument('--dataset', type=str, default='celebdf_v2',
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0', 'dfdc'],
                       help='Dataset to use for training')
    parser.add_argument('--dataset-mode', type=str, default='original',
                       choices=['original', 'anonymized', 'anonymized_balanced', 'balanced'],
                       help='Dataset mode: original, anonymized, anonymized_balanced, or balanced')
    parser.add_argument('--use-dataset-weights', type=str2bool, nargs='?', const=True, default=None,
                       metavar='{true,false}',
                       help='Override dataset-level balancing weights (true/false)')
    parser.add_argument('--multi-dataset', action='store_true',
                       help='Enable multi-dataset training (uses all configured datasets)')
    parser.add_argument('--exclude-dataset', type=str, action='append', default=[],
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0', 'dfdc'],
                       help='Exclude specific dataset(s) from multi-dataset training (can be used multiple times)')
    parser.add_argument('--strict-batch-balance', action='store_true',
                       help='Enforce equal per-dataset and per-class samples within each batch')
    parser.add_argument('--use-class-weights', type=str2bool, nargs='?', const=True, default=None,
                       metavar='{true,false}',
                       help='Override class weighting when computing BCE loss (true/false)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                       help='Logging level for console output')

    # Evaluation-only mode parameters
    parser.add_argument('--eval-only', action='store_true',
                       help='Evaluation mode: load checkpoint and test on dataset (no training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation (required with --eval-only)')
    parser.add_argument('--test-dataset', type=str, default=None,
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0', 'dfdc'],
                       help='Dataset to test on for evaluation (required with --eval-only)')

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Configuration file not found: {config_path}")

    if args.eval_only:
        logger.info("")
        logger.info("==> Entering evaluation-only mode")
        run_evaluation(args)
        return None

    # Base configuration seeded from CLI defaults
    config = {
        'experiment': {
            'name': args.experiment_name,
            'description': 'EfficientNetV2 baseline training',
            'tags': ['baseline', 'efficientnet', 'stage0']
        },
        'data': {
            'dataset_config': args.dataset_config or 'configs/datasets.json',
            'dataset_name': args.dataset,
            'dataset_mode': args.dataset_mode,
            'multi_dataset': args.multi_dataset,
            'exclude_datasets': args.exclude_dataset,
            'image_size': 256,
            'subset_ratio': args.subset_ratio,
            'augmentation': True,
            'strict_batch_balance': args.strict_batch_balance
        },
        'model': {
            'name': args.model,
            'architecture': args.model,
            'num_classes': 1,
            'pretrained': not args.no_pretrained,
            'dropout_rate': 0.2
        },
        'training': {
            'weight_decay': 1e-4,
            'num_workers': 4,
            'use_class_weights': True,
            'early_stopping_patience': 10,
            'save_checkpoints': True
        }
    }

    # Load configuration file first
    logger.info("Loading configuration overrides from %s", config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        file_config = json.load(f)
    config.update(file_config)

    # Override with CLI arguments only if explicitly provided (not default values)
    training_cfg = config.setdefault('training', {})

    # Only override config file if command line args are explicitly provided
    if args.batch_size is not None:  # User specified this parameter
        training_cfg['batch_size'] = args.batch_size
        logger.info(f"Override config batch_size with command line argument: {args.batch_size}")

    if args.epochs is not None:  # User specified this parameter
        training_cfg['epochs'] = args.epochs
        logger.info(f"Override config epochs with command line argument: {args.epochs}")

    if args.learning_rate is not None:  # User specified this parameter
        training_cfg['learning_rate'] = args.learning_rate
        logger.info(f"Override config learning_rate with command line argument: {args.learning_rate}")
    if args.use_class_weights is not None:
        training_cfg['use_class_weights'] = args.use_class_weights

    experiment_cfg = config.setdefault('experiment', {})
    experiment_cfg['name'] = args.experiment_name

    model_cfg = config.setdefault('model', {})

    # Only override model config if user specified
    if args.model is not None:
        model_cfg.update({
            'name': args.model,
            'architecture': args.model,
        })
        logger.info(f"Override config model with command line argument: {args.model}")

    # Always set pretrained flag based on command line
    model_cfg['pretrained'] = not args.no_pretrained

    data_cfg = config.setdefault('data', {})

    # Only override config file if command line args are explicitly provided
    if args.dataset is not None:
        data_cfg['dataset_name'] = args.dataset
        logger.info(f"Override config dataset_name with command line argument: {args.dataset}")

    if args.dataset_mode is not None:
        data_cfg['dataset_mode'] = args.dataset_mode
        logger.info(f"Override config dataset_mode with command line argument: {args.dataset_mode}")

    if args.multi_dataset:  # Flag, only set if True
        data_cfg['multi_dataset'] = args.multi_dataset

    data_cfg['exclude_datasets'] = args.exclude_dataset

    if args.subset_ratio is not None:
        data_cfg['subset_ratio'] = args.subset_ratio
        logger.info(f"Override config subset_ratio with command line argument: {args.subset_ratio}")

    if args.use_dataset_weights is not None:
        data_cfg['use_dataset_weights'] = args.use_dataset_weights
    else:
        data_cfg.setdefault('use_dataset_weights', False)

    if args.dataset_config:
        data_cfg['dataset_config'] = args.dataset_config
    else:
        data_cfg.setdefault('dataset_config', 'configs/datasets.json')

    logger.debug("Effective configuration after overrides:\n%s", json.dumps(config, indent=2))

    exp_config = ExperimentConfig(
        experiment_name=experiment_cfg['name'],
        model_name=model_cfg.get('name', model_cfg.get('architecture', args.model)),
        dataset_name=data_cfg.get('dataset_name', 'celebdf_v2'),
        description=experiment_cfg.get('description', ''),
        tags=experiment_cfg.get('tags', []),
        batch_size=training_cfg['batch_size'],
        learning_rate=training_cfg['learning_rate'],
        num_epochs=training_cfg.get('epochs', 30),
        weight_decay=training_cfg.get('weight_decay', 1e-4)
    )

    exp_manager = ExperimentManager(base_path="experiments")

    # Create experiment manually and manage context manually for better control
    experiment_id = exp_manager.create_experiment(exp_config)
    logger.info("Started experiment: %s", experiment_id)

    # Store experiment_id for fallback in checkpoint saving
    exp_manager.experiment_id = experiment_id

    # Setup matplotlib before training
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']

    # Setup training components
    model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, device = setup_training(config)

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

    visualizer = AcademicVisualizer()
    training_logger = TrainingLogger(exp_manager, visualizer, experiment_id)
    logger.info("Training logger initialized")
    logger.info("Starting training for %d epochs", training_cfg['epochs'])
    logger.info("%s", "=" * 80)

    start_time = datetime.datetime.now()
    try:
        for epoch in range(1, training_cfg['epochs'] + 1):
            epoch_start_time = time.time()

            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
            scheduler.step()

            exp_manager.log_metrics(train_metrics, split='train', epoch=epoch)
            exp_manager.log_metrics({
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy'],
                'auc': val_metrics['auc'],
                'f1': val_metrics['f1']
            }, split='val', epoch=epoch)

            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_auc'].append(val_metrics['auc'])
            training_history['val_f1'].append(val_metrics['f1'])

          # Enhanced checkpoint saving logic with improved floating point comparison
            current_auc = float(val_metrics['auc'])  # Ensure Python float for comparison
            is_best = current_auc > best_val_auc or (abs(current_auc - best_val_auc) < 1e-8 and epoch == 0)

            # Detailed debugging information
            logger.info("Checkpoint evaluation - Epoch %d:", epoch)
            logger.info("  Current AUC: %.8f", current_auc)
            logger.info("  Best AUC:    %.8f", best_val_auc)
            logger.info("  Difference:  %.8f", abs(current_auc - best_val_auc))
            logger.info("  Is Best:     %s", is_best)

            if is_best:
                best_val_auc = current_auc
                patience_counter = 0

                logger.info("🎯 SAVING NEW BEST MODEL!")
                logger.info("   Epoch: %d", epoch)
                logger.info("   AUC: %.8f", best_val_auc)

                # Ensure experiment context is properly set
                if hasattr(exp_manager, 'current_experiment') and exp_manager.current_experiment is None:
                    logger.warning("⚠️ No active experiment context, setting it now...")
                    if hasattr(exp_manager, 'experiment_id') and exp_manager.experiment_id:
                        exp_manager.current_experiment = exp_manager.experiment_id
                        logger.info("✅ Experiment context set to: %s", exp_manager.experiment_id)
                    else:
                        logger.error("❌ No experiment_id available!")
                        continue

                try:
                    # Call save_checkpoint with detailed error tracking
                    checkpoint_path = exp_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        is_best=True
                    )

                    # Verify checkpoint was actually saved
                    if checkpoint_path and Path(checkpoint_path).exists():
                        file_size = Path(checkpoint_path).stat().st_size
                        logger.info("✅ Checkpoint saved successfully!")
                        logger.info("   Path: %s", checkpoint_path)
                        logger.info("   Size: %s bytes", f"{file_size:,}")
                    else:
                        logger.error("❌ Checkpoint file verification failed!")
                        logger.error("   Returned path: %s", checkpoint_path)

                except Exception as e:
                    logger.error("❌ Failed to save checkpoint: %s", str(e))
                    logger.error("   Exception type: %s", type(e).__name__)
                    import traceback
                    logger.error("   Traceback: %s", traceback.format_exc())
                    # Don't raise - continue training
            else:
                patience_counter += 1
                logger.debug("Not best model - patience increased to %d", patience_counter)

            epoch_time = time.time() - epoch_start_time
            logger.info("Epoch %02d Summary:", epoch)
            logger.info("  Train - Loss: %.4f, Acc: %.4f", train_metrics['loss'], train_metrics['accuracy'])
            logger.info(
                "  Val   - Loss: %.4f, Acc: %.4f, AUC: %.4f, F1: %.4f",
                val_metrics['loss'],
                val_metrics['accuracy'],
                val_metrics['auc'],
                val_metrics['f1'],
            )

            if 'per_dataset_metrics' in val_metrics:
                logger.debug("  Per-dataset validation breakdown:")
                for dataset_name, metrics in val_metrics['per_dataset_metrics'].items():
                    logger.debug(
                        "    %s → AUC: %.4f, F1: %.4f, Acc: %.4f (%s samples)",
                        dataset_name,
                        metrics['auc'],
                        metrics['f1'],
                        metrics['accuracy'],
                        f"{metrics['num_samples']:,}",
                    )

            logger.info(
                "  Time: %.2fs, LR: %.2e | Best AUC: %.4f, Patience: %d/%d",
                epoch_time,
                optimizer.param_groups[0]['lr'],
                best_val_auc,
                patience_counter,
                training_cfg['early_stopping_patience'],
            )
            logger.info("%s", "-" * 80)

            training_logger.log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=optimizer.param_groups[0]['lr'],
                per_dataset_metrics=val_metrics.get('per_dataset_metrics')
            )

            if patience_counter >= training_cfg['early_stopping_patience']:
                logger.warning("Early stopping triggered after %d epochs", epoch)
                break

        logger.info("Evaluating on test set...")
        test_metrics = validate_epoch(model, test_loader, criterion, device, epoch=-1)

        exp_manager.log_metrics({
            'loss': test_metrics['loss'],
            'accuracy': test_metrics['accuracy'],
            'auc': test_metrics['auc'],
            'f1': test_metrics['f1']
        }, split='test')

        logger.info("Final Test Results:")
        logger.info("  Loss: %.4f", test_metrics['loss'])
        logger.info("  Accuracy: %.4f", test_metrics['accuracy'])
        logger.info("  AUC: %.4f", test_metrics['auc'])
        logger.info("  F1: %.4f", test_metrics['f1'])

        if 'per_dataset_metrics' in test_metrics:
            logger.debug("  Per-dataset test breakdown:")
            for dataset_name, metrics in test_metrics['per_dataset_metrics'].items():
                logger.debug(
                    "    %s → AUC: %.4f, F1: %.4f, Acc: %.4f (%s samples)",
                    dataset_name,
                    metrics['auc'],
                    metrics['f1'],
                    metrics['accuracy'],
                    f"{metrics['num_samples']:,}",
                )

        training_logger.generate_final_report(training_history, test_metrics)

        # Mark experiment as completed successfully
        exp_manager.current_result.success = True
        exp_manager.current_result.finished_at = datetime.datetime.now().isoformat()
        exp_manager.current_result.total_training_time = (datetime.datetime.now() - start_time).total_seconds()
        exp_manager.registry["experiments"][experiment_id]["status"] = "completed"
        exp_manager._save_registry()

        logger.info("Experiment %s completed successfully!", experiment_id)
        return experiment_id

    except Exception as e:
        # Mark experiment as failed
        exp_manager.current_result.success = False
        exp_manager.current_result.error_message = str(e)
        exp_manager.current_result.finished_at = datetime.datetime.now().isoformat()
        exp_manager.registry["experiments"][experiment_id]["status"] = "failed"
        exp_manager.registry["experiments"][experiment_id]["error"] = str(e)
        exp_manager._save_registry()

        logger.error("Training failed with error: %s", str(e))
        raise



if __name__ == "__main__":
    experiment_id = main()
    if experiment_id is not None:
        logger.info("")
        logger.info("Training finished. Experiment ID: %s", experiment_id)
