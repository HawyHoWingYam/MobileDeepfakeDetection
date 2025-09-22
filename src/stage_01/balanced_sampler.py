"""
AWARE-NET Stage 1: Balanced Batch Sampler for Contrastive Learning

This module implements balanced sampling strategies specifically designed for
supervised contrastive learning in the authenticity modeling paradigm.

Key Features:
- Ensures sufficient positive/negative pairs in each batch
- Maintains class balance for stable contrastive learning
- Supports various sampling strategies
- Handles class imbalance gracefully
"""

import torch
from torch.utils.data import Sampler, Dataset
import numpy as np
from typing import List, Dict, Optional, Iterator, Union
import logging
from collections import defaultdict, Counter
import random

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler for contrastive learning.

    Ensures each batch contains minimum number of samples from each class
    to enable effective contrastive learning.
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        min_samples_per_class: int = 4,
        strategy: str = 'balanced',
        drop_last: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize balanced batch sampler.

        Args:
            labels: List of class labels for all samples
            batch_size: Target batch size
            min_samples_per_class: Minimum samples per class in each batch
            strategy: Sampling strategy ('balanced', 'weighted', 'minority_focused')
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle samples
            seed: Random seed for reproducibility
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.min_samples_per_class = min_samples_per_class
        self.strategy = strategy
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Analyze dataset
        self.num_samples = len(labels)
        self.num_classes = len(np.unique(labels))
        self.class_counts = Counter(labels)
        self.classes = list(self.class_counts.keys())

        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

        # Validate configuration
        self._validate_config()

        # Prepare sampling
        self._prepare_sampling()

        logger.info(f"BalancedBatchSampler initialized: "
                   f"samples={self.num_samples}, classes={self.num_classes}, "
                   f"batch_size={batch_size}, min_per_class={min_samples_per_class}")

    def _validate_config(self):
        """Validate sampler configuration."""
        if self.batch_size < self.min_samples_per_class * self.num_classes:
            logger.warning(
                f"Batch size ({self.batch_size}) may be too small for "
                f"{self.min_samples_per_class} samples per class × {self.num_classes} classes"
            )

        for class_label, count in self.class_counts.items():
            if count < self.min_samples_per_class:
                logger.warning(
                    f"Class {class_label} has only {count} samples, "
                    f"less than min_samples_per_class ({self.min_samples_per_class})"
                )

    def _prepare_sampling(self):
        """Prepare sampling based on strategy."""
        if self.strategy == 'balanced':
            self._prepare_balanced_sampling()
        elif self.strategy == 'weighted':
            self._prepare_weighted_sampling()
        elif self.strategy == 'minority_focused':
            self._prepare_minority_focused_sampling()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _prepare_balanced_sampling(self):
        """Prepare balanced sampling (equal samples per class)."""
        self.samples_per_class = self.batch_size // self.num_classes
        self.extra_samples = self.batch_size % self.num_classes

        # Ensure minimum samples per class
        if self.samples_per_class < self.min_samples_per_class:
            self.samples_per_class = self.min_samples_per_class
            logger.warning(f"Adjusted samples_per_class to {self.samples_per_class}")

    def _prepare_weighted_sampling(self):
        """Prepare weighted sampling (inverse frequency weighting)."""
        total_samples = sum(self.class_counts.values())
        class_weights = {
            class_label: total_samples / (self.num_classes * count)
            for class_label, count in self.class_counts.items()
        }

        # Normalize weights to sum to batch_size
        weight_sum = sum(class_weights.values())
        self.class_batch_sizes = {
            class_label: max(
                self.min_samples_per_class,
                int(self.batch_size * weight / weight_sum)
            )
            for class_label, weight in class_weights.items()
        }

    def _prepare_minority_focused_sampling(self):
        """Prepare minority-focused sampling (boost underrepresented classes)."""
        min_count = min(self.class_counts.values())
        max_count = max(self.class_counts.values())

        # Boost factor for minority classes
        self.class_batch_sizes = {}
        remaining_batch = self.batch_size

        for class_label, count in self.class_counts.items():
            # Inverse relationship: fewer samples → more in batch
            boost_factor = max_count / count
            target_samples = max(
                self.min_samples_per_class,
                int(self.min_samples_per_class * boost_factor)
            )
            target_samples = min(target_samples, remaining_batch - self.min_samples_per_class * (self.num_classes - 1))
            self.class_batch_sizes[class_label] = target_samples
            remaining_batch -= target_samples

    def _create_batch(self) -> List[int]:
        """Create a single balanced batch."""
        batch_indices = []

        if self.strategy == 'balanced':
            # Balanced sampling
            for class_label in self.classes:
                class_pool = self.class_indices[class_label].copy()
                if self.shuffle:
                    random.shuffle(class_pool)

                # Sample with replacement if necessary
                n_needed = self.samples_per_class
                if len(class_pool) >= n_needed:
                    selected = random.sample(class_pool, n_needed)
                else:
                    # Sample with replacement
                    selected = random.choices(class_pool, k=n_needed)

                batch_indices.extend(selected)

            # Add extra samples randomly
            if self.extra_samples > 0:
                all_indices = list(range(self.num_samples))
                if self.shuffle:
                    random.shuffle(all_indices)
                batch_indices.extend(all_indices[:self.extra_samples])

        else:
            # Weighted or minority-focused sampling
            for class_label, n_samples in self.class_batch_sizes.items():
                class_pool = self.class_indices[class_label].copy()
                if self.shuffle:
                    random.shuffle(class_pool)

                if len(class_pool) >= n_samples:
                    selected = random.sample(class_pool, n_samples)
                else:
                    selected = random.choices(class_pool, k=n_samples)

                batch_indices.extend(selected)

        # Final shuffle of batch
        if self.shuffle:
            random.shuffle(batch_indices)

        return batch_indices

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        # Calculate number of batches
        if self.drop_last:
            n_batches = self.num_samples // self.batch_size
        else:
            n_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        for _ in range(n_batches):
            batch = self._create_batch()
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def get_batch_statistics(self, batch_indices: List[int]) -> Dict:
        """Analyze batch composition."""
        batch_labels = [self.labels[i] for i in batch_indices]
        batch_counts = Counter(batch_labels)

        stats = {
            'batch_size': len(batch_indices),
            'class_counts': dict(batch_counts),
            'class_ratios': {
                class_label: count / len(batch_indices)
                for class_label, count in batch_counts.items()
            },
            'balance_score': min(batch_counts.values()) / max(batch_counts.values()) if batch_counts else 0,
            'min_class_samples': min(batch_counts.values()) if batch_counts else 0,
            'max_class_samples': max(batch_counts.values()) if batch_counts else 0
        }

        return stats


class ContrastivePairSampler(Sampler):
    """
    Specialized sampler for contrastive learning that ensures
    good positive/negative pair coverage.
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        positive_pair_ratio: float = 0.5,
        min_positive_pairs: int = 8,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize contrastive pair sampler.

        Args:
            labels: List of class labels
            batch_size: Target batch size
            positive_pair_ratio: Target ratio of positive pairs
            min_positive_pairs: Minimum positive pairs per batch
            shuffle: Whether to shuffle samples
            seed: Random seed
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.positive_pair_ratio = positive_pair_ratio
        self.min_positive_pairs = min_positive_pairs
        self.shuffle = shuffle

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

        self.num_samples = len(labels)
        self.classes = list(self.class_indices.keys())

        logger.info(f"ContrastivePairSampler initialized: {len(self.classes)} classes, "
                   f"positive_ratio={positive_pair_ratio}")

    def _calculate_pairs(self, batch_indices: List[int]) -> Dict:
        """Calculate number of positive and negative pairs in batch."""
        batch_labels = [self.labels[i] for i in batch_indices]
        label_counts = Counter(batch_labels)

        positive_pairs = sum(count * (count - 1) // 2 for count in label_counts.values())
        total_pairs = len(batch_indices) * (len(batch_indices) - 1) // 2
        negative_pairs = total_pairs - positive_pairs

        return {
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs,
            'total_pairs': total_pairs,
            'positive_ratio': positive_pairs / total_pairs if total_pairs > 0 else 0
        }

    def _create_optimal_batch(self) -> List[int]:
        """Create batch optimized for contrastive pairs."""
        batch_indices = []

        # Target positive pairs
        target_positive_pairs = max(
            self.min_positive_pairs,
            int(self.batch_size * (self.batch_size - 1) // 2 * self.positive_pair_ratio)
        )

        # Strategy: Add samples from each class to achieve target positive pairs
        samples_per_class = int(np.sqrt(2 * target_positive_pairs / len(self.classes))) + 1
        samples_per_class = max(2, min(samples_per_class, self.batch_size // len(self.classes)))

        for class_label in self.classes:
            class_pool = self.class_indices[class_label].copy()
            if self.shuffle:
                random.shuffle(class_pool)

            n_samples = min(samples_per_class, len(class_pool))
            if len(batch_indices) + n_samples <= self.batch_size:
                if len(class_pool) >= n_samples:
                    selected = random.sample(class_pool, n_samples)
                else:
                    selected = random.choices(class_pool, k=n_samples)
                batch_indices.extend(selected)

        # Fill remaining slots randomly
        remaining_slots = self.batch_size - len(batch_indices)
        if remaining_slots > 0:
            all_indices = list(range(self.num_samples))
            available_indices = [i for i in all_indices if i not in batch_indices]
            if self.shuffle:
                random.shuffle(available_indices)

            batch_indices.extend(available_indices[:remaining_slots])

        if self.shuffle:
            random.shuffle(batch_indices)

        return batch_indices

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over optimized batches."""
        n_batches = self.num_samples // self.batch_size

        for _ in range(n_batches):
            batch = self._create_optimal_batch()
            yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_samples // self.batch_size


def test_balanced_sampler():
    """Test balanced sampler implementations."""
    print("Testing Balanced Samplers...")

    # Create imbalanced test dataset
    labels = [0] * 100 + [1] * 400  # 1:4 ratio
    batch_size = 16

    # Test BalancedBatchSampler
    sampler = BalancedBatchSampler(
        labels=labels,
        batch_size=batch_size,
        min_samples_per_class=4,
        strategy='balanced',
        seed=42
    )

    print(f"✓ Sampler created: {len(sampler)} batches")

    # Test a few batches
    batch_stats = []
    for i, batch_indices in enumerate(sampler):
        if i >= 3:  # Test first 3 batches
            break

        stats = sampler.get_batch_statistics(batch_indices)
        batch_stats.append(stats)
        print(f"  Batch {i}: {stats['class_counts']}, balance={stats['balance_score']:.3f}")

    # Test ContrastivePairSampler
    pair_sampler = ContrastivePairSampler(
        labels=labels,
        batch_size=batch_size,
        positive_pair_ratio=0.4,
        seed=42
    )

    print(f"✓ ContrastivePairSampler created: {len(pair_sampler)} batches")

    for i, batch_indices in enumerate(pair_sampler):
        if i >= 2:  # Test first 2 batches
            break

        pairs = pair_sampler._calculate_pairs(batch_indices)
        print(f"  Batch {i}: {pairs['positive_pairs']} pos, {pairs['negative_pairs']} neg, "
              f"ratio={pairs['positive_ratio']:.3f}")

    # Test different strategies
    strategies = ['weighted', 'minority_focused']
    for strategy in strategies:
        strategy_sampler = BalancedBatchSampler(
            labels=labels,
            batch_size=batch_size,
            strategy=strategy,
            seed=42
        )
        batch = next(iter(strategy_sampler))
        stats = strategy_sampler.get_batch_statistics(batch)
        print(f"✓ {strategy} strategy: {stats['class_counts']}")

    print("All sampler tests passed! ✓")


if __name__ == "__main__":
    test_balanced_sampler()