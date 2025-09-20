"""
AWARE-NET Test Suite
Comprehensive testing framework for Stage 0 components
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path


def setup_test_environment():
    """Setup test environment with necessary configurations"""
    torch.manual_seed(42)
    np.random.seed(42)


def create_mock_dataset(n_samples=100, n_real=50, n_fake=50):
    """Create mock dataset for testing"""
    assert n_real + n_fake == n_samples
    
    # Generate mock data
    images = np.random.rand(n_samples, 256, 256, 3).astype(np.uint8)
    labels = np.concatenate([np.zeros(n_real), np.ones(n_fake)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    images = images[indices]
    labels = labels[indices]
    
    return images, labels


def create_mock_predictions(n_samples=100, auc_target=0.9):
    """Create mock predictions with controlled AUC"""
    labels = np.random.binomial(1, 0.5, n_samples)
    
    # Generate predictions correlated with labels to achieve target AUC
    base_probs = np.random.rand(n_samples)
    
    # Adjust probabilities to match labels more closely
    adjustment = (labels - 0.5) * (auc_target - 0.5) * 2
    predictions = np.clip(base_probs + adjustment, 0, 1)
    
    return labels, predictions


def create_temp_manifest(images, labels, temp_dir):
    """Create temporary manifest file for testing"""
    import pandas as pd
    
    manifest_data = []
    for i, (label) in enumerate(labels):
        image_path = temp_dir / f"image_{i:04d}.png"
        manifest_data.append({
            'path': str(image_path),
            'label': int(label),
            'real': int(label == 0),
            'fake': int(label == 1)
        })
    
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = temp_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    return manifest_path, manifest_df