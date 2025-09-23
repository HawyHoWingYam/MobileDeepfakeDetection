#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for AWARE-NET tests
"""

import pytest
import sys
import tempfile
import pandas as pd
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def temp_manifest():
    """Create a temporary manifest file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create sample manifest data
        data = {
            'image_path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'label': [0, 1, 0, 1],
            'split': ['train', 'train', 'val', 'val'],
            'md5': ['hash1', 'hash2', 'hash3', 'hash4'],
            'valid': [True, True, True, True],
            'width': [256, 256, 256, 256],
            'height': [256, 256, 256, 256],
            'file_size': [15000, 16000, 15500, 15800]
        }

        df = pd.DataFrame(data)
        df.to_csv(f.name, index=False)

        yield f.name

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

@pytest.fixture
def sample_predictions():
    """Generate sample prediction data for testing"""
    import numpy as np

    np.random.seed(42)
    n_samples = 100

    y_true = np.random.binomial(1, 0.6, n_samples)
    y_scores = np.random.beta(2, 3, n_samples)
    y_pred = (y_scores > 0.5).astype(int)

    return {
        'y_true': y_true,
        'y_scores': y_scores,
        'y_pred': y_pred,
        'n_samples': n_samples
    }

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )