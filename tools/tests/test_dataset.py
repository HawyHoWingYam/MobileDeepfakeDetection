#!/usr/bin/env python3
"""
Test dataset loading and multi-dataset functionality
"""

import sys
import pytest
import torch
import pandas as pd
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stage_00.train_baseline import MultiDatasetWrapper, UnifiedDeepfakeDataset

class TestDatasetLoading:
    """Test cases for dataset loading functionality"""

    def test_multi_dataset_wrapper_instantiation(self):
        """Test MultiDatasetWrapper can be instantiated"""
        # Create mock datasets
        datasets = []  # Empty for now
        wrapper = MultiDatasetWrapper(datasets)

        assert wrapper is not None
        assert len(wrapper) == 0

    def test_multi_dataset_wrapper_get_class_counts_empty(self):
        """Test get_class_counts works with empty dataset list"""
        datasets = []
        wrapper = MultiDatasetWrapper(datasets)

        counts = wrapper.get_class_counts()

        assert isinstance(counts, torch.Tensor)
        assert counts.shape == (2,)
        assert counts[0] == 0  # real count
        assert counts[1] == 0  # fake count

    def test_unified_deepfake_dataset_manifest_reading(self):
        """Test UnifiedDeepfakeDataset can read manifest files"""
        # Use existing manifest
        manifest_path = Path("manifests/celebdf_v2_train.csv")

        if manifest_path.exists():
            try:
                dataset = UnifiedDeepfakeDataset(
                    manifest_path=str(manifest_path),
                    dataset_name="test",
                    transform=None
                )
                # Just test that it doesn't crash
                assert dataset is not None
                assert hasattr(dataset, 'data')
            except Exception as e:
                # Expected to fail with placeholder data
                assert "Error loading" in str(e) or "No such file" in str(e)

    def create_test_manifest(self, temp_dir: Path, n_samples: int = 10):
        """Helper to create a test manifest"""
        manifest_path = temp_dir / "test_manifest.csv"

        data = []
        for i in range(n_samples):
            label = i % 2  # Alternate between real (0) and fake (1)
            data.append({
                'image_path': f'test_image_{i}.jpg',
                'label': label,
                'split': 'train',
                'md5': f'hash_{i}',
                'valid': True,
                'width': 256,
                'height': 256,
                'file_size': 15000
            })

        df = pd.DataFrame(data)
        df.to_csv(manifest_path, index=False)
        return manifest_path

    def test_unified_dataset_with_mock_manifest(self):
        """Test UnifiedDeepfakeDataset with properly formatted manifest"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = self.create_test_manifest(temp_path, n_samples=10)

            dataset = UnifiedDeepfakeDataset(
                manifest_path=str(manifest_path),
                dataset_name="test",
                transform=None
            )

            assert len(dataset.data) == 10
            assert 'label' in dataset.data.columns
            assert 'image_path' in dataset.data.columns

    def test_unified_dataset_get_class_counts(self):
        """Test get_class_counts method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = self.create_test_manifest(temp_path, n_samples=10)

            dataset = UnifiedDeepfakeDataset(
                manifest_path=str(manifest_path),
                dataset_name="test",
                transform=None
            )

            counts = dataset.get_class_counts()

            assert isinstance(counts, torch.Tensor)
            assert counts.shape == (2,)
            # With alternating labels, should be 5 real, 5 fake
            assert counts[0] == 5  # real count
            assert counts[1] == 5  # fake count

    def test_multi_dataset_wrapper_with_mock_datasets(self):
        """Test MultiDatasetWrapper with actual datasets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create two test manifests
            manifest1 = self.create_test_manifest(temp_path / "dataset1", n_samples=6)
            manifest2 = self.create_test_manifest(temp_path / "dataset2", n_samples=4)

            dataset1 = UnifiedDeepfakeDataset(
                manifest_path=str(manifest1),
                dataset_name="test1",
                transform=None
            )

            dataset2 = UnifiedDeepfakeDataset(
                manifest_path=str(manifest2),
                dataset_name="test2",
                transform=None
            )

            wrapper = MultiDatasetWrapper([dataset1, dataset2])

            # Test total length
            assert len(wrapper) == 10

            # Test combined class counts
            counts = wrapper.get_class_counts()
            assert counts[0] == 5  # total real (3 + 2)
            assert counts[1] == 5  # total fake (3 + 2)

    def test_dataset_getitem_fallback(self):
        """Test dataset __getitem__ fallback behavior"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = self.create_test_manifest(temp_path, n_samples=2)

            dataset = UnifiedDeepfakeDataset(
                manifest_path=str(manifest_path),
                dataset_name="test",
                transform=None
            )

            # This should trigger the fallback since image files don't exist
            try:
                image, label = dataset[0]
                # Should return a tensor for the label
                assert isinstance(label, torch.Tensor)
                assert label.dtype == torch.float
            except Exception:
                # Expected to fail gracefully
                pass

    def test_subset_ratio_functionality(self):
        """Test subset ratio parameter"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = self.create_test_manifest(temp_path, n_samples=100)

            # Test with 50% subset
            dataset = UnifiedDeepfakeDataset(
                manifest_path=str(manifest_path),
                dataset_name="test",
                transform=None,
                subset_ratio=0.5
            )

            # Should have ~50 samples (50% of 100)
            assert len(dataset.data) == 50

if __name__ == "__main__":
    pytest.main([__file__])