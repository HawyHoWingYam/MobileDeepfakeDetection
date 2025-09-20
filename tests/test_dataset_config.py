"""
Test suite for dataset configuration and data loading
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import cv2
from unittest.mock import patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.dataset_config import DatasetConfig
from utils.manifest_generator import ManifestGenerator
from utils.data_validator import DataValidator


class TestDatasetConfig:
    """Test DatasetConfig class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_data = {
            "metadata": {
                "name": "test_dataset",
                "version": "1.0",
                "description": "Test dataset for unit tests",
                "dataset_type": "celebdf",
                "total_samples": 100,
                "real_samples": 50,
                "fake_samples": 50,
                "image_format": "png",
                "image_size": [256, 256],
                "created_at": "2025-01-15T00:00:00"
            },
            "root_path": str(self.temp_dir),
            "dataset_path": "data",
            "splits": {
                "train": {
                    "ratio": 0.7,
                    "min_samples": 50,
                    "manifest_path": "manifests/train.csv"
                },
                "val": {
                    "ratio": 0.15,
                    "min_samples": 10,
                    "manifest_path": "manifests/val.csv"
                },
                "test": {
                    "ratio": 0.15,
                    "min_samples": 10,
                    "manifest_path": "manifests/test.csv"
                }
            },
            "preprocessing": {
                "resize": True,
                "normalize": True,
                "augmentation": {
                    "enabled": True,
                    "rotation": 15,
                    "flip": True,
                    "color_jitter": 0.1
                }
            },
            "paths": {
                "real_images": str(self.temp_dir / "real"),
                "fake_images": str(self.temp_dir / "fake"),
                "metadata_file": "metadata.json"
            }
        }
        
        # Create config file
        self.config_path = self.temp_dir / "config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_dataset_config_initialization(self):
        """Test DatasetConfig initialization"""
        config = DatasetConfig(str(self.config_path))
        
        assert config.name == "test_dataset"
        assert config.version == "1.0"
        assert config.dataset_type == "celebdf"
        assert config.total_samples == 100
        assert config.real_samples == 50
        assert config.fake_samples == 50
        assert config.image_format == "png"
        assert config.image_size == [256, 256]
    
    def test_dataset_config_paths(self):
        """Test path handling in DatasetConfig"""
        config = DatasetConfig(str(self.config_path))
        
        assert Path(config.root_path) == self.temp_dir
        assert config.splits["train"]["ratio"] == 0.7
        assert config.splits["val"]["ratio"] == 0.15
        assert config.splits["test"]["ratio"] == 0.15
        
        # Test path resolution
        train_manifest_path = config.get_manifest_path("train")
        assert "manifests/train.csv" in str(train_manifest_path)
    
    def test_dataset_config_validation(self):
        """Test dataset configuration validation"""
        config = DatasetConfig(str(self.config_path))
        
        # Test split ratios sum to 1.0
        total_ratio = sum(split["ratio"] for split in config.splits.values())
        assert abs(total_ratio - 1.0) < 1e-6
        
        # Test minimum samples requirements
        for split_name, split_config in config.splits.items():
            assert split_config["min_samples"] > 0
            assert 0 < split_config["ratio"] < 1
    
    def test_preprocessing_config(self):
        """Test preprocessing configuration"""
        config = DatasetConfig(str(self.config_path))
        
        preprocessing = config.preprocessing
        assert preprocessing["resize"] is True
        assert preprocessing["normalize"] is True
        assert preprocessing["augmentation"]["enabled"] is True
        assert preprocessing["augmentation"]["rotation"] == 15
        assert preprocessing["augmentation"]["flip"] is True
        assert preprocessing["augmentation"]["color_jitter"] == 0.1
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file"""
        # Create invalid JSON file
        invalid_config_path = self.temp_dir / "invalid_config.json"
        with open(invalid_config_path, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises((json.JSONDecodeError, ValueError)):
            DatasetConfig(str(invalid_config_path))
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file"""
        missing_path = self.temp_dir / "missing_config.json"
        
        with pytest.raises(FileNotFoundError):
            DatasetConfig(str(missing_path))
    
    def test_config_update_and_save(self):
        """Test configuration update and save functionality"""
        config = DatasetConfig(str(self.config_path))
        
        # Update some values
        original_samples = config.total_samples
        config.total_samples = 200
        config.real_samples = 100
        config.fake_samples = 100
        
        # Save updated config
        updated_config_path = self.temp_dir / "updated_config.json"
        config.save(str(updated_config_path))
        
        # Load updated config and verify changes
        updated_config = DatasetConfig(str(updated_config_path))
        assert updated_config.total_samples == 200
        assert updated_config.real_samples == 100
        assert updated_config.fake_samples == 100
        
        # Original config should be unchanged
        original_config = DatasetConfig(str(self.config_path))
        assert original_config.total_samples == original_samples


class TestManifestGenerator:
    """Test ManifestGenerator class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample image directories and files
        self.real_dir = self.temp_dir / "real"
        self.fake_dir = self.temp_dir / "fake"
        self.real_dir.mkdir(parents=True)
        self.fake_dir.mkdir(parents=True)
        
        # Create sample images
        self.create_sample_images()
        
        self.generator = ManifestGenerator(
            real_image_dir=str(self.real_dir),
            fake_image_dir=str(self.fake_dir),
            output_dir=str(self.temp_dir / "manifests")
        )
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_sample_images(self, n_real=20, n_fake=15):
        """Create sample images for testing"""
        # Create real images
        for i in range(n_real):
            image = Image.new('RGB', (256, 256), color=(i*10 % 256, i*5 % 256, i*7 % 256))
            image.save(self.real_dir / f"real_{i:04d}.png")
        
        # Create fake images
        for i in range(n_fake):
            image = Image.new('RGB', (256, 256), color=(i*15 % 256, i*12 % 256, i*3 % 256))
            image.save(self.fake_dir / f"fake_{i:04d}.png")
    
    def test_manifest_generation(self):
        """Test manifest file generation"""
        manifest_path = self.generator.generate_manifest()
        
        assert manifest_path.exists()
        
        # Load and verify manifest
        manifest_df = pd.read_csv(manifest_path)
        
        assert len(manifest_df) == 35  # 20 real + 15 fake
        assert set(manifest_df.columns) >= {'path', 'label', 'real', 'fake'}
        
        # Check label distribution
        real_count = len(manifest_df[manifest_df['label'] == 0])
        fake_count = len(manifest_df[manifest_df['label'] == 1])
        
        assert real_count == 20
        assert fake_count == 15
        
        # Check that paths exist
        for path_str in manifest_df['path']:
            path = Path(path_str)
            assert path.exists(), f"Path does not exist: {path}"
    
    def test_manifest_with_splits(self):
        """Test manifest generation with train/val/test splits"""
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        
        manifest_paths = self.generator.generate_split_manifests(
            split_ratios=split_ratios,
            stratify=True,
            random_state=42
        )
        
        assert 'train' in manifest_paths
        assert 'val' in manifest_paths
        assert 'test' in manifest_paths
        
        # Load all manifests and check sizes
        total_samples = 0
        for split_name, path in manifest_paths.items():
            assert path.exists()
            df = pd.read_csv(path)
            total_samples += len(df)
            
            # Check class balance in each split
            real_count = len(df[df['label'] == 0])
            fake_count = len(df[df['label'] == 1])
            assert real_count > 0
            assert fake_count > 0
            
            print(f"{split_name}: {len(df)} samples ({real_count} real, {fake_count} fake)")
        
        assert total_samples == 35  # Should equal total images
    
    def test_manifest_with_metadata(self):
        """Test manifest generation with metadata inclusion"""
        # Create metadata file
        metadata = {
            f"real_{i:04d}.png": {"quality": "high", "source": "original"}
            for i in range(20)
        }
        metadata.update({
            f"fake_{i:04d}.png": {"quality": "medium", "source": "deepfake", "method": "faceswap"}
            for i in range(15)
        })
        
        metadata_path = self.temp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        self.generator.metadata_path = str(metadata_path)
        manifest_path = self.generator.generate_manifest(include_metadata=True)
        
        manifest_df = pd.read_csv(manifest_path)
        
        # Check that metadata columns are present
        assert 'quality' in manifest_df.columns
        assert 'source' in manifest_df.columns
        
        # Check metadata for fake images
        fake_rows = manifest_df[manifest_df['label'] == 1]
        assert 'method' in fake_rows.columns
        assert all(fake_rows['method'] == 'faceswap')
    
    def test_manifest_validation(self):
        """Test manifest validation functionality"""
        manifest_path = self.generator.generate_manifest()
        
        # Validate the generated manifest
        is_valid, errors = self.generator.validate_manifest(str(manifest_path))
        
        assert is_valid
        assert len(errors) == 0
        
        # Create invalid manifest for testing
        invalid_manifest_path = self.temp_dir / "invalid_manifest.csv"
        invalid_data = {
            'path': ['/nonexistent/file1.png', '/nonexistent/file2.png'],
            'label': [0, 1],
            'real': [1, 0],
            'fake': [0, 1]
        }
        pd.DataFrame(invalid_data).to_csv(invalid_manifest_path, index=False)
        
        is_valid, errors = self.generator.validate_manifest(str(invalid_manifest_path))
        assert not is_valid
        assert len(errors) > 0
    
    def test_image_format_filtering(self):
        """Test filtering by image format"""
        # Create images with different formats
        mixed_dir = self.temp_dir / "mixed"
        mixed_dir.mkdir()
        
        # Create PNG, JPG, and other files
        Image.new('RGB', (100, 100)).save(mixed_dir / "image1.png")
        Image.new('RGB', (100, 100)).save(mixed_dir / "image2.jpg")
        Image.new('RGB', (100, 100)).save(mixed_dir / "image3.jpeg")
        with open(mixed_dir / "not_image.txt", 'w') as f:
            f.write("test")
        
        generator = ManifestGenerator(
            real_image_dir=str(mixed_dir),
            fake_image_dir=str(self.fake_dir),
            supported_formats=['.png', '.jpg', '.jpeg']
        )
        
        manifest_path = generator.generate_manifest()
        manifest_df = pd.read_csv(manifest_path)
        
        # Should include PNG, JPG, JPEG but not TXT
        paths = manifest_df['path'].tolist()
        image_extensions = [Path(p).suffix.lower() for p in paths]
        
        assert '.png' in image_extensions
        assert '.jpg' in image_extensions or '.jpeg' in image_extensions
        assert '.txt' not in image_extensions


class TestDataValidator:
    """Test DataValidator class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = DataValidator()
        
        # Create test manifest
        self.create_test_manifest()
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_manifest(self):
        """Create test manifest with sample data"""
        # Create sample images
        image_dir = self.temp_dir / "images"
        image_dir.mkdir()
        
        manifest_data = []
        for i in range(10):
            # Create actual image file
            image = Image.new('RGB', (256, 256), color=(i*25, i*25, i*25))
            image_path = image_dir / f"image_{i:04d}.png"
            image.save(image_path)
            
            manifest_data.append({
                'path': str(image_path),
                'label': i % 2,  # Alternate between 0 and 1
                'real': 1 - (i % 2),
                'fake': i % 2
            })
        
        self.manifest_path = self.temp_dir / "test_manifest.csv"
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_csv(self.manifest_path, index=False)
    
    def test_image_validation(self):
        """Test individual image validation"""
        # Test valid image
        valid_image_path = self.temp_dir / "valid.png"
        Image.new('RGB', (256, 256)).save(valid_image_path)
        
        is_valid, error = self.validator.validate_image(str(valid_image_path))
        assert is_valid
        assert error is None
        
        # Test invalid image (corrupted)
        invalid_image_path = self.temp_dir / "invalid.png"
        with open(invalid_image_path, 'wb') as f:
            f.write(b"not an image")
        
        is_valid, error = self.validator.validate_image(str(invalid_image_path))
        assert not is_valid
        assert error is not None
        
        # Test missing image
        missing_image_path = self.temp_dir / "missing.png"
        
        is_valid, error = self.validator.validate_image(str(missing_image_path))
        assert not is_valid
        assert "does not exist" in error.lower()
    
    def test_manifest_validation(self):
        """Test manifest file validation"""
        validation_result = self.validator.validate_manifest(str(self.manifest_path))
        
        assert validation_result['is_valid']
        assert validation_result['total_samples'] == 10
        assert validation_result['valid_samples'] == 10
        assert validation_result['invalid_samples'] == 0
        assert len(validation_result['errors']) == 0
        
        # Check class distribution
        assert validation_result['class_distribution'][0] == 5  # Real
        assert validation_result['class_distribution'][1] == 5  # Fake
    
    def test_manifest_with_missing_files(self):
        """Test manifest validation with missing image files"""
        # Create manifest with non-existent files
        invalid_data = [
            {'path': '/nonexistent/file1.png', 'label': 0, 'real': 1, 'fake': 0},
            {'path': '/nonexistent/file2.png', 'label': 1, 'real': 0, 'fake': 1}
        ]
        
        invalid_manifest_path = self.temp_dir / "invalid_manifest.csv"
        pd.DataFrame(invalid_data).to_csv(invalid_manifest_path, index=False)
        
        validation_result = self.validator.validate_manifest(str(invalid_manifest_path))
        
        assert not validation_result['is_valid']
        assert validation_result['valid_samples'] == 0
        assert validation_result['invalid_samples'] == 2
        assert len(validation_result['errors']) == 2
    
    def test_image_statistics(self):
        """Test image statistics calculation"""
        stats = self.validator.calculate_image_statistics(str(self.manifest_path))
        
        assert 'total_images' in stats
        assert 'valid_images' in stats
        assert 'average_dimensions' in stats
        assert 'format_distribution' in stats
        assert 'size_distribution' in stats
        
        assert stats['total_images'] == 10
        assert stats['valid_images'] == 10
        assert stats['format_distribution']['png'] == 10  # All images are PNG
    
    def test_data_quality_check(self):
        """Test comprehensive data quality check"""
        quality_report = self.validator.comprehensive_quality_check(
            str(self.manifest_path),
            check_duplicates=True,
            check_corruption=True,
            check_dimensions=True
        )
        
        assert 'manifest_validation' in quality_report
        assert 'image_statistics' in quality_report
        assert 'quality_issues' in quality_report
        assert 'recommendations' in quality_report
        
        # Should have no quality issues with our clean test data
        assert len(quality_report['quality_issues']) == 0
        
        # Should pass manifest validation
        assert quality_report['manifest_validation']['is_valid']


class TestDatasetIntegration:
    """Integration tests for dataset components"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.setup_complete_dataset()
    
    def teardown_method(self):
        """Cleanup after integration tests"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def setup_complete_dataset(self):
        """Setup complete dataset structure for integration testing"""
        # Create directory structure
        (self.temp_dir / "real").mkdir(parents=True)
        (self.temp_dir / "fake").mkdir(parents=True)
        (self.temp_dir / "manifests").mkdir(parents=True)
        
        # Create sample images
        n_real, n_fake = 50, 30
        
        for i in range(n_real):
            image = Image.new('RGB', (256, 256), color=(100, 150, 200))
            image.save(self.temp_dir / "real" / f"real_{i:04d}.png")
        
        for i in range(n_fake):
            image = Image.new('RGB', (256, 256), color=(200, 100, 150))
            image.save(self.temp_dir / "fake" / f"fake_{i:04d}.png")
        
        # Create dataset configuration
        self.config_data = {
            "metadata": {
                "name": "integration_test_dataset",
                "version": "1.0",
                "description": "Integration test dataset",
                "dataset_type": "celebdf",
                "total_samples": n_real + n_fake,
                "real_samples": n_real,
                "fake_samples": n_fake,
                "image_format": "png",
                "image_size": [256, 256],
                "created_at": "2025-01-15T00:00:00"
            },
            "root_path": str(self.temp_dir),
            "splits": {
                "train": {"ratio": 0.7, "min_samples": 30},
                "val": {"ratio": 0.15, "min_samples": 5},
                "test": {"ratio": 0.15, "min_samples": 5}
            },
            "paths": {
                "real_images": str(self.temp_dir / "real"),
                "fake_images": str(self.temp_dir / "fake")
            }
        }
        
        self.config_path = self.temp_dir / "config.json"
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
    
    def test_complete_dataset_pipeline(self):
        """Test complete dataset preparation pipeline"""
        # 1. Load dataset configuration
        config = DatasetConfig(str(self.config_path))
        assert config.total_samples == 80
        
        # 2. Generate manifests
        generator = ManifestGenerator(
            real_image_dir=config.paths["real_images"],
            fake_image_dir=config.paths["fake_images"],
            output_dir=str(self.temp_dir / "manifests")
        )
        
        split_manifests = generator.generate_split_manifests(
            split_ratios={split: info["ratio"] for split, info in config.splits.items()},
            stratify=True,
            random_state=42
        )
        
        # 3. Validate all manifests
        validator = DataValidator()
        total_samples = 0
        
        for split_name, manifest_path in split_manifests.items():
            validation_result = validator.validate_manifest(str(manifest_path))
            
            assert validation_result['is_valid']
            assert validation_result['invalid_samples'] == 0
            total_samples += validation_result['total_samples']
            
            # Check minimum sample requirements
            assert validation_result['total_samples'] >= config.splits[split_name]["min_samples"]
            
            print(f"{split_name}: {validation_result['total_samples']} samples")
        
        assert total_samples == config.total_samples
        
        # 4. Generate quality report
        train_manifest_path = split_manifests['train']
        quality_report = validator.comprehensive_quality_check(str(train_manifest_path))
        
        assert quality_report['manifest_validation']['is_valid']
        assert len(quality_report['quality_issues']) == 0
        
        return config, split_manifests, quality_report
    
    def test_dataset_configuration_update(self):
        """Test dataset configuration updates after manifest generation"""
        config = DatasetConfig(str(self.config_path))
        
        # Generate manifests
        generator = ManifestGenerator(
            real_image_dir=config.paths["real_images"],
            fake_image_dir=config.paths["fake_images"],
            output_dir=str(self.temp_dir / "manifests")
        )
        
        split_manifests = generator.generate_split_manifests(
            split_ratios={split: info["ratio"] for split, info in config.splits.items()},
            stratify=True,
            random_state=42
        )
        
        # Update configuration with manifest paths
        for split_name, manifest_path in split_manifests.items():
            config.splits[split_name]["manifest_path"] = str(manifest_path.relative_to(self.temp_dir))
        
        # Save updated configuration
        updated_config_path = self.temp_dir / "updated_config.json"
        config.save(str(updated_config_path))
        
        # Verify updated configuration
        updated_config = DatasetConfig(str(updated_config_path))
        for split_name in config.splits:
            assert "manifest_path" in updated_config.splits[split_name]
            manifest_path = self.temp_dir / updated_config.splits[split_name]["manifest_path"]
            assert manifest_path.exists()


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])