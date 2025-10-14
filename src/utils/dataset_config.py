"""
AWARE-NET Dataset Configuration Management
Configuration-driven dataset loading and validation system
"""

import json
import hashlib
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from PIL import Image
import warnings

class DatasetType(Enum):
    """Supported dataset types"""
    DF40 = "df40"
    FACEFORENSICS = "faceforensics"
    CELEBDF = "celebdf"
    DFDC = "dfdc"
    CUSTOM = "custom"

class DataSplitType(Enum):
    """Data split types"""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

@dataclass
class DatasetSplit:
    """Dataset split configuration"""
    name: str
    ratio: float
    manifest_path: Optional[str] = None
    min_samples: int = 100
    max_samples: Optional[int] = None

@dataclass
class DatasetMeta:
    """Dataset metadata"""
    name: str
    version: str
    description: str
    total_samples: int
    real_samples: int
    fake_samples: int
    image_format: str = "png"
    image_size: Tuple[int, int] = (256, 256)
    created_at: str = ""
    updated_at: str = ""

class DatasetConfig:
    """
    Configuration-driven dataset management system
    
    Features:
    - JSON-based configuration files
    - Automatic manifest generation and validation
    - Cross-platform path handling
    - Data integrity checks with MD5 validation
    - Support for multiple dataset formats
    """
    
    def __init__(self,
                 config_path: Union[str, Path],
                 dataset_name: Optional[str] = None):
        """
        Initialize dataset configuration
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.raw_config = self.config.copy() if isinstance(self.config, dict) else {}
        self.dataset_name = dataset_name
        self.dataset_entry = None

        if dataset_name and isinstance(self.config, dict) and 'datasets' in self.config:
            datasets = self.config.get('datasets', {})
            if dataset_name not in datasets:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in configuration {self.config_path}"
                )
            self.dataset_entry = datasets[dataset_name]

        self.root_path = Path(self.config.get("root_path", self.raw_config.get("root_path", ".")))
        self.metadata = self._parse_metadata()
        self.splits = self._parse_splits()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def _parse_metadata(self) -> DatasetMeta:
        """Parse dataset metadata from configuration"""
        if self.dataset_entry:
            statistics = self.dataset_entry.get("statistics", {})
            default_settings = self.raw_config.get("default_settings", {})
            default_size = default_settings.get("image_size", [256, 256])
            default_format = default_settings.get("image_format", "png")

            return DatasetMeta(
                name=self.dataset_entry.get("name", self.dataset_name or "unknown"),
                version=self.raw_config.get("metadata", {}).get("version", "1.0"),
                description=self.dataset_entry.get("description", ""),
                total_samples=statistics.get("total_samples", 0),
                real_samples=statistics.get("real_samples", 0),
                fake_samples=statistics.get("fake_samples", 0),
                image_format=self.dataset_entry.get("image_format", default_format),
                image_size=tuple(self.dataset_entry.get("image_size", default_size)),
                created_at=self.dataset_entry.get("created_at", ""),
                updated_at=self.dataset_entry.get("updated_at", "")
            )

        meta_config = self.config.get("metadata", {})
        return DatasetMeta(
            name=meta_config.get("name", "unknown"),
            version=meta_config.get("version", "1.0"),
            description=meta_config.get("description", ""),
            total_samples=meta_config.get("total_samples", 0),
            real_samples=meta_config.get("real_samples", 0),
            fake_samples=meta_config.get("fake_samples", 0),
            image_format=meta_config.get("image_format", "png"),
            image_size=tuple(meta_config.get("image_size", [256, 256])),
            created_at=meta_config.get("created_at", ""),
            updated_at=meta_config.get("updated_at", "")
        )
    
    def _parse_splits(self) -> Dict[str, DatasetSplit]:
        """Parse data splits from configuration"""
        if self.dataset_entry:
            manifests = self.dataset_entry.get("manifests", {})
            statistics = self.dataset_entry.get("statistics", {})
            default_split_config = self.raw_config.get("default_settings", {}).get("splits", {})
            ratio_map = {
                key.replace('_ratio', ''): value
                for key, value in default_split_config.items()
                if key.endswith('_ratio')
            }

            splits = {}
            for split_name, manifest_path in manifests.items():
                splits[split_name] = DatasetSplit(
                    name=split_name,
                    ratio=ratio_map.get(split_name, 0.0),
                    manifest_path=manifest_path,
                    min_samples=statistics.get(f"{split_name}_samples", 100),
                    max_samples=None
                )
            return splits

        splits_config = self.config.get("splits", {})
        splits = {}
        
        for split_name, split_config in splits_config.items():
            splits[split_name] = DatasetSplit(
                name=split_name,
                ratio=split_config.get("ratio", 0.0),
                manifest_path=split_config.get("manifest_path"),
                min_samples=split_config.get("min_samples", 100),
                max_samples=split_config.get("max_samples")
            )
        
        return splits
    
    def get_dataset_path(self, relative_path: str = "") -> Path:
        """
        Get absolute dataset path with cross-platform compatibility
        
        Args:
            relative_path: Relative path within dataset
            
        Returns:
            Absolute path to dataset location
        """
        if self.dataset_entry:
            dataset_path = self.dataset_entry.get("dataset_path")
            if dataset_path:
                base_path = Path(dataset_path)
                if not base_path.is_absolute():
                    base_path = self.root_path / base_path
            else:
                base_path = self.root_path
        else:
            base_path = self.root_path / self.config.get("dataset_path", "")
        if relative_path:
            return base_path / relative_path
        return base_path
    
    def get_manifest_path(self, split: str) -> Path:
        """
        Get manifest file path for specific split
        
        Args:
            split: Split name (train/val/test)
            
        Returns:
            Path to manifest file
        """
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}")
        
        split_config = self.splits[split]
        if split_config.manifest_path:
            return self.root_path / split_config.manifest_path
        else:
            # Generate default manifest path
            manifest_dir = self.root_path / "manifests"
            manifest_dir.mkdir(exist_ok=True)
            return manifest_dir / f"{self.metadata.name}_{split}.csv"

    def available_splits(self) -> List[str]:
        """Return list of configured split names."""
        return list(self.splits.keys())

    def has_manifest(self, split: str) -> bool:
        """Check whether manifest exists on disk for given split."""
        try:
            manifest_path = self.get_manifest_path(split)
        except ValueError:
            return False
        return manifest_path.exists()
    
    def validate_paths(self) -> Tuple[bool, List[str]]:
        """
        Validate all configured paths exist
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check root path
        if not self.root_path.exists():
            errors.append(f"Root path does not exist: {self.root_path}")
        
        # Check dataset path
        dataset_path = self.get_dataset_path()
        if not dataset_path.exists():
            errors.append(f"Dataset path does not exist: {dataset_path}")
        
        # Check manifest paths
        for split_name in self.splits:
            manifest_path = self.get_manifest_path(split_name)
            manifest_dir = manifest_path.parent
            if not manifest_dir.exists():
                manifest_dir.mkdir(parents=True, exist_ok=True)
        
        return len(errors) == 0, errors
    
    def calculate_md5(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of file for integrity checking
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            warnings.warn(f"Failed to calculate MD5 for {file_path}: {e}")
            return ""
    
    def validate_image(self, image_path: Path) -> Tuple[bool, str]:
        """
        Validate image file format and properties
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with Image.open(image_path) as img:
                # Check format
                expected_format = self.metadata.image_format.upper()
                if img.format != expected_format:
                    return False, f"Expected {expected_format}, got {img.format}"
                
                # Check size
                expected_size = self.metadata.image_size
                if img.size != expected_size:
                    return False, f"Expected size {expected_size}, got {img.size}"
                
                return True, ""
                
        except Exception as e:
            return False, f"Image validation error: {str(e)}"
    
    def scan_dataset_directory(self, 
                             directory: Path,
                             pattern: str = "**/*.png") -> List[Path]:
        """
        Scan directory for image files matching pattern
        
        Args:
            directory: Directory to scan
            pattern: Glob pattern for files
            
        Returns:
            List of image file paths
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        return sorted(list(directory.glob(pattern)))
    
    def generate_manifest(self, 
                         split: str,
                         image_paths: List[Path],
                         labels: List[int],
                         validate_images: bool = True) -> Path:
        """
        Generate manifest CSV file for dataset split
        
        Args:
            split: Split name
            image_paths: List of image file paths
            labels: List of labels (0=real, 1=fake)
            validate_images: Whether to validate each image
            
        Returns:
            Path to generated manifest file
        """
        if len(image_paths) != len(labels):
            raise ValueError("Number of images and labels must match")
        
        manifest_path = self.get_manifest_path(split)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare manifest data
        manifest_data = []
        
        for image_path, label in zip(image_paths, labels):
            # Convert to relative path for portability
            try:
                rel_path = image_path.relative_to(self.root_path)
            except ValueError:
                rel_path = image_path  # Use absolute if relative fails
            
            row = {
                'image_path': str(rel_path),
                'label': label,
                'split': split,
                'md5': self.calculate_md5(image_path) if image_path.exists() else "",
                'valid': True
            }
            
            # Validate image if requested
            if validate_images and image_path.exists():
                is_valid, error_msg = self.validate_image(image_path)
                row['valid'] = is_valid
                if not is_valid:
                    row['error'] = error_msg
                    warnings.warn(f"Invalid image {image_path}: {error_msg}")
            
            manifest_data.append(row)
        
        # Write to CSV
        df = pd.DataFrame(manifest_data)
        df.to_csv(manifest_path, index=False)
        
        print(f"Generated manifest: {manifest_path}")
        print(f"Total samples: {len(manifest_data)}")
        print(f"Real samples: {(df['label'] == 0).sum()}")
        print(f"Fake samples: {(df['label'] == 1).sum()}")
        
        return manifest_path
    
    def load_manifest(self, split: str) -> pd.DataFrame:
        """
        Load manifest file for specified split
        
        Args:
            split: Split name
            
        Returns:
            DataFrame with manifest data
        """
        manifest_path = self.get_manifest_path(split)
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        try:
            df = pd.read_csv(manifest_path)
            
            # Validate required columns
            required_cols = ['image_path', 'label', 'split']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load manifest {manifest_path}: {e}")
    
    def verify_manifest_integrity(self, split: str, 
                                check_md5: bool = False) -> Tuple[bool, List[str]]:
        """
        Verify integrity of manifest file and referenced images
        
        Args:
            split: Split name
            check_md5: Whether to verify MD5 hashes
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            df = self.load_manifest(split)
            
            for idx, row in df.iterrows():
                image_path = self.root_path / row['image_path']
                
                # Check if file exists
                if not image_path.exists():
                    errors.append(f"Missing image: {image_path}")
                    continue
                
                # Check MD5 if requested and available
                if check_md5 and 'md5' in row and row['md5']:
                    current_md5 = self.calculate_md5(image_path)
                    if current_md5 != row['md5']:
                        errors.append(f"MD5 mismatch for {image_path}")
                
                # Validate image format
                is_valid, error_msg = self.validate_image(image_path)
                if not is_valid:
                    errors.append(f"Invalid image {image_path}: {error_msg}")
            
        except Exception as e:
            errors.append(f"Manifest verification failed: {e}")
        
        return len(errors) == 0, errors
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'metadata': self.metadata.__dict__,
            'splits': {},
            'total_samples': 0,
            'total_real': 0,
            'total_fake': 0
        }
        
        for split_name in self.splits:
            try:
                df = self.load_manifest(split_name)
                split_stats = {
                    'samples': len(df),
                    'real': (df['label'] == 0).sum(),
                    'fake': (df['label'] == 1).sum(),
                    'valid': df.get('valid', pd.Series([True] * len(df))).sum()
                }
                stats['splits'][split_name] = split_stats
                stats['total_samples'] += split_stats['samples']
                stats['total_real'] += split_stats['real']
                stats['total_fake'] += split_stats['fake']
                
            except FileNotFoundError:
                stats['splits'][split_name] = {
                    'samples': 0,
                    'real': 0,
                    'fake': 0,
                    'valid': 0,
                    'error': 'Manifest not found'
                }
        
        return stats
    
    def export_config(self, output_path: Path) -> None:
        """
        Export current configuration to JSON file
        
        Args:
            output_path: Path for exported configuration
        """
        # Update metadata with current statistics
        stats = self.get_dataset_statistics()
        
        export_config = self.config.copy()
        export_config['metadata'].update({
            'total_samples': stats['total_samples'],
            'real_samples': stats['total_real'],
            'fake_samples': stats['total_fake'],
            'updated_at': pd.Timestamp.now().isoformat()
        })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_config, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration exported to: {output_path}")

def create_dataset_config_template(dataset_name: str, 
                                 dataset_type: DatasetType,
                                 output_path: Path) -> None:
    """
    Create a template configuration file for a new dataset
    
    Args:
        dataset_name: Name of the dataset
        dataset_type: Type of dataset
        output_path: Path for the template file
    """
    template = {
        "metadata": {
            "name": dataset_name,
            "version": "1.0",
            "description": f"{dataset_type.value.upper()} dataset configuration",
            "dataset_type": dataset_type.value,
            "total_samples": 0,
            "real_samples": 0,
            "fake_samples": 0,
            "image_format": "png",
            "image_size": [256, 256],
            "created_at": pd.Timestamp.now().isoformat()
        },
        "root_path": ".",
        "dataset_path": f"datasets/{dataset_name}",
        "splits": {
            "train": {
                "ratio": 0.7,
                "min_samples": 1000,
                "manifest_path": f"manifests/{dataset_name}_train.csv"
            },
            "val": {
                "ratio": 0.15,
                "min_samples": 200,
                "manifest_path": f"manifests/{dataset_name}_val.csv"
            },
            "test": {
                "ratio": 0.15,
                "min_samples": 200,
                "manifest_path": f"manifests/{dataset_name}_test.csv"
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
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset configuration template created: {output_path}")
