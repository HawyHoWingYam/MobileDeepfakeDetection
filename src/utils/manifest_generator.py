"""
AWARE-NET Manifest Generation and Validation Tools
Automated manifest file creation and dataset validation
"""

import os
import csv
import json
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import warnings

from .dataset_config import DatasetConfig, DatasetType

@dataclass
class ManifestEntry:
    """Single entry in a manifest file"""
    image_path: str
    label: int  # 0=real, 1=fake
    split: str
    md5: str = ""
    valid: bool = True
    error: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0

class ManifestGenerator:
    """
    Automated manifest generation for deepfake datasets
    
    Features:
    - Automatic dataset scanning and validation
    - MD5 integrity checking
    - Balanced train/val/test splitting
    - Support for multiple dataset formats
    - Cross-platform path handling
    """
    
    def __init__(self, 
                 config: DatasetConfig,
                 validate_images: bool = True,
                 calculate_md5: bool = True,
                 seed: int = 42):
        """
        Initialize manifest generator
        
        Args:
            config: Dataset configuration
            validate_images: Whether to validate image files
            calculate_md5: Whether to calculate MD5 hashes
            seed: Random seed for reproducible splits
        """
        self.config = config
        self.validate_images = validate_images
        self.calculate_md5 = calculate_md5
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Create manifest directory
        self.manifest_dir = Path("manifests")
        self.manifest_dir.mkdir(exist_ok=True)
    
    def scan_directory(self, 
                      directory: Path, 
                      label: int,
                      extensions: List[str] = None) -> List[ManifestEntry]:
        """
        Scan directory for images and create manifest entries
        
        Args:
            directory: Directory to scan
            label: Label for images (0=real, 1=fake)
            extensions: Allowed file extensions
            
        Returns:
            List of manifest entries
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        if not directory.exists():
            warnings.warn(f"Directory does not exist: {directory}")
            return []
        
        print(f"Scanning directory: {directory}")
        print(f"Label: {'Real' if label == 0 else 'Fake'}")
        
        entries = []
        image_files = []
        
        # Collect all image files
        for ext in extensions:
            pattern = f"**/*{ext}"
            image_files.extend(directory.glob(pattern))
        
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} image files")
        
        # Process each image
        for image_path in tqdm(image_files, desc=f"Processing {'real' if label == 0 else 'fake'} images"):
            entry = self._process_image(image_path, label)
            entries.append(entry)
        
        valid_entries = [e for e in entries if e.valid]
        print(f"Valid images: {len(valid_entries)}/{len(entries)}")
        
        return entries
    
    def _process_image(self, image_path: Path, label: int) -> ManifestEntry:
        """
        Process single image file and create manifest entry
        
        Args:
            image_path: Path to image file
            label: Image label
            
        Returns:
            Manifest entry
        """
        # Convert to relative path from root
        try:
            if self.config.root_path in image_path.parents:
                rel_path = image_path.relative_to(self.config.root_path)
            else:
                rel_path = image_path
        except ValueError:
            rel_path = image_path
        
        entry = ManifestEntry(
            image_path=str(rel_path).replace('\\', '/'),  # Use forward slashes for cross-platform
            label=label,
            split="",  # Will be assigned later
            file_size=image_path.stat().st_size if image_path.exists() else 0
        )
        
        # Calculate MD5 if requested
        if self.calculate_md5 and image_path.exists():
            entry.md5 = self._calculate_md5(image_path)
        
        # Validate image if requested
        if self.validate_images and image_path.exists():
            entry.valid, entry.error, entry.width, entry.height = self._validate_image(image_path)
        
        return entry
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _validate_image(self, image_path: Path) -> Tuple[bool, str, int, int]:
        """
        Validate image file
        
        Returns:
            Tuple of (is_valid, error_message, width, height)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Check if image can be loaded
                img.verify()
                
                # Check minimum size
                if width < 64 or height < 64:
                    return False, f"Image too small: {width}x{height}", width, height
                
                # Check if image is corrupted
                with Image.open(image_path) as img2:
                    img2.load()  # Force loading to detect corruption
                
                return True, "", width, height
                
        except Exception as e:
            return False, f"Image validation error: {str(e)}", 0, 0
    
    def split_data(self, 
                   entries: List[ManifestEntry],
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   stratify: bool = True) -> Dict[str, List[ManifestEntry]]:
        """
        Split data into train/val/test sets
        
        Args:
            entries: List of manifest entries
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify: Whether to maintain label balance across splits
            
        Returns:
            Dictionary with split assignments
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Filter valid entries
        valid_entries = [e for e in entries if e.valid]
        print(f"Splitting {len(valid_entries)} valid entries")
        
        if stratify:
            # Split by label to maintain balance
            real_entries = [e for e in valid_entries if e.label == 0]
            fake_entries = [e for e in valid_entries if e.label == 1]
            
            print(f"Real images: {len(real_entries)}")
            print(f"Fake images: {len(fake_entries)}")
            
            # Shuffle independently
            random.shuffle(real_entries)
            random.shuffle(fake_entries)
            
            splits = {"train": [], "val": [], "test": []}
            
            for entries_list in [real_entries, fake_entries]:
                n = len(entries_list)
                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)
                
                train_split = entries_list[:train_end]
                val_split = entries_list[train_end:val_end]
                test_split = entries_list[val_end:]
                
                # Assign split names
                for entry in train_split:
                    entry.split = "train"
                for entry in val_split:
                    entry.split = "val"
                for entry in test_split:
                    entry.split = "test"
                
                splits["train"].extend(train_split)
                splits["val"].extend(val_split)
                splits["test"].extend(test_split)
        else:
            # Simple random split
            random.shuffle(valid_entries)
            n = len(valid_entries)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits = {
                "train": valid_entries[:train_end],
                "val": valid_entries[train_end:val_end],
                "test": valid_entries[val_end:]
            }
            
            # Assign split names
            for split_name, split_entries in splits.items():
                for entry in split_entries:
                    entry.split = split_name
        
        # Print split statistics
        for split_name, split_entries in splits.items():
            real_count = sum(1 for e in split_entries if e.label == 0)
            fake_count = sum(1 for e in split_entries if e.label == 1)
            print(f"{split_name}: {len(split_entries)} samples ({real_count} real, {fake_count} fake)")
        
        return splits
    
    def generate_manifest_file(self, 
                              entries: List[ManifestEntry],
                              output_path: Path) -> None:
        """
        Write manifest entries to CSV file
        
        Args:
            entries: List of manifest entries
            output_path: Output CSV file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        data = []
        for entry in entries:
            data.append({
                'image_path': entry.image_path,
                'label': entry.label,
                'split': entry.split,
                'md5': entry.md5,
                'valid': entry.valid,
                'error': entry.error,
                'width': entry.width,
                'height': entry.height,
                'file_size': entry.file_size
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"Manifest saved: {output_path}")
        print(f"Total entries: {len(entries)}")
    
    def generate_full_dataset_manifest(self, 
                                     real_dir: Optional[Path] = None,
                                     fake_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Generate complete dataset manifest with all splits
        
        Args:
            real_dir: Directory containing real images (optional, uses config)
            fake_dir: Directory containing fake images (optional, uses config)
            
        Returns:
            Dictionary mapping split names to manifest file paths
        """
        print("=== Generating Full Dataset Manifest ===")
        
        # Use provided directories or fall back to config
        if real_dir is None:
            real_dir = Path(self.config.config["paths"]["real_images"])
        if fake_dir is None:
            fake_dir = Path(self.config.config["paths"]["fake_images"])
        
        print(f"Real images directory: {real_dir}")
        print(f"Fake images directory: {fake_dir}")
        
        # Scan directories
        all_entries = []
        
        # Scan real images
        if real_dir.exists():
            real_entries = self.scan_directory(real_dir, label=0)
            all_entries.extend(real_entries)
        else:
            warnings.warn(f"Real images directory not found: {real_dir}")
        
        # Scan fake images
        if fake_dir.exists():
            fake_entries = self.scan_directory(fake_dir, label=1)
            all_entries.extend(fake_entries)
        else:
            warnings.warn(f"Fake images directory not found: {fake_dir}")
        
        if not all_entries:
            raise ValueError("No images found in specified directories")
        
        # Split data
        splits_config = self.config.splits
        train_ratio = splits_config["train"].ratio
        val_ratio = splits_config["val"].ratio
        test_ratio = splits_config["test"].ratio
        
        splits = self.split_data(all_entries, train_ratio, val_ratio, test_ratio)
        
        # Generate manifest files for each split
        manifest_paths = {}
        
        for split_name, split_entries in splits.items():
            manifest_path = self.config.get_manifest_path(split_name)
            self.generate_manifest_file(split_entries, manifest_path)
            manifest_paths[split_name] = manifest_path
        
        # Update dataset statistics in config
        self._update_dataset_statistics(splits)
        
        return manifest_paths
    
    def _update_dataset_statistics(self, splits: Dict[str, List[ManifestEntry]]) -> None:
        """Update dataset statistics in configuration"""
        total_samples = sum(len(entries) for entries in splits.values())
        total_real = sum(sum(1 for e in entries if e.label == 0) for entries in splits.values())
        total_fake = sum(sum(1 for e in entries if e.label == 1) for entries in splits.values())
        
        # Update config metadata
        self.config.config["metadata"].update({
            "total_samples": total_samples,
            "real_samples": total_real,
            "fake_samples": total_fake,
            "updated_at": pd.Timestamp.now().isoformat()
        })
        
        print(f"\nDataset Statistics:")
        print(f"Total samples: {total_samples}")
        print(f"Real samples: {total_real}")
        print(f"Fake samples: {total_fake}")
    
    def validate_existing_manifest(self, manifest_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate existing manifest file
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not manifest_path.exists():
            errors.append(f"Manifest file does not exist: {manifest_path}")
            return False, errors
        
        try:
            df = pd.read_csv(manifest_path)
            
            # Check required columns
            required_cols = ['image_path', 'label', 'split']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")
            
            # Check image files
            missing_files = []
            for idx, row in df.iterrows():
                image_path = Path(self.config.root_path) / row['image_path']
                if not image_path.exists():
                    missing_files.append(str(image_path))
            
            if missing_files:
                errors.append(f"Missing image files: {len(missing_files)} files")
                if len(missing_files) <= 5:  # Show first 5
                    errors.extend(missing_files)
            
            # Check label distribution
            if 'label' in df.columns:
                label_dist = df['label'].value_counts()
                if len(label_dist) != 2:
                    errors.append("Dataset should have exactly 2 labels (0=real, 1=fake)")
                
                # Check balance
                if len(label_dist) == 2:
                    ratio = min(label_dist) / max(label_dist)
                    if ratio < 0.1:  # Very imbalanced
                        errors.append(f"Dataset is very imbalanced (ratio: {ratio:.3f})")
            
        except Exception as e:
            errors.append(f"Error reading manifest: {str(e)}")
        
        return len(errors) == 0, errors

def main():
    """Command-line interface for manifest generation"""
    parser = argparse.ArgumentParser(description="Generate dataset manifests for AWARE-NET")
    parser.add_argument("--config", required=True, help="Dataset configuration file")
    parser.add_argument("--real-dir", help="Directory containing real images")
    parser.add_argument("--fake-dir", help="Directory containing fake images")
    parser.add_argument("--no-validate", action="store_true", help="Skip image validation")
    parser.add_argument("--no-md5", action="store_true", help="Skip MD5 calculation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DatasetConfig(args.config)
    
    # Create generator
    generator = ManifestGenerator(
        config=config,
        validate_images=not args.no_validate,
        calculate_md5=not args.no_md5,
        seed=args.seed
    )
    
    # Generate manifests
    real_dir = Path(args.real_dir) if args.real_dir else None
    fake_dir = Path(args.fake_dir) if args.fake_dir else None
    
    manifest_paths = generator.generate_full_dataset_manifest(real_dir, fake_dir)
    
    print(f"\n=== Manifest Generation Complete ===")
    for split_name, manifest_path in manifest_paths.items():
        print(f"{split_name}: {manifest_path}")

if __name__ == "__main__":
    main()