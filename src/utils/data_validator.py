"""
AWARE-NET Data Validation Tools
Comprehensive data integrity and quality checking
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings

from .dataset_config import DatasetConfig

@dataclass
class ValidationResult:
    """Results of data validation"""
    is_valid: bool
    total_files: int
    valid_files: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ImageValidationResult:
    """Results of single image validation"""
    path: str
    is_valid: bool
    width: int = 0
    height: int = 0
    format: str = ""
    file_size: int = 0
    md5: str = ""
    error: str = ""

class DataValidator:
    """
    Comprehensive data validation for deepfake datasets
    
    Features:
    - Image format and integrity validation
    - MD5 hash verification
    - Dataset balance analysis
    - Path consistency checks
    - Performance statistics
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize data validator
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.validation_results = {}
    
    def validate_image_file(self, image_path: Path) -> ImageValidationResult:
        """
        Validate single image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image validation result
        """
        result = ImageValidationResult(path=str(image_path), is_valid=False)
        
        try:
            if not image_path.exists():
                result.error = "File does not exist"
                return result
            
            # Get file size
            result.file_size = image_path.stat().st_size
            
            # Calculate MD5
            result.md5 = self._calculate_md5(image_path)
            
            # Validate image with PIL
            with Image.open(image_path) as img:
                result.width, result.height = img.size
                result.format = img.format
                
                # Check minimum dimensions
                if result.width < 32 or result.height < 32:
                    result.error = f"Image too small: {result.width}x{result.height}"
                    return result
                
                # Check maximum dimensions (reasonable limit)
                if result.width > 4096 or result.height > 4096:
                    result.error = f"Image too large: {result.width}x{result.height}"
                    return result
                
                # Verify image integrity
                img.verify()
                
                # Try to load image data
                with Image.open(image_path) as img2:
                    img2.load()
                
                result.is_valid = True
                
        except Exception as e:
            result.error = f"Validation error: {str(e)}"
        
        return result
    
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
    
    def validate_manifest_file(self, manifest_path: Path) -> ValidationResult:
        """
        Validate manifest file and referenced images
        
        Args:
            manifest_path: Path to manifest CSV file
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=False, total_files=0, valid_files=0)
        
        if not manifest_path.exists():
            result.errors.append(f"Manifest file does not exist: {manifest_path}")
            return result
        
        try:
            # Load manifest
            df = pd.read_csv(manifest_path)
            result.total_files = len(df)
            
            # Check required columns
            required_cols = ['image_path', 'label', 'split']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                result.errors.append(f"Missing required columns: {missing_cols}")
                return result
            
            # Validate each image
            valid_count = 0
            image_results = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating images"):
                # Convert relative path to absolute
                image_path = self.config.root_path / row['image_path']
                
                # Validate image
                img_result = self.validate_image_file(image_path)
                image_results.append(img_result)
                
                if img_result.is_valid:
                    valid_count += 1
                else:
                    result.errors.append(f"Invalid image {image_path}: {img_result.error}")
                
                # Verify MD5 if available
                if 'md5' in row and row['md5'] and img_result.is_valid:
                    if img_result.md5 != row['md5']:
                        result.warnings.append(f"MD5 mismatch for {image_path}")
            
            result.valid_files = valid_count
            
            # Calculate statistics
            result.statistics = self._calculate_manifest_statistics(df, image_results)
            
            # Check if validation passed
            success_rate = valid_count / result.total_files if result.total_files > 0 else 0
            result.is_valid = success_rate >= 0.95  # 95% success rate threshold
            
            if success_rate < 1.0:
                result.warnings.append(f"Validation success rate: {success_rate:.2%}")
            
        except Exception as e:
            result.errors.append(f"Error validating manifest: {str(e)}")
        
        return result
    
    def _calculate_manifest_statistics(self, 
                                     df: pd.DataFrame,
                                     image_results: List[ImageValidationResult]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for manifest"""
        stats = {}
        
        # Basic counts
        stats['total_samples'] = len(df)
        stats['valid_samples'] = sum(1 for r in image_results if r.is_valid)
        stats['invalid_samples'] = stats['total_samples'] - stats['valid_samples']
        
        # Label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts().to_dict()
            stats['real_samples'] = label_counts.get(0, 0)
            stats['fake_samples'] = label_counts.get(1, 0)
            
            # Balance ratio
            if stats['real_samples'] > 0 and stats['fake_samples'] > 0:
                stats['balance_ratio'] = min(stats['real_samples'], stats['fake_samples']) / max(stats['real_samples'], stats['fake_samples'])
            else:
                stats['balance_ratio'] = 0.0
        
        # Split distribution
        if 'split' in df.columns:
            stats['split_distribution'] = df['split'].value_counts().to_dict()
        
        # Image statistics (only for valid images)
        valid_results = [r for r in image_results if r.is_valid]
        if valid_results:
            widths = [r.width for r in valid_results]
            heights = [r.height for r in valid_results]
            file_sizes = [r.file_size for r in valid_results]
            
            stats['image_stats'] = {
                'width_mean': np.mean(widths),
                'width_std': np.std(widths),
                'height_mean': np.mean(heights),
                'height_std': np.std(heights),
                'file_size_mean': np.mean(file_sizes),
                'file_size_std': np.std(file_sizes),
                'total_size_mb': sum(file_sizes) / (1024 * 1024)
            }
            
            # Format distribution
            formats = [r.format for r in valid_results]
            stats['format_distribution'] = pd.Series(formats).value_counts().to_dict()
        
        return stats
    
    def validate_dataset_splits(self) -> Dict[str, ValidationResult]:
        """
        Validate all dataset splits
        
        Returns:
            Dictionary mapping split names to validation results
        """
        results = {}
        
        for split_name in self.config.splits:
            print(f"Validating {split_name} split...")
            
            manifest_path = self.config.get_manifest_path(split_name)
            result = self.validate_manifest_file(manifest_path)
            results[split_name] = result
            
            # Print summary
            if result.is_valid:
                print(f"✅ {split_name}: {result.valid_files}/{result.total_files} files valid")
            else:
                print(f"❌ {split_name}: {result.valid_files}/{result.total_files} files valid")
                if result.errors:
                    print(f"   Errors: {len(result.errors)}")
                if result.warnings:
                    print(f"   Warnings: {len(result.warnings)}")
        
        return results
    
    def check_dataset_balance(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Check dataset balance across splits
        
        Args:
            results: Validation results for each split
            
        Returns:
            Balance analysis
        """
        balance_info = {
            'overall_balance': {},
            'split_balance': {},
            'recommendations': []
        }
        
        # Overall balance
        total_real = sum(r.statistics.get('real_samples', 0) for r in results.values())
        total_fake = sum(r.statistics.get('fake_samples', 0) for r in results.values())
        
        balance_info['overall_balance'] = {
            'real_samples': total_real,
            'fake_samples': total_fake,
            'total_samples': total_real + total_fake,
            'balance_ratio': min(total_real, total_fake) / max(total_real, total_fake) if max(total_real, total_fake) > 0 else 0
        }
        
        # Per-split balance
        for split_name, result in results.items():
            if 'real_samples' in result.statistics and 'fake_samples' in result.statistics:
                real_count = result.statistics['real_samples']
                fake_count = result.statistics['fake_samples']
                
                balance_info['split_balance'][split_name] = {
                    'real_samples': real_count,
                    'fake_samples': fake_count,
                    'balance_ratio': min(real_count, fake_count) / max(real_count, fake_count) if max(real_count, fake_count) > 0 else 0
                }
        
        # Generate recommendations
        overall_ratio = balance_info['overall_balance']['balance_ratio']
        if overall_ratio < 0.8:
            balance_info['recommendations'].append(
                f"Dataset is imbalanced (ratio: {overall_ratio:.3f}). Consider balancing real/fake samples."
            )
        
        # Check split sizes
        total_samples = balance_info['overall_balance']['total_samples']
        for split_name, result in results.items():
            split_size = result.statistics.get('total_samples', 0)
            split_ratio = split_size / total_samples if total_samples > 0 else 0
            
            expected_ratio = self.config.splits[split_name].ratio
            if abs(split_ratio - expected_ratio) > 0.05:  # 5% tolerance
                balance_info['recommendations'].append(
                    f"{split_name} split size ({split_ratio:.1%}) differs from expected ({expected_ratio:.1%})"
                )
        
        return balance_info
    
    def generate_validation_report(self, 
                                 results: Dict[str, ValidationResult],
                                 output_path: Path) -> None:
        """
        Generate comprehensive validation report
        
        Args:
            results: Validation results for each split
            output_path: Path for report file
        """
        report = {
            'dataset_info': {
                'name': self.config.metadata.name,
                'version': self.config.metadata.version,
                'description': self.config.metadata.description
            },
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'validation_results': {},
            'overall_statistics': {},
            'balance_analysis': {},
            'recommendations': []
        }
        
        # Add validation results
        for split_name, result in results.items():
            report['validation_results'][split_name] = {
                'is_valid': result.is_valid,
                'total_files': result.total_files,
                'valid_files': result.valid_files,
                'success_rate': result.valid_files / result.total_files if result.total_files > 0 else 0,
                'errors': result.errors,
                'warnings': result.warnings,
                'statistics': result.statistics
            }
        
        # Calculate overall statistics
        report['overall_statistics'] = {
            'total_files': sum(r.total_files for r in results.values()),
            'total_valid': sum(r.valid_files for r in results.values()),
            'overall_success_rate': sum(r.valid_files for r in results.values()) / sum(r.total_files for r in results.values()) if sum(r.total_files for r in results.values()) > 0 else 0,
            'total_errors': sum(len(r.errors) for r in results.values()),
            'total_warnings': sum(len(r.warnings) for r in results.values())
        }
        
        # Add balance analysis
        report['balance_analysis'] = self.check_dataset_balance(results)
        
        # Add recommendations
        report['recommendations'].extend(report['balance_analysis']['recommendations'])
        
        # Overall recommendations
        if report['overall_statistics']['overall_success_rate'] < 0.95:
            report['recommendations'].append("Consider cleaning dataset - low validation success rate")
        
        if report['overall_statistics']['total_errors'] > 0:
            report['recommendations'].append("Fix validation errors before training")
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Validation report saved: {output_path}")
    
    def quick_validate(self) -> bool:
        """
        Quick validation check for all splits
        
        Returns:
            True if all splits pass basic validation
        """
        print("Running quick validation...")
        
        all_valid = True
        
        for split_name in self.config.splits:
            manifest_path = self.config.get_manifest_path(split_name)
            
            if not manifest_path.exists():
                print(f"❌ {split_name}: Manifest file missing")
                all_valid = False
                continue
            
            try:
                df = pd.read_csv(manifest_path)
                
                # Check required columns
                required_cols = ['image_path', 'label', 'split']
                if not all(col in df.columns for col in required_cols):
                    print(f"❌ {split_name}: Missing required columns")
                    all_valid = False
                    continue
                
                # Check if files exist (sample check)
                sample_size = min(10, len(df))
                sample_df = df.sample(n=sample_size)
                
                missing_count = 0
                for _, row in sample_df.iterrows():
                    image_path = self.config.root_path / row['image_path']
                    if not image_path.exists():
                        missing_count += 1
                
                if missing_count > 0:
                    print(f"⚠️ {split_name}: {missing_count}/{sample_size} sampled files missing")
                    if missing_count > sample_size / 2:
                        all_valid = False
                else:
                    print(f"✅ {split_name}: Basic validation passed")
                
            except Exception as e:
                print(f"❌ {split_name}: Error reading manifest - {str(e)}")
                all_valid = False
        
        return all_valid

def main():
    """Command-line interface for data validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate AWARE-NET dataset")
    parser.add_argument("--config", required=True, help="Dataset configuration file")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--quick", action="store_true", help="Quick validation only")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DatasetConfig(args.config)
    
    # Create validator
    validator = DataValidator(config)
    
    if args.quick:
        # Quick validation
        is_valid = validator.quick_validate()
        print(f"\nQuick validation: {'PASSED' if is_valid else 'FAILED'}")
    else:
        # Full validation
        results = validator.validate_dataset_splits()
        
        # Generate report
        if args.output:
            validator.generate_validation_report(results, Path(args.output))
        else:
            validator.generate_validation_report(results, Path("validation_report.json"))

if __name__ == "__main__":
    main()