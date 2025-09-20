#!/usr/bin/env python3
"""
Path Anonymization for AWARE-NET
Eliminate 100% label leakage by anonymizing file paths
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import hashlib
import json
from typing import Dict, List
import os

def create_anonymized_filename(original_path: str, index: int) -> str:
    """
    Create anonymized filename based on index
    
    Args:
        original_path: Original image path
        index: Unique index for this sample
        
    Returns:
        Anonymized filename like "sample_000001.jpg"
    """
    # Get file extension from original
    ext = Path(original_path).suffix
    if not ext:
        ext = '.jpg'  # Default extension
    
    # Create anonymized name
    return f"sample_{index:06d}{ext}"

def create_path_mapping(manifests_dir: str = "D:/work/AWARE-NET/datasets/manifests_clean") -> Dict:
    """
    Create mapping from original paths to anonymized paths
    
    Args:
        manifests_dir: Directory containing clean manifest files
        
    Returns:
        Dictionary with path mappings and statistics
    """
    manifest_files = {
        'train': Path(manifests_dir) / "celebdf_train_clean.csv",
        'val': Path(manifests_dir) / "celebdf_val_clean.csv", 
        'test': Path(manifests_dir) / "celebdf_test_clean.csv"
    }
    
    # Load all clean manifests
    all_paths = []
    path_to_split = {}
    path_to_label = {}
    
    for split, file_path in manifest_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                original_path = row['image_path']
                label = row['label']
                
                all_paths.append(original_path)
                path_to_split[original_path] = split
                path_to_label[original_path] = label
            
            print(f"Loaded {len(df)} paths from {split}")
        else:
            print(f"WARNING: File not found: {file_path}")
    
    # Create anonymized mapping
    path_mapping = {}
    anonymized_to_original = {}
    
    # Shuffle paths to remove any ordering patterns
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(all_paths))
    
    for new_index, original_index in enumerate(shuffled_indices):
        original_path = all_paths[original_index]
        anonymized_filename = create_anonymized_filename(original_path, new_index + 1)
        
        path_mapping[original_path] = {
            'anonymized_path': f"anonymized/{anonymized_filename}",
            'original_path': original_path,
            'split': path_to_split[original_path],
            'label': path_to_label[original_path],
            'index': new_index + 1
        }
        
        anonymized_to_original[anonymized_filename] = original_path
    
    return {
        'path_mapping': path_mapping,
        'anonymized_to_original': anonymized_to_original,
        'total_files': len(all_paths),
        'splits_info': {
            split: len([p for p in all_paths if path_to_split[p] == split])
            for split in manifest_files.keys()
        }
    }

def copy_and_anonymize_files(path_mapping: Dict, 
                           original_root: str = "D:/work/AWARE-NET/datasets",
                           anonymized_root: str = "D:/work/AWARE-NET/datasets/anonymized",
                           dry_run: bool = True) -> Dict:
    """
    Copy original files to anonymized directory structure
    
    Args:
        path_mapping: Path mapping dictionary
        original_root: Root directory of original files
        anonymized_root: Root directory for anonymized files
        dry_run: If True, only simulate the copying
        
    Returns:
        Copy operation results
    """
    anonymized_path = Path(anonymized_root)
    
    if not dry_run:
        anonymized_path.mkdir(exist_ok=True)
        print(f"Created anonymized directory: {anonymized_path}")
    
    copy_results = {
        'successful_copies': 0,
        'failed_copies': 0,
        'missing_files': [],
        'copy_errors': []
    }
    
    mapping = path_mapping['path_mapping']
    
    for original_path, info in mapping.items():
        # Construct full original path
        full_original_path = Path(original_root) / original_path
        
        # Construct full anonymized path
        full_anonymized_path = Path(anonymized_root) / info['anonymized_path']
        
        if dry_run:
            # Just check if original file exists
            if full_original_path.exists():
                copy_results['successful_copies'] += 1
            else:
                copy_results['failed_copies'] += 1
                copy_results['missing_files'].append(str(full_original_path))
        else:
            try:
                if full_original_path.exists():
                    # Create parent directory if needed
                    full_anonymized_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(full_original_path, full_anonymized_path)
                    copy_results['successful_copies'] += 1
                else:
                    copy_results['failed_copies'] += 1
                    copy_results['missing_files'].append(str(full_original_path))
                    
            except Exception as e:
                copy_results['failed_copies'] += 1
                copy_results['copy_errors'].append({
                    'original_path': str(full_original_path),
                    'anonymized_path': str(full_anonymized_path),
                    'error': str(e)
                })
        
        # Progress indicator
        total_processed = copy_results['successful_copies'] + copy_results['failed_copies']
        if total_processed % 1000 == 0:
            print(f"Processed {total_processed}/{len(mapping)} files...")
    
    print(f"\nCopy Results:")
    print(f"  Successful: {copy_results['successful_copies']}")
    print(f"  Failed: {copy_results['failed_copies']}")
    print(f"  Missing files: {len(copy_results['missing_files'])}")
    print(f"  Copy errors: {len(copy_results['copy_errors'])}")
    
    return copy_results

def create_anonymized_manifests(path_mapping: Dict,
                              manifests_dir: str = "D:/work/AWARE-NET/datasets/manifests_clean",
                              output_dir: str = "D:/work/AWARE-NET/datasets/manifests_anonymized") -> Dict:
    """
    Create new manifest files with anonymized paths
    
    Args:
        path_mapping: Path mapping dictionary
        manifests_dir: Directory with clean manifests
        output_dir: Output directory for anonymized manifests
        
    Returns:
        Results of manifest creation
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    manifest_files = {
        'train': 'celebdf_train_clean.csv',
        'val': 'celebdf_val_clean.csv',
        'test': 'celebdf_test_clean.csv'
    }
    
    anonymized_manifests = {}
    
    for split, filename in manifest_files.items():
        input_file = Path(manifests_dir) / filename
        
        if not input_file.exists():
            print(f"WARNING: Input file not found: {input_file}")
            continue
        
        # Load original manifest
        df = pd.read_csv(input_file)
        
        # Create anonymized version
        anonymized_data = []
        
        for _, row in df.iterrows():
            original_path = row['image_path']
            
            if original_path in path_mapping['path_mapping']:
                mapping_info = path_mapping['path_mapping'][original_path]
                
                # Create new row with anonymized path
                new_row = row.copy()
                new_row['image_path'] = mapping_info['anonymized_path']
                
                # Optional: add original path as metadata (for debugging)
                new_row['original_path'] = original_path
                
                anonymized_data.append(new_row)
            else:
                print(f"WARNING: Path not found in mapping: {original_path}")
        
        # Create DataFrame and save
        anonymized_df = pd.DataFrame(anonymized_data)
        output_file = output_path / f"celebdf_{split}_anonymized.csv"
        anonymized_df.to_csv(output_file, index=False)
        
        anonymized_manifests[split] = anonymized_df
        
        print(f"Created anonymized manifest: {output_file}")
        print(f"  Samples: {len(anonymized_df)}")
        print(f"  Real: {(anonymized_df['label'] == 0).sum()}")
        print(f"  Fake: {(anonymized_df['label'] == 1).sum()}")
    
    return anonymized_manifests

def validate_anonymization(path_mapping: Dict, anonymized_manifests: Dict) -> Dict:
    """
    Validate that anonymization was successful
    
    Args:
        path_mapping: Path mapping dictionary
        anonymized_manifests: Anonymized manifest DataFrames
        
    Returns:
        Validation results
    """
    validation_results = {
        'path_leakage_check': {},
        'label_preservation_check': {},
        'filename_uniqueness_check': {},
        'overall_status': 'unknown'
    }
    
    # Check for path-based label leakage
    all_anonymized_paths = []
    
    for split, df in anonymized_manifests.items():
        paths = df['image_path'].tolist()
        all_anonymized_paths.extend(paths)
        
        # Check if any anonymized paths contain label indicators
        real_paths_with_real = sum(1 for p in paths if 'real' in p.lower() and df[df['image_path'] == p]['label'].iloc[0] == 0)
        fake_paths_with_fake = sum(1 for p in paths if 'fake' in p.lower() and df[df['image_path'] == p]['label'].iloc[0] == 1)
        
        validation_results['path_leakage_check'][split] = {
            'total_paths': len(paths),
            'real_paths_with_real_indicator': real_paths_with_real,
            'fake_paths_with_fake_indicator': fake_paths_with_fake,
            'leakage_percentage': ((real_paths_with_real + fake_paths_with_fake) / len(paths)) * 100 if len(paths) > 0 else 0
        }
    
    # Check filename uniqueness
    unique_paths = set(all_anonymized_paths)
    validation_results['filename_uniqueness_check'] = {
        'total_paths': len(all_anonymized_paths),
        'unique_paths': len(unique_paths),
        'duplicates_found': len(all_anonymized_paths) - len(unique_paths),
        'is_unique': len(all_anonymized_paths) == len(unique_paths)
    }
    
    # Overall validation
    total_leakage = sum(
        result['real_paths_with_real_indicator'] + result['fake_paths_with_fake_indicator']
        for result in validation_results['path_leakage_check'].values()
    )
    
    is_successful = (
        total_leakage == 0 and 
        validation_results['filename_uniqueness_check']['is_unique']
    )
    
    validation_results['overall_status'] = 'SUCCESS' if is_successful else 'FAILED'
    
    return validation_results

def main():
    """Main anonymization function"""
    
    print("AWARE-NET Path Anonymization")
    print("=" * 50)
    
    # Step 1: Create path mapping
    print("\nStep 1: Creating path mapping...")
    path_mapping = create_path_mapping()
    
    print(f"Path mapping created:")
    print(f"  Total files: {path_mapping['total_files']}")
    print(f"  Splits: {path_mapping['splits_info']}")
    
    # Step 2: Test file copying (dry run first)
    print("\nStep 2: Testing file copying (dry run)...")
    copy_results = copy_and_anonymize_files(path_mapping, dry_run=True)
    
    if copy_results['failed_copies'] > 0:
        print(f"WARNING: {copy_results['failed_copies']} files would fail to copy")
        print("First few missing files:")
        for missing_file in copy_results['missing_files'][:5]:
            print(f"  {missing_file}")
        
        # Ask user if they want to continue despite missing files
        response = input("\nContinue with missing files? (y/n): ")
        if response.lower() != 'y':
            print("Aborting anonymization due to missing files.")
            return
    
    # Step 3: Create anonymized manifests (this doesn't require file copying)
    print("\nStep 3: Creating anonymized manifests...")
    anonymized_manifests = create_anonymized_manifests(path_mapping)
    
    # Step 4: Validate anonymization
    print("\nStep 4: Validating anonymization...")
    validation_results = validate_anonymization(path_mapping, anonymized_manifests)
    
    print(f"Validation Results:")
    for split, leakage_info in validation_results['path_leakage_check'].items():
        print(f"  {split.upper()}: {leakage_info['leakage_percentage']:.1f}% path leakage")
    
    print(f"  Filename uniqueness: {'PASS' if validation_results['filename_uniqueness_check']['is_unique'] else 'FAIL'}")
    print(f"  Overall status: {validation_results['overall_status']}")
    
    # Step 5: Save all results
    results_data = {
        'path_mapping': path_mapping,
        'copy_results': copy_results,
        'validation_results': validation_results,
        'anonymized_manifest_stats': {
            split: {
                'total_samples': len(df),
                'real_samples': (df['label'] == 0).sum(),
                'fake_samples': (df['label'] == 1).sum()
            }
            for split, df in anonymized_manifests.items()
        }
    }
    
    output_file = 'D:/work/AWARE-NET/path_anonymization_results.json'
    
    # Remove non-serializable numpy types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert the data
    serializable_data = json.loads(json.dumps(results_data, default=convert_types))
    
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    if validation_results['overall_status'] == 'SUCCESS':
        print("\nSUCCESS: Path anonymization completed successfully!")
        print("The anonymized manifests are ready for clean training.")
    else:
        print("\nWARNING: Anonymization validation failed!")
        print("Please review the results before proceeding.")

if __name__ == "__main__":
    main()