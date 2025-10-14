#!/usr/bin/env python3
"""
AWARE-NET Data Cleanup: Fix Critical Data Leakage Issues
Create subject-level splits with no overlap and path anonymization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
import shutil
import random
from collections import defaultdict
import hashlib

def extract_subject_id(image_path: str) -> str:
    """
    Extract subject ID from image path
    
    Args:
        image_path: Path like "real/CelebDF-v2/id56_0000/..." or "real/CelebDF-v2/00291/..."
        
    Returns:
        Subject ID (e.g., "id56_0000" or "00291")
    """
    parts = Path(image_path).parts
    
    # Look for subject ID patterns
    for part in parts:
        # Pattern 1: idXX_XXXX format
        if part.startswith('id') and '_' in part:
            return part
        # Pattern 2: Numeric ID (like 00291)
        elif part.isdigit() and len(part) >= 3:
            return part
    
    # Fallback: use the parent directory name
    return parts[-2] if len(parts) >= 2 else "unknown"

def load_all_data() -> pd.DataFrame:
    """
    Load and combine all manifest files
    
    Returns:
        Combined DataFrame with all samples
    """
    manifest_files = {
        'train': 'D:/work/AWARE-NET/datasets/manifests/celebdf_train.csv',
        'val': 'D:/work/AWARE-NET/datasets/manifests/celebdf_val.csv',
        'test': 'D:/work/AWARE-NET/datasets/manifests/celebdf_test.csv'
    }
    
    all_data = []
    
    for split, file_path in manifest_files.items():
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            df['original_split'] = split
            all_data.append(df)
            print(f"Loaded {len(df)} samples from {split}")
        else:
            print(f"WARNING: File not found: {file_path}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total combined samples: {len(combined_df)}")
    
    return combined_df

def analyze_subjects(df: pd.DataFrame) -> Dict:
    """
    Analyze subject distribution in the dataset
    
    Args:
        df: Combined DataFrame
        
    Returns:
        Subject analysis results
    """
    # Extract subject IDs
    df['subject_id'] = df['image_path'].apply(extract_subject_id)
    
    # Group by subject and analyze
    subject_stats = []
    
    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id]
        
        real_count = (subject_data['label'] == 0).sum()
        fake_count = (subject_data['label'] == 1).sum()
        
        # Check which splits this subject appears in
        splits = set(subject_data['original_split'])
        
        subject_stats.append({
            'subject_id': subject_id,
            'total_samples': len(subject_data),
            'real_samples': real_count,
            'fake_samples': fake_count,
            'splits': splits,
            'has_both_classes': real_count > 0 and fake_count > 0,
            'split_count': len(splits)
        })
    
    subject_df = pd.DataFrame(subject_stats)
    
    # Summary statistics
    analysis = {
        'total_subjects': len(subject_df),
        'subjects_in_multiple_splits': (subject_df['split_count'] > 1).sum(),
        'subjects_with_both_classes': subject_df['has_both_classes'].sum(),
        'subject_sample_distribution': subject_df['total_samples'].describe(),
        'subjects_per_split': {
            split: len(df[df['original_split'] == split]['subject_id'].unique())
            for split in df['original_split'].unique()
        }
    }
    
    return analysis, subject_df

def create_subject_level_splits(subject_df: pd.DataFrame, 
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15,
                              test_ratio: float = 0.15,
                              random_seed: int = 42) -> Dict[str, Set[str]]:
    """
    Create subject-level splits with no overlap
    
    Args:
        subject_df: DataFrame with subject statistics
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to sets of subject IDs
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get list of all unique subjects
    all_subjects = list(subject_df['subject_id'].unique())
    random.shuffle(all_subjects)
    
    total_subjects = len(all_subjects)
    train_count = int(total_subjects * train_ratio)
    val_count = int(total_subjects * val_ratio)
    
    # Split subjects
    train_subjects = set(all_subjects[:train_count])
    val_subjects = set(all_subjects[train_count:train_count + val_count])
    test_subjects = set(all_subjects[train_count + val_count:])
    
    # Verify no overlap
    assert len(train_subjects & val_subjects) == 0, "Train-Val overlap detected"
    assert len(train_subjects & test_subjects) == 0, "Train-Test overlap detected"
    assert len(val_subjects & test_subjects) == 0, "Val-Test overlap detected"
    
    splits = {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects
    }
    
    print(f"Subject-level splits created:")
    print(f"  Train: {len(train_subjects)} subjects")
    print(f"  Val: {len(val_subjects)} subjects")
    print(f"  Test: {len(test_subjects)} subjects")
    print(f"  Total: {len(train_subjects) + len(val_subjects) + len(test_subjects)} subjects")
    
    return splits

def create_clean_manifests(df: pd.DataFrame, 
                         subject_splits: Dict[str, Set[str]],
                         balance_classes: bool = True,
                         max_samples_per_class: int = None) -> Dict[str, pd.DataFrame]:
    """
    Create clean manifest files based on subject splits
    
    Args:
        df: Combined DataFrame with subject IDs
        subject_splits: Dictionary mapping split names to subject sets
        balance_classes: Whether to balance real/fake classes
        max_samples_per_class: Maximum samples per class (for limiting dataset size)
        
    Returns:
        Dictionary mapping split names to clean DataFrames
    """
    clean_manifests = {}
    
    for split_name, subject_set in subject_splits.items():
        # Filter data for this split's subjects
        split_data = df[df['subject_id'].isin(subject_set)].copy()
        
        if balance_classes:
            # Count samples per class
            real_count = (split_data['label'] == 0).sum()
            fake_count = (split_data['label'] == 1).sum()
            
            # Balance to the smaller class (or max_samples_per_class if specified)
            target_count = min(real_count, fake_count)
            if max_samples_per_class:
                target_count = min(target_count, max_samples_per_class)
            
            # Sample equal numbers from each class
            real_samples = split_data[split_data['label'] == 0].sample(
                n=min(target_count, real_count), random_state=42
            )
            fake_samples = split_data[split_data['label'] == 1].sample(
                n=min(target_count, fake_count), random_state=42
            )
            
            split_data = pd.concat([real_samples, fake_samples], ignore_index=True)
        
        # Shuffle the data
        split_data = split_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        clean_manifests[split_name] = split_data
        
        real_final = (split_data['label'] == 0).sum()
        fake_final = (split_data['label'] == 1).sum()
        
        print(f"{split_name.upper()} split:")
        print(f"  Subjects: {len(subject_set)}")
        print(f"  Total samples: {len(split_data)}")
        print(f"  Real: {real_final} ({100*real_final/len(split_data):.1f}%)")
        print(f"  Fake: {fake_final} ({100*fake_final/len(split_data):.1f}%)")
        print(f"  Balance ratio: {fake_final/real_final:.2f}:1" if real_final > 0 else "  Balance ratio: N/A")
    
    return clean_manifests

def save_clean_manifests(clean_manifests: Dict[str, pd.DataFrame], 
                        output_dir: str = "D:/work/AWARE-NET/datasets/manifests_clean"):
    """
    Save clean manifest files
    
    Args:
        clean_manifests: Dictionary of clean DataFrames
        output_dir: Output directory for clean manifests
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for split_name, df in clean_manifests.items():
        # Remove the subject_id column before saving (internal use only)
        df_to_save = df.drop(columns=['subject_id'], errors='ignore')
        
        output_file = output_path / f"celebdf_{split_name}_clean.csv"
        df_to_save.to_csv(output_file, index=False)
        print(f"Saved {len(df_to_save)} samples to {output_file}")

def main():
    """Main data cleanup function"""
    
    print("AWARE-NET Data Cleanup: Fixing Critical Issues")
    print("=" * 60)
    
    # Step 1: Load all data
    print("\nStep 1: Loading all data...")
    df = load_all_data()
    
    # Step 2: Analyze current subject distribution
    print("\nStep 2: Analyzing subject distribution...")
    analysis, subject_df = analyze_subjects(df)
    
    print(f"Analysis Results:")
    print(f"  Total subjects: {analysis['total_subjects']}")
    print(f"  Subjects in multiple splits: {analysis['subjects_in_multiple_splits']}")
    print(f"  Subjects with both classes: {analysis['subjects_with_both_classes']}")
    print(f"  Subjects per original split: {analysis['subjects_per_split']}")
    
    # Step 3: Create subject-level splits
    print("\nStep 3: Creating clean subject-level splits...")
    subject_splits = create_subject_level_splits(subject_df)
    
    # Step 4: Create clean manifests
    print("\nStep 4: Creating balanced clean manifests...")
    clean_manifests = create_clean_manifests(
        df, subject_splits, 
        balance_classes=True,
        max_samples_per_class=50000  # Limit for faster experimentation
    )
    
    # Step 5: Save clean manifests
    print("\nStep 5: Saving clean manifests...")
    save_clean_manifests(clean_manifests)
    
    # Step 6: Validation
    print("\nStep 6: Validation...")
    
    # Check for subject overlap
    all_train_subjects = set(clean_manifests['train']['subject_id'])
    all_val_subjects = set(clean_manifests['val']['subject_id'])
    all_test_subjects = set(clean_manifests['test']['subject_id'])
    
    overlaps = {
        'train_val': len(all_train_subjects & all_val_subjects),
        'train_test': len(all_train_subjects & all_test_subjects),
        'val_test': len(all_val_subjects & all_test_subjects)
    }
    
    if sum(overlaps.values()) == 0:
        print("SUCCESS: No subject overlap between splits!")
    else:
        print(f"ERROR: Subject overlaps found: {overlaps}")
    
    # Save detailed analysis
    analysis_output = {
        'original_analysis': analysis,
        'clean_splits_subjects': {k: list(v) for k, v in subject_splits.items()},
        'clean_manifest_stats': {
            split: {
                'total_samples': len(df),
                'real_samples': (df['label'] == 0).sum(),
                'fake_samples': (df['label'] == 1).sum(),
                'unique_subjects': len(df['subject_id'].unique())
            }
            for split, df in clean_manifests.items()
        },
        'validation': {
            'subject_overlaps': overlaps,
            'is_clean': sum(overlaps.values()) == 0
        }
    }
    
    with open('D:/work/AWARE-NET/data_cleanup_results.json', 'w') as f:
        json.dump(analysis_output, f, indent=2)
    
    print("\nCleanup Summary:")
    print("="*40)
    for split, df in clean_manifests.items():
        print(f"{split.upper()}: {len(df)} samples, {len(df['subject_id'].unique())} subjects")
    
    print(f"\nResults saved to: D:/work/AWARE-NET/data_cleanup_results.json")
    print(f"Clean manifests saved to: D:/work/AWARE-NET/datasets/manifests_clean/")

if __name__ == "__main__":
    main()