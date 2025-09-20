#!/usr/bin/env python3
"""
Data Distribution Analysis for AWARE-NET
Analyze manifest files to identify critical issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json

def analyze_manifest_file(manifest_path: str) -> Dict[str, Any]:
    """
    Analyze a single manifest file
    
    Args:
        manifest_path: Path to CSV manifest file
        
    Returns:
        Dictionary with analysis results
    """
    df = pd.read_csv(manifest_path)
    
    # Basic statistics
    total_samples = len(df)
    real_count = (df['label'] == 0).sum()
    fake_count = (df['label'] == 1).sum()
    
    # Calculate percentages
    real_pct = (real_count / total_samples) * 100 if total_samples > 0 else 0
    fake_pct = (fake_count / total_samples) * 100 if total_samples > 0 else 0
    
    # Check for data quality issues
    missing_labels = df['label'].isna().sum()
    invalid_labels = (~df['label'].isin([0, 1])).sum()
    
    # Extract subject IDs to check for overlap
    subjects = set()
    for path in df['image_path']:
        # Extract subject ID from path patterns like "id56_0000" or "00291"
        parts = Path(path).parts
        for part in parts:
            if 'id' in part or part.isdigit():
                subjects.add(part)
                break
    
    return {
        'total_samples': total_samples,
        'real_count': real_count,
        'fake_count': fake_count,
        'real_percentage': real_pct,
        'fake_percentage': fake_pct,
        'class_ratio': fake_count / real_count if real_count > 0 else float('inf'),
        'missing_labels': missing_labels,
        'invalid_labels': invalid_labels,
        'unique_subjects': len(subjects),
        'subjects': subjects,
        'sample_paths': df['image_path'].head(5).tolist()
    }

def check_subject_overlap(train_subjects: set, val_subjects: set, test_subjects: set) -> Dict[str, Any]:
    """
    Check for subject overlap between splits
    
    Returns:
        Dictionary with overlap analysis
    """
    train_val_overlap = train_subjects & val_subjects
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects
    
    total_unique = len(train_subjects | val_subjects | test_subjects)
    
    return {
        'train_val_overlap': len(train_val_overlap),
        'train_test_overlap': len(train_test_overlap),
        'val_test_overlap': len(val_test_overlap),
        'train_val_overlap_subjects': list(train_val_overlap)[:10],  # Show first 10
        'train_test_overlap_subjects': list(train_test_overlap)[:10],
        'val_test_overlap_subjects': list(val_test_overlap)[:10],
        'total_unique_subjects': total_unique,
        'has_leakage': len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
    }

def analyze_path_patterns(manifest_files: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyze path patterns to check for potential data leakage
    
    Args:
        manifest_files: Dict mapping split names to file paths
        
    Returns:
        Path pattern analysis
    """
    patterns = {}
    
    for split, file_path in manifest_files.items():
        df = pd.read_csv(file_path)
        
        # Check if paths contain label information
        real_paths = df[df['label'] == 0]['image_path']
        fake_paths = df[df['label'] == 1]['image_path']
        
        real_contains_real = real_paths.str.contains('real').sum()
        fake_contains_fake = fake_paths.str.contains('fake').sum()
        
        patterns[split] = {
            'real_paths_contain_real': real_contains_real,
            'fake_paths_contain_fake': fake_contains_fake,
            'total_real': len(real_paths),
            'total_fake': len(fake_paths),
            'path_leakage_risk': (real_contains_real / len(real_paths)) if len(real_paths) > 0 else 0
        }
    
    return patterns

def main():
    """Main analysis function"""
    
    # Define manifest file paths
    manifest_files = {
        'train': 'D:/work/AWARE-NET/datasets/manifests/celebdf_train.csv',
        'val': 'D:/work/AWARE-NET/datasets/manifests/celebdf_val.csv',
        'test': 'D:/work/AWARE-NET/datasets/manifests/celebdf_test.csv'
    }
    
    print("AWARE-NET Data Distribution Analysis")
    print("=" * 60)
    
    # Analyze each split
    results = {}
    for split, file_path in manifest_files.items():
        print(f"\nAnalyzing {split.upper()} split:")
        print("-" * 30)
        
        if not Path(file_path).exists():
            print(f"ERROR: File not found: {file_path}")
            continue
            
        analysis = analyze_manifest_file(file_path)
        results[split] = analysis
        
        print(f"Total samples: {analysis['total_samples']:,}")
        print(f"Real samples (0): {analysis['real_count']:,} ({analysis['real_percentage']:.1f}%)")
        print(f"Fake samples (1): {analysis['fake_count']:,} ({analysis['fake_percentage']:.1f}%)")
        print(f"Class ratio (fake:real): {analysis['class_ratio']:.2f}:1")
        print(f"Unique subjects: {analysis['unique_subjects']}")
        
        if analysis['missing_labels'] > 0:
            print(f"WARNING: Missing labels: {analysis['missing_labels']}")
        if analysis['invalid_labels'] > 0:
            print(f"WARNING: Invalid labels: {analysis['invalid_labels']}")
            
        # Show sample paths
        print(f"Sample paths:")
        for i, path in enumerate(analysis['sample_paths'], 1):
            print(f"  {i}. {path}")
    
    # Check for subject overlap
    if len(results) >= 2:
        print(f"\nSubject Overlap Analysis:")
        print("-" * 30)
        
        train_subjects = results.get('train', {}).get('subjects', set())
        val_subjects = results.get('val', {}).get('subjects', set())
        test_subjects = results.get('test', {}).get('subjects', set())
        
        overlap_analysis = check_subject_overlap(train_subjects, val_subjects, test_subjects)
        
        if overlap_analysis['has_leakage']:
            print("CRITICAL: DATA LEAKAGE DETECTED!")
            print(f"Train-Val overlap: {overlap_analysis['train_val_overlap']} subjects")
            print(f"Train-Test overlap: {overlap_analysis['train_test_overlap']} subjects")
            print(f"Val-Test overlap: {overlap_analysis['val_test_overlap']} subjects")
        else:
            print("GOOD: No subject overlap detected")
        
        print(f"Total unique subjects across all splits: {overlap_analysis['total_unique_subjects']}")
    
    # Analyze path patterns
    print(f"\nPath Pattern Analysis:")
    print("-" * 30)
    
    path_patterns = analyze_path_patterns(manifest_files)
    for split, patterns in path_patterns.items():
        print(f"\n{split.upper()} split:")
        if patterns['total_real'] > 0:
            leakage_pct = (patterns['real_paths_contain_real'] / patterns['total_real']) * 100
            print(f"  Real paths containing 'real': {patterns['real_paths_contain_real']}/{patterns['total_real']} ({leakage_pct:.1f}%)")
        if patterns['total_fake'] > 0:
            leakage_pct = (patterns['fake_paths_contain_fake'] / patterns['total_fake']) * 100
            print(f"  Fake paths containing 'fake': {patterns['fake_paths_contain_fake']}/{patterns['total_fake']} ({leakage_pct:.1f}%)")
    
    # Overall assessment
    print(f"\nCritical Issues Summary:")
    print("-" * 30)
    
    issues = []
    
    # Check for extreme imbalance
    for split, analysis in results.items():
        if analysis['class_ratio'] > 5 or analysis['class_ratio'] < 0.2:
            issues.append(f"ISSUE: Extreme class imbalance in {split}: {analysis['class_ratio']:.1f}:1")
    
    # Check for path leakage
    total_path_leakage = sum(p.get('path_leakage_risk', 0) for p in path_patterns.values())
    if total_path_leakage > 0.8:  # If >80% of paths contain label info
        issues.append("ISSUE: High risk of path-based label leakage")
    
    # Check for subject leakage
    if 'has_leakage' in locals() and overlap_analysis['has_leakage']:
        issues.append("ISSUE: Subject overlap between splits (data leakage)")
    
    if issues:
        print("CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nCRITICAL: Training results are likely INVALID due to these issues!")
    else:
        print("GOOD: No critical issues detected")
    
    # Save detailed results
    output_file = 'D:/work/AWARE-NET/data_analysis_results.json'
    detailed_results = {
        'split_analysis': results,
        'overlap_analysis': overlap_analysis if 'overlap_analysis' in locals() else {},
        'path_patterns': path_patterns,
        'issues': issues,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        # Convert sets to lists for JSON serialization
        for split, analysis in detailed_results['split_analysis'].items():
            if 'subjects' in analysis:
                analysis['subjects'] = list(analysis['subjects'])
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()