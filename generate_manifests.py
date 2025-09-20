#!/usr/bin/env python3
"""
Quick script to generate CelebDF-v2 manifests for AWARE-NET
"""

import os
import csv
import json
import random
import hashlib
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""

def scan_celebdf_directory(directory: Path, label: int) -> List[dict]:
    """
    Scan directory and collect all image files (works for CelebDF-v2, FF++, DeeperForensics)
    
    Args:
        directory: Directory to scan
        label: Label for images (0=real, 1=fake)
        
    Returns:
        List of image entries
    """
    print(f"Scanning {'real' if label == 0 else 'fake'} images in: {directory}")
    
    entries = []
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return entries
    
    # Get all video directories (including nested ones for DeeperForensics)
    video_dirs = []
    
    # Check if it's DeeperForensics structure (has subdirectories like end_to_end)
    if "DeeperForensics" in str(directory):
        # For DeeperForensics, scan recursively for any directory containing JPG files
        for root in directory.rglob("*"):
            if root.is_dir() and any(f.suffix.lower() == '.jpg' for f in root.iterdir() if f.is_file()):
                video_dirs.append(root)
    else:
        # For CelebDF-v2 and FF++, use direct subdirectories
        video_dirs = [d for d in directory.iterdir() if d.is_dir()]
    
    print(f"Found {len(video_dirs)} video directories")
    
    for video_dir in tqdm(video_dirs, desc=f"Processing {'real' if label == 0 else 'fake'} videos"):
        # Get all JPG files in this video directory
        jpg_files = list(video_dir.glob("*.jpg"))
        
        for jpg_file in jpg_files:
            # Convert to relative path from project root
            try:
                rel_path = jpg_file.relative_to(Path.cwd())
            except ValueError:
                # If relative_to fails, use the path as is
                rel_path = jpg_file
            
            entry = {
                'image_path': str(rel_path).replace('\\', '/'),
                'label': label,
                'split': '',  # Will be assigned later
                'md5': '',  # Skip MD5 for speed
                'valid': True,
                'error': '',
                'width': 0,
                'height': 0,
                'file_size': jpg_file.stat().st_size
            }
            
            entries.append(entry)
    
    print(f"Found {len(entries)} images")
    return entries

def split_data(entries: List[dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42) -> dict:
    """Split data into train/val/test sets with stratification"""
    
    # Set random seed
    random.seed(seed)
    
    # Separate by label
    real_entries = [e for e in entries if e['label'] == 0]
    fake_entries = [e for e in entries if e['label'] == 1]
    
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
            entry['split'] = "train"
        for entry in val_split:
            entry['split'] = "val"
        for entry in test_split:
            entry['split'] = "test"
        
        splits["train"].extend(train_split)
        splits["val"].extend(val_split)
        splits["test"].extend(test_split)
    
    # Print split statistics
    for split_name, split_entries in splits.items():
        real_count = sum(1 for e in split_entries if e['label'] == 0)
        fake_count = sum(1 for e in split_entries if e['label'] == 1)
        print(f"{split_name}: {len(split_entries)} samples ({real_count} real, {fake_count} fake)")
    
    return splits

def save_manifest(entries: List[dict], output_path: Path):
    """Save manifest to CSV file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(entries)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(entries)} entries to {output_path}")

def main():
    """Generate manifests for all datasets"""
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].lower()
    else:
        dataset_name = "celebdf"  # Default
    
    print(f"=== {dataset_name.upper()} Manifest Generation ===")
    
    if dataset_name == "celebdf":
        # CelebDF-v2 paths
        real_dir = Path("dataset/real/CelebDF-v2")
        fake_dir = Path("dataset/fake/CelebDF-v2")
        config_path = Path("configs/dataset_paths.json")
        manifest_prefix = "celebdf_v2"
    elif dataset_name == "faceforensics" or dataset_name == "ff++":
        # FF++ paths
        real_dir = Path("dataset/real/FF++")
        fake_dir = Path("dataset/fake/FF++")
        config_path = Path("configs/faceforensics_config.json")
        manifest_prefix = "faceforensics"
    elif dataset_name == "deeperforensics":
        # DeeperForensics paths
        real_dir = Path("dataset/real/DeeperForensics-1.0")
        fake_dir = Path("dataset/fake/DeeperForensics-1.0")
        config_path = Path("configs/deeperforensics_config.json")
        manifest_prefix = "deeperforensics"
    else:
        print(f"Unknown dataset: {dataset_name}")
        print("Available: celebdf, faceforensics, deeperforensics")
        return
    
    manifests_dir = Path("manifests")
    
    # Scan directories
    print("\\nScanning real images...")
    real_entries = scan_celebdf_directory(real_dir, label=0)
    
    print("\\nScanning fake images...")
    fake_entries = scan_celebdf_directory(fake_dir, label=1)
    
    # Combine all entries
    all_entries = real_entries + fake_entries
    print(f"\\nTotal images: {len(all_entries)}")
    
    if not all_entries:
        print("No images found! Check directory paths.")
        return
    
    # Split data
    print("\\nSplitting data...")
    splits = split_data(all_entries)
    
    # Save manifests
    print("\\nSaving manifest files...")
    for split_name, split_entries in splits.items():
        manifest_path = manifests_dir / f"{manifest_prefix}_{split_name}.csv"
        save_manifest(split_entries, manifest_path)
    
    # Update config with statistics
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        total_samples = len(all_entries)
        total_real = len(real_entries)
        total_fake = len(fake_entries)
        
        config['metadata'].update({
            'total_samples': total_samples,
            'real_samples': total_real,
            'fake_samples': total_fake,
            'updated_at': pd.Timestamp.now().isoformat()
        })
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\\nUpdated config with statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Real samples: {total_real}")
        print(f"  Fake samples: {total_fake}")
    
    print("\\n=== Manifest Generation Complete ===")

if __name__ == "__main__":
    main()