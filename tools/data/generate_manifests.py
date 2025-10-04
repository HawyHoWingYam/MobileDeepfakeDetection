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

# ============================================================================
# DEPRECATED: Frame-level split (DO NOT USE - causes data leakage!)
# ============================================================================
# def split_data_frame_level(entries: List[dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42) -> dict:
#     """
#     OLD IMPLEMENTATION - DEPRECATED
#
#     WARNING: This function splits frames randomly, causing severe data leakage!
#     Frames from the same video can appear in both train and val sets.
#     This leads to artificially inflated performance (AUC 0.99+).
#
#     Use split_data_video_level() instead!
#     """
#     random.seed(seed)
#     real_entries = [e for e in entries if e['label'] == 0]
#     fake_entries = [e for e in entries if e['label'] == 1]
#     random.shuffle(real_entries)  # ❌ WRONG: shuffles frames, not videos
#     random.shuffle(fake_entries)
#     ...
# ============================================================================

def split_data_video_level(entries: List[dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42) -> dict:
    """
    Split data into train/val/test sets at VIDEO LEVEL (fixes data leakage)

    This ensures all frames from the same video only appear in one split,
    preventing the model from learning video-specific features.

    Args:
        entries: List of image entries with paths and labels
        train_ratio: Proportion of videos for training (default 0.7)
        val_ratio: Proportion of videos for validation (default 0.15)
        test_ratio: Proportion of videos for testing (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with train/val/test splits
    """
    random.seed(seed)

    print("\n🎬 Grouping frames by video ID...")

    # Step 1: Group all frames by video ID
    video_groups = {}
    for entry in entries:
        # Extract video ID: use second-to-last path component
        # Works for all datasets (CelebDF, FF++, DeeperForensics)
        # Example: dataset/fake/CelebDF-v2/id39_id40_0000/xxx.jpg -> id39_id40_0000
        path_parts = entry['image_path'].split('/')
        video_id = path_parts[-2] if len(path_parts) >= 2 else None

        if video_id:
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(entry)

    print(f"Found {len(video_groups)} unique videos")

    # Step 2: Separate videos by label
    real_videos = {}
    fake_videos = {}

    for video_id, frames in video_groups.items():
        if not frames:
            continue
        label = frames[0]['label']  # All frames in a video have the same label
        if label == 0:
            real_videos[video_id] = frames
        else:
            fake_videos[video_id] = frames

    print(f"Real videos: {len(real_videos)}")
    print(f"Fake videos: {len(fake_videos)}")

    # Step 3: Shuffle and split VIDEO IDs (not frames!)
    print("\n📊 Splitting videos into train/val/test...")
    splits = {"train": [], "val": [], "test": []}

    for video_dict, label_name in [(real_videos, "real"), (fake_videos, "fake")]:
        video_ids = list(video_dict.keys())
        random.shuffle(video_ids)  # ✅ CORRECT: shuffle video IDs, not frames

        n = len(video_ids)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_video_ids = video_ids[:train_end]
        val_video_ids = video_ids[train_end:val_end]
        test_video_ids = video_ids[val_end:]

        print(f"  {label_name} - train: {len(train_video_ids)} videos, "
              f"val: {len(val_video_ids)} videos, test: {len(test_video_ids)} videos")

        # Step 4: Collect ALL frames from selected videos
        for vid in train_video_ids:
            for frame in video_dict[vid]:
                frame['split'] = 'train'
            splits["train"].extend(video_dict[vid])

        for vid in val_video_ids:
            for frame in video_dict[vid]:
                frame['split'] = 'val'
            splits["val"].extend(video_dict[vid])

        for vid in test_video_ids:
            for frame in video_dict[vid]:
                frame['split'] = 'test'
            splits["test"].extend(video_dict[vid])

    # Print split statistics
    print("\n✅ Split statistics:")
    for split_name, split_entries in splits.items():
        real_count = sum(1 for e in split_entries if e['label'] == 0)
        fake_count = sum(1 for e in split_entries if e['label'] == 1)
        print(f"  {split_name}: {len(split_entries)} samples ({real_count} real, {fake_count} fake)")

    return splits

# Alias for backward compatibility
split_data = split_data_video_level

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