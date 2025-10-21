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
    if "DeeperForensics" in str(directory) or "DFDC" in str(directory):
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

def split_data_video_level(
    entries: List[dict],
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    balanced=False,
) -> dict:
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
    split_video_ids = {
        "train": {0: [], 1: []},
        "val": {0: [], 1: []},
        "test": {0: [], 1: []},
    }

    for video_dict, label_value in [(real_videos, 0), (fake_videos, 1)]:
        video_ids = list(video_dict.keys())
        random.shuffle(video_ids)

        n = len(video_ids)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_video_ids = video_ids[:train_end]
        val_video_ids = video_ids[train_end:val_end]
        test_video_ids = video_ids[val_end:]

        label_name = "real" if label_value == 0 else "fake"
        print(f"  {label_name} - train: {len(train_video_ids)} videos, "
              f"val: {len(val_video_ids)} videos, test: {len(test_video_ids)} videos")

        split_video_ids["train"][label_value] = train_video_ids
        split_video_ids["val"][label_value] = val_video_ids
        split_video_ids["test"][label_value] = test_video_ids

    if balanced:
        print("\n⚖️  Applying class-balanced video sampling (per split)...")
        for split_name, class_dict in split_video_ids.items():
            real_ids = class_dict.get(0, [])
            fake_ids = class_dict.get(1, [])
            if not real_ids or not fake_ids:
                continue
            target = min(len(real_ids), len(fake_ids))
            if target == 0:
                continue
            class_dict[0] = real_ids[:target]
            class_dict[1] = fake_ids[:target]
            print(f"  {split_name}: retaining {target} videos per class")

    # Step 4: Collect ALL frames from selected videos
    for split_name, class_dict in split_video_ids.items():
        for label_value, video_ids in class_dict.items():
            source_dict = real_videos if label_value == 0 else fake_videos
            for vid in video_ids:
                if vid not in source_dict:
                    continue
                for frame in source_dict[vid]:
                    frame['split'] = split_name
                splits[split_name].extend(source_dict[vid])

    # Print split statistics
    print("\n✅ Split statistics:")
    for split_name, split_entries in splits.items():
        real_count = sum(1 for e in split_entries if e['label'] == 0)
        fake_count = sum(1 for e in split_entries if e['label'] == 1)
        print(f"  {split_name}: {len(split_entries)} samples ({real_count} real, {fake_count} fake)")

    return splits

# Alias for backward compatibility
split_data = split_data_video_level


def simple_balance_datasets(real_entries: List[dict], fake_entries: List[dict], seed: int = 42) -> List[dict]:
    """
    Simple balancing: Use ALL Real samples + equal number of Fake samples.

    Args:
        real_entries: List of real image entries
        fake_entries: List of fake image entries
        seed: Random seed for reproducibility

    Returns:
        Balanced list with 50:50 Real:Fake ratio
    """
    random.seed(seed)

    # Use all Real samples
    selected_real = real_entries.copy()

    # Randomly sample equal number of Fake samples
    if len(fake_entries) >= len(real_entries):
        random.shuffle(fake_entries)
        selected_fake = fake_entries[:len(real_entries)]
    else:
        # Not enough Fake samples, use all available
        selected_fake = fake_entries.copy()
        print(f"Warning: Not enough Fake samples ({len(fake_entries)}) to match Real samples ({len(real_entries)})")

    # Combine and shuffle
    balanced_entries = selected_real + selected_fake
    random.shuffle(balanced_entries)

    print(f"Simple balancing: {len(selected_real)} Real + {len(selected_fake)} Fake = {len(balanced_entries)} total")

    return balanced_entries



def save_manifest(entries: List[dict], output_path: Path):
    """Save manifest to CSV file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(entries)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(entries)} entries to {output_path}")

def main():
    """Generate manifests for all datasets with simple balancing"""
    import sys

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].lower()
    else:
        dataset_name = "celebdf"  # Default

    print(f"=== {dataset_name.upper()} Simple Manifest Generation ===")

    # Dataset paths configuration
    if dataset_name == "celebdf":
        real_dir = Path("dataset/real/CelebDF-v2")
        fake_dir = Path("dataset/fake/CelebDF-v2")
        config_path = Path("configs/datasets.json")
        manifest_prefix = "celebdf_v2"
    elif dataset_name == "faceforensics" or dataset_name == "ff++":
        real_dir = Path("dataset/real/FF++")
        fake_dir = Path("dataset/fake/FF++")
        config_path = Path("configs/faceforensics_config.json")
        manifest_prefix = "faceforensics"
    elif dataset_name == "deeperforensics":
        real_dir = Path("dataset/real/DeeperForensics-1.0")
        fake_dir = Path("dataset/fake/DeeperForensics-1.0")
        config_path = Path("configs/deeperforensics_config.json")
        manifest_prefix = "deeperforensics"
    elif dataset_name == "dfdc":
        real_dir = Path("dataset/real/DFDC")
        fake_dir = Path("dataset/fake/DFDC")
        config_path = Path("configs/datasets.json")
        manifest_prefix = "dfdc"
    else:
        print(f"Unknown dataset: {dataset_name}")
        print("Available: celebdf, faceforensics, deeperforensics, dfdc")
        return

    manifests_dir = Path("manifests")

    # Scan directories
    print("\\nScanning real images...")
    real_entries = scan_celebdf_directory(real_dir, label=0)

    print("\\nScanning fake images...")
    fake_entries = scan_celebdf_directory(fake_dir, label=1)

    if not real_entries and not fake_entries:
        print("No images found! Check directory paths.")
        return

    print(f"\\nFound {len(real_entries)} Real images, {len(fake_entries)} Fake images")

    # Apply simple balancing: ALL Real samples + equal Fake samples
    print("\\nApplying simple balancing (All Real + Equal Fake)...")
    balanced_entries = simple_balance_datasets(real_entries, fake_entries, seed=42)

    # Split balanced data at video level to prevent leakage
    print("\\nSplitting balanced data at video level...")
    splits = split_data_video_level(balanced_entries, seed=42, balanced=False)

    # Save manifests
    print("\\nSaving balanced manifest files...")
    for split_name, split_entries in splits.items():
        manifest_path = manifests_dir / f"{manifest_prefix}_{split_name}_balanced.csv"
        save_manifest(split_entries, manifest_path)

    # Print final statistics
    print("\\n=== Final Statistics ===")
    for split_name, split_entries in splits.items():
        real_count = sum(1 for e in split_entries if e['label'] == 0)
        fake_count = sum(1 for e in split_entries if e['label'] == 1)
        total_count = len(split_entries)
        real_pct = real_count / total_count * 100 if total_count > 0 else 0
        fake_pct = fake_count / total_count * 100 if total_count > 0 else 0
        print(f"{split_name}: {total_count} samples ({real_count} Real, {fake_count} Fake) - {real_pct:.1f}% Real, {fake_pct:.1f}% Fake")

    # Update config with statistics
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        config['metadata'].update({
            'total_samples': len(balanced_entries),
            'real_samples': len(real_entries),
            'fake_samples': min(len(fake_entries), len(real_entries)),
            'balancing_method': 'simple_all_real_equal_fake',
            'updated_at': pd.Timestamp.now().isoformat()
        })

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\\nUpdated config with simple balancing statistics")

    print("\\n=== Simple Manifest Generation Complete ===")

if __name__ == "__main__":
    main()
