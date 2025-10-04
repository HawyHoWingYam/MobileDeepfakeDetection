#!/usr/bin/env python3
"""
AWARE-NET Dataset Utilities - Unified Tool
Generate manifests, anonymize data, and create balanced datasets
"""

import os
import sys
import json
import hashlib
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

def generate_manifests():
    """Generate original manifests using the existing script"""
    print("🔄 Generating original manifests...")

    try:
        # Run manifest generation for CelebDF-v2
        os.system("PYTHONPATH=. python tools/data/generate_manifests.py celebdf --config configs/datasets.json")
        print("✅ CelebDF-v2 manifests generated")

        # Run manifest generation for FaceForensics++
        os.system("PYTHONPATH=. python tools/data/generate_manifests.py faceforensics --config configs/datasets.json")
        print("✅ FaceForensics++ manifests generated")

        # Run manifest generation for DeeperForensics-1.0
        os.system("PYTHONPATH=. python tools/data/generate_manifests.py deeperforensics --config configs/datasets.json")
        print("✅ DeeperForensics-1.0 manifests generated")

        return True
    except Exception as e:
        print(f"❌ Error generating manifests: {e}")
        return False

def anonymize_dataset(dataset_name: str, splits: List[str] = ['train', 'val', 'test']):
    """Anonymize a specific dataset"""
    print(f"🔒 Anonymizing {dataset_name}...")

    # Mapping for original path -> anonymous path
    mapping_table = {}
    counter = 0

    for split in splits:
        manifest_path = f"manifests/{dataset_name}_{split}.csv"

        if not Path(manifest_path).exists():
            print(f"⚠️  Manifest not found: {manifest_path}")
            continue

        # Read original manifest
        df = pd.read_csv(manifest_path)
        anonymized_paths = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Anonymizing {split}"):
            original_path = row['image_path']

            if original_path not in mapping_table:
                counter += 1
                # Generate anonymous filename
                hash_part = hashlib.md5(original_path.encode()).hexdigest()[:8]
                anonymous_id = f"img_{counter:06d}_{hash_part}.jpg"
                mapping_table[original_path] = anonymous_id

            anonymous_path = f"dataset/anonymized/{dataset_name}/images/{mapping_table[original_path]}"
            anonymized_paths.append(anonymous_path)

        # Create anonymized manifest
        df_anon = df.copy()
        df_anon['image_path'] = anonymized_paths

        output_path = f"manifests/{dataset_name}_{split}_anonymized.csv"
        df_anon.to_csv(output_path, index=False)
        print(f"✅ Created: {output_path}")

def create_balanced_dataset(dataset_name: str):
    """Create balanced version of any dataset using original paths"""
    print(f"⚖️  Creating balanced {dataset_name}...")

    for split in ['train', 'val', 'test']:
        # Use ORIGINAL manifest (not anonymized)
        original_manifest = f"manifests/{dataset_name}_{split}.csv"

        if not Path(original_manifest).exists():
            print(f"⚠️  Original manifest not found: {original_manifest}")
            continue

        df = pd.read_csv(original_manifest)
        real_samples = df[df['label'] == 0]
        fake_samples = df[df['label'] == 1]

        # Balance to minimum class size
        min_samples = min(len(real_samples), len(fake_samples))
        real_balanced = real_samples.sample(n=min_samples, random_state=42)
        fake_balanced = fake_samples.sample(n=min_samples, random_state=42)

        # Combine and shuffle
        balanced_df = pd.concat([real_balanced, fake_balanced])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save with original paths (not anonymized paths)
        output_path = f"manifests/{dataset_name}_{split}_balanced.csv"
        balanced_df.to_csv(output_path, index=False)
        print(f"✅ Created balanced: {output_path} ({len(balanced_df)} samples)")
        print(f"   Real: {min_samples:,} | Fake: {min_samples:,} (50/50 balanced)")

def check_path_leakage(manifest_path: str):
    """Check for potential path leakage in manifest"""
    df = pd.read_csv(manifest_path)
    leakage_indicators = ['real', 'fake', 'authentic', 'forged']

    leakage_count = 0
    for _, row in df.iterrows():
        path_lower = str(row['image_path']).lower()
        if any(indicator in path_lower for indicator in leakage_indicators):
            leakage_count += 1

    if leakage_count > 0:
        print(f"⚠️  Path leakage detected in {leakage_count}/{len(df)} samples")
        return False
    else:
        print(f"✅ No path leakage detected")
        return True

def check_video_overlap(dataset_name: str) -> bool:
    """
    Check for video ID overlap between train/val/test splits

    This is critical to detect data leakage: if the same video's frames
    appear in both train and val, the model can cheat by learning
    video-specific features (scene, person, lighting).

    Args:
        dataset_name: Dataset to check (celebdf_v2, faceforensics, deeperforensics)

    Returns:
        True if no overlap (clean), False if overlap found (data leakage)
    """
    print(f"\n🔍 Checking video ID overlap for {dataset_name}...")

    splits = ['train', 'val', 'test']
    video_sets = {}

    # Extract video IDs from each split
    for split in splits:
        manifest_path = f"manifests/{dataset_name}_{split}.csv"

        if not Path(manifest_path).exists():
            print(f"⚠️  Manifest not found: {manifest_path}")
            continue

        df = pd.read_csv(manifest_path)

        # Extract video ID (second-to-last path component)
        # Example: dataset/fake/CelebDF-v2/id39_id40_0000/xxx.jpg -> id39_id40_0000
        video_ids = set(df['image_path'].apply(
            lambda x: x.split('/')[-2] if len(x.split('/')) >= 2 else None
        ).dropna())

        video_sets[split] = video_ids
        print(f"  {split}: {len(video_ids)} unique videos, {len(df)} frames")

    # Check for overlaps between all pairs
    overlap_found = False
    total_overlaps = 0

    for i, split1 in enumerate(splits):
        if split1 not in video_sets:
            continue
        for split2 in splits[i+1:]:
            if split2 not in video_sets:
                continue

            overlap = video_sets[split1] & video_sets[split2]
            if overlap:
                print(f"\n❌ OVERLAP DETECTED: {split1} ↔ {split2}: {len(overlap)} videos")
                print(f"   Examples: {list(overlap)[:5]}")
                overlap_found = True
                total_overlaps += len(overlap)

    if not overlap_found:
        print(f"\n✅ No video ID overlap detected - splits are clean!")
        print(f"   Train/val/test are completely separated at video level.")
        return True
    else:
        print(f"\n❌ VIDEO ID OVERLAP DETECTED - DATA LEAKAGE PRESENT!")
        print(f"   Total overlapping videos: {total_overlaps}")
        print(f"   This will cause inflated performance (AUC 0.99+)")
        print(f"   Please regenerate manifests with video-level split!")
        return False

def main():
    """Main execution with command line interface"""
    parser = argparse.ArgumentParser(description='AWARE-NET Dataset Utilities')
    parser.add_argument('--action',
                       choices=['generate', 'anonymize', 'balance', 'check', 'check-overlap', 'all'],
                       default='all',
                       help='Action to perform')
    parser.add_argument('--dataset', default='celebdf_v2',
                       help='Dataset name (celebdf_v2, faceforensics, deeperforensics)')

    args = parser.parse_args()

    print("🔧 AWARE-NET Dataset Utilities")
    print("=" * 40)

    if args.action in ['generate', 'all']:
        generate_manifests()

        # Automatically check for video overlap after generation
        print("\n" + "="*60)
        print("AUTOMATIC VIDEO OVERLAP CHECK")
        print("="*60)

        all_clean = True
        for dataset in ['celebdf_v2', 'faceforensics', 'deeperforensics']:
            is_clean = check_video_overlap(dataset)
            if not is_clean:
                all_clean = False

        if not all_clean:
            print("\n⚠️  WARNING: Some datasets have video ID overlap!")
            print("   This indicates frame-level split (data leakage).")
            print("   Performance metrics will be artificially inflated.")

    if args.action in ['anonymize', 'all']:
        if args.dataset == 'celebdf_v2':
            anonymize_dataset('celebdf_v2')
        elif args.dataset == 'faceforensics':
            anonymize_dataset('faceforensics', ['train', 'val', 'test'])
        elif args.dataset == 'deeperforensics':
            anonymize_dataset('deeperforensics', ['train', 'val', 'test'])

    if args.action in ['balance', 'all']:
        if args.dataset == 'celebdf_v2':
            create_balanced_dataset('celebdf_v2')
        elif args.dataset == 'faceforensics':
            create_balanced_dataset('faceforensics')
        elif args.dataset == 'deeperforensics':
            create_balanced_dataset('deeperforensics')
        else:
            # Create for all datasets when using 'all'
            create_balanced_dataset('celebdf_v2')
            create_balanced_dataset('faceforensics')
            create_balanced_dataset('deeperforensics')

    if args.action == 'check':
        manifest_path = f"manifests/{args.dataset}_train.csv"
        check_path_leakage(manifest_path)

    if args.action == 'check-overlap':
        # Check video overlap for all datasets or specific one
        if args.dataset == 'all':
            for dataset in ['celebdf_v2', 'faceforensics', 'deeperforensics']:
                check_video_overlap(dataset)
        else:
            check_video_overlap(args.dataset)

    print("\n✅ Dataset utilities complete!")
    print("📁 Available manifests:")
    manifest_dir = Path("manifests")
    if manifest_dir.exists():
        for file in sorted(manifest_dir.glob("*.csv")):
            print(f"   {file.name}")

if __name__ == "__main__":
    main()