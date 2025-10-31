#!/usr/bin/env python3
"""
Build and merge manifests for Deepfake-Eval-2024 dataset.

This tool:
1. Reads image manifest from downloaded images
2. Reads video frames manifest (if exists)
3. Merges and deduplicates records
4. Performs stratified train/val/test split
5. Validates data integrity
6. Saves three manifest CSVs for project use

Example:
    python -m src.tools.build_deepfake_eval_manifests \
        --image-manifest dataset/deepfake_eval_2024/images/manifest.csv \
        --video-manifest dataset/deepfake_eval_2024/video_frames_manifest.csv \
        --output-full dataset/deepfake_eval_2024/full_manifest.csv \
        --output-val manifests/deepfake_eval_2024_val.csv \
        --output-test manifests/deepfake_eval_2024_test.csv \
        --val-fraction 0.5 \
        --test-fraction 0.5 \
        --random-seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("build_deepfake_eval_manifests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and split Deepfake-Eval-2024 manifests"
    )
    parser.add_argument(
        "--image-manifest",
        required=True,
        help="Path to image manifest CSV",
    )
    parser.add_argument(
        "--video-manifest",
        default=None,
        help="Path to video frames manifest CSV (optional)",
    )
    parser.add_argument(
        "--output-full",
        default=None,
        help="Path to save full merged manifest",
    )
    parser.add_argument(
        "--output-val",
        required=True,
        help="Path to save validation set manifest",
    )
    parser.add_argument(
        "--output-test",
        required=True,
        help="Path to save test set manifest",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.5,
        help="Fraction of data to use for validation (rest goes to test)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.5,
        help="Fraction of data to use for test (only used if val+test < 1.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for stratified split",
    )
    parser.add_argument(
        "--group-by-video",
        action="store_true",
        help="Group frames from same video together (avoid data leakage)",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        default=True,
        help="Remove duplicate records (same image_path)",
    )
    parser.add_argument(
        "--validate-files",
        action="store_true",
        help="Check that referenced image files exist",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    """Load manifest CSV file."""
    if not path.exists():
        logger.error("Manifest file not found: %s", path)
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded manifest from %s: %d rows", path, len(df))
    return df


def merge_manifests(
    image_manifest: pd.DataFrame,
    video_manifest: Optional[pd.DataFrame] = None,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Merge image and video manifests."""
    # Ensure required columns exist
    for col in ["image_path", "label"]:
        if col not in image_manifest.columns:
            raise ValueError(f"Missing required column '{col}' in image manifest")

    # Add source_type if not exists
    if "source_type" not in image_manifest.columns:
        image_manifest["source_type"] = "image"

    # Add dataset field if not exists
    if "dataset" not in image_manifest.columns:
        image_manifest["dataset"] = "deepfake_eval_2024"

    # If video manifest provided, merge it
    if video_manifest is not None:
        if "source_type" not in video_manifest.columns:
            video_manifest["source_type"] = "video_frame"
        if "dataset" not in video_manifest.columns:
            video_manifest["dataset"] = "deepfake_eval_2024"

        df_combined = pd.concat([image_manifest, video_manifest], ignore_index=True)
    else:
        df_combined = image_manifest.copy()

    logger.info("Combined manifests: %d rows", len(df_combined))

    # Remove duplicates if requested
    if remove_duplicates:
        before = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=["image_path"], keep="first")
        after = len(df_combined)
        logger.info("Removed %d duplicate entries", before - after)

    # Validate labels
    unique_labels = df_combined["label"].unique()
    logger.info("Unique labels: %s", sorted(unique_labels))

    # Check label distribution
    label_counts = df_combined["label"].value_counts().to_dict()
    logger.info("Label distribution: %s", label_counts)

    return df_combined


def stratified_split(
    df: pd.DataFrame,
    val_fraction: float = 0.5,
    test_fraction: float = 0.5,
    random_seed: int = 42,
    group_by_video: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified split preserving label distribution.

    Args:
        df: Data to split
        val_fraction: Fraction for validation
        test_fraction: Fraction for test (rest goes to train if val+test < 1.0)
        random_seed: Random seed for reproducibility
        group_by_video: Whether to group by video_name to avoid leakage

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if group_by_video and "video_name" in df.columns:
        # Group by video to avoid train/test leakage
        logger.info("Grouping by video_name to prevent frame leakage")

        # Get unique videos
        df_videos = df.groupby("video_name").agg({"label": "first"}).reset_index()
        df_videos.columns = ["video_name", "label"]

        # Split videos first (stratified by label)
        df_val_videos, df_test_videos = train_test_split(
            df_videos,
            test_size=test_fraction,
            stratify=df_videos["label"],
            random_state=random_seed,
        )

        # Further split validation from rest if needed
        if val_fraction > 0 and val_fraction < 1.0:
            remaining_fraction = 1.0 - test_fraction
            if remaining_fraction > 0:
                val_of_remaining = val_fraction / remaining_fraction
                df_train_videos, df_val_videos = train_test_split(
                    df_val_videos,
                    test_size=val_of_remaining,
                    stratify=df_val_videos["label"],
                    random_state=random_seed + 1,
                )
            else:
                df_train_videos = df.iloc[:0]  # Empty
        else:
            df_train_videos = df.iloc[:0]  # Empty

        # Get all frames for each set of videos
        val_videos = set(df_val_videos["video_name"])
        test_videos = set(df_test_videos["video_name"])

        df_val = df[df["video_name"].isin(val_videos)].copy()
        df_test = df[df["video_name"].isin(test_videos)].copy()
        df_train = df[~df["video_name"].isin(val_videos | test_videos)].copy()

    else:
        # Simple stratified split without grouping
        logger.info("Performing simple stratified split")
        # First split: val vs (train+test)
        df_val, df_rest = train_test_split(
            df,
            test_size=1.0 - val_fraction,
            stratify=df["label"],
            random_state=random_seed,
        )

        # Second split: split remaining into train and test
        if test_fraction > 0:
            test_of_rest = test_fraction / (1.0 - val_fraction)
            df_test, df_train = train_test_split(
                df_rest,
                test_size=test_of_rest,
                stratify=df_rest["label"],
                random_state=random_seed + 1,
            )
        else:
            df_train = df_rest
            df_test = df.iloc[:0]  # Empty

    logger.info("Split sizes - Train: %d, Val: %d, Test: %d", len(df_train), len(df_val), len(df_test))
    logger.info("Val label distribution: %s", df_val["label"].value_counts().to_dict())
    logger.info("Test label distribution: %s", df_test["label"].value_counts().to_dict())

    return df_train, df_val, df_test


def validate_files(df: pd.DataFrame, root_dir: Path) -> pd.DataFrame:
    """Check if image files exist and filter invalid rows.

    Args:
        df: Manifest dataframe
        root_dir: Root directory for relative paths

    Returns:
        Filtered dataframe with only existing files
    """
    valid_rows = []
    missing_count = 0

    for idx, row in df.iterrows():
        image_path = root_dir / row["image_path"]
        if image_path.exists():
            valid_rows.append(row)
        else:
            missing_count += 1

    if missing_count > 0:
        logger.warning("Found %d missing image files", missing_count)

    result = pd.DataFrame(valid_rows)
    logger.info("After file validation: %d rows remain", len(result))
    return result


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load manifests
    try:
        df_images = load_manifest(Path(args.image_manifest))
    except FileNotFoundError:
        return 1

    df_video = None
    if args.video_manifest:
        try:
            df_video = load_manifest(Path(args.video_manifest))
        except FileNotFoundError:
            logger.warning("Video manifest not found, proceeding with images only")

    # Merge manifests
    df_combined = merge_manifests(df_images, df_video, remove_duplicates=args.remove_duplicates)

    # Validate files if requested
    if args.validate_files:
        root_dir = Path(args.image_manifest).parent.parent
        df_combined = validate_files(df_combined, root_dir)

    # Save full manifest if requested
    if args.output_full:
        output_path = Path(args.output_full)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(output_path, index=False)
        logger.info("Saved full manifest: %s", output_path)

    # Perform split
    # For OOD evaluation, use val/test split (no train)
    val_frac = args.val_fraction
    test_frac = args.test_fraction

    # Normalize fractions
    total = val_frac + test_frac
    if total > 1.0:
        val_frac = val_frac / total
        test_frac = test_frac / total

    df_train, df_val, df_test = stratified_split(
        df_combined,
        val_fraction=val_frac,
        test_fraction=test_frac,
        random_seed=args.random_seed,
        group_by_video=args.group_by_video,
    )

    # Save validation and test manifests
    for output_path, df, name in [
        (Path(args.output_val), df_val, "validation"),
        (Path(args.output_test), df_test, "test"),
    ]:
        if not df.empty:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info("Saved %s manifest: %s (%d rows)", name, output_path, len(df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
