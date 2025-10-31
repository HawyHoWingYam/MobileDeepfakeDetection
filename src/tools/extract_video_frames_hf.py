#!/usr/bin/env python3
"""
Extract frames from videos in HuggingFace Deepfake-Eval-2024 dataset.

This tool:
1. Loads video files from local directory
2. Uses OpenCV to extract frames at specified FPS or max frame count
3. Saves frames as JPEG images with organized directory structure
4. Loads metadata CSV to retrieve labels and other info
5. Generates a manifest CSV compatible with project dataset format

Example:
    python -m src.tools.extract_video_frames_hf \
        --video-dir dataset/deepfake_eval_2024/raw/video-data \
        --output-dir dataset/deepfake_eval_2024/video_frames \
        --metadata-csv dataset/deepfake_eval_2024/raw/video-metadata-publish-with-links.csv \
        --manifest-output dataset/deepfake_eval_2024/video_frames_manifest.csv \
        --fps 1 \
        --max-frames-per-video 30 \
        --image-size 256 \
        --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("extract_video_frames_hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from video files and create manifest CSV"
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store extracted frames",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="CSV file with video metadata (filename, label, etc.)",
    )
    parser.add_argument(
        "--manifest-output",
        required=True,
        help="Path to save output manifest CSV",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frame extraction rate (fps): extract 1 frame per N seconds. 1 = every frame, 2 = every 2 sec, etc.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=30,
        help="Maximum number of frames to extract per video",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize extracted frames to this size (square)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have frames extracted",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract frames even if they exist (overrides --skip-existing)",
    )
    parser.add_argument(
        "--root-prefix",
        default="",
        help="Prefix to prepend to relative frame paths in manifest",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_metadata_csv(metadata_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load metadata CSV and index by filename.

    Expected columns: filename (or video_filename), label, etc.
    """
    metadata = {}
    if not metadata_csv.exists():
        logger.warning("Metadata CSV not found: %s", metadata_csv)
        return metadata

    try:
        df = pd.read_csv(metadata_csv)
        # Try common filename column names
        filename_col = None
        for col in ["filename", "video_filename", "file_path", "path"]:
            if col in df.columns:
                filename_col = col
                break
        if not filename_col:
            logger.warning("No recognized filename column in metadata CSV. Columns: %s", df.columns.tolist())
            return metadata

        for _, row in df.iterrows():
            fname = str(row[filename_col]).strip()
            # Extract just the basename if full path provided
            fname_base = Path(fname).name
            metadata[fname_base] = row.to_dict()

        logger.info("Loaded metadata for %d videos", len(metadata))
    except Exception as e:
        logger.error("Error loading metadata CSV: %s", e)

    return metadata


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: int = 1,
    max_frames: int = 30,
    image_size: int = 256,
) -> Tuple[list[Path], Dict[str, Any]]:
    """Extract frames from a video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        target_fps: Target FPS for extraction (1 = every frame, 2 = every 2 sec, etc.)
        max_frames: Maximum frames to extract
        image_size: Size to resize frames to (square)

    Returns:
        Tuple of (list of extracted frame paths, extraction stats dict)
    """
    stats = {
        "video_path": str(video_path),
        "frames_extracted": 0,
        "total_frames": 0,
        "video_fps": 0.0,
        "duration_sec": 0.0,
        "error": None,
    }

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            stats["error"] = "Failed to open video"
            return [], stats

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats["video_fps"] = float(video_fps)
        stats["total_frames"] = total_frames
        stats["duration_sec"] = float(total_frames / video_fps) if video_fps > 0 else 0

        if video_fps <= 0:
            stats["error"] = f"Invalid video FPS: {video_fps}"
            cap.release()
            return [], stats

        # Calculate frame interval based on target FPS
        frame_interval = max(1, int(video_fps / target_fps))

        ensure_dir(output_dir)
        extracted_frames = []
        frame_idx = 0
        extracted_count = 0

        while extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Resize frame
                frame_resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

                # Save frame
                frame_filename = f"frame_{extracted_count:05d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_resized)
                extracted_frames.append(frame_path)
                extracted_count += 1

            frame_idx += 1

        cap.release()
        stats["frames_extracted"] = extracted_count

    except Exception as e:
        logger.error("Error extracting frames from %s: %s", video_path, e)
        stats["error"] = str(e)

    return extracted_frames, stats


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_output)

    if not video_dir.exists():
        logger.error("Video directory does not exist: %s", video_dir)
        return 1

    ensure_dir(output_dir)
    ensure_dir(manifest_path.parent)

    # Load metadata if provided
    metadata_dict = {}
    if args.metadata_csv:
        metadata_dict = load_metadata_csv(Path(args.metadata_csv))

    # Find all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    video_files = sorted(
        [f for f in video_dir.rglob("*") if f.is_file() and f.suffix.lower() in video_extensions]
    )

    if not video_files:
        logger.warning("No video files found in %s", video_dir)
        return 1

    logger.info("Found %d video files to process", len(video_files))

    manifest_rows = []
    skipped_videos = []
    failed_videos = []

    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = video_path.stem
        frames_output_dir = output_dir / f"{video_name}_frames"

        # Check if already extracted
        if frames_output_dir.exists() and args.skip_existing and not args.force:
            logger.debug("Skipping existing frames for video: %s", video_name)
            skipped_videos.append(video_name)
            # Still load metadata for manifest
            metadata = metadata_dict.get(video_path.name, {})
            label = metadata.get("label", metadata.get("fake", "unknown"))
            if isinstance(label, str) and label.lower() in ["real", "true", "0"]:
                label = 0
            elif isinstance(label, str) and label.lower() in ["fake", "false", "1"]:
                label = 1
            else:
                try:
                    label = int(float(label))
                except (ValueError, TypeError):
                    label = -1

            # Add manifest rows for existing frames
            frame_files = sorted(frames_output_dir.glob("frame_*.jpg"))
            for frame_idx, frame_path in enumerate(frame_files):
                rel_path = frame_path.relative_to(output_dir.parent)
                if args.root_prefix:
                    rel_path_str = args.root_prefix.rstrip("/") + "/" + str(rel_path).replace("\\", "/")
                else:
                    rel_path_str = str(rel_path).replace("\\", "/")

                manifest_rows.append(
                    {
                        "image_path": rel_path_str,
                        "label": label,
                        "video_name": video_name,
                        "frame_index": frame_idx,
                        "source_type": "video_frame",
                    }
                )
            continue

        # Extract frames
        ensure_dir(frames_output_dir)
        frame_paths, stats = extract_frames(
            video_path,
            frames_output_dir,
            target_fps=args.fps,
            max_frames=args.max_frames_per_video,
            image_size=args.image_size,
        )

        if stats["error"]:
            logger.warning("Failed to extract frames from %s: %s", video_name, stats["error"])
            failed_videos.append((video_name, stats["error"]))
            continue

        logger.debug("Extracted %d frames from %s", stats["frames_extracted"], video_name)

        # Get label from metadata
        metadata = metadata_dict.get(video_path.name, {})
        label = metadata.get("label", metadata.get("fake", "unknown"))

        # Convert label to int if possible
        if isinstance(label, str) and label.lower() in ["real", "true", "0"]:
            label = 0
        elif isinstance(label, str) and label.lower() in ["fake", "false", "1"]:
            label = 1
        else:
            try:
                label = int(float(label))
            except (ValueError, TypeError):
                logger.warning("Unable to parse label for video %s: %s", video_name, label)
                label = -1

        # Add manifest rows for extracted frames
        for frame_idx, frame_path in enumerate(frame_paths):
            rel_path = frame_path.relative_to(output_dir.parent)
            if args.root_prefix:
                rel_path_str = args.root_prefix.rstrip("/") + "/" + str(rel_path).replace("\\", "/")
            else:
                rel_path_str = str(rel_path).replace("\\", "/")

            manifest_rows.append(
                {
                    "image_path": rel_path_str,
                    "label": label,
                    "video_name": video_name,
                    "frame_index": frame_idx,
                    "source_type": "video_frame",
                    "fps": stats["video_fps"],
                    "original_video_path": str(video_path.relative_to(video_dir.parent)),
                }
            )

    # Save manifest
    if manifest_rows:
        fieldnames = sorted(set().union(*[row.keys() for row in manifest_rows]))
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in manifest_rows:
                # Fill missing fields with empty string
                filled_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(filled_row)
        logger.info("Manifest saved: %s (%d frames from %d videos)", manifest_path, len(manifest_rows), len(video_files))
    else:
        logger.warning("No frames were extracted")

    # Save extraction log
    log_summary = {
        "total_videos": len(video_files),
        "skipped_videos": len(skipped_videos),
        "failed_videos": len(failed_videos),
        "total_frames_extracted": len(manifest_rows),
        "failed_details": [(v, e) for v, e in failed_videos],
    }

    log_path = output_dir / "extraction_log.json"
    with log_path.open("w") as f:
        json.dump(log_summary, f, indent=2)
    logger.info("Extraction log saved: %s", log_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
