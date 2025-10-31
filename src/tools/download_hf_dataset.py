#!/usr/bin/env python3
"""
Download a HuggingFace dataset split, export images, and build a manifest CSV.

Example:
    python -m src.tools.download_hf_dataset \
        --dataset-id nuriachandra/Deepfake-Eval-2024 \
        --split train \
        --output-dir data/deepfake_eval_2024/train
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from datasets import Image, load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "The 'datasets' package is required. Install with `pip install datasets`."
    ) from exc


logger = logging.getLogger("download_hf_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset split and export images + manifest."
    )
    parser.add_argument("--dataset-id", required=True, help="Dataset repository id (e.g., user/dataset)")
    parser.add_argument("--split", default="train", help="Dataset split to download")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset configuration name (if the dataset exposes multiple configs)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store images and manifest.csv",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column name containing the image data",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column name containing label values",
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="Comma-separated label mapping (e.g., 'audio-data=0,video-data=1' or 'real=0,fake=1'). If not provided, uses sequential mapping.",
    )
    parser.add_argument(
        "--root-prefix",
        default="",
        help="Prefix to prepend to relative image paths in manifest (e.g., 'dataset/deepfake_eval_2024/images')",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token. Falls back to HF_TOKEN or HUGGINGFACEHUB_API_TOKEN environment variables.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to download (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-downloading images that already exist on disk.",
    )
    parser.add_argument(
        "--compression",
        choices=["png", "jpg"],
        default="png",
        help="File format to save images (default: png).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON file to store additional metadata per sample.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode to load dataset iteratively (prevents OOM errors on large datasets).",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming mode (not recommended for large datasets).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100,
        help="Save checkpoint manifest every N samples (default: 100). Set to 0 to disable checkpointing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if it exists.",
    )
    parser.add_argument(
        "--disable-caching",
        action="store_true",
        help="Disable HuggingFace dataset caching to reduce memory footprint.",
    )
    return parser.parse_args()


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def encode_label(
    value: Any,
    encoder_cache: Dict[str, int],
    label_features: Optional[Any],
    label_map: Optional[Dict[str, int]] = None,
) -> int:
    """Encode label value to integer.

    Args:
        value: The raw label value
        encoder_cache: Cache for auto-discovered labels
        label_features: HF dataset feature info
        label_map: Explicit label -> int mapping (highest priority)

    Returns:
        Encoded label as integer
    """
    if label_map:
        key = str(value)
        if key in label_map:
            return label_map[key]
        else:
            logger.warning("Label '%s' not in provided label_map %s, falling back to encoder", key, label_map)

    if isinstance(value, (int, float)):
        return int(value)

    if label_features is not None and hasattr(label_features, "names"):
        try:
            return int(label_features.str2int(str(value)))
        except (ValueError, KeyError):
            pass

    key = str(value)
    if key not in encoder_cache:
        encoder_cache[key] = len(encoder_cache)
        logger.debug("Discovered new label '%s' -> %d", key, encoder_cache[key])
    return encoder_cache[key]


def save_manifest(manifest_rows: list[Dict[str, Any]], manifest_path: Path) -> None:
    if not manifest_rows:
        logger.warning("No rows collected; manifest not written.")
        return

    fieldnames = sorted({k for row in manifest_rows for k in row.keys()})
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    logger.info("Manifest saved: %s (%d rows)", manifest_path, len(manifest_rows))


def load_checkpoint_manifest(checkpoint_path: Path) -> tuple[list[Dict[str, Any]], set[str]]:
    """Load existing checkpoint manifest and return rows + processed sample IDs."""
    if not checkpoint_path.exists():
        return [], set()

    rows = []
    processed_ids = set()
    try:
        with checkpoint_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                processed_ids.add(row.get("image_path", ""))
        logger.info("Loaded checkpoint from %s (%d rows)", checkpoint_path, len(rows))
        return rows, processed_ids
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s. Starting fresh.", checkpoint_path, e)
        return [], set()


def get_checkpoint_path(output_dir: Path) -> Path:
    """Get the checkpoint manifest path for resuming."""
    return output_dir / "manifest_checkpoint.csv"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    token = resolve_token(args.token)
    output_dir = ensure_dir(Path(args.output_dir))
    images_dir = ensure_dir(output_dir / "images")
    manifest_path = output_dir / "manifest.csv"
    checkpoint_path = get_checkpoint_path(output_dir)
    metadata_path = Path(args.metadata) if args.metadata else None
    metadata_records: Dict[str, Dict[str, Any]] = {}

    # Parse label mapping if provided
    label_map: Optional[Dict[str, int]] = None
    if args.label_map:
        try:
            label_map = {}
            pairs = args.label_map.split(",")
            for pair in pairs:
                key, val = pair.strip().split("=")
                label_map[key.strip()] = int(val.strip())
            logger.info("Using custom label mapping: %s", label_map)
        except (ValueError, KeyError) as e:
            logger.error("Invalid label_map format '%s': %s", args.label_map, e)
            return 1

    # Handle resume from checkpoint
    manifest_rows: list[Dict[str, Any]] = []
    processed_ids: set[str] = set()
    resume_idx = 0
    if args.resume:
        manifest_rows, processed_ids = load_checkpoint_manifest(checkpoint_path)
        resume_idx = len(manifest_rows)
        if manifest_rows:
            logger.info("Resuming from checkpoint with %d already processed samples", resume_idx)

    logger.info(
        "Downloading dataset '%s' (config=%s, split=%s, streaming=%s)",
        args.dataset_id,
        args.config or "<default>",
        args.split,
        args.streaming,
    )

    load_dataset_kwargs: Dict[str, Any] = {
        "path": args.dataset_id,
        "name": args.config,
        "split": args.split,
        "streaming": args.streaming,
    }
    if token:
        load_dataset_kwargs["token"] = token
    if args.disable_caching:
        load_dataset_kwargs["keep_in_memory"] = False

    try:
        ds = load_dataset(**load_dataset_kwargs)
    except TypeError:
        # Older datasets versions expect use_auth_token instead of token
        if token:
            load_dataset_kwargs.pop("token", None)
            load_dataset_kwargs["use_auth_token"] = token
        try:
            ds = load_dataset(**load_dataset_kwargs)
        except TypeError:
            # Fallback: remove streaming parameter for older versions
            load_dataset_kwargs.pop("streaming", None)
            ds = load_dataset(**load_dataset_kwargs)

    # Cast image column only if not in streaming mode (streaming mode handles this differently)
    if not args.streaming:
        ds = ds.cast_column(args.image_column, Image())

    label_features = ds.features.get(args.label_column) if hasattr(ds, "features") else None
    label_encoder: Dict[str, int] = {}

    compression = args.compression.lower()
    if compression == "jpg":
        compression = "jpeg"

    try:
        for idx, sample in enumerate(ds):
            # Skip samples if resuming
            if idx < resume_idx:
                continue

            if args.max_samples is not None and idx >= args.max_samples:
                break

            image = sample.get(args.image_column)
            if image is None:
                logger.warning("Sample %d missing image column '%s'; skipping", idx, args.image_column)
                continue

            # Handle PIL Image conversion in streaming mode
            if args.streaming and isinstance(image, dict):
                try:
                    from PIL import Image as PILImage
                    image = PILImage.open(image.get("path") or image.get("bytes"))
                except Exception as e:
                    logger.warning("Sample %d failed to open image: %s; skipping", idx, e)
                    continue

            label_raw = sample.get(args.label_column)
            label = encode_label(label_raw, label_encoder, label_features, label_map=label_map)

            label_dir = ensure_dir(images_dir / f"class_{label}")
            file_name = f"{idx:07d}.{compression}"
            image_path = label_dir / file_name

            if image_path.exists() and args.skip_existing:
                logger.debug("Skipping existing image: %s", image_path)
            else:
                pil_image = image.convert("RGB") if hasattr(image, "convert") else image
                pil_image.save(image_path, quality=95 if compression == "jpeg" else None)

            rel_path = image_path.relative_to(output_dir)
            rel_path_str = str(rel_path).replace("\\", "/")

            # Prepend root prefix if provided
            if args.root_prefix:
                rel_path_str = args.root_prefix.rstrip("/") + "/" + rel_path_str

            # Skip if already processed (for resume functionality)
            if rel_path_str in processed_ids:
                logger.debug("Skipping already processed sample: %s", rel_path_str)
                continue

            manifest_row = {
                "image_path": rel_path_str,
                "label": int(label),
                "source_split": args.split,
            }
            manifest_rows.append(manifest_row)

            if metadata_path:
                metadata_records[rel_path_str] = {
                    "label_raw": str(label_raw),
                    "label_encoded": int(label),
                }

            # Periodic checkpoint save
            if args.checkpoint_freq > 0 and (idx + 1) % args.checkpoint_freq == 0:
                save_manifest(manifest_rows, checkpoint_path)
                logger.info("Checkpoint saved: %d samples processed", idx + 1)

            # Log progress (without total count in streaming mode)
            if args.streaming:
                if (idx + 1) % 100 == 0:
                    logger.info("Processed %d samples...", idx + 1)
            else:
                total = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
                if (idx + 1) % 100 == 0 or (idx + 1) == total:
                    logger.info("Processed %d/%d samples", idx + 1, total)

    except Exception as e:
        if "bad_alloc" in str(e).lower() or "out of memory" in str(e).lower():
            logger.error(
                "Memory exhaustion error: %s\n"
                "This usually happens with large datasets when not using streaming mode.\n"
                "Try using --streaming (enabled by default) or --max-samples to limit the download.",
                e,
            )
            return 1
        else:
            logger.exception("Unexpected error during dataset processing: %s", e)
            return 1

    # Save final manifest
    save_manifest(manifest_rows, manifest_path)

    # Clean up checkpoint file after successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug("Removed checkpoint file: %s", checkpoint_path)
        except Exception as e:
            logger.warning("Could not remove checkpoint file %s: %s", checkpoint_path, e)

    if metadata_path and metadata_records:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_records, f, indent=2)
        logger.info("Sample metadata saved: %s", metadata_path)

    logger.info("Done. Images stored under %s (%d samples processed)", images_dir, len(manifest_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
