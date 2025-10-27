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
) -> int:
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
    metadata_path = Path(args.metadata) if args.metadata else None
    metadata_records: Dict[str, Dict[str, Any]] = {}

    logger.info(
        "Downloading dataset '%s' (config=%s, split=%s)",
        args.dataset_id,
        args.config or "<default>",
        args.split,
    )

    ds = load_dataset(
        path=args.dataset_id,
        name=args.config,
        split=args.split,
        use_auth_token=token,
    )
    ds = ds.cast_column(args.image_column, Image())
    label_features = ds.features.get(args.label_column)
    label_encoder: Dict[str, int] = {}

    compression = args.compression.lower()
    if compression == "jpg":
        compression = "jpeg"

    manifest_rows: list[Dict[str, Any]] = []

    total = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
    for idx, sample in enumerate(ds):
        if args.max_samples is not None and idx >= args.max_samples:
            break

        image = sample.get(args.image_column)
        if image is None:
            logger.warning("Sample %d missing image column '%s'; skipping", idx, args.image_column)
            continue

        label_raw = sample.get(args.label_column)
        label = encode_label(label_raw, label_encoder, label_features)

        label_dir = ensure_dir(images_dir / f"class_{label}")
        file_name = f"{idx:07d}.{compression}"
        image_path = label_dir / file_name

        if image_path.exists() and args.skip_existing:
            logger.debug("Skipping existing image: %s", image_path)
        else:
            pil_image = image.convert("RGB") if hasattr(image, "convert") else image
            pil_image.save(image_path, quality=95 if compression == "jpeg" else None)

        rel_path = image_path.relative_to(output_dir)
        manifest_row = {
            "image_path": str(rel_path).replace("\\", "/"),
            "label": int(label),
            "source_split": args.split,
        }
        manifest_rows.append(manifest_row)

        if metadata_path:
            metadata_records[str(rel_path)] = {
                "label_raw": label_raw,
                "label_encoded": int(label),
            }

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            logger.info("Processed %d/%d samples", idx + 1, total)

    save_manifest(manifest_rows, manifest_path)

    if metadata_path and metadata_records:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_records, f, indent=2)
        logger.info("Sample metadata saved: %s", metadata_path)

    logger.info("Done. Images stored under %s", images_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
