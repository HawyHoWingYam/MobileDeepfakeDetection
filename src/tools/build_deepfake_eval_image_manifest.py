#!/usr/bin/env python3
"""
Build an image manifest for Deepfake‑Eval‑2024 images.

Reads the local snapshot directory (e.g., Deepfake-Eval-2024/image-data)
and the published image metadata CSV to map filenames to labels.

Outputs a manifest CSV with at least: image_path,label and optional fields.

Example:
  python -m src.tools.build_deepfake_eval_image_manifest \
    --image-dir Deepfake-Eval-2024/image-data \
    --metadata-csv Deepfake-Eval-2024/image-metadata-publish.csv \
    --output Deepfake-Eval-2024/images_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("build_deepfake_eval_image_manifest")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Deepfake‑Eval‑2024 image manifest")
    ap.add_argument("--image-dir", required=True, help="Directory with image files (flat)")
    ap.add_argument(
        "--metadata-csv",
        required=True,
        help="Image metadata CSV (e.g., image-metadata-publish.csv or *_with_links.csv)",
    )
    ap.add_argument("--output", required=True, help="Output manifest CSV path")
    ap.add_argument(
        "--root-prefix",
        default="",
        help="Prefix to prepend to image_path (e.g., 'Deepfake-Eval-2024')",
    )
    return ap.parse_args()


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {c.lower(): c for c in df.columns}


def _parse_label(row: Dict[str, Any]) -> int:
    lower = {k.lower(): v for k, v in row.items()}
    candidates = ["label", "ground truth", "ground_truth", "fake", "is_fake"]
    val: Any = None
    for key in candidates:
        if key in lower:
            val = lower[key]
            break
    if val is None:
        return -1
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"real", "true", "0", "r"}:
            return 0
        if v in {"fake", "false", "1", "f"}:
            return 1
        try:
            val = int(float(v))
        except Exception:
            return -1
    try:
        iv = int(val)
        return iv if iv in (0, 1) else -1
    except Exception:
        return -1


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    image_dir = Path(args.image_dir)
    metadata_csv = Path(args.metadata_csv)
    output_path = Path(args.output)

    if not image_dir.exists():
        logger.error("Image directory not found: %s", image_dir)
        return 1
    if not metadata_csv.exists():
        logger.error("Metadata CSV not found: %s", metadata_csv)
        return 1

    # Read metadata and index by basename
    df = pd.read_csv(metadata_csv)
    col_map = _normalize_columns(df)
    filename_key = None
    for key in ["filename", "image_filename", "file_path", "path", "file", "name"]:
        if key in col_map:
            filename_key = col_map[key]
            break
    if not filename_key:
        logger.error("No recognized filename column in %s (columns=%s)", metadata_csv, df.columns.tolist())
        return 1

    index: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        fname = str(r[filename_key]).strip()
        index[Path(fname).name] = r.to_dict()

    # Enumerate images
    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])
    if not image_files:
        logger.warning("No image files found in %s", image_dir)

    rows = []
    missing_meta = 0
    unlabeled = 0

    for img_path in tqdm(image_files, desc="Indexing images"):
        meta = index.get(img_path.name)
        if not meta:
            missing_meta += 1
            continue
        label = _parse_label(meta)
        if label not in (0, 1):
            unlabeled += 1
            continue

        # Build relative path from project root
        rel_path = img_path
        rel_str = str(rel_path).replace("\\", "/")
        if args.root_prefix:
            rel_str = args.root_prefix.rstrip("/") + "/" + rel_str

        row = {
            "image_path": rel_str,
            "label": label,
            "source_type": "image",
            "dataset": "deepfake_eval_2024",
            "original_filename": img_path.name,
        }
        # Optional: include finetuning split if present
        lower = {k.lower(): v for k, v in meta.items()}
        if "finetuning set" in lower:
            row["finetuning_set"] = lower["finetuning set"]

        rows.append(row)

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted(set().union(*[r.keys() for r in rows]))
        with output_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        logger.info("Saved image manifest: %s (rows=%d, missing_meta=%d, unlabeled=%d)", output_path, len(rows), missing_meta, unlabeled)
    else:
        logger.warning("No labeled rows collected (missing_meta=%d, unlabeled=%d)", missing_meta, unlabeled)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

