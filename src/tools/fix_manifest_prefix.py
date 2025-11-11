#!/usr/bin/env python3
"""
Fix image_path prefix in a manifest CSV in-place or to a new file.

Usage example:
  python -m src.tools.fix_manifest_prefix \
    --manifest manifests/deepfake_eval_2024_val.csv \
    --old-prefix "video-frames/" \
    --new-prefix "Deepfake-Eval-2024/video-frames/" \
    --inplace

Notes:
 - Expects a CSV with at least the column 'image_path'.
 - Only rows whose image_path startswith old-prefix will be changed.
 - Writes atomically by saving to a temp file and moving over when --inplace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replace image_path prefix in manifest CSV")
    p.add_argument("--manifest", required=True, help="Path to manifest CSV to modify")
    p.add_argument("--old-prefix", required=True, help="Old prefix to replace (e.g., 'video-frames/')")
    p.add_argument("--new-prefix", required=True, help="New prefix (e.g., 'Deepfake-Eval-2024/video-frames/')")
    p.add_argument("--output", default=None, help="Optional output CSV (default: overwrite when --inplace)")
    p.add_argument("--inplace", action="store_true", default=False, help="Overwrite the input manifest")
    p.add_argument("--dry-run", action="store_true", default=False, help="Print a few before/after paths and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        return 1

    df = pd.read_csv(manifest)
    if 'image_path' not in df.columns:
        print("Manifest missing 'image_path' column")
        return 1

    old = args.old_prefix
    new = args.new_prefix

    # Track changes for a small sample
    sample_before_after = []
    def _replace(p: str) -> str:
        if isinstance(p, str) and p.startswith(old):
            return new + p[len(old):]
        return p

    # Apply replacement
    df['image_path'] = df['image_path'].map(_replace)

    # Collect up to 5 samples where change occurred
    for _, row in df.iterrows():
        ip = row['image_path']
        if isinstance(ip, str) and ip.startswith(new):
            sample_before_after.append(ip)
            if len(sample_before_after) >= 5:
                break

    if args.dry_run:
        print("Dry run: showing first 5 updated image_path entries (if any):")
        for s in sample_before_after:
            print("  ", s)
        return 0

    # Decide output path
    if args.inplace and not args.output:
        tmp = manifest.with_suffix(manifest.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(manifest)
        print(f"Updated in-place: {manifest}")
    else:
        out = Path(args.output) if args.output else (manifest.parent / (manifest.stem + "_fixed" + manifest.suffix))
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

