#!/usr/bin/env python3
"""
Build a balanced "difficult subset" manifest for Stage 5 training.

Input: predictions CSV from cascade_infer.py (e.g., outputs/stage5/train_preds.csv)
Expected columns:
  - path, label, prediction, confidence, stage_used, stage1_confidence, stage2_confidence

Selection logic (default):
  - Base hard set: (stage_used == 2) OR (prediction != label)
  - Optional edge set: stage1_confidence within [low±margin] or [high±margin]
  - Union(base, edge), then drop duplicates by path

Balancing:
  - Per-class cap (default 15000). Target per class = min(n_fake, n_real, cap)
  - Downsample majority class to target; keep minority class as-is up to target

Output: CSV with at least [image_path, label]
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build balanced difficult subset manifest for Stage 5"
    )
    p.add_argument("--preds", type=str, default="outputs/stage5/train_preds.csv",
                   help="Input predictions CSV from cascade_infer.py")
    p.add_argument("--output", type=str, default="outputs/stage5/train_difficult_subset.csv",
                   help="Output manifest CSV path")
    p.add_argument("--low", type=float, default=0.03,
                   help="Low threshold (for edge samples)")
    p.add_argument("--high", type=float, default=0.55,
                   help="High threshold (for edge samples)")
    p.add_argument("--margin", type=float, default=0.02,
                   help="Margin around thresholds to include edge samples")
    p.add_argument("--per-class-cap", type=int, default=15000,
                   help="Max samples per class in the balanced subset (set <=0 for unlimited)")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    p.add_argument("--no-edge", action="store_true", default=False,
                   help="Do not include Stage1 edge samples around thresholds")
    return p.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    required = {"path", "label", "prediction", "stage_used", "stage1_confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in preds CSV: {sorted(missing)}")


def select_difficult(df: pd.DataFrame, low: float, high: float, margin: float, include_edge: bool) -> pd.DataFrame:
    # Base hard: escalated samples or misclassified
    hard_mask = (df["stage_used"] == 2) | (df["prediction"] != df["label"])
    if include_edge:
        edge_mask = df["stage1_confidence"].between(low - margin, low + margin) | \
                    df["stage1_confidence"].between(high - margin, high + margin)
        sel = df[hard_mask | edge_mask]
    else:
        sel = df[hard_mask]
    # Deduplicate by image path
    sel = sel.drop_duplicates(subset=["path"]).copy()
    return sel


def balance_by_label(sel: pd.DataFrame, per_class_cap: int, seed: int) -> Tuple[pd.DataFrame, int, int, int]:
    sel = sel.copy()
    counts = sel["label"].value_counts().to_dict()
    n_fake = int(counts.get(1, 0))
    n_real = int(counts.get(0, 0))
    if n_fake == 0 or n_real == 0:
        # Nothing to balance; return as-is
        return sel, n_real, n_fake, min(n_real, n_fake)

    cap = per_class_cap if per_class_cap and per_class_cap > 0 else min(n_fake, n_real)
    target = min(n_fake, n_real, cap)

    # Downsample if needed
    rng = np.random.default_rng(seed)
    fake_sel = sel[sel["label"] == 1]
    real_sel = sel[sel["label"] == 0]

    if len(fake_sel) > target:
        fake_sel = fake_sel.sample(n=target, random_state=seed)
    if len(real_sel) > target:
        real_sel = real_sel.sample(n=target, random_state=seed)

    balanced = pd.concat([fake_sel, real_sel], axis=0).sample(frac=1.0, random_state=seed)
    return balanced, n_real, n_fake, target


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    preds_path = Path(args.preds)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        print(f"Predictions CSV not found: {preds_path}")
        return 1

    # Read predictions with consistent dtypes and coerce numeric columns
    df = pd.read_csv(preds_path, low_memory=False)
    # Best-effort dtype coercion to avoid object/str vs float comparison errors
    for col in ["label", "prediction", "stage_used"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    if "stage1_confidence" in df.columns:
        df["stage1_confidence"] = pd.to_numeric(df["stage1_confidence"], errors="coerce")
    if "stage2_confidence" in df.columns:
        df["stage2_confidence"] = pd.to_numeric(df["stage2_confidence"], errors="coerce")
    validate_columns(df)

    sel = select_difficult(df, args.low, args.high, args.margin, include_edge=not args.no_edge)
    balanced, n_real, n_fake, target = balance_by_label(sel, args.per_class_cap, args.seed)

    # Save manifest with required columns
    manifest = balanced[["path", "label"]].rename(columns={"path": "image_path"})
    manifest.to_csv(out_path, index=False)

    print(f"Selected raw hard: {len(sel)}  (real={n_real}, fake={n_fake})")
    print(f"Balanced subset  : {len(balanced)}  (per-class target={target}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
