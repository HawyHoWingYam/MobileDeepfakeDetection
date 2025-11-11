#!/usr/bin/env python3
"""
Split a video-frame manifest into val/test at VIDEO level (no leakage).

Input CSV (at least): image_path,label[,video_name]
Output CSVs: manifests/deepfake_eval_2024_val.csv, manifests/deepfake_eval_2024_test.csv

Notes
- Group by video_name if present; else derive from the parent directory of image_path.
- Optionally class-balance per split by capping videos to the minority count.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def infer_video_id(row: pd.Series) -> str:
    vid = row.get('video_name')
    if isinstance(vid, str) and vid:
        return vid
    path = str(row['image_path'])
    parts = path.replace('\\', '/').split('/')
    # Example: video-frames/<video_id>_frames/frame_00001.jpg → use parent dir name
    return parts[-2] if len(parts) >= 2 else path


def split_by_video(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    balanced: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert abs(val_ratio + test_ratio - 1.0) < 1e-6, "val_ratio + test_ratio must equal 1.0"

    # Build mapping: video_id -> label
    df = df.copy()
    df['video_id'] = df.apply(infer_video_id, axis=1)

    # We assume all frames of a video share the same label; enforce by majority if noisy
    vid_label: Dict[str, int] = (
        df.groupby('video_id')['label']
          .agg(lambda s: int(round(float(s.mean()))))
          .to_dict()
    )
    real_vids = [v for v, y in vid_label.items() if y == 0]
    fake_vids = [v for v, y in vid_label.items() if y == 1]

    rng = np.random.default_rng(seed)
    rng.shuffle(real_vids)
    rng.shuffle(fake_vids)

    def take_split(vids: List[str]) -> Tuple[List[str], List[str]]:
        n = len(vids)
        n_val = int(round(n * val_ratio))
        val_ids = vids[:n_val]
        test_ids = vids[n_val:]
        return val_ids, test_ids

    r_val, r_test = take_split(real_vids)
    f_val, f_test = take_split(fake_vids)

    if balanced:
        # Cap each split to the minority class video count
        n_val = min(len(r_val), len(f_val))
        n_test = min(len(r_test), len(f_test))
        r_val, f_val = r_val[:n_val], f_val[:n_val]
        r_test, f_test = r_test[:n_test], f_test[:n_test]
        logger.info("Applied per-split class balancing at video level: val=%d/test=%d videos per class", n_val, n_test)

    val_ids = set(r_val) | set(f_val)
    test_ids = set(r_test) | set(f_test)

    val_df = df[df['video_id'].isin(val_ids)][['image_path', 'label']].copy()
    test_df = df[df['video_id'].isin(test_ids)][['image_path', 'label']].copy()

    return val_df, test_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Split a video-frame manifest into val/test without leakage')
    p.add_argument('--input', required=True, help='Input CSV with image_path,label[,video_name]')
    p.add_argument('--val-out', default='manifests/deepfake_eval_2024_val.csv')
    p.add_argument('--test-out', default='manifests/deepfake_eval_2024_test.csv')
    p.add_argument('--val-ratio', type=float, default=0.5)
    p.add_argument('--test-ratio', type=float, default=0.5)
    p.add_argument('--balanced', action='store_true', default=True, help='Balance classes per split by video count')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        logger.error('Input not found: %s', inp)
        return 1

    df = pd.read_csv(inp)
    missing = {'image_path', 'label'} - set(df.columns)
    if missing:
        logger.error('Missing required columns: %s', sorted(missing))
        return 1

    val_df, test_df = split_by_video(df, args.val_ratio, args.test_ratio, args.seed, args.balanced)

    Path(args.val_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_out).parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(args.val_out, index=False)
    test_df.to_csv(args.test_out, index=False)

    logger.info('Wrote val: %s (rows=%d)', args.val_out, len(val_df))
    logger.info('Wrote test: %s (rows=%d)', args.test_out, len(test_df))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

