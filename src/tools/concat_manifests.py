#!/usr/bin/env python3
"""
Concatenate one or more dataset manifests into a single CSV.

By default, reads dataset names and split keys from configs/datasets.json and
produces a combined CSV with columns: image_path,label[,dataset].

Examples:
  # Combine default 4 training sets into a single training manifest
  python src/tools/concat_manifests.py \
    --datasets celebdf_v2 faceforensics deeperforensics dfdc \
    --splits train \
    --output outputs/stage5/train_all.csv

  # Combine multiple CSV files directly
  python src/tools/concat_manifests.py --files manifests/a.csv manifests/b.csv --output combined.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Concatenate dataset manifests into a single CSV')
    p.add_argument('--datasets', nargs='*', default=['celebdf_v2','faceforensics','deeperforensics','dfdc'],
                   help='Dataset names defined in configs/datasets.json')
    p.add_argument('--splits', nargs='*', default=['train'], choices=['train','val','test'],
                   help='Split keys to include from each dataset (default: train)')
    p.add_argument('--files', nargs='*', default=None, help='Explicit CSVs to concat (overrides --datasets/--splits)')
    p.add_argument('--output', required=True, help='Output CSV path')
    p.add_argument('--add-dataset-col', action='store_true', default=True,
                   help='Add a dataset column indicating source dataset')
    return p.parse_args()


def load_from_config(datasets: List[str], splits: List[str]) -> List[pd.DataFrame]:
    cfg = json.loads(Path('configs/datasets.json').read_text())
    out: List[pd.DataFrame] = []
    for ds_name in datasets:
        ds_cfg = cfg.get('datasets', {}).get(ds_name)
        if not ds_cfg:
            logger.warning('Dataset not found in config: %s', ds_name)
            continue
        root = Path(ds_cfg.get('root_path', '.'))
        for split in splits:
            man_rel = ds_cfg.get('splits', {}).get(split)
            if not man_rel:
                continue
            man = root / man_rel
            if not man.exists():
                logger.warning('Manifest not found: %s (%s/%s)', man, ds_name, split)
                continue
            df = pd.read_csv(man)
            if 'image_path' not in df.columns or 'label' not in df.columns:
                logger.warning('Skipping invalid manifest (missing image_path/label): %s', man)
                continue
            df = df[['image_path', 'label']].copy()
            df['dataset'] = ds_name
            out.append(df)
            logger.info('Loaded %s (%s): %d rows', man, ds_name, len(df))
    return out


def load_from_files(files: List[str]) -> List[pd.DataFrame]:
    out: List[pd.DataFrame] = []
    for fp in files:
        p = Path(fp)
        if not p.exists():
            logger.warning('File not found: %s', p)
            continue
        df = pd.read_csv(p)
        if 'image_path' not in df.columns or 'label' not in df.columns:
            logger.warning('Skipping invalid manifest (missing image_path/label): %s', p)
            continue
        out.append(df[['image_path', 'label']].copy())
    return out


def main() -> int:
    args = parse_args()
    parts: List[pd.DataFrame]
    if args.files:
        parts = load_from_files(args.files)
    else:
        parts = load_from_config(args.datasets, args.splits)

    if not parts:
        logger.error('No inputs loaded.')
        return 1

    df = pd.concat(parts, axis=0, ignore_index=True)
    # Ensure required columns
    cols = ['image_path', 'label'] + (['dataset'] if args.add_dataset_col and 'dataset' in df.columns else [])
    df = df[cols]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info('Wrote combined manifest: %s (rows=%d)', out, len(df))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

