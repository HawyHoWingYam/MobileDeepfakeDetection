#!/usr/bin/env python3
"""
Evaluate the Stage 4 cascade across multiple datasets (val/test) and summarize metrics.

- Loads the cascade once (Stage 1 + Stage 2) using tuned thresholds JSON
- Runs inference on each dataset split manifest
- Saves per-split predictions CSV and a summary.csv with Acc/F1/Recall/FNR/FPR and stage2 rate

New: "Stage2-only" diagnostic mode to measure Stage 2 upper bound by escalating
all samples to Stage 2 (auto-generates thresholds with low=0.0, high=1.0).

Usage (example):
  # Regular evaluation
  python src/tools/eval_cascade_across_datasets.py \
    --stage1-ckpt outputs/stage1/run_20251023_025539/best_model.pth \
    --stage2-ckpt outputs/stage5/run_20251031_041013/best_model.pth \
    --thresholds outputs/stage4/best_config.json \
    --device cuda:0 \
    --stage1-size 256 --stage2-size 384 \
    --splits val --datasets celebdf_v2 faceforensics deeperforensics dfdc \
    --output-dir outputs/stage4/evals

  # Stage2-only diagnostic (escalate all to Stage 2)
  python src/tools/eval_cascade_across_datasets.py \
    --stage1-ckpt outputs/stage1/run_.../best_model.pth \
    --stage2-ckpt outputs/stage5/finetune_.../best_model.pth \
    --thresholds outputs/stage4/best_config.json \
    --stage2-only \
    --datasets deepfake_eval_2024 --splits val \
    --stage1-size 256 --stage2-size 384 \
    --output-dir outputs/stage5/evals_stage2only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Make sure 'src' is on import path, then import CascadeEngine
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from tools.cascade_infer import CascadeEngine  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eval_cascade_across_datasets')


DEFAULT_MANIFESTS = {
    'celebdf_v2': {
        'val': 'manifests/celebdf_v2_val_balanced.csv',
        'test': 'manifests/celebdf_v2_test_balanced.csv',
    },
    'faceforensics': {
        'val': 'manifests/faceforensics_val_balanced.csv',
        'test': 'manifests/faceforensics_test_balanced.csv',
    },
    'deeperforensics': {
        'val': 'manifests/deeperforensics_val_balanced.csv',
        'test': 'manifests/deeperforensics_test_balanced.csv',
    },
    'dfdc': {
        'val': 'manifests/dfdc_val_balanced.csv',
        'test': 'manifests/dfdc_test_balanced.csv',
    },
    'deepfake_eval_2024': {
        'val': 'manifests/deepfake_eval_2024_val.csv',
        'test': 'manifests/deepfake_eval_2024_test.csv',
    },
}


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute metrics safely. If predictions are missing/empty, return zeros."""
    # Normalize column names (strip spaces)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {'label', 'prediction'}
    n = int(len(df))
    if n == 0 or not required_cols.issubset(set(df.columns)):
        logging.warning("No predictions available for metrics (samples=%d, cols=%s)", n, list(df.columns))
        return {
            'samples': n,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'fnr': 0.0,
            'fpr': 0.0,
            'stage2_rate': 0.0,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
        }

    tp = int(((df['prediction'] == 1) & (df['label'] == 1)).sum())
    tn = int(((df['prediction'] == 0) & (df['label'] == 0)).sum())
    fp = int(((df['prediction'] == 1) & (df['label'] == 0)).sum())
    fn = int(((df['prediction'] == 0) & (df['label'] == 1)).sum())
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    s2_rate = float((df['stage_used'] == 2).mean()) if 'stage_used' in df.columns and n else 0.0
    return {
        'samples': n,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fnr': fnr,
        'fpr': fpr,
        'stage2_rate': s2_rate,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def eval_manifest(
    engine: CascadeEngine,
    manifest: Path,
    out_csv: Path,
    show_progress: bool = True,
    use_existing: bool = False,
) -> Dict[str, float]:
    """Run inference on a manifest or reuse existing predictions CSV when requested."""
    if use_existing and out_csv.exists():
        try:
            preds = pd.read_csv(out_csv)
            logger.info("Reusing existing predictions: %s (rows=%d)", out_csv, len(preds))
            return compute_metrics(preds)
        except Exception as exc:
            logger.warning("Failed to read existing predictions (%s): %s. Recomputing...", out_csv, exc)

    df = pd.read_csv(manifest)
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]

    rows: List[Dict] = []
    iterator = df.itertuples(index=False)
    iterator = tqdm(iterator, total=len(df), desc=f"Infer {manifest.name}") if show_progress else iterator

    for row in iterator:
        # Accept common path column names
        path = getattr(row, 'image_path', None)
        if path is None and hasattr(row, 'path'):
            path = getattr(row, 'path')
        label = getattr(row, 'label', None)
        if path is None or label is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Skip unreadable: %s", path)
            continue
        res = engine.infer_image(img)
        rows.append({'path': path, 'label': int(label), **res})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        preds = pd.DataFrame(rows)
    else:
        # Write an empty file with expected columns for downstream tools
        preds = pd.DataFrame(columns=['path','label','prediction','confidence','stage_used','stage1_confidence','stage2_confidence'])
    preds.to_csv(out_csv, index=False)
    return compute_metrics(preds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate cascade across multiple datasets')
    p.add_argument('--stage1-ckpt', required=True)
    p.add_argument('--stage2-ckpt', required=True)
    p.add_argument('--thresholds', required=True)
    p.add_argument('--stage1-model', default='mobilenetv4_hybrid_medium')
    p.add_argument('--stage2-model', default='tf_efficientnetv2_b0')
    p.add_argument('--device', default='auto')
    p.add_argument('--stage1-size', type=int, default=256)
    p.add_argument('--stage2-size', type=int, default=384)
    p.add_argument('--stage2-temperature', type=float, default=1.0,
                   help='Temperature for Stage 2 calibration (logits/T before sigmoid; default: 1.0)')
    p.add_argument('--stage2-decision-threshold', type=float, default=None,
                   help='Decision threshold for Stage 2 (overrides thresholds JSON if provided). Default: use config value or 0.5')
    p.add_argument('--stage2-tta', default='none', choices=['none','hflip'],
                   help='Stage 2 test-time augmentation (default: none)')
    p.add_argument('--datasets', nargs='+', default=['celebdf_v2','faceforensics','deeperforensics','dfdc'])
    p.add_argument('--splits', nargs='+', default=['val'], choices=['val','test'])
    p.add_argument('--output-dir', default='outputs/stage4/evals')
    p.add_argument('--no-progress', action='store_true', default=False)
    p.add_argument('--resume', action='store_true', default=False,
                   help='Reuse existing predictions CSVs when available (skip re-inference).')
    p.add_argument('--stage2-only', action='store_true', default=False,
                   help='Diagnostic mode: escalate all samples to Stage 2 by using thresholds low=0.0, high=1.0.')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally override thresholds with Stage2-only diagnostics
    thresholds_path = args.thresholds
    if args.stage2_only:
        try:
            stage2_only_cfg = {"low_thresh": 0.0, "high_thresh": 1.0, "escalation_rate": 1.0}
            tmp_cfg_path = out_dir / 'stage2_only_thresholds.json'
            tmp_cfg_path.write_text(json.dumps(stage2_only_cfg, indent=2))
            logger.info("Stage2-only mode enabled: using thresholds %s", tmp_cfg_path)
            thresholds_path = str(tmp_cfg_path)
        except Exception as exc:
            logger.error("Failed to create Stage2-only thresholds: %s", exc)
            return 1

    engine = CascadeEngine(
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        thresholds_path=thresholds_path,
        stage1_model=args.stage1_model,
        stage2_model=args.stage2_model,
        device=args.device,
        stage1_size=args.stage1_size,
        stage2_size=args.stage2_size,
        stage2_temperature=args.stage2_temperature,
        stage2_decision_threshold=args.stage2_decision_threshold,
        stage2_tta=args.stage2_tta,
    )

    summary_rows: List[Dict] = []

    for ds in args.datasets:
        ds_cfg = DEFAULT_MANIFESTS.get(ds, {})
        for split in args.splits:
            man_path = ds_cfg.get(split)
            if not man_path:
                logger.warning("No manifest configured for %s/%s, skipping", ds, split)
                continue
            man = Path(man_path)
            if not man.exists():
                logger.warning("Manifest not found: %s, skipping", man)
                continue
            out_csv = out_dir / f"{ds}_{split}_preds.csv"
            logger.info("Evaluating %s/%s (%s)", ds, split, man)
            m = eval_manifest(engine, man, out_csv, show_progress=not args.no_progress, use_existing=args.resume)
            m.update({'dataset': ds, 'split': split, 'file': str(out_csv)})
            summary_rows.append(m)

    if not summary_rows:
        logger.error("No evaluations were run. Check dataset names/splits/manifests.")
        return 1

    summary = pd.DataFrame(summary_rows).sort_values(['dataset','split'])
    summary.to_csv(out_dir / 'summary.csv', index=False)
    logger.info("Summary written: %s", out_dir / 'summary.csv')
    try:
        # Pretty print
        with pd.option_context('display.max_columns', None):
            print(summary.to_string(index=False))
    except Exception:
        pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
