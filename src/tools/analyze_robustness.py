#!/usr/bin/env python3
"""
Analyze robustness of the cascade (or Stage2-only) under common perturbations.

Generates CSV metrics and plots (AUC/F1/Accuracy/FNR vs perturbation strength)
for multiple distortion types, and writes LaTeX include snippets for the paper.

Usage (example):
  python -m src.tools.analyze_robustness \
    --stage1-ckpt outputs/stage1/run_20251023_034316/best_model.pth \
    --stage2-ckpt outputs/stage3/run_20251030_081127/best_model.pth \
    --thresholds outputs/stage4/run_20251030_185050/best_config.json \
    --manifest manifests/deepfake_eval_2024_val.csv \
    --output-dir outputs/stage5/robustness \
    --stage1-model mobilenetv4_hybrid_medium \
    --stage2-model tf_efficientnetv2_b3 \
    --stage1-size 256 --stage2-size 384 \
    --types jpeg,gaussian,motion,brightness \
    --jpeg-qualities 95,80,60,40,20 \
    --gaussian-sigmas 2,4,8,12 \
    --motion-kernels 3,5,9,13 \
    --brightness-factors 0.7,0.85,1.15,1.3 \
    --sample-limit 5000 --device auto --no-progress
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from tools.cascade_infer import CascadeEngine  # noqa: E402


def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def parse_list_ints(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(',') if x.strip()]


def distort_image(img: np.ndarray, kind: str, level: float | int) -> np.ndarray:
    if kind == 'jpeg':
        quality = int(level)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality))]
        ok, enc = cv2.imencode('.jpg', img, encode_param)
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else img
    if kind == 'gaussian':
        sigma = float(level)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out
    if kind == 'motion':
        k = int(level)
        k = max(3, k | 1)  # odd >=3
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(img, -1, kernel)
        return out
    if kind == 'brightness':
        alpha = float(level)  # brightness/contrast factor
        out = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        return out
    if kind == 'downscale':
        scale = float(level)
        h, w = img.shape[:2]
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return out
    if kind == 'gamma':
        gamma = float(level)
        gamma = max(0.01, gamma)
        # normalize to [0,1], apply gamma, scale back to [0,255]
        x = (img.astype(np.float32) / 255.0) ** gamma
        out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return out
    return img


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = {'label', 'prediction'}
    n = int(len(df))
    if n == 0 or not required.issubset(df.columns):
        return {'samples': n, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'fnr': 0.0, 'fpr': 0.0}
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
    # Optional: stage2 escalation rate if present
    s2_rate = 0.0
    if 'stage_used' in df.columns:
        try:
            s2_rate = float((df['stage_used'] == 2).mean()) if n else 0.0
        except Exception:
            s2_rate = 0.0
    return {
        'samples': n,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fnr': fnr,
        'fpr': fpr,
        'stage2_rate': s2_rate,
    }


def run_eval(
    engine: CascadeEngine,
    manifest: Path,
    kind: str,
    levels: List[float | int],
    sample_limit: int = 0,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict[float, Dict[str, float]]]:
    import tqdm as _tqdm
    df = pd.read_csv(manifest)
    rows: List[Dict[str, any]] = []
    # Build a cache of readable images up to sample_limit (if provided)
    cache: List[Tuple[str, int, np.ndarray | None]] = []
    max_needed = int(sample_limit) if (sample_limit and sample_limit > 0) else len(df)
    it_all = _tqdm.tqdm(df.itertuples(index=False), total=len(df), desc=f"cache:{kind}") if show_progress else df.itertuples(index=False)
    base_root = manifest.parent.parent  # assume manifests/ under repo root
    for r in it_all:
        if len(cache) >= max_needed:
            break
        path = getattr(r, 'image_path', None)
        label = int(getattr(r, 'label', -1))
        if path is None or label not in (0, 1):
            continue
        # Try read as-is, else try relative to repo root
        img = cv2.imread(str(path))
        if img is None:
            alt = (base_root / str(path)).as_posix()
            img = cv2.imread(alt)
        if img is None:
            continue
        cache.append((str(path), label, img))
    if not cache:
        print(f"[WARN] No readable images found for {kind}. Skipping.")
        return pd.DataFrame(), {}

    metrics_per_level: Dict[float, Dict[str, float]] = {}
    for lv in levels:
        preds = []
        iterable = None
        try:
            import tqdm as _tqdm
            iterable = _tqdm.tqdm(cache, total=len(cache), desc=f"{kind}[{lv}]", unit="img") if show_progress else cache
        except Exception:
            iterable = cache

        for path, label, img in iterable:
            if img is None:
                continue
            dimg = distort_image(img, kind, lv)
            res = engine.infer_image(dimg)
            preds.append({'path': path, 'label': label, **res, 'level': lv})

        if not preds:
            continue

        pdf = pd.DataFrame(preds)
        m = compute_metrics(pdf.rename(columns={'prediction': 'prediction'}))
        metrics_per_level[float(lv)] = m
        rows.extend(preds)

        # Console summary per level (helps long CPU runs)
        try:
            print(f"robust:{kind} level={lv} samples={len(preds)} f1={m['f1']:.4f} acc={m['accuracy']:.4f} fnr={m['fnr']:.4f}")
        except Exception:
            pass

    return pd.DataFrame(rows), metrics_per_level


def plot_metric_curve(metrics: Dict[float, Dict[str, float]], metric: str, title: str, out_path: Path) -> None:
    xs = sorted(metrics.keys())
    ys = [metrics[x][metric] for x in xs]
    plt.figure(figsize=(4.2, 3.2))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('Strength')
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_tex_snippet(fig_paths: Dict[str, Path], out_tex: Path) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with out_tex.open('w') as f:
        f.write('% Auto-generated robustness figures\n')
        f.write('\\begin{figure}[h]\n  \\centering\n')
        for name, p in fig_paths.items():
            f.write(f"  \\includegraphics[width=.32\\textwidth]{{{p.as_posix()}}}% {name}\n")
        f.write('\\\caption{Robustness: metric curves under common perturbations.}\n')
        f.write('\\\label{fig:robustness_curves}\n\\end{figure}\n')


def main() -> int:
    ap = argparse.ArgumentParser(description='Analyze cascade robustness under perturbations')
    ap.add_argument('--stage1-ckpt', required=True)
    ap.add_argument('--stage2-ckpt', required=True)
    ap.add_argument('--thresholds', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-dir', default='outputs/stage5/robustness')
    ap.add_argument('--stage1-model', default='mobilenetv4_hybrid_medium')
    ap.add_argument('--stage2-model', default='tf_efficientnetv2_b3')
    ap.add_argument('--stage1-size', type=int, default=256)
    ap.add_argument('--stage2-size', type=int, default=384)
    ap.add_argument('--stage2-temperature', type=float, default=1.0)
    ap.add_argument('--stage2-threshold', type=float, default=None)
    ap.add_argument('--types', default='jpeg,gaussian,motion,brightness,downscale,gamma')
    ap.add_argument('--jpeg-qualities', default='95,80,60,40,20')
    ap.add_argument('--gaussian-sigmas', default='2,4,8,12')
    ap.add_argument('--motion-kernels', default='3,5,9,13')
    ap.add_argument('--brightness-factors', default='0.7,0.85,1.15,1.3')
    ap.add_argument('--downscale-scales', default='0.75,0.5,0.33')
    ap.add_argument('--gamma-values', default='0.7,0.85,1.15,1.3')
    ap.add_argument('--sample-limit', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--no-progress', action='store_true', default=False)
    ap.add_argument('--stage2-only', action='store_true', default=False, help='Escalate all samples to Stage 2 by setting thresholds low=0, high=1')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresh_path = Path(args.thresholds)
    if args.stage2_only:
        # Create a temporary thresholds file to escalate all to Stage 2
        tmp = out_dir / 'stage2_only_thresholds.json'
        tmp.write_text(json.dumps({'low_thresh': 0.0, 'high_thresh': 1.0, 'escalation_rate': 1.0}, indent=2))
        thresh_path = tmp

    engine = CascadeEngine(
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        thresholds_path=str(thresh_path),
        stage1_model=args.stage1_model,
        stage2_model=args.stage2_model,
        device=args.device,
        stage1_size=int(args.stage1_size),
        stage2_size=int(args.stage2_size),
        stage2_temperature=float(args.stage2_temperature),
        stage2_decision_threshold=args.stage2_threshold,
    )

    kinds = [t.strip() for t in args.types.split(',') if t.strip()]
    levels_map: Dict[str, List[float | int]] = {
        'jpeg': parse_list_ints(args.jpeg_qualities),
        'gaussian': parse_list_floats(args.gaussian_sigmas),
        'motion': parse_list_ints(args.motion_kernels),
        'brightness': parse_list_floats(args.brightness_factors),
        'downscale': parse_list_floats(args.downscale_scales),
        'gamma': parse_list_floats(args.gamma_values),
    }

    manifest = Path(args.manifest)
    fig_paths: Dict[str, Path] = {}

    for kind in kinds:
        levels = levels_map.get(kind, [])
        if not levels:
            continue
        pred_csv = out_dir / f'{kind}_preds.csv'
        _, metrics = run_eval(
            engine,
            manifest=manifest,
            kind=kind,
            levels=levels,
            sample_limit=max(0, int(args.sample_limit)),
            show_progress=not args.no_progress,
        )
        # Write metrics CSV (incrementally)
        mrows = []
        for lv in sorted(metrics):
            row = {'level': lv, **metrics[lv]}
            mrows.append(row)
        mdf = pd.DataFrame(mrows)
        out_csv = out_dir / f'{kind}_metrics.csv'
        mdf.to_csv(out_csv, index=False)

        # Plot AUC/F1/Acc/FNR curves (AUC not available from cascade; use F1/Acc/FNR)
        for metric, title_suffix in [('f1', 'F1'), ('accuracy', 'Accuracy'), ('fnr', 'FNR')]:
            plot_metric_curve(metrics, metric, f'{kind} - {title_suffix}', out_path=out_dir / f'{kind}_{metric}.png')
        fig_paths[kind] = out_dir / f'{kind}_f1.png'

    # Write LaTeX snippet for inclusion
    write_tex_snippet(fig_paths, out_tex=Path('MobileDeepfakeDetection/paper/generated/robustness_figs.tex'))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
