#!/usr/bin/env python3
"""
Robustness threshold sweep: generate FNR vs Stage-2 rate views under optional distortions.

For a given manifest (e.g., Deepfake-Eval-2024 val), optionally apply a distortion family
and level to all images, then compute Stage 1 probability p1 and Stage 2 probability p2
for each sample (independent of thresholds). For a grid of (low, high) pairs, compute
final cascade predictions and summarize metrics, including Stage-2 escalation rate.

Outputs:
- CSV with grid results: out_dir/fnr_vs_s2_{kind}_{level}.csv
- Scatter plot FNR vs S2 Rate: out_dir/fnr_vs_s2_{kind}_{level}.png
- LaTeX snippet including available figures: paper/generated/robustness_tradeoff.tex

Usage example:
  PYTHONPATH=MobileDeepfakeDetection/src python -m tools.robustness_threshold_sweep \
    --stage1-ckpt MobileDeepfakeDetection/outputs/stage1/run_.../best_model.pth \
    --stage2-ckpt MobileDeepfakeDetection/outputs/stage3/run_.../best_model.pth \
    --manifest MobileDeepfakeDetection/manifests/deepfake_eval_2024_val.csv \
    --kinds jpeg,gaussian --jpeg-qualities 95,80 --gaussian-sigmas 4,12 \
    --lows 0.01,0.03,0.05,0.1 --highs 0.5,0.55,0.6,0.7 \
    --sample-limit 500 --output-dir MobileDeepfakeDetection/outputs/stage5/robustness
"""

from __future__ import annotations

import argparse
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
from tools.analyze_robustness import distort_image, parse_list_floats, parse_list_ints  # noqa: E402


def parse_list_floats_str(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def compute_p1_p2(engine: CascadeEngine, img_bgr: np.ndarray) -> Tuple[float, float]:
    import torch
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x1 = engine.tr1(image=img_rgb)['image'].unsqueeze(0).to(engine.device)
    with torch.no_grad():
        logit1 = engine.s1(x1).view(-1)
        p1 = torch.sigmoid(logit1).item()
    x2 = engine.tr2(image=img_rgb)['image'].unsqueeze(0).to(engine.device)
    with torch.no_grad():
        logit2 = engine.s2(x2).view(-1)
        if engine.stage2_temperature and engine.stage2_temperature != 1.0:
            logit2 = logit2 / engine.stage2_temperature
        p2 = torch.sigmoid(logit2).item()
    return float(p1), float(p2)


def sweep_grid(labels: np.ndarray, p1: np.ndarray, p2: np.ndarray, lows: List[float], highs: List[float], s2_thresh: float = 0.5) -> pd.DataFrame:
    rows = []
    y = labels.astype(int)
    for lo in lows:
        for hi in highs:
            if lo >= hi:
                continue
            accept_real = p1 <= lo
            accept_fake = p1 >= hi
            escalate = ~(accept_real | accept_fake)
            final_pred = np.zeros_like(y)
            final_pred[accept_real] = 0
            final_pred[accept_fake] = 1
            if escalate.any():
                final_pred[escalate] = (p2[escalate] >= s2_thresh).astype(int)
            # metrics
            n = len(y)
            tp = int(((final_pred == 1) & (y == 1)).sum())
            tn = int(((final_pred == 0) & (y == 0)).sum())
            fp = int(((final_pred == 1) & (y == 0)).sum())
            fn = int(((final_pred == 0) & (y == 1)).sum())
            acc = (tp + tn)/n if n else 0.0
            prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
            fnr = fn/(fn+tp) if (fn+tp)>0 else 0.0
            s2_rate = float(escalate.mean()) if n else 0.0
            rows.append({'low': lo, 'high': hi, 'f1': f1, 'accuracy': acc, 'fnr': fnr, 'stage2_rate': s2_rate,
                         'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})
    return pd.DataFrame(rows)


def write_tradeoff_figure(df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(4.2, 3.2))
    plt.scatter(df['stage2_rate'], df['fnr'], s=12, alpha=0.7)
    plt.xlabel('Stage-2 rate')
    plt.ylabel('FNR')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def append_tradeoff_snippet(figs: List[Tuple[str, Path]]) -> None:
    snip = Path('MobileDeepfakeDetection/paper/generated/robustness_tradeoff.tex')
    snip.parent.mkdir(parents=True, exist_ok=True)
    with snip.open('w') as f:
        f.write('% Auto-generated robustness trade-off figures\n')
        f.write('\\begin{figure}[h]\n  \\centering\n')
        for name, p in figs:
            f.write(f"  \\includegraphics[width=.32\\textwidth]{{{p.as_posix()}}}% {name}\n")
        f.write('\\\caption{FNR vs Stage-2 rate under threshold sweeps.}\n')
        f.write('\\\label{fig:robustness_tradeoff}\n\\end{figure}\n')


def main() -> int:
    ap = argparse.ArgumentParser(description='Robustness threshold sweep: FNR vs Stage-2 rate')
    ap.add_argument('--stage1-ckpt', required=True)
    ap.add_argument('--stage2-ckpt', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-dir', default='outputs/stage5/robustness')
    ap.add_argument('--stage1-model', default='mobilenetv4_hybrid_medium.ix_e550_r256_in1k')
    ap.add_argument('--stage2-model', default='efficientnetv2_b3.in21k_ft_in1k')
    ap.add_argument('--stage1-size', type=int, default=256)
    ap.add_argument('--stage2-size', type=int, default=384)
    ap.add_argument('--stage2-temperature', type=float, default=1.0)
    ap.add_argument('--stage2-threshold', type=float, default=0.5)
    ap.add_argument('--kinds', default='jpeg,gaussian,motion,brightness,downscale,gamma')
    ap.add_argument('--jpeg-qualities', default='95,80,60,40,20')
    ap.add_argument('--gaussian-sigmas', default='2,4,8,12')
    ap.add_argument('--motion-kernels', default='3,5,9,13')
    ap.add_argument('--brightness-factors', default='0.7,0.85,1.15,1.3')
    ap.add_argument('--downscale-scales', default='0.75,0.5,0.33')
    ap.add_argument('--gamma-values', default='0.7,0.85,1.15,1.3')
    ap.add_argument('--lows', default='0.01,0.03,0.05,0.1')
    ap.add_argument('--highs', default='0.5,0.55,0.6,0.7')
    ap.add_argument('--sample-limit', type=int, default=500)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary thresholds JSON (not used for decisions; sweep is offline)
    tmp_thr = out_dir / 'tmp_thresholds.json'
    tmp_thr.write_text('{"low_thresh": 0.0, "high_thresh": 1.0, "escalation_rate": 1.0}')

    engine = CascadeEngine(
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        thresholds_path=str(tmp_thr),  # required by engine; sweep uses offline p1/p2
        stage1_model=args.stage1_model,
        stage2_model=args.stage2_model,
        device=args.device,
        stage1_size=args.stage1_size,
        stage2_size=args.stage2_size,
        stage2_temperature=args.stage2_temperature,
        stage2_decision_threshold=args.stage2_threshold,
    )

    kinds = [x.strip() for x in args.kinds.split(',') if x.strip()]
    levels_map: Dict[str, List[float]] = {
        'jpeg': [float(x) for x in parse_list_ints(args.jpeg_qualities)],
        'gaussian': parse_list_floats(args.gaussian_sigmas),
        'motion': [float(x) for x in parse_list_ints(args.motion_kernels)],
        'brightness': parse_list_floats(args.brightness_factors),
        'downscale': parse_list_floats(args.downscale_scales),
        'gamma': parse_list_floats(args.gamma_values),
    }
    lows = parse_list_floats_str(args.lows)
    highs = parse_list_floats_str(args.highs)

    # Load manifest and cache readable images (up to sample-limit)
    df = pd.read_csv(args.manifest)
    cache = []
    for _, r in df.iterrows():
        if len(cache) >= max(1, args.sample_limit):
            break
        path = str(r['image_path'])
        label = int(r['label']) if 'label' in r else -1
        if label not in (0, 1):
            continue
        img = cv2.imread(path)
        if img is None:
            alt = (Path(args.manifest).parent.parent / path).as_posix()
            img = cv2.imread(alt)
        if img is None:
            continue
        cache.append((path, label, img))
    if not cache:
        print('[WARN] No readable images found; aborting sweep')
        return 0

    figs: List[Tuple[str, Path]] = []
    for kind in kinds:
        levels = levels_map.get(kind, [])
        for lv in levels:
            labels = []
            p1_list = []
            p2_list = []
            for path, label, img in cache:
                dimg = distort_image(img, kind, lv)
                p1, p2 = compute_p1_p2(engine, dimg)
                labels.append(label)
                p1_list.append(p1)
                p2_list.append(p2)
            labels_np = np.array(labels)
            p1_np = np.array(p1_list, dtype=float)
            p2_np = np.array(p2_list, dtype=float)
            grid = sweep_grid(labels_np, p1_np, p2_np, lows, highs, s2_thresh=float(args.stage2_threshold))
            grid_out = out_dir / f'fnr_vs_s2_{kind}_{lv}.csv'
            grid.to_csv(grid_out, index=False)
            fig_out = out_dir / f'fnr_vs_s2_{kind}_{lv}.png'
            write_tradeoff_figure(grid, f'{kind} level={lv}', fig_out)
            figs.append((f'{kind}_{lv}', fig_out))

    append_tradeoff_snippet(figs)
    print('Trade-off figures generated and snippet written to paper/generated/robustness_tradeoff.tex')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
