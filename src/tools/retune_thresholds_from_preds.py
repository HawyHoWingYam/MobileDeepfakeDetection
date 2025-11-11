#!/usr/bin/env python3
"""
Retune cascade thresholds using existing per-image preds CSV (no re-inference).

Inputs: a CSV produced by cascade_infer/eval (columns include at least:
  path,label,prediction,confidence,stage_used,stage1_confidence,stage2_confidence)

For each (low, high) pair, simulate cascade decisions:
  if stage1_conf <= low -> REAL (0)
  elif stage1_conf >= high -> FAKE (1)
  else -> use stage2_confidence >= 0.5

Saves metrics_grid.csv and best_config.json. Optionally evaluates the selected
thresholds on a separate test preds CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retune cascade thresholds from preds CSV")
    p.add_argument("--val-preds", required=True, help="Validation preds CSV (with stage1_confidence)")
    p.add_argument("--test-preds", default=None, help="Optional test preds CSV for reporting")
    p.add_argument("--low-start", type=float, default=0.01)
    p.add_argument("--low-stop", type=float, default=0.05)
    p.add_argument("--low-step", type=float, default=0.01)
    p.add_argument("--high-start", type=float, default=0.50)
    p.add_argument("--high-stop", type=float, default=0.70)
    p.add_argument("--high-step", type=float, default=0.01)
    p.add_argument("--opt-metric", default="f1", choices=["f1","recall","accuracy","auc"], help="Primary metric")
    p.add_argument("--min-recall", type=float, default=0.0, help="Constraint: minimum recall")
    p.add_argument("--min-precision", type=float, default=0.0, help="Constraint: minimum precision")
    p.add_argument("--s2-start", type=float, default=0.30, help="Stage 2 decision threshold start (inclusive)")
    p.add_argument("--s2-stop", type=float, default=0.70, help="Stage 2 decision threshold stop (inclusive)")
    p.add_argument("--s2-step", type=float, default=0.02, help="Stage 2 decision threshold step")
    p.add_argument("--output-dir", default="outputs/stage4/run_re_tune_from_preds")
    p.add_argument("--no-progress", action="store_true", default=False,
                   help="Disable progress bar during grid search")
    return p.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    n = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"samples": n, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fnr": fnr, "fpr": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _simulate(df: pd.DataFrame, low: float, high: float, s2_thr: float) -> Tuple[np.ndarray, float]:
    # Prepare columns
    s1 = pd.to_numeric(df["stage1_confidence"], errors="coerce").values
    # Stage 2 confidence: prefer explicit col; fallback to 'confidence' for stage_used==2
    if "stage2_confidence" in df.columns:
        s2 = pd.to_numeric(df["stage2_confidence"], errors="coerce").values
    else:
        s2 = np.full_like(s1, np.nan, dtype=float)
    if "stage_used" in df.columns:
        used2 = (df["stage_used"].astype(str) == "2").values
    else:
        used2 = np.zeros_like(s1, dtype=bool)
    conf = pd.to_numeric(df.get("confidence", pd.Series(np.nan, index=df.index)), errors="coerce").values
    s2 = np.where(np.isnan(s2) & used2, conf, s2)

    # Cascade decision
    pred = np.empty_like(s1, dtype=int)
    # S1 direct
    pred[s1 <= low] = 0
    pred[s1 >= high] = 1
    # S2 path
    mid = (s1 > low) & (s1 < high)
    # If s2 missing for a mid sample, fallback to stage1 threshold midpoint decision to be safe
    s2_mid = s2[mid]
    s2_missing = np.isnan(s2_mid)
    s2_mid_bin = (s2_mid >= s2_thr).astype(int)
    # For missing s2, consider 1 if s1 >= 0.5 else 0 (weak fallback)
    s1_mid = s1[mid]
    s2_mid_bin[s2_missing] = (s1_mid[s2_missing] >= 0.5).astype(int)
    pred[mid] = s2_mid_bin

    escalation_rate = float(mid.mean())
    return pred, escalation_rate


def _grid(values: Dict[str, np.ndarray], low_vals: np.ndarray, high_vals: np.ndarray, s2_vals: np.ndarray,
          min_recall: float, min_precision: float, opt_metric: str, show_progress: bool = True) -> Tuple[pd.DataFrame, Dict]:
    y_true = values["label"]
    rows = []
    best = None
    total = int(len(low_vals) * len(high_vals) * len(s2_vals))
    pbar = tqdm(total=total, desc="Retuning thresholds", unit="comb", disable=not show_progress)
    for low in low_vals:
        for high in high_vals:
            if not (0.0 <= low < high <= 1.0):
                # Skip invalid pair but still progress for s2 loop length
                for _ in s2_vals:
                    pbar.update(1)
                continue
            for s2_thr in s2_vals:
                y_pred, esc = _simulate(values["df"], low, high, float(s2_thr))
                m = _metrics(y_true, y_pred)
                m.update({
                    "low": float(low),
                    "high": float(high),
                    "stage2_threshold": float(s2_thr),
                    "escalation_rate": float(esc),
                })
                if m["recall"] + 1e-9 < min_recall or m["precision"] + 1e-9 < min_precision:
                    rows.append(m)
                    pbar.update(1)
                    continue
                if best is None or m.get(opt_metric, 0.0) > best.get(opt_metric, 0.0):
                    best = dict(m)
                    # Update postfix with current best
                    if show_progress:
                        pbar.set_postfix({
                            "best_f1": f"{best.get('f1',0):.4f}",
                            "best_rec": f"{best.get('recall',0):.4f}",
                            "low": f"{best.get('low',0):.3f}",
                            "high": f"{best.get('high',0):.3f}",
                            "s2": f"{best.get('stage2_threshold',0):.2f}",
                        })
                rows.append(m)
                pbar.update(1)
    pbar.close()
    grid = pd.DataFrame(rows)
    return grid, (best or {})


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(args.val_preds)
    if not {"label","stage1_confidence"}.issubset(set(val.columns)):
        print("Validation preds must contain label and stage1_confidence")
        return 1
    values = {"label": val["label"].to_numpy().astype(int), "df": val}

    low_vals = np.arange(args.low_start, args.low_stop + 1e-8, args.low_step)
    high_vals = np.arange(args.high_start, args.high_stop + 1e-8, args.high_step)
    s2_vals = np.arange(args.s2_start, args.s2_stop + 1e-8, args.s2_step)
    show_progress = not bool(args.no_progress)
    grid, best = _grid(values, low_vals, high_vals, s2_vals, args.min_recall, args.min_precision, args.opt_metric, show_progress=show_progress)
    grid.to_csv(out_dir / "metrics_grid.csv", index=False)
    if best:
        (out_dir / "best_config.json").write_text(json.dumps({
            "low_thresh": best["low"],
            "high_thresh": best["high"],
            "stage2_threshold": best.get("stage2_threshold", 0.5),
            "metrics": best,
            "escalation_rate": best.get("escalation_rate", 0.0),
            "source": str(Path(args.val_preds).resolve()),
        }, indent=2))

    # Optional: Evaluate chosen thresholds on test preds
    if args.test_preds and best:
        test = pd.read_csv(args.test_preds)
        if {"label","stage1_confidence"}.issubset(set(test.columns)):
            s2_thr = float(best.get("stage2_threshold", 0.5))
            y_pred, esc = _simulate(test, best["low"], best["high"], s2_thr)
            m = _metrics(test["label"].to_numpy().astype(int), y_pred)
            m.update({
                "escalation_rate": esc,
                "low": best["low"],
                "high": best["high"],
                "stage2_threshold": s2_thr,
            })
            (out_dir / "test_metrics.json").write_text(json.dumps(m, indent=2))

    print("Done. Grid saved to:", out_dir / "metrics_grid.csv")
    if best:
        print("Best config:", json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
