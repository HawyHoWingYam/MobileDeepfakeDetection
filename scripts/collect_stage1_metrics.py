#!/usr/bin/env python3
"""
Collect Stage 1 evaluation metrics across datasets and emit CSV + LaTeX table.

Looks under outputs/stage1/evaluation/<dataset>/evaluation_results.json
for datasets: faceforensics, celebdf_v2, dfdc, deeperforensics (by default).

Outputs:
- outputs/stage1/evaluation/stage1_eval_summary.csv
- paper/generated/stage1_eval_summary.tex

Run from repo root.
"""

import json
import os
from pathlib import Path

DATASETS = [
    "faceforensics",
    "celebdf_v2",
    "dfdc",
    "deeperforensics",
]

OUTPUT_ROOT = Path("outputs/stage1/evaluation")
TEX_OUT = Path("paper/generated/stage1_eval_summary.tex")
CSV_OUT = OUTPUT_ROOT / "stage1_eval_summary.csv"


def load_metrics(ds: str):
    res_path = OUTPUT_ROOT / ds / "evaluation_results.json"
    if not res_path.exists():
        return None
    with open(res_path, "r") as f:
        data = json.load(f)
    m = data.get("performance_metrics", {})
    # FNR is 1 - recall if not provided
    recall = float(m.get("recall", 0.0))
    fnr = 1.0 - recall
    row = {
        "dataset": ds,
        "auc": float(m.get("auc", 0.0)),
        "f1": float(m.get("f1", 0.0)),
        "acc": float(m.get("accuracy", 0.0)),
        "prec": float(m.get("precision", 0.0)),
        "rec": recall,
        "fnr": fnr,
        "total": int(data.get("dataset_info", {}).get("total_samples", 0)),
        "real": int(data.get("dataset_info", {}).get("real_samples", 0)),
        "fake": int(data.get("dataset_info", {}).get("fake_samples", 0)),
        "manifest": str(data.get("dataset_info", {}).get("test_manifest", "")),
    }
    return row


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main():
    rows = []
    for ds in DATASETS:
        r = load_metrics(ds)
        if r:
            rows.append(r)

    if not rows:
        print("No evaluation_results.json files found. Run the batch evaluation first.")
        return

    # CSV
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w") as f:
        f.write("dataset,auc,f1,acc,prec,rec,fnr,total,real,fake,manifest\n")
        for r in rows:
            f.write(
                f"{r['dataset']},{fmt(r['auc'])},{fmt(r['f1'])},{fmt(r['acc'])},{fmt(r['prec'])},{fmt(r['rec'])},{fmt(r['fnr'])},{r['total']},{r['real']},{r['fake']},{r['manifest']}\n"
            )

    # Macro (unweighted) and micro (weighted by total)
    macro = {
        k: sum(r[k] for r in rows) / len(rows)
        for k in ["auc", "f1", "acc", "prec", "rec", "fnr"]
    }
    total_sum = sum(r["total"] for r in rows)
    if total_sum > 0:
        micro = {
            k: sum(r[k] * r["total"] for r in rows) / total_sum
            for k in ["auc", "f1", "acc", "prec", "rec", "fnr"]
        }
    else:
        micro = macro.copy()

    # LaTeX table
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    def ltx(text: str) -> str:
        # Minimal LaTeX escaping for dataset names
        return str(text).replace('_', r'\_')

    with open(TEX_OUT, "w") as f:
        f.write("% Auto-generated Stage 1 per-dataset evaluation (calibrated)\n")
        f.write("\\begin{table}[h]\n  \\centering\n")
        f.write("  \\caption{Stage~1 per-dataset evaluation (calibrated).}\\label{tab:stage1_per_dataset}\n  {\\small\n")
        f.write("  \\begin{tabular}{lcccccc}\\toprule\n")
        f.write("  Dataset & AUC & F1 & Acc & Prec & Rec & FNR \\\\ \\midrule\n")
        for r in rows:
            f.write(
                f"  {ltx(r['dataset'])} & {fmt(r['auc'])} & {fmt(r['f1'])} & {fmt(r['acc'])} & {fmt(r['prec'])} & {fmt(r['rec'])} & {fmt(r['fnr'])} \\\\ \n"
            )
        f.write(
            f"  \\midrule\n  Macro Avg & {fmt(macro['auc'])} & {fmt(macro['f1'])} & {fmt(macro['acc'])} & {fmt(macro['prec'])} & {fmt(macro['rec'])} & {fmt(macro['fnr'])} \\\\ \n"
        )
        f.write(
            f"  Micro Avg & {fmt(micro['auc'])} & {fmt(micro['f1'])} & {fmt(micro['acc'])} & {fmt(micro['prec'])} & {fmt(micro['rec'])} & {fmt(micro['fnr'])} \\\\ \n"
        )
        f.write("  \\bottomrule\n  \\end{tabular}}\n\\end{table}\n")

    print(f"Wrote: {CSV_OUT}")
    print(f"Wrote: {TEX_OUT}")


if __name__ == "__main__":
    main()
