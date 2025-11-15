#!/usr/bin/env python3
"""
Generate LaTeX tables for the paper from existing outputs. Writes to
MobileDeepfakeDetection/paper/generated/*.tex and reuses plots already under outputs/.

Usage:
  python -m src.tools.generate_paper_assets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import csv
import os

ROOT = Path(__file__).resolve().parents[2]
PAPER_GEN = ROOT / 'paper' / 'generated'


def read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def find_best_stage1() -> Tuple[Path, float]:
    best_auc = -1.0
    best_run = None
    for js in (ROOT / 'outputs' / 'stage1').glob('run_*/training_summary.json'):
        d = read_json(js)
        auc = float(d.get('best_validation_auc', -1.0))
        if auc > best_auc:
            best_auc = auc
            best_run = js.parent
    return best_run or Path('.'), best_auc


def find_best_stage2() -> Tuple[Path, Dict[str, float]]:
    best_auc = -1.0
    best = {}
    best_run = None
    for js in (ROOT / 'outputs' / 'stage3').glob('run_*/evaluation_summary.json'):
        d = read_json(js)
        m = d.get('metrics') or {}
        auc = float(m.get('auc', -1.0))
        if auc > best_auc:
            best_auc = auc
            best = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
            best_run = js.parent
    return best_run or Path('.'), best


def select_best_cascade(n: int = 2) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for js in (ROOT / 'outputs' / 'stage4').glob('run_*/best_config.json'):
        d = read_json(js)
        try:
            rec = {
                'low': float(d['low_thresh']),
                'high': float(d['high_thresh']),
                'auc': float(d['metrics']['auc']),
                'f1': float(d['metrics']['f1']),
                'fnr': float(d['metrics']['fnr']),
                'escalation_rate': float(d.get('escalation_rate', 0.0)),
                'path': js.as_posix(),
            }
            records.append(rec)
        except Exception:
            continue
    # Sort by FNR asc, then escalation asc
    records.sort(key=lambda r: (r['fnr'], r['escalation_rate']))
    return records[: max(0, n)]


def read_summary_csv(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    with p.open() as f:
        rdr = csv.DictReader(f)
        return [r for r in rdr]


def write_tables_tex() -> None:
    PAPER_GEN.mkdir(parents=True, exist_ok=True)
    # Stage1/Stage2
    s1_run, s1_auc = find_best_stage1()
    s2_run, s2_m = find_best_stage2()

    # Cascade bests
    casc = select_best_cascade(2)

    # Deepfake-Eval-2024 summaries
    cal = read_summary_csv(ROOT / 'outputs' / 'stage5' / 'evals_calibrated' / 'summary.csv')
    s2o = read_summary_csv(ROOT / 'outputs' / 'stage5' / 'evals_r4_512_s2only' / 'summary.csv')

    # Mobile artifacts sizes
    a1 = ROOT / 'outputs' / 'stage6' / 'export_ts' / 'stage1_mobilenetv4_ts.pt'
    a2 = ROOT / 'outputs' / 'stage6' / 'export_ts' / 'stage2_efficientnetv2_ts.pt'
    meta = ROOT / 'outputs' / 'stage6' / 'export_ts' / 'bundle_meta.json'
    def size_mb(p: Path) -> float:
        return round(os.path.getsize(p) / (1024 * 1024), 2) if p.exists() else 0.0

    with (PAPER_GEN / 'auto_tables.tex').open('w') as f:
        f.write('% Auto-generated tables\n')

        # Single-model table
        f.write('\\begin{table}[h]\n  \\centering\n')
        f.write('  \\caption{Single-model performance on combined validation (auto-generated).}\\label{tab:single_models_auto}\n')
        f.write('  \\begin{tabular}{lcccccc}\\toprule\n  Model & AUC & F1 & Acc & Prec & Rec & FNR \\\\ \\midrule\n')
        f.write(f'  MobileNetV4 (Stage~1) & {s1_auc:.4f} & -- & -- & -- & -- & -- \\\\ \n')
        if s2_m:
            f.write('  EfficientNetV2 (Stage~2, b3) & ' +
                    f"{s2_m.get('auc',0):.4f} & {s2_m.get('f1',0):.4f} & {s2_m.get('accuracy',0):.4f} & {s2_m.get('precision',0):.4f} & {s2_m.get('recall',0):.4f} & {s2_m.get('fnr',0):.4f} \\\\ \n")
        f.write('  \\bottomrule\\end{tabular}\\end{table}\n\n')

        # Cascade table
        f.write('\\begin{table}[h]\n  \\centering\n')
        f.write('  \\caption{Best cascade configurations (auto-generated).}\\label{tab:cascade_best_auto}\n')
        f.write('  \\begin{tabular}{cccccc}\\toprule\n Low & High & AUC & F1 & FNR & Escalation \\\\ \\midrule\n')
        for r in casc:
            f.write(f"  {r['low']:.2f} & {r['high']:.2f} & {r['auc']:.4f} & {r['f1']:.4f} & {r['fnr']:.4f} & {r['escalation_rate']:.4f} \\\\ \n")
        f.write('  \\bottomrule\\end{tabular}\\end{table}\n\n')

        # Deepfake-Eval-2024 tables
        def write_de_rows(rows: List[Dict[str, Any]], label: str):
            f.write('\\begin{table}[h]\n  \\centering\n')
            f.write(f'  \\caption{{Deepfake-Eval-2024 ({label})}}\\label{{tab:deepeval_{label}}}\n')
            f.write('  \\begin{tabular}{lccccc}\\toprule\n Split & Acc & F1 & FNR & FPR & Stage2 Rate \\\\ \\midrule\n')
            for r in rows:
                if r.get('dataset') != 'deepfake_eval_2024':
                    continue
                f.write(f"  {r.get('split')} & {float(r.get('accuracy',0)):.4f} & {float(r.get('f1',0)):.4f} & {float(r.get('fnr',0)):.4f} & {float(r.get('fpr',0)):.4f} & {float(r.get('stage2_rate',0)):.4f} \\\\ \n")
            f.write('  \\bottomrule\\end{tabular}\\end{table}\n\n')

        write_de_rows(cal, 'calibrated')
        write_de_rows(s2o, 'stage2only')

    with (PAPER_GEN / 'mobile_tables.tex').open('w') as f:
        f.write('% Auto-generated mobile artifact sizes\n')
        f.write('\\begin{table}[h]\n  \\centering\n')
        f.write('  \\caption{Exported on-device artifacts (auto).}\\label{tab:mobile_artifacts_auto}\n')
        f.write('  \\begin{tabular}{lcc}\\toprule\n Artifact & Path & Size (MB) \\\\ \\midrule\n')
        f.write(f'  Stage~1 TS & {a1.as_posix()} & {size_mb(a1):.2f} \\\\ \n')
        f.write(f'  Stage~2 TS & {a2.as_posix()} & {size_mb(a2):.2f} \\\\ \n')
        f.write(f'  Bundle meta & {meta.as_posix()} & {size_mb(meta):.2f} \\\\ \n')
        f.write('  \\bottomrule\\end{tabular}\\end{table}\n')

    # Robustness table (if metrics CSVs are present)
    rob_dir = ROOT / 'outputs' / 'stage5' / 'robustness'
    rows: List[Dict[str, Any]] = []
    for kind in ['jpeg','gaussian','motion','brightness']:
        mfile = rob_dir / f'{kind}_metrics.csv'
        if not mfile.exists():
            continue
        with mfile.open() as mf:
            rdr = csv.DictReader(mf)
            for r in rdr:
                try:
                    rows.append({
                        'kind': kind,
                        'level': r.get('level'),
                        'samples': r.get('samples'),
                        'f1': float(r.get('f1', 0.0)),
                        'accuracy': float(r.get('accuracy', 0.0)),
                        'fnr': float(r.get('fnr', 0.0)),
                        'stage2_rate': float(r.get('stage2_rate', 0.0)) if r.get('stage2_rate') not in (None, '') else 0.0,
                    })
                except Exception:
                    pass
    if rows:
        rows_sorted = sorted(rows, key=lambda x: (x['kind'], str(x['level'])))
        with (PAPER_GEN / 'robustness_table.tex').open('w') as f:
            f.write('% Auto-generated robustness table\n')
            f.write('\\begin{table}[h]\n  \\centering\n')
            f.write('  \\caption{Robustness summary by perturbation and level (auto-generated).}\\label{tab:robustness_table}\n')
            f.write('  \\begin{tabular}{lcrrrr}\\toprule\n Perturbation & Level & F1 & Acc & FNR & S2 Rate \\\\ \\midrule\n')
            for r in rows_sorted:
                f.write(f"  {r['kind']} & {r['level']} & {r['f1']:.4f} & {r['accuracy']:.4f} & {r['fnr']:.4f} & {r.get('stage2_rate',0.0):.4f} \\\\ \n")
            f.write('  \\bottomrule\\end{tabular}\\end{table}\n')


def main() -> int:
    write_tables_tex()
    print('Generated LaTeX tables under paper/generated/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
