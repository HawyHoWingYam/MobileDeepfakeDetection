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


def summarize_mobile_detection_logs(log_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Summarise on-device detection_logs.csv into simple runtime/use stats.

    We distinguish between image-based detections and video-frame detections
    using the presence of a `frame=` query parameter in the source URI.
    """
    if not log_path.exists():
        return {}

    with log_path.open() as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        return {}

    groups: Dict[str, List[Dict[str, Any]]] = {'image': [], 'video': []}
    for r in rows:
        uri = r.get('source_uri', '') or ''
        key = 'video' if 'frame=' in uri else 'image'
        groups.setdefault(key, []).append(r)

    def safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def safe_int(v: Any, default: int = 0) -> int:
        try:
            return int(float(v))
        except Exception:
            return default

    stats: Dict[str, Dict[str, float]] = {}
    for key, items in groups.items():
        if not items:
            continue
        n = len(items)
        deepfake_count = sum(1 for r in items if safe_int(r.get('is_deepfake', 0)) == 1)
        stage2_count = sum(1 for r in items if safe_int(r.get('stage2_used', 0)) == 1)

        total_pre = sum(safe_float(r.get('preprocess_ms', 0.0)) for r in items)
        total_inf = sum(safe_float(r.get('inference_ms', 0.0)) for r in items)
        total_tot = sum(safe_float(r.get('total_ms', 0.0)) for r in items)

        # Derive ground-truth labels from path when possible
        tp = tn = fp = fn = 0
        labeled = 0
        for r in items:
            uri = (r.get('source_uri') or '').lower()
            if 'images_fake' in uri:
                y = 1
            elif 'images_real' in uri:
                y = 0
            else:
                continue
            y_pred = safe_int(r.get('is_deepfake', 0))
            labeled += 1
            if y == 1 and y_pred == 1:
                tp += 1
            elif y == 0 and y_pred == 0:
                tn += 1
            elif y == 1 and y_pred == 0:
                fn += 1
            elif y == 0 and y_pred == 1:
                fp += 1

        def ratio(num: float, den: float) -> float:
            return num / den if den > 0 else 0.0

        acc = ratio(tp + tn, labeled)
        fnr = ratio(fn, fn + tp)
        fpr = ratio(fp, fp + tn)
        prec = ratio(tp, tp + fp)
        rec = ratio(tp, tp + fn)

        stats[key] = {
            'samples': float(n),
            'samples_labeled': float(labeled),
            'deepfake_rate': deepfake_count / float(n),
            'stage2_rate': stage2_count / float(n),
            'avg_pre_ms': total_pre / float(n),
            'avg_inf_ms': total_inf / float(n),
            'avg_total_ms': total_tot / float(n),
            'accuracy': acc,
            'fnr': fnr,
            'fpr': fpr,
            'precision': prec,
            'recall': rec,
        }

    return stats


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

    # Mobile artifacts sizes (ONNX files used on device)
    a1 = ROOT / 'android' / 'app' / 'src' / 'main' / 'assets' / 'models' / 'aware_cascade_stage1.onnx'
    a2 = ROOT / 'android' / 'app' / 'src' / 'main' / 'assets' / 'models' / 'aware_cascade_stage2.onnx'
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
        f.write('  \\caption{Exported on-device model artifacts.}\\label{tab:mobile_artifacts_auto}\n')
        f.write('  \\begin{tabular}{lcc}\\toprule\n  Artifact & Format & Size (MB) \\\\ \\midrule\n')
        f.write(f'  Stage~1 (MobileNetV4) & ONNX & {size_mb(a1):.2f} \\\\ \n')
        f.write(f'  Stage~2 (EfficientNetV2-B3) & ONNX & {size_mb(a2):.2f} \\\\ \n')
        f.write(f'  \\textbf{{Total}} & -- & \\textbf{{{size_mb(a1) + size_mb(a2):.2f}}} \\\\ \n')
        f.write('  \\bottomrule\\end{tabular}\\end{table}\n')

        # Optional: on-device detection log summary (if available)
        mobile_log = ROOT / 'ref' / 'detection_logs.csv'
        stats = summarize_mobile_detection_logs(mobile_log)
        if stats:
            mode_labels = {
                'image': 'Images (single/batch)',
                'video': 'Video frames',
            }
            f.write('\n% Auto-generated on-device detection summary\n')
            f.write('\\begin{table}[h]\n  \\centering\n')
            f.write('  \\caption{On-device cascade runtime and usage (auto).}\\label{tab:mobile_runtime_auto}\n')
            f.write('  \\begin{tabular}{lrrrrrr}\\toprule\n')
            f.write('  Mode & $N$ & Deepfake Rate & Stage2 Rate & Preproc (ms) & Inference (ms) & Total (ms) \\\\ \\midrule\n')
            for key in ('image', 'video'):
                if key not in stats:
                    continue
                st = stats[key]
                f.write(
                    f"  {mode_labels.get(key, key)} & "
                    f"{int(st['samples'])} & "
                    f"{st['deepfake_rate']:.3f} & "
                    f"{st['stage2_rate']:.3f} & "
                    f"{st['avg_pre_ms']:.1f} & "
                    f"{st['avg_inf_ms']:.1f} & "
                    f"{st['avg_total_ms']:.1f} \\\\ \n"
                )
            f.write('  \\bottomrule\\end{tabular}\\end{table}\n')

            # Accuracy / FNR summary (only for samples where labels can be inferred)
            f.write('\n% Auto-generated on-device accuracy summary (requires folder-structured datasets)\n')
            f.write('\\begin{table}[h]\n  \\centering\n')
            f.write('  \\caption{On-device accuracy and error rates (auto).}\\label{tab:mobile_accuracy_auto}\n')
            f.write('  \\begin{tabular}{lrrrrrrr}\\toprule\n')
            f.write('  Mode & $N_{lbl}$ & Acc & FNR & FPR & Prec & Rec & Stage2 Rate \\\\ \\midrule\n')
            for key in ('image', 'video'):
                if key not in stats:
                    continue
                st = stats[key]
                if st.get('samples_labeled', 0.0) <= 0:
                    continue
                f.write(
                    f"  {mode_labels.get(key, key)} & "
                    f"{int(st['samples_labeled'])} & "
                    f"{st['accuracy']:.3f} & "
                    f"{st['fnr']:.3f} & "
                    f"{st['fpr']:.3f} & "
                    f"{st['precision']:.3f} & "
                    f"{st['recall']:.3f} & "
                    f"{st['stage2_rate']:.3f} \\\\ \n"
                )
            f.write('  \\bottomrule\\end{tabular}\\end{table}\n')

    # Robustness table (if metrics CSVs are present)
    rob_dir = ROOT / 'outputs' / 'stage5' / 'robustness'
    rows: List[Dict[str, Any]] = []
    for kind in ['jpeg','gaussian','motion','brightness','downscale']:
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
        rows_sorted = sorted(rows, key=lambda x: (x['kind'], float(x['level'])))
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
