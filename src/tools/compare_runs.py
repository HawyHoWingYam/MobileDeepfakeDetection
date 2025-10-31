#!/usr/bin/env python3
"""
AWARE-NET: Run Comparison CLI Tool (Phase 4)

Command-line tool to compare multiple training runs and generate
comparative analysis reports.

Usage:
    # Compare two runs
    python compare_runs.py \
        outputs/stage1/run_20251020_162308 \
        outputs/stage1/run_20251020_165000

    # Compare multiple runs with custom output
    python compare_runs.py run1 run2 run3 \
        --output-dir comparisons \
        --format html

    # Generate comparison metrics table
    python compare_runs.py run1 run2 \
        --show-metrics \
        --metric auc f1 accuracy

Features:
    - Load multiple evaluation_summary.json files
    - Compare key metrics across runs
    - Identify best performing run
    - Generate comparison tables and plots
    - Export comparison reports (JSON, HTML, CSV)
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RunComparator:
    """Compare multiple training runs."""

    def __init__(self, run_dirs: List[str]):
        """
        Initialize run comparator.

        Args:
            run_dirs: List of paths to run directories
        """
        self.run_dirs = [Path(d) for d in run_dirs]
        self.run_data = {}
        self.run_names = {}

        # Load all run data
        for run_dir in self.run_dirs:
            self._load_run(run_dir)

        if not self.run_data:
            raise RuntimeError("No valid runs loaded")

        logger.info(f"📊 Loaded {len(self.run_data)} runs for comparison")

    def _load_run(self, run_dir: Path) -> None:
        """
        Load evaluation summary from a run directory.

        Args:
            run_dir: Path to run directory
        """
        # Try multiple possible locations for evaluation_summary.json
        summary_paths = [
            run_dir / 'artifacts' / 'evaluation_summary.json',
            run_dir / 'evaluation_summary.json',
        ]

        summary_path = None
        for path in summary_paths:
            if path.exists():
                summary_path = path
                break

        if not summary_path:
            logger.warning(f"⚠️  evaluation_summary.json not found in {run_dir}")
            return

        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)

            # Use directory name as run identifier
            run_id = run_dir.name
            self.run_data[run_id] = data
            self.run_names[run_id] = run_id

            logger.info(f"  ✓ Loaded run: {run_id}")

        except Exception as e:
            logger.error(f"  ✗ Failed to load {summary_path}: {e}")

    def get_comparison_table(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comparison table of all runs and their metrics.

        Returns:
            Dictionary with run comparisons
        """
        comparison = {}

        key_metrics = ['auc', 'f1', 'accuracy', 'precision', 'recall']

        for run_id, data in self.run_data.items():
            metrics = data.get('metrics', {})
            comparison[run_id] = {
                'total_samples': data.get('total_samples', 'N/A'),
                'evaluation_mode': data.get('evaluation_mode', 'N/A'),
            }

            for metric in key_metrics:
                if metric in metrics:
                    comparison[run_id][metric] = round(metrics[metric], 4)

        return comparison

    def get_best_run(self, metric: str = 'auc') -> tuple[str, float]:
        """
        Find the best performing run for a given metric.

        Args:
            metric: Metric to compare (default: 'auc')

        Returns:
            Tuple of (run_id, metric_value)
        """
        best_run = None
        best_value = -1

        for run_id, data in self.run_data.items():
            metrics = data.get('metrics', {})
            if metric in metrics:
                value = metrics[metric]
                if value > best_value:
                    best_value = value
                    best_run = run_id

        return best_run, best_value

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of all runs.

        Returns:
            Dictionary with statistics
        """
        key_metrics = ['auc', 'f1', 'accuracy', 'precision', 'recall']
        statistics = {}

        for metric in key_metrics:
            values = []
            for data in self.run_data.values():
                metrics = data.get('metrics', {})
                if metric in metrics:
                    values.append(metrics[metric])

            if values:
                statistics[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'num_runs': len(values),
                }

        return statistics

    def get_top_runs(self, metric: str = 'auc', n: int = 3) -> List[tuple[str, float]]:
        """
        Get top N performing runs for a metric.

        Args:
            metric: Metric to rank by
            n: Number of top runs to return

        Returns:
            List of (run_id, metric_value) tuples sorted descending
        """
        ranked = []

        for run_id, data in self.run_data.items():
            metrics = data.get('metrics', {})
            if metric in metrics:
                ranked.append((run_id, metrics[metric]))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def generate_comparison_report(self, output_path: str) -> str:
        """
        Generate comprehensive comparison report.

        Args:
            output_path: Path to save report (JSON or CSV)

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get comparison data
        comparison = self.get_comparison_table()
        statistics = self.get_statistics()
        best_auc_run, best_auc_val = self.get_best_run('auc')
        best_f1_run, best_f1_val = self.get_best_run('f1')

        # Create report
        report = {
            'comparison_timestamp': str(Path.cwd()),
            'num_runs': len(self.run_data),
            'run_ids': list(self.run_data.keys()),
            'comparison_metrics': comparison,
            'statistics': statistics,
            'best_run_auc': {'run_id': best_auc_run, 'auc': best_auc_val},
            'best_run_f1': {'run_id': best_f1_run, 'f1': best_f1_val},
            'top_3_by_auc': [{'run_id': rid, 'auc': val} for rid, val in self.get_top_runs('auc', 3)],
            'top_3_by_f1': [{'run_id': rid, 'f1': val} for rid, val in self.get_top_runs('f1', 3)],
        }

        # Save report
        if str(output_path).endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        elif str(output_path).endswith('.csv'):
            import csv
            with open(output_path, 'w', newline='') as f:
                # Write header
                all_keys = set()
                for run_data in comparison.values():
                    all_keys.update(run_data.keys())

                writer = csv.DictWriter(f, fieldnames=['run_id'] + sorted(all_keys))
                writer.writeheader()

                # Write data
                for run_id, data in comparison.items():
                    row = {'run_id': run_id}
                    row.update(data)
                    writer.writerow(row)

        logger.info(f"📄 Generated comparison report: {output_path}")

        return str(output_path)

    def print_comparison_table(self) -> None:
        """Print comparison table to console."""
        comparison = self.get_comparison_table()
        statistics = self.get_statistics()

        print("\n" + "=" * 100)
        print("RUN COMPARISON TABLE")
        print("=" * 100)

        # Print header
        headers = ['Run ID'] + list(next(iter(comparison.values())).keys())
        col_widths = [max(len(str(h)), 15) for h in headers]

        header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        print(header_str)
        print("-" * len(header_str))

        # Print data
        for run_id, data in comparison.items():
            row = [run_id] + [str(v) for v in data.values()]
            row_str = " | ".join(f"{v:<{w}}" for v, w in zip(row, col_widths))
            print(row_str)

        # Print statistics
        print("\n" + "=" * 100)
        print("STATISTICAL SUMMARY")
        print("=" * 100)

        for metric, stats in statistics.items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean:    {stats['mean']:.4f}")
            print(f"  Std Dev: {stats['std']:.4f}")
            print(f"  Min:     {stats['min']:.4f}")
            print(f"  Max:     {stats['max']:.4f}")

        # Print best runs
        print("\n" + "=" * 100)
        print("BEST PERFORMING RUNS")
        print("=" * 100)

        best_auc, auc_val = self.get_best_run('auc')
        best_f1, f1_val = self.get_best_run('f1')

        print(f"\nBest AUC:  {best_auc} ({auc_val:.4f})")
        print(f"Best F1:   {best_f1} ({f1_val:.4f})")

        print(f"\nTop 3 by AUC:")
        for i, (run_id, val) in enumerate(self.get_top_runs('auc', 3), 1):
            print(f"  {i}. {run_id}: {val:.4f}")

        print(f"\nTop 3 by F1:")
        for i, (run_id, val) in enumerate(self.get_top_runs('f1', 3), 1):
            print(f"  {i}. {run_id}: {val:.4f}")

        print("\n" + "=" * 100 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AWARE-NET Run Comparison Tool - Compare multiple training runs"
    )

    parser.add_argument(
        "run_dirs",
        type=str,
        nargs='+',
        help="Paths to run directories (at least 2 required)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for comparison report"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=['json', 'csv', 'print'],
        default='print',
        help="Output format: json, csv, or print to console"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default='auc',
        help="Metric to use for ranking (default: auc)"
    )

    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        parser.error("At least 2 run directories required for comparison")

    try:
        # Create comparator
        logger.info("=" * 70)
        logger.info("AWARE-NET Run Comparison Tool")
        logger.info("=" * 70)

        comparator = RunComparator(args.run_dirs)

        # Display comparison
        if args.format == 'print':
            comparator.print_comparison_table()
        else:
            # Generate report file
            if not args.output_dir:
                args.output_dir = Path.cwd() / "comparisons"

            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

            report_path = Path(args.output_dir) / f"comparison.{args.format}"
            comparator.generate_comparison_report(str(report_path))

            logger.info(f"\n✅ Comparison complete! Report saved: {report_path}")

    except RuntimeError as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
