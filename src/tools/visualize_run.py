#!/usr/bin/env python3
"""
AWARE-NET: Visualization CLI Tool (Phase 3)

Command-line tool to visualize training runs and evaluation results.
Generates comprehensive visualization reports from experiment outputs.

Usage:
    # Visualize a specific run
    python visualize_run.py outputs/stage1/run_20251020_162308

    # Generate all plots and save to custom directory
    python visualize_run.py outputs/stage1/run_20251020_162308 \
        --output-dir outputs/visualizations \
        --show-plots

    # Generate HTML report
    python visualize_run.py outputs/stage1/run_20251020_162308 \
        --format html

Features:
    - Loads evaluation_summary.json from run directory
    - Generates 10+ visualization plots
    - Creates HTML report (optional)
    - Interactive preview (optional)
    - Comparison plots for multiple runs (future)
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plotting import (
    plot_metrics_summary,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_threshold_analysis,
    create_comprehensive_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """Generate visualizations from training run results."""

    def __init__(self, run_dir: str, output_dir: Optional[str] = None):
        """
        Initialize visualization generator.

        Args:
            run_dir: Path to run directory (contains evaluation_summary.json)
            output_dir: Output directory for visualizations (default: run_dir/figures)
        """
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir) if output_dir else self.run_dir / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load evaluation summary
        self.summary_path = self.run_dir / 'artifacts' / 'evaluation_summary.json'
        if not self.summary_path.exists():
            self.summary_path = self.run_dir / 'evaluation_summary.json'

        if not self.summary_path.exists():
            raise FileNotFoundError(f"evaluation_summary.json not found in {self.run_dir}")

        with open(self.summary_path, 'r') as f:
            self.summary = json.load(f)

        logger.info(f"📂 Loaded evaluation summary from: {self.summary_path}")

    def generate_all_plots(self, dpi: int = 150) -> Dict[str, str]:
        """
        Generate all available visualization plots.

        Args:
            dpi: DPI for saved figures

        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.info(f"🎨 Generating visualizations to: {self.output_dir}")

        plots_generated = {}

        try:
            # 1. Metrics Summary
            logger.info("  - Generating metrics summary...")
            if 'metrics' in self.summary:
                plot_metrics_summary(
                    self.summary['metrics'],
                    output_path=str(self.output_dir / '01_metrics_summary.png'),
                    dpi=dpi
                )
                plots_generated['metrics_summary'] = str(self.output_dir / '01_metrics_summary.png')

            # 2. Confusion Matrix
            logger.info("  - Generating confusion matrix...")
            if 'confusion_matrix' in self.summary:
                cm = self.summary['confusion_matrix']
                import numpy as np
                cm_array = np.array([
                    [cm.get('true_negatives', 0), cm.get('false_positives', 0)],
                    [cm.get('false_negatives', 0), cm.get('true_positives', 0)]
                ])
                plot_confusion_matrix(
                    cm_array,
                    output_path=str(self.output_dir / '02_confusion_matrix.png'),
                    dpi=dpi
                )
                plots_generated['confusion_matrix'] = str(self.output_dir / '02_confusion_matrix.png')

            # 3. Class Distribution
            logger.info("  - Generating class distribution...")
            if 'class_distribution' in self.summary:
                cd = self.summary['class_distribution']
                plot_class_distribution(
                    cd.get('real_count', 0),
                    cd.get('fake_count', 0),
                    output_path=str(self.output_dir / '03_class_distribution.png'),
                    dpi=dpi
                )
                plots_generated['class_distribution'] = str(self.output_dir / '03_class_distribution.png')

            # 4. Threshold Analysis
            logger.info("  - Generating threshold analysis...")
            if 'threshold_analysis' in self.summary:
                plot_threshold_analysis(
                    self.summary['threshold_analysis'],
                    output_path=str(self.output_dir / '04_threshold_analysis.png'),
                    dpi=dpi
                )
                plots_generated['threshold_analysis'] = str(self.output_dir / '04_threshold_analysis.png')

            logger.info(f"✅ Generated {len(plots_generated)} plots")

            return plots_generated

        except Exception as e:
            logger.error(f"❌ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
            return plots_generated

    def generate_html_report(
        self,
        plots: Dict[str, str],
        title: str = "AWARE-NET Visualization Report"
    ) -> str:
        """
        Generate HTML report with embedded plots.

        Args:
            plots: Dictionary of plot names to file paths
            title: HTML report title

        Returns:
            Path to generated HTML file
        """
        import base64
        from datetime import datetime

        html_path = self.output_dir / 'report.html'

        html_content = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        html_content += "    <meta charset=\"UTF-8\">\n"
        html_content += "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        html_content += f"    <title>{title}</title>\n"
        html_content += "    <style>\n"
        html_content += "        * { margin: 0; padding: 0; box-sizing: border-box; }\n"
        html_content += "        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; "
        html_content += "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
        html_content += "padding: 20px; min-height: 100vh; }\n"
        html_content += "        .container { max-width: 1400px; margin: 0 auto; background: white; "
        html_content += "border-radius: 10px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2); overflow: hidden; }\n"
        html_content += "        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
        html_content += "color: white; padding: 40px; text-align: center; }\n"
        html_content += "        .header h1 { font-size: 32px; margin-bottom: 10px; }\n"
        html_content += "        .header p { font-size: 14px; opacity: 0.9; }\n"
        html_content += "        .content { padding: 40px; }\n"
        html_content += "        .section { margin-bottom: 40px; border-bottom: 1px solid #e0e0e0; padding-bottom: 40px; }\n"
        html_content += "        .section:last-child { border-bottom: none; }\n"
        html_content += "        .section h2 { color: #333; margin-bottom: 20px; font-size: 20px; "
        html_content += "border-left: 4px solid #667eea; padding-left: 15px; }\n"
        html_content += "        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); "
        html_content += "gap: 20px; margin-bottom: 30px; }\n"
        html_content += "        .metric-card { background: #f5f5f5; padding: 20px; border-radius: 5px; "
        html_content += "text-align: center; border-left: 4px solid #667eea; }\n"
        html_content += "        .metric-card .label { font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 10px; }\n"
        html_content += "        .metric-card .value { font-size: 28px; font-weight: bold; color: #333; }\n"
        html_content += "        .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 30px; }\n"
        html_content += "        .plot { border: 1px solid #e0e0e0; border-radius: 5px; overflow: hidden; background: white; }\n"
        html_content += "        .plot img { width: 100%; height: auto; display: block; }\n"
        html_content += "        .footer { background: #f5f5f5; padding: 20px; text-align: center; "
        html_content += "font-size: 12px; color: #666; border-top: 1px solid #e0e0e0; }\n"
        html_content += "        .footer p { margin: 5px 0; }\n"
        html_content += "    </style>\n</head>\n<body>\n"
        html_content += "    <div class=\"container\">\n"
        html_content += "        <div class=\"header\">\n"
        html_content += f"            <h1>{title}</h1>\n"
        html_content += f"            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n"
        html_content += f"            <p>Run Directory: {self.run_dir}</p>\n"
        html_content += "        </div>\n\n        <div class=\"content\">\n"

        # Add metrics summary
        if 'metrics' in self.summary:
            html_content += '<div class="section">\n<h2>📊 Evaluation Metrics</h2>\n<div class="metrics">\n'
            for metric, value in self.summary['metrics'].items():
                html_content += f'''
            <div class="metric-card">
                <div class="label">{metric}</div>
                <div class="value">{value:.4f}</div>
            </div>
'''
            html_content += '</div>\n</div>\n'

        # Add confusion matrix info
        if 'confusion_matrix' in self.summary:
            cm = self.summary['confusion_matrix']
            html_content += '<div class="section">\n<h2>🎯 Classification Matrix</h2>\n<div class="metrics">\n'
            for key in ['true_negatives', 'true_positives', 'false_negatives', 'false_positives']:
                if key in cm:
                    label = key.replace('_', ' ').title()
                    html_content += f'''
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{cm[key]}</div>
            </div>
'''
            html_content += '</div>\n</div>\n'

        # Add plots
        if plots:
            html_content += '<div class="section">\n<h2>📈 Visualizations</h2>\n<div class="plot-grid">\n'
            for plot_name, plot_path in plots.items():
                plot_title = plot_name.replace('_', ' ').title()
                try:
                    with open(plot_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    html_content += f'''
            <div class="plot">
                <img src="data:image/png;base64,{img_data}" alt="{plot_title}">
            </div>
'''
                except Exception as e:
                    logger.warning(f"Could not embed plot {plot_path}: {e}")

            html_content += '</div>\n</div>\n'

        # Add footer
        html_content += '        <div class="section">\n'
        html_content += '            <h2>📝 Run Information</h2>\n'
        html_content += '            <div class="metrics">\n'
        html_content += '                <div class="metric-card">\n'
        html_content += '                    <div class="label">Total Samples</div>\n'
        html_content += f'                    <div class="value">{self.summary.get("total_samples", "N/A")}</div>\n'
        html_content += '                </div>\n'
        html_content += '                <div class="metric-card">\n'
        html_content += '                    <div class="label">Evaluation Mode</div>\n'
        html_content += f'                    <div class="value">{self.summary.get("evaluation_mode", "N/A")}</div>\n'
        html_content += '                </div>\n'
        html_content += '            </div>\n'
        html_content += '        </div>\n'
        html_content += '    </div>\n\n'
        html_content += '    <div class="footer">\n'
        html_content += '        <p>Generated by AWARE-NET Visualization Tool</p>\n'
        html_content += f'        <p>Run Directory: {self.run_dir}</p>\n'
        html_content += '    </div>\n'
        html_content += '</body>\n</html>\n'

        with open(html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"📄 Generated HTML report: {html_path}")

        return str(html_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AWARE-NET Visualization Tool - Generate reports from training runs"
    )

    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to run directory (contains evaluation_summary.json)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: run_dir/figures)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=['plots', 'html', 'both'],
        default='plots',
        help="Output format: plots (PNG), html, or both"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures (default: 150)"
    )

    parser.add_argument(
        "--show-plots",
        action='store_true',
        help="Show plots interactively (requires display)"
    )

    args = parser.parse_args()

    try:
        # Generate visualizations
        logger.info("=" * 70)
        logger.info("AWARE-NET Visualization Tool")
        logger.info("=" * 70)

        visualizer = VisualizationGenerator(args.run_dir, args.output_dir)
        plots = visualizer.generate_all_plots(dpi=args.dpi)

        # Generate HTML report if requested
        if args.format in ['html', 'both']:
            html_path = visualizer.generate_html_report(plots)
            logger.info(f"\n✅ HTML report generated: {html_path}")

        # Show plots if requested
        if args.show_plots:
            logger.info("\n📊 Opening plots interactively...")
            import matplotlib.pyplot as plt
            plt.show()

        logger.info("\n" + "=" * 70)
        logger.info(f"✅ Visualization complete! Results: {visualizer.output_dir}")
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
