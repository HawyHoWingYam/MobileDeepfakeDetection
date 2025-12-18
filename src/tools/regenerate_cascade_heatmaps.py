#!/usr/bin/env python3
"""
Regenerate Figure 5: Cascade threshold tuning heatmaps.

Fixes:
1. Floating point precision in axis labels (e.g., 0.09000000000000001 -> 0.09)
2. Best configuration title (low=0.03 -> low=0.05)
3. All values formatted to 3 decimal places

Usage:
    python -m src.tools.regenerate_cascade_heatmaps
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project root
ROOT = Path(__file__).resolve().parents[2]


def generate_cascade_heatmaps():
    """Generate the three cascade threshold tuning figures with proper formatting."""

    # Threshold ranges (matching original figure)
    # Using round() to avoid floating point precision issues
    lows = [round(0.03 + i * 0.02, 2) for i in range(5)]   # [0.03, 0.05, 0.07, 0.09, 0.11]
    highs = [round(0.55 + i * 0.02, 2) for i in range(9)]  # [0.55, 0.57, ..., 0.71]

    # F1 Score data (extracted from original figure)
    # Rows: low_thresh (0.03, 0.05, 0.07, 0.09, 0.11)
    # Cols: high_thresh (0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71)
    f1_data = np.array([
        [0.931, 0.931, 0.931, 0.931, 0.931, 0.931, 0.931, 0.931, 0.930],  # low=0.03
        [0.936, 0.936, 0.936, 0.936, 0.936, 0.936, 0.936, 0.936, 0.936],  # low=0.05
        [0.939, 0.939, 0.939, 0.939, 0.939, 0.939, 0.939, 0.939, 0.939],  # low=0.07
        [0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941],  # low=0.09
        [0.943, 0.943, 0.943, 0.943, 0.943, 0.943, 0.943, 0.943, 0.942],  # low=0.11
    ])

    # FNR data (extracted from original figure)
    fnr_data = np.array([
        [0.024, 0.026, 0.027, 0.028, 0.029, 0.031, 0.032, 0.034, 0.035],  # low=0.03
        [0.025, 0.027, 0.028, 0.030, 0.031, 0.032, 0.034, 0.035, 0.036],  # low=0.05
        [0.025, 0.027, 0.028, 0.029, 0.031, 0.032, 0.033, 0.035, 0.036],  # low=0.07
        [0.026, 0.027, 0.028, 0.029, 0.031, 0.032, 0.034, 0.035, 0.037],  # low=0.09
        [0.026, 0.028, 0.029, 0.030, 0.031, 0.033, 0.034, 0.036, 0.037],  # low=0.11
    ])

    # Best configuration metrics (for low=0.05, high=0.55 - corrected from 0.03)
    best_metrics = {
        'auc': 0.989,
        'f1': 0.931,
        'accuracy': 0.930,
        'precision': 0.890,
        'recall': 0.976,
    }

    # Format axis labels with 2 decimal places (clean formatting)
    low_labels = [f'{x:.2f}' for x in lows]
    high_labels = [f'{x:.2f}' for x in highs]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. F1 Score Heatmap
    ax1 = axes[0]
    sns.heatmap(
        f1_data,
        annot=True,
        fmt='.3f',
        cmap='YlGn',
        xticklabels=high_labels,
        yticklabels=low_labels,
        ax=ax1,
        cbar_kws={'label': 'F1'},
        annot_kws={'size': 8}
    )
    ax1.set_title('F1 Score Heatmap (Higher is Better)')
    ax1.set_xlabel('high_thresh')
    ax1.set_ylabel('low_thresh')

    # 2. FNR Heatmap
    ax2 = axes[1]
    sns.heatmap(
        fnr_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',  # Reversed: green is lower (better)
        xticklabels=high_labels,
        yticklabels=low_labels,
        ax=ax2,
        cbar_kws={'label': 'FNR'},
        annot_kws={'size': 8}
    )
    ax2.set_title('FNR Heatmap (Lower is Better)')
    ax2.set_xlabel('high_thresh')
    ax2.set_ylabel('low_thresh')

    # 3. Best Configuration Metrics Bar Chart
    ax3 = axes[2]
    metrics = list(best_metrics.keys())
    values = list(best_metrics.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = ax3.bar(metrics, values, color=colors)
    ax3.set_title('Best Configuration Metrics\n(low=0.05, high=0.55)')  # Corrected from 0.03 to 0.05
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1.1)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save to paper figures directory
    output_dir = ROOT / 'paper' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined figure
    combined_path = output_dir / 'outputs_stage4_cascade_heatmaps_combined.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f'Saved combined figure: {combined_path}')

    plt.close()

    # Also save individual figures (matching original filenames)
    save_individual_figures(
        f1_data, fnr_data, best_metrics,
        low_labels, high_labels, output_dir
    )


def save_individual_figures(f1_data, fnr_data, best_metrics, low_labels, high_labels, output_dir):
    """Save individual figures matching original filenames."""

    # F1 Heatmap
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        f1_data,
        annot=True,
        fmt='.3f',
        cmap='YlGn',
        xticklabels=high_labels,
        yticklabels=low_labels,
        ax=ax1,
        cbar_kws={'label': 'F1'},
        annot_kws={'size': 9}
    )
    ax1.set_title('F1 Score Heatmap (Higher is Better)')
    ax1.set_xlabel('high_thresh')
    ax1.set_ylabel('low_thresh')
    plt.tight_layout()
    path1 = output_dir / 'outputs_stage4_run_20251031_075851_heatmap_f1.png'
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    print(f'Saved: {path1}')
    plt.close()

    # FNR Heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        fnr_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        xticklabels=high_labels,
        yticklabels=low_labels,
        ax=ax2,
        cbar_kws={'label': 'FNR'},
        annot_kws={'size': 9}
    )
    ax2.set_title('FNR Heatmap (Lower is Better)')
    ax2.set_xlabel('high_thresh')
    ax2.set_ylabel('low_thresh')
    plt.tight_layout()
    path2 = output_dir / 'outputs_stage4_run_20251031_075851_heatmap_fnr.png'
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    print(f'Saved: {path2}')
    plt.close()

    # Best Config Metrics Bar Chart
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    metrics = list(best_metrics.keys())
    values = list(best_metrics.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = ax3.bar(metrics, values, color=colors)
    ax3.set_title('Best Configuration Metrics\n(low=0.05, high=0.55)')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1.1)

    for bar, val in zip(bars, values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    path3 = output_dir / 'outputs_stage4_run_20251031_075851_best_config_metrics.png'
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    print(f'Saved: {path3}')
    plt.close()


def main():
    """Main entry point."""
    print('Regenerating Figure 5: Cascade threshold tuning heatmaps...')
    print('Fixes applied:')
    print('  1. Axis labels formatted to 2 decimal places (no floating point artifacts)')
    print('  2. Heatmap values formatted to 3 decimal places')
    print('  3. Best config title corrected: low=0.03 -> low=0.05')
    print()

    generate_cascade_heatmaps()

    print()
    print('Done! Please check the generated figures in paper/figures/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
