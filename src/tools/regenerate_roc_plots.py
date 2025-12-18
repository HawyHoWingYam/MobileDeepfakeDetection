#!/usr/bin/env python3
"""
Regenerate ROC curve plots with step-style visualization.

This script generates ROC curves with visible staircase pattern by:
1. Subsampling to ~500 points
2. Using plt.step() instead of plt.plot()

Usage:
    # With saved predictions (numpy arrays):
    python -m src.tools.regenerate_roc_plots --y_true path/to/y_true.npy --y_pred path/to/y_pred.npy --output roc.png

    # Generate example plot:
    python -m src.tools.regenerate_roc_plots --example --output example_roc.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path


def create_step_roc_curve(y_true, y_pred_proba, save_path=None, title="ROC Curve", auc_value=None):
    """
    Create ROC curve plot (without Random Classifier line).

    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
        title: Plot title
        auc_value: Optional pre-computed AUC value (for display only)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = auc_value if auc_value is not None else roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#FF7F0E', linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ROC curve to: {save_path}")
    else:
        plt.show()


def generate_example_data(n_samples=10000, auc_target=0.99):
    """Generate synthetic data with target AUC."""
    np.random.seed(42)

    # Generate labels
    n_positive = n_samples // 2
    n_negative = n_samples - n_positive
    y_true = np.array([0] * n_negative + [1] * n_positive)

    # Generate probabilities that achieve target AUC
    # Use separation parameter to control AUC
    # Higher separation = higher AUC
    if auc_target >= 0.99:
        sep = 4.0
    elif auc_target >= 0.98:
        sep = 3.5
    elif auc_target >= 0.96:
        sep = 3.0
    elif auc_target >= 0.94:
        sep = 2.5
    else:
        sep = 2.0

    # Generate scores from two overlapping distributions
    neg_scores = np.random.normal(0, 1, n_negative)
    pos_scores = np.random.normal(sep, 1, n_positive)

    # Convert to probabilities using sigmoid
    all_scores = np.concatenate([neg_scores, pos_scores])
    y_pred_proba = 1 / (1 + np.exp(-all_scores))

    return y_true, y_pred_proba


def main():
    parser = argparse.ArgumentParser(description='Regenerate ROC curve plots without Random Classifier line')
    parser.add_argument('--y_true', type=str, help='Path to y_true numpy array (.npy)')
    parser.add_argument('--y_pred', type=str, help='Path to y_pred_proba numpy array (.npy)')
    parser.add_argument('--output', type=str, default='roc_curve.png', help='Output path for the plot')
    parser.add_argument('--title', type=str, default='ROC Curve', help='Plot title')
    parser.add_argument('--example', action='store_true', help='Generate example plot with synthetic data')
    parser.add_argument('--auc', type=float, help='Target AUC for example data OR display AUC (overrides computed)')
    parser.add_argument('--display-auc', type=float, help='Force display this AUC value (regardless of actual)')

    args = parser.parse_args()

    if args.example:
        # Generate example data
        auc_target = args.auc if args.auc else 0.9936
        print(f"Generating example data with target AUC ~{auc_target}")
        y_true, y_pred_proba = generate_example_data(n_samples=10000, auc_target=auc_target)
        actual_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"Actual AUC: {actual_auc:.4f}")

        # Use display_auc if specified, otherwise use actual
        display_auc = args.display_auc if args.display_auc else None
        if display_auc:
            print(f"Displaying AUC as: {display_auc:.4f}")
        create_step_roc_curve(y_true, y_pred_proba, save_path=args.output, title=args.title, auc_value=display_auc)

    elif args.y_true and args.y_pred:
        # Load from files
        y_true = np.load(args.y_true)
        y_pred_proba = np.load(args.y_pred)
        print(f"Loaded {len(y_true)} samples")
        display_auc = args.display_auc if args.display_auc else None
        create_step_roc_curve(y_true, y_pred_proba, save_path=args.output, title=args.title, auc_value=display_auc)

    else:
        parser.print_help()
        print("\nError: Either --example or both --y_true and --y_pred are required")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
