"""
AWARE-NET: Comprehensive Visualization Utilities (Phase 3)

Provides extensive plotting and visualization functions for analyzing
training results, evaluation metrics, and model performance across all stages.

Includes 15+ plot types:
- Learning curves (loss, AUC, metrics over epochs)
- Confusion matrix and classification metrics
- ROC curve and Precision-Recall curves
- Probability distribution analysis
- Per-dataset comparisons
- Threshold sensitivity analysis
- Calibration plots
- Error analysis visualization

Usage:
    from utils.plotting import plot_learning_curves, plot_confusion_matrix

    # Plot learning curves
    plot_learning_curves(
        train_loss, val_loss, train_auc, val_auc,
        output_path='outputs/learning_curves.png'
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        cm_data, output_path='outputs/confusion_matrix.png'
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_learning_curves(
    train_loss: List[float],
    val_loss: List[float],
    train_auc: List[float] = None,
    val_auc: List[float] = None,
    train_f1: List[float] = None,
    val_f1: List[float] = None,
    train_accuracy: List[float] = None,
    val_accuracy: List[float] = None,
    train_precision: List[float] = None,
    val_precision: List[float] = None,
    train_recall: List[float] = None,
    val_recall: List[float] = None,
    train_specificity: List[float] = None,
    val_specificity: List[float] = None,
    train_fnr: List[float] = None,
    val_fnr: List[float] = None,
    output_path: str = None,
    title: str = "Training Learning Curves",
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 150,
    best_epoch: int = None,
    include_loss: bool = True,
    include_auc: bool = True
) -> Optional[plt.Figure]:
    """
    Plot comprehensive training curves (loss and metrics).

    Args:
        train_loss: Training loss history
        val_loss: Validation loss history
        train_auc: Training AUC history (optional)
        val_auc: Validation AUC history (optional)
        train_f1: Training F1 history (optional)
        val_f1: Validation F1 history (optional)
        train_accuracy: Training accuracy history (optional)
        val_accuracy: Validation accuracy history (optional)
        train_precision: Training precision history (optional)
        val_precision: Validation precision history (optional)
        train_recall: Training recall history (optional)
        val_recall: Validation recall history (optional)
        train_specificity: Training specificity history (optional)
        val_specificity: Validation specificity history (optional)
        train_fnr: Training false negative rate history (optional)
        val_fnr: Validation false negative rate history (optional)
        output_path: Path to save figure (optional)
        title: Figure title
        figsize: Figure size (width, height)
        dpi: DPI for saving
        best_epoch: Index of best epoch to mark (optional)
        include_loss: Whether to include the loss panel
        include_auc: Whether to include the AUC panel

    Returns:
        Matplotlib figure object
    """
    epochs = np.arange(1, len(train_loss) + 1)
    num_epochs = len(epochs)

    def _validated_series(name: str, series: Optional[List[float]]) -> Optional[List[float]]:
        if series is None:
            return None
        if len(series) != num_epochs:
            logger.warning(
                "Skipping %s panel: expected %d points, got %d",
                name, num_epochs, len(series)
            )
            return None
        return series

    def _add_metric_panel(metric_name: str,
                          train_series: Optional[List[float]],
                          val_series: Optional[List[float]],
                          ylabel: str,
                          ylim: Optional[Tuple[float, float]] = None,
                          baseline: Optional[float] = None):
        train_valid = _validated_series(metric_name, train_series)
        val_valid = _validated_series(metric_name, val_series)
        if train_valid is None and val_valid is None:
            return
        metric_panels.append({
            "metric_name": metric_name,
            "train": train_valid,
            "val": val_valid,
            "ylabel": ylabel,
            "ylim": ylim,
            "baseline": baseline
        })

    metric_panels: List[Dict[str, Any]] = []

    if include_loss:
        loss_train = _validated_series("Loss", train_loss)
        loss_val = _validated_series("Loss", val_loss)
        if loss_train is not None or loss_val is not None:
            metric_panels.append({
                "metric_name": "Loss",
                "train": loss_train,
                "val": loss_val,
                "ylabel": "Loss",
                "ylim": None,
                "baseline": None
            })

    _add_metric_panel("Specificity", train_specificity, val_specificity, ylabel="Specificity", ylim=(0.0, 1.0))
    _add_metric_panel("False Negative Rate", train_fnr, val_fnr, ylabel="FNR", ylim=(0.0, 1.0))
    _add_metric_panel("F1-Score", train_f1, val_f1, ylabel="F1-Score", ylim=(0.3, 1.0))
    _add_metric_panel("Accuracy", train_accuracy, val_accuracy, ylabel="Accuracy", ylim=(0.0, 1.0))
    _add_metric_panel("Precision", train_precision, val_precision, ylabel="Precision", ylim=(0.0, 1.0))
    _add_metric_panel("Recall", train_recall, val_recall, ylabel="Recall", ylim=(0.0, 1.0))
    if include_auc:
        _add_metric_panel("AUC", train_auc, val_auc, ylabel="AUC", ylim=(0.4, 1.0), baseline=0.5)

    # Filter out any None entries (in case loss failed validation)
    metric_panels = [panel for panel in metric_panels if panel["train"] is not None or panel["val"] is not None]

    if not metric_panels:
        logger.warning("No valid metric data provided for plotting learning curves.")
        return None

    num_plots = len(metric_panels)
    if num_plots <= 3:
        num_cols = num_plots
    elif num_plots == 4:
        num_cols = 2
    else:
        num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes_array = np.atleast_1d(axes).ravel()

    for plot_idx, panel in enumerate(metric_panels):
        ax = axes_array[plot_idx]
        train_series = panel["train"]
        val_series = panel["val"]

        if train_series is not None:
            ax.plot(epochs, train_series, 'b-', label=f'Train {panel["metric_name"]}',
                    linewidth=2, marker='o', markersize=4)
        if val_series is not None:
            ax.plot(epochs, val_series, 'r-', label=f'Val {panel["metric_name"]}',
                    linewidth=2, marker='s', markersize=4)

        if best_epoch is not None:
            ax.axvline(best_epoch + 1, color='g', linestyle='--', alpha=0.7,
                       label=f'Best (Epoch {best_epoch+1})')

        if panel["baseline"] is not None:
            ax.axhline(panel["baseline"], color='k', linestyle=':', alpha=0.5, label='Baseline')

        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel(panel["ylabel"], fontsize=11, fontweight='bold')
        ax.set_title(panel["metric_name"], fontsize=12, fontweight='bold')

        if panel["ylim"] is not None:
            ax.set_ylim(panel["ylim"])

        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    # Hide any unused axes
    for ax in axes_array[num_plots:]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved learning curves to: {output_path}")

    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    output_path: str = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    annotations: Dict[str, str] = None
) -> Optional[plt.Figure]:
    """
    Plot confusion matrix heatmap.

    Args:
        confusion_matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]]
        output_path: Path to save figure (optional)
        title: Figure title
        figsize: Figure size
        dpi: DPI for saving
        annotations: Custom annotations for matrix cells (optional)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize for visualization
    cm_display = confusion_matrix.astype('float') / confusion_matrix.sum() * 100

    sns.heatmap(
        cm_display,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        cbar=True,
        ax=ax,
        xticklabels=['Predicted Real', 'Predicted Fake'],
        yticklabels=['Actual Real', 'Actual Fake'],
        cbar_kws={'label': 'Percentage (%)'}
    )

    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    # Add count annotations
    for i in range(2):
        for j in range(2):
            count = confusion_matrix[i, j]
            ax.text(j + 0.5, i + 0.7, f'n={count}', ha='center', va='center',
                   fontsize=9, color='gray', style='italic')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved confusion matrix to: {output_path}")

    return fig


def plot_roc_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    auc_score: float = None,
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot ROC curve with AUC score.

    Args:
        targets: True labels
        probabilities: Model probability predictions
        auc_score: AUC score to display (optional, will calculate if not provided)
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)

    fpr, tpr, thresholds = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr) if auc_score is None else auc_score

    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved ROC curve to: {output_path}")

    return fig


def plot_precision_recall_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot Precision-Recall curve.

    Args:
        targets: True labels
        probabilities: Model probability predictions
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=figsize)

    precision, recall, _ = precision_recall_curve(targets, probabilities)
    ap = average_precision_score(targets, probabilities)

    ax.plot(recall, precision, color='darkblue', lw=2.5, label=f'PR curve (AP = {ap:.4f})')
    ax.fill_between(recall, precision, alpha=0.2, color='darkblue')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved PR curve to: {output_path}")

    return fig


def plot_probability_distribution(
    probabilities_real: np.ndarray,
    probabilities_fake: np.ndarray,
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot distribution of predicted probabilities for each class.

    Args:
        probabilities_real: Probabilities for real samples
        probabilities_fake: Probabilities for fake samples
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(probabilities_real, bins=50, alpha=0.6, label='Real Samples', color='blue', edgecolor='black')
    ax.hist(probabilities_fake, bins=50, alpha=0.6, label='Fake Samples', color='red', edgecolor='black')

    ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Classification Threshold (0.5)')
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Predicted Probability Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved probability distribution to: {output_path}")

    return fig


def plot_threshold_analysis(
    threshold_metrics: Dict[str, Dict[str, float]],
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 7),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot model performance across different classification thresholds.

    Args:
        threshold_metrics: Dictionary of threshold -> metrics
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    thresholds = sorted([float(k) for k in threshold_metrics.keys()])
    metrics_to_plot = {
        'f1': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
    }

    for threshold in thresholds:
        key = f'{threshold:.1f}'
        if key in threshold_metrics:
            for metric in metrics_to_plot:
                if metric in threshold_metrics[key]:
                    metrics_to_plot[metric].append(threshold_metrics[key][metric])

    colors = {'f1': 'red', 'accuracy': 'blue', 'precision': 'green', 'recall': 'purple'}
    for metric, values in metrics_to_plot.items():
        if values:
            ax.plot(thresholds, values, marker='o', linewidth=2, label=metric.capitalize(), color=colors[metric])

    ax.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance vs Classification Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved threshold analysis to: {output_path}")

    return fig


def plot_per_dataset_metrics(
    dataset_metrics: Dict[str, Dict[str, float]],
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot metrics comparison across multiple datasets.

    Args:
        dataset_metrics: Dictionary of dataset_name -> metrics
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    datasets = list(dataset_metrics.keys())
    metrics = ['auc', 'f1', 'accuracy']

    # Prepare data
    data_by_metric = {metric: [] for metric in metrics}
    for dataset in datasets:
        for metric in metrics:
            if metric in dataset_metrics[dataset]:
                data_by_metric[metric].append(dataset_metrics[dataset][metric])

    # Plot 1: Metrics comparison
    x = np.arange(len(datasets))
    width = 0.25

    for i, metric in enumerate(metrics):
        axes[0].bar(x + i * width, data_by_metric[metric], width, label=metric.upper())

    axes[0].set_xlabel('Dataset', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[0].set_title('Metrics Comparison Across Datasets', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(datasets, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Sample counts
    sample_counts = [dataset_metrics[ds].get('total_samples', 0) for ds in datasets]
    axes[1].bar(datasets, sample_counts, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    axes[1].set_title('Dataset Sizes', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved per-dataset metrics to: {output_path}")

    return fig


def plot_class_distribution(
    real_count: int,
    fake_count: int,
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot class distribution pie chart.

    Args:
        real_count: Number of real samples
        fake_count: Number of fake samples
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    sizes = [real_count, fake_count]
    labels = [f'Real\n({real_count:,})', f'Fake\n({fake_count:,})']
    colors = ['#66c2a5', '#fc8d62']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'}
    )

    ax.set_title('Class Distribution', fontsize=13, fontweight='bold', pad=20)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved class distribution to: {output_path}")

    return fig


def plot_calibration_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    n_bins: int = 10
) -> Optional[plt.Figure]:
    """
    Plot calibration curve (reliability diagram).

    Args:
        targets: True labels
        probabilities: Model probability predictions
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving
        n_bins: Number of bins for calibration

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate calibration curve
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    true_positives = []
    confidences = []

    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        if mask.sum() > 0:
            true_pos_rate = targets[mask].mean()
            avg_confidence = probabilities[mask].mean()
            true_positives.append(true_pos_rate)
            confidences.append(avg_confidence)

    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.plot(confidences, true_positives, 'o-', linewidth=2.5, markersize=8,
           color='darkblue', label='Model')

    ax.set_xlabel('Confidence (Mean Predicted Probability)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved calibration curve to: {output_path}")

    return fig


def plot_error_analysis(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Analyze and visualize classification errors (FP and FN).

    Args:
        targets: True labels
        predictions: Model predictions
        probabilities: Model probabilities
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # False Positives
    fp_mask = (predictions == 1) & (targets == 0)
    fp_probs = probabilities[fp_mask]

    # False Negatives
    fn_mask = (predictions == 0) & (targets == 1)
    fn_probs = probabilities[fn_mask]

    # Plot distributions
    axes[0].hist(fp_probs, bins=30, alpha=0.7, color='red', edgecolor='black', label=f'FP (n={fp_mask.sum()})')
    axes[0].axvline(fp_probs.mean() if len(fp_probs) > 0 else 0, color='darkred', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[0].set_title(f'False Positives (n={fp_mask.sum()})', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].hist(fn_probs, bins=30, alpha=0.7, color='orange', edgecolor='black', label=f'FN (n={fn_mask.sum()})')
    axes[1].axvline(fn_probs.mean() if len(fn_probs) > 0 else 0, color='darkorange', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[1].set_title(f'False Negatives (n={fn_mask.sum()})', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Error Analysis: Classification Errors', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved error analysis to: {output_path}")

    return fig


def plot_metrics_summary(
    metrics: Dict[str, float],
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot summary of all evaluation metrics as a bar chart.

    Args:
        metrics: Dictionary of metric_name -> value
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter metrics for visualization
    metric_names = [k for k in metrics.keys() if k not in ['loss'] and isinstance(metrics[k], (int, float))]
    metric_values = [metrics[k] for k in metric_names]

    colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Metrics Summary', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved metrics summary to: {output_path}")

    return fig


def plot_train_val_gap(
    epoch_metrics: List[Dict],
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    PR-3: Plot train/val gap to visualize overfitting.

    Args:
        epoch_metrics: List of epoch metric dictionaries with train/val data
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    if not epoch_metrics or len(epoch_metrics) == 0:
        logger.warning("No epoch metrics to plot")
        return None

    epochs = [m['epoch'] for m in epoch_metrics]
    train_loss = [m['train'].get('loss', 0) for m in epoch_metrics]
    val_loss = [m['val'].get('loss', 0) for m in epoch_metrics]
    train_auc = [m['train'].get('auc', 0.5) for m in epoch_metrics]
    val_auc = [m['val'].get('auc', 0.5) for m in epoch_metrics]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss gap
    loss_gap = np.array(val_loss) - np.array(train_loss)
    axes[0].plot(epochs, train_loss, 'b-', marker='o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', marker='s', label='Val Loss', linewidth=2)
    axes[0].fill_between(epochs, train_loss, val_loss, alpha=0.2, color='orange', label='Overfitting Gap')
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0].set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC gap
    auc_gap = np.array(val_auc) - np.array(train_auc)
    axes[1].plot(epochs, train_auc, 'b-', marker='o', label='Train AUC', linewidth=2)
    axes[1].plot(epochs, val_auc, 'r-', marker='s', label='Val AUC', linewidth=2)
    axes[1].fill_between(epochs, train_auc, val_auc, alpha=0.2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('AUC', fontsize=11, fontweight='bold')
    axes[1].set_title('Training vs Validation AUC', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0.4, 1.0])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Train/Val Gap Analysis (Overfitting Indicator)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved train/val gap to: {output_path}")

    return fig


def plot_lr_schedule(
    epoch_metrics: List[Dict],
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    PR-3: Plot learning rate schedule across epochs.

    Args:
        epoch_metrics: List of epoch metric dictionaries with lr data
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    if not epoch_metrics or len(epoch_metrics) == 0:
        logger.warning("No epoch metrics to plot")
        return None

    epochs = [m['epoch'] for m in epoch_metrics]
    lrs = [m.get('lr', 0.001) for m in epoch_metrics]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(epochs, lrs, 'o-', color='darkgreen', linewidth=2.5, markersize=8, label='Learning Rate')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Format y-axis in scientific notation if needed
    if min(lrs) < 0.0001:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved LR schedule to: {output_path}")

    return fig


def plot_image_grid(
    image_paths: List[str],
    title: str = "Error Samples",
    output_path: str = None,
    cols: int = 8,
    img_size: Tuple[int, int] = (112, 112),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    PR-3: Create grid of images from file paths.

    Args:
        image_paths: List of image file paths
        title: Title for the grid
        output_path: Path to save figure (optional)
        cols: Number of columns in grid
        img_size: Size to resize images to
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    import cv2
    from pathlib import Path as PathlibPath

    if not image_paths or len(image_paths) == 0:
        logger.warning("No images to plot")
        return None

    # Filter out non-existent images
    valid_paths = [p for p in image_paths if PathlibPath(p).exists()]
    if not valid_paths:
        logger.warning(f"No valid image paths found (checked {len(image_paths)} paths)")
        return None

    # Limit to first 32 images
    valid_paths = valid_paths[:32]
    num_images = len(valid_paths)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))

    # Flatten axes for easier iteration
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, img_path in enumerate(valid_paths):
        ax = axes[idx]
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, img_size)
                ax.imshow(img_resized)
                ax.set_title(PathlibPath(img_path).name, fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', fontsize=8)

        ax.axis('off')

    # Hide remaining subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'{title} ({num_images} samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved image grid to: {output_path}")

    return fig


def plot_roc_curve_precomputed(
    fpr: List[float],
    tpr: List[float],
    auc_score: float,
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot pre-computed ROC curve data.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score value
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved ROC curve to: {output_path}")

    return fig


def plot_precision_recall_precomputed(
    precision: List[float],
    recall: List[float],
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot pre-computed Precision-Recall curve data.

    Args:
        precision: Precision values
        recall: Recall values
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, color='darkblue', lw=2.5, label='PR curve')
    ax.fill_between(recall, precision, alpha=0.2, color='darkblue')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved PR curve to: {output_path}")

    return fig


def plot_calibration_curve_precomputed(
    bin_centers: List[float],
    avg_confidence: List[float],
    accuracy: List[float],
    output_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot pre-computed calibration curve data.

    Args:
        bin_centers: Center of confidence bins
        avg_confidence: Average confidence in each bin
        accuracy: Accuracy in each bin
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.plot(avg_confidence, accuracy, 'o-', linewidth=2.5, markersize=8,
            color='darkblue', label='Model')

    ax.set_xlabel('Confidence (Mean Predicted Probability)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved calibration curve to: {output_path}")

    return fig


def plot_confidence_histogram(
    bins: List[float],
    real_hist: List[int],
    fake_hist: List[int],
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Plot confidence histogram for real and fake samples.

    Robust to list inputs from JSON (converts to numpy arrays before math).

    Args:
        bins: Bin edges for histogram
        real_hist: Histogram counts for real samples
        fake_hist: Histogram counts for fake samples
        output_path: Path to save figure (optional)
        figsize: Figure size
        dpi: DPI for saving

    Returns:
        Matplotlib figure object
    """
    # Ensure array types for arithmetic
    bins_arr = np.asarray(bins, dtype=float).reshape(-1)
    real_arr = np.asarray(real_hist, dtype=float).reshape(-1)
    fake_arr = np.asarray(fake_hist, dtype=float).reshape(-1)

    if bins_arr.size < 2:
        logger.warning("Not enough bins to plot confidence histogram")
        return None

    # Use bin centers for bar positions
    bin_centers = (bins_arr[:-1] + bins_arr[1:]) / 2.0
    width = float(bins_arr[1] - bins_arr[0]) * 0.35  # group width

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(bin_centers - width / 2.0, real_arr, width, alpha=0.7,
           label='Real Samples', color='blue', edgecolor='black')
    ax.bar(bin_centers + width / 2.0, fake_arr, width, alpha=0.7,
           label='Fake Samples', color='red', edgecolor='black')

    ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Classification Threshold (0.5)')
    ax.set_xlabel('Model Confidence (Probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Distribution by Class', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"📊 Saved confidence histogram to: {output_path}")

    return fig


def create_comprehensive_report(
    evaluation_summary: Dict[str, Any],
    output_dir: str,
    dpi: int = 150
) -> None:
    """
    Create a comprehensive visualization report with all plots.

    Args:
        evaluation_summary: Evaluation summary dictionary from full_evaluation()
        output_dir: Directory to save all plots
        dpi: DPI for all saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📊 Creating comprehensive visualization report in: {output_dir}")

    metrics = evaluation_summary.get('metrics', {})
    cm = evaluation_summary.get('confusion_matrix', {})

    # 1. Metrics Summary
    if metrics:
        plot_metrics_summary(metrics, output_path=output_dir / 'metrics_summary.png', dpi=dpi)

    # 2. Confusion Matrix
    if cm:
        cm_array = np.array([[cm.get('true_negatives', 0), cm.get('false_positives', 0)],
                            [cm.get('false_negatives', 0), cm.get('true_positives', 0)]])
        plot_confusion_matrix(cm_array, output_path=output_dir / 'confusion_matrix.png', dpi=dpi)

    # 3. Class Distribution
    cd = evaluation_summary.get('class_distribution', {})
    if cd:
        plot_class_distribution(
            cd.get('real_count', 0),
            cd.get('fake_count', 0),
            output_path=output_dir / 'class_distribution.png',
            dpi=dpi
        )

    # 4. Threshold Analysis
    ta = evaluation_summary.get('threshold_analysis', {})
    if ta:
        plot_threshold_analysis(ta, output_path=output_dir / 'threshold_analysis.png', dpi=dpi)

    logger.info(f"✅ Comprehensive report created: {output_dir}")
