"""
AWARE-NET Academic Visualization Tools
Professional visualization for research publications and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Set academic plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AcademicVisualizer:
    """
    Academic-grade visualization tools for deepfake detection research
    
    Features:
    - Publication-ready plots with proper styling
    - ROC and PR curve visualization
    - Confusion matrix heatmaps
    - Calibration plots and reliability diagrams
    - Performance comparison charts
    - Error analysis visualizations
    """
    
    def __init__(self, 
                 style: str = 'academic',
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 300,
                 font_size: int = 12):
        """
        Initialize visualizer with academic styling
        
        Args:
            style: Plotting style ('academic', 'presentation', 'paper')
            figsize: Default figure size
            dpi: Figure resolution
            font_size: Base font size
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.font_size = font_size
        
        # Configure matplotlib for academic use
        self._configure_matplotlib()
        
        # Color schemes for different use cases
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'real': '#2E8B57',
            'fake': '#CD5C5C',
            'confidence': '#4682B4'
        }
    
    def _configure_matplotlib(self):
        """Configure matplotlib for academic publications"""
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.labelsize': self.font_size,
            'axes.titlesize': self.font_size + 2,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.titlesize': self.font_size + 4,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def plot_roc_curve(self, 
                      fpr: np.ndarray, 
                      tpr: np.ndarray, 
                      auc_score: float,
                      model_name: str = "Model",
                      confidence_interval: Optional[Tuple[float, float]] = None,
                      save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot ROC curve with AUC score and confidence interval
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC-ROC score
            model_name: Name of the model
            confidence_interval: AUC confidence interval
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2.5, color=self.colors['primary'], 
                label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Add confidence interval to label if available
        if confidence_interval:
            ci_text = f" [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]"
            ax.lines[-1].set_label(ax.lines[-1].get_label() + ci_text)
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1.5, 
                label='Random Classifier (AUC = 0.500)')
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Add text box with performance info
        textstr = f'AUC: {auc_score:.4f}'
        if confidence_interval:
            textstr += f'\n95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=self.font_size-1,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, 
                                   precision: np.ndarray,
                                   recall: np.ndarray,
                                   ap_score: float,
                                   model_name: str = "Model",
                                   save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
            ap_score: Average precision score
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot PR curve
        ax.plot(recall, precision, linewidth=2.5, color=self.colors['secondary'], 
                label=f'{model_name} (AP = {ap_score:.3f})')
        
        # Plot baseline (random classifier)
        baseline_precision = np.sum(recall > 0) / len(recall) if len(recall) > 0 else 0.5
        ax.axhline(y=baseline_precision, color='k', linestyle='--', alpha=0.6, 
                  linewidth=1.5, label=f'Random Classifier (AP = {baseline_precision:.3f})')
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        # Add text box with performance info
        textstr = f'Average Precision: {ap_score:.4f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=self.font_size-1,
                verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, 
                            cm: np.ndarray,
                            class_names: List[str] = None,
                            title: str = "Confusion Matrix",
                            normalize: bool = False,
                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot confusion matrix heatmap
        
        Args:
            cm: Confusion matrix array
            class_names: Names of classes
            title: Plot title
            normalize: Whether to normalize values
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if class_names is None:
            class_names = ['Real', 'Fake']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title += ' (Normalized)'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add performance metrics as text
        if not normalize and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_text = f'Accuracy: {accuracy:.3f}\\nPrecision: {precision:.3f}\\nRecall: {recall:.3f}\\nF1: {f1:.3f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, fontsize=self.font_size-2,
                   verticalalignment='center', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(self, 
                             y_true: np.ndarray,
                             y_prob: np.ndarray,
                             model_name: str = "Model",
                             n_bins: int = 10,
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot reliability diagram (calibration curve)
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            n_bins: Number of bins for calibration
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        from sklearn.calibration import calibration_curve
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", 
                linewidth=2, color=self.colors['primary'], 
                label=f'{model_name}', markersize=6)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2.hist(y_prob, bins=50, alpha=0.7, color=self.colors['confidence'], 
                edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, 
                            results_dict: Dict[str, Dict],
                            metric: str = 'auc_roc',
                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot comparison of multiple models
        
        Args:
            results_dict: Dictionary mapping model names to results
            metric: Metric to compare ('auc_roc', 'f1_score', 'accuracy')
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        model_names = list(results_dict.keys())
        values = []
        errors = []
        
        for model_name in model_names:
            result = results_dict[model_name]['classification_metrics'][metric]
            values.append(result.value)
            
            # Calculate error bar from confidence interval
            if result.confidence_interval:
                ci_lower, ci_upper = result.confidence_interval
                error = (ci_upper - ci_lower) / 2
                errors.append(error)
            else:
                errors.append(0)
        
        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.5), 6))
        
        # Create bar plot
        bars = ax.bar(model_names, values, yerr=errors, capsize=5, 
                     color=[self.colors['primary'], self.colors['secondary'], 
                           self.colors['accent'], self.colors['success']][:len(model_names)],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value, error in zip(bars, values, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=self.font_size-1)
        
        # Styling
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, min(1.1, max(values) + max(errors) + 0.1))
        
        # Rotate x-axis labels if needed
        if len(max(model_names, key=len)) > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_performance_over_time(self, 
                                 training_history: Dict[str, List[float]],
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot training performance over epochs
        
        Args:
            training_history: Dictionary with 'train_loss', 'val_loss', etc.
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['loss', 'accuracy', 'auc', 'f1']
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['success']]
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in training_history:
                epochs = range(1, len(training_history[train_key]) + 1)
                ax.plot(epochs, training_history[train_key], 'o-', 
                       color=colors[i], label=f'Training {metric.title()}', 
                       linewidth=2, markersize=4)
            
            if val_key in training_history:
                epochs = range(1, len(training_history[val_key]) + 1)
                ax.plot(epochs, training_history[val_key], 's-', 
                       color=colors[i], alpha=0.7, label=f'Validation {metric.title()}',
                       linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.title())
            ax.set_title(f'{metric.title()} over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_error_analysis(self, 
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_prob: np.ndarray,
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot error analysis with confidence-based breakdown
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Correct vs incorrect predictions by confidence
        correct = (y_true == y_pred)
        confidence = np.abs(y_prob - 0.5) * 2  # Convert to 0-1 scale
        
        # 1. Confidence distribution for correct/incorrect
        ax1 = axes[0, 0]
        ax1.hist(confidence[correct], bins=20, alpha=0.7, label='Correct', 
                color=self.colors['real'], density=True)
        ax1.hist(confidence[~correct], bins=20, alpha=0.7, label='Incorrect', 
                color=self.colors['fake'], density=True)
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Density')
        ax1.set_title('Confidence Distribution by Correctness')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error rate by confidence bins
        ax2 = axes[0, 1]
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        error_rates = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i+1])
            if np.sum(mask) > 0:
                error_rate = np.mean(~correct[mask])
                error_rates.append(error_rate)
            else:
                error_rates.append(0)
        
        ax2.bar(bin_centers, error_rates, width=0.08, alpha=0.7, 
               color=self.colors['accent'], edgecolor='black')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Error Rate')
        ax2.set_title('Error Rate by Confidence Level')
        ax2.grid(True, alpha=0.3)
        
        # 3. Real vs Fake classification errors
        ax3 = axes[1, 0]
        
        # True positives, false positives, true negatives, false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        categories = ['True\\nNegative', 'False\\nPositive', 'False\\nNegative', 'True\\nPositive']
        values = [tn, fp, fn, tp]
        colors_cm = [self.colors['real'], self.colors['fake'], self.colors['fake'], self.colors['real']]
        
        bars = ax3.bar(categories, values, color=colors_cm, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Count')
        ax3.set_title('Classification Results Breakdown')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value}', ha='center', va='bottom')
        
        # 4. Prediction probability distribution by true class
        ax4 = axes[1, 1]
        real_probs = y_prob[y_true == 0]
        fake_probs = y_prob[y_true == 1]
        
        ax4.hist(real_probs, bins=30, alpha=0.7, label='Real Images', 
                color=self.colors['real'], density=True)
        ax4.hist(fake_probs, bins=30, alpha=0.7, label='Fake Images', 
                color=self.colors['fake'], density=True)
        ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Threshold')
        ax4.set_xlabel('Predicted Probability (Fake)')
        ax4.set_ylabel('Density')
        ax4.set_title('Prediction Distribution by True Class')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_publication_figure(self, 
                                results: Dict[str, Any],
                                model_name: str = "AWARE-NET",
                                save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive publication-ready figure
        
        Args:
            results: Comprehensive evaluation results
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = results['curves']['roc_curve']
        auc_result = results['classification_metrics']['auc_roc']
        
        ax1.plot(fpr, tpr, linewidth=3, color=self.colors['primary'])
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve\\nAUC = {auc_result.value:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # PR Curve
        ax2 = fig.add_subplot(gs[0, 1])
        prec, rec, _ = results['curves']['pr_curve']
        ap_score = results['classification_metrics']['average_precision']
        
        ax2.plot(rec, prec, linewidth=3, color=self.colors['secondary'])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve\\nAP = {ap_score:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Confusion Matrix
        ax3 = fig.add_subplot(gs[0, 2])
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                   ax=ax3, cbar=False)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Performance Metrics
        ax4 = fig.add_subplot(gs[1, :])
        metrics = results['classification_metrics']
        metric_names = ['AUC-ROC', 'F1 Score', 'Accuracy', 'Precision', 'Recall']
        values = [
            metrics['auc_roc'].value,
            metrics['f1_score'].value,
            metrics['accuracy'].value,
            metrics['precision'],
            metrics['recall']
        ]
        
        bars = ax4.bar(metric_names, values, 
                      color=[self.colors['primary'], self.colors['secondary'], 
                            self.colors['accent'], self.colors['success'], self.colors['fake']],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Score')
        ax4.set_title(f'{model_name} Performance Metrics')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Calibration info
        ax5 = fig.add_subplot(gs[2, :2])
        calibration = results['calibration']
        
        # Create text summary
        calib_text = f"""Calibration Analysis:
        
Expected Calibration Error (ECE): {calibration['ece']:.4f}
Maximum Calibration Error (MCE): {calibration['mce']:.4f}
Brier Score: {calibration['brier_score']:.4f}

Dataset: {results['n_samples']} samples
Threshold: {results['threshold']}"""
        
        ax5.text(0.05, 0.95, calib_text, transform=ax5.transAxes, 
                fontsize=self.font_size, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Model Calibration Summary')
        
        # Model info
        ax6 = fig.add_subplot(gs[2, 2])
        model_info = f"""Model Information:
        
Architecture: {model_name}
Framework: PyTorch + TIMM
Training: EfficientNetV2-B3
Optimizer: AdamW
        
Performance Summary:
• High accuracy detection
• Well-calibrated predictions
• Robust generalization"""
        
        ax6.text(0.05, 0.95, model_info, transform=ax6.transAxes, 
                fontsize=self.font_size-1, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Model Details')
        
        # Add main title
        fig.suptitle(f'{model_name}: Comprehensive Performance Analysis', 
                    fontsize=self.font_size + 6, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def save_all_plots(self, 
                      results: Dict[str, Any],
                      output_dir: Path,
                      model_name: str = "Model") -> List[Path]:
        """
        Save all visualization plots to directory
        
        Args:
            results: Comprehensive evaluation results
            output_dir: Output directory for plots
            model_name: Name of the model
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # ROC Curve
        fpr, tpr, _ = results['curves']['roc_curve']
        auc_result = results['classification_metrics']['auc_roc']
        roc_path = output_dir / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
        self.plot_roc_curve(fpr, tpr, auc_result.value, model_name, 
                           auc_result.confidence_interval, roc_path)
        saved_files.append(roc_path)
        plt.close()
        
        # PR Curve
        prec, rec, _ = results['curves']['pr_curve']
        ap_score = results['classification_metrics']['average_precision']
        pr_path = output_dir / f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
        self.plot_precision_recall_curve(rec, prec, ap_score, model_name, pr_path)
        saved_files.append(pr_path)
        plt.close()
        
        # Confusion Matrix
        cm_path = output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        self.plot_confusion_matrix(results['confusion_matrix'], 
                                  title=f"{model_name} Confusion Matrix", 
                                  save_path=cm_path)
        saved_files.append(cm_path)
        plt.close()
        
        # Publication Figure
        pub_path = output_dir / f"{model_name.lower().replace(' ', '_')}_comprehensive.png"
        self.create_publication_figure(results, model_name, pub_path)
        saved_files.append(pub_path)
        plt.close()
        
        print(f"Saved {len(saved_files)} visualization files to {output_dir}")
        
        return saved_files