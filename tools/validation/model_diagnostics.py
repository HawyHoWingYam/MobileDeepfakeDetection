#!/usr/bin/env python3
"""
AWARE-NET Model Diagnostics Tool

Comprehensive diagnostic analysis for trained deepfake detection models.
Supports baseline and SupCon models with detailed performance visualization.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stage_00.baseline_model import EfficientNetV2B3Baseline
from src.stage_00.train_baseline import UnifiedDeepfakeDataset, setup_device_with_fallback


class ModelDiagnostics:
    """
    Comprehensive model diagnostic analysis

    Features:
    - Checkpoint loading (baseline and supcon models)
    - ROC curve analysis with optimal threshold
    - Confusion matrix at different thresholds
    - Precision-Recall curve
    - Prediction distribution analysis
    - Threshold optimization
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = 'baseline',
        model_name: str = 'tf_efficientnetv2_b0',
        device: Optional[torch.device] = None
    ):
        """
        Initialize diagnostics tool

        Args:
            checkpoint_path: Path to model checkpoint
            model_type: 'baseline' or 'supcon'
            model_name: Model architecture name
            device: torch device (auto-detected if None)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.model_name = model_name

        # Setup device
        if device is None:
            self.device, _ = setup_device_with_fallback()
        else:
            self.device = device

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Storage for predictions
        self.predictions = None
        self.targets = None
        self.probabilities = None

    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        print(f"\n{'='*80}")
        print(f"Loading Model Checkpoint")
        print(f"{'='*80}\n")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Model type: {self.model_type}")
        print(f"Architecture: {self.model_name}")

        # Create model
        if self.model_type == 'baseline':
            model = EfficientNetV2B3Baseline(
                num_classes=1,
                pretrained=False,
                model_name=self.model_name
            )
        elif self.model_type == 'supcon':
            # Import SupCon model
            from src.stage_01.train_supcon_quick import SupConModel
            model = SupConModel(
                model_name=self.model_name,
                projection_dim=512,
                pretrained=False
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        print(f"✓ Model loaded successfully\n")
        return model

    def evaluate_on_dataset(
        self,
        dataset_name: str,
        batch_size: int = 128,
        num_workers: int = 4
    ) -> Dict:
        """
        Evaluate model on dataset and collect predictions

        Args:
            dataset_name: Dataset to evaluate on
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*80}")
        print(f"Evaluating on Dataset: {dataset_name}")
        print(f"{'='*80}\n")

        # Create dataset
        dataset_short_names = {
            'celebdf_v2': 'celebdf_v2',
            'faceforensics_plus_plus': 'faceforensics',
            'deeperforensics_1_0': 'deeperforensics'
        }

        short_name = dataset_short_names[dataset_name]
        manifest_path = f'manifests/{short_name}_test.csv'

        print(f"Loading dataset from: {manifest_path}")

        test_dataset = UnifiedDeepfakeDataset(
            manifest_path=manifest_path,
            dataset_name=f'{dataset_name}_test',
            transform=None,
            use_augmentation=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"Dataset loaded: {len(test_dataset):,} samples\n")

        # Collect predictions
        all_predictions = []
        all_targets = []
        all_logits = []

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            for batch_data in pbar:
                # Unpack batch
                if len(batch_data) == 3:
                    data, targets, _ = batch_data
                else:
                    data, targets = batch_data

                data = data.to(self.device)
                targets = targets.cpu().numpy()

                # Forward pass
                logits = self.model(data)
                logits = logits.cpu().numpy().flatten()

                # Store results
                all_logits.extend(logits)
                all_targets.extend(targets)

        # Convert to numpy arrays
        self.targets = np.array(all_targets)
        logits_array = np.array(all_logits)

        # Apply sigmoid to get probabilities
        self.probabilities = 1 / (1 + np.exp(-logits_array))

        # Default threshold predictions
        self.predictions = (self.probabilities >= 0.5).astype(int)

        # Calculate basic metrics
        accuracy = (self.predictions == self.targets).mean()

        # Calculate AUC
        try:
            fpr, tpr, _ = roc_curve(self.targets, self.probabilities)
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.0

        results = {
            'num_samples': len(self.targets),
            'accuracy': accuracy,
            'auc': roc_auc
        }

        print(f"\n✓ Evaluation complete")
        print(f"  Samples: {results['num_samples']:,}")
        print(f"  Accuracy (threshold=0.5): {results['accuracy']:.4f}")
        print(f"  AUC-ROC: {results['auc']:.4f}\n")

        return results

    def plot_roc_curve(self, output_path: str):
        """Plot ROC curve with optimal threshold marked"""
        if self.targets is None or self.probabilities is None:
            raise ValueError("Must run evaluate_on_dataset() first")

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.targets, self.probabilities)
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold (Youden's index)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

        # Mark optimal threshold
        plt.scatter([optimal_fpr], [optimal_tpr], color='red', s=100, zorder=5,
                   label=f'Optimal threshold = {optimal_threshold:.3f}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Analysis', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ ROC curve saved to: {output_path}")
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  TPR at optimal: {optimal_tpr:.4f}")
        print(f"  FPR at optimal: {optimal_fpr:.4f}\n")

        return optimal_threshold

    def plot_confusion_matrix(self, output_path: str, threshold: float = 0.5):
        """Plot confusion matrix at given threshold"""
        if self.targets is None or self.probabilities is None:
            raise ValueError("Must run evaluate_on_dataset() first")

        # Get predictions at threshold
        predictions = (self.probabilities >= threshold).astype(int)

        # Calculate confusion matrix
        cm = confusion_matrix(self.targets, predictions)

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   square=True, ax=ax, annot_kws={'size': 14})

        # Add percentage annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                             ha="center", va="center", color="gray", fontsize=10)

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix (Threshold = {threshold:.3f})',
                    fontsize=14, fontweight='bold')
        ax.set_xticklabels(['Real (0)', 'Fake (1)'])
        ax.set_yticklabels(['Real (0)', 'Fake (1)'])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"✓ Confusion matrix saved to: {output_path}")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  TN: {tn:,}, FP: {fp:,}, FN: {fn:,}, TP: {tp:,}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}\n")

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def plot_threshold_analysis(self, output_path: str):
        """Plot accuracy/F1 vs threshold to find optimal working point"""
        if self.targets is None or self.probabilities is None:
            raise ValueError("Must run evaluate_on_dataset() first")

        # Try different thresholds
        thresholds = np.linspace(0, 1, 101)
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []

        for thresh in thresholds:
            preds = (self.probabilities >= thresh).astype(int)

            # Accuracy
            acc = (preds == self.targets).mean()
            accuracies.append(acc)

            # Confusion matrix metrics
            cm = confusion_matrix(self.targets, preds)
            tn, fp, fn, tp = cm.ravel()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)

        # Find best thresholds
        best_acc_idx = np.argmax(accuracies)
        best_f1_idx = np.argmax(f1_scores)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: Accuracy and F1
        ax1.plot(thresholds, accuracies, label='Accuracy', linewidth=2, color='blue')
        ax1.plot(thresholds, f1_scores, label='F1-Score', linewidth=2, color='green')
        ax1.axvline(thresholds[best_acc_idx], color='blue', linestyle='--', alpha=0.5,
                   label=f'Best Acc threshold = {thresholds[best_acc_idx]:.3f}')
        ax1.axvline(thresholds[best_f1_idx], color='green', linestyle='--', alpha=0.5,
                   label=f'Best F1 threshold = {thresholds[best_f1_idx]:.3f}')
        ax1.axvline(0.5, color='red', linestyle=':', alpha=0.5, label='Default (0.5)')
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Accuracy & F1-Score vs Threshold', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right plot: Precision and Recall
        ax2.plot(thresholds, precisions, label='Precision', linewidth=2, color='orange')
        ax2.plot(thresholds, recalls, label='Recall', linewidth=2, color='purple')
        ax2.axvline(0.5, color='red', linestyle=':', alpha=0.5, label='Default (0.5)')
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Precision & Recall vs Threshold', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Threshold analysis saved to: {output_path}")
        print(f"  Best accuracy threshold: {thresholds[best_acc_idx]:.3f} (Acc={accuracies[best_acc_idx]:.4f})")
        print(f"  Best F1 threshold: {thresholds[best_f1_idx]:.3f} (F1={f1_scores[best_f1_idx]:.4f})")
        print(f"  Default (0.5): Acc={accuracies[50]:.4f}, F1={f1_scores[50]:.4f}\n")

        return {
            'best_acc_threshold': float(thresholds[best_acc_idx]),
            'best_acc_value': float(accuracies[best_acc_idx]),
            'best_f1_threshold': float(thresholds[best_f1_idx]),
            'best_f1_value': float(f1_scores[best_f1_idx])
        }

    def plot_prediction_distribution(self, output_path: str):
        """Plot histogram of prediction probabilities for real vs fake"""
        if self.targets is None or self.probabilities is None:
            raise ValueError("Must run evaluate_on_dataset() first")

        # Separate by true label
        real_probs = self.probabilities[self.targets == 0]
        fake_probs = self.probabilities[self.targets == 1]

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Overlapping histograms
        ax1.hist(real_probs, bins=50, alpha=0.6, label='Real (label=0)', color='blue', density=True)
        ax1.hist(fake_probs, bins=50, alpha=0.6, label='Fake (label=1)', color='red', density=True)
        ax1.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
        ax1.set_xlabel('Predicted Probability (P(Fake))', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Prediction Distribution by True Label', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Box plots
        ax2.boxplot([real_probs, fake_probs], labels=['Real', 'Fake'],
                   showmeans=True, meanline=True)
        ax2.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
        ax2.set_ylabel('Predicted Probability (P(Fake))', fontsize=12)
        ax2.set_title('Prediction Distribution Box Plot', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate statistics
        real_mean = np.mean(real_probs)
        real_std = np.std(real_probs)
        fake_mean = np.mean(fake_probs)
        fake_std = np.std(fake_probs)

        print(f"✓ Prediction distribution saved to: {output_path}")
        print(f"  Real samples: mean={real_mean:.4f}, std={real_std:.4f}")
        print(f"  Fake samples: mean={fake_mean:.4f}, std={fake_std:.4f}")
        print(f"  Separation: {abs(fake_mean - real_mean):.4f}\n")

        return {
            'real_mean': float(real_mean),
            'real_std': float(real_std),
            'fake_mean': float(fake_mean),
            'fake_std': float(fake_std)
        }

    def generate_full_report(
        self,
        dataset_name: str,
        output_dir: str,
        batch_size: int = 128
    ):
        """
        Generate comprehensive diagnostic report

        Args:
            dataset_name: Dataset to evaluate
            output_dir: Directory to save outputs
            batch_size: Batch size for evaluation
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"GENERATING COMPREHENSIVE DIAGNOSTIC REPORT")
        print(f"{'='*80}\n")

        # 1. Evaluate on dataset
        eval_results = self.evaluate_on_dataset(dataset_name, batch_size=batch_size)

        # 2. Generate visualizations
        roc_path = output_dir / 'roc_curve.png'
        optimal_threshold = self.plot_roc_curve(str(roc_path))

        conf_matrix_path_default = output_dir / 'confusion_matrix_threshold_0.5.png'
        metrics_default = self.plot_confusion_matrix(str(conf_matrix_path_default), threshold=0.5)

        conf_matrix_path_optimal = output_dir / 'confusion_matrix_threshold_optimal.png'
        metrics_optimal = self.plot_confusion_matrix(str(conf_matrix_path_optimal),
                                                     threshold=optimal_threshold)

        threshold_path = output_dir / 'threshold_analysis.png'
        threshold_results = self.plot_threshold_analysis(str(threshold_path))

        dist_path = output_dir / 'prediction_distribution.png'
        dist_stats = self.plot_prediction_distribution(str(dist_path))

        # 3. Save JSON report
        report = {
            'checkpoint': str(self.checkpoint_path),
            'model_type': self.model_type,
            'model_name': self.model_name,
            'dataset': dataset_name,
            'num_samples': eval_results['num_samples'],
            'metrics': {
                'auc_roc': float(eval_results['auc']),
                'optimal_threshold': float(optimal_threshold),
                'default_threshold_0.5': metrics_default,
                'optimal_threshold_metrics': metrics_optimal,
                'best_accuracy_threshold': threshold_results['best_acc_threshold'],
                'best_f1_threshold': threshold_results['best_f1_threshold']
            },
            'distribution_stats': dist_stats
        }

        report_path = output_dir / 'diagnostic_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*80}")
        print(f"REPORT GENERATION COMPLETE")
        print(f"{'='*80}\n")
        print(f"Output directory: {output_dir}")
        print(f"Files generated:")
        print(f"  - {roc_path.name}")
        print(f"  - {conf_matrix_path_default.name}")
        print(f"  - {conf_matrix_path_optimal.name}")
        print(f"  - {threshold_path.name}")
        print(f"  - {dist_path.name}")
        print(f"  - {report_path.name}")
        print(f"\n{'='*80}\n")

        return report


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description='Model Diagnostics Tool for AWARE-NET',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--model-type', type=str, default='baseline',
                       choices=['baseline', 'supcon'],
                       help='Model type')
    parser.add_argument('--model-name', type=str, default='tf_efficientnetv2_b0',
                       help='Model architecture name')
    parser.add_argument('--test-dataset', type=str, required=True,
                       choices=['celebdf_v2', 'faceforensics_plus_plus', 'deeperforensics_1_0'],
                       help='Dataset to evaluate on')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for diagnostic results')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')

    args = parser.parse_args()

    # Create diagnostics tool
    diagnostics = ModelDiagnostics(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        model_name=args.model_name
    )

    # Generate full report
    diagnostics.generate_full_report(
        dataset_name=args.test_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
