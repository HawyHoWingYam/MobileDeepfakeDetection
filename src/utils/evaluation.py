"""
AWARE-NET: Unified Evaluation Module (Extracted from Stage 00)

This module provides comprehensive evaluation functionality extracted from
evaluate_baseline.py and made reusable across all stages.

Key Features:
- Multiple metrics: AUC, F1, Accuracy, Precision, Recall
- Confusion matrix calculation
- Detailed analysis capabilities
- Cross-stage compatibility

Usage:
    evaluator = ModelEvaluator(device=device)
    metrics = evaluator.evaluate(model, data_loader, mode='validation')
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Unified model evaluation system extracted from Stage 00.

    Provides comprehensive metrics calculation and analysis capabilities
    suitable for all training stages.
    """

    def __init__(self, device: torch.device, threshold: float = 0.5):
        """
        Initialize model evaluator.

        Args:
            device: Device for model evaluation
            threshold: Classification threshold for binary decisions
        """
        self.device = device
        self.threshold = threshold
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all stored metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []

    def evaluate_model(
        self,
        model: nn.Module,
        data_loader,
        criterion: nn.Module = None,
        mode: str = 'validation',
        return_detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data
            criterion: Loss function (optional)
            mode: Evaluation mode ('train', 'val', 'test')
            return_detailed: Whether to return detailed analysis

        Returns:
            Dictionary containing all calculated metrics
        """
        model.eval()
        self.reset_metrics()

        total_loss = 0.0
        num_batches = 0

        logger.info(f"Evaluating model on {mode} set...")

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"{mode.capitalize()} Evaluation")):
                images = images.to(self.device)
                targets = targets.float().to(self.device)

                # Forward pass
                outputs = model(images)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities >= self.threshold).float()

                # Calculate loss if criterion provided
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1

                # Store results
                self.predictions.extend(predictions.cpu().numpy())
                self.targets.extend(targets.cpu().numpy())
                self.probabilities.extend(probabilities.cpu().numpy())
                self.losses.extend([loss.item()] if criterion is not None else [])

        # Calculate comprehensive metrics
        metrics = self._calculate_all_metrics()

        # Add loss information
        if criterion is not None:
            avg_loss = total_loss / max(num_batches, 1)
            metrics['loss'] = avg_loss
            logger.info(f"{mode.capitalize()} Loss: {avg_loss:.4f}")

        # Log results
        self._log_metrics(metrics, mode)

        if return_detailed:
            metrics['detailed'] = self._get_detailed_analysis()
            metrics['confusion_matrix'] = self._get_confusion_matrix()

        return metrics

    def _calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        if len(self.targets) == 0:
            return {'auc': 0.0, 'f1': 0.0, 'accuracy': 0.0}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)

        # Core metrics
        auc = roc_auc_score(targets, probabilities)
        f1 = f1_score(targets, predictions)
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)

        # Additional metrics
        try:
            specificity = recall_score(targets, predictions, pos_label=0)
        except:
            specificity = 0.0

        # Calculate confusion matrix components
        tn, fp, fn, tp = self._get_confusion_components()

        # False Negative Rate (FNR) - important for cascade systems
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            'auc': auc,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'fnr': fnr,  # False Negative Rate
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

    def _get_confusion_components(self) -> Tuple[int, int, int, int]:
        """Get confusion matrix components."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        return tn, fp, fn, tp

    def _get_confusion_matrix(self) -> Dict[str, int]:
        """Get detailed confusion matrix."""
        tn, fp, fn, tp = self._get_confusion_components()
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': len(self.targets)
        }

    def _get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis for research purposes."""
        if len(self.targets) == 0:
            return {}

        targets = np.array(self.targets)
        predictions = np.array(self.predictions)
        probabilities = np.array(self.probabilities)

        try:
            report = classification_report(targets, predictions, output_dict=True)
            return report
        except Exception as e:
            logger.warning(f"Could not generate detailed report: {e}")
            return {}

    def _log_metrics(self, metrics: Dict[str, float], mode: str):
        """Log evaluation metrics."""
        logger.info(f"=== {mode.upper()} Results ===")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"False Negative Rate: {metrics['fnr']:.4f}")

        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            logger.info(f"Confusion Matrix:")
            logger.info(f"  True Negatives: {cm['true_negatives']}")
            logger.info(f"  False Positives: {cm['false_positives']}")
            logger.info(f"  False Negatives: {cm['false_negatives']}")
            logger.info(f"  True Positives: {cm['true_positives']}")

    def compare_models(
        self,
        model1: nn.Module,
        model2: nn.Module,
        data_loader,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        Compare two models on the same dataset.

        Args:
            model1: First model
            model2: Second model
            data_loader: DataLoader for evaluation
            model1_name: Name for first model
            model2_name: Name for second model

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {model1_name} vs {model2_name}...")

        # Evaluate both models
        metrics1 = self.evaluate_model(model1, data_loader, mode=f"{model1_name}_eval")
        metrics2 = self.evaluate_model(model2, data_loader, mode=f"{model2_name}_eval")

        # Calculate improvements
        improvements = {}
        for metric in ['auc', 'f1', 'accuracy']:
            if metric in metrics1 and metric in metrics2:
                diff = metrics2[metric] - metrics1[metric]
                improvements[f'{metric}_improvement'] = diff
                improvements[f'{metric}_improvement_pct'] = (diff / metrics1[metric]) * 100 if metrics1[metric] > 0 else 0.0

        return {
            model1_name: metrics1,
            model2_name: metrics2,
            'comparison': improvements,
            'winner': model2_name if improvements.get('auc_improvement', 0) > 0 else model1_name
        }

    def calculate_cascade_metrics(
        self,
        stage1_predictions: List[int],
        stage2_predictions: List[int],
        targets: List[int]
    ) -> Dict[str, float]:
        """
        Calculate cascade system specific metrics.

        Args:
            stage1_predictions: Predictions from first stage (binary)
            stage2_predictions: Predictions from second stage (binary)
            targets: True labels (binary)

        Returns:
            Dictionary with cascade-specific metrics
        """
        stage1_pred = np.array(stage1_predictions)
        stage2_pred = np.array(stage2_predictions)
        targets_np = np.array(targets)

        # Calculate stage-wise metrics
        stage1_f1 = f1_score(targets_np, stage1_pred)
        stage2_f1 = f1_score(targets_np, stage2_pred)
        stage1_auc = roc_auc_score(targets_np, stage1_pred)
        stage2_auc = roc_auc_score(targets_np, stage2_pred)

        # Calculate cascade efficiency
        stage1_direct = stage1_pred  # Stage 1 direct decisions
        stage2_used = np.where((stage1_pred == 0.5) & (stage2_pred != stage1_pred), stage2_pred, stage1_direct)  # Stage 2 intervention

        cascade_f1 = f1_score(targets_np, stage2_used)
        cascade_auc = roc_auc_score(targets_np, stage2_used)

        # Calculate cascade efficiency metrics
        total_samples = len(targets)
        stage1_only = np.sum(stage1_pred == stage2_used)
        stage2_intervention_rate = (total_samples - stage1_only) / total_samples

        return {
            'stage1_f1': stage1_f1,
            'stage2_f1': stage2_f1,
            'cascade_f1': cascade_f1,
            'stage1_auc': stage1_auc,
            'stage2_auc': stage2_auc,
            'cascade_auc': cascade_auc,
            'stage2_intervention_rate': stage2_intervention_rate,
            'total_samples': total_samples
        }