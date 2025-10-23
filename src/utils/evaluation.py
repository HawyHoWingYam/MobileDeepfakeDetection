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

        # Warn if predictions collapsed to single class (degenerate prediction)
        if tp == 0 or fp == 0 or tn == 0 or fn == 0:
            logger.warning("⚠️  DEGENERATE PREDICTION DETECTED!")
            logger.warning(f"   Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            logger.warning(f"   Predictions may have collapsed to a single class")
            logger.warning(f"   This is common in early epochs but monitor for persistence")
            if precision == 0 or recall == 0:
                logger.warning(f"   Precision={precision:.4f}, Recall={recall:.4f}")

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

    def full_evaluation(
        self,
        model: nn.Module,
        data_loader,
        criterion: nn.Module = None,
        mode: str = 'validation',
        thresholds: List[float] = None,
        top_k_errors: int = 32
    ) -> Dict[str, Any]:
        """
        PR-3: Comprehensive evaluation with ROC/PR curves, calibration data, and error paths.

        Consolidates all evaluation data into a single unified output for easier
        artifact management and reproducibility.

        Args:
            model: Model to evaluate
            data_loader: DataLoader for evaluation data (supports 2-tuple or 3-tuple with metadata)
            criterion: Loss function (optional)
            mode: Evaluation mode ('train', 'val', 'test')
            thresholds: List of thresholds to analyze (default: [0.3, 0.5, 0.7])
            top_k_errors: Number of top errors to track by path (default: 32)

        Returns:
            Dictionary with full evaluation summary ready for JSON serialization
        """
        if thresholds is None:
            thresholds = [0.3, 0.5, 0.7]

        logger.info(f"📊 Running full evaluation on {mode} set...")

        # Extended evaluation to handle metadata
        model.eval()
        self.reset_metrics()
        self.sample_paths = []  # PR-3: Track sample paths

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"{mode.capitalize()} Evaluation")):
                # Support both (img, target) and (img, target, metadata) with default collate
                metadata = None
                try:
                    images, targets, metadata = batch_data
                except (ValueError, TypeError):
                    images, targets = batch_data

                # Collect sample paths when available
                if metadata is not None:
                    try:
                        # Case 1: metadata is dict of lists (default PyTorch collate for list[dict])
                        if isinstance(metadata, dict) and 'path' in metadata:
                            paths = metadata['path']
                            if isinstance(paths, (list, tuple)):
                                self.sample_paths.extend([str(p) for p in paths])
                            else:
                                self.sample_paths.append(str(paths))
                        else:
                            # Case 2: iterable of dicts or strings
                            for m in metadata:
                                if isinstance(m, dict) and 'path' in m:
                                    self.sample_paths.append(str(m['path']))
                                else:
                                    self.sample_paths.append(str(m))
                    except Exception:
                        # Be permissive: don't let metadata parsing break evaluation
                        pass

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

        # Calculate metrics
        metrics = self._calculate_all_metrics()
        if criterion is not None:
            avg_loss = total_loss / max(num_batches, 1)
            metrics['loss'] = avg_loss

        # Get arrays for curve analysis
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities).flatten()

        # PR-3: Collect ROC curve data
        from sklearn.metrics import roc_curve, precision_recall_curve
        try:
            fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
            if len(fpr) > 512:
                indices = np.linspace(0, len(fpr)-1, 512, dtype=int)
                fpr, tpr, roc_thresholds = fpr[indices], tpr[indices], roc_thresholds[indices]
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }
        except Exception as e:
            logger.warning(f"⚠️  Could not compute ROC curve: {e}")
            roc_data = {'fpr': [], 'tpr': [], 'thresholds': []}

        # PR-3: Collect PR curve data
        try:
            precision, recall, pr_thresholds = precision_recall_curve(targets, probabilities)
            if len(precision) > 512:
                indices = np.linspace(0, len(precision)-1, 512, dtype=int)
                precision, recall = precision[indices], recall[indices]
            pr_data = {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        except Exception as e:
            logger.warning(f"⚠️  Could not compute PR curve: {e}")
            pr_data = {'precision': [], 'recall': []}

        # PR-3: Collect calibration curve data
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        calibration_data = {
            'bin_edges': bin_edges.tolist(),
            'bin_centers': bin_centers.tolist(),
            'avg_confidence': [],
            'accuracy': []
        }

        for i in range(n_bins):
            mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            if mask.sum() > 0:
                calibration_data['avg_confidence'].append(float(probabilities[mask].mean()))
                calibration_data['accuracy'].append(float(targets[mask].mean()))
            else:
                calibration_data['avg_confidence'].append(float(bin_centers[i]))
                calibration_data['accuracy'].append(0.0)

        # PR-3: Collect confidence histograms by class
        real_mask = targets == 0
        fake_mask = targets == 1
        bins = np.linspace(0, 1, 50)
        real_hist, _ = np.histogram(probabilities[real_mask], bins=bins)
        fake_hist, _ = np.histogram(probabilities[fake_mask], bins=bins)

        confidence_hist = {
            'bins': bins.tolist(),
            'real_hist': real_hist.tolist(),
            'fake_hist': fake_hist.tolist()
        }

        # PR-3: Collect top-K error paths
        topk_fp_paths = []
        topk_fn_paths = []

        if hasattr(self, 'sample_paths') and len(self.sample_paths) == len(targets):
            fp_indices = np.where((predictions == 1) & (targets == 0))[0]
            fn_indices = np.where((predictions == 0) & (targets == 1))[0]

            if len(fp_indices) > 0:
                fp_probs = probabilities[fp_indices]
                top_fp_idx = fp_indices[np.argsort(-fp_probs)[:top_k_errors]]
                topk_fp_paths = [self.sample_paths[i] for i in top_fp_idx if i < len(self.sample_paths)]

            if len(fn_indices) > 0:
                fn_probs = 1 - probabilities[fn_indices]
                top_fn_idx = fn_indices[np.argsort(-fn_probs)[:top_k_errors]]
                topk_fn_paths = [self.sample_paths[i] for i in top_fn_idx if i < len(self.sample_paths)]

        # Build comprehensive summary
        summary = {
            'evaluation_mode': mode,
            'total_samples': len(targets),
            'metrics': {
                'auc': float(metrics.get('auc', 0.0)),
                'f1': float(metrics.get('f1', 0.0)),
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'precision': float(metrics.get('precision', 0.0)),
                'recall': float(metrics.get('recall', 0.0)),
                'specificity': float(metrics.get('specificity', 0.0)),
                'fnr': float(metrics.get('fnr', 0.0)),
            },
            'loss': float(metrics.get('loss', 0.0)) if 'loss' in metrics else None,
            # Include explicit confusion matrix for downstream plotting
            'confusion_matrix': self._get_confusion_matrix(),
            'class_distribution': {
                'real_count': int(np.sum(targets == 0)),
                'fake_count': int(np.sum(targets == 1)),
                'real_percentage': float(np.mean(targets == 0) * 100),
                'fake_percentage': float(np.mean(targets == 1) * 100),
            },
            'probability_statistics': {
                'min': float(probabilities.min()),
                'max': float(probabilities.max()),
                'mean': float(probabilities.mean()),
                'median': float(np.median(probabilities)),
                'std': float(probabilities.std()),
            },
            'threshold_analysis': self._analyze_thresholds(targets, probabilities, thresholds),
            'classification_report': metrics.get('detailed', {}),
            # PR-3: New fields for plotting
            'roc_curve': roc_data,
            'pr_curve': pr_data,
            'calibration_bins': calibration_data,
            'confidence_hist': confidence_hist,
            'topk_fp_paths': topk_fp_paths,
            'topk_fn_paths': topk_fn_paths,
            # Non-serialized convenience fields for additional plots (used immediately by caller)
            'targets_array': targets,
            'probabilities_array': probabilities,
            'predictions_array': predictions,
        }

        logger.info(f"✅ Full evaluation completed: {summary['total_samples']} samples analyzed")
        logger.info(f"   Collected {len(topk_fp_paths)} top FP paths, {len(topk_fn_paths)} top FN paths")

        return summary

    def _analyze_thresholds(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        thresholds: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze model performance at different classification thresholds.

        Args:
            targets: True labels
            probabilities: Model probability predictions
            thresholds: List of thresholds to analyze

        Returns:
            Dictionary with threshold analysis
        """
        threshold_results = {}

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)

            try:
                metrics = {
                    'threshold': float(threshold),
                    'f1': float(f1_score(targets, predictions)),
                    'accuracy': float(accuracy_score(targets, predictions)),
                    'precision': float(precision_score(targets, predictions)),
                    'recall': float(recall_score(targets, predictions)),
                    'specificity': float(recall_score(targets, predictions, pos_label=0)),
                }
                threshold_results[f'{threshold:.1f}'] = metrics
            except Exception as e:
                logger.warning(f"Could not calculate metrics for threshold {threshold}: {e}")

        return threshold_results

    def save_predictions(
        self,
        output_path: str,
        include_metadata: bool = True
    ) -> None:
        """
        Phase 2: Save detailed predictions for post-hoc analysis.

        Args:
            output_path: Path to save predictions (CSV or JSON)
            include_metadata: Whether to include sample metadata
        """
        import json
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions_data = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)

        # Create records for each sample
        records = []
        for i in range(len(targets)):
            records.append({
                'sample_id': int(i),
                'target': int(targets[i]),
                'probability': float(probabilities[i]),
                'prediction': int(predictions_data[i]),
                'correct': int(predictions_data[i]) == int(targets[i]),
            })

        # Save as JSON
        if str(output_path).endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)
        # Save as CSV
        elif str(output_path).endswith('.csv'):
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)

        logger.info(f"💾 Predictions saved to: {output_path}")


class PredictionsCollector:
    """
    Phase 2: Collects and manages predictions for comprehensive analysis.

    Tracks predictions across multiple evaluation passes and datasets,
    enabling post-hoc analysis and per-dataset metrics calculation.
    """

    def __init__(self):
        """Initialize predictions collector."""
        self.predictions = []
        self.metadata = []

    def add_batch(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
        dataset_name: str = None,
        batch_id: int = None
    ) -> None:
        """
        Add a batch of predictions.

        Args:
            predictions: Binary predictions
            targets: True labels
            probabilities: Probability predictions
            dataset_name: Name of dataset this batch belongs to (optional)
            batch_id: Batch identifier (optional)
        """
        for i in range(len(targets)):
            self.predictions.append({
                'prediction': int(predictions[i]),
                'target': int(targets[i]),
                'probability': float(probabilities[i]),
                'correct': int(predictions[i]) == int(targets[i]),
                'dataset': dataset_name,
                'batch_id': batch_id,
                'sample_id': len(self.predictions),
            })

    def get_metrics_by_dataset(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics grouped by dataset.

        Returns:
            Dictionary with per-dataset metrics
        """
        datasets = set(p.get('dataset') for p in self.predictions if p.get('dataset'))

        results = {}
        for dataset in datasets:
            dataset_preds = [p for p in self.predictions if p.get('dataset') == dataset]

            if not dataset_preds:
                continue

            targets = np.array([p['target'] for p in dataset_preds])
            predictions = np.array([p['prediction'] for p in dataset_preds])
            probabilities = np.array([p['probability'] for p in dataset_preds])

            try:
                results[dataset] = {
                    'total_samples': len(dataset_preds),
                    'auc': float(roc_auc_score(targets, probabilities)),
                    'f1': float(f1_score(targets, predictions)),
                    'accuracy': float(accuracy_score(targets, predictions)),
                    'precision': float(precision_score(targets, predictions)),
                    'recall': float(recall_score(targets, predictions)),
                }
            except Exception as e:
                logger.warning(f"Could not calculate metrics for {dataset}: {e}")
                results[dataset] = {'error': str(e)}

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics from collected predictions.

        Returns:
            Dictionary with statistics
        """
        if not self.predictions:
            return {}

        predictions = np.array([p['prediction'] for p in self.predictions])
        targets = np.array([p['target'] for p in self.predictions])
        probabilities = np.array([p['probability'] for p in self.predictions])
        correct = np.array([p['correct'] for p in self.predictions])

        return {
            'total_samples': len(self.predictions),
            'correct_predictions': int(np.sum(correct)),
            'incorrect_predictions': int(np.sum(~correct)),
            'accuracy': float(np.mean(correct)),
            'real_samples': int(np.sum(targets == 0)),
            'fake_samples': int(np.sum(targets == 1)),
            'true_positives': int(np.sum((predictions == 1) & (targets == 1))),
            'true_negatives': int(np.sum((predictions == 0) & (targets == 0))),
            'false_positives': int(np.sum((predictions == 1) & (targets == 0))),
            'false_negatives': int(np.sum((predictions == 0) & (targets == 1))),
            'probability_mean': float(probabilities.mean()),
            'probability_std': float(probabilities.std()),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            'total_predictions': len(self.predictions),
            'statistics': self.get_statistics(),
            'per_dataset_metrics': self.get_metrics_by_dataset(),
            'predictions': self.predictions[:100],  # Include first 100 for inspection
        }
