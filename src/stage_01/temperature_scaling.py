"""
AWARE-NET Stage 1: Temperature Scaling Calibration Framework

This module implements temperature scaling for probability calibration in the
authenticity modeling paradigm. Proper calibration is crucial for cascade
system confidence thresholds.

Key Features:
- Temperature scaling optimization
- Calibration metrics (ECE, MCE, Brier Score)
- Reliability diagrams
- Conservative threshold strategies
- Cascade system integration

Mathematical Foundation:
Calibrated probability = softmax(logits / T)
where T > 1 reduces confidence, T < 1 increases confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Union
import logging
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class TemperatureScaling(nn.Module):
    """
    Temperature scaling module for probability calibration.

    Learns a single temperature parameter T to calibrate model predictions
    by minimizing the negative log-likelihood on a validation set.
    """

    def __init__(self, initial_temperature: float = 1.0):
        """
        Initialize temperature scaling module.

        Args:
            initial_temperature: Initial value for temperature parameter
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model logits (batch_size, num_classes)

        Returns:
            Temperature-scaled probabilities (batch_size, num_classes)
        """
        return torch.softmax(logits / self.temperature, dim=1)

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


class CalibrationMetrics:
    """
    Compute calibration metrics for model evaluation.
    """

    @staticmethod
    def expected_calibration_error(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 15
    ) -> Tuple[float, Dict]:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            y_prob: Predicted probabilities (n_samples,)
            y_true: True binary labels (n_samples,)
            n_bins: Number of bins for calibration

        Returns:
            ECE value and detailed bin statistics
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        bin_stats = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                bin_ece = abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                ece += bin_ece

                bin_stats.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'prop_in_bin': prop_in_bin,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'bin_ece': bin_ece,
                    'count': in_bin.sum()
                })

        return ece, {'bins': bin_stats, 'n_bins': n_bins}

    @staticmethod
    def maximum_calibration_error(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 15
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        Args:
            y_prob: Predicted probabilities
            y_true: True binary labels
            n_bins: Number of bins

        Returns:
            MCE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_error = 0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                max_error = max(max_error, abs(avg_confidence_in_bin - accuracy_in_bin))

        return max_error

    @staticmethod
    def brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Brier Score.

        Args:
            y_prob: Predicted probabilities
            y_true: True binary labels

        Returns:
            Brier score (lower is better)
        """
        return brier_score_loss(y_true, y_prob)

    @staticmethod
    def reliability_diagram_data(
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Compute data for reliability diagram.

        Args:
            y_prob: Predicted probabilities
            y_true: True labels
            n_bins: Number of bins

        Returns:
            Dictionary with bin data for plotting
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )

        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'n_bins': n_bins
        }


class ModelCalibrator:
    """
    Main calibration class that handles the full calibration pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        temperature_bounds: Tuple[float, float] = (0.1, 5.0),
        max_iter: int = 100,
        tolerance: float = 1e-6
    ):
        """
        Initialize model calibrator.

        Args:
            model: PyTorch model to calibrate
            temperature_bounds: (min_temp, max_temp) search bounds
            max_iter: Maximum optimization iterations
            tolerance: Optimization tolerance
        """
        self.model = model
        self.temperature_bounds = temperature_bounds
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.temperature_scaler = None
        self.calibration_results = None

        logger.info(f"ModelCalibrator initialized with bounds {temperature_bounds}")

    def calibrate(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device = None
    ) -> Dict:
        """
        Calibrate model using temperature scaling.

        Args:
            val_loader: Validation data loader
            device: Device to use for computation

        Returns:
            Calibration results dictionary
        """
        if device is None:
            device = next(self.model.parameters()).device

        logger.info("Starting temperature calibration...")

        # Collect predictions and labels
        all_logits = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Get model predictions
                if hasattr(self.model, 'forward'):
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                else:
                    logits = self.model(inputs)

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Processed {batch_idx + 1} batches")

        # Concatenate all predictions
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        logger.info(f"Collected {len(all_labels)} validation samples")

        # Compute pre-calibration metrics
        pre_probs = torch.softmax(all_logits, dim=1)
        if all_logits.shape[1] == 2:  # Binary classification
            pre_probs_np = pre_probs[:, 1].numpy()  # Probability of positive class
        else:
            pre_probs_np = pre_probs.max(dim=1)[0].numpy()  # Max probability

        pre_metrics = self._compute_calibration_metrics(pre_probs_np, all_labels.numpy())

        # Optimize temperature
        self.temperature_scaler = TemperatureScaling()
        optimizer = LBFGS(
            self.temperature_scaler.parameters(),
            lr=0.01,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance,
            tolerance_change=self.tolerance
        )

        def eval_loss():
            optimizer.zero_grad()
            calibrated_probs = self.temperature_scaler(all_logits)
            loss = F.nll_loss(torch.log(calibrated_probs + 1e-8), all_labels.long())
            loss.backward()
            return loss

        # Run optimization
        optimizer.step(eval_loss)

        # Compute post-calibration metrics
        with torch.no_grad():
            post_probs = self.temperature_scaler(all_logits)
            if all_logits.shape[1] == 2:
                post_probs_np = post_probs[:, 1].numpy()
            else:
                post_probs_np = post_probs.max(dim=1)[0].numpy()

        post_metrics = self._compute_calibration_metrics(post_probs_np, all_labels.numpy())

        # Store results
        self.calibration_results = {
            'temperature': self.temperature_scaler.get_temperature(),
            'pre_calibration': pre_metrics,
            'post_calibration': post_metrics,
            'improvement': {
                'ece': pre_metrics['ece'] - post_metrics['ece'],
                'mce': pre_metrics['mce'] - post_metrics['mce'],
                'brier': pre_metrics['brier'] - post_metrics['brier']
            },
            'num_samples': len(all_labels)
        }

        logger.info(f"Calibration completed: T={self.temperature_scaler.get_temperature():.4f}")
        logger.info(f"ECE improvement: {pre_metrics['ece']:.4f} → {post_metrics['ece']:.4f}")
        logger.info(f"MCE improvement: {pre_metrics['mce']:.4f} → {post_metrics['mce']:.4f}")

        return self.calibration_results

    def _compute_calibration_metrics(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """Compute all calibration metrics."""
        ece, ece_details = CalibrationMetrics.expected_calibration_error(y_prob, y_true)
        mce = CalibrationMetrics.maximum_calibration_error(y_prob, y_true)
        brier = CalibrationMetrics.brier_score(y_prob, y_true)
        reliability_data = CalibrationMetrics.reliability_diagram_data(y_prob, y_true)

        return {
            'ece': ece,
            'mce': mce,
            'brier': brier,
            'ece_details': ece_details,
            'reliability_data': reliability_data
        }

    def apply_calibration(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply learned temperature scaling to new logits.

        Args:
            logits: Raw model logits

        Returns:
            Calibrated probabilities
        """
        if self.temperature_scaler is None:
            raise ValueError("Model has not been calibrated yet. Call calibrate() first.")

        return self.temperature_scaler(logits)

    def get_temperature(self) -> float:
        """Get the learned temperature value."""
        if self.temperature_scaler is None:
            raise ValueError("Model has not been calibrated yet.")
        return self.temperature_scaler.get_temperature()

    def plot_reliability_diagram(
        self,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot reliability diagram before and after calibration.

        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if self.calibration_results is None:
            raise ValueError("No calibration results available")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Before calibration
        pre_data = self.calibration_results['pre_calibration']['reliability_data']
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax1.plot(
            pre_data['mean_predicted_value'],
            pre_data['fraction_of_positives'],
            'bo-',
            label=f"Before (ECE={self.calibration_results['pre_calibration']['ece']:.3f})"
        )
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Before Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # After calibration
        post_data = self.calibration_results['post_calibration']['reliability_data']
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax2.plot(
            post_data['mean_predicted_value'],
            post_data['fraction_of_positives'],
            'ro-',
            label=f"After (ECE={self.calibration_results['post_calibration']['ece']:.3f})"
        )
        ax2.set_xlabel('Mean Predicted Probability')
        ax2.set_ylabel('Fraction of Positives')
        ax2.set_title(f'After Calibration (T={self.get_temperature():.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reliability diagram saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()


class CascadeThresholdOptimizer:
    """
    Optimize conservative thresholds for cascade system integration.
    """

    def __init__(
        self,
        filter_rate_target: float = 0.90,
        false_negative_rate_max: float = 0.05
    ):
        """
        Initialize threshold optimizer for cascade system.

        Args:
            filter_rate_target: Target filtering rate (fraction passed to next stage)
            false_negative_rate_max: Maximum acceptable false negative rate
        """
        self.filter_rate_target = filter_rate_target
        self.false_negative_rate_max = false_negative_rate_max

    def optimize_thresholds(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """
        Find optimal conservative thresholds.

        Args:
            y_prob: Calibrated probabilities
            y_true: True labels

        Returns:
            Optimal threshold configuration
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        threshold_stats = []

        for threshold in thresholds:
            # Conservative prediction: predict fake if confidence < threshold
            y_pred = (y_prob < threshold).astype(int)

            # Calculate metrics
            true_negatives = ((y_true == 0) & (y_pred == 0)).sum()
            false_negatives = ((y_true == 1) & (y_pred == 0)).sum()
            true_positives = ((y_true == 1) & (y_pred == 1)).sum()
            false_positives = ((y_true == 0) & (y_pred == 1)).sum()

            # Cascade metrics
            filtered = (y_prob >= threshold).sum()  # Samples passed to next stage
            filter_rate = filtered / len(y_prob)

            fnr = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            fpr = false_positives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

            threshold_stats.append({
                'threshold': threshold,
                'filter_rate': filter_rate,
                'false_negative_rate': fnr,
                'false_positive_rate': fpr,
                'filtered_count': filtered,
                'meets_filter_target': filter_rate >= self.filter_rate_target,
                'meets_fnr_target': fnr <= self.false_negative_rate_max
            })

        # Find optimal threshold
        valid_thresholds = [
            stats for stats in threshold_stats
            if stats['meets_filter_target'] and stats['meets_fnr_target']
        ]

        if valid_thresholds:
            # Choose threshold with highest filter rate among valid options
            optimal = max(valid_thresholds, key=lambda x: x['filter_rate'])
        else:
            # Fallback: choose best compromise
            optimal = min(threshold_stats, key=lambda x: (
                abs(x['filter_rate'] - self.filter_rate_target) +
                max(0, x['false_negative_rate'] - self.false_negative_rate_max)
            ))

        return {
            'optimal_threshold': optimal['threshold'],
            'achieved_filter_rate': optimal['filter_rate'],
            'achieved_fnr': optimal['false_negative_rate'],
            'all_thresholds': threshold_stats,
            'meets_targets': optimal['meets_filter_target'] and optimal['meets_fnr_target']
        }


def test_temperature_scaling():
    """Test temperature scaling functionality."""
    print("Testing Temperature Scaling...")

    # Create synthetic test data
    batch_size = 100
    num_classes = 2

    # Overconfident logits
    logits = torch.randn(batch_size, num_classes) * 3  # High magnitude = overconfident
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test TemperatureScaling module
    temp_scaler = TemperatureScaling(initial_temperature=1.0)
    calibrated_probs = temp_scaler(logits)

    print(f"✓ Temperature scaling module works: shape {calibrated_probs.shape}")
    print(f"  Initial temperature: {temp_scaler.get_temperature():.3f}")

    # Test calibration metrics
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # Binary case
    labels_np = labels.numpy()

    ece, _ = CalibrationMetrics.expected_calibration_error(probs, labels_np)
    mce = CalibrationMetrics.maximum_calibration_error(probs, labels_np)
    brier = CalibrationMetrics.brier_score(probs, labels_np)

    print(f"✓ Calibration metrics: ECE={ece:.4f}, MCE={mce:.4f}, Brier={brier:.4f}")

    # Test threshold optimization
    threshold_optimizer = CascadeThresholdOptimizer(
        filter_rate_target=0.8,
        false_negative_rate_max=0.1
    )

    threshold_results = threshold_optimizer.optimize_thresholds(probs, labels_np)
    print(f"✓ Optimal threshold: {threshold_results['optimal_threshold']:.3f}")
    print(f"  Filter rate: {threshold_results['achieved_filter_rate']:.3f}")
    print(f"  FNR: {threshold_results['achieved_fnr']:.3f}")

    print("Temperature scaling tests passed! ✓")


if __name__ == "__main__":
    test_temperature_scaling()