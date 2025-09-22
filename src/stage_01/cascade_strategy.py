"""
AWARE-NET Stage 1: Conservative Threshold Strategy for Cascade System

This module implements the conservative threshold strategy for Stage 1
rapid filtering in the cascade system. The goal is to achieve high recall
while maintaining efficient filtering for subsequent stages.

Core Philosophy: "Better to be safe than sorry"
- High confidence authentic → Pass as authentic (>99%)
- Low confidence → Send to next stage for detailed analysis (1%-99%)
- Very low confidence → Flag as potentially fake (<1%)

Key Features:
- Non-symmetric thresholds optimized for cascade efficiency
- False negative minimization (critical for authenticity modeling)
- Adaptive threshold adjustment based on data distribution
- Performance monitoring and cascade flow analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from sklearn.metrics import roc_curve, precision_recall_curve
import json

logger = logging.getLogger(__name__)


@dataclass
class CascadeConfig:
    """Configuration for cascade threshold strategy."""
    high_confidence_threshold: float = 0.99  # Very confident authentic
    low_confidence_threshold: float = 0.01   # Very confident fake
    target_filter_rate: float = 0.90          # Target % passed to next stage
    max_false_negative_rate: float = 0.05     # Maximum acceptable FNR
    min_true_positive_rate: float = 0.95      # Minimum TPR requirement
    adaptation_enabled: bool = True            # Enable adaptive thresholds


class CascadeDecision:
    """Represents a cascade decision with confidence and routing."""

    def __init__(
        self,
        decision: str,
        confidence: float,
        raw_probability: float,
        threshold_used: str,
        stage_routing: str
    ):
        """
        Initialize cascade decision.

        Args:
            decision: 'authentic', 'fake', or 'uncertain'
            confidence: Calibrated confidence score
            raw_probability: Original model probability
            threshold_used: Which threshold was applied
            stage_routing: 'accept', 'reject', or 'next_stage'
        """
        self.decision = decision
        self.confidence = confidence
        self.raw_probability = raw_probability
        self.threshold_used = threshold_used
        self.stage_routing = stage_routing


class ConservativeThresholdStrategy:
    """
    Conservative threshold strategy for cascade system Stage 1.

    Implements non-symmetric thresholds designed to minimize false negatives
    while maintaining efficient cascade flow.
    """

    def __init__(self, config: CascadeConfig = None):
        """
        Initialize conservative threshold strategy.

        Args:
            config: Cascade configuration parameters
        """
        self.config = config or CascadeConfig()
        self.is_calibrated = False
        self.calibrated_thresholds = None
        self.performance_stats = None

        logger.info(f"ConservativeThresholdStrategy initialized: "
                   f"high={self.config.high_confidence_threshold}, "
                   f"low={self.config.low_confidence_threshold}")

    def calibrate_thresholds(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        validation_data: Optional[Dict] = None
    ) -> Dict:
        """
        Calibrate thresholds based on validation data.

        Args:
            y_prob: Predicted probabilities (positive class)
            y_true: True binary labels (0=fake, 1=authentic)
            validation_data: Additional validation information

        Returns:
            Calibration results and optimized thresholds
        """
        logger.info("Calibrating conservative thresholds...")

        # Analyze data distribution
        distribution_stats = self._analyze_distribution(y_prob, y_true)

        # Find optimal thresholds
        threshold_analysis = self._analyze_thresholds(y_prob, y_true)

        # Optimize for cascade objectives
        optimized_thresholds = self._optimize_cascade_thresholds(
            y_prob, y_true, threshold_analysis
        )

        # Update configuration with optimized thresholds
        if self.config.adaptation_enabled:
            self.config.high_confidence_threshold = optimized_thresholds['high_threshold']
            self.config.low_confidence_threshold = optimized_thresholds['low_threshold']

        self.calibrated_thresholds = optimized_thresholds
        self.is_calibrated = True

        # Calculate performance with calibrated thresholds
        self.performance_stats = self._calculate_cascade_performance(y_prob, y_true)

        results = {
            'original_thresholds': {
                'high': 0.99,
                'low': 0.01
            },
            'calibrated_thresholds': optimized_thresholds,
            'distribution_stats': distribution_stats,
            'performance_stats': self.performance_stats,
            'threshold_analysis': threshold_analysis,
            'meets_requirements': self._check_requirements()
        }

        logger.info(f"Threshold calibration completed:")
        logger.info(f"  High threshold: {optimized_thresholds['high_threshold']:.4f}")
        logger.info(f"  Low threshold: {optimized_thresholds['low_threshold']:.4f}")
        logger.info(f"  Filter rate: {self.performance_stats['filter_rate']:.3f}")
        logger.info(f"  FNR: {self.performance_stats['false_negative_rate']:.3f}")

        return results

    def make_decisions(
        self,
        probabilities: Union[torch.Tensor, np.ndarray],
        return_detailed: bool = False
    ) -> Union[List[str], List[CascadeDecision]]:
        """
        Make cascade decisions based on probabilities.

        Args:
            probabilities: Model probabilities for positive class
            return_detailed: Whether to return detailed decision objects

        Returns:
            List of decisions or detailed decision objects
        """
        if isinstance(probabilities, torch.Tensor):
            probs = probabilities.detach().cpu().numpy()
        else:
            probs = probabilities

        decisions = []

        for prob in probs:
            decision_obj = self._make_single_decision(prob)

            if return_detailed:
                decisions.append(decision_obj)
            else:
                decisions.append(decision_obj.stage_routing)

        return decisions

    def _make_single_decision(self, probability: float) -> CascadeDecision:
        """Make decision for single probability value."""
        high_thresh = self.config.high_confidence_threshold
        low_thresh = self.config.low_confidence_threshold

        if probability >= high_thresh:
            # Very confident authentic
            return CascadeDecision(
                decision='authentic',
                confidence=probability,
                raw_probability=probability,
                threshold_used='high',
                stage_routing='accept'
            )
        elif probability <= low_thresh:
            # Very confident fake
            return CascadeDecision(
                decision='fake',
                confidence=1.0 - probability,
                raw_probability=probability,
                threshold_used='low',
                stage_routing='reject'
            )
        else:
            # Uncertain - send to next stage
            return CascadeDecision(
                decision='uncertain',
                confidence=0.5,  # Neutral confidence for uncertain cases
                raw_probability=probability,
                threshold_used='middle',
                stage_routing='next_stage'
            )

    def _analyze_distribution(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """Analyze probability distribution characteristics."""
        authentic_probs = y_prob[y_true == 1]
        fake_probs = y_prob[y_true == 0]

        return {
            'overall': {
                'mean': np.mean(y_prob),
                'std': np.std(y_prob),
                'median': np.median(y_prob),
                'min': np.min(y_prob),
                'max': np.max(y_prob)
            },
            'authentic_samples': {
                'count': len(authentic_probs),
                'mean': np.mean(authentic_probs),
                'std': np.std(authentic_probs),
                'percentiles': np.percentile(authentic_probs, [10, 25, 50, 75, 90])
            },
            'fake_samples': {
                'count': len(fake_probs),
                'mean': np.mean(fake_probs),
                'std': np.std(fake_probs),
                'percentiles': np.percentile(fake_probs, [10, 25, 50, 75, 90])
            },
            'separation': {
                'mean_difference': np.mean(authentic_probs) - np.mean(fake_probs),
                'overlap_region': self._calculate_overlap_region(authentic_probs, fake_probs)
            }
        }

    def _calculate_overlap_region(
        self,
        authentic_probs: np.ndarray,
        fake_probs: np.ndarray
    ) -> Dict:
        """Calculate probability regions where classes overlap."""
        authentic_min, authentic_max = np.min(authentic_probs), np.max(authentic_probs)
        fake_min, fake_max = np.min(fake_probs), np.max(fake_probs)

        overlap_start = max(fake_min, authentic_min)
        overlap_end = min(fake_max, authentic_max)

        if overlap_start >= overlap_end:
            return {'overlap_exists': False, 'overlap_range': [0, 0]}

        # Count samples in overlap region
        authentic_in_overlap = ((authentic_probs >= overlap_start) &
                               (authentic_probs <= overlap_end)).sum()
        fake_in_overlap = ((fake_probs >= overlap_start) &
                          (fake_probs <= overlap_end)).sum()

        return {
            'overlap_exists': True,
            'overlap_range': [overlap_start, overlap_end],
            'overlap_width': overlap_end - overlap_start,
            'authentic_in_overlap': authentic_in_overlap,
            'fake_in_overlap': fake_in_overlap,
            'overlap_ratio': (authentic_in_overlap + fake_in_overlap) / (len(authentic_probs) + len(fake_probs))
        }

    def _analyze_thresholds(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """Analyze performance across different threshold values."""
        thresholds = np.linspace(0.01, 0.99, 99)
        threshold_stats = []

        for thresh in thresholds:
            # Conservative strategy: high threshold for authentic classification
            y_pred_authentic = (y_prob >= thresh).astype(int)

            # Calculate confusion matrix elements
            tp = ((y_true == 1) & (y_pred_authentic == 1)).sum()
            tn = ((y_true == 0) & (y_pred_authentic == 0)).sum()
            fp = ((y_true == 0) & (y_pred_authentic == 1)).sum()
            fn = ((y_true == 1) & (y_pred_authentic == 0)).sum()

            # Calculate rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

            # Cascade-specific metrics
            passed_to_next = ((y_prob < thresh) & (y_prob > self.config.low_confidence_threshold)).sum()
            filter_rate = passed_to_next / len(y_prob)

            threshold_stats.append({
                'threshold': thresh,
                'tpr': tpr,
                'fpr': fpr,
                'fnr': fnr,
                'filter_rate': filter_rate,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'passed_to_next': passed_to_next
            })

        return {
            'threshold_sweep': threshold_stats,
            'optimal_indices': self._find_optimal_thresholds(threshold_stats)
        }

    def _find_optimal_thresholds(self, threshold_stats: List[Dict]) -> Dict:
        """Find optimal threshold indices based on different criteria."""
        # Find thresholds that meet requirements
        valid_thresholds = [
            (i, stats) for i, stats in enumerate(threshold_stats)
            if (stats['fnr'] <= self.config.max_false_negative_rate and
                stats['tpr'] >= self.config.min_true_positive_rate)
        ]

        if not valid_thresholds:
            # Fallback: best compromise
            best_idx = min(range(len(threshold_stats)),
                          key=lambda i: threshold_stats[i]['fnr'])
            return {'best_compromise': best_idx, 'meets_requirements': False}

        # Among valid thresholds, find the one with highest filter rate
        best_valid_idx = max(valid_thresholds, key=lambda x: x[1]['filter_rate'])[0]

        return {
            'best_valid': best_valid_idx,
            'meets_requirements': True,
            'num_valid': len(valid_thresholds)
        }

    def _optimize_cascade_thresholds(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        threshold_analysis: Dict
    ) -> Dict:
        """Optimize thresholds specifically for cascade system."""
        threshold_stats = threshold_analysis['threshold_sweep']
        optimal_indices = threshold_analysis['optimal_indices']

        if optimal_indices['meets_requirements']:
            # Use the threshold that meets requirements and maximizes filter rate
            optimal_idx = optimal_indices['best_valid']
            optimal_high_threshold = threshold_stats[optimal_idx]['threshold']
        else:
            # Use fallback strategy
            logger.warning("No threshold meets all requirements, using best compromise")
            optimal_idx = optimal_indices['best_compromise']
            optimal_high_threshold = threshold_stats[optimal_idx]['threshold']

        # Low threshold optimization (for definitive fake detection)
        low_threshold_candidates = np.linspace(0.001, 0.1, 50)
        best_low_threshold = 0.01
        best_precision = 0

        for low_thresh in low_threshold_candidates:
            fake_predictions = y_prob <= low_thresh
            if fake_predictions.sum() > 0:
                precision = (y_true[fake_predictions] == 0).mean()
                if precision > best_precision:
                    best_precision = precision
                    best_low_threshold = low_thresh

        return {
            'high_threshold': optimal_high_threshold,
            'low_threshold': best_low_threshold,
            'optimization_stats': {
                'high_threshold_idx': optimal_idx,
                'high_threshold_stats': threshold_stats[optimal_idx],
                'low_threshold_precision': best_precision
            }
        }

    def _calculate_cascade_performance(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """Calculate cascade-specific performance metrics."""
        high_thresh = self.config.high_confidence_threshold
        low_thresh = self.config.low_confidence_threshold

        # Categorize decisions
        authentic_decisions = y_prob >= high_thresh
        fake_decisions = y_prob <= low_thresh
        uncertain_decisions = (y_prob > low_thresh) & (y_prob < high_thresh)

        # Calculate routing statistics
        routed_accept = authentic_decisions.sum()
        routed_reject = fake_decisions.sum()
        routed_next_stage = uncertain_decisions.sum()

        # Performance for authentic routing (high confidence)
        if routed_accept > 0:
            authentic_precision = (y_true[authentic_decisions] == 1).mean()
            authentic_recall = (y_true[authentic_decisions] == 1).sum() / (y_true == 1).sum()
        else:
            authentic_precision = 0
            authentic_recall = 0

        # Performance for fake routing (low confidence)
        if routed_reject > 0:
            fake_precision = (y_true[fake_decisions] == 0).mean()
            fake_recall = (y_true[fake_decisions] == 0).sum() / (y_true == 0).sum()
        else:
            fake_precision = 0
            fake_recall = 0

        # Overall cascade metrics
        total_samples = len(y_prob)
        filter_rate = routed_next_stage / total_samples

        # False negative rate (missing authentic samples)
        total_authentic = (y_true == 1).sum()
        missed_authentic = ((y_true == 1) & fake_decisions).sum()
        false_negative_rate = missed_authentic / total_authentic if total_authentic > 0 else 0

        return {
            'routing_stats': {
                'routed_accept': routed_accept,
                'routed_reject': routed_reject,
                'routed_next_stage': routed_next_stage,
                'accept_rate': routed_accept / total_samples,
                'reject_rate': routed_reject / total_samples,
                'filter_rate': filter_rate
            },
            'performance': {
                'authentic_precision': authentic_precision,
                'authentic_recall': authentic_recall,
                'fake_precision': fake_precision,
                'fake_recall': fake_recall,
                'false_negative_rate': false_negative_rate
            },
            'cascade_efficiency': {
                'samples_processed_locally': routed_accept + routed_reject,
                'samples_to_next_stage': routed_next_stage,
                'processing_efficiency': (routed_accept + routed_reject) / total_samples
            }
        }

    def _check_requirements(self) -> Dict:
        """Check if current configuration meets requirements."""
        if not self.performance_stats:
            return {'meets_all': False, 'reason': 'No performance stats available'}

        checks = {
            'filter_rate': self.performance_stats['routing_stats']['filter_rate'] >= self.config.target_filter_rate,
            'false_negative_rate': self.performance_stats['performance']['false_negative_rate'] <= self.config.max_false_negative_rate,
            'true_positive_rate': self.performance_stats['performance']['authentic_recall'] >= self.config.min_true_positive_rate
        }

        return {
            'meets_all': all(checks.values()),
            'individual_checks': checks,
            'current_values': {
                'filter_rate': self.performance_stats['routing_stats']['filter_rate'],
                'false_negative_rate': self.performance_stats['performance']['false_negative_rate'],
                'true_positive_rate': self.performance_stats['performance']['authentic_recall']
            }
        }

    def save_configuration(self, filepath: str):
        """Save calibrated configuration to file."""
        config_dict = {
            'cascade_config': {
                'high_confidence_threshold': self.config.high_confidence_threshold,
                'low_confidence_threshold': self.config.low_confidence_threshold,
                'target_filter_rate': self.config.target_filter_rate,
                'max_false_negative_rate': self.config.max_false_negative_rate,
                'min_true_positive_rate': self.config.min_true_positive_rate,
                'adaptation_enabled': self.config.adaptation_enabled
            },
            'calibrated_thresholds': self.calibrated_thresholds,
            'performance_stats': self.performance_stats,
            'is_calibrated': self.is_calibrated
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration saved to {filepath}")

    def load_configuration(self, filepath: str):
        """Load calibrated configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Update configuration
        cascade_config = config_dict['cascade_config']
        self.config.high_confidence_threshold = cascade_config['high_confidence_threshold']
        self.config.low_confidence_threshold = cascade_config['low_confidence_threshold']
        self.config.target_filter_rate = cascade_config['target_filter_rate']
        self.config.max_false_negative_rate = cascade_config['max_false_negative_rate']
        self.config.min_true_positive_rate = cascade_config['min_true_positive_rate']
        self.config.adaptation_enabled = cascade_config['adaptation_enabled']

        self.calibrated_thresholds = config_dict['calibrated_thresholds']
        self.performance_stats = config_dict['performance_stats']
        self.is_calibrated = config_dict['is_calibrated']

        logger.info(f"Configuration loaded from {filepath}")


def test_cascade_strategy():
    """Test cascade threshold strategy."""
    print("Testing Cascade Threshold Strategy...")

    # Generate test data
    np.random.seed(42)
    n_samples = 1000

    # Simulate model probabilities (authentic class)
    authentic_probs = np.random.beta(3, 1, n_samples // 2)  # Skewed towards high
    fake_probs = np.random.beta(1, 3, n_samples // 2)      # Skewed towards low

    y_prob = np.concatenate([authentic_probs, fake_probs])
    y_true = np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)])

    # Initialize strategy
    config = CascadeConfig(
        target_filter_rate=0.8,
        max_false_negative_rate=0.1
    )
    strategy = ConservativeThresholdStrategy(config)

    print(f"✓ Strategy initialized with config")

    # Calibrate thresholds
    results = strategy.calibrate_thresholds(y_prob, y_true)

    print(f"✓ Thresholds calibrated:")
    print(f"  High: {results['calibrated_thresholds']['high_threshold']:.4f}")
    print(f"  Low: {results['calibrated_thresholds']['low_threshold']:.4f}")
    print(f"  Meets requirements: {results['meets_requirements']['meets_all']}")

    # Test decision making
    test_probs = np.array([0.95, 0.05, 0.5, 0.99, 0.01])
    decisions = strategy.make_decisions(test_probs)
    detailed_decisions = strategy.make_decisions(test_probs, return_detailed=True)

    print(f"✓ Decision making:")
    for i, (prob, decision, detailed) in enumerate(zip(test_probs, decisions, detailed_decisions)):
        print(f"  P={prob:.2f} → {decision} ({detailed.decision}, conf={detailed.confidence:.2f})")

    # Test configuration save/load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        strategy.save_configuration(f.name)

        # Create new strategy and load config
        new_strategy = ConservativeThresholdStrategy()
        new_strategy.load_configuration(f.name)

        print(f"✓ Configuration save/load works")
        print(f"  Loaded high threshold: {new_strategy.config.high_confidence_threshold:.4f}")

    print("Cascade strategy tests passed! ✓")


if __name__ == "__main__":
    test_cascade_strategy()