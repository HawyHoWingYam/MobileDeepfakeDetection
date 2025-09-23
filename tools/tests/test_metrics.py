#!/usr/bin/env python3
"""
Test academic metrics functionality
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.metrics import AcademicMetrics, MetricResult

class TestAcademicMetrics:
    """Test cases for academic metrics computation"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 100

        # Create realistic test data
        self.y_true = np.random.binomial(1, 0.6, self.n_samples)

        # Create correlated predictions
        base_scores = np.random.rand(self.n_samples)
        correlation = 0.7
        self.y_scores = (1 - correlation) * base_scores + correlation * self.y_true
        self.y_scores += np.random.normal(0, 0.1, self.n_samples)
        self.y_scores = np.clip(self.y_scores, 0, 1)

        self.y_pred = (self.y_scores > 0.5).astype(int)

        # Initialize metrics with reduced bootstrap for speed
        self.metrics = AcademicMetrics(n_bootstrap=50, random_state=42)

    def test_academic_metrics_instantiation(self):
        """Test AcademicMetrics can be instantiated"""
        metrics = AcademicMetrics()
        assert metrics is not None

    def test_auc_calculation(self):
        """Test AUC calculation with confidence intervals"""
        result = self.metrics.calculate_auc_with_ci(self.y_true, self.y_scores)

        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.value <= result.confidence_interval[1]
        assert result.n_samples == self.n_samples

    def test_accuracy_calculation(self):
        """Test accuracy calculation with confidence intervals"""
        result = self.metrics.calculate_accuracy_with_ci(self.y_true, self.y_pred)

        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None
        assert result.n_samples == self.n_samples

    def test_f1_calculation(self):
        """Test F1 score calculation"""
        result = self.metrics.calculate_f1_with_ci(self.y_true, self.y_pred)

        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None

    def test_precision_calculation(self):
        """Test precision calculation"""
        result = self.metrics.calculate_precision_with_ci(self.y_true, self.y_pred)

        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None

    def test_recall_calculation(self):
        """Test recall calculation"""
        result = self.metrics.calculate_recall_with_ci(self.y_true, self.y_pred)

        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = AcademicMetrics(n_bootstrap=10)

        auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
        acc_result = metrics.calculate_accuracy_with_ci(y_true, y_pred)

        assert auc_result.value == 1.0
        assert acc_result.value == 1.0

    def test_random_predictions(self):
        """Test metrics with random predictions"""
        np.random.seed(123)
        y_true = np.random.binomial(1, 0.5, 200)
        y_scores = np.random.rand(200)
        y_pred = (y_scores > 0.5).astype(int)

        metrics = AcademicMetrics(n_bootstrap=20)

        auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)

        # Random predictions should have AUC around 0.5
        assert 0.3 <= auc_result.value <= 0.7

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with all same class
        y_true_same = np.ones(10)
        y_scores_same = np.random.rand(10)

        metrics = AcademicMetrics(n_bootstrap=10)

        # AUC should handle this gracefully (may return NaN)
        try:
            result = metrics.calculate_auc_with_ci(y_true_same, y_scores_same)
            # If it succeeds, result should be reasonable or NaN
            assert np.isnan(result.value) or 0 <= result.value <= 1
        except Exception:
            # Expected to fail gracefully with single class
            pass

    def test_confidence_level_parameter(self):
        """Test different confidence levels"""
        metrics_95 = AcademicMetrics(confidence_level=0.95, n_bootstrap=30)
        metrics_99 = AcademicMetrics(confidence_level=0.99, n_bootstrap=30)

        result_95 = metrics_95.calculate_auc_with_ci(self.y_true, self.y_scores)
        result_99 = metrics_99.calculate_auc_with_ci(self.y_true, self.y_scores)

        # 99% CI should be wider than 95% CI
        width_95 = result_95.confidence_interval[1] - result_95.confidence_interval[0]
        width_99 = result_99.confidence_interval[1] - result_99.confidence_interval[0]

        assert width_99 >= width_95

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap results are reproducible with same seed"""
        metrics1 = AcademicMetrics(n_bootstrap=20, random_state=42)
        metrics2 = AcademicMetrics(n_bootstrap=20, random_state=42)

        result1 = metrics1.calculate_auc_with_ci(self.y_true, self.y_scores)
        result2 = metrics2.calculate_auc_with_ci(self.y_true, self.y_scores)

        assert abs(result1.value - result2.value) < 1e-10
        assert abs(result1.confidence_interval[0] - result2.confidence_interval[0]) < 1e-10

if __name__ == "__main__":
    pytest.main([__file__])