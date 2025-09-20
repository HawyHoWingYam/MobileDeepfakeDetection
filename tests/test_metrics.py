"""
Test suite for academic evaluation metrics
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.metrics import AcademicMetrics, MetricResult, ComparisonResult


class TestMetricResult:
    """Test MetricResult dataclass"""
    
    def test_metric_result_creation(self):
        """Test MetricResult creation and attributes"""
        result = MetricResult(
            value=0.85,
            confidence_interval=(0.80, 0.90),
            p_value=0.001,
            std_error=0.025,
            n_samples=1000
        )
        
        assert result.value == 0.85
        assert result.confidence_interval == (0.80, 0.90)
        assert result.p_value == 0.001
        assert result.std_error == 0.025
        assert result.n_samples == 1000
    
    def test_metric_result_defaults(self):
        """Test MetricResult with default values"""
        result = MetricResult(value=0.75)
        
        assert result.value == 0.75
        assert result.confidence_interval is None
        assert result.p_value is None
        assert result.std_error is None
        assert result.n_samples == 0


class TestComparisonResult:
    """Test ComparisonResult dataclass"""
    
    def test_comparison_result_creation(self):
        """Test ComparisonResult creation"""
        result = ComparisonResult(
            metric_name="AUC",
            model1_value=0.85,
            model2_value=0.90,
            difference=0.05,
            p_value=0.01,
            confidence_interval=(0.01, 0.09),
            is_significant=True,
            effect_size=0.2
        )
        
        assert result.metric_name == "AUC"
        assert result.model1_value == 0.85
        assert result.model2_value == 0.90
        assert result.difference == 0.05
        assert result.p_value == 0.01
        assert result.is_significant is True
        assert result.effect_size == 0.2


class TestAcademicMetrics:
    """Test AcademicMetrics class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.metrics = AcademicMetrics(confidence_level=0.95, n_bootstrap=100, random_state=42)
        
        # Create sample data
        np.random.seed(42)
        self.n_samples = 500
        self.y_true = np.random.binomial(1, 0.6, self.n_samples)
        self.y_scores = np.random.rand(self.n_samples)
        # Make scores somewhat correlated with true labels for realistic AUC
        self.y_scores = 0.3 * self.y_scores + 0.7 * self.y_true + np.random.normal(0, 0.1, self.n_samples)
        self.y_scores = np.clip(self.y_scores, 0, 1)
        self.y_pred = (self.y_scores > 0.5).astype(int)
    
    def test_initialization(self):
        """Test AcademicMetrics initialization"""
        metrics = AcademicMetrics(confidence_level=0.99, n_bootstrap=1000, random_state=123)
        assert metrics.confidence_level == 0.99
        assert metrics.n_bootstrap == 1000
        assert metrics.random_state == 123
        assert metrics.alpha == 0.01
    
    def test_auc_calculation_with_ci(self):
        """Test AUC calculation with confidence interval"""
        result = self.metrics.calculate_auc_with_ci(self.y_true, self.y_scores)
        
        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.value <= result.confidence_interval[1]
        assert result.n_samples == self.n_samples
        
        # Compare with sklearn
        sklearn_auc = roc_auc_score(self.y_true, self.y_scores)
        assert abs(result.value - sklearn_auc) < 1e-10  # Should be identical
    
    def test_accuracy_calculation_with_ci(self):
        """Test accuracy calculation with confidence interval"""
        result = self.metrics.calculate_accuracy_with_ci(self.y_true, self.y_pred)
        
        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None
        assert result.confidence_interval[0] <= result.value <= result.confidence_interval[1]
        
        # Compare with sklearn
        sklearn_accuracy = accuracy_score(self.y_true, self.y_pred)
        assert abs(result.value - sklearn_accuracy) < 1e-10
    
    def test_f1_calculation_with_ci(self):
        """Test F1 score calculation with confidence interval"""
        result = self.metrics.calculate_f1_with_ci(self.y_true, self.y_pred)
        
        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None
        assert result.confidence_interval[0] <= result.value <= result.confidence_interval[1]
        
        # Compare with sklearn
        sklearn_f1 = f1_score(self.y_true, self.y_pred)
        assert abs(result.value - sklearn_f1) < 1e-10
    
    def test_precision_calculation_with_ci(self):
        """Test precision calculation with confidence interval"""
        result = self.metrics.calculate_precision_with_ci(self.y_true, self.y_pred)
        
        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None
        
        # Compare with sklearn
        sklearn_precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        assert abs(result.value - sklearn_precision) < 1e-10
    
    def test_recall_calculation_with_ci(self):
        """Test recall calculation with confidence interval"""
        result = self.metrics.calculate_recall_with_ci(self.y_true, self.y_pred)
        
        assert isinstance(result, MetricResult)
        assert 0 <= result.value <= 1
        assert result.confidence_interval is not None
        
        # Compare with sklearn
        sklearn_recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        assert abs(result.value - sklearn_recall) < 1e-10
    
    def test_bootstrap_metric_general(self):
        """Test general bootstrap metric calculation"""
        def dummy_metric(y_true, y_pred):
            return accuracy_score(y_true, y_pred)
        
        result = self.metrics._bootstrap_metric(
            self.y_true, self.y_pred, dummy_metric, "dummy"
        )
        
        assert isinstance(result, MetricResult)
        assert result.value >= 0
        assert result.confidence_interval is not None
        assert result.std_error is not None
        assert result.n_samples == self.n_samples
    
    def test_input_validation(self):
        """Test input validation for metrics"""
        # Test mismatched lengths
        with pytest.raises(ValueError, match="must have the same length"):
            self.metrics.calculate_auc_with_ci([0, 1], [0.5, 0.6, 0.7])
        
        # Test empty arrays
        with pytest.raises(ValueError, match="cannot be empty"):
            self.metrics.calculate_auc_with_ci([], [])
        
        # Test invalid labels for AUC
        with pytest.raises(ValueError, match="AUC requires both classes"):
            self.metrics.calculate_auc_with_ci([0, 0, 0], [0.1, 0.2, 0.3])
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        y_pred = (y_scores > 0.5).astype(int)
        
        auc_result = self.metrics.calculate_auc_with_ci(y_true, y_scores)
        acc_result = self.metrics.calculate_accuracy_with_ci(y_true, y_pred)
        
        assert auc_result.value == 1.0
        assert acc_result.value == 1.0
        
        # All same class predictions
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        
        # These should handle the edge case gracefully
        precision_result = self.metrics.calculate_precision_with_ci(y_true, y_pred)
        recall_result = self.metrics.calculate_recall_with_ci(y_true, y_pred)
        
        assert precision_result.value == 1.0
        assert recall_result.value == 1.0
    
    def test_statistical_significance_test(self):
        """Test statistical significance testing between models"""
        np.random.seed(42)
        n_samples = 300
        
        # Create two sets of predictions with known difference
        y_true = np.random.binomial(1, 0.5, n_samples)
        
        # Model 1: Better performance
        y_scores1 = np.where(y_true == 1, 
                           np.random.uniform(0.6, 1.0, n_samples),
                           np.random.uniform(0.0, 0.4, n_samples))
        
        # Model 2: Worse performance  
        y_scores2 = np.where(y_true == 1,
                           np.random.uniform(0.5, 0.8, n_samples), 
                           np.random.uniform(0.2, 0.5, n_samples))
        
        # This is a simplified test - would need actual implementation
        auc1 = self.metrics.calculate_auc_with_ci(y_true, y_scores1)
        auc2 = self.metrics.calculate_auc_with_ci(y_true, y_scores2)
        
        assert auc1.value > auc2.value  # Model 1 should be better
        
        # Test if confidence intervals overlap (simple significance test)
        ci1_lower, ci1_upper = auc1.confidence_interval
        ci2_lower, ci2_upper = auc2.confidence_interval
        
        # Non-overlapping CIs suggest significant difference
        significant = ci1_lower > ci2_upper or ci2_lower > ci1_upper
        print(f"Model 1 AUC: {auc1.value:.4f} [{ci1_lower:.4f}, {ci1_upper:.4f}]")
        print(f"Model 2 AUC: {auc2.value:.4f} [{ci2_lower:.4f}, {ci2_upper:.4f}]")
        print(f"Significant difference: {significant}")
    
    def test_multiple_confidence_levels(self):
        """Test metrics with different confidence levels"""
        confidence_levels = [0.90, 0.95, 0.99]
        
        for conf_level in confidence_levels:
            metrics = AcademicMetrics(confidence_level=conf_level, random_state=42)
            result = metrics.calculate_auc_with_ci(self.y_true, self.y_scores)
            
            # Higher confidence should give wider intervals
            ci_width = result.confidence_interval[1] - result.confidence_interval[0]
            assert ci_width > 0
            
            # Store results to compare widths
            if conf_level == 0.90:
                ci_width_90 = ci_width
            elif conf_level == 0.95:
                ci_width_95 = ci_width
            elif conf_level == 0.99:
                ci_width_99 = ci_width
        
        # Higher confidence should give wider intervals
        assert ci_width_90 < ci_width_95 < ci_width_99
    
    def test_bootstrap_reproducibility(self):
        """Test that bootstrap results are reproducible"""
        metrics1 = AcademicMetrics(random_state=42, n_bootstrap=50)
        metrics2 = AcademicMetrics(random_state=42, n_bootstrap=50)
        
        result1 = metrics1.calculate_auc_with_ci(self.y_true, self.y_scores)
        result2 = metrics2.calculate_auc_with_ci(self.y_true, self.y_scores)
        
        # Results should be identical with same random seed
        assert result1.value == result2.value
        assert result1.confidence_interval == result2.confidence_interval
        assert abs(result1.std_error - result2.std_error) < 1e-10
    
    def test_sample_size_effects(self):
        """Test how sample size affects confidence intervals"""
        np.random.seed(42)
        
        # Small sample
        n_small = 50
        y_true_small = np.random.binomial(1, 0.5, n_small)
        y_scores_small = np.random.rand(n_small)
        
        # Large sample
        n_large = 1000
        y_true_large = np.random.binomial(1, 0.5, n_large)
        y_scores_large = np.random.rand(n_large)
        
        metrics = AcademicMetrics(n_bootstrap=100, random_state=42)
        
        result_small = metrics.calculate_auc_with_ci(y_true_small, y_scores_small)
        result_large = metrics.calculate_auc_with_ci(y_true_large, y_scores_large)
        
        # Larger sample should have narrower confidence interval
        ci_width_small = result_small.confidence_interval[1] - result_small.confidence_interval[0]
        ci_width_large = result_large.confidence_interval[1] - result_large.confidence_interval[0]
        
        assert ci_width_large < ci_width_small
        assert result_small.std_error > result_large.std_error
    
    def test_imbalanced_data(self):
        """Test metrics with imbalanced data"""
        np.random.seed(42)
        n_samples = 1000
        
        # Highly imbalanced (10% positive)
        y_true_imbalanced = np.random.binomial(1, 0.1, n_samples)
        y_scores_imbalanced = np.random.rand(n_samples)
        y_pred_imbalanced = (y_scores_imbalanced > 0.9).astype(int)  # Conservative threshold
        
        # All metrics should handle imbalanced data
        auc_result = self.metrics.calculate_auc_with_ci(y_true_imbalanced, y_scores_imbalanced)
        precision_result = self.metrics.calculate_precision_with_ci(y_true_imbalanced, y_pred_imbalanced)
        recall_result = self.metrics.calculate_recall_with_ci(y_true_imbalanced, y_pred_imbalanced)
        f1_result = self.metrics.calculate_f1_with_ci(y_true_imbalanced, y_pred_imbalanced)
        
        assert 0 <= auc_result.value <= 1
        assert 0 <= precision_result.value <= 1
        assert 0 <= recall_result.value <= 1
        assert 0 <= f1_result.value <= 1
        
        # Confidence intervals should be valid
        for result in [auc_result, precision_result, recall_result, f1_result]:
            if result.confidence_interval is not None:
                assert result.confidence_interval[0] <= result.value <= result.confidence_interval[1]


class TestMetricsIntegration:
    """Integration tests for academic metrics"""
    
    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation pipeline with all metrics"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic classification scenario
        y_true = np.random.binomial(1, 0.4, n_samples)
        
        # Create predictions with realistic correlation
        base_scores = np.random.rand(n_samples)
        # Make scores correlated with true labels
        correlation_strength = 0.6
        y_scores = (1 - correlation_strength) * base_scores + correlation_strength * y_true
        y_scores += np.random.normal(0, 0.1, n_samples)  # Add noise
        y_scores = np.clip(y_scores, 0, 1)
        
        y_pred = (y_scores > 0.5).astype(int)
        
        metrics = AcademicMetrics(confidence_level=0.95, n_bootstrap=200, random_state=42)
        
        # Calculate all metrics
        results = {
            'auc': metrics.calculate_auc_with_ci(y_true, y_scores),
            'accuracy': metrics.calculate_accuracy_with_ci(y_true, y_pred),
            'precision': metrics.calculate_precision_with_ci(y_true, y_pred),
            'recall': metrics.calculate_recall_with_ci(y_true, y_pred),
            'f1': metrics.calculate_f1_with_ci(y_true, y_pred)
        }
        
        # Verify all results are valid
        for metric_name, result in results.items():
            assert isinstance(result, MetricResult)
            assert 0 <= result.value <= 1
            assert result.confidence_interval is not None
            assert result.confidence_interval[0] <= result.value <= result.confidence_interval[1]
            assert result.n_samples == n_samples
            print(f"{metric_name.upper()}: {result.value:.4f} "
                  f"[{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        
        # Verify relationships between metrics
        # AUC should be reasonable for correlated data
        assert results['auc'].value > 0.6
        
        return results
    
    def test_cross_validation_metrics(self):
        """Test metrics across multiple CV folds (simulated)"""
        np.random.seed(42)
        n_folds = 5
        fold_size = 200
        
        metrics = AcademicMetrics(random_state=42)
        fold_results = []
        
        for fold in range(n_folds):
            # Generate data for this fold
            y_true = np.random.binomial(1, 0.5, fold_size)
            y_scores = np.random.rand(fold_size)
            
            # Make it somewhat realistic
            y_scores = 0.3 * y_scores + 0.7 * y_true + np.random.normal(0, 0.2, fold_size)
            y_scores = np.clip(y_scores, 0, 1)
            
            auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
            fold_results.append(auc_result.value)
        
        # Calculate cross-fold statistics
        mean_auc = np.mean(fold_results)
        std_auc = np.std(fold_results)
        
        assert len(fold_results) == n_folds
        assert 0 <= mean_auc <= 1
        assert std_auc >= 0
        
        print(f"Cross-validation AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Individual fold AUCs: {[f'{auc:.4f}' for auc in fold_results]}")
        
        return fold_results


# Performance and stress tests
class TestMetricsPerformance:
    """Performance tests for metrics calculation"""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test metrics calculation on large datasets"""
        import time
        
        np.random.seed(42)
        n_samples = 10000  # Large dataset
        
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_scores = np.random.rand(n_samples)
        
        metrics = AcademicMetrics(n_bootstrap=100)
        
        start_time = time.time()
        result = metrics.calculate_auc_with_ci(y_true, y_scores)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        print(f"AUC calculation time for {n_samples} samples: {calculation_time:.4f} seconds")
        
        # Should complete in reasonable time (less than 10 seconds)
        assert calculation_time < 10.0
        assert isinstance(result, MetricResult)
        assert result.n_samples == n_samples
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        metrics = AcademicMetrics(n_bootstrap=50)
        
        # Run multiple calculations
        for i in range(10):
            np.random.seed(i)
            y_true = np.random.binomial(1, 0.5, 1000)
            y_scores = np.random.rand(1000)
            
            metrics.calculate_auc_with_ci(y_true, y_scores)
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage increase: {memory_increase:.2f} MB")
        
        # Memory usage shouldn't increase dramatically (less than 100MB)
        assert memory_increase < 100


@pytest.fixture
def sample_binary_classification_data():
    """Fixture providing sample binary classification data"""
    np.random.seed(42)
    n_samples = 300
    
    # Create somewhat realistic data
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    
    # Create correlated predictions
    base_scores = np.random.beta(2, 5, n_samples)  # Realistic score distribution
    
    # Make scores correlated with labels
    correlation = 0.7
    y_scores = (1 - correlation) * base_scores + correlation * y_true
    y_scores += np.random.normal(0, 0.1, n_samples)
    y_scores = np.clip(y_scores, 0, 1)
    
    y_pred = (y_scores > 0.5).astype(int)
    
    return y_true, y_scores, y_pred


def test_academic_metrics_with_fixture(sample_binary_classification_data):
    """Test using pytest fixture"""
    y_true, y_scores, y_pred = sample_binary_classification_data
    metrics = AcademicMetrics()
    
    auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
    assert isinstance(auc_result, MetricResult)
    assert 0 <= auc_result.value <= 1
    assert auc_result.n_samples == 300


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])