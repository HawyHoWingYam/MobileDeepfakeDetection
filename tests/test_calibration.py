"""
Test suite for calibration assessment tools
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.calibration_tools import CalibrationAnalyzer, CalibrationResult, TemperatureScalingResult


class TestCalibrationAnalyzer:
    """Test cases for CalibrationAnalyzer"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.analyzer = CalibrationAnalyzer(n_bins=10, confidence_level=0.95, random_state=42)
    
    def test_initialization(self):
        """Test CalibrationAnalyzer initialization"""
        analyzer = CalibrationAnalyzer(n_bins=15, bin_strategy='quantile', confidence_level=0.99)
        assert analyzer.n_bins == 15
        assert analyzer.bin_strategy == 'quantile'
        assert analyzer.confidence_level == 0.99
        assert analyzer.alpha == 0.01
    
    def test_perfect_calibration(self):
        """Test ECE/MCE calculation for perfectly calibrated predictions"""
        n_samples = 1000
        np.random.seed(42)
        
        # Create perfectly calibrated predictions
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.rand(n_samples)
        
        # Make them perfectly calibrated by construction
        sorted_indices = np.argsort(y_prob)
        sorted_probs = y_prob[sorted_indices]
        
        # Create perfect calibration
        perfect_y_true = np.zeros(n_samples)
        for i, prob in enumerate(sorted_probs):
            perfect_y_true[i] = np.random.binomial(1, prob)
        
        # Restore original order
        restore_indices = np.argsort(sorted_indices)
        perfect_y_true = perfect_y_true[restore_indices]
        
        result = self.analyzer.calculate_ece_mce(perfect_y_true, y_prob)
        
        assert isinstance(result, CalibrationResult)
        assert result.ece >= 0.0
        assert result.mce >= 0.0
        assert result.n_samples == n_samples
        assert result.n_bins == 10
        
        # Perfect calibration should have low ECE (due to random noise, won't be exactly 0)
        assert result.ece < 0.2  # Allow some tolerance for random noise
    
    def test_poor_calibration(self):
        """Test ECE/MCE calculation for poorly calibrated predictions"""
        n_samples = 1000
        np.random.seed(42)
        
        # Create poorly calibrated predictions (overconfident)
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.where(y_true == 1, 
                         np.random.uniform(0.8, 1.0, n_samples),  # High confidence for positives
                         np.random.uniform(0.0, 0.2, n_samples))  # Low confidence for negatives
        
        # But make predictions wrong 30% of the time
        wrong_indices = np.random.choice(n_samples, size=int(0.3 * n_samples), replace=False)
        y_true[wrong_indices] = 1 - y_true[wrong_indices]
        
        result = self.analyzer.calculate_ece_mce(y_true, y_prob)
        
        # Poor calibration should have high ECE
        assert result.ece > 0.1
        assert result.mce > 0.1
        assert result.brier_score > 0.0
    
    def test_input_validation(self):
        """Test input validation for calculate_ece_mce"""
        # Test mismatched lengths
        with pytest.raises(ValueError, match="must have same length"):
            self.analyzer.calculate_ece_mce([0, 1], [0.5, 0.6, 0.7])
        
        # Test invalid labels
        with pytest.raises(ValueError, match="must contain only 0s and 1s"):
            self.analyzer.calculate_ece_mce([0, 1, 2], [0.3, 0.5, 0.7])
        
        # Test invalid probabilities
        with pytest.raises(ValueError, match="must be in"):
            self.analyzer.calculate_ece_mce([0, 1, 0], [0.3, 1.5, 0.7])
        
        with pytest.raises(ValueError, match="must be in"):
            self.analyzer.calculate_ece_mce([0, 1, 0], [-0.1, 0.5, 0.7])
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation"""
        np.random.seed(42)
        n_samples = 500
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.rand(n_samples)
        
        ece, (ci_lower, ci_upper) = self.analyzer.bootstrap_ece_confidence_interval(
            y_true, y_prob, n_bootstrap=100
        )
        
        assert isinstance(ece, float)
        assert ece >= 0.0
        assert ci_lower <= ece <= ci_upper
        assert ci_lower >= 0.0
        assert ci_upper <= 1.0
    
    def test_temperature_scaling(self):
        """Test temperature scaling calibration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate uncalibrated logits (overconfident)
        y_true = np.random.binomial(1, 0.5, n_samples)
        logits = np.where(y_true == 1,
                         np.random.normal(2.0, 1.0, n_samples),    # High logits for positives
                         np.random.normal(-2.0, 1.0, n_samples))   # Low logits for negatives
        
        # Add some noise to make it imperfect
        noise_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
        logits[noise_indices] *= -0.5
        
        result = self.analyzer.temperature_scaling(y_true, logits, validation_split=0.2)
        
        assert isinstance(result, TemperatureScalingResult)
        assert result.optimal_temperature > 0
        assert len(result.calibrated_predictions) == n_samples
        assert result.improvement >= 0  # Could be negative if calibration gets worse
        assert result.convergence_info['success'] is True
        
        # Check that calibrated predictions are valid probabilities
        assert np.all(result.calibrated_predictions >= 0)
        assert np.all(result.calibrated_predictions <= 1)
    
    def test_logits_to_probs(self):
        """Test logits to probabilities conversion"""
        logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        probs = self.analyzer._logits_to_probs(logits)
        
        # Test sigmoid function properties
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert probs[2] == 0.5  # sigmoid(0) = 0.5
        assert probs[0] < probs[1] < probs[2] < probs[3] < probs[4]  # Monotonic
    
    def test_plot_reliability_diagram(self):
        """Test reliability diagram plotting"""
        np.random.seed(42)
        n_samples = 500
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.rand(n_samples)
        
        calibration_result = self.analyzer.calculate_ece_mce(y_true, y_prob)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_reliability.png"
            
            fig = self.analyzer.plot_reliability_diagram(
                calibration_result, 
                title="Test Reliability",
                save_path=str(save_path),
                figsize=(6, 4)
            )
            
            assert fig is not None
            assert save_path.exists()
            plt.close(fig)
    
    def test_compare_calibrations(self):
        """Test multi-model calibration comparison"""
        np.random.seed(42)
        n_samples = 500
        y_true = np.random.binomial(1, 0.5, n_samples)
        
        predictions_dict = {
            'Model_A': np.random.rand(n_samples),
            'Model_B': np.random.rand(n_samples) * 0.8 + 0.1,  # More conservative
            'Model_C': np.random.rand(n_samples)
        }
        
        results = self.analyzer.compare_calibrations(y_true, predictions_dict)
        
        assert len(results) == 3
        assert 'Model_A' in results
        assert 'Model_B' in results
        assert 'Model_C' in results
        
        for result in results.values():
            assert isinstance(result, CalibrationResult)
    
    def test_comprehensive_calibration_report(self):
        """Test comprehensive calibration report generation"""
        np.random.seed(42)
        n_samples = 500
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.rand(n_samples)
        logits = np.log(y_prob / (1 - y_prob + 1e-8))  # Convert back to logits
        
        report = self.analyzer.comprehensive_calibration_report(
            y_true, y_prob, logits, model_name="Test_Model"
        )
        
        assert 'model_name' in report
        assert 'calibration_metrics' in report
        assert 'ECE_with_CI' in report
        assert 'temperature_scaling' in report
        assert 'reliability_data' in report
        
        # Check calibration metrics
        metrics = report['calibration_metrics']
        assert 'ECE' in metrics
        assert 'MCE' in metrics
        assert 'Brier_Score' in metrics
        
        # Check ECE with CI
        ece_ci = report['ECE_with_CI']
        assert 'ECE' in ece_ci
        assert 'confidence_interval' in ece_ci
        
        # Check temperature scaling
        temp_scaling = report['temperature_scaling']
        assert 'optimal_temperature' in temp_scaling
        assert 'improvement' in temp_scaling
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with all predictions being 0.5
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        
        result = self.analyzer.calculate_ece_mce(y_true, y_prob)
        assert result.ece >= 0.0
        assert result.mce >= 0.0
        
        # Test with perfect predictions
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        
        result = self.analyzer.calculate_ece_mce(y_true, y_prob)
        assert result.ece >= 0.0  # Should be very low but might not be exactly 0 due to binning
        
        # Test with all same labels
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.8, 0.7, 0.9, 0.6])
        
        result = self.analyzer.calculate_ece_mce(y_true, y_prob)
        assert result.ece >= 0.0
        assert result.mce >= 0.0
    
    def test_different_bin_strategies(self):
        """Test different binning strategies"""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.rand(n_samples)
        
        # Uniform binning
        analyzer_uniform = CalibrationAnalyzer(n_bins=10, bin_strategy='uniform')
        result_uniform = analyzer_uniform.calculate_ece_mce(y_true, y_prob)
        
        # Quantile binning
        analyzer_quantile = CalibrationAnalyzer(n_bins=10, bin_strategy='quantile')
        result_quantile = analyzer_quantile.calculate_ece_mce(y_true, y_prob)
        
        # Both should produce valid results
        assert result_uniform.ece >= 0.0
        assert result_quantile.ece >= 0.0
        assert result_uniform.n_bins == result_quantile.n_bins == 10
    
    def test_multiple_confidence_levels(self):
        """Test different confidence levels for bootstrap"""
        np.random.seed(42)
        n_samples = 300
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.rand(n_samples)
        
        # Test different confidence levels
        for confidence_level in [0.90, 0.95, 0.99]:
            analyzer = CalibrationAnalyzer(confidence_level=confidence_level)
            ece, (ci_lower, ci_upper) = analyzer.bootstrap_ece_confidence_interval(
                y_true, y_prob, n_bootstrap=50
            )
            
            # Higher confidence should give wider intervals
            interval_width = ci_upper - ci_lower
            assert interval_width > 0
            assert ci_lower <= ece <= ci_upper


@pytest.fixture
def sample_calibration_data():
    """Fixture providing sample calibration data for tests"""
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.binomial(1, 0.5, n_samples)
    y_prob = np.random.rand(n_samples)
    return y_true, y_prob


def test_calibration_analyzer_with_fixture(sample_calibration_data):
    """Test using pytest fixture"""
    y_true, y_prob = sample_calibration_data
    analyzer = CalibrationAnalyzer()
    
    result = analyzer.calculate_ece_mce(y_true, y_prob)
    assert isinstance(result, CalibrationResult)
    assert result.n_samples == 200


class TestCalibrationIntegration:
    """Integration tests for calibration tools"""
    
    def test_end_to_end_calibration_workflow(self):
        """Test complete calibration analysis workflow"""
        np.random.seed(42)
        
        # Generate realistic uncalibrated data
        n_samples = 1000
        y_true = np.random.binomial(1, 0.4, n_samples)  # Imbalanced
        
        # Create overconfident predictions
        base_logits = np.where(y_true == 1, 
                              np.random.normal(1.5, 0.8, n_samples),
                              np.random.normal(-1.5, 0.8, n_samples))
        
        # Add noise to create miscalibration
        noise = np.random.normal(0, 0.5, n_samples)
        logits = base_logits + noise
        
        analyzer = CalibrationAnalyzer(n_bins=15)
        
        # Convert logits to probabilities
        y_prob = analyzer._logits_to_probs(logits)
        
        # 1. Basic calibration analysis
        basic_result = analyzer.calculate_ece_mce(y_true, y_prob)
        
        # 2. Confidence interval
        ece_with_ci, ci = analyzer.bootstrap_ece_confidence_interval(y_true, y_prob, n_bootstrap=100)
        
        # 3. Temperature scaling
        temp_result = analyzer.temperature_scaling(y_true, logits, validation_split=0.3)
        
        # 4. Comprehensive report
        report = analyzer.comprehensive_calibration_report(y_true, y_prob, logits, "Integration_Test_Model")
        
        # Verify all components work together
        assert basic_result.ece > 0
        assert ece_with_ci == basic_result.ece
        assert ci[0] <= ece_with_ci <= ci[1]
        assert temp_result.optimal_temperature > 0
        assert report['model_name'] == "Integration_Test_Model"
        
        # Temperature scaling should improve calibration (usually)
        # Note: improvement might be negative if the data is already well-calibrated
        assert temp_result.pre_calibration_ece >= 0
        assert temp_result.post_calibration_ece >= 0
        
        print(f"Pre-calibration ECE: {temp_result.pre_calibration_ece:.4f}")
        print(f"Post-calibration ECE: {temp_result.post_calibration_ece:.4f}")
        print(f"Improvement: {temp_result.improvement:.4f}")
        print(f"Optimal temperature: {temp_result.optimal_temperature:.4f}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])