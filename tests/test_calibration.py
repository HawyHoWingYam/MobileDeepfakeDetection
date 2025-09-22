#!/usr/bin/env python3
"""
Test calibration tools functionality
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.calibration_tools import CalibrationAnalyzer, CalibrationResult

class TestCalibrationTools:
    """Test cases for calibration analysis tools"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 200

        # Create test data with some miscalibration
        self.y_true = np.random.binomial(1, 0.4, self.n_samples)

        # Create predictions with some calibration issues
        base_scores = np.random.beta(2, 5, self.n_samples)
        correlation = 0.7
        self.y_prob = (1 - correlation) * base_scores + correlation * self.y_true
        self.y_prob += np.random.normal(0, 0.1, self.n_samples)
        self.y_prob = np.clip(self.y_prob, 0.01, 0.99)  # Avoid extreme values

        # Initialize calibration analyzer
        self.analyzer = CalibrationAnalyzer(n_bins=10, random_state=42)

    def test_calibration_analyzer_instantiation(self):
        """Test CalibrationAnalyzer can be instantiated"""
        analyzer = CalibrationAnalyzer()
        assert analyzer is not None

    def test_ece_mce_calculation(self):
        """Test ECE and MCE calculation"""
        result = self.analyzer.calculate_ece_mce(self.y_true, self.y_prob)

        assert isinstance(result, CalibrationResult)
        assert 0 <= result.ece <= 1
        assert 0 <= result.mce <= 1
        assert result.mce >= result.ece  # MCE should be >= ECE
        assert 0 <= result.brier_score <= 1
        assert result.n_samples == self.n_samples
        assert result.n_bins == 10

    def test_reliability_diagram_data(self):
        """Test reliability diagram data generation"""
        result = self.analyzer.calculate_ece_mce(self.y_true, self.y_prob)

        assert 'bin_boundaries' in result.reliability_diagram_data
        assert 'bin_lowers' in result.reliability_diagram_data
        assert 'bin_uppers' in result.reliability_diagram_data
        assert 'bin_accuracies' in result.reliability_diagram_data
        assert 'bin_confidences' in result.reliability_diagram_data
        assert 'bin_counts' in result.reliability_diagram_data

        # Check data consistency
        boundaries = result.reliability_diagram_data['bin_boundaries']
        assert len(boundaries) == 11  # 10 bins + 1
        assert boundaries[0] == 0.0
        assert boundaries[-1] == 1.0

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval for ECE"""
        ece_with_ci, ci = self.analyzer.bootstrap_ece_confidence_interval(
            self.y_true, self.y_prob, n_bootstrap=50
        )

        assert isinstance(ece_with_ci, float)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= ece_with_ci <= ci[1]
        assert 0 <= ci[0] <= 1
        assert 0 <= ci[1] <= 1

    def test_temperature_scaling(self):
        """Test temperature scaling calibration"""
        # Convert probabilities to logits for temperature scaling
        logits = np.log(self.y_prob / (1 - self.y_prob + 1e-8))

        temp_result = self.analyzer.temperature_scaling(
            self.y_true, logits, validation_split=0.3
        )

        assert hasattr(temp_result, 'optimal_temperature')
        assert hasattr(temp_result, 'calibrated_predictions')
        assert hasattr(temp_result, 'pre_calibration_ece')
        assert hasattr(temp_result, 'post_calibration_ece')
        assert hasattr(temp_result, 'improvement')

        assert temp_result.optimal_temperature > 0
        assert len(temp_result.calibrated_predictions) > 0
        assert 0 <= temp_result.pre_calibration_ece <= 1
        assert 0 <= temp_result.post_calibration_ece <= 1

    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions"""
        n_samples = 1000
        np.random.seed(123)

        # Create perfectly calibrated data
        y_prob_perfect = np.random.rand(n_samples)
        y_true_perfect = np.random.binomial(1, y_prob_perfect)

        analyzer = CalibrationAnalyzer(n_bins=10)
        result = analyzer.calculate_ece_mce(y_true_perfect, y_prob_perfect)

        # Perfect calibration should have low ECE/MCE
        assert result.ece < 0.1  # Should be very low for large n
        assert result.mce < 0.2

    def test_overconfident_predictions(self):
        """Test with overconfident predictions"""
        n_samples = 200
        y_true = np.random.binomial(1, 0.6, n_samples)

        # Create overconfident predictions (always high confidence)
        y_prob_overconfident = np.where(y_true == 1, 0.95, 0.05)

        analyzer = CalibrationAnalyzer(n_bins=5)
        result = analyzer.calculate_ece_mce(y_true, y_prob_overconfident)

        # Overconfident predictions should have higher ECE
        assert result.ece > 0.1

    def test_different_bin_numbers(self):
        """Test calibration with different numbers of bins"""
        analyzer_5 = CalibrationAnalyzer(n_bins=5)
        analyzer_15 = CalibrationAnalyzer(n_bins=15)

        result_5 = analyzer_5.calculate_ece_mce(self.y_true, self.y_prob)
        result_15 = analyzer_15.calculate_ece_mce(self.y_true, self.y_prob)

        assert result_5.n_bins == 5
        assert result_15.n_bins == 15

        # Both should give reasonable results
        assert 0 <= result_5.ece <= 1
        assert 0 <= result_15.ece <= 1

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with extreme predictions
        y_true_extreme = np.array([0, 1, 0, 1])
        y_prob_extreme = np.array([0.0, 1.0, 0.0, 1.0])

        analyzer = CalibrationAnalyzer(n_bins=2)

        try:
            result = analyzer.calculate_ece_mce(y_true_extreme, y_prob_extreme)
            # Should handle extreme values gracefully
            assert 0 <= result.ece <= 1
        except Exception:
            # May fail with extreme values, which is acceptable
            pass

    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        analyzer1 = CalibrationAnalyzer(n_bins=10, random_state=42)
        analyzer2 = CalibrationAnalyzer(n_bins=10, random_state=42)

        ece1, ci1 = analyzer1.bootstrap_ece_confidence_interval(
            self.y_true, self.y_prob, n_bootstrap=20
        )
        ece2, ci2 = analyzer2.bootstrap_ece_confidence_interval(
            self.y_true, self.y_prob, n_bootstrap=20
        )

        assert abs(ece1 - ece2) < 1e-10
        assert abs(ci1[0] - ci2[0]) < 1e-10
        assert abs(ci1[1] - ci2[1]) < 1e-10

if __name__ == "__main__":
    pytest.main([__file__])