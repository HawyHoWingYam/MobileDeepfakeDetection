"""
AWARE-NET Calibration Assessment Tools
Comprehensive model calibration analysis with academic rigor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class CalibrationResult:
    """Container for calibration assessment results"""
    ece: float
    mce: float
    brier_score: float
    reliability_diagram_data: Dict[str, np.ndarray]
    confidence_interval: Optional[Tuple[float, float]] = None
    n_bins: int = 10
    n_samples: int = 0

@dataclass
class TemperatureScalingResult:
    """Container for temperature scaling results"""
    optimal_temperature: float
    calibrated_predictions: np.ndarray
    pre_calibration_ece: float
    post_calibration_ece: float
    improvement: float
    convergence_info: Dict[str, Any]

class CalibrationAnalyzer:
    """
    Academic-grade model calibration analysis toolkit
    
    Features:
    - Expected Calibration Error (ECE) with confidence intervals
    - Maximum Calibration Error (MCE) assessment
    - Reliability diagram generation
    - Temperature scaling calibration
    - Before/after calibration comparison
    - Brier score decomposition
    """
    
    def __init__(self, 
                 n_bins: int = 15,
                 bin_strategy: str = 'uniform',
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        """
        Initialize calibration analyzer
        
        Args:
            n_bins: Number of bins for reliability analysis
            bin_strategy: Binning strategy ('uniform' or 'quantile')
            confidence_level: Confidence level for intervals
            random_state: Random seed for reproducibility
        """
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.alpha = 1 - confidence_level
        
        np.random.seed(random_state)
    
    def calculate_ece_mce(self, 
                         y_true: np.ndarray, 
                         y_prob: np.ndarray,
                         return_details: bool = True) -> CalibrationResult:
        """
        Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
        
        ECE measures the weighted average of absolute differences between accuracy and confidence
        MCE measures the maximum difference between accuracy and confidence across bins
        
        Args:
            y_true: True binary labels (0 or 1)
            y_prob: Predicted probabilities
            return_details: Whether to return detailed bin information
            
        Returns:
            CalibrationResult with ECE, MCE, and reliability diagram data
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        # Input validation
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have same length")
        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y_true must contain only 0s and 1s")
        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            raise ValueError("y_prob must be in [0, 1]")
        
        # Create bins
        if self.bin_strategy == 'uniform':
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        else:  # quantile
            bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, self.n_bins + 1))
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        
        # Calculate per-bin statistics
        ece = 0.0
        mce = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        bin_positions = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Handle edge case for last bin
            if bin_lower == bin_uppers[-1]:
                in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            else:
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
                bin_positions.append((bin_lower + bin_upper) / 2)
                
                # Calculate calibration error for this bin
                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)
                bin_positions.append((bin_lower + bin_upper) / 2)
        
        # Calculate Brier score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Prepare reliability diagram data
        reliability_data = {
            'bin_boundaries': bin_boundaries,
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts),
            'bin_positions': np.array(bin_positions),
            'perfect_calibration': np.array(bin_positions)  # Perfect calibration line
        }
        
        return CalibrationResult(
            ece=ece,
            mce=mce,
            brier_score=brier_score,
            reliability_diagram_data=reliability_data,
            n_bins=self.n_bins,
            n_samples=len(y_true)
        )
    
    def bootstrap_ece_confidence_interval(self, 
                                        y_true: np.ndarray, 
                                        y_prob: np.ndarray,
                                        n_bootstrap: int = 1000) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate ECE with bootstrap confidence interval
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (ECE, confidence_interval)
        """
        def ece_statistic(indices):
            result = self.calculate_ece_mce(y_true[indices], y_prob[indices], return_details=False)
            return result.ece
        
        # Generate bootstrap samples
        n_samples = len(y_true)
        bootstrap_eces = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_ece = ece_statistic(indices)
            bootstrap_eces.append(bootstrap_ece)
        
        bootstrap_eces = np.array(bootstrap_eces)
        
        # Calculate confidence interval
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_eces, lower_percentile)
        ci_upper = np.percentile(bootstrap_eces, upper_percentile)
        
        original_ece = ece_statistic(np.arange(n_samples))
        
        return original_ece, (ci_lower, ci_upper)
    
    def temperature_scaling(self, 
                          y_true: np.ndarray, 
                          logits: np.ndarray,
                          validation_split: float = 0.2) -> TemperatureScalingResult:
        """
        Apply temperature scaling for calibration improvement
        
        Temperature scaling applies a single scalar parameter T to the logits:
        p_i = softmax(z_i / T) where z_i are the original logits
        
        Args:
            y_true: True binary labels
            logits: Raw model logits (before softmax)
            validation_split: Fraction of data to use for temperature optimization
            
        Returns:
            TemperatureScalingResult with optimal temperature and calibrated predictions
        """
        y_true = np.asarray(y_true)
        logits = np.asarray(logits)
        
        # Split data for temperature optimization
        n_samples = len(y_true)
        n_val = int(n_samples * validation_split)
        
        # Random split
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        
        y_val = y_true[val_indices]
        logits_val = logits[val_indices]
        
        # Convert logits to probabilities before calibration
        original_probs = self._logits_to_probs(logits)
        
        # Calculate pre-calibration ECE
        pre_calibration_result = self.calculate_ece_mce(y_true, original_probs, return_details=False)
        pre_calibration_ece = pre_calibration_result.ece
        
        # Define objective function for temperature optimization
        def temperature_objective(temperature):
            """Negative log-likelihood for temperature optimization"""
            if temperature <= 0:
                return np.inf
            
            scaled_logits = logits_val / temperature
            scaled_probs = self._logits_to_probs(scaled_logits)
            
            # Avoid log(0) by clipping
            scaled_probs = np.clip(scaled_probs, 1e-7, 1-1e-7)
            
            # Binary cross-entropy loss
            nll = -(y_val * np.log(scaled_probs) + (1 - y_val) * np.log(1 - scaled_probs)).mean()
            return nll
        
        # Optimize temperature
        optimization_result = minimize_scalar(
            temperature_objective,
            bounds=(0.1, 10.0),
            method='bounded'
        )
        
        optimal_temperature = optimization_result.x
        
        # Apply optimal temperature to all data
        calibrated_logits = logits / optimal_temperature
        calibrated_probs = self._logits_to_probs(calibrated_logits)
        
        # Calculate post-calibration ECE
        post_calibration_result = self.calculate_ece_mce(y_true, calibrated_probs, return_details=False)
        post_calibration_ece = post_calibration_result.ece
        
        improvement = pre_calibration_ece - post_calibration_ece
        
        return TemperatureScalingResult(
            optimal_temperature=optimal_temperature,
            calibrated_predictions=calibrated_probs,
            pre_calibration_ece=pre_calibration_ece,
            post_calibration_ece=post_calibration_ece,
            improvement=improvement,
            convergence_info={
                'success': optimization_result.success,
                'iterations': optimization_result.nit if hasattr(optimization_result, 'nit') else None,
                'message': optimization_result.message if hasattr(optimization_result, 'message') else None
            }
        )
    
    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """
        Convert logits to probabilities using sigmoid for binary classification
        
        Args:
            logits: Raw model logits
            
        Returns:
            Probabilities in [0, 1]
        """
        return 1 / (1 + np.exp(-logits))
    
    def plot_reliability_diagram(self, 
                                calibration_result: CalibrationResult,
                                title: str = "Reliability Diagram",
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create reliability diagram (calibration plot)
        
        Args:
            calibration_result: Result from calculate_ece_mce
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        data = calibration_result.reliability_diagram_data
        
        # Plot reliability curve
        mask = data['bin_counts'] > 0  # Only plot bins with samples
        ax.plot(
            data['bin_confidences'][mask], 
            data['bin_accuracies'][mask], 
            'o-', 
            label='Model Reliability',
            markersize=8,
            linewidth=2
        )
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, label='Perfect Calibration')
        
        # Add confidence intervals or error bars if available
        if calibration_result.confidence_interval:
            ci_lower, ci_upper = calibration_result.confidence_interval
            ax.fill_between([0, 1], [ci_lower, ci_lower], [ci_upper, ci_upper], 
                           alpha=0.2, color='blue', label=f'{self.confidence_level*100:.0f}% CI')
        
        # Formatting
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'{title}\nECE: {calibration_result.ece:.4f}, MCE: {calibration_result.mce:.4f}', 
                    fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add histogram of prediction distribution
        ax2 = ax.twinx()
        # Create histogram data
        bin_edges = data['bin_boundaries']
        bin_counts = data['bin_counts']
        
        # Plot histogram
        ax2.bar(data['bin_positions'][mask], bin_counts[mask], 
                width=1.0/self.n_bins, alpha=0.3, color='orange',
                label='Sample Distribution')
        ax2.set_ylabel('Number of Samples', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_calibrations(self, 
                           y_true: np.ndarray,
                           predictions_dict: Dict[str, np.ndarray],
                           save_path: Optional[str] = None) -> Dict[str, CalibrationResult]:
        """
        Compare calibration of multiple models
        
        Args:
            y_true: True binary labels
            predictions_dict: Dictionary of model_name -> predictions
            save_path: Path to save comparison plot
            
        Returns:
            Dictionary of model_name -> CalibrationResult
        """
        results = {}
        
        # Calculate calibration for each model
        for model_name, predictions in predictions_dict.items():
            results[model_name] = self.calculate_ece_mce(y_true, predictions)
        
        # Create comparison plot
        if len(predictions_dict) > 1:
            self._plot_calibration_comparison(results, save_path)
        
        return results
    
    def _plot_calibration_comparison(self, 
                                   results: Dict[str, CalibrationResult],
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8)):
        """
        Plot calibration comparison for multiple models
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        # Reliability diagram comparison
        for (model_name, result), color in zip(results.items(), colors):
            data = result.reliability_diagram_data
            mask = data['bin_counts'] > 0
            
            ax1.plot(
                data['bin_confidences'][mask], 
                data['bin_accuracies'][mask], 
                'o-', 
                label=f'{model_name} (ECE: {result.ece:.4f})',
                color=color,
                markersize=6,
                linewidth=2
            )
        
        ax1.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, label='Perfect Calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ECE/MCE comparison bar chart
        model_names = list(results.keys())
        ece_values = [results[name].ece for name in model_names]
        mce_values = [results[name].mce for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax2.bar(x - width/2, ece_values, width, label='ECE', alpha=0.8)
        ax2.bar(x + width/2, mce_values, width, label='MCE', alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Calibration Error')
        ax2.set_title('ECE and MCE Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def comprehensive_calibration_report(self, 
                                       y_true: np.ndarray,
                                       y_prob: np.ndarray,
                                       logits: Optional[np.ndarray] = None,
                                       model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate comprehensive calibration assessment report
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            logits: Optional raw logits for temperature scaling
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing all calibration metrics and analysis
        """
        report = {"model_name": model_name}
        
        # Basic calibration metrics
        calibration_result = self.calculate_ece_mce(y_true, y_prob)
        report["calibration_metrics"] = {
            "ECE": calibration_result.ece,
            "MCE": calibration_result.mce,
            "Brier_Score": calibration_result.brier_score,
            "n_samples": calibration_result.n_samples,
            "n_bins": calibration_result.n_bins
        }
        
        # ECE with confidence interval
        ece_with_ci, ci = self.bootstrap_ece_confidence_interval(y_true, y_prob)
        report["ECE_with_CI"] = {
            "ECE": ece_with_ci,
            "confidence_interval": ci,
            "confidence_level": self.confidence_level
        }
        
        # Temperature scaling if logits provided
        if logits is not None:
            temp_scaling_result = self.temperature_scaling(y_true, logits)
            report["temperature_scaling"] = {
                "optimal_temperature": temp_scaling_result.optimal_temperature,
                "pre_calibration_ECE": temp_scaling_result.pre_calibration_ece,
                "post_calibration_ECE": temp_scaling_result.post_calibration_ece,
                "improvement": temp_scaling_result.improvement,
                "convergence": temp_scaling_result.convergence_info
            }
        
        # Reliability diagram data for visualization
        report["reliability_data"] = calibration_result.reliability_diagram_data
        
        return report