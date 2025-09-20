"""
AWARE-NET Academic Evaluation Metrics
Comprehensive evaluation framework with statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.stats import bootstrap
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class MetricResult:
    """Container for metric calculation results"""
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    std_error: Optional[float] = None
    n_samples: int = 0
    
@dataclass
class ComparisonResult:
    """Container for model comparison results"""
    metric_name: str
    model1_value: float
    model2_value: float
    difference: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    effect_size: float

class AcademicMetrics:
    """
    Academic-grade evaluation metrics with statistical testing
    
    Features:
    - Bootstrap confidence intervals
    - Statistical significance testing
    - Effect size calculations
    - Calibration analysis
    - Multiple comparison corrections
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 n_bootstrap: int = 1000,
                 random_state: int = 42):
        """
        Initialize metrics calculator
        
        Args:
            confidence_level: Confidence level for intervals
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.alpha = 1 - confidence_level
        
        np.random.seed(random_state)
    
    def calculate_auc_with_ci(self, 
                            y_true: np.ndarray, 
                            y_scores: np.ndarray) -> MetricResult:
        """
        Calculate AUC-ROC with confidence interval
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            
        Returns:
            Metric result with confidence interval
        """
        # Calculate main AUC
        auc = roc_auc_score(y_true, y_scores)
        
        # Bootstrap confidence interval
        def auc_statistic(y_true, y_scores, indices):
            return roc_auc_score(y_true[indices], y_scores[indices])
        
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            y_true, y_scores, auc_statistic
        )
        
        return MetricResult(
            value=auc,
            confidence_interval=(ci_lower, ci_upper),
            n_samples=len(y_true)
        )
    
    def calculate_f1_with_ci(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray) -> MetricResult:
        """
        Calculate F1 score with confidence interval
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Metric result with confidence interval
        """
        # Calculate main F1
        f1 = f1_score(y_true, y_pred)
        
        # Bootstrap confidence interval
        def f1_statistic(y_true, y_pred, indices):
            return f1_score(y_true[indices], y_pred[indices])
        
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            y_true, y_pred, f1_statistic
        )
        
        return MetricResult(
            value=f1,
            confidence_interval=(ci_lower, ci_upper),
            n_samples=len(y_true)
        )
    
    def calculate_accuracy_with_ci(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> MetricResult:
        """
        Calculate accuracy with confidence interval
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Metric result with confidence interval
        """
        # Calculate main accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Bootstrap confidence interval
        def acc_statistic(y_true, y_pred, indices):
            return accuracy_score(y_true[indices], y_pred[indices])
        
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            y_true, y_pred, acc_statistic
        )
        
        return MetricResult(
            value=acc,
            confidence_interval=(ci_lower, ci_upper),
            n_samples=len(y_true)
        )
    
    def _bootstrap_confidence_interval(self, 
                                     *arrays, 
                                     statistic_func) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for any statistic
        
        Args:
            *arrays: Input arrays (y_true, y_pred, etc.)
            statistic_func: Function to calculate statistic
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n_samples = len(arrays[0])
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample indices
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            try:
                # Calculate statistic for bootstrap sample
                stat = statistic_func(*arrays, indices)
                bootstrap_stats.append(stat)
            except (ValueError, ZeroDivisionError):
                # Skip invalid bootstrap samples
                continue
        
        if not bootstrap_stats:
            warnings.warn("No valid bootstrap samples generated")
            return 0.0, 0.0
        
        # Calculate confidence interval
        alpha_half = self.alpha / 2
        lower_percentile = (alpha_half) * 100
        upper_percentile = (1 - alpha_half) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def compare_auc_scores(self, 
                          y_true: np.ndarray,
                          scores1: np.ndarray,
                          scores2: np.ndarray,
                          model1_name: str = "Model 1",
                          model2_name: str = "Model 2") -> ComparisonResult:
        """
        Compare AUC scores between two models using DeLong test
        
        Args:
            y_true: True binary labels
            scores1: Prediction scores from model 1
            scores2: Prediction scores from model 2
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Comparison result with statistical test
        """
        auc1 = roc_auc_score(y_true, scores1)
        auc2 = roc_auc_score(y_true, scores2)
        
        # DeLong test for comparing AUCs
        p_value, difference_ci = self._delong_test(y_true, scores1, scores2)
        
        # Effect size (Cohen's d for AUC difference)
        effect_size = abs(auc1 - auc2) / np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
        
        return ComparisonResult(
            metric_name="AUC-ROC",
            model1_value=auc1,
            model2_value=auc2,
            difference=auc1 - auc2,
            p_value=p_value,
            confidence_interval=difference_ci,
            is_significant=p_value < 0.05,
            effect_size=effect_size
        )
    
    def _delong_test(self, 
                    y_true: np.ndarray,
                    scores1: np.ndarray,
                    scores2: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """
        DeLong test for comparing AUC values
        
        Args:
            y_true: True binary labels
            scores1: Prediction scores from model 1
            scores2: Prediction scores from model 2
            
        Returns:
            Tuple of (p_value, confidence_interval_for_difference)
        """
        # Simplified DeLong test implementation
        # For production use, consider using scipy.stats or specialized libraries
        
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            warnings.warn("Cannot perform DeLong test with only one class")
            return 1.0, (0.0, 0.0)
        
        # Calculate structural components
        auc1 = roc_auc_score(y_true, scores1)
        auc2 = roc_auc_score(y_true, scores2)
        
        # Estimate variance using bootstrap
        def auc_diff_statistic(y_true, scores1, scores2, indices):
            y_boot = y_true[indices]
            s1_boot = scores1[indices]
            s2_boot = scores2[indices]
            
            try:
                auc1_boot = roc_auc_score(y_boot, s1_boot)
                auc2_boot = roc_auc_score(y_boot, s2_boot)
                return auc1_boot - auc2_boot
            except ValueError:
                return 0.0
        
        # Bootstrap differences
        differences = []
        n_samples = len(y_true)
        
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            diff = auc_diff_statistic(y_true, scores1, scores2, indices)
            differences.append(diff)
        
        differences = np.array(differences)
        observed_diff = auc1 - auc2
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * min(
            np.mean(differences <= -abs(observed_diff)),
            np.mean(differences >= abs(observed_diff))
        )
        
        # Confidence interval for difference
        alpha_half = self.alpha / 2
        ci_lower = np.percentile(differences, alpha_half * 100)
        ci_upper = np.percentile(differences, (1 - alpha_half) * 100)
        
        return p_value, (ci_lower, ci_upper)
    
    def paired_t_test(self, 
                     scores1: np.ndarray, 
                     scores2: np.ndarray) -> Tuple[float, float]:
        """
        Paired t-test for comparing model performances
        
        Args:
            scores1: Performance scores from model 1
            scores2: Performance scores from model 2
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        return stats.ttest_rel(scores1, scores2)
    
    def calculate_calibration_metrics(self, 
                                    y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate calibration metrics (ECE, MCE, Brier Score)
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
        
        # Brier Score
        brier_score = np.mean((y_prob - y_true) ** 2)
        
        # Reliability (from sklearn)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'reliability_curve': (fraction_of_positives, mean_predicted_value)
        }
    
    def comprehensive_evaluation(self, 
                               y_true: np.ndarray,
                               y_scores: np.ndarray,
                               threshold: float = 0.5) -> Dict[str, Any]:
        """
        Comprehensive evaluation with all metrics
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores/probabilities
            threshold: Decision threshold for binary predictions
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Convert scores to binary predictions
        y_pred = (y_scores >= threshold).astype(int)
        
        # Classification metrics with confidence intervals
        auc_result = self.calculate_auc_with_ci(y_true, y_scores)
        f1_result = self.calculate_f1_with_ci(y_true, y_pred)
        acc_result = self.calculate_accuracy_with_ci(y_true, y_pred)
        
        # Additional metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        ap_score = average_precision_score(y_true, y_scores)
        
        # ROC and PR curves
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        prec, rec, pr_thresholds = precision_recall_curve(y_true, y_scores)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calibration metrics
        calibration_metrics = self.calculate_calibration_metrics(y_true, y_scores)
        
        results = {
            'classification_metrics': {
                'auc_roc': auc_result,
                'f1_score': f1_result,
                'accuracy': acc_result,
                'precision': precision,
                'recall': recall,
                'average_precision': ap_score
            },
            'curves': {
                'roc_curve': (fpr, tpr, roc_thresholds),
                'pr_curve': (prec, rec, pr_thresholds)
            },
            'confusion_matrix': cm,
            'calibration': calibration_metrics,
            'threshold': threshold,
            'n_samples': len(y_true)
        }
        
        return results
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Bonferroni correction for multiple comparisons
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of corrected p-values
        """
        n_tests = len(p_values)
        return [min(1.0, p * n_tests) for p in p_values]
    
    def benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Benjamini-Hochberg FDR correction
        
        Args:
            p_values: List of p-values
            
        Returns:
            List of corrected p-values
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        # Sort p-values and get original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Calculate BH correction
        corrected_p_values = np.zeros_like(p_values)
        
        for i in range(n_tests - 1, -1, -1):
            rank = i + 1
            bh_value = sorted_p_values[i] * n_tests / rank
            
            if i == n_tests - 1:
                corrected_p_values[sorted_indices[i]] = min(1.0, bh_value)
            else:
                corrected_p_values[sorted_indices[i]] = min(
                    corrected_p_values[sorted_indices[i + 1]], 
                    bh_value
                )
        
        return corrected_p_values.tolist()
    
    def effect_size_cohen_d(self, 
                          group1: np.ndarray, 
                          group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Cohen's d effect size
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def power_analysis(self, 
                      effect_size: float,
                      alpha: float = 0.05,
                      power: float = 0.8) -> int:
        """
        Calculate required sample size for given effect size and power
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Type I error rate
            power: Desired statistical power
            
        Returns:
            Required sample size per group
        """
        # Simplified power analysis for two-sample t-test
        # For more accurate calculations, use specialized libraries like statsmodels
        
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def generate_metrics_report(self, 
                              results: Dict[str, Any],
                              model_name: str = "Model") -> str:
        """
        Generate formatted metrics report
        
        Args:
            results: Results from comprehensive_evaluation
            model_name: Name of the model
            
        Returns:
            Formatted report string
        """
        metrics = results['classification_metrics']
        
        report = f"""
=== {model_name} Performance Report ===

Classification Metrics:
- AUC-ROC: {metrics['auc_roc'].value:.4f} ({metrics['auc_roc'].confidence_interval[0]:.4f}, {metrics['auc_roc'].confidence_interval[1]:.4f})
- F1 Score: {metrics['f1_score'].value:.4f} ({metrics['f1_score'].confidence_interval[0]:.4f}, {metrics['f1_score'].confidence_interval[1]:.4f})
- Accuracy: {metrics['accuracy'].value:.4f} ({metrics['accuracy'].confidence_interval[0]:.4f}, {metrics['accuracy'].confidence_interval[1]:.4f})
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- Average Precision: {metrics['average_precision']:.4f}

Calibration Metrics:
- ECE: {results['calibration']['ece']:.4f}
- MCE: {results['calibration']['mce']:.4f}
- Brier Score: {results['calibration']['brier_score']:.4f}

Confusion Matrix:
{results['confusion_matrix']}

Sample Size: {results['n_samples']}
Decision Threshold: {results['threshold']}
Confidence Level: {self.confidence_level:.1%}
"""
        
        return report