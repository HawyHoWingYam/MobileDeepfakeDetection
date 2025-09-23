"""
Expert Comparison Framework for Stage 02 Heterogeneous System

This module provides comprehensive comparison tools for evaluating and analyzing
the complementarity between spatial and generative experts in the AWARE-NET Stage 02 system.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import pearsonr, spearmanr, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from collections import defaultdict

from .unified_feature_extractor import BaseExpert, ExpertOutput, ExpertType


@dataclass
class ComparisonMetrics:
    """Metrics for comparing expert performance and complementarity"""
    # Individual performance
    spatial_auc: float
    generative_auc: float
    spatial_f1: float
    generative_f1: float

    # Complementarity metrics
    correlation_coefficient: float
    error_pattern_overlap: float
    ensemble_improvement: float

    # Efficiency metrics
    spatial_inference_time: float
    generative_inference_time: float
    spatial_memory_usage: float
    generative_memory_usage: float

    # Statistical significance
    performance_p_value: float
    complementarity_significance: bool


@dataclass
class ExpertAnalysisResult:
    """Results from expert analysis and comparison"""
    individual_metrics: Dict[str, Dict[str, float]]
    complementarity_analysis: Dict[str, float]
    efficiency_comparison: Dict[str, Dict[str, float]]
    error_pattern_analysis: Dict[str, Any]
    fusion_potential: Dict[str, float]
    recommendations: List[str]


class ExpertPerformanceAnalyzer:
    """Analyzes individual expert performance with detailed metrics"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.results_cache = {}

    def _load_config(self, config_path: str) -> Dict:
        """Load analysis configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _default_config(self) -> Dict:
        """Default configuration for expert analysis"""
        return {
            "metrics": {
                "classification": ["auc", "f1", "precision", "recall"],
                "efficiency": ["inference_time", "memory_usage", "throughput"],
                "specialization": ["artifact_detection", "domain_specificity"]
            },
            "statistical_tests": {
                "significance_level": 0.05,
                "bootstrap_iterations": 1000,
                "confidence_interval": 0.95
            },
            "visualization": {
                "save_plots": True,
                "plot_format": "png",
                "dpi": 300
            }
        }

    def evaluate_expert(
        self,
        expert: BaseExpert,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        expert_name: str
    ) -> Dict[str, float]:
        """Comprehensive evaluation of a single expert"""
        expert.eval()

        predictions = []
        targets = []
        inference_times = []
        memory_usage = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)

                # Measure inference time
                start_time = time.time()

                # Memory tracking
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                # Expert prediction
                output = expert(images)
                predictions.extend(output.confidence_scores.cpu().numpy())
                targets.extend(labels.cpu().numpy())

                # Record timing and memory
                inference_times.append(time.time() - start_time)
                if device.type == 'cuda':
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB

        # Calculate comprehensive metrics
        metrics = self._calculate_performance_metrics(
            np.array(predictions),
            np.array(targets),
            inference_times,
            memory_usage
        )

        # Add expert-specific specialization metrics
        if expert_name == "spatial":
            metrics.update(self._calculate_spatial_specialization_metrics(predictions, targets))
        elif expert_name == "generative":
            metrics.update(self._calculate_generative_specialization_metrics(predictions, targets))

        self.results_cache[expert_name] = metrics
        return metrics

    def _calculate_performance_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        inference_times: List[float],
        memory_usage: List[float]
    ) -> Dict[str, float]:
        """Calculate standard performance metrics"""
        # Convert predictions to binary
        binary_preds = (predictions > 0.5).astype(int)

        metrics = {
            "auc": roc_auc_score(targets, predictions),
            "f1": f1_score(targets, binary_preds),
            "precision": precision_score(targets, binary_preds),
            "recall": recall_score(targets, binary_preds),
            "mean_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "mean_memory_usage": np.mean(memory_usage) if memory_usage else 0.0,
            "throughput": len(predictions) / np.sum(inference_times)
        }

        return metrics

    def _calculate_spatial_specialization_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics specific to spatial expert specialization"""
        # Placeholder for spatial-specific metrics
        # In real implementation, this would analyze spatial artifact detection capability
        return {
            "edge_artifact_detection_score": 0.85,  # Placeholder
            "texture_inconsistency_score": 0.82,    # Placeholder
            "spatial_frequency_analysis_score": 0.78  # Placeholder
        }

    def _calculate_generative_specialization_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics specific to generative expert specialization"""
        # Placeholder for generative-specific metrics
        # In real implementation, this would analyze generative structure detection capability
        return {
            "gan_detection_score": 0.88,        # Placeholder
            "diffusion_detection_score": 0.85,  # Placeholder
            "reconstruction_quality_score": 0.83  # Placeholder
        }


class ComplementarityAnalyzer:
    """Analyzes complementarity and fusion potential between experts"""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def analyze_complementarity(
        self,
        spatial_predictions: np.ndarray,
        generative_predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Comprehensive complementarity analysis"""

        # Correlation analysis
        correlation_pearson, p_pearson = pearsonr(spatial_predictions, generative_predictions)
        correlation_spearman, p_spearman = spearmanr(spatial_predictions, generative_predictions)

        # Error pattern analysis
        spatial_errors = self._get_error_patterns(spatial_predictions, targets)
        generative_errors = self._get_error_patterns(generative_predictions, targets)
        error_overlap = self._calculate_error_overlap(spatial_errors, generative_errors)

        # Ensemble improvement analysis
        ensemble_improvement = self._calculate_ensemble_improvement(
            spatial_predictions, generative_predictions, targets
        )

        # Diversity metrics
        diversity_score = self._calculate_diversity_score(
            spatial_predictions, generative_predictions, targets
        )

        # Fusion potential
        fusion_potential = self._estimate_fusion_potential(
            spatial_predictions, generative_predictions, targets
        )

        return {
            "correlation_pearson": correlation_pearson,
            "correlation_spearman": correlation_spearman,
            "correlation_significance": p_pearson < self.significance_level,
            "error_pattern_overlap": error_overlap,
            "ensemble_improvement": ensemble_improvement,
            "diversity_score": diversity_score,
            "fusion_potential": fusion_potential,
            "complementarity_index": self._calculate_complementarity_index(
                correlation_pearson, error_overlap, ensemble_improvement, diversity_score
            )
        }

    def _get_error_patterns(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Identify error patterns in predictions"""
        binary_preds = (predictions > 0.5).astype(int)
        errors = (binary_preds != targets).astype(int)
        return errors

    def _calculate_error_overlap(self, errors1: np.ndarray, errors2: np.ndarray) -> float:
        """Calculate overlap in error patterns between two experts"""
        intersection = np.sum(errors1 * errors2)
        union = np.sum((errors1 + errors2) > 0)
        return intersection / union if union > 0 else 0.0

    def _calculate_ensemble_improvement(
        self,
        spatial_preds: np.ndarray,
        generative_preds: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate improvement from simple ensemble"""
        # Simple average ensemble
        ensemble_preds = (spatial_preds + generative_preds) / 2

        # Individual AUCs
        spatial_auc = roc_auc_score(targets, spatial_preds)
        generative_auc = roc_auc_score(targets, generative_preds)
        ensemble_auc = roc_auc_score(targets, ensemble_preds)

        # Calculate improvement over best individual expert
        best_individual = max(spatial_auc, generative_auc)
        improvement = ensemble_auc - best_individual

        return improvement

    def _calculate_diversity_score(
        self,
        spatial_preds: np.ndarray,
        generative_preds: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate diversity score between experts"""
        # Q-statistic for diversity
        spatial_correct = (spatial_preds > 0.5) == targets
        generative_correct = (generative_preds > 0.5) == targets

        n11 = np.sum(spatial_correct & generative_correct)
        n10 = np.sum(spatial_correct & ~generative_correct)
        n01 = np.sum(~spatial_correct & generative_correct)
        n00 = np.sum(~spatial_correct & ~generative_correct)

        q_statistic = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)

        # Convert to diversity score (lower Q means higher diversity)
        diversity_score = 1 - abs(q_statistic)
        return diversity_score

    def _estimate_fusion_potential(
        self,
        spatial_preds: np.ndarray,
        generative_preds: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Estimate potential for sophisticated fusion methods"""
        # Test multiple fusion strategies
        fusion_scores = []

        # Weighted average with different weights
        for weight in [0.3, 0.4, 0.5, 0.6, 0.7]:
            fused = weight * spatial_preds + (1 - weight) * generative_preds
            score = roc_auc_score(targets, fused)
            fusion_scores.append(score)

        # Maximum fusion performance
        max_fusion_score = max(fusion_scores)

        # Best individual performance
        best_individual = max(
            roc_auc_score(targets, spatial_preds),
            roc_auc_score(targets, generative_preds)
        )

        # Fusion potential as improvement over best individual
        fusion_potential = max_fusion_score - best_individual
        return fusion_potential

    def _calculate_complementarity_index(
        self,
        correlation: float,
        error_overlap: float,
        ensemble_improvement: float,
        diversity_score: float
    ) -> float:
        """Calculate overall complementarity index"""
        # Lower correlation and error overlap = better complementarity
        # Higher ensemble improvement and diversity = better complementarity

        complementarity = (
            (1 - abs(correlation)) * 0.25 +
            (1 - error_overlap) * 0.25 +
            ensemble_improvement * 2.0 +  # Scale up improvement
            diversity_score * 0.5
        )

        return max(0.0, min(1.0, complementarity))


class ExpertComparisonFramework:
    """Main framework for comprehensive expert comparison"""

    def __init__(self, config_path: Optional[str] = None, output_dir: str = "comparison_results"):
        self.performance_analyzer = ExpertPerformanceAnalyzer(config_path)
        self.complementarity_analyzer = ComplementarityAnalyzer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def compare_experts(
        self,
        spatial_expert: BaseExpert,
        generative_expert: BaseExpert,
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> ExpertAnalysisResult:
        """Comprehensive comparison of spatial and generative experts"""

        print("Evaluating spatial expert...")
        spatial_metrics = self.performance_analyzer.evaluate_expert(
            spatial_expert, test_dataloader, device, "spatial"
        )

        print("Evaluating generative expert...")
        generative_metrics = self.performance_analyzer.evaluate_expert(
            generative_expert, test_dataloader, device, "generative"
        )

        print("Analyzing complementarity...")
        # Get predictions for complementarity analysis
        spatial_preds, generative_preds, targets = self._get_expert_predictions(
            spatial_expert, generative_expert, test_dataloader, device
        )

        complementarity_analysis = self.complementarity_analyzer.analyze_complementarity(
            spatial_preds, generative_preds, targets
        )

        # Efficiency comparison
        efficiency_comparison = self._compare_efficiency(spatial_metrics, generative_metrics)

        # Error pattern analysis
        error_pattern_analysis = self._analyze_error_patterns(
            spatial_preds, generative_preds, targets
        )

        # Fusion potential analysis
        fusion_potential = self._analyze_fusion_potential(
            spatial_preds, generative_preds, targets
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            spatial_metrics, generative_metrics, complementarity_analysis
        )

        # Create comprehensive result
        result = ExpertAnalysisResult(
            individual_metrics={
                "spatial": spatial_metrics,
                "generative": generative_metrics
            },
            complementarity_analysis=complementarity_analysis,
            efficiency_comparison=efficiency_comparison,
            error_pattern_analysis=error_pattern_analysis,
            fusion_potential=fusion_potential,
            recommendations=recommendations
        )

        # Save results
        self._save_results(result)

        # Generate visualizations
        self._generate_visualizations(result, spatial_preds, generative_preds, targets)

        return result

    def _get_expert_predictions(
        self,
        spatial_expert: BaseExpert,
        generative_expert: BaseExpert,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions from both experts for comparison"""

        spatial_expert.eval()
        generative_expert.eval()

        spatial_preds = []
        generative_preds = []
        targets = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Get predictions from both experts
                spatial_output = spatial_expert(images)
                generative_output = generative_expert(images)

                spatial_preds.extend(spatial_output.confidence_scores.cpu().numpy())
                generative_preds.extend(generative_output.confidence_scores.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        return np.array(spatial_preds), np.array(generative_preds), np.array(targets)

    def _compare_efficiency(
        self,
        spatial_metrics: Dict[str, float],
        generative_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare efficiency metrics between experts"""

        efficiency_metrics = ["mean_inference_time", "mean_memory_usage", "throughput"]

        comparison = {
            "spatial": {},
            "generative": {},
            "ratio": {}
        }

        for metric in efficiency_metrics:
            spatial_val = spatial_metrics.get(metric, 0.0)
            generative_val = generative_metrics.get(metric, 0.0)

            comparison["spatial"][metric] = spatial_val
            comparison["generative"][metric] = generative_val

            # Calculate ratio (spatial / generative)
            if generative_val > 0:
                comparison["ratio"][metric] = spatial_val / generative_val
            else:
                comparison["ratio"][metric] = float('inf') if spatial_val > 0 else 1.0

        return comparison

    def _analyze_error_patterns(
        self,
        spatial_preds: np.ndarray,
        generative_preds: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze error patterns and failure modes"""

        spatial_errors = self.complementarity_analyzer._get_error_patterns(spatial_preds, targets)
        generative_errors = self.complementarity_analyzer._get_error_patterns(generative_preds, targets)

        # Categorize errors
        spatial_only_errors = spatial_errors & ~generative_errors
        generative_only_errors = generative_errors & ~spatial_errors
        common_errors = spatial_errors & generative_errors

        error_analysis = {
            "spatial_only_error_rate": np.mean(spatial_only_errors),
            "generative_only_error_rate": np.mean(generative_only_errors),
            "common_error_rate": np.mean(common_errors),
            "total_error_reduction_potential": np.mean(spatial_errors | generative_errors) - np.mean(common_errors),
            "error_pattern_diversity": 1 - np.mean(common_errors) / max(np.mean(spatial_errors), np.mean(generative_errors))
        }

        return error_analysis

    def _analyze_fusion_potential(
        self,
        spatial_preds: np.ndarray,
        generative_preds: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Analyze potential for different fusion strategies"""

        fusion_strategies = {
            "simple_average": (spatial_preds + generative_preds) / 2,
            "weighted_average_spatial": 0.6 * spatial_preds + 0.4 * generative_preds,
            "weighted_average_generative": 0.4 * spatial_preds + 0.6 * generative_preds,
            "max_confidence": np.maximum(spatial_preds, generative_preds),
            "min_confidence": np.minimum(spatial_preds, generative_preds)
        }

        fusion_scores = {}
        for strategy_name, fused_preds in fusion_strategies.items():
            auc_score = roc_auc_score(targets, fused_preds)
            fusion_scores[f"{strategy_name}_auc"] = auc_score

        # Best individual scores for comparison
        spatial_auc = roc_auc_score(targets, spatial_preds)
        generative_auc = roc_auc_score(targets, generative_preds)
        best_individual = max(spatial_auc, generative_auc)

        # Calculate improvements
        for strategy_name in fusion_strategies.keys():
            improvement = fusion_scores[f"{strategy_name}_auc"] - best_individual
            fusion_scores[f"{strategy_name}_improvement"] = improvement

        return fusion_scores

    def _generate_recommendations(
        self,
        spatial_metrics: Dict[str, float],
        generative_metrics: Dict[str, float],
        complementarity_analysis: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""

        recommendations = []

        # Performance recommendations
        spatial_auc = spatial_metrics.get("auc", 0.0)
        generative_auc = generative_metrics.get("auc", 0.0)

        if spatial_auc > generative_auc + 0.02:
            recommendations.append("Spatial expert shows superior performance. Consider prioritizing spatial expert in cascade system.")
        elif generative_auc > spatial_auc + 0.02:
            recommendations.append("Generative expert shows superior performance. Consider prioritizing generative expert in cascade system.")
        else:
            recommendations.append("Both experts show comparable performance. Ensemble approach recommended.")

        # Complementarity recommendations
        complementarity_index = complementarity_analysis.get("complementarity_index", 0.0)
        correlation = complementarity_analysis.get("correlation_pearson", 0.0)

        if complementarity_index > 0.7:
            recommendations.append("High complementarity detected. Strong candidate for ensemble fusion.")
        elif complementarity_index < 0.3:
            recommendations.append("Low complementarity. Consider specialized deployment or cascade system.")

        if abs(correlation) < 0.3:
            recommendations.append("Low correlation between experts. Good diversity for ensemble methods.")
        elif abs(correlation) > 0.7:
            recommendations.append("High correlation detected. Consider expert specialization or different architectures.")

        # Efficiency recommendations
        spatial_time = spatial_metrics.get("mean_inference_time", 0.0)
        generative_time = generative_metrics.get("mean_inference_time", 0.0)

        if spatial_time < generative_time * 0.7:
            recommendations.append("Spatial expert is significantly faster. Suitable for real-time applications.")
        elif generative_time < spatial_time * 0.7:
            recommendations.append("Generative expert is significantly faster. Optimize spatial expert for efficiency.")

        # Fusion recommendations
        ensemble_improvement = complementarity_analysis.get("ensemble_improvement", 0.0)

        if ensemble_improvement > 0.03:
            recommendations.append("Significant ensemble improvement detected. Implement advanced fusion methods.")
        elif ensemble_improvement < 0.01:
            recommendations.append("Limited ensemble benefit. Consider separate deployment strategies.")

        return recommendations

    def _save_results(self, result: ExpertAnalysisResult):
        """Save analysis results to files"""

        # Save JSON summary
        summary = {
            "individual_metrics": result.individual_metrics,
            "complementarity_analysis": result.complementarity_analysis,
            "efficiency_comparison": result.efficiency_comparison,
            "error_pattern_analysis": result.error_pattern_analysis,
            "fusion_potential": result.fusion_potential,
            "recommendations": result.recommendations
        }

        with open(self.output_dir / "expert_comparison_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed report
        self._generate_detailed_report(result)

    def _generate_detailed_report(self, result: ExpertAnalysisResult):
        """Generate detailed markdown report"""

        report_lines = [
            "# Expert Comparison Analysis Report",
            "",
            "## Executive Summary",
            "",
        ]

        # Add recommendations
        report_lines.extend([
            "### Key Recommendations",
            ""
        ])

        for i, rec in enumerate(result.recommendations, 1):
            report_lines.append(f"{i}. {rec}")

        report_lines.extend([
            "",
            "## Individual Expert Performance",
            ""
        ])

        # Performance metrics table
        spatial_metrics = result.individual_metrics["spatial"]
        generative_metrics = result.individual_metrics["generative"]

        report_lines.extend([
            "| Metric | Spatial Expert | Generative Expert |",
            "|--------|----------------|-------------------|"
        ])

        key_metrics = ["auc", "f1", "precision", "recall", "mean_inference_time"]
        for metric in key_metrics:
            spatial_val = spatial_metrics.get(metric, 0.0)
            generative_val = generative_metrics.get(metric, 0.0)
            report_lines.append(f"| {metric} | {spatial_val:.4f} | {generative_val:.4f} |")

        # Complementarity analysis
        report_lines.extend([
            "",
            "## Complementarity Analysis",
            ""
        ])

        comp_analysis = result.complementarity_analysis
        for key, value in comp_analysis.items():
            if isinstance(value, float):
                report_lines.append(f"- **{key}**: {value:.4f}")
            else:
                report_lines.append(f"- **{key}**: {value}")

        # Save report
        with open(self.output_dir / "expert_comparison_report.md", 'w') as f:
            f.write('\n'.join(report_lines))

    def _generate_visualizations(
        self,
        result: ExpertAnalysisResult,
        spatial_preds: np.ndarray,
        generative_preds: np.ndarray,
        targets: np.ndarray
    ):
        """Generate visualization plots"""

        # Performance comparison plot
        self._plot_performance_comparison(result.individual_metrics)

        # Prediction correlation plot
        self._plot_prediction_correlation(spatial_preds, generative_preds, targets)

        # Error pattern analysis plot
        self._plot_error_patterns(spatial_preds, generative_preds, targets)

        # Fusion potential plot
        self._plot_fusion_potential(result.fusion_potential)

    def _plot_performance_comparison(self, individual_metrics: Dict[str, Dict[str, float]]):
        """Plot performance comparison between experts"""

        metrics_to_plot = ["auc", "f1", "precision", "recall"]
        spatial_values = [individual_metrics["spatial"].get(m, 0.0) for m in metrics_to_plot]
        generative_values = [individual_metrics["generative"].get(m, 0.0) for m in metrics_to_plot]

        x = np.arange(len(metrics_to_plot))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, spatial_values, width, label='Spatial Expert', alpha=0.8)
        bars2 = ax.bar(x + width/2, generative_values, width, label='Generative Expert', alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison: Spatial vs Generative Expert')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300)
        plt.close()

    def _plot_prediction_correlation(
        self,
        spatial_preds: np.ndarray,
        generative_preds: np.ndarray,
        targets: np.ndarray
    ):
        """Plot correlation between expert predictions"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot of predictions
        colors = ['red' if t == 1 else 'blue' for t in targets]
        ax1.scatter(spatial_preds, generative_preds, c=colors, alpha=0.6)
        ax1.set_xlabel('Spatial Expert Predictions')
        ax1.set_ylabel('Generative Expert Predictions')
        ax1.set_title('Expert Prediction Correlation')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr_coeff = np.corrcoef(spatial_preds, generative_preds)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr_coeff:.3f}',
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

        # Prediction difference histogram
        pred_diff = spatial_preds - generative_preds
        ax2.hist(pred_diff, bins=50, alpha=0.7, color='green')
        ax2.set_xlabel('Prediction Difference (Spatial - Generative)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Prediction Differences')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_correlation.png", dpi=300)
        plt.close()

    def _plot_error_patterns(
        self,
        spatial_preds: np.ndarray,
        generative_preds: np.ndarray,
        targets: np.ndarray
    ):
        """Plot error pattern analysis"""

        spatial_errors = (spatial_preds > 0.5) != targets
        generative_errors = (generative_preds > 0.5) != targets

        # Create confusion matrix style plot for error patterns
        spatial_only = spatial_errors & ~generative_errors
        generative_only = generative_errors & ~spatial_errors
        both_correct = ~spatial_errors & ~generative_errors
        both_wrong = spatial_errors & generative_errors

        categories = ['Both Correct', 'Spatial Only Wrong', 'Generative Only Wrong', 'Both Wrong']
        counts = [
            np.sum(both_correct),
            np.sum(spatial_only),
            np.sum(generative_only),
            np.sum(both_wrong)
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green', 'orange', 'blue', 'red']
        bars = ax.bar(categories, counts, color=colors, alpha=0.7)

        ax.set_ylabel('Number of Samples')
        ax.set_title('Error Pattern Distribution')
        ax.grid(True, alpha=0.3)

        # Add percentage labels
        total_samples = len(targets)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total_samples) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + total_samples*0.01,
                   f'{percentage:.1f}%', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_patterns.png", dpi=300)
        plt.close()

    def _plot_fusion_potential(self, fusion_potential: Dict[str, float]):
        """Plot fusion strategy comparison"""

        # Extract AUC scores for different fusion strategies
        fusion_aucs = {k.replace('_auc', ''): v for k, v in fusion_potential.items()
                      if k.endswith('_auc')}

        strategies = list(fusion_aucs.keys())
        scores = list(fusion_aucs.values())

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(strategies, scores, color='skyblue', alpha=0.8)

        ax.set_ylabel('AUC Score')
        ax.set_title('Fusion Strategy Performance Comparison')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{score:.4f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "fusion_potential.png", dpi=300)
        plt.close()


def main():
    """Example usage of the expert comparison framework"""

    # This would be used after training both experts
    print("Expert Comparison Framework initialized")
    print("Ready for comprehensive expert analysis")

    # Example workflow:
    # 1. Load trained spatial and generative experts
    # 2. Prepare test dataset
    # 3. Run comparison framework
    # 4. Analyze results and recommendations

    framework = ExpertComparisonFramework()
    print(f"Results will be saved to: {framework.output_dir}")


if __name__ == "__main__":
    main()