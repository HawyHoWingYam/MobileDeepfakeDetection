"""
Heterogeneous Validation Tools for Stage 02 Expert System

This module provides comprehensive validation tools specifically designed for
heterogeneous expert systems, including cross-expert validation, ensemble testing,
and specialized metric analysis for multi-expert deepfake detection.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import logging
from collections import defaultdict
import warnings

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_rel, chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sns

from .unified_feature_extractor import BaseExpert, ExpertOutput, ExpertType, HeterogeneousOutput
from .expert_comparison import ExpertComparisonFramework


@dataclass
class ValidationConfig:
    """Configuration for heterogeneous validation"""
    # Statistical settings
    significance_level: float = 0.05
    confidence_interval: float = 0.95
    bootstrap_iterations: int = 1000
    cross_validation_folds: int = 5

    # Performance thresholds
    min_individual_auc: float = 0.88
    min_ensemble_auc: float = 0.92
    min_complementarity_score: float = 0.3
    max_correlation_threshold: float = 0.7

    # Efficiency requirements
    max_inference_time_ms: float = 200.0
    max_memory_usage_mb: float = 3000.0
    min_throughput_fps: float = 5.0

    # Validation strategies
    enable_cross_expert_validation: bool = True
    enable_ensemble_validation: bool = True
    enable_robustness_testing: bool = True
    enable_fairness_validation: bool = True
    enable_interpretability_validation: bool = True


@dataclass
class ValidationResult:
    """Comprehensive validation result for heterogeneous system"""
    # Overall system validation
    system_passed: bool
    individual_expert_results: Dict[str, Dict[str, Any]]
    ensemble_results: Dict[str, Any]

    # Detailed analysis
    statistical_tests: Dict[str, Any]
    robustness_analysis: Dict[str, Any]
    fairness_analysis: Dict[str, Any]
    interpretability_analysis: Dict[str, Any]

    # Performance breakdown
    performance_breakdown: Dict[str, Dict[str, float]]
    failure_analysis: Dict[str, Any]

    # Recommendations
    validation_recommendations: List[str]
    improvement_suggestions: List[str]

    # Metadata
    validation_timestamp: str
    config_used: ValidationConfig


class StatisticalValidator:
    """Statistical validation for expert system performance"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_statistical_significance(
        self,
        expert_predictions: Dict[str, np.ndarray],
        targets: np.ndarray,
        baseline_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Validate statistical significance of expert performance"""

        results = {
            "significance_tests": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "power_analysis": {}
        }

        expert_names = list(expert_predictions.keys())

        # Pairwise comparisons between experts
        for i, expert1 in enumerate(expert_names):
            for expert2 in expert_names[i+1:]:
                comparison_key = f"{expert1}_vs_{expert2}"

                # AUC comparison using bootstrap
                auc1_bootstrap = self._bootstrap_auc(expert_predictions[expert1], targets)
                auc2_bootstrap = self._bootstrap_auc(expert_predictions[expert2], targets)

                # Statistical test
                stat, p_value = mannwhitneyu(auc1_bootstrap, auc2_bootstrap, alternative='two-sided')

                results["significance_tests"][comparison_key] = {
                    "test_statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.config.significance_level,
                    "effect_size": self._calculate_effect_size(auc1_bootstrap, auc2_bootstrap)
                }

        # Baseline comparisons if provided
        if baseline_predictions is not None:
            baseline_auc_bootstrap = self._bootstrap_auc(baseline_predictions, targets)

            for expert_name, predictions in expert_predictions.items():
                expert_auc_bootstrap = self._bootstrap_auc(predictions, targets)

                stat, p_value = mannwhitneyu(
                    expert_auc_bootstrap, baseline_auc_bootstrap, alternative='greater'
                )

                results["significance_tests"][f"{expert_name}_vs_baseline"] = {
                    "test_statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.config.significance_level,
                    "improvement_significant": p_value < self.config.significance_level
                }

        # Calculate confidence intervals
        for expert_name, predictions in expert_predictions.items():
            auc_bootstrap = self._bootstrap_auc(predictions, targets)
            ci_lower = np.percentile(auc_bootstrap, (1 - self.config.confidence_interval) / 2 * 100)
            ci_upper = np.percentile(auc_bootstrap, (1 + self.config.confidence_interval) / 2 * 100)

            results["confidence_intervals"][expert_name] = {
                "auc_mean": float(np.mean(auc_bootstrap)),
                "auc_std": float(np.std(auc_bootstrap)),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "ci_width": float(ci_upper - ci_lower)
            }

        return results

    def _bootstrap_auc(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Bootstrap AUC scores for statistical testing"""
        n_samples = len(predictions)
        bootstrap_aucs = []

        for _ in range(self.config.bootstrap_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_pred = predictions[indices]
            boot_target = targets[indices]

            # Calculate AUC if both classes present
            if len(np.unique(boot_target)) > 1:
                auc = roc_auc_score(boot_target, boot_pred)
                bootstrap_aucs.append(auc)

        return np.array(bootstrap_aucs)

    def _calculate_effect_size(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

        # Pooled standard deviation
        n1, n2 = len(sample1), len(sample2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std


class RobustnessValidator:
    """Validate robustness of expert system to various perturbations"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_robustness(
        self,
        experts: Dict[str, BaseExpert],
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Comprehensive robustness validation"""

        results = {
            "noise_robustness": {},
            "compression_robustness": {},
            "resolution_robustness": {},
            "adversarial_robustness": {},
            "distribution_shift_robustness": {}
        }

        # Noise robustness testing
        results["noise_robustness"] = self._test_noise_robustness(
            experts, test_dataloader, device
        )

        # Compression robustness testing
        results["compression_robustness"] = self._test_compression_robustness(
            experts, test_dataloader, device
        )

        # Resolution robustness testing
        results["resolution_robustness"] = self._test_resolution_robustness(
            experts, test_dataloader, device
        )

        # Basic adversarial robustness (FGSM)
        results["adversarial_robustness"] = self._test_adversarial_robustness(
            experts, test_dataloader, device
        )

        return results

    def _test_noise_robustness(
        self,
        experts: Dict[str, BaseExpert],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Test robustness to various noise types"""

        noise_types = {
            "gaussian": lambda x, std: x + torch.randn_like(x) * std,
            "uniform": lambda x, scale: x + (torch.rand_like(x) - 0.5) * 2 * scale,
            "salt_pepper": lambda x, prob: self._add_salt_pepper_noise(x, prob)
        }

        noise_levels = [0.01, 0.02, 0.05, 0.1, 0.15]
        results = {}

        for expert_name, expert in experts.items():
            expert.eval()
            expert_results = {}

            for noise_type, noise_func in noise_types.items():
                noise_results = []

                for noise_level in noise_levels:
                    total_samples = 0
                    correct_predictions = 0

                    with torch.no_grad():
                        for images, labels in dataloader:
                            images = images.to(device)
                            labels = labels.to(device)

                            # Add noise
                            if noise_type == "salt_pepper":
                                noisy_images = noise_func(images, noise_level)
                            else:
                                noisy_images = noise_func(images, noise_level)

                            # Clamp to valid range
                            noisy_images = torch.clamp(noisy_images, 0, 1)

                            # Get predictions
                            output = expert(noisy_images)
                            predictions = (output.confidence_scores > 0.5).float()

                            correct_predictions += (predictions == labels).sum().item()
                            total_samples += labels.size(0)

                    accuracy = correct_predictions / total_samples
                    noise_results.append({
                        "noise_level": noise_level,
                        "accuracy": accuracy
                    })

                expert_results[noise_type] = noise_results

            results[expert_name] = expert_results

        return results

    def _add_salt_pepper_noise(self, images: torch.Tensor, noise_prob: float) -> torch.Tensor:
        """Add salt and pepper noise"""
        noise_mask = torch.rand_like(images) < noise_prob
        salt_mask = torch.rand_like(images) < 0.5

        noisy_images = images.clone()
        noisy_images[noise_mask & salt_mask] = 1.0  # Salt
        noisy_images[noise_mask & ~salt_mask] = 0.0  # Pepper

        return noisy_images

    def _test_compression_robustness(
        self,
        experts: Dict[str, BaseExpert],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Test robustness to JPEG compression"""

        # Simplified compression simulation using noise
        compression_levels = [95, 90, 80, 70, 60, 50]
        results = {}

        for expert_name, expert in experts.items():
            expert.eval()
            compression_results = []

            for quality in compression_levels:
                # Simulate compression artifacts with noise
                noise_std = (100 - quality) / 1000.0  # Simple approximation

                total_samples = 0
                correct_predictions = 0

                with torch.no_grad():
                    for images, labels in dataloader:
                        images = images.to(device)
                        labels = labels.to(device)

                        # Simulate compression artifacts
                        compressed_images = images + torch.randn_like(images) * noise_std
                        compressed_images = torch.clamp(compressed_images, 0, 1)

                        # Get predictions
                        output = expert(compressed_images)
                        predictions = (output.confidence_scores > 0.5).float()

                        correct_predictions += (predictions == labels).sum().item()
                        total_samples += labels.size(0)

                accuracy = correct_predictions / total_samples
                compression_results.append({
                    "quality": quality,
                    "accuracy": accuracy
                })

            results[expert_name] = compression_results

        return results

    def _test_resolution_robustness(
        self,
        experts: Dict[str, BaseExpert],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Test robustness to different resolutions"""

        resolution_scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # Relative to original
        results = {}

        for expert_name, expert in experts.items():
            expert.eval()
            resolution_results = []

            for scale in resolution_scales:
                total_samples = 0
                correct_predictions = 0

                with torch.no_grad():
                    for images, labels in dataloader:
                        images = images.to(device)
                        labels = labels.to(device)

                        # Resize images
                        if scale != 1.0:
                            original_size = images.shape[-2:]
                            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))

                            resized_images = torch.nn.functional.interpolate(
                                images, size=new_size, mode='bilinear', align_corners=False
                            )

                            # Resize back to original size
                            resized_images = torch.nn.functional.interpolate(
                                resized_images, size=original_size, mode='bilinear', align_corners=False
                            )
                        else:
                            resized_images = images

                        # Get predictions
                        output = expert(resized_images)
                        predictions = (output.confidence_scores > 0.5).float()

                        correct_predictions += (predictions == labels).sum().item()
                        total_samples += labels.size(0)

                accuracy = correct_predictions / total_samples
                resolution_results.append({
                    "scale": scale,
                    "accuracy": accuracy
                })

            results[expert_name] = resolution_results

        return results

    def _test_adversarial_robustness(
        self,
        experts: Dict[str, BaseExpert],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Any]:
        """Test basic adversarial robustness using FGSM"""

        epsilon_values = [0.001, 0.01, 0.03, 0.05, 0.1]
        results = {}

        for expert_name, expert in experts.items():
            expert.eval()
            adversarial_results = []

            for epsilon in epsilon_values:
                total_samples = 0
                correct_predictions = 0

                for images, labels in dataloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # Generate adversarial examples using FGSM
                    adversarial_images = self._fgsm_attack(expert, images, labels, epsilon)

                    with torch.no_grad():
                        output = expert(adversarial_images)
                        predictions = (output.confidence_scores > 0.5).float()

                        correct_predictions += (predictions == labels).sum().item()
                        total_samples += labels.size(0)

                    # Limit samples for efficiency
                    if total_samples >= 1000:
                        break

                accuracy = correct_predictions / total_samples
                adversarial_results.append({
                    "epsilon": epsilon,
                    "accuracy": accuracy
                })

            results[expert_name] = adversarial_results

        return results

    def _fgsm_attack(
        self,
        expert: BaseExpert,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack"""

        images.requires_grad_(True)

        output = expert(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output.confidence_scores.squeeze(), labels.float()
        )

        # Calculate gradients
        expert.zero_grad()
        loss.backward()

        # Create adversarial examples
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()

        adversarial_images = images + epsilon * sign_data_grad
        adversarial_images = torch.clamp(adversarial_images, 0, 1)

        return adversarial_images.detach()


class FairnessValidator:
    """Validate fairness and bias in expert predictions"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_fairness(
        self,
        experts: Dict[str, BaseExpert],
        test_dataloader: torch.utils.data.DataLoader,
        demographic_attributes: Optional[Dict[str, np.ndarray]],
        device: torch.device
    ) -> Dict[str, Any]:
        """Validate fairness across demographic groups"""

        if demographic_attributes is None:
            return {"warning": "No demographic attributes provided for fairness analysis"}

        results = {
            "demographic_parity": {},
            "equalized_odds": {},
            "calibration": {},
            "individual_fairness": {}
        }

        # Get predictions for all experts
        expert_predictions = {}
        all_targets = []

        for expert_name, expert in experts.items():
            expert.eval()
            predictions = []
            targets = []

            with torch.no_grad():
                for images, labels in test_dataloader:
                    images = images.to(device)
                    output = expert(images)
                    predictions.extend(output.confidence_scores.cpu().numpy())
                    targets.extend(labels.cpu().numpy())

            expert_predictions[expert_name] = np.array(predictions)
            if not all_targets:  # Only set once
                all_targets = np.array(targets)

        # Analyze fairness for each demographic attribute
        for attr_name, attr_values in demographic_attributes.items():
            attr_results = {}

            for expert_name, predictions in expert_predictions.items():
                expert_fairness = self._analyze_expert_fairness(
                    predictions, all_targets, attr_values, attr_name
                )
                attr_results[expert_name] = expert_fairness

            results[attr_name] = attr_results

        return results

    def _analyze_expert_fairness(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        demographic_attr: np.ndarray,
        attr_name: str
    ) -> Dict[str, Any]:
        """Analyze fairness for a single expert"""

        unique_groups = np.unique(demographic_attr)
        group_metrics = {}

        # Calculate metrics for each demographic group
        for group in unique_groups:
            group_mask = demographic_attr == group
            group_preds = predictions[group_mask]
            group_targets = targets[group_mask]

            if len(group_targets) > 0:
                binary_preds = (group_preds > 0.5).astype(int)

                metrics = {
                    "sample_size": len(group_targets),
                    "positive_rate": np.mean(binary_preds),
                    "true_positive_rate": np.mean(binary_preds[group_targets == 1]) if np.sum(group_targets) > 0 else 0,
                    "false_positive_rate": np.mean(binary_preds[group_targets == 0]) if np.sum(group_targets == 0) > 0 else 0,
                    "auc": roc_auc_score(group_targets, group_preds) if len(np.unique(group_targets)) > 1 else 0.5
                }

                group_metrics[f"group_{group}"] = metrics

        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(group_metrics)

        return {
            "group_metrics": group_metrics,
            "fairness_metrics": fairness_metrics
        }

    def _calculate_fairness_metrics(self, group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate fairness metrics from group statistics"""

        groups = list(group_metrics.keys())
        if len(groups) < 2:
            return {"warning": "Need at least 2 groups for fairness analysis"}

        # Demographic parity (difference in positive rates)
        positive_rates = [group_metrics[g]["positive_rate"] for g in groups]
        demographic_parity_diff = max(positive_rates) - min(positive_rates)

        # Equalized odds (difference in TPR and FPR)
        tpr_values = [group_metrics[g]["true_positive_rate"] for g in groups]
        fpr_values = [group_metrics[g]["false_positive_rate"] for g in groups]

        tpr_diff = max(tpr_values) - min(tpr_values)
        fpr_diff = max(fpr_values) - min(fpr_values)
        equalized_odds_diff = max(tpr_diff, fpr_diff)

        # AUC difference
        auc_values = [group_metrics[g]["auc"] for g in groups]
        auc_diff = max(auc_values) - min(auc_values)

        return {
            "demographic_parity_difference": demographic_parity_diff,
            "equalized_odds_difference": equalized_odds_diff,
            "auc_difference": auc_diff,
            "max_group_size_ratio": max([group_metrics[g]["sample_size"] for g in groups]) /
                                   min([group_metrics[g]["sample_size"] for g in groups])
        }


class HeterogeneousSystemValidator:
    """Main validator for heterogeneous expert systems"""

    def __init__(self, config_path: Optional[str] = None, output_dir: str = "validation_results"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize component validators
        self.statistical_validator = StatisticalValidator(self.config)
        self.robustness_validator = RobustnessValidator(self.config)
        self.fairness_validator = FairnessValidator(self.config)

        # Setup logging
        self._setup_logging()

    def _load_config(self, config_path: Optional[str]) -> ValidationConfig:
        """Load validation configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return ValidationConfig(**config_dict)
        else:
            return ValidationConfig()

    def _setup_logging(self):
        """Setup logging for validation process"""
        log_file = self.output_dir / "validation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_heterogeneous_system(
        self,
        experts: Dict[str, BaseExpert],
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        baseline_predictions: Optional[np.ndarray] = None,
        demographic_attributes: Optional[Dict[str, np.ndarray]] = None
    ) -> ValidationResult:
        """Comprehensive validation of heterogeneous expert system"""

        self.logger.info("Starting comprehensive heterogeneous system validation")

        # Individual expert validation
        individual_results = self._validate_individual_experts(experts, test_dataloader, device)

        # Get predictions for ensemble validation
        expert_predictions, targets = self._get_all_predictions(experts, test_dataloader, device)

        # Ensemble validation
        ensemble_results = self._validate_ensemble_performance(expert_predictions, targets)

        # Statistical validation
        statistical_tests = self.statistical_validator.validate_statistical_significance(
            expert_predictions, targets, baseline_predictions
        )

        # Robustness validation
        robustness_analysis = {}
        if self.config.enable_robustness_testing:
            robustness_analysis = self.robustness_validator.validate_robustness(
                experts, test_dataloader, device
            )

        # Fairness validation
        fairness_analysis = {}
        if self.config.enable_fairness_validation and demographic_attributes:
            fairness_analysis = self.fairness_validator.validate_fairness(
                experts, test_dataloader, demographic_attributes, device
            )

        # Performance breakdown
        performance_breakdown = self._calculate_performance_breakdown(
            individual_results, ensemble_results, expert_predictions, targets
        )

        # Failure analysis
        failure_analysis = self._analyze_failures(expert_predictions, targets)

        # Overall system validation
        system_passed = self._evaluate_system_pass_criteria(
            individual_results, ensemble_results, statistical_tests
        )

        # Generate recommendations
        recommendations = self._generate_validation_recommendations(
            individual_results, ensemble_results, statistical_tests,
            robustness_analysis, fairness_analysis
        )

        # Create comprehensive result
        result = ValidationResult(
            system_passed=system_passed,
            individual_expert_results=individual_results,
            ensemble_results=ensemble_results,
            statistical_tests=statistical_tests,
            robustness_analysis=robustness_analysis,
            fairness_analysis=fairness_analysis,
            performance_breakdown=performance_breakdown,
            failure_analysis=failure_analysis,
            validation_recommendations=recommendations["validation"],
            improvement_suggestions=recommendations["improvements"],
            validation_timestamp=pd.Timestamp.now().isoformat(),
            config_used=self.config
        )

        # Save results
        self._save_validation_results(result)

        # Generate report
        self._generate_validation_report(result)

        self.logger.info(f"Validation completed. System passed: {system_passed}")

        return result

    def _validate_individual_experts(
        self,
        experts: Dict[str, BaseExpert],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, Dict[str, Any]]:
        """Validate individual expert performance"""

        results = {}

        for expert_name, expert in experts.items():
            self.logger.info(f"Validating expert: {expert_name}")

            expert.eval()
            predictions = []
            targets = []
            inference_times = []

            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    start_time = time.time()
                    output = expert(images)
                    inference_time = time.time() - start_time

                    predictions.extend(output.confidence_scores.cpu().numpy())
                    targets.extend(labels.cpu().numpy())
                    inference_times.append(inference_time)

            predictions = np.array(predictions)
            targets = np.array(targets)

            # Calculate metrics
            binary_preds = (predictions > 0.5).astype(int)
            auc = roc_auc_score(targets, predictions)
            f1 = f1_score(targets, binary_preds)
            precision = precision_score(targets, binary_preds)
            recall = recall_score(targets, binary_preds)

            # Performance validation
            passed_auc = auc >= self.config.min_individual_auc
            passed_inference_time = np.mean(inference_times) <= self.config.max_inference_time_ms / 1000.0

            results[expert_name] = {
                "metrics": {
                    "auc": auc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "mean_inference_time": np.mean(inference_times),
                    "std_inference_time": np.std(inference_times)
                },
                "validation": {
                    "passed_auc_threshold": passed_auc,
                    "passed_inference_time": passed_inference_time,
                    "overall_passed": passed_auc and passed_inference_time
                },
                "predictions": predictions,
                "targets": targets
            }

        return results

    def _get_all_predictions(
        self,
        experts: Dict[str, BaseExpert],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Get predictions from all experts"""

        expert_predictions = {}
        targets = None

        for expert_name, expert in experts.items():
            expert.eval()
            predictions = []
            expert_targets = []

            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(device)
                    output = expert(images)
                    predictions.extend(output.confidence_scores.cpu().numpy())
                    expert_targets.extend(labels.cpu().numpy())

            expert_predictions[expert_name] = np.array(predictions)
            if targets is None:
                targets = np.array(expert_targets)

        return expert_predictions, targets

    def _validate_ensemble_performance(
        self,
        expert_predictions: Dict[str, np.ndarray],
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Validate ensemble performance"""

        ensemble_strategies = {
            "simple_average": lambda preds: np.mean(list(preds.values()), axis=0),
            "weighted_average": lambda preds: 0.6 * list(preds.values())[0] + 0.4 * list(preds.values())[1] if len(preds) == 2 else np.mean(list(preds.values()), axis=0),
            "max_confidence": lambda preds: np.max(list(preds.values()), axis=0)
        }

        ensemble_results = {}

        for strategy_name, strategy_func in ensemble_strategies.items():
            ensemble_preds = strategy_func(expert_predictions)
            ensemble_binary = (ensemble_preds > 0.5).astype(int)

            auc = roc_auc_score(targets, ensemble_preds)
            f1 = f1_score(targets, ensemble_binary)

            passed_auc = auc >= self.config.min_ensemble_auc

            ensemble_results[strategy_name] = {
                "auc": auc,
                "f1": f1,
                "passed_threshold": passed_auc,
                "predictions": ensemble_preds
            }

        return ensemble_results

    def _calculate_performance_breakdown(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        ensemble_results: Dict[str, Any],
        expert_predictions: Dict[str, np.ndarray],
        targets: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate detailed performance breakdown"""

        breakdown = {
            "individual_experts": {},
            "ensemble_strategies": {},
            "complementarity": {}
        }

        # Individual expert breakdown
        for expert_name, results in individual_results.items():
            breakdown["individual_experts"][expert_name] = results["metrics"]

        # Ensemble strategy breakdown
        for strategy_name, results in ensemble_results.items():
            breakdown["ensemble_strategies"][strategy_name] = {
                "auc": results["auc"],
                "f1": results["f1"]
            }

        # Complementarity analysis
        if len(expert_predictions) == 2:
            expert_names = list(expert_predictions.keys())
            preds1, preds2 = expert_predictions[expert_names[0]], expert_predictions[expert_names[1]]

            correlation = np.corrcoef(preds1, preds2)[0, 1]

            breakdown["complementarity"] = {
                "correlation": correlation,
                "low_correlation": abs(correlation) < self.config.max_correlation_threshold
            }

        return breakdown

    def _analyze_failures(
        self,
        expert_predictions: Dict[str, np.ndarray],
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze failure patterns"""

        failure_analysis = {}

        for expert_name, predictions in expert_predictions.items():
            binary_preds = (predictions > 0.5).astype(int)
            errors = binary_preds != targets

            false_positives = (binary_preds == 1) & (targets == 0)
            false_negatives = (binary_preds == 0) & (targets == 1)

            failure_analysis[expert_name] = {
                "total_error_rate": np.mean(errors),
                "false_positive_rate": np.mean(false_positives),
                "false_negative_rate": np.mean(false_negatives),
                "error_indices": np.where(errors)[0].tolist()
            }

        return failure_analysis

    def _evaluate_system_pass_criteria(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        ensemble_results: Dict[str, Any],
        statistical_tests: Dict[str, Any]
    ) -> bool:
        """Evaluate if system meets pass criteria"""

        # Check individual expert performance
        individual_passed = all(
            result["validation"]["overall_passed"]
            for result in individual_results.values()
        )

        # Check ensemble performance
        ensemble_passed = any(
            result["passed_threshold"]
            for result in ensemble_results.values()
        )

        # Check statistical significance if baseline provided
        statistical_passed = True  # Default to True if no baseline
        if "significance_tests" in statistical_tests:
            for test_name, test_result in statistical_tests["significance_tests"].items():
                if "vs_baseline" in test_name:
                    statistical_passed = statistical_passed and test_result.get("improvement_significant", True)

        overall_passed = individual_passed and ensemble_passed and statistical_passed

        return overall_passed

    def _generate_validation_recommendations(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        ensemble_results: Dict[str, Any],
        statistical_tests: Dict[str, Any],
        robustness_analysis: Dict[str, Any],
        fairness_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate validation recommendations"""

        validation_recommendations = []
        improvement_suggestions = []

        # Individual expert recommendations
        for expert_name, results in individual_results.items():
            if not results["validation"]["passed_auc_threshold"]:
                validation_recommendations.append(
                    f"{expert_name} expert failed AUC threshold ({results['metrics']['auc']:.3f} < {self.config.min_individual_auc})"
                )
                improvement_suggestions.append(
                    f"Improve {expert_name} expert architecture or training strategy"
                )

            if not results["validation"]["passed_inference_time"]:
                validation_recommendations.append(
                    f"{expert_name} expert exceeded inference time limit"
                )
                improvement_suggestions.append(
                    f"Optimize {expert_name} expert for faster inference"
                )

        # Ensemble recommendations
        best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]["auc"])
        if best_ensemble[1]["passed_threshold"]:
            validation_recommendations.append(
                f"Ensemble validation passed with {best_ensemble[0]} strategy (AUC: {best_ensemble[1]['auc']:.3f})"
            )
        else:
            validation_recommendations.append(
                "All ensemble strategies failed to meet threshold"
            )
            improvement_suggestions.append(
                "Investigate advanced ensemble methods or improve individual experts"
            )

        # Robustness recommendations
        if robustness_analysis:
            validation_recommendations.append("Robustness testing completed")
            improvement_suggestions.append("Review robustness results for deployment considerations")

        # Fairness recommendations
        if fairness_analysis:
            validation_recommendations.append("Fairness analysis completed")
            improvement_suggestions.append("Address any identified fairness issues before deployment")

        return {
            "validation": validation_recommendations,
            "improvements": improvement_suggestions
        }

    def _save_validation_results(self, result: ValidationResult):
        """Save validation results to files"""

        # Save comprehensive JSON results
        result_dict = {
            "system_passed": result.system_passed,
            "individual_expert_results": {
                k: {**v, "predictions": v["predictions"].tolist(), "targets": v["targets"].tolist()}
                for k, v in result.individual_expert_results.items()
            },
            "ensemble_results": {
                k: {**v, "predictions": v["predictions"].tolist() if "predictions" in v else []}
                for k, v in result.ensemble_results.items()
            },
            "statistical_tests": result.statistical_tests,
            "robustness_analysis": result.robustness_analysis,
            "fairness_analysis": result.fairness_analysis,
            "performance_breakdown": result.performance_breakdown,
            "failure_analysis": result.failure_analysis,
            "validation_recommendations": result.validation_recommendations,
            "improvement_suggestions": result.improvement_suggestions,
            "validation_timestamp": result.validation_timestamp,
            "config_used": result.config_used.__dict__
        }

        with open(self.output_dir / "validation_results.json", 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

    def _generate_validation_report(self, result: ValidationResult):
        """Generate comprehensive validation report"""

        report_lines = [
            "# Heterogeneous Expert System Validation Report",
            "",
            f"**Validation Date**: {result.validation_timestamp}",
            f"**Overall System Status**: {'✅ PASSED' if result.system_passed else '❌ FAILED'}",
            "",
            "## Executive Summary",
            ""
        ]

        # Add key findings
        if result.system_passed:
            report_lines.append("The heterogeneous expert system has successfully passed all validation criteria.")
        else:
            report_lines.append("The heterogeneous expert system has failed one or more validation criteria.")

        report_lines.extend([
            "",
            "## Individual Expert Performance",
            ""
        ])

        # Individual expert results table
        report_lines.extend([
            "| Expert | AUC | F1 | Precision | Recall | Inference Time (s) | Status |",
            "|--------|-----|----|-----------|---------|--------------------|--------|"
        ])

        for expert_name, results in result.individual_expert_results.items():
            metrics = results["metrics"]
            status = "✅ Pass" if results["validation"]["overall_passed"] else "❌ Fail"
            report_lines.append(
                f"| {expert_name} | {metrics['auc']:.3f} | {metrics['f1']:.3f} | "
                f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                f"{metrics['mean_inference_time']:.4f} | {status} |"
            )

        # Ensemble results
        report_lines.extend([
            "",
            "## Ensemble Performance",
            "",
            "| Strategy | AUC | F1 | Status |",
            "|----------|-----|----|---------",
        ])

        for strategy, results in result.ensemble_results.items():
            status = "✅ Pass" if results["passed_threshold"] else "❌ Fail"
            report_lines.append(
                f"| {strategy} | {results['auc']:.3f} | {results['f1']:.3f} | {status} |"
            )

        # Recommendations
        report_lines.extend([
            "",
            "## Validation Recommendations",
            ""
        ])

        for i, rec in enumerate(result.validation_recommendations, 1):
            report_lines.append(f"{i}. {rec}")

        report_lines.extend([
            "",
            "## Improvement Suggestions",
            ""
        ])

        for i, suggestion in enumerate(result.improvement_suggestions, 1):
            report_lines.append(f"{i}. {suggestion}")

        # Save report
        with open(self.output_dir / "validation_report.md", 'w') as f:
            f.write('\n'.join(report_lines))


def main():
    """Example usage of heterogeneous validation tools"""

    config = ValidationConfig(
        min_individual_auc=0.88,
        min_ensemble_auc=0.92,
        enable_robustness_testing=True,
        enable_fairness_validation=True
    )

    validator = HeterogeneousSystemValidator()
    print("Heterogeneous System Validator initialized")
    print("Ready for comprehensive validation of expert systems")


if __name__ == "__main__":
    main()