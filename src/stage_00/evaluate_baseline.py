#!/usr/bin/env python3
"""
AWARE-NET Stage 0: Baseline Model Comprehensive Evaluation
Multi-dataset evaluation with failure analysis and academic reporting
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stage_00.baseline_model import EfficientNetV2B3Baseline
from stage_00.dataset import CelebDFDataset, create_data_loaders
from utils.metrics import AcademicMetrics, MetricResult
from utils.calibration_tools import CalibrationAnalyzer, CalibrationResult
from utils.visualization import AcademicVisualizer
from utils.experiment_utils import ExperimentManager, set_reproducible_mode
from utils.dataset_config import DatasetConfig

@dataclass
class DatasetEvaluationResult:
    """Container for single dataset evaluation results"""
    dataset_name: str
    n_samples: int
    n_real: int
    n_fake: int
    
    # Performance metrics
    auc_roc: MetricResult
    accuracy: MetricResult
    f1_score: MetricResult
    precision: MetricResult
    recall: MetricResult
    
    # Calibration metrics
    calibration: CalibrationResult
    
    # Per-class performance
    real_precision: float
    real_recall: float
    fake_precision: float
    fake_recall: float
    
    # Failure analysis
    false_positives: List[str]  # Real images classified as fake
    false_negatives: List[str]  # Fake images classified as real
    high_confidence_errors: List[Tuple[str, float, int, int]]  # (path, confidence, true_label, pred_label)

@dataclass
class CrossDatasetEvaluationResult:
    """Container for cross-dataset evaluation results"""
    train_dataset: str
    test_datasets: List[str]
    per_dataset_results: Dict[str, DatasetEvaluationResult]
    
    # Cross-dataset metrics
    average_auc: float
    auc_std: float
    performance_variance: float
    generalization_gap: float  # Performance drop from validation to test datasets
    
    # Statistical analysis
    dataset_comparison: Dict[str, Any]
    significance_tests: Dict[str, Any]

class BaselineEvaluator:
    """
    Comprehensive baseline model evaluation framework
    
    Features:
    - Multi-dataset evaluation with statistical rigor
    - Failure case analysis and visualization
    - Cross-dataset generalization assessment
    - Calibration analysis and improvement
    - Academic-grade reporting with LaTeX output
    - Performance benchmarking against Stage-Gate criteria
    """
    
    def __init__(self,
                 model_path: str,
                 config_path: str,
                 output_dir: str = "results/baseline_evaluation",
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 num_workers: int = 4):
        """
        Initialize baseline evaluator
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to dataset configuration
            output_dir: Directory for evaluation results
            device: Computing device ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize evaluation tools
        self.metrics = AcademicMetrics()
        self.calibration = CalibrationAnalyzer()
        self.visualizer = AcademicVisualizer()
        
        # Load model
        self.model = self._load_model()
        
        # Load dataset configuration
        self.dataset_config = self._load_dataset_config()
        
        # Setup reproducibility
        set_reproducible_mode(42)
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained baseline model"""
        print(f"Loading model from {self.model_path}")
        
        # Initialize model architecture for BCE
        model = EfficientNetV2B3Baseline(
            num_classes=1,  # Changed to 1 for true BCE
            pretrained=False,
            dropout_rate=0.2
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def _load_dataset_config(self) -> Dict[str, Any]:
        """Load dataset configuration"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def evaluate_single_dataset(self, 
                              manifest_path: str,
                              dataset_name: str,
                              save_predictions: bool = True) -> DatasetEvaluationResult:
        """
        Evaluate model on a single dataset
        
        Args:
            manifest_path: Path to dataset manifest
            dataset_name: Name of the dataset
            save_predictions: Whether to save detailed predictions
            
        Returns:
            DatasetEvaluationResult with comprehensive metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = CelebDFDataset(
            manifest_path=manifest_path,
            root_path=self.dataset_config.get("root_path", "."),
            image_size=self.dataset_config.get("image_size", 256),
            augmentation=False,  # No augmentation during evaluation
            normalize=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        
        # Collect predictions
        all_predictions = []
        all_probabilities = []
        all_logits = []
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                paths = batch['path']
                
                # Forward pass
                outputs = self.model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # For BCE model, use sigmoid activation
                probabilities = torch.sigmoid(logits).squeeze()  # Remove extra dimension
                predictions = (probabilities > 0.5).long()
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_logits.extend(logits.squeeze().cpu().numpy())  # Raw logits for BCE
                all_labels.extend(labels.cpu().numpy())
                all_paths.extend(paths)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        logits = np.array(all_logits)
        labels = np.array(all_labels)
        paths = np.array(all_paths)
        
        # Calculate dataset statistics
        n_samples = len(labels)
        n_real = int((labels == 0).sum())
        n_fake = int((labels == 1).sum())
        
        print(f"Real samples: {n_real}, Fake samples: {n_fake}")
        
        # Calculate performance metrics with confidence intervals
        auc_roc = self.metrics.calculate_auc_with_ci(labels, probabilities)
        accuracy = self.metrics.calculate_accuracy_with_ci(labels, predictions)
        f1_score = self.metrics.calculate_f1_with_ci(labels, predictions)
        precision = self.metrics.calculate_precision_with_ci(labels, predictions)
        recall = self.metrics.calculate_recall_with_ci(labels, predictions)
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        per_class_precision, per_class_recall, _, _ = precision_recall_fscore_support(
            labels, predictions, labels=[0, 1]
        )
        
        # Calibration analysis
        calibration_result = self.calibration.calculate_ece_mce(labels, probabilities)
        
        # Failure analysis
        false_positives = []  # Real images classified as fake
        false_negatives = []  # Fake images classified as real
        high_confidence_errors = []
        
        for i, (true_label, pred_label, prob, path) in enumerate(zip(labels, predictions, probabilities, paths)):
            if true_label != pred_label:
                if true_label == 0 and pred_label == 1:  # Real classified as fake
                    false_positives.append(str(path))
                elif true_label == 1 and pred_label == 0:  # Fake classified as real
                    false_negatives.append(str(path))
                
                # High confidence errors (>0.8 confidence)
                if prob > 0.8 or prob < 0.2:
                    confidence = max(prob, 1 - prob)
                    high_confidence_errors.append((str(path), confidence, int(true_label), int(pred_label)))
        
        # Sort high confidence errors by confidence
        high_confidence_errors.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Performance Summary:")
        print(f"  AUC-ROC: {auc_roc.value:.4f} ± {(auc_roc.confidence_interval[1] - auc_roc.confidence_interval[0])/2:.4f}")
        print(f"  Accuracy: {accuracy.value:.4f} ± {(accuracy.confidence_interval[1] - accuracy.confidence_interval[0])/2:.4f}")
        print(f"  F1-Score: {f1_score.value:.4f}")
        print(f"  ECE: {calibration_result.ece:.4f}")
        print(f"  False Positives: {len(false_positives)}")
        print(f"  False Negatives: {len(false_negatives)}")
        print(f"  High Confidence Errors: {len(high_confidence_errors)}")
        
        # Save detailed predictions if requested
        if save_predictions:
            predictions_df = pd.DataFrame({
                'path': paths,
                'true_label': labels,
                'predicted_label': predictions,
                'probability_fake': probabilities,
                'logit_fake': logits,
                'correct': labels == predictions
            })
            
            predictions_file = self.output_dir / f"{dataset_name}_predictions.csv"
            predictions_df.to_csv(predictions_file, index=False)
            print(f"Detailed predictions saved to: {predictions_file}")
        
        return DatasetEvaluationResult(
            dataset_name=dataset_name,
            n_samples=n_samples,
            n_real=n_real,
            n_fake=n_fake,
            auc_roc=auc_roc,
            accuracy=accuracy,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            calibration=calibration_result,
            real_precision=per_class_precision[0],
            real_recall=per_class_recall[0],
            fake_precision=per_class_precision[1],
            fake_recall=per_class_recall[1],
            false_positives=false_positives[:50],  # Limit to first 50
            false_negatives=false_negatives[:50],
            high_confidence_errors=high_confidence_errors[:20]  # Limit to top 20
        )
    
    def evaluate_cross_dataset(self, 
                             datasets: Dict[str, str],
                             train_dataset_name: str) -> CrossDatasetEvaluationResult:
        """
        Evaluate model across multiple datasets
        
        Args:
            datasets: Dictionary of dataset_name -> manifest_path
            train_dataset_name: Name of the dataset used for training
            
        Returns:
            CrossDatasetEvaluationResult with cross-dataset analysis
        """
        print(f"\n{'='*80}")
        print(f"Cross-Dataset Evaluation")
        print(f"Training dataset: {train_dataset_name}")
        print(f"Test datasets: {list(datasets.keys())}")
        print(f"{'='*80}")
        
        # Evaluate each dataset
        per_dataset_results = {}
        for dataset_name, manifest_path in datasets.items():
            result = self.evaluate_single_dataset(manifest_path, dataset_name)
            per_dataset_results[dataset_name] = result
        
        # Calculate cross-dataset statistics
        auc_values = [result.auc_roc.value for result in per_dataset_results.values()]
        average_auc = np.mean(auc_values)
        auc_std = np.std(auc_values)
        performance_variance = np.var(auc_values)
        
        # Calculate generalization gap (performance drop from train to test datasets)
        if train_dataset_name in per_dataset_results:
            train_auc = per_dataset_results[train_dataset_name].auc_roc.value
            test_aucs = [result.auc_roc.value for name, result in per_dataset_results.items() 
                        if name != train_dataset_name]
            generalization_gap = train_auc - np.mean(test_aucs) if test_aucs else 0.0
        else:
            generalization_gap = 0.0
        
        # Statistical comparison between datasets
        dataset_comparison = self._compare_datasets(per_dataset_results)
        
        # Significance tests
        significance_tests = self._perform_significance_tests(per_dataset_results)
        
        print(f"\nCross-Dataset Summary:")
        print(f"  Average AUC: {average_auc:.4f} ± {auc_std:.4f}")
        print(f"  Performance Variance: {performance_variance:.6f}")
        print(f"  Generalization Gap: {generalization_gap:.4f}")
        
        return CrossDatasetEvaluationResult(
            train_dataset=train_dataset_name,
            test_datasets=list(datasets.keys()),
            per_dataset_results=per_dataset_results,
            average_auc=average_auc,
            auc_std=auc_std,
            performance_variance=performance_variance,
            generalization_gap=generalization_gap,
            dataset_comparison=dataset_comparison,
            significance_tests=significance_tests
        )
    
    def _compare_datasets(self, results: Dict[str, DatasetEvaluationResult]) -> Dict[str, Any]:
        """Compare performance across datasets"""
        comparison = {
            'dataset_names': list(results.keys()),
            'auc_values': [result.auc_roc.value for result in results.values()],
            'accuracy_values': [result.accuracy.value for result in results.values()],
            'ece_values': [result.calibration.ece for result in results.values()],
            'sample_sizes': [result.n_samples for result in results.values()],
            'class_balance': [result.n_fake / result.n_samples for result in results.values()]
        }
        
        # Find best and worst performing datasets
        auc_values = comparison['auc_values']
        best_idx = np.argmax(auc_values)
        worst_idx = np.argmin(auc_values)
        
        comparison['best_dataset'] = {
            'name': comparison['dataset_names'][best_idx],
            'auc': auc_values[best_idx]
        }
        comparison['worst_dataset'] = {
            'name': comparison['dataset_names'][worst_idx],
            'auc': auc_values[worst_idx]
        }
        
        return comparison
    
    def _perform_significance_tests(self, results: Dict[str, DatasetEvaluationResult]) -> Dict[str, Any]:
        """Perform statistical significance tests between datasets"""
        # This is a simplified version - in practice, you'd need access to the raw predictions
        # for proper statistical testing
        
        dataset_names = list(results.keys())
        auc_values = [result.auc_roc.value for result in results.values()]
        
        # Pairwise comparisons (simplified)
        pairwise_comparisons = []
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i < j:  # Only compare each pair once
                    auc1, auc2 = auc_values[i], auc_values[j]
                    # Simplified significance test based on confidence intervals
                    ci1 = results[name1].auc_roc.confidence_interval
                    ci2 = results[name2].auc_roc.confidence_interval
                    
                    # Check if confidence intervals overlap
                    overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
                    
                    pairwise_comparisons.append({
                        'dataset1': name1,
                        'dataset2': name2,
                        'auc1': auc1,
                        'auc2': auc2,
                        'difference': abs(auc1 - auc2),
                        'ci_overlap': overlap,
                        'likely_significant': not overlap
                    })
        
        return {
            'pairwise_comparisons': pairwise_comparisons,
            'n_comparisons': len(pairwise_comparisons)
        }
    
    def generate_failure_analysis_report(self, 
                                       results: Dict[str, DatasetEvaluationResult],
                                       save_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive failure analysis report
        
        Args:
            results: Dictionary of dataset results
            save_visualizations: Whether to save visualization plots
            
        Returns:
            Dictionary containing failure analysis insights
        """
        print(f"\n{'='*60}")
        print("Generating Failure Analysis Report")
        print(f"{'='*60}")
        
        report = {
            'summary': {},
            'per_dataset_analysis': {},
            'common_failure_patterns': [],
            'recommendations': []
        }
        
        # Overall failure statistics
        total_samples = sum(result.n_samples for result in results.values())
        total_fp = sum(len(result.false_positives) for result in results.values())
        total_fn = sum(len(result.false_negatives) for result in results.values())
        total_high_conf_errors = sum(len(result.high_confidence_errors) for result in results.values())
        
        report['summary'] = {
            'total_samples': total_samples,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'total_high_confidence_errors': total_high_conf_errors,
            'false_positive_rate': total_fp / total_samples,
            'false_negative_rate': total_fn / total_samples
        }
        
        # Per-dataset analysis
        for dataset_name, result in results.items():
            fp_rate = len(result.false_positives) / result.n_real if result.n_real > 0 else 0
            fn_rate = len(result.false_negatives) / result.n_fake if result.n_fake > 0 else 0
            
            report['per_dataset_analysis'][dataset_name] = {
                'false_positive_rate': fp_rate,
                'false_negative_rate': fn_rate,
                'high_confidence_error_rate': len(result.high_confidence_errors) / result.n_samples,
                'calibration_quality': 'Good' if result.calibration.ece < 0.1 else 'Poor',
                'worst_errors': result.high_confidence_errors[:5]  # Top 5 worst errors
            }
        
        # Identify common failure patterns
        all_high_conf_errors = []
        for result in results.values():
            all_high_conf_errors.extend(result.high_confidence_errors)
        
        # Sort by confidence and analyze top errors
        all_high_conf_errors.sort(key=lambda x: x[1], reverse=True)
        
        # Simple pattern analysis (could be more sophisticated)
        fp_high_conf = [err for err in all_high_conf_errors if err[2] == 0 and err[3] == 1]  # Real -> Fake
        fn_high_conf = [err for err in all_high_conf_errors if err[2] == 1 and err[3] == 0]  # Fake -> Real
        
        report['common_failure_patterns'] = [
            f"High confidence false positives: {len(fp_high_conf)} cases",
            f"High confidence false negatives: {len(fn_high_conf)} cases",
            f"Most confident error: {all_high_conf_errors[0][1]:.4f}" if all_high_conf_errors else "No high confidence errors"
        ]
        
        # Generate recommendations
        recommendations = []
        if total_fp > total_fn:
            recommendations.append("Model tends to over-detect fakes. Consider adjusting threshold or improving real sample training.")
        elif total_fn > total_fp:
            recommendations.append("Model tends to under-detect fakes. Consider more diverse fake samples or harder negative mining.")
        
        avg_ece = np.mean([result.calibration.ece for result in results.values()])
        if avg_ece > 0.1:
            recommendations.append(f"Poor calibration detected (avg ECE: {avg_ece:.4f}). Consider temperature scaling or calibration techniques.")
        
        if report['summary']['false_positive_rate'] > 0.1:
            recommendations.append("High false positive rate. Review real samples that are frequently misclassified.")
            
        if report['summary']['false_negative_rate'] > 0.1:
            recommendations.append("High false negative rate. Consider incorporating more challenging fake samples in training.")
        
        report['recommendations'] = recommendations
        
        # Save visualizations if requested
        if save_visualizations:
            self._create_failure_visualizations(results, report)
        
        print("Failure analysis completed")
        return report
    
    def _create_failure_visualizations(self, 
                                     results: Dict[str, DatasetEvaluationResult],
                                     report: Dict[str, Any]):
        """Create visualizations for failure analysis"""
        
        # Error rate comparison across datasets
        plt.figure(figsize=(12, 8))
        
        datasets = list(results.keys())
        fp_rates = [report['per_dataset_analysis'][ds]['false_positive_rate'] for ds in datasets]
        fn_rates = [report['per_dataset_analysis'][ds]['false_negative_rate'] for ds in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        plt.subplot(2, 2, 1)
        plt.bar(x - width/2, fp_rates, width, label='False Positive Rate', alpha=0.8)
        plt.bar(x + width/2, fn_rates, width, label='False Negative Rate', alpha=0.8)
        plt.xlabel('Dataset')
        plt.ylabel('Error Rate')
        plt.title('False Positive vs False Negative Rates')
        plt.xticks(x, datasets, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calibration comparison
        plt.subplot(2, 2, 2)
        ece_values = [result.calibration.ece for result in results.values()]
        mce_values = [result.calibration.mce for result in results.values()]
        
        plt.bar(x - width/2, ece_values, width, label='ECE', alpha=0.8)
        plt.bar(x + width/2, mce_values, width, label='MCE', alpha=0.8)
        plt.xlabel('Dataset')
        plt.ylabel('Calibration Error')
        plt.title('Calibration Quality Comparison')
        plt.xticks(x, datasets, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Performance vs calibration scatter
        plt.subplot(2, 2, 3)
        auc_values = [result.auc_roc.value for result in results.values()]
        plt.scatter(ece_values, auc_values, s=100, alpha=0.7)
        for i, dataset in enumerate(datasets):
            plt.annotate(dataset, (ece_values[i], auc_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        plt.xlabel('Expected Calibration Error')
        plt.ylabel('AUC-ROC')
        plt.title('Performance vs Calibration')
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(2, 2, 4)
        high_conf_error_rates = [report['per_dataset_analysis'][ds]['high_confidence_error_rate'] 
                                for ds in datasets]
        plt.bar(datasets, high_conf_error_rates, alpha=0.8, color='red')
        plt.xlabel('Dataset')
        plt.ylabel('High Confidence Error Rate')
        plt.title('High Confidence Errors by Dataset')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'failure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Failure analysis visualizations saved to: {self.output_dir / 'failure_analysis.png'}")
    
    def generate_academic_report(self, 
                               cross_dataset_result: CrossDatasetEvaluationResult,
                               failure_analysis: Dict[str, Any],
                               format: str = 'markdown') -> str:
        """
        Generate academic-style evaluation report
        
        Args:
            cross_dataset_result: Cross-dataset evaluation results
            failure_analysis: Failure analysis results
            format: Output format ('markdown' or 'latex')
            
        Returns:
            Formatted report string
        """
        print(f"\n{'='*60}")
        print("Generating Academic Report")
        print(f"{'='*60}")
        
        if format == 'latex':
            return self._generate_latex_report(cross_dataset_result, failure_analysis)
        else:
            return self._generate_markdown_report(cross_dataset_result, failure_analysis)
    
    def _generate_markdown_report(self, 
                                cross_dataset_result: CrossDatasetEvaluationResult,
                                failure_analysis: Dict[str, Any]) -> str:
        """Generate markdown format report"""
        
        report = []
        report.append("# AWARE-NET Stage 0: Baseline Model Evaluation Report")
        report.append("")
        report.append(f"**Evaluation Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Training Dataset:** {cross_dataset_result.train_dataset}")
        report.append(f"**Test Datasets:** {', '.join(cross_dataset_result.test_datasets)}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"The EfficientNetV2-B3 baseline model achieved an average AUC-ROC of "
                     f"**{cross_dataset_result.average_auc:.4f} ± {cross_dataset_result.auc_std:.4f}** "
                     f"across {len(cross_dataset_result.test_datasets)} datasets.")
        
        # Stage-Gate Assessment
        stage_gate_pass = cross_dataset_result.average_auc >= 0.88
        report.append("")
        report.append("### Stage-Gate Assessment")
        report.append(f"- **Target AUC ≥ 0.88:** {'✅ PASS' if stage_gate_pass else '❌ FAIL'} "
                     f"({cross_dataset_result.average_auc:.4f})")
        report.append(f"- **Performance Variance < 0.05:** "
                     f"{'✅ PASS' if cross_dataset_result.performance_variance < 0.05 else '❌ FAIL'} "
                     f"({cross_dataset_result.performance_variance:.6f})")
        report.append(f"- **Generalization Gap:** {cross_dataset_result.generalization_gap:.4f}")
        report.append("")
        
        # Detailed Results Table
        report.append("## Detailed Performance Results")
        report.append("")
        report.append("| Dataset | Samples | AUC-ROC | Accuracy | F1-Score | Precision | Recall | ECE |")
        report.append("|---------|---------|---------|----------|----------|-----------|--------|-----|")
        
        for dataset_name, result in cross_dataset_result.per_dataset_results.items():
            report.append(f"| {dataset_name} | {result.n_samples} | "
                         f"{result.auc_roc.value:.4f} | {result.accuracy.value:.4f} | "
                         f"{result.f1_score.value:.4f} | {result.precision.value:.4f} | "
                         f"{result.recall.value:.4f} | {result.calibration.ece:.4f} |")
        
        report.append("")
        
        # Failure Analysis
        report.append("## Failure Analysis")
        report.append("")
        report.append(f"**Total Samples Analyzed:** {failure_analysis['summary']['total_samples']:,}")
        report.append(f"**False Positive Rate:** {failure_analysis['summary']['false_positive_rate']:.4f}")
        report.append(f"**False Negative Rate:** {failure_analysis['summary']['false_negative_rate']:.4f}")
        report.append(f"**High Confidence Errors:** {failure_analysis['summary']['total_high_confidence_errors']}")
        report.append("")
        
        # Common Failure Patterns
        report.append("### Common Failure Patterns")
        for pattern in failure_analysis['common_failure_patterns']:
            report.append(f"- {pattern}")
        report.append("")
        
        # Recommendations
        report.append("### Recommendations for Improvement")
        for rec in failure_analysis['recommendations']:
            report.append(f"- {rec}")
        report.append("")
        
        # Statistical Analysis
        report.append("## Statistical Analysis")
        report.append("")
        best_ds = cross_dataset_result.dataset_comparison['best_dataset']
        worst_ds = cross_dataset_result.dataset_comparison['worst_dataset']
        
        report.append(f"**Best Performing Dataset:** {best_ds['name']} (AUC: {best_ds['auc']:.4f})")
        report.append(f"**Worst Performing Dataset:** {worst_ds['name']} (AUC: {worst_ds['auc']:.4f})")
        report.append("")
        
        # Conclusion
        report.append("## Conclusion")
        report.append("")
        if stage_gate_pass:
            report.append("The baseline model **meets the Stage-Gate criteria** for advancement to Stage 1. "
                         "The model demonstrates consistent performance across multiple datasets with "
                         "acceptable variance and calibration quality.")
        else:
            report.append("The baseline model **does not meet the Stage-Gate criteria**. "
                         "Performance improvements are required before advancing to Stage 1. "
                         "Consider the recommendations provided in the failure analysis section.")
        
        report.append("")
        report.append("---")
        report.append("*Report generated by AWARE-NET Stage 0 Baseline Evaluator*")
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.output_dir / "baseline_evaluation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Academic report saved to: {report_file}")
        return report_text
    
    def _generate_latex_report(self, 
                             cross_dataset_result: CrossDatasetEvaluationResult,
                             failure_analysis: Dict[str, Any]) -> str:
        """Generate LaTeX format report for academic papers"""
        
        latex = []
        latex.append("\\documentclass{article}")
        latex.append("\\usepackage{booktabs}")
        latex.append("\\usepackage{array}")
        latex.append("\\usepackage{multirow}")
        latex.append("\\begin{document}")
        latex.append("")
        
        # Results Table
        latex.append("\\begin{table}[h!]")
        latex.append("\\centering")
        latex.append("\\caption{EfficientNetV2-B3 Baseline Performance Across Datasets}")
        latex.append("\\label{tab:baseline_results}")
        latex.append("\\begin{tabular}{lrrrrrr}")
        latex.append("\\toprule")
        latex.append("Dataset & Samples & AUC-ROC & Accuracy & F1-Score & Precision & Recall \\\\")
        latex.append("\\midrule")
        
        for dataset_name, result in cross_dataset_result.per_dataset_results.items():
            latex.append(f"{dataset_name} & {result.n_samples} & "
                        f"{result.auc_roc.value:.3f} & {result.accuracy.value:.3f} & "
                        f"{result.f1_score.value:.3f} & {result.precision.value:.3f} & "
                        f"{result.recall.value:.3f} \\\\")
        
        latex.append("\\midrule")
        latex.append(f"Average & - & {cross_dataset_result.average_auc:.3f} & - & - & - & - \\\\")
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")
        latex.append("\\end{document}")
        
        latex_text = "\n".join(latex)
        latex_file = self.output_dir / "baseline_results_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_text)
        
        print(f"LaTeX table saved to: {latex_file}")
        return latex_text
    
    def run_comprehensive_evaluation(self, 
                                   datasets: Dict[str, str],
                                   train_dataset_name: str,
                                   generate_report: bool = True) -> Dict[str, Any]:
        """
        Run complete comprehensive evaluation pipeline
        
        Args:
            datasets: Dictionary of dataset_name -> manifest_path
            train_dataset_name: Name of training dataset
            generate_report: Whether to generate academic report
            
        Returns:
            Complete evaluation results dictionary
        """
        print(f"\n{'='*100}")
        print("AWARE-NET Stage 0: Comprehensive Baseline Evaluation")
        print(f"{'='*100}")
        
        # Run cross-dataset evaluation
        cross_dataset_result = self.evaluate_cross_dataset(datasets, train_dataset_name)
        
        # Generate failure analysis
        failure_analysis = self.generate_failure_analysis_report(
            cross_dataset_result.per_dataset_results
        )
        
        # Generate calibration visualizations
        self._generate_calibration_plots(cross_dataset_result.per_dataset_results)
        
        # Generate academic report
        if generate_report:
            markdown_report = self.generate_academic_report(
                cross_dataset_result, failure_analysis, format='markdown'
            )
            latex_report = self.generate_academic_report(
                cross_dataset_result, failure_analysis, format='latex'
            )
        
        # Compile final results
        final_results = {
            'cross_dataset_evaluation': asdict(cross_dataset_result),
            'failure_analysis': failure_analysis,
            'stage_gate_assessment': {
                'meets_auc_requirement': cross_dataset_result.average_auc >= 0.88,
                'meets_variance_requirement': cross_dataset_result.performance_variance < 0.05,
                'average_auc': cross_dataset_result.average_auc,
                'performance_variance': cross_dataset_result.performance_variance,
                'generalization_gap': cross_dataset_result.generalization_gap,
                'overall_pass': (cross_dataset_result.average_auc >= 0.88 and 
                               cross_dataset_result.performance_variance < 0.05)
            },
            'evaluation_metadata': {
                'evaluation_date': datetime.datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'config_path': str(self.config_path),
                'output_directory': str(self.output_dir),
                'device': str(self.device)
            }
        }
        
        # Save complete results
        results_file = self.output_dir / "complete_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n{'='*100}")
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'='*100}")
        print(f"Results saved to: {self.output_dir}")
        print(f"Overall Stage-Gate Status: {'✅ PASS' if final_results['stage_gate_assessment']['overall_pass'] else '❌ FAIL'}")
        print(f"Average AUC: {cross_dataset_result.average_auc:.4f}")
        print(f"Performance Variance: {cross_dataset_result.performance_variance:.6f}")
        
        return final_results
    
    def _generate_calibration_plots(self, results: Dict[str, DatasetEvaluationResult]):
        """Generate calibration plots for all datasets"""
        
        n_datasets = len(results)
        fig, axes = plt.subplots(2, (n_datasets + 1) // 2, figsize=(15, 10))
        if n_datasets == 1:
            axes = [axes]
        axes = axes.flatten() if n_datasets > 2 else axes
        
        for idx, (dataset_name, result) in enumerate(results.items()):
            self.calibration.plot_reliability_diagram(
                result.calibration,
                title=f"{dataset_name} Calibration",
                save_path=None,
                figsize=(6, 4)
            )
            
            # Save individual plots as well
            individual_fig = plt.gcf()
            individual_fig.savefig(
                self.output_dir / f"{dataset_name}_calibration.png",
                dpi=300, bbox_inches='tight'
            )
            plt.close()
        
        print(f"Calibration plots saved to: {self.output_dir}")


def main():
    """Main evaluation script with command line interface"""
    parser = argparse.ArgumentParser(description="AWARE-NET Stage 0 Baseline Evaluation")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to dataset configuration file")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                       help="Dataset manifest paths (format: name=path)")
    parser.add_argument("--train_dataset", type=str, required=True,
                       help="Name of the training dataset")
    parser.add_argument("--output_dir", type=str, default="results/baseline_evaluation",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, auto-detect if None)")
    parser.add_argument("--no_report", action="store_true",
                       help="Skip generating academic report")
    
    args = parser.parse_args()
    
    # Parse datasets argument
    datasets = {}
    for dataset_spec in args.datasets:
        if "=" not in dataset_spec:
            raise ValueError(f"Dataset specification must be in format 'name=path', got: {dataset_spec}")
        name, path = dataset_spec.split("=", 1)
        datasets[name] = path
    
    # Initialize evaluator
    evaluator = BaselineEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        datasets=datasets,
        train_dataset_name=args.train_dataset,
        generate_report=not args.no_report
    )
    
    # Print final summary
    stage_gate_status = results['stage_gate_assessment']
    print(f"\n{'='*80}")
    print("FINAL STAGE-GATE ASSESSMENT")
    print(f"{'='*80}")
    print(f"AUC Requirement (≥0.88): {'✅' if stage_gate_status['meets_auc_requirement'] else '❌'} "
          f"({stage_gate_status['average_auc']:.4f})")
    print(f"Variance Requirement (<0.05): {'✅' if stage_gate_status['meets_variance_requirement'] else '❌'} "
          f"({stage_gate_status['performance_variance']:.6f})")
    print(f"Overall Status: {'✅ READY FOR STAGE 1' if stage_gate_status['overall_pass'] else '❌ IMPROVEMENTS NEEDED'}")
    
    return results


if __name__ == "__main__":
    import datetime
    main()