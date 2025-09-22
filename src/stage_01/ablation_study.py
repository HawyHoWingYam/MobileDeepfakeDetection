"""
AWARE-NET Stage 1: Small-scale Ablation Study

This module implements the critical small-scale validation experiment to
determine whether SupCon loss provides meaningful improvements over BCE
baseline before proceeding with full-scale implementation.

CRITICAL DECISION GATE:
- If SupCon > BCE + 1% AUC → Proceed with caution
- If SupCon > BCE + 3% AUC → Proceed with full implementation
- If SupCon ≤ BCE + 1% AUC → STOP and activate contingency plans

Key Features:
- Fast 10-epoch validation on 1000 samples
- Statistical significance testing
- Feature space analysis (t-SNE visualization)
- Comprehensive comparison metrics
- Decision gate automation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy import stats
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# Local imports
from .supcon_loss import SupConLoss, SupConLossWithLogging
from .mobilenetv4_model import MobileNetV4SupCon
from .balanced_sampler import BalancedBatchSampler

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    n_samples: int = 1000
    n_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42

    # Decision thresholds
    min_improvement_threshold: float = 0.01  # 1% minimum improvement
    target_improvement_threshold: float = 0.03  # 3% target improvement
    statistical_significance_alpha: float = 0.05


@dataclass
class ExperimentResults:
    """Results from a single experiment."""
    model_name: str
    loss_type: str
    final_auc: float
    final_f1: float
    final_accuracy: float
    training_time: float
    convergence_epoch: int
    feature_separation: float
    loss_history: List[float]
    auc_history: List[float]


class QuickDatasetSampler:
    """
    Create balanced small dataset for quick validation.
    """

    @staticmethod
    def sample_balanced_dataset(
        full_dataset,
        n_samples: int = 1000,
        seed: int = 42
    ) -> Tuple[Subset, List[int]]:
        """
        Sample balanced dataset from full dataset.

        Args:
            full_dataset: Full dataset to sample from
            n_samples: Total number of samples to extract
            seed: Random seed for reproducibility

        Returns:
            Subset dataset and corresponding labels
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Get all labels
        all_labels = []
        for i in range(len(full_dataset)):
            _, label = full_dataset[i]
            all_labels.append(label)

        all_labels = np.array(all_labels)
        unique_labels = np.unique(all_labels)

        # Sample equally from each class
        samples_per_class = n_samples // len(unique_labels)
        selected_indices = []

        for label in unique_labels:
            label_indices = np.where(all_labels == label)[0]
            if len(label_indices) >= samples_per_class:
                selected = np.random.choice(
                    label_indices,
                    samples_per_class,
                    replace=False
                )
            else:
                # Sample with replacement if not enough samples
                selected = np.random.choice(
                    label_indices,
                    samples_per_class,
                    replace=True
                )
            selected_indices.extend(selected)

        # Shuffle final selection
        np.random.shuffle(selected_indices)

        # Create subset
        subset = Subset(full_dataset, selected_indices)
        subset_labels = [all_labels[i] for i in selected_indices]

        logger.info(f"Sampled {len(selected_indices)} balanced samples")
        logger.info(f"Class distribution: {np.bincount(subset_labels)}")

        return subset, subset_labels


class FeatureAnalyzer:
    """
    Analyze feature space quality and separation.
    """

    @staticmethod
    def extract_features(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from model."""
        model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)

                # Get features (before final classification)
                features = model.get_features(inputs)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        return np.vstack(all_features), np.concatenate(all_labels)

    @staticmethod
    def calculate_feature_separation(
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Calculate feature space separation metrics."""
        unique_labels = np.unique(labels)

        if len(unique_labels) != 2:
            logger.warning(f"Expected 2 classes, got {len(unique_labels)}")
            return {'separation_score': 0.0}

        # Split features by class
        class_0_features = features[labels == unique_labels[0]]
        class_1_features = features[labels == unique_labels[1]]

        # Calculate centroids
        centroid_0 = np.mean(class_0_features, axis=0)
        centroid_1 = np.mean(class_1_features, axis=0)

        # Inter-class distance (distance between centroids)
        inter_class_distance = np.linalg.norm(centroid_0 - centroid_1)

        # Intra-class distances (average distance within each class)
        intra_class_0 = np.mean([
            np.linalg.norm(f - centroid_0) for f in class_0_features
        ])
        intra_class_1 = np.mean([
            np.linalg.norm(f - centroid_1) for f in class_1_features
        ])
        avg_intra_class = (intra_class_0 + intra_class_1) / 2

        # Separation score: higher is better
        separation_score = inter_class_distance / (avg_intra_class + 1e-8)

        return {
            'separation_score': separation_score,
            'inter_class_distance': inter_class_distance,
            'avg_intra_class_distance': avg_intra_class,
            'class_0_intra_distance': intra_class_0,
            'class_1_intra_distance': intra_class_1,
            'centroid_0': centroid_0,
            'centroid_1': centroid_1
        }

    @staticmethod
    def create_tsne_visualization(
        features: np.ndarray,
        labels: np.ndarray,
        title: str = "Feature Space Visualization",
        save_path: Optional[str] = None
    ) -> None:
        """Create t-SNE visualization of feature space."""
        if len(features) > 1000:
            # Sample for faster t-SNE
            indices = np.random.choice(len(features), 1000, replace=False)
            features = features[indices]
            labels = labels[indices]

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # Create plot
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = ['red', 'blue', 'green', 'orange']

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = 'Authentic' if label == 1 else 'Fake'
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=colors[i % len(colors)],
                label=label_name,
                alpha=0.7,
                s=20
            )

        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"t-SNE plot saved to {save_path}")

        plt.show()


class SmallScaleExperiment:
    """
    Main class for conducting small-scale ablation experiments.
    """

    def __init__(self, config: AblationConfig = None):
        """Initialize experiment with configuration."""
        self.config = config or AblationConfig()
        self.device = torch.device(self.config.device)

        # Set random seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.results = {}

        logger.info(f"SmallScaleExperiment initialized on {self.device}")

    def run_bce_baseline(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> ExperimentResults:
        """Run BCE baseline experiment."""
        logger.info("Running BCE baseline experiment...")

        # Create model with BCE loss
        model = MobileNetV4SupCon(
            pretrained=False,  # Faster for ablation
            use_projection_head=False,
            num_classes=2
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        return self._train_and_evaluate(
            model, train_loader, val_loader,
            criterion, optimizer, "BCE_Baseline", "BCE"
        )

    def run_supcon_experiment(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        temperature: float = 0.07
    ) -> ExperimentResults:
        """Run SupCon experiment."""
        logger.info("Running SupCon experiment...")

        # Create model with SupCon setup
        model = MobileNetV4SupCon(
            pretrained=False,
            use_projection_head=True,
            projection_dim=256,
            num_classes=2
        ).to(self.device)

        # Two-stage training: SupCon + fine-tuning
        supcon_criterion = SupConLoss(temperature=temperature)
        ce_criterion = nn.CrossEntropyLoss()

        # Stage 1: SupCon training on projections
        projection_optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        start_time = time.time()
        loss_history = []
        auc_history = []

        for epoch in range(self.config.n_epochs // 2):  # Half epochs for SupCon
            model.train()
            epoch_loss = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                projection_optimizer.zero_grad()

                # Get projections for SupCon
                projections = model.get_projections(inputs)
                projections = projections.unsqueeze(1)  # Add view dimension

                loss = supcon_criterion(projections, labels)
                loss.backward()
                projection_optimizer.step()

                epoch_loss += loss.item()

            # Validation
            val_auc = self._evaluate_model(model, val_loader)
            loss_history.append(epoch_loss / len(train_loader))
            auc_history.append(val_auc)

            logger.info(f"SupCon Epoch {epoch + 1}: Loss={epoch_loss/len(train_loader):.4f}, AUC={val_auc:.4f}")

        # Stage 2: Fine-tune classification head
        classifier_optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate / 2)

        for epoch in range(self.config.n_epochs // 2, self.config.n_epochs):
            model.train()
            epoch_loss = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                classifier_optimizer.zero_grad()

                logits = model(inputs)
                loss = ce_criterion(logits, labels.long())
                loss.backward()
                classifier_optimizer.step()

                epoch_loss += loss.item()

            # Validation
            val_auc = self._evaluate_model(model, val_loader)
            loss_history.append(epoch_loss / len(train_loader))
            auc_history.append(val_auc)

            logger.info(f"Fine-tune Epoch {epoch + 1}: Loss={epoch_loss/len(train_loader):.4f}, AUC={val_auc:.4f}")

        training_time = time.time() - start_time

        # Final evaluation
        final_metrics = self._get_detailed_metrics(model, val_loader)

        # Feature separation analysis
        features, labels = FeatureAnalyzer.extract_features(model, val_loader, self.device)
        separation_stats = FeatureAnalyzer.calculate_feature_separation(features, labels)

        return ExperimentResults(
            model_name="MobileNetV4_SupCon",
            loss_type="SupCon+CE",
            final_auc=final_metrics['auc'],
            final_f1=final_metrics['f1'],
            final_accuracy=final_metrics['accuracy'],
            training_time=training_time,
            convergence_epoch=len(auc_history),
            feature_separation=separation_stats['separation_score'],
            loss_history=loss_history,
            auc_history=auc_history
        )

    def _train_and_evaluate(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        model_name: str,
        loss_type: str
    ) -> ExperimentResults:
        """Generic training and evaluation loop."""
        start_time = time.time()
        loss_history = []
        auc_history = []

        for epoch in range(self.config.n_epochs):
            model.train()
            epoch_loss = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Validation
            val_auc = self._evaluate_model(model, val_loader)
            loss_history.append(epoch_loss / len(train_loader))
            auc_history.append(val_auc)

            logger.info(f"{model_name} Epoch {epoch + 1}: Loss={epoch_loss/len(train_loader):.4f}, AUC={val_auc:.4f}")

        training_time = time.time() - start_time

        # Final evaluation
        final_metrics = self._get_detailed_metrics(model, val_loader)

        # Feature separation analysis
        features, labels = FeatureAnalyzer.extract_features(model, val_loader, self.device)
        separation_stats = FeatureAnalyzer.calculate_feature_separation(features, labels)

        return ExperimentResults(
            model_name=model_name,
            loss_type=loss_type,
            final_auc=final_metrics['auc'],
            final_f1=final_metrics['f1'],
            final_accuracy=final_metrics['accuracy'],
            training_time=training_time,
            convergence_epoch=len(auc_history),
            feature_separation=separation_stats['separation_score'],
            loss_history=loss_history,
            auc_history=auc_history
        )

    def _evaluate_model(
        self,
        model: nn.Module,
        val_loader: DataLoader
    ) -> float:
        """Quick AUC evaluation."""
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)

                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class probability
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        return roc_auc_score(all_labels, all_probs)

    def _get_detailed_metrics(
        self,
        model: nn.Module,
        val_loader: DataLoader
    ) -> Dict:
        """Get detailed evaluation metrics."""
        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)

                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return {
            'auc': roc_auc_score(all_labels, all_probs),
            'f1': f1_score(all_labels, all_preds),
            'accuracy': accuracy_score(all_labels, all_preds)
        }

    def run_ablation_study(
        self,
        dataset,
        test_split: float = 0.2
    ) -> Dict:
        """
        Run complete ablation study.

        Returns:
            Complete results with decision gate recommendation
        """
        logger.info("Starting small-scale ablation study...")

        # Sample balanced dataset
        small_dataset, labels = QuickDatasetSampler.sample_balanced_dataset(
            dataset, self.config.n_samples, self.config.seed
        )

        # Split into train/val
        n_train = int(len(small_dataset) * (1 - test_split))
        n_val = len(small_dataset) - n_train

        train_dataset, val_dataset = torch.utils.data.random_split(
            small_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.config.seed)
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )

        # Run experiments
        bce_results = self.run_bce_baseline(train_loader, val_loader)
        supcon_results = self.run_supcon_experiment(train_loader, val_loader)

        # Statistical comparison
        comparison = self._compare_results(bce_results, supcon_results)

        # Decision gate analysis
        decision = self._make_decision_gate_recommendation(comparison)

        # Compile final results
        results = {
            'experiment_config': {
                'n_samples': self.config.n_samples,
                'n_epochs': self.config.n_epochs,
                'batch_size': self.config.batch_size,
                'device': str(self.device)
            },
            'bce_results': bce_results,
            'supcon_results': supcon_results,
            'comparison': comparison,
            'decision_gate': decision,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Log summary
        self._log_experiment_summary(results)

        return results

    def _compare_results(
        self,
        bce_results: ExperimentResults,
        supcon_results: ExperimentResults
    ) -> Dict:
        """Compare BCE and SupCon results."""
        auc_improvement = supcon_results.final_auc - bce_results.final_auc
        f1_improvement = supcon_results.final_f1 - bce_results.final_f1
        separation_improvement = supcon_results.feature_separation - bce_results.feature_separation

        # Statistical significance test (t-test on AUC histories)
        if len(bce_results.auc_history) > 5 and len(supcon_results.auc_history) > 5:
            t_stat, p_value = stats.ttest_ind(
                bce_results.auc_history[-5:],  # Last 5 epochs
                supcon_results.auc_history[-5:]
            )
            is_significant = p_value < self.config.statistical_significance_alpha
        else:
            t_stat, p_value, is_significant = None, None, False

        return {
            'auc_improvement': auc_improvement,
            'auc_improvement_percent': (auc_improvement / bce_results.final_auc) * 100,
            'f1_improvement': f1_improvement,
            'separation_improvement': separation_improvement,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'alpha': self.config.statistical_significance_alpha
            },
            'training_time_ratio': supcon_results.training_time / bce_results.training_time,
            'performance_summary': {
                'bce_final_auc': bce_results.final_auc,
                'supcon_final_auc': supcon_results.final_auc,
                'supcon_better': supcon_results.final_auc > bce_results.final_auc
            }
        }

    def _make_decision_gate_recommendation(self, comparison: Dict) -> Dict:
        """Make recommendation for decision gate."""
        auc_improvement = comparison['auc_improvement']
        is_significant = comparison['statistical_test']['is_significant']

        # Decision logic
        if auc_improvement >= self.config.target_improvement_threshold and is_significant:
            recommendation = "PROCEED_FULL"
            confidence = "HIGH"
            reason = f"SupCon shows {auc_improvement:.1%} improvement (≥{self.config.target_improvement_threshold:.1%}) with statistical significance"
        elif auc_improvement >= self.config.min_improvement_threshold:
            recommendation = "PROCEED_CAUTIOUS"
            confidence = "MEDIUM"
            reason = f"SupCon shows {auc_improvement:.1%} improvement (≥{self.config.min_improvement_threshold:.1%}) but below target"
        elif auc_improvement > 0:
            recommendation = "CONSIDER_OPTIMIZATION"
            confidence = "LOW"
            reason = f"SupCon shows minimal improvement ({auc_improvement:.1%}). Consider hyperparameter tuning"
        else:
            recommendation = "STOP_ACTIVATE_PLAN_B"
            confidence = "HIGH"
            reason = f"SupCon shows negative improvement ({auc_improvement:.1%}). Activate contingency plans"

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reason': reason,
            'auc_improvement': auc_improvement,
            'meets_min_threshold': auc_improvement >= self.config.min_improvement_threshold,
            'meets_target_threshold': auc_improvement >= self.config.target_improvement_threshold,
            'is_statistically_significant': is_significant
        }

    def _log_experiment_summary(self, results: Dict):
        """Log comprehensive experiment summary."""
        decision = results['decision_gate']
        comparison = results['comparison']

        logger.info("="*60)
        logger.info("SMALL-SCALE ABLATION STUDY RESULTS")
        logger.info("="*60)
        logger.info(f"BCE Baseline AUC: {results['bce_results'].final_auc:.4f}")
        logger.info(f"SupCon Method AUC: {results['supcon_results'].final_auc:.4f}")
        logger.info(f"AUC Improvement: {comparison['auc_improvement']:.4f} ({comparison['auc_improvement_percent']:.2f}%)")
        logger.info(f"Feature Separation Improvement: {comparison['separation_improvement']:.4f}")
        logger.info(f"Statistical Significance: {comparison['statistical_test']['is_significant']} (p={comparison['statistical_test']['p_value']:.4f})")
        logger.info("-"*60)
        logger.info(f"DECISION GATE RECOMMENDATION: {decision['recommendation']}")
        logger.info(f"Confidence: {decision['confidence']}")
        logger.info(f"Reason: {decision['reason']}")
        logger.info("="*60)


def test_ablation_study():
    """Test ablation study with synthetic data."""
    print("Testing Ablation Study...")

    # Create synthetic dataset
    n_total = 2000
    feature_dim = 3 * 224 * 224  # Simulate image dimensions

    X = torch.randn(n_total, feature_dim)
    y = torch.randint(0, 2, (n_total,))
    dataset = TensorDataset(X, y)

    # Initialize experiment
    config = AblationConfig(
        n_samples=200,  # Small for testing
        n_epochs=3,
        batch_size=16
    )
    experiment = SmallScaleExperiment(config)

    print(f"✓ Experiment initialized")

    # Run ablation study
    results = experiment.run_ablation_study(dataset)

    print(f"✓ Ablation study completed")
    print(f"  Recommendation: {results['decision_gate']['recommendation']}")
    print(f"  AUC Improvement: {results['comparison']['auc_improvement']:.4f}")
    print(f"  Statistical Significance: {results['comparison']['statistical_test']['is_significant']}")

    print("Ablation study test passed! ✓")


if __name__ == "__main__":
    test_ablation_study()