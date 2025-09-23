"""
Complementarity Analysis and Fusion Prototypes
Advanced analysis of expert complementarity and adaptive fusion mechanisms

This module implements sophisticated analysis tools to understand how spatial and
generative experts complement each other, and provides adaptive fusion strategies
based on complementarity metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import math
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from .unified_feature_extractor import ExpertOutput, ExpertType


class ComplementarityMetric(Enum):
    MUTUAL_INFORMATION = "mutual_information"
    CORRELATION_DISTANCE = "correlation_distance"
    DECISION_DIVERSITY = "decision_diversity"
    UNCERTAINTY_OVERLAP = "uncertainty_overlap"
    FEATURE_ORTHOGONALITY = "feature_orthogonality"


class FusionStrategy(Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    ATTENTION_BASED = "attention_based"
    GATING_NETWORK = "gating_network"
    STACKING = "stacking"
    DYNAMIC_SELECTION = "dynamic_selection"


@dataclass
class ComplementarityConfig:
    metrics: List[ComplementarityMetric] = None
    correlation_threshold: float = 0.7
    diversity_threshold: float = 0.3
    uncertainty_bins: int = 20
    feature_dim_reduction: str = "pca"  # pca, umap, tsne
    visualization_enabled: bool = True

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                ComplementarityMetric.MUTUAL_INFORMATION,
                ComplementarityMetric.DECISION_DIVERSITY,
                ComplementarityMetric.UNCERTAINTY_OVERLAP
            ]


@dataclass
class FusionConfig:
    strategy: FusionStrategy = FusionStrategy.GATING_NETWORK
    hidden_dim: int = 128
    num_experts: int = 2
    temperature: float = 1.0
    dropout_rate: float = 0.1
    adaptive_weighting: bool = True
    uncertainty_aware: bool = True


@dataclass
class ComplementarityAnalysisResult:
    """Results from complementarity analysis"""
    mutual_information: float
    correlation_distance: float
    decision_diversity: float
    uncertainty_overlap: float
    feature_orthogonality: float
    overall_complementarity: float
    recommendations: Dict[str, Any]


class FeatureAnalyzer:
    """
    Analyze feature-level complementarity between experts
    """
    def __init__(self, config: ComplementarityConfig):
        self.config = config

    def compute_mutual_information(self,
                                 features_a: torch.Tensor,
                                 features_b: torch.Tensor,
                                 bins: int = 20) -> float:
        """
        Compute mutual information between feature representations
        """
        # Convert to numpy and flatten
        feat_a = features_a.detach().cpu().numpy().flatten()
        feat_b = features_b.detach().cpu().numpy().flatten()

        # Discretize features
        feat_a_disc = np.digitize(feat_a, np.histogram_bin_edges(feat_a, bins=bins))
        feat_b_disc = np.digitize(feat_b, np.histogram_bin_edges(feat_b, bins=bins))

        # Compute mutual information
        mi = mutual_info_score(feat_a_disc, feat_b_disc)

        # Normalize by joint entropy
        joint_entropy = entropy(np.histogram2d(feat_a_disc, feat_b_disc)[0].flatten())
        normalized_mi = mi / joint_entropy if joint_entropy > 0 else 0

        return normalized_mi

    def compute_correlation_distance(self,
                                   features_a: torch.Tensor,
                                   features_b: torch.Tensor) -> float:
        """
        Compute correlation-based distance between features
        """
        # Global average pooling if spatial features
        if features_a.dim() > 2:
            features_a = F.adaptive_avg_pool2d(features_a, (1, 1)).flatten(1)
        if features_b.dim() > 2:
            features_b = F.adaptive_avg_pool2d(features_b, (1, 1)).flatten(1)

        # Compute correlation matrix
        feat_a = features_a.detach().cpu().numpy()
        feat_b = features_b.detach().cpu().numpy()

        correlations = []
        for i in range(feat_a.shape[1]):
            for j in range(feat_b.shape[1]):
                corr, _ = pearsonr(feat_a[:, i], feat_b[:, j])
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        # Correlation distance (1 - max correlation)
        max_correlation = np.max(correlations) if correlations else 0
        correlation_distance = 1 - max_correlation

        return correlation_distance

    def compute_feature_orthogonality(self,
                                    features_a: torch.Tensor,
                                    features_b: torch.Tensor) -> float:
        """
        Compute orthogonality between feature spaces
        """
        # Normalize features
        feat_a = F.normalize(features_a.flatten(1), dim=1)
        feat_b = F.normalize(features_b.flatten(1), dim=1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(feat_a, feat_b.t())

        # Orthogonality score (lower similarity = higher orthogonality)
        mean_similarity = similarity_matrix.abs().mean().item()
        orthogonality = 1 - mean_similarity

        return orthogonality


class DecisionAnalyzer:
    """
    Analyze decision-level complementarity between experts
    """
    def __init__(self, config: ComplementarityConfig):
        self.config = config

    def compute_decision_diversity(self,
                                 predictions_a: torch.Tensor,
                                 predictions_b: torch.Tensor,
                                 labels: Optional[torch.Tensor] = None) -> float:
        """
        Compute decision diversity between experts
        """
        # Convert predictions to binary decisions
        decisions_a = (predictions_a > 0.5).float()
        decisions_b = (predictions_b > 0.5).float()

        # Compute disagreement rate
        disagreement = (decisions_a != decisions_b).float().mean().item()

        # If labels available, compute Q-statistic
        if labels is not None:
            correct_a = (decisions_a == labels).float()
            correct_b = (decisions_b == labels).float()

            # Q-statistic for diversity
            n11 = ((correct_a == 1) & (correct_b == 1)).sum().item()
            n10 = ((correct_a == 1) & (correct_b == 0)).sum().item()
            n01 = ((correct_a == 0) & (correct_b == 1)).sum().item()
            n00 = ((correct_a == 0) & (correct_b == 0)).sum().item()

            q_statistic = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10 + 1e-8)
            diversity = 1 - q_statistic  # Convert to diversity measure

            return diversity

        return disagreement

    def compute_uncertainty_overlap(self,
                                  predictions_a: torch.Tensor,
                                  predictions_b: torch.Tensor) -> float:
        """
        Compute overlap in uncertainty regions
        """
        # Compute prediction uncertainty (distance from 0.5)
        uncertainty_a = 1 - 2 * torch.abs(predictions_a - 0.5)
        uncertainty_b = 1 - 2 * torch.abs(predictions_b - 0.5)

        # Compute uncertainty overlap
        uncertainty_threshold = 0.5  # High uncertainty threshold
        uncertain_a = uncertainty_a > uncertainty_threshold
        uncertain_b = uncertainty_b > uncertainty_threshold

        # Overlap ratio
        overlap = (uncertain_a & uncertain_b).float().sum()
        total_uncertain = (uncertain_a | uncertain_b).float().sum()

        overlap_ratio = overlap / (total_uncertain + 1e-8)

        # Complementarity is inverse of overlap
        complementarity = 1 - overlap_ratio.item()

        return complementarity


class ComplementarityAnalyzer:
    """
    Main complementarity analysis coordinator
    """
    def __init__(self, config: ComplementarityConfig):
        self.config = config
        self.feature_analyzer = FeatureAnalyzer(config)
        self.decision_analyzer = DecisionAnalyzer(config)

    def analyze_complementarity(self,
                               expert_a_output: ExpertOutput,
                               expert_b_output: ExpertOutput,
                               labels: Optional[torch.Tensor] = None) -> ComplementarityAnalysisResult:
        """
        Perform comprehensive complementarity analysis
        """
        results = {}

        # Extract features and predictions
        features_a = expert_a_output.features.get('fused_features',
                                                expert_a_output.features.get('final_features'))
        features_b = expert_b_output.features.get('fused_features',
                                                expert_b_output.features.get('final_features'))

        pred_a = expert_a_output.predictions.get('classification',
                                               expert_a_output.predictions.get('probability'))
        pred_b = expert_b_output.predictions.get('classification',
                                               expert_b_output.predictions.get('probability'))

        # Feature-level analysis
        if ComplementarityMetric.MUTUAL_INFORMATION in self.config.metrics:
            results['mutual_information'] = self.feature_analyzer.compute_mutual_information(
                features_a, features_b
            )

        if ComplementarityMetric.CORRELATION_DISTANCE in self.config.metrics:
            results['correlation_distance'] = self.feature_analyzer.compute_correlation_distance(
                features_a, features_b
            )

        if ComplementarityMetric.FEATURE_ORTHOGONALITY in self.config.metrics:
            results['feature_orthogonality'] = self.feature_analyzer.compute_feature_orthogonality(
                features_a, features_b
            )

        # Decision-level analysis
        if ComplementarityMetric.DECISION_DIVERSITY in self.config.metrics:
            results['decision_diversity'] = self.decision_analyzer.compute_decision_diversity(
                pred_a, pred_b, labels
            )

        if ComplementarityMetric.UNCERTAINTY_OVERLAP in self.config.metrics:
            results['uncertainty_overlap'] = self.decision_analyzer.compute_uncertainty_overlap(
                pred_a, pred_b
            )

        # Fill missing metrics with defaults
        for metric in ComplementarityMetric:
            if metric.value not in results:
                results[metric.value] = 0.0

        # Compute overall complementarity score
        overall_complementarity = self._compute_overall_score(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        return ComplementarityAnalysisResult(
            mutual_information=results['mutual_information'],
            correlation_distance=results['correlation_distance'],
            decision_diversity=results['decision_diversity'],
            uncertainty_overlap=results['uncertainty_overlap'],
            feature_orthogonality=results['feature_orthogonality'],
            overall_complementarity=overall_complementarity,
            recommendations=recommendations
        )

    def _compute_overall_score(self, results: Dict[str, float]) -> float:
        """
        Compute overall complementarity score
        """
        # Weighted combination of metrics
        weights = {
            'mutual_information': 0.15,
            'correlation_distance': 0.2,
            'decision_diversity': 0.3,
            'uncertainty_overlap': 0.2,
            'feature_orthogonality': 0.15
        }

        overall_score = sum(
            weights.get(metric, 0) * value
            for metric, value in results.items()
        )

        return overall_score

    def _generate_recommendations(self, results: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate recommendations based on analysis results
        """
        recommendations = {
            'fusion_strategy': 'weighted_average',
            'expert_weighting': [0.5, 0.5],
            'training_suggestions': [],
            'architecture_suggestions': []
        }

        # High complementarity -> complex fusion
        if results['overall_complementarity'] > 0.7:
            recommendations['fusion_strategy'] = 'gating_network'
            recommendations['training_suggestions'].append(
                "High complementarity detected. Use sophisticated fusion mechanisms."
            )

        # Low feature orthogonality -> feature diversification
        if results['feature_orthogonality'] < 0.3:
            recommendations['architecture_suggestions'].append(
                "Low feature orthogonality. Consider different backbone architectures."
            )

        # High decision correlation -> ensemble diversity
        if results['decision_diversity'] < 0.2:
            recommendations['training_suggestions'].append(
                "Low decision diversity. Consider diverse training strategies."
            )

        # Uncertainty analysis
        if results['uncertainty_overlap'] > 0.6:
            recommendations['fusion_strategy'] = 'dynamic_selection'
            recommendations['training_suggestions'].append(
                "High uncertainty overlap. Use dynamic expert selection."
            )

        return recommendations


class GatingNetwork(nn.Module):
    """
    Gating network for adaptive expert fusion
    """
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        # Input feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(config.hidden_dim * config.num_experts, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )

        # Gating layers
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(config.num_experts)
        ])

        # Uncertainty-aware weighting
        if config.uncertainty_aware:
            self.uncertainty_processor = nn.Sequential(
                nn.Linear(config.num_experts, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 4, config.num_experts),
                nn.Softmax(dim=-1)
            )

    def forward(self, expert_features: List[torch.Tensor],
                expert_predictions: List[torch.Tensor],
                expert_uncertainties: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for gating network
        """
        # Concatenate expert features
        combined_features = torch.cat(expert_features, dim=-1)
        processed_features = self.feature_processor(combined_features)

        # Compute gates for each expert
        gates = []
        for gate_layer in self.gate_layers:
            gate = gate_layer(processed_features)
            gates.append(gate)

        gates = torch.cat(gates, dim=-1)

        # Uncertainty-aware adjustment
        if self.config.uncertainty_aware and expert_uncertainties is not None:
            uncertainty_tensor = torch.stack(expert_uncertainties, dim=-1)
            uncertainty_weights = self.uncertainty_processor(uncertainty_tensor)
            gates = gates * uncertainty_weights

        # Normalize gates
        gates = F.softmax(gates / self.config.temperature, dim=-1)

        # Compute weighted predictions
        weighted_predictions = []
        for i, pred in enumerate(expert_predictions):
            weighted_pred = gates[:, i:i+1] * pred
            weighted_predictions.append(weighted_pred)

        final_prediction = sum(weighted_predictions)

        return {
            'prediction': final_prediction,
            'gates': gates,
            'expert_weights': gates
        }


class AttentionFusion(nn.Module):
    """
    Attention-based fusion mechanism
    """
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=config.dropout_rate,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, expert_features: List[torch.Tensor],
                expert_predictions: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Attention-based fusion
        """
        # Stack expert features for attention
        # expert_features: List[B, D] -> [B, num_experts, D]
        stacked_features = torch.stack(expert_features, dim=1)

        # Self-attention across experts
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )

        # Global average pooling and prediction
        global_features = attended_features.mean(dim=1)
        final_prediction = self.output_proj(global_features)

        return {
            'prediction': torch.sigmoid(final_prediction),
            'attention_weights': attention_weights,
            'fused_features': global_features
        }


class AdaptiveFusionSystem:
    """
    Adaptive fusion system that selects fusion strategy based on complementarity
    """
    def __init__(self, fusion_config: FusionConfig, complementarity_config: ComplementarityConfig):
        self.fusion_config = fusion_config
        self.complementarity_config = complementarity_config

        # Initialize fusion modules
        self.gating_network = GatingNetwork(fusion_config)
        self.attention_fusion = AttentionFusion(fusion_config)

        # Complementarity analyzer
        self.complementarity_analyzer = ComplementarityAnalyzer(complementarity_config)

    def fuse_experts(self,
                    expert_outputs: List[ExpertOutput],
                    labels: Optional[torch.Tensor] = None,
                    strategy: Optional[FusionStrategy] = None) -> Dict[str, torch.Tensor]:
        """
        Adaptively fuse expert outputs based on complementarity analysis
        """
        # Analyze complementarity if more than one expert
        if len(expert_outputs) >= 2:
            complementarity_result = self.complementarity_analyzer.analyze_complementarity(
                expert_outputs[0], expert_outputs[1], labels
            )

            # Select fusion strategy based on complementarity
            if strategy is None:
                if complementarity_result.overall_complementarity > 0.7:
                    strategy = FusionStrategy.GATING_NETWORK
                elif complementarity_result.decision_diversity > 0.5:
                    strategy = FusionStrategy.ATTENTION_BASED
                else:
                    strategy = FusionStrategy.WEIGHTED_AVERAGE

        else:
            strategy = FusionStrategy.WEIGHTED_AVERAGE

        # Extract features and predictions
        expert_features = []
        expert_predictions = []
        expert_uncertainties = []

        for output in expert_outputs:
            # Get main features
            features = output.features.get('fused_features',
                                         output.features.get('final_features'))
            expert_features.append(features)

            # Get predictions
            pred = output.predictions.get('classification',
                                        output.predictions.get('probability'))
            expert_predictions.append(pred)

            # Get uncertainty (1 - confidence)
            uncertainty = 1 - output.confidence
            expert_uncertainties.append(torch.tensor(uncertainty))

        # Apply fusion strategy
        if strategy == FusionStrategy.GATING_NETWORK:
            result = self.gating_network(expert_features, expert_predictions, expert_uncertainties)
        elif strategy == FusionStrategy.ATTENTION_BASED:
            result = self.attention_fusion(expert_features, expert_predictions)
        else:  # WEIGHTED_AVERAGE
            weights = torch.softmax(torch.stack(expert_uncertainties), dim=0)
            weighted_predictions = sum(w * pred for w, pred in zip(weights, expert_predictions))
            result = {'prediction': weighted_predictions, 'weights': weights}

        # Add complementarity info
        if len(expert_outputs) >= 2:
            result['complementarity_score'] = complementarity_result.overall_complementarity
            result['fusion_strategy'] = strategy.value

        return result


def create_fusion_system(hidden_dim: int = 256,
                        num_experts: int = 2,
                        uncertainty_aware: bool = True) -> AdaptiveFusionSystem:
    """
    Factory function to create adaptive fusion system
    """
    fusion_config = FusionConfig(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        uncertainty_aware=uncertainty_aware
    )

    complementarity_config = ComplementarityConfig(
        visualization_enabled=False  # Disable for production
    )

    return AdaptiveFusionSystem(fusion_config, complementarity_config)