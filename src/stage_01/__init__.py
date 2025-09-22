"""
AWARE-NET Stage 1: SupCon快速過濾器 (Rapid Filter)

This package implements the core innovation of AWARE-NET project: a paradigm shift
from "fake detection" to "authenticity modeling" using Supervised Contrastive Learning.

Architecture Overview:
- MobileNetV4-Hybrid-Medium backbone for mobile-friendly inference
- SupCon loss for building "truthfulness fortress" in feature space
- Temperature scaling for probability calibration
- Conservative threshold strategy for cascade system integration

Key Components:
- supcon_loss: Supervised contrastive learning implementation
- mobilenetv4_model: Mobile-optimized architecture with projection heads
- balanced_sampler: Contrastive learning batch construction
- temperature_scaling: Probability calibration framework
- cascade_strategy: Conservative thresholds for cascade routing
- ablation_study: Small-scale validation (CRITICAL DECISION GATE)
- stage02_integration: Interface for Stage 2 handoff
- evaluate_stage1: Comprehensive evaluation and stage-gate validation

Usage Example:
    from stage_01 import Stage1RapidFilter, create_mobilenetv4_supcon
    from stage_01.supcon_loss import SupConLoss

    # Create model
    model = create_mobilenetv4_supcon(
        pretrained=True,
        projection_dim=512
    )

    # Create loss
    criterion = SupConLoss(temperature=0.07)

    # Create rapid filter for inference
    rapid_filter = Stage1RapidFilter(model, temp_scaler, threshold_strategy)

    # Make cascade decisions
    outputs = rapid_filter.predict(inputs, return_features=True)

Academic Contribution:
This implementation represents the first systematic application of supervised
contrastive learning to deepfake detection, establishing a new paradigm of
"authenticity modeling" that focuses on learning the intrinsic geometry of
authentic content rather than detecting fake artifacts.

Performance Targets:
- AUC ≥ 0.90 (baseline improvement ≥ 3%)
- Inference time ≤ 50ms per image
- Memory usage ≤ 2GB
- Cascade filter rate ≥ 90%
- False negative rate ≤ 5%
- ECE after calibration ≤ 0.05

Stage-Gate Validation:
All components include comprehensive stage-gate validation against technical,
academic, and system requirements. Use evaluate_stage1.Stage1Evaluator for
complete validation before proceeding to Stage 2.
"""

from .supcon_loss import SupConLoss, SupConLossWithLogging
from .mobilenetv4_model import (
    MobileNetV4SupCon,
    ProjectionHead,
    ClassificationHead,
    create_mobilenetv4_supcon
)
from .balanced_sampler import BalancedBatchSampler, ContrastivePairSampler
from .temperature_scaling import (
    TemperatureScaling,
    CalibrationMetrics,
    ModelCalibrator,
    CascadeThresholdOptimizer
)
from .cascade_strategy import (
    ConservativeThresholdStrategy,
    CascadeDecision,
    CascadeConfig
)
from .ablation_study import (
    SmallScaleExperiment,
    AblationConfig,
    ExperimentResults,
    FeatureAnalyzer,
    QuickDatasetSampler
)
from .stage02_integration import (
    Stage1RapidFilter,
    Stage1Output,
    CascadeStatistics,
    Stage1Interface,
    Stage1ArtifactPackager
)
from .evaluate_stage1 import (
    Stage1Evaluator,
    EvaluationReport,
    StageGateResults,
    PerformanceProfile
)

# Version and metadata
__version__ = "1.0.0"
__stage__ = "stage_01"
__paradigm__ = "authenticity_modeling"

# Key innovation summary
__innovation__ = "SupCon-based authenticity modeling for deepfake detection"
__contribution__ = "First systematic application of supervised contrastive learning to deepfake detection"

# Performance specifications
PERFORMANCE_TARGETS = {
    'technical': {
        'min_auc': 0.90,
        'baseline_improvement_min': 0.03,
        'inference_time_max_ms': 50,
        'memory_max_gb': 2,
        'calibration_ece_max': 0.05
    },
    'system': {
        'cascade_filter_rate_min': 0.90,
        'false_negative_rate_max': 0.05,
        'mobile_deployment_ready': True,
        'package_size_max_mb': 50
    }
}

# Export convenience functions
def create_stage1_pipeline(
    model_name: str = 'mobilenetv4_hybrid_medium',
    projection_dim: int = 512,
    temperature: float = 0.07,
    pretrained: bool = True
):
    """
    Create complete Stage 1 pipeline with all components.

    Returns:
        Tuple of (model, supcon_loss, temp_scaler, threshold_strategy)
    """
    # Create model
    model = create_mobilenetv4_supcon(
        model_name=model_name,
        projection_dim=projection_dim,
        pretrained=pretrained
    )

    # Create loss
    supcon_loss = SupConLoss(temperature=temperature)

    # Create calibration components
    temp_scaler = TemperatureScaling()
    threshold_strategy = ConservativeThresholdStrategy()

    return model, supcon_loss, temp_scaler, threshold_strategy


def run_small_scale_validation(
    dataset,
    n_samples: int = 1000,
    n_epochs: int = 10,
    device: str = 'auto'
) -> dict:
    """
    Run critical small-scale validation experiment.

    This is the DECISION GATE for Stage 1 implementation.

    Returns:
        Dictionary with decision gate recommendation and detailed results
    """
    from .ablation_study import SmallScaleExperiment, AblationConfig

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = AblationConfig(
        n_samples=n_samples,
        n_epochs=n_epochs,
        device=device
    )

    experiment = SmallScaleExperiment(config)
    results = experiment.run_ablation_study(dataset)

    return results


# Validation utilities
def validate_stage_gates(
    rapid_filter,
    test_loader,
    baseline_results=None
) -> bool:
    """
    Quick stage-gate validation.

    Returns:
        True if all gates pass, False otherwise
    """
    evaluator = Stage1Evaluator()
    report = evaluator.evaluate_comprehensive(
        rapid_filter, test_loader, baseline_results
    )

    return report.stage_gate_results.overall_pass


# Import torch for convenience functions
import torch

__all__ = [
    # Core components
    'SupConLoss', 'SupConLossWithLogging',
    'MobileNetV4SupCon', 'ProjectionHead', 'ClassificationHead',
    'BalancedBatchSampler', 'ContrastivePairSampler',
    'TemperatureScaling', 'ModelCalibrator',
    'ConservativeThresholdStrategy', 'CascadeDecision',

    # Experiment and evaluation
    'SmallScaleExperiment', 'AblationConfig', 'FeatureAnalyzer',
    'Stage1Evaluator', 'EvaluationReport', 'StageGateResults',

    # Integration
    'Stage1RapidFilter', 'Stage1Output', 'Stage1ArtifactPackager',

    # Factory functions
    'create_mobilenetv4_supcon', 'create_stage1_pipeline',

    # Validation utilities
    'run_small_scale_validation', 'validate_stage_gates',

    # Constants
    'PERFORMANCE_TARGETS',

    # Metadata
    '__version__', '__stage__', '__paradigm__', '__innovation__', '__contribution__'
]