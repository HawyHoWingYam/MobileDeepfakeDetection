"""
MobileDeepfakeDetection package initializer.

This module intentionally keeps the import surface lean so that importing
``src`` does not pull in optional research utilities that may or may not be
present in a given deployment.  Only the commonly used APIs that actually
exist in the repository are re-exported here for convenience.
"""

from .models.mobilenetv4_model import (
    MobileNetV4SupCon,
    ProjectionHead,
    ClassificationHead,
    create_mobilenetv4_simple,
    create_mobilenetv4_supcon,
)
from .models.efficientnetv2_model import (
    EfficientNetV2B3Baseline,
    create_baseline_model,
)
from .utils.supcon_loss import SupConLoss, SupConLossWithLogging
from .utils.balanced_sampler import BalancedBatchSampler, ContrastivePairSampler
from .utils.experiment_framework import ExperimentFramework, setup_reproducible_environment
from .utils.evaluation import ModelEvaluator

__all__ = [
    # Models
    "MobileNetV4SupCon",
    "ProjectionHead",
    "ClassificationHead",
    "create_mobilenetv4_simple",
    "create_mobilenetv4_supcon",
    "EfficientNetV2B3Baseline",
    "create_baseline_model",
    # Losses & samplers
    "SupConLoss",
    "SupConLossWithLogging",
    "BalancedBatchSampler",
    "ContrastivePairSampler",
    # Experiment utilities
    "ExperimentFramework",
    "setup_reproducible_environment",
    "ModelEvaluator",
]

__version__ = "1.0.0"
