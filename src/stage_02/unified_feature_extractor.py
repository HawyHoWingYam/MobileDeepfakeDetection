"""
AWARE-NET Stage 02: Unified Feature Extraction Interface

This module defines the unified interface for heterogeneous expert feature extraction,
supporting both spatial (EfficientNetV2-B0) and generative (GenConViT) experts with
flexible resolution support and standardized output formats.

Design Principles:
- Unified API for different expert architectures
- Flexible multi-resolution input processing
- Standardized feature output format for fusion
- Efficient parallel expert inference
- Integration with Stage 01 rapid filter
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ExpertType(Enum):
    """Expert model types in the heterogeneous system."""
    SPATIAL = "spatial"           # EfficientNetV2-B0 spatial artifact expert
    GENERATIVE = "generative"     # GenConViT generative structure expert
    RAPID_FILTER = "rapid_filter" # Stage 01 SupCon rapid filter

class ResolutionMode(Enum):
    """Supported input resolution modes."""
    STANDARD = 256      # 256x256 - standard training resolution
    SMALL = 224         # 224x224 - efficiency mode
    MEDIUM = 288        # 288x288 - balanced mode
    LARGE = 320         # 320x320 - detail preservation mode

@dataclass
class ExpertOutput:
    """Standardized output format for all experts."""
    # Core predictions
    logits: torch.Tensor                    # Raw classification logits
    probabilities: torch.Tensor             # Calibrated probabilities
    predictions: torch.Tensor               # Binary predictions
    confidence: torch.Tensor                # Prediction confidence scores

    # Feature representations
    features: torch.Tensor                  # Raw feature embeddings
    normalized_features: torch.Tensor       # L2-normalized features for fusion

    # Expert-specific outputs
    expert_type: ExpertType                 # Which expert generated this output
    resolution: ResolutionMode              # Input resolution used

    # Optional outputs
    attention_maps: Optional[torch.Tensor] = None      # Spatial attention (Grad-CAM, etc.)
    reconstruction: Optional[torch.Tensor] = None      # GenConViT reconstruction
    reconstruction_error: Optional[torch.Tensor] = None # Reconstruction quality metrics

    # Metadata
    inference_time_ms: Optional[float] = None          # Inference latency
    memory_usage_mb: Optional[float] = None            # Peak memory usage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'logits': self.logits.cpu().numpy() if isinstance(self.logits, torch.Tensor) else self.logits,
            'probabilities': self.probabilities.cpu().numpy() if isinstance(self.probabilities, torch.Tensor) else self.probabilities,
            'predictions': self.predictions.cpu().numpy() if isinstance(self.predictions, torch.Tensor) else self.predictions,
            'confidence': self.confidence.cpu().numpy() if isinstance(self.confidence, torch.Tensor) else self.confidence,
            'features': self.features.cpu().numpy() if isinstance(self.features, torch.Tensor) else self.features,
            'expert_type': self.expert_type.value,
            'resolution': self.resolution.value,
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_mb': self.memory_usage_mb
        }

@dataclass
class HeterogeneousOutput:
    """Combined output from multiple experts."""
    expert_outputs: Dict[ExpertType, ExpertOutput]     # Individual expert outputs
    fusion_logits: Optional[torch.Tensor] = None       # Fused logits
    fusion_probabilities: Optional[torch.Tensor] = None # Fused probabilities
    fusion_predictions: Optional[torch.Tensor] = None   # Fused predictions

    # Complementarity analysis
    expert_agreement: Optional[float] = None            # Inter-expert agreement
    confidence_variance: Optional[float] = None         # Confidence variance across experts

    # System performance
    total_inference_time_ms: Optional[float] = None     # Total pipeline latency
    peak_memory_usage_mb: Optional[float] = None        # Peak memory across all experts

class BaseExpert(ABC):
    """Abstract base class for all expert models."""

    def __init__(self, expert_type: ExpertType, config: Dict[str, Any]):
        self.expert_type = expert_type
        self.config = config
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load the expert model."""
        pass

    @abstractmethod
    def preprocess_input(self,
                        images: torch.Tensor,
                        target_resolution: ResolutionMode = ResolutionMode.STANDARD) -> torch.Tensor:
        """Preprocess input images to target resolution."""
        pass

    @abstractmethod
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract feature representations."""
        pass

    @abstractmethod
    def predict(self, images: torch.Tensor, return_features: bool = True) -> ExpertOutput:
        """Generate predictions with optional feature extraction."""
        pass

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

class AdaptiveInputProcessor:
    """Handles flexible multi-resolution input processing."""

    def __init__(self, supported_resolutions: List[ResolutionMode]):
        self.supported_resolutions = supported_resolutions

    def process_batch(self,
                     images: torch.Tensor,
                     target_resolution: ResolutionMode,
                     preserve_aspect_ratio: bool = True) -> torch.Tensor:
        """
        Process a batch of images to target resolution.

        Args:
            images: Input tensor [B, C, H, W]
            target_resolution: Target resolution mode
            preserve_aspect_ratio: Whether to preserve aspect ratio during resize

        Returns:
            Processed tensor [B, C, target_size, target_size]
        """
        if target_resolution not in self.supported_resolutions:
            raise ValueError(f"Unsupported resolution: {target_resolution}")

        target_size = target_resolution.value

        if preserve_aspect_ratio:
            # Resize with aspect ratio preservation + center crop
            _, _, h, w = images.shape
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize
            images = torch.nn.functional.interpolate(
                images, size=(new_h, new_w), mode='bilinear', align_corners=False
            )

            # Center crop
            if new_h > target_size:
                start_h = (new_h - target_size) // 2
                images = images[:, :, start_h:start_h + target_size, :]
            if new_w > target_size:
                start_w = (new_w - target_size) // 2
                images = images[:, :, :, start_w:start_w + target_size]
        else:
            # Direct resize
            images = torch.nn.functional.interpolate(
                images, size=(target_size, target_size), mode='bilinear', align_corners=False
            )

        return images

    def get_optimal_resolution(self,
                              input_size: Tuple[int, int],
                              expert_type: ExpertType) -> ResolutionMode:
        """
        Determine optimal resolution based on input size and expert type.

        Args:
            input_size: Original image size (H, W)
            expert_type: Type of expert model

        Returns:
            Recommended resolution mode
        """
        h, w = input_size
        original_size = min(h, w)

        # Expert-specific resolution recommendations
        if expert_type == ExpertType.SPATIAL:
            # Spatial expert benefits from higher resolution for artifact detection
            if original_size >= 320:
                return ResolutionMode.LARGE
            elif original_size >= 288:
                return ResolutionMode.MEDIUM
            else:
                return ResolutionMode.STANDARD

        elif expert_type == ExpertType.GENERATIVE:
            # GenConViT uses standard resolution for reconstruction consistency
            return ResolutionMode.STANDARD

        else:  # RAPID_FILTER
            # Rapid filter optimized for 256x256
            return ResolutionMode.STANDARD

class UnifiedFeatureExtractor:
    """
    Unified interface for extracting features from heterogeneous experts.

    This class coordinates multiple expert models and provides a standardized
    interface for feature extraction, prediction, and analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experts: Dict[ExpertType, BaseExpert] = {}
        self.input_processor = AdaptiveInputProcessor(
            supported_resolutions=list(ResolutionMode)
        )
        self.is_initialized = False

    def register_expert(self, expert: BaseExpert) -> None:
        """Register an expert model."""
        self.experts[expert.expert_type] = expert

    def load_experts(self, checkpoint_paths: Dict[ExpertType, str]) -> None:
        """Load all registered experts."""
        for expert_type, expert in self.experts.items():
            if expert_type in checkpoint_paths:
                expert.load_model(checkpoint_paths[expert_type])
        self.is_initialized = True

    def extract_features_single_expert(self,
                                     images: torch.Tensor,
                                     expert_type: ExpertType,
                                     resolution: Optional[ResolutionMode] = None) -> ExpertOutput:
        """
        Extract features using a single expert.

        Args:
            images: Input images [B, C, H, W]
            expert_type: Which expert to use
            resolution: Target resolution (auto-selected if None)

        Returns:
            Expert output with features and predictions
        """
        if not self.is_initialized:
            raise RuntimeError("Experts not loaded. Call load_experts() first.")

        if expert_type not in self.experts:
            raise ValueError(f"Expert {expert_type} not registered.")

        expert = self.experts[expert_type]

        # Auto-select resolution if not specified
        if resolution is None:
            h, w = images.shape[-2:]
            resolution = self.input_processor.get_optimal_resolution((h, w), expert_type)

        # Preprocess input
        processed_images = expert.preprocess_input(images, resolution)

        # Generate predictions
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

        output = expert.predict(processed_images, return_features=True)

        if start_time:
            end_time.record()
            torch.cuda.synchronize()
            output.inference_time_ms = start_time.elapsed_time(end_time)

        output.memory_usage_mb = expert.get_memory_usage()
        output.resolution = resolution

        return output

    def extract_features_all_experts(self,
                                   images: torch.Tensor,
                                   expert_types: Optional[List[ExpertType]] = None,
                                   parallel: bool = True) -> HeterogeneousOutput:
        """
        Extract features using multiple experts.

        Args:
            images: Input images [B, C, H, W]
            expert_types: Which experts to use (all if None)
            parallel: Whether to run experts in parallel

        Returns:
            Combined output from all experts
        """
        if expert_types is None:
            expert_types = list(self.experts.keys())

        expert_outputs = {}
        total_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if total_start_time:
            total_end_time = torch.cuda.Event(enable_timing=True)
            total_start_time.record()

        if parallel and len(expert_types) > 1:
            # TODO: Implement parallel processing
            # For now, sequential processing
            for expert_type in expert_types:
                expert_outputs[expert_type] = self.extract_features_single_expert(
                    images, expert_type
                )
        else:
            # Sequential processing
            for expert_type in expert_types:
                expert_outputs[expert_type] = self.extract_features_single_expert(
                    images, expert_type
                )

        if total_start_time:
            total_end_time.record()
            torch.cuda.synchronize()
            total_inference_time = total_start_time.elapsed_time(total_end_time)
        else:
            total_inference_time = None

        # Analyze expert complementarity
        expert_agreement = self._compute_expert_agreement(expert_outputs)
        confidence_variance = self._compute_confidence_variance(expert_outputs)

        return HeterogeneousOutput(
            expert_outputs=expert_outputs,
            expert_agreement=expert_agreement,
            confidence_variance=confidence_variance,
            total_inference_time_ms=total_inference_time,
            peak_memory_usage_mb=max([out.memory_usage_mb or 0 for out in expert_outputs.values()])
        )

    def _compute_expert_agreement(self, expert_outputs: Dict[ExpertType, ExpertOutput]) -> float:
        """Compute agreement between expert predictions."""
        if len(expert_outputs) < 2:
            return 1.0

        predictions = [output.predictions for output in expert_outputs.values()]

        # Compute pairwise agreement
        total_pairs = 0
        total_agreement = 0

        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                agreement = (predictions[i] == predictions[j]).float().mean()
                total_agreement += agreement.item()
                total_pairs += 1

        return total_agreement / total_pairs if total_pairs > 0 else 1.0

    def _compute_confidence_variance(self, expert_outputs: Dict[ExpertType, ExpertOutput]) -> float:
        """Compute variance in confidence scores across experts."""
        if len(expert_outputs) < 2:
            return 0.0

        confidences = torch.stack([output.confidence for output in expert_outputs.values()])
        return torch.var(confidences, dim=0).mean().item()

    def get_supported_expert_types(self) -> List[ExpertType]:
        """Get list of supported expert types."""
        return list(self.experts.keys())

    def get_expert_config(self, expert_type: ExpertType) -> Dict[str, Any]:
        """Get configuration for a specific expert."""
        if expert_type in self.experts:
            return self.experts[expert_type].config
        raise ValueError(f"Expert {expert_type} not found.")

# Utility functions for feature extraction interface

def create_unified_feature_extractor(config_path: str) -> UnifiedFeatureExtractor:
    """Factory function to create unified feature extractor from config."""
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    return UnifiedFeatureExtractor(config)

def normalize_features_for_fusion(features: torch.Tensor) -> torch.Tensor:
    """Normalize features for fusion (L2 normalization)."""
    return torch.nn.functional.normalize(features, p=2, dim=-1)

def compute_feature_similarity(features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between feature vectors."""
    features1_norm = normalize_features_for_fusion(features1)
    features2_norm = normalize_features_for_fusion(features2)
    return torch.sum(features1_norm * features2_norm, dim=-1)

# Export interface
__all__ = [
    'ExpertType', 'ResolutionMode', 'ExpertOutput', 'HeterogeneousOutput',
    'BaseExpert', 'AdaptiveInputProcessor', 'UnifiedFeatureExtractor',
    'create_unified_feature_extractor', 'normalize_features_for_fusion',
    'compute_feature_similarity'
]