"""
Stage 3 Integration Interface
Seamless integration interface for temporal modeling expert

This module provides standardized interfaces and protocols for integrating
Stage 2 heterogeneous experts with Stage 3 temporal modeling capabilities.
Enables smooth data flow, feature sharing, and backward compatibility.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import numpy as np

from .unified_feature_extractor import BaseExpert, ExpertOutput, ExpertType
from .complementarity_analysis import AdaptiveFusionSystem
from .concurrent_testing_framework import TestMetrics


class TemporalMode(Enum):
    FRAME_SEQUENCE = "frame_sequence"
    VIDEO_CLIPS = "video_clips"
    OPTICAL_FLOW = "optical_flow"
    MOTION_VECTORS = "motion_vectors"


class IntegrationLevel(Enum):
    FEATURE_LEVEL = "feature_level"
    DECISION_LEVEL = "decision_level"
    HYBRID = "hybrid"


@dataclass
class TemporalInput:
    """Standardized temporal input format"""
    frames: torch.Tensor  # [B, T, C, H, W] or [B, C, T, H, W]
    frame_timestamps: Optional[List[float]] = None
    sequence_metadata: Dict[str, Any] = field(default_factory=dict)
    optical_flow: Optional[torch.Tensor] = None  # [B, T-1, 2, H, W]
    motion_vectors: Optional[torch.Tensor] = None


@dataclass
class Stage2Output:
    """Standardized Stage 2 output format"""
    spatial_features: torch.Tensor
    generative_features: torch.Tensor
    fused_features: torch.Tensor
    spatial_predictions: torch.Tensor
    generative_predictions: torch.Tensor
    final_predictions: torch.Tensor
    confidence_scores: torch.Tensor
    attention_maps: Optional[torch.Tensor] = None
    complementarity_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationConfig:
    """Configuration for Stage 2-3 integration"""
    integration_level: IntegrationLevel = IntegrationLevel.HYBRID
    temporal_mode: TemporalMode = TemporalMode.FRAME_SEQUENCE
    feature_alignment: bool = True
    backward_compatibility: bool = True
    enable_caching: bool = True
    batch_processing: bool = True
    max_sequence_length: int = 16
    feature_dimension: int = 256
    fusion_strategy: str = "attention_based"


class TemporalExpertProtocol(Protocol):
    """Protocol definition for temporal experts"""

    def process_temporal_sequence(self,
                                temporal_input: TemporalInput,
                                stage2_output: Stage2Output) -> ExpertOutput:
        """Process temporal sequence with Stage 2 context"""
        ...

    def extract_temporal_features(self,
                                temporal_input: TemporalInput) -> torch.Tensor:
        """Extract temporal features from input sequence"""
        ...

    def get_temporal_attention(self,
                             temporal_input: TemporalInput) -> torch.Tensor:
        """Get temporal attention weights"""
        ...


class FeatureAlignmentModule(nn.Module):
    """
    Align features between Stage 2 and Stage 3
    """
    def __init__(self, config: IntegrationConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dimension

        # Feature projection layers
        self.spatial_projector = nn.Sequential(
            nn.Linear(512, self.feature_dim),  # Assume 512 from spatial expert
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.generative_projector = nn.Sequential(
            nn.Linear(384, self.feature_dim),  # Assume 384 from generative expert
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.temporal_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Feature normalization
        self.layer_norm = nn.LayerNorm(self.feature_dim)

    def align_spatial_features(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Align spatial features to common dimension"""
        # Handle different spatial feature dimensions
        if spatial_features.dim() > 2:
            spatial_features = torch.flatten(spatial_features, 1)

        aligned = self.spatial_projector(spatial_features)
        return self.layer_norm(aligned)

    def align_generative_features(self, generative_features: torch.Tensor) -> torch.Tensor:
        """Align generative features to common dimension"""
        if generative_features.dim() > 2:
            generative_features = torch.flatten(generative_features, 1)

        aligned = self.generative_projector(generative_features)
        return self.layer_norm(aligned)

    def align_temporal_features(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """Align temporal features to common dimension"""
        aligned = self.temporal_projector(temporal_features)
        return self.layer_norm(aligned)


class SequenceProcessor:
    """
    Process video sequences for temporal modeling
    """
    def __init__(self, config: IntegrationConfig):
        self.config = config

    def prepare_temporal_input(self,
                             video_frames: torch.Tensor,
                             metadata: Optional[Dict[str, Any]] = None) -> TemporalInput:
        """
        Prepare temporal input from video frames
        """
        # Ensure correct tensor format [B, T, C, H, W]
        if video_frames.dim() == 5:
            if video_frames.size(1) == 3:  # [B, C, T, H, W] -> [B, T, C, H, W]
                video_frames = video_frames.permute(0, 2, 1, 3, 4)
        elif video_frames.dim() == 4:
            # Add temporal dimension
            video_frames = video_frames.unsqueeze(1)

        # Limit sequence length
        if video_frames.size(1) > self.config.max_sequence_length:
            video_frames = video_frames[:, :self.config.max_sequence_length]

        # Generate timestamps if not provided
        frame_timestamps = list(range(video_frames.size(1)))

        # Compute optical flow if needed
        optical_flow = None
        if self.config.temporal_mode == TemporalMode.OPTICAL_FLOW:
            optical_flow = self._compute_optical_flow(video_frames)

        return TemporalInput(
            frames=video_frames,
            frame_timestamps=frame_timestamps,
            sequence_metadata=metadata or {},
            optical_flow=optical_flow
        )

    def _compute_optical_flow(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute optical flow between consecutive frames
        """
        B, T, C, H, W = frames.shape
        optical_flow = torch.zeros(B, T-1, 2, H, W, device=frames.device)

        # Simplified optical flow computation (placeholder)
        for t in range(T-1):
            frame1 = frames[:, t]
            frame2 = frames[:, t+1]

            # Compute frame difference as proxy for optical flow
            diff = frame2 - frame1
            flow_x = diff.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            flow_y = torch.zeros_like(flow_x)  # Placeholder

            optical_flow[:, t] = torch.cat([flow_x, flow_y], dim=1)

        return optical_flow

    def batch_process_sequences(self,
                              sequences: List[torch.Tensor]) -> List[TemporalInput]:
        """
        Batch process multiple sequences
        """
        temporal_inputs = []
        for sequence in sequences:
            temporal_input = self.prepare_temporal_input(sequence)
            temporal_inputs.append(temporal_input)
        return temporal_inputs


class Stage2ExpertWrapper:
    """
    Wrapper for Stage 2 experts to provide consistent interface
    """
    def __init__(self,
                 spatial_expert: BaseExpert,
                 generative_expert: BaseExpert,
                 fusion_system: AdaptiveFusionSystem,
                 config: IntegrationConfig):
        self.spatial_expert = spatial_expert
        self.generative_expert = generative_expert
        self.fusion_system = fusion_system
        self.config = config

        # Feature alignment
        self.feature_aligner = FeatureAlignmentModule(config)

        # Caching for efficiency
        self.feature_cache = {} if config.enable_caching else None

    def process_single_frame(self, frame: torch.Tensor) -> Stage2Output:
        """
        Process single frame through Stage 2 experts
        """
        # Check cache first
        if self.config.enable_caching and self.feature_cache is not None:
            frame_hash = hash(frame.data_ptr())
            if frame_hash in self.feature_cache:
                return self.feature_cache[frame_hash]

        # Run spatial expert
        spatial_output = self.spatial_expert(frame)

        # Run generative expert
        generative_output = self.generative_expert(frame)

        # Fusion
        fusion_result = self.fusion_system.fuse_experts([spatial_output, generative_output])

        # Align features
        spatial_features = self.feature_aligner.align_spatial_features(
            spatial_output.features.get('fused_features',
                                      spatial_output.features.get('final_features'))
        )

        generative_features = self.feature_aligner.align_generative_features(
            generative_output.features.get('fused_features',
                                         generative_output.features.get('final_features'))
        )

        # Create Stage 2 output
        stage2_output = Stage2Output(
            spatial_features=spatial_features,
            generative_features=generative_features,
            fused_features=spatial_features + generative_features,  # Simple fusion
            spatial_predictions=spatial_output.predictions.get('classification'),
            generative_predictions=generative_output.predictions.get('classification'),
            final_predictions=fusion_result['prediction'],
            confidence_scores=torch.tensor([spatial_output.confidence, generative_output.confidence]),
            complementarity_score=fusion_result.get('complementarity_score'),
            metadata={
                'spatial_metadata': spatial_output.features,
                'generative_metadata': generative_output.features
            }
        )

        # Cache result
        if self.config.enable_caching and self.feature_cache is not None:
            self.feature_cache[frame_hash] = stage2_output

        return stage2_output

    def process_sequence(self, temporal_input: TemporalInput) -> List[Stage2Output]:
        """
        Process entire temporal sequence through Stage 2 experts
        """
        sequence_outputs = []

        B, T, C, H, W = temporal_input.frames.shape

        for t in range(T):
            frame = temporal_input.frames[:, t]  # [B, C, H, W]
            stage2_output = self.process_single_frame(frame)
            sequence_outputs.append(stage2_output)

        return sequence_outputs


class TemporalIntegrationHub:
    """
    Central hub for Stage 2-3 integration
    """
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.sequence_processor = SequenceProcessor(config)
        self.registered_temporal_experts = {}

        # Integration metrics
        self.integration_stats = {
            'processed_sequences': 0,
            'average_processing_time': 0.0,
            'cache_hit_rate': 0.0
        }

    def register_temporal_expert(self, name: str, expert: TemporalExpertProtocol):
        """Register a temporal expert"""
        self.registered_temporal_experts[name] = expert

    def create_stage2_wrapper(self,
                            spatial_expert: BaseExpert,
                            generative_expert: BaseExpert,
                            fusion_system: AdaptiveFusionSystem) -> Stage2ExpertWrapper:
        """Create Stage 2 expert wrapper"""
        return Stage2ExpertWrapper(
            spatial_expert, generative_expert, fusion_system, self.config
        )

    def integrated_inference(self,
                           video_input: torch.Tensor,
                           stage2_wrapper: Stage2ExpertWrapper,
                           temporal_expert_name: str,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform integrated inference across Stage 2 and Stage 3
        """
        if temporal_expert_name not in self.registered_temporal_experts:
            raise ValueError(f"Temporal expert '{temporal_expert_name}' not registered")

        temporal_expert = self.registered_temporal_experts[temporal_expert_name]

        # Prepare temporal input
        temporal_input = self.sequence_processor.prepare_temporal_input(video_input, metadata)

        # Process through Stage 2
        stage2_outputs = stage2_wrapper.process_sequence(temporal_input)

        # Process through temporal expert
        temporal_results = []
        for i, stage2_output in enumerate(stage2_outputs):
            # Create single-frame temporal input for this frame
            single_frame_input = TemporalInput(
                frames=temporal_input.frames[:, i:i+1],
                frame_timestamps=[temporal_input.frame_timestamps[i]] if temporal_input.frame_timestamps else None,
                sequence_metadata=temporal_input.sequence_metadata
            )

            temporal_result = temporal_expert.process_temporal_sequence(
                single_frame_input, stage2_output
            )
            temporal_results.append(temporal_result)

        # Aggregate temporal results
        aggregated_result = self._aggregate_temporal_results(temporal_results, stage2_outputs)

        # Update stats
        self.integration_stats['processed_sequences'] += 1

        return aggregated_result

    def _aggregate_temporal_results(self,
                                  temporal_results: List[ExpertOutput],
                                  stage2_outputs: List[Stage2Output]) -> Dict[str, Any]:
        """
        Aggregate results from temporal processing
        """
        # Extract predictions and features
        temporal_predictions = [result.predictions for result in temporal_results]
        temporal_features = [result.features for result in temporal_results]

        # Compute sequence-level aggregations
        final_prediction = torch.stack([
            pred.get('classification', pred.get('probability', torch.tensor(0.5)))
            for pred in temporal_predictions
        ]).mean()

        # Confidence aggregation
        confidences = [result.confidence for result in temporal_results]
        final_confidence = np.mean(confidences)

        # Feature aggregation
        if temporal_features and 'fused_features' in temporal_features[0]:
            aggregated_features = torch.stack([
                feat['fused_features'] for feat in temporal_features
            ]).mean(dim=0)
        else:
            aggregated_features = None

        return {
            'final_prediction': final_prediction,
            'confidence': final_confidence,
            'temporal_predictions': temporal_predictions,
            'stage2_outputs': stage2_outputs,
            'aggregated_features': aggregated_features,
            'sequence_length': len(temporal_results),
            'integration_level': self.config.integration_level.value
        }

    def validate_integration(self,
                           stage2_wrapper: Stage2ExpertWrapper,
                           temporal_expert_name: str,
                           test_input: torch.Tensor) -> Dict[str, Any]:
        """
        Validate Stage 2-3 integration
        """
        validation_results = {
            'compatibility_check': True,
            'performance_metrics': {},
            'error_messages': []
        }

        try:
            # Test basic integration
            result = self.integrated_inference(
                test_input, stage2_wrapper, temporal_expert_name
            )

            # Validate output format
            required_keys = ['final_prediction', 'confidence', 'temporal_predictions']
            for key in required_keys:
                if key not in result:
                    validation_results['compatibility_check'] = False
                    validation_results['error_messages'].append(f"Missing required key: {key}")

            # Performance validation
            if validation_results['compatibility_check']:
                validation_results['performance_metrics'] = {
                    'sequence_length': result['sequence_length'],
                    'prediction_range': (float(result['final_prediction'].min()),
                                        float(result['final_prediction'].max())),
                    'confidence_score': result['confidence']
                }

        except Exception as e:
            validation_results['compatibility_check'] = False
            validation_results['error_messages'].append(str(e))

        return validation_results

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration performance statistics"""
        return self.integration_stats.copy()


class BackwardCompatibilityLayer:
    """
    Ensure backward compatibility with existing Stage 2 implementations
    """
    def __init__(self):
        self.compatibility_mappings = {
            'legacy_spatial_output': 'spatial_features',
            'legacy_generative_output': 'generative_features',
            'legacy_prediction': 'final_predictions'
        }

    def convert_legacy_output(self, legacy_output: Dict[str, Any]) -> Stage2Output:
        """Convert legacy Stage 2 output to standard format"""

        # Default values
        spatial_features = torch.zeros(1, 256)
        generative_features = torch.zeros(1, 256)
        fused_features = torch.zeros(1, 256)
        spatial_predictions = torch.tensor([0.5])
        generative_predictions = torch.tensor([0.5])
        final_predictions = torch.tensor([0.5])
        confidence_scores = torch.tensor([0.5, 0.5])

        # Map legacy fields
        for legacy_key, standard_key in self.compatibility_mappings.items():
            if legacy_key in legacy_output:
                if standard_key == 'spatial_features':
                    spatial_features = legacy_output[legacy_key]
                elif standard_key == 'generative_features':
                    generative_features = legacy_output[legacy_key]
                elif standard_key == 'final_predictions':
                    final_predictions = legacy_output[legacy_key]

        return Stage2Output(
            spatial_features=spatial_features,
            generative_features=generative_features,
            fused_features=fused_features,
            spatial_predictions=spatial_predictions,
            generative_predictions=generative_predictions,
            final_predictions=final_predictions,
            confidence_scores=confidence_scores
        )


def create_integration_hub(integration_level: str = "hybrid",
                         temporal_mode: str = "frame_sequence",
                         max_sequence_length: int = 16) -> TemporalIntegrationHub:
    """
    Factory function to create integration hub
    """
    config = IntegrationConfig(
        integration_level=IntegrationLevel(integration_level),
        temporal_mode=TemporalMode(temporal_mode),
        max_sequence_length=max_sequence_length,
        feature_alignment=True,
        backward_compatibility=True,
        enable_caching=True
    )

    return TemporalIntegrationHub(config)


# Example usage for Stage 3 temporal expert
class ExampleTemporalExpert:
    """
    Example implementation of temporal expert for demonstration
    """
    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim

        # Simple LSTM for temporal modeling
        self.temporal_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def process_temporal_sequence(self,
                                temporal_input: TemporalInput,
                                stage2_output: Stage2Output) -> ExpertOutput:
        """Process temporal sequence with Stage 2 context"""

        # Use Stage 2 fused features as input
        features = stage2_output.fused_features.unsqueeze(0)  # Add sequence dimension

        # LSTM processing
        lstm_output, _ = self.temporal_encoder(features)

        # Classification
        prediction = self.classifier(lstm_output[:, -1])  # Use last hidden state

        return ExpertOutput(
            predictions={'classification': prediction},
            features={'temporal_features': lstm_output},
            confidence=float(prediction.max()),
            losses={}
        )

    def extract_temporal_features(self, temporal_input: TemporalInput) -> torch.Tensor:
        """Extract temporal features from input sequence"""
        B, T, C, H, W = temporal_input.frames.shape

        # Simple temporal pooling (placeholder)
        temporal_features = temporal_input.frames.mean(dim=1).flatten(1)

        return temporal_features

    def get_temporal_attention(self, temporal_input: TemporalInput) -> torch.Tensor:
        """Get temporal attention weights"""
        T = temporal_input.frames.size(1)

        # Uniform attention as placeholder
        attention_weights = torch.ones(1, T) / T

        return attention_weights