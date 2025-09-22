"""
AWARE-NET Stage 1 → Stage 2 Integration Interface

This module defines the standard interface for integrating Stage 1 rapid filter
with Stage 2 heterogeneous expert models. It provides clean abstractions for
cascade system communication and feature passing.

Key Components:
- Standardized feature extraction interface
- Cascade decision routing protocols
- Performance monitoring and statistics
- Artifact packaging for Stage 2 handoff
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
import pickle
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Stage1Output:
    """Standardized output from Stage 1 rapid filter."""
    sample_id: str
    decision: str  # 'accept', 'reject', 'next_stage'
    confidence: float
    raw_probability: float
    calibrated_probability: float
    features: Optional[np.ndarray] = None
    projections: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.features is not None:
            result['features'] = self.features.tolist()
        if self.projections is not None:
            result['projections'] = self.projections.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'Stage1Output':
        """Create from dictionary."""
        if 'features' in data and data['features'] is not None:
            data['features'] = np.array(data['features'])
        if 'projections' in data and data['projections'] is not None:
            data['projections'] = np.array(data['projections'])
        return cls(**data)


@dataclass
class CascadeStatistics:
    """Statistics for cascade system monitoring."""
    total_samples: int
    accepted_locally: int
    rejected_locally: int
    sent_to_stage2: int
    acceptance_rate: float
    rejection_rate: float
    filter_rate: float
    average_processing_time_ms: float
    confidence_distribution: Dict[str, float]
    performance_metrics: Dict[str, float]


class Stage1Interface(ABC):
    """Abstract interface for Stage 1 models."""

    @abstractmethod
    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray],
        return_features: bool = False,
        return_projections: bool = False
    ) -> List[Stage1Output]:
        """Make predictions with cascade decisions."""
        pass

    @abstractmethod
    def extract_features(
        self,
        inputs: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """Extract features for Stage 2."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get model information and metadata."""
        pass


class Stage1RapidFilter(Stage1Interface):
    """
    Production implementation of Stage 1 rapid filter.

    Combines MobileNetV4 model, temperature calibration, and conservative
    threshold strategy into a unified interface for cascade system integration.
    """

    def __init__(
        self,
        model: nn.Module,
        temperature_scaler: Optional[nn.Module] = None,
        threshold_strategy: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Stage 1 rapid filter.

        Args:
            model: Trained MobileNetV4 model
            temperature_scaler: Calibrated temperature scaling module
            threshold_strategy: Conservative threshold strategy
            device: Computation device
        """
        self.model = model
        self.temperature_scaler = temperature_scaler
        self.threshold_strategy = threshold_strategy
        self.device = device or torch.device('cpu')

        self.model.to(self.device)
        if self.temperature_scaler:
            self.temperature_scaler.to(self.device)

        self.processing_stats = {
            'total_predictions': 0,
            'total_processing_time': 0.0,
            'cascade_decisions': {'accept': 0, 'reject': 0, 'next_stage': 0}
        }

        logger.info("Stage1RapidFilter initialized")

    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray],
        return_features: bool = False,
        return_projections: bool = False,
        sample_ids: Optional[List[str]] = None
    ) -> List[Stage1Output]:
        """
        Make predictions with cascade decisions.

        Args:
            inputs: Input tensor or array
            return_features: Whether to include backbone features
            return_projections: Whether to include projection head outputs
            sample_ids: Optional sample identifiers

        Returns:
            List of Stage1Output objects
        """
        import time

        start_time = time.time()

        # Convert inputs to tensor
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()

        inputs = inputs.to(self.device)
        batch_size = inputs.shape[0]

        # Generate sample IDs if not provided
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(batch_size)]

        self.model.eval()
        outputs = []

        with torch.no_grad():
            # Get model outputs
            model_outputs = self.model(
                inputs,
                return_features=return_features,
                return_projections=return_projections
            )

            if isinstance(model_outputs, dict):
                logits = model_outputs['logits']
                features = model_outputs.get('features')
                projections = model_outputs.get('projections')
            else:
                logits = model_outputs
                features = None
                projections = None

            # Apply temperature scaling if available
            if self.temperature_scaler:
                calibrated_probs = self.temperature_scaler(logits)
            else:
                calibrated_probs = torch.softmax(logits, dim=1)

            raw_probs = torch.softmax(logits, dim=1)

            # Process each sample
            for i in range(batch_size):
                sample_raw_prob = raw_probs[i, 1].item()  # Positive class probability
                sample_calibrated_prob = calibrated_probs[i, 1].item()

                # Make cascade decision
                if self.threshold_strategy:
                    decision_obj = self.threshold_strategy._make_single_decision(
                        sample_calibrated_prob
                    )
                    decision = decision_obj.stage_routing
                    confidence = decision_obj.confidence
                else:
                    # Default conservative strategy
                    if sample_calibrated_prob >= 0.99:
                        decision = 'accept'
                        confidence = sample_calibrated_prob
                    elif sample_calibrated_prob <= 0.01:
                        decision = 'reject'
                        confidence = 1.0 - sample_calibrated_prob
                    else:
                        decision = 'next_stage'
                        confidence = 0.5

                # Extract features if requested
                sample_features = None
                if return_features and features is not None:
                    sample_features = features[i].cpu().numpy()

                sample_projections = None
                if return_projections and projections is not None:
                    sample_projections = projections[i].cpu().numpy()

                # Create output object
                output = Stage1Output(
                    sample_id=sample_ids[i],
                    decision=decision,
                    confidence=confidence,
                    raw_probability=sample_raw_prob,
                    calibrated_probability=sample_calibrated_prob,
                    features=sample_features,
                    projections=sample_projections,
                    processing_time_ms=0.0,  # Will be updated after batch processing
                    metadata={
                        'temperature_scaled': self.temperature_scaler is not None,
                        'threshold_strategy_applied': self.threshold_strategy is not None
                    }
                )

                outputs.append(output)

                # Update cascade decision stats
                self.processing_stats['cascade_decisions'][decision] += 1

        # Update timing statistics
        total_time = time.time() - start_time
        time_per_sample = (total_time * 1000) / batch_size  # Convert to ms

        for output in outputs:
            output.processing_time_ms = time_per_sample

        # Update global stats
        self.processing_stats['total_predictions'] += batch_size
        self.processing_stats['total_processing_time'] += total_time

        return outputs

    def extract_features(
        self,
        inputs: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """Extract backbone features for Stage 2."""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()

        inputs = inputs.to(self.device)

        self.model.eval()
        with torch.no_grad():
            features = self.model.get_features(inputs)
            return features.cpu().numpy()

    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        base_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}

        return {
            **base_info,
            'stage': 'stage_01',
            'role': 'rapid_filter',
            'cascade_integration': True,
            'temperature_calibrated': self.temperature_scaler is not None,
            'threshold_strategy_configured': self.threshold_strategy is not None,
            'processing_stats': self.processing_stats.copy(),
            'interface_version': '1.0.0'
        }

    def get_cascade_statistics(self) -> CascadeStatistics:
        """Get detailed cascade performance statistics."""
        total = self.processing_stats['total_predictions']
        if total == 0:
            return CascadeStatistics(
                total_samples=0,
                accepted_locally=0,
                rejected_locally=0,
                sent_to_stage2=0,
                acceptance_rate=0.0,
                rejection_rate=0.0,
                filter_rate=0.0,
                average_processing_time_ms=0.0,
                confidence_distribution={},
                performance_metrics={}
            )

        decisions = self.processing_stats['cascade_decisions']
        avg_time = (self.processing_stats['total_processing_time'] * 1000) / total

        return CascadeStatistics(
            total_samples=total,
            accepted_locally=decisions['accept'],
            rejected_locally=decisions['reject'],
            sent_to_stage2=decisions['next_stage'],
            acceptance_rate=decisions['accept'] / total,
            rejection_rate=decisions['reject'] / total,
            filter_rate=decisions['next_stage'] / total,
            average_processing_time_ms=avg_time,
            confidence_distribution={
                'high_confidence_accept': decisions['accept'] / total,
                'high_confidence_reject': decisions['reject'] / total,
                'uncertain': decisions['next_stage'] / total
            },
            performance_metrics={
                'throughput_samples_per_second': total / (self.processing_stats['total_processing_time'] + 1e-8),
                'cascade_efficiency': (decisions['accept'] + decisions['reject']) / total,
                'filter_efficiency': decisions['next_stage'] / total
            }
        )


class Stage1ArtifactPackager:
    """
    Package Stage 1 artifacts for Stage 2 integration.
    """

    @staticmethod
    def package_artifacts(
        model: nn.Module,
        temperature_scaler: Optional[nn.Module],
        threshold_strategy: Optional[Any],
        model_config: Dict,
        calibration_results: Dict,
        performance_stats: Dict,
        output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """
        Package all Stage 1 artifacts for handoff to Stage 2.

        Args:
            model: Trained model
            temperature_scaler: Calibration module
            threshold_strategy: Threshold strategy
            model_config: Model configuration
            calibration_results: Calibration results
            performance_stats: Performance statistics
            output_dir: Output directory for artifacts

        Returns:
            Dictionary mapping artifact types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_paths = {}

        # 1. Model weights
        model_path = output_dir / "stage1_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }, model_path)
        artifact_paths['model'] = str(model_path)

        # 2. Temperature scaler
        if temperature_scaler:
            temp_path = output_dir / "stage1_temperature_scaler.pth"
            torch.save({
                'temperature_scaler_state_dict': temperature_scaler.state_dict(),
                'temperature_value': temperature_scaler.get_temperature()
            }, temp_path)
            artifact_paths['temperature_scaler'] = str(temp_path)

        # 3. Threshold strategy
        if threshold_strategy:
            threshold_path = output_dir / "stage1_threshold_strategy.pkl"
            with open(threshold_path, 'wb') as f:
                pickle.dump(threshold_strategy, f)
            artifact_paths['threshold_strategy'] = str(threshold_path)

        # 4. Calibration results
        calibration_path = output_dir / "stage1_calibration_results.json"
        with open(calibration_path, 'w') as f:
            json.dump(calibration_results, f, indent=2, default=str)
        artifact_paths['calibration_results'] = str(calibration_path)

        # 5. Performance statistics
        performance_path = output_dir / "stage1_performance_stats.json"
        with open(performance_path, 'w') as f:
            json.dump(performance_stats, f, indent=2, default=str)
        artifact_paths['performance_stats'] = str(performance_path)

        # 6. Integration metadata
        integration_metadata = {
            'stage': 'stage_01',
            'version': '1.0.0',
            'artifacts': artifact_paths,
            'interface_specs': {
                'input_format': 'torch.Tensor or np.ndarray',
                'output_format': 'List[Stage1Output]',
                'feature_dimension': model.feature_dim if hasattr(model, 'feature_dim') else 'unknown',
                'projection_dimension': model.projection_dim if hasattr(model, 'projection_dim') else 'unknown'
            },
            'cascade_specifications': {
                'high_confidence_threshold': threshold_strategy.config.high_confidence_threshold if threshold_strategy else 0.99,
                'low_confidence_threshold': threshold_strategy.config.low_confidence_threshold if threshold_strategy else 0.01,
                'expected_filter_rate': threshold_strategy.config.target_filter_rate if threshold_strategy else 0.90
            },
            'performance_requirements': {
                'max_inference_time_ms': 50,
                'max_memory_gb': 2,
                'min_filter_rate': 0.90,
                'max_false_negative_rate': 0.05
            }
        }

        metadata_path = output_dir / "stage1_integration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(integration_metadata, f, indent=2, default=str)
        artifact_paths['integration_metadata'] = str(metadata_path)

        logger.info(f"Stage 1 artifacts packaged in {output_dir}")
        logger.info(f"Artifact files: {list(artifact_paths.keys())}")

        return artifact_paths

    @staticmethod
    def load_packaged_artifacts(artifact_dir: Union[str, Path]) -> Dict:
        """
        Load packaged Stage 1 artifacts.

        Args:
            artifact_dir: Directory containing packaged artifacts

        Returns:
            Dictionary containing loaded artifacts
        """
        artifact_dir = Path(artifact_dir)

        # Load integration metadata first
        metadata_path = artifact_dir / "stage1_integration_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        artifacts = {'metadata': metadata}

        # Load model (requires separate instantiation)
        model_path = artifact_dir / "stage1_model.pth"
        if model_path.exists():
            artifacts['model_checkpoint'] = torch.load(model_path, map_location='cpu')

        # Load temperature scaler
        temp_path = artifact_dir / "stage1_temperature_scaler.pth"
        if temp_path.exists():
            artifacts['temperature_scaler_checkpoint'] = torch.load(temp_path, map_location='cpu')

        # Load threshold strategy
        threshold_path = artifact_dir / "stage1_threshold_strategy.pkl"
        if threshold_path.exists():
            with open(threshold_path, 'rb') as f:
                artifacts['threshold_strategy'] = pickle.load(f)

        # Load calibration results
        calibration_path = artifact_dir / "stage1_calibration_results.json"
        if calibration_path.exists():
            with open(calibration_path, 'r') as f:
                artifacts['calibration_results'] = json.load(f)

        # Load performance stats
        performance_path = artifact_dir / "stage1_performance_stats.json"
        if performance_path.exists():
            with open(performance_path, 'r') as f:
                artifacts['performance_stats'] = json.load(f)

        logger.info(f"Loaded Stage 1 artifacts from {artifact_dir}")

        return artifacts


def test_integration_interface():
    """Test Stage 1 integration interface."""
    print("Testing Stage 1 Integration Interface...")

    # Create mock components for testing
    from .mobilenetv4_model import MobileNetV4SupCon
    from .temperature_scaling import TemperatureScaling
    from .cascade_strategy import ConservativeThresholdStrategy

    # Initialize components
    model = MobileNetV4SupCon(pretrained=False, projection_dim=128)
    temp_scaler = TemperatureScaling()
    threshold_strategy = ConservativeThresholdStrategy()

    print("✓ Components initialized")

    # Create rapid filter
    rapid_filter = Stage1RapidFilter(
        model=model,
        temperature_scaler=temp_scaler,
        threshold_strategy=threshold_strategy
    )

    print("✓ Rapid filter created")

    # Test prediction
    test_input = torch.randn(4, 3, 256, 256)
    outputs = rapid_filter.predict(
        test_input,
        return_features=True,
        return_projections=True
    )

    print(f"✓ Prediction test: {len(outputs)} outputs")
    for i, output in enumerate(outputs[:2]):
        print(f"  Sample {i}: {output.decision}, conf={output.confidence:.3f}")

    # Test feature extraction
    features = rapid_filter.extract_features(test_input)
    print(f"✓ Feature extraction: shape {features.shape}")

    # Test statistics
    stats = rapid_filter.get_cascade_statistics()
    print(f"✓ Statistics: {stats.total_samples} samples processed")

    # Test artifact packaging
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_paths = Stage1ArtifactPackager.package_artifacts(
            model=model,
            temperature_scaler=temp_scaler,
            threshold_strategy=threshold_strategy,
            model_config={'test': True},
            calibration_results={'ece': 0.05},
            performance_stats={'auc': 0.95},
            output_dir=temp_dir
        )

        print(f"✓ Artifacts packaged: {len(artifact_paths)} files")

        # Test loading
        loaded_artifacts = Stage1ArtifactPackager.load_packaged_artifacts(temp_dir)
        print(f"✓ Artifacts loaded: {len(loaded_artifacts)} items")

    print("Integration interface tests passed! ✓")


if __name__ == "__main__":
    test_integration_interface()