"""
AWARE-NET Stage 02: EfficientNetV2-B0 Spatial Artifact Expert

This module implements the spatial artifact detection expert using EfficientNetV2-B0
with specialized optimizations for detecting spatial domain forgeries, edge blending
artifacts, and texture inconsistencies.

Design Principles:
- Flexible multi-resolution input processing (224x224 to 320x320)
- Spatial artifact-specific data augmentation strategies
- Grad-CAM visualization for spatial attention analysis
- Graduated learning rates for backbone and classifier
- Professional domain specialization validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import timm
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .unified_feature_extractor import BaseExpert, ExpertType, ResolutionMode, ExpertOutput, AdaptiveInputProcessor

class SpatialArtifactType(Enum):
    """Types of spatial artifacts to detect."""
    EDGE_BLENDING = "edge_blending"           # Face swap edge inconsistencies
    TEXTURE_MISMATCH = "texture_mismatch"     # Texture/lighting inconsistencies
    SPATIAL_FREQUENCY = "spatial_frequency"   # Frequency domain artifacts
    COMPRESSION_ARTIFACTS = "compression"     # Compression-related artifacts
    GEOMETRIC_DISTORTION = "geometric"       # Geometric inconsistencies

@dataclass
class SpatialExpertConfig:
    """Configuration for EfficientNetV2-B0 spatial expert."""
    # Model architecture
    model_name: str = "efficientnetv2_rw_s"      # Start with smaller model for concept validation
    pretrained: bool = True
    num_classes: int = 1                          # Binary classification
    dropout_rate: float = 0.2

    # Multi-resolution support
    supported_resolutions: List[int] = None       # [224, 256, 288, 320]
    default_resolution: int = 256
    adaptive_resolution: bool = True

    # Spatial specialization
    spatial_attention: bool = True                # Enable spatial attention mechanisms
    edge_enhancement: bool = True                 # Edge detection enhancement
    texture_analysis: bool = True                 # Texture consistency analysis
    frequency_analysis: bool = False              # Frequency domain analysis (optional)

    # Training optimization
    graduated_learning_rates: bool = True        # Different LR for backbone vs classifier
    backbone_lr_multiplier: float = 0.1          # Backbone LR = base_lr * 0.1
    focal_loss: bool = True                       # Handle hard examples
    label_smoothing: float = 0.1                 # Regularization

    # Inference optimization
    enable_mixed_precision: bool = True
    enable_grad_cam: bool = True                  # Attention visualization
    batch_size_optimization: bool = True

    def __post_init__(self):
        if self.supported_resolutions is None:
            self.supported_resolutions = [224, 256, 288, 320]

class SpatialAttentionModule(nn.Module):
    """Spatial attention module for focusing on artifact regions."""

    def __init__(self, in_channels: int, reduction: int = 8):
        super(SpatialAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x

class EdgeEnhancementModule(nn.Module):
    """Edge detection and enhancement for spatial artifact detection."""

    def __init__(self, in_channels: int):
        super(EdgeEnhancementModule, self).__init__()

        # Sobel edge detection filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))

        self.edge_conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply Sobel filters
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))

        # Combine edge information
        edges = torch.cat([edge_x, edge_y], dim=1)
        edges = self.relu(self.bn(self.edge_conv(edges)))

        return edges

class TextureAnalysisModule(nn.Module):
    """Texture consistency analysis module."""

    def __init__(self, in_channels: int, patch_size: int = 8):
        super(TextureAnalysisModule, self).__init__()
        self.patch_size = patch_size

        # Local texture descriptors
        self.texture_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        # Texture consistency scoring
        self.consistency_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels // 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        texture_features = self.texture_conv(x)
        consistency_score = self.consistency_head(texture_features)

        return texture_features, consistency_score

class EfficientNetV2SpatialExpert(BaseExpert):
    """
    EfficientNetV2-B0 based spatial artifact detection expert.

    This expert specializes in detecting spatial domain forgeries including:
    - Edge blending artifacts in face swaps
    - Texture and lighting inconsistencies
    - Spatial frequency anomalies
    - Geometric distortions
    """

    def __init__(self, config: SpatialExpertConfig):
        super().__init__(ExpertType.SPATIAL, config.__dict__)
        self.config = config

        # Initialize adaptive input processor
        resolution_modes = [ResolutionMode(res) for res in config.supported_resolutions
                          if res in [mode.value for mode in ResolutionMode]]
        self.input_processor = AdaptiveInputProcessor(resolution_modes)

        # Model components will be initialized in _build_model
        self.backbone = None
        self.spatial_attention = None
        self.edge_enhancement = None
        self.texture_analysis = None
        self.classifier = None

        # Grad-CAM for visualization
        self.grad_cam_hooks = []
        self.grad_cam_enabled = config.enable_grad_cam

        # Build model
        self._build_model()

    def _build_model(self):
        """Build the spatial expert model architecture."""
        # Load EfficientNetV2 backbone
        self.backbone = timm.create_model(
            self.config.model_name,
            pretrained=self.config.pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=self.config.dropout_rate
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.config.default_resolution, self.config.default_resolution)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Add spatial specialization modules
        if self.config.spatial_attention:
            self.spatial_attention = SpatialAttentionModule(feature_dim)

        if self.config.edge_enhancement:
            self.edge_enhancement = EdgeEnhancementModule(3)  # Input channels

        if self.config.texture_analysis:
            self.texture_analysis = TextureAnalysisModule(feature_dim)

        # Classification head
        classifier_layers = []

        # Add texture analysis features if enabled
        classifier_input_dim = feature_dim
        if self.config.texture_analysis:
            classifier_input_dim += feature_dim // 4  # From texture analysis

        classifier_layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.config.num_classes)
        ])

        self.classifier = nn.Sequential(*classifier_layers)

        # Setup Grad-CAM hooks if enabled
        if self.grad_cam_enabled:
            self._setup_grad_cam_hooks()

    def _setup_grad_cam_hooks(self):
        """Setup hooks for Grad-CAM visualization."""
        self.gradients = []
        self.activations = []

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            self.activations.append(output)

        # Hook to the last convolutional layer
        target_layer = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module

        if target_layer is not None:
            self.grad_cam_hooks.append(target_layer.register_forward_hook(forward_hook))
            self.grad_cam_hooks.append(target_layer.register_backward_hook(backward_hook))

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load the expert model from checkpoint."""
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            print(f"Loaded spatial expert from {checkpoint_path}")

        self.is_loaded = True

    def preprocess_input(self,
                        images: torch.Tensor,
                        target_resolution: ResolutionMode = ResolutionMode.STANDARD) -> torch.Tensor:
        """
        Preprocess input images for spatial expert.

        Args:
            images: Input tensor [B, C, H, W]
            target_resolution: Target resolution mode

        Returns:
            Processed tensor [B, C, target_size, target_size]
        """
        # Apply adaptive input processing
        processed = self.input_processor.process_batch(
            images, target_resolution, preserve_aspect_ratio=True
        )

        # Normalize to ImageNet statistics (EfficientNet requirement)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(processed.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(processed.device)
        processed = (processed - mean) / std

        return processed

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from preprocessed images."""
        # Edge enhancement (applied to input if enabled)
        if self.config.edge_enhancement and self.edge_enhancement is not None:
            # Apply edge enhancement to input
            enhanced_images = images + 0.1 * self.edge_enhancement(images)
        else:
            enhanced_images = images

        # Extract backbone features
        features = self.backbone(enhanced_images)

        # Apply spatial attention if enabled
        if self.config.spatial_attention and self.spatial_attention is not None:
            features = self.spatial_attention(features)

        return features

    def predict(self, images: torch.Tensor, return_features: bool = True) -> ExpertOutput:
        """
        Generate predictions with the spatial expert.

        Args:
            images: Preprocessed input tensor [B, C, H, W]
            return_features: Whether to return feature representations

        Returns:
            ExpertOutput with predictions and features
        """
        batch_size = images.shape[0]
        device = images.device

        # Clear Grad-CAM data
        if self.grad_cam_enabled:
            self.gradients.clear()
            self.activations.clear()

        # Extract features
        features = self.extract_features(images)

        # Texture analysis if enabled
        texture_features = None
        texture_consistency = None
        if self.config.texture_analysis and self.texture_analysis is not None:
            texture_features, texture_consistency = self.texture_analysis(features)
            # Concatenate texture features with main features for classification
            pooled_texture = F.adaptive_avg_pool2d(texture_features, 1).flatten(1)
            pooled_main = F.adaptive_avg_pool2d(features, 1).flatten(1)
            classifier_input = torch.cat([pooled_main, pooled_texture], dim=1)
        else:
            classifier_input = F.adaptive_avg_pool2d(features, 1).flatten(1)

        # Classification
        logits = self.classifier(classifier_input)

        # Convert to probabilities
        probabilities = torch.sigmoid(logits)

        # Generate binary predictions (threshold = 0.5)
        predictions = (probabilities > 0.5).float()

        # Confidence scores (distance from decision boundary)
        confidence = torch.abs(probabilities - 0.5) * 2

        # Prepare output
        output = ExpertOutput(
            logits=logits.squeeze(-1),
            probabilities=probabilities.squeeze(-1),
            predictions=predictions.squeeze(-1),
            confidence=confidence.squeeze(-1),
            features=F.adaptive_avg_pool2d(features, 1).flatten(1) if return_features else None,
            normalized_features=F.normalize(F.adaptive_avg_pool2d(features, 1).flatten(1), p=2, dim=1) if return_features else None,
            expert_type=ExpertType.SPATIAL,
            resolution=ResolutionMode.STANDARD  # Will be updated by caller
        )

        # Add spatial-specific outputs
        if self.grad_cam_enabled and len(self.activations) > 0:
            output.attention_maps = self._generate_grad_cam(logits)

        return output

    def _generate_grad_cam(self, logits: torch.Tensor) -> torch.Tensor:
        """Generate Grad-CAM attention maps."""
        if not self.gradients or not self.activations:
            return None

        # Get gradients and activations
        gradients = self.gradients[-1]  # Most recent
        activations = self.activations[-1]

        # Compute weights
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam / (torch.max(cam.view(cam.size(0), -1), dim=1, keepdim=True)[0].view(cam.size(0), 1, 1, 1) + 1e-8)

        return cam

    def get_spatial_artifact_scores(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get specialized scores for different types of spatial artifacts.

        Args:
            images: Preprocessed input tensor

        Returns:
            Dictionary mapping artifact types to detection scores
        """
        features = self.extract_features(images)

        scores = {}

        # Edge blending score (based on edge enhancement if available)
        if self.config.edge_enhancement and self.edge_enhancement is not None:
            edge_features = self.edge_enhancement(images)
            edge_score = torch.mean(torch.abs(edge_features), dim=[1, 2, 3])
            scores[SpatialArtifactType.EDGE_BLENDING.value] = edge_score

        # Texture consistency score
        if self.config.texture_analysis and self.texture_analysis is not None:
            _, texture_consistency = self.texture_analysis(features)
            scores[SpatialArtifactType.TEXTURE_MISMATCH.value] = 1.0 - texture_consistency.squeeze()

        # Overall spatial anomaly score
        spatial_anomaly = torch.var(features, dim=[2, 3]).mean(dim=1)
        scores[SpatialArtifactType.SPATIAL_FREQUENCY.value] = spatial_anomaly

        return scores

    def analyze_resolution_performance(self,
                                     images: torch.Tensor,
                                     resolutions: Optional[List[int]] = None) -> Dict[int, ExpertOutput]:
        """
        Analyze performance across different input resolutions.

        Args:
            images: Input images at original resolution
            resolutions: List of resolutions to test

        Returns:
            Dictionary mapping resolutions to expert outputs
        """
        if resolutions is None:
            resolutions = self.config.supported_resolutions

        results = {}

        for resolution in resolutions:
            try:
                resolution_mode = ResolutionMode(resolution)
                processed_images = self.preprocess_input(images, resolution_mode)
                output = self.predict(processed_images, return_features=True)
                output.resolution = resolution_mode
                results[resolution] = output
            except ValueError:
                # Skip unsupported resolutions
                continue

        return results

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration for training."""
        config = {
            'model_parameters': {
                'backbone': list(self.backbone.parameters()),
                'classifier': list(self.classifier.parameters()),
                'spatial_modules': []
            },
            'learning_rates': {
                'backbone': 1e-4 * self.config.backbone_lr_multiplier,
                'classifier': 1e-4,
                'spatial_modules': 1e-4
            },
            'loss_config': {
                'focal_loss': self.config.focal_loss,
                'label_smoothing': self.config.label_smoothing
            }
        }

        # Add spatial module parameters
        if self.spatial_attention is not None:
            config['model_parameters']['spatial_modules'].extend(list(self.spatial_attention.parameters()))
        if self.edge_enhancement is not None:
            config['model_parameters']['spatial_modules'].extend(list(self.edge_enhancement.parameters()))
        if self.texture_analysis is not None:
            config['model_parameters']['spatial_modules'].extend(list(self.texture_analysis.parameters()))

        return config

    def cleanup_grad_cam_hooks(self):
        """Clean up Grad-CAM hooks."""
        for hook in self.grad_cam_hooks:
            hook.remove()
        self.grad_cam_hooks.clear()

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'grad_cam_hooks'):
            self.cleanup_grad_cam_hooks()

# Factory functions
def create_spatial_expert(config: Optional[SpatialExpertConfig] = None) -> EfficientNetV2SpatialExpert:
    """Factory function to create spatial expert."""
    if config is None:
        config = SpatialExpertConfig()

    return EfficientNetV2SpatialExpert(config)

def create_spatial_expert_from_dict(config_dict: Dict[str, Any]) -> EfficientNetV2SpatialExpert:
    """Create spatial expert from configuration dictionary."""
    config = SpatialExpertConfig(**config_dict)
    return EfficientNetV2SpatialExpert(config)

# Export interface
__all__ = [
    'SpatialArtifactType', 'SpatialExpertConfig',
    'SpatialAttentionModule', 'EdgeEnhancementModule', 'TextureAnalysisModule',
    'EfficientNetV2SpatialExpert',
    'create_spatial_expert', 'create_spatial_expert_from_dict'
]