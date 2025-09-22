"""
AWARE-NET Stage 1: MobileNetV4-based SupCon Model

This module implements MobileNetV4 architecture with projection head for
supervised contrastive learning in authenticity modeling paradigm.

Architecture:
- MobileNetV4-Hybrid-Medium backbone (pretrained)
- Projection head for contrastive learning (512D)
- Classification head for final prediction
- Temperature scaling support for calibration

Key Features:
- Mobile-friendly inference (<50ms, <2GB memory)
- Dual-head architecture (projection + classification)
- L2 normalization for contrastive learning
- Flexible backbone freezing options
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Tuple, Optional, Union
import logging
import math

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps backbone features to normalized embedding space for SupCon loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout_rate: float = 0.1,
        use_bn: bool = True
    ):
        """
        Initialize projection head.

        Args:
            input_dim: Dimension of input features from backbone
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout_rate: Dropout probability
            use_bn: Whether to use batch normalization
        """
        super().__init__()

        layers = []

        # First projection layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Second projection layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.projection = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize projection head weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            L2-normalized projections (batch_size, output_dim)
        """
        projections = self.projection(x)
        # L2 normalize for contrastive learning
        return F.normalize(projections, dim=1)


class ClassificationHead(nn.Module):
    """
    Classification head for binary authenticity prediction.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        dropout_rate: float = 0.2
    ):
        """
        Initialize classification head.

        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, num_classes)
        )

        # Initialize weights
        nn.init.kaiming_normal_(self.classifier[1].weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.classifier[1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head."""
        return self.classifier(x)


class MobileNetV4SupCon(nn.Module):
    """
    MobileNetV4-based model for supervised contrastive learning.

    This model implements the authenticity modeling paradigm by learning
    a structured feature space where authentic content forms a "truthfulness fortress".
    """

    def __init__(
        self,
        model_name: str = 'mobilenetv4_hybrid_medium',
        pretrained: bool = True,
        num_classes: int = 2,
        projection_dim: int = 512,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        use_projection_head: bool = True
    ):
        """
        Initialize MobileNetV4 SupCon model.

        Args:
            model_name: MobileNetV4 variant to use
            pretrained: Whether to load pretrained weights
            num_classes: Number of classes for classification
            projection_dim: Dimension of projection head output
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone weights
            use_projection_head: Whether to include projection head
        """
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.use_projection_head = use_projection_head

        # Load MobileNetV4 backbone
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='avg'
            )
            logger.info(f"Loaded {model_name} with pretrained={pretrained}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            # Fallback to a known working model
            logger.info("Falling back to mobilenetv3_large_100")
            self.backbone = timm.create_model(
                'mobilenetv3_large_100',
                pretrained=pretrained,
                num_classes=0,
                global_pool='avg'
            )
            self.model_name = 'mobilenetv3_large_100'

        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        logger.info(f"Backbone feature dimension: {self.feature_dim}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")

        # Projection head for contrastive learning
        if use_projection_head:
            self.projection_head = ProjectionHead(
                input_dim=self.feature_dim,
                output_dim=projection_dim,
                dropout_rate=dropout_rate
            )
        else:
            self.projection_head = None

        # Classification head
        self.classification_head = ClassificationHead(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

        # Temperature parameter for calibration
        self.register_parameter(
            'temperature',
            nn.Parameter(torch.ones(1))
        )

        logger.info(f"Model initialized: {model_name}, features={self.feature_dim}, "
                   f"projection={projection_dim if use_projection_head else 'None'}")

    def forward(
        self,
        x: torch.Tensor,
        return_projections: bool = False,
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor (batch_size, channels, height, width)
            return_projections: Whether to return projection head outputs
            return_features: Whether to return backbone features

        Returns:
            Classification logits or dictionary of outputs
        """
        # Extract features from backbone
        features = self.backbone(x)  # (batch_size, feature_dim)

        # Classification logits
        logits = self.classification_head(features)

        # Prepare outputs
        outputs = {'logits': logits}

        if return_features:
            outputs['features'] = features

        if return_projections and self.projection_head is not None:
            projections = self.projection_head(features)
            outputs['projections'] = projections

        # Return single tensor or dictionary based on what's requested
        if not return_projections and not return_features:
            return logits
        else:
            return outputs

    def get_projections(self, x: torch.Tensor) -> torch.Tensor:
        """Get projection head outputs for contrastive learning."""
        if self.projection_head is None:
            raise ValueError("Model was initialized without projection head")

        features = self.backbone(x)
        return self.projection_head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone features."""
        return self.backbone(x)

    def get_calibrated_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get temperature-calibrated predictions."""
        logits = self.forward(x)
        return torch.softmax(logits / self.temperature, dim=1)

    def set_temperature(self, temperature: float):
        """Set temperature parameter for calibration."""
        self.temperature.data.fill_(temperature)
        logger.info(f"Temperature set to {temperature}")

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())

        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'feature_dimension': self.feature_dim,
            'projection_dimension': self.projection_dim,
            'has_projection_head': self.projection_head is not None,
            'temperature': self.temperature.item()
        }


def create_mobilenetv4_supcon(
    model_name: str = 'mobilenetv4_hybrid_medium',
    pretrained: bool = True,
    **kwargs
) -> MobileNetV4SupCon:
    """
    Factory function to create MobileNetV4 SupCon model.

    Args:
        model_name: MobileNetV4 variant name
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for model initialization

    Returns:
        MobileNetV4SupCon model instance
    """
    return MobileNetV4SupCon(
        model_name=model_name,
        pretrained=pretrained,
        **kwargs
    )


def test_mobilenetv4_model():
    """Test function to validate MobileNetV4 model implementation."""
    print("Testing MobileNetV4 SupCon Model...")

    # Test parameters
    batch_size = 4
    input_size = (3, 256, 256)

    # Create test input
    x = torch.randn(batch_size, *input_size)

    # Test model creation
    model = create_mobilenetv4_supcon(
        pretrained=False,  # Faster for testing
        projection_dim=128
    )

    print(f"✓ Model created: {model.model_name}")

    # Test forward pass
    with torch.no_grad():
        # Basic forward pass
        logits = model(x)
        assert logits.shape == (batch_size, 2), f"Expected (4, 2), got {logits.shape}"
        print(f"✓ Basic forward pass: logits shape {logits.shape}")

        # Forward with projections
        outputs = model(x, return_projections=True, return_features=True)
        assert 'logits' in outputs
        assert 'projections' in outputs
        assert 'features' in outputs
        print(f"✓ Multi-output forward: projections shape {outputs['projections'].shape}")

        # Test projection head directly
        projections = model.get_projections(x)
        assert projections.shape == (batch_size, 128)
        # Check L2 normalization
        norms = torch.norm(projections, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        print(f"✓ Projections L2 normalized: {norms[:3]}")

        # Test calibrated predictions
        calibrated = model.get_calibrated_predictions(x)
        assert calibrated.shape == (batch_size, 2)
        # Check probabilities sum to 1
        prob_sums = calibrated.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
        print(f"✓ Calibrated predictions sum to 1: {prob_sums[:3]}")

    # Test model info
    info = model.get_model_info()
    print(f"✓ Model info: {info['total_parameters']:,} parameters")

    # Test backbone freezing
    model.freeze_backbone()
    backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
    assert not any(backbone_grads), "Backbone should be frozen"
    print("✓ Backbone freezing works")

    model.unfreeze_backbone()
    backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
    assert all(backbone_grads), "Backbone should be unfrozen"
    print("✓ Backbone unfreezing works")

    # Test temperature setting
    model.set_temperature(2.0)
    assert abs(model.temperature.item() - 2.0) < 1e-6
    print("✓ Temperature setting works")

    print("All tests passed! ✓")

    return model


if __name__ == "__main__":
    test_mobilenetv4_model()