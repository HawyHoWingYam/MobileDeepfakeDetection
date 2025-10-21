"""
AWARE-NET: MobileNetV4 Model Library

This module implements MobileNetV4 architecture for deepfake detection with
both simple classification and advanced supervised contrastive learning options.

Architecture:
- MobileNetV4-Hybrid-Medium backbone (pretrained)
- Simple classification head for binary classification (Stage 01 requirement)
- Optional projection head for contrastive learning (advanced feature)
- Temperature scaling support for calibration

Key Features:
- Mobile-friendly inference (<50ms, <2GB memory)
- Simple binary classification (perfect for Stage 01 baseline)
- Advanced supervised contrastive learning (optional, for research)
- Flexible backbone freezing options
- Dual implementation modes to meet different requirements
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


class MobileNetV4Simple(nn.Module):
    """
    Simple MobileNetV4 model for binary classification (Stage 01 requirement).

    This is a simplified version focused on standard binary classification
    with BCE loss, perfect for Stage 01 baseline requirements.

    Architecture:
    - MobileNetV4 backbone (pretrained)
    - Simple binary classification head
    - No projection head (simpler than SupCon version)

    Key Features:
    - Mobile-friendly inference (<50ms, <2GB memory)
    - Simple binary classification for fake/real detection
    - Flexible backbone freezing options
    - Direct logits output for BCE loss
    """

    def __init__(
        self,
        model_name: str = 'mobilenetv4_hybrid_medium',
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False
    ):
        """
        Initialize simple MobileNetV4 classifier.

        Args:
            model_name: MobileNetV4 variant to use
            pretrained: Whether to load pretrained weights
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()

        self.model_name = model_name

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

        # Stage 01: CRITICAL FIX - Verify actual feature dimension before creating classifier
        logger.info("=== MOBILENETV4 INITIALIZATION VALIDATION ===")

        # Get expected feature dimension from backbone metadata
        expected_feature_dim = self.backbone.num_features
        logger.info(f"Expected feature dimension (backbone.num_features): {expected_feature_dim}")

        # Verify by performing a test forward pass with dummy input
        logger.info("🔍 Performing test forward pass to verify actual output dimensions...")

        # CRITICAL FIX: Handle BatchNorm with batch_size=1 by temporarily switching to eval mode
        logger.info("⚙️  Switching to eval mode for dimension validation (BatchNorm compatibility)")
        original_training_mode = self.training  # Save original training state

        try:
            self.eval()  # Switch to eval mode to use pretrained BatchNorm statistics
            logger.info(f"   ✅ Model mode: eval (training={self.training})")

            with torch.no_grad():
                # Create dummy input for testing
                dummy_input = torch.randn(1, 3, 224, 224)  # Standard input size
                logger.info(f"   📥 Dummy input shape: {dummy_input.shape}")

                # Perform test forward pass
                logger.info("   🚀 Executing backbone forward pass...")
                dummy_features = self.backbone(dummy_input)
                actual_feature_dim = dummy_features.shape[1]

                logger.info(f"   ✅ Test forward pass successful")
                logger.info(f"   📤 Backbone output shape: {dummy_features.shape}")
                logger.info(f"   📊 Actual feature dimension: {actual_feature_dim}")

        except Exception as e:
            logger.error(f"   ❌ Test forward pass failed: {e}")
            logger.error(f"   🆘 Attempting fallback dimension detection...")

            # Fallback: Use backbone metadata with warning
            actual_feature_dim = expected_feature_dim
            logger.warning(f"   ⚠️  Using backbone.num_features as fallback: {actual_feature_dim}")
            logger.warning(f"   ⚠️  This may cause classifier dimension mismatch!")

        finally:
            # Restore original training mode
            if original_training_mode:
                self.train()
                logger.info(f"   🔄 Restored training mode: {self.training}")
            else:
                logger.info(f"   ✅ Kept eval mode: {self.training}")

        logger.info(f"🎯 Dimension validation completed: Expected={expected_feature_dim}, Actual={actual_feature_dim}")

        # CRITICAL: Use actual dimension, not expected
        if actual_feature_dim != expected_feature_dim:
            logger.warning(f"⚠️  DIMENSION MISMATCH DETECTED during initialization!")
            logger.warning(f"   Expected: {expected_feature_dim}, Actual: {actual_feature_dim}")
            logger.warning(f"   Using actual dimension for classifier initialization")
            self.feature_dim = actual_feature_dim
        else:
            logger.info("✅ Feature dimensions match - using backbone.num_features")
            self.feature_dim = expected_feature_dim

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("🧊 Backbone frozen - only classifier will be trained")
        else:
            logger.info("🔥 Backbone unfrozen - full model will be trained")

        # Create binary classification head with CORRECT dimensions
        logger.info(f"🏗️  Creating classifier with validated dimensions...")
        logger.info(f"   📐 Input dimension: {self.feature_dim}")
        logger.info(f"   🎯 Output dimension: 1 (binary classification)")
        logger.info(f"   🛡️  Dropout rate: {dropout_rate}")

        try:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, 1)  # Single output for BCE
            )
            logger.info(f"   ✅ Classifier layers created successfully")

            # Initialize weights properly with detailed logging
            logger.info("   🎲 Initializing classifier weights...")
            nn.init.kaiming_normal_(self.classifier[1].weight, mode='fan_out', nonlinearity='linear')
            nn.init.constant_(self.classifier[1].bias, 0)
            logger.info(f"   ✅ Weight initialization completed")

            # Verify classifier structure and parameters
            logger.info(f"🔍 Classifier structure verification:")
            logger.info(f"   📦 Total layers: {len(self.classifier)}")

            for i, layer in enumerate(self.classifier):
                layer_type = type(layer).__name__
                if hasattr(layer, 'in_features'):
                    logger.info(f"   Layer {i}: {layer_type} ({layer.in_features} -> {layer.out_features})")
                    if hasattr(layer, 'weight') and layer.weight is not None:
                        weight_stats = {
                            'mean': layer.weight.mean().item(),
                            'std': layer.weight.std().item(),
                            'min': layer.weight.min().item(),
                            'max': layer.weight.max().item()
                        }
                        logger.info(f"      Weight stats: mean={weight_stats['mean']:.4f}, std={weight_stats['std']:.4f}")
                        logger.info(f"      Weight range: [{weight_stats['min']:.4f}, {weight_stats['max']:.4f}]")
                else:
                    logger.info(f"   Layer {i}: {layer_type}")

            if hasattr(self.classifier[1], 'bias') and self.classifier[1].bias is not None:
                bias_mean = self.classifier[1].bias.mean().item()
                logger.info(f"   Bias initialized: {bias_mean:.4f}")

            logger.info(f"✅ Classifier creation and verification completed successfully!")

        except Exception as e:
            logger.error(f"❌ Classifier creation failed: {e}")
            logger.error(f"🆘 This is a critical error - cannot proceed with training")
            raise

        # Final model summary with comprehensive statistics
        logger.info("📋 Generating comprehensive model summary...")

        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            classifier_params = sum(p.numel() for p in self.classifier.parameters())
            backbone_params = sum(p.numel() for p in self.backbone.parameters())
            trainable_backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            trainable_classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)

            # Calculate memory usage estimate
            param_memory_mb = total_params * 4 / (1024**2)  # Assuming float32
            classifier_memory_mb = classifier_params * 4 / (1024**2)

            logger.info(f"📊 COMPREHENSIVE MODEL SUMMARY:")
            logger.info(f"   🏷️  Model Name: {model_name}")
            logger.info(f"   📐 Feature Dimension: {self.feature_dim}")
            logger.info(f"   🔧 Training Mode: {self.training}")

            logger.info(f"   📈 PARAMETER COUNTS:")
            logger.info(f"      Total Parameters: {total_params:,}")
            logger.info(f"      Trainable Parameters: {trainable_params:,}")
            logger.info(f"      Backbone Parameters: {backbone_params:,} (trainable: {trainable_backbone_params:,})")
            logger.info(f"      Classifier Parameters: {classifier_params:,} (trainable: {trainable_classifier_params:,})")

            logger.info(f"   💾 MEMORY ESTIMATES:")
            logger.info(f"      Total Parameters Memory: {param_memory_mb:.2f} MB")
            logger.info(f"      Classifier Memory: {classifier_memory_mb:.2f} MB")

            # Calculate parameter ratios
            classifier_ratio = (classifier_params / total_params) * 100 if total_params > 0 else 0
            trainable_ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0

            logger.info(f"   📊 PARAMETER RATIOS:")
            logger.info(f"      Classifier vs Total: {classifier_ratio:.2f}%")
            logger.info(f"      Trainable vs Total: {trainable_ratio:.2f}%")

            # Model complexity indicators
            logger.info(f"   🎯 TRAINING COMPLEXITY:")
            logger.info(f"      Primary Learning Components: Classifier ({trainable_classifier_params:,} params)")
            if trainable_backbone_params > 0:
                logger.info(f"      Fine-tuning Components: Backbone ({trainable_backbone_params:,} params)")
            else:
                logger.info(f"      Feature Extraction Only: Backbone frozen")

            logger.info("✅ MODEL INITIALIZATION SUCCESSFULLY COMPLETED!")
            logger.info("=== INITIALIZATION VALIDATION FINISHED ===")

        except Exception as e:
            logger.error(f"❌ Model summary generation failed: {e}")
            logger.warning("⚠️  Model may still be functional, but statistics could not be calculated")
            logger.info("=== INITIALIZATION COMPLETED (with warnings) ===")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Single logit (batch_size, 1) for BCE loss
        """
        # Extract features from backbone
        features = self.backbone(x)  # (batch_size, feature_dim)

        # Stage 01: Verify dimensions (should always match now due to initialization fix)
        if not hasattr(self, '_forward_passes_completed'):
            self._forward_passes_completed = 0
            logger.info("🚀 First forward pass - verifying dimensions...")
            logger.info(f"   Input shape: {x.shape}")
            logger.info(f"   Features shape: {features.shape}")
            logger.info(f"   Expected feature_dim: {self.feature_dim}")
            logger.info(f"   Classifier input dim: {self.classifier[1].in_features}")

            if features.shape[1] != self.feature_dim:
                logger.error("🚨 CRITICAL ERROR: Dimension mismatch despite initialization fix!")
                logger.error(f"   features.shape[1]={features.shape[1]} != feature_dim={self.feature_dim}")
                raise RuntimeError(f"Feature dimension mismatch: {features.shape[1]} != {self.feature_dim}")
            else:
                logger.info("✅ Dimensions verified - proceeding with classification")

        self._forward_passes_completed = getattr(self, '_forward_passes_completed', 0) + 1

        # Periodic logging (every 1000 forward passes)
        if self._forward_passes_completed % 1000 == 1:
            logger.info(f"📈 Forward pass #{self._forward_passes_completed} - dimensions stable")

        # Classification logits
        logits = self.classifier(features)  # (batch_size, 1)

        return logits.squeeze(-1)  # (batch_size,) for BCE loss

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone features."""
        return self.backbone(x)

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
            'model_type': 'simple_classification'
        }


def create_mobilenetv4_simple(
    model_name: str = 'mobilenetv4_hybrid_medium',
    pretrained: bool = True,
    **kwargs
) -> MobileNetV4Simple:
    """
    Factory function to create simple MobileNetV4 classifier.

    Stage 01: Simple binary classification model

    Args:
        model_name: MobileNetV4 variant name
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for model initialization

    Returns:
        MobileNetV4Simple model instance
    """
    return MobileNetV4Simple(
        model_name=model_name,
        pretrained=pretrained,
        **kwargs
    )


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
    """Test function to validate both MobileNetV4 model implementations."""
    print("Testing MobileNetV4 Models...")
    print("=" * 50)

    # Test parameters
    batch_size = 4
    input_size = (3, 256, 256)

    # Create test input
    x = torch.randn(batch_size, *input_size)

    # Test 1: Simple Classifier (Stage 01 requirement)
    print("\n1. Testing MobileNetV4Simple (Stage 01)...")
    simple_model = create_mobilenetv4_simple(
        pretrained=False,  # Faster for testing
        dropout_rate=0.1
    )

    print(f"✓ Simple model created: {simple_model.model_name}")

    with torch.no_grad():
        # Test forward pass - single logit for BCE
        logits = simple_model(x)
        assert logits.shape == (batch_size,), f"Expected ({batch_size},), got {logits.shape}"
        print(f"✓ Simple forward pass: logits shape {logits.shape}")

        # Test feature extraction
        features = simple_model.get_features(x)
        assert features.shape == (batch_size, simple_model.feature_dim)
        print(f"✓ Feature extraction: {features.shape}")

        # Test sigmoid range (should be reasonable)
        sigmoid_output = torch.sigmoid(logits)
        assert torch.all(sigmoid_output >= 0) and torch.all(sigmoid_output <= 1)
        print(f"✓ Sigmoid output range: {sigmoid_output[:3]}")

    # Test simple model info
    simple_info = simple_model.get_model_info()
    print(f"✓ Simple model info: {simple_info['total_parameters']:,} parameters")

    # Test 2: SupCon Model (Advanced feature)
    print("\n2. Testing MobileNetV4SupCon (Advanced)...")
    supcon_model = create_mobilenetv4_supcon(
        pretrained=False,  # Faster for testing
        projection_dim=128
    )

    print(f"✓ SupCon model created: {supcon_model.model_name}")

    with torch.no_grad():
        # Basic forward pass
        logits = supcon_model(x)
        assert logits.shape == (batch_size, 2), f"Expected (4, 2), got {logits.shape}"
        print(f"✓ SupCon forward pass: logits shape {logits.shape}")

        # Forward with projections
        outputs = supcon_model(x, return_projections=True, return_features=True)
        assert 'logits' in outputs
        assert 'projections' in outputs
        assert 'features' in outputs
        print(f"✓ Multi-output forward: projections shape {outputs['projections'].shape}")

        # Test projection head directly
        projections = supcon_model.get_projections(x)
        assert projections.shape == (batch_size, 128)
        # Check L2 normalization
        norms = torch.norm(projections, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        print(f"✓ Projections L2 normalized: {norms[:3]}")

        # Test calibrated predictions
        calibrated = supcon_model.get_calibrated_predictions(x)
        assert calibrated.shape == (batch_size, 2)
        # Check probabilities sum to 1
        prob_sums = calibrated.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
        print(f"✓ Calibrated predictions sum to 1: {prob_sums[:3]}")

    # Test SupCon model info
    supcon_info = supcon_model.get_model_info()
    print(f"✓ SupCon model info: {supcon_info['total_parameters']:,} parameters")

    # Test 3: Backbone freezing/unfreezing (both models)
    print("\n3. Testing backbone freezing...")
    for model_name, model in [("Simple", simple_model), ("SupCon", supcon_model)]:
        model.freeze_backbone()
        backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
        assert not any(backbone_grads), f"{model_name} backbone should be frozen"
        print(f"✓ {model_name} backbone freezing works")

        model.unfreeze_backbone()
        backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
        assert all(backbone_grads), f"{model_name} backbone should be unfrozen"
        print(f"✓ {model_name} backbone unfreezing works")

    # Test 4: Temperature setting (SupCon only)
    if hasattr(supcon_model, 'temperature'):
        supcon_model.set_temperature(2.0)
        assert abs(supcon_model.temperature.item() - 2.0) < 1e-6
        print("✓ Temperature setting works")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print(f"Simple model: {simple_info['total_parameters']:,} parameters")
    print(f"SupCon model: {supcon_info['total_parameters']:,} parameters")
    print("=" * 50)

    return simple_model, supcon_model


if __name__ == "__main__":
    test_mobilenetv4_model()