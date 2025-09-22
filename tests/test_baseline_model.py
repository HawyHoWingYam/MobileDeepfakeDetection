#!/usr/bin/env python3
"""
Test baseline model architecture and functionality
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stage_00.baseline_model import EfficientNetV2B3Baseline

class TestBaselineModel:
    """Test cases for EfficientNetV2B3Baseline model"""

    def test_model_instantiation_b0(self):
        """Test model can be instantiated with B0 variant"""
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            model_name='tf_efficientnetv2_b0'
        )
        assert model is not None
        assert model.num_classes == 1

    def test_model_instantiation_b3(self):
        """Test model can be instantiated with B3 variant"""
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            model_name='tf_efficientnetv2_b3'
        )
        assert model is not None
        assert model.num_classes == 1

    def test_unsupported_model_variant(self):
        """Test that unsupported model variants raise error"""
        with pytest.raises(ValueError):
            EfficientNetV2B3Baseline(
                num_classes=1,
                pretrained=False,
                model_name='unsupported_model'
            )

    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape"""
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            model_name='tf_efficientnetv2_b0'
        )

        # Test single sample
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_forward_pass_batch(self):
        """Test forward pass with batch input"""
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            model_name='tf_efficientnetv2_b0'
        )

        # Test batch
        batch_size = 4
        x = torch.randn(batch_size, 3, 256, 256)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1)
        assert torch.isfinite(output).all()

    def test_model_training_mode(self):
        """Test model can switch between train/eval modes"""
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            model_name='tf_efficientnetv2_b0'
        )

        # Test training mode
        model.train()
        assert model.training == True

        # Test eval mode
        model.eval()
        assert model.training == False

    def test_model_parameters_require_grad(self):
        """Test model parameters require gradients by default"""
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            model_name='tf_efficientnetv2_b0'
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        assert trainable_params > 0, "No trainable parameters found"
        assert trainable_params == total_params, "Some parameters frozen unexpectedly"

    def test_model_freeze_backbone(self):
        """Test backbone freezing functionality"""
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            freeze_backbone=True,
            model_name='tf_efficientnetv2_b0'
        )

        # Check that backbone parameters are frozen
        backbone_params_frozen = all(not p.requires_grad for p in model.backbone.parameters())

        # Check that classifier parameters are still trainable
        classifier_params_trainable = all(p.requires_grad for p in model.classifier.parameters())

        assert backbone_params_frozen, "Backbone parameters not frozen"
        assert classifier_params_trainable, "Classifier parameters frozen unexpectedly"

    def test_model_dropout_configuration(self):
        """Test dropout rate configuration"""
        dropout_rate = 0.3
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            dropout_rate=dropout_rate,
            model_name='tf_efficientnetv2_b0'
        )

        assert model.dropout_rate == dropout_rate

    def test_model_feature_extraction(self):
        """Test feature extraction functionality"""
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            model_name='tf_efficientnetv2_b0'
        )

        x = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            features = model.extract_features(x)

        # Features should have correct batch dimension
        assert features.shape[0] == 2
        # Features should be 1D after global pooling
        assert len(features.shape) == 2
        # Features should be finite
        assert torch.isfinite(features).all()

if __name__ == "__main__":
    pytest.main([__file__])