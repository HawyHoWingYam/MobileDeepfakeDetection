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

# Add tools to path for generalization testing
sys.path.insert(0, str(Path(__file__).parent.parent))

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


class TestGeneralization:
    """Test cases for model generalization on real-world datasets"""

    @pytest.mark.slow
    def test_deepfake_eval_generalization(self):
        """Test model generalization on Deepfake-Eval-2024 dataset"""
        try:
            from tools.performance.test_generalization import GeneralizationTester, GeneralizationConfig
        except ImportError:
            pytest.skip("Generalization testing module not available")

        # Test configuration
        model_path = "experiments/test_final_fix_*/checkpoints/best_model.pth"
        dataset_path = "/workspace/Deepfake-Eval-2024"

        # Find the actual model file (handle wildcard)
        import glob
        model_files = glob.glob(model_path)
        if not model_files:
            pytest.skip(f"No model file found at {model_path}")
        model_path = model_files[0]

        if not Path(dataset_path).exists():
            pytest.skip(f"Deepfake-Eval-2024 dataset not found at {dataset_path}")

        # Create config
        config = GeneralizationConfig(
            model_path=model_path,
            dataset_path=dataset_path,
            batch_size=16,  # Smaller batch size for testing
            save_predictions=False,  # Don't save predictions in tests
            save_visualizations=False,  # Don't generate visualizations in tests
            generate_detailed_report=False  # Don't generate report in tests
        )

        # Initialize tester
        tester = GeneralizationTester(config)

        # Load model
        tester.load_model()

        # Load dataset
        samples = tester.load_dataset()

        # Test on a subset for faster testing
        test_samples = samples[:50]  # Only test on first 50 samples

        # Run testing
        results = tester.test_model(test_samples)

        # Basic assertions
        assert 'performance_metrics' in results
        assert 'accuracy' in results['performance_metrics']
        assert 'f1_score' in results['performance_metrics']
        assert 'auc_roc' in results['performance_metrics']

        # Check that metrics are reasonable
        accuracy = results['performance_metrics']['accuracy']
        f1_score = results['performance_metrics']['f1_score']
        auc_roc = results['performance_metrics']['auc_roc']

        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= f1_score <= 1.0
        assert 0.0 <= auc_roc <= 1.0

        # Print results for manual inspection
        print(f"Generalization Test Results (50 samples):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")

    def test_generalization_adapter_functionality(self):
        """Test Deepfake-Eval-2024 adapter functionality"""
        try:
            from tools.performance.test_generalization import DeepfakeEvalDatasetAdapter, GeneralizationConfig
        except ImportError:
            pytest.skip("Generalization testing module not available")

        dataset_path = "/workspace/Deepfake-Eval-2024"
        if not Path(dataset_path).exists():
            pytest.skip(f"Deepfake-Eval-2024 dataset not found at {dataset_path}")

        # Create config and adapter
        config = GeneralizationConfig(dataset_path=dataset_path)
        adapter = DeepfakeEvalDatasetAdapter(dataset_path, config)

        # Test metadata loading
        assert adapter.metadata_df is not None
        assert len(adapter.metadata_df) > 0

        # Test label mapping
        assert 'label' in adapter.metadata_df.columns
        unique_labels = adapter.metadata_df['label'].unique()
        assert all(label in [0, 1] for label in unique_labels)

        # Test dataset info
        info = adapter.get_dataset_info()
        assert 'total_samples' in info
        assert 'real_samples' in info
        assert 'fake_samples' in info
        assert info['total_samples'] > 0

        # Test sample loading
        samples = adapter.get_test_samples()
        assert len(samples) > 0

        # Test sample structure
        sample = samples[0]
        assert 'image_path' in sample
        assert 'label' in sample
        assert 'filename' in sample
        assert Path(sample['image_path']).exists()


if __name__ == "__main__":
    pytest.main([__file__])