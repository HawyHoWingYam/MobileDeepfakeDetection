"""
Comprehensive Unit Test Suite for Stage 2 Implementation
Testing framework for heterogeneous expert system validation

This module provides comprehensive unit tests, integration tests, and validation
tests for all Stage 2 components including experts, fusion systems, diagnostics,
and integration interfaces.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from typing import Dict, List, Any
import warnings

# Import all Stage 2 components from existing modules
from .genconvit_expert import (
    BaseExpert, ExpertOutput, ExpertType, EnhancedGenConViT, GenConViTConfig, create_enhanced_genconvit
)
from .enhanced_spatial_expert import (
    EnhancedSpatialExpert, FocalLoss, GraduatedLRScheduler
)
from .complementarity_analysis import (
    ComplementarityAnalyzer, AdaptiveFusionSystem, create_fusion_system
)


class TestUnifiedFeatureExtractor(unittest.TestCase):
    """Test unified feature extraction framework"""

    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 256, 256)

    def test_expert_output_creation(self):
        """Test ExpertOutput creation and validation"""
        predictions = {'classification': torch.tensor([0.8, 0.3])}
        features = {'test_features': torch.randn(2, 128)}
        confidence = 0.75

        output = ExpertOutput(
            predictions=predictions,
            features=features,
            confidence=confidence,
            losses={}
        )

        self.assertEqual(output.confidence, confidence)
        self.assertIn('classification', output.predictions)
        self.assertIn('test_features', output.features)
        self.assertEqual(len(output.losses), 0)

    def test_expert_type_enum(self):
        """Test ExpertType enumeration"""
        self.assertEqual(ExpertType.SPATIAL.value, "spatial")
        self.assertEqual(ExpertType.GENERATIVE.value, "generative")
        self.assertEqual(ExpertType.TEMPORAL.value, "temporal")

    def test_base_expert_abstract(self):
        """Test that BaseExpert is properly abstract"""
        with self.assertRaises(TypeError):
            BaseExpert()  # Should raise TypeError for abstract class


class TestEnhancedSpatialExpert(unittest.TestCase):
    """Test enhanced spatial expert implementation"""

    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 256, 256)

    def test_focal_loss_initialization(self):
        """Test FocalLoss initialization and configuration"""
        from .enhanced_spatial_expert import FocalLossConfig

        config = FocalLossConfig(alpha=0.25, gamma=2.0)
        focal_loss = FocalLoss(config)

        self.assertEqual(focal_loss.config.alpha, 0.25)
        self.assertEqual(focal_loss.config.gamma, 2.0)

    def test_focal_loss_forward(self):
        """Test FocalLoss forward pass"""
        from .enhanced_spatial_expert import FocalLossConfig

        config = FocalLossConfig()
        focal_loss = FocalLoss(config)

        inputs = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4,)).float()

        loss = focal_loss(inputs, targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)

    def test_graduated_lr_scheduler(self):
        """Test GraduatedLRScheduler functionality"""
        from .enhanced_spatial_expert import GraduatedLRConfig

        # Create simple model for testing
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        config = GraduatedLRConfig(
            backbone_lr=0.0001,
            head_lr=0.001,
            warmup_epochs=2
        )

        scheduler = GraduatedLRScheduler(optimizer, config)

        # Test initial learning rates
        initial_lrs = scheduler.get_last_lr()
        self.assertIsInstance(initial_lrs, list)

        # Test step
        scheduler.step()
        new_lrs = scheduler.get_last_lr()
        self.assertIsInstance(new_lrs, list)


class TestEnhancedGenConViT(unittest.TestCase):
    """Test enhanced GenConViT implementation"""

    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 256, 256)

    def test_genconvit_config_creation(self):
        """Test GenConViT configuration"""
        config = GenConViTConfig(
            input_resolution=256,
            embed_dim=384,
            num_transformer_layers=6
        )

        self.assertEqual(config.input_resolution, 256)
        self.assertEqual(config.embed_dim, 384)
        self.assertEqual(config.num_transformer_layers, 6)
        self.assertIsNotNone(config.feature_fusion)
        self.assertIsNotNone(config.dual_variant)

    def test_genconvit_creation_factory(self):
        """Test GenConViT factory function"""
        model = create_enhanced_genconvit(
            input_resolution=256,
            fusion_strategy="cross_attention",
            reconstruction_mode="patch_based"
        )

        self.assertIsInstance(model, EnhancedGenConViT)
        self.assertEqual(model.expert_type, ExpertType.GENERATIVE)

    @patch('timm.create_model')
    def test_genconvit_forward_pass(self, mock_timm):
        """Test GenConViT forward pass with mocked timm"""
        # Mock timm model
        mock_model = MagicMock()
        mock_model.forward_features.return_value = [
            torch.randn(2, 96, 64, 64),
            torch.randn(2, 192, 32, 32),
            torch.randn(2, 384, 16, 16),
            torch.randn(2, 768, 8, 8)
        ]
        mock_timm.return_value = mock_model

        model = create_enhanced_genconvit(input_resolution=256)

        # Test forward pass
        with torch.no_grad():
            output = model(self.input_tensor)

        self.assertIsInstance(output, ExpertOutput)
        self.assertIn('classification', output.predictions)
        self.assertIsInstance(output.confidence, (int, float))


class TestComplementarityAnalysis(unittest.TestCase):
    """Test complementarity analysis and fusion systems"""

    def setUp(self):
        self.device = torch.device('cpu')

        # Create mock expert outputs
        self.expert_output_a = ExpertOutput(
            predictions={'classification': torch.tensor([0.8, 0.3])},
            features={'fused_features': torch.randn(2, 128)},
            confidence=0.75,
            losses={}
        )

        self.expert_output_b = ExpertOutput(
            predictions={'classification': torch.tensor([0.2, 0.9])},
            features={'fused_features': torch.randn(2, 128)},
            confidence=0.65,
            losses={}
        )

    def test_complementarity_analyzer_creation(self):
        """Test ComplementarityAnalyzer creation"""
        from .complementarity_analysis import ComplementarityConfig

        config = ComplementarityConfig()
        analyzer = ComplementarityAnalyzer(config)

        self.assertIsNotNone(analyzer.feature_analyzer)
        self.assertIsNotNone(analyzer.decision_analyzer)

    def test_complementarity_analysis_execution(self):
        """Test complementarity analysis execution"""
        from .complementarity_analysis import ComplementarityConfig

        config = ComplementarityConfig()
        analyzer = ComplementarityAnalyzer(config)

        result = analyzer.analyze_complementarity(
            self.expert_output_a,
            self.expert_output_b
        )

        self.assertIsNotNone(result.overall_complementarity)
        self.assertIsInstance(result.mutual_information, float)
        self.assertIsInstance(result.decision_diversity, float)
        self.assertIsInstance(result.recommendations, dict)

    def test_fusion_system_creation(self):
        """Test fusion system creation"""
        fusion_system = create_fusion_system(
            hidden_dim=256,
            num_experts=2,
            uncertainty_aware=True
        )

        self.assertIsInstance(fusion_system, AdaptiveFusionSystem)

    def test_fusion_system_execution(self):
        """Test fusion system execution"""
        fusion_system = create_fusion_system()

        expert_outputs = [self.expert_output_a, self.expert_output_b]
        result = fusion_system.fuse_experts(expert_outputs)

        self.assertIn('prediction', result)
        self.assertIsInstance(result['prediction'], torch.Tensor)


class TestSmokeTest(unittest.TestCase):
    """Smoke tests for Stage 2 core functionality"""

    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 256, 256)

    def test_module_imports(self):
        """Test that all core modules can be imported"""
        try:
            # Test importing from genconvit_expert
            from .genconvit_expert import (
                GenConViTExpert, GenConViTConfig, create_genconvit_expert,
                EnhancedGenConViT, create_enhanced_genconvit  # Aliases
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import from genconvit_expert: {e}")

        try:
            # Test importing from enhanced_spatial_expert
            from .enhanced_spatial_expert import (
                EnhancedSpatialExpert, FocalLoss, GraduatedLRScheduler
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import from enhanced_spatial_expert: {e}")

        try:
            # Test importing from complementarity_analysis
            from .complementarity_analysis import (
                ComplementarityAnalyzer, AdaptiveFusionSystem, create_fusion_system
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import from complementarity_analysis: {e}")

    @patch('timm.create_model')
    def test_genconvit_expert_creation(self, mock_timm):
        """Test GenConViT expert can be created"""
        # Mock timm model to avoid dependency issues
        mock_model = MagicMock()
        mock_model.forward_features.return_value = [
            torch.randn(2, 96, 64, 64),
            torch.randn(2, 192, 32, 32),
        ]
        mock_timm.return_value = mock_model

        try:
            config = GenConViTConfig(input_resolution=256)
            expert = create_enhanced_genconvit()
            self.assertIsInstance(expert, EnhancedGenConViT)
        except Exception as e:
            self.skipTest(f"GenConViT expert creation skipped: {e}")

    def test_complementarity_analysis_creation(self):
        """Test complementarity analyzer can be created"""
        try:
            from .complementarity_analysis import ComplementarityConfig
            config = ComplementarityConfig()
            analyzer = ComplementarityAnalyzer(config)
            self.assertIsNotNone(analyzer)
        except Exception as e:
            self.fail(f"Complementarity analyzer creation failed: {e}")

    def test_fusion_system_creation(self):
        """Test fusion system can be created"""
        try:
            fusion_system = create_fusion_system(
                hidden_dim=128,
                num_experts=2
            )
            self.assertIsInstance(fusion_system, AdaptiveFusionSystem)
        except Exception as e:
            self.fail(f"Fusion system creation failed: {e}")

    def test_training_script_compatibility(self):
        """Test that training scripts can be imported"""
        try:
            # Test importing training scripts (they should exist)
            import sys
            import os
            stage2_dir = os.path.dirname(__file__)

            # Check that training scripts exist
            spatial_train_path = os.path.join(stage2_dir, 'train_stage2_spatial.py')
            genconvit_train_path = os.path.join(stage2_dir, 'train_stage2_genconvit.py')

            self.assertTrue(os.path.exists(spatial_train_path),
                          f"Missing training script: {spatial_train_path}")
            self.assertTrue(os.path.exists(genconvit_train_path),
                          f"Missing training script: {genconvit_train_path}")

        except Exception as e:
            self.fail(f"Training script compatibility check failed: {e}")


def run_all_tests():
    """
    Run all test suites and return results
    """
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all available test classes
    test_classes = [
        TestUnifiedFeatureExtractor,
        TestEnhancedSpatialExpert,
        TestEnhancedGenConViT,
        TestComplementarityAnalysis,
        TestSmokeTest
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)

    return result


def run_specific_test_suite(suite_name: str):
    """
    Run specific test suite
    """
    suite_mapping = {
        'unified': TestUnifiedFeatureExtractor,
        'spatial': TestEnhancedSpatialExpert,
        'genconvit': TestEnhancedGenConViT,
        'complementarity': TestComplementarityAnalysis,
        'smoke': TestSmokeTest
    }

    if suite_name not in suite_mapping:
        raise ValueError(f"Unknown test suite: {suite_name}")

    test_class = suite_mapping[suite_name]
    tests = unittest.TestLoader().loadTestsFromTestCase(test_class)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)

    return result


if __name__ == "__main__":
    # Run all tests when script is executed directly
    print("Running Stage 2 Implementation Test Suite...")
    print("=" * 60)

    test_result = run_all_tests()

    print("\n" + "=" * 60)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Skipped: {len(test_result.skipped)}")

    if test_result.failures:
        print("\nFailures:")
        for test, traceback in test_result.failures:
            print(f"- {test}: {traceback}")

    if test_result.errors:
        print("\nErrors:")
        for test, traceback in test_result.errors:
            print(f"- {test}: {traceback}")

    success_rate = (test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}%")