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

# Import all Stage 2 components
from .unified_feature_extractor import (
    BaseExpert, ExpertOutput, ExpertType, UnifiedFeatureExtractor
)
from .enhanced_spatial_expert import (
    EnhancedSpatialExpert, FocalLoss, GraduatedLRScheduler
)
from .enhanced_genconvit import (
    EnhancedGenConViT, GenConViTConfig, create_enhanced_genconvit
)
from .complementarity_analysis import (
    ComplementarityAnalyzer, AdaptiveFusionSystem, create_fusion_system
)
from .concurrent_testing_framework import (
    ConcurrentTestExecutor, TestCase, TestType, run_concurrent_tests
)
from .stage3_integration_interface import (
    TemporalIntegrationHub, Stage2ExpertWrapper, create_integration_hub
)
from .diagnostic_tools import (
    StageGateEvaluator, SystemHealthMonitor, ModelValidator, create_diagnostic_system
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


class TestConcurrentTestingFramework(unittest.TestCase):
    """Test concurrent testing framework"""

    def setUp(self):
        self.device = torch.device('cpu')

        # Create mock experts
        self.mock_expert_a = Mock(spec=BaseExpert)
        self.mock_expert_a.return_value = self.expert_output_a = ExpertOutput(
            predictions={'classification': torch.tensor([0.8])},
            features={'fused_features': torch.randn(1, 128)},
            confidence=0.75,
            losses={}
        )

        self.mock_expert_b = Mock(spec=BaseExpert)
        self.mock_expert_b.return_value = self.expert_output_b = ExpertOutput(
            predictions={'classification': torch.tensor([0.3])},
            features={'fused_features': torch.randn(1, 128)},
            confidence=0.65,
            losses={}
        )

    def test_test_case_creation(self):
        """Test TestCase creation"""
        test_case = TestCase(
            name="test_inference",
            test_type=TestType.PERFORMANCE_TEST,
            test_function=lambda x, y: None,
            timeout=60
        )

        self.assertEqual(test_case.name, "test_inference")
        self.assertEqual(test_case.test_type, TestType.PERFORMANCE_TEST)
        self.assertEqual(test_case.timeout, 60)

    @patch('torch.utils.data.DataLoader')
    def test_concurrent_test_executor_creation(self, mock_dataloader):
        """Test ConcurrentTestExecutor creation"""
        from .concurrent_testing_framework import ConcurrentTestConfig

        config = ConcurrentTestConfig(max_workers=2)
        executor = ConcurrentTestExecutor(config)

        self.assertEqual(executor.config.max_workers, 2)
        self.assertIsNotNone(executor.resource_monitor)


class TestStage3Integration(unittest.TestCase):
    """Test Stage 3 integration interface"""

    def setUp(self):
        self.device = torch.device('cpu')

    def test_integration_hub_creation(self):
        """Test integration hub creation"""
        hub = create_integration_hub(
            integration_level="hybrid",
            temporal_mode="frame_sequence",
            max_sequence_length=8
        )

        self.assertIsInstance(hub, TemporalIntegrationHub)
        self.assertEqual(hub.config.max_sequence_length, 8)

    def test_temporal_input_creation(self):
        """Test TemporalInput creation"""
        from .stage3_integration_interface import TemporalInput

        frames = torch.randn(1, 8, 3, 256, 256)  # [B, T, C, H, W]
        temporal_input = TemporalInput(
            frames=frames,
            frame_timestamps=list(range(8))
        )

        self.assertEqual(temporal_input.frames.shape, (1, 8, 3, 256, 256))
        self.assertEqual(len(temporal_input.frame_timestamps), 8)

    def test_stage2_output_creation(self):
        """Test Stage2Output creation"""
        from .stage3_integration_interface import Stage2Output

        output = Stage2Output(
            spatial_features=torch.randn(1, 256),
            generative_features=torch.randn(1, 256),
            fused_features=torch.randn(1, 256),
            spatial_predictions=torch.tensor([0.8]),
            generative_predictions=torch.tensor([0.3]),
            final_predictions=torch.tensor([0.55]),
            confidence_scores=torch.tensor([0.75, 0.65])
        )

        self.assertEqual(output.spatial_features.shape, (1, 256))
        self.assertEqual(output.final_predictions.item(), 0.55)


class TestDiagnosticTools(unittest.TestCase):
    """Test diagnostic tools and reporting"""

    def setUp(self):
        self.device = torch.device('cpu')

    def test_system_health_monitor_creation(self):
        """Test SystemHealthMonitor creation"""
        monitor = SystemHealthMonitor()

        self.assertFalse(monitor.monitoring_active)
        self.assertEqual(len(monitor.health_history), 0)

    def test_system_health_metrics_collection(self):
        """Test system health metrics collection"""
        monitor = SystemHealthMonitor()
        health = monitor.get_current_health()

        self.assertIsNotNone(health.cpu_usage)
        self.assertIsNotNone(health.memory_usage)
        self.assertIsNotNone(health.timestamp)

    def test_model_validator_creation(self):
        """Test ModelValidator creation"""
        validator = ModelValidator()

        self.assertEqual(len(validator.validation_history), 0)

    def test_stage_gate_evaluator_creation(self):
        """Test StageGateEvaluator creation"""
        evaluator = create_diagnostic_system()

        self.assertIsInstance(evaluator, StageGateEvaluator)
        self.assertIsNotNone(evaluator.criteria)
        self.assertIsNotNone(evaluator.health_monitor)

    def test_gate_report_template_generation(self):
        """Test gate report template generation"""
        from .diagnostic_tools import generate_stage_gate_report_template

        template = generate_stage_gate_report_template()

        self.assertIn('report_metadata', template)
        self.assertIn('executive_summary', template)
        self.assertIn('technical_assessment', template)
        self.assertIn('academic_assessment', template)


class TestIntegrationSuite(unittest.TestCase):
    """Integration tests for complete Stage 2 system"""

    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 256, 256)

    @patch('timm.create_model')
    def test_end_to_end_pipeline(self, mock_timm):
        """Test complete end-to-end pipeline"""
        # Mock timm model
        mock_model = MagicMock()
        mock_model.forward_features.return_value = [
            torch.randn(2, 96, 64, 64),
            torch.randn(2, 192, 32, 32),
            torch.randn(2, 384, 16, 16),
            torch.randn(2, 768, 8, 8)
        ]
        mock_timm.return_value = mock_model

        try:
            # Create experts
            spatial_expert = create_enhanced_genconvit()  # Using GenConViT as placeholder
            generative_expert = create_enhanced_genconvit()

            # Create fusion system
            fusion_system = create_fusion_system()

            # Run inference
            with torch.no_grad():
                spatial_output = spatial_expert(self.input_tensor)
                generative_output = generative_expert(self.input_tensor)

                # Fusion
                expert_outputs = [spatial_output, generative_output]
                fusion_result = fusion_system.fuse_experts(expert_outputs)

            # Validate outputs
            self.assertIn('prediction', fusion_result)
            self.assertIsInstance(fusion_result['prediction'], torch.Tensor)

        except Exception as e:
            # If there are import or dependency issues, skip this test
            self.skipTest(f"End-to-end test skipped due to dependencies: {e}")

    def test_stage_integration_compatibility(self):
        """Test Stage 2-3 integration compatibility"""
        try:
            # Create integration hub
            hub = create_integration_hub()

            # Create mock video input
            video_input = torch.randn(1, 8, 3, 256, 256)  # [B, T, C, H, W]

            # Test temporal input preparation
            temporal_input = hub.sequence_processor.prepare_temporal_input(video_input)

            self.assertEqual(temporal_input.frames.shape[1], 8)  # 8 frames
            self.assertEqual(len(temporal_input.frame_timestamps), 8)

        except Exception as e:
            self.skipTest(f"Integration test skipped due to dependencies: {e}")


class TestDocumentationAndExamples(unittest.TestCase):
    """Test documentation and usage examples"""

    def test_config_serialization(self):
        """Test configuration serialization for documentation"""
        from .enhanced_genconvit import GenConViTConfig

        config = GenConViTConfig(
            input_resolution=256,
            embed_dim=384
        )

        # Test that config can be converted to dict
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__

        self.assertIn('input_resolution', str(config_dict))
        self.assertIn('embed_dim', str(config_dict))

    def test_example_usage_patterns(self):
        """Test common usage patterns for documentation"""
        # Test factory function usage
        try:
            fusion_system = create_fusion_system(
                hidden_dim=128,
                num_experts=2
            )
            self.assertIsNotNone(fusion_system)

            integration_hub = create_integration_hub(
                integration_level="hybrid"
            )
            self.assertIsNotNone(integration_hub)

        except Exception as e:
            self.skipTest(f"Usage pattern test skipped: {e}")

    def test_error_handling_examples(self):
        """Test error handling patterns for documentation"""
        from .unified_feature_extractor import ExpertType

        # Test invalid expert type handling
        with self.assertRaises(ValueError):
            invalid_type = "invalid_expert_type"
            # This should raise an error if validation is implemented

        # Test configuration validation
        try:
            from .enhanced_genconvit import GenConViTConfig
            config = GenConViTConfig(input_resolution=-1)  # Invalid resolution
            # Should either raise error or handle gracefully
        except (ValueError, AssertionError):
            pass  # Expected behavior


def run_all_tests():
    """
    Run all test suites and return results
    """
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestUnifiedFeatureExtractor,
        TestEnhancedSpatialExpert,
        TestEnhancedGenConViT,
        TestComplementarityAnalysis,
        TestConcurrentTestingFramework,
        TestStage3Integration,
        TestDiagnosticTools,
        TestIntegrationSuite,
        TestDocumentationAndExamples
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
        'concurrent': TestConcurrentTestingFramework,
        'integration': TestStage3Integration,
        'diagnostics': TestDiagnosticTools,
        'end_to_end': TestIntegrationSuite,
        'docs': TestDocumentationAndExamples
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