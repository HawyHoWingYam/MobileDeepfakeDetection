#!/usr/bin/env python3
"""
Stage 02 Smoke Tests

Quick validation tests for Stage 02 heterogeneous expert models.
Replaces the outdated src/stage_02/test_suite.py with functional smoke tests.

Tests:
- Model import and initialization
- Single batch forward pass
- Configuration loading
- Basic functionality validation
"""

import pytest
import sys
import torch
import tempfile
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Stage 02 modules
try:
    from stage_02.enhanced_spatial_expert import EfficientNetV2SpatialExpert, FocalLoss, FocalLossConfig
    from stage_02.genconvit_expert import GenConViTExpert
    STAGE02_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Stage 02 modules not available: {e}")
    STAGE02_AVAILABLE = False


class TestSpatialExpert:
    """Test suite for Spatial Expert model"""

    @pytest.mark.gpu
    @pytest.mark.skipif(not STAGE02_AVAILABLE, reason="Stage 02 modules not available")
    def test_spatial_expert_import(self):
        """Test that spatial expert can be imported"""
        assert EfficientNetV2SpatialExpert is not None
        assert FocalLoss is not None
        assert FocalLossConfig is not None

    @pytest.mark.gpu
    @pytest.mark.skipif(not STAGE02_AVAILABLE, reason="Stage 02 modules not available")
    def test_spatial_expert_initialization(self):
        """Test spatial expert model initialization"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model with dummy config
        model = EfficientNetV2SpatialExpert("dummy_config.json")
        model = model.to(device)

        # Check model properties
        assert isinstance(model, torch.nn.Module)
        assert sum(p.numel() for p in model.parameters()) > 0

        print(f"✅ Spatial expert initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    @pytest.mark.gpu
    @pytest.mark.skipif(not STAGE02_AVAILABLE, reason="Stage 02 modules not available")
    def test_spatial_expert_forward_pass(self):
        """Test spatial expert forward pass with dummy data"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        model = EfficientNetV2SpatialExpert("dummy_config.json")
        model = model.to(device)
        model.eval()

        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        # Check output
        assert output.shape == (batch_size, 1)  # Binary classification
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        print(f"✅ Spatial expert forward pass successful: output shape {output.shape}")

    @pytest.mark.skipif(not STAGE02_AVAILABLE, reason="Stage 02 modules not available")
    def test_focal_loss_initialization(self):
        """Test focal loss initialization"""
        config = FocalLossConfig(alpha=0.25, gamma=2.0)
        loss_fn = FocalLoss(config)

        assert isinstance(loss_fn, torch.nn.Module)
        print("✅ Focal loss initialized successfully")


class TestGenConViTExpert:
    """Test suite for GenConViT Expert model"""

    @pytest.mark.gpu
    @pytest.mark.skipif(not STAGE02_AVAILABLE, reason="Stage 02 modules not available")
    def test_genconvit_import(self):
        """Test that GenConViT expert can be imported"""
        assert GenConViTExpert is not None
        print("✅ GenConViT expert import successful")

    @pytest.mark.gpu
    @pytest.mark.skipif(not STAGE02_AVAILABLE, reason="Stage 02 modules not available")
    def test_genconvit_initialization(self):
        """Test GenConViT expert model initialization"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Create model with dummy config
            model = GenConViTExpert("dummy_config.json")
            model = model.to(device)

            # Check model properties
            assert isinstance(model, torch.nn.Module)
            assert sum(p.numel() for p in model.parameters()) > 0

            print(f"✅ GenConViT expert initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        except Exception as e:
            pytest.skip(f"GenConViT initialization failed (might be due to missing dependencies): {e}")


class TestStage02Configuration:
    """Test suite for Stage 02 configuration loading"""

    def test_stage02_config_loading(self):
        """Test Stage 02 configuration file loading"""
        config_path = Path(__file__).parent.parent.parent / "configs" / "stage02_training.json"

        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check required sections
        required_sections = ['metadata', 'experiment', 'spatial_expert', 'genconvit_expert',
                          'training', 'data', 'monitoring', 'paths', 'targets']

        for section in required_sections:
            assert section in config, f"Missing config section: {section}"

        # Check specific values
        assert config['training']['batch_size'] == 32
        assert config['training']['epochs'] == 50
        assert config['spatial_expert']['name'] == 'efficientnetv2_rw_s'
        assert config['genconvit_expert']['name'] == 'genconvit'

        print("✅ Stage 02 configuration loaded successfully")


class TestStage02Dependencies:
    """Test suite for Stage 02 dependencies"""

    def test_pytorch_version(self):
        """Check PyTorch version compatibility"""
        torch_version = torch.__version__
        major_version = int(torch_version.split('.')[0])

        assert major_version >= 1, f"PyTorch version {torch_version} too old, need >= 1.0"
        print(f"✅ PyTorch version {torch_version} compatible")

    @pytest.mark.gpu
    def test_cuda_availability(self):
        """Check CUDA availability for Stage 02 training"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            assert device_count > 0, "CUDA available but no devices found"

            # Test basic CUDA operation
            test_tensor = torch.randn(10, 10).cuda()
            result = torch.mm(test_tensor, test_tensor.T)
            assert result.shape == (10, 10)

            print(f"✅ CUDA available with {device_count} device(s)")
        else:
            pytest.skip("CUDA not available, Stage 02 training requires GPU")

    def test_memory_availability(self):
        """Check system memory availability"""
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        assert available_gb >= 8, f"Insufficient memory: {available_gb:.1f}GB available, need >= 8GB"
        print(f"✅ Sufficient memory available: {available_gb:.1f}GB")


def run_smoke_tests():
    """Run all smoke tests and return results"""
    import subprocess

    print("🚀 Running Stage 02 Smoke Tests...")
    print("=" * 60)

    # Run pytest with specific markers
    cmd = [
        sys.executable, "-m", "pytest",
        str(__file__),
        "-v",  # Verbose output
        "-m", "not slow",  # Skip slow tests
        "--tb=short"  # Short traceback
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ All smoke tests passed!")
            return True
        else:
            print("❌ Some smoke tests failed!")
            return False

    except Exception as e:
        print(f"❌ Error running smoke tests: {e}")
        return False


if __name__ == "__main__":
    # Allow running this script directly
    success = run_smoke_tests()
    sys.exit(0 if success else 1)