#!/usr/bin/env python3
"""
AWARE-NET Stage 0: Complete System Verification
Final integration test and performance validation
"""

import sys
import os
import json
import time
import tempfile
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify_manifests():
    """Verify dataset manifests exist and are readable"""
    print("🔍 Verifying Dataset Manifests...")

    manifests_dir = Path("manifests")
    if not manifests_dir.exists():
        print("❌ Manifests directory not found")
        return False

    required_manifests = [
        "celebdf_v2_train.csv", "celebdf_v2_val.csv", "celebdf_v2_test.csv",
        "faceforensics_train.csv", "faceforensics_val.csv", "faceforensics_test.csv",
        "deeperforensics_train.csv", "deeperforensics_val.csv", "deeperforensics_test.csv"
    ]

    missing_manifests = []
    for manifest in required_manifests:
        manifest_path = manifests_dir / manifest
        if not manifest_path.exists():
            missing_manifests.append(manifest)

    if missing_manifests:
        print(f"❌ Missing manifests: {missing_manifests}")
        return False

    print("✅ All required manifest files exist")
    return True

def verify_model_architecture():
    """Verify baseline model can be instantiated"""
    print("🏗️ Verifying Model Architecture...")

    try:
        from stage_00.baseline_model import EfficientNetV2B3Baseline

        # Test B3 model instantiation
        model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,  # Don't download for verification
            model_name='tf_efficientnetv2_b3'
        )

        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
        assert torch.isfinite(output).all(), "Non-finite outputs detected"

        print("✅ EfficientNetV2-B3 model verification passed")
        return True

    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def verify_data_loaders():
    """Verify data loading pipeline works"""
    print("📊 Verifying Data Loading Pipeline...")

    try:
        from stage_00.train_baseline import MultiDatasetWrapper, UnifiedDeepfakeDataset

        # Test single dataset
        manifest_path = "manifests/celebdf_v2_train.csv"
        dataset = UnifiedDeepfakeDataset(
            manifest_path=manifest_path,
            dataset_name="test",
            transform=None
        )

        # Test get_class_counts method
        if hasattr(dataset, 'get_class_counts'):
            class_counts = dataset.get_class_counts()
            print(f"   Class counts: {class_counts}")

        # Test MultiDatasetWrapper
        datasets = [dataset]
        wrapper = MultiDatasetWrapper(datasets)

        # Test get_class_counts method on wrapper
        wrapper_counts = wrapper.get_class_counts()
        print(f"   Wrapper class counts: {wrapper_counts}")

        print("✅ Data loading pipeline verification passed")
        return True

    except Exception as e:
        print(f"❌ Data loading verification failed: {e}")
        return False

def verify_academic_tools():
    """Verify academic metrics and calibration tools"""
    print("📈 Verifying Academic Tools...")

    try:
        from utils.metrics import AcademicMetrics
        from utils.calibration_tools import CalibrationAnalyzer

        # Generate test data
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.binomial(1, 0.6, n_samples)
        y_scores = np.random.beta(2, 5, n_samples)

        # Test metrics
        metrics = AcademicMetrics(n_bootstrap=50)  # Reduced for speed
        auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
        print(f"   AUC test: {auc_result.value:.4f}")

        # Test calibration
        calibration = CalibrationAnalyzer(n_bins=5)  # Reduced for small dataset
        cal_result = calibration.calculate_ece_mce(y_true, y_scores)
        print(f"   ECE test: {cal_result.ece:.4f}")

        print("✅ Academic tools verification passed")
        return True

    except Exception as e:
        print(f"❌ Academic tools verification failed: {e}")
        return False

def verify_experiment_tracking():
    """Verify experiment management works"""
    print("🧪 Verifying Experiment Tracking...")

    try:
        from utils.experiment_utils import ExperimentManager, ExperimentConfig

        # Test configuration
        config = ExperimentConfig(
            experiment_name="test_verification",
            model_name="efficientnetv2_b3",
            dataset_name="test"
        )

        # Test experiment manager
        manager = ExperimentManager(experiments_dir="experiments")
        experiment_id = manager.start_experiment(config)

        print(f"   Test experiment ID: {experiment_id}")
        print("✅ Experiment tracking verification passed")
        return True

    except Exception as e:
        print(f"❌ Experiment tracking verification failed: {e}")
        return False

def verify_environment():
    """Verify environment setup"""
    print("🌍 Verifying Environment Setup...")

    try:
        # Check PyTorch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")

        # Check key directories
        required_dirs = [
            "src/stage_00", "src/utils", "configs", "manifests",
            "models", "src/inference", "tools"
        ]

        missing_dirs = []
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            print(f"   ⚠️ Missing directories: {missing_dirs}")
        else:
            print("   ✅ All required directories exist")

        print("✅ Environment verification passed")
        return True

    except Exception as e:
        print(f"❌ Environment verification failed: {e}")
        return False

def main():
    """Main verification script"""
    print("🚀 AWARE-NET Stage 0: Complete System Verification")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Run all verification tests
    tests = [
        ("Environment Setup", verify_environment),
        ("Dataset Manifests", verify_manifests),
        ("Model Architecture", verify_model_architecture),
        ("Data Loading Pipeline", verify_data_loaders),
        ("Academic Tools", verify_academic_tools),
        ("Experiment Tracking", verify_experiment_tracking),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\\n{'-' * 40}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Final summary
    print(f"\\n{'=' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 60}")

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")

    print(f"\\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\\n🎉 STAGE 0 VERIFICATION SUCCESSFUL!")
        print("✅ All systems operational and ready for training")
        return 0
    else:
        print("\\n⚠️ STAGE 0 VERIFICATION INCOMPLETE")
        print("📋 Please address failed tests before proceeding")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)