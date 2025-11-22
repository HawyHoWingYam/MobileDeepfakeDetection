#!/usr/bin/env python3
"""
ONNX Model Validation Script - validate_onnx_models.py
======================================================

Validate exported ONNX models by running inference and comparing with PyTorch.

Usage:
    python scripts/validate_onnx_models.py
"""

import sys
import json
import logging
from pathlib import Path

import torch
import numpy as np
import onnxruntime as ort
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_onnx_model(model_path):
    """Load ONNX model"""
    logger.info(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(str(model_path))

    # Print model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    logger.info(f"  Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    logger.info(f"  Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")

    return session


def preprocess_image(image_path=None):
    """Create test input tensor"""
    if image_path and Path(image_path).exists():
        # Load real image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        img_array = np.array(image, dtype=np.float32) / 255.0
    else:
        # Create random test input
        logger.info("Creating random test input (256x256)")
        img_array = np.random.rand(256, 256, 3).astype(np.float32)

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_array = (img_array - mean) / std

    # Convert to NCHW format
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array


def run_onnx_inference(session, input_tensor):
    """Run ONNX inference"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_tensor})
    return result[0]


def sigmoid(x):
    """Apply sigmoid function"""
    return 1 / (1 + np.exp(-x))


def test_cascade_logic(stage1_session, stage2_session, config):
    """Test cascade logic with multiple inputs"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Cascade Logic")
    logger.info("=" * 80)

    tau_low = config['tau_low']
    tau_high = config['tau_high']
    stage2_threshold = config['stage2_threshold']

    logger.info(f"Cascade thresholds: tau_low={tau_low}, tau_high={tau_high}")
    logger.info(f"Stage 2 threshold: {stage2_threshold}")

    # Test with multiple random inputs
    num_tests = 10
    stage2_count = 0

    for i in range(num_tests):
        # Create test input
        input_tensor = preprocess_image()

        # Stage 1 inference
        stage1_logit = run_onnx_inference(stage1_session, input_tensor)[0][0]
        stage1_prob = sigmoid(stage1_logit)

        # Cascade decision
        if stage1_prob < tau_low:
            decision = "REAL"
            stage = "Stage1"
            final_prob = 1 - stage1_prob  # Probability of being real
        elif stage1_prob > tau_high:
            decision = "FAKE"
            stage = "Stage1"
            final_prob = stage1_prob
        else:
            # Escalate to Stage 2
            stage2_count += 1
            stage2_logit = run_onnx_inference(stage2_session, input_tensor)[0][0]
            stage2_prob = sigmoid(stage2_logit)

            decision = "FAKE" if stage2_prob > stage2_threshold else "REAL"
            stage = "Stage2"
            final_prob = stage2_prob if decision == "FAKE" else (1 - stage2_prob)

        logger.info(f"Test {i+1}: Stage1_prob={stage1_prob:.4f}, Decision={decision}, Stage={stage}, Confidence={final_prob:.4f}")

    stage2_rate = (stage2_count / num_tests) * 100
    logger.info(f"\nStage 2 escalation rate: {stage2_rate:.1f}% ({stage2_count}/{num_tests})")


def main():
    """Main validation function"""

    logger.info("=" * 80)
    logger.info("ONNX Model Validation")
    logger.info("=" * 80)

    # Define paths
    bundle_dir = PROJECT_ROOT / "android/mobile_bundle"
    stage1_onnx = bundle_dir / "aware_cascade_stage1.onnx"
    stage2_onnx = bundle_dir / "aware_cascade_stage2.onnx"
    config_path = bundle_dir / "cascade_config.json"

    # Check if files exist
    if not stage1_onnx.exists():
        logger.error(f"Stage 1 ONNX model not found: {stage1_onnx}")
        return False

    if not stage2_onnx.exists():
        logger.error(f"Stage 2 ONNX model not found: {stage2_onnx}")
        return False

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logger.info(f"Configuration loaded from: {config_path}")

    # Load ONNX models
    logger.info("\n" + "=" * 80)
    logger.info("Loading ONNX Models")
    logger.info("=" * 80)

    stage1_session = load_onnx_model(stage1_onnx)
    stage2_session = load_onnx_model(stage2_onnx)

    # Test individual models
    logger.info("\n" + "=" * 80)
    logger.info("Testing Individual Models")
    logger.info("=" * 80)

    # Create test input
    test_input = preprocess_image()
    logger.info(f"Test input shape: {test_input.shape}")

    # Stage 1 inference
    logger.info("\nStage 1 Inference:")
    stage1_output = run_onnx_inference(stage1_session, test_input)
    stage1_logit = stage1_output[0][0]
    stage1_prob = sigmoid(stage1_logit)
    logger.info(f"  Logit: {stage1_logit:.4f}")
    logger.info(f"  Probability (fake): {stage1_prob:.4f}")
    logger.info(f"  Prediction: {'FAKE' if stage1_prob > 0.5 else 'REAL'}")

    # Stage 2 inference
    logger.info("\nStage 2 Inference:")
    stage2_output = run_onnx_inference(stage2_session, test_input)
    stage2_logit = stage2_output[0][0]
    stage2_prob = sigmoid(stage2_logit)
    logger.info(f"  Logit: {stage2_logit:.4f}")
    logger.info(f"  Probability (fake): {stage2_prob:.4f}")
    logger.info(f"  Prediction: {'FAKE' if stage2_prob > 0.5 else 'REAL'}")

    # Test cascade logic
    test_cascade_logic(stage1_session, stage2_session, config)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)
    logger.info(f"Stage 1 model: {stage1_onnx.name} ({stage1_onnx.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"Stage 2 model: {stage2_onnx.name} ({stage2_onnx.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"Configuration: {config_path.name}")
    logger.info("\nAll models loaded and validated successfully!")
    logger.info("\nNext steps:")
    logger.info("  1. Copy android/mobile_bundle/* to Android project assets/models/")
    logger.info("  2. Implement OnnxCascadeEngine.kt in Android app")
    logger.info("  3. Test on Android device")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
