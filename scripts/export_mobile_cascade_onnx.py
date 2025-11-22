#!/usr/bin/env python3
"""
Mobile Cascade ONNX Export Script - export_mobile_cascade_onnx.py
==================================================================

Export Stage 1 (MobileNetV4) and Stage 2 (EfficientNetV2-B3) models to ONNX format
for Android deployment using ONNX Runtime.

This script:
1. Loads trained PyTorch models from outputs/stage1 and outputs/stage5
2. Exports them to ONNX format using the ONNXExporter
3. Creates a cascade configuration file for Android
4. Validates ONNX models against PyTorch originals

Usage:
    python scripts/export_mobile_cascade_onnx.py

Output:
    android/mobile_bundle/
        aware_cascade_stage1.onnx
        aware_cascade_stage2.onnx
        aware_cascade_manifest.json
        cascade_config.json
"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepfakeClassifier(nn.Module):
    """
    Wrapper class that matches the training checkpoint structure.

    The checkpoint has:
    - backbone.* : Feature extractor (timm model with num_classes=0)
    - classifier.* : Classification head
    - temperature (optional): Temperature scaling parameter
    """

    def __init__(self, backbone_model_name, num_classes=1, use_two_layer_classifier=False, hidden_dim=512):
        super().__init__()

        # Create backbone with num_classes=0 to get feature extractor only
        # But first create a temporary model with num_classes=1 to get the correct feature dimension
        temp_model = timm.create_model(backbone_model_name, pretrained=False, num_classes=1)

        # Get the feature dimension from the original classifier
        if hasattr(temp_model, 'classifier') and isinstance(temp_model.classifier, nn.Linear):
            num_features = temp_model.classifier.in_features
        elif hasattr(temp_model, 'fc') and isinstance(temp_model.fc, nn.Linear):
            num_features = temp_model.fc.in_features
        elif hasattr(temp_model, 'head') and isinstance(temp_model.head, nn.Linear):
            num_features = temp_model.head.in_features
        else:
            num_features = 1280  # Default fallback

        logger.info(f"Detected feature dimension: {num_features}")

        # Now create the actual backbone without classifier
        self.backbone = timm.create_model(backbone_model_name, pretrained=False, num_classes=0)

        # Create classifier matching the checkpoint structure
        if use_two_layer_classifier:
            # Two-layer classifier (Stage 2 style)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(hidden_dim, num_classes)
            )
            # Temperature scaling parameter (for Stage 2)
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            # Single-layer classifier (Stage 1 style)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, num_classes)
            )

    def forward(self, x):
        features = self.backbone(x)
        # Handle different output formats
        if isinstance(features, tuple):
            features = features[0]
        output = self.classifier(features)
        return output


def create_stage1_model(model_path, device='cpu'):
    """
    Create and load Stage 1 MobileNetV4 model

    Args:
        model_path: Path to best_model.pth
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info("Creating Stage 1 MobileNetV4 model...")

    # Create model with wrapper to match checkpoint structure
    model = DeepfakeClassifier(
        backbone_model_name='mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
        num_classes=1
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_metric', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Stage 1 model loaded - Parameters: {num_params:,}")

    return model


def create_stage2_model(model_path, device='cpu'):
    """
    Create and load Stage 2 EfficientNetV2-B3 model

    Args:
        model_path: Path to best_model.pth
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info("Creating Stage 2 EfficientNetV2-B3 model...")

    # Create model with wrapper to match checkpoint structure
    # Stage 2 uses a two-layer classifier with hidden_dim=512
    model = DeepfakeClassifier(
        backbone_model_name='tf_efficientnetv2_b3',
        num_classes=1,
        use_two_layer_classifier=True,
        hidden_dim=512
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_metric', 'unknown')}")
        if hasattr(model, 'temperature'):
            logger.info(f"Temperature scaling: {model.temperature.item():.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Stage 2 model loaded - Parameters: {num_params:,}")

    return model


def create_cascade_config():
    """
    Create cascade configuration file for Android deployment

    Returns:
        Dictionary with cascade configuration
    """
    config = {
        "input_size": [1, 3, 256, 256],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "tau_low": 0.02,
        "tau_high": 0.98,
        "stage2_threshold": 0.5,
        "description": "Cascade configuration for two-stage deepfake detection",
        "notes": {
            "tau_low": "Stage1 fake probability below this -> classify as real",
            "tau_high": "Stage1 fake probability above this -> classify as fake",
            "stage2_threshold": "Stage2 fake probability threshold for final decision",
            "input_preprocessing": "Resize to 256x256, normalize with ImageNet stats (mean/std)"
        }
    }
    return config


def test_onnx_inference(onnx_path, pytorch_model, device='cpu'):
    """
    Test ONNX model inference and compare with PyTorch

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        device: Device for PyTorch inference
    """
    import onnxruntime as ort

    logger.info(f"Testing ONNX inference: {onnx_path}")

    # Create test input
    test_input = torch.randn(1, 3, 256, 256).to(device)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).cpu().numpy()

    # ONNX inference
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    logger.info(f"  PyTorch output: {pytorch_output.flatten()[:3]}...")
    logger.info(f"  ONNX output:    {onnx_output.flatten()[:3]}...")
    logger.info(f"  Max difference: {max_diff:.2e}")
    logger.info(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-5:
        logger.info("  ✅ ONNX model validated successfully!")
    else:
        logger.warning(f"  ⚠️ Large difference detected: {max_diff:.2e}")

    return max_diff < 1e-5


def main():
    """Main export function"""

    logger.info("=" * 80)
    logger.info("Mobile Cascade ONNX Export")
    logger.info("=" * 80)

    # Define paths
    stage1_model_path = PROJECT_ROOT / "outputs/stage1/run_20251023_034316/best_model.pth"
    stage2_model_path = PROJECT_ROOT / "outputs/stage5/finetune_s2_b3_r2/run_20251109_161118/best_model.pth"
    output_dir = PROJECT_ROOT / "android/mobile_bundle"

    # Verify model files exist
    if not stage1_model_path.exists():
        logger.error(f"Stage 1 model not found: {stage1_model_path}")
        return False

    if not stage2_model_path.exists():
        logger.error(f"Stage 2 model not found: {stage2_model_path}")
        return False

    logger.info(f"Stage 1 model: {stage1_model_path}")
    logger.info(f"Stage 2 model: {stage2_model_path}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Step 1: Load PyTorch models
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading PyTorch Models")
    logger.info("=" * 80)

    stage1_model = create_stage1_model(stage1_model_path, device)
    stage2_model = create_stage2_model(stage2_model_path, device)

    # Step 2: Export to ONNX
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Exporting to ONNX Format")
    logger.info("=" * 80)

    exporter = ONNXExporter(optimize_for_mobile=True, verbose=True)

    models = {
        'stage1': stage1_model,
        'stage2': stage2_model
    }

    bundle_result = exporter.export_cascade_bundle(
        models=models,
        output_dir=output_dir,
        bundle_name="aware_cascade"
    )

    if not bundle_result['success']:
        logger.error(f"Export failed: {bundle_result.get('error', 'Unknown error')}")
        return False

    logger.info(f"\n✅ Bundle exported successfully!")
    logger.info(f"   Total size: {bundle_result['total_size_mb']:.2f} MB")
    logger.info(f"   Models exported: {bundle_result['models_exported']}")
    logger.info(f"   Manifest: {bundle_result['manifest_path']}")

    # Step 3: Create cascade configuration
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Creating Cascade Configuration")
    logger.info("=" * 80)

    cascade_config = create_cascade_config()
    config_path = output_dir / "cascade_config.json"

    with open(config_path, 'w') as f:
        json.dump(cascade_config, f, indent=2)

    logger.info(f"✅ Configuration saved: {config_path}")
    logger.info(f"   tau_low: {cascade_config['tau_low']}")
    logger.info(f"   tau_high: {cascade_config['tau_high']}")
    logger.info(f"   stage2_threshold: {cascade_config['stage2_threshold']}")

    # Step 4: Validate ONNX models
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Validating ONNX Models")
    logger.info("=" * 80)

    stage1_onnx = output_dir / "aware_cascade_stage1.onnx"
    stage2_onnx = output_dir / "aware_cascade_stage2.onnx"

    stage1_valid = test_onnx_inference(stage1_onnx, stage1_model, device)
    stage2_valid = test_onnx_inference(stage2_onnx, stage2_model, device)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Export Summary")
    logger.info("=" * 80)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nGenerated files:")
    logger.info(f"  - aware_cascade_stage1.onnx ({os.path.getsize(stage1_onnx) / (1024*1024):.2f} MB)")
    logger.info(f"  - aware_cascade_stage2.onnx ({os.path.getsize(stage2_onnx) / (1024*1024):.2f} MB)")
    logger.info(f"  - aware_cascade_manifest.json")
    logger.info(f"  - cascade_config.json")
    logger.info(f"  - deploy_aware_cascade.py (deployment script)")
    logger.info(f"  - requirements.txt")
    logger.info(f"  - README.md")

    logger.info(f"\nValidation status:")
    logger.info(f"  Stage 1: {'✅ PASSED' if stage1_valid else '❌ FAILED'}")
    logger.info(f"  Stage 2: {'✅ PASSED' if stage2_valid else '❌ FAILED'}")

    if stage1_valid and stage2_valid:
        logger.info("\n🎉 All models exported and validated successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Copy android/mobile_bundle/* to Android project assets/models/")
        logger.info("  2. Implement OnnxCascadeEngine.kt in Android app")
        logger.info("  3. Test on Android device")
        return True
    else:
        logger.error("\n❌ Validation failed for some models")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
