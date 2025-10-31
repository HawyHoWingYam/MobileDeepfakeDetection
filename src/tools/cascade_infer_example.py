#!/usr/bin/env python3
"""
Example: Using Stage 4 Tuned Thresholds in Inference

This script demonstrates how to load trained models and apply optimized
cascade thresholds from Stage 4 threshold tuning.

Usage:
    python src/tools/cascade_infer_example.py \
        --stage1-ckpt outputs/stage1/.../best_model.pth \
        --stage2-ckpt outputs/stage3/.../best_model.pth \
        --thresholds outputs/stage4/run_.../best_config.json \
        --image-path /path/to/test/image.jpg

This is meant as a reference implementation. In production, integrate this
logic into your inference API/pipeline.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CascadeInference:
    """
    Production-ready cascade inference with Stage 4 tuned thresholds.

    Architecture:
    1. Load trained Stage 1 and Stage 2 models
    2. Apply learned threshold boundaries
    3. Route samples to Stage 2 only if needed
    4. Return final predictions with confidence scores
    """

    def __init__(
        self,
        stage1_ckpt: str,
        stage2_ckpt: str,
        best_config_json: str,
        device: str = 'cuda:0',
    ):
        """
        Initialize cascade inference engine.

        Args:
            stage1_ckpt: Path to Stage 1 checkpoint
            stage2_ckpt: Path to Stage 2 checkpoint
            best_config_json: Path to best_config.json from Stage 4 tuning
            device: Device for computation
        """
        self.device = torch.device(device)

        # Load thresholds from Stage 4 tuning
        logger.info(f"Loading thresholds from {best_config_json}")
        with open(best_config_json, 'r') as f:
            config = json.load(f)

        self.low_thresh = config['low_thresh']
        self.high_thresh = config['high_thresh']
        self.escalation_rate = config['escalation_rate']

        logger.info(f"Thresholds: low={self.low_thresh:.4f}, high={self.high_thresh:.4f}")
        logger.info(f"Expected escalation rate: {self.escalation_rate:.2%}")

        # Load models
        logger.info("Loading Stage 1 (MobileNetV4) ...")
        self.stage1_model = self._load_model(stage1_ckpt, size=256)

        logger.info("Loading Stage 2 (EfficientNetV2) ...")
        self.stage2_model = self._load_model(stage2_ckpt, size=384)

        # Setup transforms
        self.stage1_transform = self._build_transform(256)
        self.stage2_transform = self._build_transform(384)

        # Performance tracking
        self.stats = {
            'total_samples': 0,
            'stage1_real': 0,
            'stage1_fake': 0,
            'stage2_used': 0,
        }

    def _load_model(self, ckpt_path: str, size: int) -> torch.nn.Module:
        """Load model from checkpoint."""
        # Import here to avoid circular dependencies
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Determine model type from checkpoint metadata or name
        if 'mobilenetv4' in str(ckpt_path).lower():
            from models.mobilenetv4_model import create_mobilenetv4_simple
            model = create_mobilenetv4_simple(model_name='mobilenetv4_hybrid_medium', pretrained=False)
        else:
            from models.efficientnetv2_model import create_baseline_model
            model = create_baseline_model(model_name='tf_efficientnetv2_b0', pretrained=False)

        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device).eval()

    def _build_transform(self, size: int) -> A.Compose:
        """Build inference transform (no augmentation)."""
        return A.Compose([
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

    def predict(self, image: np.ndarray) -> Dict:
        """
        Perform cascade prediction on a single image.

        Args:
            image: Input image (BGR or RGB, HxWxC)

        Returns:
            Dictionary with:
            - prediction: 0 (real) or 1 (fake)
            - confidence: Score for fake class [0, 1]
            - stage_used: 1 or 2
            - stage1_conf: Confidence from Stage 1
            - stage2_conf: Confidence from Stage 2 (if used)
        """
        # Stage 1: Fast inference
        with torch.no_grad():
            # Preprocess
            if image.shape[2] == 3:  # Assume BGR from cv2.imread
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Apply transform
            transformed = self.stage1_transform(image=image_rgb)
            img_tensor = transformed['image'].unsqueeze(0).to(self.device)

            # Stage 1 inference
            stage1_logit = self.stage1_model(img_tensor)
            stage1_conf = torch.sigmoid(stage1_logit).item()

        # Decision based on threshold
        if stage1_conf <= self.low_thresh:
            # Confident real - use Stage 1 decision
            prediction = 0
            confidence = stage1_conf
            stage_used = 1
            stage2_conf = None

            self.stats['stage1_real'] += 1

        elif stage1_conf >= self.high_thresh:
            # Confident fake - use Stage 1 decision
            prediction = 1
            confidence = stage1_conf
            stage_used = 1
            stage2_conf = None

            self.stats['stage1_fake'] += 1

        else:
            # Ambiguous - escalate to Stage 2
            with torch.no_grad():
                # Preprocess for Stage 2
                transformed = self.stage2_transform(image=image_rgb)
                img_tensor = transformed['image'].unsqueeze(0).to(self.device)

                # Stage 2 inference
                stage2_logit = self.stage2_model(img_tensor)
                stage2_conf = torch.sigmoid(stage2_logit).item()

            # Stage 2 decision
            prediction = 1 if stage2_conf >= 0.5 else 0
            confidence = stage2_conf
            stage_used = 2

            self.stats['stage2_used'] += 1

        self.stats['total_samples'] += 1

        return {
            'prediction': prediction,
            'confidence': confidence,
            'stage_used': stage_used,
            'stage1_confidence': stage1_conf,
            'stage2_confidence': stage2_conf,
            'label': 'FAKE' if prediction == 1 else 'REAL',
            'threshold_low': self.low_thresh,
            'threshold_high': self.high_thresh,
        }

    def predict_batch(self, images: list) -> list:
        """
        Perform cascade prediction on a batch of images.

        Args:
            images: List of image arrays (HxWxC each)

        Returns:
            List of prediction dictionaries
        """
        results = []
        for img in images:
            result = self.predict(img)
            results.append(result)
        return results

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['total_samples'] > 0:
            stats['stage2_rate'] = stats['stage2_used'] / stats['total_samples']
            stats['stage1_rate'] = 1.0 - stats['stage2_rate']
        return stats

    def reset_stats(self):
        """Reset performance counters."""
        for key in self.stats:
            self.stats[key] = 0


def main():
    """Example usage of cascade inference."""
    parser = argparse.ArgumentParser(
        description='Stage 4 Cascade Inference Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python src/tools/cascade_infer_example.py \\
    --stage1-ckpt outputs/stage1/.../best_model.pth \\
    --stage2-ckpt outputs/stage3/.../best_model.pth \\
    --thresholds outputs/stage4/run_.../best_config.json \\
    --image-path /path/to/image.jpg

  # Batch inference
  python src/tools/cascade_infer_example.py \\
    --stage1-ckpt outputs/stage1/.../best_model.pth \\
    --stage2-ckpt outputs/stage3/.../best_model.pth \\
    --thresholds outputs/stage4/run_.../best_config.json \\
    --image-dir /path/to/images/
        """
    )

    parser.add_argument('--stage1-ckpt', required=True, help='Stage 1 checkpoint path')
    parser.add_argument('--stage2-ckpt', required=True, help='Stage 2 checkpoint path')
    parser.add_argument('--thresholds', required=True, help='Path to best_config.json from Stage 4')
    parser.add_argument('--image-path', type=str, help='Path to single image')
    parser.add_argument('--image-dir', type=str, help='Path to directory of images')
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')

    args = parser.parse_args()

    # Initialize inference engine
    cascade = CascadeInference(
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        best_config_json=args.thresholds,
        device=args.device,
    )

    logger.info(f"Cascade inference initialized")
    logger.info(f"  Low threshold: {cascade.low_thresh:.4f}")
    logger.info(f"  High threshold: {cascade.high_thresh:.4f}")

    # Process single image
    if args.image_path:
        logger.info(f"\nProcessing: {args.image_path}")
        image = cv2.imread(args.image_path)

        if image is None:
            logger.error(f"Could not load image: {args.image_path}")
            return 1

        result = cascade.predict(image)

        logger.info("\n" + "="*60)
        logger.info("PREDICTION RESULT")
        logger.info("="*60)
        logger.info(f"Label: {result['label']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        logger.info(f"Stage Used: {result['stage_used']}")
        if result['stage_used'] == 2:
            logger.info(f"  Stage 1 Confidence: {result['stage1_confidence']:.4f}")
            logger.info(f"  Stage 2 Confidence: {result['stage2_confidence']:.4f}")
        logger.info("="*60)

    # Process directory of images
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        logger.info(f"\nProcessing {len(image_files)} images from {image_dir}")

        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Skipped (could not load): {img_path}")
                continue

            result = cascade.predict(image)
            logger.info(f"{img_path.name}: {result['label']} (conf={result['confidence']:.4f}, stage={result['stage_used']})")

        # Print statistics
        stats = cascade.get_stats()
        logger.info("\n" + "="*60)
        logger.info("BATCH STATISTICS")
        logger.info("="*60)
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Stage 1 decisions: {stats['total_samples'] - stats['stage2_used']} ({1-stats['stage2_rate']:.1%})")
        logger.info(f"Stage 2 escalations: {stats['stage2_used']} ({stats['stage2_rate']:.1%})")
        logger.info(f"Stage 1 Real predictions: {stats['stage1_real']}")
        logger.info(f"Stage 1 Fake predictions: {stats['stage1_fake']}")
        logger.info("="*60)

    else:
        logger.error("Provide either --image-path or --image-dir")
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
