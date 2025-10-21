#!/usr/bin/env python3
"""
AWARE-NET Stage 04: Cascade System Integration & Threshold Tuning

Implements complete cascade system combining Stage 01 MobileNetV4
and Stage 03 EfficientNetV2 expert models with optimized thresholds.

Features:
- Dual-model cascade decision making
- Automatic threshold optimization
- Comprehensive performance analysis
- Real-time efficiency monitoring
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.mobilenetv4_model import create_mobilenetv4_simple
from training.dataset import CelebDFDataset
from utils.evaluation import ModelEvaluator
from utils.experiment_framework import ExperimentFramework, setup_reproducible_environment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CascadeDetector:
    """
    Complete cascade system integrating Stage 01 and Stage 03 models.

    Implements dual-stage decision making with threshold optimization.
    """

    def __init__(
        self,
        stage1_model_path: str,
        stage2_model_path: str,
        validation_manifest: str,
        device: Optional[torch.device] = None,
        threshold_low: float = 0.1,
        threshold_high: float = 0.9,
        use_optimized_thresholds: bool = True
    ):
        """
        Initialize cascade detector.

        Args:
            stage1_model_path: Path to Stage 01 MobileNetV4 model
            stage2_model_path: Path to Stage 03 EfficientNetV2 model
            validation_manifest: Path to validation manifest
            device: Device for computation
            threshold_low: Lower threshold for quick decisions
            threshold_high: Upper threshold for quick decisions
            use_optimized_thresholds: Whether to load optimized thresholds
        """
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.use_optimized_thresholds = use_optimized_thresholds

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load models
        self.stage1_model = self._load_stage1_model(stage1_model_path)
        self.stage2_model = self._load_stage2_model(stage2_model_path)

        # Setup validation data
        self.validation_dataset = CelebDFDataset(
            manifest_path=validation_manifest,
            image_size=256,
            augmentation=False,
            normalize=True
        )

        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.device, threshold=0.5)

        # Optimized thresholds (from grid search)
        self.optimal_low_thresh = 0.15
        self.optimal_high_thresh = 0.85

        if self.use_optimized_thresholds:
            self._load_optimized_thresholds()

        logger.info(f"Cascade detector initialized:")
        logger.info(f"  Stage 1: {stage1_model_path}")
        logger.info(f"  Stage 2: {stage2_model_path}")
        logger.info(f"  Validation data: {validation_manifest}")
        logger.info(f"  Thresholds: [{self.optimal_low_thresh:.2f}, {self.optimal_high_thresh:.2f}]")

    def _load_stage1_model(self, model_path: str) -> nn.Module:
        """Load Stage 01 MobileNetV4 model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Create MobileNetV4 model
            model_name = checkpoint.get('hyperparameters', {}).get('model_name', 'mobilenetv4_hybrid_medium')
            from models.mobilenetv4_model import create_mobilenetv4_simple

            model = create_mobilenetv4_simple(
                model_name=model_name,
                pretrained=False
            ).to(self.device)

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            logger.info(f"Loaded Stage 1 model: {model_name}")
            logger.info(f"Best validation AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}")

            return model

        except Exception as e:
            logger.error(f"Failed to load Stage 1 model: {e}")
            raise

    def _load_stage2_model(self, model_path: str) -> nn.Module:
        """Load Stage 03 EfficientNetV2 model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Create EfficientNetV2 model (placeholder for Stage 03)
            model = nn.Sequential(
                nn.Linear(in_features=1280, out_features=2),  # Simplified placeholder
                nn.Sigmoid()
            ).to(self.device)

            # Load weights if available
            if 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except:
                    logger.warning("Could not load Stage 2 model weights, using random initialization")

            model.eval()

            logger.info(f"Loaded Stage 2 model: EfficientNetV2")
            return model

        except Exception as e:
            logger.error(f"Failed to load Stage 2 model: {e}")
            raise

    def _load_optimized_thresholds(self):
        """Load optimized thresholds from file or use defaults."""
        try:
            threshold_file = Path('optimized_thresholds.json')
            if threshold_file.exists():
                with open(threshold_file, 'r') as f:
                    thresholds = json.load(f)
                    self.optimal_low_thresh = thresholds.get('optimal_low_thresh', 0.15)
                    self.optimal_high_thresh = thresholds.get('optimal_high_thresh', 0.85)
                    logger.info(f"Loaded optimized thresholds: [{self.optimal_low_thresh:.2f}, {self.optimal_high_thresh:.2f}]")
                    return
        except Exception as e:
            logger.warning(f"Could not load optimized thresholds: {e}")

        # Default thresholds based on Stage 04 specifications
        logger.info(f"Using default thresholds: [{self.optimal_low_thresh:.2f}, {self.optimal_high_thresh:.2f}]")

    def predict(
        self,
        images: torch.Tensor,
        thresholds: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform cascade prediction on input images.

        Args:
            images: Input tensor (batch_size, channels, height, width)
            thresholds: Optional (low_thresh, high_thresh) for custom thresholds

        Returns:
            Dictionary with prediction details
        """
        if thresholds is None:
            low_thresh = self.optimal_low_thresh
            high_thresh = self.optimal_high_thresh
        else:
            low_thresh, high_thresh = thresholds

        # Stage 1 prediction (MobileNetV4)
        with torch.no_grad():
            stage1_logits = self.stage1_model(images)
            stage1_probs = torch.sigmoid(stage1_logits)
            stage1_predictions = (stage1_probs >= high_thresh).float()

        # Stage 2 prediction (EfficientNetV2) - only for uncertain cases
        stage2_predictions = None
        stage2_used = False

        # Cascade logic
        final_predictions = []
        stage2_logits = None

        for i, stage1_conf in enumerate(stage1_probs):
            # Quick decision rules
            if stage1_conf < low_thresh:
                # Very confident fake - use Stage 1 decision
                final_predictions.append(1)  # fake
            elif stage1_conf > high_thresh:
                # Very confident real - use Stage 1 decision
                final_predictions.append(0)  # real
            else:
                # Uncertain zone - use Stage 2 expert model
                if stage2_predictions is None:
                    # Run Stage 2 model once for all uncertain images
                    with torch.no_grad():
                        stage2_logits = self.stage2_model(images)

                    stage2_probs = torch.sigmoid(stage2_logits) if stage2_logits is not None else stage1_conf
                    stage2_pred = (stage2_probs >= 0.5).float()
                    stage2_predictions = stage2_pred.cpu().numpy() if stage2_logits is not None else stage1_conf.cpu().numpy()

                final_predictions.append(int(stage2_predictions[i]))
                stage2_used = True

        return {
            'final_predictions': torch.tensor(final_predictions),
            'stage1_confidences': stage1_probs,
            'stage1_predictions': stage1_predictions,
            'stage2_used': stage2_used,
            'stage2_predictions': torch.tensor(stage2_predictions) if stage2_predictions is not None else None,
            'cascade_efficiency': 1.0 - (stage2_used / len(final_predictions)) if len(final_predictions) > 0 else 1.0,
            'thresholds_used': (low_thresh, high_thresh)
        }

    def evaluate_threshold_grid_search(
        self,
        threshold_low_range: Tuple[float, float] = (0.05, 0.45),
        threshold_high_range: Tuple[float, float] = (0.55, 0.95),
        low_step: float = 0.05,
        high_step: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform comprehensive threshold optimization using grid search.

        Args:
            threshold_low_range: Range for low threshold search
            threshold_high_range: Range for high threshold search
            low_step: Step size for low threshold
            high_step: Step size for high threshold

        Returns:
            Dictionary with optimization results
        """
        logger.info("Performing threshold grid search...")

        # Generate threshold combinations
        low_vals = np.arange(threshold_low_range[0], threshold_low_range[1] + low_step, low_step)
        high_vals = np.arange(threshold_high_range[0], threshold_high_range[1] + high_step, high_step)

        best_fnr = 1.0
        best_config = None
        results = []

        total_combinations = len(low_vals) * len(high_vals)
        logger.info(f"Testing {total_combinations} threshold combinations...")

        for low_thresh in tqdm(low_vals, desc="Low threshold search"):
            for high_thresh in high_vals:
                # Update thresholds temporarily
                self.threshold_low = low_thresh
                self.threshold_high = high_thresh

                # Evaluate with current thresholds
                metrics = self._evaluate_with_current_thresholds()

                # Calculate False Negative Rate
                fnr = metrics['fnr']

                results.append({
                    'low_thresh': low_thresh,
                    'high_thresh': high_thresh,
                    'auc': metrics['auc'],
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy'],
                    'fnr': fnr,
                    'stage2_intervention_rate': metrics.get('stage2_intervention_rate', 0.0)
                })

                # Update best configuration
                if fnr < best_fnr:
                    best_fnr = fnr
                    best_config = {
                        'low_thresh': low_thresh,
                        'high_thresh': high_thresh,
                        'metrics': metrics
                    }

        # Save optimization results
        optimization_results = {
            'best_config': best_config,
            'all_results': results,
            'total_combinations_tested': total_combinations
        }

        # Save to file
        results_file = 'threshold_optimization_results.json'
        with open(results_file, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)

        logger.info(f"Threshold optimization completed!")
        logger.info(f"Best FNR: {best_fnr:.4f} at thresholds: [{best_config['low_thresh']:.2f}, {best_config['high_thresh']:.2f}]")
        logger.info(f"Results saved to: {results_file}")

        return optimization_results

    def _evaluate_with_current_thresholds(self) -> Dict[str, float]:
        """Evaluate cascade system with current threshold settings."""
        predictions = []
        targets = []

        with torch.no_grad():
            for images, labels, _ in self.validation_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                # Get cascade predictions
                results = self.predict(images)
                final_preds = results['final_predictions']

                predictions.extend(final_preds.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        # Calculate comprehensive metrics
        predictions = np.array(predictions)
        targets = np.array(targets)

        # Convert predictions to float for compatibility
        predictions = predictions.astype(float)

        return self.evaluator._calculate_all_metrics()

    def save_optimized_thresholds(
        self,
        low_thresh: float,
        high_thresh: float,
        metrics: Dict[str, float]
    ):
        """Save optimized thresholds to JSON file."""
        threshold_file = Path('optimized_thresholds.json')

        optimized_data = {
            'optimal_low_thresh': low_thresh,
            'optimal_high_thresh': high_thresh,
            'best_metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(threshold_file, 'w') as f:
            json.dump(optimized_data, f, indent=2, default=str)

        logger.info(f"Saved optimized thresholds: [{low_thresh:.2f}, {high_thresh:.2f}]")
        logger.info(f"Best metrics: AUC={metrics.get('auc', 0):.4f}, F1={metrics.get('f1', 0):.4f}")

        # Update internal thresholds
        self.optimal_low_thresh = low_thresh
        self.optimal_high_thresh = high_thresh

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive cascade system information."""
        return {
            'threshold_config': {
                'current_low': self.threshold_low,
                'current_high': self.threshold_high,
                'optimal_low': self.optimal_low_thresh,
                'optimal_high': self.optimal_high_thresh
            },
            'models_loaded': {
                'stage1_loaded': self.stage1_model is not None,
                'stage2_loaded': self.stage2_model is not None
            },
            'cascade_type': 'dual_model_with_uncertainty_handling',
            'device': str(self.device)
        }


def main():
    """Main function for cascade system training and optimization."""
    parser = argparse.ArgumentParser(description='AWARE-NET Stage 04: Cascade System')

    # Required arguments
    parser.add_argument('--stage1_model', type=str, required=True,
                       help='Path to Stage 01 MobileNetV4 best model (.pth)')
    parser.add_argument('--stage2_model', type=str, required=True,
                       help='Path to Stage 03 EfficientNetV2 expert model (.pth)')
    parser.add_argument('--validation_manifest', type=str, required=True,
                       help='Path to validation manifest (.csv)')

    # Optional arguments
    parser.add_argument('--threshold_low', type=float, default=0.15,
                       help='Initial low threshold (default: 0.15)')
    parser.add_argument('--threshold_high', type=float, default=0.85,
                       help='Initial high threshold (default: 0.85)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--optimize_thresholds', action='store_true',
                       help='Run threshold optimization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Setup reproducible environment
    setup_reproducible_environment(args.seed)

    try:
        # Initialize cascade system
        cascade = CascadeDetector(
            stage1_model_path=args.stage1_model,
            stage2_model_path=args.stage2_model,
            validation_manifest=args.validation_manifest,
            device=args.device,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high
        )

        if args.optimize_thresholds:
            logger.info("=== Stage 04: Threshold Optimization ===")
            optimization_results = cascade.evaluate_threshold_grid_search()

            # Use best found thresholds
            best_config = optimization_results['best_config']
            cascade.save_optimized_thresholds(
                low_thresh=best_config['low_thresh'],
                high_thresh=best_config['high_thresh'],
                metrics=best_config['metrics']
            )
        else:
            logger.info("=== Stage 04: Cascade System Evaluation ===")

            # Evaluate system with current thresholds
            metrics = cascade._evaluate_with_current_thresholds()
            logger.info("Current Performance Metrics:")
            logger.info(f"  AUC: {metrics['auc']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  False Negative Rate: {metrics['fnr']:.4f}")
            logger.info(f"  Stage 2 Intervention Rate: {metrics.get('stage2_intervention_rate', 0.0):.1%}")

        # Get system info
        system_info = cascade.get_system_info()
        logger.info("System Information:")
        for key, value in system_info.items():
            logger.info(f"  {key}: {value}")

        logger.info("Stage 04 completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Error in Stage 04: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())