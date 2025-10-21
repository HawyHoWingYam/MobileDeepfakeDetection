#!/usr/bin/env python3
"""
AWARE-NET Stage 02: Difficult Sample Subset Creation

Creates a subset of difficult samples for Stage 03 expert model training.
Uses Stage 01 best model to identify challenging samples.

Input:
- Stage 01 best model (MobileNetV4)
- Complete training manifest (train_manifest.csv)

Output:
- Difficult subset manifest (train_difficult_subset_manifest.csv)
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
from typing import List, Tuple, Dict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.dataset import CelebDFDataset
from utils.experiment_framework import setup_reproducible_environment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DifficultSubsetCreator:
    """
    Creates difficult sample subset using Stage 01 model for Stage 03 training.

    Implements Stage 02 requirements:
    1. Load Stage 01 best model
    2. Run inference on complete training set
    3. Identify difficult samples based on confusion zone
    4. Create filtered subset manifest
    """

    def __init__(
        self,
        model_path: str,
        train_manifest: str,
        output_path: str,
        confidence_low: float = 0.3,
        confidence_high: float = 0.7,
        image_size: int = 256,
        batch_size: int = 32,
        seed: int = 42
    ):
        """
        Initialize difficult subset creator.

        Args:
            model_path: Path to Stage 01 best model
            train_manifest: Path to complete training manifest
            output_path: Output path for difficult subset
            confidence_low: Lower bound for confusion zone
            confidence_high: Upper bound for confusion zone
            image_size: Image size for processing
            batch_size: Batch size for inference
            seed: Random seed for reproducibility
        """
        self.model_path = Path(model_path)
        self.train_manifest = Path(train_manifest)
        self.output_path = Path(output_path)
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

        # Setup reproducible environment
        setup_reproducible_environment(seed)

        logger.info(f"Difficult subset creator initialized:")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Train manifest: {self.train_manifest}")
        logger.info(f"  Confidence zone: [{self.confidence_low}, {self.confidence_high}]")
        logger.info(f"  Output: {self.output_path}")

    def load_model(self) -> nn.Module:
        """Load Stage 01 best model (MobileNetV4)."""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')

            # Create model based on saved architecture
            model_name = checkpoint.get('hyperparameters', {}).get('model_name', 'mobilenetv4_hybrid_medium')

            # Import MobileNetV4 model
            from models.mobilenetv4_model import create_mobilenetv4_simple

            model = create_mobilenetv4_simple(
                model_name=model_name,
                pretrained=False  # Use saved weights
            )

            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])

            logger.info(f"Loaded model: {model_name}")
            logger.info(f"Best validation AUC: {checkpoint.get('best_auc', 'Unknown'):.4f}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def create_data_loader(self) -> DataLoader:
        """Create data loader for inference (no augmentation)."""
        dataset = CelebDFDataset(
            manifest_path=str(self.train_manifest),
            image_size=self.image_size,
            augmentation=False,  # No augmentation for evaluation
            normalize=True
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep original order
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    def run_inference_and_collect_results(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> pd.DataFrame:
        """Run inference on complete training set and collect predictions."""
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        all_results = []

        logger.info("Running inference on training set...")

        with torch.no_grad():
            for images, targets, image_paths in tqdm(data_loader, desc="Processing samples"):
                images = images.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(images)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities >= 0.5).astype(int)

                # Collect results
                batch_results = []
                for i, (image_path, target, prob, pred) in enumerate(zip(image_paths, targets, probabilities, predictions)):
                    batch_results.append({
                        'image_path': image_path,
                        'label': int(target.cpu().numpy()),
                        'confidence': float(prob),
                        'prediction': int(pred),
                        'model_correct': (int(pred) == int(target.cpu().numpy()))
                    })

                all_results.extend(batch_results)

        logger.info(f"Processed {len(all_results)} samples")
        return pd.DataFrame(all_results)

    def filter_difficult_samples(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter difficult samples based on confidence zone and prediction errors.

        Implements Stage 02 criteria:
        1. Model confusion zone samples
        2. Model prediction errors
        """
        # Criteria 1: Confidence in confusion zone
        confusion_mask = (
            (results_df['confidence'] >= self.confidence_low) &
            (results_df['confidence'] <= self.confidence_high)
        )

        # Criteria 2: Model prediction is wrong
        error_mask = results_df['model_correct'] == False

        # Combine criteria (satisfy at least one)
        difficult_mask = confusion_mask | error_mask

        difficult_samples = results_df[difficult_mask].copy()

        logger.info(f"Filtering criteria:")
        logger.info(f"  Confidence zone [{self.confidence_low}, {self.confidence_high}]: {confusion_mask.sum()} samples")
        logger.info(f"  Prediction errors: {error_mask.sum()} samples")
        logger.info(f"  Combined difficult samples: {len(difficult_samples)} samples")

        # Analyze difficult sample distribution
        if len(difficult_samples) > 0:
            real_difficult = (difficult_samples['label'] == 0).sum()
            fake_difficult = (difficult_samples['label'] == 1).sum()
            total_real = results_df[results_df['label'] == 0].shape[0]
            total_fake = results_df[results_df['label'] == 1].shape[0]

            logger.info(f"  Difficult real samples: {real_difficult}/{total_real} ({real_difficult/total_real*100:.1f}%)")
            logger.info(f"  Difficult fake samples: {fake_difficult}/{total_fake} ({fake_difficult/total_fake*100:.1f}%)")

        return difficult_samples

    def save_difficult_subset(self, difficult_df: pd.DataFrame) -> None:
        """Save difficult subset manifest with same structure as original."""
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        difficult_df.to_csv(self.output_path, index=False)

        # Save analysis report
        analysis_path = self.output_path.parent / "difficult_subset_analysis.json"
        analysis = {
            'total_samples': len(difficult_df),
            'filtering_criteria': {
                'confidence_zone': [self.confidence_low, self.confidence_high],
                'include_prediction_errors': True
            },
            'output_manifest': str(self.output_path),
            'source_model': str(self.model_path),
            'source_train_manifest': str(self.train_manifest)
        }

        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"Saved {len(difficult_df)} difficult samples to: {self.output_path}")
        logger.info(f"Analysis report saved to: {analysis_path}")

    def create_subset(self):
        """Execute the complete difficult subset creation pipeline."""
        logger.info("=== Stage 02: Difficult Sample Subset Creation ===")

        # Step 1: Load Stage 01 model
        logger.info("Step 1: Loading Stage 01 best model...")
        model = self.load_model()

        # Step 2: Create data loader
        logger.info("Step 2: Creating data loader...")
        data_loader = self.create_data_loader()

        # Step 3: Run inference
        logger.info("Step 3: Running inference on training set...")
        results_df = self.run_inference_and_collect_results(model, data_loader)

        # Step 4: Filter difficult samples
        logger.info("Step 4: Filtering difficult samples...")
        difficult_df = self.filter_difficult_samples(results_df)

        # Step 5: Save subset
        logger.info("Step 5: Saving difficult subset...")
        self.save_difficult_subset(difficult_df)

        logger.info("=== Stage 02 Completed Successfully ===")

        return difficult_df


def main():
    """Main function for difficult subset creation."""
    parser = argparse.ArgumentParser(description='Create difficult subset for Stage 03')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Stage 01 best model (.pth)')
    parser.add_argument('--train_manifest', type=str, required=True,
                       help='Path to complete training manifest (.csv)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for difficult subset manifest (.csv)')

    # Optional arguments
    parser.add_argument('--confidence_low', type=float, default=0.3,
                       help='Lower bound for confusion zone (default: 0.3)')
    parser.add_argument('--confidence_high', type=float, default=0.7,
                       help='Upper bound for confusion zone (default: 0.7)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for processing (default: 256)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Validate input files
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1

    if not Path(args.train_manifest).exists():
        logger.error(f"Train manifest not found: {args.train_manifest}")
        return 1

    try:
        # Create subset creator
        creator = DifficultSubsetCreator(
            model_path=args.model_path,
            train_manifest=args.train_manifest,
            output_path=args.output_path,
            confidence_low=args.confidence_low,
            confidence_high=args.confidence_high,
            image_size=args.image_size,
            batch_size=args.batch_size,
            seed=args.seed
        )

        # Execute pipeline
        difficult_df = creator.create_subset()

        logger.info("Stage 02: Difficult subset creation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in Stage 02: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())