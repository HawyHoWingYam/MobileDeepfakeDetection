#!/usr/bin/env python3
"""
Generalization Testing Script for Deepfake Detection Models

This script tests the generalization ability of trained models on unseen datasets,
specifically designed for evaluating out-of-distribution performance on real-world data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from stage_00.baseline_model import EfficientNetV2B3Baseline


@dataclass
class GeneralizationConfig:
    """Configuration for generalization testing"""
    # Required arguments (no defaults)
    model_path: str
    dataset_path: str

    # Optional arguments with defaults
    model_architecture: str = 'tf_efficientnetv2_b0'
    metadata_file: str = 'image-metadata-publish.csv'
    image_dir: str = 'image-data'
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 256
    output_dir: str = "generalization_results"
    save_predictions: bool = True
    save_visualizations: bool = True
    generate_detailed_report: bool = True


class DeepfakeEvalDatasetAdapter:
    """Adapter for Deepfake-Eval-2024 dataset"""

    def __init__(self, dataset_path: str, config: GeneralizationConfig):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.metadata_df = None
        self.image_dir = self.dataset_path / config.image_dir

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load and preprocess metadata"""
        metadata_path = self.dataset_path / self.config.metadata_file

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.metadata_df = pd.read_csv(metadata_path)

        # Preprocess
        self.metadata_df['label'] = self.metadata_df['Ground Truth'].map({
            'Real': 0, 'Fake': 1, 'real': 0, 'fake': 1,
            'REAL': 0, 'FAKE': 1, 'Real': 0, 'Fake': 1
        })

        # Filter out invalid entries
        self.metadata_df = self.metadata_df.dropna(subset=['label'])
        self.metadata_df['label'] = self.metadata_df['label'].astype(int)

        logging.info(f"Loaded {len(self.metadata_df)} samples from metadata")
        logging.info(f"Label distribution: {self.metadata_df['label'].value_counts().to_dict()}")

    def get_test_samples(self) -> List[Dict[str, Any]]:
        """Get all samples for testing"""
        samples = []

        for _, row in self.metadata_df.iterrows():
            image_path = self.image_dir / row['Filename']

            if image_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'label': row['label'],
                    'filename': row['Filename']
                })
            else:
                logging.warning(f"Image file not found: {image_path}")

        logging.info(f"Found {len(samples)} valid image files")
        return samples

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        if self.metadata_df is None:
            return {}

        return {
            'total_samples': len(self.metadata_df),
            'real_samples': (self.metadata_df['label'] == 0).sum(),
            'fake_samples': (self.metadata_df['label'] == 1).sum(),
            'unique_formats': self.metadata_df['Filename'].apply(lambda x: Path(x).suffix).unique().tolist(),
            'train_samples': (self.metadata_df.get('Finetuning Set', '') == 'Train').sum(),
            'test_samples': (self.metadata_df.get('Finetuning Set', '') == 'Test').sum()
        }


class GeneralizationTester:
    """Main class for testing model generalization"""

    def __init__(self, config: GeneralizationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dataset_adapter = None
        self.results = {}

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "generalization_test.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def load_model(self):
        """Load trained model"""
        logging.info(f"Loading model from {self.config.model_path}")

        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

        # Create model
        self.model = EfficientNetV2B3Baseline(
            num_classes=1,
            pretrained=False,
            model_name=self.config.model_architecture
        )

        # Load state dict (weights_only=False for compatibility with older checkpoints)
        checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        logging.info(f"Model loaded successfully on {self.device}")

    def load_dataset(self):
        """Load and prepare dataset"""
        logging.info(f"Loading dataset from {self.config.dataset_path}")

        self.dataset_adapter = DeepfakeEvalDatasetAdapter(
            self.config.dataset_path,
            self.config
        )

        dataset_info = self.dataset_adapter.get_dataset_info()
        logging.info("Dataset info:")
        for key, value in dataset_info.items():
            logging.info(f"  {key}: {value}")

        return self.dataset_adapter.get_test_samples()

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image with robust error handling"""
        try:
            # Check if file exists and is readable
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Check file size (skip if too large or too small)
            file_size = Path(image_path).stat().st_size
            if file_size < 100:  # Less than 100 bytes
                raise ValueError(f"Image file too small: {file_size} bytes")
            if file_size > 50 * 1024 * 1024:  # More than 50MB
                raise ValueError(f"Image file too large: {file_size/1024/1024:.1f} MB")

            # Open image with error handling
            image = Image.open(image_path)

            # Convert to RGB with mode checking
            if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
                logging.warning(f"Unusual image mode {image.mode} for {image_path}")

            image = image.convert('RGB')

            # Check image dimensions
            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError(f"Image too small: {image.size}")
            if image.size[0] > 4096 or image.size[1] > 4096:
                logging.warning(f"Large image size {image.size} for {image_path}")

            # Resize with antialiasing for better quality
            resize_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BILINEAR
            image = image.resize((self.config.image_size, self.config.image_size), resize_method)

            # Convert to tensor with error checking
            image_array = np.array(image, dtype=np.float32)

            # Validate array
            if image_array.size == 0:
                raise ValueError("Empty image array")

            # Normalize and convert to tensor
            image_array = image_array / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC -> CHW

            # Validate tensor
            if image_tensor.shape != (3, self.config.image_size, self.config.image_size):
                raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")

            return image_tensor

        except FileNotFoundError as e:
            logging.warning(f"Skipping missing image: {e}")
            return torch.zeros(3, self.config.image_size, self.config.image_size)
        except (IOError, OSError) as e:
            logging.warning(f"Skipping corrupted image {image_path}: {e}")
            return torch.zeros(3, self.config.image_size, self.config.image_size)
        except Exception as e:
            logging.warning(f"Error processing image {image_path}: {e}")
            return torch.zeros(3, self.config.image_size, self.config.image_size)

    def test_model(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test model on samples with progress tracking"""
        logging.info(f"Testing model on {len(samples)} samples")

        all_predictions = []
        all_labels = []
        all_filenames = []
        processing_times = []
        error_count = 0
        skipped_count = 0

        # Calculate number of batches
        num_batches = (len(samples) + self.config.batch_size - 1) // self.config.batch_size
        logging.info(f"Processing {len(samples)} samples in {num_batches} batches (batch_size={self.config.batch_size})")

        # Process in batches with progress bar
        with tqdm(total=len(samples), desc="Processing samples", unit="samples",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

            for i in range(0, len(samples), self.config.batch_size):
                batch_samples = samples[i:i + self.config.batch_size]
                batch_start_time = time.time()

                batch_images = []
                batch_labels = []
                batch_filenames = []
                batch_errors = 0

                # Preprocess batch with error handling
                for sample in batch_samples:
                    try:
                        image = self.preprocess_image(sample['image_path'])
                        batch_images.append(image)
                        batch_labels.append(sample['label'])
                        batch_filenames.append(sample['filename'])
                    except Exception as e:
                        logging.error(f"Failed to process {sample['filename']}: {e}")
                        # Add a black image as fallback
                        batch_images.append(torch.zeros(3, self.config.image_size, self.config.image_size))
                        batch_labels.append(sample['label'])
                        batch_filenames.append(sample['filename'])
                        batch_errors += 1

                # Skip batch if too many errors
                if batch_errors > len(batch_samples) * 0.8:  # More than 80% errors
                    logging.warning(f"Skipping batch {i//self.config.batch_size + 1} due to too many errors ({batch_errors}/{len(batch_samples)})")
                    skipped_count += len(batch_samples)
                    pbar.update(len(batch_samples))
                    continue

                try:
                    # Stack images and move to device
                    batch_tensor = torch.stack(batch_images).to(self.device)

                    # Inference with error handling
                    with torch.no_grad():
                        outputs = self.model(batch_tensor)
                        predictions = torch.sigmoid(outputs).cpu().numpy().flatten()

                    # Validate predictions
                    if len(predictions) != len(batch_samples):
                        raise ValueError(f"Prediction length mismatch: {len(predictions)} vs {len(batch_samples)}")

                    processing_time = time.time() - batch_start_time
                    processing_times.append(processing_time)

                    # Store results
                    all_predictions.extend(predictions)
                    all_labels.extend(batch_labels)
                    all_filenames.extend(batch_filenames)

                    # Update progress bar
                    pbar.set_postfix({
                        'batch': f"{i//self.config.batch_size + 1}/{num_batches}",
                        'errors': f"{error_count}",
                        'time': f"{processing_time:.2f}s"
                    })
                    pbar.update(len(batch_samples))

                    # Log progress every 10 batches
                    if (i // self.config.batch_size + 1) % 10 == 0:
                        current_time = time.time()
                        elapsed = current_time - batch_start_time
                        samples_per_sec = len(batch_samples) / elapsed
                        logging.info(f"Batch {i//self.config.batch_size + 1}/{num_batches}: "
                                   f"{len(batch_samples)} samples, {samples_per_sec:.1f} samples/sec")

                except Exception as e:
                    logging.error(f"Failed to process batch {i//self.config.batch_size + 1}: {e}")
                    error_count += len(batch_samples)
                    pbar.update(len(batch_samples))
                    continue

        # Final statistics
        total_processed = len(all_predictions)
        total_errors = len(samples) - total_processed
        logging.info(f"Processing completed: {total_processed} successful, {total_errors} errors, {skipped_count} skipped")

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Convert predictions to binary labels
        binary_predictions = (all_predictions > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, binary_predictions)
        f1 = f1_score(all_labels, binary_predictions)
        auc = roc_auc_score(all_labels, all_predictions)

        # Additional metrics
        try:
            precision = precision_score(all_labels, binary_predictions)
            recall = recall_score(all_labels, binary_predictions)
        except:
            precision = recall = 0.0

        # Performance metrics (handle case where no batches were processed)
        if processing_times:
            avg_processing_time = np.mean(processing_times)
            total_processing_time = sum(processing_times)
            throughput = total_processed / total_processing_time
            avg_time_per_sample = avg_processing_time / min(self.config.batch_size, len(batch_samples))
        else:
            avg_processing_time = 0.0
            total_processing_time = 0.0
            throughput = 0.0
            avg_time_per_sample = 0.0

        results = {
            'dataset_info': self.dataset_adapter.get_dataset_info(),
            'performance_metrics': {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_roc': auc,
                'precision': precision,
                'recall': recall,
                'total_samples': len(all_labels),
                'successful_samples': total_processed,
                'failed_samples': total_errors,
                'skipped_samples': skipped_count,
                'correct_predictions': (binary_predictions == all_labels).sum(),
                'false_positives': ((binary_predictions == 1) & (all_labels == 0)).sum(),
                'false_negatives': ((binary_predictions == 0) & (all_labels == 1)).sum(),
                'true_positives': ((binary_predictions == 1) & (all_labels == 1)).sum(),
                'true_negatives': ((binary_predictions == 0) & (all_labels == 0)).sum()
            },
            'performance_stats': {
                'avg_processing_time_per_batch': avg_processing_time,
                'throughput_samples_per_second': throughput,
                'avg_time_per_sample': avg_time_per_sample,
                'total_processing_time': total_processing_time,
                'num_successful_batches': len(processing_times)
            },
            'predictions': {
                'filenames': all_filenames,
                'labels': all_labels.tolist(),
                'probabilities': all_predictions.tolist(),
                'binary_predictions': binary_predictions.tolist()
            } if self.config.save_predictions else None,
            'confusion_matrix': confusion_matrix(all_labels, binary_predictions).tolist(),
            'classification_report': classification_report(all_labels, binary_predictions, output_dict=True)
        }

        self.results = results
        logging.info("Testing completed!")
        logging.info(f"Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        return results

    def generate_visualizations(self) -> Dict[str, str]:
        """Generate visualization plots"""
        if not self.results:
            raise ValueError("No results available. Run test_model first.")

        visualization_paths = {}

        if not self.config.save_visualizations:
            return visualization_paths

        # 1. ROC Curve
        plt.figure(figsize=(8, 6))
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.results['predictions']['labels'],
                                self.results['predictions']['probabilities'])
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {self.results["performance_metrics"]["auc_roc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Generalization Test')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        roc_path = self.output_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['roc_curve'] = str(roc_path)

        # 2. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(self.results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix - Generalization Test')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['confusion_matrix'] = str(cm_path)

        # 3. Prediction Distribution
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        real_probs = [p for p, l in zip(self.results['predictions']['probabilities'],
                                       self.results['predictions']['labels']) if l == 0]
        fake_probs = [p for p, l in zip(self.results['predictions']['probabilities'],
                                       self.results['predictions']['labels']) if l == 1]

        plt.hist(real_probs, bins=30, alpha=0.7, label='Real', color='green')
        plt.hist(fake_probs, bins=30, alpha=0.7, label='Fake', color='red')
        plt.xlabel('Predicted Probability (Fake)')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution by True Label')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Performance comparison bar chart
        metrics = ['Accuracy', 'F1-Score', 'AUC']
        values = [self.results['performance_metrics']['accuracy'],
                 self.results['performance_metrics']['f1_score'],
                 self.results['performance_metrics']['auc_roc']]

        bars = plt.bar(metrics, values, color=['blue', 'green', 'red'], alpha=0.7)
        plt.ylabel('Score')
        plt.title('Performance Metrics')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        dist_path = self.output_dir / "prediction_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['prediction_distribution'] = str(dist_path)

        return visualization_paths

    def generate_report(self, visualization_paths: Dict[str, str]) -> str:
        """Generate comprehensive report"""
        if not self.results:
            raise ValueError("No results available. Run test_model first.")

        report_lines = [
            "# Deepfake Detection Model Generalization Test Report",
            "",
            f"**Generated on**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model**: {self.config.model_architecture}",
            f"**Model Path**: {self.config.model_path}",
            f"**Dataset**: Deepfake-Eval-2024",
            f"**Device**: {self.device}",
            "",
            "## Executive Summary",
            ""
        ]

        # Performance summary
        metrics = self.results['performance_metrics']
        report_lines.extend([
            "### Key Performance Metrics",
            f"- **Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
            f"- **F1-Score**: {metrics['f1_score']:.4f}",
            f"- **AUC-ROC**: {metrics['auc_roc']:.4f}",
            f"- **Precision**: {metrics['precision']:.4f}",
            f"- **Recall**: {metrics['recall']:.4f}",
            "",
        ])

        # Dataset info
        dataset_info = self.results['dataset_info']
        report_lines.extend([
            "### Dataset Information",
            f"- **Total Samples**: {dataset_info['total_samples']}",
            f"- **Real Samples**: {dataset_info['real_samples']}",
            f"- **Fake Samples**: {dataset_info['fake_samples']}",
            f"- **Image Formats**: {', '.join(dataset_info['unique_formats'])}",
            "",
        ])

        # Performance analysis
        report_lines.extend([
            "### Performance Analysis",
            f"- **Correct Predictions**: {metrics['correct_predictions']}/{metrics['total_samples']}",
            f"- **False Positives**: {metrics['false_positives']} (Real incorrectly classified as Fake)",
            f"- **False Negatives**: {metrics['false_negatives']} (Fake incorrectly classified as Real)",
            "",
        ])

        # Confusion matrix details
        cm = self.results['confusion_matrix']
        report_lines.extend([
            "### Confusion Matrix",
            "|              | Predicted Real | Predicted Fake |",
            "|--------------|----------------|----------------|",
            f"| Actual Real  | {cm[0][0]:14d} | {cm[0][1]:14d} |",
            f"| Actual Fake  | {cm[1][0]:14d} | {cm[1][1]:14d} |",
            "",
        ])

        # Performance stats
        perf_stats = self.results['performance_stats']
        report_lines.extend([
            "### Processing Performance",
            f"- **Average Time per Batch**: {perf_stats['avg_processing_time_per_batch']:.3f} seconds",
            f"- **Throughput**: {perf_stats['throughput_samples_per_second']:.1f} samples/second",
            f"- **Average Time per Sample**: {perf_stats['avg_time_per_sample']*1000:.1f} ms",
            "",
        ])

        # Generalization assessment
        auc_score = metrics['auc_roc']
        if auc_score > 0.90:
            assessment = "Excellent generalization ability"
        elif auc_score > 0.80:
            assessment = "Good generalization ability"
        elif auc_score > 0.70:
            assessment = "Moderate generalization ability"
        else:
            assessment = "Poor generalization ability"

        report_lines.extend([
            "### Generalization Assessment",
            f"**Overall Assessment**: {assessment}",
            f"**AUC Interpretation**: The model achieves {auc_score:.3f} AUC on unseen real-world data.",
            "",
        ])

        # Recommendations based on performance
        report_lines.extend([
            "### Recommendations",
            ""
        ])

        if auc_score > 0.90:
            report_lines.extend([
                "✅ **Strong Performance**: The model generalizes well to real-world data.",
                "✅ **Proceed to Stage 02**: Consider moving directly to heterogeneous expert systems.",
                "✅ **Deployment Ready**: Model shows promising deployment potential.",
                ""
            ])
        elif auc_score > 0.80:
            report_lines.extend([
                "⚠️ **Moderate Performance**: Model shows some generalization but has room for improvement.",
                "🔍 **Consider Stage 01**: SupCon-based feature learning might help.",
                "🔍 **Error Analysis**: Investigate failure cases for insights.",
                ""
            ])
        else:
            report_lines.extend([
                "❌ **Limited Generalization**: Model struggles with real-world data.",
                "🚨 **Stage 01 Recommended**: Feature learning improvements are needed.",
                "🚨 **Architecture Review**: Consider model architecture changes.",
                ""
            ])

        # Visualizations
        if visualization_paths:
            report_lines.extend([
                "### Visualizations",
                ""
            ])
            for viz_name, viz_path in visualization_paths.items():
                if viz_path:
                    viz_filename = Path(viz_path).name
                    report_lines.extend([
                        f"#### {viz_name.replace('_', ' ').title()}",
                        f"![{viz_name}]({viz_filename})",
                        ""
                    ])

        # Save report
        report_path = self.output_dir / "generalization_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        return str(report_path)

    def save_results(self, visualization_paths: Dict[str, str]):
        """Save all results to files"""
        if not self.results:
            raise ValueError("No results available. Run test_model first.")

        # Save detailed results as JSON
        results_file = self.output_dir / "generalization_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save predictions as CSV
        if self.config.save_predictions and self.results['predictions']:
            pred_df = pd.DataFrame({
                'filename': self.results['predictions']['filenames'],
                'true_label': self.results['predictions']['labels'],
                'predicted_probability': self.results['predictions']['probabilities'],
                'predicted_label': self.results['predictions']['binary_predictions']
            })
            pred_file = self.output_dir / "predictions.csv"
            pred_df.to_csv(pred_file, index=False)

        # Generate and save report
        if self.config.generate_detailed_report:
            report_path = self.generate_report(visualization_paths)

        logging.info(f"Results saved to {self.output_dir}")
        return {
            'results_file': str(results_file),
            'predictions_file': str(self.output_dir / "predictions.csv") if self.config.save_predictions else None,
            'report_file': str(self.output_dir / "generalization_report.md") if self.config.generate_detailed_report else None,
            'visualizations': visualization_paths
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Test model generalization on Deepfake-Eval-2024")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data", required=True, help="Path to Deepfake-Eval-2024 dataset")
    parser.add_argument("--output", default="generalization_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--architecture", default="tf_efficientnetv2_b0",
                       help="Model architecture")
    parser.add_argument("--no-predictions", action="store_true",
                       help="Don't save individual predictions")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Don't generate visualizations")
    parser.add_argument("--no-report", action="store_true",
                       help="Don't generate detailed report")

    args = parser.parse_args()

    # Create configuration
    config = GeneralizationConfig(
        model_path=args.model,
        dataset_path=args.data,
        output_dir=args.output,
        batch_size=args.batch_size,
        model_architecture=args.architecture,
        save_predictions=not args.no_predictions,
        save_visualizations=not args.no_visualizations,
        generate_detailed_report=not args.no_report
    )

    print("=== Deepfake Detection Generalization Test ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Output: {args.output}")
    print()

    # Initialize tester
    tester = GeneralizationTester(config)

    try:
        # Load model and dataset
        tester.load_model()
        samples = tester.load_dataset()

        if not samples:
            print("Error: No valid samples found in dataset")
            return 1

        # Run testing
        print("Starting generalization test...")
        results = tester.test_model(samples)

        # Generate visualizations
        print("Generating visualizations...")
        visualization_paths = tester.generate_visualizations()

        # Save results
        print("Saving results...")
        saved_files = tester.save_results(visualization_paths)

        print("\n=== Generalization Test Completed ===")
        print(f"Accuracy: {results['performance_metrics']['accuracy']:.4f}")
        print(f"F1-Score: {results['performance_metrics']['f1_score']:.4f}")
        print(f"AUC-ROC: {results['performance_metrics']['auc_roc']:.4f}")
        print(f"\nResults saved to: {config.output_dir}")

        if saved_files['report_file']:
            print(f"Report: {saved_files['report_file']}")

        return 0

    except Exception as e:
        print(f"\nError during generalization test: {e}")
        logging.error(f"Generalization test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())