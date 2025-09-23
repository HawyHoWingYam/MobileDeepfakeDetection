"""
Reconstruction Analysis Tools for GenConViT Expert

This module provides comprehensive analysis tools for evaluating reconstruction quality
in the GenConViT generative expert, including SSIM, LPIPS, and perceptual quality metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from PIL import Image
import io
import base64
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import scipy.stats as stats

# For LPIPS - we'll implement a simplified version
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available. Using simplified perceptual metrics.")


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction analysis"""
    # Quality metrics
    enable_ssim: bool = True
    enable_psnr: bool = True
    enable_lpips: bool = True
    enable_mse: bool = True
    enable_perceptual_metrics: bool = True

    # SSIM settings
    ssim_window_size: int = 11
    ssim_multichannel: bool = True
    ssim_data_range: float = 1.0

    # Visualization settings
    save_reconstruction_pairs: bool = True
    save_difference_maps: bool = True
    save_quality_distributions: bool = True

    # Analysis settings
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "ssim_good": 0.85,
        "ssim_fair": 0.70,
        "psnr_good": 25.0,
        "psnr_fair": 20.0,
        "lpips_good": 0.1,
        "lpips_fair": 0.2
    })

    # Statistical analysis
    enable_statistical_analysis: bool = True
    statistical_tests: List[str] = field(default_factory=lambda: [
        "normality_test", "correlation_analysis", "outlier_detection"
    ])

    # Output settings
    output_dir: str = "reconstruction_analysis"
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 300


class ReconstructionQualityMetrics:
    """Comprehensive reconstruction quality assessment"""

    def __init__(self, config: ReconstructionConfig):
        self.config = config

        # Initialize LPIPS if available
        if LPIPS_AVAILABLE and config.enable_lpips:
            try:
                self.lpips_metric = lpips.LPIPS(net='alex', verbose=False)
                self.lpips_available = True
            except Exception as e:
                logging.warning(f"Failed to initialize LPIPS: {e}")
                self.lpips_available = False
        else:
            self.lpips_available = False

    def calculate_all_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate all reconstruction quality metrics"""
        metrics = {}

        # Convert tensors to numpy for skimage metrics
        orig_np = self._tensor_to_numpy(original)
        recon_np = self._tensor_to_numpy(reconstructed)

        # SSIM
        if self.config.enable_ssim:
            metrics['ssim'] = self._calculate_ssim(orig_np, recon_np)

        # PSNR
        if self.config.enable_psnr:
            metrics['psnr'] = self._calculate_psnr(orig_np, recon_np)

        # MSE
        if self.config.enable_mse:
            metrics['mse'] = self._calculate_mse(original, reconstructed)

        # LPIPS
        if self.config.enable_lpips and self.lpips_available:
            metrics['lpips'] = self._calculate_lpips(original, reconstructed)

        # Perceptual metrics
        if self.config.enable_perceptual_metrics:
            perceptual_metrics = self._calculate_perceptual_metrics(orig_np, recon_np)
            metrics.update(perceptual_metrics)

        return metrics

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for skimage"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Denormalize if needed (assuming ImageNet normalization)
        if tensor.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean

        tensor = torch.clamp(tensor, 0, 1)
        numpy_array = tensor.permute(1, 2, 0).cpu().numpy()

        return numpy_array

    def _calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate SSIM score"""
        return ssim(
            original,
            reconstructed,
            win_size=self.config.ssim_window_size,
            multichannel=self.config.ssim_multichannel,
            data_range=self.config.ssim_data_range
        )

    def _calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate PSNR score"""
        return psnr(original, reconstructed, data_range=self.config.ssim_data_range)

    def _calculate_mse(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Calculate MSE"""
        mse = F.mse_loss(original, reconstructed)
        return float(mse.item())

    def _calculate_lpips(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Calculate LPIPS perceptual distance"""
        if not self.lpips_available:
            return 0.0

        try:
            # Ensure tensors are in correct format for LPIPS
            if original.dim() == 3:
                original = original.unsqueeze(0)
            if reconstructed.dim() == 3:
                reconstructed = reconstructed.unsqueeze(0)

            # LPIPS expects values in [-1, 1]
            original_norm = original * 2.0 - 1.0
            reconstructed_norm = reconstructed * 2.0 - 1.0

            with torch.no_grad():
                lpips_score = self.lpips_metric(original_norm, reconstructed_norm)

            return float(lpips_score.item())

        except Exception as e:
            logging.warning(f"LPIPS calculation failed: {e}")
            return 0.0

    def _calculate_perceptual_metrics(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> Dict[str, float]:
        """Calculate additional perceptual quality metrics"""
        metrics = {}

        # Color histogram correlation
        orig_hist = self._calculate_color_histogram(original)
        recon_hist = self._calculate_color_histogram(reconstructed)
        metrics['color_histogram_correlation'] = np.corrcoef(orig_hist, recon_hist)[0, 1]

        # Edge preservation
        metrics['edge_preservation'] = self._calculate_edge_preservation(original, reconstructed)

        # Texture similarity
        metrics['texture_similarity'] = self._calculate_texture_similarity(original, reconstructed)

        # Contrast preservation
        metrics['contrast_preservation'] = self._calculate_contrast_preservation(original, reconstructed)

        return metrics

    def _calculate_color_histogram(self, image: np.ndarray, bins: int = 256) -> np.ndarray:
        """Calculate color histogram"""
        histograms = []
        for channel in range(image.shape[2]):
            hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0, 1))
            histograms.extend(hist)
        return np.array(histograms)

    def _calculate_edge_preservation(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate edge preservation metric"""
        # Convert to grayscale
        orig_gray = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        recon_gray = cv2.cvtColor((reconstructed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Edge detection
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        recon_edges = cv2.Canny(recon_gray, 50, 150)

        # Calculate correlation
        correlation = np.corrcoef(orig_edges.flatten(), recon_edges.flatten())[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def _calculate_texture_similarity(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate texture similarity using Local Binary Pattern"""
        from skimage.feature import local_binary_pattern

        # Convert to grayscale
        orig_gray = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        recon_gray = cv2.cvtColor((reconstructed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        orig_lbp = local_binary_pattern(orig_gray, n_points, radius, method='uniform')
        recon_lbp = local_binary_pattern(recon_gray, n_points, radius, method='uniform')

        # Calculate histogram correlation
        orig_hist, _ = np.histogram(orig_lbp.flatten(), bins=50)
        recon_hist, _ = np.histogram(recon_lbp.flatten(), bins=50)

        correlation = np.corrcoef(orig_hist, recon_hist)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def _calculate_contrast_preservation(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate contrast preservation metric"""
        # Calculate contrast using standard deviation
        orig_contrast = np.std(original)
        recon_contrast = np.std(reconstructed)

        # Calculate preservation ratio
        contrast_ratio = min(orig_contrast, recon_contrast) / max(orig_contrast, recon_contrast)
        return contrast_ratio


class ReconstructionAnalyzer:
    """Comprehensive reconstruction analysis and monitoring"""

    def __init__(self, config: ReconstructionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.quality_metrics = ReconstructionQualityMetrics(config)
        self.logger = logging.getLogger(__name__)

        # Storage for batch analysis
        self.batch_metrics = defaultdict(list)
        self.batch_counter = 0

    def analyze_reconstruction_batch(
        self,
        original_batch: torch.Tensor,
        reconstructed_batch: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        save_samples: bool = True
    ) -> Dict[str, Any]:
        """Analyze a batch of reconstructions"""
        batch_size = original_batch.size(0)
        individual_results = []
        batch_metrics = defaultdict(list)

        for i in range(batch_size):
            # Calculate metrics for individual sample
            original = original_batch[i:i+1]
            reconstructed = reconstructed_batch[i:i+1]

            metrics = self.quality_metrics.calculate_all_metrics(original, reconstructed)

            # Store individual result
            result = {
                "sample_index": i,
                "metrics": metrics,
                "label": labels[i].item() if labels is not None else None
            }
            individual_results.append(result)

            # Aggregate for batch statistics
            for metric_name, value in metrics.items():
                batch_metrics[metric_name].append(value)

        # Calculate batch statistics
        batch_stats = self._calculate_batch_statistics(batch_metrics)

        # Quality classification
        quality_classification = self._classify_reconstruction_quality(batch_metrics)

        # Create comprehensive analysis
        analysis_result = {
            "batch_size": batch_size,
            "individual_results": individual_results,
            "batch_statistics": batch_stats,
            "quality_classification": quality_classification
        }

        # Save visualizations if requested
        if save_samples:
            visualization_result = self._create_batch_visualization(
                original_batch, reconstructed_batch, batch_metrics, analysis_result
            )
            analysis_result["visualization"] = visualization_result

        # Update running statistics
        self._update_running_statistics(batch_metrics)

        return analysis_result

    def _calculate_batch_statistics(self, batch_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate comprehensive batch statistics"""
        stats = {}

        for metric_name, values in batch_metrics.items():
            if values:
                stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75))
                }

                # Statistical tests if enabled
                if self.config.enable_statistical_analysis:
                    stats[metric_name]["statistical_tests"] = self._perform_statistical_tests(values)

        return stats

    def _perform_statistical_tests(self, values: List[float]) -> Dict[str, Any]:
        """Perform statistical tests on metric values"""
        tests = {}

        # Normality test
        if "normality_test" in self.config.statistical_tests and len(values) > 3:
            statistic, p_value = stats.shapiro(values)
            tests["normality_test"] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05
            }

        # Outlier detection
        if "outlier_detection" in self.config.statistical_tests:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            outlier_bounds = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
            outliers = [v for v in values if v < outlier_bounds[0] or v > outlier_bounds[1]]

            tests["outlier_detection"] = {
                "num_outliers": len(outliers),
                "outlier_ratio": len(outliers) / len(values),
                "outlier_bounds": outlier_bounds,
                "outliers": outliers
            }

        return tests

    def _classify_reconstruction_quality(self, batch_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Classify reconstruction quality based on thresholds"""
        classification = {
            "overall_quality": "unknown",
            "quality_distribution": {},
            "metric_classifications": {}
        }

        # Classify based on individual metrics
        for metric_name, values in batch_metrics.items():
            if metric_name in self.config.quality_thresholds:
                thresholds = self.config.quality_thresholds
                avg_value = np.mean(values)

                if metric_name.startswith('ssim'):
                    if avg_value >= thresholds.get(f"{metric_name}_good", 0.85):
                        quality = "good"
                    elif avg_value >= thresholds.get(f"{metric_name}_fair", 0.70):
                        quality = "fair"
                    else:
                        quality = "poor"

                elif metric_name.startswith('psnr'):
                    if avg_value >= thresholds.get(f"{metric_name}_good", 25.0):
                        quality = "good"
                    elif avg_value >= thresholds.get(f"{metric_name}_fair", 20.0):
                        quality = "fair"
                    else:
                        quality = "poor"

                elif metric_name.startswith('lpips'):
                    if avg_value <= thresholds.get(f"{metric_name}_good", 0.1):
                        quality = "good"
                    elif avg_value <= thresholds.get(f"{metric_name}_fair", 0.2):
                        quality = "fair"
                    else:
                        quality = "poor"

                else:
                    quality = "unknown"

                classification["metric_classifications"][metric_name] = quality

        # Overall quality assessment
        metric_qualities = list(classification["metric_classifications"].values())
        if metric_qualities:
            good_count = metric_qualities.count("good")
            fair_count = metric_qualities.count("fair")
            poor_count = metric_qualities.count("poor")

            if good_count >= len(metric_qualities) * 0.7:
                classification["overall_quality"] = "good"
            elif (good_count + fair_count) >= len(metric_qualities) * 0.7:
                classification["overall_quality"] = "fair"
            else:
                classification["overall_quality"] = "poor"

            classification["quality_distribution"] = {
                "good": good_count,
                "fair": fair_count,
                "poor": poor_count
            }

        return classification

    def _create_batch_visualization(
        self,
        original_batch: torch.Tensor,
        reconstructed_batch: torch.Tensor,
        batch_metrics: Dict[str, List[float]],
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive visualization for batch analysis"""

        # Select representative samples
        num_samples = min(4, original_batch.size(0))
        sample_indices = np.linspace(0, original_batch.size(0) - 1, num_samples, dtype=int)

        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Create grid layout
        rows = 3
        cols = max(4, num_samples)

        # Plot sample reconstructions
        for i, idx in enumerate(sample_indices):
            # Original image
            plt.subplot(rows, cols, i + 1)
            orig_img = self._tensor_to_display_image(original_batch[idx])
            plt.imshow(orig_img)
            plt.title(f"Original {idx}")
            plt.axis('off')

            # Reconstructed image
            plt.subplot(rows, cols, i + 1 + cols)
            recon_img = self._tensor_to_display_image(reconstructed_batch[idx])
            plt.imshow(recon_img)

            # Add quality metrics as title
            sample_metrics = analysis_result["individual_results"][idx]["metrics"]
            title = f"Recon {idx}\n"
            if 'ssim' in sample_metrics:
                title += f"SSIM: {sample_metrics['ssim']:.3f}\n"
            if 'psnr' in sample_metrics:
                title += f"PSNR: {sample_metrics['psnr']:.1f}"

            plt.title(title, fontsize=8)
            plt.axis('off')

            # Difference map
            plt.subplot(rows, cols, i + 1 + 2 * cols)
            diff_map = self._create_difference_map(original_batch[idx], reconstructed_batch[idx])
            plt.imshow(diff_map, cmap='hot')
            plt.title(f"Diff {idx}")
            plt.axis('off')

        # Plot metric distributions
        self._plot_metric_distributions(batch_metrics, fig, rows, cols)

        plt.tight_layout()

        # Save visualization
        save_path = self.output_dir / f"reconstruction_analysis_batch_{self.batch_counter}.png"
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        self.batch_counter += 1

        return {
            "visualization_path": str(save_path),
            "base64_image": image_base64
        }

    def _tensor_to_display_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Denormalize if needed
        if tensor.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean

        tensor = torch.clamp(tensor, 0, 1)
        numpy_array = tensor.permute(1, 2, 0).cpu().numpy()

        return numpy_array

    def _create_difference_map(self, original: torch.Tensor, reconstructed: torch.Tensor) -> np.ndarray:
        """Create difference map between original and reconstructed images"""
        orig_img = self._tensor_to_display_image(original)
        recon_img = self._tensor_to_display_image(reconstructed)

        # Calculate absolute difference
        diff = np.abs(orig_img - recon_img)

        # Convert to grayscale for visualization
        diff_gray = np.mean(diff, axis=2)

        return diff_gray

    def _plot_metric_distributions(
        self,
        batch_metrics: Dict[str, List[float]],
        fig: plt.Figure,
        rows: int,
        cols: int
    ):
        """Plot metric distributions"""
        metric_names = ['ssim', 'psnr', 'lpips', 'mse']
        available_metrics = [m for m in metric_names if m in batch_metrics]

        for i, metric_name in enumerate(available_metrics[:4]):
            if i < cols:
                plt.subplot(rows, cols, (rows - 1) * cols + i + 1)
                values = batch_metrics[metric_name]

                plt.hist(values, bins=min(20, len(values)), alpha=0.7, edgecolor='black')
                plt.axvline(np.mean(values), color='red', linestyle='--', label='Mean')
                plt.xlabel(metric_name.upper())
                plt.ylabel('Frequency')
                plt.title(f'{metric_name.upper()} Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)

    def _update_running_statistics(self, batch_metrics: Dict[str, List[float]]):
        """Update running statistics across batches"""
        for metric_name, values in batch_metrics.items():
            self.batch_metrics[metric_name].extend(values)

    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall statistics across all processed batches"""
        if not self.batch_metrics:
            return {}

        overall_stats = {}
        for metric_name, all_values in self.batch_metrics.items():
            if all_values:
                overall_stats[metric_name] = {
                    "total_samples": len(all_values),
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                    "min": float(np.min(all_values)),
                    "max": float(np.max(all_values)),
                    "median": float(np.median(all_values))
                }

        return overall_stats


def test_reconstruction_analysis():
    """Test function for reconstruction analysis tools"""
    print("Testing Reconstruction Analysis Tools...")

    # Create dummy data
    original = torch.randn(4, 3, 256, 256)
    reconstructed = original + torch.randn_like(original) * 0.1  # Add some noise

    # Create configuration
    config = ReconstructionConfig(
        output_dir="test_reconstruction_analysis",
        enable_lpips=False  # Disable LPIPS for testing
    )

    try:
        # Test analyzer
        analyzer = ReconstructionAnalyzer(config)
        result = analyzer.analyze_reconstruction_batch(original, reconstructed, save_samples=True)

        print("✓ Reconstruction Analysis completed successfully")
        print(f"✓ Batch statistics calculated: {len(result['batch_statistics'])} metrics")
        print(f"✓ Quality classification: {result['quality_classification']['overall_quality']}")

        # Test overall statistics
        overall_stats = analyzer.get_overall_statistics()
        print(f"✓ Overall statistics: {len(overall_stats)} metrics")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")


if __name__ == "__main__":
    test_reconstruction_analysis()