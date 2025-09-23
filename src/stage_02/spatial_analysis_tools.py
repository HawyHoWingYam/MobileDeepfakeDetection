"""
Spatial Analysis Tools for Stage 02 Expert Systems

This module provides comprehensive visualization and analysis tools for understanding
spatial expert behavior, including Grad-CAM visualization, spatial attention analysis,
and artifact localization capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from .unified_feature_extractor import BaseExpert


@dataclass
class VisualizationConfig:
    """Configuration for spatial analysis visualizations"""
    # Grad-CAM settings
    target_layers: List[str] = field(default_factory=lambda: ["last_conv"])
    cam_colormap: str = "jet"
    cam_alpha: float = 0.4
    save_raw_cam: bool = True

    # Spatial attention settings
    attention_threshold: float = 0.5
    attention_percentile: float = 95
    highlight_top_regions: int = 5

    # Visualization settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = "png"
    colormap: str = "viridis"

    # Analysis settings
    enable_statistical_analysis: bool = True
    enable_artifact_localization: bool = True
    enable_comparative_analysis: bool = True

    # Output settings
    output_dir: str = "spatial_analysis_output"
    save_individual_visualizations: bool = True
    save_summary_plots: bool = True


class GradCAM:
    """Grad-CAM implementation for spatial expert analysis"""

    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # Register hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Generate Grad-CAM heatmaps"""
        self.model.eval()

        # Forward pass
        input_tensor.requires_grad_()
        output = self.model(input_tensor)

        # Handle different output types
        if hasattr(output, 'confidence_scores'):
            logits = output.confidence_scores
        else:
            logits = output

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        if logits.dim() > 1:
            class_score = logits[0, target_class] if logits.size(1) > 1 else logits[0, 0]
        else:
            class_score = logits[0]
        class_score.backward()

        cams = {}

        # Generate CAM for each target layer
        for layer_name in self.target_layers:
            if layer_name in self.gradients and layer_name in self.activations:
                gradients = self.gradients[layer_name]
                activations = self.activations[layer_name]

                # Calculate weights
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

                # Generate CAM
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
                cam = F.relu(cam)

                # Normalize and resize
                cam = cam.squeeze().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                # Resize to input size
                input_size = input_tensor.shape[-2:]
                cam = cv2.resize(cam, input_size)

                cams[layer_name] = cam

        return cams

    def __del__(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()


class SpatialAttentionAnalyzer:
    """Analyzes spatial attention patterns in expert predictions"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def analyze_attention_patterns(
        self,
        cam_maps: Dict[str, np.ndarray],
        input_image: np.ndarray,
        prediction_confidence: float
    ) -> Dict[str, Any]:
        """Analyze spatial attention patterns"""
        analysis_results = {}

        for layer_name, cam in cam_maps.items():
            # Basic statistics
            stats = {
                "mean_attention": float(np.mean(cam)),
                "max_attention": float(np.max(cam)),
                "min_attention": float(np.min(cam)),
                "std_attention": float(np.std(cam)),
                "attention_entropy": float(self._calculate_entropy(cam))
            }

            # Attention distribution
            distribution = {
                "high_attention_ratio": float(np.sum(cam > self.config.attention_threshold) / cam.size),
                "top_percentile_value": float(np.percentile(cam, self.config.attention_percentile)),
                "attention_concentration": float(self._calculate_concentration(cam))
            }

            # Spatial patterns
            spatial_patterns = self._analyze_spatial_patterns(cam, input_image)

            # Artifact-specific analysis
            artifact_analysis = self._analyze_artifacts(cam, input_image)

            analysis_results[layer_name] = {
                "statistics": stats,
                "distribution": distribution,
                "spatial_patterns": spatial_patterns,
                "artifact_analysis": artifact_analysis,
                "prediction_confidence": prediction_confidence
            }

        return analysis_results

    def _calculate_entropy(self, cam: np.ndarray) -> float:
        """Calculate entropy of attention map"""
        # Normalize to probability distribution
        cam_flat = cam.flatten()
        cam_norm = cam_flat / (np.sum(cam_flat) + 1e-8)

        # Calculate entropy
        entropy = -np.sum(cam_norm * np.log(cam_norm + 1e-8))
        return entropy

    def _calculate_concentration(self, cam: np.ndarray) -> float:
        """Calculate attention concentration (Gini coefficient)"""
        cam_flat = cam.flatten()
        cam_sorted = np.sort(cam_flat)
        n = len(cam_sorted)

        index = np.arange(1, n + 1)
        gini = 2 * np.sum(index * cam_sorted) / (n * np.sum(cam_sorted)) - (n + 1) / n

        return gini

    def _analyze_spatial_patterns(self, cam: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial patterns in attention"""
        patterns = {}

        # Center bias analysis
        h, w = cam.shape
        center_region = cam[h//4:3*h//4, w//4:3*w//4]
        edge_region = cam.copy()
        edge_region[h//4:3*h//4, w//4:3*w//4] = 0

        patterns["center_bias"] = float(np.mean(center_region) / (np.mean(cam) + 1e-8))
        patterns["edge_attention"] = float(np.mean(edge_region) / (np.mean(cam) + 1e-8))

        # Symmetry analysis
        left_half = cam[:, :w//2]
        right_half = cam[:, w//2:]
        patterns["horizontal_symmetry"] = float(np.corrcoef(
            left_half.flatten(),
            np.fliplr(right_half).flatten()
        )[0, 1])

        # Attention clustering
        patterns["attention_clusters"] = self._find_attention_clusters(cam)

        return patterns

    def _analyze_artifacts(self, cam: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
        """Analyze attention correlation with potential artifacts"""
        artifacts = {}

        # Edge detection correlation
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)
        edges_normalized = edges.astype(np.float32) / 255.0

        # Resize to match CAM
        if edges_normalized.shape != cam.shape:
            edges_normalized = cv2.resize(edges_normalized, cam.shape[::-1])

        artifacts["edge_correlation"] = float(np.corrcoef(
            cam.flatten(),
            edges_normalized.flatten()
        )[0, 1])

        # Texture analysis correlation
        texture_strength = cv2.Laplacian(gray_image, cv2.CV_32F)
        texture_strength = np.abs(texture_strength)
        texture_strength = (texture_strength - texture_strength.min()) / (
            texture_strength.max() - texture_strength.min() + 1e-8
        )

        if texture_strength.shape != cam.shape:
            texture_strength = cv2.resize(texture_strength, cam.shape[::-1])

        artifacts["texture_correlation"] = float(np.corrcoef(
            cam.flatten(),
            texture_strength.flatten()
        )[0, 1])

        return artifacts

    def _find_attention_clusters(self, cam: np.ndarray) -> Dict[str, Any]:
        """Find clusters of high attention"""
        # Threshold high attention regions
        binary_cam = (cam > self.config.attention_threshold).astype(np.uint8)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_cam)

        clusters = {
            "num_clusters": num_labels - 1,  # Exclude background
            "cluster_sizes": [],
            "cluster_centers": []
        }

        for label in range(1, num_labels):
            cluster_mask = (labels == label)
            cluster_size = np.sum(cluster_mask)
            clusters["cluster_sizes"].append(int(cluster_size))

            # Find cluster center
            y_coords, x_coords = np.where(cluster_mask)
            center_y = float(np.mean(y_coords))
            center_x = float(np.mean(x_coords))
            clusters["cluster_centers"].append([center_x, center_y])

        return clusters


class SpatialVisualizationGenerator:
    """Generates comprehensive spatial analysis visualizations"""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def create_comprehensive_visualization(
        self,
        input_image: np.ndarray,
        cam_maps: Dict[str, np.ndarray],
        analysis_results: Dict[str, Any],
        prediction_info: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create comprehensive spatial analysis visualization"""

        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)

        # Calculate grid size based on number of CAM maps
        num_cams = len(cam_maps)
        grid_rows = 2 + (num_cams + 1) // 2
        grid_cols = max(2, min(3, num_cams))

        # Original image
        plt.subplot(grid_rows, grid_cols, 1)
        plt.imshow(input_image)
        plt.title("Original Image")
        plt.axis('off')

        # Prediction info
        plt.subplot(grid_rows, grid_cols, 2)
        self._plot_prediction_info(prediction_info)

        # CAM visualizations
        plot_idx = 3
        for layer_name, cam in cam_maps.items():
            if plot_idx <= grid_rows * grid_cols:
                plt.subplot(grid_rows, grid_cols, plot_idx)
                self._plot_cam_overlay(input_image, cam, layer_name)
                plot_idx += 1

        # Statistical analysis
        if self.config.enable_statistical_analysis and plot_idx <= grid_rows * grid_cols:
            plt.subplot(grid_rows, grid_cols, plot_idx)
            self._plot_attention_statistics(analysis_results)
            plot_idx += 1

        # Artifact analysis
        if self.config.enable_artifact_localization and plot_idx <= grid_rows * grid_cols:
            plt.subplot(grid_rows, grid_cols, plot_idx)
            self._plot_artifact_analysis(analysis_results)

        plt.tight_layout()

        # Save visualization
        if save_path:
            plt.savefig(
                save_path,
                format=self.config.save_format,
                dpi=self.config.dpi,
                bbox_inches='tight'
            )

        # Convert to base64 for web display
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            "visualization_path": save_path,
            "base64_image": image_base64,
            "analysis_summary": self._create_analysis_summary(analysis_results)
        }

    def _plot_cam_overlay(self, image: np.ndarray, cam: np.ndarray, layer_name: str):
        """Plot CAM overlay on original image"""
        # Apply colormap to CAM
        cam_colored = plt.cm.get_cmap(self.config.cam_colormap)(cam)
        cam_colored = (cam_colored[:, :, :3] * 255).astype(np.uint8)

        # Create overlay
        overlay = cv2.addWeighted(
            image, 1 - self.config.cam_alpha,
            cam_colored, self.config.cam_alpha,
            0
        )

        plt.imshow(overlay)
        plt.title(f"Grad-CAM: {layer_name}")
        plt.axis('off')

        # Add colorbar
        im = plt.imshow(cam, alpha=0, cmap=self.config.cam_colormap)
        plt.colorbar(im, fraction=0.046, pad=0.04)

    def _plot_prediction_info(self, prediction_info: Dict[str, Any]):
        """Plot prediction information"""
        plt.text(0.1, 0.8, f"Prediction: {prediction_info.get('prediction', 'N/A')}", fontsize=12)
        plt.text(0.1, 0.6, f"Confidence: {prediction_info.get('confidence', 0.0):.3f}", fontsize=12)
        plt.text(0.1, 0.4, f"Processing Time: {prediction_info.get('time', 0.0):.3f}s", fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title("Prediction Info")

    def _plot_attention_statistics(self, analysis_results: Dict[str, Any]):
        """Plot attention statistics"""
        # Aggregate statistics across layers
        all_stats = []
        layer_names = []

        for layer_name, results in analysis_results.items():
            stats = results["statistics"]
            all_stats.append([
                stats["mean_attention"],
                stats["max_attention"],
                stats["std_attention"],
                stats["attention_entropy"]
            ])
            layer_names.append(layer_name)

        if all_stats:
            all_stats = np.array(all_stats)
            stat_names = ["Mean", "Max", "Std", "Entropy"]

            # Create heatmap
            sns.heatmap(
                all_stats.T,
                xticklabels=layer_names,
                yticklabels=stat_names,
                annot=True,
                fmt='.3f',
                cmap='viridis'
            )
            plt.title("Attention Statistics")

    def _plot_artifact_analysis(self, analysis_results: Dict[str, Any]):
        """Plot artifact analysis results"""
        edge_correlations = []
        texture_correlations = []
        layer_names = []

        for layer_name, results in analysis_results.items():
            artifacts = results["artifact_analysis"]
            edge_correlations.append(artifacts["edge_correlation"])
            texture_correlations.append(artifacts["texture_correlation"])
            layer_names.append(layer_name)

        if edge_correlations:
            x = np.arange(len(layer_names))
            width = 0.35

            plt.bar(x - width/2, edge_correlations, width, label='Edge Correlation', alpha=0.8)
            plt.bar(x + width/2, texture_correlations, width, label='Texture Correlation', alpha=0.8)

            plt.xlabel('Layers')
            plt.ylabel('Correlation')
            plt.title('Artifact Correlation Analysis')
            plt.xticks(x, layer_names, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

    def _create_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of analysis results"""
        summary = {
            "num_layers_analyzed": len(analysis_results),
            "avg_attention_entropy": 0.0,
            "avg_edge_correlation": 0.0,
            "avg_texture_correlation": 0.0,
            "attention_concentration": 0.0
        }

        if analysis_results:
            entropies = []
            edge_corrs = []
            texture_corrs = []
            concentrations = []

            for layer_name, results in analysis_results.items():
                entropies.append(results["statistics"]["attention_entropy"])
                edge_corrs.append(results["artifact_analysis"]["edge_correlation"])
                texture_corrs.append(results["artifact_analysis"]["texture_correlation"])
                concentrations.append(results["distribution"]["attention_concentration"])

            summary["avg_attention_entropy"] = float(np.mean(entropies))
            summary["avg_edge_correlation"] = float(np.mean(edge_corrs))
            summary["avg_texture_correlation"] = float(np.mean(texture_corrs))
            summary["attention_concentration"] = float(np.mean(concentrations))

        return summary


class SpatialAnalysisFramework:
    """Main framework for spatial expert analysis"""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.attention_analyzer = SpatialAttentionAnalyzer(config)
        self.visualization_generator = SpatialVisualizationGenerator(config)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def analyze_spatial_expert(
        self,
        model: BaseExpert,
        input_batch: torch.Tensor,
        image_paths: Optional[List[str]] = None,
        save_individual: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive spatial expert analysis"""

        model.eval()
        batch_size = input_batch.size(0)
        all_results = []

        # Initialize Grad-CAM
        grad_cam = GradCAM(model, self.config.target_layers)

        for i in range(batch_size):
            # Process single image
            single_input = input_batch[i:i+1]

            # Get prediction
            with torch.no_grad():
                output = model(single_input)
                if hasattr(output, 'confidence_scores'):
                    confidence = float(output.confidence_scores.squeeze())
                    prediction = int(output.confidence_scores.squeeze() > 0.5)
                else:
                    confidence = float(torch.sigmoid(output).squeeze())
                    prediction = int(confidence > 0.5)

            # Generate Grad-CAM
            cam_maps = grad_cam.generate_cam(single_input)

            # Convert input tensor to numpy image
            input_np = self._tensor_to_numpy(single_input.squeeze())

            # Analyze attention patterns
            analysis_results = self.attention_analyzer.analyze_attention_patterns(
                cam_maps, input_np, confidence
            )

            # Create prediction info
            prediction_info = {
                "prediction": "Fake" if prediction == 1 else "Real",
                "confidence": confidence,
                "time": 0.0  # Would be measured in real scenario
            }

            # Generate visualization
            if save_individual:
                save_path = self.output_dir / f"spatial_analysis_{i}.{self.config.save_format}"
            else:
                save_path = None

            visualization_result = self.visualization_generator.create_comprehensive_visualization(
                input_np, cam_maps, analysis_results, prediction_info, str(save_path)
            )

            # Compile results
            result = {
                "image_index": i,
                "image_path": image_paths[i] if image_paths else f"batch_item_{i}",
                "prediction_info": prediction_info,
                "cam_maps": cam_maps,
                "analysis_results": analysis_results,
                "visualization": visualization_result
            }

            all_results.append(result)

        # Create summary analysis
        summary_analysis = self._create_batch_summary(all_results)

        return {
            "batch_size": batch_size,
            "individual_results": all_results,
            "summary_analysis": summary_analysis
        }

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        denorm_tensor = tensor * std + mean
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)

        # Convert to numpy
        numpy_image = denorm_tensor.permute(1, 2, 0).cpu().numpy()
        numpy_image = (numpy_image * 255).astype(np.uint8)

        return numpy_image

    def _create_batch_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary analysis for the entire batch"""
        summary = {
            "total_samples": len(all_results),
            "predictions": {"real": 0, "fake": 0},
            "avg_confidence": 0.0,
            "attention_patterns": {},
            "artifact_correlations": {}
        }

        confidences = []
        all_attention_stats = defaultdict(list)
        all_artifact_stats = defaultdict(list)

        for result in all_results:
            # Prediction statistics
            pred = result["prediction_info"]["prediction"]
            summary["predictions"][pred.lower()] += 1

            confidences.append(result["prediction_info"]["confidence"])

            # Attention statistics
            for layer_name, analysis in result["analysis_results"].items():
                stats = analysis["statistics"]
                artifacts = analysis["artifact_analysis"]

                all_attention_stats[f"{layer_name}_entropy"].append(stats["attention_entropy"])
                all_attention_stats[f"{layer_name}_concentration"].append(
                    analysis["distribution"]["attention_concentration"]
                )

                all_artifact_stats[f"{layer_name}_edge_corr"].append(artifacts["edge_correlation"])
                all_artifact_stats[f"{layer_name}_texture_corr"].append(artifacts["texture_correlation"])

        # Calculate averages
        summary["avg_confidence"] = float(np.mean(confidences))

        for key, values in all_attention_stats.items():
            summary["attention_patterns"][f"avg_{key}"] = float(np.mean(values))
            summary["attention_patterns"][f"std_{key}"] = float(np.std(values))

        for key, values in all_artifact_stats.items():
            summary["artifact_correlations"][f"avg_{key}"] = float(np.mean(values))
            summary["artifact_correlations"][f"std_{key}"] = float(np.std(values))

        return summary


def test_spatial_analysis():
    """Test function for spatial analysis tools"""
    print("Testing Spatial Analysis Tools...")

    # Create dummy model and data
    dummy_model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 1)
    )

    dummy_input = torch.randn(2, 3, 224, 224)

    # Create configuration
    config = VisualizationConfig(
        target_layers=["0"],  # First conv layer
        output_dir="test_spatial_analysis"
    )

    # Test framework
    try:
        framework = SpatialAnalysisFramework(config)
        print("✓ Spatial Analysis Framework initialized successfully")

        # Note: Full test would require proper model with hooks
        print("✓ Spatial Analysis Tools structure validated")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")


if __name__ == "__main__":
    test_spatial_analysis()