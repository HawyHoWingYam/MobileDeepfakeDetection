"""
Spatial Artifact-Preserving Data Augmentation for Stage 02 Experts

This module provides specialized data augmentation strategies that preserve spatial artifacts
critical for deepfake detection while providing necessary data diversity for robust training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from pathlib import Path


@dataclass
class AugmentationConfig:
    """Configuration for artifact-preserving augmentations"""
    # Spatial augmentations (preserve spatial artifacts)
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 5.0
    scale_range: Tuple[float, float] = (0.9, 1.0)
    shear_range: Tuple[float, float] = (-2, 2)

    # Photometric augmentations (mild to preserve artifacts)
    brightness_range: float = 0.1
    contrast_range: float = 0.1
    saturation_range: float = 0.1
    hue_range: float = 0.05

    # Noise augmentations (artifact-aware)
    gaussian_noise_std: float = 0.01
    gaussian_noise_prob: float = 0.3
    motion_blur_limit: int = 3
    motion_blur_prob: float = 0.2

    # Compression simulation
    jpeg_quality_range: Tuple[int, int] = (85, 100)
    jpeg_compression_prob: float = 0.3

    # Edge-preserving augmentations
    edge_preservation_ratio: float = 0.9
    texture_consistency_check: bool = True

    # Expert-specific configurations
    spatial_expert_mode: bool = True
    generative_expert_mode: bool = False


class EdgePreservingAugmentation:
    """Augmentations that preserve edge information critical for spatial experts"""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def apply_edge_preserving_blur(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply blur while preserving strong edges"""
        # Use bilateral filter to preserve edges
        blurred = cv2.bilateralFilter(image, kernel_size, 75, 75)

        # Blend with original based on edge strength
        if self.config.edge_preservation_ratio < 1.0:
            # Calculate edge strength
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_mask = edges.astype(np.float32) / 255.0
            edge_mask = np.stack([edge_mask] * 3, axis=-1)

            # Preserve edges based on edge strength
            result = (blurred * (1 - edge_mask * self.config.edge_preservation_ratio) +
                     image * edge_mask * self.config.edge_preservation_ratio)
            return result.astype(np.uint8)

        return blurred

    def apply_texture_preserving_noise(self, image: np.ndarray, noise_std: float) -> np.ndarray:
        """Add noise while preserving texture consistency"""
        # Generate noise
        noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)

        if self.config.texture_consistency_check:
            # Reduce noise in high-texture areas
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            texture_strength = cv2.Laplacian(gray, cv2.CV_32F)
            texture_strength = np.abs(texture_strength)
            texture_strength = (texture_strength - texture_strength.min()) / (texture_strength.max() - texture_strength.min() + 1e-8)

            # Reduce noise where texture is strong
            noise_reduction = np.stack([texture_strength] * 3, axis=-1)
            noise = noise * (1 - noise_reduction * 0.5)

        # Apply noise
        noisy_image = image.astype(np.float32) + noise * 255
        return np.clip(noisy_image, 0, 255).astype(np.uint8)


class CompressionSimulation:
    """Simulate compression artifacts that affect deepfake detection"""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def apply_jpeg_compression(self, image: np.ndarray, quality: int) -> np.ndarray:
        """Simulate JPEG compression"""
        # Convert to PIL format for JPEG simulation
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

    def apply_adaptive_compression(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive compression based on image content"""
        # Analyze image complexity
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        complexity = np.std(gray)

        # Adjust compression quality based on complexity
        if complexity > 50:  # High complexity
            quality = random.randint(self.config.jpeg_quality_range[1] - 5, self.config.jpeg_quality_range[1])
        else:  # Low complexity
            quality = random.randint(self.config.jpeg_quality_range[0], self.config.jpeg_quality_range[1] - 5)

        return self.apply_jpeg_compression(image, quality)


class SpatialExpertAugmentation:
    """Augmentations specifically designed for spatial artifact detection expert"""

    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.edge_preserving = EdgePreservingAugmentation(config)
        self.compression_sim = CompressionSimulation(config)

        # Build albumentations pipeline
        self.spatial_pipeline = A.Compose([
            # Geometric transformations (mild to preserve spatial relationships)
            A.HorizontalFlip(p=config.horizontal_flip_prob),
            A.Rotate(
                limit=config.rotation_degrees,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=0.7
            ),
            A.Affine(
                scale=config.scale_range,
                shear=config.shear_range,
                interpolation=cv2.INTER_LINEAR,
                p=0.5
            ),

            # Photometric transformations (conservative)
            A.ColorJitter(
                brightness=config.brightness_range,
                contrast=config.contrast_range,
                saturation=config.saturation_range,
                hue=config.hue_range,
                p=0.6
            ),

            # Specialized noise augmentations
            A.OneOf([
                A.GaussNoise(var_limit=(0, config.gaussian_noise_std * 255), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=config.gaussian_noise_prob),

            # Motion blur (mild)
            A.MotionBlur(blur_limit=config.motion_blur_limit, p=config.motion_blur_prob),

            # Final normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply spatial expert augmentations"""
        # Apply custom edge-preserving augmentations first
        if random.random() < self.config.motion_blur_prob:
            image = self.edge_preserving.apply_edge_preserving_blur(image)

        if random.random() < self.config.gaussian_noise_prob:
            image = self.edge_preserving.apply_texture_preserving_noise(
                image, self.config.gaussian_noise_std
            )

        # Apply compression simulation
        if random.random() < self.config.jpeg_compression_prob:
            image = self.compression_sim.apply_adaptive_compression(image)

        # Apply standard augmentations
        augmented = self.spatial_pipeline(image=image)
        return augmented['image']


class GenerativeExpertAugmentation:
    """Augmentations specifically designed for generative structure detection expert"""

    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.compression_sim = CompressionSimulation(config)

        # Build pipeline for generative expert (more aggressive augmentations)
        self.generative_pipeline = A.Compose([
            # More aggressive geometric transformations
            A.HorizontalFlip(p=config.horizontal_flip_prob),
            A.Rotate(
                limit=config.rotation_degrees * 1.5,  # More rotation for generative
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=0.8
            ),
            A.Affine(
                scale=(config.scale_range[0] - 0.05, config.scale_range[1] + 0.05),
                shear=(config.shear_range[0] - 1, config.shear_range[1] + 1),
                interpolation=cv2.INTER_LINEAR,
                p=0.6
            ),

            # More diverse photometric changes
            A.ColorJitter(
                brightness=config.brightness_range * 1.2,
                contrast=config.contrast_range * 1.2,
                saturation=config.saturation_range * 1.2,
                hue=config.hue_range * 1.2,
                p=0.7
            ),

            # Additional augmentations for generative patterns
            A.OneOf([
                A.GaussNoise(var_limit=(0, config.gaussian_noise_std * 255 * 1.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, p=1.0),
                A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8), p=1.0),
            ], p=config.gaussian_noise_prob),

            # More aggressive blur
            A.OneOf([
                A.MotionBlur(blur_limit=config.motion_blur_limit + 2, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=config.motion_blur_prob),

            # Additional augmentations for generative artifacts
            A.OneOf([
                A.Downscale(scale_min=0.75, scale_max=0.95, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.ImageCompression(quality_lower=70, quality_upper=100, p=1.0),
            ], p=0.3),

            # Final normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply generative expert augmentations"""
        # Apply compression simulation with higher probability
        if random.random() < self.config.jpeg_compression_prob * 1.5:
            image = self.compression_sim.apply_adaptive_compression(image)

        # Apply standard augmentations
        augmented = self.generative_pipeline(image=image)
        return augmented['image']


class AdaptiveAugmentationScheduler:
    """Adaptive augmentation strength based on training progress"""

    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.base_config = config

    def update_strength(self, epoch: int, total_epochs: int, current_loss: float) -> AugmentationConfig:
        """Update augmentation strength based on training progress"""
        # Calculate training progress
        progress = epoch / total_epochs

        # Adaptive strength based on progress
        if progress < 0.3:  # Early training - mild augmentations
            strength_factor = 0.7
        elif progress < 0.7:  # Mid training - full augmentations
            strength_factor = 1.0
        else:  # Late training - reduced augmentations
            strength_factor = 0.8

        # Adaptive strength based on loss
        if current_loss > 0.7:  # High loss - more augmentations
            strength_factor *= 1.2
        elif current_loss < 0.3:  # Low loss - less augmentations
            strength_factor *= 0.8

        # Create updated config
        updated_config = AugmentationConfig()
        updated_config.__dict__.update(self.base_config.__dict__)

        # Apply strength factor
        updated_config.rotation_degrees *= strength_factor
        updated_config.brightness_range *= strength_factor
        updated_config.contrast_range *= strength_factor
        updated_config.gaussian_noise_std *= strength_factor

        return updated_config


class MultiResolutionAugmentation:
    """Augmentations that work across multiple resolutions"""

    def __init__(self, config: AugmentationConfig, target_sizes: List[int] = [224, 256, 288, 320]):
        self.config = config
        self.target_sizes = target_sizes

    def get_augmentation_for_size(self, target_size: int, expert_type: str = "spatial") -> Callable:
        """Get augmentation pipeline for specific resolution"""
        # Adjust augmentation parameters based on resolution
        scale_factor = target_size / 256.0  # Base resolution

        adjusted_config = AugmentationConfig()
        adjusted_config.__dict__.update(self.config.__dict__)

        # Scale-aware adjustments
        adjusted_config.motion_blur_limit = max(1, int(self.config.motion_blur_limit * scale_factor))
        adjusted_config.rotation_degrees = self.config.rotation_degrees * (1.0 + (scale_factor - 1.0) * 0.5)

        # Create appropriate augmentation
        if expert_type == "spatial":
            return SpatialExpertAugmentation(adjusted_config)
        else:
            return GenerativeExpertAugmentation(adjusted_config)


class AugmentationFactory:
    """Factory for creating appropriate augmentations"""

    @staticmethod
    def create_augmentation(
        expert_type: str,
        config_path: Optional[str] = None,
        **kwargs
    ) -> Union[SpatialExpertAugmentation, GenerativeExpertAugmentation]:
        """Create augmentation pipeline for specified expert type"""

        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = AugmentationConfig(**{**config_dict, **kwargs})
        else:
            config = AugmentationConfig(**kwargs)

        # Create appropriate augmentation
        if expert_type.lower() == "spatial":
            config.spatial_expert_mode = True
            config.generative_expert_mode = False
            return SpatialExpertAugmentation(config)
        elif expert_type.lower() == "generative":
            config.spatial_expert_mode = False
            config.generative_expert_mode = True
            return GenerativeExpertAugmentation(config)
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

    @staticmethod
    def create_multi_resolution_augmentation(
        expert_type: str,
        target_sizes: List[int],
        config_path: Optional[str] = None,
        **kwargs
    ) -> MultiResolutionAugmentation:
        """Create multi-resolution augmentation pipeline"""

        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = AugmentationConfig(**{**config_dict, **kwargs})
        else:
            config = AugmentationConfig(**kwargs)

        return MultiResolutionAugmentation(config, target_sizes)


def test_augmentations():
    """Test function for augmentation pipelines"""
    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test spatial expert augmentation
    spatial_aug = AugmentationFactory.create_augmentation("spatial")
    spatial_result = spatial_aug(test_image)
    print(f"Spatial augmentation output shape: {spatial_result.shape}")

    # Test generative expert augmentation
    generative_aug = AugmentationFactory.create_augmentation("generative")
    generative_result = generative_aug(test_image)
    print(f"Generative augmentation output shape: {generative_result.shape}")

    # Test multi-resolution
    multi_res_aug = AugmentationFactory.create_multi_resolution_augmentation(
        "spatial", [224, 256, 288, 320]
    )

    for size in [224, 256, 288, 320]:
        aug_pipeline = multi_res_aug.get_augmentation_for_size(size, "spatial")
        result = aug_pipeline(test_image)
        print(f"Multi-resolution augmentation for {size}x{size}: {result.shape}")


if __name__ == "__main__":
    test_augmentations()