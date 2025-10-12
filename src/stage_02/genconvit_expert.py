"""
AWARE-NET Stage 02: GenConViT Generative Structure Expert

This module implements the GenConViT (Generative-Contrastive Vision Transformer) expert
that combines generative reconstruction with discriminative classification for detecting
GAN/Diffusion-generated content and structural inconsistencies.

Design Principles:
- ConvNeXt-Swin hybrid architecture for multi-scale feature extraction
- Dual-task training: classification + reconstruction
- ED (Encoder-Decoder) and VAE (Variational AutoEncoder) variants
- Reconstruction quality analysis for generative detection
- Joint loss optimization with balanced weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import timm
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math

# Simplified enums and base classes to replace deleted dependencies
from enum import Enum
from dataclasses import dataclass

class ExpertType(Enum):
    GENERATIVE = "generative"

class ResolutionMode(Enum):
    STANDARD = "standard"

@dataclass
class ExpertOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    confidence: torch.Tensor
    features: torch.Tensor
    normalized_features: torch.Tensor
    expert_type: ExpertType
    resolution: ResolutionMode
    reconstruction: torch.Tensor = None
    reconstruction_error: Dict = None
    # For test compatibility
    losses: Dict = None

class GenConViTVariant(Enum):
    """GenConViT architecture variants."""
    ED = "encoder_decoder"           # Standard encoder-decoder
    VAE = "variational_autoencoder"  # Variational autoencoder variant

class ReconstructionMetric(Enum):
    """Reconstruction quality metrics."""
    MSE = "mse"           # Mean Squared Error
    SSIM = "ssim"         # Structural Similarity Index
    LPIPS = "lpips"       # Learned Perceptual Image Patch Similarity
    VGG = "vgg"           # VGG perceptual loss

@dataclass
class GenConViTConfig:
    """Configuration for GenConViT generative expert."""
    # Architecture variant
    variant: GenConViTVariant = GenConViTVariant.ED
    input_resolution: int = 256
    latent_dim: int = 512

    # ConvNeXt backbone configuration
    convnext_model: str = "convnext_small"
    convnext_pretrained: bool = True
    convnext_freeze_epochs: int = 3

    # Swin Transformer configuration
    swin_patch_size: int = 4
    swin_window_size: int = 8
    swin_depths: List[int] = None
    swin_num_heads: List[int] = None

    # Encoder-Decoder configuration
    encoder_depths: List[int] = None
    decoder_depths: List[int] = None
    skip_connections: bool = True

    # VAE-specific configuration (for VAE variant)
    vae_beta: float = 1.0                    # KL divergence weight
    vae_sample_during_inference: bool = False

    # Dual-task training configuration
    classification_weight: float = 1.0       # α in paper
    reconstruction_weight: float = 0.5       # β in paper
    perceptual_weight: float = 0.3           # γ in paper
    kl_weight: float = 0.1                  # δ in paper (VAE only)

    # Training strategy
    staged_training: bool = True             # Pretraining → joint → fine-tuning
    reconstruction_pretraining_epochs: int = 10
    joint_training_epochs: int = 30
    classification_finetuning_epochs: int = 10

    # Quality thresholds
    real_image_ssim_target: float = 0.85
    fake_image_ssim_target: float = 0.70
    reconstruction_quality_threshold: float = 0.75

    def __post_init__(self):
        if self.swin_depths is None:
            self.swin_depths = [2, 2, 6, 2]
        if self.swin_num_heads is None:
            self.swin_num_heads = [3, 6, 12, 24]
        if self.encoder_depths is None:
            self.encoder_depths = [2, 2, 6, 2]
        if self.decoder_depths is None:
            self.decoder_depths = [2, 6, 2, 2]

class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-based feature encoder for local feature extraction."""

    def __init__(self, model_name: str = "convnext_small", pretrained: bool = True):
        super(ConvNeXtEncoder, self).__init__()

        # Load ConvNeXt backbone without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''  # Remove global pooling
        )

        # Get feature dimensions at different scales
        self.feature_dims = self._get_feature_dimensions()

    def _get_feature_dimensions(self):
        """Get feature dimensions at different scales."""
        with torch.no_grad():
            x = torch.randn(1, 3, 256, 256)
            features = self.forward_features(x)
            return [f.shape[1] for f in features]

    def forward_features(self, x):
        """Extract multi-scale features."""
        features = []

        # Forward through ConvNeXt stages
        x = self.backbone.stem(x)
        features.append(x)

        for i, stage in enumerate(self.backbone.stages):
            x = stage(x)
            features.append(x)

        return features

    def forward(self, x):
        """Forward pass returning all feature scales."""
        return self.forward_features(x)

class SwinTransformerGlobal(nn.Module):
    """Swin Transformer for global context modeling."""

    def __init__(self,
                 input_dim: int,
                 depths: List[int] = [2, 2, 6, 2],
                 num_heads: List[int] = [3, 6, 12, 24],
                 window_size: int = 8):
        super(SwinTransformerGlobal, self).__init__()

        self.input_projection = nn.Conv2d(input_dim, num_heads[0] * 32, 1)

        # Create Swin Transformer blocks
        self.transformer_blocks = nn.ModuleList()

        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            dim = heads * 32

            # Window-based multi-head self-attention blocks
            blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=window_size,
                    shift_size=0 if j % 2 == 0 else window_size // 2
                ) for j in range(depth)
            ])

            self.transformer_blocks.append(blocks)

            # Downsample between stages (except last)
            if i < len(depths) - 1:
                self.transformer_blocks.append(
                    nn.Conv2d(dim, num_heads[i+1] * 32, 2, stride=2)
                )

    def forward(self, x):
        """Forward pass through Swin Transformer."""
        x = self.input_projection(x)

        for stage in self.transformer_blocks:
            if isinstance(stage, nn.ModuleList):
                for block in stage:
                    x = block(x)
            else:
                x = stage(x)

        return x

class SwinTransformerBlock(nn.Module):
    """Simplified Swin Transformer block."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int):
        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x):
        """Forward pass with window-based attention."""
        B, C, H, W = x.shape

        # Reshape to (B, H*W, C) for attention
        x_flat = x.flatten(2).transpose(1, 2)

        # Apply attention
        attn_out = self.attn(self.norm1(x_flat), H, W)
        x_flat = x_flat + attn_out

        # Apply MLP
        mlp_out = self.mlp(self.norm2(x_flat))
        x_flat = x_flat + mlp_out

        # Reshape back to (B, C, H, W)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)

        return x

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        """Forward pass with window partitioning."""
        B, N, C = x.shape

        # Generate QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        # Scaled dot-product attention (simplified)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out

class MLP(nn.Module):
    """Multi-layer perceptron for transformer blocks."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class ReconstructionDecoder(nn.Module):
    """Decoder for image reconstruction."""

    def __init__(self,
                 latent_dim: int,
                 output_channels: int = 3,
                 depths: List[int] = [2, 6, 2, 2],
                 skip_connections: bool = True):
        super(ReconstructionDecoder, self).__init__()

        self.skip_connections = skip_connections

        # Progressive upsampling layers
        dims = [latent_dim, 256, 128, 64, 32]

        self.decoder_stages = nn.ModuleList()

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            # Add skip connection dimension if enabled
            if skip_connections and i > 0:
                in_dim *= 2  # Concatenate with skip connection

            stage = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                *[ResidualBlock(out_dim) for _ in range(depths[i])]
            )

            self.decoder_stages.append(stage)

        # Final output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dims[-1], output_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, latent, skip_features=None):
        """Decode latent representation to image."""
        x = latent

        for i, stage in enumerate(self.decoder_stages):
            # Add skip connection if available
            if self.skip_connections and skip_features and i < len(skip_features):
                skip = skip_features[-(i+1)]  # Reverse order
                # Resize skip to match current resolution
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)

            x = stage(x)

        output = self.output_conv(x)
        return output

class ResidualBlock(nn.Module):
    """Residual block for decoder."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + residual)

class VAEBottleneck(nn.Module):
    """VAE bottleneck for variational encoding."""

    def __init__(self, feature_dim: int, latent_dim: int):
        super(VAEBottleneck, self).__init__()

        self.mu_head = nn.Linear(feature_dim, latent_dim)
        self.logvar_head = nn.Linear(feature_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, features):
        """Encode features to latent distribution."""
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)

        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

class GenConViTExpert(nn.Module):
    """
    GenConViT expert for generative structure analysis.

    Combines ConvNeXt local feature extraction with Swin Transformer global modeling
    and reconstruction-based generative detection.
    """

    def __init__(self, config: GenConViTConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self._build_model()

        # Reconstruction metrics tracking
        self.reconstruction_metrics = {}

    def _build_model(self):
        """Build the GenConViT architecture."""
        # ConvNeXt encoder for local features
        self.convnext_encoder = ConvNeXtEncoder(
            self.config.convnext_model,
            self.config.convnext_pretrained
        )

        # Get the last feature dimension from ConvNeXt
        last_feature_dim = self.convnext_encoder.feature_dims[-1]

        # Swin Transformer for global context
        self.swin_global = SwinTransformerGlobal(
            input_dim=last_feature_dim,
            depths=self.config.swin_depths,
            num_heads=self.config.swin_num_heads,
            window_size=self.config.swin_window_size
        )

        # Get global feature dimension
        global_feature_dim = self.config.swin_num_heads[-1] * 32

        # VAE bottleneck for VAE variant
        if self.config.variant == GenConViTVariant.VAE:
            self.vae_bottleneck = VAEBottleneck(global_feature_dim, self.config.latent_dim)
        else:
            # Standard bottleneck for ED variant
            self.bottleneck = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(global_feature_dim, self.config.latent_dim),
                nn.ReLU(inplace=True)
            )

        # Reconstruction decoder
        self.decoder = ReconstructionDecoder(
            latent_dim=self.config.latent_dim,
            output_channels=3,
            depths=self.config.decoder_depths,
            skip_connections=self.config.skip_connections
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  # Binary classification
        )

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load the expert model from checkpoint."""
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            print(f"Loaded GenConViT expert from {checkpoint_path}")

        self.is_loaded = True

    def preprocess_input(self,
                        images: torch.Tensor,
                        target_resolution: ResolutionMode = ResolutionMode.STANDARD) -> torch.Tensor:
        """
        Preprocess input images for GenConViT expert.

        Args:
            images: Input tensor [B, C, H, W]
            target_resolution: Target resolution (GenConViT uses standard 256x256)

        Returns:
            Processed tensor [B, C, 256, 256]
        """
        # GenConViT works with fixed 256x256 resolution for reconstruction consistency
        target_size = 256

        # Resize to target resolution
        if images.shape[-1] != target_size or images.shape[-2] != target_size:
            images = F.interpolate(images, size=(target_size, target_size),
                                 mode='bilinear', align_corners=False)

        # Normalize to [-1, 1] for reconstruction consistency
        if images.max() > 1.0:
            images = images / 255.0
        images = images * 2.0 - 1.0

        return images

    def extract_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract features using ConvNeXt + Swin Transformer."""
        # ConvNeXt local feature extraction
        convnext_features = self.convnext_encoder(images)

        # Use the last feature map for global processing
        last_features = convnext_features[-1]

        # Swin Transformer global context modeling
        global_features = self.swin_global(last_features)

        return global_features, convnext_features[:-1]  # Skip features for decoder

    def encode_to_latent(self, global_features: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode global features to latent representation."""
        if self.config.variant == GenConViTVariant.VAE:
            z, mu, logvar = self.vae_bottleneck(global_features)
            return z, mu, logvar
        else:
            # Standard encoding for ED variant
            latent = self.bottleneck(global_features)
            return latent

    def reconstruct_image(self, latent: torch.Tensor, skip_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Reconstruct image from latent representation."""
        # Reshape latent for decoder input
        if len(latent.shape) == 2:
            # Reshape to spatial dimensions for decoder
            spatial_size = int(math.sqrt(latent.shape[1] // 16))  # Assuming 16 channels
            if spatial_size * spatial_size * 16 == latent.shape[1]:
                latent = latent.view(latent.shape[0], 16, spatial_size, spatial_size)
            else:
                # Fallback: create spatial representation
                latent = latent.unsqueeze(-1).unsqueeze(-1)
                latent = latent.expand(-1, -1, 8, 8)

        reconstructed = self.decoder(latent, skip_features)
        return reconstructed

    def predict(self, images: torch.Tensor, return_features: bool = True) -> ExpertOutput:
        """
        Generate predictions with dual-task GenConViT expert.

        Args:
            images: Preprocessed input tensor [B, C, H, W]
            return_features: Whether to return feature representations

        Returns:
            ExpertOutput with predictions, features, and reconstruction
        """
        batch_size = images.shape[0]
        device = images.device

        # Extract features
        global_features, skip_features = self.extract_features(images)

        # Encode to latent
        if self.config.variant == GenConViTVariant.VAE:
            latent, mu, logvar = self.encode_to_latent(global_features)
        else:
            latent = self.encode_to_latent(global_features)
            mu, logvar = None, None

        # Classification
        logits = self.classifier(latent)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        confidence = torch.abs(probabilities - 0.5) * 2

        # Reconstruction
        reconstructed = self.reconstruct_image(latent, skip_features if self.config.skip_connections else None)

        # Compute reconstruction error
        reconstruction_error = self._compute_reconstruction_error(images, reconstructed)

        # Prepare output
        output = ExpertOutput(
            logits=logits.squeeze(-1),
            probabilities=probabilities.squeeze(-1),
            predictions=predictions.squeeze(-1),
            confidence=confidence.squeeze(-1),
            features=latent if return_features else None,
            normalized_features=F.normalize(latent, p=2, dim=1) if return_features else None,
            expert_type=ExpertType.GENERATIVE,
            resolution=ResolutionMode.STANDARD,
            reconstruction=reconstructed,
            reconstruction_error=reconstruction_error
        )

        return output

    def _compute_reconstruction_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various reconstruction error metrics."""
        # MSE
        mse = F.mse_loss(reconstructed, original, reduction='none').mean(dim=[1, 2, 3])

        # SSIM (simplified implementation)
        ssim = self._compute_ssim(original, reconstructed)

        # L1 loss
        l1 = F.l1_loss(reconstructed, original, reduction='none').mean(dim=[1, 2, 3])

        return {
            'mse': mse,
            'ssim': ssim,
            'l1': l1,
            'combined': mse + (1 - ssim) + l1  # Combined error score
        }

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Simplified SSIM computation."""
        # Constants for numerical stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Convert to grayscale if needed
        if img1.shape[1] == 3:
            img1_gray = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
            img2_gray = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        else:
            img1_gray = img1.squeeze(1)
            img2_gray = img2.squeeze(1)

        # Compute means
        mu1 = F.avg_pool2d(img1_gray.unsqueeze(1), window_size, stride=1, padding=window_size//2).squeeze(1)
        mu2 = F.avg_pool2d(img2_gray.unsqueeze(1), window_size, stride=1, padding=window_size//2).squeeze(1)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = F.avg_pool2d((img1_gray * img1_gray).unsqueeze(1), window_size, stride=1, padding=window_size//2).squeeze(1) - mu1_sq
        sigma2_sq = F.avg_pool2d((img2_gray * img2_gray).unsqueeze(1), window_size, stride=1, padding=window_size//2).squeeze(1) - mu2_sq
        sigma12 = F.avg_pool2d((img1_gray * img2_gray).unsqueeze(1), window_size, stride=1, padding=window_size//2).squeeze(1) - mu1_mu2

        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        return ssim_map.mean(dim=[1, 2])  # Average over spatial dimensions

    def analyze_reconstruction_quality(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze reconstruction quality for different types of images.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Dictionary with reconstruction quality analysis
        """
        output = self.predict(images, return_features=True)

        # Compute detailed metrics
        reconstruction_metrics = output.reconstruction_error

        # Classify as real/fake based on reconstruction quality
        reconstruction_predictions = []
        for metric_name, values in reconstruction_metrics.items():
            if metric_name == 'ssim':
                # Higher SSIM = better reconstruction = more likely real
                pred = values > self.config.reconstruction_quality_threshold
            else:
                # Lower error = better reconstruction = more likely real
                pred = values < torch.median(values)
            reconstruction_predictions.append(pred.float())

        reconstruction_consensus = torch.stack(reconstruction_predictions).mean(dim=0)

        return {
            'reconstruction_metrics': reconstruction_metrics,
            'reconstruction_predictions': reconstruction_consensus,
            'quality_thresholds': {
                'real_ssim_target': self.config.real_image_ssim_target,
                'fake_ssim_target': self.config.fake_image_ssim_target
            },
            'classification_output': output
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for dual-task learning."""
        config = {
            'loss_weights': {
                'classification': self.config.classification_weight,
                'reconstruction': self.config.reconstruction_weight,
                'perceptual': self.config.perceptual_weight,
                'kl_divergence': self.config.kl_weight if self.config.variant == GenConViTVariant.VAE else 0.0
            },
            'staged_training': {
                'enabled': self.config.staged_training,
                'reconstruction_pretraining_epochs': self.config.reconstruction_pretraining_epochs,
                'joint_training_epochs': self.config.joint_training_epochs,
                'classification_finetuning_epochs': self.config.classification_finetuning_epochs
            },
            'optimization': {
                'convnext_lr_multiplier': 0.1,  # Lower LR for pretrained ConvNeXt
                'transformer_lr_multiplier': 1.0,
                'decoder_lr_multiplier': 1.0,
                'classifier_lr_multiplier': 1.0
            },
            'variant_specific': {
                'is_vae': self.config.variant == GenConViTVariant.VAE,
                'vae_beta': self.config.vae_beta,
                'vae_warmup_epochs': 5 if self.config.variant == GenConViTVariant.VAE else 0
            }
        }

        return config

# Factory functions
def create_genconvit_expert(config: Optional[GenConViTConfig] = None) -> GenConViTExpert:
    """Factory function to create GenConViT expert."""
    if config is None:
        config = GenConViTConfig()

    return GenConViTExpert(config)

def create_genconvit_expert_from_dict(config_dict: Dict[str, Any]) -> GenConViTExpert:
    """Create GenConViT expert from configuration dictionary."""
    # Handle enum conversion
    if 'variant' in config_dict and isinstance(config_dict['variant'], str):
        config_dict['variant'] = GenConViTVariant(config_dict['variant'])

    config = GenConViTConfig(**config_dict)
    return GenConViTExpert(config)

# Export interface
__all__ = [
    'GenConViTVariant', 'ReconstructionMetric', 'GenConViTConfig',
    'ConvNeXtEncoder', 'SwinTransformerGlobal', 'ReconstructionDecoder', 'VAEBottleneck',
    'GenConViTExpert',
    'create_genconvit_expert', 'create_genconvit_expert_from_dict',
    # Aliases for test_suite.py compatibility
    'EnhancedGenConViT', 'create_enhanced_genconvit'
]

# Aliases for test_suite.py compatibility
EnhancedGenConViT = GenConViTExpert
create_enhanced_genconvit = create_genconvit_expert

# Also add missing ExpertType values for compatibility
ExpertType.SPATIAL = "spatial"
ExpertType.TEMPORAL = "temporal"

# Add BaseExpert abstract class if needed
class BaseExpert:
    """Abstract base class for experts"""
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError