"""
Enhanced GenConViT Implementation
Generative-Contrastive Vision Transformer with Advanced Feature Fusion

This module implements an enhanced version of GenConViT (Generative-Contrastive Vision Transformer)
for generative structure analysis in deepfake detection. Features include:
- Multi-scale feature fusion with cross-attention mechanisms
- Dual-variant training (classification + reconstruction)
- Contrastive learning for authentic/synthetic structure discrimination
- Adaptive reconstruction quality assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

from .unified_feature_extractor import BaseExpert, ExpertOutput, ExpertType


class FusionStrategy(Enum):
    CROSS_ATTENTION = "cross_attention"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    RESIDUAL_DENSE = "residual_dense"


class ReconstructionMode(Enum):
    FULL_IMAGE = "full_image"
    PATCH_BASED = "patch_based"
    SELECTIVE_REGIONS = "selective_regions"


@dataclass
class ContrastiveLearningConfig:
    temperature: float = 0.1
    negative_sampling_ratio: float = 2.0
    hard_negative_mining: bool = True
    momentum_update: float = 0.999
    queue_size: int = 65536


@dataclass
class FeatureFusionConfig:
    strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    num_scales: int = 4
    fusion_dim: int = 256
    attention_heads: int = 8
    dropout_rate: float = 0.1
    temperature_scaling: bool = True


@dataclass
class DualVariantConfig:
    classification_weight: float = 0.6
    reconstruction_weight: float = 0.4
    contrastive_weight: float = 0.3
    perceptual_weight: float = 0.2
    reconstruction_mode: ReconstructionMode = ReconstructionMode.PATCH_BASED
    adaptive_weighting: bool = True


@dataclass
class GenConViTConfig:
    backbone_type: str = "convnext_tiny"
    input_resolution: int = 256
    patch_size: int = 16
    embed_dim: int = 384
    num_transformer_layers: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0

    feature_fusion: FeatureFusionConfig = None
    dual_variant: DualVariantConfig = None
    contrastive: ContrastiveLearningConfig = None

    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True

    def __post_init__(self):
        if self.feature_fusion is None:
            self.feature_fusion = FeatureFusionConfig()
        if self.dual_variant is None:
            self.dual_variant = DualVariantConfig()
        if self.contrastive is None:
            self.contrastive = ContrastiveLearningConfig()


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction with ConvNeXt backbone
    """
    def __init__(self, config: GenConViTConfig):
        super().__init__()
        self.config = config

        # ConvNeXt backbone for multi-scale features
        try:
            import timm
            self.backbone = timm.create_model(
                config.backbone_type,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3, 4]  # Multiple scales
            )
        except ImportError:
            # Fallback implementation
            self.backbone = self._create_fallback_backbone()

        # Feature projection layers
        self.feature_dims = self._get_feature_dimensions()
        self.projectors = nn.ModuleList([
            nn.Conv2d(dim, config.feature_fusion.fusion_dim, 1)
            for dim in self.feature_dims
        ])

        # Adaptive pooling for consistent spatial dimensions
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((config.input_resolution // (4 * (2**i)),
                                config.input_resolution // (4 * (2**i))))
            for i in range(len(self.feature_dims))
        ])

    def _get_feature_dimensions(self) -> List[int]:
        """Get feature dimensions from backbone"""
        if hasattr(self.backbone, 'feature_info'):
            return [info['num_chs'] for info in self.backbone.feature_info]
        else:
            # Default ConvNeXt tiny dimensions
            return [96, 192, 384, 768]

    def _create_fallback_backbone(self):
        """Fallback ConvNeXt-style backbone"""
        layers = []
        in_channels = 3
        dims = [96, 192, 384, 768]

        for i, dim in enumerate(dims):
            layers.append(nn.Conv2d(in_channels, dim, 4, 4 if i == 0 else 2, 0 if i == 0 else 1))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.GELU())
            in_channels = dim

        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features"""
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            # Fallback forward
            features = []
            for i in range(0, len(self.backbone), 3):
                x = self.backbone[i:i+3](x)
                features.append(x)

        # Project and normalize features
        projected_features = []
        for feat, proj, pool in zip(features, self.projectors, self.adaptive_pools):
            feat_proj = proj(feat)
            feat_pooled = pool(feat_proj)
            projected_features.append(feat_pooled)

        return projected_features


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based feature fusion
    """
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_dim = config.fusion_dim
        self.num_heads = config.attention_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=self.num_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )

        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, config.num_scales, self.fusion_dim))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim)
        )

        # Temperature scaling
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.ones(1))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features using cross-attention
        """
        # Flatten spatial dimensions and prepare for attention
        B = features[0].size(0)
        flattened_features = []

        for feat in features:
            # feat: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = feat.size()
            feat_flat = feat.view(B, C, H * W).transpose(1, 2)
            # Global average pooling to get scale representation
            feat_global = feat_flat.mean(dim=1, keepdim=True)  # [B, 1, C]
            flattened_features.append(feat_global)

        # Stack features: [B, num_scales, C]
        features_stack = torch.cat(flattened_features, dim=1)

        # Add position encoding
        features_stack = features_stack + self.pos_encoding

        # Self-attention across scales
        attended_features, attention_weights = self.attention(
            features_stack, features_stack, features_stack
        )

        # Apply temperature scaling
        attended_features = attended_features / self.temperature

        # Global pooling and projection
        fused_feature = attended_features.mean(dim=1)  # [B, C]
        output = self.output_proj(fused_feature)

        return output, attention_weights


class HierarchicalFusion(nn.Module):
    """
    Hierarchical feature fusion from coarse to fine
    """
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_dim = config.fusion_dim

        # Hierarchical fusion layers
        self.fusion_layers = nn.ModuleList()
        for i in range(config.num_scales - 1):
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                nn.GELU(),
                nn.Dropout(config.dropout_rate)
            ))

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 2),
            nn.GELU(),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Hierarchically fuse features from coarse to fine
        """
        # Global average pooling for each scale
        pooled_features = []
        for feat in features:
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            pooled_features.append(pooled)

        # Hierarchical fusion (coarse to fine)
        fused = pooled_features[0]  # Start with coarsest scale

        for i, (feat, fusion_layer) in enumerate(zip(pooled_features[1:], self.fusion_layers)):
            # Concatenate and fuse
            combined = torch.cat([fused, feat], dim=1)
            fused = fusion_layer(combined)

        output = self.final_proj(fused)
        return output


class DualVariantHead(nn.Module):
    """
    Dual-variant head for classification and reconstruction
    """
    def __init__(self, config: GenConViTConfig):
        super().__init__()
        self.config = config
        fusion_dim = config.feature_fusion.fusion_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 1)  # Binary classification
        )

        # Reconstruction head
        self.reconstruction_head = self._build_reconstruction_head(fusion_dim)

        # Adaptive weight network
        if config.dual_variant.adaptive_weighting:
            self.weight_network = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 4),
                nn.GELU(),
                nn.Linear(fusion_dim // 4, 2),  # [cls_weight, recon_weight]
                nn.Softmax(dim=1)
            )

        # Reconstruction quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.GELU(),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid()
        )

    def _build_reconstruction_head(self, fusion_dim: int) -> nn.Module:
        """Build reconstruction head based on mode"""
        mode = self.config.dual_variant.reconstruction_mode

        if mode == ReconstructionMode.FULL_IMAGE:
            # Full image reconstruction
            return nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 4),
                nn.GELU(),
                nn.Linear(fusion_dim * 4, self.config.input_resolution * self.config.input_resolution * 3),
                nn.Tanh()
            )
        elif mode == ReconstructionMode.PATCH_BASED:
            # Patch-based reconstruction
            patch_size = self.config.patch_size
            return nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Linear(fusion_dim * 2, patch_size * patch_size * 3),
                nn.Tanh()
            )
        else:  # SELECTIVE_REGIONS
            # Selective region reconstruction
            return nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Linear(fusion_dim * 2, 64 * 64 * 3),  # 64x64 regions
                nn.Tanh()
            )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for dual-variant prediction
        """
        # Classification prediction
        cls_logits = self.classifier(features)

        # Reconstruction prediction
        recon_output = self.reconstruction_head(features)

        # Quality prediction
        quality_score = self.quality_predictor(features)

        # Adaptive weighting
        outputs = {
            'classification': cls_logits,
            'reconstruction': recon_output,
            'quality_score': quality_score
        }

        if hasattr(self, 'weight_network'):
            weights = self.weight_network(features)
            outputs['adaptive_weights'] = weights

        return outputs


class ContrastiveLearningModule(nn.Module):
    """
    Contrastive learning for authentic/synthetic discrimination
    """
    def __init__(self, config: ContrastiveLearningConfig, feature_dim: int):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )

        # Momentum encoder
        self.momentum_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )

        # Initialize momentum encoder with projection head parameters
        for param_q, param_k in zip(self.projection_head.parameters(),
                                   self.momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Feature queue for negative sampling
        self.register_buffer("queue", torch.randn(feature_dim // 2, config.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the momentum encoder"""
        for param_q, param_k in zip(self.projection_head.parameters(),
                                   self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.config.momentum_update + \
                          param_q.data * (1. - self.config.momentum_update)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the feature queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.config.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, features: torch.Tensor,
                momentum_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Query features
        q = self.projection_head(features)
        q = F.normalize(q, dim=1)

        # Key features (momentum encoder)
        with torch.no_grad():
            self._momentum_update()

            if momentum_features is not None:
                momentum_features = F.normalize(momentum_features, dim=1)
                k = self.momentum_encoder(momentum_features)
                k = F.normalize(k, dim=1)
            else:
                k = self.momentum_encoder(features)
                k = F.normalize(k, dim=1)

        # Compute logits
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.config.temperature

        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Update queue
        self._dequeue_and_enqueue(k)

        return {
            'logits': logits,
            'labels': labels,
            'queries': q,
            'keys': k
        }


class EnhancedGenConViT(BaseExpert):
    """
    Enhanced GenConViT with advanced feature fusion and dual-variant mechanisms
    """
    def __init__(self, config: GenConViTConfig):
        super().__init__(expert_type=ExpertType.GENERATIVE)
        self.config = config

        # Multi-scale feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(config)

        # Feature fusion module
        if config.feature_fusion.strategy == FusionStrategy.CROSS_ATTENTION:
            self.feature_fusion = CrossAttentionFusion(config.feature_fusion)
        elif config.feature_fusion.strategy == FusionStrategy.HIERARCHICAL:
            self.feature_fusion = HierarchicalFusion(config.feature_fusion)
        else:
            raise ValueError(f"Unsupported fusion strategy: {config.feature_fusion.strategy}")

        # Dual-variant head
        self.dual_head = DualVariantHead(config)

        # Contrastive learning module
        self.contrastive_module = ContrastiveLearningModule(
            config.contrastive,
            config.feature_fusion.fusion_dim
        )

        # Loss functions
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.contrastive_loss = nn.CrossEntropyLoss()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor,
                targets: Optional[Dict[str, torch.Tensor]] = None) -> ExpertOutput:
        """
        Forward pass with dual-variant output
        """
        # Extract multi-scale features
        multi_scale_features = self.feature_extractor(x)

        # Feature fusion
        if isinstance(self.feature_fusion, CrossAttentionFusion):
            fused_features, attention_weights = self.feature_fusion(multi_scale_features)
        else:
            fused_features = self.feature_fusion(multi_scale_features)
            attention_weights = None

        # Dual-variant predictions
        dual_outputs = self.dual_head(fused_features)

        # Contrastive learning (during training)
        contrastive_outputs = None
        if self.training and targets is not None:
            contrastive_outputs = self.contrastive_module(fused_features)

        # Compute losses if targets provided
        losses = {}
        if targets is not None:
            losses = self._compute_losses(dual_outputs, contrastive_outputs, targets)

        # Prepare expert output
        features = {
            'fused_features': fused_features,
            'multi_scale_features': multi_scale_features,
            'attention_weights': attention_weights
        }

        predictions = {
            'classification': torch.sigmoid(dual_outputs['classification']),
            'reconstruction': dual_outputs['reconstruction'],
            'quality_score': dual_outputs['quality_score']
        }

        if 'adaptive_weights' in dual_outputs:
            predictions['adaptive_weights'] = dual_outputs['adaptive_weights']

        confidence = predictions['quality_score'].mean().item()

        return ExpertOutput(
            predictions=predictions,
            features=features,
            confidence=confidence,
            losses=losses
        )

    def _compute_losses(self, dual_outputs: Dict[str, torch.Tensor],
                       contrastive_outputs: Optional[Dict[str, torch.Tensor]],
                       targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for dual-variant training
        """
        losses = {}

        # Classification loss
        if 'labels' in targets:
            cls_loss = self.classification_loss(
                dual_outputs['classification'].squeeze(),
                targets['labels'].float()
            )
            losses['classification'] = cls_loss

        # Reconstruction loss
        if 'reconstruction_targets' in targets:
            recon_loss = self.reconstruction_loss(
                dual_outputs['reconstruction'],
                targets['reconstruction_targets']
            )
            losses['reconstruction'] = recon_loss

        # Contrastive loss
        if contrastive_outputs is not None:
            contrastive_loss = self.contrastive_loss(
                contrastive_outputs['logits'],
                contrastive_outputs['labels']
            )
            losses['contrastive'] = contrastive_loss

        # Adaptive weighting
        config = self.config.dual_variant
        if config.adaptive_weighting and 'adaptive_weights' in dual_outputs:
            weights = dual_outputs['adaptive_weights']
            cls_weight = weights[:, 0].mean()
            recon_weight = weights[:, 1].mean()
        else:
            cls_weight = config.classification_weight
            recon_weight = config.reconstruction_weight

        # Total loss
        total_loss = 0
        if 'classification' in losses:
            total_loss += cls_weight * losses['classification']
        if 'reconstruction' in losses:
            total_loss += recon_weight * losses['reconstruction']
        if 'contrastive' in losses:
            total_loss += config.contrastive_weight * losses['contrastive']

        losses['total'] = total_loss

        return losses

    def get_attention_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention maps for visualization
        """
        with torch.no_grad():
            multi_scale_features = self.feature_extractor(x)
            if isinstance(self.feature_fusion, CrossAttentionFusion):
                _, attention_weights = self.feature_fusion(multi_scale_features)
                return attention_weights
        return None


class GenConViTLoss(nn.Module):
    """
    Combined loss function for GenConViT training
    """
    def __init__(self, config: DualVariantConfig):
        super().__init__()
        self.config = config

        # Individual loss functions
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.perceptual_loss = self._create_perceptual_loss()
        self.contrastive_loss = nn.CrossEntropyLoss()

    def _create_perceptual_loss(self):
        """Create perceptual loss using VGG features"""
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16]
            for param in vgg.parameters():
                param.requires_grad = False
            return vgg
        except:
            return None

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        """
        losses = {}
        total_loss = 0

        # Classification loss
        if 'classification' in outputs and 'labels' in targets:
            cls_loss = self.classification_loss(
                outputs['classification'].squeeze(),
                targets['labels'].float()
            )
            losses['classification'] = cls_loss
            total_loss += self.config.classification_weight * cls_loss

        # Reconstruction loss
        if 'reconstruction' in outputs and 'reconstruction_targets' in targets:
            recon_loss = self.reconstruction_loss(
                outputs['reconstruction'],
                targets['reconstruction_targets']
            )
            losses['reconstruction'] = recon_loss
            total_loss += self.config.reconstruction_weight * recon_loss

            # Perceptual loss
            if self.perceptual_loss is not None:
                recon_features = self.perceptual_loss(outputs['reconstruction'])
                target_features = self.perceptual_loss(targets['reconstruction_targets'])
                perceptual_loss = F.mse_loss(recon_features, target_features)
                losses['perceptual'] = perceptual_loss
                total_loss += self.config.perceptual_weight * perceptual_loss

        # Contrastive loss
        if 'contrastive_logits' in outputs and 'contrastive_labels' in targets:
            contrastive_loss = self.contrastive_loss(
                outputs['contrastive_logits'],
                targets['contrastive_labels']
            )
            losses['contrastive'] = contrastive_loss
            total_loss += self.config.contrastive_weight * contrastive_loss

        losses['total'] = total_loss
        return losses


def create_enhanced_genconvit(
    input_resolution: int = 256,
    fusion_strategy: str = "cross_attention",
    reconstruction_mode: str = "patch_based"
) -> EnhancedGenConViT:
    """
    Factory function to create enhanced GenConViT model
    """
    feature_fusion_config = FeatureFusionConfig(
        strategy=FusionStrategy(fusion_strategy),
        num_scales=4,
        fusion_dim=256,
        attention_heads=8
    )

    dual_variant_config = DualVariantConfig(
        reconstruction_mode=ReconstructionMode(reconstruction_mode),
        adaptive_weighting=True
    )

    config = GenConViTConfig(
        input_resolution=input_resolution,
        feature_fusion=feature_fusion_config,
        dual_variant=dual_variant_config
    )

    return EnhancedGenConViT(config)