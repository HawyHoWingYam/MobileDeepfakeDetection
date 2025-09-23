# AWARE-NET API 參考文檔

## 📚 概述

本文檔提供AWARE-NET框架的詳細API參考，涵蓋Stage 00-02的所有核心組件和接口。

## 🏗️ 核心架構

### 基礎接口

#### BaseExpert
```python
from src.stage_02.unified_feature_extractor import BaseExpert, ExpertOutput

class BaseExpert(nn.Module):
    """所有專家模型的基礎類別"""

    def __init__(self, expert_type: ExpertType, config: Dict[str, Any])

    def forward(self, input_tensor: torch.Tensor) -> ExpertOutput

    def get_features(self, input_tensor: torch.Tensor) -> torch.Tensor

    def calibrate_predictions(self, logits: torch.Tensor) -> torch.Tensor
```

#### ExpertOutput
```python
@dataclass
class ExpertOutput:
    """專家模型標準化輸出格式"""

    predictions: torch.Tensor          # 主要預測 [B, num_classes]
    features: torch.Tensor             # 特徵表示 [B, feature_dim]
    confidence: torch.Tensor           # 預測置信度 [B]
    attention_maps: Optional[torch.Tensor]  # 注意力圖 [B, H, W]
    auxiliary_outputs: Dict[str, torch.Tensor]  # 輔助輸出
    metadata: Dict[str, Any]           # 元數據
```

## 🎯 Stage 01: SupCon快速過濾器

### SupCon損失函數

#### SupConLoss
```python
from src.stage_01.supcon_loss import SupConLoss

class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = 'all',
        base_temperature: float = 0.07
    )

    def forward(
        self,
        features: torch.Tensor,    # [B, feature_dim]
        labels: torch.Tensor = None,  # [B]
        mask: torch.Tensor = None     # [B, B]
    ) -> torch.Tensor
```

**使用示例:**
```python
# 監督對比學習
criterion = SupConLoss(temperature=0.1)
loss = criterion(features, labels)

# 自監督對比學習
loss = criterion(features)
```

### MobileNetV4模型

#### MobileNetV4SupCon
```python
from src.stage_01.mobilenetv4_model import MobileNetV4SupCon

class MobileNetV4SupCon(nn.Module):
    def __init__(
        self,
        variant: str = 'medium',      # 'small', 'medium', 'large'
        projection_dim: int = 128,
        num_classes: int = 1,
        pretrained: bool = True
    )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]

    def get_features(self, x: torch.Tensor) -> torch.Tensor

    def get_projection(self, x: torch.Tensor) -> torch.Tensor
```

**使用示例:**
```python
model = MobileNetV4SupCon(
    variant='medium',
    projection_dim=128,
    num_classes=1
)

output = model(input_tensor)
# output: {
#   'logits': torch.Tensor,      # 分類輸出
#   'features': torch.Tensor,    # 骨幹特徵
#   'projections': torch.Tensor  # 投影特徵
# }
```

### 平衡採樣器

#### BalancedContrastiveSampler
```python
from src.stage_01.balanced_sampler import BalancedContrastiveSampler

class BalancedContrastiveSampler:
    def __init__(
        self,
        dataset_labels: List[int],
        batch_size: int = 32,
        n_positive: int = 2,
        n_negative: int = 2
    )

    def __iter__(self) -> Iterator[List[int]]

    def get_balanced_batch(self) -> List[int]
```

**使用示例:**
```python
sampler = BalancedContrastiveSampler(
    dataset_labels=train_labels,
    batch_size=32,
    n_positive=2,
    n_negative=2
)

dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4
)
```

### 溫度校準

#### TemperatureScaling
```python
from src.stage_01.temperature_scaling import TemperatureScaling

class TemperatureScaling(nn.Module):
    def __init__(self, init_temperature: float = 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor

    def calibrate_model(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device
    ) -> float  # 返回最佳溫度
```

**使用示例:**
```python
temp_scaling = TemperatureScaling()
optimal_temp = temp_scaling.calibrate_model(model, val_loader, device)

# 應用校準
calibrated_probs = temp_scaling(logits)
```

## 🎯 Stage 02: 異構專家模型

### 空間專家

#### EnhancedSpatialExpert
```python
from src.stage_02.enhanced_spatial_expert import (
    EnhancedSpatialExpert, SpatialExpertConfig, FocalLossConfig
)

class EnhancedSpatialExpert(BaseExpert):
    def __init__(
        self,
        config: SpatialExpertConfig,
        focal_config: FocalLossConfig,
        lr_config: GraduatedLRConfig
    )

    def forward(self, x: torch.Tensor) -> ExpertOutput

    def get_grad_cam(
        self,
        input_tensor: torch.Tensor,
        target_layer: str = 'features'
    ) -> torch.Tensor

    def multi_resolution_inference(
        self,
        input_tensor: torch.Tensor,
        resolutions: List[int] = [224, 256, 288, 320]
    ) -> Dict[int, ExpertOutput]
```

**配置示例:**
```python
spatial_config = SpatialExpertConfig(
    backbone="efficientnetv2_rw_s",
    input_resolution=256,
    num_classes=1,
    dropout_rate=0.2
)

focal_config = FocalLossConfig(
    alpha=0.25,
    gamma=2.0,
    label_smoothing=0.1
)

lr_config = GraduatedLRConfig(
    backbone_lr=1e-4,
    head_lr=1e-3,
    warmup_epochs=5
)

spatial_expert = EnhancedSpatialExpert(spatial_config, focal_config, lr_config)
```

### 生成專家

#### EnhancedGenConViT
```python
from src.stage_02.enhanced_genconvit import (
    EnhancedGenConViT, GenConViTConfig, FeatureFusionConfig
)

class EnhancedGenConViT(BaseExpert):
    def __init__(self, config: GenConViTConfig)

    def forward(self, x: torch.Tensor) -> ExpertOutput

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor

    def get_reconstruction_quality(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, float]  # SSIM, LPIPS等指標

    def dual_variant_training(
        self,
        x: torch.Tensor,
        classification_weight: float = 0.6,
        reconstruction_weight: float = 0.4
    ) -> Dict[str, torch.Tensor]
```

**配置示例:**
```python
fusion_config = FeatureFusionConfig(
    strategy=FusionStrategy.CROSS_ATTENTION,
    num_scales=4,
    fusion_dim=256,
    attention_heads=8
)

genconvit_config = GenConViTConfig(
    input_resolution=256,
    feature_fusion=fusion_config,
    dual_variant=dual_config
)

generative_expert = EnhancedGenConViT(genconvit_config)
```

### 互補性分析

#### ComplementarityAnalyzer
```python
from src.stage_02.complementarity_analysis import (
    ComplementarityAnalyzer, ComplementarityConfig, ComplementarityResult
)

class ComplementarityAnalyzer:
    def __init__(self, config: ComplementarityConfig)

    def analyze_complementarity(
        self,
        expert1_output: ExpertOutput,
        expert2_output: ExpertOutput
    ) -> ComplementarityResult

    def compute_mutual_information(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> float

    def compute_decision_diversity(
        self,
        predictions1: torch.Tensor,
        predictions2: torch.Tensor
    ) -> float
```

**使用示例:**
```python
config = ComplementarityConfig(
    metrics=[
        ComplementarityMetric.MUTUAL_INFORMATION,
        ComplementarityMetric.DECISION_DIVERSITY,
        ComplementarityMetric.FEATURE_ORTHOGONALITY
    ]
)

analyzer = ComplementarityAnalyzer(config)
result = analyzer.analyze_complementarity(spatial_output, generative_output)

print(f"互補性分數: {result.overall_complementarity:.3f}")
print(f"建議: {result.recommendations}")
```

### 融合系統

#### AdaptiveFusionSystem
```python
from src.stage_02.complementarity_analysis import (
    AdaptiveFusionSystem, FusionStrategy
)

class AdaptiveFusionSystem:
    def __init__(
        self,
        hidden_dim: int = 256,
        num_experts: int = 2,
        uncertainty_aware: bool = True
    )

    def fuse_experts(
        self,
        expert_outputs: List[ExpertOutput]
    ) -> Dict[str, torch.Tensor]

    def select_fusion_strategy(
        self,
        complementarity_score: float
    ) -> FusionStrategy

    def uncertainty_weighted_fusion(
        self,
        predictions: List[torch.Tensor],
        uncertainties: List[torch.Tensor]
    ) -> torch.Tensor
```

**使用示例:**
```python
fusion_system = AdaptiveFusionSystem(
    hidden_dim=256,
    num_experts=2,
    uncertainty_aware=True
)

fusion_result = fusion_system.fuse_experts([spatial_output, generative_output])
# fusion_result: {
#   'prediction': torch.Tensor,
#   'confidence': torch.Tensor,
#   'complementarity_score': float,
#   'fusion_weights': torch.Tensor
# }
```

## 🔧 診斷和監控

### 系統健康監控

#### SystemHealthMonitor
```python
from src.stage_02.diagnostic_tools import SystemHealthMonitor, SystemHealth

class SystemHealthMonitor:
    def __init__(self, update_interval: float = 1.0)

    def get_current_health(self) -> SystemHealth

    def start_monitoring(self) -> None

    def stop_monitoring(self) -> None

    def get_health_history(self) -> List[SystemHealth]
```

**使用示例:**
```python
monitor = SystemHealthMonitor()
health = monitor.get_current_health()

print(f"CPU: {health.cpu_usage:.1f}%")
print(f"Memory: {health.memory_usage:.1f}%")
print(f"GPU: {health.gpu_usage:.1f}%")
```

### Stage-Gate評估

#### StageGateEvaluator
```python
from src.stage_02.diagnostic_tools import (
    StageGateEvaluator, GateReport, GateStatus
)

class StageGateEvaluator:
    def __init__(self, criteria_config: Dict[str, Any])

    def evaluate_stage_2(
        self,
        spatial_expert: EnhancedSpatialExpert,
        generative_expert: EnhancedGenConViT,
        test_dataloader: DataLoader,
        complementarity_result: ComplementarityResult
    ) -> GateReport

    def validate_technical_criteria(self, **kwargs) -> Dict[str, bool]

    def validate_academic_criteria(self, **kwargs) -> Dict[str, bool]

    def validate_system_criteria(self, **kwargs) -> Dict[str, bool]
```

**使用示例:**
```python
evaluator = StageGateEvaluator(criteria_config)

gate_report = evaluator.evaluate_stage_2(
    spatial_expert=spatial_expert,
    generative_expert=generative_expert,
    test_dataloader=test_loader,
    complementarity_result=complementarity_analysis
)

print(f"Gate狀態: {gate_report.gate_status}")
print(f"技術標準: {gate_report.technical_status}")
print(f"學術標準: {gate_report.academic_status}")
```

## 🔗 Stage 3 整合接口

### TemporalIntegrationHub
```python
from src.stage_02.stage3_integration_interface import (
    TemporalIntegrationHub, TemporalInput, IntegratedOutput
)

class TemporalIntegrationHub:
    def __init__(
        self,
        integration_level: str = "hybrid",
        temporal_mode: str = "frame_sequence",
        max_sequence_length: int = 16
    )

    def create_stage2_wrapper(
        self,
        spatial_expert: EnhancedSpatialExpert,
        generative_expert: EnhancedGenConViT,
        fusion_system: AdaptiveFusionSystem
    ) -> Stage2ExpertWrapper

    def register_temporal_expert(
        self,
        name: str,
        expert: Any
    ) -> None

    def integrated_inference(
        self,
        video_input: torch.Tensor,  # [B, T, C, H, W]
        stage2_wrapper: Stage2ExpertWrapper,
        temporal_expert_name: str
    ) -> IntegratedOutput
```

**使用示例:**
```python
hub = TemporalIntegrationHub(
    integration_level="hybrid",
    temporal_mode="frame_sequence",
    max_sequence_length=16
)

stage2_wrapper = hub.create_stage2_wrapper(
    spatial_expert=spatial_expert,
    generative_expert=generative_expert,
    fusion_system=fusion_system
)

# 註冊時序專家
hub.register_temporal_expert("my_temporal", temporal_expert)

# 整合推理
video_input = torch.randn(1, 16, 3, 256, 256)
result = hub.integrated_inference(
    video_input=video_input,
    stage2_wrapper=stage2_wrapper,
    temporal_expert_name="my_temporal"
)
```

## 🧪 測試框架

### 並發測試

#### ConcurrentTestingFramework
```python
from src.stage_02.concurrent_testing_framework import (
    ConcurrentTestingFramework, TestConfig, TestResults
)

async def run_concurrent_tests(
    experts: Dict[str, BaseExpert],
    fusion_system: AdaptiveFusionSystem,
    dataloader: DataLoader,
    config: TestConfig = None
) -> TestResults
```

**使用示例:**
```python
experts = {
    'spatial': spatial_expert,
    'generative': generative_expert
}

test_results = await run_concurrent_tests(
    experts=experts,
    fusion_system=fusion_system,
    dataloader=test_loader
)

print(f"平均推理時間: {test_results['avg_inference_time']:.3f}s")
print(f"平均準確率: {test_results['avg_accuracy']:.3f}")
```

## 📊 數據處理

### 多分辨率數據加載器

#### MultiResolutionDataLoader
```python
from src.stage_02.multi_resolution_dataloader import (
    MultiResolutionDataLoader, ResolutionConfig
)

class MultiResolutionDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        resolutions: List[int] = [224, 256, 288, 320],
        batch_size: int = 32,
        validation_enabled: bool = True
    )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]

    def validate_batch_quality(self, batch: Dict[str, torch.Tensor]) -> bool

    def get_curriculum_schedule(self) -> List[int]
```

### 數據增強

#### ExpertAugmentationFactory
```python
from src.stage_02.data_augmentation import (
    ExpertAugmentationFactory, AugmentationStrategy
)

class ExpertAugmentationFactory:
    @staticmethod
    def create_spatial_augmentation(
        config: Dict[str, Any]
    ) -> AugmentationStrategy

    @staticmethod
    def create_generative_augmentation(
        config: Dict[str, Any]
    ) -> AugmentationStrategy

    @staticmethod
    def create_adaptive_augmentation(
        expert_type: ExpertType,
        difficulty_level: float = 0.5
    ) -> AugmentationStrategy
```

## 🛠️ 工具函數

### 性能分析

#### ProfileStage2Performance
```python
from scripts.profile_stage2_performance import (
    profile_expert_performance,
    profile_fusion_performance,
    generate_performance_report
)

def profile_expert_performance(
    expert: BaseExpert,
    dataloader: DataLoader,
    device: torch.device,
    num_runs: int = 100
) -> Dict[str, float]

def profile_fusion_performance(
    fusion_system: AdaptiveFusionSystem,
    expert_outputs: List[ExpertOutput],
    num_runs: int = 100
) -> Dict[str, float]
```

### 可視化工具

#### SpatialAnalysisTools
```python
from src.stage_02.spatial_analysis_tools import (
    generate_grad_cam,
    analyze_attention_patterns,
    compute_artifact_correlation
)

def generate_grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str,
    target_class: int = None
) -> np.ndarray

def analyze_attention_patterns(
    attention_maps: torch.Tensor,
    ground_truth_masks: torch.Tensor = None
) -> Dict[str, float]
```

## 📝 配置管理

### 配置類別

所有配置類別都支持JSON序列化和驗證：

```python
from dataclasses import dataclass, asdict
import json

@dataclass
class SpatialExpertConfig:
    backbone: str = "efficientnetv2_rw_s"
    input_resolution: int = 256
    num_classes: int = 1
    dropout_rate: float = 0.2

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'SpatialExpertConfig':
        return cls(**json.loads(json_str))
```

### 配置文件位置

- `configs/spatial_expert_config.json` - 空間專家配置
- `configs/genconvit_expert_config.json` - GenConViT配置
- `configs/stage2_augmentation_unified.json` - 數據增強配置
- `configs/experiment_tracking.json` - 實驗追蹤配置

## 🔍 錯誤處理

### 異常類別

```python
from src.utils.exceptions import (
    ExpertInitializationError,
    FusionError,
    ConfigurationError,
    ModelValidationError
)

class ExpertInitializationError(Exception):
    """專家模型初始化失敗"""
    pass

class FusionError(Exception):
    """專家融合過程錯誤"""
    pass
```

### 錯誤處理最佳實踐

```python
try:
    spatial_expert = create_enhanced_spatial_expert(config)
except ExpertInitializationError as e:
    logger.error(f"空間專家初始化失敗: {e}")
    # 回退到簡單模型
    spatial_expert = create_simple_spatial_expert(fallback_config)

try:
    fusion_result = fusion_system.fuse_experts(expert_outputs)
except FusionError as e:
    logger.warning(f"融合失敗，使用平均策略: {e}")
    # 使用簡單平均作為回退
    fusion_result = simple_average_fusion(expert_outputs)
```

## 📚 類型提示

本框架完全支持類型提示，提升IDE支持和代碼品質：

```python
from typing import List, Dict, Optional, Union, Tuple, Any
import torch
from torch import nn

def process_expert_outputs(
    outputs: List[ExpertOutput],
    fusion_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """處理專家輸出並返回融合結果"""
    pass
```

---

*API版本: v0.3.0*
*最後更新: 2025-09-23*
*支持Stage: 00-02*