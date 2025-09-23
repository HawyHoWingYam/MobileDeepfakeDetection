# AWARE-NET Configuration Files

## 📁 File Structure

### Core Configuration Files
- `datasets.json` - All dataset configurations (original + anonymized)
- `training.json` - All training configurations 
- `README.md` - This usage guide

## 🚀 Quick Start

### 1. Train with Original Dataset (Imbalanced, Path Leakage)
```bash
python src/stage_00/train_baseline.py --dataset original_imbalanced --training original_imbalanced
```

### 2. Train with Clean Balanced Dataset (Recommended)
```bash
python src/stage_00/train_baseline.py --dataset balanced_clean --training balanced_clean
```

### 3. Quick 3-Epoch Test
```bash
python src/stage_00/train_baseline.py --dataset original_imbalanced --training quick_test
```

## 📊 Dataset Options

### `celebdf_v2_original`
- **407,712 samples** (59,557 real + 348,155 fake)
- **5.85:1 imbalance** (natural CelebDF-v2 distribution)  
- **Path leakage present** (will show unrealistic performance)
- **Subject-level clean splits** (zero overlap)

### `celebdf_v2_balanced_anonymized`
- **119,114 samples** (59,557 real + 59,557 fake)
- **Perfect 50/50 balance**
- **Zero path leakage** (anonymized file names)
- **Subject-level clean splits** (zero overlap)

## 🎯 Training Configurations

### `original_imbalanced`
- For testing with original dataset structure
- Expects unrealistic performance due to path leakage
- pos_weight = 0.190 for class imbalance

### `balanced_clean` 
- For realistic academic training
- Expected AUC: 0.75-0.85, F1: 0.70-0.82
- pos_weight = 1.0 (balanced data)

### `quick_test`
- 3 epochs only for validation
- Fast testing of setup

## 📝 Expected Performance

| Dataset | Config | Expected AUC | Expected F1 | Notes |
|---------|--------|--------------|-------------|-------|
| Original | original_imbalanced | 0.95-0.999 | 0.90-0.99 | Path leakage |
| Balanced | balanced_clean | 0.75-0.85 | 0.70-0.82 | Realistic |
| Any | quick_test | Variable | Variable | Testing only |

## 🔧 Configuration Structure

Both `datasets.json` and `training.json` use nested structures:

```json
{
  "datasets": {
    "dataset_name": { ... },
    "another_dataset": { ... }
  },
  "default_dataset": "dataset_name"
}
```
# Model Storage Directory

This directory is intended for storing trained model weights and checkpoints.

## Structure

```
models/
├── stage_00/
│   ├── baseline_efficientnet_b3.pth
│   └── checkpoints/
├── stage_01/
│   └── supcon_filter.pth
└── final/
    └── aware_net_final.pth
```

## Usage

- **Training**: Models are automatically saved here during training
- **Inference**: Load models from this directory for evaluation
- **Docker**: Contents copied to inference container

## Current Status

⚠️ **PLACEHOLDER**: This directory is currently empty. Models will be saved here after training.

## File Formats

- `.pth`: PyTorch model weights
- `.onnx`: ONNX format for deployment
- `.json`: Model metadata and configuration

# Stage 2: Heterogeneous Expert Models Implementation

## Overview

This module implements Stage 2 of the AWARE-NET deepfake detection framework, featuring a sophisticated heterogeneous expert system with spatial and generative specialists, advanced feature fusion mechanisms, and comprehensive diagnostic tools.

## 🏗️ Architecture

### Core Components

1. **Unified Feature Extraction Framework** (`unified_feature_extractor.py`)
   - Base classes and interfaces for expert models
   - Standardized input/output formats
   - Expert type definitions and protocols

2. **Enhanced Spatial Expert** (`enhanced_spatial_expert.py`)
   - EfficientNetV2-B0 based spatial artifact detection
   - Focal loss with adaptive scheduling
   - Graduated learning rate optimization
   - Multi-resolution inference pipeline

3. **Enhanced GenConViT** (`enhanced_genconvit.py`)
   - Generative-Contrastive Vision Transformer
   - Dual-variant training (classification + reconstruction)
   - Multi-scale feature fusion with cross-attention
   - Contrastive learning for structure analysis

4. **Complementarity Analysis** (`complementarity_analysis.py`)
   - Feature-level and decision-level complementarity metrics
   - Adaptive fusion strategies based on expert diversity
   - Gating networks and attention-based fusion

5. **Integration Interface** (`stage3_integration_interface.py`)
   - Seamless Stage 2-3 integration protocols
   - Temporal input preparation and sequence processing
   - Backward compatibility layer

6. **Diagnostic Tools** (`diagnostic_tools.py`)
   - System health monitoring and performance analysis
   - Stage-gate evaluation with comprehensive reporting
   - Model validation and risk assessment

7. **Testing Framework** (`concurrent_testing_framework.py`)
   - Concurrent expert testing and validation
   - Performance benchmarking and system validation
   - Integration testing across components

## 🚀 Quick Start

### Basic Usage

```python
# Import core components
from src.stage_02.enhanced_spatial_expert import create_enhanced_spatial_expert
from src.stage_02.enhanced_genconvit import create_enhanced_genconvit
from src.stage_02.complementarity_analysis import create_fusion_system

# Create experts
spatial_expert = create_enhanced_spatial_expert(
    input_resolution=256,
    num_classes=1,
    use_focal_loss=True
)

generative_expert = create_enhanced_genconvit(
    input_resolution=256,
    fusion_strategy="cross_attention",
    reconstruction_mode="patch_based"
)

# Create fusion system
fusion_system = create_fusion_system(
    hidden_dim=256,
    num_experts=2,
    uncertainty_aware=True
)

# Run inference
import torch
input_tensor = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    spatial_output = spatial_expert(input_tensor)
    generative_output = generative_expert(input_tensor)

    # Fuse expert outputs
    fusion_result = fusion_system.fuse_experts([spatial_output, generative_output])

print(f"Final prediction: {fusion_result['prediction']}")
print(f"Complementarity score: {fusion_result.get('complementarity_score', 'N/A')}")
```

### Advanced Configuration

```python
# Configure spatial expert with focal loss
from src.stage_02.enhanced_spatial_expert import (
    SpatialExpertConfig, FocalLossConfig, GraduatedLRConfig
)

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

# Configure GenConViT with advanced fusion
from src.stage_02.enhanced_genconvit import (
    GenConViTConfig, FeatureFusionConfig, DualVariantConfig
)

fusion_config = FeatureFusionConfig(
    strategy=FusionStrategy.CROSS_ATTENTION,
    num_scales=4,
    fusion_dim=256,
    attention_heads=8
)

dual_config = DualVariantConfig(
    classification_weight=0.6,
    reconstruction_weight=0.4,
    contrastive_weight=0.3,
    adaptive_weighting=True
)

genconvit_config = GenConViTConfig(
    input_resolution=256,
    feature_fusion=fusion_config,
    dual_variant=dual_config
)

generative_expert = EnhancedGenConViT(genconvit_config)
```

## 🔧 Configuration

### Environment Setup

```bash
# Install required dependencies
pip install torch torchvision timm albumentations opencv-python
pip install scikit-learn matplotlib seaborn psutil
pip install lpips  # For perceptual loss
```

### GPU Configuration

The system automatically detects and configures GPU usage:

```python
# Automatic GPU detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Multi-GPU support
if torch.cuda.device_count() > 1:
    spatial_expert = nn.DataParallel(spatial_expert)
    generative_expert = nn.DataParallel(generative_expert)
```

## 📊 Testing and Validation

### Running Tests

```bash
# Run all tests
python src/stage_02/test_suite.py

# Run specific test suites
python -c "
from src.stage_02.test_suite import run_specific_test_suite
run_specific_test_suite('spatial')      # Test spatial expert
run_specific_test_suite('genconvit')    # Test GenConViT
run_specific_test_suite('integration') # Test integration
"
```

### Performance Benchmarking

```python
from src.stage_02.concurrent_testing_framework import run_concurrent_tests
from torch.utils.data import DataLoader

# Create test data
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Run concurrent performance tests
experts = {
    'spatial': spatial_expert,
    'generative': generative_expert
}

test_results = await run_concurrent_tests(
    experts=experts,
    fusion_system=fusion_system,
    dataloader=test_loader
)

print("Performance Report:")
print(f"Average inference time: {test_results['report']['performance_metrics']['avg_inference_time']:.3f}s")
print(f"Average accuracy: {test_results['report']['performance_metrics']['avg_accuracy']:.3f}")
```

### System Health Monitoring

```python
from src.stage_02.diagnostic_tools import SystemHealthMonitor

monitor = SystemHealthMonitor()
health = monitor.get_current_health()

print(f"CPU Usage: {health.cpu_usage:.1f}%")
print(f"Memory Usage: {health.memory_usage:.1f}%")
print(f"GPU Usage: {health.gpu_usage:.1f}%")
```

## 🔍 Diagnostic and Monitoring

### Stage-Gate Evaluation

```python
from src.stage_02.diagnostic_tools import create_diagnostic_system

# Create diagnostic system
evaluator = create_diagnostic_system()

# Run comprehensive evaluation
gate_report = evaluator.evaluate_stage_2(
    spatial_expert=spatial_expert,
    generative_expert=generative_expert,
    test_dataloader=test_loader,
    complementarity_result=complementarity_analysis
)

print(f"Gate Status: {gate_report.gate_status}")
print(f"Overall Score: {gate_report.overall_score:.3f}")
print(f"Technical Status: {gate_report.technical_status}")

# Save report
gate_report.save_report("stage2_evaluation_report.json")
```

### Complementarity Analysis

```python
from src.stage_02.complementarity_analysis import ComplementarityAnalyzer, ComplementarityConfig

config = ComplementarityConfig(
    metrics=[
        ComplementarityMetric.MUTUAL_INFORMATION,
        ComplementarityMetric.DECISION_DIVERSITY,
        ComplementarityMetric.FEATURE_ORTHOGONALITY
    ]
)

analyzer = ComplementarityAnalyzer(config)
result = analyzer.analyze_complementarity(spatial_output, generative_output)

print(f"Complementarity Score: {result.overall_complementarity:.3f}")
print(f"Decision Diversity: {result.decision_diversity:.3f}")
print("Recommendations:", result.recommendations)
```

## 🔗 Stage 3 Integration

### Temporal Integration Setup

```python
from src.stage_02.stage3_integration_interface import create_integration_hub

# Create integration hub
hub = create_integration_hub(
    integration_level="hybrid",
    temporal_mode="frame_sequence",
    max_sequence_length=16
)

# Create Stage 2 wrapper
stage2_wrapper = hub.create_stage2_wrapper(
    spatial_expert=spatial_expert,
    generative_expert=generative_expert,
    fusion_system=fusion_system
)

# Process video sequence
video_input = torch.randn(1, 16, 3, 256, 256)  # [B, T, C, H, W]

# Example temporal expert (implement according to Stage 3)
class MyTemporalExpert:
    def process_temporal_sequence(self, temporal_input, stage2_output):
        # Your temporal processing logic here
        pass

temporal_expert = MyTemporalExpert()
hub.register_temporal_expert("my_temporal", temporal_expert)

# Integrated inference
result = hub.integrated_inference(
    video_input=video_input,
    stage2_wrapper=stage2_wrapper,
    temporal_expert_name="my_temporal"
)
```

## 📈 Performance Optimization

### Memory Optimization

```python
# Enable gradient checkpointing for large models
spatial_expert.enable_gradient_checkpointing()
generative_expert.enable_gradient_checkpointing()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = loss_function(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Inference Optimization

```python
# Enable inference optimization
torch.backends.cudnn.benchmark = True

# Use TorchScript for deployment
spatial_expert_scripted = torch.jit.script(spatial_expert)
generative_expert_scripted = torch.jit.script(generative_expert)

# Batch processing for efficiency
def batch_inference(model, inputs, batch_size=8):
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        with torch.no_grad():
            output = model(batch)
        results.append(output)
    return torch.cat(results)
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or enable gradient checkpointing
   # Use gradient accumulation for effective larger batches
   ```

2. **Import Errors**
   ```python
   # Ensure all dependencies are installed
   # Check Python path includes src directory
   import sys
   sys.path.append('path/to/MobileDeepfakeDetection')
   ```

3. **Performance Issues**
   ```python
   # Enable performance monitoring
   from src.stage_02.diagnostic_tools import SystemHealthMonitor
   monitor = SystemHealthMonitor()
   monitor.start_monitoring()
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use diagnostic tools for detailed analysis
from src.stage_02.diagnostic_tools import ModelValidator

validator = ModelValidator()
metrics = validator.validate_model_performance(model, dataloader, device)
print("Debug metrics:", metrics)
```

## 📚 API Reference

### Core Classes

- **`BaseExpert`**: Abstract base class for all expert models
- **`ExpertOutput`**: Standardized output format for expert predictions
- **`UnifiedFeatureExtractor`**: Central feature extraction coordinator

### Expert Models

- **`EnhancedSpatialExpert`**: Spatial artifact detection specialist
- **`EnhancedGenConViT`**: Generative structure analysis specialist

### Fusion Systems

- **`AdaptiveFusionSystem`**: Intelligent expert fusion coordinator
- **`GatingNetwork`**: Learnable expert weighting system
- **`AttentionFusion`**: Attention-based fusion mechanism

### Integration

- **`TemporalIntegrationHub`**: Stage 2-3 integration coordinator
- **`Stage2ExpertWrapper`**: Compatibility wrapper for Stage 2 experts

### Diagnostics

- **`StageGateEvaluator`**: Comprehensive system evaluation
- **`SystemHealthMonitor`**: Real-time system monitoring
- **`ModelValidator`**: Model performance validation

## 🤝 Contributing

### Development Guidelines

1. Follow the established code structure and naming conventions
2. Add comprehensive tests for new functionality
3. Update documentation for API changes
4. Run the full test suite before submitting changes

### Testing New Components

```python
# Create unit tests
class TestNewComponent(unittest.TestCase):
    def setUp(self):
        # Setup test environment
        pass

    def test_functionality(self):
        # Test core functionality
        pass

# Add to test suite
from src.stage_02.test_suite import run_all_tests
run_all_tests()
```

## 📄 License

This implementation is part of the AWARE-NET academic research project. Please refer to the main project license for usage terms.

## 🔗 References

- EfficientNetV2: Smaller Models and Faster Training
- Vision Transformer (ViT) Architecture
- Focal Loss for Dense Object Detection
- Contrastive Learning for Visual Representations
- Multi-Modal Fusion Techniques

---

For more detailed information about specific components, refer to the inline documentation in each module file.