# AWARE-NET Tools Directory

Tools目錄按功能分類組織，提供項目所需的各種實用工具。

## 目錄結構

```
tools/
├── data/                    # 數據處理工具
│   ├── generate_manifests.py    - 生成數據集清單文件
│   └── diagnose_path_leakage.py - 診斷數據集路徑洩漏問題
├── setup/                   # 環境設置工具
│   ├── environment_manager.py   - 智能PyTorch安裝器
│   └── setup_environment.py     - 全面環境驗證器
├── performance/             # 性能分析工具
│   ├── profile_stage2_performance.py - Stage 2性能分析
│   └── stage01_threshold_optimizer.py - Stage 01阈值优化
├── validation/              # 驗證工具
│   ├── stage_gate_validator.py      - Stage Gate驗證器
│   └── verify_stage_0_completion.py - Stage 0完成驗證
└── tests/                   # 測試文件
    ├── test_baseline_model.py
    ├── test_dataset.py
    ├── test_metrics.py
    ├── test_stage02_smoke.py - Stage 02煙霧測試
    └── ...
```

## 工具使用指南

### 數據處理工具 (data/)

#### 1. 數據集清單生成器
```bash
python tools/data/generate_manifests.py celebdf
python tools/data/generate_manifests.py faceforensics
python tools/data/generate_manifests.py deeperforensics
```

#### 2. 路徑洩漏診斷器
```bash
python tools/data/diagnose_path_leakage.py
```

### 環境設置工具 (setup/)

#### 1. 智能PyTorch安裝器
```bash
# 檢查安裝計劃 (dry run)
python tools/setup/environment_manager.py --dry-run

# 安裝PyTorch
python tools/setup/environment_manager.py

# 僅驗證現有安裝
python tools/setup/environment_manager.py --verify-only

# 強制CPU版本
python tools/setup/environment_manager.py --force-cpu
```

#### 2. 全面環境驗證器
```bash
python tools/setup/setup_environment.py
```

### 性能分析工具 (performance/)

#### Stage 2性能分析
```bash
python tools/performance/profile_stage2_performance.py
```

#### Stage 01阈值优化
```bash
python tools/performance/stage01_threshold_optimizer.py \
  --checkpoint experiments/full_training_20251014_100257_cf074fe9/checkpoints/best_model.pth \
  --dataset celebdf_v2 \
  --output-dir analysis/threshold_optimization
```

### 驗證工具 (validation/)

#### 1. Stage Gate驗證器
```bash
python tools/validation/stage_gate_validator.py --stage 1
python tools/validation/stage_gate_validator.py --stage 2
```

#### 2. Stage 0完成驗證
```bash
python tools/validation/verify_stage_0_completion.py
```

## 模塊化使用

所有工具也可以作為Python模塊導入：

```python
# 數據工具
from tools.data import generate_manifests, diagnose_path_leakage

# 環境工具
from tools.setup import environment_manager, setup_environment

# 性能工具
from tools.performance import profile_stage2_performance

# 驗證工具
from tools.validation import stage_gate_validator, verify_stage_0_completion
```

## 開發與貢獻

- 每個新工具應放在合適的功能目錄下
- 更新對應的`__init__.py`文件以導出新功能
- 為工具添加適當的命令行接口
- 編寫測試文件到`tests/`目錄

## 測試

運行所有測試：
```bash
pytest tools/tests/
```

運行特定測試：
```bash
pytest tools/tests/test_dataset.py
pytest tools/tests/test_baseline_model.py
pytest tools/tests/test_stage02_smoke.py  # Stage 02煙霧測試
```

# Stage 02 Heterogeneous Expert Tools

## Stage 02 Configuration

Stage 02 uses a dedicated configuration file for heterogeneous expert training:

```bash
# Configuration location
configs/stage02_training.json
```

### Key Configuration Sections

- **spatial_expert**: EfficientNetV2-B0 with focal loss and graduated learning rates
- **genconvit_expert**: Generative-Contrastive Vision Transformer with dual-variant training
- **training**: Optimized for expert models (batch_size=32, epochs=50)
- **expert_fusion**: Adaptive weighted fusion strategies
- **progressive_validation**: Concept validation and resolution comparison

## Stage 02 Smoke Testing

Quick validation tests to ensure Stage 02 components are working before training:

```bash
# Run all Stage 02 smoke tests
python tools/tests/test_stage02_smoke.py

# Or run with pytest
pytest tools/tests/test_stage02_smoke.py -v
```

### Smoke Test Coverage

1. **Model Import Tests**: Verify spatial and GenConViT experts can be imported
2. **Initialization Tests**: Check models can be created and moved to GPU
3. **Forward Pass Tests**: Validate single batch inference
4. **Configuration Tests**: Ensure stage02_training.json loads correctly
5. **Dependency Tests**: Check PyTorch version, CUDA availability, system memory

### Test Markers

- `@pytest.mark.gpu`: Tests requiring GPU (skip on CPU-only systems)
- `@pytest.mark.skipif`: Conditional skipping based on module availability

## Stage 02 Training Commands

### Spatial Expert Training

```bash
# Concept validation (10 epochs)
PYTHONPATH=. python src/stage_02/train_stage2_spatial.py \
  --mode concept_validation \
  --config configs/stage02_training.json \
  --epochs 10 \
  --batch_size 32

# Full training
PYTHONPATH=. python src/stage_02/train_stage2_spatial.py \
  --mode full_training \
  --config configs/stage02_training.json \
  --epochs 50 \
  --batch_size 32 \
  --dataset_config configs/datasets.json \
  --manifest_dataset celebdf_v2 \
  --manifest_mode balanced
```

### GenConViT Expert Training

```bash
# Concept validation
PYTHONPATH=. python src/stage_02/train_stage2_genconvit.py \
  --config configs/stage02_training.json \
  --epochs 10 \
  --batch_size 16 \
  --dataset_config configs/datasets.json \
  --manifest_dataset celebdf_v2 \
  --manifest_mode balanced
```

### Using Manifests vs ImageFolder

Stage 02 supports both manifest-based and ImageFolder data loading:

```bash
# Manifest-based (recommended, uses existing datasets.json)
--dataset_config configs/datasets.json --manifest_dataset celebdf_v2 --manifest_mode balanced

# Fallback ImageFolder (for testing)
--data_path /path/to/ImageFolder --train_manifest --val_manifest --test_manifest
```

## Integration with Stage 01 Results

Stage 02 builds on the high-performance baseline from Stage 01:

- **Baseline Reference**: Stage 01 AUC 0.99144 (full_training_20251014_100257_cf074fe9)
- **Performance Target**: Stage 02 aims for OOD improvement over Stage 01
- **Cascade Integration**: Stage 02 processes ~5% samples routed from Stage 01

## Troubleshooting

### Common Issues

1. **Import Errors**: Run smoke tests to verify all dependencies
2. **CUDA Out of Memory**: Reduce batch_size in config
3. **Manifest Not Found**: Check configs/datasets.json paths
4. **Model Loading Failures**: Verify checkpoint paths and model compatibility

### Performance Tips

- Use `--mode concept_validation` for quick testing (10 epochs)
- Monitor GPU memory usage during multi-expert training
- Validate data loading with small subsets first
- Use existing manifests from Stage 01 for consistency

# Model Diagnostics Tool

Comprehensive diagnostic analysis for trained deepfake detection models.

## Features

- **ROC Curve Analysis**: Plot ROC curve with optimal threshold identification
- **Confusion Matrix**: Visualize confusion matrix at different thresholds
- **Threshold Optimization**: Find best threshold for accuracy and F1-score
- **Prediction Distribution**: Analyze prediction confidence for real vs fake
- **Comprehensive Reports**: JSON report with all metrics

## Usage

### Command Line Interface

```bash
python -m tools.validation.model_diagnostics \
  --checkpoint experiments/baseline_lodo_df_comparison_20251004_194410_440eaa09/checkpoints/best_model.pth \
  --model-type baseline \
  --model-name tf_efficientnetv2_b0 \
  --test-dataset deeperforensics_1_0 \
  --output-dir experiments/baseline_lodo_df_comparison_20251004_194410_440eaa09/diagnostics \
  --batch-size 128
```

### Parameters

- `--checkpoint`: Path to model checkpoint (.pth file)
- `--model-type`: Model type (`baseline` or `supcon`)
- `--model-name`: Model architecture (`tf_efficientnetv2_b0`, `tf_efficientnetv2_b3`, etc.)
- `--test-dataset`: Dataset to evaluate (`celebdf_v2`, `faceforensics_plus_plus`, `deeperforensics_1_0`)
- `--output-dir`: Directory to save diagnostic outputs
- `--batch-size`: Batch size for evaluation (default: 128)

### Python API

```python
from tools.validation.model_diagnostics import ModelDiagnostics

# Create diagnostics tool
diagnostics = ModelDiagnostics(
    checkpoint_path='path/to/checkpoint.pth',
    model_type='baseline',
    model_name='tf_efficientnetv2_b0'
)

# Generate full report
report = diagnostics.generate_full_report(
    dataset_name='deeperforensics_1_0',
    output_dir='diagnostics/',
    batch_size=128
)
```

## Output Files

The tool generates the following files in the output directory:

1. **roc_curve.png** - ROC curve with optimal threshold marked
2. **confusion_matrix_threshold_0.5.png** - Confusion matrix at default threshold
3. **confusion_matrix_threshold_optimal.png** - Confusion matrix at optimal threshold
4. **threshold_analysis.png** - Accuracy/F1/Precision/Recall vs threshold curves
5. **prediction_distribution.png** - Histogram and boxplot of predictions
6. **diagnostic_report.json** - Comprehensive JSON report with all metrics

## Example: Diagnosing Baseline Model

```bash
# Diagnose baseline model on DeeperForensics OOD test
python -m tools.validation.model_diagnostics \
  --checkpoint experiments/baseline_lodo_df_comparison_20251004_194410_440eaa09/checkpoints/best_model.pth \
  --model-type baseline \
  --test-dataset deeperforensics_1_0 \
  --output-dir experiments/baseline_lodo_df_comparison_20251004_194410_440eaa09/diagnostics
```

## Example: Diagnosing SupCon Model

```bash
# Diagnose SupCon model on DeeperForensics OOD test
python -m tools.validation.model_diagnostics \
  --checkpoint experiments/supcon_medium_training_20251004_160336/checkpoints/best_model.pth \
  --model-type supcon \
  --test-dataset deeperforensics_1_0 \
  --output-dir experiments/supcon_medium_training_20251004_160336/diagnostics
```

## Interpreting Results

### ROC Curve
- **AUC close to 1.0**: Excellent discrimination
- **AUC around 0.7-0.8**: Moderate discrimination
- **AUC around 0.5**: Random guessing
- **Optimal threshold**: Maximizes TPR - FPR (Youden's index)

### Confusion Matrix
- **High TN & TP**: Good classification
- **High FP**: Model too aggressive (many false alarms)
- **High FN**: Model too conservative (misses fakes)

### Threshold Analysis
- **Best accuracy threshold**: Maximizes overall correctness
- **Best F1 threshold**: Balances precision and recall
- Often differ from default 0.5 threshold

### Prediction Distribution
- **Good separation**: Real and Fake peaks are far apart
- **Overlap**: Model struggles to distinguish
- **Skewed distributions**: May indicate calibration issues


# AWARE-NET Test Suite

Unit tests for Stage 0 components to ensure functionality and catch regressions.

## Test Coverage

### Core Model Tests (`test_baseline_model.py`)
- ✅ Model instantiation (B0, B3 variants)
- ✅ Forward pass validation
- ✅ Batch processing
- ✅ Training/evaluation modes
- ✅ Parameter management
- ✅ Feature extraction

### Dataset Tests (`test_dataset.py`)
- ✅ Multi-dataset wrapper functionality
- ✅ Manifest loading and parsing
- ✅ Class count calculations
- ✅ Data loading pipeline
- ✅ Subset ratio handling

### Academic Metrics Tests (`test_metrics.py`)
- ✅ AUC calculation with confidence intervals
- ✅ Accuracy, F1, precision, recall metrics
- ✅ Bootstrap sampling
- ✅ Edge case handling
- ✅ Reproducibility validation

### Calibration Tests (`test_calibration.py`)
- ✅ ECE/MCE calculation
- ✅ Reliability diagram data
- ✅ Bootstrap confidence intervals
- ✅ Temperature scaling
- ✅ Different calibration scenarios

### Experiment Management Tests (`test_experiment_utils.py`)
- ✅ Experiment configuration
- ✅ Experiment lifecycle management
- ✅ Registry operations
- ✅ History tracking
- ✅ Reproducibility utilities

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Test File
```bash
pytest tests/test_baseline_model.py
```

### With Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Verbose Output
```bash
pytest tests/ -v
```

### Skip Slow Tests
```bash
pytest tests/ -m "not slow"
```

## Test Markers

- `slow`: Tests that take longer to run
- `gpu`: Tests requiring GPU hardware
- `integration`: End-to-end integration tests

## Test Data

Tests use:
- **Mock manifests**: Temporary CSV files with sample data
- **Generated predictions**: Synthetic but realistic prediction data
- **Temporary directories**: Isolated test environments
- **Fixtures**: Shared test data and setup

## Expected Behavior

### Passing Tests
- All tests should pass on CPU-only systems
- GPU tests are optional (marked with `@pytest.mark.gpu`)
- Integration tests may be skipped if dependencies missing

### Known Limitations
- Model tests use `pretrained=False` to avoid downloads
- Dataset tests use placeholder data (real images don't exist)
- Some edge cases intentionally test error handling

## Continuous Integration

These tests are designed to run in CI environments:
- No external data dependencies
- Reasonable execution time (< 5 minutes total)
- Graceful handling of missing optional dependencies
- Cross-platform compatibility

## Adding New Tests

When adding new components to Stage 0:

1. Create corresponding test file in `tests/`
2. Follow naming convention: `test_<module_name>.py`
3. Use fixtures from `conftest.py` for common data
4. Add appropriate markers for slow/GPU tests
5. Test both success and failure cases
6. Ensure tests are deterministic and reproducible