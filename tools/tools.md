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
│   └── profile_stage2_performance.py - Stage 2性能分析
├── validation/              # 驗證工具
│   ├── stage_gate_validator.py      - Stage Gate驗證器
│   └── verify_stage_0_completion.py - Stage 0完成驗證
└── tests/                   # 測試文件
    ├── test_baseline_model.py
    ├── test_dataset.py
    ├── test_metrics.py
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
```

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