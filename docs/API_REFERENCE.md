# AWARE-NET Stage 0 API Reference

## Overview

This document provides comprehensive API reference for AWARE-NET Stage 0 components, including calibration tools, academic metrics, dataset management, and baseline model evaluation.

---

## Calibration Tools (`src/utils/calibration_tools.py`)

### CalibrationAnalyzer

Comprehensive model calibration analysis toolkit with academic rigor.

#### Constructor

```python
CalibrationAnalyzer(
    n_bins: int = 15,
    bin_strategy: str = 'uniform',
    confidence_level: float = 0.95,
    random_state: int = 42
)
```

**Parameters:**
- `n_bins`: Number of bins for reliability analysis
- `bin_strategy`: Binning strategy ('uniform' or 'quantile')
- `confidence_level`: Confidence level for intervals (0-1)
- `random_state`: Random seed for reproducibility

#### Core Methods

##### `calculate_ece_mce(y_true, y_prob, return_details=True)`

Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

**Parameters:**
- `y_true` (np.ndarray): True binary labels (0 or 1)
- `y_prob` (np.ndarray): Predicted probabilities [0, 1]
- `return_details` (bool): Whether to return detailed bin information

**Returns:**
- `CalibrationResult`: Contains ECE, MCE, Brier score, and reliability diagram data

**Example:**
```python
analyzer = CalibrationAnalyzer(n_bins=10)
result = analyzer.calculate_ece_mce(y_true, y_prob)
print(f"ECE: {result.ece:.4f}, MCE: {result.mce:.4f}")
```

##### `bootstrap_ece_confidence_interval(y_true, y_prob, n_bootstrap=1000)`

Calculate ECE with bootstrap confidence interval.

**Parameters:**
- `y_true` (np.ndarray): True binary labels
- `y_prob` (np.ndarray): Predicted probabilities
- `n_bootstrap` (int): Number of bootstrap samples

**Returns:**
- `Tuple[float, Tuple[float, float]]`: (ECE, confidence_interval)

##### `temperature_scaling(y_true, logits, validation_split=0.2)`

Apply temperature scaling for calibration improvement.

**Parameters:**
- `y_true` (np.ndarray): True binary labels
- `logits` (np.ndarray): Raw model logits (before softmax)
- `validation_split` (float): Fraction for temperature optimization

**Returns:**
- `TemperatureScalingResult`: Contains optimal temperature and calibrated predictions

##### `plot_reliability_diagram(calibration_result, title="Reliability Diagram", save_path=None, figsize=(8, 6))`

Create reliability diagram (calibration plot).

**Parameters:**
- `calibration_result` (CalibrationResult): Result from calculate_ece_mce
- `title` (str): Plot title
- `save_path` (Optional[str]): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

**Returns:**
- `plt.Figure`: Matplotlib figure

### Data Classes

#### CalibrationResult
```python
@dataclass
class CalibrationResult:
    ece: float                    # Expected Calibration Error
    mce: float                    # Maximum Calibration Error
    brier_score: float            # Brier score
    reliability_diagram_data: Dict[str, np.ndarray]  # Plot data
    confidence_interval: Optional[Tuple[float, float]] = None
    n_bins: int = 10
    n_samples: int = 0
```

#### TemperatureScalingResult
```python
@dataclass
class TemperatureScalingResult:
    optimal_temperature: float     # Optimal temperature parameter
    calibrated_predictions: np.ndarray  # Calibrated probabilities
    pre_calibration_ece: float     # ECE before calibration
    post_calibration_ece: float    # ECE after calibration
    improvement: float             # ECE improvement
    convergence_info: Dict[str, Any]  # Optimization details
```

---

## Academic Metrics (`src/utils/metrics.py`)

### AcademicMetrics

Academic-grade evaluation metrics with statistical testing.

#### Constructor

```python
AcademicMetrics(
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 42
)
```

#### Core Methods

##### `calculate_auc_with_ci(y_true, y_scores)`

Calculate AUC-ROC with confidence interval.

**Parameters:**
- `y_true` (np.ndarray): True binary labels
- `y_scores` (np.ndarray): Predicted scores/probabilities

**Returns:**
- `MetricResult`: AUC value with confidence interval and statistics

**Example:**
```python
metrics = AcademicMetrics()
auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
print(f"AUC: {auc_result.value:.4f} [{auc_result.confidence_interval[0]:.4f}, {auc_result.confidence_interval[1]:.4f}]")
```

##### `calculate_accuracy_with_ci(y_true, y_pred)` 
##### `calculate_f1_with_ci(y_true, y_pred)`
##### `calculate_precision_with_ci(y_true, y_pred)`
##### `calculate_recall_with_ci(y_true, y_pred)`

Calculate classification metrics with confidence intervals.

**Parameters:**
- `y_true` (np.ndarray): True binary labels
- `y_pred` (np.ndarray): Predicted binary labels

**Returns:**
- `MetricResult`: Metric value with confidence interval

### Data Classes

#### MetricResult
```python
@dataclass
class MetricResult:
    value: float                           # Metric value
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None        # Statistical significance
    std_error: Optional[float] = None      # Standard error
    n_samples: int = 0                     # Sample size
```

---

## Dataset Management (`src/utils/dataset_config.py`)

### DatasetConfig

Configuration-driven dataset management.

#### Constructor

```python
DatasetConfig(config_path: str)
```

**Parameters:**
- `config_path`: Path to JSON configuration file

#### Properties

```python
config = DatasetConfig("config.json")
print(f"Dataset: {config.name}")
print(f"Total samples: {config.total_samples}")
print(f"Image size: {config.image_size}")
```

#### Methods

##### `get_manifest_path(split_name)`

Get manifest path for a data split.

**Parameters:**
- `split_name` (str): Split name ('train', 'val', 'test')

**Returns:**
- `Path`: Path to manifest file

##### `save(output_path)`

Save updated configuration to file.

**Parameters:**
- `output_path` (str): Output file path

---

## Manifest Generation (`src/utils/manifest_generator.py`)

### ManifestGenerator

Generate and validate dataset manifest files.

#### Constructor

```python
ManifestGenerator(
    real_image_dir: str,
    fake_image_dir: str,
    output_dir: str = "manifests",
    supported_formats: List[str] = ['.png', '.jpg', '.jpeg']
)
```

#### Methods

##### `generate_manifest(include_metadata=False)`

Generate complete dataset manifest.

**Returns:**
- `Path`: Path to generated manifest file

##### `generate_split_manifests(split_ratios, stratify=True, random_state=42)`

Generate train/val/test split manifests.

**Parameters:**
- `split_ratios` (Dict[str, float]): Split ratios (must sum to 1.0)
- `stratify` (bool): Whether to stratify by class
- `random_state` (int): Random seed

**Returns:**
- `Dict[str, Path]`: Dictionary of split_name -> manifest_path

**Example:**
```python
generator = ManifestGenerator("real/", "fake/", "manifests/")
splits = generator.generate_split_manifests({
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
})
```

##### `validate_manifest(manifest_path)`

Validate manifest file integrity.

**Returns:**
- `Tuple[bool, List[str]]`: (is_valid, error_list)

---

## Data Validation (`src/utils/data_validator.py`)

### DataValidator

Comprehensive data quality validation.

#### Methods

##### `validate_image(image_path)`

Validate individual image file.

**Returns:**
- `Tuple[bool, Optional[str]]`: (is_valid, error_message)

##### `validate_manifest(manifest_path)`

Validate complete manifest file.

**Returns:**
- `Dict[str, Any]`: Validation results with statistics

##### `calculate_image_statistics(manifest_path)`

Calculate comprehensive image statistics.

**Returns:**
- `Dict[str, Any]`: Image statistics and distributions

##### `comprehensive_quality_check(manifest_path, check_duplicates=True, check_corruption=True, check_dimensions=True)`

Perform comprehensive data quality assessment.

**Returns:**
- `Dict[str, Any]`: Complete quality report

---

## Baseline Model (`src/stage_00/baseline_model.py`)

### EfficientNetV2B3Baseline

EfficientNetV2-B3 based baseline model for deepfake detection.

#### Constructor

```python
EfficientNetV2B3Baseline(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    freeze_backbone: bool = False
)
```

**Parameters:**
- `num_classes`: Number of output classes
- `pretrained`: Whether to use pretrained weights
- `dropout_rate`: Dropout rate for regularization
- `freeze_backbone`: Whether to freeze backbone weights

#### Methods

##### `forward(x)`

Forward pass through the model.

**Parameters:**
- `x` (torch.Tensor): Input tensor [batch_size, 3, height, width]

**Returns:**
- `torch.Tensor` or `Dict`: Model outputs (logits)

**Example:**
```python
model = EfficientNetV2B3Baseline(num_classes=2, pretrained=False)
x = torch.randn(4, 3, 256, 256)
outputs = model(x)
print(f"Output shape: {outputs.shape}")  # [4, 2]
```

---

## Baseline Evaluation (`src/stage_00/evaluate_baseline.py`)

### BaselineEvaluator

Comprehensive baseline model evaluation framework.

#### Constructor

```python
BaselineEvaluator(
    model_path: str,
    config_path: str,
    output_dir: str = "results/baseline_evaluation",
    device: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4
)
```

#### Methods

##### `evaluate_single_dataset(manifest_path, dataset_name, save_predictions=True)`

Evaluate model on a single dataset.

**Returns:**
- `DatasetEvaluationResult`: Comprehensive evaluation results

##### `evaluate_cross_dataset(datasets, train_dataset_name)`

Evaluate model across multiple datasets.

**Parameters:**
- `datasets` (Dict[str, str]): dataset_name -> manifest_path mapping
- `train_dataset_name` (str): Name of training dataset

**Returns:**
- `CrossDatasetEvaluationResult`: Cross-dataset analysis results

##### `generate_failure_analysis_report(results, save_visualizations=True)`

Generate comprehensive failure analysis.

**Returns:**
- `Dict[str, Any]`: Failure analysis insights

##### `generate_academic_report(cross_dataset_result, failure_analysis, format='markdown')`

Generate academic-style evaluation report.

**Parameters:**
- `format` (str): Output format ('markdown' or 'latex')

**Returns:**
- `str`: Formatted report string

##### `run_comprehensive_evaluation(datasets, train_dataset_name, generate_report=True)`

Run complete evaluation pipeline.

**Returns:**
- `Dict[str, Any]`: Complete evaluation results

**Example:**
```python
evaluator = BaselineEvaluator(
    model_path="model.pth",
    config_path="config.json"
)

datasets = {
    "celebdf": "manifests/celebdf_test.csv",
    "ff++": "manifests/ffpp_test.csv"
}

results = evaluator.run_comprehensive_evaluation(
    datasets=datasets,
    train_dataset_name="celebdf"
)
```

### Data Classes

#### DatasetEvaluationResult
```python
@dataclass
class DatasetEvaluationResult:
    dataset_name: str
    n_samples: int
    n_real: int
    n_fake: int
    
    # Performance metrics
    auc_roc: MetricResult
    accuracy: MetricResult
    f1_score: MetricResult
    precision: MetricResult
    recall: MetricResult
    
    # Calibration metrics
    calibration: CalibrationResult
    
    # Per-class performance
    real_precision: float
    real_recall: float
    fake_precision: float
    fake_recall: float
    
    # Failure analysis
    false_positives: List[str]
    false_negatives: List[str] 
    high_confidence_errors: List[Tuple[str, float, int, int]]
```

---

## Testing Framework

### Running Tests

```bash
# Run all tests
python tests/run_tests.py --suite all

# Run specific test categories
python tests/run_tests.py --suite unit
python tests/run_tests.py --suite performance
python tests/run_tests.py --suite stage-gate

# Run individual test files
python -m pytest tests/test_calibration.py -v
python -m pytest tests/test_metrics.py -v
python -m pytest tests/test_baseline_model.py -v
```

### Test Configuration

Tests are configured via `tests/pytest.ini` with the following markers:
- `slow`: Tests that may take several minutes
- `gpu`: Tests requiring GPU
- `integration`: Integration tests
- `performance`: Performance benchmark tests
- `stage_gate`: Stage-Gate validation tests

---

## Stage-Gate Validation

### Automatic Validation

```bash
# Run complete Stage-Gate validation
python stage_gate_validator.py --category all

# Run specific validation categories
python stage_gate_validator.py --category technical
python stage_gate_validator.py --category academic
python stage_gate_validator.py --category system
```

### Validation Criteria

#### Technical Gates (50% weight)
- Environment Success Rate ≥ 95%
- Data Management: 4+ dataset formats supported  
- Tool Library: All evaluation tools operational
- Baseline Model: Complete implementation with AUC ≥ 0.88
- Project Structure: 10-stage architecture + utilities

#### Academic Gates (30% weight)  
- Reproducibility: Seed management + deterministic training + config
- Statistical Rigor: CI + significance tests + bootstrap methods
- Baseline Analysis: Failure analysis + cross-dataset + recommendations
- Documentation: API docs + technical specs + user guides

#### System Gates (20% weight)
- Cross-Platform: Docker + conda/pip support
- Performance: Inference < 100ms, Memory < 4GB
- Extensibility: Modular architecture + plugin support + APIs
- Operational: Monitoring + logging + error handling + testing

### Quantified Success Metrics

| Metric | Minimum | Target | Test Method |
|--------|---------|---------|-------------|
| Environment Success Rate | ≥ 95% | ≥ 99% | 10 independent installations |
| Baseline Model AUC | ≥ 0.88 | ≥ 0.90 | 4 dataset average |
| Data Loading Speed | ≥ 100/sec | ≥ 200/sec | 1000 sample test |
| Code Coverage | ≥ 80% | ≥ 90% | pytest automation |
| Documentation Completeness | ≥ 90% | ≥ 95% | API coverage check |

---

## Error Handling

### Common Error Types

All components implement comprehensive error handling:

```python
try:
    result = analyzer.calculate_ece_mce(y_true, y_prob)
except ValueError as e:
    # Handle input validation errors
    print(f"Input validation failed: {e}")
except RuntimeError as e:
    # Handle computation errors
    print(f"Computation failed: {e}")
```

### Validation Errors

- **Input Validation**: Array shapes, value ranges, data types
- **Configuration Errors**: Missing files, invalid JSON, path errors
- **Resource Errors**: Memory limits, GPU availability, file permissions

---

## Performance Optimization

### Memory Management

```python
# Use appropriate batch sizes
evaluator = BaselineEvaluator(batch_size=16)  # Reduce if OOM

# Enable memory monitoring
import torch
torch.cuda.empty_cache()  # Clear GPU memory
```

### Speed Optimization

```python
# Use fewer bootstrap samples for faster CI
metrics = AcademicMetrics(n_bootstrap=100)  # Default: 1000

# Reduce calibration bins for speed
calibration = CalibrationAnalyzer(n_bins=10)  # Default: 15

# Parallelize data loading
evaluator = BaselineEvaluator(num_workers=8)
```

---

## Best Practices

### Reproducibility

```python
# Always set random seeds
import numpy as np
import torch
import random

def set_reproducible_mode(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### Academic Rigor

```python
# Always use confidence intervals
auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
print(f"AUC: {auc_result.value:.4f} ± {(auc_result.confidence_interval[1] - auc_result.confidence_interval[0])/2:.4f}")

# Perform calibration analysis
calibration = CalibrationAnalyzer()
cal_result = calibration.calculate_ece_mce(y_true, y_prob)
print(f"ECE: {cal_result.ece:.4f}")
```

### Configuration Management

```python
# Use JSON configuration files
config = {
    "model": {"architecture": "efficientnet", "dropout": 0.2},
    "training": {"batch_size": 32, "learning_rate": 1e-3},
    "data": {"image_size": 256, "augmentation": True}
}

# Save configurations with results
results["config"] = config
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Import Errors**: Check environment installation
3. **Path Errors**: Use absolute paths for reliability
4. **Permission Errors**: Check file system permissions
5. **Configuration Errors**: Validate JSON syntax

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use development settings
config.update({
    "n_bootstrap": 10,     # Reduce for faster debugging  
    "batch_size": 4,       # Small batch for memory
    "num_workers": 0       # Single-threaded for debugging
})
```

---

This API reference covers all major components of AWARE-NET Stage 0. For additional examples and tutorials, see the test files in `tests/` directory and the implementation files in `src/`.