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