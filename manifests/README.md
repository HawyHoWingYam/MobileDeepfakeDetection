# Dataset Manifests

This directory contains manifest files for the AWARE-NET multi-dataset training pipeline.

## Current Status

⚠️ **PLACEHOLDER MANIFESTS**: These files currently contain placeholder entries and must be regenerated with actual dataset paths when datasets are available.

## Manifest Format

Each CSV manifest file contains the following columns:
- `image_path`: Relative path to image file
- `label`: Classification label (0=real, 1=fake)
- `split`: Data split (train/val/test)
- `md5`: MD5 hash for integrity checking
- `valid`: Boolean indicating if file is valid
- `width`: Image width in pixels
- `height`: Image height in pixels
- `file_size`: File size in bytes

## Dataset Coverage

### CelebDF-v2
- `celebdf_v2_train.csv`
- `celebdf_v2_val.csv`
- `celebdf_v2_test.csv`

### FaceForensics++
- `faceforensics_train.csv`
- `faceforensics_val.csv`
- `faceforensics_test.csv`

### DeeperForensics-1.0
- `deeperforensics_train.csv`
- `deeperforensics_val.csv`
- `deeperforensics_test.csv`

## Regenerating Manifests

To generate actual manifests from dataset directories:

```python
from src.utils.manifest_generator import ManifestGenerator
from src.utils.dataset_config import DatasetConfig

# Load dataset configuration
config = DatasetConfig("configs/dataset_paths.json")

# Generate manifests
generator = ManifestGenerator(config)
generator.generate_all_manifests()
```

## Integration with Training Pipeline

These manifests are automatically loaded by:
- `src/stage_00/train_baseline.py` via `create_multi_dataset_loaders()`
- Multi-dataset configuration in `configs/unified_dataset_config.json`

## Next Steps

1. **Manual Task**: Place actual datasets in appropriate directories
2. **Manual Task**: Run manifest generation scripts with real data paths
3. **Manual Task**: Validate generated manifests for completeness and accuracy
4. **Automatic**: Training pipeline will use these manifests for multi-dataset training