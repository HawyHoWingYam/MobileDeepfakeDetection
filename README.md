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