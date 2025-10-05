# AWARE-NET: Advanced Deepfake Detection Framework

## 📋 Project Overview

**AWARE-NET** is a comprehensive deep fake detection framework built as a 10-stage academic research project. The project implements a sophisticated multi-stage approach combining rapid filtering, heterogeneous expert models, and advanced fusion techniques for mobile-optimized deepfake detection.

### 🎯 Key Features

- **10-Stage Implementation Strategy**: Progressive development with rigorous stage-gate validation
- **Paradigm Innovation**: Shift from "detect fake" to "model authenticity" using SupCon learning
- **Heterogeneous Expert System**: Spatial, generative, and temporal detection specialists
- **SAT Framework**: Novel self-supervised adversarial training approach
- **Mobile Optimization**: Designed for deployment on mid-range mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (GPU support)
- 16GB+ RAM
- 50GB+ available storage

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd MobileDeepfakeDetection
```

2. **Set up environment**
```bash
# Create conda environment
conda env create -f environment.yml
conda activate aware_net_rtx50

# Verify installation
python tools/scripts/setup_environment.py --validate-only
```

3. **Quick test**
```bash
# Test with 3-epoch baseline training
python src/stage_00/train_baseline.py --dataset original_imbalanced --training quick_test
```

## 📦 DFDC Onboarding Guide

To integrate the DFDC dataset into the AWARE-NET pipeline:

1. **Prepare Frames** – extract frames into `dataset/real/DFDC/<video_id>/...` and `dataset/fake/DFDC/<video_id>/...` (e.g. `ffmpeg -i video.mp4 -vf fps=5 dataset/real/DFDC/id/%06d.jpg`).
2. **Generate Manifests** – create CSV manifests and validate them:
   ```bash
   python -m src.utils.manifest_generator \
     --config configs/dfdc.json \
     --real-dir dataset/real/DFDC \
     --fake-dir dataset/fake/DFDC \
     --no-validate --no-md5
   python -m src.utils.data_validator --config configs/dfdc.json --quick
   ```
   (Remove `--no-validate/--no-md5` if you want full integrity checks.)
3. **Register Configuration** – add a `dfdc` entry to `configs/datasets.json` and include it in `multi_dataset_configs.unified_training.datasets_included` when running multi-dataset training.
4. **Update Training Config** – point `configs/training.json` to the desired dataset (`data.dataset_name: "dfdc"`) or enable `multi_dataset: true` to blend DFDC with other datasets.
5. **Rebuild Baselines** – rerun Stage 00 baselines (LODO Config 3, 3-dataset mix, 4-dataset mix) so checkpoints/metrics reflect the new data.

Record the regenerated metrics in the Stage 00 status report once DFDC integration is complete.

## 🏗️ Project Architecture

```
MobileDeepfakeDetection/
├── src/                    # Source code organized by stages
│   ├── stage_00/          # Infrastructure & baseline models
│   ├── stage_01/          # SupCon rapid filtering system
│   ├── stage_02/          # Heterogeneous expert models
│   ├── stage_03/          # Temporal modeling expert
│   ├── stage_04/          # Feature fusion system
│   ├── stage_05/          # SAT adversarial training
│   ├── stage_06/          # Cascade system integration
│   ├── stage_07/          # Continual learning mechanisms
│   ├── stage_08/          # Mobile deployment optimization
│   └── stage_09/          # Comprehensive evaluation
├── configs/               # Configuration files by stage and function
├── tools/                 # Development tools and utilities
├── project_instruction/   # Complete project documentation
└── models/               # Trained model storage
```

## 📊 Current Status

| Stage | Name | Status | Completion | Key Deliverables |
|-------|------|--------|------------|------------------|
| **Stage 00** | Infrastructure & Baseline | 🔄 | 83% | Environment, data management, baseline model |
| **Stage 01** | SupCon Rapid Filter | 🔄 | 90% | Architecture complete, needs training validation |
| **Stage 02** | Heterogeneous Experts | 🔄 | 85% | Architecture complete, needs training scripts |
| **Stage 03** | Temporal Modeling | 🚀 | 0% | Integration interface ready |
| **Stage 04+** | Advanced Stages | ⏳ | 0% | Awaiting earlier stage completion |

## 📖 Documentation

- **[Configuration Guide](docs/configuration-guide.md)** - Dataset and training configurations
- **[Model Storage Guide](docs/model-storage-guide.md)** - Model management and deployment
- **[Stage Documentation](project_instruction/stage/)** - Detailed specifications for each stage
- **[Implementation Guide](project_instruction/general/implementation_plan.md)** - Complete implementation roadmap
- **[API Reference](project_instruction/general/api_reference.md)** - Code API documentation

## 🔧 Key Components

### Stage 0: Foundation
- EfficientNetV2-B3 baseline model (target AUC: 0.88-0.92)
- Comprehensive data management system
- Academic-grade evaluation tools

### Stage 1: SupCon Filtering
- MobileNetV4-based rapid filtering
- Supervised contrastive learning approach
- Temperature scaling for probability calibration

### Stage 2: Expert Models
- **Spatial Expert**: EfficientNetV2-B3 for artifact detection
- **Generative Expert**: GenConViT for structure analysis
- Advanced complementarity analysis and fusion

## ⚖️ Data Sampling & Weight Configuration

### Understanding Data Balancing

AWARE-NET implements **three-level data balancing** to ensure fair and effective training:

#### 1. **Dataset-Level Weights** (数据集权重)
Controls the contribution of each dataset in multi-dataset training.

**Problem**:
- CelebDF-v2: 83K samples (7.3%)
- FaceForensics++: 225K samples (19.8%)
- DeeperForensics-1.0: 828K samples (72.9%)
- **Without balancing**: DeeperForensics dominates training

**Solution**: `WeightedRandomSampler` ensures equal dataset contributions
- Each dataset sampled with equal probability: 33.3% each
- Prevents any single dataset from dominating the learned features

**Configuration** (`configs/training.json`):
```json
{
  "data": {
    "use_dataset_weights": true,  // Enable dataset-level balancing
    "multi_dataset": true          // Required for multi-dataset training
  }
}
```

#### 2. **Class-Level Weights** (类别权重)
Balances Real vs Fake samples within the training data.

**Problem**:
- Imbalanced datasets may have more real or fake samples
- Model bias towards majority class

**Solution**: `BCEWithLogitsLoss(pos_weight=real_count/fake_count)`
- Adjusts loss function to weight minority class more heavily

**Configuration** (`configs/training.json`):
```json
{
  "training": {
    "use_class_weights": false  // Set false for balanced datasets
                                 // Set true for imbalanced datasets
  }
}
```

**Note**: For `anonymized_balanced` datasets, this should be `false` as data is already 50/50 balanced.

#### 3. **Dataset Modes** (数据集模式)
Two modes available for different training scenarios:

| Mode | Description | Real:Fake Ratio | Path Leakage | Use Case |
|------|-------------|-----------------|--------------|----------|
| `original` | Raw dataset | Varies | ⚠️ Possible | Baseline comparison |
| `balanced` | Original paths + balanced | 50:50 | ✅ Code-level prevention | **Recommended for training** |

**Configuration**:
```json
{
  "data": {
    "dataset_mode": "balanced"  // Recommended
  }
}
```

**Path Leakage Prevention**:
- The `balanced` mode uses original image paths (no image copying needed)
- Path leakage is prevented at code level: dataset only returns `(image_tensor, label)`, paths are never exposed to the model
- Saves 150-200GB storage and 30-60 minutes setup time compared to physical anonymization

### Configuration Examples

#### Single Dataset Training (CelebDF-v2)
```bash
python src/stage_00/train_baseline.py \
  --config configs/training.json \
  --experiment-name celebdf_baseline \
  --model tf_efficientnetv2_b0 \
  --dataset celebdf_v2 \
  --dataset-mode balanced \
  --epochs 20 \
  --batch-size 32
```

**Config file** (`configs/training.json`):
```json
{
  "training": {
    "use_class_weights": false  // Data already balanced
  },
  "data": {
    "dataset_name": "celebdf_v2",
    "dataset_mode": "balanced",
    "multi_dataset": false,
    "use_dataset_weights": false  // N/A for single dataset
  }
}
```

#### Multi-Dataset Training (All 3 datasets) - **Recommended**
```bash
python src/stage_00/train_baseline.py \
  --config configs/training.json \
  --experiment-name multi_baseline \
  --model tf_efficientnetv2_b0 \
  --multi-dataset \
  --dataset-mode balanced \
  --epochs 30 \
  --batch-size 32
```

**Config file** (`configs/training.json`):
```json
{
  "training": {
    "use_class_weights": false  // Data already balanced
  },
  "data": {
    "dataset_mode": "balanced",
    "multi_dataset": true,
    "use_dataset_weights": true  // ✅ Balance dataset contributions
  }
}
```

**Expected output**:
```
⚖️  Calculating dataset-level weights for balanced sampling...
  Dataset contributions (with balanced sampling):
    celebdf_v2_train: 83,378 samples (7.3% → 33.3%)
    faceforensics_plus_plus_train: 225,418 samples (19.8% → 33.3%)
    deeperforensics_1_0_train: 828,200 samples (72.9% → 33.3%)
  ✓ Weighted sampler created (balances dataset contributions)
```

### Best Practices

1. **For Production Training** (推荐设置):
   - Use `balanced` mode to prevent path leakage with 50/50 class balance
   - Enable `use_dataset_weights: true` for multi-dataset training
   - Disable `use_class_weights: false` (data already balanced)

2. **For Imbalanced Research**:
   - Use `original` mode
   - Enable `use_class_weights: true`
   - Consider manual pos_weight tuning

3. **For Quick Testing**:
   - Use `subset_ratio: 0.1` to train on 10% of data
   - Single dataset mode for faster iteration

### Advanced Configuration

**Manual Class Weight Specification**:
```json
{
  "training": {
    "use_class_weights": true,
    "pos_weight": 2.33  // Manual override: real_count/fake_count
  }
}
```

**Subset Training** (for debugging):
```bash
python src/stage_00/train_baseline.py \
  --subset-ratio 0.1 \  # Use only 10% of data
  --epochs 5
```

## 🧪 Running Experiments

### Basic Training
```bash
# Stage 0 baseline training (single dataset)
python src/stage_00/train_baseline.py \
  --config configs/training.json \
  --experiment-name celebdf_b0 \
  --model tf_efficientnetv2_b0 \
  --dataset celebdf_v2 \
  --dataset-mode balanced \
  --epochs 20 \
  --batch-size 32

# Stage 0 multi-dataset training (recommended)
python src/stage_00/train_baseline.py \
  --config configs/training.json \
  --experiment-name multi_b0 \
  --model tf_efficientnetv2_b0 \
  --multi-dataset \
  --dataset-mode balanced \
  --epochs 30 \
  --batch-size 32

# Stage 1 SupCon training (when available)
python src/stage_01/train_stage1_supcon.py

# Stage 2 expert training (when available)
python src/stage_02/train_stage2_spatial.py
python src/stage_02/train_stage2_genconvit.py
```

### Evaluation
```bash
# Baseline evaluation
python src/stage_00/evaluate_baseline.py

# Stage validation
python tools/validation/stage_gate_validator.py --stage 1
```

## 📈 Expected Performance

### Baseline Targets
- **AUC-ROC**: ≥ 0.88-0.92 across all test datasets
- **F1-Score**: ≥ 0.85-0.88
- **Inference Speed**: < 100ms/image
- **Cross-dataset Stability**: Variance < 0.05

### Advanced Stages
- **Stage 1 SupCon**: +3% AUC improvement over baseline
- **Stage 2 Experts**: +3% AUC improvement over Stage 1
- **Final System**: Target AUC ≥ 0.95 with mobile optimization

## 🤝 Contributing

1. Follow the 10-stage development methodology
2. Ensure all code passes stage-gate criteria
3. Maintain academic documentation standards
4. Add comprehensive tests for new functionality

## 📄 License

This implementation is part of the AWARE-NET academic research project. Please refer to the project license for usage terms.

## 🔗 References

- [AWARE-NET Technical Documentation](project_instruction/)
- [Implementation Roadmap](project_instruction/general/implementation_plan.md)
- [Stage-Gate Methodology](project_instruction/stage/)

---

For detailed implementation guidance, refer to the [project instruction documentation](project_instruction/) and stage-specific guides.
