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

## 🧪 Running Experiments

### Basic Training
```bash
# Stage 0 baseline training
python src/stage_00/train_baseline.py --dataset balanced_clean --training balanced_clean

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