# MobileDeepfakeDetection

A two-stage cascaded deepfake detection system designed for mobile deployment.

**Paper:** *MobileDeepfake: A Lightweight Cascaded Deepfake Detection System for Mobile Deployment*

## Overview

- **Stage 1**: MobileNetV4-Hybrid-Medium lightweight filter for fast initial classification
- **Stage 2**: EfficientNetV2-B3 expert model for ambiguous samples
- **Target**: On-device detection with strict latency, footprint, and privacy constraints
- **Focus**: Minimizing false negatives (missed fakes) under escalation-rate budgets

The system is trained across multiple academic datasets and tuned with a cost-aware threshold grid search. Compact ONNX models are exported for mobile integration.

---

## Key Results

### Stage 1 Performance (MobileNetV4, Calibrated)

| Dataset | AUC | F1 | Accuracy | Precision | Recall | FNR |
|---------|-----|-----|----------|-----------|--------|-----|
| FaceForensics++ | 0.9683 | 0.9171 | 0.9138 | 0.8905 | 0.9453 | 0.0547 |
| CelebDF-v2 | 0.9982 | 0.9771 | 0.9769 | 0.9611 | 0.9936 | 0.0064 |
| DFDC | 0.9931 | 0.9546 | 0.9543 | 0.9512 | 0.9580 | 0.0420 |
| DeeperForensics 1.0 | 1.0000 | 0.9975 | 0.9976 | 1.0000 | 0.9951 | 0.0049 |
| **Macro Average** | **0.9899** | **0.9616** | **0.9607** | **0.9507** | **0.9730** | **0.0270** |
| **Micro Average** | **0.9934** | **0.9700** | **0.9695** | **0.9658** | **0.9744** | **0.0256** |

### Cascade Performance (Best Configuration)

| Configuration | AUC | F1 | FNR | Stage-2 Rate |
|---------------|-----|-----|-----|--------------|
| tau_low=0.05, tau_high=0.55 | 0.9941 | 0.9654 | 0.0060 | 1.16% |

### Mobile Deployment Metrics

| Artifact | Format | Size |
|----------|--------|------|
| Stage 1 (MobileNetV4) | ONNX | 37.45 MB |
| Stage 2 (EfficientNetV2-B3) | ONNX | 51.89 MB |
| **Total** | -- | **89.34 MB** |

**On-Device Performance (Android, 152 test images):**

| Metric | Value |
|--------|-------|
| Accuracy | 92.8% |
| False Negative Rate | 2.5% |
| False Positive Rate | 12.5% |
| Precision | 89.7% |
| Recall | 97.5% |
| Stage-2 Usage | 7.2% |
| Total Inference Time | 179.7 ms |

---

## Repository Structure

```text
MobileDeepfakeDetection/
├── src/                    # Training, evaluation, and cascade code
│   ├── stage1/             # MobileNetV4 filter (train/calibrate/evaluate)
│   ├── stage2/             # EfficientNetV2-B3 expert training
│   ├── stage3/             # LightGBM meta-model (research tool, not deployed)
│   ├── stage4/             # CascadeDetector, threshold tuning, benchmarking
│   ├── tools/              # Robustness analysis, paper asset generation
│   ├── common/             # Shared utilities (losses, etc.)
│   └── utils/              # Dataset configuration helpers
├── paper/                  # LaTeX source, figures, generated tables
│   ├── sections/           # Paper section .tex files
│   ├── figures/            # Figures and diagrams
│   └── generated/          # Auto-generated tables from experiments
├── android/                # Android demo application
│   ├── app/                # Main Android app source
│   └── mobile_bundle/      # ONNX models and cascade config
├── scripts/                # Preprocessing and export utilities
├── config/                 # Dataset configuration (dataset_paths.json)
├── test/                   # Unit tests for Stage 1/2
├── weights/                # Face detector weights (Caffe SSD)
├── environment.yml         # Conda environment specification
└── requirements.txt        # Python dependencies
```

**Note:** `manifests/` and `outputs/` directories are generated during training and evaluation but are gitignored due to size.

---

## Datasets and Task

### Training and Validation Datasets

The system trains on four academic deepfake datasets:

- **CelebDF-v2**: High-quality celebrity face-swap videos
- **FaceForensics++ (FF++)**: Multiple manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
- **DFDC**: Crowd-sourced dataset with diverse actors and compression pipelines
- **DeeperForensics-1.0**: Real-world perturbations and challenging textures

All datasets are preprocessed into **256x256 face crops** using MTCNN face detection.

### Out-of-Distribution Evaluation

- **Deepfake-Eval-2024**: In-the-wild benchmark from real social platforms, used strictly as a held-out OOD test set

### Task Definition

Frame-level binary classification:
- Label `0` = Real face
- Label `1` = Fake face

Video-level predictions can be aggregated from frame scores (e.g., max or average probability).

---

## Pipeline Overview

The system follows a six-stage pipeline aligned with the paper's methodology:

### Stage 0: Data Preparation
- Configure datasets via `config/dataset_paths.json`
- Preprocess videos into 256x256 PNG face crops using `scripts/preprocess_datasets_v2.py`
- Generate train/val/test manifests

### Stage 1: Lightweight Filter (MobileNetV4)
- Train with `src/stage1/train_stage1.py`
- Light augmentation, AdamW optimizer, cosine LR schedule
- Calibrate probabilities via temperature scaling
- Select best checkpoint by validation AUC

### Stage 2: Expert Model (EfficientNetV2-B3)
- Train with `src/stage2/train_stage2_effnet.py`
- Stronger augmentation (RandAugment + Mixup/CutMix)
- Focal Loss with cosine-annealing warm restarts
- Cache logits/embeddings for cascade tuning

### Stage 3: Meta-Model (Optional Research Tool)
- LightGBM meta-model for analysis
- **Not part of default deployment** - research tool only

### Stage 4: Cascade Threshold Tuning
- Grid search over (tau_low, tau_high) thresholds
- Minimize FNR under Stage-2 usage budget
- Implement cascade logic in `src/stage4/cascade_detector.py`

### Stage 5: Robustness Evaluation
- Cross-dataset generalization testing
- Robustness analysis (JPEG compression, noise, blur, brightness)
- Generate LaTeX tables/figures for paper

### Stage 6: Mobile Export (ONNX)
- Export Stage 1/2 to ONNX format
- Bundle models with cascade configuration
- Output to `android/mobile_bundle/`

---

## Quick Start

### Environment Setup

```bash
conda env create -f environment.yml
conda activate aware-net
```

### Dataset Configuration

1. Edit `config/dataset_paths.json` with your dataset locations
2. Run preprocessing:

```bash
python scripts/preprocess_datasets_v2.py \
  --video-backend decord \
  --face-detector mtcnn \
  --workers 4
```

### Training Stage 1

```bash
python src/stage1/train_stage1.py --data_dir processed_data
python src/stage1/calibrate_model.py --data_dir processed_data
python src/stage1/evaluate_stage1.py --data_dir processed_data
```

### Training Stage 2 and Cascade Tuning

```bash
python src/stage2/train_stage2_effnet.py --data_dir processed_data
python src/stage4/benchmark_cascade.py --benchmark_all
```

### Mobile Export (ONNX)

```bash
python scripts/export_mobile_cascade_onnx.py
```

This generates ONNX models in `android/mobile_bundle/`.

---

## Android Application

The project includes a demo Android app for on-device deepfake detection.

### Features
- Two-stage cascade detection using ONNX Runtime
- Jetpack Compose UI with Material 3 design
- Single image detection with confidence scores
- Timing breakdown (preprocessing, inference)

### Requirements
- Android Studio Hedgehog (2023.1.1) or later
- Min SDK: 24 (Android 7.0)
- Target SDK: 34
- Device with 2GB+ RAM recommended

### Model Setup

After running `scripts/export_mobile_cascade_onnx.py`, copy models to the app:

```bash
cp android/mobile_bundle/*.onnx android/app/src/main/assets/models/
cp android/mobile_bundle/cascade_config.json android/app/src/main/assets/models/
```

### Expected Performance
- Stage 1 inference: 5-10 ms
- Stage 2 inference: 15-25 ms (when triggered)
- Total time: 10-30 ms per image (mid-range device)
- Stage 2 usage rate: ~1-5%

See `android/README.md` for detailed build and usage instructions.

---

## Implementation Notes

### Cascade Threshold Semantics

Stage 1 outputs P(fake) after sigmoid. The cascade decision rule:
- `p1(x) < tau_low` -> Predict **real** (high confidence)
- `p1(x) > tau_high` -> Predict **fake** (high confidence)
- Otherwise -> Escalate to Stage 2

Default thresholds: `tau_low=0.05`, `tau_high=0.55`

### Stage 3 Meta-Model

The LightGBM meta-model in `src/stage3/` is a **research tool** for offline analysis. The deployed mobile system uses only Stage 1 + Stage 2 (two-stage cascade).

### Hard Example Mining (HEM)

The released training scripts use **uniform sampling** by default. HEM is available for ablation studies but is not part of the main reported results.

### Dynamic Thresholds

`CascadeConfig.enable_dynamic_thresholds` provides a prototype hook for adaptive thresholds. All paper experiments use **static thresholds** from grid search.

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{ho2024mobiledeepfake,
  title={MobileDeepfake: A Lightweight Cascaded Deepfake Detection System for Mobile Deployment},
  author={Ho, Wing Yam},
  institution={The Hong Kong Polytechnic University},
  year={2024}
}
```

The paper PDF is available at `paper/main.pdf`.

---

## Acknowledgments

This project builds upon several open-source frameworks and datasets:

**Frameworks:**
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference

**Datasets:**
- [CelebDF-v2](https://github.com/yuezunli/celeb-deepfakeforensics)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC](https://ai.facebook.com/datasets/dfdc/)
- [DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0)

---

## References

- Paper source: `paper/main.tex`
- Full bibliography: `paper/references.bib`
- Code repository: [GitHub](https://github.com/HawyHoWingYam/MobileDeepfakeDetection)
