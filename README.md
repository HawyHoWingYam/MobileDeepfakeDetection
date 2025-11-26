# MobileDeepfakeDetection

Two‑stage cascaded deepfake detection system designed for mobile deployment.  
Stage 1 is a lightweight MobileNetV4 filter; Stage 2 is an EfficientNetV2‑B3 expert.  
The project includes a full training → tuning → evaluation → export pipeline and a LaTeX paper.

This README reflects the **current** implementation and is aligned with:

- The paper in `paper/`  
- The Chinese phase documents in `project_instruction/`  
- The Python code in `src/`

---

## 1. High‑Level Overview

- **Goal**: Detect face‑swap deepfakes in videos/images with **high recall** and **controlled compute** on resource‑limited devices.  
- **Main idea**:  
  - Stage 1 (MobileNetV4) quickly filters easy real/fake cases.  
  - Stage 2 (EfficientNetV2‑B3) only processes ambiguous samples.  
  - Stage 4 tunes two thresholds to trade FNR against Stage‑2 usage.  
  - Stage 5 evaluates robustness and cross‑dataset generalization.  
  - Stage 6 exports TorchScript + quantized models for mobile integration.
- **Optional components**:  
  - Stage 3 LightGBM meta‑model and GenConViT expert are kept as **research tools**, not part of the default deployment path.

---

## 2. Repository Structure

Only the most important directories are briefly described here (see `project_instruction/` and `docs/` for more details):

```text
MobileDeepfakeDetection/
├── src/                    # All training / evaluation / cascade / export code
│   ├── stage1/             # Stage 1: MobileNetV4 filter (train / calibrate / evaluate)
│   ├── stage2/             # Stage 2: EfficientNetV2-B3 expert (+ optional GenConViT)
│   ├── stage3/             # Stage 3: LightGBM meta-model (optional analysis)
│   ├── stage4/             # CascadeDetector, threshold tuning, mobile export helpers
│   ├── tools/              # Robustness analysis, paper asset generation, etc.
│   └── utils/, training/   # Dataset config, experiment helpers
├── paper/                  # LaTeX source, figures, auto-generated tables
├── project_instruction/    # Chinese phase docs (Stage 0–5), aligned with paper/code
├── docs/                   # Environment, preprocessing, Stage 1 user guide
├── config/                 # Dataset configuration files (dataset_paths.json, etc.)
├── manifests/              # Generated CSV manifests (train/val/test splits)
├── outputs/                # Training/eval outputs (logs, JSON/CSV, figures)
├── test/                   # Minimal unit tests for Stage 1/2
├── weights/                # Optional face detector weights (Caffe)
├── environment.yml         # Conda environment (Python 3.12 + PyTorch 2.1+)
└── dataset_paths*.json     # Example dataset path configs (no private paths)
```

Some secondary or historical files (e.g., logs, zip archives, early Overleaf bundles) are kept only as backups and are not required for the main pipeline; they can be ignored or pruned when preparing a submission.

---

## 3. Datasets and Task

### 3.1 Training / Validation Datasets

The final implementation trains and validates on four academic datasets:

- **CelebDF‑v2** – High‑quality celebrity face‑swap videos.  
- **FaceForensics++ (FF++)** – Multiple manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures, etc.).  
- **DFDC** – Crowd‑sourced dataset with diverse actors and compression pipelines.  
- **DeeperForensics‑1.0** – Emphasizes real‑world perturbations and challenging textures.

All video datasets are preprocessed into **256×256 face crops** using a unified pipeline (MTCNN face detection + resizing), and described by CSV manifests.

> **DF40** is used only to analyze the desired **256×256 PNG** specification in early planning.  
> It does **not** participate in the final training pipeline.

### 3.2 OOD Evaluation Dataset

- **Deepfake‑Eval‑2024** – In‑the‑wild benchmark composed of real social‑platform videos.  
  Used strictly as a **held‑out OOD benchmark** for Stage‑1/2 and cascade evaluation.

### 3.3 Task Definition

Detection is framed as **frame‑level binary classification**:

- Label `0` = real face, `1` = fake face.  
- For videos, applications may aggregate frame‑level predictions into a video‑level score (e.g., max or average probability), but the core pipeline operates on face crops.

---

## 4. Six‑Stage Pipeline (Summary)

This matches the paper’s Method section.

1. **Stage 0 – Data & Environment (project setup)**  
   - Configure datasets via `config/dataset_paths.json` (see `docs/dataset_configuration_guide.md`).  
   - Preprocess raw videos into 256×256 PNG face crops using `scripts/preprocess_datasets_v2.py`.  
   - Generate train/val/test manifests under `manifests/`.

2. **Stage 1 – Lightweight Filter (MobileNetV4)**  
   - Train with `src/stage1/train_stage1.py` on the combined manifest (default batch size 128).  
   - Apply light augmentation (resize, flip, colour jitter, small affine, blur), AdamW, cosine LR; select the best checkpoint by validation AUC.  
   - Calibrate probabilities via temperature scaling using `src/stage1/calibrate_model.py`, then evaluate per‑dataset and combined metrics (`outputs/stage1/evaluation/...`).

3. **Stage 2 – Expert Model (EfficientNetV2‑B3)**  
   - Train with `src/stage2/train_stage2_effnet.py` on the same combined manifest using EfficientNetV2‑B3, stronger augmentation (RandAugment + Mixup/CutMix), and Focal Loss with cosine‑annealing warm restarts (defaults match the paper’s hyperparameters).  
   - Optional hard‑example mining (HEM) ablations oversample a difficult subset derived from Stage‑1 scores; the released training script defaults to uniform sampling and corresponds to the main reported EfficientNetV2‑B3 results.  
   - Calibrate Stage‑2 probabilities and cache logits/embeddings for cascade tuning and optional meta‑model experiments.

4. **Stage 3 – Meta‑Model (optional)**  
   - Create meta‑dataset from Stage‑2 (and optional GenConViT) features using `src/stage3/create_meta_dataset.py`.  
   - Train LightGBM meta‑model with `src/stage3/train_meta_model.py`.  
   - In current experiments, Stage‑3 slightly underperforms the best Stage‑2 expert,  
     so it is **not** used in the default deployment, but remains a research tool.

5. **Stage 4 – Cascade Threshold Tuning**  
   - Use cached logits and labels on the combined validation set.  
   - Run grid search over $(\tau_{\mathrm{low}},\tau_{\mathrm{high}})$ using scripts in `src/stage4/` / `src/tools/`.  
   - Select operating points that minimize FNR under Stage‑2 usage budgets (e.g., FNR≈0.006 with Stage‑2 rate≈1.16%).  
   - Implement the cascade logic in `src/stage4/cascade_detector.py` and evaluate via `benchmark_cascade.py`.

6. **Stage 5 – Robustness & Cross‑Dataset Evaluation**  
   - Evaluate Stage‑1/2 and the cascade on test splits of all datasets and on Deepfake‑Eval‑2024.  
   - Use `src/tools/analyze_robustness.py` and `src/tools/robustness_threshold_sweep.py` to generate robustness tables/plots (JPEG, noise, blur, brightness, etc.).  
   - Generated LaTeX tables and figures are written to `paper/generated/` and used directly in the paper.

7. **Stage 6 – Mobile Export (TorchScript + PTQ)**  
   - Export Stage‑1/2 to TorchScript, apply post‑training dynamic quantization on linear layers.  
   - Bundle quantized models, thresholds, and calibration parameters into a simple mobile bundle (see `src/stage4/optimize_for_mobile.py` and `src/stage4/mobile_deployment/`).

---

## 5. Minimum Usage Guide

> 這裡只提供一條「最少步驟」的路線圖，具體參數與詳細命令請參考 `docs/` 和 `project_instruction/`。

### 5.1 Environment

```bash
conda env create -f environment.yml
conda activate aware-net
```

### 5.2 Dataset Configuration & Preprocessing

1. Edit `config/dataset_paths.json` (or start from `dataset_paths_example.json`).  
2. Run preprocessing (example, adjust flags to your environment):

```bash
python scripts/preprocess_datasets_v2.py \
  --video-backend decord \
  --face-detector mtcnn \
  --workers 4
```

This populates `processed_data/` and `manifests/`.

### 5.3 Train & Evaluate Stage 1

```bash
python src/stage1/train_stage1.py --data_dir processed_data
python src/stage1/calibrate_model.py --data_dir processed_data
python src/stage1/evaluate_stage1.py --data_dir processed_data
```

### 5.4 Train Stage 2 and Tune Cascade

```bash
python src/stage2/train_stage2_effnet.py --data_dir processed_data
# optional: cascade benchmarking and mobile‑oriented analysis
python src/stage4/benchmark_cascade.py --benchmark_all
```

These commands update `outputs/` and `paper/generated/` so that the LaTeX paper reflects the latest experiments.

### 5.5 Android Demo App (optional)

- After training Stage‑1/2 and selecting cascade thresholds, you can export ONNX models and a cascade config for Android via:

  ```bash
  python scripts/export_mobile_cascade_onnx.py
  ```

  This populates `android/mobile_bundle/` with ONNX models and JSON configs. For how to load these into the sample app and run on‑device evaluation, see `android/README.md`.

---

## 6. How This Repository Relates to the Paper

- The **paper text** lives in `paper/sections/*.tex`.  
- All **tables/figures** that summarize metrics are generated from `outputs/` via scripts in `src/tools/` and written into `paper/generated/*.tex`.  
- The **Chinese project_instruction documents** (`project_instruction/階段*.md`) explain the same pipeline stage‑by‑stage and are now kept consistent with the paper and code.

If you update experiments (e.g., retrain Stage‑2 or re‑tune the cascade), run the appropriate tools in `src/tools/` to regenerate `paper/generated/` before re‑compiling the paper.

---

## 7. Notes for Organizing and Submitting the Project

When submitting this project together with the paper:

- **Must keep**: `src/`, `paper/`, `project_instruction/`, `docs/`, `config/`, `manifests/`, `environment.yml`, and the key `outputs/` subdirectories (at least those runs that back the current paper tables).  
- **Recommended to keep**: `test/` (unit tests) and `dataset_paths_example.json` (example dataset configuration).  
- **Optional**: `weights/` (Caffe face detector) and larger `outputs/` subdirectories, depending on storage constraints.  
- **Safe to compress/remove**: outdated logs, zip bundles, and early Overleaf exports, as long as they are no longer referenced by scripts.

For your own cloud backups, a suggested priority is:

1. `paper/` (final LaTeX + PDF)  
2. Key `outputs/` (runs that correspond to the paper’s tables)  
3. `manifests/` (dataset splits)  
4. `config/dataset_paths.json` / `dataset_paths_example.json`

---

For more detailed documentation (in Chinese) and the development history of the project, refer to the stage documents under `project_instruction/` and the `Midterm Progress Report.md`.  
Together with this README, they provide a complete narrative from initial research concept to the final implementation.

---

## 8. Implementation Notes & Alignment with the Paper

This section summarizes a few potentially confusing details that are now explicitly aligned between the paper and the codebase, to make future reading and maintenance easier.

- **Stage‑1 probability semantics and cascade thresholds**  
  - Stage‑1 (MobileNetV4) outputs a single logit which, after `sigmoid`, is interpreted as **P(fake)**, matching the paper’s definition $p_1(x)=\Pr(y=1\mid x)$.  
  - In `src/stage4/cascade_detector.py`, `CascadeConfig.stage1_real_threshold` corresponds to $\tau_{\mathrm{low}}$ in the paper (default `0.05`), and `stage1_fake_threshold` corresponds to $\tau_{\mathrm{high}}$ (default `0.55`). The decision rule is:
    - `p1(x) < tau_low` → predict **real**, with confidence approximately `1 - p1(x)`;  
    - `p1(x) > tau_high` → predict **fake**, with confidence approximately `p1(x)`;  
    - values in between → escalate to Stage‑2/3.  
  - Early internal versions briefly treated the output as P(real), which inverted the Stage‑1 decision logic; the current implementation has been corrected to match the paper’s formulation.

- **Role of the Stage‑3 meta‑model**  
  - The LightGBM meta‑model and optional GenConViT expert under `src/stage3/` are primarily **research and offline analysis components**.  
  - The “default deployment” and mobile export pipeline described in both the paper and this README are a **two‑stage cascade** (Stage‑1 + Stage‑2); Stage‑3 is **not** part of the Android/on‑device runtime.  
  - `src/stage4/cascade_detector.py` implements a three‑stage Python prototype `CascadeDetector` for desktop‑side analysis and benchmarking across Stage‑1→2→3; this does not imply Stage‑3 must be executed on mobile.

- **Dynamic / adaptive thresholds**  
  - The configuration field `CascadeConfig.enable_dynamic_thresholds` provides a prototype hook for dynamic thresholds based on an `uncertainty_score`, intended for experimenting with alternative routing policies.  
  - This option is currently `False` by default, and all experiments/tables in the paper use **static thresholds** $(\tau_{\mathrm{low}},\tau_{\mathrm{high}})$ obtained via the Stage‑4 grid search.  
  - To experiment with adaptive thresholds you would:
    - set `enable_dynamic_thresholds=True`, and  
    - supply a meaningful `uncertainty_score` when calling `CascadeDetector.predict(..., uncertainty_score=...)` (or the batch/video variants), e.g., derived from video complexity indicators.

- **Hard Example Mining (HEM)**  
  - The paper and Chinese documentation discuss how hard‑example mining affects Stage‑2, but the **public training script** `src/stage2/train_stage2_effnet.py` uses a simple `shuffle=True` loader, i.e., **uniform sampling**.  
  - HEM is only enabled in specific ablation experiments to construct a “difficult subset” and compare against the standard configuration. The main reported results and the released training script correspond to the “no‑HEM, uniform sampling” setting.  
  - If a dedicated HEM training script is added in the future, it should live alongside the existing script in `src/stage2/` and be clearly marked in the README as ablation‑only.

- **Use of DF40**  
  - As noted in Section 3.1, DF40 is used only to analyze the target image specification (256×256 PNG) and **does not participate in any Stage‑1/2/3 training or evaluation**.  
  - The preprocessing pipeline deliberately skips DF40 for re‑processing and treats it as a “format reference”; the final train/val/test manifests only use CelebDF‑v2, FF++, DFDC, DeeperForensics‑1.0, plus the OOD Deepfake‑Eval‑2024 benchmark.

The goal of these notes is to keep the mathematical definitions and experimental settings in the paper aligned with the actual implementation and default scripts in this repository, while clearly separating research components (Stage‑3, HEM, adaptive thresholds) from the final deployment path. If you notice any apparent mismatch while reading the paper or reproducing experiments, this section and the corresponding `paper/sections/*.tex` passages are the first places to cross‑check.  
