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

Only最重要的目錄簡要說明如下（詳細內容見 `project_instruction/` 和 `docs/`）：

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

一些次要或歷史性文件（例如部分 log、zip 壓縮包、早期 Overleaf bundle）只作為備份存在，對主流程不是必需，可以在提交作業時選擇忽略。

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

- 必須保留：`src/`, `paper/`, `project_instruction/`, `docs/`, `config/`, `manifests/`, `environment.yml`, 關鍵 `outputs/`（至少包含目前論文所用的 run）。  
- 建議保留：`test/`（單元測試）、`dataset_paths_example.json`（示例配置）。  
- 可選保留：`weights/`（Caffe face detector）、較大的 `outputs/` 子目錄（視空間而定）。  
- 可以忽略或壓縮：舊的 log、zip bundle、早期 Overleaf 導出檔，只要已不再被腳本引用。

對於你自己要備份到雲端的內容，優先順序建議是：

1. `paper/`（最終 LaTeX + PDF）  
2. 關鍵 `outputs/`（對應論文表格的 run）  
3. `manifests/`（資料切分）  
4. `config/dataset_paths.json` / `dataset_paths_example.json`

---

如需更細節的中文說明與開發歷史，請參考 `project_instruction/` 下的各個階段文檔，以及 `Midterm Progress Report.md`。  
這些文檔加上本 README 構成了一條從研究構想到最終實作的完整故事線。  
