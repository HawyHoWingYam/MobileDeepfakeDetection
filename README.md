# AWARE-NET Framework Guide

Complete documentation for the AWARE-NET experiment management and visualization framework.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Phase 1: Core Architecture](#phase-1-core-architecture)
5. [Phase 2: Enhanced Evaluation](#phase-2-enhanced-evaluation)
6. [Phase 3: Visualization System](#phase-3-visualization-system)
7. [Phase 4: Run Comparison](#phase-4-run-comparison)
8. [Phase 5: Advanced Features](#phase-5-advanced-features)
9. [CLI Tools](#cli-tools)
10. [Best Practices](#best-practices)

## Overview

The AWARE-NET Framework is a comprehensive experiment management system designed to address three critical challenges in deep learning research:

### Problems Solved

1. **God Script Problem** ("上帝脚本")
   - Training scripts accumulating too many responsibilities
   - Solution: ExperimentManager centralizes all non-training concerns

2. **File Ocean Problem** ("文件海洋")
   - 13+ scattered artifact files causing management complexity
   - Solution: evaluation_summary.json consolidates all evaluation data

3. **Tracker Abstraction Timing**
   - Flexible switching between local and cloud-based experiment tracking
   - Solution: Unified tracker interface with LocalTracker and WandbTracker

## Architecture

### Core Components

```
AWARE-NET Framework
├── Phase 1: Core Architecture
│   ├── ExperimentManager (experiment_manager.py)
│   ├── Tracker Abstraction (tracker.py)
│   ├── Configuration System (experiment_default.yaml)
│   └── Config Utilities (experiment_framework.py)
├── Phase 2: Enhanced Evaluation
│   ├── ModelEvaluator Extensions (evaluation.py)
│   └── PredictionsCollector (evaluation.py)
├── Phase 3: Visualization
│   ├── Plotting Module (plotting.py)
│   └── Visualization CLI (visualize_run.py)
├── Phase 4: Run Comparison
│   └── Comparison Tool (compare_runs.py)
└── Phase 5: Advanced Features
    ├── WandbTracker (tracker.py)
    └── Documentation (this file)
```

### Artifact Structure

```
outputs/stage1/run_20251020_162308/
├── artifacts/
│   ├── metrics.csv              # Training metrics (CSV format)
│   ├── evaluation_summary.json   # Consolidated evaluation data
│   └── predictions.json         # Detailed predictions (optional)
├── figures/                      # Generated visualizations
│   ├── 01_metrics_summary.png
│   ├── 02_confusion_matrix.png
│   ├── 03_class_distribution.png
│   ├── 04_threshold_analysis.png
│   └── report.html             # Interactive HTML report
├── logs/                         # Training logs
│   └── log.md                   # Markdown formatted logs
├── config_snapshot.yaml         # Complete configuration snapshot
├── meta.json                    # Run metadata (git, timestamp, etc.)
└── best_model.pth              # Best model checkpoint
```

## Quick Start

### 1. Basic Training with Configuration

```bash
# Using default configuration
python src/training/train_mobilenet.py

# With custom configuration
python src/training/train_mobilenet.py \
    --config configs/experiment_default.yaml \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 100
```

### 2. Generate Visualizations

```bash
# Visualize a training run
python src/tools/visualize_run.py outputs/stage1/run_20251020_162308

# Generate HTML report
python src/tools/visualize_run.py outputs/stage1/run_20251020_162308 \
    --format html \
    --output-dir reports/
```

### 3. Compare Multiple Runs

```bash
# Compare runs side-by-side
python src/tools/compare_runs.py \
    outputs/stage1/run_20251020_162308 \
    outputs/stage1/run_20251020_165000 \
    outputs/stage1/run_20251021_100000

# Export comparison report
python src/tools/compare_runs.py \
    run1 run2 run3 \
    --format json \
    --output-dir comparisons/
```

## Phase 1: Core Architecture

### ExperimentManager

Central coordinator for all experiment management concerns.

#### Key Features

- **Configuration Management**: Load YAML/JSON configs with CLI overrides
- **Metadata Collection**: Automatic git commit, branch, and hostname tracking
- **Artifact Saving**: Configurable save policies (on_best, periodic, retention)
- **Metrics Logging**: CSV-based metrics tracking
- **Tracker Integration**: Unified interface for local/cloud tracking

#### Usage

```python
from utils.experiment_manager import ExperimentManager
from utils.experiment_framework import load_config

# Load configuration
config = load_config("configs/experiment_default.yaml")

# Create experiment manager
manager = ExperimentManager(
    base_config=config,
    cli_args={},
    output_dir="outputs/stage1",
    experiment_name="my_experiment"
)

# Log metrics during training
manager.log_epoch_metrics(epoch=0, metrics_dict={
    'loss': 0.5,
    'auc': 0.85,
    'f1': 0.82
}, split='train')

# Check if should save artifacts
if manager.should_save_artifacts(epoch=0, metrics={'auc': 0.85}, is_best=True):
    manager.save_evaluation_summary(epoch=0, eval_summary={...})

# Finalize experiment
manager.finalize()
```

### Configuration System

#### YAML Structure

```yaml
model:
  name: mobilenetv4_hybrid_medium
  dropout: 0.2
  freeze_backbone: false

training:
  lr: 0.001
  batch_size: 32
  epochs: 20
  optimizer: adam

data:
  datasets: [celeb-df, faceforensics, deeperforensics, dfdc]
  image_size: 224

save_policy:
  on_best: f1           # Save when F1 improves
  every_n_epochs: 0     # Don't save periodically
  last_n_epochs: 2      # Keep last 2 epochs

early_stopping:
  patience: 5
  min_delta: 0.001
  restore_best: true

tracker:
  type: local           # or "wandb"
  project: aware-net
```

#### CLI Overrides

```bash
# Override nested keys with dot notation
python train_mobilenet.py \
    --config configs/experiment_default.yaml \
    --training.lr 0.0001 \
    --training.batch_size 64 \
    --model.dropout 0.3
```

### Tracker Abstraction

#### Local Tracker (Default)

No-op adapter. Actual logging handled by other components:

```python
from utils.tracker import LocalTracker

tracker = LocalTracker(run_dir="outputs/stage1/run_xxx")
tracker.log_metrics({'auc': 0.92}, step=0)     # No-op
tracker.log_config({'lr': 0.001})              # No-op
```

#### WandB Tracker (Optional)

Requires: `pip install wandb`

```python
from utils.tracker import WandbTracker

tracker = WandbTracker(
    run_dir="outputs/stage1/run_xxx",
    project="aware-net-stage1",
    name="my_experiment",
    config={'lr': 0.001}
)

tracker.log_metrics({'auc': 0.92}, step=0)     # Logs to WandB
tracker.log_artifact('best_model.pth')         # Uploads to WandB
tracker.finish()                               # Finalize run
```

## Phase 2: Enhanced Evaluation

### Full Evaluation

Comprehensive evaluation generating `evaluation_summary.json`:

```python
from utils.evaluation import ModelEvaluator

evaluator = ModelEvaluator(device='cuda')

# Generate full evaluation summary
summary = evaluator.full_evaluation(
    model=model,
    data_loader=val_loader,
    criterion=loss_fn,
    mode='validation',
    thresholds=[0.3, 0.5, 0.7]
)

# summary contains:
# - metrics (auc, f1, accuracy, precision, recall, specificity, fnr)
# - confusion_matrix
# - class_distribution
# - probability_statistics
# - threshold_analysis
# - classification_report
```

### PredictionsCollector

Track predictions across multiple datasets:

```python
from utils.evaluation import PredictionsCollector

collector = PredictionsCollector()

# Add predictions from each dataset
for dataset_name in ['celeb-df', 'faceforensics']:
    predictions = model(batch_images)
    targets = batch_labels

    collector.add_batch(
        predictions=predictions,
        targets=targets,
        probabilities=torch.sigmoid(predictions),
        dataset_name=dataset_name
    )

# Get per-dataset metrics
per_dataset = collector.get_metrics_by_dataset()
for dataset, metrics in per_dataset.items():
    print(f"{dataset}: AUC={metrics['auc']:.4f}")

# Export to JSON
data = collector.to_dict()
```

## Phase 3: Visualization System

### Plotting Functions

11 comprehensive visualization functions:

1. **plot_learning_curves** - Training/validation curves
2. **plot_confusion_matrix** - Classification confusion matrix
3. **plot_roc_curve** - ROC curve with AUC
4. **plot_precision_recall_curve** - PR curve
5. **plot_probability_distribution** - Probability histograms
6. **plot_threshold_analysis** - Metrics vs threshold
7. **plot_per_dataset_metrics** - Cross-dataset comparison
8. **plot_class_distribution** - Real/fake split pie chart
9. **plot_calibration_curve** - Model calibration
10. **plot_error_analysis** - FP/FN analysis
11. **plot_metrics_summary** - All metrics bar chart

#### Usage

```python
from utils.plotting import plot_learning_curves

fig = plot_learning_curves(
    train_loss=train_losses,
    val_loss=val_losses,
    train_auc=train_aucs,
    val_auc=val_aucs,
    train_f1=train_f1_scores,
    val_f1=val_f1_scores,
    train_accuracy=train_accuracy,
    val_accuracy=val_accuracy,
    train_precision=train_precision,
    val_precision=val_precision,
    train_recall=train_recall,
    val_recall=val_recall,
    train_specificity=train_specificity,
    val_specificity=val_specificity,
    train_fnr=train_fnr,
    val_fnr=val_fnr,
    output_path='figures/learning_curves.png',
    best_epoch=15,
    include_loss=False,
    include_auc=False
)
```

### Visualization CLI

#### Generate Visualizations

```bash
python src/tools/visualize_run.py \
    outputs/stage1/run_20251020_162308 \
    --output-dir visualizations/ \
    --format both              # plots, html, or both
    --dpi 150
```

#### HTML Report Generation

Creates interactive HTML report with:
- Embedded PNG plots
- Metric cards showing key values
- Professional CSS styling
- Run metadata display

## Phase 4: Run Comparison

### Compare Runs

```bash
python src/tools/compare_runs.py \
    outputs/stage1/run_20251020_162308 \
    outputs/stage1/run_20251020_165000 \
    outputs/stage1/run_20251021_100000 \
    --metric auc                    # Metric to rank by
    --format json                   # json, csv, or print
    --output-dir comparisons/
```

### Comparison Output

Generates comprehensive report with:
- Side-by-side metrics comparison
- Statistical summary (mean, std, min, max)
- Top 3 runs by each metric
- Best run identification

## Phase 5: Advanced Features

### WandbTracker Integration

Enable cloud-based experiment tracking:

```bash
# In configs/experiment_default.yaml
tracker:
  type: wandb
  project: aware-net-stage1
  entity: your-team-name

# Then training automatically logs to W&B
python train_mobilenet.py --config configs/experiment_default.yaml
```

## CLI Tools

### visualize_run.py

```bash
python src/tools/visualize_run.py <run_dir> [options]

Options:
  --output-dir DIR     Output directory for visualizations
  --format FMT        Output format: plots, html, both
  --dpi NUM           DPI for saved figures (default: 150)
  --show-plots        Display plots interactively
```

### compare_runs.py

```bash
python src/tools/compare_runs.py <run1> <run2> [run3...] [options]

Options:
  --output-dir DIR     Output directory for report
  --format FMT        Output format: json, csv, print
  --metric METRIC     Metric for ranking (default: auc)
```

### download_hf_dataset.py

```bash
python src/tools/download_hf_dataset.py \
    --dataset-id nuriachandra/Deepfake-Eval-2024 \
    --split train \
    --output-dir data/deepfake_eval/train \
    --max-samples 5000

Options:
  --config NAME         Dataset config name (if multiple configs)
  --image-column COL    Image column (default: image)
  --label-column COL    Label column (default: label)
  --token TOKEN         HuggingFace token (falls back to HF_TOKEN env var)
  --max-samples N       Limit number of samples for smoke tests
  --compression {png,jpg}  Image format for export (default: png)
  --skip-existing       Skip downloads if files already exist
  --metadata PATH       Optional JSON output capturing raw label metadata
```

The script exports all images into `<output-dir>/images/` grouped by encoded label
and produces a `manifest.csv` compatible with `CelebDFDataset`.  Use the resulting
manifest as a validation split by pointing Stage 03 evaluation to the saved CSV.

## Best Practices

### 1. Configuration Management

- Store all hyperparameters in YAML
- Use CLI overrides only for quick experiments
- Commit config snapshots to version control
- Review config_snapshot.yaml after runs

### 2. Experiment Naming

- Use descriptive experiment names
- Include model variant in name
- Example: `mobilenetv4_simple_celeb_df_only`

### 3. Artifact Management

- Use `save_policy` to control what gets saved
- Set `on_best` to save only improved models
- Use `last_n_epochs` to limit disk usage

### 4. Metrics Logging

- Log all relevant metrics every epoch
- Include both train and validation metrics
- Use consistent metric names across experiments

### 5. Visualization

- Generate HTML reports for quick review
- Compare runs when validating improvements
- Identify best hyperparameters using comparison tool

### 6. Reproducibility

- Always set seed in config
- Track git commit in metadata
- Save full config snapshot
- Document any manual changes

## Example Workflow

### Step 1: Prepare Configuration

```bash
# Edit configs/experiment_default.yaml with desired hyperparameters
nano configs/experiment_default.yaml
```

### Step 2: Run Training

```bash
python src/training/train_mobilenet.py \
    --config configs/experiment_default.yaml \
    --output_dir outputs/stage1
```

### Step 3: Generate Visualizations

```bash
python src/tools/visualize_run.py \
    outputs/stage1/run_20251020_162308 \
    --format html \
    --output-dir reports/
```

### Step 4: Compare with Previous Runs

```bash
python src/tools/compare_runs.py \
    outputs/stage1/run_20251020_162308 \
    outputs/stage1/run_20251020_165000 \
    --format print
```

### Step 5: Analyze Results

- Open `reports/report.html` in browser
- Review comparison metrics
- Identify improvements/regressions
- Decide on next experiment parameters

## Troubleshooting

### Q: "evaluation_summary.json not found"

A: Make sure you ran evaluation with `full_evaluation()` method:
```python
summary = evaluator.full_evaluation(model, data_loader, mode='validation')
manager.save_evaluation_summary(epoch, summary)
```

### Q: "WandbTracker: wandb not installed"

A: Install wandb:
```bash
pip install wandb
```

Then login:
```bash
wandb login
```

### Q: "Config file not found"

A: Verify path is correct:
```bash
ls -la configs/experiment_default.yaml
```

## Performance Notes

- **Metrics CSV**: ~10KB per epoch (1000 samples)
- **evaluation_summary.json**: ~50KB per epoch
- **PNG plots**: 100-300KB per plot
- **HTML report**: ~1-2MB with embedded plots

## References

- Stage 00 Documentation: See `project_instruction/stage/stage_00.md`
- Framework Design: See `project_instruction/implementation/implementation_plan.md`
- Configuration Schema: See `configs/experiment_default.yaml` comments

---

**Version**: 1.0
**Last Updated**: 2025-10-20
**Framework Version**: Phase 5 Complete
