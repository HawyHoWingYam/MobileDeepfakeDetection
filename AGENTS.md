# Repository Guidelines

## Project Structure & Module Organization
- Core code: `MobileDeepfakeDetection/src`
  - Models: `src/models/*` (e.g., `mobilenetv4_model.py`, `efficientnetv2_model.py`)
  - Training: `src/training/*` (primary entrypoint: `train_mobilenet.py`)
  - Utilities: `src/utils/*` (experiment logging, evaluation)
  - Tools: `src/tools/generate_manifests.py` (CSV manifest helper)
- Data manifests: `MobileDeepfakeDetection/manifests/*.csv` (balanced train/val/test)
- Dataset config: `MobileDeepfakeDetection/configs/datasets.json`
- Outputs: `MobileDeepfakeDetection/outputs/<stage>/run_YYYYMMDD_HHMMSS/`

## Build, Test, and Development Commands
- Create venv (example): `python -m venv .venv && source .venv/bin/activate`
- Install deps (example): `pip install torch timm albumentations opencv-python pandas scikit-learn matplotlib pillow tqdm tensorboard`
- Train (stage 01, multi‑dataset):
  `python MobileDeepfakeDetection/src/training/train_mobilenet.py --epochs 5 --batch_size 32 --output_dir MobileDeepfakeDetection/outputs/stage1`
- View logs: `tensorboard --logdir MobileDeepfakeDetection/outputs/stage1`
- Update manifests (if datasets change): `python MobileDeepfakeDetection/src/tools/generate_manifests.py`

## Coding Style & Naming Conventions
- Python 3.10+; PEP 8; 4‑space indent; max line length ~100.
- Modules/files: `snake_case.py`; classes: `CamelCase`; functions/vars: `snake_case`.
- Prefer type hints and docstrings; avoid `print`—use module loggers (`logging.getLogger(__name__)`).
- Keep imports explicit (no wildcard). Place `src` on path as done in training scripts when needed.

## Testing Guidelines
- No formal test suite yet. Add lightweight checks:
  - Dataset smoke test: load first batch from each manifest.
  - Determinism: set seed via `utils.experiment_framework.setup_reproducible_environment` and run 1–2 epochs.
  - Evaluation: use `utils.evaluation.ModelEvaluator` on val loader.

## Commit & Pull Request Guidelines
- Follow Conventional Commits: `feat|fix|docs|refactor|chore(scope): short summary` (e.g., `feat(stage_01): balance multi‑dataset sampling`).
- PRs should include: clear description, rationale, config changes (`configs/datasets.json`), sample command to reproduce, and screenshots/metrics (TensorBoard or AUC/F1) when applicable.

## Security & Configuration Tips
- Manifests: keep labels in CSV columns; avoid leaking labels in file paths. See `src/training/dataset.py` for leakage checks.
- Edit datasets via `configs/datasets.json`; paths default to `MobileDeepfakeDetection/` root.
- Large artifacts: write to `outputs/` only; don’t commit generated checkpoints or event files.
