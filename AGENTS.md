# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src/`, split into numbered stages (`stage_00`…`stage_09`) that mirror the research roadmap; new components should sit inside the relevant stage package. Re-usable training and evaluation settings live in `configs/`, while experiment artifacts belong in `experiments/` and dataset manifests in `manifests/`. Utility scripts reside under `tools/` (`tools/setup` for environment automation, `tools/tests` for unit suites). Use the top-level `train.py` for interactive runs or stage-specific trainers such as `src/stage_00/train_baseline.py` when scripting pipelines.

## Build, Test, and Development Commands
```bash
conda env create -f environment.yml  # provision aware_net_rtx50 env
python tools/setup/setup_environment.py --validate-only  # verify deps & manifests
python src/stage_00/train_baseline.py --config configs/training.json --training quick_test  # 3-epoch smoke run
pytest tools/tests/  # execute Stage 0 regression suite
```
Execute commands from the repo root; prefer GPU runs when available but confirm CPU fallbacks via the quick test before large jobs.

## Coding Style & Naming Conventions
Python 3.10+ with 4-space indentation, type hints, and docstrings for public APIs. Run `black` (default 88-char line width), `flake8`, and `mypy` before submitting. Modules follow snake_case filenames that match their contained class or function groups; stage-level packages should mirror the roadmap naming (`stage_01_supcon`, `stage_02_experts`, etc.). Configuration files use lower-case hyphenated names and live in `configs/`.

## Testing Guidelines
Unit tests sit in `tools/tests` and follow the `test_<module>.py` naming pattern. Run `pytest tools/tests/ --cov=src --cov-report=term-missing` and keep coverage at or above the 90% target noted in Stage 00 docs. Mark expensive suites with `@pytest.mark.slow` and GPU-dependent cases with `@pytest.mark.gpu`; ensure both paths degrade gracefully on CPU-only hardware. Include fixtures for temporary data and manifest mocks to keep tests deterministic.

## Commit & Pull Request Guidelines
Git history uses Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, etc.) plus concise summaries; reference the impacted stage or subsystem when possible (e.g., `feat(stage_00): add sampler weights`). PRs should describe the motivation, list runnable commands and results (tests, training metrics, coverage), and link any related experiment folders or issues. Provide screenshots or table snippets when adding dashboards, and call out configuration changes affecting reproducibility.

## Environment & Data Notes
Large datasets are referenced via manifests only—store raw media outside the repository. Keep sensitive keys in your personal environment and use `.env.example` patterns if sharing defaults becomes necessary. When introducing new datasets, update `manifests/` and document balancing requirements inside `configs/` so downstream stages inherit consistent sampling behavior.
