Stage 02 – Difficult Sample Subset Creation

Status: 100% complete

Evidence (artifacts)
- Tool: `MobileDeepfakeDetection/src/tools/create_difficult_subset.py`
- Outputs: `MobileDeepfakeDetection/outputs/stage2/run_*/stage2_summary.json`, per‑dataset train preds CSVs
- Manifests: `MobileDeepfakeDetection/manifests/train_difficult_subset.csv`, `MobileDeepfakeDetection/manifests/train_stage3_mix.csv`

Instruction vs implementation
- Load Stage 1 best model and set eval(): Implemented (`_load_stage1_model`, `model.eval()`).
- Build non‑augmented DataLoader over full train splits: Implemented (CelebDFDataset with `augmentation=False`, `return_meta=True`).
- Collect logits → sigmoid → probability per sample: Implemented.
- Define difficulty by ambiguity range and misclassification: Implemented with configurable `[ambiguity_lower, ambiguity_upper]` and decision threshold.
- Save difficult subset manifest CSV: Implemented (`--out` → `manifests/train_difficult_subset.csv`).
- Optional: save per‑dataset preds, per‑dataset min/quota, mixed manifest for Stage 03: Implemented (`--save_intermediate_preds`, `--dataset_min`, `--dataset_quota`, `--mix_manifest`).

Differences/gaps
- None material. Implementation matches and extends the spec (mix manifest support and dataset caps).

Action items
- None required.
