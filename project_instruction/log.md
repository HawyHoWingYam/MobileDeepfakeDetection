Here’s a concrete, end‑to‑end plan to implement and run Stage 4 threshold tuning for the two‑stage cascade (MobileNetV4 → EfficientNetV2).

  Scope & Goals

  - Tune low/high confidence thresholds for Stage 1 (MobileNetV4) so only ambiguous samples escalate to Stage 2 (EfficientNetV2).
  - Optimize for minimal FNR, subject to accuracy/F1 constraints; report metrics, stage2 usage, and latency estimates.
  - Persist metrics, plots, selected thresholds, and a concise run README under outputs/stage4/run_<timestamp>/.

  Prerequisite Inventory

  - Checkpoints
      - Confirm Stage 1 best checkpoint path and training settings: outputs/stage1/**/checkpoint_best.pt; note if EMA or custom heads were used.
      - Confirm Stage 3 best checkpoint path and input config: outputs/stage3/**/checkpoint_best.pt; verify input size and normalization.
  - Manifests
      - Open manifests/val_manifest.csv; verify schema matches src/training/dataset.py (columns like path,label), paths resolvable (absolute or
        repo‑relative), and near‑balanced labels.
  - Device & batch
      - Record CUDA availability and prior default batch size/num_workers used stably on your machine; reuse for Stage 4 if feasible.
  - Transforms
      - From model factories in src/models/, confirm each model’s expected input (H×W, normalization mean/std, interpolation).

  Script Scaffolding (src/tools/tune_cascade_system.py)

  - Entrypoint layout
      - parse_args() → setup logger/seed → resolve output run dir → build dataloader → load models → precompute/cache (optional) → grid search →
        save results/plots → write README.
  - Logging/seed
      - Use logging.getLogger(__name__), INFO default, DEBUG optional; call utils.experiment_framework.setup_reproducible_environment(seed).
  - Pathing
      - Add repo root to sys.path only if import of src.* fails; mirror pattern used in src/tools/generate_manifests.py.

  CLI Interface

  - Required
      - --stage1-ckpt, --stage2-ckpt, --manifest, --output-dir (default MobileDeepfakeDetection/outputs/stage4)
  - Threshold search
      - --low-start 0.05 --low-stop 0.45 --high-start 0.55 --high-stop 0.95 --step 0.05
      - --primary-metric fnr --min-accuracy 0.9 --min-f1 0.9 (constraints configurable)
  - Inference & data
      - --batch-size 64 --device cuda:0 --num-workers 4 --pin-memory --amp
      - --stage1-input-size 256 --stage2-input-size 384 (if not inferrable from checkpoints)
      - --stage1-mean 0.5 0.5 0.5 --stage1-std 0.5 0.5 0.5 and analogous for stage2 (override if needed)
      - --seed 1337 --save-probs --save-cm --save-roc
  - Performance/levers
      - --precompute-stage1 (default true), --precompute-stage2 (default true), --estimate-latency (measure per-sample times once, then
        compute).
      - --escalation-class real|fake (if you want confidence thresholds applied to the probability of the “fake” class; default “fake”).

  Example:
  python MobileDeepfakeDetection/src/tools/tune_cascade_system.py \
  --stage1-ckpt outputs/stage1/run_YYYY.../checkpoint_best.pt \
  --stage2-ckpt outputs/stage3/run_YYYY.../checkpoint_best.pt \
  --manifest MobileDeepfakeDetection/manifests/val_manifest.csv \
  --output-dir MobileDeepfakeDetection/outputs/stage4 \
  --low-start 0.05 --low-stop 0.45 --high-start 0.55 --high-stop 0.95 --step 0.05 \
  --batch-size 64 --device cuda:0 --num-workers 4 --pin-memory --amp \
  --min-accuracy 0.9 --min-f1 0.9 --primary-metric fnr

  CascadeDetector Design

  - Responsibilities
      - Load models from src/models/* factories with checkpoints; eval(), torch.no_grad().
      - Maintain two transform pipelines (stage1, stage2) consistent with each model config.
      - Provide predict_batch(samples) returning:
          - stage1_logits (or probs), stage1_confidence (per sample), escalate_mask (low < p < high), stage2_logits for escalated ones.
  - Caching
      - Precompute on the full val set (recommended for grid search scale):
          - stage1_logits_all [N, C] on CPU; optionally stage2_logits_all [N, C].
          - Memory estimate: for N=10k, C=2, float32 ≈ 80 KB per 1k samples; safe to store; consider float16 if needed.
      - Alternative lazy mode: compute stage2 only for unique escalations encountered; memoize per index.
  - Confidence and temperature
      - Allow --temperature (default 1.0) and softmax on the target class to compute confidence p_fake (or p_real if configured).
      - Ensure consistent handling if heads output logits; avoid double‑softmax.

  Minimal sketch:
  class CascadeDetector:
  def init(...):
  self.stage1 = build_mobilenetv4(...); self.stage2 = build_efficientnetv2(...)
  self.tr1 = build_transform(stage1_config); self.tr2 = build_transform(stage2_config)
  self.cache1 = None; self.cache2 = None
  def confidence(self, logits, temp=1.0, target=1):
  probs = torch.softmax(logits / temp, dim=1)
  return probs[:, target]
  def precompute(self, dataloader):
  # Fill cache1 (and cache2 if requested)
  def predict_batch(self, images_or_indices, thresholds):
  # Use cache if indices provided; otherwise run transforms+models
  # Return stage1 logits, escalate mask, stage2 logits for escalations

  Validation Data Pipeline

  - Build a val dataset with no augmentations, shuffle=False, deterministic transforms.
      - Prefer using src/training/dataset.py dataset class; pass a no‑aug eval transform for stage1 precompute.
      - If dataset can return raw image paths, apply stage‑specific transforms inside the detector; otherwise, create two eval datasets (one per
        model) sharing ordering and indices.
  - Dataloader settings
      - batch_size from CLI; num_workers tuned to disk; pin_memory=True if CUDA; persistent_workers=True.
      - AMP (--amp) with torch.autocast(device_type) for speed; ensure numerical equivalence for metrics.
  - Transform fidelity
      - Match each model’s resize, crop, interpolation, mean/std; ensure channel order (RGB) and dtype are correct.

  Threshold Grid Search

  - Search space
      - Iterate low in [low_start, low_stop] and high in [high_start, high_stop] with step; enforce low < high each pair.
  - Evaluation loop
      - Use cached stage1_logits_all to compute p per sample; split indices:
          - accept_fake = p >= high; accept_real = p <= low; ambiguous = (p > low) & (p < high).
      - For ambiguous, get stage2_logits from cache or run batched inference via stage2 transforms; combine decisions:
          - Final pred = stage1 for confident, stage2 for ambiguous.
      - Compute metrics using utils.evaluation.ModelEvaluator if available; else fallback to scikit‑learn:
          - ROC‑AUC (using final soft scores), accuracy, precision, recall, F1, FNR, confusion matrix.
      - Track per‑pair stats:
          - thresholds, metrics, escalation_rate = len(ambiguous)/N, estimated latency, counts (TP, FP, TN, FN).
  - Latency estimation
      - Measure once: average per‑sample time for stage1 and stage2 (t1_ps, t2_ps) using warmup+timed batches.
      - Estimate: latency = N*t1_ps + N*escalation_rate*t2_ps; also include empirical variance if possible.

  Pseudocode core:
  for low in lows:
  for high in highs:
  if low >= high: continue
  p = confidence(stage1_logits_all)
  accept_real = p <= low; accept_fake = p >= high; ambig = ~(accept_real | accept_fake)
  preds = np.empty(N); scores = np.empty(N)
  preds[accept_real] = 0; scores[accept_real] = p[accept_real]
  preds[accept_fake] = 1; scores[accept_fake] = p[accept_fake]
  s2_idx = np.where(ambig)[0]
  if precompute_stage2:
  s2_logits = stage2_logits_all[s2_idx]
  else:
  s2_logits = run_stage2_inference(s2_idx)
  s2_p = softmax(s2_logits)[:, target_class]
  preds[s2_idx] = (s2_p >= 0.5).astype(int)
  scores[s2_idx] = s2_p
  metrics = evaluate(y_true, preds, scores)
  table.append({... thresholds, metrics, escalation_rate, latency ...})

  Result Persistence & Logging

  - Run directory
      - Create outputs/stage4/run_YYYYMMDD_HHMMSS/ with subfiles:
          - metrics_grid.csv and metrics_grid.json (all pairs)
          - best_config.json (selected thresholds, constraints, metrics)
          - confusion_matrix.png, roc_curve.png (for best)
          - Heatmaps: heatmap_f1.png, heatmap_fnr.png (axes low/high)
          - Optional: scores_stage1.npy, scores_stage2.npy if --save-probs
  - Selection policy
      - Filter grid by accuracy >= min_accuracy and f1 >= min_f1.
      - Choose minimal FNR; tie‑break by higher F1 then lower latency then lower escalation rate.
  - Logging
      - Log summary lines: best thresholds, metrics, escalation %, estimated latency, ckpt ids, dataset size.

  Finalize Defaults & Documentation

  - Defaults in code
      - If a best config exists, write the chosen low/high into CascadeDetector defaults (but keep CLI overrides).
      - Expose a JSON load path --defaults-from to reuse thresholds later.
  - Run README
      - Write README.txt in the run dir with:
          - checkpoints used (relative paths), dataset manifest, device/batch/AMP, search grid, constraints, selection rule.
          - summary metrics and plots list; exact command line for reproducibility.
  - Repo docs
      - Add a short section to the main README or a docs snippet:
          - prerequisites (checkpoints), how to run tuning, how to interpret thresholds, and how to integrate into inference.

  Implementation Notes

  - Model loading
      - Use existing factories in src/models/* so head shapes and preprocessing match; pass checkpoint and config inferred from the run dir or
        an embedded hparams file if present.
  - Dataset alignment
      - Ensure consistent sample ordering across any dual pipelines; keep a single source of truth list of file paths and labels; index
        everywhere by that order.
  - Mixed precision
      - Safe for inference; wrap both stages with autocast and keep caches in float32 for stability unless memory is tight.
  - Numerics
      - Use logits→softmax once; avoid converting to probs then re‑softmax; temperature should divide logits directly.

  Validation & Smoke Tests

  - Dataset smoke test
      - Load first batch and verify shapes after each transform (stage1, stage2).
  - Determinism
      - Fix seed; run with --step 0.2 to validate end‑to‑end grid and ensure stable metrics runs twice.
  - Metric parity
      - Cross‑check ModelEvaluator vs scikit‑learn on a sample to confirm consistency.
  - Cache integrity
      - Turn off --precompute-stage2 and compare best config/metrics to the precompute variant; tolerances within rounding.

  Risk/Assumptions

  - If model factories don’t expose canonical transforms, add a small helper in the tool to construct them from hparams or use standard ImageNet
    eval transforms with model‑specific sizes.
  - If dataset.py can’t return raw images, define a tiny manifest dataset in the tool for inference‑only use while keeping ordering stable.
  - Checkpoint compatibility: if heads differ (num classes), assert num_classes == 2 and raise with a clear message.

  Work Breakdown (suggested)

  - Day 1: Scaffolding, CLI, logging, seeding, dataloader integration; load both checkpoints; print configs.
  - Day 2: CascadeDetector with transforms, AMP, and stage1 precompute; baseline single pair eval.
  - Day 3: Grid search with caching; selection logic; metrics table; latency estimation.
  - Day 4: Persistence to CSV/JSON; heatmaps and plots; README writer; defaults update.
  - Day 5: Polishing, smoke tests, docs update; prepare example command.

  Next Steps After Tuning

  - Integrate thresholds into the inference pipeline (e.g., a src/tools/cascade_infer.py script).
  - Add a smoke test that asserts cascade pass/fail behavior around the learned thresholds on 10–20 samples.
  - Optionally add a small benchmark script to validate escalation rate and latency on your deployment hardware.

  If you want, I can generate the script skeleton now with the argparse/structure and stubs for the core functions, and we can fill in model/
  dataloader specifics to match your repo utilities.

  Great work shipping Stage 4. I did a detailed pass across code, docs, and repo structure. Below are findings ordered by severity with
  file:line references, followed by a few open questions and suggested fixes.

  Findings

  Critical

  - Stage-specific preprocessing not applied; Stage 2 gets Stage 1–sized, non-ImageNet-normalized tensors
      - src/tools/tune_cascade_system.py:116,117 define transforms but they’re never used during inference/caching.
      - src/tools/tune_cascade_system.py:831–834 val dataset is created with normalize=False, so images are only scaled to [0,1] (no ImageNet
        mean/std).
      - src/tools/tune_cascade_system.py:219, 234 both stages run directly on the same tensor batch; Stage 2 receives 256×256 instead of 384×384
        and no ImageNet normalization. This will degrade Stage 2 performance and invalidate threshold search metrics.

  High

  - Can’t disable Stage 2 precompute; lazy escalation not implemented
      - src/tools/tune_cascade_system.py:304 raises if Stage 2 logits were not precomputed. CLI has --precompute-stage2 but there’s no way to
        set it to False (see next bullet), and “lazy evaluation” for escalated samples is missing.
  - CLI flags use action='store_true' with default=True, making them impossible to disable
      - src/tools/tune_cascade_system.py:788, 790, 806, 808 set default=True for --pin-memory, --amp, --precompute-stage2, --measure-latency.
        You can’t turn these off from the CLI. Typical pattern is default False.
  - Device default may fail on CPU-only machines
      - src/tools/tune_cascade_system.py:784, 815 default device='cuda:0' and no fallback. Will error if CUDA isn’t available.
  - y_true extraction re-reads/augments dataset, hurting determinism and performance
      - src/tools/tune_cascade_system.py:864 iterates the dataset to build labels. That re-runs transforms and I/O. It’s cheaper and safer to
        read labels from dataset.df['label'].

  Medium

  - Unused imports and light inconsistencies
      - src/tools/tune_cascade_system.py: top import of plot_roc_curve, plot_confusion_matrix but they’re not used. ModelEvaluator is
        constructed but the code relies on sklearn metrics; either remove the dependency or use it.
  - Hard-coded model variants; no CLI override
      - src/tools/tune_cascade_system.py:103–121 _load_model hard-codes 'mobilenetv4_hybrid_medium' and 'tf_efficientnetv2_b0'. If your Stage 3
        best run used a different EfficientNetV2 variant (e.g., rw_t or b3), results may be off. Provide CLI knobs or load from saved hparams.
  - README (run) shows “Total samples” incorrectly
      - src/tools/tune_cascade_system.py:709–735 the READMEs “Total samples” is derived from results_df; that’s the number of threshold rows,
        not dataset size.
  - Heatmap pivot axes are low on y, high on x; fine but double-check if your doc uses that convention (to avoid confusion when comparing).

  Low

  - Docs path and content nits
      - docs/STAGE4_QUICKSTART.md references “Full Stage 4 documentation: project_instruction/stage/stage_04.md” but the actual file is
        project_instruction/stage_04.md (no extra “stage/” directory).
      - The sample log says “Grid search: 81 x 81 = 6561 combinations” while earlier you say “9×9 grid” and your defaults produce 9×9=81, not
        6561. Adjust the sample output.
      - STAGE4_QUICK_REFERENCE.md appears missing from the repo.
  - cascade_infer_example.py model-type detection is brittle
      - src/tools/cascade_infer_example.py: if 'mobilenetv4' in checkpoint path → MobileNet; else EfficientNet. This relies on path naming.
        Consider explicit CLI flags or embedding metadata in the checkpoint.

  Questions

  - Which exact EfficientNetV2 variant was used in Stage 3 best_model.pth? If not tf_efficientnetv2_b0, please expose a CLI arg like --stage2-
    model tf_efficientnetv2_b0 and pass to create_baseline_model to match the trained backbone.
  - Did you intend to support disabling AMP, Stage 2 precompute, and latency measurement via CLI? If yes, we should switch to paired flags
    (e.g., --amp/--no-amp) or default False with store_true.
  - Is the Stage 1 MobileNetV4 head true-BCE (1 output) or 2-class softmax? Your confidence() handles both, but if any runs used a 2-logit head,
    we should verify that the target_class mapping matches the “fake” class.

  Change Summary (proposed)

  1. Fix preprocessing and input size per stage (most important)

  - Keep a single dataloader at Stage 1 size for I/O efficiency, then:
      - Normalize to ImageNet for both stages on-tensor:
          - images = (images - mean) / std with mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
      - For Stage 2, upsample on GPU before passing to the model:
          - images_s2 = torch.nn.functional.interpolate(images, size=(stage2_size, stage2_size), mode='bilinear', align_corners=False).
  - Apply this in both precompute_logits and measure_latency:
      - Stage 1: use normalized images at stage1_size.
      - Stage 2: use upsampled+normalized images.
  - Alternatively, build two eval dataloaders using CelebDFDataset with normalize=True and sizes stage1_size and stage2_size, both
    shuffle=False. Precompute with each loader, relying on consistent sample order. This is a bit more I/O but simplest and fully correct.

  2. Enable lazy Stage 2 inference and fix the CLI flags

  - Make --precompute-stage2 default False (or keep default True but allow opt-out) and implement lazy mode in predict_batch:
      - If self.stage2_logits_cache is None: run Stage 2 only on escalation_indices:
          - Take the original batch tensors indexed by escalation_indices, normalize and resize to stage2_size as above, then run the model.
            Cache per index in a dict to avoid recomputing across threshold pairs.
  - For flags: use a pair or store_true with default False:
      - --amp (default False)
      - --precompute-stage2 (default False)
      - --measure-latency (default False)
      - --pin-memory (default False)

  3. Robust device handling

  - Detect device automatically:
      - device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      - Let CLI override if the user provides --device.

  4. Faster, deterministic labels

  - Replace y_true extraction:
      - y_true = dataset.df['label'].to_numpy().astype(int)
  - That avoids re-reading or touching imaging transforms.

  5. Align Stage 2 model variant

  - Add --stage2-model with default matching the actual trained variant; pass to create_baseline_model.
  - If you saved hparams with checkpoint, try reading them from the checkpoint dict (e.g., checkpoint['hparams']['model_name']).

  6. Cleanups and doc fixes

  - Remove unused imports in tune_cascade_system.py or add ROC/CM plots for the selected best config (optional).
  - Fix docs:
      - docs/STAGE4_QUICKSTART.md: correct “Full Stage 4 documentation” path and grid-search example counts.
      - Add the missing STAGE4_QUICK_REFERENCE.md or update your summary to match existing docs.

  What I Verified

  - Tools present:
      - src/tools/tune_cascade_system.py (≈35 KB, ~900 lines)
      - src/tools/cascade_infer_example.py (≈12 KB, ~340 lines)
  - Docs present:
      - project_instruction/stage_04.md
      - docs/STAGE4_QUICKSTART.md
      - STAGE4_QUICK_REFERENCE.md not found
  - Manifests present with expected schema (‘image_path’, ‘label’, ‘split’, etc.); celebdf_v2_val_balanced.csv exists and appears balanced.
  - Stage 1 and Stage 3 best_model.pth files exist under outputs/stage1/... and outputs/stage3/....
  - No outputs/stage4/ run found yet in this workspace snapshot; if you ran it elsewhere, share the run dir path and I can sanity-check the CSV/
    JSON/README quickly.

  Suggested Next Steps

  1. Pick one of these options to fix preprocessing:
      1. Minimal code change: keep the single dataloader, normalize on-tensor for both stages and upsample to stage2_size inside
         precompute_logits and measure_latency.
      2. Most explicit: add a second val dataloader at stage2_size with normalize=True and precompute Stage 2 logits using that loader. Keep
         shuffle=False so ordering aligns.
  2. Implement lazy Stage 2 evaluation and invert CLI defaults for toggles so they’re truly optional.
  3. Add device fallback to CPU automatically; expose --device for overrides.
  4. Replace y_true extraction with labels from dataset.df to reduce I/O and ensure determinism.
  5. Fix docs references and add STAGE4_QUICK_REFERENCE.md (or remove its mention).
  6. Optional: include ROC/Confusion Matrix plots for the best config to leverage utils.plotting and justify the ModelEvaluator import.

  If you want, I can prepare a small patch that:

  - Applies ImageNet normalization and Stage 2 resizing on-tensor,
  - Implements lazy Stage 2 inference with caching,
  - Switches CLI flags to be disable-able,
  - Adds CPU fallback,
  - Simplifies y_true extraction,
  - Fixes the docs path nits.

  Reply 1 to go with the minimal on-tensor normalization + resize fix, or 2 to switch to dual-dataloader precompute (cleanest correctness).