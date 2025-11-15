# References To Collect (Phase B Web Search)

Goal: Add 70–80 authoritative citations across detection methods, mobile/lightweight, compression (PTQ/QAT/pruning/distillation), cascaded/adaptive inference, generalization/robustness/OOD, domain adaptation, datasets/benchmarks, and deployment tooling.

For each item, include BibTeX and short 1–2 line rationale. Prefer peer‑reviewed venues (CVPR/ICCV/ECCV/NeurIPS/ICML/AAAI/TIFS/TPAMI), then arXiv if necessary.

## Buckets and Target Counts
- Detection architectures and surveys (15)
- Mobile/lightweight architectures (10)
- Compression: PTQ/QAT/pruning/distillation (12)
- Cascaded/adaptive/early‑exit inference + cost‑sensitive optimization (12)
- Generalization and robustness for deepfake (12)
- Domain adaptation / OOD in detection (6)
- Datasets and benchmarking (5)
- Deployment/tooling (3)

## Candidate Queries (to refine during search)

### Detection Architectures & Surveys
- "Deepfake detection survey"; "Media forensics survey"
- "FaceForensics++ deepfake detection baseline Xception"
- "ViT deepfake detection", "transformer deepfake detection"
- "Hybrid CNN transformer deepfake detection"
- "Temporal deepfake detection 3D CNN LSTM"

### Mobile / Lightweight
- "MobileNetV4 paper", "MobileNetV3", "EfficientNet‑Lite"
- "ShuffleNet", "SqueezeNet edge deployment"

### Compression & Distillation
- "Quantization Aware Training survey", "PTQ vs QAT"
- "Integer‑only quantization Jacob 2018"
- "Neural network pruning survey"
- "Knowledge Distillation Hinton 2015", "KD survey", "dual/teacher distillation deepfake"

### Cascaded / Adaptive / Early‑Exit / Cost‑Sensitive
- "early exit networks BranchyNet", "MSDNet", "Shallow‑Deep Networks"
- "cascaded classification cost sensitive optimization"
- "entropy routing threshold", "margin‑based routing"

### Generalization & Robustness
- "deepfake detector generalization cross‑dataset"
- "frequency artifacts deepfake detection", "JPEG robustness deepfake"
- "noise/blur robustness deepfake"

### Domain Adaptation / OOD
- "domain adaptation deepfake detection"
- "unsupervised domain adaptation face forensics"

### Datasets & Benchmarking
- "Celeb‑DF v2 paper", "FaceForensics++ paper", "DFDC dataset paper"
- "DeeperForensics paper", "Deepfake‑Eval‑2024 paper"
- "Benchmarking deepfake detection"

### Deployment / Tooling
- "PyTorch Mobile", "TorchScript", "torch.ao quantization"

## Mapping to Citation Keys (to replace todo_*)
- todo_curriculum_survey, todo_hard_mining
- todo_cost_sensitive, todo_cascade_adaptive, todo_bayes_opt
- todo_entropy_routing, todo_margin_routing
- todo_qat_survey, todo_ptq_qat_survey, todo_int4_quant
- todo_xception, todo_vit_df, todo_hybrid_df, todo_temporal_df
- todo_shufflenet, todo_pruning_survey, todo_kd_survey
- todo_cascade_survey, todo_early_exit
- todo_generalization_gap, todo_robustness_survey
- todo_domain_adapt_survey

## Process
1) Search per bucket; shortlist 8–15 per bucket; pick top N by venue/impact.
2) Add BibTeX to `paper/references.bib`, maintain coherent `ieeetr` style fields.
3) Replace `todo_*` placeholders across sections; recompile and check link warnings.
4) Cap to ~80 total; prefer recent (2019–2025), ensure dataset papers included.

