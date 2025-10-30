#!/usr/bin/env python3
"""
Stage 02: Difficult Sample Mining Tool

This tool loads the Stage 01 best model and runs inference over the full
training splits of all enabled datasets defined in configs/datasets.json.
It tags samples as difficult if they fall into an ambiguity range
([ambiguity.lower, ambiguity.upper]) and/or are predicted incorrectly at a
decision threshold (default 0.5). It then emits a manifest that contains
only the difficult samples (subset), preserving original manifest columns
and adding helpful fields: prob, pred, difficulty, dataset.

Why in src/tools?: This is a data preprocessing/mining utility, not a
training loop.

Usage (example):
    python -m src.tools.create_difficult_subset \
      --stage1_model_path outputs/stage1/<run>/best_model.pth \
      --config configs/datasets.json \
      --stage2_config configs/stage_02_config.yaml \
      --out manifests/train_difficult_subset.csv \
      --save_intermediate_preds true

Notes
- For very large datasets, enable --save_intermediate_preds to stream per-dataset
  predictions into outputs/stage2/run_<ts>/ and avoid keeping everything in memory.
- Ambiguity bounds and decision threshold are configurable via YAML or CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure 'src' is importable when executed as a script
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
import sys as _sys
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from training.dataset import CelebDFDataset  # noqa: E402
from utils.experiment_framework import (  # noqa: E402
    ExperimentFramework,
    load_config,
    merge_configs,
    setup_reproducible_environment,
)

logger = logging.getLogger(__name__)


def _parse_kv_pairs(pairs: Optional[List[str]]) -> Dict[str, int]:
    """Parse CLI key=value pairs into a dict."""
    parsed: Dict[str, int] = {}
    if not pairs:
        return parsed
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected key=value, got '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        try:
            parsed[key] = int(float(value))
        except ValueError as exc:  # noqa: BLE001
            raise argparse.ArgumentTypeError(
                f"Value for '{key}' must be numeric, got '{value}'"
            ) from exc
    return parsed


def _str2bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


@dataclass
class Stage2Config:
    ambiguity_lower: float = 0.3
    ambiguity_upper: float = 0.7
    decision_threshold: float = 0.5
    batch_size: int = 256
    num_workers: int = 8
    device: str = "auto"
    save_intermediate_preds: bool = False
    pin_memory: bool = True
    use_amp: bool = True


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_stage2_config(path: Optional[str], cli_overrides: Dict[str, object]) -> Stage2Config:
    base: Dict[str, object] = {}

    # Load YAML/JSON if provided
    if path:
        try:
            base = load_config(path) or {}
        except FileNotFoundError:
            logger.warning("Stage 02 config file not found: %s (using defaults/CLI)", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error loading Stage 02 config '%s': %s (using defaults/CLI)", path, exc)

    # Accept both flat and nested keys under a stage_02 root
    # Example YAML structure:
    # stage_02:
    #   ambiguity_lower: 0.3
    #   ambiguity_upper: 0.7
    #   decision_threshold: 0.5
    #   batch_size: 256
    #   num_workers: 8
    #   save_intermediate_preds: false
    #   use_amp: true
    cfg = base.get("stage_02", base)

    # Merge with CLI (CLI wins)
    merged = merge_configs(cfg, cli_overrides)

    return Stage2Config(
        ambiguity_lower=float(merged.get("ambiguity_lower", 0.3)),
        ambiguity_upper=float(merged.get("ambiguity_upper", 0.7)),
        decision_threshold=float(merged.get("decision_threshold", 0.5)),
        batch_size=int(merged.get("batch_size", 256)),
        num_workers=int(merged.get("num_workers", 8)),
        device=str(merged.get("device", "auto")),
        save_intermediate_preds=bool(merged.get("save_intermediate_preds", False)),
        pin_memory=bool(merged.get("pin_memory", True)),
        use_amp=bool(merged.get("use_amp", True)),
    )


def _load_stage1_model(stage1_model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load Stage 01 MobileNetV4 model from checkpoint saved by ExperimentFramework.

    Robust to different metadata placements in the checkpoint.
    """
    from models.mobilenetv4_model import create_mobilenetv4_simple

    ckpt = torch.load(stage1_model_path, map_location=device)

    # Try to get model_name from additional_info.hyperparameters; fallback to default
    model_name = "mobilenetv4_hybrid_medium"
    try:
        add = ckpt.get("additional_info") or {}
        h = add.get("hyperparameters") or {}
        model_name = h.get("model_name", model_name)
    except Exception:  # noqa: BLE001
        pass

    model = create_mobilenetv4_simple(
        model_name=model_name,
        pretrained=False,
        dropout_rate=0.2,
        freeze_backbone=False,
    ).to(device)

    state = ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _rel_path_from_root(path_str: str, root: Path) -> str:
    """Convert an absolute or relative path back to a manifest-like relative path.

    We try to make paths relative to the dataset root to match the 'image_path'
    keys stored in manifests. If that fails, we return the original string.
    """
    try:
        p = Path(path_str).resolve()
        r = root.resolve()
        rel = p.relative_to(r)
        return rel.as_posix()
    except Exception:
        # Already relative or on different drive; return as-is
        return Path(path_str).as_posix()


def _infer_dataset(
    name: str,
    cfg: Dict[str, object],
    model: torch.nn.Module,
    device: torch.device,
    s2cfg: Stage2Config,
    out_dir: Path,
    save_intermediate: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run inference over one dataset's train split and return (all_preds, difficult_subset).

    all_preds columns: [image_path, label, prob, pred, dataset]
    difficult_subset is filtered on ambiguity/error and includes the same columns plus 'difficulty'.
    """
    root_path = Path(cfg.get("root_path", "."))
    train_manifest = root_path / cfg.get("splits", {}).get(
        "train", f"manifests/{name}_train_balanced.csv"
    )

    # Build dataset without augmentation; include metadata path
    ds = CelebDFDataset(
        manifest_path=train_manifest,
        root_path=root_path,
        image_size=int(cfg.get("metadata", {}).get("image_size", [256])[0]),
        augmentation=False,
        normalize=True,
        return_meta=True,
    )

    loader = DataLoader(
        ds,
        batch_size=s2cfg.batch_size,
        shuffle=False,
        num_workers=s2cfg.num_workers,
        pin_memory=s2cfg.pin_memory,
        drop_last=False,
    )

    records: List[Dict[str, object]] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (s2cfg.use_amp and device.type == "cuda")
        else torch.cpu.amp.autocast(enabled=False)
    )

    model.eval()
    with torch.no_grad():
        with amp_ctx:
            for batch in tqdm(loader, desc=f"infer:{name}"):
                # Support (img, target) and (img, target, meta)
                try:
                    images, targets, meta = batch
                except (ValueError, TypeError):
                    images, targets = batch
                    meta = None

                images = images.to(device, non_blocking=True)
                targets_np = targets.cpu().numpy().astype(int)

                logits = model(images)
                # Robustly convert logits to 1D per-sample probabilities
                probs_t = torch.sigmoid(logits)
                # Common cases: shape [B] or [B,1]. Handle both; fallback to first logit per sample.
                if probs_t.dim() == 2 and probs_t.size(1) == 1:
                    probs_t = probs_t[:, 0]
                elif probs_t.dim() > 2:
                    probs_t = probs_t.view(probs_t.size(0), -1)[:, 0]
                # If probs_t.dim() == 1, keep as-is
                probs = probs_t.detach().cpu().numpy()
                preds = (probs >= s2cfg.decision_threshold).astype(int)

                # Derive manifest-like paths
                if meta is not None:
                    if isinstance(meta, dict) and "path" in meta:
                        paths = meta["path"]
                    else:
                        # Fallback: try to iterate meta and extract 'path'
                        paths = []
                        for m in meta:
                            if isinstance(m, dict) and "path" in m:
                                paths.append(m["path"])
                            else:
                                paths.append(str(m))
                else:
                    # If metadata is missing, leave empty and rely on join later (not ideal)
                    paths = [""] * len(preds)

                for p, y, pr, pd_ in zip(paths, targets_np, probs, preds):
                    rel = _rel_path_from_root(str(p), root_path)
                    records.append(
                        {
                            "image_path": rel,
                            "label": int(y),
                            "prob": float(pr),
                            "pred": int(pd_),
                            "dataset": name,
                        }
                    )

    all_preds = pd.DataFrame.from_records(records)

    # Join back with original manifest to preserve columns and ensure consistency
    df_src = pd.read_csv(train_manifest)
    merged = df_src.merge(
        all_preds[["image_path", "prob", "pred", "dataset"]],
        on="image_path",
        how="inner",
    )

    # Apply difficulty rules
    amb = (merged["prob"] >= s2cfg.ambiguity_lower) & (merged["prob"] <= s2cfg.ambiguity_upper)
    err = merged["pred"] != merged["label"]
    difficulty = np.where(amb & err, "both", np.where(amb, "ambiguous", np.where(err, "error", "none")))
    merged["difficulty"] = difficulty

    difficult = merged[merged["difficulty"] != "none"].copy()

    # Optionally write per-dataset predictions to stage2 output directory
    if save_intermediate:
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_path = out_dir / f"{name}_train_preds.csv"
        try:
            # Persist only needed columns to keep size in check
            all_preds.to_csv(preds_path, index=False)
            logger.info("Saved intermediate predictions: %s", preds_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save intermediate preds '%s': %s", preds_path, exc)

    merged["dataset"] = name
    return merged, difficult


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 02: Mine difficult samples from training sets")
    parser.add_argument(
        "--stage1_model_path",
        type=str,
        required=True,
        help="Path to Stage 01 best model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets.json",
        help="Datasets configuration JSON (Stage 01 format)",
    )
    parser.add_argument(
        "--stage2_config",
        type=str,
        default="configs/stage_02_config.yaml",
        help="Stage 02 YAML/JSON with ambiguity and I/O settings",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="manifests/train_difficult_subset.csv",
        help="Output manifest (CSV) containing only difficult samples",
    )
    parser.add_argument(
        "--save_intermediate_preds",
        type=_str2bool,
        default=None,
        help="Whether to save per-dataset predictions CSVs (override config)",
    )
    # Optional CLI overrides for common params
    parser.add_argument("--ambiguity_lower", type=float, default=None)
    parser.add_argument("--ambiguity_upper", type=float, default=None)
    parser.add_argument("--decision_threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", type=_str2bool, default=None)
    parser.add_argument(
        "--dataset_quota",
        nargs="*",
        default=None,
        metavar="DATASET=COUNT",
        help="Per-dataset cap applied to difficult samples (e.g. dfdc=15000)",
    )
    parser.add_argument(
        "--dataset_min",
        nargs="*",
        default=None,
        metavar="DATASET=COUNT",
        help="Per-dataset minimum to retain (e.g. celebdf_v2=3000)",
    )

    args = parser.parse_args()

    # Configure logging (console + file via ExperimentFramework)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load datasets config
    datasets_cfg = load_config(args.config)
    datasets = datasets_cfg.get("datasets", {})

    # Build Stage 02 config from file + CLI overrides
    cli_overrides = {}
    for k in [
        "ambiguity_lower",
        "ambiguity_upper",
        "decision_threshold",
        "batch_size",
        "num_workers",
        "device",
        "use_amp",
        "save_intermediate_preds",
    ]:
        v = getattr(args, k)
        if v is not None:
            cli_overrides[k] = v

    s2cfg = _load_stage2_config(args.stage2_config, cli_overrides)
    device = _resolve_device(s2cfg.device)
    dataset_quota_cfg = _parse_kv_pairs(args.dataset_quota)
    dataset_min_cfg = _parse_kv_pairs(args.dataset_min)

    # Reproducibility
    setup_reproducible_environment(42)

    # Create an experiment dir for stage2 artifacts/logs
    framework = ExperimentFramework(output_dir="outputs/stage2", experiment_name="stage2_difficult_subset")
    log_dir = Path(framework.experiment_dir)
    logger.info("Stage 02 outputs: %s", log_dir)

    # Load Stage 01 best model
    model = _load_stage1_model(Path(args.stage1_model_path), device)

    # Iterate datasets and process train split
    dataset_full_records: Dict[str, pd.DataFrame] = {}
    difficult_parts: List[pd.DataFrame] = []
    for name in ["celebdf_v2", "faceforensics", "deeperforensics", "dfdc"]:
        ds_cfg = datasets.get(name)
        if not ds_cfg or not ds_cfg.get("enabled", True):
            logger.info("Dataset '%s' disabled or missing in config. Skipping.", name)
            continue

        logger.info("Processing dataset '%s'", name)
        merged_preds, difficult = _infer_dataset(
            name=name,
            cfg=ds_cfg,
            model=model,
            device=device,
            s2cfg=s2cfg,
            out_dir=log_dir,
            save_intermediate=s2cfg.save_intermediate_preds,
        )

        dataset_full_records[name] = merged_preds
        logger.info(
            "Dataset '%s': total=%d, difficult=%d (amb>=%.2f<=%.2f or error at thr=%.2f)",
            name,
            len(merged_preds),
            len(difficult),
            s2cfg.ambiguity_lower,
            s2cfg.ambiguity_upper,
            s2cfg.decision_threshold,
        )

        difficult_parts.append(difficult)

    # Concatenate and de-duplicate by (dataset, image_path)
    if difficult_parts:
        df_difficult = pd.concat(difficult_parts, ignore_index=True)
        df_difficult = df_difficult.drop_duplicates(subset=["dataset", "image_path"], keep="first")

        if dataset_min_cfg:
            for dataset_name, minimum in dataset_min_cfg.items():
                if minimum is None or minimum <= 0:
                    continue
                current_subset = df_difficult[df_difficult["dataset"] == dataset_name]
                current_count = len(current_subset)
                if current_count >= minimum:
                    continue
                full_preds = dataset_full_records.get(dataset_name)
                if full_preds is None or full_preds.empty:
                    logger.warning(
                        "Dataset '%s' missing prediction records; cannot satisfy minimum %d.",
                        dataset_name,
                        minimum,
                    )
                    continue

                available = full_preds.drop_duplicates(subset=["dataset", "image_path"], keep="first")
                if not current_subset.empty:
                    available = available[~available["image_path"].isin(current_subset["image_path"])]

                if available.empty:
                    logger.warning(
                        "Dataset '%s' has no additional samples to reach minimum %d (current=%d).",
                        dataset_name,
                        minimum,
                        current_count,
                    )
                    continue

                needed = minimum - current_count
                needed = min(needed, len(available))

                available = available.copy()
                if "difficulty" not in available.columns:
                    available["difficulty"] = "none"
                available["distance_to_threshold"] = np.abs(
                    available["prob"] - s2cfg.decision_threshold
                )
                available = available.sort_values("distance_to_threshold", ascending=True)
                selected = available.head(needed).copy()
                if selected.empty:
                    logger.warning(
                        "Dataset '%s' could not provide additional samples despite available entries.",
                        dataset_name,
                    )
                    continue

                # Mark padding source so downstream analysis can filter if needed
                selected.loc[selected["difficulty"] == "none", "difficulty"] = "min_padding"
                selected = selected.drop(columns=["distance_to_threshold"], errors="ignore")

                df_difficult = pd.concat([df_difficult, selected], ignore_index=True)
                logger.info(
                    "Dataset '%s' padded with %d near-threshold samples to reach minimum %d (total=%d).",
                    dataset_name,
                    len(selected),
                    minimum,
                    len(df_difficult[df_difficult["dataset"] == dataset_name]),
                )

                if len(df_difficult[df_difficult["dataset"] == dataset_name]) < minimum:
                    logger.warning(
                        "Dataset '%s' still below requested minimum %d after padding (current=%d).",
                        dataset_name,
                        minimum,
                        len(df_difficult[df_difficult["dataset"] == dataset_name]),
                    )

        df_difficult = df_difficult.drop_duplicates(subset=["dataset", "image_path"], keep="first")

        if dataset_quota_cfg or dataset_min_cfg:
            rng = np.random.default_rng(42)
            capped_parts: List[pd.DataFrame] = []
            for dataset_name, subset in df_difficult.groupby("dataset", sort=False):
                original_len = len(subset)
                quota = (dataset_quota_cfg or {}).get(dataset_name)
                minimum = (dataset_min_cfg or {}).get(dataset_name)

                target = original_len
                if quota is not None:
                    target = min(target, quota)
                if minimum is not None:
                    if original_len < minimum:
                        logger.warning(
                            "Dataset '%s' only has %d difficult samples (< requested min %d); keeping all.",
                            dataset_name,
                            original_len,
                            minimum,
                        )
                    else:
                        target = max(target, minimum)

                target = min(target, original_len)
                if target < original_len:
                    indices = rng.choice(original_len, size=target, replace=False)
                    subset = subset.iloc[np.sort(indices)]
                    logger.info(
                        "Dataset '%s' capped from %d to %d difficult samples",
                        dataset_name,
                        original_len,
                        target,
                    )

                capped_parts.append(subset)

            df_difficult = pd.concat(capped_parts, ignore_index=True)
    else:
        df_difficult = pd.DataFrame(columns=["image_path", "label", "prob", "pred", "difficulty", "dataset"])  # noqa: E501

    # Filter valid if column exists and equals True
    if "valid" in df_difficult.columns:
        df_difficult = df_difficult[df_difficult["valid"] == True]  # noqa: E712

    # Save final subset manifest (ONLY difficult samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_difficult.to_csv(out_path, index=False)
    logger.info("Saved difficult subset: %s (rows=%d)", out_path, len(df_difficult))

    # Write a concise summary JSON
    summary = {
        "stage": 2,
        "output_manifest": str(out_path),
        "total_difficult": int(len(df_difficult)),
        "by_dataset": df_difficult.groupby("dataset").size().to_dict() if not df_difficult.empty else {},
        "ambiguity_lower": s2cfg.ambiguity_lower,
        "ambiguity_upper": s2cfg.ambiguity_upper,
        "decision_threshold": s2cfg.decision_threshold,
        "save_intermediate_preds": s2cfg.save_intermediate_preds,
    }
    with open(log_dir / "stage2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Stage 02 summary saved: %s", log_dir / "stage2_summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
