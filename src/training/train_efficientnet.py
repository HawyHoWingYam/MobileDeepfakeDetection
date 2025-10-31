#!/usr/bin/env python3
"""
train_efficientnet.py – Stage 03 EfficientNetV2 Expert Training

Trains an EfficientNetV2 expert model using the difficult subset manifest
generated in Stage 02. Evaluates on full multi-dataset validation to ensure
generalization. Uses the Stage 00 ExperimentFramework for logging and model
checkpointing.

Default run (after Stage 02):
    python -m src.training.train_efficientnet \
      --train_manifest manifests/train_difficult_subset.csv \
      --model_name tf_efficientnetv2_b0 \
      --epochs 10 --batch_size 256 --learning_rate 1e-4

Notes
- The training manifest is expected to include at least: image_path,label; other
  fields like dataset,difficulty are optional but enable weighted sampling.
- Validation is performed on the combined validation splits from configs/datasets.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure 'src' on path
import sys as _sys
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from models.efficientnetv2_model import create_baseline_model  # noqa: E402
from training.dataset import CelebDFDataset  # noqa: E402
from utils.experiment_framework import (  # noqa: E402
    ExperimentFramework,
    setup_reproducible_environment,
)
from utils.evaluation import ModelEvaluator  # noqa: E402
from utils.plotting import (  # noqa: E402
    plot_learning_curves,
    plot_roc_curve_precomputed,
    plot_precision_recall_precomputed,
    plot_calibration_curve_precomputed,
    plot_confidence_histogram,
    plot_train_val_gap,
    plot_lr_schedule,
    plot_image_grid,
    plot_probability_distribution,
    plot_error_analysis,
    create_comprehensive_report,
)
from training.train_mobilenet import create_multi_dataset_loader  # reuse val loader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score


logger = logging.getLogger("train_efficientnet")


def str2bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}: return True
    if s in {"false", "f", "0", "no", "n"}: return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


@dataclass
class TrainConfig:
    train_manifest: Path
    model_name: str = "tf_efficientnetv2_b0"
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 4
    min_delta: float = 0.0005
    restore_best: bool = True
    save_on_metric: str = "f1"
    output_dir: Path = Path("outputs/stage3")
    device: str = "auto"
    num_workers: int = 12
    image_size: Optional[int] = 256
    dataset_balance: str = "equal_by_dataset"  # none|equal_by_dataset
    balance_exponent: float = 0.75
    dropout_rate: float = 0.3
    freeze_backbone_epochs: int = 3
    backbone_lr_scale: float = 0.5
    neg_class_weight: float = 1.25
    pos_class_weight: float = 1.0
    real_aug_prob: float = 0.2
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def make_train_loader(cfg: TrainConfig) -> Tuple[DataLoader, Optional[np.ndarray], Dict[str, int]]:
    """Create a training loader from the difficult subset manifest.

    If dataset_balance == equal_by_dataset and the manifest has a 'dataset' column,
    use a WeightedRandomSampler to equalize dataset contributions per epoch.
    Returns (loader, weights_used or None, class_counts).
    """
    # Read manifest for weighting decisions
    df = pd.read_csv(cfg.train_manifest)
    root_path = Path(".")

    ds = CelebDFDataset(
        manifest_path=str(cfg.train_manifest),
        root_path=str(root_path),
        image_size=int(cfg.image_size or 256),
        augmentation=True,
        normalize=True,
        return_meta=False,
        real_aug_prob=cfg.real_aug_prob,
    )

    sampler = None
    weights = None
    class_counts = df["label"].value_counts().to_dict()
    logger.info("Training manifest class distribution: %s", class_counts)
    if cfg.dataset_balance == "equal_by_dataset" and "dataset" in df.columns:
        counts = df["dataset"].value_counts().to_dict()
        # Equalize datasets: each dataset shares 1/K of samples; per-sample weight inversely proportional to its dataset count
        inv = df["dataset"].map(lambda x: 1.0 / counts.get(x, 1)).astype(float)
        inv = inv / inv.mean()
        if abs(cfg.balance_exponent - 1.0) > 1e-6:
            inv = np.power(inv, cfg.balance_exponent)
        weights = inv.values
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        logger.info("Using WeightedRandomSampler to equalize datasets: %s (exponent=%.2f)", counts, cfg.balance_exponent)

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, weights, class_counts


def sanitize_evaluation_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Remove non-serializable objects (numpy arrays) from evaluation summary."""
    sanitized = dict(summary)
    for key in ("targets_array", "probabilities_array", "predictions_array"):
        sanitized.pop(key, None)
    return sanitized


def to_float_list(values: List[float]) -> List[float]:
    """Ensure all values are plain Python floats for JSON serialization."""
    return [float(x) for x in values]


def create_validation_loader_with_metadata(
    config_path: str,
    batch_size: int,
    num_workers: int,
    image_size: Optional[int] = None,
) -> Optional[DataLoader]:
    """Build validation loader that yields metadata for error visualization."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        logger.warning("Validation config not found for metadata loader: %s", cfg_path)
        return None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            datasets_cfg = json.load(f)
    except Exception as exc:
        logger.error("Failed to read dataset config %s: %s", cfg_path, exc)
        return None

    dataset_names = ["celebdf_v2", "faceforensics", "deeperforensics", "dfdc"]
    datasets_cfg = datasets_cfg.get("datasets", {})

    val_datasets: List[CelebDFDataset] = []
    for dataset_name in dataset_names:
        dataset_cfg = datasets_cfg.get(dataset_name)
        if not dataset_cfg or not dataset_cfg.get("enabled", True):
            continue

        root_path = Path(dataset_cfg.get("root_path", "."))
        splits = dataset_cfg.get("splits", {})
        ds_image_size = dataset_cfg.get("metadata", {}).get("image_size", [256])[0]
        if image_size is not None:
            ds_image_size = int(image_size)
        val_manifest = root_path / splits.get("val", f"manifests/{dataset_name}_val_balanced.csv")

        try:
            val_ds = CelebDFDataset(
                manifest_path=val_manifest,
                root_path=root_path,
                image_size=ds_image_size,
                augmentation=False,
                normalize=True,
                return_meta=True,
            )
            val_datasets.append(val_ds)
        except Exception as exc:
            logger.warning("Failed to prepare metadata dataset for %s: %s", dataset_name, exc)

    if not val_datasets:
        logger.warning("No validation datasets available for metadata-enabled loader")
        return None

    combined = ConcatDataset(val_datasets)
    logger.info("Metadata validation loader size: %s samples", len(combined))
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 03: Train EfficientNetV2 expert on difficult subset")
    parser.add_argument("--train_manifest", type=str, default="manifests/train_difficult_subset.csv")
    parser.add_argument("--model_name", type=str, default="tf_efficientnetv2_b0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min_delta", type=float, default=0.0005)
    parser.add_argument("--restore_best", type=str2bool, default=True)
    parser.add_argument("--save_on_metric", type=str, default="f1")
    parser.add_argument("--output_dir", type=str, default="outputs/stage3")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dataset_balance", type=str, default="equal_by_dataset", choices=["none","equal_by_dataset"])
    parser.add_argument("--balance_exponent", type=float, default=0.75)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.5)
    parser.add_argument("--neg_class_weight", type=float, default=1.25)
    parser.add_argument("--pos_class_weight", type=float, default=1.0)
    parser.add_argument("--real_aug_prob", type=float, default=0.2)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_scheduler", type=str2bool, default=True)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=2)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)

    args = parser.parse_args()

    # Basic validation
    train_manifest = Path(args.train_manifest)
    if not train_manifest.exists():
        print(f"Train manifest not found: {train_manifest}")
        return 1

    # Setup
    setup_reproducible_environment(42)
    device = resolve_device(args.device)
    cfg = TrainConfig(
        train_manifest=train_manifest,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best=args.restore_best,
        save_on_metric=args.save_on_metric.lower(),
        output_dir=Path(args.output_dir),
        device=args.device,
        num_workers=args.num_workers,
        image_size=args.image_size,
        dataset_balance=args.dataset_balance,
        balance_exponent=args.balance_exponent,
        dropout_rate=args.dropout_rate,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        backbone_lr_scale=args.backbone_lr_scale,
        neg_class_weight=args.neg_class_weight,
        pos_class_weight=args.pos_class_weight,
        real_aug_prob=args.real_aug_prob,
        grad_clip_norm=args.grad_clip_norm,
        label_smoothing=args.label_smoothing,
        use_scheduler=args.use_scheduler,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
    )

    # Experiment directory
    framework = ExperimentFramework(output_dir=str(cfg.output_dir), experiment_name="stage3_efficientnet_expert")
    logger.info("Stage 03 outputs: %s", framework.experiment_dir)

    # Model
    model = create_baseline_model(
        pretrained=True,
        dropout_rate=cfg.dropout_rate,
        model_name=cfg.model_name
    ).to(device)

    if cfg.freeze_backbone_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for first %d epochs", cfg.freeze_backbone_epochs)

    # Optimizer with differential learning rates
    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    if not head_params or not backbone_params:
        logger.warning("Could not separate backbone/head parameters cleanly; using single parameter group.")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            [
                {"params": head_params, "lr": cfg.learning_rate},
                {"params": backbone_params, "lr": cfg.learning_rate * cfg.backbone_lr_scale},
            ],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    pos_weight_tensor = torch.tensor([cfg.pos_class_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")

    # Setup scheduler
    scheduler = None
    if cfg.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )
        logger.info("ReduceLROnPlateau scheduler enabled: factor=%.2f, patience=%d, min_lr=%.2e",
                   cfg.scheduler_factor, cfg.scheduler_patience, cfg.scheduler_min_lr)

    # Data
    train_loader, _, class_counts = make_train_loader(cfg)
    logger.info("Initial class counts: %s", class_counts)

    # Validation loader (reuse Stage 01 loaders, but we only need val)
    _, val_loader, _ = create_multi_dataset_loader(
        config_path="configs/datasets.json",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        seed=42,
        override_image_size=cfg.image_size,
    )

    evaluator = ModelEvaluator(device=device, threshold=0.7)

    best_metric = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    backbone_frozen = cfg.freeze_backbone_epochs > 0
    neg_class_weight_active = abs(cfg.neg_class_weight - 1.0) > 1e-6

    train_loss_history: List[float] = []
    train_auc_history: List[float] = []
    train_f1_history: List[float] = []
    train_accuracy_history: List[float] = []
    train_precision_history: List[float] = []
    train_recall_history: List[float] = []
    train_specificity_history: List[float] = []
    train_fnr_history: List[float] = []
    val_loss_history: List[float] = []
    val_auc_history: List[float] = []
    val_f1_history: List[float] = []
    val_accuracy_history: List[float] = []
    val_precision_history: List[float] = []
    val_recall_history: List[float] = []
    val_specificity_history: List[float] = []
    val_fnr_history: List[float] = []
    epoch_metrics_history: List[Dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        if backbone_frozen and epoch >= cfg.freeze_backbone_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = True
            backbone_frozen = False
            logger.info("Unfroze EfficientNet backbone at epoch %d", epoch + 1)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        epoch_loss = 0.0
        n_batches = 0
        train_prob_buffer: List[float] = []
        train_target_buffer: List[float] = []

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            hard_targets = targets.detach().clone()

            # Apply label smoothing
            if cfg.label_smoothing > 0:
                targets = targets * (1 - cfg.label_smoothing) + cfg.label_smoothing * 0.5

            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss_values = criterion(logits, targets).view(-1)
            if neg_class_weight_active:
                sample_weights = torch.where(
                    targets < 0.5,
                    torch.full_like(loss_values, cfg.neg_class_weight),
                    torch.ones_like(loss_values)
                )
                loss = (loss_values * sample_weights).mean()
            else:
                loss = loss_values.mean()
            loss.backward()
            if cfg.grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            current_loss = epoch_loss / max(n_batches, 1)
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            probs = torch.sigmoid(logits)
            train_prob_buffer.extend(probs.detach().cpu().tolist())
            train_target_buffer.extend(hard_targets.cpu().tolist())

        # Validation
        val_metrics = evaluator.evaluate_model(model, val_loader, criterion=criterion, mode="validation")
        avg_train_loss = epoch_loss / max(n_batches, 1)
        try:
            train_probs_np = np.asarray(train_prob_buffer, dtype=np.float32)
            train_targets_np = np.asarray(train_target_buffer, dtype=np.float32)
            train_preds_np = (train_probs_np >= 0.7).astype(np.float32)
            train_auc = roc_auc_score(train_targets_np, train_probs_np) if train_targets_np.size and len(np.unique(train_targets_np)) > 1 else 0.5
            train_f1 = f1_score(train_targets_np, train_preds_np, zero_division=0)
            train_accuracy = accuracy_score(train_targets_np, train_preds_np)
            train_tp = int(((train_preds_np == 1) & (train_targets_np == 1)).sum())
            train_tn = int(((train_preds_np == 0) & (train_targets_np == 0)).sum())
            train_fp = int(((train_preds_np == 1) & (train_targets_np == 0)).sum())
            train_fn = int(((train_preds_np == 0) & (train_targets_np == 1)).sum())
            train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0.0
            train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0.0
            train_specificity = train_tn / (train_tn + train_fp) if (train_tn + train_fp) > 0 else 0.0
            train_fnr = train_fn / (train_fn + train_tp) if (train_fn + train_tp) > 0 else 0.0
        except Exception as exc:
            logger.warning("Failed to compute training metrics: %s", exc)
            train_auc = 0.5
            train_f1 = 0.0
            train_accuracy = 0.0
            train_precision = 0.0
            train_recall = 0.0
            train_specificity = 0.0
            train_fnr = 0.0

        metric_name = cfg.save_on_metric
        metric_value = float(val_metrics.get(metric_name, 0.0))
        framework.log_metrics(epoch, {"train_loss": avg_train_loss})
        framework.log_metrics(epoch, val_metrics, mode="validation")
        framework.save_best_model(model, metric_value, metric_name=metric_name, optimizer=optimizer,
                                  additional_info={"hyperparameters": {
                                      "model_name": cfg.model_name,
                                      "learning_rate": cfg.learning_rate,
                                      "batch_size": cfg.batch_size,
                                      "weight_decay": cfg.weight_decay,
                                      "dropout_rate": cfg.dropout_rate,
                                      "neg_class_weight": cfg.neg_class_weight,
                                      "pos_class_weight": cfg.pos_class_weight,
                                      "freeze_backbone_epochs": cfg.freeze_backbone_epochs,
                                      "real_aug_prob": cfg.real_aug_prob,
                                  }})

        train_loss_history.append(float(avg_train_loss))
        train_auc_history.append(float(train_auc))
        train_f1_history.append(float(train_f1))
        train_accuracy_history.append(float(train_accuracy))
        train_precision_history.append(float(train_precision))
        train_recall_history.append(float(train_recall))
        train_specificity_history.append(float(train_specificity))
        train_fnr_history.append(float(train_fnr))
        val_loss_history.append(float(val_metrics.get("loss", 0.0)))
        val_auc_history.append(float(val_metrics.get("auc", 0.0)))
        val_f1_history.append(float(val_metrics.get("f1", 0.0)))
        val_accuracy_history.append(float(val_metrics.get("accuracy", 0.0)))
        val_precision_history.append(float(val_metrics.get("precision", 0.0)))
        val_recall_history.append(float(val_metrics.get("recall", 0.0)))
        val_specificity_history.append(float(val_metrics.get("specificity", 0.0)))
        val_fnr_history.append(float(val_metrics.get("fnr", 0.0)))
        current_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        epoch_metrics_history.append({
            "epoch": int(epoch + 1),
            "lr": current_lrs[0],
            "lr_groups": current_lrs,
            "train": {
                "loss": float(avg_train_loss),
                "auc": float(train_auc),
                "f1": float(train_f1),
                "accuracy": float(train_accuracy),
                "precision": float(train_precision),
                "recall": float(train_recall),
                "specificity": float(train_specificity),
                "fnr": float(train_fnr),
            },
            "val": {k: float(v) for k, v in val_metrics.items()},
        })

        # Early stopping (use AUC by default for stability)
        monitor = float(val_metrics.get("auc", 0.0))
        if monitor > best_metric + cfg.min_delta:
            best_metric = monitor
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info("New best AUC: %.4f at epoch %d", best_metric, epoch+1)
        else:
            patience_counter += 1
            logger.info("No AUC improvement. Patience %d/%d", patience_counter, cfg.patience)
            if cfg.patience > 0 and patience_counter >= cfg.patience:
                logger.info("Early stopping triggered at epoch %d", epoch+1)
                early_stopped = True
                break

        # Scheduler step (on AUC metric)
        if scheduler is not None:
            prev_lrs = [pg["lr"] for pg in optimizer.param_groups]
            scheduler.step(monitor)
            new_lrs = [pg["lr"] for pg in optimizer.param_groups]
            if any(abs(a - b) > 1e-12 for a, b in zip(prev_lrs, new_lrs)):
                logger.info("Scheduler adjusted learning rates: %s -> %s", prev_lrs, new_lrs)

    # Restore best
    if best_state is not None and cfg.restore_best:
        model.load_state_dict(best_state)
        logger.info("Restored best weights from epoch %d (AUC=%.4f)", best_epoch+1, best_metric)

    epochs_ran = len(train_loss_history)
    summary = {
        "training_completed": True,
        "early_stopped": bool(early_stopped),
        "epochs_configured": int(cfg.epochs),
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch + 1),
        "best_val_auc": float(best_metric),
        "train_manifest": str(cfg.train_manifest),
        "model_name": cfg.model_name,
        "class_counts": class_counts,
        "training_history": {
            "train_loss": to_float_list(train_loss_history),
            "train_auc": to_float_list(train_auc_history),
            "train_f1": to_float_list(train_f1_history),
            "train_accuracy": to_float_list(train_accuracy_history),
            "train_precision": to_float_list(train_precision_history),
            "train_recall": to_float_list(train_recall_history),
            "train_specificity": to_float_list(train_specificity_history),
            "train_fnr": to_float_list(train_fnr_history),
            "val_loss": to_float_list(val_loss_history),
            "val_auc": to_float_list(val_auc_history),
            "val_f1": to_float_list(val_f1_history),
            "val_accuracy": to_float_list(val_accuracy_history),
            "val_precision": to_float_list(val_precision_history),
            "val_recall": to_float_list(val_recall_history),
            "val_specificity": to_float_list(val_specificity_history),
            "val_fnr": to_float_list(val_fnr_history),
        },
        "epoch_metrics": epoch_metrics_history,
        "hyperparameters": {
            "model_name": cfg.model_name,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "patience": cfg.patience,
            "min_delta": cfg.min_delta,
            "dataset_balance": cfg.dataset_balance,
            "balance_exponent": cfg.balance_exponent,
            "dropout_rate": cfg.dropout_rate,
            "freeze_backbone_epochs": cfg.freeze_backbone_epochs,
            "backbone_lr_scale": cfg.backbone_lr_scale,
            "neg_class_weight": cfg.neg_class_weight,
            "pos_class_weight": cfg.pos_class_weight,
            "real_aug_prob": cfg.real_aug_prob,
            "grad_clip_norm": cfg.grad_clip_norm,
            "label_smoothing": cfg.label_smoothing,
            "use_scheduler": cfg.use_scheduler,
            "scheduler_factor": cfg.scheduler_factor,
            "scheduler_patience": cfg.scheduler_patience,
            "scheduler_min_lr": cfg.scheduler_min_lr,
            "evaluation_threshold": evaluator.threshold,
        },
    }

    summary_path = Path(framework.experiment_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved: %s", summary_path)

    plots_generated: List[str] = []
    plots_failed: List[Dict[str, str]] = []

    learning_curve_path = Path(framework.experiment_dir) / "learning_curves.png"
    try:
        plot_learning_curves(
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            train_auc=train_auc_history if len(train_auc_history) > 0 else None,
            val_auc=val_auc_history if len(val_auc_history) > 0 else None,
            train_f1=train_f1_history if len(train_f1_history) > 0 else None,
            val_f1=val_f1_history if len(val_f1_history) > 0 else None,
            train_accuracy=train_accuracy_history if len(train_accuracy_history) > 0 else None,
            val_accuracy=val_accuracy_history if len(val_accuracy_history) > 0 else None,
            train_precision=train_precision_history if len(train_precision_history) > 0 else None,
            val_precision=val_precision_history if len(val_precision_history) > 0 else None,
            train_recall=train_recall_history if len(train_recall_history) > 0 else None,
            val_recall=val_recall_history if len(val_recall_history) > 0 else None,
            train_specificity=train_specificity_history if len(train_specificity_history) > 0 else None,
            val_specificity=val_specificity_history if len(val_specificity_history) > 0 else None,
            train_fnr=train_fnr_history if len(train_fnr_history) > 0 else None,
            val_fnr=val_fnr_history if len(val_fnr_history) > 0 else None,
            output_path=str(learning_curve_path),
            best_epoch=best_epoch if epochs_ran else None,
            include_loss=False,
            include_auc=False,
        )
        plots_generated.append(learning_curve_path.name)
    except Exception as exc:
        logger.warning("Could not generate learning curves: %s", exc)
        plots_failed.append({"plot": learning_curve_path.name, "error": str(exc)})

    evaluation_summary_path = Path(framework.experiment_dir) / "evaluation_summary.json"

    try:
        final_evaluator = ModelEvaluator(device=device, threshold=0.7)
        val_loader_with_meta = create_validation_loader_with_metadata(
            config_path="configs/datasets.json",
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image_size=cfg.image_size,
        )
        eval_loader = val_loader_with_meta or val_loader
        val_summary = final_evaluator.full_evaluation(
            model, eval_loader, criterion=criterion, mode="validation"
        )

        with open(evaluation_summary_path, "w") as f:
            json.dump(sanitize_evaluation_summary(val_summary), f, indent=2)
        logger.info("Evaluation summary saved: %s", evaluation_summary_path)

        roc_data = val_summary.get("roc_curve", {})
        if roc_data.get("fpr") and roc_data.get("tpr"):
            try:
                fig = plot_roc_curve_precomputed(
                    roc_data["fpr"],
                    roc_data["tpr"],
                    val_summary.get("metrics", {}).get("auc", 0.0),
                    output_path=framework.experiment_dir / "01_roc_curve.png",
                )
                if fig is not None:
                    plt.close(fig)
                plots_generated.append("01_roc_curve.png")
            except Exception as exc:  # pragma: no cover - plotting safety
                logger.warning("Failed to generate ROC curve: %s", exc)
                plots_failed.append({"plot": "01_roc_curve.png", "error": str(exc)})

        pr_data = val_summary.get("pr_curve", {})
        if pr_data.get("precision") and pr_data.get("recall"):
            try:
                fig = plot_precision_recall_precomputed(
                    pr_data["precision"],
                    pr_data["recall"],
                    output_path=framework.experiment_dir / "02_pr_curve.png",
                )
                if fig is not None:
                    plt.close(fig)
                plots_generated.append("02_pr_curve.png")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to generate precision-recall curve: %s", exc)
                plots_failed.append({"plot": "02_pr_curve.png", "error": str(exc)})

        calibration = val_summary.get("calibration_bins", {})
        if calibration.get("bin_centers") and calibration.get("avg_confidence"):
            try:
                fig = plot_calibration_curve_precomputed(
                    calibration["bin_centers"],
                    calibration["avg_confidence"],
                    calibration.get("accuracy", []),
                    output_path=framework.experiment_dir / "03_calibration.png",
                )
                if fig is not None:
                    plt.close(fig)
                plots_generated.append("03_calibration.png")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to generate calibration plot: %s", exc)
                plots_failed.append({"plot": "03_calibration.png", "error": str(exc)})

        confidence_hist = val_summary.get("confidence_hist", {})
        if confidence_hist.get("bins"):
            try:
                fig = plot_confidence_histogram(
                    confidence_hist["bins"],
                    confidence_hist.get("real_hist", []),
                    confidence_hist.get("fake_hist", []),
                    output_path=framework.experiment_dir / "04_confidence_hist.png",
                )
                if fig is not None:
                    plt.close(fig)
                plots_generated.append("04_confidence_hist.png")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to generate confidence histogram: %s", exc)
                plots_failed.append({"plot": "04_confidence_hist.png", "error": str(exc)})

        try:
            fig = plot_train_val_gap(
                epoch_metrics_history,
                output_path=framework.experiment_dir / "05_train_val_gap.png",
            )
            if fig is not None:
                plt.close(fig)
            plots_generated.append("05_train_val_gap.png")
        except Exception as exc:
            logger.warning("Failed to generate train/val gap plot: %s", exc)
            plots_failed.append({"plot": "05_train_val_gap.png", "error": str(exc)})

        try:
            fig = plot_lr_schedule(
                epoch_metrics_history,
                output_path=framework.experiment_dir / "06_lr_schedule.png",
            )
            if fig is not None:
                plt.close(fig)
            plots_generated.append("06_lr_schedule.png")
        except Exception as exc:
            logger.warning("Failed to generate LR schedule plot: %s", exc)
            plots_failed.append({"plot": "06_lr_schedule.png", "error": str(exc)})

        try:
            if val_summary.get("topk_fp_paths") or val_summary.get("topk_fn_paths"):
                fig = plot_image_grid(
                    val_summary.get("topk_fp_paths", [])[:32] + val_summary.get("topk_fn_paths", [])[:32],
                    title="Error Samples (FP & FN)",
                    output_path=framework.experiment_dir / "07_error_samples.png",
                )
                if fig is not None:
                    plt.close(fig)
                    plots_generated.append("07_error_samples.png")

                if val_summary.get("topk_fp_paths"):
                    fig = plot_image_grid(
                        val_summary["topk_fp_paths"][:32],
                        title="Top False Positives",
                        output_path=framework.experiment_dir / "07a_top_fp_samples.png",
                    )
                    if fig is not None:
                        plt.close(fig)
                        plots_generated.append("07a_top_fp_samples.png")

                if val_summary.get("topk_fn_paths"):
                    fig = plot_image_grid(
                        val_summary["topk_fn_paths"][:32],
                        title="Top False Negatives",
                        output_path=framework.experiment_dir / "07b_top_fn_samples.png",
                    )
                    if fig is not None:
                        plt.close(fig)
                        plots_generated.append("07b_top_fn_samples.png")
        except Exception as exc:
            logger.warning("Failed to generate error sample grids: %s", exc)
            plots_failed.append({"plot": "07_error_samples.png", "error": str(exc)})

        try:
            if "targets_array" in val_summary and "probabilities_array" in val_summary:
                targets_np = np.asarray(val_summary["targets_array"])
                probs_np = np.asarray(val_summary["probabilities_array"]).reshape(-1)
                preds_np = np.asarray(
                    val_summary.get("predictions_array", (probs_np >= 0.5).astype(int))
                ).reshape(-1)

                real_probs = probs_np[targets_np == 0]
                fake_probs = probs_np[targets_np == 1]

                fig = plot_probability_distribution(
                    probabilities_real=real_probs,
                    probabilities_fake=fake_probs,
                    output_path=framework.experiment_dir / "08_probability_distribution.png",
                )
                if fig is not None:
                    plt.close(fig)
                    plots_generated.append("08_probability_distribution.png")

                fig = plot_error_analysis(
                    targets=targets_np,
                    predictions=preds_np,
                    probabilities=probs_np,
                    output_path=framework.experiment_dir / "09_error_analysis.png",
                )
                if fig is not None:
                    plt.close(fig)
                    plots_generated.append("09_error_analysis.png")
        except Exception as exc:
            logger.warning("Failed to generate probability/error analysis plots: %s", exc)
            plots_failed.append({"plot": "probability_analysis", "error": str(exc)})

        try:
            create_comprehensive_report(
                val_summary,
                output_dir=framework.experiment_dir,
            )
            plots_generated.extend(
                [
                    "metrics_summary.png",
                    "confusion_matrix.png",
                    "class_distribution.png",
                    "threshold_analysis.png",
                ]
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to create comprehensive report plots: %s", exc)
            plots_failed.append({"plot": "comprehensive_report", "error": str(exc)})

        summary["val_final"] = {
            "metrics": val_summary.get("metrics", {}),
            "loss": val_summary.get("loss"),
            "confusion_matrix": val_summary.get("confusion_matrix", {}),
            "total_samples": val_summary.get("total_samples"),
            "class_distribution": val_summary.get("class_distribution", {}),
            "probability_statistics": val_summary.get("probability_statistics", {}),
        }
        summary["evaluation_summary_file"] = evaluation_summary_path.name

        artifacts_dict = summary.setdefault("artifacts", {})
        artifacts_dict.update(
            {
                "plots": plots_generated,
                "plots_failed": plots_failed,
                "learning_curves": learning_curve_path.name,
                "best_model": "best_model.pth",
            }
        )
        artifacts_dict.setdefault("threshold_sweep_csv", None)
        artifacts_dict.setdefault("calibration_table_csv", None)

        # Save threshold sweep CSV if available (Phase B: B1 implementation)
        try:
            if "threshold_sweep_csv" in val_summary and val_summary["threshold_sweep_csv"]:
                threshold_sweep_csv_path = Path(framework.experiment_dir) / "threshold_sweep.csv"
                with open(threshold_sweep_csv_path, "w") as f:
                    f.write(val_summary["threshold_sweep_csv"])
                logger.info("Saved threshold sweep to: %s", threshold_sweep_csv_path)
                artifacts_dict["threshold_sweep_csv"] = "threshold_sweep.csv"
        except Exception as exc:
            logger.warning("Failed to save threshold sweep CSV: %s", exc)

        # Save calibration table CSV if available (Phase B: B2 implementation)
        try:
            if "calibration_table_csv" in val_summary and val_summary["calibration_table_csv"]:
                calibration_table_csv_path = Path(framework.experiment_dir) / "calibration_table.csv"
                with open(calibration_table_csv_path, "w") as f:
                    f.write(val_summary["calibration_table_csv"])
                logger.info("Saved calibration table to: %s", calibration_table_csv_path)
                artifacts_dict["calibration_table_csv"] = "calibration_table.csv"
        except Exception as exc:
            logger.warning("Failed to save calibration table CSV: %s", exc)

    except Exception as exc:
        logger.error("Post-training evaluation failed: %s", exc)
        plots_failed.append({"plot": "evaluation", "error": str(exc)})

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Updated training summary with evaluation artifacts: %s", summary_path)

    framework.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
