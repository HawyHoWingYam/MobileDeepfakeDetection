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
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

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
from training.train_mobilenet import create_multi_dataset_loader  # reuse val loader


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
    patience: int = 5
    min_delta: float = 0.0005
    restore_best: bool = True
    save_on_metric: str = "f1"
    output_dir: Path = Path("outputs/stage3")
    device: str = "auto"
    num_workers: int = 12
    image_size: Optional[int] = 256
    dataset_balance: str = "equal_by_dataset"  # none|equal_by_dataset


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def make_train_loader(cfg: TrainConfig) -> Tuple[DataLoader, Optional[np.ndarray]]:
    """Create a training loader from the difficult subset manifest.

    If dataset_balance == equal_by_dataset and the manifest has a 'dataset' column,
    use a WeightedRandomSampler to equalize dataset contributions per epoch.
    Returns (loader, weights_used or None).
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
    )

    sampler = None
    weights = None
    if cfg.dataset_balance == "equal_by_dataset" and "dataset" in df.columns:
        counts = df["dataset"].value_counts().to_dict()
        # Equalize datasets: each dataset shares 1/K of samples; per-sample weight inversely proportional to its dataset count
        inv = df["dataset"].map(lambda x: 1.0 / counts.get(x, 1))
        # Normalize weights to mean 1.0 (not necessary but stable)
        inv = inv / inv.mean()
        weights = inv.astype(float).values
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        logger.info("Using WeightedRandomSampler to equalize datasets: %s", counts)

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 03: Train EfficientNetV2 expert on difficult subset")
    parser.add_argument("--train_manifest", type=str, default="manifests/train_difficult_subset.csv")
    parser.add_argument("--model_name", type=str, default="tf_efficientnetv2_b0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0005)
    parser.add_argument("--restore_best", type=str2bool, default=True)
    parser.add_argument("--save_on_metric", type=str, default="f1")
    parser.add_argument("--output_dir", type=str, default="outputs/stage3")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dataset_balance", type=str, default="equal_by_dataset", choices=["none","equal_by_dataset"])

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
    )

    # Experiment directory
    framework = ExperimentFramework(output_dir=str(cfg.output_dir), experiment_name="stage3_efficientnet_expert")
    logger.info("Stage 03 outputs: %s", framework.experiment_dir)

    # Model
    model = create_baseline_model(pretrained=True, dropout_rate=0.2, model_name=cfg.model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Data
    train_loader, _ = make_train_loader(cfg)

    # Validation loader (reuse Stage 01 loaders, but we only need val)
    _, val_loader, _ = create_multi_dataset_loader(
        config_path="configs/datasets.json",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        seed=42,
        override_image_size=cfg.image_size,
    )

    evaluator = ModelEvaluator(device=device)

    best_metric = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        epoch_loss = 0.0
        n_batches = 0

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{(epoch_loss/max(n_batches,1)):.4f}"})

        # Validation
        val_metrics = evaluator.evaluate_model(model, val_loader, criterion=criterion, mode="validation")
        metric_name = cfg.save_on_metric
        metric_value = float(val_metrics.get(metric_name, 0.0))
        framework.log_metrics(epoch, {"train_loss": epoch_loss / max(n_batches,1)})
        framework.log_metrics(epoch, val_metrics, mode="validation")
        framework.save_best_model(model, metric_value, metric_name=metric_name, optimizer=optimizer,
                                  additional_info={"hyperparameters": {
                                      "model_name": cfg.model_name,
                                      "learning_rate": cfg.learning_rate,
                                      "batch_size": cfg.batch_size,
                                      "weight_decay": cfg.weight_decay,
                                  }})

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
                break

    # Restore best
    if best_state is not None and cfg.restore_best:
        model.load_state_dict(best_state)
        logger.info("Restored best weights from epoch %d (AUC=%.4f)", best_epoch+1, best_metric)

    # Save simple summary
    summary = {
        "training_completed": True,
        "best_val_auc": float(best_metric),
        "best_epoch": int(best_epoch+1),
        "epochs_configured": int(cfg.epochs),
        "train_manifest": str(cfg.train_manifest),
        "model_name": cfg.model_name,
    }
    with open(Path(framework.experiment_dir) / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved: %s", Path(framework.experiment_dir)/"training_summary.json")

    framework.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
