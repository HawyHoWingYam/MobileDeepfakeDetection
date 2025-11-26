#!/usr/bin/env python3
"""
Stage 2 Model Calibration Script - calibrate_stage2.py
======================================================

Temperature scaling calibration for the Stage 2 EfficientNetV2-B3 expert.

This mirrors the Stage 1 calibration flow:
 1) Load the trained Stage 2 checkpoint and validation manifest.
 2) Run inference to collect logits and labels.
 3) Fit a single temperature parameter T by minimising NLL.
 4) Save T to a small JSON file for later use (PC analysis or mobile export).

Usage:
    python src/stage2/calibrate_stage2.py \\
      --model_path output/stage2/effnet/best_model.pth \\
      --data_dir processed_data \\
      --val_manifest processed_data/manifests/val_manifest.csv
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm

import timm


def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / f"calibration_stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


class ManifestDataset(Dataset):
    """Minimal dataset for calibration / evaluation based on a manifest CSV."""

    def __init__(self, manifest_path: str, data_root: str, transform=None):
        import pandas as pd

        self.data_root = Path(data_root)
        self.transform = transform

        df = pd.read_csv(manifest_path)
        valid_rows = []
        for _, row in df.iterrows():
            img_path = self.data_root / row["image_path"]
            if img_path.exists():
                valid_rows.append(row)
        self.rows = valid_rows

        logging.info(
            f"Loaded {len(self.rows)} samples for calibration from {manifest_path}"
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = self.data_root / row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            img = Image.new("RGB", (256, 256), color="black")

        if self.transform:
            img = self.transform(img)

        label = float(row["label"])
        return img, label


def collect_logits_and_labels(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Calib Inference [Stage 2]"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    return logits_tensor, labels_tensor


def optimise_temperature(
    logits: torch.Tensor, labels: torch.Tensor, bounds=(0.1, 10.0)
) -> dict:
    """Fit temperature T by minimising NLL on validation logits."""
    logits_np = logits.numpy()
    labels_np = labels.numpy()

    def nll_objective(temp: float) -> float:
        scaled_logits = torch.tensor(logits_np / temp, dtype=torch.float32)
        target = torch.tensor(labels_np, dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(scaled_logits, target)
        return loss.item()

    result = minimize(
        lambda t: nll_objective(t[0]),
        x0=[1.0],
        bounds=[bounds],
        method="L-BFGS-B",
    )

    optimal_temp = float(result.x[0])
    logging.info("Stage 2 temperature calibration:")
    logging.info(f"  Optimal T: {optimal_temp:.4f}")
    logging.info(f"  Success: {result.success}")
    logging.info(f"  Final NLL: {result.fun:.6f}")

    return {
        "optimal_temperature": optimal_temp,
        "success": bool(result.success),
        "final_objective": float(result.fun),
    }


def evaluate_with_temperature(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> dict:
    """Compute basic metrics before/after temperature scaling."""
    probs_raw = torch.sigmoid(logits).numpy()
    probs_calib = torch.sigmoid(logits / temperature).numpy()
    y = labels.numpy().astype(int)

    def metrics_for(p):
        preds = (p > 0.5).astype(int)
        auc = roc_auc_score(y, p)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)
        return {"auc": float(auc), "accuracy": float(acc), "f1": float(f1)}

    return {
        "before": metrics_for(probs_raw),
        "after": metrics_for(probs_calib),
        "temperature": float(temperature),
    }


def main():
    ap = argparse.ArgumentParser(description="Stage 2 Temperature Scaling Calibration")
    ap.add_argument(
        "--model_path",
        type=str,
        default="output/stage2/effnet/best_model.pth",
        help="Path to Stage 2 checkpoint (best_model.pth)",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="efficientnetv2_b3.in21k_ft_in1k",
        help="timm model name for Stage 2",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default="processed_data",
        help="Root directory for processed data",
    )
    ap.add_argument(
        "--val_manifest",
        type=str,
        default="processed_data/manifests/val_manifest.csv",
        help="Validation manifest CSV path",
    )
    ap.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for calibration inference"
    )
    ap.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="output/stage2/effnet",
        help="Output directory for calibration results",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info("=== Stage 2 Temperature Calibration ===")
    logging.info(f"Arguments: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    if not Path(args.model_path).exists():
        logging.error(f"Checkpoint not found: {args.model_path}")
        return 1

    logging.info(f"Loading model from: {args.model_path}")
    model = timm.create_model(args.model_name, pretrained=False, num_classes=1)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    # Data
    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = ManifestDataset(args.val_manifest, args.data_dir, val_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Collect logits
    logits, labels = collect_logits_and_labels(model, dataloader, device)

    # Optimise temperature
    calib_info = optimise_temperature(logits, labels)
    T_opt = calib_info["optimal_temperature"]

    # Evaluate before/after
    metrics = evaluate_with_temperature(logits, labels, T_opt)

    # Save calibration JSON
    calib_json_path = output_dir / "calibration_temp_stage2.json"
    with calib_json_path.open("w") as f:
        json.dump(
            {
                "optimal_temperature": T_opt,
                "metrics": metrics,
                "optimization": calib_info,
            },
            f,
            indent=2,
        )

    logging.info(f"Saved Stage 2 calibration to: {calib_json_path}")
    logging.info(f"Temperature: {T_opt:.4f}")
    logging.info(f"Before metrics: {metrics['before']}")
    logging.info(f"After metrics:  {metrics['after']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

