#!/usr/bin/env python3
"""
Calibrate Stage 2 temperature (binary) on a given manifest by minimizing NLL.

It optimizes a single positive scalar T so that BCEWithLogitsLoss(logits/T, y)
on the calibration set (e.g., val_all) is minimized. The resulting temperature
can be used in Stage 4 tuning via --temperature, and for inference (if the
inference tool supports a stage2 temperature flag).

Usage (example):
  python src/tools/calibrate_temperature.py \
    --stage2-ckpt outputs/stage5/run_20251031_041013/best_model.pth \
    --manifest outputs/stage4/manifests/val_all.csv \
    --stage2-model tf_efficientnetv2_b0 \
    --stage2-size 384 --batch-size 256 --device cuda:0 \
    --output outputs/stage4/calibration.json

Then, apply in Stage 4 tuning:
  python src/tools/tune_cascade_system.py ... --temperature <VALUE>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# import repo modules
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from models.efficientnetv2_model import create_baseline_model  # noqa: E402
from training.dataset import CelebDFDataset  # noqa: E402

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Please install albumentations and opencv-python") from exc


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('calibrate_temperature')


def build_transform(size: int):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def load_logits_and_labels(
    model: nn.Module,
    manifest: str,
    size: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Stage 2 model once on the manifest and collect logits/labels."""
    ds = CelebDFDataset(
        manifest_path=manifest,
        image_size=size,
        augmentation=False,
        normalize=True,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logits_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        iterator = tqdm(dl, total=len(dl), desc="Collecting logits", unit="batch") if show_progress else dl
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.view(-1).to(device)
            out = model(images)
            out = out.view(-1)
            logits_list.append(out.detach().cpu())
            labels_list.append(labels.detach().cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return logits, labels


def load_cached_logits_and_labels(
    logits_npy: str,
    manifest: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load cached stage2 logits from npy and labels from manifest CSV."""
    import pandas as pd
    arr = np.load(logits_npy)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    logits = torch.from_numpy(arr.astype(np.float32))
    df = pd.read_csv(manifest)
    if 'label' not in df.columns:
        raise ValueError("Manifest must contain 'label' column for calibration")
    labels = torch.from_numpy(df['label'].to_numpy().astype(np.float32))
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"Logits count ({logits.shape[0]}) != labels count ({labels.shape[0]})")
    return logits, labels


def optimize_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_steps: int = 300,
    lr: float = 0.01,
    device: torch.device = torch.device('cpu'),
) -> Tuple[float, float, float]:
    """Optimize scalar temperature T>0 by minimizing BCEWithLogitsLoss(logits/T, y)."""
    logits = logits.to(device)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Optimize over log_T to keep T>0
    log_T = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=50, line_search_fn='strong_wolfe')

    with torch.no_grad():
        base_loss = criterion(logits, labels).item()

    def closure():  # LBFGS requires a closure
        opt.zero_grad(set_to_none=True)
        T = torch.exp(log_T)  # positive
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    # First LBFGS, then a few Adam steps if needed
    opt.step(closure)
    # polish with Adam
    adam = torch.optim.Adam([log_T], lr=lr)
    for _ in range(max_steps):
        adam.zero_grad(set_to_none=True)
        T = torch.exp(log_T)
        loss = criterion(logits / T, labels)
        loss.backward()
        adam.step()

    T = float(torch.exp(log_T).item())
    final_loss = criterion(logits / torch.tensor(T, device=device), labels).item()
    return T, base_loss, final_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Calibrate Stage 2 temperature on a manifest')
    p.add_argument('--stage2-ckpt', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--stage2-model', default='tf_efficientnetv2_b0')
    p.add_argument('--stage2-size', type=int, default=384)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--stage2-logits-npy', type=str, default=None, help='Use cached stage2 logits (skip inference)')
    p.add_argument('--device', default='auto')
    p.add_argument('--max-steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--output', default='outputs/stage4/calibration.json')
    p.add_argument('--no-progress', action='store_true', default=False, help='Disable progress bars during calibration')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if (args.device == 'auto' and torch.cuda.is_available()) else args.device)

    if args.stage2_logits_npy:
        logits, labels = load_cached_logits_and_labels(args.stage2_logits_npy, args.manifest)
        logger.info("Loaded cached logits: %s and labels: %s", tuple(logits.shape), tuple(labels.shape))
        logits = logits.to(device)
        labels = labels.to(device)
    else:
        # Load Stage 2 model
        ckpt = torch.load(args.stage2_ckpt, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model = create_baseline_model(model_name=args.stage2_model, pretrained=False)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

        # Collect logits/labels
        logits, labels = load_logits_and_labels(
            model,
            args.manifest,
            args.stage2_size,
            args.batch_size,
            device,
            show_progress=not args.no_progress,
        )
        logger.info("Collected logits/labels: %s, %s", tuple(logits.shape), tuple(labels.shape))

    # Optimize temperature
    T, base_nll, final_nll = optimize_temperature(logits, labels, max_steps=args.max_steps, lr=args.lr, device=device)
    logger.info("Optimal temperature: %.6f | base NLL: %.6f -> final NLL: %.6f", T, base_nll, final_nll)

    out = {
        'stage2_temperature': float(T),
        'stage2_ckpt': str(Path(args.stage2_ckpt).resolve()),
        'manifest': str(Path(args.manifest).resolve()),
        'stage2_model': args.stage2_model,
        'stage2_size': int(args.stage2_size),
        'batch_size': int(args.batch_size),
        'base_nll': float(base_nll),
        'final_nll': float(final_nll),
        'device': str(device),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved calibration: %s", out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
