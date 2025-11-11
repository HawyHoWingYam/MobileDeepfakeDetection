#!/usr/bin/env python3
"""
Export Stage 1/2 models to TorchScript for mobile/edge deployment.

Goals (Stage 6 minimal path):
- Load trained checkpoints for Stage 1 (MobileNetV4) and/or Stage 2 (EfficientNetV2)
- Optionally apply dynamic quantization to Linear layers (keeps conv float)
- Trace to TorchScript with a fixed input size
- Save small deployment bundle metadata (thresholds/temperature) alongside models

Examples:
  # Export both stages with dynamic quantization
  python -m src.tools.export_torchscript \
    --stage both \
    --stage1-ckpt outputs/stage1/run_20251023_034316/best_model.pth \
    --stage2-ckpt outputs/stage5/finetune_s2_r2/run_20251108_043257/best_model.pth \
    --stage1-model mobilenetv4_hybrid_medium \
    --stage2-model tf_efficientnetv2_b0 \
    --stage1-size 256 --stage2-size 384 \
    --thresholds outputs/stage4/run_20251103_093455/best_config.json \
    --temperature 4.8354101181 \
    --quantize-dynamic true \
    --output-dir outputs/stage6/export_ts

  # Export only Stage 2 (expert) without quantization
  python -m src.tools.export_torchscript \
    --stage stage2 \
    --stage2-ckpt outputs/stage5/finetune_s2/run_*/best_model.pth \
    --stage2-model tf_efficientnetv2_b0 \
    --stage2-size 384 \
    --quantize-dynamic false \
    --output-dir outputs/stage6/export_ts
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Ensure repository src on path then import model factories
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from models.mobilenetv4_model import create_mobilenetv4_simple  # noqa: E402
from models.efficientnetv2_model import create_baseline_model   # noqa: E402

logger = logging.getLogger("export_torchscript")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def str2bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true","t","1","yes","y"}: return True
    if s in {"false","f","0","no","n"}: return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def load_model_and_ckpt(
    stage: str,
    ckpt_path: Optional[str],
    model_name: str,
    device: torch.device,
) -> nn.Module:
    """Instantiate model by stage and load checkpoint state_dict (strict=False)."""
    if stage == "stage1":
        model = create_mobilenetv4_simple(model_name=model_name, pretrained=False).to(device)
    elif stage == "stage2":
        model = create_baseline_model(model_name=model_name, pretrained=False).to(device)
    else:
        raise ValueError(f"Unknown stage '{stage}'")

    if not ckpt_path:
        raise ValueError(f"Checkpoint path required for {stage}")
    ckpt_path = str(ckpt_path)
    logger.info("Loading checkpoint for %s: %s", stage, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys while loading %s: %s", stage, missing)
    if unexpected:
        logger.warning("Unexpected keys while loading %s: %s", stage, unexpected)
    model.eval()
    return model


def to_torchscript(
    model: nn.Module,
    input_size: int,
    device: torch.device,
    quantize_dynamic: bool = True,
    trace_only: bool = False,
) -> torch.jit.ScriptModule:
    """Export model to TorchScript via tracing.

    Strategy:
    1) Try dynamic quantization (Linear) + trace
    2) If trace fails, retry in fp32 (no quantization)
    3) If still fails and trace_only=False, fallback to scripting; otherwise raise
    """
    m = model
    # Step 0: prepare example
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        # Step 1: quantized trace
        if quantize_dynamic:
            try:
                m_q = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
                ts = torch.jit.trace(m_q, dummy, strict=False, check_trace=False)
                ts = torch.jit.freeze(ts)
                logger.info("Traced with dynamic quantization")
                return ts
            except Exception as exc:
                logger.warning("Trace failed with quantization (%s); retrying in fp32", exc)

        # Step 2: fp32 trace
        try:
            ts = torch.jit.trace(model, dummy, strict=False, check_trace=False)
            ts = torch.jit.freeze(ts)
            logger.info("Traced in fp32 (no quantization)")
            return ts
        except Exception as exc:
            if trace_only:
                logger.error("Trace failed in fp32 as well: %s", exc)
                raise
            logger.info("Falling back to scripting due to trace error: %s", exc)
            # Step 3: fallback script (may fail for some python constructs)
            ts = torch.jit.script(model)
            ts = torch.jit.freeze(ts)
            return ts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Stage 1/2 to TorchScript")
    p.add_argument("--stage", default="both", choices=["stage1","stage2","both"], help="Which stage to export")
    p.add_argument("--stage1-ckpt", default=None, help="Path to Stage 1 checkpoint (best_model.pth)")
    p.add_argument("--stage2-ckpt", default=None, help="Path to Stage 2 checkpoint (best_model.pth)")
    p.add_argument("--stage1-model", default="mobilenetv4_hybrid_medium")
    p.add_argument("--stage2-model", default="tf_efficientnetv2_b0")
    p.add_argument("--stage1-size", type=int, default=256)
    p.add_argument("--stage2-size", type=int, default=384)
    p.add_argument("--thresholds", default=None, help="best_config.json with low/high for cascade (optional)")
    p.add_argument("--temperature", type=float, default=1.0, help="Stage 2 temperature for deployment (optional)")
    p.add_argument("--quantize-dynamic", type=str2bool, default=True)
    p.add_argument("--trace-only", type=str2bool, default=True,
                   help="If true, do not fallback to scripting; fail fast if tracing fails.")
    p.add_argument("--device", default="cpu", help="Export device (cpu recommended)")
    p.add_argument("--output-dir", default="outputs/stage6/export_ts")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = {}
    if args.stage in ("stage1", "both"):
        s1 = load_model_and_ckpt("stage1", args.stage1_ckpt, args.stage1_model, device)
        ts1 = to_torchscript(s1, args.stage1_size, device, quantize_dynamic=args.quantize_dynamic, trace_only=args.trace_only)
        s1_path = out_dir / "stage1_mobilenetv4_ts.pt"
        ts1.save(str(s1_path))
        exported["stage1_model"] = str(s1_path)
        exported["stage1_model_name"] = args.stage1_model
        exported["stage1_size"] = int(args.stage1_size)
        logger.info("Saved Stage 1 TorchScript: %s", s1_path)

    if args.stage in ("stage2", "both"):
        s2 = load_model_and_ckpt("stage2", args.stage2_ckpt, args.stage2_model, device)
        ts2 = to_torchscript(s2, args.stage2_size, device, quantize_dynamic=args.quantize_dynamic, trace_only=args.trace_only)
        s2_path = out_dir / "stage2_efficientnetv2_ts.pt"
        ts2.save(str(s2_path))
        exported["stage2_model"] = str(s2_path)
        exported["stage2_model_name"] = args.stage2_model
        exported["stage2_size"] = int(args.stage2_size)
        exported["stage2_temperature"] = float(args.temperature)
        logger.info("Saved Stage 2 TorchScript: %s", s2_path)

    # Bundle minimal cascade metadata
    meta = {
        "thresholds_path": args.thresholds if args.thresholds else None,
        "low_thresh": None,
        "high_thresh": None,
        "stage2_temperature": float(args.temperature),
        **exported,
    }
    try:
        if args.thresholds and Path(args.thresholds).exists():
            cfg = json.loads(Path(args.thresholds).read_text())
            meta["low_thresh"] = float(cfg.get("low_thresh", 0.03))
            meta["high_thresh"] = float(cfg.get("high_thresh", 0.55))
    except Exception as exc:
        logger.warning("Failed to read thresholds: %s", exc)

    (out_dir / "bundle_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote metadata: %s", out_dir / "bundle_meta.json")

    print("Export complete. Artifacts:")
    for k, v in exported.items():
        print(f" - {k}: {v}")
    print(" - metadata:", out_dir / "bundle_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
