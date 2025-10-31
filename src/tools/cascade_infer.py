#!/usr/bin/env python3
"""
Stage 4 Cascade Inference Tool (production‑ready)

- Loads Stage 1 and Stage 2 models
- Applies tuned thresholds from best_config.json
- Supports single image, directory, or manifest inference
- Reports cascade stats and optional CSV output

Usage examples:
  # Single image
  python src/tools/cascade_infer.py \
    --stage1-ckpt outputs/stage1/.../best_model.pth \
    --stage2-ckpt outputs/stage3/.../best_model.pth \
    --thresholds outputs/stage4/run_.../best_config.json \
    --image-path /path/to/image.jpg

  # Directory
  python src/tools/cascade_infer.py \
    --stage1-ckpt outputs/stage1/.../best_model.pth \
    --stage2-ckpt outputs/stage3/.../best_model.pth \
    --thresholds outputs/stage4/run_.../best_config.json \
    --image-dir /path/to/images \
    --output preds.csv

  # Manifest CSV (columns: image_path[,label])
  python src/tools/cascade_infer.py \
    --stage1-ckpt outputs/stage1/.../best_model.pth \
    --stage2-ckpt outputs/stage3/.../best_model.pth \
    --thresholds outputs/stage4/run_.../best_config.json \
    --manifest manifests/celebdf_v2_val_balanced.csv \
    --output preds.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mobilenetv4_model import create_mobilenetv4_simple
from models.efficientnetv2_model import create_baseline_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cascade_infer')


def build_transform(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])


class CascadeEngine:
    def __init__(
        self,
        stage1_ckpt: str,
        stage2_ckpt: str,
        thresholds_path: str,
        stage1_model: str = 'mobilenetv4_hybrid_medium',
        stage2_model: str = 'tf_efficientnetv2_b0',
        device: str = 'auto',
    ):
        if device == 'auto':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load thresholds
        cfg = json.loads(Path(thresholds_path).read_text())
        self.low = float(cfg['low_thresh'])
        self.high = float(cfg['high_thresh'])
        self.escalation_est = float(cfg.get('escalation_rate', 0.0))

        logger.info(f"Thresholds: low={self.low:.4f}, high={self.high:.4f}; expected escalation ~{self.escalation_est:.1%}")

        # Load models (eval)
        self.s1 = self._load_model(stage1_ckpt, stage1_model)
        self.s2 = self._load_model(stage2_ckpt, stage2_model)
        self.s1.eval(); self.s2.eval()

        # Preprocessing
        self.tr1 = build_transform(256)
        self.tr2 = build_transform(384)

        # Stats
        self.stats = {'total': 0, 's1_real': 0, 's1_fake': 0, 's2_used': 0}

    def _load_model(self, ckpt: str, model_name: str) -> torch.nn.Module:
        checkpoint = torch.load(ckpt, map_location=self.device)
        state = checkpoint.get('model_state_dict', checkpoint)
        if 'mobilenetv4' in model_name:
            model = create_mobilenetv4_simple(model_name=model_name, pretrained=False)
        else:
            model = create_baseline_model(model_name=model_name, pretrained=False)
        model.load_state_dict(state, strict=False)
        return model.to(self.device)

    @torch.no_grad()
    def infer_image(self, image_bgr: np.ndarray) -> Dict:
        # Stage 1
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x1 = self.tr1(image=img_rgb)['image'].unsqueeze(0).to(self.device)
        logit1 = self.s1(x1)
        p1 = torch.sigmoid(logit1.view(-1)).item()

        if p1 <= self.low:
            self.stats['s1_real'] += 1
            self.stats['total'] += 1
            return {'prediction': 0, 'confidence': p1, 'stage_used': 1, 'stage1_confidence': p1, 'stage2_confidence': None}
        if p1 >= self.high:
            self.stats['s1_fake'] += 1
            self.stats['total'] += 1
            return {'prediction': 1, 'confidence': p1, 'stage_used': 1, 'stage1_confidence': p1, 'stage2_confidence': None}

        # Stage 2
        x2 = self.tr2(image=img_rgb)['image'].unsqueeze(0).to(self.device)
        logit2 = self.s2(x2)
        p2 = torch.sigmoid(logit2.view(-1)).item()
        pred = int(p2 >= 0.5)
        self.stats['s2_used'] += 1
        self.stats['total'] += 1
        return {'prediction': pred, 'confidence': p2, 'stage_used': 2, 'stage1_confidence': p1, 'stage2_confidence': p2}

    def report(self) -> Dict:
        s = self.stats.copy()
        if s['total']:
            s['s2_rate'] = s['s2_used'] / s['total']
        return s


def main() -> int:
    ap = argparse.ArgumentParser(description='Stage 4 Cascade Inference')
    ap.add_argument('--stage1-ckpt', required=True)
    ap.add_argument('--stage2-ckpt', required=True)
    ap.add_argument('--thresholds', required=True)
    ap.add_argument('--stage1-model', default='mobilenetv4_hybrid_medium')
    ap.add_argument('--stage2-model', default='tf_efficientnetv2_b0')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--image-path')
    ap.add_argument('--image-dir')
    ap.add_argument('--manifest')
    ap.add_argument('--output')
    ap.add_argument('--no-progress', action='store_true', default=False,
                   help='Disable progress bar output')
    args = ap.parse_args()

    engine = CascadeEngine(
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        thresholds_path=args.thresholds,
        stage1_model=args.stage1_model,
        stage2_model=args.stage2_model,
        device=args.device,
    )

    rows: List[Dict] = []
    if args.image_path:
        img = cv2.imread(args.image_path)
        if img is None:
            logger.error(f"Cannot read image: {args.image_path}")
            return 1
        r = engine.infer_image(img)
        logger.info(f"Result: pred={'FAKE' if r['prediction'] else 'REAL'} conf={r['confidence']:.4f} stage={r['stage_used']}")
        rows.append({'path': args.image_path, **r})

    elif args.image_dir:
        p = Path(args.image_dir)
        imgs = list(p.glob('*.jpg')) + list(p.glob('*.png'))
        iterator = imgs if args.no_progress else tqdm(imgs, desc=f"Infer dir ({len(imgs)} imgs)")
        for fp in iterator:
            img = cv2.imread(str(fp))
            if img is None:
                logger.warning(f"Skip unreadable: {fp}")
                continue
            r = engine.infer_image(img)
            rows.append({'path': str(fp), **r})
        stats = engine.report()
        logger.info(f"Total={stats['total']} s2_used={stats['s2_used']} ({stats.get('s2_rate',0.0):.1%})")

    elif args.manifest:
        df = pd.read_csv(args.manifest)
        iterator = df.iterrows() if args.no_progress else tqdm(df.iterrows(), total=len(df), desc=f"Infer manifest ({len(df)} rows)")
        for _, row in iterator:
            path = row['image_path']
            img = cv2.imread(path)
            if img is None:
                logger.warning(f"Skip unreadable: {path}")
                continue
            r = engine.infer_image(img)
            rows.append({'path': path, 'label': int(row.get('label', -1)), **r})
        stats = engine.report()
        logger.info(f"Total={stats['total']} s2_used={stats['s2_used']} ({stats.get('s2_rate',0.0):.1%})")
    else:
        logger.error('Provide one of --image-path, --image-dir, --manifest')
        return 1

    if args.output and rows:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info(f"Saved predictions to {out}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
