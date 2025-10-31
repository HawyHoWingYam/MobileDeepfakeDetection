#!/usr/bin/env python3
"""
AWARE-NET Stage 4: Comprehensive Threshold Tuning for Two-Stage Cascade System

Implements end-to-end threshold optimization for MobileNetV4 (Stage 1) → EfficientNetV2 (Stage 3)
cascade with:
- Advanced caching strategies (precomputed stage1/stage2 logits)
- Confidence-based decision making with temperature scaling
- Comprehensive grid search with constraint-based filtering
- Detailed result persistence (CSV, JSON, plots, heatmaps)
- Latency estimation and analysis
- Reproducible run management with timestamped directories

Usage:
    python src/tools/tune_cascade_system.py \
        --stage1-ckpt outputs/stage1/run_YYYY.../best_model.pth \
        --stage2-ckpt outputs/stage3/run_YYYY.../best_model.pth \
        --manifest manifests/celebdf_v2_val_balanced.csv \
        --output-dir outputs/stage4 \
        --low-start 0.05 --low-stop 0.45 --high-start 0.55 --high-stop 0.95 \
        --step 0.05 --min-accuracy 0.90 --min-f1 0.90 --primary-metric fnr
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mobilenetv4_model import create_mobilenetv4_simple
from models.efficientnetv2_model import create_baseline_model
from training.dataset import CelebDFDataset
from utils.evaluation import ModelEvaluator
from utils.experiment_framework import setup_reproducible_environment
from utils.plotting import plot_roc_curve, plot_confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CascadeDetector:
    """
    Two-stage cascade detector with caching and confidence-based decisions.

    Architecture:
    - Stage 1: MobileNetV4 (fast, ~50ms per batch)
    - Stage 2: EfficientNetV2 (precise, ~100ms per batch)

    Decision logic:
    - High confidence (p >= high_thresh): Use Stage 1 decision
    - Low confidence (p <= low_thresh): Use Stage 1 decision
    - Ambiguous (low < p < high): Escalate to Stage 2
    """

    def __init__(
        self,
        stage1_ckpt: str,
        stage2_ckpt: str,
        device: torch.device,
        stage1_size: int = 256,
        stage2_size: int = 384,
        temperature: float = 1.0,
        target_class: int = 1,  # Class to compute confidence for (1=fake)
        amp: bool = False,
        dataset: CelebDFDataset | None = None,
        stage1_model_name: str = "mobilenetv4_hybrid_medium",
        stage2_model_name: str = "tf_efficientnetv2_b0",
    ):
        """
        Initialize cascade detector.

        Args:
            stage1_ckpt: Path to Stage 1 MobileNetV4 checkpoint
            stage2_ckpt: Path to Stage 3 EfficientNetV2 checkpoint
            device: Computation device
            stage1_size: Input size for Stage 1
            stage2_size: Input size for Stage 2
            temperature: Temperature for confidence scaling
            target_class: Class index to compute confidence for
            amp: Whether to use automatic mixed precision
        """
        self.device = device
        self.stage1_size = stage1_size
        self.stage2_size = stage2_size
        self.temperature = temperature
        self.target_class = target_class
        self.amp = amp

        # Persist dataset reference for lazy Stage 2 (optional path reload)
        self.dataset = dataset

        # Load models
        logger.info("Loading Stage 1 (MobileNetV4)...")
        self.stage1_model = self._load_model(stage1_ckpt, stage1_model_name)

        logger.info("Loading Stage 2 (EfficientNetV2)...")
        self.stage2_model = self._load_model(stage2_ckpt, stage2_model_name)

        # Setup transform pipelines
        self.stage1_transform = self._build_transform(stage1_size)
        self.stage2_transform = self._build_transform(stage2_size)

        # Cache storage
        self.stage1_logits_cache = None
        self.stage2_logits_cache = None
        self.cache_indices = None
        self._stage2_memo: dict[int, np.ndarray] = {}

        logger.info(f"CascadeDetector initialized on {device}")

    def _load_model(self, ckpt_path: str, model_name: str) -> nn.Module:
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Create model
            if 'mobilenetv4' in model_name.lower():
                model = create_mobilenetv4_simple(model_name=model_name, pretrained=False)
            else:
                model = create_baseline_model(model_name=model_name, pretrained=False)

            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device).eval()

            logger.info(f"Loaded {model_name} from {Path(ckpt_path).name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _build_transform(self, size: int):
        """Build inference transform pipeline (no augmentation)."""
        import cv2
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

 

    def confidence(self, logits: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Compute confidence probability for target class.

        Args:
            logits: Raw model outputs (batch_size, 1) or (batch_size, 2)
            temperature: Optional temperature override

        Returns:
            Confidence scores for target class (batch_size,)
        """
        if temperature is None:
            temperature = self.temperature

        # Handle both single logit and binary logit outputs
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            # Single logit - use sigmoid
            logits = logits.view(-1)
            return torch.sigmoid(logits / temperature)
        else:
            # Binary logits - use softmax
            probs = torch.softmax(logits / temperature, dim=1)
            return probs[:, self.target_class]

    def precompute_logits(
        self,
        dataloader: DataLoader,
        precompute_stage2: bool = True,
    ) -> None:
        """
        Precompute logits for all samples in dataloader.

        Args:
            dataloader: Validation dataloader
            precompute_stage2: Whether to precompute Stage 2 logits
        """
        logger.info("Precomputing Stage 1 logits...")
        stage1_logits_list = []
        stage2_logits_list = [] if precompute_stage2 else None

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Precomputing logits")):
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 3:
                        images, labels, meta = batch
                    else:
                        images, labels = batch[:2]
                else:
                    images = batch

                images = images.to(self.device)

                # Stage 1 inference
                if self.amp:
                    with torch.autocast(device_type=self.device.type):
                        stage1_logits = self.stage1_model(images)
                else:
                    stage1_logits = self.stage1_model(images)

                stage1_logits_list.append(stage1_logits.detach().cpu())

                # Stage 2 inference (optional) - upsample to stage2_size before forward
                if precompute_stage2:
                    imgs2 = torch.nn.functional.interpolate(
                        images,
                        size=(self.stage2_size, self.stage2_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if self.amp:
                        with torch.autocast(device_type=self.device.type):
                            stage2_logits = self.stage2_model(imgs2)
                    else:
                        stage2_logits = self.stage2_model(imgs2)
                    stage2_logits_list.append(stage2_logits.detach().cpu())

        # Concatenate all logits
        self.stage1_logits_cache = torch.cat(stage1_logits_list, dim=0)
        if precompute_stage2:
            self.stage2_logits_cache = torch.cat(stage2_logits_list, dim=0)

        logger.info(f"Precomputed Stage 1: {self.stage1_logits_cache.shape}")
        if precompute_stage2:
            logger.info(f"Precomputed Stage 2: {self.stage2_logits_cache.shape}")

    def predict_batch(
        self,
        low_thresh: float,
        high_thresh: float,
        sample_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform cascade predictions for a set of samples.

        Args:
            low_thresh: Lower confidence threshold
            high_thresh: Upper confidence threshold
            sample_indices: Optional indices to use from cache

        Returns:
            Dictionary with predictions and metadata
        """
        if self.stage1_logits_cache is None:
            raise RuntimeError("Logits not precomputed. Call precompute_logits first.")

        # Use all cached logits or subset
        if sample_indices is not None:
            stage1_logits = self.stage1_logits_cache[sample_indices]
            stage2_logits = self.stage2_logits_cache[sample_indices] if self.stage2_logits_cache is not None else None
        else:
            stage1_logits = self.stage1_logits_cache
            stage2_logits = self.stage2_logits_cache

        # Compute Stage 1 confidence
        stage1_conf = self.confidence(stage1_logits)
        stage1_conf_np = stage1_conf.numpy()

        # Decision masks
        accept_real = stage1_conf_np <= low_thresh  # p_fake low → predict real
        accept_fake = stage1_conf_np >= high_thresh  # p_fake high → predict fake
        ambiguous = ~(accept_real | accept_fake)    # Need Stage 2

        # Initialize predictions and scores
        n = len(stage1_conf_np)
        final_preds = np.zeros(n, dtype=int)
        final_scores = np.zeros(n, dtype=float)

        # Stage 1 decisions for confident samples
        final_preds[accept_real] = 0
        final_scores[accept_real] = stage1_conf_np[accept_real]

        final_preds[accept_fake] = 1
        final_scores[accept_fake] = stage1_conf_np[accept_fake]

        # Stage 2 decisions for ambiguous samples
        escalation_indices = np.where(ambiguous)[0]
        if len(escalation_indices) > 0:
            if stage2_logits is not None:
                stage2_conf = self.confidence(stage2_logits[escalation_indices])
                stage2_conf_np = stage2_conf.numpy()
            else:
                # Lazy Stage 2: compute for escalated indices only (reload images)
                stage2_conf_np = self._compute_stage2_conf_for_indices(escalation_indices)

            final_preds[escalation_indices] = (stage2_conf_np >= 0.5).astype(int)
            final_scores[escalation_indices] = stage2_conf_np

        return {
            'predictions': final_preds,
            'scores': final_scores,
            'stage1_conf': stage1_conf_np,
            'escalation_mask': ambiguous,
            'escalation_count': len(escalation_indices),
            'escalation_rate': len(escalation_indices) / n,
        }

    def _compute_stage2_conf_for_indices(self, indices: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Run Stage 2 only for selected dataset indices, with memoization.

        Loads images from disk using dataset.df['image_path'], applies stage2_transform,
        and runs batched inference. Results are memoized per index.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset reference not set; cannot perform lazy Stage 2 inference.")

        import cv2
        conf_out = np.zeros(len(indices), dtype=np.float32)

        # Determine which indices still need to be computed
        to_compute = []
        map_pos: list[tuple[int, int]] = []  # (global_idx, local_pos)
        for pos, idx in enumerate(indices.tolist()):
            cached = self._stage2_memo.get(int(idx))
            if cached is not None:
                conf_out[pos] = float(cached)
            else:
                to_compute.append(int(idx))
                map_pos.append((int(idx), pos))

        if not to_compute:
            return conf_out

        # Batch loading and inference
        # Build list of transformed tensors
        imgs: list[torch.Tensor] = []
        for i in to_compute:
            row = self.dataset.df.iloc[i]
            img_path = Path(self.dataset.root_path) / str(row['image_path'])
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Could not load image for Stage 2 lazy eval: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = self.stage2_transform(image=img)
            imgs.append(transformed['image'])

        # Stack and run in chunks
        imgs_tensor = torch.stack(imgs, dim=0).to(self.device)
        probs: list[float] = []
        with torch.no_grad():
            for start in range(0, imgs_tensor.shape[0], batch_size):
                end = min(start + batch_size, imgs_tensor.shape[0])
                batch = imgs_tensor[start:end]
                if self.amp:
                    with torch.autocast(device_type=self.device.type):
                        logits = self.stage2_model(batch)
                else:
                    logits = self.stage2_model(batch)
                p = self.confidence(logits).detach().cpu().numpy().astype(np.float32)
                probs.extend(p.tolist())

        # Write back to outputs and memo
        for (gidx, pos), p in zip(map_pos, probs):
            conf_out[pos] = p
            self._stage2_memo[gidx] = float(p)

        return conf_out


# -------------------------------
# Cache persistence helpers
# -------------------------------

def _try_load_caches(
    cache_dir: Path,
    signature: Dict[str, Any],
    cascade: 'CascadeDetector',
    dataset: CelebDFDataset,
) -> bool:
    """Try to load stage1/stage2 logits from disk if signature and ordering match."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sig_path = cache_dir / 'signature.json'
        idx_path = cache_dir / 'index_to_path.json'
        s1_path = cache_dir / 'stage1_logits.npy'
        s2_path = cache_dir / 'stage2_logits.npy'

        if not (sig_path.exists() and idx_path.exists() and s1_path.exists()):
            return False

        on_disk_sig = json.loads(sig_path.read_text())
        if on_disk_sig != signature:
            logger.info("Cache signature mismatch; skipping persisted cache")
            return False

        # Verify ordering
        on_disk_index = json.loads(idx_path.read_text())
        cur_index = dataset.df['image_path'].astype(str).tolist()
        if on_disk_index != cur_index:
            logger.info("Cache index order mismatch; skipping persisted cache")
            return False

        # Load stage1
        s1 = np.load(s1_path)
        cascade.stage1_logits_cache = torch.from_numpy(s1)
        logger.info(f"Loaded persisted Stage 1 logits: {cascade.stage1_logits_cache.shape}")

        # Stage2 optional
        if s2_path.exists():
            s2 = np.load(s2_path)
            cascade.stage2_logits_cache = torch.from_numpy(s2)
            logger.info(f"Loaded persisted Stage 2 logits: {cascade.stage2_logits_cache.shape}")
        return True
    except Exception as exc:
        logger.warning(f"Failed to load caches: {exc}")
        return False


def _save_caches(
    cache_dir: Path,
    signature: Dict[str, Any],
    cascade: 'CascadeDetector',
    dataset: CelebDFDataset,
    include_stage2: bool,
) -> None:
    """Save stage1 (and optional stage2) logits along with signature and index mapping."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sig_path = cache_dir / 'signature.json'
        idx_path = cache_dir / 'index_to_path.json'
        s1_path = cache_dir / 'stage1_logits.npy'
        s2_path = cache_dir / 'stage2_logits.npy'

        # Write signature and index mapping
        sig_path.write_text(json.dumps(signature, indent=2))
        idx = dataset.df['image_path'].astype(str).tolist()
        idx_path.write_text(json.dumps(idx))

        # Save numpy arrays
        if cascade.stage1_logits_cache is not None:
            np.save(s1_path, cascade.stage1_logits_cache.detach().cpu().numpy())
        if include_stage2 and cascade.stage2_logits_cache is not None:
            np.save(s2_path, cascade.stage2_logits_cache.detach().cpu().numpy())
        logger.info(f"Persisted caches to {cache_dir}")
    except Exception as exc:
        logger.warning(f"Failed to save caches: {exc}")


class ThresholdGridSearch:
    """
    Comprehensive threshold grid search with metrics evaluation.
    """

    def __init__(
        self,
        cascade: CascadeDetector,
        y_true: np.ndarray,
        batch_size: int = 64,
        device: torch.device = None,
    ):
        """
        Initialize grid search.

        Args:
            cascade: CascadeDetector instance
            y_true: Ground truth labels
            batch_size: Batch size for inference
            device: Computation device
        """
        self.cascade = cascade
        self.y_true = y_true
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        self.evaluator = ModelEvaluator(self.device)

        # Latency measurements
        self.stage1_time_per_sample = None
        self.stage2_time_per_sample = None

    def measure_latency(self, dataloader: DataLoader, num_warmup: int = 5) -> None:
        """
        Measure per-sample latency for stages.

        Args:
            dataloader: Validation dataloader (first batch used)
            num_warmup: Number of warmup iterations
        """
        logger.info("Measuring latency...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                batch_size = images.shape[0]

                # Warmup
                for _ in range(num_warmup):
                    _ = self.cascade.stage1_model(images)
                    if self.cascade.device.type == 'cuda':
                        torch.cuda.synchronize()

                # Measure Stage 1
                torch.cuda.reset_peak_memory_stats() if self.cascade.device.type == 'cuda' else None
                start = time.perf_counter()
                for _ in range(10):
                    _ = self.cascade.stage1_model(images)
                if self.cascade.device.type == 'cuda':
                    torch.cuda.synchronize()
                stage1_time = (time.perf_counter() - start) / 10
                self.stage1_time_per_sample = stage1_time / batch_size

                # Measure Stage 2 (upsampled to stage2_size)
                imgs2 = torch.nn.functional.interpolate(
                    images,
                    size=(self.cascade.stage2_size, self.cascade.stage2_size),
                    mode="bilinear",
                    align_corners=False,
                )
                start = time.perf_counter()
                for _ in range(10):
                    _ = self.cascade.stage2_model(imgs2)
                if self.cascade.device.type == 'cuda':
                    torch.cuda.synchronize()
                stage2_time = (time.perf_counter() - start) / 10
                self.stage2_time_per_sample = stage2_time / batch_size

                logger.info(f"Stage 1: {self.stage1_time_per_sample*1000:.2f}ms/sample")
                logger.info(f"Stage 2: {self.stage2_time_per_sample*1000:.2f}ms/sample")
                break

    def evaluate_thresholds(
        self,
        low_vals: np.ndarray,
        high_vals: np.ndarray,
        min_accuracy: float = 0.90,
        min_f1: float = 0.90,
        primary_metric: str = 'fnr',
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate all threshold combinations.

        Args:
            low_vals: Low threshold values to test
            high_vals: High threshold values to test
            min_accuracy: Minimum accuracy constraint
            min_f1: Minimum F1 constraint
            primary_metric: Primary metric to optimize (fnr, f1, balanced, etc.)

        Returns:
            DataFrame with all results and best configuration
        """
        logger.info(f"Grid search: {len(low_vals)} x {len(high_vals)} = {len(low_vals)*len(high_vals)} combinations")

        results = []
        best_value = float('inf') if primary_metric == 'fnr' else float('-inf')
        best_config = None

        with tqdm(total=len(low_vals)*len(high_vals), desc="Grid search") as pbar:
            for low_thresh in low_vals:
                for high_thresh in high_vals:
                    if low_thresh >= high_thresh:
                        pbar.update(1)
                        continue

                    # Get predictions
                    result = self.cascade.predict_batch(low_thresh, high_thresh)
                    y_pred = result['predictions']
                    y_score = result['scores']

                    # Compute metrics
                    metrics = self._compute_metrics(y_pred, y_score)

                    # Check constraints
                    passes_constraints = (
                        metrics['accuracy'] >= min_accuracy and
                        metrics['f1'] >= min_f1
                    )

                    # Update best config
                    if passes_constraints:
                        metric_value = metrics[primary_metric]
                        is_better = (metric_value < best_value) if primary_metric == 'fnr' else (metric_value > best_value)

                        if is_better:
                            best_value = metric_value
                            best_config = {
                                'low_thresh': low_thresh,
                                'high_thresh': high_thresh,
                                'metrics': metrics,
                                'escalation_rate': result['escalation_rate'],
                            }

                    # Store result
                    results.append({
                        'low_thresh': low_thresh,
                        'high_thresh': high_thresh,
                        'auc': metrics['auc'],
                        'f1': metrics['f1'],
                        'accuracy': metrics['accuracy'],
                        'fnr': metrics['fnr'],
                        'fpr': metrics['fpr'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'escalation_rate': result['escalation_rate'],
                        'estimated_latency_ms': self._estimate_latency(result['escalation_rate']),
                        'passes_constraints': passes_constraints,
                    })

                    pbar.update(1)

        results_df = pd.DataFrame(results)

        if best_config is None:
            logger.warning("No configuration passed constraints. Using best FNR anyway.")
            idx = results_df['fnr'].idxmin()
            best_config = {
                'low_thresh': results_df.loc[idx, 'low_thresh'],
                'high_thresh': results_df.loc[idx, 'high_thresh'],
                'metrics': results_df.loc[idx].to_dict(),
                'escalation_rate': results_df.loc[idx, 'escalation_rate'],
            }

        return results_df, best_config

    def _compute_metrics(self, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        from sklearn.metrics import (
            roc_auc_score, f1_score, accuracy_score, precision_score,
            recall_score, confusion_matrix
        )

        auc = roc_auc_score(self.y_true, y_score)
        f1 = f1_score(self.y_true, y_pred)
        accuracy = accuracy_score(self.y_true, y_pred)
        precision = precision_score(self.y_true, y_pred)
        recall = recall_score(self.y_true, y_pred)

        # FNR = FN / (FN + TP) = 1 - recall for positive class
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            'auc': auc,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fnr': fnr,
            'fpr': fpr,
        }

    def _estimate_latency(self, escalation_rate: float) -> float:
        """Estimate total latency in milliseconds."""
        if self.stage1_time_per_sample is None:
            return None

        n = len(self.y_true)
        latency_s = (
            n * self.stage1_time_per_sample +
            n * escalation_rate * self.stage2_time_per_sample
        )
        return latency_s * 1000  # Convert to ms


def create_plots(results_df: pd.DataFrame, best_config: Dict[str, Any], output_dir: Path) -> None:
    """
    Create comprehensive visualization plots.

    Args:
        results_df: DataFrame with all threshold combinations
        best_config: Best configuration
        output_dir: Output directory for plots
    """
    logger.info("Creating visualization plots...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Heatmap - FNR
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_fnr = results_df.pivot_table(
        index='low_thresh', columns='high_thresh', values='fnr', aggfunc='mean'
    )
    sns.heatmap(pivot_fnr, cmap='RdYlGn_r', annot=True, fmt='.3f', ax=ax, cbar_kws={'label': 'FNR'})
    ax.set_title('FNR Heatmap (Lower is Better)')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_fnr.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Heatmap - F1
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_f1 = results_df.pivot_table(
        index='low_thresh', columns='high_thresh', values='f1', aggfunc='mean'
    )
    sns.heatmap(pivot_f1, cmap='YlGn', annot=True, fmt='.3f', ax=ax, cbar_kws={'label': 'F1'})
    ax.set_title('F1 Score Heatmap (Higher is Better)')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_f1.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Heatmap - Escalation Rate
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_esc = results_df.pivot_table(
        index='low_thresh', columns='high_thresh', values='escalation_rate', aggfunc='mean'
    )
    sns.heatmap(pivot_esc, cmap='Blues', annot=True, fmt='.1%', ax=ax, cbar_kws={'label': 'Escalation Rate'})
    ax.set_title('Stage 2 Escalation Rate Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_escalation_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Scatter - Accuracy vs FNR
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        results_df['accuracy'], results_df['fnr'],
        c=results_df['escalation_rate'], cmap='viridis', s=100, alpha=0.6
    )
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('FNR')
    ax.set_title('Accuracy vs FNR (colored by escalation rate)')
    plt.colorbar(scatter, ax=ax, label='Escalation Rate')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_accuracy_vs_fnr.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Scatter - F1 vs FNR
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        results_df['f1'], results_df['fnr'],
        c=results_df['escalation_rate'], cmap='plasma', s=100, alpha=0.6
    )
    ax.set_xlabel('F1 Score')
    ax.set_ylabel('FNR')
    ax.set_title('F1 Score vs FNR (colored by escalation rate)')
    plt.colorbar(scatter, ax=ax, label='Escalation Rate')
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_f1_vs_fnr.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Best config metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = best_config['metrics']
    metric_names = ['auc', 'f1', 'accuracy', 'precision', 'recall']
    metric_values = [metrics.get(name, 0) for name in metric_names]

    bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('Score')
    ax.set_title(f"Best Configuration Metrics\n(low={best_config['low_thresh']:.2f}, high={best_config['high_thresh']:.2f})")
    ax.set_ylim([0, 1.0])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'best_config_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("Plots saved to output directory")


def write_run_readme(
    output_dir: Path,
    stage1_ckpt: str,
    stage2_ckpt: str,
    manifest: str,
    best_config: Dict[str, Any],
    results_df: pd.DataFrame,
    grid_search_args: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    n_samples: int,
) -> None:
    """Write comprehensive README for the run."""

    readme_text = f"""# Stage 4: Cascade Threshold Tuning Run

**Run Date**: {datetime.now().isoformat()}

## Configuration

### Checkpoints
- Stage 1 (MobileNetV4): {Path(stage1_ckpt).name}
- Stage 2 (EfficientNetV2): {Path(stage2_ckpt).name}

### Validation Data
- Manifest: {Path(manifest).name}
- Total samples: {n_samples}

### Inference Settings
- Device: {device}
- Batch Size: {batch_size}
- AMP: {grid_search_args.get('amp', True)}
- Seed: {grid_search_args.get('seed', 42)}

### Grid Search Parameters
- Low threshold range: [{grid_search_args['low_start']}, {grid_search_args['low_stop']}]
- High threshold range: [{grid_search_args['high_start']}, {grid_search_args['high_stop']}]
- Step size: {grid_search_args['step']}
- Total combinations: {len(results_df)}

### Constraints
- Minimum Accuracy: {grid_search_args.get('min_accuracy', 0.90)}
- Minimum F1: {grid_search_args.get('min_f1', 0.90)}
- Primary Metric: {grid_search_args.get('primary_metric', 'fnr')}

## Best Configuration

**Thresholds**:
- Low (Stage 1 → Real): {best_config['low_thresh']:.4f}
- High (Stage 1 → Fake): {best_config['high_thresh']:.4f}

**Performance Metrics**:
"""

    for metric_name, metric_value in best_config['metrics'].items():
        if isinstance(metric_value, float):
            readme_text += f"- {metric_name.upper()}: {metric_value:.4f}\n"

    # Cascade efficiency section with safe latency rendering
    try:
        row = results_df[(results_df['low_thresh'] == best_config['low_thresh']) &
                         (results_df['high_thresh'] == best_config['high_thresh'])].iloc[0]
        est_ms = row.get('estimated_latency_ms', None)
    except Exception:
        est_ms = None

    latency_line = f"- Estimated Latency: {est_ms:.1f}ms" if isinstance(est_ms, (int, float)) else "- Estimated Latency: N/A (enable --measure-latency)"

    readme_text += f"""
**Cascade Efficiency**:
- Stage 2 Escalation Rate: {best_config['escalation_rate']:.2%}
{latency_line}

## Output Files

- `metrics_grid.csv` - All threshold combinations with metrics
- `metrics_grid.json` - Same data in JSON format
- `best_config.json` - Selected thresholds and metrics
- `heatmap_*.png` - Visualization heatmaps (FNR, F1, escalation rate)
- `scatter_*.png` - Scatter plots for metric relationships
- `best_config_metrics.png` - Bar chart of best config metrics

## Reproduction

To reproduce this run:
```bash
python src/tools/tune_cascade_system.py \\
  --stage1-ckpt {Path(stage1_ckpt).name} \\
  --stage2-ckpt {Path(stage2_ckpt).name} \\
  --manifest {Path(manifest).name} \\
  --output-dir outputs/stage4 \\
  --low-start {grid_search_args['low_start']} \\
  --low-stop {grid_search_args['low_stop']} \\
  --high-start {grid_search_args['high_start']} \\
  --high-stop {grid_search_args['high_stop']} \\
  --step {grid_search_args['step']} \\
  --min-accuracy {grid_search_args.get('min_accuracy', 0.90)} \\
  --min-f1 {grid_search_args.get('min_f1', 0.90)} \\
  --primary-metric {grid_search_args.get('primary_metric', 'fnr')} \\
  --seed {grid_search_args.get('seed', 42)}
```

## Notes

- Thresholds should be applied to the confidence score for the "fake" class
- Samples with p_fake < low_thresh are classified as Real (Stage 1)
- Samples with p_fake > high_thresh are classified as Fake (Stage 1)
- Samples with low_thresh <= p_fake <= high_thresh are escalated to Stage 2
- All results are saved with full reproducibility in mind

"""

    (output_dir / 'README.txt').write_text(readme_text)
    logger.info(f"README saved: {output_dir / 'README.txt'}")


def main():
    """Main entry point for threshold tuning."""

    parser = argparse.ArgumentParser(
        description='AWARE-NET Stage 4: Cascade Threshold Tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument('--stage1-ckpt', type=str, required=True,
                       help='Path to Stage 1 (MobileNetV4) checkpoint')
    parser.add_argument('--stage2-ckpt', type=str, required=True,
                       help='Path to Stage 2 (EfficientNetV2) checkpoint')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to validation manifest CSV')

    # Output directory
    parser.add_argument('--output-dir', type=str, default='outputs/stage4',
                       help='Base output directory (default: outputs/stage4)')

    # Threshold search parameters
    parser.add_argument('--low-start', type=float, default=0.05,
                       help='Low threshold range start (default: 0.05)')
    parser.add_argument('--low-stop', type=float, default=0.45,
                       help='Low threshold range stop (default: 0.45)')
    parser.add_argument('--high-start', type=float, default=0.55,
                       help='High threshold range start (default: 0.55)')
    parser.add_argument('--high-stop', type=float, default=0.95,
                       help='High threshold range stop (default: 0.95)')
    parser.add_argument('--step', type=float, default=0.05,
                       help='Step size for threshold search (default: 0.05)')

    # Constraints
    parser.add_argument('--min-accuracy', type=float, default=0.90,
                       help='Minimum accuracy constraint (default: 0.90)')
    parser.add_argument('--min-f1', type=float, default=0.90,
                       help='Minimum F1 constraint (default: 0.90)')
    parser.add_argument('--primary-metric', type=str, default='fnr',
                       choices=['fnr', 'f1', 'auc', 'accuracy'],
                       help='Primary metric to optimize (default: fnr)')

    # Inference settings
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference (default: 64)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for computation (default: auto; picks cuda:0 if available else cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers (default: 4)')
    parser.add_argument('--pin-memory', action='store_true', default=False,
                       help='Pin memory in dataloader')
    parser.add_argument('--amp', action='store_true', default=False,
                       help='Use automatic mixed precision')

    # Model configuration
    parser.add_argument('--stage1-size', type=int, default=256,
                       help='Input size for Stage 1 (default: 256)')
    parser.add_argument('--stage2-size', type=int, default=384,
                       help='Input size for Stage 2 (default: 384)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for confidence scaling (default: 1.0)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # Pre-compute options
    parser.add_argument('--precompute-stage2', action='store_true', default=False,
                       help='Precompute Stage 2 logits (default: False; enable for fastest grid search)')
    parser.add_argument('--measure-latency', action='store_true', default=False,
                       help='Measure inference latency (default: False)')

    # Optional on-disk cache for repeated tuning on the same data
    parser.add_argument('--persist-cache', action='store_true', default=False,
                       help='Persist precomputed logits to disk for reuse across runs (default: False)')
    parser.add_argument('--cache-dir', type=str, default='outputs/stage4/cache',
                       help='Directory to store/load persisted caches (default: outputs/stage4/cache)')

    # Model variants
    parser.add_argument('--stage1-model', type=str, default='mobilenetv4_hybrid_medium',
                       help='Stage 1 model variant (default: mobilenetv4_hybrid_medium)')
    parser.add_argument('--stage2-model', type=str, default='tf_efficientnetv2_b0',
                       help='Stage 2 model variant (default: tf_efficientnetv2_b0)')

    args = parser.parse_args()

    # Setup
    setup_reproducible_environment(args.seed)
    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Create output directory with timestamp
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_base / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    try:
        # Load validation dataset
        logger.info(f"Loading validation dataset: {args.manifest}")
        dataset = CelebDFDataset(
            manifest_path=args.manifest,
            image_size=args.stage1_size,  # Will be resized in cascade
            augmentation=False,
            normalize=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        logger.info(f"Dataset: {len(dataset)} samples")

        # Initialize cascade detector
        cascade = CascadeDetector(
            stage1_ckpt=args.stage1_ckpt,
            stage2_ckpt=args.stage2_ckpt,
            device=device,
            stage1_size=args.stage1_size,
            stage2_size=args.stage2_size,
            temperature=args.temperature,
            amp=args.amp,
            dataset=dataset,
            stage1_model_name=args.stage1_model,
            stage2_model_name=args.stage2_model,
        )

        # Attempt to load persisted caches (optional)
        def _signature_dict() -> Dict[str, Any]:
            man = Path(args.manifest)
            try:
                mtime = man.stat().st_mtime
                size = man.stat().st_size
            except Exception:
                mtime = None
                size = None
            return {
                'stage1_ckpt': str(Path(args.stage1_ckpt).resolve()),
                'stage2_ckpt': str(Path(args.stage2_ckpt).resolve()),
                'stage1_model': args.stage1_model,
                'stage2_model': args.stage2_model,
                'stage1_size': int(args.stage1_size),
                'stage2_size': int(args.stage2_size),
                'manifest': str(man.resolve()),
                'manifest_mtime': mtime,
                'manifest_size': size,
                'dataset_length': len(dataset),
                'normalize': True,
            }

        caches_loaded = False
        if args.persist_cache:
            caches_loaded = _try_load_caches(
                cache_dir=Path(args.cache_dir),
                signature=_signature_dict(),
                cascade=cascade,
                dataset=dataset,
            )

        # Precompute logits if needed
        if not caches_loaded:
            cascade.precompute_logits(
                dataloader,
                precompute_stage2=args.precompute_stage2,
            )
            if args.persist_cache:
                _save_caches(
                    cache_dir=Path(args.cache_dir),
                    signature=_signature_dict(),
                    cascade=cascade,
                    dataset=dataset,
                    include_stage2=bool(args.precompute_stage2),
                )

        # Extract ground truth labels deterministically from manifest
        y_true = dataset.df['label'].to_numpy().astype(int)
        logger.info(f"Labels: {(y_true == 0).sum()} real, {(y_true == 1).sum()} fake")

        # Initialize grid search
        grid_search = ThresholdGridSearch(
            cascade=cascade,
            y_true=y_true,
            batch_size=args.batch_size,
            device=device,
        )

        # Measure latency (optional)
        if args.measure_latency:
            grid_search.measure_latency(dataloader)

        # Run grid search
        logger.info("Starting threshold grid search...")
        low_vals = np.arange(args.low_start, args.low_stop + args.step, args.step)
        high_vals = np.arange(args.high_start, args.high_stop + args.step, args.step)

        results_df, best_config = grid_search.evaluate_thresholds(
            low_vals=low_vals,
            high_vals=high_vals,
            min_accuracy=args.min_accuracy,
            min_f1=args.min_f1,
            primary_metric=args.primary_metric,
        )

        # Save results
        logger.info("Saving results...")
        results_df.to_csv(output_dir / 'metrics_grid.csv', index=False)
        results_df.to_json(output_dir / 'metrics_grid.json', orient='records', indent=2)

        # Save best config
        best_config_save = {
            'low_thresh': best_config['low_thresh'],
            'high_thresh': best_config['high_thresh'],
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                       for k, v in best_config['metrics'].items()},
            'escalation_rate': float(best_config['escalation_rate']),
        }
        (output_dir / 'best_config.json').write_text(json.dumps(best_config_save, indent=2))

        # Create plots
        create_plots(results_df, best_config, output_dir)

        # Write README
        write_run_readme(
            output_dir=output_dir,
            stage1_ckpt=args.stage1_ckpt,
            stage2_ckpt=args.stage2_ckpt,
            manifest=args.manifest,
            best_config=best_config,
            results_df=results_df,
            grid_search_args=vars(args),
            device=device,
            batch_size=args.batch_size,
            n_samples=len(y_true),
        )

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("THRESHOLD TUNING SUMMARY")
        logger.info("="*60)
        logger.info(f"Best Low Threshold: {best_config['low_thresh']:.4f}")
        logger.info(f"Best High Threshold: {best_config['high_thresh']:.4f}")
        logger.info(f"Stage 2 Escalation Rate: {best_config['escalation_rate']:.2%}")
        logger.info("\nBest Metrics:")
        for metric_name, metric_value in best_config['metrics'].items():
            if isinstance(metric_value, float):
                logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")
        logger.info("="*60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"Error during threshold tuning: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
