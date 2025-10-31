"""
AWARE-NET: Universal Experiment Framework (Stage 00 Integration)

This module provides the unified experiment tracking framework used across all stages.
Extracted from Stage 00 train_baseline.py and made reusable for Stage 01-06.

Key Features:
- Automatic timestamped experiment directories
- File-based logging and hyperparameter tracking
- Best model saving and checkpointing
- Reproducibility controls
- Cross-stage compatibility

Usage:
    framework = ExperimentFramework(
        output_dir="outputs/stage1",
        experiment_name="mobilenetv4_simple"
    )

    # Log training metrics
    framework.log_metrics(epoch, metrics, mode='train')
    framework.save_best_model(model, auc_score)
"""

import os
import sys
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


class ExperimentFramework:
    """
    Unified experiment tracking framework extracted from Stage 00 baseline.

    Provides consistent experiment management across all training stages.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        log_level: int = logging.INFO
    ):
        """
        Initialize experiment framework.

        Args:
            output_dir: Base directory for experiment outputs
            experiment_name: Name for this type of experiment
            log_level: Logging level
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.log_level = log_level

        # Create timestamped experiment directory (Stage 00 feature)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"run_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # File logging - create live-updating log.md
        self.log_path = self.experiment_dir / "log.md"

        # Create markdown header
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# Training Log\n\n")
            f.write(f"- **Experiment**: {self.experiment_name}\n")
            f.write(f"- **Run ID**: {self.timestamp}\n")
            f.write(f"- **Output Directory**: {self.experiment_dir}\n\n")
            f.write("---\n\n")

        # Attach FileHandler to root logger (captures all modules)
        self._file_handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
        self._file_handler.setLevel(self.log_level)
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        )
        logging.getLogger().addHandler(self._file_handler)

        # Training state
        self.best_metric = 0.0
        self.best_epoch = 0

        logger.info(f"Experiment initialized: {self.experiment_dir}")
        logger.info(f"Results will be saved to: {self.experiment_dir}")

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters to file for reproducibility.

        Args:
            hparams: Dictionary of hyperparameters
        """
        # Save hparams to file for reproducibility
        hparams_file = self.experiment_dir / "hyperparameters.json"
        with open(hparams_file, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'timestamp': self.timestamp,
                'hyperparameters': hparams
            }, f, indent=2, default=str)

        logger.info(f"Logged hyperparameters: {len(hparams)} parameters")

    def log_metrics(self, epoch: int, metrics: Dict[str, float], mode: str = 'train'):
        """
        Log training metrics to console.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric names and values
            mode: 'train', 'validation', or 'test'
        """
        # Log epoch summary to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch+1} [{mode.upper()}] {metrics_str}")

    def save_best_model(
        self,
        model: torch.nn.Module,
        metric_value: float,
        metric_name: str = 'auc',
        optimizer: Optional[torch.optim.Optimizer] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Save model if it achieves new best metric (Stage 00 feature).

        Args:
            model: Model to save
            metric_value: Current metric value
            metric_name: Name of the metric (e.g., 'auc', 'accuracy')
            optimizer: Optimizer state to save
            additional_info: Additional information to save
        """
        if metric_value > self.best_metric:
            self.best_metric = metric_value

            # Create save data
            save_data = {
                'epoch': self.best_epoch,
                'model_state_dict': model.state_dict(),
                'best_metric': metric_value,
                'metric_name': metric_name,
                'experiment_name': self.experiment_name,
                'timestamp': self.timestamp
            }

            # Add optimizer state if provided
            if optimizer is not None:
                save_data['optimizer_state_dict'] = optimizer.state_dict()

            # Add additional info if provided
            if additional_info is not None:
                save_data['additional_info'] = additional_info

            # Save model checkpoint
            checkpoint_path = self.experiment_dir / 'best_model.pth'
            torch.save(save_data, checkpoint_path)

            logger.info(f"✓ New best model saved! {metric_name.upper()}: {metric_value:.4f}")
        else:
            logger.debug(f"Model not saved. Current {metric_name}: {metric_value:.4f}, Best: {self.best_metric:.4f}")

    def set_best_epoch(self, epoch: int):
        """Set the current epoch number for model saving."""
        self.best_epoch = epoch

    def close(self):
        """Finalize experiment."""
        logger.info(f"Experiment completed. Best metric: {self.best_metric:.4f}")
        logger.info(f"Results saved to: {self.experiment_dir}")

        # Remove and close file handler to prevent duplicate logs
        if hasattr(self, "_file_handler"):
            root_logger = logging.getLogger()
            try:
                root_logger.removeHandler(self._file_handler)
            except Exception:
                pass  # Handler may have been already removed
            try:
                self._file_handler.close()
            except Exception:
                pass  # Handler may have been already closed

    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get comprehensive experiment information.

        Returns:
            Dictionary with experiment metadata
        """
        return {
            'experiment_name': self.experiment_name,
            'output_dir': str(self.output_dir),
            'experiment_dir': str(self.experiment_dir),
            'timestamp': self.timestamp,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }

    def create_metric_tracker(self):
        """
        Create a simple metric tracker for training progress.

        Returns:
            MetricTracker instance
        """
        return MetricTracker()


class MetricTracker:
    """
    Simple metric tracker for training progress monitoring.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metric tracker.

        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.metrics = {}

    def update(self, metric_name: str, value: float):
        """
        Update metric value.

        Args:
            metric_name: Name of the metric
            value: New metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append(value)

        # Keep only recent values
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name] = self.metrics[metric_name][-self.window_size:]

    def get_average(self, metric_name: str) -> float:
        """
        Get moving average of metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Moving average value
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])

    def get_latest(self, metric_name: str) -> float:
        """
        Get latest value of metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Latest metric value
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return self.metrics[metric_name][-1]


def setup_reproducible_environment(seed: int = 42):
    """
    Setup reproducible training environment (Stage 00 feature).

    Args:
        seed: Random seed for reproducibility
    """
    import random
    import numpy as np

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set deterministic behavior for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Reproducible environment setup with seed: {seed}")


# ==============================================================================
# CONFIGURATION LOADING & MERGING UTILITIES (Phase 1)
# ==============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file (.yaml, .yml, or .json)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If unsupported file format
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded YAML config: {config_path}")
            return config
        except ImportError:
            logger.warning("PyYAML not installed. Install with: pip install pyyaml")
            raise
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Loaded JSON config: {config_path}")
        return config
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def merge_configs(
    base_config: Dict[str, Any],
    cli_args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge configurations with CLI overrides.

    Priority: base_config → cli_args (CLI values override base)

    Args:
        base_config: Base configuration dictionary
        cli_args: CLI arguments dictionary (may contain nested keys with dots)

    Returns:
        Merged configuration

    Example:
        >>> base = {"lr": 0.001, "batch_size": 32}
        >>> cli = {"lr": 0.0001}
        >>> merge_configs(base, cli)
        {'lr': 0.0001, 'batch_size': 32}
    """
    config = dict(base_config)

    for key, value in cli_args.items():
        if value is None:
            continue  # Skip None values

        # Handle nested keys like "training.lr"
        if '.' in key:
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        else:
            config[key] = value

    logger.info(f"✅ Merged configurations")
    return config