"""
AWARE-NET: Centralized Experiment Manager (Phase 1 Architecture)

Coordinates all experiment management concerns:
- Configuration loading, merging, validation, and snapshots
- Metadata collection (git, hostname, timestamps)
- Artifact saving policies and rotation
- Metrics CSV logging
- Tracker integration (local or external like WandB)
- Directory structure management

This centralizes experiment management, keeping train_mobilenet.py focused
on training logic only.
"""

import json
import csv
import logging
import datetime
import subprocess
import socket
from pathlib import Path
from typing import Dict, Any, Optional, List

from .tracker import ExperimentTracker, LocalTracker

logger = logging.getLogger(__name__)


class ArtifactSaver:
    """
    Manages artifact saving with configurable policies:
    - on_best: Save when a specific metric improves
    - every_n_epochs: Periodic saving every N epochs
    - last_n_epochs: Keep only the last N epochs of large artifacts
    """

    def __init__(
        self,
        run_dir: Path,
        save_policy: Dict[str, Any],
        artifact_dir_name: str = "artifacts"
    ):
        """
        Args:
            run_dir: Base run directory
            save_policy: Dict with 'on_best', 'every_n_epochs', 'last_n_epochs'
            artifact_dir_name: Name of artifacts subdirectory
        """
        self.run_dir = Path(run_dir)
        self.artifact_dir = self.run_dir / artifact_dir_name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.on_best_metric = save_policy.get('on_best', 'f1')
        self.every_n_epochs = save_policy.get('every_n_epochs', 0)
        self.last_n_epochs = save_policy.get('last_n_epochs', 2)

        self.best_metric_value = -float('inf')
        self.saved_epochs = []

        logger.info(f"📦 ArtifactSaver initialized:")
        logger.info(f"   - on_best: {self.on_best_metric}")
        logger.info(f"   - every_n_epochs: {self.every_n_epochs}")
        logger.info(f"   - last_n_epochs: {self.last_n_epochs}")

    def should_save(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> bool:
        """
        Determine if artifacts should be saved for this epoch.

        Args:
            epoch: Current epoch number
            metrics: Current metrics dictionary
            is_best: Whether this is the best epoch so far

        Returns:
            Whether artifacts should be saved
        """
        # Check if on_best metric improved
        if is_best and self.on_best_metric in metrics:
            metric_val = metrics[self.on_best_metric]
            if metric_val > self.best_metric_value:
                self.best_metric_value = metric_val
                self._record_saved_epoch(epoch)
                return True

        # Check if periodic saving should occur
        if self.every_n_epochs > 0 and epoch % self.every_n_epochs == 0:
            self._record_saved_epoch(epoch)
            return True

        return False

    def _record_saved_epoch(self, epoch: int):
        """Record an epoch as saved and clean up old ones."""
        self.saved_epochs.append(epoch)

        # Keep only last_n_epochs
        if len(self.saved_epochs) > self.last_n_epochs:
            old_epoch = self.saved_epochs.pop(0)
            logger.info(f"🗑️  Cleaning up artifacts from epoch {old_epoch} (keeping last {self.last_n_epochs})")

    def get_artifact_path(self, epoch: int, suffix: str) -> Path:
        """
        Get the path for an artifact file.

        Args:
            epoch: Epoch number
            suffix: File suffix (e.g., 'predictions.parquet', 'summary.json')

        Returns:
            Full path to artifact
        """
        filename = f"evaluation_epoch_{epoch}_{suffix}"
        return self.artifact_dir / filename


class MetricsCSVLogger:
    """
    Manages writing training metrics to CSV in append mode.
    Columns: epoch, split, loss, acc, f1, auc, bal_acc, lr, epoch_time_s,
             grad_norm_global, grad_norm_head, grad_norm_backbone
    """

    FIELDNAMES = [
        'epoch', 'split', 'loss', 'acc', 'f1', 'auc', 'bal_acc',
        'lr', 'epoch_time_s', 'grad_norm_global', 'grad_norm_head',
        'grad_norm_backbone'
    ]

    def __init__(self, csv_path: Path):
        """
        Args:
            csv_path: Path to metrics CSV file
        """
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Write header if file doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            logger.info(f"📊 Created metrics CSV: {self.csv_path}")

    def append(self, epoch: int, metrics_dict: Dict[str, float], split: str = 'train'):
        """
        Append metrics for an epoch to the CSV.

        Args:
            epoch: Epoch number
            metrics_dict: Dictionary of metrics
            split: 'train' or 'validation'
        """
        row = {'epoch': epoch, 'split': split}

        # Add known fields, using 0 as default for missing ones
        for field in self.FIELDNAMES:
            if field not in ('epoch', 'split'):
                row[field] = metrics_dict.get(field, 0.0)

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

    def close(self):
        """No-op for compatibility."""
        pass




class ExperimentManager:
    """
    Centralized experiment management.

    Handles:
    1. Configuration loading, merging, validation, snapshots
    2. Run directory creation and structure
    3. Metadata collection (git, hostname, timestamps)
    4. Tracker initialization (LocalTracker or WandbTracker)
    5. ArtifactSaver coordination
    6. Metrics logging

    This keeps train_mobilenet.py focused on training logic only.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        cli_args: Dict[str, Any],
        output_dir: Path,
        experiment_name: str = "mobilenetv4_experiment"
    ):
        """
        Args:
            base_config: Base configuration dict (from YAML/JSON)
            cli_args: CLI arguments dict
            output_dir: Base output directory
            experiment_name: Name of experiment
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

        # 1. Create timestamped run directory
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

        logger.info(f"🚀 ExperimentManager initialized")
        logger.info(f"   Run directory: {self.run_dir}")

        # 2. Resolve configuration (defaults → file → CLI)
        self.config = self._resolve_config(base_config, cli_args)
        self._save_config_snapshot()

        # 3. Initialize tracker
        self.tracker = self._create_tracker()

        # 4. Initialize artifact saver
        self.artifact_saver = ArtifactSaver(
            self.run_dir,
            self.config.get('save_policy', {}),
            artifact_dir_name="artifacts"
        )

        # 5. Initialize metrics logger
        metrics_csv_path = self.run_dir / "artifacts" / "metrics.csv"
        self.metrics_logger = MetricsCSVLogger(metrics_csv_path)

        # 6. Collect and save metadata
        self.metadata = self._collect_metadata()
        self._save_metadata()

        # 7. Log configuration to tracker
        self.tracker.log_config(self.config)

        logger.info(f"✅ ExperimentManager fully initialized")

    def _resolve_config(
        self,
        base_config: Dict[str, Any],
        cli_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve configuration with priority: defaults → file → CLI.

        Args:
            base_config: Base configuration from file
            cli_args: CLI arguments

        Returns:
            Merged configuration
        """
        config = dict(base_config)  # Start with base

        # Override with CLI arguments
        for key, value in cli_args.items():
            if value is not None:  # Only override if CLI value provided
                if '.' in key:  # Nested key like "training.lr"
                    keys = key.split('.')
                    d = config
                    for k in keys[:-1]:
                        d = d.setdefault(k, {})
                    d[keys[-1]] = value
                else:
                    config[key] = value

        logger.info(f"📋 Configuration resolved")
        return config

    def _save_config_snapshot(self):
        """Save resolved configuration to config_snapshot.yaml."""
        import yaml

        snapshot_path = self.run_dir / "config_snapshot.yaml"
        try:
            with open(snapshot_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"💾 Config snapshot saved: {snapshot_path}")
        except ImportError:
            # Fallback to JSON if YAML not available
            snapshot_path = self.run_dir / "config_snapshot.json"
            with open(snapshot_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"💾 Config snapshot saved (JSON fallback): {snapshot_path}")

    def _create_tracker(self) -> ExperimentTracker:
        """Create appropriate tracker based on config."""
        tracker_config = self.config.get('tracker', {})
        tracker_type = tracker_config.get('type', 'local')

        if tracker_type == 'wandb':
            try:
                from .tracker import WandbTracker
                return WandbTracker(
                    run_dir=self.run_dir,
                    project=tracker_config.get('project', 'aware-net'),
                    name=f"{self.experiment_name}_{self.timestamp}",
                    config=self.config,
                    entity=tracker_config.get('entity', None)
                )
            except ImportError as e:
                logger.warning(f"WandB tracker unavailable ({e}), falling back to LocalTracker")
                return LocalTracker(self.run_dir)
        else:
            return LocalTracker(self.run_dir)

    def _collect_metadata(self) -> Dict[str, Any]:
        """
        Collect run metadata.

        Returns:
            Metadata dictionary
        """
        metadata = {
            'timestamp': self.timestamp,
            'run_dir': str(self.run_dir),
            'experiment_name': self.experiment_name,
            'hostname': socket.gethostname(),
            'start_time': datetime.datetime.now().isoformat()
        }

        # Try to get git commit
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd='/workspace/MobileDeepfakeDetection',
                stderr=subprocess.DEVNULL
            ).decode().strip()
            metadata['git_commit'] = git_commit
        except Exception:
            metadata['git_commit'] = 'unknown'

        # Try to get git branch
        try:
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd='/workspace/MobileDeepfakeDetection',
                stderr=subprocess.DEVNULL
            ).decode().strip()
            metadata['git_branch'] = git_branch
        except Exception:
            metadata['git_branch'] = 'unknown'

        logger.info(f"📌 Metadata collected: git_commit={metadata.get('git_commit')}, "
                   f"hostname={metadata['hostname']}")

        return metadata

    def _save_metadata(self):
        """Save metadata to meta.json."""
        meta_path = self.run_dir / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"📄 Metadata saved: {meta_path}")

    def log_epoch_metrics(self, epoch: int, metrics_dict: Dict[str, float], split: str = 'train'):
        """
        Log metrics for an epoch (unified interface).

        Args:
            epoch: Epoch number
            metrics_dict: Dictionary of metrics
            split: 'train' or 'validation'
        """
        self.metrics_logger.append(epoch, metrics_dict, split=split)
        self.tracker.log_metrics(metrics_dict, step=epoch)

    def should_save_artifacts(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> bool:
        """
        Determine if artifacts should be saved.

        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best epoch

        Returns:
            Whether to save artifacts
        """
        return self.artifact_saver.should_save(epoch, metrics, is_best)

    def save_evaluation_summary(self, epoch: int, eval_summary: Dict[str, Any]):
        """
        Save the integrated evaluation_summary.json.

        Args:
            epoch: Epoch number
            eval_summary: Evaluation summary dictionary
        """
        summary_path = self.run_dir / f"artifacts/evaluation_summary_epoch_{epoch}.json"
        with open(summary_path, 'w') as f:
            json.dump(eval_summary, f, indent=2)

        self.tracker.log_artifact(summary_path)
        logger.info(f"💾 Evaluation summary saved: {summary_path}")

    def finalize(self):
        """Finalize experiment and cleanup."""
        self.metadata['end_time'] = datetime.datetime.now().isoformat()
        self._save_metadata()

        self.metrics_logger.close()
        self.tracker.finish()

        logger.info(f"✅ Experiment finalized. Results: {self.run_dir}")
