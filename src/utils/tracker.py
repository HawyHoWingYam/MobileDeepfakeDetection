"""
AWARE-NET: Experiment Tracker Abstraction Layer (Phase 1)

Provides a unified tracker interface supporting:
- LocalTracker: Local file-based tracking (default)
- WandbTracker: Weights & Biases integration (optional)

Design philosophy: Implement TrackerInterface first, then specific backends.
This allows seamless switching between local development and collaborative
experiment tracking without modifying training code.
"""

import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment trackers.

    Defines a unified interface for logging metrics, artifacts, configs, etc.
    Allows swapping between different tracking backends (local, WandB, MLflow, etc.)
    without modifying training code.
    """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log scalar metrics.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Global step (e.g., epoch number)
        """
        pass

    @abstractmethod
    def log_artifact(self, file_path: Path, artifact_type: str = "file"):
        """
        Log a file artifact.

        Args:
            file_path: Path to file artifact
            artifact_type: Type of artifact (file, model, data, etc.)
        """
        pass

    @abstractmethod
    def log_figure(self, figure, name: str):
        """
        Log a matplotlib figure.

        Args:
            figure: matplotlib figure object
            name: Name for the figure
        """
        pass

    @abstractmethod
    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def finish(self):
        """
        Finish tracking and cleanup resources.

        Called at the end of training.
        """
        pass


class LocalTracker(ExperimentTracker):
    """
    Local file-based tracker (default).

    Lightweight no-op adapter. Actual work is handled by:
    - ExperimentManager (config, metadata)
    - MetricsCSVLogger (metrics.csv)
    - ArtifactSaver (artifact files)
    - Plotting utilities (figures)

    This tracker simply ensures compatibility with the unified interface.
    """

    def __init__(self, run_dir: Path):
        """
        Args:
            run_dir: Base run directory
        """
        self.run_dir = Path(run_dir)
        logger.info(f"📍 LocalTracker initialized for: {self.run_dir}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        No-op: Metrics already logged by MetricsCSVLogger.

        Args:
            metrics: Metrics dictionary
            step: Step number
        """
        # Actual logging handled by MetricsCSVLogger
        pass

    def log_artifact(self, file_path: Path, artifact_type: str = "file"):
        """
        No-op: Artifact already saved to run_dir.

        Args:
            file_path: Path to artifact
            artifact_type: Type of artifact
        """
        # File already in run_dir/artifacts/
        pass

    def log_figure(self, figure, name: str):
        """
        No-op: Figure already saved.

        Args:
            figure: matplotlib figure
            name: Figure name
        """
        # Figure already saved to run_dir/figures/
        pass

    def log_config(self, config: Dict[str, Any]):
        """
        No-op: Config already snapshotted.

        Args:
            config: Configuration dictionary
        """
        # Config already saved as config_snapshot.yaml
        pass

    def finish(self):
        """Finish tracking (no-op for local)."""
        logger.info(f"✅ LocalTracker finished")


class WandbTracker(ExperimentTracker):
    """
    Weights & Biases tracker integration.

    Syncs training data to wandb.ai for collaborative experiment tracking,
    interactive visualization, and advanced analysis.

    Installation: pip install wandb
    """

    def __init__(
        self,
        run_dir: Path,
        project: str,
        name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        tags: Optional[list] = None
    ):
        """
        Args:
            run_dir: Base run directory
            project: WandB project name
            name: WandB run name
            config: Configuration to log
            entity: WandB entity (username or team)
            tags: Tags for this run
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "WandB not installed. Install with: pip install wandb\n"
                "Or use LocalTracker (default) for local-only tracking."
            )

        self.run_dir = Path(run_dir)

        # Initialize WandB run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            dir=str(self.run_dir),
            entity=entity,
            tags=tags or [],
            reinit=False  # Prevent re-initialization
        )

        logger.info(f"🌐 WandbTracker initialized")
        logger.info(f"   Project: {project}")
        logger.info(f"   Run URL: {self.run.get_url()}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metrics
            step: Step number (epoch)
        """
        # Add step information
        log_dict = {**metrics, "step": step}
        self.run.log(log_dict, step=step)

    def log_artifact(self, file_path: Path, artifact_type: str = "file"):
        """
        Log artifact to WandB.

        Args:
            file_path: Path to artifact file
            artifact_type: Type of artifact
        """
        artifact = self.run.log_artifact(
            str(file_path),
            type=artifact_type
        )
        logger.info(f"📦 Artifact logged to WandB: {file_path.name}")

    def log_figure(self, figure, name: str):
        """
        Log matplotlib figure to WandB.

        Args:
            figure: matplotlib figure object
            name: Name for the figure
        """
        import wandb as wb
        self.run.log({name: wb.Image(figure)})

    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration to WandB.

        Args:
            config: Configuration dictionary
        """
        self.run.config.update(config)
        logger.info(f"⚙️  Configuration logged to WandB")

    def finish(self):
        """Finish WandB run."""
        self.run.finish()
        logger.info(f"✅ WandbTracker finished. Results available at: {self.run.get_url()}")
