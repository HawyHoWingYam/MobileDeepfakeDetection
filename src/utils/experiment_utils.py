"""
AWARE-NET Experiment Management and Reproducibility Tools
Comprehensive experiment tracking and reproducible research framework
"""

import os
import json
import pickle
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
import torch
import random
import warnings
from contextlib import contextmanager

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    experiment_name: str
    model_name: str
    dataset_name: str
    version: str = "1.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 50
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    
    # Model parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data parameters
    image_size: int = 256
    augmentation: bool = True
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Paths
    data_path: str = ""
    output_path: str = "experiments"
    
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class ExperimentResult:
    """Results from experiment execution"""
    experiment_id: str
    config: ExperimentConfig
    
    # Performance metrics
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics: Dict[str, List[float]] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model info
    best_epoch: int = 0
    best_val_score: float = 0.0
    total_training_time: float = 0.0
    
    # Paths
    model_path: str = ""
    log_path: str = ""
    
    # System info
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    finished_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    success: bool = True
    error_message: str = ""

class ExperimentManager:
    """
    Comprehensive experiment management system
    
    Features:
    - Automatic experiment tracking and logging
    - Reproducible seed management
    - Model checkpoint handling
    - Performance monitoring
    - Result comparison and analysis
    """
    
    def __init__(self, 
                 base_path: Union[str, Path] = "experiments",
                 auto_save: bool = True):
        """
        Initialize experiment manager
        
        Args:
            base_path: Base directory for experiments
            auto_save: Whether to automatically save results
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        
        # Current experiment tracking
        self.current_experiment = None
        self.current_result = None
        
        # Experiment registry
        self.registry_path = self.base_path / "experiment_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load registry: {e}")
        
        return {"experiments": {}, "metadata": {"created": datetime.datetime.now().isoformat()}}
    
    def _save_registry(self):
        """Save experiment registry"""
        self.registry["metadata"]["updated"] = datetime.datetime.now().isoformat()
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID"""
        # Create hash from config and timestamp
        config_str = json.dumps(asdict(config), sort_keys=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        hash_obj = hashlib.md5(config_str.encode())
        config_hash = hash_obj.hexdigest()[:8]
        
        return f"{config.experiment_name}_{timestamp}_{config_hash}"
    
    def setup_reproducibility(self, seed: int = 42, deterministic: bool = True):
        """
        Setup reproducible environment
        
        Args:
            seed: Random seed
            deterministic: Whether to use deterministic algorithms
        """
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Set CUBLAS workspace config for deterministic behavior
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # For newer PyTorch versions
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
        
        # Set environment variables
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print(f"Reproducibility setup complete (seed: {seed})")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create new experiment with configuration
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        # Generate unique ID
        experiment_id = self.generate_experiment_id(config)
        
        # Create experiment directory
        exp_dir = self.base_path / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        
        # Setup reproducibility
        self.setup_reproducibility(config.seed, config.deterministic)
        
        # Save configuration
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Initialize result tracking
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            system_info=self._get_system_info()
        )
        
        # Update registry
        self.registry["experiments"][experiment_id] = {
            "config": asdict(config),
            "created_at": config.created_at,
            "status": "running",
            "directory": str(exp_dir)
        }
        self._save_registry()
        
        # Set as current experiment
        self.current_experiment = experiment_id
        self.current_result = result
        
        print(f"Created experiment: {experiment_id}")
        print(f"Directory: {exp_dir}")
        
        return experiment_id
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility"""
        import platform
        import psutil
        
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
        
        # Memory info
        try:
            memory = psutil.virtual_memory()
            info["total_memory_gb"] = memory.total / (1024**3)
            info["available_memory_gb"] = memory.available / (1024**3)
        except:
            pass
        
        # PyTorch info
        if torch:
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_names"] = [torch.cuda.get_device_name(i) 
                                   for i in range(torch.cuda.device_count())]
        
        return info
    
    @contextmanager
    def experiment_context(self, config: ExperimentConfig):
        """
        Context manager for experiment execution
        
        Args:
            config: Experiment configuration
        """
        experiment_id = self.create_experiment(config)
        start_time = datetime.datetime.now()
        
        try:
            yield experiment_id
            
            # Mark as completed
            self.current_result.success = True
            self.current_result.finished_at = datetime.datetime.now().isoformat()
            self.current_result.total_training_time = (datetime.datetime.now() - start_time).total_seconds()
            
            self.registry["experiments"][experiment_id]["status"] = "completed"
            
        except Exception as e:
            # Mark as failed
            self.current_result.success = False
            self.current_result.error_message = str(e)
            self.current_result.finished_at = datetime.datetime.now().isoformat()
            
            self.registry["experiments"][experiment_id]["status"] = "failed"
            self.registry["experiments"][experiment_id]["error"] = str(e)
            
            raise
            
        finally:
            # Save final results
            if self.auto_save:
                self.save_experiment_result(experiment_id)
            
            self._save_registry()
            
            # Clear current experiment
            self.current_experiment = None
            self.current_result = None
    
    def log_metrics(self, 
                   metrics: Dict[str, float], 
                   split: str = "train",
                   epoch: Optional[int] = None):
        """
        Log metrics for current experiment
        
        Args:
            metrics: Dictionary of metric values
            split: Data split ('train', 'val', 'test')
            epoch: Current epoch number
        """
        if not self.current_result:
            warnings.warn("No active experiment to log metrics to")
            return
        
        if split == "train":
            target_dict = self.current_result.train_metrics
        elif split == "val":
            target_dict = self.current_result.val_metrics
        elif split == "test":
            target_dict = self.current_result.test_metrics
        else:
            warnings.warn(f"Unknown split: {split}")
            return
        
        # Log metrics
        for metric_name, value in metrics.items():
            if split in ["train", "val"]:
                if metric_name not in target_dict:
                    target_dict[metric_name] = []
                target_dict[metric_name].append(value)
            else:  # test
                target_dict[metric_name] = value
        
        # Update best validation score
        if split == "val" and "auc" in metrics:
            if metrics["auc"] > self.current_result.best_val_score:
                self.current_result.best_val_score = metrics["auc"]
                self.current_result.best_epoch = epoch if epoch is not None else len(target_dict.get("auc", [])) - 1
        
        # Auto-save periodically
        if self.auto_save and split == "val":
            self._save_current_progress()
    
    def _save_current_progress(self):
        """Save current experiment progress"""
        if not self.current_experiment or not self.current_result:
            return
        
        exp_dir = self.base_path / self.current_experiment
        progress_path = exp_dir / "progress.json"
        
        # Create serializable version
        progress_data = {
            "experiment_id": self.current_result.experiment_id,
            "train_metrics": self.current_result.train_metrics,
            "val_metrics": self.current_result.val_metrics,
            "test_metrics": self.current_result.test_metrics,
            "best_epoch": self.current_result.best_epoch,
            "best_val_score": self.current_result.best_val_score,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Performance metrics
            is_best: Whether this is the best checkpoint
        """
        if not self.current_experiment:
            warnings.warn("No active experiment to save checkpoint to")
            return
        
        exp_dir = self.base_path / self.current_experiment
        checkpoint_dir = exp_dir / "checkpoints"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'experiment_id': self.current_experiment,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.current_result.model_path = str(best_path)
            
            print(f"Saved best checkpoint: {best_path}")
    
    def load_checkpoint(self, 
                       experiment_id: str,
                       checkpoint_name: str = "best_model.pth") -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            experiment_id: ID of experiment
            checkpoint_name: Name of checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        exp_dir = self.base_path / experiment_id
        checkpoint_path = exp_dir / "checkpoints" / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def save_experiment_result(self, experiment_id: str):
        """
        Save complete experiment result
        
        Args:
            experiment_id: ID of experiment to save
        """
        if experiment_id != self.current_experiment:
            warnings.warn("Can only save current experiment")
            return
        
        exp_dir = self.base_path / experiment_id
        result_path = exp_dir / "results" / "experiment_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert result to serializable format
        result_dict = asdict(self.current_result)
        
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Saved experiment result: {result_path}")
    
    def load_experiment_result(self, experiment_id: str) -> ExperimentResult:
        """
        Load experiment result
        
        Args:
            experiment_id: ID of experiment
            
        Returns:
            Experiment result object
        """
        exp_dir = self.base_path / experiment_id
        result_path = exp_dir / "results" / "experiment_result.json"
        
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")
        
        with open(result_path, 'r') as f:
            result_dict = json.load(f)
        
        # Convert back to ExperimentResult object
        config_dict = result_dict.pop('config')
        config = ExperimentConfig(**config_dict)
        
        result = ExperimentResult(config=config, **result_dict)
        
        return result
    
    def list_experiments(self, 
                        status: Optional[str] = None,
                        tag: Optional[str] = None) -> pd.DataFrame:
        """
        List all experiments with filtering
        
        Args:
            status: Filter by status ('running', 'completed', 'failed')
            tag: Filter by tag
            
        Returns:
            DataFrame with experiment information
        """
        experiments = []
        
        for exp_id, exp_info in self.registry["experiments"].items():
            # Apply filters
            if status and exp_info.get("status") != status:
                continue
            
            if tag and tag not in exp_info.get("config", {}).get("tags", []):
                continue
            
            experiments.append({
                "experiment_id": exp_id,
                "experiment_name": exp_info.get("config", {}).get("experiment_name", ""),
                "model_name": exp_info.get("config", {}).get("model_name", ""),
                "dataset_name": exp_info.get("config", {}).get("dataset_name", ""),
                "status": exp_info.get("status", "unknown"),
                "created_at": exp_info.get("created_at", ""),
                "tags": exp_info.get("config", {}).get("tags", [])
            })
        
        return pd.DataFrame(experiments)
    
    def compare_experiments(self, 
                           experiment_ids: List[str],
                           metric: str = "auc") -> pd.DataFrame:
        """
        Compare multiple experiments
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric: Metric to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            try:
                result = self.load_experiment_result(exp_id)
                
                # Get best validation score
                best_val_score = 0.0
                if metric in result.val_metrics:
                    best_val_score = max(result.val_metrics[metric])
                
                # Get test score
                test_score = result.test_metrics.get(metric, 0.0)
                
                comparison_data.append({
                    "experiment_id": exp_id,
                    "experiment_name": result.config.experiment_name,
                    "model_name": result.config.model_name,
                    f"best_val_{metric}": best_val_score,
                    f"test_{metric}": test_score,
                    "best_epoch": result.best_epoch,
                    "training_time": result.total_training_time,
                    "success": result.success
                })
                
            except Exception as e:
                warnings.warn(f"Failed to load experiment {exp_id}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def cleanup_experiments(self, 
                           keep_best_n: int = 5,
                           max_age_days: int = 30):
        """
        Clean up old experiments
        
        Args:
            keep_best_n: Number of best experiments to keep
            max_age_days: Maximum age of experiments to keep
        """
        print("Cleaning up experiments...")
        
        # Get all experiments
        experiments_df = self.list_experiments()
        
        if len(experiments_df) == 0:
            print("No experiments to clean up")
            return
        
        # Determine which experiments to keep
        keep_experiments = set()
        
        # Keep recent experiments
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
        recent_experiments = experiments_df[
            pd.to_datetime(experiments_df['created_at']) > cutoff_date
        ]
        keep_experiments.update(recent_experiments['experiment_id'])
        
        # Keep best performing experiments
        completed_experiments = experiments_df[experiments_df['status'] == 'completed']
        if len(completed_experiments) > 0:
            # This would need to be implemented based on actual performance metrics
            # For now, keep the most recent completed experiments
            best_experiments = completed_experiments.nlargest(keep_best_n, 'created_at')
            keep_experiments.update(best_experiments['experiment_id'])
        
        # Remove experiments not in keep list
        removed_count = 0
        for exp_id in experiments_df['experiment_id']:
            if exp_id not in keep_experiments:
                exp_dir = self.base_path / exp_id
                if exp_dir.exists():
                    import shutil
                    shutil.rmtree(exp_dir)
                    removed_count += 1
                
                # Remove from registry
                if exp_id in self.registry["experiments"]:
                    del self.registry["experiments"][exp_id]
        
        self._save_registry()
        
        print(f"Removed {removed_count} old experiments")
        print(f"Kept {len(keep_experiments)} experiments")

def create_latex_table(comparison_df: pd.DataFrame, 
                      output_path: Optional[Path] = None) -> str:
    """
    Generate LaTeX table from experiment comparison
    
    Args:
        comparison_df: DataFrame from compare_experiments
        output_path: Optional path to save LaTeX file
        
    Returns:
        LaTeX table string
    """
    # Generate LaTeX table
    latex_str = comparison_df.to_latex(
        index=False,
        float_format="%.4f",
        escape=False,
        column_format="l" + "c" * (len(comparison_df.columns) - 1)
    )
    
    # Improve formatting
    latex_str = latex_str.replace('\\toprule', '\\hline')
    latex_str = latex_str.replace('\\midrule', '\\hline')
    latex_str = latex_str.replace('\\bottomrule', '\\hline')
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"Saved LaTeX table: {output_path}")
    
    return latex_str