"""
Training Monitoring and Performance Profiling Tools for Stage 02

This module provides comprehensive monitoring capabilities for training progress,
performance profiling, and resource utilization tracking for the heterogeneous
expert system.
"""

import torch
import torch.nn as nn
import torch.profiler
import psutil
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from pathlib import Path
import io
import base64
import warnings
from datetime import datetime, timedelta
import gc


@dataclass
class MonitoringConfig:
    """Configuration for training monitoring and profiling"""
    # Performance monitoring
    enable_gpu_monitoring: bool = True
    enable_cpu_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_disk_monitoring: bool = True

    # Profiling settings
    enable_pytorch_profiler: bool = True
    profiler_schedule_wait: int = 1
    profiler_schedule_warmup: int = 1
    profiler_schedule_active: int = 3
    profiler_schedule_repeat: int = 2

    # Training metrics
    track_learning_rates: bool = True
    track_gradient_norms: bool = True
    track_weight_norms: bool = True
    track_loss_components: bool = True

    # Monitoring frequency
    system_monitor_interval: float = 1.0  # seconds
    metrics_log_frequency: int = 10  # steps
    checkpoint_monitor_frequency: int = 100  # steps

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_loss"

    # Visualization
    enable_live_plotting: bool = True
    plot_update_frequency: int = 50  # steps
    save_plots: bool = True

    # Output settings
    output_dir: str = "training_monitoring"
    log_file: str = "training.log"
    metrics_file: str = "metrics.json"


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    temperature_celsius: float = 0.0


@dataclass
class TrainingMetrics:
    """Training progress metrics"""
    epoch: int
    step: int
    timestamp: float
    loss: float
    learning_rate: float
    batch_size: int
    samples_per_second: float
    gradient_norm: float = 0.0
    weight_norm: float = 0.0
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None


class SystemResourceMonitor:
    """Monitors system resources during training"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.monitoring = False
        self.metrics_history = deque(maxlen=1000)
        self.monitor_thread = None

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()

    def start_monitoring(self):
        """Start system resource monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logging.info("System resource monitoring started")

    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logging.info("System resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.config.system_monitor_interval)
            except Exception as e:
                logging.warning(f"Error in system monitoring: {e}")
                time.sleep(self.config.system_monitor_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent
        )

        # GPU metrics
        if self.gpu_available and self.config.enable_gpu_monitoring:
            try:
                metrics.gpu_memory_used_mb = torch.cuda.memory_allocated(self.gpu_device) / (1024**2)
                metrics.gpu_memory_total_mb = torch.cuda.get_device_properties(self.gpu_device).total_memory / (1024**2)
                metrics.gpu_utilization_percent = torch.cuda.utilization(self.gpu_device) if hasattr(torch.cuda, 'utilization') else 0.0

                # Try to get GPU temperature (may not be available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_device)
                    metrics.temperature_celsius = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    pass

            except Exception as e:
                logging.debug(f"GPU monitoring error: {e}")

        return metrics

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics"""
        if not self.metrics_history:
            return {}

        metrics_arrays = {
            'cpu_percent': [m.cpu_percent for m in self.metrics_history],
            'memory_percent': [m.memory_percent for m in self.metrics_history],
            'gpu_memory_used_mb': [m.gpu_memory_used_mb for m in self.metrics_history],
            'gpu_utilization_percent': [m.gpu_utilization_percent for m in self.metrics_history]
        }

        summary = {}
        for metric_name, values in metrics_arrays.items():
            if values and not all(v == 0 for v in values):
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values)
                }

        return summary


class PerformanceProfiler:
    """PyTorch profiler integration for detailed performance analysis"""

    def __init__(self, config: MonitoringConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.profiler = None

    def start_profiling(self):
        """Start PyTorch profiling"""
        if not self.config.enable_pytorch_profiler:
            return

        try:
            schedule = torch.profiler.schedule(
                wait=self.config.profiler_schedule_wait,
                warmup=self.config.profiler_schedule_warmup,
                active=self.config.profiler_schedule_active,
                repeat=self.config.profiler_schedule_repeat
            )

            self.profiler = torch.profiler.profile(
                schedule=schedule,
                on_trace_ready=self._on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )

            self.profiler.start()
            logging.info("PyTorch profiling started")

        except Exception as e:
            logging.warning(f"Failed to start profiling: {e}")

    def stop_profiling(self):
        """Stop PyTorch profiling"""
        if self.profiler:
            try:
                self.profiler.stop()
                logging.info("PyTorch profiling stopped")
            except Exception as e:
                logging.warning(f"Error stopping profiler: {e}")
            finally:
                self.profiler = None

    def step(self):
        """Step the profiler"""
        if self.profiler:
            self.profiler.step()

    def _on_trace_ready(self, prof):
        """Handle profiler trace ready event"""
        try:
            # Save trace
            trace_path = self.output_dir / f"profiler_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            prof.export_chrome_trace(str(trace_path))

            # Print top operations
            print("Top 10 operations by CPU time:")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

            if torch.cuda.is_available():
                print("\nTop 10 operations by GPU time:")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        except Exception as e:
            logging.warning(f"Error processing profiler trace: {e}")


class TrainingProgressTracker:
    """Tracks training progress and metrics"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.training_history = []
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = None
        self.epoch_start_time = None

        # Early stopping
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.should_stop_early = False

        # Loss components tracking
        self.loss_components = defaultdict(list)

    def start_training(self):
        """Mark start of training"""
        self.start_time = time.time()
        logging.info("Training progress tracking started")

    def start_epoch(self, epoch: int):
        """Mark start of epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def log_step(
        self,
        loss: float,
        learning_rate: float,
        batch_size: int,
        model: Optional[nn.Module] = None,
        **kwargs
    ):
        """Log training step metrics"""
        current_time = time.time()

        # Calculate throughput
        if self.epoch_start_time:
            time_elapsed = current_time - self.epoch_start_time
            samples_per_second = (self.current_step + 1) * batch_size / max(time_elapsed, 1e-6)
        else:
            samples_per_second = 0.0

        # Calculate gradient and weight norms
        gradient_norm = 0.0
        weight_norm = 0.0

        if model and self.config.track_gradient_norms:
            gradient_norm = self._calculate_gradient_norm(model)

        if model and self.config.track_weight_norms:
            weight_norm = self._calculate_weight_norm(model)

        # Create metrics object
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            timestamp=current_time,
            loss=loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            samples_per_second=samples_per_second,
            gradient_norm=gradient_norm,
            weight_norm=weight_norm,
            **kwargs
        )

        self.training_history.append(metrics)
        self.current_step += 1

        # Log loss components
        for key, value in kwargs.items():
            if 'loss' in key.lower() and isinstance(value, (int, float)):
                self.loss_components[key].append(value)

        return metrics

    def log_validation(self, val_loss: float, val_accuracy: Optional[float] = None):
        """Log validation metrics"""
        if self.training_history:
            last_metrics = self.training_history[-1]
            last_metrics.validation_loss = val_loss
            last_metrics.validation_accuracy = val_accuracy

        # Check early stopping
        if self.config.enable_early_stopping:
            self._check_early_stopping(val_loss)

    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calculate gradient norm"""
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1. / 2)

        return total_norm

    def _calculate_weight_norm(self, model: nn.Module) -> float:
        """Calculate weight norm"""
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.data is not None:
                param_norm = param.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1. / 2)

        return total_norm

    def _check_early_stopping(self, current_metric: float):
        """Check if early stopping criteria are met"""
        if current_metric < self.best_metric - self.config.early_stopping_min_delta:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.config.early_stopping_patience:
            self.should_stop_early = True
            logging.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        if not self.training_history:
            return {}

        losses = [m.loss for m in self.training_history]
        learning_rates = [m.learning_rate for m in self.training_history]
        throughputs = [m.samples_per_second for m in self.training_history]

        summary = {
            "total_steps": len(self.training_history),
            "current_epoch": self.current_epoch,
            "loss_statistics": {
                "current": losses[-1],
                "min": min(losses),
                "max": max(losses),
                "mean": np.mean(losses),
                "std": np.std(losses)
            },
            "learning_rate_statistics": {
                "current": learning_rates[-1],
                "min": min(learning_rates),
                "max": max(learning_rates),
                "mean": np.mean(learning_rates)
            },
            "throughput_statistics": {
                "current": throughputs[-1],
                "min": min(throughputs),
                "max": max(throughputs),
                "mean": np.mean(throughputs)
            },
            "early_stopping": {
                "should_stop": self.should_stop_early,
                "patience_counter": self.patience_counter,
                "best_metric": self.best_metric
            }
        }

        # Add validation metrics if available
        val_losses = [m.validation_loss for m in self.training_history if m.validation_loss is not None]
        if val_losses:
            summary["validation_loss_statistics"] = {
                "current": val_losses[-1],
                "min": min(val_losses),
                "max": max(val_losses),
                "mean": np.mean(val_losses)
            }

        return summary


class TrainingVisualizer:
    """Creates real-time visualizations of training progress"""

    def __init__(self, config: MonitoringConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def create_training_dashboard(
        self,
        training_tracker: TrainingProgressTracker,
        system_monitor: SystemResourceMonitor
    ) -> Dict[str, str]:
        """Create comprehensive training dashboard"""
        if not training_tracker.training_history:
            return {}

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Dashboard', fontsize=16)

        # Training loss
        self._plot_training_loss(training_tracker, axes[0, 0])

        # Learning rate
        self._plot_learning_rate(training_tracker, axes[0, 1])

        # Throughput
        self._plot_throughput(training_tracker, axes[0, 2])

        # System resources
        self._plot_system_resources(system_monitor, axes[1, 0])

        # Gradient norms
        self._plot_gradient_norms(training_tracker, axes[1, 1])

        # Loss components
        self._plot_loss_components(training_tracker, axes[1, 2])

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "training_dashboard.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return {
            "dashboard_path": str(plot_path),
            "base64_image": image_base64
        }

    def _plot_training_loss(self, tracker: TrainingProgressTracker, ax):
        """Plot training and validation loss"""
        steps = [m.step for m in tracker.training_history]
        train_losses = [m.loss for m in tracker.training_history]

        ax.plot(steps, train_losses, label='Training Loss', alpha=0.7)

        # Validation loss
        val_steps = [m.step for m in tracker.training_history if m.validation_loss is not None]
        val_losses = [m.validation_loss for m in tracker.training_history if m.validation_loss is not None]

        if val_losses:
            ax.plot(val_steps, val_losses, label='Validation Loss', alpha=0.7)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_learning_rate(self, tracker: TrainingProgressTracker, ax):
        """Plot learning rate schedule"""
        steps = [m.step for m in tracker.training_history]
        lrs = [m.learning_rate for m in tracker.training_history]

        ax.plot(steps, lrs, color='orange')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    def _plot_throughput(self, tracker: TrainingProgressTracker, ax):
        """Plot training throughput"""
        steps = [m.step for m in tracker.training_history]
        throughputs = [m.samples_per_second for m in tracker.training_history]

        ax.plot(steps, throughputs, color='green', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Samples/Second')
        ax.set_title('Training Throughput')
        ax.grid(True, alpha=0.3)

    def _plot_system_resources(self, monitor: SystemResourceMonitor, ax):
        """Plot system resource usage"""
        if not monitor.metrics_history:
            ax.text(0.5, 0.5, 'No system metrics available', ha='center', va='center')
            ax.set_title('System Resources')
            return

        timestamps = [m.timestamp for m in monitor.metrics_history]
        start_time = timestamps[0]
        times = [(t - start_time) / 60 for t in timestamps]  # Convert to minutes

        cpu_usage = [m.cpu_percent for m in monitor.metrics_history]
        memory_usage = [m.memory_percent for m in monitor.metrics_history]
        gpu_memory = [m.gpu_memory_used_mb / 1024 for m in monitor.metrics_history]  # Convert to GB

        ax.plot(times, cpu_usage, label='CPU %', alpha=0.7)
        ax.plot(times, memory_usage, label='Memory %', alpha=0.7)

        if any(gpu > 0 for gpu in gpu_memory):
            ax2 = ax.twinx()
            ax2.plot(times, gpu_memory, label='GPU Memory (GB)', color='red', alpha=0.7)
            ax2.set_ylabel('GPU Memory (GB)')
            ax2.legend(loc='upper right')

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Usage (%)')
        ax.set_title('System Resources')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_norms(self, tracker: TrainingProgressTracker, ax):
        """Plot gradient norms"""
        steps = [m.step for m in tracker.training_history]
        grad_norms = [m.gradient_norm for m in tracker.training_history]

        if any(gn > 0 for gn in grad_norms):
            ax.plot(steps, grad_norms, color='purple', alpha=0.7)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'Gradient norms not tracked', ha='center', va='center')

        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms')
        ax.grid(True, alpha=0.3)

    def _plot_loss_components(self, tracker: TrainingProgressTracker, ax):
        """Plot loss components"""
        if not tracker.loss_components:
            ax.text(0.5, 0.5, 'No loss components tracked', ha='center', va='center')
            ax.set_title('Loss Components')
            return

        for component_name, values in tracker.loss_components.items():
            if values and len(values) > 1:
                steps = list(range(len(values)))
                ax.plot(steps, values, label=component_name, alpha=0.7)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss Value')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)


class TrainingMonitor:
    """Main training monitoring system"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.system_monitor = SystemResourceMonitor(config)
        self.profiler = PerformanceProfiler(config, str(self.output_dir))
        self.progress_tracker = TrainingProgressTracker(config)
        self.visualizer = TrainingVisualizer(config, str(self.output_dir))

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / self.config.log_file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def start_monitoring(self):
        """Start all monitoring components"""
        self.system_monitor.start_monitoring()
        self.profiler.start_profiling()
        self.progress_tracker.start_training()
        logging.info("Training monitoring started")

    def stop_monitoring(self):
        """Stop all monitoring components"""
        self.system_monitor.stop_monitoring()
        self.profiler.stop_profiling()
        self._save_final_report()
        logging.info("Training monitoring stopped")

    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        batch_size: int,
        model: Optional[nn.Module] = None,
        **kwargs
    ) -> TrainingMetrics:
        """Log a training step"""
        metrics = self.progress_tracker.log_step(
            loss, learning_rate, batch_size, model, **kwargs
        )

        # Step profiler
        self.profiler.step()

        # Update visualizations periodically
        if (metrics.step % self.config.plot_update_frequency == 0 and
            self.config.enable_live_plotting):
            self._update_visualizations()

        return metrics

    def log_validation(self, val_loss: float, val_accuracy: Optional[float] = None):
        """Log validation results"""
        self.progress_tracker.log_validation(val_loss, val_accuracy)

    def should_stop_early(self) -> bool:
        """Check if training should stop early"""
        return self.progress_tracker.should_stop_early

    def _update_visualizations(self):
        """Update training visualizations"""
        try:
            dashboard = self.visualizer.create_training_dashboard(
                self.progress_tracker, self.system_monitor
            )
            logging.debug("Training dashboard updated")
        except Exception as e:
            logging.warning(f"Failed to update visualizations: {e}")

    def _save_final_report(self):
        """Save final training report"""
        try:
            # Training summary
            training_summary = self.progress_tracker.get_training_summary()

            # System summary
            system_summary = self.system_monitor.get_metrics_summary()

            # Combined report
            report = {
                "training_summary": training_summary,
                "system_summary": system_summary,
                "config": self.config.__dict__,
                "timestamp": datetime.now().isoformat()
            }

            # Save to file
            report_file = self.output_dir / "final_training_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Create final dashboard
            final_dashboard = self.visualizer.create_training_dashboard(
                self.progress_tracker, self.system_monitor
            )

            logging.info(f"Final training report saved to {report_file}")

        except Exception as e:
            logging.error(f"Failed to save final report: {e}")


def test_training_monitor():
    """Test function for training monitoring"""
    print("Testing Training Monitor...")

    config = MonitoringConfig(
        output_dir="test_training_monitor",
        enable_pytorch_profiler=False,  # Disable for testing
        system_monitor_interval=0.5
    )

    monitor = TrainingMonitor(config)

    try:
        # Start monitoring
        monitor.start_monitoring()
        print("✓ Monitoring started successfully")

        # Simulate training steps
        for step in range(5):
            loss = 1.0 - step * 0.1
            lr = 0.001 * (0.9 ** step)

            metrics = monitor.log_training_step(
                loss=loss,
                learning_rate=lr,
                batch_size=32
            )

            print(f"✓ Step {step}: Loss = {loss:.3f}, LR = {lr:.6f}")
            time.sleep(0.1)

        # Simulate validation
        monitor.log_validation(0.8, 0.85)
        print("✓ Validation logged successfully")

        # Stop monitoring
        monitor.stop_monitoring()
        print("✓ Monitoring stopped successfully")

        print("✓ Training Monitor test completed successfully")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")


if __name__ == "__main__":
    test_training_monitor()