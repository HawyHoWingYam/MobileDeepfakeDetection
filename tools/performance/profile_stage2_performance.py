"""
Performance Profiling Script for Stage 02 Heterogeneous Expert System

This script provides comprehensive performance profiling for the Stage 02 system,
including latency analysis, memory usage, throughput measurement, and bottleneck identification.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import json
import logging
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import psutil
import gc

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from stage_02.spatial_expert import EfficientNetV2SpatialExpert
from stage_02.genconvit_expert import GenConViTExpert
from stage_02.unified_feature_extractor import UnifiedFeatureExtractor
from stage_02.multi_resolution_dataloader import MultiResolutionDataLoaderFactory, DataLoaderConfig


@dataclass
class ProfileConfig:
    """Configuration for performance profiling"""
    # Test parameters
    batch_sizes: List[int] = None
    resolutions: List[int] = None
    num_warmup_batches: int = 5
    num_test_batches: int = 20

    # Profiling options
    profile_memory: bool = True
    profile_latency: bool = True
    profile_throughput: bool = True
    profile_concurrent: bool = True

    # Output settings
    output_dir: str = "performance_profiles"
    save_detailed_logs: bool = True
    save_visualizations: bool = True

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.resolutions is None:
            self.resolutions = [224, 256, 288, 320]


class ExpertProfiler:
    """Profiles individual expert performance"""

    def __init__(self, config: ProfileConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def profile_expert(
        self,
        expert: nn.Module,
        expert_name: str,
        expert_config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Profile a single expert across different configurations"""

        print(f"\n=== Profiling {expert_name} Expert ===")
        expert_results = {
            "expert_name": expert_name,
            "device": str(self.device),
            "latency_results": {},
            "memory_results": {},
            "throughput_results": {},
            "detailed_stats": {}
        }

        expert = expert.to(self.device)
        expert.eval()

        # Profile across different batch sizes and resolutions
        for resolution in self.config.resolutions:
            for batch_size in self.config.batch_sizes:
                config_key = f"res_{resolution}_batch_{batch_size}"
                print(f"  Testing {config_key}...")

                config_results = self._profile_configuration(
                    expert, resolution, batch_size
                )

                expert_results["latency_results"][config_key] = config_results["latency"]
                expert_results["memory_results"][config_key] = config_results["memory"]
                expert_results["throughput_results"][config_key] = config_results["throughput"]

        # Calculate aggregated statistics
        expert_results["detailed_stats"] = self._calculate_expert_statistics(expert_results)

        return expert_results

    def _profile_configuration(
        self,
        expert: nn.Module,
        resolution: int,
        batch_size: int
    ) -> Dict[str, Any]:
        """Profile expert for specific resolution and batch size"""

        # Create dummy input
        dummy_input = torch.randn(
            batch_size, 3, resolution, resolution,
            device=self.device, dtype=torch.float32
        )

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.num_warmup_batches):
                _ = expert(dummy_input)

        # Clear cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Measure memory before
        memory_before = self._get_memory_usage()

        # Latency measurement
        latencies = []
        with torch.no_grad():
            for _ in range(self.config.num_test_batches):
                start_time = time.time()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                output = expert(dummy_input)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Measure memory after
        memory_after = self._get_memory_usage()

        # Calculate results
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)

        throughput = batch_size / (avg_latency / 1000)  # images per second
        memory_usage = memory_after - memory_before

        return {
            "latency": {
                "mean_ms": avg_latency,
                "std_ms": std_latency,
                "min_ms": min_latency,
                "max_ms": max_latency,
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99)
            },
            "memory": {
                "usage_mb": memory_usage,
                "per_sample_mb": memory_usage / batch_size
            },
            "throughput": {
                "images_per_second": throughput,
                "batches_per_second": 1 / (avg_latency / 1000)
            }
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)

    def _calculate_expert_statistics(self, expert_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregated statistics for expert"""

        # Extract all latency values
        all_latencies = []
        all_throughputs = []
        all_memory_usage = []

        for config_key, results in expert_results["latency_results"].items():
            all_latencies.append(results["mean_ms"])
            all_throughputs.append(expert_results["throughput_results"][config_key]["images_per_second"])
            all_memory_usage.append(expert_results["memory_results"][config_key]["usage_mb"])

        stats = {
            "latency_stats": {
                "overall_mean_ms": np.mean(all_latencies),
                "overall_std_ms": np.std(all_latencies),
                "best_latency_ms": np.min(all_latencies),
                "worst_latency_ms": np.max(all_latencies)
            },
            "throughput_stats": {
                "max_throughput_ips": np.max(all_throughputs),
                "min_throughput_ips": np.min(all_throughputs),
                "mean_throughput_ips": np.mean(all_throughputs)
            },
            "memory_stats": {
                "max_memory_mb": np.max(all_memory_usage),
                "min_memory_mb": np.min(all_memory_usage),
                "mean_memory_mb": np.mean(all_memory_usage)
            }
        }

        return stats


class ConcurrentProfiler:
    """Profiles concurrent execution of multiple experts"""

    def __init__(self, config: ProfileConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def profile_concurrent_experts(
        self,
        spatial_expert: nn.Module,
        generative_expert: nn.Module,
        unified_extractor: Optional[UnifiedFeatureExtractor] = None
    ) -> Dict[str, Any]:
        """Profile concurrent execution of experts"""

        print("\n=== Profiling Concurrent Expert Execution ===")

        spatial_expert = spatial_expert.to(self.device)
        generative_expert = generative_expert.to(self.device)

        spatial_expert.eval()
        generative_expert.eval()

        concurrent_results = {
            "sequential_execution": {},
            "parallel_execution": {},
            "unified_extractor_execution": {},
            "efficiency_analysis": {}
        }

        # Test different configurations
        for resolution in [256, 320]:  # Focus on key resolutions
            for batch_size in [1, 8, 16]:  # Focus on key batch sizes
                config_key = f"res_{resolution}_batch_{batch_size}"
                print(f"  Testing concurrent {config_key}...")

                # Create dummy input
                dummy_input = torch.randn(
                    batch_size, 3, resolution, resolution,
                    device=self.device, dtype=torch.float32
                )

                # Sequential execution
                sequential_results = self._profile_sequential_execution(
                    spatial_expert, generative_expert, dummy_input
                )
                concurrent_results["sequential_execution"][config_key] = sequential_results

                # Parallel execution (simulated)
                parallel_results = self._profile_parallel_execution(
                    spatial_expert, generative_expert, dummy_input
                )
                concurrent_results["parallel_execution"][config_key] = parallel_results

                # Unified extractor (if available)
                if unified_extractor:
                    unified_results = self._profile_unified_execution(
                        unified_extractor, dummy_input
                    )
                    concurrent_results["unified_extractor_execution"][config_key] = unified_results

        # Calculate efficiency analysis
        concurrent_results["efficiency_analysis"] = self._analyze_concurrent_efficiency(
            concurrent_results
        )

        return concurrent_results

    def _profile_sequential_execution(
        self,
        spatial_expert: nn.Module,
        generative_expert: nn.Module,
        dummy_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Profile sequential execution of experts"""

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = spatial_expert(dummy_input)
                _ = generative_expert(dummy_input)

        # Clear cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Measure execution time
        execution_times = []
        memory_usage = []

        with torch.no_grad():
            for _ in range(self.config.num_test_batches):
                memory_before = self._get_memory_usage()

                start_time = time.time()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                # Sequential execution
                spatial_output = spatial_expert(dummy_input)
                generative_output = generative_expert(dummy_input)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                memory_after = self._get_memory_usage()

                execution_times.append((end_time - start_time) * 1000)
                memory_usage.append(memory_after - memory_before)

        return {
            "mean_time_ms": np.mean(execution_times),
            "std_time_ms": np.std(execution_times),
            "min_time_ms": np.min(execution_times),
            "max_time_ms": np.max(execution_times),
            "mean_memory_mb": np.mean(memory_usage),
            "throughput_ips": dummy_input.size(0) / (np.mean(execution_times) / 1000)
        }

    def _profile_parallel_execution(
        self,
        spatial_expert: nn.Module,
        generative_expert: nn.Module,
        dummy_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Profile simulated parallel execution"""

        # For single GPU, this simulates the best-case scenario
        # In practice, would need multiple GPUs for true parallelism

        # Measure individual expert times
        spatial_times = []
        generative_times = []

        with torch.no_grad():
            # Spatial expert timing
            for _ in range(self.config.num_test_batches):
                start_time = time.time()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                _ = spatial_expert(dummy_input)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                spatial_times.append((end_time - start_time) * 1000)

            # Generative expert timing
            for _ in range(self.config.num_test_batches):
                start_time = time.time()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                _ = generative_expert(dummy_input)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                generative_times.append((end_time - start_time) * 1000)

        # Simulated parallel time (max of the two)
        spatial_mean = np.mean(spatial_times)
        generative_mean = np.mean(generative_times)
        parallel_time = max(spatial_mean, generative_mean)

        return {
            "spatial_time_ms": spatial_mean,
            "generative_time_ms": generative_mean,
            "simulated_parallel_time_ms": parallel_time,
            "theoretical_speedup": (spatial_mean + generative_mean) / parallel_time,
            "throughput_ips": dummy_input.size(0) / (parallel_time / 1000)
        }

    def _profile_unified_execution(
        self,
        unified_extractor: UnifiedFeatureExtractor,
        dummy_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Profile unified extractor execution"""

        execution_times = []
        memory_usage = []

        with torch.no_grad():
            for _ in range(self.config.num_test_batches):
                memory_before = self._get_memory_usage()

                start_time = time.time()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                output = unified_extractor.extract_features_parallel(dummy_input)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                memory_after = self._get_memory_usage()

                execution_times.append((end_time - start_time) * 1000)
                memory_usage.append(memory_after - memory_before)

        return {
            "mean_time_ms": np.mean(execution_times),
            "std_time_ms": np.std(execution_times),
            "mean_memory_mb": np.mean(memory_usage),
            "throughput_ips": dummy_input.size(0) / (np.mean(execution_times) / 1000)
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)

    def _analyze_concurrent_efficiency(self, concurrent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze efficiency of concurrent execution strategies"""

        efficiency_analysis = {
            "sequential_vs_parallel": {},
            "unified_extractor_efficiency": {},
            "recommendations": []
        }

        # Compare sequential vs parallel for each configuration
        for config_key in concurrent_results["sequential_execution"].keys():
            sequential = concurrent_results["sequential_execution"][config_key]
            parallel = concurrent_results["parallel_execution"][config_key]

            speedup = sequential["mean_time_ms"] / parallel["simulated_parallel_time_ms"]
            throughput_improvement = parallel["throughput_ips"] / sequential["throughput_ips"]

            efficiency_analysis["sequential_vs_parallel"][config_key] = {
                "speedup": speedup,
                "throughput_improvement": throughput_improvement,
                "sequential_time_ms": sequential["mean_time_ms"],
                "parallel_time_ms": parallel["simulated_parallel_time_ms"]
            }

        # Unified extractor analysis
        if concurrent_results["unified_extractor_execution"]:
            for config_key in concurrent_results["unified_extractor_execution"].keys():
                if config_key in concurrent_results["sequential_execution"]:
                    sequential = concurrent_results["sequential_execution"][config_key]
                    unified = concurrent_results["unified_extractor_execution"][config_key]

                    unified_speedup = sequential["mean_time_ms"] / unified["mean_time_ms"]

                    efficiency_analysis["unified_extractor_efficiency"][config_key] = {
                        "speedup": unified_speedup,
                        "sequential_time_ms": sequential["mean_time_ms"],
                        "unified_time_ms": unified["mean_time_ms"]
                    }

        # Generate recommendations
        avg_speedup = np.mean([
            data["speedup"] for data in efficiency_analysis["sequential_vs_parallel"].values()
        ])

        if avg_speedup > 1.5:
            efficiency_analysis["recommendations"].append(
                "High parallel efficiency detected. Recommend multi-GPU deployment."
            )
        elif avg_speedup > 1.2:
            efficiency_analysis["recommendations"].append(
                "Moderate parallel efficiency. Consider parallel deployment for high-throughput scenarios."
            )
        else:
            efficiency_analysis["recommendations"].append(
                "Limited parallel efficiency. Sequential execution may be sufficient."
            )

        return efficiency_analysis


class PerformanceReporter:
    """Generates comprehensive performance reports"""

    def __init__(self, config: ProfileConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def generate_report(
        self,
        spatial_results: Dict[str, Any],
        generative_results: Dict[str, Any],
        concurrent_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate comprehensive performance report"""

        # Create summary
        summary = self._create_summary(spatial_results, generative_results, concurrent_results)

        # Save detailed results
        detailed_results = {
            "spatial_expert": spatial_results,
            "generative_expert": generative_results,
            "concurrent_execution": concurrent_results,
            "summary": summary
        }

        results_file = self.output_dir / "performance_results.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)

        # Generate visualizations
        visualization_paths = self._generate_visualizations(
            spatial_results, generative_results, concurrent_results
        )

        # Generate markdown report
        report_path = self._generate_markdown_report(summary, visualization_paths)

        return {
            "results_file": str(results_file),
            "report_file": str(report_path),
            "visualizations": visualization_paths
        }

    def _create_summary(
        self,
        spatial_results: Dict[str, Any],
        generative_results: Dict[str, Any],
        concurrent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create performance summary"""

        summary = {
            "spatial_expert_summary": {
                "best_latency_ms": spatial_results["detailed_stats"]["latency_stats"]["best_latency_ms"],
                "max_throughput_ips": spatial_results["detailed_stats"]["throughput_stats"]["max_throughput_ips"],
                "memory_range_mb": [
                    spatial_results["detailed_stats"]["memory_stats"]["min_memory_mb"],
                    spatial_results["detailed_stats"]["memory_stats"]["max_memory_mb"]
                ]
            },
            "generative_expert_summary": {
                "best_latency_ms": generative_results["detailed_stats"]["latency_stats"]["best_latency_ms"],
                "max_throughput_ips": generative_results["detailed_stats"]["throughput_stats"]["max_throughput_ips"],
                "memory_range_mb": [
                    generative_results["detailed_stats"]["memory_stats"]["min_memory_mb"],
                    generative_results["detailed_stats"]["memory_stats"]["max_memory_mb"]
                ]
            }
        }

        # Add concurrent execution summary
        if concurrent_results.get("efficiency_analysis"):
            efficiency = concurrent_results["efficiency_analysis"]["sequential_vs_parallel"]
            if efficiency:
                avg_speedup = np.mean([data["speedup"] for data in efficiency.values()])
                summary["concurrent_execution_summary"] = {
                    "average_parallel_speedup": avg_speedup,
                    "recommendations": concurrent_results["efficiency_analysis"]["recommendations"]
                }

        return summary

    def _generate_visualizations(
        self,
        spatial_results: Dict[str, Any],
        generative_results: Dict[str, Any],
        concurrent_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate performance visualizations"""

        visualization_paths = {}

        # Latency comparison plot
        latency_plot = self._plot_latency_comparison(spatial_results, generative_results)
        visualization_paths["latency_comparison"] = latency_plot

        # Throughput analysis plot
        throughput_plot = self._plot_throughput_analysis(spatial_results, generative_results)
        visualization_paths["throughput_analysis"] = throughput_plot

        # Memory usage plot
        memory_plot = self._plot_memory_usage(spatial_results, generative_results)
        visualization_paths["memory_usage"] = memory_plot

        # Concurrent execution plot
        if concurrent_results:
            concurrent_plot = self._plot_concurrent_performance(concurrent_results)
            visualization_paths["concurrent_performance"] = concurrent_plot

        return visualization_paths

    def _plot_latency_comparison(
        self,
        spatial_results: Dict[str, Any],
        generative_results: Dict[str, Any]
    ) -> str:
        """Plot latency comparison between experts"""

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Extract data
        spatial_latencies = []
        generative_latencies = []
        configs = []

        for config_key in spatial_results["latency_results"].keys():
            spatial_latencies.append(spatial_results["latency_results"][config_key]["mean_ms"])
            generative_latencies.append(generative_results["latency_results"][config_key]["mean_ms"])
            configs.append(config_key.replace("res_", "").replace("_batch_", "x"))

        # Bar plot
        x = np.arange(len(configs))
        width = 0.35

        axes[0].bar(x - width/2, spatial_latencies, width, label='Spatial Expert', alpha=0.8)
        axes[0].bar(x + width/2, generative_latencies, width, label='Generative Expert', alpha=0.8)

        axes[0].set_xlabel('Configuration (Resolution x Batch Size)')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Latency Comparison by Configuration')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(configs, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter plot
        axes[1].scatter(spatial_latencies, generative_latencies, alpha=0.7, s=100)

        # Add diagonal line
        max_latency = max(max(spatial_latencies), max(generative_latencies))
        axes[1].plot([0, max_latency], [0, max_latency], 'r--', alpha=0.5)

        axes[1].set_xlabel('Spatial Expert Latency (ms)')
        axes[1].set_ylabel('Generative Expert Latency (ms)')
        axes[1].set_title('Latency Correlation')
        axes[1].grid(True, alpha=0.3)

        # Add text annotations
        for i, config in enumerate(configs):
            axes[1].annotate(config, (spatial_latencies[i], generative_latencies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()

        plot_path = self.output_dir / "latency_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _plot_throughput_analysis(
        self,
        spatial_results: Dict[str, Any],
        generative_results: Dict[str, Any]
    ) -> str:
        """Plot throughput analysis"""

        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract batch sizes and throughputs for different resolutions
        resolutions = self.config.resolutions
        batch_sizes = self.config.batch_sizes

        for resolution in resolutions:
            spatial_throughputs = []
            generative_throughputs = []
            valid_batch_sizes = []

            for batch_size in batch_sizes:
                config_key = f"res_{resolution}_batch_{batch_size}"
                if config_key in spatial_results["throughput_results"]:
                    spatial_throughputs.append(
                        spatial_results["throughput_results"][config_key]["images_per_second"]
                    )
                    generative_throughputs.append(
                        generative_results["throughput_results"][config_key]["images_per_second"]
                    )
                    valid_batch_sizes.append(batch_size)

            if spatial_throughputs:
                ax.plot(valid_batch_sizes, spatial_throughputs,
                       marker='o', label=f'Spatial {resolution}px', alpha=0.8)
                ax.plot(valid_batch_sizes, generative_throughputs,
                       marker='s', label=f'Generative {resolution}px', alpha=0.8)

        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (images/second)')
        ax.set_title('Throughput Analysis by Batch Size and Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

        plt.tight_layout()

        plot_path = self.output_dir / "throughput_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _plot_memory_usage(
        self,
        spatial_results: Dict[str, Any],
        generative_results: Dict[str, Any]
    ) -> str:
        """Plot memory usage analysis"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Memory vs batch size for each resolution
        resolutions = self.config.resolutions[:2]  # Use first 2 resolutions for clarity

        for i, resolution in enumerate(resolutions):
            ax = axes[i, 0]

            batch_sizes = []
            spatial_memory = []
            generative_memory = []

            for batch_size in self.config.batch_sizes:
                config_key = f"res_{resolution}_batch_{batch_size}"
                if config_key in spatial_results["memory_results"]:
                    batch_sizes.append(batch_size)
                    spatial_memory.append(spatial_results["memory_results"][config_key]["usage_mb"])
                    generative_memory.append(generative_results["memory_results"][config_key]["usage_mb"])

            if batch_sizes:
                ax.plot(batch_sizes, spatial_memory, marker='o', label='Spatial Expert')
                ax.plot(batch_sizes, generative_memory, marker='s', label='Generative Expert')
                ax.set_xlabel('Batch Size')
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title(f'Memory Usage vs Batch Size ({resolution}px)')
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Memory per sample
        ax = axes[0, 1]
        configs = []
        spatial_per_sample = []
        generative_per_sample = []

        for config_key in spatial_results["memory_results"].keys():
            configs.append(config_key.replace("res_", "").replace("_batch_", "x"))
            spatial_per_sample.append(spatial_results["memory_results"][config_key]["per_sample_mb"])
            generative_per_sample.append(generative_results["memory_results"][config_key]["per_sample_mb"])

        x = np.arange(len(configs))
        width = 0.35

        ax.bar(x - width/2, spatial_per_sample, width, label='Spatial Expert', alpha=0.8)
        ax.bar(x + width/2, generative_per_sample, width, label='Generative Expert', alpha=0.8)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Memory per Sample (MB)')
        ax.set_title('Memory Usage per Sample')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Memory efficiency (throughput / memory)
        ax = axes[1, 1]
        spatial_efficiency = []
        generative_efficiency = []

        for config_key in configs:
            full_config_key = config_key.replace("x", "_batch_").replace(configs[0].split("x")[0], f"res_{configs[0].split('x')[0]}")
            if full_config_key in spatial_results["memory_results"]:
                spatial_eff = (spatial_results["throughput_results"][full_config_key]["images_per_second"] /
                              spatial_results["memory_results"][full_config_key]["usage_mb"])
                generative_eff = (generative_results["throughput_results"][full_config_key]["images_per_second"] /
                                 generative_results["memory_results"][full_config_key]["usage_mb"])
                spatial_efficiency.append(spatial_eff)
                generative_efficiency.append(generative_eff)

        if spatial_efficiency:
            x = np.arange(len(spatial_efficiency))
            ax.bar(x - width/2, spatial_efficiency, width, label='Spatial Expert', alpha=0.8)
            ax.bar(x + width/2, generative_efficiency, width, label='Generative Expert', alpha=0.8)
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Efficiency (images/second/MB)')
            ax.set_title('Memory Efficiency')
            ax.set_xticks(x)
            ax.set_xticklabels(configs[:len(spatial_efficiency)], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.output_dir / "memory_usage.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _plot_concurrent_performance(self, concurrent_results: Dict[str, Any]) -> str:
        """Plot concurrent execution performance"""

        if not concurrent_results.get("efficiency_analysis"):
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Speedup analysis
        ax = axes[0, 0]
        efficiency = concurrent_results["efficiency_analysis"]["sequential_vs_parallel"]

        configs = list(efficiency.keys())
        speedups = [efficiency[config]["speedup"] for config in configs]

        bars = ax.bar(range(len(configs)), speedups, alpha=0.8, color='skyblue')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Parallel Execution Speedup')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace("res_", "").replace("_batch_", "x") for c in configs], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{speedup:.2f}x', ha='center', va='bottom')

        # Execution time comparison
        ax = axes[0, 1]
        sequential_times = [efficiency[config]["sequential_time_ms"] for config in configs]
        parallel_times = [efficiency[config]["parallel_time_ms"] for config in configs]

        x = np.arange(len(configs))
        width = 0.35

        ax.bar(x - width/2, sequential_times, width, label='Sequential', alpha=0.8)
        ax.bar(x + width/2, parallel_times, width, label='Parallel', alpha=0.8)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Sequential vs Parallel Execution Time')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("res_", "").replace("_batch_", "x") for c in configs], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Resource utilization (if data available)
        ax = axes[1, 0]
        ax.text(0.5, 0.5, 'Resource utilization data\nwould be shown here\n(requires additional monitoring)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Resource Utilization')

        # Recommendations
        ax = axes[1, 1]
        recommendations = concurrent_results["efficiency_analysis"]["recommendations"]
        rec_text = '\n\n'.join([f"• {rec}" for rec in recommendations])
        ax.text(0.05, 0.95, rec_text, ha='left', va='top', transform=ax.transAxes,
               fontsize=10, wrap=True)
        ax.set_title('Performance Recommendations')
        ax.axis('off')

        plt.tight_layout()

        plot_path = self.output_dir / "concurrent_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _generate_markdown_report(
        self,
        summary: Dict[str, Any],
        visualization_paths: Dict[str, str]
    ) -> str:
        """Generate markdown performance report"""

        report_lines = [
            "# Stage 02 Performance Profiling Report",
            "",
            f"**Generated on**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Device**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
            "",
            "## Executive Summary",
            ""
        ]

        # Spatial expert summary
        spatial = summary["spatial_expert_summary"]
        report_lines.extend([
            "### Spatial Expert Performance",
            f"- **Best Latency**: {spatial['best_latency_ms']:.2f} ms",
            f"- **Max Throughput**: {spatial['max_throughput_ips']:.1f} images/second",
            f"- **Memory Range**: {spatial['memory_range_mb'][0]:.1f} - {spatial['memory_range_mb'][1]:.1f} MB",
            ""
        ])

        # Generative expert summary
        generative = summary["generative_expert_summary"]
        report_lines.extend([
            "### Generative Expert Performance",
            f"- **Best Latency**: {generative['best_latency_ms']:.2f} ms",
            f"- **Max Throughput**: {generative['max_throughput_ips']:.1f} images/second",
            f"- **Memory Range**: {generative['memory_range_mb'][0]:.1f} - {generative['memory_range_mb'][1]:.1f} MB",
            ""
        ])

        # Concurrent execution summary
        if "concurrent_execution_summary" in summary:
            concurrent = summary["concurrent_execution_summary"]
            report_lines.extend([
                "### Concurrent Execution Analysis",
                f"- **Average Parallel Speedup**: {concurrent['average_parallel_speedup']:.2f}x",
                "",
                "**Recommendations**:",
            ])

            for rec in concurrent["recommendations"]:
                report_lines.append(f"- {rec}")

            report_lines.append("")

        # Add visualizations
        report_lines.extend([
            "## Performance Visualizations",
            ""
        ])

        for viz_name, viz_path in visualization_paths.items():
            if viz_path:
                report_lines.extend([
                    f"### {viz_name.replace('_', ' ').title()}",
                    f"![{viz_name}]({Path(viz_path).name})",
                    ""
                ])

        # Configuration details
        report_lines.extend([
            "## Test Configuration",
            "",
            f"- **Batch Sizes Tested**: {self.config.batch_sizes}",
            f"- **Resolutions Tested**: {self.config.resolutions}",
            f"- **Warmup Batches**: {self.config.num_warmup_batches}",
            f"- **Test Batches**: {self.config.num_test_batches}",
            ""
        ])

        # Save report
        report_path = self.output_dir / "performance_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        return str(report_path)


def main():
    """Main profiling execution"""
    parser = argparse.ArgumentParser(description="Profile Stage 02 Expert System Performance")
    parser.add_argument("--output-dir", default="performance_profiles",
                       help="Output directory for results")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32],
                       help="Batch sizes to test")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[224, 256, 288, 320],
                       help="Image resolutions to test")
    parser.add_argument("--test-batches", type=int, default=20,
                       help="Number of test batches per configuration")
    parser.add_argument("--skip-concurrent", action="store_true",
                       help="Skip concurrent execution profiling")

    args = parser.parse_args()

    # Create configuration
    config = ProfileConfig(
        batch_sizes=args.batch_sizes,
        resolutions=args.resolutions,
        num_test_batches=args.test_batches,
        output_dir=args.output_dir,
        profile_concurrent=not args.skip_concurrent
    )

    print("=== Stage 02 Performance Profiling ===")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Resolutions: {config.resolutions}")
    print(f"Test batches per config: {config.num_test_batches}")

    # Initialize profilers
    expert_profiler = ExpertProfiler(config)
    concurrent_profiler = ConcurrentProfiler(config) if config.profile_concurrent else None
    reporter = PerformanceReporter(config)

    try:
        # Create dummy experts for testing
        print("\nCreating test models...")

        # Simple dummy models for testing
        spatial_expert = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

        generative_expert = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

        # Profile individual experts
        spatial_results = expert_profiler.profile_expert(spatial_expert, "Spatial")
        generative_results = expert_profiler.profile_expert(generative_expert, "Generative")

        # Profile concurrent execution
        concurrent_results = {}
        if concurrent_profiler:
            concurrent_results = concurrent_profiler.profile_concurrent_experts(
                spatial_expert, generative_expert
            )

        # Generate report
        print("\nGenerating performance report...")
        report_files = reporter.generate_report(
            spatial_results, generative_results, concurrent_results
        )

        print("\n=== Profiling Completed ===")
        print(f"Results saved to: {report_files['results_file']}")
        print(f"Report saved to: {report_files['report_file']}")
        print(f"Visualizations: {len(report_files['visualizations'])} files generated")

    except Exception as e:
        print(f"\nError during profiling: {e}")
        logging.error(f"Profiling failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())