"""
Concurrent System Testing Framework
Advanced testing infrastructure for multi-expert system validation

This module provides comprehensive testing capabilities for the heterogeneous expert
system, including concurrent expert evaluation, integration testing, performance
benchmarking, and system validation.
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import threading
import asyncio
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

from .unified_feature_extractor import BaseExpert, ExpertOutput, ExpertType
from .complementarity_analysis import AdaptiveFusionSystem, ComplementarityAnalysisResult


class TestType(Enum):
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    STRESS_TEST = "stress_test"
    VALIDATION_TEST = "validation_test"


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestMetrics:
    """Metrics collected during testing"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0


@dataclass
class TestCase:
    """Individual test case definition"""
    name: str
    test_type: TestType
    test_function: Callable
    expected_metrics: Optional[TestMetrics] = None
    timeout: int = 300  # seconds
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    status: TestStatus
    metrics: TestMetrics
    execution_time: float
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcurrentTestConfig:
    """Configuration for concurrent testing"""
    max_workers: int = 4
    gpu_memory_limit: int = 4096  # MB
    cpu_count: int = mp.cpu_count()
    timeout_global: int = 3600  # seconds
    memory_threshold: float = 0.9  # 90% memory usage limit
    enable_profiling: bool = True
    save_intermediate_results: bool = True
    visualization_enabled: bool = True


class ResourceMonitor:
    """Monitor system resources during testing"""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.metrics_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        if not self.metrics_history:
            return {}

        # Aggregate metrics
        metrics_array = np.array(self.metrics_history)
        return {
            'cpu_usage': {
                'mean': np.mean(metrics_array[:, 0]),
                'max': np.max(metrics_array[:, 0]),
                'std': np.std(metrics_array[:, 0])
            },
            'memory_usage': {
                'mean': np.mean(metrics_array[:, 1]),
                'max': np.max(metrics_array[:, 1]),
                'std': np.std(metrics_array[:, 1])
            },
            'gpu_usage': {
                'mean': np.mean(metrics_array[:, 2]),
                'max': np.max(metrics_array[:, 2]),
                'std': np.std(metrics_array[:, 2])
            }
        }

    def _monitor_resources(self):
        """Background resource monitoring"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # GPU usage (if available)
                gpu_percent = 0.0
                if torch.cuda.is_available():
                    gpu_percent = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0

                self.metrics_history.append([cpu_percent, memory_percent, gpu_percent])

            except Exception as e:
                logging.warning(f"Resource monitoring error: {e}")

            time.sleep(self.monitoring_interval)

    @contextmanager
    def monitor_context(self):
        """Context manager for resource monitoring"""
        self.start_monitoring()
        try:
            yield self
        finally:
            metrics = self.stop_monitoring()
            return metrics


class ExpertTester:
    """Individual expert testing utilities"""

    def __init__(self, expert: BaseExpert, device: torch.device):
        self.expert = expert
        self.device = device
        self.expert.to(device)

    async def test_inference_speed(self, dataloader: DataLoader, num_batches: int = 10) -> Dict[str, float]:
        """Test inference speed asynchronously"""
        self.expert.eval()
        inference_times = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                # Move data to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)

                # Time inference
                start_time = time.time()
                _ = self.expert(inputs)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()

                inference_times.append(end_time - start_time)

        return {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times)
        }

    async def test_memory_usage(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Test memory usage"""
        if not torch.cuda.is_available():
            return {'gpu_memory': 0.0}

        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(self.device)

        # Forward pass
        with torch.no_grad():
            _ = self.expert(input_tensor)

        peak_memory = torch.cuda.max_memory_allocated(self.device)
        memory_usage = (peak_memory - initial_memory) / 1024**2  # MB

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        return {'gpu_memory': memory_usage}

    async def test_accuracy(self, dataloader: DataLoader, labels: torch.Tensor) -> TestMetrics:
        """Test accuracy metrics"""
        self.expert.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, batch_labels = batch[0].to(self.device), batch[1]
                else:
                    inputs = batch.to(self.device)
                    batch_labels = labels[:inputs.size(0)]

                output = self.expert(inputs)
                predictions = output.predictions.get('classification',
                                                   output.predictions.get('probability'))

                all_predictions.append(predictions.cpu())
                all_labels.append(batch_labels)

        # Compute metrics
        predictions = torch.cat(all_predictions)
        labels_tensor = torch.cat(all_labels)

        # Convert to binary predictions
        binary_pred = (predictions > 0.5).float()

        # Calculate metrics
        tp = ((binary_pred == 1) & (labels_tensor == 1)).sum().item()
        tn = ((binary_pred == 0) & (labels_tensor == 0)).sum().item()
        fp = ((binary_pred == 1) & (labels_tensor == 0)).sum().item()
        fn = ((binary_pred == 0) & (labels_tensor == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score
            auc_roc = roc_auc_score(labels_tensor.numpy(), predictions.numpy())
        except:
            auc_roc = 0.0

        return TestMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc
        )


class ConcurrentTestExecutor:
    """Execute tests concurrently across multiple experts"""

    def __init__(self, config: ConcurrentTestConfig):
        self.config = config
        self.test_results = {}
        self.resource_monitor = ResourceMonitor()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def run_expert_tests(self,
                              experts: Dict[str, BaseExpert],
                              test_cases: List[TestCase],
                              dataloader: DataLoader) -> Dict[str, TestResult]:
        """Run tests concurrently across multiple experts"""

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

        try:
            # Create expert testers
            expert_testers = {}
            for name, expert in experts.items():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                expert_testers[name] = ExpertTester(expert, device)

            # Execute tests concurrently
            tasks = []
            for test_case in test_cases:
                for expert_name, tester in expert_testers.items():
                    task = self._execute_test_case(expert_name, tester, test_case, dataloader)
                    tasks.append(task)

            # Wait for all tests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Test failed with exception: {result}")
                    continue

                expert_name, test_result = result
                key = f"{expert_name}_{test_result.test_name}"
                processed_results[key] = test_result

            return processed_results

        finally:
            # Stop resource monitoring
            resource_metrics = self.resource_monitor.stop_monitoring()
            self.logger.info(f"Resource usage: {resource_metrics}")

    async def _execute_test_case(self,
                                expert_name: str,
                                tester: ExpertTester,
                                test_case: TestCase,
                                dataloader: DataLoader) -> Tuple[str, TestResult]:
        """Execute individual test case"""

        start_time = time.time()

        try:
            # Execute test function
            if test_case.test_type == TestType.PERFORMANCE_TEST:
                metrics_dict = await tester.test_inference_speed(dataloader)
                metrics = TestMetrics(
                    inference_time=metrics_dict['mean_inference_time'],
                    throughput=1.0 / metrics_dict['mean_inference_time']
                )
            elif test_case.test_type == TestType.VALIDATION_TEST:
                # Assume labels are available in metadata
                labels = test_case.metadata.get('labels')
                metrics = await tester.test_accuracy(dataloader, labels)
            else:
                # Custom test function
                metrics = await test_case.test_function(tester, dataloader)

            execution_time = time.time() - start_time

            # Check if metrics meet expectations
            status = TestStatus.PASSED
            if test_case.expected_metrics:
                status = self._validate_metrics(metrics, test_case.expected_metrics)

            return expert_name, TestResult(
                test_name=test_case.name,
                status=status,
                metrics=metrics,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return expert_name, TestResult(
                test_name=test_case.name,
                status=TestStatus.FAILED,
                metrics=TestMetrics(),
                execution_time=execution_time,
                error_message=str(e)
            )

    def _validate_metrics(self, actual: TestMetrics, expected: TestMetrics) -> TestStatus:
        """Validate metrics against expectations"""
        tolerance = 0.1  # 10% tolerance

        # Check key metrics
        if expected.accuracy > 0:
            if abs(actual.accuracy - expected.accuracy) > tolerance:
                return TestStatus.FAILED

        if expected.inference_time > 0:
            if actual.inference_time > expected.inference_time * (1 + tolerance):
                return TestStatus.FAILED

        return TestStatus.PASSED

    def run_integration_tests(self,
                            experts: Dict[str, BaseExpert],
                            fusion_system: AdaptiveFusionSystem,
                            dataloader: DataLoader) -> Dict[str, TestResult]:
        """Run integration tests for the complete system"""

        integration_results = {}

        # Test 1: Expert coordination
        try:
            test_result = self._test_expert_coordination(experts, dataloader)
            integration_results['expert_coordination'] = test_result
        except Exception as e:
            integration_results['expert_coordination'] = TestResult(
                test_name='expert_coordination',
                status=TestStatus.FAILED,
                metrics=TestMetrics(),
                execution_time=0.0,
                error_message=str(e)
            )

        # Test 2: Fusion system functionality
        try:
            test_result = self._test_fusion_system(experts, fusion_system, dataloader)
            integration_results['fusion_system'] = test_result
        except Exception as e:
            integration_results['fusion_system'] = TestResult(
                test_name='fusion_system',
                status=TestStatus.FAILED,
                metrics=TestMetrics(),
                execution_time=0.0,
                error_message=str(e)
            )

        # Test 3: End-to-end pipeline
        try:
            test_result = self._test_end_to_end_pipeline(experts, fusion_system, dataloader)
            integration_results['end_to_end'] = test_result
        except Exception as e:
            integration_results['end_to_end'] = TestResult(
                test_name='end_to_end',
                status=TestStatus.FAILED,
                metrics=TestMetrics(),
                execution_time=0.0,
                error_message=str(e)
            )

        return integration_results

    def _test_expert_coordination(self, experts: Dict[str, BaseExpert], dataloader: DataLoader) -> TestResult:
        """Test that experts can run concurrently without conflicts"""
        start_time = time.time()

        # Run experts simultaneously
        with ThreadPoolExecutor(max_workers=len(experts)) as executor:
            futures = []
            for name, expert in experts.items():
                future = executor.submit(self._run_expert_inference, expert, dataloader)
                futures.append((name, future))

            results = {}
            for name, future in futures:
                try:
                    result = future.result(timeout=60)
                    results[name] = result
                except Exception as e:
                    raise RuntimeError(f"Expert {name} failed: {e}")

        execution_time = time.time() - start_time

        return TestResult(
            test_name='expert_coordination',
            status=TestStatus.PASSED,
            metrics=TestMetrics(inference_time=execution_time),
            execution_time=execution_time
        )

    def _test_fusion_system(self,
                           experts: Dict[str, BaseExpert],
                           fusion_system: AdaptiveFusionSystem,
                           dataloader: DataLoader) -> TestResult:
        """Test fusion system functionality"""
        start_time = time.time()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get expert outputs
        expert_outputs = []
        for expert in experts.values():
            expert.to(device)
            expert.eval()

            with torch.no_grad():
                batch = next(iter(dataloader))
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)

                output = expert(inputs)
                expert_outputs.append(output)

        # Test fusion
        fusion_result = fusion_system.fuse_experts(expert_outputs)

        execution_time = time.time() - start_time

        # Validate fusion output
        if 'prediction' not in fusion_result:
            raise ValueError("Fusion system did not produce predictions")

        return TestResult(
            test_name='fusion_system',
            status=TestStatus.PASSED,
            metrics=TestMetrics(inference_time=execution_time),
            execution_time=execution_time
        )

    def _test_end_to_end_pipeline(self,
                                 experts: Dict[str, BaseExpert],
                                 fusion_system: AdaptiveFusionSystem,
                                 dataloader: DataLoader) -> TestResult:
        """Test complete end-to-end pipeline"""
        start_time = time.time()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total_samples = 0
        correct_predictions = 0

        # Test on multiple batches
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Test on first 5 batches
                break

            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                labels = torch.randint(0, 2, (inputs.size(0),)).to(device)  # Mock labels

            # Run experts
            expert_outputs = []
            for expert in experts.values():
                expert.to(device)
                expert.eval()

                with torch.no_grad():
                    output = expert(inputs)
                    expert_outputs.append(output)

            # Fusion
            fusion_result = fusion_system.fuse_experts(expert_outputs, labels)
            predictions = fusion_result['prediction']

            # Calculate accuracy
            binary_pred = (predictions > 0.5).float()
            correct = (binary_pred.squeeze() == labels.float()).sum().item()

            correct_predictions += correct
            total_samples += labels.size(0)

        execution_time = time.time() - start_time
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        return TestResult(
            test_name='end_to_end',
            status=TestStatus.PASSED,
            metrics=TestMetrics(
                accuracy=accuracy,
                inference_time=execution_time,
                throughput=total_samples / execution_time
            ),
            execution_time=execution_time
        )

    def _run_expert_inference(self, expert: BaseExpert, dataloader: DataLoader) -> Any:
        """Helper function to run expert inference"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        expert.to(device)
        expert.eval()

        with torch.no_grad():
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)

            return expert(inputs)

    def generate_test_report(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report"""

        report = {
            'summary': {
                'total_tests': len(test_results),
                'passed': sum(1 for r in test_results.values() if r.status == TestStatus.PASSED),
                'failed': sum(1 for r in test_results.values() if r.status == TestStatus.FAILED),
                'skipped': sum(1 for r in test_results.values() if r.status == TestStatus.SKIPPED)
            },
            'performance_metrics': {},
            'detailed_results': {},
            'recommendations': []
        }

        # Aggregate performance metrics
        all_metrics = [r.metrics for r in test_results.values() if r.status == TestStatus.PASSED]
        if all_metrics:
            report['performance_metrics'] = {
                'avg_accuracy': np.mean([m.accuracy for m in all_metrics if m.accuracy > 0]),
                'avg_inference_time': np.mean([m.inference_time for m in all_metrics if m.inference_time > 0]),
                'avg_throughput': np.mean([m.throughput for m in all_metrics if m.throughput > 0])
            }

        # Detailed results
        for test_name, result in test_results.items():
            report['detailed_results'][test_name] = {
                'status': result.status.value,
                'execution_time': result.execution_time,
                'metrics': {
                    'accuracy': result.metrics.accuracy,
                    'inference_time': result.metrics.inference_time,
                    'memory_usage': result.metrics.memory_usage
                },
                'error_message': result.error_message
            }

        # Generate recommendations
        failed_tests = [name for name, result in test_results.items()
                       if result.status == TestStatus.FAILED]

        if failed_tests:
            report['recommendations'].append(
                f"Review failed tests: {', '.join(failed_tests)}"
            )

        slow_tests = [name for name, result in test_results.items()
                     if result.metrics.inference_time > 1.0]

        if slow_tests:
            report['recommendations'].append(
                f"Optimize performance for: {', '.join(slow_tests)}"
            )

        return report


def create_test_suite(experts: Dict[str, BaseExpert],
                     fusion_system: AdaptiveFusionSystem,
                     dataloader: DataLoader) -> List[TestCase]:
    """Create comprehensive test suite"""

    test_cases = [
        # Performance tests
        TestCase(
            name="inference_speed",
            test_type=TestType.PERFORMANCE_TEST,
            test_function=lambda tester, dl: tester.test_inference_speed(dl),
            expected_metrics=TestMetrics(inference_time=0.1),  # 100ms expectation
            timeout=120
        ),

        # Memory tests
        TestCase(
            name="memory_usage",
            test_type=TestType.PERFORMANCE_TEST,
            test_function=lambda tester, dl: tester.test_memory_usage(next(iter(dl))[0]),
            expected_metrics=TestMetrics(memory_usage=1000),  # 1GB expectation
            timeout=60
        ),

        # Validation tests
        TestCase(
            name="accuracy_validation",
            test_type=TestType.VALIDATION_TEST,
            test_function=lambda tester, dl: tester.test_accuracy(dl, torch.randint(0, 2, (100,))),
            expected_metrics=TestMetrics(accuracy=0.8),  # 80% accuracy expectation
            timeout=300
        )
    ]

    return test_cases


async def run_concurrent_tests(experts: Dict[str, BaseExpert],
                              fusion_system: AdaptiveFusionSystem,
                              dataloader: DataLoader,
                              config: Optional[ConcurrentTestConfig] = None) -> Dict[str, Any]:
    """
    Main function to run comprehensive concurrent testing
    """
    if config is None:
        config = ConcurrentTestConfig()

    # Create test executor
    executor = ConcurrentTestExecutor(config)

    # Create test suite
    test_cases = create_test_suite(experts, fusion_system, dataloader)

    # Run expert tests
    expert_results = await executor.run_expert_tests(experts, test_cases, dataloader)

    # Run integration tests
    integration_results = executor.run_integration_tests(experts, fusion_system, dataloader)

    # Combine results
    all_results = {**expert_results, **integration_results}

    # Generate report
    report = executor.generate_test_report(all_results)

    return {
        'test_results': all_results,
        'report': report,
        'config': config
    }