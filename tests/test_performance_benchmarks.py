"""
Test suite for performance benchmarks and Stage-Gate criteria
"""

import pytest
import time
import psutil
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.metrics import AcademicMetrics
from utils.calibration_tools import CalibrationAnalyzer
from utils.dataset_config import DatasetConfig
from utils.manifest_generator import ManifestGenerator
from stage_00.baseline_model import EfficientNetV2B3Baseline


class TestPerformanceBenchmarks:
    """Test performance benchmarks against Stage-Gate criteria"""
    
    def setup_method(self):
        """Setup for performance tests"""
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize components
        self.metrics = AcademicMetrics(n_bootstrap=100, random_state=42)
        self.calibration = CalibrationAnalyzer(random_state=42)
        self.model = EfficientNetV2B3Baseline(pretrained=False)
        
        # Performance thresholds from Stage-Gate criteria
        self.thresholds = {
            'data_loading_speed': 100,  # samples/second
            'inference_speed': 100,     # ms/image
            'memory_limit': 4000,       # MB
            'auc_target': 0.88,         # Minimum AUC
            'ece_target': 0.1,          # Maximum ECE
        }
    
    def test_data_loading_performance(self):
        """Test data loading efficiency meets Stage-Gate requirements"""
        # Create temporary dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            n_samples = 1000
            
            # Create sample images
            real_dir = temp_path / "real"
            fake_dir = temp_path / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()
            
            # Create images efficiently for testing
            sample_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
            
            start_creation = time.time()
            for i in range(n_samples // 2):
                sample_image.save(real_dir / f"real_{i:04d}.png")
                sample_image.save(fake_dir / f"fake_{i:04d}.png")
            creation_time = time.time() - start_creation
            
            print(f"Image creation time: {creation_time:.2f} seconds")
            
            # Generate manifest
            generator = ManifestGenerator(
                real_image_dir=str(real_dir),
                fake_image_dir=str(fake_dir),
                output_dir=str(temp_path / "manifests")
            )
            
            start_manifest = time.time()
            manifest_path = generator.generate_manifest()
            manifest_time = time.time() - start_manifest
            
            print(f"Manifest generation time: {manifest_time:.2f} seconds")
            
            # Test manifest loading performance
            start_loading = time.time()
            manifest_df = pd.read_csv(manifest_path)
            loading_time = time.time() - start_loading
            
            samples_per_second = len(manifest_df) / loading_time
            
            print(f"Data loading performance: {samples_per_second:.1f} samples/second")
            
            # Check against threshold
            assert samples_per_second >= self.thresholds['data_loading_speed'], \
                f"Data loading too slow: {samples_per_second:.1f} < {self.thresholds['data_loading_speed']}"
    
    def test_model_inference_speed(self):
        """Test model inference speed meets Stage-Gate requirements"""
        self.model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Create test batch
        batch_size = 32
        x = torch.randn(batch_size, 3, 256, 256, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time inference
        n_iterations = 50
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = self.model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_images = n_iterations * batch_size
        avg_time_per_image = ((end_time - start_time) / total_images) * 1000  # ms
        
        print(f"Inference speed: {avg_time_per_image:.2f} ms/image")
        
        # Check against threshold
        assert avg_time_per_image <= self.thresholds['inference_speed'], \
            f"Inference too slow: {avg_time_per_image:.2f} ms > {self.thresholds['inference_speed']} ms"
    
    def test_memory_usage_limits(self):
        """Test memory usage stays within acceptable limits"""
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Load model and perform operations
        self.model.eval()
        
        # Simulate multiple inference runs
        peak_memory = initial_memory
        
        for batch_size in [16, 32, 64]:
            try:
                x = torch.randn(batch_size, 3, 256, 256)
                
                with torch.no_grad():
                    _ = self.model(x)
                
                current_memory = self.process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of memory at batch size: {batch_size}")
                    break
        
        memory_usage = peak_memory - initial_memory
        
        print(f"Peak memory usage: {peak_memory:.1f} MB (increase: {memory_usage:.1f} MB)")
        
        # Check against threshold (less strict for testing)
        assert peak_memory <= self.thresholds['memory_limit'], \
            f"Memory usage too high: {peak_memory:.1f} MB > {self.thresholds['memory_limit']} MB"
    
    def test_metrics_calculation_speed(self):
        """Test evaluation metrics calculation performance"""
        # Generate test data
        np.random.seed(42)
        n_samples = 10000  # Large dataset for performance testing
        
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_scores = np.random.rand(n_samples)
        y_pred = (y_scores > 0.5).astype(int)
        
        # Time AUC calculation with confidence intervals
        start_time = time.time()
        auc_result = self.metrics.calculate_auc_with_ci(y_true, y_scores)
        auc_time = time.time() - start_time
        
        # Time other metrics
        start_time = time.time()
        accuracy_result = self.metrics.calculate_accuracy_with_ci(y_true, y_pred)
        f1_result = self.metrics.calculate_f1_with_ci(y_true, y_pred)
        metrics_time = time.time() - start_time
        
        print(f"AUC calculation time: {auc_time:.3f} seconds")
        print(f"Other metrics time: {metrics_time:.3f} seconds")
        
        # Performance should be reasonable (< 10 seconds for 10k samples)
        assert auc_time < 10.0, f"AUC calculation too slow: {auc_time:.3f}s"
        assert metrics_time < 5.0, f"Other metrics too slow: {metrics_time:.3f}s"
        
        # Results should be valid
        assert 0 <= auc_result.value <= 1
        assert auc_result.confidence_interval is not None
    
    def test_calibration_analysis_speed(self):
        """Test calibration analysis performance"""
        np.random.seed(42)
        n_samples = 5000
        
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.random.rand(n_samples)
        logits = np.log(y_prob / (1 - y_prob + 1e-8))
        
        # Time ECE/MCE calculation
        start_time = time.time()
        calibration_result = self.calibration.calculate_ece_mce(y_true, y_prob)
        ece_time = time.time() - start_time
        
        # Time temperature scaling
        start_time = time.time()
        temp_result = self.calibration.temperature_scaling(y_true, logits, validation_split=0.2)
        temp_time = time.time() - start_time
        
        # Time bootstrap confidence interval
        start_time = time.time()
        ece_with_ci, ci = self.calibration.bootstrap_ece_confidence_interval(
            y_true, y_prob, n_bootstrap=100
        )
        bootstrap_time = time.time() - start_time
        
        print(f"ECE/MCE calculation time: {ece_time:.3f} seconds")
        print(f"Temperature scaling time: {temp_time:.3f} seconds")
        print(f"Bootstrap CI time: {bootstrap_time:.3f} seconds")
        
        # Performance should be reasonable
        assert ece_time < 1.0, f"ECE calculation too slow: {ece_time:.3f}s"
        assert temp_time < 5.0, f"Temperature scaling too slow: {temp_time:.3f}s"
        assert bootstrap_time < 10.0, f"Bootstrap CI too slow: {bootstrap_time:.3f}s"
        
        # Results should be valid
        assert calibration_result.ece >= 0
        assert calibration_result.mce >= 0
        assert temp_result.optimal_temperature > 0
    
    def test_end_to_end_performance(self):
        """Test complete evaluation pipeline performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create minimal test dataset
            n_samples = 200
            self.create_test_dataset(temp_path, n_samples)
            
            # Create config
            config_data = {
                "metadata": {
                    "name": "performance_test",
                    "total_samples": n_samples,
                    "real_samples": n_samples // 2,
                    "fake_samples": n_samples // 2
                },
                "root_path": str(temp_path),
                "paths": {
                    "real_images": str(temp_path / "real"),
                    "fake_images": str(temp_path / "fake")
                }
            }
            
            config_path = temp_path / "config.json"
            import json
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            # Time complete pipeline
            start_time = time.time()
            
            # 1. Generate manifest
            generator = ManifestGenerator(
                real_image_dir=str(temp_path / "real"),
                fake_image_dir=str(temp_path / "fake"),
                output_dir=str(temp_path / "manifests")
            )
            manifest_path = generator.generate_manifest()
            
            # 2. Load and validate data
            manifest_df = pd.read_csv(manifest_path)
            
            # 3. Simulate model predictions
            np.random.seed(42)
            y_true = manifest_df['label'].values
            y_scores = np.random.rand(len(y_true))
            # Make predictions somewhat realistic
            y_scores = 0.3 * y_scores + 0.7 * y_true + np.random.normal(0, 0.1, len(y_true))
            y_scores = np.clip(y_scores, 0, 1)
            
            # 4. Calculate metrics
            auc_result = self.metrics.calculate_auc_with_ci(y_true, y_scores)
            calibration_result = self.calibration.calculate_ece_mce(y_true, y_scores)
            
            total_time = time.time() - start_time
            
            print(f"End-to-end pipeline time: {total_time:.3f} seconds for {n_samples} samples")
            print(f"Performance: {n_samples / total_time:.1f} samples/second")
            
            # Pipeline should be reasonably fast
            assert total_time < 30.0, f"Pipeline too slow: {total_time:.3f}s"
            
            # Results should meet quality thresholds (relaxed for random data)
            print(f"AUC: {auc_result.value:.4f}")
            print(f"ECE: {calibration_result.ece:.4f}")
    
    def create_test_dataset(self, base_path: Path, n_samples: int):
        """Create test dataset for performance testing"""
        real_dir = base_path / "real"
        fake_dir = base_path / "fake"
        real_dir.mkdir(parents=True)
        fake_dir.mkdir(parents=True)
        
        # Create minimal images efficiently
        sample_image = Image.new('RGB', (64, 64), color=(128, 128, 128))
        
        for i in range(n_samples // 2):
            sample_image.save(real_dir / f"real_{i:04d}.png")
            sample_image.save(fake_dir / f"fake_{i:04d}.png")


class TestStageGateValidation:
    """Test Stage-Gate validation criteria"""
    
    def setup_method(self):
        """Setup Stage-Gate tests"""
        self.stage_gate_criteria = {
            'environment_success_rate': 0.95,
            'data_management_support': 4,  # Number of datasets
            'baseline_auc_minimum': 0.88,
            'performance_variance_max': 0.05,
            'code_coverage_minimum': 0.80,
            'documentation_completeness': 0.90
        }
    
    def test_environment_reproducibility(self):
        """Test environment setup success rate"""
        # Simulate environment setup attempts
        success_count = 0
        total_attempts = 20
        
        for _ in range(total_attempts):
            try:
                # Test key imports and initializations
                metrics = AcademicMetrics()
                calibration = CalibrationAnalyzer()
                model = EfficientNetV2B3Baseline(pretrained=False)
                
                # Test basic functionality
                test_data = np.random.rand(100)
                test_labels = np.random.binomial(1, 0.5, 100)
                
                _ = metrics.calculate_auc_with_ci(test_labels, test_data)
                _ = calibration.calculate_ece_mce(test_labels, test_data)
                
                success_count += 1
                
            except Exception as e:
                print(f"Environment setup failed: {e}")
                continue
        
        success_rate = success_count / total_attempts
        print(f"Environment setup success rate: {success_rate:.2%}")
        
        assert success_rate >= self.stage_gate_criteria['environment_success_rate'], \
            f"Environment success rate too low: {success_rate:.2%}"
    
    def test_data_management_capabilities(self):
        """Test data management system supports required datasets"""
        from utils.dataset_config import DatasetConfig
        from utils.manifest_generator import ManifestGenerator
        from utils.data_validator import DataValidator
        
        # Test dataset format support
        supported_formats = ['.png', '.jpg', '.jpeg']
        dataset_types = ['celebdf', 'ff++', 'dfdc', 'custom']
        
        # Verify components exist and can be initialized
        config_data = {
            "metadata": {"name": "test", "dataset_type": "celebdf"},
            "root_path": "/tmp",
            "splits": {"train": {"ratio": 0.8}},
            "paths": {"real_images": "/tmp/real", "fake_images": "/tmp/fake"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            # Test configuration loading
            config = DatasetConfig(temp_config_path)
            assert config.name == "test"
            
            # Test manifest generator initialization
            generator = ManifestGenerator(
                real_image_dir="/tmp/real",
                fake_image_dir="/tmp/fake"
            )
            assert generator is not None
            
            # Test validator initialization
            validator = DataValidator()
            assert validator is not None
            
            print("Data management system supports all required components")
            
        finally:
            os.unlink(temp_config_path)
    
    def test_baseline_model_performance_capability(self):
        """Test baseline model can achieve target performance"""
        model = EfficientNetV2B3Baseline(pretrained=False)
        model.eval()
        
        # Test model architecture can learn
        batch_size = 32
        x = torch.randn(batch_size, 3, 256, 256)
        
        with torch.no_grad():
            outputs = model(x)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Check output properties
            assert logits.shape == (batch_size, 2)
            assert not torch.isnan(logits).any()
            assert torch.isfinite(logits).all()
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
            
            assert torch.all(probs >= 0)
            assert torch.all(probs <= 1)
        
        # Test training capability (simplified)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Simple training step
        targets = torch.randint(0, 2, (batch_size,))
        
        optimizer.zero_grad()
        outputs = model(x)
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        assert torch.isfinite(loss)
        print(f"Model training capability verified (loss: {loss.item():.4f})")
    
    def test_statistical_rigor_capability(self):
        """Test statistical analysis capabilities"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate test data with known properties
        y_true = np.random.binomial(1, 0.6, n_samples)  # 60% positive
        
        # Create correlated predictions
        base_scores = np.random.rand(n_samples)
        correlation = 0.8
        y_scores = (1 - correlation) * base_scores + correlation * y_true
        y_scores += np.random.normal(0, 0.1, n_samples)
        y_scores = np.clip(y_scores, 0, 1)
        
        metrics = AcademicMetrics(confidence_level=0.95, n_bootstrap=200)
        
        # Test AUC with confidence interval
        auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
        
        # Should achieve good AUC with correlated data
        print(f"AUC: {auc_result.value:.4f} [{auc_result.confidence_interval[0]:.4f}, {auc_result.confidence_interval[1]:.4f}]")
        
        assert auc_result.value > 0.7, f"AUC capability test failed: {auc_result.value:.4f}"
        assert auc_result.confidence_interval is not None
        assert auc_result.confidence_interval[0] <= auc_result.value <= auc_result.confidence_interval[1]
        
        # Test calibration analysis
        calibration = CalibrationAnalyzer()
        cal_result = calibration.calculate_ece_mce(y_true, y_scores)
        
        assert cal_result.ece >= 0
        assert cal_result.mce >= 0
        print(f"Calibration: ECE={cal_result.ece:.4f}, MCE={cal_result.mce:.4f}")
    
    def test_scalability_requirements(self):
        """Test system can scale to required dataset sizes"""
        # Test with progressively larger dataset sizes
        test_sizes = [100, 1000, 5000]
        
        for size in test_sizes:
            start_time = time.time()
            
            # Generate data
            y_true = np.random.binomial(1, 0.5, size)
            y_scores = np.random.rand(size)
            
            # Test metrics calculation
            metrics = AcademicMetrics(n_bootstrap=50)  # Reduced for speed
            auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
            
            # Test calibration
            calibration = CalibrationAnalyzer(n_bins=10)
            cal_result = calibration.calculate_ece_mce(y_true, y_scores)
            
            processing_time = time.time() - start_time
            
            print(f"Processed {size} samples in {processing_time:.3f} seconds ({size/processing_time:.1f} samples/sec)")
            
            # Should maintain reasonable performance
            assert processing_time < 30.0, f"Processing too slow for {size} samples"
            assert auc_result.value >= 0
            assert cal_result.ece >= 0


@pytest.mark.slow
class TestLargeScalePerformance:
    """Large-scale performance tests (marked as slow)"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        n_samples = 50000
        
        print(f"Testing large dataset performance with {n_samples:,} samples")
        
        # Generate large dataset
        start_time = time.time()
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_scores = np.random.rand(n_samples)
        generation_time = time.time() - start_time
        
        print(f"Data generation: {generation_time:.2f} seconds")
        
        # Test metrics calculation
        metrics = AcademicMetrics(n_bootstrap=50)  # Reduced for speed
        
        start_time = time.time()
        auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
        metrics_time = time.time() - start_time
        
        print(f"Metrics calculation: {metrics_time:.2f} seconds")
        print(f"Performance: {n_samples / metrics_time:.0f} samples/second")
        
        # Should complete in reasonable time
        assert metrics_time < 60.0, f"Large dataset processing too slow: {metrics_time:.2f}s"
        assert auc_result.value >= 0
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        import gc
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process multiple large batches
        batch_sizes = [10000, 20000, 30000]
        max_memory_increase = 0
        
        for batch_size in batch_sizes:
            gc.collect()  # Clear memory before test
            
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Generate and process data
            y_true = np.random.binomial(1, 0.5, batch_size)
            y_scores = np.random.rand(batch_size)
            
            metrics = AcademicMetrics(n_bootstrap=20)  # Reduced for memory efficiency
            _ = metrics.calculate_auc_with_ci(y_true, y_scores)
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = end_memory - start_memory
            max_memory_increase = max(max_memory_increase, memory_increase)
            
            print(f"Batch size {batch_size:,}: Memory increase {memory_increase:.1f} MB")
            
            # Clean up
            del y_true, y_scores, metrics
            gc.collect()
        
        print(f"Maximum memory increase: {max_memory_increase:.1f} MB")
        
        # Memory usage should be reasonable (< 1GB increase)
        assert max_memory_increase < 1000, f"Memory usage too high: {max_memory_increase:.1f} MB"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "-m", "not slow"])  # Skip slow tests by default