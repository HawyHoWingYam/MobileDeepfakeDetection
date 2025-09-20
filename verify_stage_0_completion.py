#!/usr/bin/env python3
"""
AWARE-NET Stage 0: Complete System Verification
Final integration test and performance validation
"""

import sys
import os
import json
import time
import tempfile
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_demo_dataset(base_dir: Path, n_samples: int = 100) -> Dict[str, str]:
    """Create demonstration dataset for verification"""
    print(f"Creating demo dataset with {n_samples} samples...")
    
    # Create directories
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"
    manifests_dir = base_dir / "manifests"
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample images
    n_real = n_samples // 2
    n_fake = n_samples - n_real
    
    # Generate real images (solid colors)
    for i in range(n_real):
        color = (100 + i % 156, 150, 200)  # Bluish tones
        image = Image.new('RGB', (256, 256), color)
        image.save(real_dir / f"real_{i:04d}.png")
    
    # Generate fake images (different colors)  
    for i in range(n_fake):
        color = (200, 100 + i % 156, 150)  # Reddish tones
        image = Image.new('RGB', (256, 256), color)
        image.save(fake_dir / f"fake_{i:04d}.png")
    
    # Generate manifests using ManifestGenerator
    from utils.manifest_generator import ManifestGenerator
    
    generator = ManifestGenerator(
        real_image_dir=str(real_dir),
        fake_image_dir=str(fake_dir),
        output_dir=str(manifests_dir)
    )
    
    # Generate split manifests
    split_manifests = generator.generate_split_manifests(
        split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
        stratify=True,
        random_state=42
    )
    
    print(f"✅ Demo dataset created successfully")
    print(f"   Real images: {n_real}, Fake images: {n_fake}")
    for split, path in split_manifests.items():
        df = pd.read_csv(path)
        print(f"   {split}: {len(df)} samples")
    
    return {split: str(path) for split, path in split_manifests.items()}


def verify_calibration_tools():
    """Verify calibration tools functionality"""
    print("\n" + "="*60)
    print("Testing Calibration Tools")
    print("="*60)
    
    from utils.calibration_tools import CalibrationAnalyzer
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.6, n_samples)
    
    # Create correlated predictions for realistic results
    base_scores = np.random.rand(n_samples)
    correlation = 0.7
    y_prob = (1 - correlation) * base_scores + correlation * y_true
    y_prob += np.random.normal(0, 0.1, n_samples)
    y_prob = np.clip(y_prob, 0, 1)
    
    # Initialize analyzer
    analyzer = CalibrationAnalyzer(n_bins=10, random_state=42)
    
    # Test ECE/MCE calculation
    start_time = time.time()
    calibration_result = analyzer.calculate_ece_mce(y_true, y_prob)
    ece_time = time.time() - start_time
    
    print(f"✅ ECE/MCE Calculation:")
    print(f"   ECE: {calibration_result.ece:.4f}")
    print(f"   MCE: {calibration_result.mce:.4f}")
    print(f"   Brier Score: {calibration_result.brier_score:.4f}")
    print(f"   Calculation Time: {ece_time:.3f}s")
    
    # Test bootstrap confidence interval
    start_time = time.time()
    ece_with_ci, ci = analyzer.bootstrap_ece_confidence_interval(
        y_true, y_prob, n_bootstrap=100
    )
    bootstrap_time = time.time() - start_time
    
    print(f"✅ Bootstrap Confidence Interval:")
    print(f"   ECE: {ece_with_ci:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"   Calculation Time: {bootstrap_time:.3f}s")
    
    # Test temperature scaling
    logits = np.log(y_prob / (1 - y_prob + 1e-8))  # Convert to logits
    
    start_time = time.time()
    temp_result = analyzer.temperature_scaling(y_true, logits, validation_split=0.2)
    temp_time = time.time() - start_time
    
    print(f"✅ Temperature Scaling:")
    print(f"   Optimal Temperature: {temp_result.optimal_temperature:.4f}")
    print(f"   Pre-calibration ECE: {temp_result.pre_calibration_ece:.4f}")
    print(f"   Post-calibration ECE: {temp_result.post_calibration_ece:.4f}")
    print(f"   Improvement: {temp_result.improvement:.4f}")
    print(f"   Calculation Time: {temp_time:.3f}s")
    
    return {
        'ece': calibration_result.ece,
        'mce': calibration_result.mce,
        'ece_with_ci': ece_with_ci,
        'temperature_improvement': temp_result.improvement,
        'performance': {
            'ece_calculation_time': ece_time,
            'bootstrap_time': bootstrap_time,
            'temp_scaling_time': temp_time
        }
    }


def verify_academic_metrics():
    """Verify academic metrics functionality"""
    print("\n" + "="*60)
    print("Testing Academic Metrics")
    print("="*60)
    
    from utils.metrics import AcademicMetrics
    
    # Generate test data
    np.random.seed(42)
    n_samples = 2000
    
    # Create realistic classification scenario
    y_true = np.random.binomial(1, 0.4, n_samples)  # 40% positive
    
    # Create correlated predictions
    base_scores = np.random.beta(2, 5, n_samples)
    correlation = 0.8
    y_scores = (1 - correlation) * base_scores + correlation * y_true
    y_scores += np.random.normal(0, 0.08, n_samples)
    y_scores = np.clip(y_scores, 0, 1)
    
    y_pred = (y_scores > 0.5).astype(int)
    
    # Initialize metrics
    metrics = AcademicMetrics(confidence_level=0.95, n_bootstrap=200, random_state=42)
    
    results = {}
    
    # Test AUC with CI
    start_time = time.time()
    auc_result = metrics.calculate_auc_with_ci(y_true, y_scores)
    auc_time = time.time() - start_time
    
    print(f"✅ AUC-ROC with CI:")
    print(f"   AUC: {auc_result.value:.4f} [{auc_result.confidence_interval[0]:.4f}, {auc_result.confidence_interval[1]:.4f}]")
    print(f"   Std Error: {auc_result.std_error:.4f}")
    print(f"   Calculation Time: {auc_time:.3f}s")
    results['auc'] = auc_result.value
    
    # Test other metrics
    metrics_to_test = [
        ('accuracy', metrics.calculate_accuracy_with_ci),
        ('f1', metrics.calculate_f1_with_ci),
        ('precision', metrics.calculate_precision_with_ci),
        ('recall', metrics.calculate_recall_with_ci)
    ]
    
    total_metrics_time = 0
    for metric_name, metric_func in metrics_to_test:
        start_time = time.time()
        result = metric_func(y_true, y_pred)
        metric_time = time.time() - start_time
        total_metrics_time += metric_time
        
        print(f"✅ {metric_name.title()}:")
        print(f"   Value: {result.value:.4f} [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        print(f"   Time: {metric_time:.3f}s")
        results[metric_name] = result.value
    
    results['performance'] = {
        'auc_time': auc_time,
        'total_metrics_time': total_metrics_time
    }
    
    return results


def verify_baseline_model():
    """Verify baseline model functionality"""
    print("\n" + "="*60)
    print("Testing Baseline Model")
    print("="*60)
    
    from stage_00.baseline_model import EfficientNetV2B3Baseline
    
    # Initialize model
    model = EfficientNetV2B3Baseline(
        num_classes=2,
        pretrained=False,  # Avoid downloading for demo
        dropout_rate=0.2
    )
    
    print(f"✅ Model initialized successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{(total_params * 4) / (1024**2):.1f} MB")
    
    # Test inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    batch_sizes = [1, 4, 16] if device.type == 'cpu' else [1, 4, 16, 32]
    inference_times = []
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 256, 256, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                outputs = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time_per_sample = ((end_time - start_time) / (10 * batch_size)) * 1000  # ms
        inference_times.append(avg_time_per_sample)
        
        print(f"✅ Batch size {batch_size:2d}: {avg_time_per_sample:.2f} ms/sample")
        
        # Verify output shape
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
            
        assert logits.shape == (batch_size, 2), f"Unexpected output shape: {logits.shape}"
        assert torch.isfinite(logits).all(), "Non-finite outputs detected"
    
    # Test training step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    x = torch.randn(8, 3, 256, 256, device=device)
    targets = torch.randint(0, 2, (8,), device=device)
    
    optimizer.zero_grad()
    outputs = model(x)
    if isinstance(outputs, dict):
        logits = outputs['logits']
    else:
        logits = outputs
    
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step successful (loss: {loss.item():.4f})")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'inference_times': inference_times,
        'avg_inference_time': np.mean(inference_times),
        'device': str(device),
        'training_loss': loss.item()
    }


def verify_integration():
    """Verify complete system integration"""
    print("\n" + "="*60)
    print("Testing Complete System Integration")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create demo dataset
        manifests = create_demo_dataset(temp_path, n_samples=200)
        
        # Create configuration
        config_data = {
            "metadata": {
                "name": "integration_test",
                "total_samples": 200,
                "real_samples": 100,
                "fake_samples": 100,
                "image_size": [256, 256]
            },
            "root_path": str(temp_path),
            "paths": {
                "real_images": str(temp_path / "real"),
                "fake_images": str(temp_path / "fake")
            }
        }
        
        config_path = temp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Test configuration created")
        
        # Test DatasetConfig
        from utils.dataset_config import DatasetConfig
        
        config = DatasetConfig(str(config_path))
        print(f"✅ DatasetConfig loaded: {config.name} ({config.total_samples} samples)")
        
        # Test data validation
        from utils.data_validator import DataValidator
        
        validator = DataValidator()
        validation_result = validator.validate_manifest(manifests['test'])
        
        print(f"✅ Data validation:")
        print(f"   Valid: {validation_result['is_valid']}")
        print(f"   Samples: {validation_result['total_samples']}")
        print(f"   Real: {validation_result['class_distribution'].get(0, 0)}")
        print(f"   Fake: {validation_result['class_distribution'].get(1, 0)}")
        
        # Simulate model evaluation (without actual training)
        test_df = pd.read_csv(manifests['test'])
        n_test_samples = len(test_df)
        
        # Generate realistic predictions
        np.random.seed(42)
        y_true = test_df['label'].values
        
        # Create somewhat correlated predictions
        base_prob = np.random.rand(n_test_samples)
        correlation = 0.75  # Good but not perfect correlation
        y_prob = (1 - correlation) * base_prob + correlation * y_true
        y_prob += np.random.normal(0, 0.1, n_test_samples)
        y_prob = np.clip(y_prob, 0, 1)
        
        # Test academic metrics
        from utils.metrics import AcademicMetrics
        metrics = AcademicMetrics(n_bootstrap=100)  # Reduced for speed
        
        auc_result = metrics.calculate_auc_with_ci(y_true, y_prob)
        accuracy_result = metrics.calculate_accuracy_with_ci(y_true, (y_prob > 0.5).astype(int))
        
        print(f"✅ Simulated evaluation results:")
        print(f"   AUC: {auc_result.value:.4f} [{auc_result.confidence_interval[0]:.4f}, {auc_result.confidence_interval[1]:.4f}]")
        print(f"   Accuracy: {accuracy_result.value:.4f}")
        
        # Test calibration analysis
        from utils.calibration_tools import CalibrationAnalyzer
        
        calibration = CalibrationAnalyzer(n_bins=8)  # Reduced for small dataset
        cal_result = calibration.calculate_ece_mce(y_true, y_prob)
        
        print(f"✅ Calibration analysis:")
        print(f"   ECE: {cal_result.ece:.4f}")
        print(f"   MCE: {cal_result.mce:.4f}")
        
        return {
            'auc': auc_result.value,
            'accuracy': accuracy_result.value,
            'ece': cal_result.ece,
            'n_test_samples': n_test_samples,
            'data_validation_passed': validation_result['is_valid']
        }


def run_stage_gate_check():
    """Run final Stage-Gate validation"""
    print("\n" + "="*60)
    print("Running Final Stage-Gate Check")
    print("="*60)
    
    try:
        # Import stage gate validator
        sys.path.insert(0, str(Path(__file__).parent))
        from stage_gate_validator import StageGateValidator
        
        validator = StageGateValidator()
        
        # Run quick validation (subset of full validation)
        print("Running quick Stage-Gate validation...")
        
        technical_results = validator.validate_technical_gates()
        academic_results = validator.validate_academic_gates()
        
        # Count passes
        technical_passed = sum(1 for r in technical_results if r.threshold_met)
        academic_passed = sum(1 for r in academic_results if r.threshold_met)
        
        print(f"✅ Technical Gates: {technical_passed}/{len(technical_results)} passed")
        for result in technical_results:
            status = "✅" if result.threshold_met else "❌"
            print(f"   {status} {result.criterion}: {result.score:.1f}/100")
        
        print(f"✅ Academic Gates: {academic_passed}/{len(academic_results)} passed")
        for result in academic_results:
            status = "✅" if result.threshold_met else "❌"
            print(f"   {status} {result.criterion}: {result.score:.1f}/100")
        
        total_passed = technical_passed + academic_passed
        total_criteria = len(technical_results) + len(academic_results)
        success_rate = total_passed / total_criteria
        
        return {
            'technical_passed': technical_passed,
            'technical_total': len(technical_results),
            'academic_passed': academic_passed,
            'academic_total': len(academic_results),
            'overall_success_rate': success_rate,
            'stage_gate_ready': success_rate >= 0.8  # 80% threshold
        }
    
    except Exception as e:
        print(f"❌ Stage-Gate validation error: {e}")
        return {
            'error': str(e),
            'stage_gate_ready': False
        }


def main():
    """Main verification script"""
    print("🚀 AWARE-NET Stage 0: Complete System Verification")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__ if 'torch' in sys.modules else 'Not loaded'}")
    
    verification_results = {}
    
    try:
        # 1. Verify Calibration Tools
        calibration_results = verify_calibration_tools()
        verification_results['calibration'] = calibration_results
        
        # 2. Verify Academic Metrics
        metrics_results = verify_academic_metrics()
        verification_results['metrics'] = metrics_results
        
        # 3. Verify Baseline Model
        model_results = verify_baseline_model()
        verification_results['baseline_model'] = model_results
        
        # 4. Verify Integration
        integration_results = verify_integration()
        verification_results['integration'] = integration_results
        
        # 5. Run Stage-Gate Check
        stage_gate_results = run_stage_gate_check()
        verification_results['stage_gate'] = stage_gate_results
        
        # Generate final report
        print("\n" + "="*80)
        print("FINAL VERIFICATION REPORT")
        print("="*80)
        
        # Performance summary
        print("📊 Performance Summary:")
        print(f"   Calibration ECE Calculation: {calibration_results['performance']['ece_calculation_time']:.3f}s")
        print(f"   Metrics AUC Calculation: {metrics_results['performance']['auc_time']:.3f}s")
        print(f"   Model Inference (avg): {model_results['avg_inference_time']:.2f} ms/sample")
        
        # Quality summary
        print("\n📈 Quality Summary:")
        print(f"   Simulated AUC: {integration_results['auc']:.4f}")
        print(f"   Simulated ECE: {integration_results['ece']:.4f}")
        print(f"   Data Validation: {'✅ Passed' if integration_results['data_validation_passed'] else '❌ Failed'}")
        
        # Stage-Gate summary
        if 'error' not in stage_gate_results:
            success_rate = stage_gate_results['overall_success_rate']
            print(f"\n🎯 Stage-Gate Summary:")
            print(f"   Overall Success Rate: {success_rate:.1%}")
            print(f"   Technical Gates: {stage_gate_results['technical_passed']}/{stage_gate_results['technical_total']}")
            print(f"   Academic Gates: {stage_gate_results['academic_passed']}/{stage_gate_results['academic_total']}")
            print(f"   Status: {'✅ READY' if stage_gate_results['stage_gate_ready'] else '⚠️ NEEDS WORK'}")
        
        # Overall assessment
        overall_success = all([
            calibration_results['ece'] < 0.5,  # Reasonable calibration
            metrics_results['auc'] > 0.6,     # Reasonable AUC
            model_results['avg_inference_time'] < 200,  # Reasonable speed (ms)
            integration_results['data_validation_passed'],
            stage_gate_results.get('stage_gate_ready', False)
        ])
        
        print(f"\n{'='*80}")
        if overall_success:
            print("🎉 VERIFICATION SUCCESSFUL: Stage 0 Implementation Complete!")
            print("✅ All systems operational and ready for Stage 1")
        else:
            print("⚠️  VERIFICATION PARTIAL: Some improvements recommended")
            print("📋 Review detailed results for optimization opportunities")
        print(f"{'='*80}")
        
        # Save results
        with open('stage_0_verification_results.json', 'w') as f:
            json.dump(verification_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: stage_0_verification_results.json")
        
        return 0 if overall_success else 1
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)