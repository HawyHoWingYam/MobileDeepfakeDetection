"""
Test script for EfficientNetV2 model selection
Tests both tf_efficientnetv2_b0 and efficientnetv2_rw_t configurations
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_model_creation():
    """Test creating both model variants"""
    from stage_00.baseline_model import create_baseline_model
    
    models_to_test = [
        ('tf_efficientnetv2_b0', 'EfficientNetV2-B0'),
        ('efficientnetv2_rw_t', 'EfficientNetV2-rw-t')
    ]
    
    print("Testing EfficientNetV2 Model Selection")
    print("=" * 50)
    
    results = []
    
    for model_name, display_name in models_to_test:
        print(f"\nTesting {display_name} ({model_name})...")
        
        try:
            # Test model creation
            model = create_baseline_model(
                pretrained=True,
                dropout_rate=0.2,
                model_name=model_name
            )
            
            # Test forward pass
            with torch.no_grad():
                x = torch.randn(2, 3, 256, 256)
                logits = model(x)
                probs = model.predict_proba(x)
                features = model.get_features(x)
            
            info = model.get_model_info()
            
            print(f"  [OK] {display_name} works correctly")
            print(f"     Input: {x.shape}")
            print(f"     Logits: {logits.shape}")
            print(f"     Probabilities: {probs.shape}")
            print(f"     Features: {features.shape}")
            print(f"     Feature dimension: {info['feature_dim']}")
            print(f"     Total parameters: {info['total_parameters']:,}")
            print(f"     Prob range: [{probs.min():.3f}, {probs.max():.3f}]")
            
            results.append({
                'model_name': model_name,
                'display_name': display_name,
                'success': True,
                'info': info,
                'logits_shape': logits.shape,
                'probs_shape': probs.shape,
                'features_shape': features.shape
            })
            
        except Exception as e:
            print(f"  [ERROR] {display_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model_name': model_name,
                'display_name': display_name,
                'success': False,
                'error': str(e)
            })
    
    print(f"\n{'=' * 50}")
    print("Summary:")
    for result in results:
        if result['success']:
            print(f"  [OK] {result['display_name']}: {result['info']['feature_dim']} features, {result['info']['total_parameters']:,} params")
        else:
            print(f"  [ERROR] {result['display_name']}: {result['error']}")
    
    successful_models = [r for r in results if r['success']]
    print(f"\nSuccessful models: {len(successful_models)}/{len(results)}")
    
    return results

def test_training_script_compatibility():
    """Test that training script accepts model parameters"""
    print(f"\n{'=' * 50}")
    print("Testing Training Script Compatibility")
    print("=" * 50)
    
    # Test help output to see if --model parameter is available
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, 'src/stage_00/train_baseline.py', '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if '--model' in result.stdout:
            print("  [OK] Training script supports --model parameter")
            
            # Extract model choices from help text
            for line in result.stdout.split('\n'):
                if '--model' in line and 'choices' in line:
                    print(f"  [INFO] {line.strip()}")
        else:
            print("  [ERROR] Training script does not support --model parameter")
            
    except Exception as e:
        print(f"  [ERROR] Error testing training script: {e}")

if __name__ == "__main__":
    print("EfficientNetV2 Model Selection Test")
    print("=" * 50)
    
    # Test model creation
    results = test_model_creation()
    
    # Test training script compatibility
    test_training_script_compatibility()
    
    # Overall result
    successful_models = [r for r in results if r['success']]
    if len(successful_models) == len(results):
        print(f"\n[SUCCESS] All tests passed! Both models are ready for training.")
    else:
        print(f"\n[WARNING] Some tests failed. Check the output above.")