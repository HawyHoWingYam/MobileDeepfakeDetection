"""
Quick test for EfficientNetV2-B0 model configuration
"""

import torch
import timm
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_model_variants():
    """Test different EfficientNetV2-B0 model names"""
    
    # Test possible model names for EfficientNetV2-B0
    possible_names = [
        'efficientnetv2_rw_b0',
        'efficientnetv2_b0', 
        'efficientnet_v2_b0',
        'efficientnetv2_s',
        'efficientnetv2_rw_s'
    ]
    
    print("Testing EfficientNetV2-B0 model variants...")
    
    working_models = []
    
    for model_name in possible_names:
        try:
            print(f"\\nTesting: {model_name}")
            model = timm.create_model(model_name, pretrained=False, num_classes=0)
            
            # Test forward pass
            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224)
                output = model(x)
                
            print(f"  SUCCESS: {model_name}")
            print(f"    Output shape: {output.shape}")
            print(f"    Num features: {model.num_features}")
            
            working_models.append((model_name, output.shape, model.num_features))
            
        except Exception as e:
            print(f"  FAILED: {model_name} - {str(e)[:50]}...")
    
    print(f"\\nWorking models ({len(working_models)}):")
    for name, shape, features in working_models:
        print(f"  {name}: {shape}, features={features}")
    
    return working_models

def test_baseline_model():
    """Test our baseline model implementation"""
    print("\\n" + "="*50)
    print("Testing AWARE-NET Baseline Model")
    print("="*50)
    
    try:
        from stage_00.baseline_model import EfficientNetV2B3Baseline
        
        # Test with pretrained
        print("\\nTesting with pretrained=True...")
        model_pretrained = EfficientNetV2B3Baseline(pretrained=True)
        
        # Test forward pass
        with torch.no_grad():
            x = torch.randn(2, 3, 256, 256)
            logits = model_pretrained(x)
            probs = model_pretrained.predict_proba(x)
            
        print(f"  Pretrained model works")
        print(f"    Input: {x.shape}")
        print(f"    Logits: {logits.shape}")
        print(f"    Probabilities: {probs.shape}")
        print(f"    Prob range: [{probs.min():.3f}, {probs.max():.3f}]")
        
        # Test without pretrained
        print("\\nTesting with pretrained=False...")
        model_no_pretrained = EfficientNetV2B3Baseline(pretrained=False)
        
        with torch.no_grad():
            logits2 = model_no_pretrained(x)
            
        print(f"  Non-pretrained model works")
        print(f"    Logits: {logits2.shape}")
        
        print(f"\\nAll tests passed!")
        return True
        
    except Exception as e:
        print(f"\\nBaseline model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("EfficientNetV2-B0 Configuration Test")
    print("="*50)
    
    # Test model variants
    working_models = test_model_variants()
    
    if working_models:
        # Test our baseline implementation
        success = test_baseline_model()
        
        if success:
            print("\\nReady for training with EfficientNetV2-B0!")
        else:
            print("\\nNeed to fix baseline model implementation")
    else:
        print("\\nNo working EfficientNetV2-B0 variants found")
        print("Consider using a different model or checking timm version")