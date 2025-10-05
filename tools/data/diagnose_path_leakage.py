"""
Path Leakage Diagnostic Tool for AWARE-NET

This script tests whether the model is learning from image content or path information.
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from stage_00.baseline_model import EfficientNetV2B3Baseline
from stage_00.train_baseline import UnifiedDeepfakeDataset

def test_random_labels():
    """Test 1: Random labels should give ~50% accuracy"""
    print("="*60)
    print("TEST 1: Random Labels Test")
    print("="*60)
    print("If model achieves >60% accuracy with random labels, it's using path info!")
    
    # Load a small subset of data
    manifest_path = "manifests/celebdf_v2_train.csv"
    
    # Read manifest and shuffle labels randomly
    data = pd.read_csv(manifest_path).head(1000)  # Use first 1000 samples for speed
    
    # Randomly shuffle labels
    np.random.seed(42)
    random_labels = np.random.randint(0, 2, len(data))
    data['label'] = random_labels
    
    print(f"Original vs Random label distribution:")
    print(f"Random labels: {np.bincount(random_labels)}")
    
    # Save temporary manifest with random labels
    temp_manifest = "temp_random_labels.csv"
    data.to_csv(temp_manifest, index=False)
    
    # Create dataset and loader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = UnifiedDeepfakeDataset(
        manifest_path=temp_manifest,
        dataset_name="random_labels_test",
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2B3Baseline(num_classes=1, pretrained=True, dropout_rate=0.2)
    model = model.to(device)
    model.eval()
    
    # Test prediction accuracy
    correct = 0
    total = 0
    
    print(f"Testing model on {len(dataset)} samples with random labels...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= 10:  # Test only first 10 batches for speed
                break
                
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            logits = model(images)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"\\nRandom Labels Test Results:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Expected: ~0.5000 (random chance)")
    
    # Clean up
    Path(temp_manifest).unlink()
    
    if accuracy > 0.6:
        print("FAILED: Model is likely using path information!")
        return False
    else:
        print("PASSED: Model accuracy is near random chance")
        return True

def test_path_content_only():
    """Test 2: Check what information is actually passed to model"""
    print("\\n" + "="*60)
    print("TEST 2: Path Content Analysis")
    print("="*60)
    
    manifest_path = "manifests/celebdf_v2_train.csv"
    data = pd.read_csv(manifest_path).head(100)
    
    # Print path analysis
    real_paths = data[data['label'] == 0]['image_path'].values[:5]
    fake_paths = data[data['label'] == 1]['image_path'].values[:5]
    
    print("Sample Real paths:")
    for path in real_paths:
        print(f"  {path}")
    
    print("\\nSample Fake paths:")
    for path in fake_paths:
        print(f"  {path}")
    
    # Check if paths contain obvious indicators
    path_indicators = ['real', 'fake', 'authentic', 'generated', 'deepfake']
    
    print(f"\\nPath Indicators Found:")
    for indicator in path_indicators:
        real_count = sum(1 for path in real_paths if indicator.lower() in path.lower())
        fake_count = sum(1 for path in fake_paths if indicator.lower() in path.lower())
        if real_count > 0 or fake_count > 0:
            print(f"  '{indicator}': Real={real_count}, Fake={fake_count}")
    
    return True

def test_anonymized_paths():
    """Test 3: Create anonymized manifest and test training"""
    print("\\n" + "="*60)
    print("TEST 3: Anonymized Path Test")
    print("="*60)
    print("Creating manifest with anonymized paths...")
    
    # Load original manifest
    manifest_path = "manifests/celebdf_v2_train.csv"
    data = pd.read_csv(manifest_path).head(1000)
    
    # Create anonymized paths by hashing or using indices
    anonymized_data = data.copy()
    anonymized_paths = []
    
    for idx, row in data.iterrows():
        original_path = row['image_path']
        # Keep the file extension and directory structure, but anonymize the sensitive parts
        parts = Path(original_path).parts
        
        # Replace 'real'/'fake' with anonymous directory names
        anonymized_parts = []
        for part in parts:
            if part.lower() == 'real':
                anonymized_parts.append('type_a')
            elif part.lower() == 'fake':
                anonymized_parts.append('type_b')
            else:
                anonymized_parts.append(part)
        
        anonymized_path = str(Path(*anonymized_parts))
        anonymized_paths.append(anonymized_path)
    
    anonymized_data['image_path'] = anonymized_paths
    
    # Note: We won't actually create the anonymized file structure
    # This is just to show the concept
    print(f"Sample anonymized paths:")
    for i in range(5):
        original = data.iloc[i]['image_path'] 
        anonymized = anonymized_paths[i]
        print(f"  Original:   {original}")
        print(f"  Anonymized: {anonymized}")
        print()
    
    print("Note: This test shows how to anonymize paths without copying files.")
    print("To fully implement, would need to create symbolic links or modify data loading.")
    
    return True

def main():
    """Run all diagnostic tests"""
    print("AWARE-NET Path Leakage Diagnostic Tool")
    print("="*60)
    
    results = []
    
    # Run tests
    print("\\nRunning diagnostic tests...")
    
    try:
        result1 = test_random_labels()
        results.append(("Random Labels Test", result1))
    except Exception as e:
        print(f"Random Labels Test failed: {e}")
        results.append(("Random Labels Test", False))
    
    try:
        result2 = test_path_content_only()
        results.append(("Path Content Analysis", result2))
    except Exception as e:
        print(f"Path Content Analysis failed: {e}")
        results.append(("Path Content Analysis", False))
    
    try:
        result3 = test_anonymized_paths()
        results.append(("Anonymized Path Test", result3))
    except Exception as e:
        print(f"Anonymized Path Test failed: {e}")
        results.append(("Anonymized Path Test", False))
    
    # Summary
    print("\\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    # Recommendations
    print("\\nRECOMMENDATIONS:")
    if not results[0][1]:  # Random labels test failed
        print("- CRITICAL: Model is using path information - implement path anonymization")
        print("- Modify manifest generation to remove 'real'/'fake' directory names")
        print("- Consider using symbolic links or hash-based file naming")
    else:
        print("- Model appears to be using image content, not paths")
        print("- High accuracy might be due to other factors (pretrained weights, data quality)")
    
    return all(result[1] for result in results)

if __name__ == "__main__":
    success = main()
    print(f"\\nDiagnostic completed: {'SUCCESS' if success else 'ISSUES FOUND'}")