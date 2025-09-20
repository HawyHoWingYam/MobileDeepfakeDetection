#!/usr/bin/env python3
"""
Clean Data Validation for AWARE-NET
Test the anonymized dataset loading and basic training functionality
"""

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stage_00.dataset import CelebDFDataset
from stage_00.baseline_model import EfficientNetV2B3Baseline
import json

def test_anonymized_manifest_loading():
    """Test loading anonymized manifests"""
    print("Testing anonymized manifest loading...")
    
    manifest_files = {
        'train': 'D:/work/AWARE-NET/datasets/manifests_anonymized/celebdf_train_anonymized.csv',
        'val': 'D:/work/AWARE-NET/datasets/manifests_anonymized/celebdf_val_anonymized.csv',
        'test': 'D:/work/AWARE-NET/datasets/manifests_anonymized/celebdf_test_anonymized.csv'
    }
    
    results = {}
    
    for split, manifest_path in manifest_files.items():
        if Path(manifest_path).exists():
            df = pd.read_csv(manifest_path)
            
            # Analyze the manifest
            total_samples = len(df)
            real_count = (df['label'] == 0).sum()
            fake_count = (df['label'] == 1).sum()
            
            # Check for path anonymization
            sample_paths = df['image_path'].head(10).tolist()
            has_real_fake_in_paths = any('real' in p or 'fake' in p for p in sample_paths)
            
            results[split] = {
                'total_samples': total_samples,
                'real_samples': real_count,
                'fake_samples': fake_count,
                'balance_ratio': fake_count / real_count if real_count > 0 else 0,
                'has_label_leakage': has_real_fake_in_paths,
                'sample_paths': sample_paths[:3]
            }
            
            print(f"{split.upper()}:")
            print(f"  Samples: {total_samples}")
            print(f"  Real: {real_count} ({100*real_count/total_samples:.1f}%)")
            print(f"  Fake: {fake_count} ({100*fake_count/total_samples:.1f}%)")
            print(f"  Balance: {fake_count/real_count:.2f}:1" if real_count > 0 else "  Balance: N/A")
            print(f"  Path leakage: {'YES' if has_real_fake_in_paths else 'NO'}")
            print(f"  Sample paths: {sample_paths[:2]}")
            
        else:
            print(f"ERROR: {manifest_path} not found")
            results[split] = {'error': 'file_not_found'}
    
    return results

def test_dataset_creation():
    """Test creating dataset objects with anonymized manifests"""
    print("\nTesting dataset creation...")
    
    manifest_path = 'D:/work/AWARE-NET/datasets/manifests_anonymized/celebdf_train_anonymized.csv'
    
    if not Path(manifest_path).exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return None
    
    try:
        # Create dataset (without actually loading images)
        dataset = CelebDFDataset(
            manifest_path=manifest_path,
            root_path="D:/work/AWARE-NET/datasets",
            image_size=256,
            augmentation=False,
            normalize=True
        )
        
        print(f"Dataset created successfully:")
        print(f"  Total samples: {len(dataset)}")
        
        # Get dataset info
        info = dataset.get_dataset_info()
        print(f"  Real samples: {info['real_samples']}")
        print(f"  Fake samples: {info['fake_samples']}")
        print(f"  Balance ratio: {info['balance_ratio']:.3f}")
        
        # Test class counts for BCE
        class_counts = dataset.get_class_counts()
        print(f"  Class counts: Real={class_counts[0]:.0f}, Fake={class_counts[1]:.0f}")
        
        # Calculate correct pos_weight
        pos_weight = class_counts[0] / class_counts[1]
        print(f"  Correct pos_weight for BCE: {pos_weight:.4f}")
        
        return dataset
        
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        return None

def test_model_creation():
    """Test creating the corrected BCE model"""
    print("\nTesting model creation...")
    
    try:
        # Create model with corrected architecture
        model = EfficientNetV2B3Baseline(
            num_classes=1,  # True BCE
            pretrained=True,
            dropout_rate=0.2
        )
        
        model_info = model.get_model_info()
        print(f"Model created successfully:")
        print(f"  Architecture: {model_info['model_name']}")
        print(f"  Output classes: {model_info['num_classes']}")
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  Forward pass successful: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Test sigmoid probabilities
        probs = torch.sigmoid(output)
        print(f"  Probabilities: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
        
        return model
        
    except Exception as e:
        print(f"ERROR creating model: {e}")
        return None

def test_simple_baseline_classifiers():
    """Test simple baseline classifiers for comparison"""
    print("\nTesting baseline classifiers...")
    
    # Load a small sample for testing
    manifest_path = 'D:/work/AWARE-NET/datasets/manifests_anonymized/celebdf_val_anonymized.csv'
    
    if not Path(manifest_path).exists():
        print(f"ERROR: Validation manifest not found")
        return
    
    df = pd.read_csv(manifest_path)
    labels = df['label'].values
    
    # Random classifier
    np.random.seed(42)
    random_predictions = np.random.randint(0, 2, len(labels))
    random_accuracy = (random_predictions == labels).mean()
    
    # Majority classifier (always predict most common class)
    majority_class = 1 if (labels == 1).sum() > (labels == 0).sum() else 0
    majority_predictions = np.full_like(labels, majority_class)
    majority_accuracy = (majority_predictions == labels).mean()
    
    print(f"Baseline Classifiers Performance:")
    print(f"  Random Classifier: {random_accuracy:.3f} accuracy")
    print(f"  Majority Classifier: {majority_accuracy:.3f} accuracy")
    print(f"  Expected Clean EfficientNet: 0.75-0.85 accuracy")
    
    return {
        'random_accuracy': random_accuracy,
        'majority_accuracy': majority_accuracy
    }

def main():
    """Main validation function"""
    
    print("AWARE-NET Clean Data Validation")
    print("=" * 50)
    
    # Test 1: Manifest loading
    manifest_results = test_anonymized_manifest_loading()
    
    # Test 2: Dataset creation
    dataset = test_dataset_creation()
    
    # Test 3: Model creation
    model = test_model_creation()
    
    # Test 4: Baseline classifiers
    baseline_results = test_simple_baseline_classifiers()
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    all_good = True
    
    # Check manifest results
    for split, result in manifest_results.items():
        if 'error' in result:
            print(f"ERROR: {split} manifest failed to load")
            all_good = False
        elif result.get('has_label_leakage', True):
            print(f"ERROR: {split} still has path label leakage")
            all_good = False
        elif abs(result.get('balance_ratio', 0) - 1.0) > 0.1:
            print(f"WARNING: {split} is not well balanced ({result['balance_ratio']:.2f}:1)")
    
    # Check dataset and model
    if dataset is None:
        print("ERROR: Dataset creation failed")
        all_good = False
    
    if model is None:
        print("ERROR: Model creation failed")
        all_good = False
    
    if all_good:
        print("SUCCESS: All validation tests passed!")
        print("The clean anonymized dataset is ready for realistic training.")
        print("\nExpected training results:")
        print("  - Gradual learning over 5-15 epochs")
        print("  - Training accuracy >= Validation accuracy")
        print("  - AUC: 0.75-0.85 (not 0.99+)")
        print("  - F1: 0.70-0.82")
    else:
        print("FAILED: Some validation tests failed!")
        print("Please fix the issues before proceeding with training.")
    
    # Save validation results
    validation_summary = {
        'manifest_results': manifest_results,
        'dataset_creation': dataset is not None,
        'model_creation': model is not None,
        'baseline_results': baseline_results,
        'overall_status': 'PASS' if all_good else 'FAIL',
        'ready_for_training': all_good
    }
    
    with open('D:/work/AWARE-NET/clean_data_validation_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(validation_summary, f, indent=2, default=convert_types)
    
    print(f"\nValidation results saved to: D:/work/AWARE-NET/clean_data_validation_results.json")

if __name__ == "__main__":
    main()