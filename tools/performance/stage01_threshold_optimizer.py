#!/usr/bin/env python3
"""
Stage 01 Conservative Threshold Strategy

Implements conservative threshold optimization for cascade filtering.
Targets false negative rate < 1% while maintaining high filtering efficiency.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.stage_00.baseline_model import EfficientNetV2B3Baseline
from src.stage_00.train_baseline import UnifiedDeepfakeDataset
from torch.utils.data import DataLoader

def setup_device():
    """Setup device for model evaluation"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    model = EfficientNetV2B3Baseline(
        num_classes=1,
        pretrained=False,
        dropout_rate=0.2,
        model_name='tf_efficientnetv2_b0'
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✅ Model loaded successfully")
    return model

def get_predictions(model, dataloader, device):
    """Get model predictions and probabilities for dataset"""
    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Computing predictions"):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            logits = model(data)
            probabilities = torch.sigmoid(logits)

            # Store results
            all_predictions.extend(probabilities.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_probabilities), np.array(all_targets)

def evaluate_threshold(probabilities, targets, threshold):
    """Evaluate performance at specific threshold"""
    predictions = (probabilities >= threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # False negative rate (key metric for cascade)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

    # Filter rate (how many samples are quickly processed)
    filter_rate = (tp + tn) / len(predictions)

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_negative_rate': fnr,
        'filter_rate': filter_rate,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'total_samples': len(predictions)
    }

def find_optimal_threshold(probabilities, targets, target_fnr=0.01):
    """Find optimal threshold that meets false negative rate target"""
    print(f"\n🎯 Finding optimal threshold for FNR ≤ {target_fnr*100:.1f}%")

    # Test thresholds from 0.01 to 0.5 (conservative range)
    thresholds = np.arange(0.01, 0.51, 0.01)
    results = []

    best_threshold = None
    best_f1 = 0

    for threshold in thresholds:
        metrics = evaluate_threshold(probabilities, targets, threshold)
        results.append(metrics)

        # Check if this threshold meets FNR target
        if metrics['false_negative_rate'] <= target_fnr:
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = threshold

    print(f"✅ Optimal threshold: {best_threshold:.3f}")
    print(f"   FNR: {results[int(best_threshold*100)-1]['false_negative_rate']*100:.2f}%")
    print(f"   F1: {best_f1:.4f}")
    print(f"   Filter Rate: {results[int(best_threshold*100)-1]['filter_rate']*100:.1f}%")

    return best_threshold, results

def generate_threshold_report(results, optimal_threshold, output_dir):
    """Generate comprehensive threshold analysis report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Save detailed results
    df_results.to_csv(output_dir / "threshold_analysis.csv", index=False)

    # Find optimal threshold results
    optimal_idx = int(optimal_threshold * 100) - 1
    optimal_metrics = results[optimal_idx]

    # Generate report
    report = f"""# Stage 01 Conservative Threshold Analysis Report

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Optimal Threshold**: {optimal_threshold:.3f}
- **False Negative Rate**: {optimal_metrics['false_negative_rate']*100:.2f}% (target: ≤1%)
- **Filter Rate**: {optimal_metrics['filter_rate']*100:.1f}%
- **F1-Score**: {optimal_metrics['f1']:.4f}
- **Accuracy**: {optimal_metrics['accuracy']:.4f}

## Performance Breakdown at Optimal Threshold

### Confusion Matrix
- **True Positives**: {optimal_metrics['true_positives']:,}
- **True Negatives**: {optimal_metrics['true_negatives']:,}
- **False Positives**: {optimal_metrics['false_positives']:,}
- **False Negatives**: {optimal_metrics['false_negatives']:,}

### Key Metrics
- **Precision**: {optimal_metrics['precision']:.4f}
- **Recall**: {optimal_metrics['recall']:.4f}
- **F1-Score**: {optimal_metrics['f1']:.4f}
- **Accuracy**: {optimal_metrics['accuracy']:.4f}
- **Filter Rate**: {optimal_metrics['filter_rate']*100:.1f}%

## Cascade System Implications

### First Layer Performance
- **Samples Fast-Tracked**: {optimal_metrics['true_positives'] + optimal_metrics['true_negatives']:,} ({optimal_metrics['filter_rate']*100:.1f}%)
- **Samples for Stage 02**: {optimal_metrics['false_positives'] + optimal_metrics['false_negatives']:,} ({(1-optimal_metrics['filter_rate'])*100:.1f}%)

### Conservative Strategy Benefits
1. **Ultra-Low Risk**: FNR ≤ 1% minimizes missed fakes
2. **High Efficiency**: >{optimal_metrics['filter_rate']*100:.0f}% samples processed instantly
3. **Stage 02 Focus**: Only ambiguous cases sent to expert system

## Recommendations

1. **Deploy with threshold {optimal_threshold:.3f}** for production cascade
2. **Monitor FNR continuously** in real deployment
3. **Stage 02 capacity planning** for ~{(1-optimal_metrics['filter_rate'])*100:.0f}% of samples
4. **Consider adaptive threshold** based on confidence requirements

## Generated Files
- `threshold_analysis.csv` - Detailed threshold analysis
- `threshold_curves.png` - Performance curves visualization
"""

    # Save report
    with open(output_dir / "threshold_analysis_report.md", 'w') as f:
        f.write(report)

    # Generate visualization
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: FNR and Filter Rate vs Threshold
    ax1.plot(df_results['threshold'], df_results['false_negative_rate']*100, 'b-', linewidth=2, label='False Negative Rate')
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Target FNR (1%)')
    ax1.axvline(x=optimal_threshold, color='g', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold:.3f})')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('False Negative Rate (%)')
    ax1.set_title('False Negative Rate vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: F1 Score vs Threshold
    ax2.plot(df_results['threshold'], df_results['f1'], 'g-', linewidth=2, label='F1-Score')
    ax2.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold:.3f})')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Precision-Recall Curve
    ax3.plot(df_results['recall'], df_results['precision'], 'purple', linewidth=2)
    ax3.scatter([optimal_metrics['recall']], [optimal_metrics['precision']],
               color='red', s=100, zorder=5, label=f'Optimal ({optimal_threshold:.3f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: All Metrics Overview
    ax4.plot(df_results['threshold'], df_results['accuracy'], label='Accuracy', linewidth=2)
    ax4.plot(df_results['threshold'], df_results['precision'], label='Precision', linewidth=2)
    ax4.plot(df_results['threshold'], df_results['recall'], label='Recall', linewidth=2)
    ax4.plot(df_results['threshold'], df_results['f1'], label='F1-Score', linewidth=2)
    ax4.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_threshold:.3f})')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('All Metrics vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "threshold_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Report saved to: {output_dir}")
    print(f"   - threshold_analysis_report.md")
    print(f"   - threshold_analysis.csv")
    print(f"   - threshold_curves.png")

    return optimal_metrics

def main():
    parser = argparse.ArgumentParser(description='Stage 01 Conservative Threshold Analysis')
    parser.add_argument('--checkpoint', required=True, help='Path to best model checkpoint')
    parser.add_argument('--dataset', required=True, help='Dataset for evaluation (val/test)')
    parser.add_argument('--manifest', help='Override manifest path')
    parser.add_argument('--target-fnr', type=float, default=0.01, help='Target false negative rate (default: 0.01)')
    parser.add_argument('--output-dir', default='analysis/stage01_threshold', help='Output directory for analysis')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')

    args = parser.parse_args()

    print("🚀 Stage 01 Conservative Threshold Analysis")
    print("=" * 60)

    # Setup
    device = setup_device()
    model = load_model(args.checkpoint, device)

    # Load dataset (you'll need to specify the actual manifest path)
    if not args.manifest:
        print("❌ Please provide --manifest path to the evaluation dataset")
        return

    print(f"Loading dataset: {args.manifest}")
    dataset = UnifiedDeepfakeDataset(
        manifest_path=args.manifest,
        dataset_name=f'{args.dataset}_threshold_analysis',
        transform=None,
        use_augmentation=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == 'cuda'
    )

    print(f"Dataset loaded: {len(dataset):,} samples")

    # Get predictions
    probabilities, targets = get_predictions(model, dataloader, device)

    # Find optimal threshold
    optimal_threshold, results = find_optimal_threshold(probabilities, targets, args.target_fnr)

    # Generate report
    optimal_metrics = generate_threshold_report(results, optimal_threshold, args.output_dir)

    print("\n🎯 Stage 01 Threshold Analysis Complete!")
    print("=" * 60)
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"False Negative Rate: {optimal_metrics['false_negative_rate']*100:.2f}%")
    print(f"Filter Rate: {optimal_metrics['filter_rate']*100:.1f}%")
    print(f"F1-Score: {optimal_metrics['f1']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()