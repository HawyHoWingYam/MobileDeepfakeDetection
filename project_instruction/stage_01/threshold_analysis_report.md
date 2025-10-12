# Stage 01 Conservative Threshold Analysis Report

**Generated**: 2025-10-12 07:15:07

## Executive Summary

- **Optimal Threshold**: 0.500
- **False Negative Rate**: 0.26% (target: ≤1%)
- **Filter Rate**: 94.9%
- **F1-Score**: 0.9509
- **Accuracy**: 0.9491

## Performance Breakdown at Optimal Threshold

### Confusion Matrix
- **True Positives**: 8,896
- **True Negatives**: 8,223
- **False Positives**: 895
- **False Negatives**: 23

### Key Metrics
- **Precision**: 0.9086
- **Recall**: 0.9974
- **F1-Score**: 0.9509
- **Accuracy**: 0.9491
- **Filter Rate**: 94.9%

## Cascade System Implications

### First Layer Performance
- **Samples Fast-Tracked**: 17,119 (94.9%)
- **Samples for Stage 02**: 918 (5.1%)

### Conservative Strategy Benefits
1. **Ultra-Low Risk**: FNR ≤ 1% minimizes missed fakes
2. **High Efficiency**: >95% samples processed instantly
3. **Stage 02 Focus**: Only ambiguous cases sent to expert system

## Recommendations

1. **Deploy with threshold 0.500** for production cascade
2. **Monitor FNR continuously** in real deployment
3. **Stage 02 capacity planning** for ~5% of samples
4. **Consider adaptive threshold** based on confidence requirements

## Generated Files
- `threshold_analysis.csv` - Detailed threshold analysis
- `threshold_curves.png` - Performance curves visualization
