# Model Storage Directory

This directory is intended for storing trained model weights and checkpoints.

## Structure

```
models/
├── stage_00/
│   ├── baseline_efficientnet_b3.pth
│   └── checkpoints/
├── stage_01/
│   └── supcon_filter.pth
└── final/
    └── aware_net_final.pth
```

## Usage

- **Training**: Models are automatically saved here during training
- **Inference**: Load models from this directory for evaluation
- **Docker**: Contents copied to inference container

## Current Status

⚠️ **PLACEHOLDER**: This directory is currently empty. Models will be saved here after training.

## File Formats

- `.pth`: PyTorch model weights
- `.onnx`: ONNX format for deployment
- `.json`: Model metadata and configuration