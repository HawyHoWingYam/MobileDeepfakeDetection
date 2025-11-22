# ONNX Models Directory

## Required Files (Not in Git)

This directory should contain the following ONNX model files:

1. `aware_cascade_stage1.onnx` (37.45 MB)
2. `aware_cascade_stage2.onnx` (52.45 MB)
3. `cascade_config.json` (included)

## Setup Instructions

### Option 1: Generate Models

From the project root directory:

```bash
# Generate ONNX models
python scripts/export_mobile_cascade_onnx.py

# Copy to assets
cp android/mobile_bundle/*.onnx android/app/src/main/assets/models/
```

### Option 2: Download Pre-built Models

Download from the releases page or shared drive, then place them in this directory.

## Verification

After copying, verify the files exist:

```bash
ls -lh android/app/src/main/assets/models/
```

You should see:
- aware_cascade_stage1.onnx (~38 MB)
- aware_cascade_stage2.onnx (~53 MB)
- cascade_config.json (~2 KB)

## Important

**Do not commit .onnx files to Git** - they are too large and should be generated or downloaded separately.
