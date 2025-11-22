# Mobile Bundle - ONNX Models

## Model Files (Not Included in Git)

Due to file size limitations, the ONNX model files are not included in the repository.

### Required Files
- `aware_cascade_stage1.onnx` (37.45 MB)
- `aware_cascade_stage2.onnx` (52.45 MB)
- `cascade_config.json` (included)

### How to Generate Models

Run the export script from the project root:

```bash
cd D:\work\MobileDeepfakeDetection
python scripts/export_mobile_cascade_onnx.py
```

This will generate the ONNX models in `android/mobile_bundle/`.

### Copy to Android Assets

After generating the models, copy them to the Android app:

```bash
# Windows
copy android\mobile_bundle\*.onnx android\app\src\main\assets\models\
copy android\mobile_bundle\cascade_config.json android\app\src\main\assets\models\

# Linux/Mac
cp android/mobile_bundle/*.onnx android/app/src/main/assets/models/
cp android/mobile_bundle/cascade_config.json android/app/src/main/assets/models/
```

### Alternative: Download Pre-exported Models

If available, download the pre-exported models from:
- [Release page](https://github.com/HawyHoWingYam/MobileDeepfakeDetection/releases)
- Or use Git LFS (if configured)

### Model Information

**Stage 1: MobileNetV4-Hybrid-Medium**
- Size: 37.45 MB
- Input: 256×256 RGB
- Purpose: Fast filter

**Stage 2: EfficientNetV2-B3**
- Size: 52.45 MB
- Input: 256×256 RGB
- Purpose: Precision analyzer

**Configuration**
- tau_low: 0.02
- tau_high: 0.98
- stage2_threshold: 0.5
