# Deepfake Detector - Android App

Android application for real-time deepfake detection using a two-stage cascade system.

## Features

- **Two-Stage Cascade Detection**
  - Stage 1: MobileNetV4 fast filter (37.45 MB)
  - Stage 2: EfficientNetV2-B3 precision analyzer (52.45 MB)
- **ONNX Runtime** for efficient mobile inference
- **Jetpack Compose** modern UI
- **Performance Metrics** display (inference time, stage used)
- **Single Image Detection** (V0)

## Architecture

### ML Pipeline
```
Input Image (256x256)
    ↓
ImagePreprocessor (ImageNet normalization)
    ↓
Stage 1 (MobileNetV4)
    ↓
Cascade Logic (tau_low=0.02, tau_high=0.98)
    ↓
Stage 2 (EfficientNetV2-B3) [if needed]
    ↓
Result (Real/Fake + Confidence)
```

### Project Structure
```
app/src/main/
├── java/com/deepfake/detector/
│   ├── MainActivity.kt                 # Main entry point
│   ├── ml/
│   │   ├── OnnxCascadeEngine.kt       # ONNX inference engine
│   │   ├── ImagePreprocessor.kt       # Image preprocessing
│   │   └── CascadeResult.kt           # Data models
│   └── ui/
│       ├── DetectionScreen.kt         # Main UI (Compose)
│       ├── DetectionViewModel.kt      # ViewModel
│       └── theme/                     # Material 3 theme
└── assets/models/
    ├── aware_cascade_stage1.onnx      # Stage 1 model
    ├── aware_cascade_stage2.onnx      # Stage 2 model
    └── cascade_config.json            # Configuration
```

## Requirements

- **Android Studio**: Hedgehog (2023.1.1) or later
- **Minimum SDK**: API 24 (Android 7.0)
- **Target SDK**: API 34 (Android 14)
- **Gradle**: 8.2
- **Kotlin**: 1.9.20

## Setup

### 1. Open in Android Studio
```bash
cd D:\work\MobileDeepfakeDetection\android
# Open this directory in Android Studio
```

### 2. Sync Gradle
Android Studio will automatically download dependencies:
- ONNX Runtime Android (1.16.0)
- Jetpack Compose (BOM 2023.10.01)
- Coil for image loading
- Kotlin Coroutines

### 3. Build and Run
1. Connect an Android device or start an emulator
2. Click "Run" (Shift+F10)
3. Grant storage permissions when prompted

## Usage

1. **Launch App**: Open "Deepfake Detector"
2. **Wait for Initialization**: Models load on startup (~2-3 seconds)
3. **Select Image**: Tap "Select Image" to choose from gallery
4. **Detect**: Tap "Detect" to run inference
5. **View Results**:
   - Prediction: Real or Fake
   - Confidence: Percentage
   - Stage: Which model made the decision
   - Timing: Preprocessing and inference time

## Performance

### Expected Performance (Mid-range Device)
- **Stage 1 Inference**: 5-10 ms
- **Stage 2 Inference**: 15-25 ms (only for ambiguous cases)
- **Preprocessing**: 2-5 ms
- **Total Time**: 10-30 ms per image
- **Stage 2 Rate**: 1-5% (most images decided by Stage 1)

### Model Sizes
- **Stage 1**: 37.45 MB (ONNX)
- **Stage 2**: 52.45 MB (ONNX)
- **Total APK**: ~100-120 MB (with dependencies)

## Configuration

Edit `assets/models/cascade_config.json` to adjust thresholds:

```json
{
  "tau_low": 0.02,      // Stage 1 real threshold
  "tau_high": 0.98,     // Stage 1 fake threshold
  "stage2_threshold": 0.5  // Stage 2 decision threshold
}
```

## Troubleshooting

### Models Not Loading
- Ensure ONNX files are in `app/src/main/assets/models/`
- Check file sizes: Stage1 ~37MB, Stage2 ~52MB
- Verify `cascade_config.json` exists

### Out of Memory
- Test on device with 2GB+ RAM
- Close other apps before running
- Consider using Stage 1 only for low-end devices

### Slow Inference
- Enable NNAPI acceleration (experimental)
- Use release build instead of debug
- Test on device with ARM64 processor

## Development

### Adding New Features

**Batch Processing**:
```kotlin
// In DetectionViewModel.kt
suspend fun detectBatch(uris: List<Uri>): List<CascadeResult> {
    return uris.map { uri ->
        val bitmap = loadBitmap(uri)
        engine.detect(bitmap)
    }
}
```

**Video Support**:
```kotlin
// Extract frames using MediaMetadataRetriever
// Process each frame with engine.detect()
// Aggregate results
```

### Testing

Run unit tests:
```bash
./gradlew test
```

Run instrumented tests:
```bash
./gradlew connectedAndroidTest
```

## License

This project is part of the MobileDeepfakeDetection research.

## References

- ONNX Runtime: https://onnxruntime.ai/
- Jetpack Compose: https://developer.android.com/jetpack/compose
- Research Paper: See `../paper/main.pdf`
