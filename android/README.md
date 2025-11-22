# Deepfake Detector – Android App

Android application for two‑stage cascade deepfake detection, built on top of the `MobileDeepfakeDetection` research project.

This README 面向“如何跑起来 / 怎么用”，更详细的整体规划与阶段目标请看：

- `android/instruction.md`

---

## Features (V0)

- **Two‑Stage Cascade Detection (ONNX)**
  - Stage 1: MobileNetV4 fast filter.
  - Stage 2: EfficientNetV2‑B3 precision analyzer.
  - Static cascade thresholds: `tau_low=0.02`, `tau_high=0.98`, `stage2_threshold=0.5`.
- **ONNX Runtime** for on‑device inference.
- **Jetpack Compose** modern UI + MVVM.
- **Single Image Detection**
  - Pick an image from gallery.
  - Run Stage1+Stage2 cascade.
  - Show label (Real/Fake), confidence, stage used, preprocessing/inference time.

> 当前版本是 **V0 单图检测 Demo**，后续扩展（批量测试、MediaPipe 人脸检测、视频等）见 `instruction.md`。

---

## Architecture Overview

### ML Pipeline

```text
Input Image (assumed 256×256 face crop)
    ↓
ImagePreprocessor (resize + ImageNet normalization, NCHW)
    ↓
Stage 1 (MobileNetV4 ONNX)
    ↓
Cascade Logic (tau_low=0.02, tau_high=0.98)
    ↓
Stage 2 (EfficientNetV2‑B3 ONNX, only if needed)
    ↓
Result (Real/Fake + Confidence + Timing)
```

### Project Structure (simplified)

```text
android/
  app/
    src/main/
      java/com/deepfake/detector/
        MainActivity.kt              # Entry point (Compose)
        ml/
          OnnxCascadeEngine.kt       # ONNX cascade engine
          ImagePreprocessor.kt       # Bitmap → NCHW float
          CascadeResult.kt           # Data models & config
        ui/
          DetectionScreen.kt         # Main detection UI
          DetectionViewModel.kt      # ViewModel (StateFlow)
          theme/                     # Material 3 theme
      assets/models/
        aware_cascade_stage1.onnx    # Stage 1 model
        aware_cascade_stage2.onnx    # Stage 2 model
        cascade_config.json          # Cascade parameters (doc only; code currently uses defaults)
      res/
        values/strings.xml
        values/themes.xml
        xml/backup_rules.xml
        xml/data_extraction_rules.xml
```

---

## Requirements

- **Android Studio**: Hedgehog (2023.1.1) or later.
- **Android SDK**:
  - Min SDK: 24 (Android 7.0)
  - Target/Compile SDK: 34
- **Gradle**: 8.2
- **Kotlin**: 1.9.20
- Device recommendation:
  - Android 7.0+ real device.
  - ≥ 2 GB RAM for stable ONNX inference.

---

## Setup & Run

### 1. Clone & Open

```bash
cd D:\work\MobileDeepfakeDetection\android
```

在 Android Studio 中选择该目录作为工程根目录打开。

### 2. Check SDK & Gradle

- 确认 `local.properties` 中的 `sdk.dir` 指向有效的 Android SDK。
- 首次打开时点击 “Sync Project with Gradle Files”，等待依赖下载完成：
  - ONNX Runtime Android 1.16.0
  - Jetpack Compose BOM 2023.10.01
  - Coil 2.5.0
  - Coroutines 1.7.3

### 3. Confirm Model Files

确保以下文件已经存在（通常由 `scripts/export_mobile_cascade_onnx.py` 生成并拷贝）：

```text
android/app/src/main/assets/models/
  aware_cascade_stage1.onnx
  aware_cascade_stage2.onnx
  cascade_config.json
```

如果缺失，可从 `android/mobile_bundle/` 复制，或重新运行 PC 端导出脚本。

### 4. Build & Run

1. 连接真机或启动模拟器（推荐真机）。
2. 在 Android Studio 中选择 `app` 配置，点击 Run（Shift+F10）。
3. 首次运行如需存储权限，请允许访问图片。

---

## Usage

1. 启动应用 “Deepfake Detector”。
2. 顶部状态显示 “Initializing models...” 时为模型加载阶段（约 2–3 秒）。
3. 状态变为 “Ready” 后：
   - 点击 “Select Image”，从相册选择一张人脸图片（最好是 256×256 预裁剪的 face crop）。
   - 点击 “Detect”。
4. 结果卡片将显示：
   - 预测：REAL 或 FAKE。
   - 置信度：0–100%。
   - Stage：由 Stage 1 或 Stage 2 决策。
   - Timing：预处理时间、推理时间、总时间。

---

## Performance (Expected, Mid‑range Device)

大致预期（Release 构建 + 真机）：

- Stage 1 inference：5–10 ms。
- Stage 2 inference：15–25 ms（仅在模糊样本时触发）。
- Preprocessing：2–5 ms。
- Total time：10–30 ms / image。
- Stage 2 usage rate：约 1–5%（大部分样本由 Stage 1 决策）。

实际数值会随设备性能和图像大小略有浮动。

---

## Troubleshooting

### Models Not Loading

- 检查 `app/src/main/assets/models/` 下 ONNX 文件是否存在，文件大小是否合理（~38 MB / ~53 MB）。
- Clean & Rebuild：

```bash
./gradlew clean
./gradlew assembleDebug
```

### Out of Memory (OOM) / Crash

- 尽量使用真机而非低配模拟器。
- 确保设备 RAM ≥ 2 GB。
- 关闭其他占用内存较大的应用。

### Slow Inference

- 使用 Release 构建（Build Variants 选择 `release`）。
- 在真机而非模拟器上测试。
- 确认设备为 ARM64 架构。

---

## Development Notes

- 更详细的开发计划与后续功能（MediaPipe、批量测试、PyTorch Mobile、视频等）请参考：
  - `android/instruction.md`
- 当前 `CascadeConfig` 在代码中使用的是默认参数，尚未动态读取 `cascade_config.json`；如需通过配置文件调整阈值，可以在后续迭代中添加 JSON 解析逻辑。

### Testing

单元测试与仪器测试命令（未来添加测试代码后使用）：

```bash
./gradlew test                # Unit tests
./gradlew connectedAndroidTest  # Instrumented tests (需要连接设备)
```

---

## References

- ONNX Runtime: https://onnxruntime.ai/
- Jetpack Compose: https://developer.android.com/jetpack/compose
- Kotlin Coroutines: https://kotlinlang.org/docs/coroutines-overview.html
- Research Paper: `../paper/main.pdf`

