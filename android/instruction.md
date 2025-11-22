Android Deepfake Detection Testing App – 实施计划（最终版）
======================================================

本文件是 `MobileDeepfakeDetection/android` 目录下的统一实施说明，用于指导后续在 Android 上实现和测试两阶段级联深度伪造检测系统。

下文在参考已有长计划的基础上，结合当前代码仓库实际情况（Stage1/Stage2 模型、Stage4 导出工具等），做了整理和调整。

---

## 1. 目标与范围

- 构建一个 **Android 测试应用**，用于验证和演示仓库中的两阶段级联深度伪造检测系统：
  - Stage 1：MobileNetV4 快速过滤器。
  - Stage 2：EfficientNetV2-B3 精细分析器。
- 支持 **单张人脸图片检测** 为第一优先目标（V0），在此基础上逐步扩展：
  - 批量图片测试。
  - 性能指标展示（推理耗时、Stage2 使用率等）。
  - 结果导出（CSV/JSON）。
- 推理引擎：
  - **首选 ONNX Runtime**（与 `src/stage4/mobile_deployment/onnx_exporter.py` 对齐）。
  - PyTorch Mobile / TorchScript 作为后续扩展，用于对比不同移动推理方案。
- 人脸检测：
  - V0 版本假定输入为预裁剪人脸（与训练数据同规格）。
  - 后续阶段集成 **MediaPipe Face Detection**，自动从任意图片中裁出 256×256 人脸。

---

## 2. 技术栈与工程约定

- **语言**：Kotlin。
- **UI 框架**：Jetpack Compose（Material Design 3）。
- **架构**：简单 MVVM，按模块划分包结构即可，无需过度复杂的 Clean Architecture。
- **推理引擎**：
  - `onnxruntime-android`（V0 必选）。
  - `org.pytorch:pytorch_android`（V2 可选）。
- **人脸检测**：`com.google.mediapipe:tasks-vision`（后续阶段引入）。
- **异步**：Kotlin Coroutines + Flow。
- **依赖注入**：可选 Hilt；若初版只做单 Activity，也可以先用手动依赖注入，之后再引入 Hilt。

建议 Gradle 主要依赖（版本可按当时最新稳定版本微调）：

```gradle
dependencies {
    // Compose
    implementation "androidx.compose.ui:ui:<latest>"
    implementation "androidx.compose.material3:material3:<latest>"
    implementation "androidx.lifecycle:lifecycle-viewmodel-compose:<latest>"

    // ONNX Runtime
    implementation "com.microsoft.onnxruntime:onnxruntime-android:1.16.0"

    // PyTorch Mobile（后续扩展）
    implementation "org.pytorch:pytorch_android:1.13.1"
    implementation "org.pytorch:pytorch_android_torchvision:1.13.1"

    // MediaPipe Face Detection（后续扩展）
    implementation "com.google.mediapipe:tasks-vision:0.10.8"

    // 图片加载
    implementation "io.coil-kt:coil-compose:2.5.0"

    // 协程
    implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3"

    // CSV / 导出工具（可选）
    implementation "com.opencsv:opencsv:5.8"
}
```

---

## 3. 模型准备（PC 端）

所有模型导出操作在 `D:\work\MobileDeepfakeDetection` 根目录下完成。

### 3.1 选定模型来源

- **Stage 1（MobileNetV4 快速过滤器）**  
  - 推荐：`outputs/stage1/run_20251023_034316/best_model.pth`。
- **Stage 2（EfficientNetV2-B3 精细分析器）**  
  - 推荐：`outputs/stage5/finetune_s2_b3_r2/run_20251109_161118/best_model.pth`  
    （这是 Stage2 的 finetune 结果，而不是 LightGBM 的 Stage3）。

> 说明：原先长计划中提到的 `outputs/stage3/.../best_model.pth` 在本仓库中属于 Stage3 元模型，与本文的移动端部署主线不完全匹配，这里统一改为使用 Stage5 中的 EfficientNetV2-B3 finetune 模型。

### 3.2 TorchScript 导出（预留给 PyTorch Mobile）

首版 Android App 可以 **不依赖 PyTorch Mobile**，只使用 ONNX Runtime。为了未来对比测试，这里规划好 TorchScript 导出步骤，但可以在后续阶段再实现。

建议新建脚本（示例名）：`scripts/export_mobile_torchscript.py`，主要工作：

1. 使用 `timm.create_model(...)` 重建 Stage1、Stage2 模型结构。
2. 加载上面选定的 `best_model.pth` 权重。
3. 如有需要，应用 `torch.quantization.quantize_dynamic` 对线性层做 INT8 动态量化。
4. 使用 `torch.jit.trace` 或 `torch.jit.script` 导出：
   - `android/models/stage1_mobilenetv4.ts`
   - `android/models/stage2_efficientnetv2.ts`
5. 用少量测试图片在 PC 上验证 TorchScript 输出与原模型一致（误差在可接受范围内）。

### 3.3 ONNX 导出（首选路径）

ONNX 导出直接利用仓库已有的工具：`src/stage4/mobile_deployment/onnx_exporter.py`。

建议新建脚本：`scripts/export_mobile_cascade_onnx.py`，大致流程：

1. 构造并加载两个 PyTorch 模型：
   - Stage1：`mobilenetv4_hybrid_medium.ix_e550_r256_in1k`，`num_classes=1`。
   - Stage2：`tf_efficientnetv2_b3.in21k_ft_in1k`，`num_classes=1`。
2. 加载对应 `best_model.pth` 的 `model_state_dict`。
3. 使用 `ONNXExporter`：

   ```python
   from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter
   from pathlib import Path

   output_dir = Path("android/mobile_bundle")
   exporter = ONNXExporter()
   bundle = exporter.export_cascade_bundle(
       models={
           "stage1": stage1_model,
           "stage2": stage2_model,
       },
       output_dir=output_dir,
       bundle_name="aware_cascade",
   )
   print(bundle)
   ```

4. 生成输出（预期在 `android/mobile_bundle/`）：
   - `aware_cascade_stage1.onnx`
   - `aware_cascade_stage2.onnx`
   - `aware_cascade_manifest.json`（由导出器自动生成）。

### 3.4 级联参数与配置文件

为了让 Android 端完全自包含，建议生成一个简单的配置文件，例如：`android/mobile_bundle/cascade_config.json`：

```json
{
  "input_size": [1, 3, 256, 256],
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "tau_low": 0.02,
  "tau_high": 0.98,
  "stage2_threshold": 0.5
}
```

其中：

- `tau_low` / `tau_high`：Stage1 fake 概率 `p1` 的两阈值：
  - `p1 < tau_low` → 直接判 real。
  - `p1 > tau_high` → 直接判 fake。
  - 中间区域 → 升级到 Stage2。
- `stage2_threshold`：Stage2 fake 概率 `p2` 的判决阈值（通常 0.5）。

具体数值可以在 PC 上使用 `benchmark_cascade.py` 做网格搜索后更新，这里先给出一个与 Stage4 文档相符的保守默认值。

---

## 4. Android 工程结构建议

在 `D:\work\MobileDeepfakeDetection\android` 目录下创建 Android Studio 工程，推荐结构：

```text
android/
  instruction.md                 # 本文件
  mobile_bundle/                 # PC 端导出的 ONNX 模型与配置
    aware_cascade_stage1.onnx
    aware_cascade_stage2.onnx
    aware_cascade_manifest.json
    cascade_config.json

  app/
    src/main/
      java/com/yourname/deepfakedetector/
        ml/
          inference/            # 推理引擎封装（ONNX / PyTorch）
          preprocessing/        # Bitmap → Tensor 预处理
          facedetection/        # MediaPipe 人脸检测（后续）
        presentation/
          main/                 # 主界面（导航）
          detection/            # 单图检测界面（V0）
          batch/                # 批量测试界面（后续）
          metrics/              # 性能指标展示（后续）
          components/           # 通用 Compose 组件
        data/                   # 如需持久化/导出，可在此放 Repository
      assets/models/
        aware_cascade_stage1.onnx
        aware_cascade_stage2.onnx
        aware_cascade_manifest.json
        cascade_config.json
```

> 说明：包结构可以按实际开发习惯调整，不强制使用完整的 Clean Architecture。优先保证 **ml 模块与 UI 解耦**，方便后续替换推理引擎。

---

## 5. 分阶段实施计划

### Phase 0 – 准备与验证（PC 端）

- [ ] 确认 Stage1/Stage2 模型训练完好，能在 Python 端跑通单图推理。
- [ ] 完成 ONNX 导出脚本 `export_mobile_cascade_onnx.py` 并生成 `android/mobile_bundle/*`。
- [ ] 在 Python 中使用 ONNX Runtime 对若干张测试图片跑级联推理，确认结果与 PyTorch 版本一致。

### Phase 1 – V0：单图 ONNX 级联 Demo（优先实现）

目标：**不做人脸检测、不做批量测试，先把“单张 256×256 人脸图片 → 级联判决”跑通。**

实现要点：

1. **工程初始化**
   - 使用 Android Studio 创建 “Empty Compose Activity” 工程。
   - 按第 2 节添加 ONNX Runtime 和 Compose 依赖。
   - 将 `android/mobile_bundle/*` 拷贝到 `app/src/main/assets/models/`。

2. **预处理模块（`ImagePreprocessor.kt`）**
   - 函数：`fun preprocess(bitmap: Bitmap): FloatArray`。
   - 步骤：
     - resize 到 256×256。
     - 转换为 `FloatArray`，范围 [0,1]。
     - 按 `cascade_config.json` 中的 mean/std 做归一化。
     - 组织成 NCHW（1×3×256×256）格式，供 ONNX Runtime 使用。

3. **ONNX 推理封装（`OnnxCascadeEngine.kt`）**
   - 加载两个 `OrtSession`：Stage1 & Stage2。
   - 级联逻辑：
     - Stage1 输出 logits → 手动 sigmoid 得到 fake 概率 `p1`。
     - 对比 `tau_low` 和 `tau_high`：
       - `p1 < tau_low` → 直接返回 real，stage=`stage1`。
       - `p1 > tau_high` → 直接返回 fake，stage=`stage1`。
       - 否则：送入 Stage2，得到 `p2`；`p2 > 0.5` → fake，否则 real。
   - 统计单次推理的耗时（预处理时间 + 推理时间）。

4. **统一接口（`InferenceEngine.kt`）**

   ```kotlin
   data class CascadeResult(
       val label: String,         // "real" or "fake"
       val confidence: Float,     // 最终阶段的概率（fake 或 real）
       val stage: String,         // "stage1" 或 "stage2"
       val preprocessMs: Long,
       val inferenceMs: Long
   )

   interface InferenceEngine {
       suspend fun predict(bitmap: Bitmap): CascadeResult
       fun release()
   }
   ```

5. **UI：单图检测界面（`DetectionScreen.kt`）**
   - 功能：
     - 从相册选择图片（可先不支持相机）。
     - 显示选中的图片缩略图。
     - 点击 “检测” 按钮后调用 `OnnxCascadeEngine.predict()`。
     - 使用卡片显示：
       - Real / Fake（颜色区分）。
       - 置信度（百分比）。
       - 使用的 Stage（Stage1 或 Stage2）。
       - 耗时信息。

> 完成本阶段后，Android 端就能对单张预裁剪人脸图片做两级联深度伪造检测，是整个项目最关键的里程碑。

### Phase 2 – 扩展：人脸检测 + 批量测试

在 V0 基础上扩展功能。

1. **MediaPipe 人脸检测（`FaceDetector.kt`）**
   - 集成 `tasks-vision` 中的 Face Detection。
   - 对任意输入图片：
     - 检测所有人脸边界框。
     - 根据面积/置信度选择一个主脸。
     - 从原始 Bitmap 中裁出人脸区域，resize 到 256×256，传入预处理模块。
   - 提供简单可视化（在 UI 上绘制边框）。

2. **批量测试界面（`BatchTestScreen.kt`）**
   - 支持从相册多选图片（例如最多 50 张）。
   - 对每张图片依次运行：人脸检测 → 级联推理。
   - 显示：处理进度、每张图片的结果列表（缩略图 + label + stage + 耗时）。
   - 汇总统计：
     - 总数量 / Real / Fake 数量。
     - 平均推理时间。
     - Stage2 触发比例。

3. **结果导出**
   - 使用 `opencsv` 或简单字符串拼接，将批量测试结果导出为 CSV：
     - `image_name,is_deepfake,confidence,stage,inference_time_ms,engine`。
   - 文件保存到 `Downloads` 目录，便于从手机拷回到 PC 分析。

### Phase 3 – 引入 PyTorch Mobile（可选，但建议）

在 ONNX 路线稳定后，可以增加一个 PyTorch Mobile 推理引擎，以便：

- 对比 TorchScript 与 ONNX 在不同设备上的性能与包体大小。
- 为未来可能的 TorchScript-only 部署做准备。

主要工作：

1. 将前文导出的 `stage1_mobilenetv4.ts` / `stage2_efficientnetv2.ts` 放入 `assets/models/`。
2. 添加 `PyTorchInferenceEngine.kt`，实现与 `OnnxCascadeEngine` 相同的接口和级联逻辑。
3. 在 UI 中加入 “推理引擎选择器”（ONNX / PyTorch / 双引擎对比）。
4. 如有需要，在性能对比页面展示两者的平均耗时、内存占用等。

### Phase 4 – 性能视图与高级功能

当核心功能（单图 + 批量 + MediaPipe + 双引擎）稳定后，可进一步增加：

- 性能对比界面：展示 ONNX vs PyTorch 的平均推理时间、内存占用、Stage2 使用率等。
- 设置界面：
  - 阈值调节（`tau_low` / `tau_high` / `stage2_threshold`）。
  - 选择默认引擎。
  - 切换 MediaPipe 开关和 NNAPI 加速开关（ONNX）。
- 错误处理与提示：无人脸、模型加载失败、内存不足等情况的用户友好提示。

   
这部分不影响当前研究作业的核心交付，可以作为后续实验性功能：

- 使用 `MediaMetadataRetriever` 或 `MediaCodec` 对视频抽帧。
- 对每帧或每 N 帧运行人脸检测 + 级联推理。
- 将帧级结果聚合为视频级判决（多数投票或时间段分析）。

---

## 6. 当前状态与下一步建议

截至目前（依据仓库内容）：

- Stage1/Stage2 已经在 PC 端训练完毕，有多个 `best_model.pth` 版本可选。
- Stage4 中已经实现了部分移动端导出工具（ONNXExporter、MobileOptimizer 等）。

**推荐的下一步具体动作**：

1. 在 PC 端完成 `scripts/export_mobile_cascade_onnx.py`，生成 `android/mobile_bundle/*`。
2. 在 `android/` 下创建 Android Studio 工程，并完成 Phase 1 的 V0 单图 ONNX Demo。
3. 用少量 `processed_data` 中的 256×256 PNG 测试图片，对比 Android 结果和 Python 结果（确保推理一致）。
4. 确认 V0 稳定后，再按本说明推进 MediaPipe、人脸裁剪、批量测试和 PyTorch Mobile 扩展。

