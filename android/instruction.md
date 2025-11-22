Android Deepfake Detection Testing App – 实施计划
==============================================

本说明文档用于指导 `MobileDeepfakeDetection/android` 子项目的开发与使用，统一记录：

- 当前 Android App 的实现状态（V0）。
- PC 端模型导出 → Android 集成的关键步骤。
- 后续阶段的开发路线与任务清单。

文档只描述本仓库 **已经存在的代码与脚本**，不再重复无关的泛泛建议。

---

## 1. 当前状态概览

子项目路径：`D:\work\MobileDeepfakeDetection\android`

已完成内容（V0）：

- 使用 **ONNX Runtime** 集成两阶段级联深度伪造检测：
  - Stage 1：MobileNetV4 快速过滤器。
  - Stage 2：EfficientNetV2-B3 精细分析器。
- 实现了 **单张图片检测**：
  - 从相册选择图片。
  - 对图片做 256×256 + ImageNet 归一化预处理。
  - Stage1 → 级联逻辑（`tau_low=0.02`, `tau_high=0.98`）→ 必要时 Stage2。
  - 显示标签（REAL/FAKE）、置信度、使用的 Stage、预处理/推理耗时。
- 工程结构、Gradle 配置、说明文档（`README.md`、`PROJECT_SUMMARY.md`、`SETUP_GUIDE.md`）均已齐全，可在 Android Studio 中直接打开运行。

尚未实现（规划中的后续阶段）：

- MediaPipe 人脸检测与裁剪。
- 批量图片测试与统计。
- 结果导出（CSV/JSON）。
- PyTorch Mobile / TorchScript 引擎对比。
- 视频抽帧与视频级聚合。

---

## 2. 模型与脚本（PC 端）

PC 端根目录：`D:\work\MobileDeepfakeDetection`

### 2.1 模型来源

- Stage 1（MobileNetV4）：
  - `outputs/stage1/run_20251023_034316/best_model.pth`
- Stage 2（EfficientNetV2-B3）：
  - `outputs/stage5/finetune_s2_b3_r2/run_20251109_161118/best_model.pth`

说明：`outputs/stage3/...` 是 LightGBM 元模型，不参与当前 Android 主线部署。

### 2.2 ONNX 导出

相关脚本位于 `scripts/`：

- `scripts/export_mobile_cascade_onnx.py`
  - 使用 `src/stage4/mobile_deployment/onnx_exporter.py` 将 Stage1/Stage2 导出为 ONNX。
  - 输出目录：`android/mobile_bundle/`：
    - `aware_cascade_stage1.onnx`
    - `aware_cascade_stage2.onnx`
    - `aware_cascade_manifest.json`
    - `cascade_config.json`

- `scripts/validate_onnx_models.py`
  - 使用 ONNX Runtime 对比 PyTorch 与 ONNX 输出，验证导出正确性。

Android 工程中实际使用的模型文件已从 `android/mobile_bundle/` 拷贝到：

- `android/app/src/main/assets/models/`

### 2.3 配置文件（级联参数）

`android/app/src/main/assets/models/cascade_config.json` 主要包含：

- `input_size`: `[1, 3, 256, 256]`
- `mean` / `std`: ImageNet 归一化参数。
- `tau_low`: 0.02
- `tau_high`: 0.98
- `stage2_threshold`: 0.5
- 模型元信息与预期性能说明。

当前代码中的 `CascadeConfig.default()` 与该 JSON 内容保持一致，但尚未动态解析 JSON（见第 6 节改进建议）。

---

## 3. Android 工程结构

关键结构如下（只列出与深伪检测相关部分）：

```text
android/
  app/
    src/main/
      java/com/deepfake/detector/
        MainActivity.kt            # 入口 Activity，承载 Compose UI
        ml/
          OnnxCascadeEngine.kt     # 两阶段 ONNX 级联引擎
          ImagePreprocessor.kt     # Bitmap → NCHW float 预处理
          CascadeResult.kt         # CascadeResult / DetectionStage / CascadeConfig
        ui/
          DetectionScreen.kt       # 单图检测界面（Compose）
          DetectionViewModel.kt    # ViewModel + StateFlow 状态
          theme/
            Color.kt
            Theme.kt
            Type.kt
      assets/models/
        aware_cascade_stage1.onnx
        aware_cascade_stage2.onnx
        cascade_config.json
        README.md
      res/
        values/strings.xml
        values/themes.xml
        xml/backup_rules.xml
        xml/data_extraction_rules.xml
    build.gradle.kts               # app 模块配置（Compose + ONNX Runtime）
  build.gradle.kts                 # 顶层 Gradle 配置
  settings.gradle.kts              # modules 配置
  README.md                        # Android App 使用说明
  PROJECT_SUMMARY.md               # Android 项目总结
  SETUP_GUIDE.md                   # 环境/构建/运行指南
  instruction.md                   # 当前文档（实施计划）
```

---

## 4. 已实现功能（V0 详细说明）

### 4.1 ML 管线

**ImagePreprocessor.kt**

- 输入：任意 `Bitmap`。
- 操作：
  - resize 到 256×256。
  - 读取像素，拆分 R/G/B 三通道。
  - 按 ImageNet 均值/方差归一化。
  - 按 NCHW 排列为 `FloatArray`，shape = `1×3×256×256`。

**OnnxCascadeEngine.kt**

- 初始化：
  - 创建 `OrtEnvironment`。
  - 从 `assets/models/` 加载 ONNX 模型为 `stage1Session` / `stage2Session`。
  - 使用 `CascadeConfig.default()` 设置级联参数（目前与 JSON 一致）。

- `detect(bitmap: Bitmap): CascadeResult`：
  - 预处理：
    - 调用 `ImagePreprocessor.preprocess()`。
    - 记录预处理耗时。
  - Stage1：
    - 构建输入张量，Run Session。
    - 取输出 logit，手动应用 sigmoid 得到 fake 概率 `p1 ∈ (0,1)`。
  - 级联决策：
    - 若 `p1 < tau_low`（0.02）：直接判 REAL，`stage=STAGE1_REAL`。
    - 若 `p1 > tau_high`（0.98）：直接判 FAKE，`stage=STAGE1_FAKE`。
    - 否则：
      - 调用 Stage2，得到 fake 概率 `p2`。
      - 若 `p2 > stage2_threshold`（0.5）→ FAKE，否则 REAL，`stage=STAGE2`。
  - 返回 `CascadeResult`，包含：
    - `isDeepfake` / `confidence` / `stage`。
    - `preprocessingTimeMs` / `inferenceTimeMs` / `totalTimeMs`。

**CascadeResult.kt**

- 封装级联结果：
  - `label`: REAL/FAKE。
  - `confidencePercent`: 置信度百分比字符串。
  - `DetectionStage`: `STAGE1_REAL` / `STAGE1_FAKE` / `STAGE2`。
  - `CascadeConfig.default()`: 当前使用的固定配置。

### 4.2 UI 与交互

**DetectionViewModel.kt**

- `AndroidViewModel`，内部持有一个 `OnnxCascadeEngine`。
- 状态流：
  - `DetectionUiState`：
    - `Idle` → `Initializing` → `Ready` → `Processing` → `Success(result)` / `Error(message)`。
- 初始化时：
  - 在 `viewModelScope.launch` 中调用 `engine.initialize()`。
  - 成功后设置状态为 `Ready`。
- 检测流程：
  - 保存用户选择的 `Uri`。
  - 在 `detectDeepfake()` 中：
    - 通过 `MediaStore.Images.Media.getBitmap()` 加载 Bitmap。
    - 调用 `engine.detect(bitmap)`。
    - 更新 UI 状态为 `Success(result)`，异常时为 `Error(...)`。

**DetectionScreen.kt**

- 使用 `Scaffold + TopAppBar` 布局。
- 使用 `rememberLauncherForActivityResult(ActivityResultContracts.GetContent())` 选择图片。
- 展示：
  - 状态卡片（当前引擎状态）。
  - 图片预览。
  - “Select Image” / “Detect” 按钮。
  - 成功结果卡片（带颜色与图标的 REAL/FAKE 提示，以及详细 metrics）。
  - 错误卡片（错误信息）。

> 总结：V0 已实现“单图 → 两阶段 ONNX 级联 → 结果 + 性能显示”的完整闭环，可以直接用于功能与性能验证。

---

## 5. 后续阶段计划（Phase 2+）

以下阶段尚未实现，是在 V0 基础上逐步增加的功能。

### Phase 2：人脸检测 + 批量测试 + 导出

1. **MediaPipe 人脸检测**
   - 引入 `com.google.mediapipe:tasks-vision`。
   - 新增 `ml/facedetection/FaceDetector.kt`：
     - 对任意图片检测所有人脸 bounding boxes。
     - 选择面积最大或置信度最高的人脸。
     - 按检测框裁剪，再 resize→256×256，交给 `ImagePreprocessor`。
   - UI 上增加简单的人脸框可视化（可选）。

2. **批量测试界面**
   - 新增 `ui/BatchTestScreen.kt` + 对应 ViewModel。
   - 支持从相册多选图片（例如 ≤ 50 张）。
   - 对每张图片执行：人脸检测 → 级联推理。
   - 展示：
     - 列表：缩略图、标签、置信度、Stage、耗时。
     - 汇总：Real/Fake 数量、平均耗时、Stage2 使用率等。

3. **结果导出**
   - 使用 `opencsv` 或 Kotlin 字符串拼接，将批量测试结果导出为 CSV/JSON。
   - 文件保存到 `Downloads` 目录。

### Phase 3：PyTorch Mobile 引擎（对比实验）

1. 使用 `scripts/export_mobile_torchscript.py` 导出 TorchScript 模型（规划中）。
2. 在 `app/build.gradle.kts` 中加入 PyTorch Mobile 依赖。
3. 新增 `ml/PyTorchInferenceEngine.kt`：
   - 接口与 `OnnxCascadeEngine` 一致。
   - 复用 `ImagePreprocessor` 与 `CascadeConfig`。
4. UI 中增加引擎选择：
   - ONNX / PyTorch / 双引擎对比。
5. 对比不同引擎的：
   - 平均耗时。
   - Stage2 使用率。
   - 精度差异。

### Phase 4：性能视图与设置界面

- 性能视图（Performance Screen）：
  - 显示最近 N 次检测的耗时、Stage 分布等。
  - 可能结合简单图表（柱状/折线）。
- 设置界面：
  - 动态调整 `tau_low` / `tau_high` / `stage2_threshold`。
  - 切换默认推理引擎。
  - 开关 MediaPipe 与 NNAPI 等。

### Phase 5：视频支持（长期扩展）

- 使用 `MediaMetadataRetriever` / `MediaCodec` 抽帧。
- 对每 N 帧执行人脸检测 + 级联推理。
- 将帧级结果聚合成视频级判决：
  - 多数投票。
  - 时间轴上标出伪造片段。

---

## 6. 已知改进点（不影响 V0 运行）

当前代码可直接运行，但有若干可以优化的点：

1. **配置来源一致化**
   - 现在 `CascadeConfig.default()` 与 `cascade_config.json` 的数值保持一致，但代码中并未真正解析 JSON。
   - 中期建议：
     - 新建 `ConfigLoader`，使用 `kotlinx-serialization-json` 从 `cascade_config.json` 读取配置。
     - 用读取结果构造 `CascadeConfig`，替换硬编码默认值。

2. **Deprecated API**
   - `DetectionViewModel` 中使用了 `MediaStore.Images.Media.getBitmap()`（新 API 中已 deprecated）。
   - 建议：
     - 在 Android 10+ 使用 `ImageDecoder.decodeBitmap()`。
     - 保留 `getBitmap` 作为旧系统 fallback。

3. **缺少单元测试**
   - 测试依赖已配置，但尚无实际测试用例。
   - 建议优先补充：
     - `ImagePreprocessorTest`：验证输出形状与归一化正确。
     - `CascadeResultTest`：验证 label / confidence / totalTime 逻辑。
     - 对级联逻辑可以用 fake 模型或 mock 替换真 ONNX，以测试决策路径。

以上均为质量和可维护性提升，不影响当前 Demo 的正常运行。

---

## 7. 使用指南（V0）

1. 在 Android Studio 中打开 `D:\work\MobileDeepfakeDetection\android`。
2. 校验 `local.properties` 中 SDK 路径正确，点击 Sync Gradle。
3. 确认 `app/src/main/assets/models/` 下存在：
   - `aware_cascade_stage1.onnx`
   - `aware_cascade_stage2.onnx`
   - `cascade_config.json`
4. 连接真机或启动模拟器（建议真机，≥2GB RAM）。
5. 运行 `app` 模块：
   - 等待模型初始化完成（状态显示 Ready）。
   - 点击 “Select Image”，从相册选择一张人脸图片（最好是 256×256 的预裁剪 PNG）。
   - 点击 “Detect”，查看结果卡片中的标签、置信度与耗时。

如需与 Python 端对齐，可在 PC 上使用相同图片，比较最终标签与概率值。

---

## 8. 下一步应该做什么？

在当前 V0 已经稳定可用的基础上，建议按照优先级推进：

1. **（小步改进）让配置真正来自 `cascade_config.json`**
   - 编写一个简单的配置加载器，从 `assets/models/cascade_config.json` 中读取参数，构造 `CascadeConfig`。
   - 在 `OnnxCascadeEngine` 中使用该配置，而不是 `CascadeConfig.default()`。
   - 好处：之后调阈值不需要改代码，只用改 JSON。

2. **（功能扩展）集成 MediaPipe 人脸检测**
   - 新增 `FaceDetector`，对任意图片裁剪出主脸后再送入当前级联逻辑。
   - 这一步完成后，应用可以对任意手机照片直接检测，而不再依赖预裁剪人脸。

3. **（实验性）设计批量测试界面**
   - 支持多选图片，跑一轮，导出 CSV，便于你用 Python / Excel 做进一步分析。

如果你希望先从某一步开始（例如：先做配置加载，或直接上 MediaPipe），我可以按你选的方向帮你具体设计代码改动方案。 

