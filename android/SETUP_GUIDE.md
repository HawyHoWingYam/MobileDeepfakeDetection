# Android Studio 项目设置指南

## 快速开始

### 1. 安装 Android Studio
下载并安装最新版本的 Android Studio：
https://developer.android.com/studio

推荐版本：Hedgehog (2023.1.1) 或更新版本

### 2. 配置 SDK 路径

编辑 `local.properties` 文件，设置您的 Android SDK 路径：

**Windows:**
```properties
sdk.dir=C\:\\Users\\YOUR_USERNAME\\AppData\\Local\\Android\\Sdk
```

**Mac/Linux:**
```properties
sdk.dir=/Users/YOUR_USERNAME/Library/Android/sdk
```

### 3. 打开项目

1. 启动 Android Studio
2. 选择 "Open an Existing Project"
3. 导航到 `D:\work\MobileDeepfakeDetection\android`
4. 点击 "OK"

### 4. Gradle 同步

Android Studio 会自动开始 Gradle 同步：
- 下载依赖项（ONNX Runtime, Compose, 等）
- 配置构建系统
- 索引项目文件

**首次同步可能需要 5-10 分钟**，取决于网络速度。

### 5. 验证模型文件

确认以下文件存在于 `app/src/main/assets/models/` 目录：
- ✅ `aware_cascade_stage1.onnx` (38 MB)
- ✅ `aware_cascade_stage2.onnx` (53 MB)
- ✅ `cascade_config.json` (2.4 KB)

### 6. 连接设备或启动模拟器

**物理设备：**
1. 在手机上启用"开发者选项"和"USB 调试"
2. 用 USB 连接手机到电脑
3. 在手机上允许 USB 调试授权

**模拟器：**
1. 在 Android Studio 中打开 "Device Manager"
2. 创建新的虚拟设备（推荐：Pixel 5, API 34）
3. 启动模拟器

### 7. 运行应用

1. 在 Android Studio 顶部工具栏选择设备
2. 点击绿色的 "Run" 按钮（或按 Shift+F10）
3. 等待应用构建和安装（首次构建需要 2-5 分钟）
4. 应用会自动在设备上启动

## 常见问题

### Q1: Gradle 同步失败
**解决方案：**
- 检查网络连接
- 在 Android Studio 中：File → Invalidate Caches → Restart
- 删除 `.gradle` 文件夹后重新同步

### Q2: 找不到 SDK
**解决方案：**
- 在 Android Studio 中：File → Project Structure → SDK Location
- 设置正确的 Android SDK 路径
- 确保已安装 Android SDK Platform 34

### Q3: 模型加载失败
**解决方案：**
- 确认 ONNX 文件在 `assets/models/` 目录
- 检查文件大小是否正确（Stage1: 38MB, Stage2: 53MB）
- 清理并重新构建项目：Build → Clean Project → Rebuild Project

### Q4: 应用崩溃或 OOM (Out of Memory)
**解决方案：**
- 在真实设备上测试（模拟器内存可能不足）
- 使用至少 2GB RAM 的设备
- 关闭其他应用释放内存

### Q5: 推理速度慢
**解决方案：**
- 使用 Release 构建而不是 Debug：Build → Select Build Variant → release
- 在真实设备上测试（模拟器性能较差）
- 确保设备有 ARM64 处理器

## 项目结构

```
android/
├── app/
│   ├── src/main/
│   │   ├── java/com/deepfake/detector/
│   │   │   ├── MainActivity.kt              # 主入口
│   │   │   ├── ml/                          # ML 模块
│   │   │   │   ├── OnnxCascadeEngine.kt    # ONNX 推理引擎
│   │   │   │   ├── ImagePreprocessor.kt    # 图像预处理
│   │   │   │   └── CascadeResult.kt        # 数据模型
│   │   │   └── ui/                          # UI 模块
│   │   │       ├── DetectionScreen.kt      # 主界面
│   │   │       ├── DetectionViewModel.kt   # ViewModel
│   │   │       └── theme/                  # Material 3 主题
│   │   ├── assets/models/                   # 模型文件
│   │   │   ├── aware_cascade_stage1.onnx
│   │   │   ├── aware_cascade_stage2.onnx
│   │   │   └── cascade_config.json
│   │   ├── res/                             # 资源文件
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts                     # App 构建配置
├── build.gradle.kts                         # 项目构建配置
├── settings.gradle.kts                      # 项目设置
└── README.md                                # 项目文档
```

## 依赖项

主要依赖（自动下载）：
- **ONNX Runtime Android**: 1.16.0 (~5 MB)
- **Jetpack Compose**: BOM 2023.10.01
- **Coil**: 2.5.0 (图像加载)
- **Kotlin Coroutines**: 1.7.3

## 构建配置

- **Minimum SDK**: API 24 (Android 7.0)
- **Target SDK**: API 34 (Android 14)
- **Compile SDK**: 34
- **Kotlin**: 1.9.20
- **Gradle**: 8.2

## 性能指标

### 预期性能（中端设备）
- **Stage 1 推理**: 5-10 ms
- **Stage 2 推理**: 15-25 ms（仅在需要时）
- **预处理**: 2-5 ms
- **总时间**: 10-30 ms/图片
- **Stage 2 触发率**: 1-5%

### APK 大小
- **Debug**: ~120 MB
- **Release**: ~100 MB（启用 ProGuard 后）

## 测试

### 单元测试
```bash
./gradlew test
```

### 仪器测试（需要设备）
```bash
./gradlew connectedAndroidTest
```

## 下一步

1. ✅ 打开项目并同步 Gradle
2. ✅ 运行应用并测试基本功能
3. 📱 在真实设备上测试性能
4. 🔧 根据需要调整阈值（`cascade_config.json`）
5. 📊 收集性能数据和准确率指标
6. 🎥 实现视频抽帧功能（Phase 2）

## 技术支持

如有问题，请参考：
- Android 开发文档：https://developer.android.com
- ONNX Runtime 文档：https://onnxruntime.ai/docs/
- Jetpack Compose 教程：https://developer.android.com/jetpack/compose/tutorial

## 许可证

本项目是 MobileDeepfakeDetection 研究的一部分。
