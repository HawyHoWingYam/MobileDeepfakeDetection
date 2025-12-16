# MobileDeepfake 硕士答辩演示文档

> 本文档是答辩演示的骨架文件，包含幻灯片结构、内容要点和讲稿。
>
> 最后更新: 2024-12-14

---

## 项目概述

**论文题目**: MobileDeepfake - A Cascaded Detection System for On-Device Deepfake Detection

**答辩时长**: 25-30分钟演讲 + Q&A

**输出格式**: LaTeX Beamer 演示文稿

**主线句**: 在保证移动端实时性的前提下，通过两阶段级联设计，实现高准确率的深度伪造检测，并系统性揭示了严重的跨数据集分布偏移问题。

---

# 第一部分：演示结构大纲 (优化版)

## Part 1: 引言与背景 (4-5分钟, Slide 1-4)

| 页码 | 标题 | 内容要点 | 时间 |
|------|------|----------|------|
| 1 | 封面 | 论文标题、姓名、导师、单位、日期 | 0.5分钟 |
| 2 | 研究背景 | Deepfake威胁、社会风险、移动端检测需求 | 1.5分钟 |
| 3 | 问题与挑战 | 轻量化vs准确率、分布偏移、工程落地 | 1.5分钟 |
| 4 | 研究目标与贡献 | 系统框架图、四大贡献点 | 1分钟 |

## Part 2: 方法设计 (7分钟, Slide 5-8)

| 页码 | 标题 | 内容要点 | 时间 |
|------|------|----------|------|
| 5 | 两阶段级联设计 | Stage1 MobileNetV4 + Stage2 EfficientNetV2-B3 | 2分钟 |
| 6 | 级联决策逻辑 | 双阈值(τ_low, τ_high)、FNR与升级率指标 | 1.5分钟 |
| 7 | 六阶段流水线 | 预处理→训练→调优→评估→导出 | 1.5分钟 |
| 8 | 移动端架构 | 量化、ONNX导出、Android集成 | 2分钟 |

## Part 3: 数据与实验设置 (3分钟, Slide 9-10)

| 页码 | 标题 | 内容要点 | 时间 |
|------|------|----------|------|
| 9 | 数据集介绍 | CelebDF-v2, FF++, DFDC, DeeperForensics (训练) | 1.5分钟 |
| 10 | 跨数据集评估 | Deepfake-Eval-2024 (OOD测试)、评估协议 | 1.5分钟 |

## Part 4: 实验结果 (8-9分钟, Slide 11-15)

| 页码 | 标题 | 内容要点 | 时间 |
|------|------|----------|------|
| 11 | In-domain结果 | AUC/F1对比表、ROC曲线 (亮点) | 2分钟 |
| 12 | 级联效率分析 | FNR ~0.6%, 升级率 ~1.2% (亮点) | 2分钟 |
| 13 | 移动端性能 | 延迟~180ms, 准确率92.8% (亮点) | 2分钟 |
| 14 | 跨数据集挑战 | Deepfake-Eval-2024 F1<0.30 (问题揭示) | 2分钟 |
| 15 | 鲁棒性与错误分析 | 典型误判案例、扰动实验 | 1-1.5分钟 |

## Part 5: 移动部署 (3分钟, Slide 16-17)

| 页码 | 标题 | 内容要点 | 时间 |
|------|------|----------|------|
| 16 | APP界面演示 | 截图、使用流程、检测结果展示 | 1.5分钟 |
| 17 | 工程实现要点 | 项目结构、可复现性、量化经验 | 1.5分钟 |

## Part 6: 总结 (2-3分钟, Slide 18-20)

| 页码 | 标题 | 内容要点 | 时间 |
|------|------|----------|------|
| 18 | 工作总结 | 贡献回顾、关键指标、实际意义 | 1分钟 |
| 19 | 局限与展望 | 分布偏移挑战、未来方向 | 1-1.5分钟 |
| 20 | 致谢与Q&A | 致谢、问答环节 | 0.5分钟 |

---

# 第二部分：关键数据

## 数据集规模

| 数据集 | Train | Val | Test |
|--------|-------|-----|------|
| CelebDF-v2 | 83,599 | 17,478 | 18,037 |
| FaceForensics++ | 223,653 | 50,945 | 47,430 |
| DFDC | 721,946 | 154,919 | 154,511 |
| DeeperForensics | 844,396 | 165,854 | 172,894 |
| **Total** | **1,873,594** | **389,196** | **392,872** |
| Deepfake-Eval-2024 (OOD) | - | 451,917 | 402,413 |

## 核心性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 级联FNR | ~0.6% | 极低漏检率 |
| Stage2升级率 | ~1.2% | 绝大多数样本用轻量模型处理 |
| Stage1 AUC | 0.9936 | 快速过滤器性能 |
| Stage2 AUC | 0.9633 | 精准专家模型性能 |
| 移动端准确率 | 92.8% | 量化后性能 |
| 移动端延迟 | ~180ms | 满足实时性要求 |
| 跨数据集F1 | <0.30 | 揭示分布偏移挑战 |

## 模型大小

| 模型 | FP32 TorchScript | INT8 ONNX |
|------|------------------|-----------|
| Stage1 (MobileNetV4) | 37.3 MB | ~37.5 MB |
| Stage2 (EfficientNetV2-B3) | 49.6 MB | ~52 MB |
| 总计 | ~87 MB | ~89 MB |

---

# 第三部分：幻灯片详细内容与讲稿

---

## Slide 1: 封面

### 内容
- **标题**: MobileDeepfake
- **副标题**: A Cascaded Detection System for On-Device Deepfake Detection
- **作者**: [姓名]
- **导师**: [导师姓名]
- **单位**: [学校/学院]
- **日期**: 2024年12月

### 视觉元素
- 简洁背景，可使用手机+人脸/AI图标的弱对比图
- 学校Logo

### 讲稿
> 各位评委老师好，我是[姓名]，我的硕士论文题目是《MobileDeepfake - 面向移动端的级联深度伪造检测系统》。下面我将用约25分钟的时间向各位老师汇报我的研究工作。

---

## Slide 2: 研究背景

### 内容要点
1. **Deepfake技术快速发展**
   - 基于GAN、扩散模型的换脸技术日益成熟
   - 生成质量已能欺骗人眼和简单检测器

2. **社会风险日益严重**
   - 政治虚假信息影响选举
   - 名人换脸用于诈骗
   - 非自愿亲密内容侵害隐私

3. **移动端检测需求**
   - 60%+的互联网流量来自移动设备
   - 用户需要在本地快速验证内容真伪
   - 隐私要求：敏感数据不上传云端

### 视觉元素
- Deepfake生成和传播路径示意图
- 3个关键风险点的图标+简短说明

### 讲稿 (中文)

> 过去十年，深伪技术从早期的 GAN 换脸，发展到扩散模型和高逼真重演系统，可以生成几乎以假乱真的人脸视频和图像，普通用户用开源工具就能轻松制作。这样的视频已经被大量用来散布政治虚假信息，冒充名人或合成身份进行金融诈骗，以及生成非自愿的色情和亲密影像。与此同时，媒体消费快速向移动端迁移，如今超过 60% 的互联网流量来自手机。用户希望在本地就能验证内容真伪，不必把敏感媒体上传云端，以保护隐私、也支持离线场景。综合来看，在移动设备上实现实时、准确的深伪检测已经变成一个紧迫的研究方向。

### English Script

> Over the last decade, deepfake technology has moved from early GAN-based face swaps to diffusion and advanced reenactment models that generate highly photorealistic faces. These synthetic videos and images are no longer harmless curiosities: they are routinely weaponized for political misinformation, large-scale financial scams using celebrity or synthetic identities, and non-consensual intimate content. At the same time, media consumption has shifted to mobile: today, over 60% of internet traffic comes from phones, and easy-to-use tools make deepfakes widely accessible. Users increasingly expect to verify content locally on their devices, without uploading sensitive media to a server, for both privacy and offline use. Together, these trends make accurate, real-time deepfake detection on mobile devices an urgent research direction.

---

## Slide 2.5: 检测研究的挑战与移动端约束

### 内容要点

**基准与现实的差距**
- 现有模型在策划好的数据集（FF++、CelebDF）上训练
- 迁移到真实数据时，AUC下降20-30%
- 数据集偏差和域偏移问题严重

**服务器级算力假设**
- 绝大多数研究在服务器级GPU上评估
- 对移动端/边缘设备关注不足
- 移动端特定指标很少被报告

**移动端部署约束**
| 约束 | 目标值 |
|------|--------|
| 模型大小 | ≤ 100 MB |
| 延迟 | < 200 ms |
| 能耗 | 电池友好 |
| 隐私 | 仅设备端处理 |

### 视觉元素
- 左侧：两组 bullet points（基准vs现实、服务器假设）
- 右侧：约束表格 + Research Gap block

### 讲稿 (中文)

> 虽然很多深伪检测模型在基准数据集上表现很好，但在真实环境中的效果往往大幅下降。比如，在 FaceForensics++、CelebDF 等精心整理的数据上训练的模型，一旦迁移到真实网络视频，AUC 往往会下降 20–30%，暴露出明显的数据集偏差和分布移位问题。现有工作大多默认使用服务器级 GPU，只报告准确率和算力指标，很少系统性地评估移动端，更缺乏面向手机的延迟和能耗指标。实际部署时，我们希望模型体积不超过 100MB，每帧延迟小于 200ms，同时节省电量，并且只在本机处理敏感媒体。不过，现有系统——包括级联检测器——普遍缺少对 FNR 的显式控制，也难以给出可审计的成本–风险权衡，这正是本研究要填补的空白。

### English Script

> Despite strong benchmark numbers, current deepfake detectors do not transfer cleanly to the real world. Models trained on curated datasets such as FaceForensics++ and CelebDF often suffer a 20–30% AUC drop on in-the-wild media because of dataset bias and distribution shift. Most prior work also assumes server-class GPUs: they report accuracy and FLOPs, but rarely mobile-specific metrics, and almost never end-to-end latency on phones. For practical deployment, however, we need detectors whose core model is no larger than about 100MB, runs in under 200ms per frame, stays battery-friendly, and keeps all media strictly on-device. Yet existing systems, including cascade-style detectors, typically lack explicit FNR control and do not offer auditable cost–risk trade-offs for mobile operators, which is the key gap this work targets.

---

## Slide 3: 问题与挑战

### 内容要点
| 挑战 | 描述 |
|------|------|
| **轻量化 vs 准确率** | 高精度模型通常计算量大，移动端资源受限 |
| **分布偏移** | 训练数据与真实应用场景分布差异大 |
| **FNR 控制** | 需在计算预算内严格控制假阴性率（漏检率） |
| **工程落地** | 从研究模型到可部署系统需要完整流水线 |

### 研究问题
1. 如何在分布偏移条件下，在资源受限的移动设备上实现高召回率检测？
2. 能否通过带有显式阈值的级联架构，实现可解释的代价-风险权衡？
3. 构建可复现、可审计的端到端系统需要哪些最小脚本和产物？
4. 如何评估鲁棒性（压缩、噪声、模糊）和跨数据集泛化以指导部署决策？

### 视觉元素
- 2×2表格或图标列表展示四大挑战
- 底部加粗总结研究目标

### 讲稿 (中文)

> 这一页我把问题具体拆成四个挑战，并用实际数字说明为什么它们难。
>
> 第一，轻量化与准确率的张力。很多最新的深伪检测模型在服务器上可以用上百 MB 的大模型、几百毫秒甚至秒级的推理时间，但在移动端我们必须遵守非常苛刻的约束：整个检测模块最好控制在 100MB 以内，单帧端侧推理时间小于 200ms，还要兼顾电池寿命。MobileDeepfake 的设计目标就是"两阶段主干模型加起来 <100MB、端侧 <200ms"。最终 Stage1 的 MobileNetV4 FP32 约 37.3MB，Stage2 的 EfficientNetV2‑B3 约 49.6MB，INT8 量化后分别压缩到 10.1MB 和 13.4MB；在小米 13 上，人脸级检测准确率 92.8%，平均延迟约 180ms。要在这样紧的算力和容量预算下维持接近 0.99 的 AUC，本身就是一个不小的工程挑战。
>
> 第二是分布偏移。文献和我们的实验都显示，基于 FaceForensics++、CelebDF 这类精心整理数据集训练的模型，一旦迁移到 in‑the‑wild 数据集（比如 WildDeepfake 或 Deepfake‑Eval‑2024），AUC 往往会下降 20–30%。在本系统中，校准后的级联在四个学术数据集的联合验证上可以做到 AUC≈0.9941、F1≈0.965、FNR≈0.60%，Stage2 升级率只有 1.16%；但当我们把同一套级联直接应用到 Deepfake‑Eval‑2024 时，F1 掉到 0.28–0.30，FNR 上升到 73–79%，Stage2 升级率飙升到约 51%。这说明即使 in‑domain 指标很漂亮，一旦遇到 2024 年真实互联网分布，模型仍然会大幅退化。
>
> 第三是假阴性率控制。在深度伪造检测里，真正危险的是"漏掉一个假"，而不是"多报几个真"。如果只用 Stage1，在 combined validation 上 FNR 约为 3.12%，Stage2 单独使用时 FNR 甚至更高，大约 9.43%，而且计算量是 2.87 GFLOPs。我们因此把"在 in‑domain 保持 FNR ≤1%"当成硬约束，通过显式的双阈值机制（τ_low=0.05、τ_high=0.55）和网格搜索来选操作点，最终把 FNR 压到 0.60%，同时把 Stage2 升级率控制在 1.16%。要做到这一点，就必须从设计之初就把 FNR 当成一等公民，而不是训练完随便挑一个阈值。
>
> 最后是工程落地。要支撑前面这些数字，我们不仅要有模型，还要有一整套可复现的流水线：多数据集 manifest 的构建，Stage1/Stage2 训练，Stage1 的温度缩放校准（最优温度 T≈1.34，把 ECE 从 4.21% 降到 0.89%，相对下降约 79%），Stage4 中对 (τ_low, τ_high) 的约束式网格搜索，以及 Stage5 上针对 JPEG、噪声和模糊的鲁棒性扫描。所有这些逻辑最终都收敛到一个 `cascade_config.json` 和一组 `CascadeConfig` 参数上，并被 Android 端的 `OnnxCascadeEngine` 直接消费。这也是为什么这页的四个挑战，在后面的实现中都能一一对应到具体的代码和配置。

### Script (English)

> This slide turns the high‑level problem into four concrete challenges, each backed by numbers.
>
> First, the tension between lightweight models and accuracy. Many state‑of‑the‑art deepfake detectors assume server‑class GPUs and hundreds of megabytes of weights, which simply do not fit into a ≤100MB mobile app bundle or a <200ms per‑frame budget. In MobileDeepfake we explicitly target "two stages under ~100MB and <200ms on device." Stage 1 MobileNetV4 is about 37.3MB in FP32, Stage 2 EfficientNetV2‑B3 is about 49.6MB, and INT8 quantization shrinks them to 10.1MB and 13.4MB. On a Xiaomi 13 (Snapdragon 8 Gen 2) the on‑device cascade reaches 92.8% face‑level accuracy with roughly 180ms end‑to‑end latency per face. Achieving near‑0.99 AUC under these constraints is already non‑trivial.
>
> Second, distribution shift. Prior work and our own experiments show that models trained on FaceForensics++ or CelebDF can lose 20–30 percentage points of AUC when evaluated on in‑the‑wild datasets such as WildDeepfake or Deepfake‑Eval‑2024. In our system, the calibrated cascade on the four academic datasets achieves about AUC=0.9941, F1≈0.965, FNR≈0.60% with only 1.16% of samples escalated to Stage 2. But when we take the exact same cascade and apply it to Deepfake‑Eval‑2024, F1 collapses to 0.28–0.30, FNR jumps to 73–79%, and the Stage‑2 escalation rate explodes to around 51%. This is a concrete, measured manifestation of dataset bias and domain shift in 2024 media.
>
> Third, explicit FNR control. For deepfake detection, the catastrophic failure is a missed fake, not an extra false alarm, so we treat FNR as a first‑class constraint. On the combined validation set, Stage 1 alone yields FNR≈3.12% at 0.54 GFLOPs; Stage 2 alone is even worse at FNR≈9.43% and 2.87 GFLOPs. We therefore formulate Stage 4 as a constrained optimization problem: search over (τ_low, τ_high) to keep FNR ≤1% while minimizing the Stage‑2 rate. The safety‑first operating point τ_low=0.05, τ_high=0.55 reaches FNR=0.60% with only 1.16% of samples escalated, giving roughly a 5× FNR reduction compared to Stage 1 for only ~9% extra compute.
>
> Finally, the engineering gap. To make these numbers reproducible and deployable, we build a six‑stage pipeline: multi‑dataset manifests, Stage‑1 and Stage‑2 training, temperature‑scaling calibration, constrained threshold grid search, robustness sweeps over JPEG/noise/blur, and mobile export. In the code, this all collapses into a `CascadeConfig` dataclass (with fields like `stage1_real_threshold` and `stage1_fake_threshold`), a `calibration_temp.json` file that stores the optimal temperature T≈1.34 (reducing Stage‑1 ECE from 4.21% to 0.89%, a 78.9% drop), and a `cascade_config.json` bundle consumed by the Android `OnnxCascadeEngine`. So each abstract challenge on this slide is tied directly to concrete configuration knobs and scripts in the implementation.

---

## Slide 3.5: 技术挑战详解

### 内容要点

**假阴性控制**
- 漏检的代价很高（虚假信息传播）
- 必须在计算预算内维持 FNR < 1%
- 需要显式、可审计的阈值调优

**概率校准**
- 原始模型分数常常校准不良
- Temperature scaling 提供可靠置信度
- 对基于阈值的决策至关重要

**扰动鲁棒性**
- 真实媒体中存在 JPEG 压缩、噪声、模糊
- 多平台效果（滤镜、字幕）
- 需要系统性的评估协议

**权衡分析图**
- X轴: Stage 2 升级率（计算成本）
- Y轴: FNR（风险）
- 显示双曲线形状的 trade-off 曲线
- 标注当前操作点 (τ_low=0.05, τ_high=0.55)

### 视觉元素
- 左侧：三组技术挑战的 bullet points
- 右侧：FNR vs Stage 2 Rate 的 trade-off 曲线图
- 底部：Core Insight 总结框

### 讲稿 (中文)

> 接下来这页，我从实现角度具体展开刚才提到的技术挑战。
>
> 首先是假阴性率控制在代码里的落地。Stage1 输出的是一个二分类 logit，我们在 Stage1 校准脚本里学习到最优温度 T≈1.34，然后在级联系统中统一用 `p1(x)=σ(z/T)` 作为假脸概率。在 Stage4，我们在 combined validation manifest 上枚举一系列 (τ_low, τ_high) 组合，计算每一对阈值对应的 FNR 和 Stage2 升级率。Safety‑first 的配置就是这样选出来的：τ_low=0.05、τ_high=0.55，把 FNR 从单独 Stage1 的 3.12% 压到 0.60%，Stage2 升级率只有 1.16%，平均计算量从 0.54 GFLOPs 升到 0.59 GFLOPs。Stage2 单独使用时 FNR 反而更高（约 9.43%、2.87 GFLOPs），说明收益来自"路由策略"，而不是某个单一更强的大模型。
>
> 在 Python 代码中，这一逻辑对应到 `CascadeConfig.stage1_real_threshold=0.05` 和 `stage1_fake_threshold=0.55`，以及 `CascadeDetector.stage1_predict()` 中对 logit 做温度缩放；在 Android 端，同样的 τ_low/τ_high 被写进 `cascade_config.json`，由 `OnnxCascadeEngine.detect()` 中的 `if (stage1Prob < tauLow) ... else if (stage1Prob > tauHigh) ...` 这种级联分支来执行。所以"FNR < 1%"不是一个口号，而是被编码进配置文件和推理流程中的硬约束。
>
> 第二个难点是概率校准。未经校准的 MobileNetV4 在联合验证集上的 Expected Calibration Error 约为 4.21%，意味着"0.9 的置信度"不能解释为"90% 的成功率"。我们用 `src/stage1/calibrate_model.py` 对 held‑out 验证集做温度缩放，拟合出最优温度 T=1.34。这个温度被写入 `calibration_temp.json`，推理时 `CascadeDetector` 读取该文件，在 `stage1_predict()` 里用 `scaled_logits = logits / T` 再过 sigmoid。校准之后，Stage1 的 ECE 降到 0.89%，相对降低约 78.9%，而 AUC 保持在 0.9936 不变，使得 τ_low、τ_high 真正可以解释为"约 5% / 55% 的假脸概率"，方便在不同数据集和设备之间迁移操作点。
>
> 第三个难点是鲁棒性评估要有系统性，而不是只看干净样本。Stage5 的脚本会对同一批验证人脸施加四类扰动：JPEG 压缩（质量从 95 降到 20）、高斯噪声（σ 从 2 到 12）、运动模糊（卷积核大小 3–13）、以及亮度/对比度缩放。每一个设置，我们都会输出 F1、FNR 和 Stage2 升级率。比如，在 JPEG 质量 q=20 的强压缩下，FNR 约 30.6%，Stage2 升级率约 5.2%；在高斯噪声 σ=12 时，FNR 约 67.4%，Stage2 升级率约 48.5%；而在运动模糊核大小 3–13 的范围内，FNR 几乎一直在 96–100% 之间，Stage2 升级率也维持在 17–38% 左右。这些数字说明：级联架构在适度压缩和噪声下还能保持一定性能，但在严重模糊等极端场景下几乎完全失效，这一点必须在早期就明确给出。
>
> 最后，右侧的 FNR–Stage2 升级率曲线，实际上就是上述机制在代码中的综合产物。我们在阈值网格搜索时，对每一对 (τ_low, τ_high) 记录 FNR、Stage2 升级率和 GFLOPs，形成一条 Pareto 前沿。Safety‑first 点（0.05, 0.55）落在曲线的左下角；如果把 τ_low 再降到 0.03，可以把 FNR 进一步压到 0.54%，但 Stage2 升级率会升到 1.50%。这种带具体数字的 trade‑off，让我们可以非常明确地回答平台方的问题：为了把 FNR 从 0.60% 再降 0.06 个百分点，需要多付出多少计算成本和端侧延迟。

### Script (English)

> Let me now unpack these technical challenges from an implementation perspective.
>
> First, FNR control is implemented as an explicit routing problem. Stage 1 produces raw logits; we interpret the calibrated fake probability as p₁(x)=σ(z/T) with a learned temperature T≈1.34, and then search over threshold pairs (τ_low, τ_high) on the combined validation manifest. At the safety‑first operating point τ_low=0.05, τ_high=0.55, the cascade drives FNR down from 3.12% for Stage‑1‑only to 0.60%, while escalating only 1.16% of samples to Stage 2 and increasing average compute from 0.54 to 0.59 GFLOPs. Stage 2 by itself actually has a higher FNR of about 9.43% at 2.87 GFLOPs, so the gain comes from smarter routing, not from a single "better" model.
>
> In the code this appears as `CascadeConfig.stage1_real_threshold=0.05` and `stage1_fake_threshold=0.55` on the Python side, plus the use of temperature scaling inside `CascadeDetector.stage1_predict()`. For mobile deployment, the same τ_low and τ_high values are written into `cascade_config.json` as `tau_low` / `tau_high` and consumed by the Android `OnnxCascadeEngine.detect()` method via a simple `if (stage1Prob < tauLow) ... else if (stage1Prob > tauHigh) ...` cascade. In other words, the "FNR < 1%" requirement is encoded directly into configuration and runtime logic, not just stated in the paper.
>
> Second, probability calibration is what makes those thresholds meaningful. The uncalibrated MobileNetV4 has an Expected Calibration Error of about 4.21% on the combined validation set, so a score of 0.9 does not correspond to a 90% success rate. The script `src/stage1/calibrate_model.py` fits a single temperature on a held‑out split, yielding an optimal T=1.34. This value is stored in `calibration_temp.json`, and during inference `CascadeDetector.stage1_predict()` loads it and applies `scaled_logits = logits / T` before the sigmoid. After calibration, Stage‑1 ECE drops to 0.89% (a 78.9% relative reduction) while AUC remains at 0.9936, so τ_low and τ_high can be interpreted as approximate probability levels rather than arbitrary scores.
>
> Third, robustness is evaluated through a systematic corruption sweep rather than a one‑off demo. In Stage 5 we take a fixed validation manifest and apply four perturbation families: JPEG compression (quality 95→20), Gaussian noise (σ from 2 to 12), motion blur (kernel sizes 3–13), and brightness/contrast scaling. For each setting we log F1, FNR, and Stage‑2 rate. For example, at JPEG quality q=20 the cascade sees FNR≈30.6% with a Stage‑2 rate of about 5.2%; with Gaussian noise σ=12, FNR is ≈67.4% and the Stage‑2 rate ≈48.5%; under motion blur, all tested kernels push FNR into the 96–100% range with Stage‑2 rates around 17–38%. These numbers show that the cascade retains some robustness under moderate compression or noise, but essentially fails under strong blur—an important limitation to surface early.
>
> Finally, the trade‑off curve on the right—FNR versus Stage‑2 escalation rate—is produced directly by the threshold search code. In `benchmark_cascade.py` we sweep a grid of (τ_low, τ_high) pairs over the calibrated probabilities, compute FNR, Stage‑2 rate, and GFLOPs for each, and plot the resulting Pareto frontier. The safety‑first point (0.05, 0.55) sits near the lower‑left; lowering τ_low to 0.03 can push FNR to about 0.54%, but at the cost of raising the Stage‑2 rate to roughly 1.50%. Presenting the curve this way lets us tell operators, in concrete numbers, exactly how much extra compute and latency they must pay to squeeze out each additional fraction of a percent of FNR.

---

## Slide 4: 研究目标与贡献

### 内容要点

**四大贡献:**

1. **成本感知的两阶段级联系统**
   - 轻量MobileNetV4快速过滤 + EfficientNetV2-B3精准分析
   - 显式控制FNR和计算成本的双阈值机制

2. **多数据集训练与跨数据集评估**
   - 4个学术数据集联合训练
   - Deepfake-Eval-2024独立OOD测试

3. **端到端可复现流水线**
   - 6阶段完整流程
   - 从原始数据到移动部署

4. **移动端部署方案**
   - 量化压缩和ONNX导出
   - Android应用集成模板

### 视觉元素
- 左侧：系统整体框图（输入→两阶段检测→结果）
- 右侧：4条贡献bullet，每条配图标

### 讲稿 (中文)

> 这一页我把刚才的问题设定，具体落到本论文的研究目标和四个贡献上。
>
> 第一项贡献是一个成本感知的两阶段级联系统。我们用 MobileNetV4 做 Stage 1 轻量过滤器，用 EfficientNetV2‑B3 做 Stage 2 专家模型，通过显式的双阈值路由策略在易样本和难样本之间分工。在四个学术数据集组成的联合验证集上（约 187 万训练样本、38.9 万验证样本），我们对 (τ_low, τ_high) 进行网格搜索，选出一个 safety‑first 操作点：级联整体 AUC 约 0.9941，系统级 FNR 仅 0.60%，只有 1.16% 的样本会被升级到 Stage 2。也就是说，在几乎不牺牲召回的前提下，把绝大部分计算量都控制在轻量级的 Stage 1 上。
>
> 第二项贡献是多数据集联合训练和系统性的跨数据集评估。Stage 1 和 Stage 2 都是在 CelebDF‑v2、FaceForensics++、DFDC 和 DeeperForensics‑1.0 四个数据集的统一 manifest 上共同训练的，训练集约 187 万张人脸，验证集约 38.9 万张。然后，我们把调好阈值的级联直接迁移到 Deepfake‑Eval‑2024 这个严格分布外的基准上，它包含大约 45.1 万验证样本和 40.2 万测试样本。通过同时报告 in‑domain 和 OOD 上的 FNR、Stage‑2 升级率和 AUC，我们可以量化「单模型 vs 级联」、「校准 vs 未校准」在 2024 年真实社交媒体分布下的性能差距，而不是只停留在单一数据集上的指标。
>
> 第三项贡献是一条端到端可复现的六阶段流水线。从 Stage 0 的预处理与 manifest 生成开始，到 Stage 1 / Stage 2 的训练和温度缩放校准，再到 Stage 4 的阈值网格搜索、Stage 5 的扰动鲁棒性扫描以及最后的移动导出，每一步都有对应的脚本和配置文件。只要给定原始数据路径和环境配置，读者可以从头重新跑一遍，自动生成论文中的绝大部分表格和图，这对于后续审计、复现和工程落地都非常关键。
>
> 第四项贡献是一个经过实际验证的移动端部署方案。我们先通过 INT8 动态量化，把 FP32 TorchScript 模型从 37.3MB / 49.6MB 压缩到 10.1MB / 13.4MB，再导出为 ONNX 并打包进 Android 应用。在小米 13 真机上，完整级联在单人脸场景下可以达到约 92.8% 的帧级准确率，端到端延迟大约 180ms。更重要的是，这个 App 使用的就是研究阶段同一套权重和阈值配置，通过 `cascade_config.json` 把 PC 端调好的 τ_low、τ_high 和温度参数原封不动地搬到设备上，实现「论文里的数字」和「手机上的体验」的一致性。

### Script (English)

> This slide turns those challenges into four concrete objectives and contributions.
>
> First, we design a cost‑aware two‑stage cascade. Stage 1 is a lightweight MobileNetV4 filter, Stage 2 is an EfficientNetV2‑B3 expert, and we route samples using an explicit pair of thresholds (τ_low, τ_high). On the combined validation set built from four academic datasets (~1.87M training and ~389k validation face crops), a grid search over these thresholds yields a safety‑first operating point with AUC≈0.9941, system‑level FNR=0.60%, and only 1.16% of samples escalated to Stage 2. In other words, the cascade achieves server‑grade recall while keeping almost all computation in the cheap Stage‑1 path.
>
> Second, we commit to multi‑dataset training and cross‑dataset evaluation. Both stages are trained jointly on CelebDF‑v2, FaceForensics++, DFDC, and DeeperForensics‑1.0 using a unified, balanced manifest, and then evaluated not only in‑domain but also on Deepfake‑Eval‑2024, a strictly held‑out OOD benchmark with roughly 451k validation and 402k test face crops. By reporting AUC, FNR, and Stage‑2 usage for single‑stage vs. cascade, and for calibrated vs. uncalibrated scores, we can quantify exactly how much robustness we gain—or still lack—when moving from curated benchmarks to 2024 social‑media videos.
>
> Third, we package everything into an end‑to‑end, scriptable six‑stage pipeline. Stage 0 handles preprocessing and manifest generation; Stages 1 and 2 train the MobileNetV4 filter and EfficientNetV2‑B3 expert; Stage 3 hosts optional meta‑models and hard‑example mining experiments; Stage 4 performs cost‑aware threshold tuning over (τ_low, τ_high); and Stages 5–6 cover robustness sweeps and mobile export. Given the raw data and the environment file, every table and figure in the thesis can be regenerated from these scripts, which is essential for reproducibility and auditability.
>
> Fourth, we deliver a practical mobile deployment path. We first compress the FP32 models (37.3MB and 49.6MB) to 10.1MB and 13.4MB using INT8 TorchScript quantization, then export them to ONNX and bundle them into an Android app. On a Xiaomi 13, this on‑device cascade reaches about 92.8% face‑level accuracy with roughly 180ms end‑to‑end latency per face. The app reads the same `cascade_config.json` used on the desktop side, so the τ_low, τ_high, and calibration temperatures tuned on the validation set transfer directly to the device without any re‑training.

---

## Slide 4.5: 贡献技术细节

### 内容要点

**1. 成本感知级联**
- Stage-1 leakage + Stage-2 escalation 作为系统级指标
- Cost-sensitive grid search 进行阈值调优
- 可选的 Stage-3 LightGBM（仅用于分析）

**2. 跨数据集评估**
- 量化跨数据集差距和失败模式
- 比较单模型 vs 级联
- 校准 vs 未校准阈值

**3. 可脚本化流水线**
- 数据预处理 + manifest 生成
- Hard-example mining (HEM)
- 校准 + 鲁棒性扫描
- 脚本可复现论文所有表格/图表

**4. 移动导出模板**
- INT8 量化 + knowledge distillation
- TorchScript（主）+ ONNX 导出
- 元数据文件：阈值 + temperature
- 可复用的 Android 集成模板

### 视觉元素
- 左右两栏布局，每栏两个贡献的技术细节
- 底部：Design Philosophy 总结框

### 讲稿 (中文)

> 这张过渡页我从工程实现的角度，再把刚才四个贡献拆得更细一点。
>
> 第一，成本感知级联。我们把 Stage‑1 leakage 定义为：在所有假样本中，被 Stage 1 当成「简单真实」直接放行、且不会升级到 Stage 2 的比例；把 Stage‑2 escalation rate 定义为：所有样本中被路由到 Stage 2 的比例。Stage 1 侧在 `src/stage1/utils.analyze_cascade_performance` 里先对一维阈值做分析，计算过滤率和 leakage；Stage 4 的 `src/stage4/benchmark_cascade.py` 则在联合验证 manifest 上对 (τ_low, τ_high) 做二维网格搜索，逐点评估 FNR、Stage‑2 升级率、leakage 和 GFLOPs。当前使用的配置在 combined validation 上达到 AUC≈0.9941、FNR=0.60%、Stage‑2 升级率 1.16%。这些数值被写入 `outputs/stage4/run_*/best_config.json`，并同步到用于移动端的 `cascade_config.json`，后续像鲁棒性扫描脚本和 Android 端的级联引擎都以这两份配置作为阈值的单一来源。
>
> 第二，跨数据集评估。数据层面，我们通过统一的 manifest 机制，把 CelebDF‑v2、FaceForensics++、DFDC 和 DeeperForensics‑1.0 四个数据集拼成约 187 万训练样本、38.9 万验证样本的联合数据集，`scripts/preprocess_datasets_v2.py` 和 `scripts/regenerate_manifests.py` 负责生成这些 CSV。Stage 1 和 Stage 2 的训练脚本只依赖「manifest + data_root」这两个参数，就可以在同一接口上完成训练。评估阶段，`src/stage1/evaluate_stage1.py`、`src/stage2/evaluate_stage2.py` 和 `src/stage4/benchmark_cascade.py` 会在四个 in‑domain 数据集和 Deepfake‑Eval‑2024（约 45.1 万验证、40.2 万测试样本）上分别跑三种配置：Stage‑1‑only、Stage‑2‑only 和 cascade，并同时记录「未校准分数 + 直接阈值」和「温度缩放后 + 同一组阈值」的结果，这样单模型 vs 级联、校准 vs 未校准的差异都可以用具体的 FNR 和 Stage‑2 升级率来刻画。
>
> 第三，可脚本化流水线。Stage 0 的预处理和 manifest 生成完全由脚本驱动；Stage 1 / Stage 2 训练脚本会把最优权重、曲线和指标统一写入 `outputs/stage*/run_*/` 目录下的 JSON 与 CSV。Stage 3 在需要时可以启用 hard‑example mining，把高损失样本或 Stage 1 / Stage 2 的错误样本抽出来，构造成一个元数据集供 LightGBM 等元模型做消融实验；但默认部署路径仍然使用标准的统一采样。Stage 1 的 `calibrate_model.py` 和 Stage 2 的 `calibrate_stage2.py` 负责温度缩放，把最优温度写入小的 JSON 文件；Stage 5 的 `src/tools/analyze_robustness.py` 则对 JPEG 压缩、高斯噪声、运动模糊、亮度变化等扰动做系统扫描，生成包含 F1、FNR、Stage‑2 升级率的 CSV 和 LaTeX 插图片段。整个链路从原始 PNG 到鲁棒性曲线，中间没有「手动点点点」这种不可追踪的步骤。
>
> 第四，移动导出。训练好的 FP32 模型首先在 PyTorch 中通过动态量化得到 10.1MB / 13.4MB 的 INT8 TorchScript 版本，作为体积受限场景的备选格式；随后在 `scripts/export_mobile_cascade_onnx.py` 中用 `DeepfakeClassifier` 统一封装 Stage 1 和 Stage 2 的骨干与分类头，交给 `src/stage4/mobile_deployment/onnx_exporter.ONNXExporter` 导出为 ONNX。Exporter 会自动做一次 PyTorch vs ONNX 输出对齐验证，为每个模型写入元数据，并在 `android/mobile_bundle/` 下生成 `aware_cascade_stage1.onnx`、`aware_cascade_stage2.onnx`、`aware_cascade_manifest.json` 和 `cascade_config.json`。Android 端的 `OnnxCascadeEngine` 从 assets 里加载这两份 ONNX 和配置 JSON，按「小于 τ_low 判真、大于 τ_high 判假、否则升级 Stage 2」的逻辑运行，并通过 `CascadeResult` 返回阶段和时间信息，这就是我们在小米 13 上测得约 92.8% 准确率和 ~180ms 延迟的直接数据来源。

### Script (English)

> On this transition slide I'd like to zoom in on how each contribution is implemented in code.
>
> First, the cost‑aware cascade. We define Stage‑1 leakage as the fraction of fake samples that Stage 1 confidently labels as "simple real" and therefore never escalates, and Stage‑2 escalation rate as the fraction of all samples routed to Stage 2. On the Python side, `src/stage1/utils.analyze_cascade_performance` analyzes one‑dimensional thresholds for Stage 1, while `src/stage4/benchmark_cascade.py` sweeps a grid of (τ_low, τ_high) over the combined validation manifest and logs FNR, Stage‑2 usage, leakage, and GFLOPs for each pair. The best configuration—AUC≈0.9941, FNR=0.60%, Stage‑2 rate=1.16%—is stored in `outputs/stage4/run_*/best_config.json` and mirrored in the mobile‑side `cascade_config.json`, so downstream components such as the robustness scripts and the Android cascade engine treat these files as the single source of truth for thresholds.
>
> Second, cross‑dataset evaluation. The data pipeline starts with unified manifests built by `scripts/preprocess_datasets_v2.py` and `scripts/regenerate_manifests.py`, which turn CelebDF‑v2, FaceForensics++, DFDC, and DeeperForensics‑1.0 into a single pool of about 1.87M training and 389k validation face crops. Stage‑1 and Stage‑2 training scripts depend only on these manifests and a data root. At evaluation time, `src/stage1/evaluate_stage1.py`, `src/stage2/evaluate_stage2.py`, and `src/stage4/benchmark_cascade.py` run three configurations—Stage‑1‑only, Stage‑2‑only, and full cascade—on each of the four in‑domain datasets and on Deepfake‑Eval‑2024 (≈451k validation and ≈402k test samples). For every run we log AUC, F1, FNR, Stage‑2 usage, and a flag indicating whether temperature scaling was enabled, which lets us quantify single vs. cascade and calibrated vs. uncalibrated performance gaps instead of just eyeballing ROC curves.
>
> Third, the scriptable pipeline. Stage 0 preprocessing and manifest generation are fully encoded in scripts; Stages 1 and 2 write all checkpoints, curves, and metrics into `outputs/stage*/run_.../` directories with JSON summaries; Stage 3 can optionally turn high‑loss or misclassified samples into a meta‑dataset for hard‑example mining and LightGBM experiments, while the default public pipeline sticks to uniform sampling for simplicity. Stage‑1 `calibrate_model.py` and Stage‑2 `calibrate_stage2.py` perform temperature scaling and store the learned temperatures in small JSON files; and Stage 5's `src/tools/analyze_robustness.py` drives corruption sweeps over JPEG compression, Gaussian noise, motion blur, and brightness changes, emitting both CSV metrics and LaTeX include snippets. There are no hidden manual steps between raw PNGs and the robustness figures in the paper.
>
> Fourth, mobile export. After training, we apply dynamic INT8 quantization in PyTorch to obtain 10.1MB and 13.4MB TorchScript variants of Stage 1 and Stage 2, then use `scripts/export_mobile_cascade_onnx.py` together with `src/stage4/mobile_deployment/onnx_exporter.ONNXExporter` to convert the same architectures to ONNX and assemble an `android/mobile_bundle/` directory. The exporter validates PyTorch vs. ONNX outputs, attaches metadata, and creates `aware_cascade_stage1.onnx`, `aware_cascade_stage2.onnx`, an `aware_cascade_manifest.json`, and a `cascade_config.json` capturing τ_low, τ_high, the Stage‑2 decision threshold, and the calibration temperature. On Android, `OnnxCascadeEngine` loads those ONNX files and the JSON, applies the "τ_low → real, τ_high → fake, else escalate" routing rule, and returns a `CascadeResult` with stage and timing information—the same information we use to report ~92.8% accuracy and ~180ms latency on the Xiaomi 13.

---

## Slide 5: 两阶段级联设计

### 内容要点

- **设计动机**: 在移动端约180ms延迟约束下，同时兼顾检测准确率和能耗
- **Stage 1: 轻量级 MobileNetV4**
  - 输入人脸裁剪图像，输出假脸概率 `p1(x)`
  - AUC = 0.9936，模型大小 37.3MB
  - 主打"快且保守不过度升级"
- **Stage 2: 高精度 EfficientNetV2-B3**
  - 只处理Stage1低置信度样本，输出 `p2(x)`
  - AUC = 0.9633，模型大小 49.6MB
  - 更关注边界样本的判别能力
- **级联效果**:
  - 双阶段组合后整体 FNR ≈ 0.6%
  - 升级到Stage2的比例约1.2%，大部分样本停留在Stage1即可完成判决
- **设计权衡**:
  - 绝大多数样本只经过一次轻量推理 → 降低平均时延和功耗
  - 少数疑难样本交给Stage2 → 提升整体鲁棒性和安全性

### 视觉元素

- **级联框图**:
  - 左侧输入视频帧/图像 → 中间Stage1(小而快的模型框) → 右侧"真实/伪造"两条粗线
  - 从Stage1中间再分一条细线到Stage2(大而精的模型框) → 再输出"真实/伪造"
- **对比表格**:
  | 模型 | AUC | 大小 | 角色 |
  |------|-----|------|------|
  | MobileNetV4 | 0.9936 | 37.3MB | 快速过滤器 |
  | EfficientNetV2-B3 | 0.9633 | 49.6MB | 精准专家 |
- 在图中标注"仅~1.2%样本进入Stage2"

### 讲稿 (中文)

> 这一部分我主要介绍整个系统的两阶段级联设计，以及为什么选择这样的架构。
>
> 在移动端场景下，我们既要在单帧大约 180 毫秒的时延预算内完成推理，又要尽可能保证深度伪造检测的准确性。单一的大模型很难同时满足这两个目标，因此我采用了两阶段级联的结构。
>
> Stage 1 使用的是 `timm` 库中的 `mobilenetv4_hybrid_medium.ix_e550_r256_in1k`，这是一个专门为移动端优化的轻量级骨干网络。在 `src/stage1/train_stage1.py` 中，我们用 `BCEWithLogitsLoss` 作为损失函数，配合 `AdamW` 优化器和 `CosineAnnealingLR` 调度器进行训练。输入是 256×256 的人脸裁剪图像，输出是一个 logit，经过温度缩放后的 sigmoid 得到假脸概率 `p1(x) = σ(z/T)`，其中 T≈1.34。Stage 1 的 AUC 达到 0.9936，模型大小 37.3MB，计算量仅 0.54 GFLOPs。
>
> Stage 2 使用的是 `efficientnetv2_b3.in21k_ft_in1k`，这是一个更大、更精细的模型。在 `src/stage2/train_stage2_effnet.py` 中，训练配置与 Stage 1 有明显区别：我们使用 Focal Loss 配合 label smoothing 来处理类别不平衡，额外加入了 `RandAugment` 和 Mixup/CutMix 增强，以及 `CosineAnnealingWarmRestarts` 调度器。Stage 2 的 AUC 为 0.9633，模型大小 49.6MB，计算量 2.87 GFLOPs。虽然单独使用时 FNR 高达 9.43%，但它专门处理 Stage 1 无法确定的边界样本。
>
> 级联的核心逻辑在 `src/stage4/cascade_detector.py` 的 `CascadeDetector.predict()` 方法中实现。`CascadeConfig` 数据类定义了两个关键阈值：`stage1_real_threshold`（τ_low=0.05）和 `stage1_fake_threshold`（τ_high=0.55）。路由规则很简单：如果 `p1(x) < 0.05`，直接判为真实；如果 `p1(x) > 0.55`，直接判为伪造；否则升级到 Stage 2。通过这样的设计，最终系统在保证整体 FNR 只有 0.60% 的情况下，升级到 Stage 2 的比例仅约 1.16%。

### Script (English)

> This section covers the two-stage cascade design and the rationale behind this architecture.
>
> On mobile devices, we need to complete inference within roughly 180ms per frame while maintaining high detection accuracy. A single large model cannot satisfy both constraints, so we adopt a two-stage cascade structure.
>
> Stage 1 uses `mobilenetv4_hybrid_medium.ix_e550_r256_in1k` from the `timm` library, a lightweight backbone optimized for mobile deployment. In `src/stage1/train_stage1.py`, we train with `BCEWithLogitsLoss`, `AdamW` optimizer, and `CosineAnnealingLR` scheduler. The input is a 256×256 face crop, and the output is a single logit. After temperature scaling, the fake probability is computed as `p1(x) = σ(z/T)` where T≈1.34. Stage 1 achieves AUC=0.9936 with only 37.3MB model size and 0.54 GFLOPs.
>
> Stage 2 uses `efficientnetv2_b3.in21k_ft_in1k`, a larger and more precise model. In `src/stage2/train_stage2_effnet.py`, the training configuration differs significantly: we use Focal Loss with label smoothing for class imbalance, add `RandAugment` and Mixup/CutMix augmentation, and use `CosineAnnealingWarmRestarts` scheduler. Stage 2 has AUC=0.9633, 49.6MB size, and 2.87 GFLOPs. While its standalone FNR is 9.43%, it specializes in handling boundary samples that Stage 1 cannot confidently classify.
>
> The cascade logic is implemented in `CascadeDetector.predict()` in `src/stage4/cascade_detector.py`. The `CascadeConfig` dataclass defines two key thresholds: `stage1_real_threshold` (τ_low=0.05) and `stage1_fake_threshold` (τ_high=0.55). The routing rule is simple: if `p1(x) < 0.05`, classify as real; if `p1(x) > 0.55`, classify as fake; otherwise escalate to Stage 2. With this design, the system achieves FNR=0.60% while only escalating 1.16% of samples to Stage 2.

---

## Slide 5.5: 级联效率与操作点

### 内容要点

**计算成本比较**
| 配置 | GFLOPs | FNR | Stage2 |
|------|--------|-----|--------|
| Stage 1 only | 0.54 | 3.12% | -- |
| Stage 2 only | 2.87 | 9.43% | 100% |
| **Cascade** | **0.59** | **0.60%** | 1.16% |

**量化影响**
- Stage 1: 37.3 MB → 10.1 MB (INT8)
- Stage 2: 49.6 MB → 13.4 MB (INT8)
- AUC 损失: < 0.01 per stage

**部署场景**
| 场景 | τ_low/τ_high | FNR |
|------|--------------|-----|
| Safety-first | 0.05 / 0.55 | 0.60% |
| Balanced | 0.10 / 0.50 | 1.2% |
| Speed-first | 0.15 / 0.45 | 2.0% |

**校准效果**
- Temperature scaling: T ≈ 1.34
- ECE 降低 79%
- 使阈值决策更可靠

### 视觉元素
- 左侧：计算成本比较表 + 量化影响
- 右侧：部署场景表 + 校准效果
- 底部：Key Insight 总结框

### 讲稿 (中文)

> 让我进一步展示级联系统的效率和不同操作点的选择，这些数字直接来自论文的消融实验。
>
> 左侧的表格比较了不同配置的计算成本。单独使用 Stage 1 需要 0.54 GFLOPs，FNR 为 3.12%。单独使用 Stage 2 需要 2.87 GFLOPs，但 FNR 反而更高，达到 9.43%——这说明 Stage 2 并不是一个"更好"的模型，而是专门处理边界样本的专家。我们的级联系统只需要 0.59 GFLOPs——因为只有 1.16% 的样本需要经过 Stage 2——但 FNR 降低到了 0.60%。计算公式是 `Cascade_GFLOPs = 0.54 + r × 2.87`，其中 r 是 Stage-2 升级率。这意味着我们用仅 9.3% 的额外计算开销，将 FNR 降低了 5.2 倍。
>
> 在量化方面，我们在 `src/stage4/optimize_for_mobile.py` 中实现了 INT8 动态量化。Stage 1 从 37.3MB 压缩到 10.1MB，Stage 2 从 49.6MB 压缩到 13.4MB，压缩率都是 73%。AUC 损失分别只有 0.003 和 0.007，完全在可接受范围内。
>
> 右侧展示了不同部署场景的阈值配置。这些操作点是通过 `src/tools/robustness_threshold_sweep.py` 中的 `sweep_grid` 函数对 (τ_low, τ_high) 进行网格搜索得到的。Safety-first 配置使用 τ_low=0.05, τ_high=0.55，实现 0.60% 的 FNR，适合内容审核场景。Balanced 配置使用 τ_low=0.10, τ_high=0.50，FNR 为 1.2%，适合一般移动部署。Speed-first 配置使用 τ_low=0.15, τ_high=0.45，FNR 为 2.0%，适合对延迟极度敏感的场景。
>
> 最后，概率校准是让这些阈值有意义的关键。在 `src/stage1/calibrate_model.py` 中，`TemperatureScalingCalibrator` 类通过最小化 NLL 来拟合最优温度 T=1.34。校准后，ECE 从 4.21% 降到 0.89%，相对降低 78.9%，而 AUC 保持在 0.9936 不变。这意味着 τ_low=0.05 真正对应"约 5% 的假脸概率"，而不是一个任意的分数阈值。

### Script (English)

> Let me elaborate on the cascade efficiency and operating point selection, with numbers directly from the ablation study.
>
> The left table compares compute costs across configurations. Stage 1 alone requires 0.54 GFLOPs with FNR=3.12%. Stage 2 alone requires 2.87 GFLOPs but has a higher FNR of 9.43%—this shows Stage 2 is not a "better" model but rather a specialist for boundary samples. Our cascade requires only 0.59 GFLOPs—because only 1.16% of samples escalate to Stage 2—while achieving FNR=0.60%. The formula is `Cascade_GFLOPs = 0.54 + r × 2.87` where r is the Stage-2 rate. This means we achieve a 5.2× FNR reduction with only 9.3% compute overhead.
>
> For quantization, we implement INT8 dynamic quantization in `src/stage4/optimize_for_mobile.py`. Stage 1 compresses from 37.3MB to 10.1MB, Stage 2 from 49.6MB to 13.4MB—both 73% reduction. AUC loss is only 0.003 and 0.007 respectively, well within acceptable bounds.
>
> The right side shows threshold configurations for different deployment scenarios. These operating points are found via grid search over (τ_low, τ_high) using the `sweep_grid` function in `src/tools/robustness_threshold_sweep.py`. Safety-first uses τ_low=0.05, τ_high=0.55 for FNR=0.60%, suitable for content moderation. Balanced uses τ_low=0.10, τ_high=0.50 for FNR=1.2%, suitable for general mobile deployment. Speed-first uses τ_low=0.15, τ_high=0.45 for FNR=2.0%, suitable for latency-critical scenarios.
>
> Finally, probability calibration is what makes these thresholds meaningful. In `src/stage1/calibrate_model.py`, the `TemperatureScalingCalibrator` class fits optimal temperature T=1.34 by minimizing NLL. After calibration, ECE drops from 4.21% to 0.89% (78.9% reduction) while AUC remains at 0.9936. This means τ_low=0.05 truly corresponds to "about 5% fake probability" rather than an arbitrary score threshold.

---

## Slide 6: 级联决策逻辑

### 内容要点

- **Stage1输出**: 假脸概率 `p1(x) = P(fake | x)`
- **双阈值机制**:
  - 低阈值 `τ_low = 0.05` → 控制漏报（假脸被判为真）
  - 高阈值 `τ_high = 0.55` → 控制误报和Stage2调用频率
- **决策规则（分段函数）**:
  - 若 `p1(x) < τ_low` → 直接判为真实
  - 若 `p1(x) > τ_high` → 直接判为伪造
  - 若 `τ_low ≤ p1(x) ≤ τ_high` → 升级到Stage2
- **Stage2决策**: `p2(x) > 0.5` 判假，否则判真
- **效果**: FNR ≈ 0.6%，升级率 ~1.2%

### 数学公式

```
ŷ(x) = { real,      if p₁(x) < τ_low
       { fake,      if p₁(x) > τ_high
       { Stage2(x), if τ_low ≤ p₁(x) ≤ τ_high
```

### 视觉元素

- **置信度-区间图**:
  - 横轴为 `p1(x)` 从0到1
  - 在0.05和0.55位置画两条竖线
  - 左区间着色标注"直接判真"，右区间着色标注"直接判假"，中间区间标注"升级到Stage2"
- 图旁边用小箭头表示Stage2再做二分类

### 讲稿

> 接下来这一页，我具体说明一下级联系统中的决策逻辑。
>
> 在第一个阶段，MobileNetV4会对每一张图像输出一个概率p1(x)，代表它认为该图像是伪造人脸的置信度。为了在移动端精细地平衡误报和漏报，我采用了一个双阈值的机制。
>
> 具体来说，我们设置了一个较低的阈值τ_low=0.05，以及一个较高的阈值τ_high=0.55。如果p1(x)非常小，小于0.05，我们认为模型对"真实"这一判断非常有信心，就直接把该样本判为真实；反过来，如果p1(x)大于0.55，则说明模型对"伪造"相对有把握，可以直接判为假脸。
>
> 真正有不确定性的，是落在中间区间[0.05, 0.55]的样本。这些样本很可能是压缩严重、光照复杂或者伪造质量较高的情况。对于这部分样本，我会将它们自动升级到Stage2，由EfficientNetV2-B3再进行一次更加精细的判断。
>
> 在实际验证中，通过这种双阈值机制和级联结构，我在整体FNR仅约0.6%的前提下，把需要升级到Stage2的样本比例控制在大约1.2%。

---

## Slide 7: 六阶段流水线

### 内容要点

| 阶段 | 名称 | 关键技术点 |
|------|------|-----------|
| 0 | 数据与预处理 | 多后端人脸检测、256×256裁剪、manifest生成 |
| 1 | Stage1训练 | MobileNetV4、BCE loss、temperature校准 |
| 2 | Stage2训练 | EfficientNetV2-B3、focal loss、强增强 |
| 3 | 元模型(可选) | LightGBM on embeddings（仅分析用，不部署） |
| 4 | 阈值调优 | 网格搜索τ_low/τ_high、约束FNR与升级率 |
| 5 | 鲁棒性评估 | JPEG、噪声、模糊、亮度、gamma扫描 |
| 6 | 移动导出 | TorchScript INT8 + FP32 ONNX、180ms/92.8% |

### 视觉元素

- **流水线时间轴**: 七个连续方块从左到右
  - Data & Preprocess → Stage1 Training → Stage2 Training → Meta-Model (opt.) → Threshold Tuning → Robustness → Mobile Export
- 每个方块下用一句话标注关键动作
- Stage 3 标注 "(optional)"
- 最后一个方块旁边标注关键数字: 180ms / 92.8%

### 讲稿

> 这一页我用一条流水线来概括整个方法从训练到落地的过程，总共分成七个阶段。
>
> 第零阶段是数据与预处理。我们使用多后端人脸检测器（InsightFace、MediaPipe、YOLOv8等）对视频帧进行人脸检测，把人脸区域裁剪到256×256，并生成统一的训练/验证/测试manifest文件。注意数据增强是在训练时才做的，不是预处理阶段。
>
> 第一阶段是Stage1的训练。使用MobileNetV4和标准的BCE loss进行训练，训练完成后进行temperature scaling校准。高召回率是通过后续的阈值调优实现的，而不是Stage1的损失函数。
>
> 第二阶段是Stage2的训练。使用EfficientNetV2-B3，配合focal loss和更强的数据增强（RandAugment、Mixup、CutMix）。Hard-example mining是可选的，默认配置使用均匀采样。
>
> 第三阶段是可选的元模型。我们实现了LightGBM元模型，基于Stage2的embeddings进行训练，但这仅用于研究分析，不在默认的移动端部署中使用。
>
> 第四阶段是阈值调优。在验证集上对τ_low和τ_high进行网格搜索，目标是在FNR约0.6%的前提下，将升级到Stage2的比例控制在约1.2%。
>
> 第五阶段是鲁棒性评估。我们在JPEG压缩、高斯噪声、运动模糊、亮度变化、gamma变换等条件下测试级联系统。
>
> 最后是移动导出阶段。我们提供TorchScript INT8量化版本（10.1MB/13.4MB）和FP32 ONNX版本（37.5MB/52MB）。当前Android应用使用FP32 ONNX，在小米13上实测延迟约180毫秒，准确率约92.8%。

---

## Slide 7.5: 移动部署详情

### 内容要点

**导出产物**
| 格式 | Stage 1 | Stage 2 |
|------|---------|---------|
| FP32 TorchScript | 37.3 MB | 49.6 MB |
| INT8 TorchScript | 10.1 MB | 13.4 MB |
| FP32 ONNX | 37.5 MB | 52 MB |

**设备端流水线**
- FaceDetector API → 256×256 裁剪
- ImageNet 归一化
- Stage 1 → 级联路由 → Stage 2
- Temperature scaling on logits

**设备端指标 (小米 13)**
| 指标 | 值 |
|------|-----|
| Accuracy | 92.8% |
| FNR | 2.5% |
| FPR | 12.5% |
| Stage-2 Rate | 7.2% |
| Preprocessing | 3.4 ms |
| Inference | 176 ms |
| **Total Latency** | ~180 ms |

**运行时栈**
- ONNX Runtime Android 1.19.0
- CPUExecutionProvider (ARM NEON)
- NNAPI/GPU delegates: 未来工作

### 视觉元素
- 左侧：导出产物表格 + 设备端流水线
- 右侧：设备端指标表格 + 运行时栈
- 底部：Deployment Note 总结框

### 讲稿 (中文)

> 让我详细介绍移动端部署的技术细节，这些内容直接对应到代码实现。
>
> 左侧展示了不同导出格式的模型大小。在 `scripts/export_mobile_cascade_onnx.py` 中，我们定义了 `DeepfakeClassifier` 包装类，它把 timm 的骨干网络和自定义分类头组合在一起。Stage 1 使用单层分类器（Dropout + Linear），Stage 2 使用两层分类器（Dropout + Linear + ReLU + Dropout + Linear）。FP32 TorchScript 是原始格式，Stage 1 为 37.3 MB，Stage 2 为 49.6 MB。通过 INT8 动态量化，可以压缩到 10.1 MB 和 13.4 MB。当前 Android 应用使用的是 FP32 ONNX 格式，大小约为 37.5 MB 和 52 MB。
>
> 导出过程使用 `src/stage4/mobile_deployment/onnx_exporter.py` 中的 `ONNXExporter` 类，它会自动进行 PyTorch vs ONNX 输出对齐验证，确保数值误差小于 1e-5。导出产物包括 `aware_cascade_stage1.onnx`、`aware_cascade_stage2.onnx`、`cascade_config.json` 和 `aware_cascade_manifest.json`，全部放在 `android/mobile_bundle/` 目录下。
>
> 设备端的处理流程在 `OnnxCascadeEngine.kt` 中实现。首先使用 Android FaceDetector API 检测人脸并裁剪到 256×256，然后通过 `ImagePreprocessor` 进行 ImageNet 标准化（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）。接着运行 Stage 1 推理，根据 `cascade_config.json` 中的阈值（tau_low, tau_high）决定是否升级到 Stage 2。
>
> 右侧是在小米 13（Snapdragon 8 Gen 2）上的实测指标。准确率达到 92.8%，FNR 为 2.5%，FPR 为 12.5%。Stage 2 的调用率约为 7.2%，高于 PC 端的 1.16%，这是因为移动端测试集更具挑战性。延迟方面，预处理约 3.4 毫秒，推理约 176 毫秒，总延迟约 180 毫秒，满足 <200ms 的设计目标。
>
> 运行时使用 ONNX Runtime Android 1.19.0，配置 4 个线程进行推理（`sessionOptions.setIntraOpNumThreads(4)`），使用 CPUExecutionProvider 并利用 ARM NEON 向量化。NNAPI 和 GPU 加速是未来的工作方向。

### Script (English)

> Let me elaborate on the mobile deployment technical details, with direct references to the code implementation.
>
> The left side shows model sizes across export formats. In `scripts/export_mobile_cascade_onnx.py`, we define a `DeepfakeClassifier` wrapper class that combines the timm backbone with a custom classifier head. Stage 1 uses a single-layer classifier (Dropout + Linear), while Stage 2 uses a two-layer classifier (Dropout + Linear + ReLU + Dropout + Linear). FP32 TorchScript is the original format: Stage 1 is 37.3 MB, Stage 2 is 49.6 MB. INT8 dynamic quantization compresses them to 10.1 MB and 13.4 MB. The current Android app uses FP32 ONNX format, approximately 37.5 MB and 52 MB.
>
> The export process uses the `ONNXExporter` class from `src/stage4/mobile_deployment/onnx_exporter.py`, which automatically validates PyTorch vs ONNX output alignment, ensuring numerical error is below 1e-5. Export artifacts include `aware_cascade_stage1.onnx`, `aware_cascade_stage2.onnx`, `cascade_config.json`, and `aware_cascade_manifest.json`, all placed in the `android/mobile_bundle/` directory.
>
> The on-device pipeline is implemented in `OnnxCascadeEngine.kt`. First, the Android FaceDetector API detects and crops faces to 256×256, then `ImagePreprocessor` applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Stage 1 inference runs, and based on thresholds (tau_low, tau_high) from `cascade_config.json`, the system decides whether to escalate to Stage 2.
>
> The right side shows on-device metrics from Xiaomi 13 (Snapdragon 8 Gen 2). Accuracy reaches 92.8%, FNR is 2.5%, FPR is 12.5%. Stage-2 rate is about 7.2%, higher than the PC's 1.16% because the mobile test set is more challenging. For latency, preprocessing takes about 3.4 ms, inference about 176 ms, total latency about 180 ms—meeting the <200ms design target.
>
> The runtime uses ONNX Runtime Android 1.19.0, configured with 4 threads for inference (`sessionOptions.setIntraOpNumThreads(4)`), using CPUExecutionProvider with ARM NEON vectorization. NNAPI and GPU acceleration are future work.

---

## Slide 8: 移动端架构

### 内容要点

**模型压缩与量化**
- FP32 TorchScript: Stage1 37.3MB / Stage2 49.6MB
- INT8 TorchScript: Stage1 10.1MB / Stage2 13.4MB
- FP32 ONNX (Android): Stage1 ~37.5MB / Stage2 ~52MB

**推理框架与导出**
- 训练端使用PyTorch完成训练
- 导出为ONNX格式
- ONNX Runtime Android 1.19.0 (CPU only)
- NNAPI/GPU delegates: 未来工作

**Android集成架构**
| 层 | 功能 |
|----|------|
| 应用层 | Camera预览 + 实时检测UI，支持帧级提示"疑似伪造" |
| 中间层 | 图像预处理（缩放、归一化）、批量打包输入 |
| 推理层 | 加载FP32 ONNX模型，异步执行Stage1/Stage2推理 |
| 结果层 | 输出标签与置信度，UI上高亮风险人脸 |

### 视觉元素

- **手机端分层结构图**:
  - 顶层: Android App（Camera预览 + UI）
  - 中间层: Pre-processing模块（人脸裁剪、归一化）
  - 底层: Inference Engine（ONNX Runtime 1.19.0）+ 两个模型框
- 底层旁边标出: FP32 ONNX、180ms、92.8%
- 用不同颜色区分"应用逻辑层"和"模型推理层"

### 讲稿 (中文)

> 这一页介绍 MobileDeepfake 在移动端的整体架构设计，以及代码层面的实现细节。
>
> 首先是模型压缩与量化。我们提供三种导出格式：FP32 TorchScript 是原始格式，Stage 1 约 37.3MB，Stage 2 约 49.6MB；INT8 TorchScript 通过 `src/stage4/optimize_for_mobile.py` 中的动态量化压缩到 10.1MB 和 13.4MB，压缩率达 73%，AUC 损失分别只有 0.003 和 0.007；当前 Android 应用使用的是 FP32 ONNX 格式，大小约为 37.5MB 和 52MB。
>
> 在推理框架方面，训练阶段使用 PyTorch 完成训练和验证，然后通过 `scripts/export_mobile_cascade_onnx.py` 将模型导出为 ONNX 格式。在手机端，使用 ONNX Runtime Android 1.19.0 进行推理，配置 4 个线程（`setIntraOpNumThreads(4)`），目前只使用 CPUExecutionProvider，内部利用 ARM NEON 向量化。NNAPI 和 GPU 加速是未来的工作方向。
>
> Android 集成方面，核心是 `OnnxCascadeEngine.kt` 类。它在 `initialize()` 方法中加载两个 ONNX 模型，在 `detect()` 方法中实现级联逻辑。具体来说，`detect()` 方法首先调用 `ImagePreprocessor.preprocess()` 将 Bitmap 转换为归一化的 FloatArray，然后调用 `runStage1()` 获取 Stage 1 概率。根据 `CascadeConfig` 中的阈值，如果 `stage1Prob < tauLow` 则直接判为真实，如果 `stage1Prob > tauHigh` 则直接判为伪造，否则调用 `runStage2()` 进行二次判断。
>
> 整体采用分层架构：最上层是应用层（Camera 预览 + 检测 UI），中间层是预处理模块（人脸检测、裁剪、归一化），底层是推理层（ONNX Runtime + 两个模型）。决策结果通过 `CascadeResult` 数据类返回，包含预测标签、置信度、决策阶段和时间信息。从实测结果看，单帧总延迟约为 180 毫秒，整体检测准确率约为 92.8%。

### Script (English)

> This slide covers the overall mobile architecture design of MobileDeepfake, with code-level implementation details.
>
> First, model compression and quantization. We provide three export formats: FP32 TorchScript is the original format, Stage 1 is about 37.3MB, Stage 2 is about 49.6MB; INT8 TorchScript via dynamic quantization in `src/stage4/optimize_for_mobile.py` compresses to 10.1MB and 13.4MB—73% reduction with AUC loss of only 0.003 and 0.007 respectively; the current Android app uses FP32 ONNX format, approximately 37.5MB and 52MB.
>
> For the inference framework, training uses PyTorch, then `scripts/export_mobile_cascade_onnx.py` exports models to ONNX format. On device, ONNX Runtime Android 1.19.0 runs inference with 4 threads (`setIntraOpNumThreads(4)`), currently using only CPUExecutionProvider with ARM NEON vectorization. NNAPI and GPU acceleration are future work.
>
> For Android integration, the core is the `OnnxCascadeEngine.kt` class. Its `initialize()` method loads both ONNX models, and `detect()` implements the cascade logic. Specifically, `detect()` first calls `ImagePreprocessor.preprocess()` to convert Bitmap to normalized FloatArray, then calls `runStage1()` to get Stage 1 probability. Based on thresholds in `CascadeConfig`, if `stage1Prob < tauLow` it classifies as real, if `stage1Prob > tauHigh` it classifies as fake, otherwise it calls `runStage2()` for the final decision.
>
> The overall architecture is layered: Application Layer (Camera preview + Detection UI), Middle Layer (face detection, cropping, normalization), and Inference Layer (ONNX Runtime + two models). Results are returned via the `CascadeResult` data class, containing prediction label, confidence, decision stage, and timing info. On-device testing shows total latency of about 180ms with 92.8% accuracy.

---

## Slide 9: 数据集介绍

### 内容要点

**数据集构成（训练: 2.66M + OOD: 0.85M 人脸裁剪）**
| 数据集 | Train | Val | Test | 特点 |
|--------|-------|-----|------|------|
| CelebDF-v2 | 83,599 | 17,478 | 18,037 | 高质量名人换脸，光照/表情丰富 |
| FaceForensics++ | 223,653 | 50,945 | 47,430 | 多种操纵方法（DeepFakes, FaceSwap等） |
| DFDC | 721,946 | 154,919 | 154,511 | Facebook大规模多样化数据集 |
| DeeperForensics-1.0 | 844,396 | 165,854 | 172,894 | 真实拍摄扰动（压缩、传输噪声等） |

**OOD评估数据集: Deepfake-Eval-2024**
- 451k val / 402k test
- 来自88个网站、52种语言
- 2024年真实媒体环境采集
- **仅用于评估，不参与训练或微调**

**设计目标**: 覆盖多种伪造方法、拍摄条件与分布，以评估泛化能力

### 视觉元素

- **左侧**: 表格，列出4个训练数据集的规模与特点
- **右侧**: OOD评估块，标示Deepfake-Eval-2024的关键特征
- 用高亮表示"88网站""52语言"等关键数字

### 讲稿

> 这一部分先介绍一下本工作的数据集构成。整体上，我们使用了约266万张人脸裁剪图像用于训练，另外85万张用于分布外评估，总计约350万张。
>
> 如左图所示，训练阶段我们主要使用CelebDF-v2、FaceForensics++、DFDC和DeeperForensics-1.0。CelebDF-v2主要是高质量的名人换脸视频；FaceForensics++包含多种操纵方法；DFDC是Facebook发布的大规模多样化数据集；DeeperForensics-1.0则特别强调真实的拍摄和传输扰动。
>
> 在评估阶段，我们引入了Deepfake-Eval-2024数据集作为分布外测试集。它来自88个网站、52种语言，是2024年真实媒体环境中采集的内容。需要强调的是，这个数据集**仅用于评估**，我们不在上面做任何训练或微调。

---

## Slide 9.5: 数据集预处理与分割

### 内容要点

**统一预处理流程**
- 256×256 PNG人脸裁剪
- 每个视频最多50张人脸
- 帧间隔: 10
- 多后端人脸检测器

**分割协议**
- 固定比例: 70% / 15% / 15%
- 保留官方测试子集（如FF++）
- 确定性hash-based分割，确保可复现

**平衡Manifests**
- 每个数据集real/fake数量大致相等
- 防止大数据集（如DFDC）主导训练
- 每个分割都保持平衡

**类别分布**
| 数据集 | Real | Fake | 比例 |
|--------|------|------|------|
| CelebDF-v2 | 59K | 60K | 0.98 |
| FF++ | 161K | 161K | 1.00 |
| DFDC | 516K | 515K | 1.00 |
| DeeperForensics | 591K | 592K | 1.00 |
| Eval-2024 | 519K | 336K | 1.55 |

**评估模式**
1. Combined validation（所有4个数据集联合）
2. Per-dataset validation（分别评估）
3. OOD evaluation（Deepfake-Eval-2024）

### 视觉元素

- **左侧**: 预处理流程、分割协议、平衡Manifests三个要点列表
- **右侧**: 类别分布表格 + 评估模式列表
- **底部**: Key Point块强调"按类别和数据集平衡"

### 讲稿

> 让我详细介绍数据集的预处理和分割协议。
>
> 首先是统一的预处理流程。所有数据集都被转换为256×256的PNG人脸裁剪图像，每个视频最多提取50张人脸，帧间隔为10。我们使用多后端人脸检测器来确保覆盖不同质量的视频。
>
> 分割协议方面，我们使用固定的70/15/15比例来划分训练/验证/测试集。对于有官方测试集的数据集（如FF++），我们保留官方的测试子集。对于其他数据集，我们使用确定性的hash-based分割来确保可复现性。
>
> 平衡Manifests是一个关键设计。我们确保每个数据集中real和fake样本数量大致相等，这样在多数据集联合训练时，不会被某个大数据集（如DFDC或DeeperForensics）主导。
>
> 右侧的表格展示了各数据集的类别分布。可以看到，四个训练数据集的real/fake比例都接近1:1。而Deepfake-Eval-2024中Real样本多于Fake样本（比例约1.55:1），反映了该基准中真实内容占多数的特点。
>
> 我们的评估分为三种模式：Combined validation在所有四个数据集上联合评估；Per-dataset validation分别在每个数据集上评估；OOD evaluation在Deepfake-Eval-2024上评估跨域泛化能力。

---

## Slide 10: 跨数据集评估协议

### 内容要点

**训练与验证**
- 仅在4个公开数据集上完成
- 训练集: CelebDF-v2 + FF++ + DFDC + DeeperForensics-1.0
- 可复现分割（保留官方测试子集，其他使用70/15/15）
- 跨数据集平衡Manifests

**跨数据集评估**
- 不在Deepfake-Eval-2024上进行任何微调或再训练
- 直接将in-domain训练好的模型应用于Deepfake-Eval-2024

**评估指标**
| 场景 | 指标 |
|------|------|
| In-domain | AUC、级联FNR、升级率 |
| OOD | F1、FNR、Stage2升级率 |

*Headline metrics; 完整表格还包括Acc, Precision/Recall, FPR*

**目标**: 模拟"训练在公开数据集 → 部署到真实互联网环境"的现实流程

### 视觉元素

- **流程图**:
  - 左边四个方块(4个训练集) → 箭头指向"训练好的级联模型"
  - 模型分两条箭头:
    - 一条指向"In-domain测试"
    - 一条指向"OOD测试(Deepfake-Eval-2024)"
- 在OOD分支上标注"无微调/无额外训练"

### 讲稿

> 这一页说明我们是如何做跨数据集评估的。左边四个方块是刚刚介绍的四个训练数据集，我们只在这些公开数据上完成模型的训练和验证，包括所有的超参数选择和模型结构设计。
>
> 需要强调的是，我们使用可复现的分割协议：对于有官方测试集的数据集保留官方划分，其他使用70/15/15比例。同时，我们使用跨数据集平衡的Manifests来确保训练的公平性。
>
> 完成训练之后，如中间的流程图所示，我们得到一个两阶段级联的深度伪造检测模型。然后我们做了两种评估：一是在原始四个数据集上做in-domain测试；二是完全不做微调，直接把这个模型应用到Deepfake-Eval-2024上。
>
> 表格中列出的是我们的headline metrics。In-domain主要看AUC、级联FNR和升级率；OOD主要看F1、FNR和Stage2升级率。完整的评估表格还包括Accuracy、Precision/Recall和FPR等指标。
>
> 这样的设置其实模拟了一种现实流程：模型在实验室里用公开数据训练好之后，直接部署到真实互联网环境中，看性能会发生什么变化。

---

## Slide 11: In-domain结果

### 内容要点

- **Stage1**: AUC = 0.9936, F1 = 0.9561
  - 高召回、低误拒真目标
- **Stage2**: AUC = 0.9633, F1 = 0.8930
  - 更精细的特征建模，修正Stage1边界样本
- **级联整体表现 (Safety-first配置)**:
  - Cascade AUC = 0.9941, F1 = 0.9654
  - 最终FNR = 0.60%（漏检率极低）
  - 升级率 = 1.16%（只有少量样本需要进入Stage2）
- **结论**: 在Combined Validation上，级联结构既保证了高检测能力，又极大节省计算

### 视觉元素

- **左图**: Stage1的ROC曲线，标注AUC数值
- **右侧**: 列表展示各阶段和级联的指标
- 标注"FNR 0.60%"、"升级率 1.16%"等关键数字

### 讲稿

> 这个页面展示的是在Combined Validation，也就是四个训练数据集联合验证集上的结果。左侧是Stage1的ROC曲线，可以看到Stage1的AUC达到0.9936，F1为0.9561；Stage2的AUC为0.9633，F1为0.8930。
>
> 在设计上，Stage1主要目标是高召回、快速筛查，它用一个相对轻量的模型捕捉大部分伪造样本；Stage2则更精细，用来处理边界样本。
>
> 整体级联之后，使用Safety-first配置（阈值0.05/0.55），级联的AUC达到0.9941，F1为0.9654。最终的漏检率FNR只有0.60%，而需要进入第二阶段的升级率只有1.16%。这意味着在公开基准数据集上，我们既能保持高精度的检测能力，又只对极少数难样本使用较重的计算。

---

## Slide 11.5: Stage-wise消融分析

### 内容要点

**为什么需要级联？**
- 单独Stage1: 速度快但FNR较高 (3.12%)
- 单独Stage2: FNR更高 (9.43%)，且计算量是Stage1的5倍
- 级联: 两全其美

**Stage-wise对比 (Combined Validation)**
| 配置 | AUC | F1 | FNR | GFLOPs |
|------|-----|-----|-----|--------|
| Stage 1 only | 0.9936 | 0.9561 | 3.12% | 0.54 |
| Stage 2 only | 0.9633 | 0.8930 | 9.43% | 2.87 |
| **Cascade** | **0.9941** | **0.9580** | **0.60%** | **0.59** |

**计算效率**
- Stage 1: 0.54 GFLOPs
- Stage 2: 2.87 GFLOPs (5.3×)
- Cascade平均: 0.59 GFLOPs
- 额外开销: 仅比Stage 1多9%

### 视觉元素

- **左侧**: Stage-wise对比表格
- **右侧**: FNR柱状图对比 (S1 vs S2 vs Cascade)
- **底部**: Key Insight块

### 讲稿

> 这一页展示为什么我们需要级联架构，而不是单独使用Stage 1或Stage 2。
>
> 首先看左侧的对比表格。如果只使用Stage 1，虽然速度快（0.54 GFLOPs），但FNR达到3.12%，意味着每100个深度伪造样本会漏掉约3个。如果只使用Stage 2，FNR更高，达到9.43%，而且计算量是Stage 1的5倍多。
>
> 这里有一个看似反直觉的结果：Stage 2单独使用时FNR比Stage 1更高。这并不是说Stage 2更差，而是说明两者的错误模式不同——Stage 1在大部分易分样本上做得很好，但在一些细节纹理或复杂操控上会出错；而Stage 2在这些难例上更有优势。正是这种**互补性**，使得级联能够显著降低系统级FNR。
>
> 我们的级联架构实现了两全其美：FNR降到0.60%，比单独使用Stage 1低了5倍；同时平均计算量只有0.59 GFLOPs，仅比Stage 1多9%。
>
> 右侧的柱状图直观展示了FNR的对比。关键在于：通过双阈值机制，我们让Stage 1处理大部分"容易"的样本，只把约1.2%的"困难"样本送到Stage 2。这样既保证了低漏检率，又控制了计算开销。

---

## Slide 12: 级联效率分析

### 内容要点

**设计思路**
- Stage1: 轻量、高吞吐、以"宁可多抓、不要漏掉"为原则
- Stage2: 相对较重、精细判别，只处理被认为"困难"的样本

**实际运行统计（Combined Validation）**
- 升级率 = 1.16% → 约98.8%样本只经过Stage1
- 在保持FNR = 0.60%的前提下，整体计算量显著降低

**对移动端/边缘侧的重要性**
- 大部分样本可以只在本地快速完成检测
- 极少数可选择上传云端或调用更强算力进行复核

### 视觉元素

- **漏斗图**: 输入样本100% → Stage1接收100% → Stage2只接收1.2% → 最终输出
- **F1 Heatmap**: 展示不同阈值组合下的F1分数

### 讲稿

> 为了更好地适配移动端和大规模部署，我们采用了两阶段级联的结构。Stage1是一个轻量、高吞吐的检测器，它的原则是宁可多抓一点嫌疑样本，也要尽量减少漏检。
>
> Stage2则相对较重，主要针对Stage1判定不确定的难样本做精细分析。右侧漏斗图可以看到，在Combined Validation场景下，只有大约1.16%的样本需要被升级到Stage2，大概98.8%的样本在Stage1就已经完成了判决。
>
> 这样设计的好处是，在保持整体漏检率0.60%的前提下，我们显著降低了平均计算成本。这为后面在移动端、甚至是端云协同的部署，提供了一个比较现实的技术基础。

---

## Slide 13: 移动端性能

### 内容要点

**实验环境**
- 设备: Xiaomi 13
- 芯片: Snapdragon 8 Gen 2

**性能指标**
| 指标 | 值 |
|------|-----|
| 人脸级检测准确率 | 92.8% |
| 单张人脸平均延迟 | ~180ms |

**应用层面**
- 对于典型25-30fps视频，可支持"近实时"抽帧检测
- 端侧完成推理，无需上传原始视频，有利于隐私保护

**结论**: 模型在当前主流旗舰手机上具备可落地的实时性和精度

### 视觉元素

- **左图**: 条形图展示延迟(ms)，标出180ms的位置
- **右侧**: 示意手机界面的截图，展示"检测中"的界面和"真实/伪造"标签效果

### 讲稿

> 这一页展示的是模型在真实手机端的表现。我们选取了一台市面上比较典型的旗舰设备，小米13，搭载Snapdragon 8 Gen 2芯片，进行端侧推理测试。
>
> 实验结果显示，在人脸级检测上，我们的准确率可以达到92.8%；同时，单张人脸的平均推理延迟大约为180毫秒。这个延迟在移动端推理中已经是一个比较可用的水平。
>
> 从应用角度看，如果对视频做抽帧，比如每几帧检测一次，在25到30帧每秒的视频场景下，可以基本实现近实时的检测体验。而且整个推理过程都在端侧完成，无需上传完整视频，有助于保护用户隐私。

---

## Slide 14: 跨数据集挑战

### 内容要点

**在Deepfake-Eval-2024上的性能显著下降**
| 指标 | In-domain | OOD (Deepfake-Eval-2024) |
|------|-----------|--------------------------|
| F1 | >0.95 | <0.30 |
| FNR | ~0.6% | 73-79% |
| FPR | ~3% | 25-27% |
| Stage2升级率 | ~1.2% | ~51% |

**问题揭示**
- 训练数据与2024年真实互联网分布差异巨大
- 新的伪造方法/后处理管线，训练集中几乎没有覆盖
- 社交媒体处理管线带来的领域偏移（重编码、滤镜、平台特定处理等）

**启示**
- 仅依赖传统公开数据集训练的模型，难以直接泛化到最新的真实场景
- 需要更系统的跨域鲁棒性设计和持续更新

### 视觉元素

- **对比表格**: In-domain vs OOD，包含F1/FNR/FPR/Stage2升级率
- 用醒目颜色强调F1大幅下降、FNR和FPR大幅上升

### 讲稿

> 但是，当我们把同样的模型直接应用到Deepfake-Eval-2024上时，情况就完全不一样了。右边的对比表格可以看到，和in-domain相比，性能出现了明显下降。
>
> 具体来说，在这个OOD数据集上，我们的F1值低于0.30，漏检率FNR大约在73%到79%之间，也就是说有大量伪造样本被误判为真实。同时，误报率FPR也从3%上升到25-27%，说明precision和recall都很差。Stage2的升级率达到大约51%，意味着有一半的样本被模型认为是'困难样本'。
>
> 这组结果说明，仅依赖传统公开数据集训练出来的模型，并不能直接适应2024年真实互联网环境。这个结果也从侧面强调了跨数据集泛化和持续更新的重要性。**这是一个负面但有价值的发现**，为后续研究提供了重要的实证基线。

---

## Slide 14.5: OOD深度分析

### 内容要点

**Deepfake-Eval-2024 校准后级联结果**
| 指标 | Val | Test |
|------|-----|------|
| F1 | 0.2996 | 0.2796 |
| Accuracy | 58.0% | 50.0% |
| FNR | 72.9% | 79.0% |
| FPR | 26.7% | 25.1% |
| Stage2 Rate | 51.1% | 52.3% |

**Stage2-Only Baseline (最大计算量)**
| 指标 | Val | Test |
|------|-----|------|
| F1 | 0.4460 | 0.5962 |
| FNR | 18.2% | 14.6% |
| FPR | 91.6% | 87.1% |
| Stage2 Rate | 97.5% | 98.9% |

**关键发现**
- 即使100%使用Stage 2，FPR仍超过87%
- 问题不在于计算量，而在于领域覆盖不足
- 当前训练数据无法覆盖2024年互联网分布

**未来方向**
- 领域泛化和周期性重训练是实际部署的关键

### 视觉元素

- **左上**: 校准后级联在OOD上的详细指标表格
- **左下**: Stage2-Only baseline表格，突出FPR飙升
- **右侧**: Key Finding警示框 + Implications列表

### 讲稿

> 这一页我们深入分析OOD结果，展示为什么跨数据集泛化是一个根本性挑战。
>
> 左上方的表格展示了在Deepfake-Eval-2024上使用校准后级联的结果。可以看到，FNR高达73-79%，意味着大部分深度伪造样本被漏检。同时FPR也有25-27%，说明precision和recall都很差。
>
> 更重要的是下方的Stage2-Only baseline。这个配置把所有样本都送到Stage 2，代表了最大计算量的情况。结果显示，虽然FNR降到了14-18%，但FPR飙升到87-92%。这意味着即使我们愿意付出100%的计算代价，也无法解决OOD问题。
>
> 这个发现的关键启示是：问题不在于级联架构的效率，而在于基础模型的领域覆盖不足。当前的训练数据无法覆盖2024年互联网上的深度伪造分布。因此，未来的工作需要关注领域泛化和持续学习，而不仅仅是模型压缩和效率优化。

---

## Slide 15: 鲁棒性与错误分析

### 内容要点

**图像扰动鲁棒性实验**
| 扰动类型 | 表现 |
|----------|------|
| JPEG压缩 | 中等压缩有时反而提升F1（抑制高频噪声，突出伪造伪影） |
| 高斯噪声 | 中等噪声下表现不稳定 |
| 运动模糊 | **近乎完全失败** (FNR ~96-100%)，鲁棒性仍是开放问题 |

**注意**: 鲁棒性结果是样本有限的初步假设，需要更大规模验证。

**典型错误案例**
- 极高质量伪造: 人眼也难以分辨，模型倾向于判为真实
- 极端压缩/重编码: 人脸纹理完全破坏，模型难以捕捉伪造线索
- 遮挡与非标准拍摄: 口罩、滤镜、强光背光等干扰人脸区域特征

**改进方向**
- 引入更多真实世界扰动的增强策略
- 针对高质量伪造和极端压缩设计专门的特征与损失

### 视觉元素

- **折线图1**: JPEG质量(高→中→低) vs F1分数
- **折线图2**: Motion Blur vs F1，显示近乎完全失败
- **示例图**: 高质量伪造、极度压缩、遮挡严重的失败样本截图
- **注释**: 标注结果为初步假设

### 讲稿

> 最后这一页，我们简单总结一下鲁棒性实验和错误分析。我们主要从图像扰动和典型失败案例两个角度来观察模型的行为。需要先说明的是，这些鲁棒性结果是样本有限的初步假设，需要更大规模的验证。
>
> 在图像扰动方面，如左图所示，当我们对图像做JPEG压缩时，发现中等程度的压缩反而可以在部分场景下提升F1。一个可能的原因是，适度压缩会抑制一些随机高频噪声，让伪造边界和不自然的纹理更突出。
>
> 对于运动模糊，这是一个比较严重的问题。我们的实验显示，在所有测试的模糊核大小下，FNR都在96%到100%之间，这意味着模型在运动模糊场景下几乎完全失效。这说明运动模糊鲁棒性仍然是一个开放问题，需要在未来工作中重点关注。
>
> 在错误案例方面，我们观察到三类比较典型的失败样本：第一类是极高质量的伪造，人眼也很难区分真伪；第二类是极端压缩导致的人脸纹理严重破坏；第三类是遮挡和非标准拍摄，比如口罩、各种滤镜等。这些分析为后续工作提供了改进方向。

---

## Slide 16: APP界面演示

### 内容要点

**简单三步流程**
1. 导入视频/图片（本地文件，不上传云端）
2. 本地检测
3. 输出结果

**功能特点**
- 显示整体真假概率 + 关键帧热力图提示
- 显示两阶段结果: Stage1快速筛查，Stage2精细判别
- 平均延迟约180ms（Xiaomi 13, Snapdragon 8 Gen 2）

### 视觉元素

- **左侧**: 手机界面截图或原型图（主界面：选择视频按钮 + 最近检测记录）
- **右侧上方**: 检测结果界面（大号"真实/疑似伪造"标签 + 概率条形图）
- **右侧下方**: 若干视频帧缩略图，某几帧叠加红色热力图或"高风险"标记
- **底部**: 标注"本地推理 / 平均延迟 ~180ms"

### 讲稿

> 这一页我简单展示一下我们在移动端的APP界面和整体使用流程。
>
> 整个使用过程非常简单，可以概括为三步：第一步，用户从本地相册中选择一段人脸视频或图片；第二步，点击"本地检测"按钮，APP会在手机上完成全部推理；第三步，几百毫秒内返回一个"真实/疑似伪造"的判断结果。
>
> 在结果展示上，我们不仅给出一个整体的真假概率，还会展示若干关键帧，并在高风险区域叠加热力图或者风险标记，帮助用户理解模型是"看到了什么"才做出这样的判断。
>
> 模型推理在本地设备上完成，不依赖云端服务器，也不会上传原始视频，有利于保护用户隐私。

---

## Slide 17: 工程实现要点

### 内容要点

**模型结构**
| 阶段 | 模型 | FP32 TS | FP32 ONNX |
|------|------|---------|-----------|
| Stage1 | MobileNetV4 | 37.3MB | ~37.5MB |
| Stage2 | EfficientNetV2-B3 | 49.6MB | ~52MB |

**注意**: INT8 TorchScript 量化后大小为 10.1MB / 13.4MB，但当前 Android 部署使用 FP32 ONNX。

**优化与部署**
- 模型优化: INT8后训练量化 + ONNX导出
- 推理框架: ONNX Runtime Android 1.19.0
- 前后处理: 人脸检测/对齐在本地实现，多线程流水线 (4线程)
- 资源管理: 目前以CPU推理为基线，GPU/NNAPI作为后续扩展方向

**工程特点**
- 端到端可复现
- 半自动化导出流水线 (导出自动化，复制到Android assets手动)

### 视觉元素

- **系统架构图**: 输入→本地人脸检测/裁剪→Stage1→决定是否进入Stage2→Stage2→输出→APP UI
- 对比"训练端/部署端"两列，展示脚本和移动端代码模块
- 角落标注模型大小和量化方式的小表格

### 讲稿

> 这一页主要介绍移动端部署在工程上的一些关键实现。
>
> 在模型设计上，我们采用两阶段级联结构：Stage1使用轻量的MobileNetV4，FP32 TorchScript约37.3MB；Stage2使用更强的EfficientNetV2-B3，FP32约49.6MB。导出为FP32 ONNX之后，Stage1大约37.5MB、Stage2约52MB。我们也评估了INT8 TorchScript量化，可以将模型压缩到10.1MB和13.4MB，但当前Android部署使用FP32 ONNX以保证兼容性。
>
> 为了适应移动端资源限制，我们统一导出为ONNX格式；推理框架选用了ONNX Runtime Android 1.19.0，配置4线程并行推理。在推理部署上，目前在Xiaomi 13上主要采用CPU基线实现；GPU和NNAPI的进一步加速留作后续工程优化方向。
>
> 在前后处理方面，我们将人脸检测、裁剪和简单对齐也放在本地实现，并与推理线程进行流水线并行。整个工程从训练脚本到ONNX导出是自动化的，但复制到Android assets仍需手动操作。

---

## Slide 17.5: PC vs Mobile 部署对比

### 内容要点

**性能对比**
| 指标 | PC | Mobile |
|------|-----|--------|
| Accuracy | 97% | 92.8% |
| FNR | 0.6% | 2.5% |
| FPR | 3% | 12.5% |
| Stage2 Rate | 1.2% | 7.2% |
| Latency | -- | ~180 ms |

**模型大小对比**
| Stage | FP32 TS | INT8 TS | FP32 ONNX |
|-------|---------|---------|-----------|
| Stage 1 | 37.3 MB | **10.1 MB** | 37.5 MB |
| Stage 2 | 49.6 MB | **13.4 MB** | 52 MB |

**延迟分解 (Xiaomi 13)**
- 预处理: ~3 ms
- ONNX 推理: ~176 ms
- 总延迟: ~180 ms

**Calibration 参数**
- Temperature: T ≈ 1.34
- ECE 降低 79%
- 阈值: τ_low=0.05, τ_high=0.55

### 视觉元素

- **左上**: PC vs Mobile 性能对比表格
- **左下**: 模型大小对比表格，突出 INT8 压缩效果
- **右上**: 延迟分解堆叠条形图
- **右下**: Calibration 参数列表 + Key Insight 框

### 讲稿

> 这一页对比了 PC 和 Mobile 部署的性能差异。
>
> 左上方的表格展示了关键指标的对比。可以看到，从 PC 到 Mobile，准确率从 97% 下降到 92.8%，FNR 从 0.6% 上升到 2.5%，FPR 从 3% 上升到 12.5%。Stage2 升级率也从 1.2% 上升到 7.2%。
>
> 这个性能下降主要来自两个因素：一是移动端使用的是 FP32 ONNX 模型而非 INT8 量化模型；二是移动端的评估数据集 mobile_eval 与 PC 端的 combined validation 有所不同。
>
> 左下方的表格展示了模型大小的对比。我们评估了 INT8 TorchScript 量化，可以将模型大小压缩到 10.1 MB 和 13.4 MB，但当前 Android 部署使用的是 FP32 ONNX 以保证兼容性。
>
> 右侧展示了延迟分解。在 Xiaomi 13 上，预处理只需要约 3 毫秒，主要时间花在 ONNX 推理上，约 176 毫秒，总延迟约 180 毫秒。
>
> 总的来说，移动端部署用约 4% 的准确率换取了端侧隐私保护和低于 200 毫秒的延迟，这是一个合理的 trade-off。

---

## Slide 18: 工作总结

### 内容要点

**四大贡献**

1. **两阶段级联检测系统**
   - FNR ≈ 0.6%，升级率 ≈ 1.2%

2. **多数据集训练与跨数据集评估**
   - 揭示分布偏移挑战

3. **端到端6阶段流水线**
   - 从预处理到移动导出全流程可复现

4. **移动端部署**
   - Xiaomi 13上达到92.8%准确率、~180ms延迟

**关键数字（分场景汇总）**
| 场景 | 指标 | 值 |
|------|------|-----|
| PC In-domain | FNR | ~0.6% |
| PC In-domain | Stage2 Rate | ~1.2% |
| Mobile (Xiaomi 13) | Accuracy | **92.8%** |
| Mobile (Xiaomi 13) | Latency | ~180ms |
| OOD (Eval-2024) | F1 | <0.30 |

### 视觉元素

- **总览示意图**: 6阶段流水线排成一条线，最后连到"Mobile App"
- **分场景数字汇总表**: 区分PC In-domain、Mobile、OOD三种场景的关键指标
- 用图标表示"模型""数据集""工程""部署"四个维度的成果

### 讲稿

> 综合来看，本课题的工作可以概括为四个方面。
>
> 第一，我们设计并实现了一套两阶段级联的深度伪造检测系统。通过在Stage1做快速筛查，在Stage2做精细判别，在保证整体效率的同时，将假阴性率控制在大约0.6%，升级率约为1.2%。
>
> 第二，我们统一多个公开数据集进行联合训练和跨数据集测试，系统地评估了在分布偏移条件下的检测性能。实验表明分布偏移仍然是深度伪造检测走向实际应用的核心挑战之一。
>
> 第三，我们搭建了一条端到端、可复现的6阶段流水线，每个阶段都可以通过统一脚本自动执行，方便后续复现和扩展。
>
> 第四，我们完成了在移动端的实际部署，在Xiaomi 13上达到了92.8%的检测准确率和约180ms的推理延迟，表明该方案在真实设备上具备一定的实用性。
>
> 右边的表格将关键指标按场景分开列出：PC端In-domain测试的FNR和Stage2升级率；移动端在Xiaomi 13上的准确率和延迟；以及跨数据集OOD测试的F1分数。这样可以更清楚地看到不同评估场景下系统的表现。

---

## Slide 19: 局限与展望

### 内容要点

**当前局限**
| 局限 | 描述 |
|------|------|
| 分布偏移 | 跨数据集F1<0.30，泛化能力不足 |
| 对抗攻击 | 尚未系统评估对抗攻击鲁棒性 |
| 设备覆盖 | 仅在单一设备（Xiaomi 13）上测试 |
| 时序建模 | 帧级检测，未做视频级时序融合 |

**未来工作方向**
- 引入领域自适应/领域泛化方法，缓解分布偏移
- 结合时序建模与视频级决策（RNN、Transformer或时序池化）
- 设计针对深度伪造的对抗训练与防御机制
- 扩展到更多设备和平台（更多SoC、iOS等）

### 视觉元素

- **左侧**: 两列表格，"现有局限" vs "改进方向"
- **右侧**: 时间轴箭头，从"研究原型"指向"工程化产品"
- 标注关键词: Domain Adaptation、Video-level Fusion、Adversarial Defense、多平台部署
- 局限用灰色图标，展望用蓝色/绿色图标形成对比

### 讲稿

> 虽然我们已经在准确率和移动部署方面取得了一定进展，但整个系统仍然存在一些明显的局限。
>
> 首先，在跨数据集场景下，F1分数仍然低于0.30，这说明当训练集和实际应用场景存在较大分布差异时，当前模型的泛化能力仍然不足。分布偏移可以说是深度伪造检测走向实际落地时面临的核心挑战之一。
>
> 其次，目前工作尚未系统评估在对抗攻击下的鲁棒性。第三，我们只在Xiaomi 13这一类设备上做了详细测试。第四，目前检测仍然基于帧级预测，没有充分利用视频的时序信息。
>
> 面向未来，我们计划从几个方向继续推进：一是引入领域自适应和领域泛化技术；二是结合时序建模，在视频级别做融合决策；三是设计面向深度伪造场景的对抗训练和防御策略；四是将系统扩展到更多设备和平台。

---

## Slide 20: 致谢与Q&A

### 内容要点

**致谢**
- 致谢导师与评委老师的指导与建议
- 致谢课题组同学在数据标注、工程实现方面的支持
- 致谢公开数据集和开源框架（如ONNX Runtime等）

**再次强调**
- 本工作为深度伪造检测落地探索提供基础平台

**Q&A**
- 欢迎各位老师批评指正

### 视觉元素

- **居中大字**: "谢谢！"或"谢谢聆听"
- **下方小字**: 指导老师、合作同学、课题组/实验室名称
- **右下角**: "Q&A"或问号图标突出提问环节
- 背景简洁、正式

### 讲稿

> 我的报告到这里基本就结束了。
>
> 首先，我要衷心感谢我的导师在整个研究过程中给予的悉心指导和耐心帮助，从问题选题、方案设计到实验分析，都提供了非常重要的建议。
>
> 同时也感谢各位评委老师在百忙之中参加我的答辩，并在前期对论文提出了很多宝贵意见，让这项工作更加完整和严谨。
>
> 另外，也非常感谢课题组的同学在数据标注、系统实现和实验调试方面给予的支持，以及各个公开数据集和开源框架的贡献者，为本课题的研究提供了坚实的基础。
>
> 总体来说，这项工作只是深度伪造检测走向实际应用的一次初步探索，仍然有很多不足，也希望在后续研究中能够不断完善。
>
> 最后，感谢各位老师和同学的聆听，下面欢迎大家提出问题和建议。

---

# 第四部分：Q&A 准备

## 10个高频问题与答案要点

### Q1: 为什么选择两阶段级联设计，而不是一个更大的单模型？

**答案要点:**
- 单大模型在移动端延迟、功耗和内存占用上都不友好
- 观察到大部分样本是"容易判"的，用轻量模型快速处理
- 实验表明：在几乎不损失FNR的前提下，把大模型调用比例压到约1.2%，显著降低平均延迟

### Q2: 双阈值是如何确定的？有没有系统的选择方法？

**答案要点:**
- 在验证集上基于ROC曲线进行网格搜索
- 优先固定FNR上限，再在可接受FPR区间内选取升级率最低的一对阈值
- 也对比过单阈值和其他配置，当前双阈值方案在精度-速度折中最优

### Q3: 跨数据集F1很低，这说明你的方法泛化性不好吗？

**答案要点:**
- 一部分体现了方法的不足，但更重要的是暴露了深度伪造检测任务本身的困难
- 不同数据集在伪造方法、压缩程度、拍摄设备等方面差异很大，导致严重分布偏移
- 这是一个"负结果但有价值的发现"，为后续研究提供基线

### Q4: 为什么移动端准确率只有92.8%，和服务器端结果有多大差距？

**答案要点:**
- 服务器端未量化模型准确率略高
- 移动端部署需要量化和模型剪枝，会带来一定精度损失
- 在180ms延迟和设备资源限制下，这是合理的折中

### Q5: 180ms的延迟是否真的足够"实时"？

**答案要点:**
- 对拍照场景来说，<200ms的实时反馈基本不影响用户体验
- 对比了主流应用（相机滤镜、实时美颜）的延迟水平，是可接受的
- 可通过异步显示、管线优化进一步掩盖延迟

### Q6: 数据集是否存在偏差？会不会影响公平性？

**答案要点:**
- 公开数据集确实存在地域、肤色、拍摄环境等方面的偏差
- 本工作聚焦在"是否被伪造"的检测准确性上，尚未系统分析群体公平性
- 后续工作可以引入更多多样化数据，专门评估不同人群上的检测性能

### Q7: 如何防御更高级的对抗攻击？

**答案要点:**
- 当前工作主要针对常规Deepfake和常见扰动场景
- 理论上存在对抗攻击风险，是深度学习安全领域的共性问题
- 后续可结合对抗训练、随机化防御等手段提升鲁棒性

### Q8: 你的系统是针对图片还是视频？

**答案要点:**
- 目前主流程是基于静态图（帧），以每帧特征为基础做判断
- 视频可通过对多帧结果做时序融合（如majority voting）
- 出于移动端实时性考虑，先实现帧级方案，视频级融合作为后续扩展

### Q9: 端到端6阶段流水线中，哪一部分是瓶颈？

**答案要点:**
- 训练阶段瓶颈：数据准备和数据I/O
- 部署阶段瓶颈：模型推理和图像预处理
- 通过多进程数据加载、缓存策略、TFLite推理引擎等优化

### Q10: 你认为自己工作中最具创新性的地方是什么？

**答案要点:**
- 在移动端场景下系统性地设计并实现了两阶段级联检测体系
- 提供了完整的端到端流水线和真实设备上的APP
- 系统性评估并揭示了跨数据集性能严重下降的问题，为未来研究提供实证基础

---

# 第五部分：可用图表

## 已有图表 (paper/figures/)

| 图表 | 文件名 | 用于幻灯片 |
|------|--------|-----------|
| Stage1 ROC曲线 | `outputs_stage1_run_20251023_034316_01_roc_curve.png` | Slide 11 |
| Stage1 校准图 | `outputs_stage1_run_20251023_034316_03_calibration.png` | Slide 11 |
| Stage1 混淆矩阵 | `outputs_stage1_run_20251023_034316_confusion_matrix.png` | Slide 11 |
| 级联最佳配置 | `outputs_stage4_run_20251031_075851_best_config_metrics.png` | Slide 12 |
| F1热力图 | `outputs_stage4_run_20251031_075851_heatmap_f1.png` | Slide 12 |
| FNR热力图 | `outputs_stage4_run_20251031_075851_heatmap_fnr.png` | Slide 12 |
| JPEG鲁棒性 | `outputs_stage5_robustness_jpeg_f1.png` | Slide 15 |
| 高斯噪声鲁棒性 | `outputs_stage5_robustness_gaussian_f1.png` | Slide 15 |
| 运动模糊鲁棒性 | `outputs_stage5_robustness_motion_f1.png` | Slide 15 |

## 需要绘制的图表

1. **系统架构图** (Slide 4-5): 两阶段级联流程示意图
2. **六阶段流水线图** (Slide 7): 端到端流程
3. **数据集统计表** (Slide 9): 训练/验证/测试样本数
4. **移动端架构图** (Slide 8): Camera→Preprocess→Inference→UI

---

# 第六部分：答辩技巧提醒

1. **把"工程实现+应用价值"讲活** - 评委对"能真机跑起来"的项目印象更好

2. **主动解释"跨数据集F1低"** - 正式报告里就主动说这是有价值的负面发现

3. **避免陷入"模型细节"泥潭** - 重点讲为什么选这些模型，而不是block细节

4. **关键数字要背熟** - FNR~0.6%、升级率~1.2%、准确率92.8%、延迟180ms、F1<0.30

5. **语速控制** - 前3-4分钟要慢、清晰；方法和实验部分略快但稳定；总结部分放慢强化记忆

---

整体结构说明（本次覆盖的部分）

  本次先完成这些页（大约前 5–7 分钟内容）：

  1. Title
  2. Outline
  3. Deepfakes & Mobile Context
  4. On-device Challenges
  5. Problem Formulation
  6. Contributions

  我用「每页：Slide 内容 + 讲稿要点」的格式写，你之后可以自己写成 Beamer 的 frame。

  ———

  Slide 1 – Title

  - 标题内容（英文）
      - Title:
          - MobileDeepfake: A Lightweight Cascaded Deepfake Detection System for Mobile Deployment
      - Subtitle block:
          - Author: Wing Yam Ho
          - Advisor: [Advisor Name]
          - Department of [Department], The Hong Kong Polytechnic University
          - Master’s Thesis Defense
          - Date: [Defense Date]
  - 讲稿要点（中文，大约 30–45 秒）
      - 各位老师好，我是 ho Wing Yam，今天汇报的硕士论文题目是「MobileDeepfake: A Lightweight Cascaded Deepfake Detection System for Mobile Deployment」。
      - 这项工作主要围绕「在移动端设备上，如何在资源受限、隐私要求很高的情况下，依然有效地检测 deepfake」这个问题展开。
      - 接下来大概 25–30 分钟，我会介绍研究背景、数据与方法、实验结果、移动端部署，以及主要贡献和局限。

  ———

  Slide 2 – Outline

  - Slide 内容（英文 bullet）
      - Motivation & Problem Setting
      - Datasets and Task Definition
      - Method: Two-Stage Cascade and Threshold Tuning
      - Experiments, Results, and Robustness
      - Mobile Deployment and On-Device Evaluation
      - Discussion, Ethics, and Conclusion
  - 讲稿要点（中文，大约 40–60 秒）
      - 整个报告大致分为六个部分：
          - 首先是 deepfake 的应用场景和移动端环境下的挑战，明确我要解决的具体问题；
          - 然后介绍使用的数据集和任务设定；
          - 接着详细讲解两阶段级联系统的结构，以及如何用阈值调优来显式控制 FNR 和计算成本；
          - 之后是主要实验结果，包括 in-distribution、跨数据集 Deepfake‑Eval‑2024，以及鲁棒性分析；
          - 然后是移动端导出和在真实手机上的测试结果；
          - 最后我会总结工作贡献，并讨论局限、伦理和未来工作。

  ———

  Slide 3 – Deepfakes & Mobile Context

  - Slide 标题：Deepfakes in the Wild and the Rise of Mobile
  - Slide 内容（英文 bullet）
      - Deepfake face-swap media is used for scams, disinformation, and harassment.
      - Modern generators (GANs, diffusion, reenactment) produce highly realistic content.
      - Billions of users now consume and share media primarily on mobile devices.
      - Practical defenses must fit into mobile apps, often without server-side support.
  - 讲稿要点（中文，大约 1 分钟）
      - 先强调 deepfake 的现实风险：现在人脸替换视频已经被广泛用于金融诈骗、政治假讯息、以及非自愿影像等场景，而且生成难度和成本都在下降。
      - 相比传统简单 Photoshop，这些 deepfake 通常由 GAN、diffusion 或复杂的 reenactment 模型生成，视觉效果可以非常逼真，这对人和传统自动检测器都是挑战。
      - 同时，信息传播的主要载体已经转向移动端：像 TikTok、Instagram、WhatsApp 等平台，大部分流量来自手机。
      - 所以，如果防御手段只能在服务器端、或者需要很大的模型和算力，往往难以真正覆盖到终端用户，这也是我这篇工作聚焦「移动端 on-device 检测」的原因。

  ———

  Slide 4 – On-Device Challenges

  - Slide 标题：On-Device Deepfake Detection: Key Challenges
  - Slide 内容（英文 bullet）
      - Limited compute, memory, and energy on mobile CPUs/NPUs.
      - Strict latency and app-size budgets for good user experience.
      - Privacy constraints: media often must stay on the device.
      - Distribution shift: real-world social media data differs from curated benchmarks.
  - 讲稿要点（中文，大约 1 分钟）
      - 在移动端部署 deepfake 检测会同时面临几类约束：
          - 第一是资源限制：手机 CPU / NPU 的算力和内存都有限，而且还要考虑电量；
          - 第二是交互体验：模型太大或太慢，会让 App 变卡，或者包体积太大，用户甚至不会安装；
          - 第三是隐私：在很多隐私和合规要求下，图像和视频不能上传到云端做分析，只能在本地推理；
          - 最后是分布偏移：真实社交媒体上的视频和公开 benchmark 数据集差异很大，比如压缩方式、噪声、拍摄条件等，这会让检测器在部署时性能明显下滑。
      - 这几方面加在一起，使得我们不能简单把一个大模型硬塞到手机上，而需要一个更精细、可控的系统设计。

  ———

  Slide 5 – Problem Formulation

  - Slide 标题：Problem Formulation
  - Slide 内容（英文 bullet）
      - Goal: detect deepfake face-swap images/frames on mobile devices.
      - Priority: minimize false negatives (FNR) – missing harmful fakes is costly.
      - Constraint: keep compute cost and latency within a strict budget.
      - System-level metrics: FNR and Stage-2 escalation rate (how often the expert is used).
  - 讲稿要点（中文，大约 1 分钟）
      - 在这样的背景下，我把问题形式化为：在移动设备上，对人脸图像或视频帧做真假二分类。
      - 目标并不是简单追求最高的准确率或 AUC，而是优先最小化假阴性率，也就是 FNR——因为漏掉真正的 deepfake 带来的风险通常比多一些误报更严重。
      - 同时，我们不能无限制地提高模型复杂度，所以需要在一个明确的计算预算和延迟预算之内工作。
      - 因此，论文中除了常规指标之外，还特别引入了系统级指标：例如 FNR、以及 Stage‑2 escalation rate，也就是有多少百分比的样本会被送到第二阶段的“专家模型”，这一点会直接决
        定计算成本。
      - 后面我会展示，我们是如何在给定 Stage‑2 使用率预算下，通过阈值调优显式地最小化 FNR。

  ———

  Slide 6 – Contributions

  - Slide 标题：Contributions
  - Slide 内容（英文 bullet）
      - A cost-aware two-stage cascade for mobile deepfake detection with explicit control over FNR and Stage-2 usage.
      - An end-to-end reproducible pipeline: multi-dataset training, calibration, robustness sweeps, and OOD evaluation.
      - A practical mobile export path: quantized artifacts and on-device validation on a consumer smartphone.
  - 讲稿要点（中文，大约 1 分钟）
      - 结合刚才的问题定义，这篇工作的主要贡献可以概括为三点：
          - 第一，提出并实现了一个两阶段级联的 deepfake 检测系统，可以在显式控制 Stage‑2 使用率的情况下，把系统级 FNR 压到很低，适合移动端场景。
          - 第二，提供了一条从多数据集训练、校准，到鲁棒性评估和分布外 Deepfake‑Eval‑2024 测试的完整可复现流水线，论文中的表格和图基本都由脚本自动生成。
          - 第三，将这个级联系统导出为适合移动端的工件，在实际手机上做了延迟和精度测试，验证它在真实设备上的可行性。
      - 接下来我会按照刚刚的 outline，从数据与任务开始，依次展开这些贡献。

Slide 7 – Task Definition

  - Slide 标题（英文）
      - Task Definition
  - Slide 内容（英文 bullet）
      - Binary classification at the frame / image level.
      - Label 0 = real face, label 1 = fake face.
      - Manifests reference extracted face crops from videos.
      - Video-level decisions can be obtained by aggregating frame scores (e.g., max / majority).
  - 讲稿（中文要点）
      - 在任务设定上，本论文把 deepfake 检测形式化为「帧 / 图像级二分类」问题：标签 0 表示真实人脸，1 表示伪造人脸。
      - 对于视频，我们先通过预处理把视频拆成多帧人脸图像，真正训练和评估都是在这些人脸 crop 的层面完成。
      - 这样做的好处是：一方面可以显著放大有效样本数量，另一方面也能跟 CelebDF‑v2、FaceForensics++、DFDC 等公开基准的常用设定保持一致。
      - 如果实际应用需要视频级的判断，可以在系统上层对帧级分数做聚合，例如取最大概率或者多数投票，但在论文中，我把核心问题限制在帧级，这样更简洁、也更便于分析。
  - Script (English key points)
      - In this work, we formulate deepfake detection as a binary classification problem at the frame or image level.
      - We use label 0 for real faces and label 1 for fake faces, following common practice in CelebDF‑v2, FaceForensics++, and DFDC.
      - For videos, our manifests reference extracted face crops, and both training and evaluation operate on these crops rather than raw videos.
      - In downstream applications, we can aggregate frame-level scores into a video-level decision, for example by taking the maximum score or a majority vote, but the
        core task in the thesis is kept at the frame level for clarity.

  ———

  Slide 8 – Training & Validation Datasets

  - Slide 标题（英文）
      - Training and Validation Datasets
  - Slide 内容（英文 bullet）
      - Four academic datasets for training/validation:
          - CelebDF-v2: high-quality celebrity face swaps.
          - FaceForensics++: multiple manipulation methods under controlled conditions.
          - DFDC: large-scale, crowd-sourced benchmark with diverse compression pipelines.
          - DeeperForensics 1.0: real-world perturbations and challenging textures.
      - Balanced manifests: ~1.87M training samples, ~0.39M validation samples across all four datasets.
  - 讲稿（中文要点）
      - 在训练和验证阶段，我主要使用四个学术数据集：CelebDF‑v2、FaceForensics++、DFDC 和 DeeperForensics‑1.0。
      - CelebDF‑v2 侧重高质量名人换脸视频，FaceForensics++ 汇集了多种经典伪造方法；DFDC 来自众包平台，人物和压缩管线都非常多样；DeeperForensics‑1.0 则强调真实世界中的扰动
        和材质挑战。
      - 在这些数据集上，我通过 manifest 的方式做了统一的切分和采样，保证 real / fake 大致平衡。四个数据集合起来，训练集大约有 187 万张人脸图像，验证集大约 39 万张。
      - 后面 Stage 1 和 Stage 2 的训练、以及级联阈值调优，都是在这四个数据集的统一 manifest 上完成的，这一点对可复现性和系统分析非常重要。
  - Script (English key points)
      - For training and validation, we rely on four academic datasets: CelebDF‑v2, FaceForensics++, DFDC, and DeeperForensics 1.0.
      - CelebDF‑v2 focuses on high-quality celebrity face swaps; FaceForensics++ aggregates multiple manipulation families under controlled conditions; DFDC is a large,
        crowd-sourced benchmark with diverse actors and compression pipelines; and DeeperForensics emphasises real-world perturbations and challenging textures.
      - We construct unified, balanced manifests over these sources, so that real and fake samples are roughly matched in each split.
      - In total, we have around 1.87 million training samples and about 0.39 million validation samples across the four datasets, and all subsequent stages—Stage 1, Stage
        2, and cascade tuning—operate consistently on these manifests.

  ———

  Slide 9 – OOD Benchmark and Preprocessing Pipeline

  - Slide 标题（英文）
      - OOD Benchmark and Preprocessing Pipeline
  - Slide 内容（英文 bullet）
      - Deepfake-Eval-2024 as a held-out OOD benchmark (≈452k val, ≈402k test samples).
      - In-the-wild social media videos from 2024, spanning 88 websites and 52 languages.
      - Unified preprocessing: face detection with MTCNN, 256×256 RGB face crops, PNG format.
      - All datasets share the same crop specification and manifest format for fair comparison.
  - 讲稿（中文要点）
      - 除了四个训练/验证数据集之外，我把 Deepfake‑Eval‑2024 作为一个严格的分布外（OOD）测试基准，只用于验证和测试，不参与训练。它由 2024 年真实社交平台上的视频构成，规模
        大约是 45 万条验证样本和 40 万条测试样本。
      - 这个数据集覆盖了 80 多个网站和 50 多种语言，内容和前面四个相对「干净」的学术数据集差异很大，因此特别适合用来衡量级联系统在真实 in‑the‑wild 场景中的泛化能力。
      - 所有数据集都经过统一的预处理：从视频中按固定间隔抽帧，用 MTCNN 做人脸检测和对齐，按一定比例扩展 bbox，然后 resize 到 256×256 的 RGB 图像，并保存为 PNG。
      - 通过统一的 crop 规格和 manifest 格式，Stage 1、Stage 2 以及鲁棒性分析都可以在完全相同的数据接口上运行，这有利于公平对比和移动端导出。
  - Script (English key points)
      - Beyond the four training and validation datasets, we use Deepfake‑Eval‑2024 as a held-out out-of-distribution benchmark. It contains roughly 452 thousand
        validation samples and 402 thousand test samples, collected from social media platforms in 2024.
      - It spans 88 websites and 52 languages, and therefore differs significantly from the more curated academic datasets, making it well-suited to probe cross-dataset
        generalization.
      - All datasets, including Deepfake‑Eval‑2024, go through a unified preprocessing pipeline: we sample frames from videos, detect faces with MTCNN, enlarge the
        bounding boxes slightly, and resize the crops to 256×256 RGB PNG images.
      - This unified crop specification and manifest format ensures that Stage 1, Stage 2, the cascade, and robustness evaluations all operate on exactly the same kind of
        inputs, which is important for fair comparison and for later mobile deployment.

Slide 10 – System Overview

  - Slide 标题（英文）
      - System Overview: Six-Stage Pipeline
  - Slide 内容（英文 bullet）
      - Stage 0: data preprocessing and manifest generation.
      - Stage 1: lightweight MobileNetV4 filter (fast frame-level classifier).
      - Stage 2: EfficientNetV2-B3 expert model for hard cases.
      - Stage 3: optional GenConViT + LightGBM for research ablations.
      - Stage 4: cascade threshold tuning and grid search over (τ_low, τ_high).
      - Stage 5–6: robustness analysis and mobile export / deployment.
  - 讲稿（中文要点）
      - 在方法部分，我把整体系统拆分为六个阶段，对应论文 Method 部分的结构：
          - Stage 0 是数据预处理与 manifest 生成，把不同数据集统一到 256×256 人脸 crop 和统一的 CSV 接口；
          - Stage 1 是轻量级的 MobileNetV4 过滤器，做快速的帧级二分类；
          - Stage 2 是 EfficientNetV2‑B3 专家模型，专门处理 Stage 1 觉得「不确定」的困难样本；
          - Stage 3 还有一个 GenConViT + LightGBM 的元模型结构，但只作为研究性组件，不在最终部署路径中；
          - Stage 4 是级联阈值调优，通过网格搜索 τ_low 和 τ_high 来显式控制 FNR 和 Stage‑2 使用率；
          - Stage 5 和 Stage 6 对应鲁棒性评估以及将系统导出到移动端。
      - 在接下来的几张 slide，我会重点关注 Stage 1、Stage 2 和 Stage 4，因为它们构成了最终部署的核心。
  - Script (English key points)
      - In the Method section, I organise the system into six stages, which mirrors the structure in the paper.
      - Stage 0 handles data preprocessing and manifest generation, unifying all datasets into 256×256 face crops with a common CSV interface.
      - Stage 1 is a lightweight MobileNetV4 filter, which performs fast frame-level binary classification.
      - Stage 2 is an EfficientNetV2‑B3 expert model that focuses on samples that Stage 1 is uncertain about.
      - Stage 3 includes an optional GenConViT plus LightGBM meta-model, but this remains a research component and is not part of the default deployment path.
      - Stage 4 tunes the cascade thresholds by grid searching over τ_low and τ_high to explicitly control the trade-off between FNR and Stage‑2 usage.
      - Finally, Stage 5 and Stage 6 cover robustness evaluation and mobile export. In the next few slides, I will focus on Stages 1, 2, and 4, which form the core of the
        deployed system.

  ———

  Slide 11 – Two-Stage Cascade Intuition

  - Slide 标题（英文）
      - Two-Stage Cascade: Intuition
  - Slide 内容（英文 bullet）
      - Single heavy model: high accuracy but too slow and expensive for mobile.
      - Single light model: fast, but FNR remains too high for safety-critical use.
      - Cascade idea: let Stage 1 handle easy cases; escalate only ambiguous samples to Stage 2.
      - System-level objective: minimise FNR under a budget on Stage-2 escalation rate.
  - 讲稿（中文要点）
      - 在设计上，我没有直接选择「一个大模型」或者「一个小模型」，而是采用两阶段级联。原因是：
          - 如果只用一个高容量的大模型，虽然精度可以比较好，但在移动端上算力和延迟成本太高，不现实；
          - 如果只用一个非常轻量的小模型，速度没问题，但 FNR 会明显偏高，在防诈骗、内容审核等高风险场景不够安全。
      - 两阶段级联的直觉是：
          - 让 Stage 1 负责「快而粗」的判断，把明显的真样本和部分明显的假样本快速处理掉；
          - 对于 Stage 1 给出中等置信度的模糊样本，再交给 Stage 2 这种高成本的专家模型做更仔细的判断。
      - 因此系统的目标不再是单一模型的 AUC，而是「在给定 Stage‑2 调用率预算下，使系统级 FNR 尽可能低」，这一点会在阈值调优时具体体现。
  - Script (English key points)
      - Instead of relying on a single heavy model or a single light model, we choose a two-stage cascade design.
      - A single large model can achieve good accuracy, but it is too slow and expensive for real-time mobile deployment.
      - A single lightweight model is fast enough, but its false negative rate is still too high for safety-critical applications like fraud detection.
      - The intuition behind the cascade is to let Stage 1 handle easy cases quickly, and to escalate only ambiguous samples to the more expensive Stage 2 expert.
      - As a result, the primary objective shifts from maximising the accuracy of a single model to minimising the system-level FNR under a budget on the Stage‑2
        escalation rate.

  ———

  Slide 12 – Stage 1: MobileNetV4 Lightweight Filter

  - Slide 标题（英文）
      - Stage 1: MobileNetV4 Lightweight Filter
  - Slide 内容（英文 bullet）
      - Backbone: MobileNetV4-Hybrid-Medium (timm), input 256×256 RGB.
      - Balanced manifests across CelebDF-v2, FF++, DFDC, and DeeperForensics.
      - Augmentation: resize, horizontal flip, colour jitter, small affine, light blur.
      - Training: AdamW, cosine learning rate schedule, early stopping on validation AUC.
      - Role: fast, conservative filter with strong AUC but FNR ≈ 3.1% when used alone.
  - 讲稿（中文要点）
      - Stage 1 使用的骨干网络是 MobileNetV4‑Hybrid‑Medium，这个结构专门为移动端优化，输入分辨率统一为 256×256 RGB。
      - 训练数据来自刚才介绍的四个数据集，通过 manifest 做了平衡采样，确保 real 和 fake 数量大致均衡。
      - 数据增强方面，我采用了相对轻量的组合：resize、随机水平翻转、颜色抖动、小幅仿射变换，再加一点 Gaussian blur，目的是增强泛化能力同时不过度扭曲图像。
      - 优化器使用 AdamW，学习率采用 cosine 衰减，并根据验证集 AUC 做早停和模型选择。
      - 在合并验证集上，Stage 1 可以达到接近 0.994 的 AUC，说明作为过滤器它的整体判别能力很强；但如果单独部署 Stage 1，系统级 FNR 仍然在约 3.1% 左右，所以需要后面 Stage 2
        和级联策略进一步压低漏报。
  - Script (English key points)
      - In Stage 1, we use MobileNetV4‑Hybrid‑Medium as the backbone, with a unified input resolution of 256×256 RGB.
      - The model is trained on balanced manifests across CelebDF‑v2, FaceForensics++, DFDC, and DeeperForensics, so that real and fake samples are roughly matched.
      - We employ a relatively light augmentation pipeline: resizing to 256×256, random horizontal flip, colour jitter, small affine transformations, and a light Gaussian
        blur.
      - Optimisation uses AdamW with a cosine learning rate schedule, and we perform early stopping and model selection based on validation AUC.
      - On the combined validation split, Stage 1 achieves an AUC close to 0.994, which is strong for a lightweight filter, but as a standalone system its FNR is around
        3.1%, motivating the need for Stage 2 and the cascade.

  ———

  Slide 13 – Stage 2: EfficientNetV2-B3 Expert

  - Slide 标题（英文）
      - Stage 2: EfficientNetV2-B3 Expert
  - Slide 内容（英文 bullet）
      - Backbone: EfficientNetV2-B3, same 256×256 input resolution.
      - Trained on the same unified manifests as Stage 1.
      - Stronger augmentation: RandAugment, Mixup, CutMix for harder decision boundaries.
      - Loss: binary classification loss with optional focal component (hard example mining enabled in ablations).
      - Standalone performance: AUC ≈ 0.9633, F1 ≈ 0.8930, FNR ≈ 9.43% at the chosen operating point, with complementary error modes to Stage 1.
  - 讲稿（中文要点）
      - Stage 2 使用 EfficientNetV2‑B3 作为专家模型，同样采用 256×256 输入分辨率，并且使用与 Stage 1 相同的统一 manifest。
      - 为了让 Stage 2 专注于困难样本，数据增强比 Stage 1 更激进，包括 RandAugment、Mixup 和 CutMix，这些方法可以帮助模型学习更复杂、更鲁棒的决策边界。
      - 损失方面，使用二分类损失，并在部分实验中结合 focal loss 和基于 loss 的 hard example mining：通过 Stage 1 得分和训练过程中的损失来识别困难样本，并在训练中对这些样本
        做过采样。
      - 在某个固定阈值下，Stage 2 stand‑alone 的 AUC 约为 0.9633，F1 约为 0.893，FNR 约 9.43%。虽然这看起来 FNR 比 Stage 1 更高，但它出错的样本类型和 Stage 1 不一样，二者
        的 error mode 具有互补性，这正是后面级联能够显著降低系统级 FNR 的原因。
  - Script (English key points)
      - In Stage 2, we use EfficientNetV2‑B3 as the expert model, with the same 256×256 input resolution and the same unified manifests as Stage 1.
      - The data augmentation pipeline is stronger: we incorporate RandAugment, Mixup, and CutMix to encourage the model to learn harder and more robust decision
        boundaries.
      - For the loss function, we use a binary classification loss and, in some ablations, a focal component combined with loss-based hard example mining, where difficult
        samples are oversampled during training.
      - At a specific operating point, the standalone EfficientNetV2 expert achieves an AUC of about 0.9633, an F1 score of about 0.893, and an FNR of around 9.43%.
      - Although this FNR is higher than Stage 1’s standalone FNR, the types of errors are different; Stage 2 tends to capture manipulation patterns that Stage 1 misses,
        which makes the two stages complementary and enables the cascade to reduce the overall FNR.

  ———

  Slide 14 – Stage 4: Cascade Threshold Tuning & Calibration

  - Slide 标题（英文）
      - Stage 4: Cascade Threshold Tuning and Calibration
  - Slide 内容（英文 bullet）
      - Stage 1 outputs calibrated fake probability p₁(x) = P(fake | x).
      - Two thresholds (τ_low, τ_high) define three regions:
          - p₁ ≤ τ_low → predict REAL (Stage 1 only).
          - p₁ ≥ τ_high → predict FAKE (Stage 1 only).
          - τ_low < p₁ < τ_high → escalate to Stage 2.
      - Grid search over (τ_low, τ_high) on the validation set to minimise FNR under a Stage-2 rate budget.
      - Best “safety-first” operating point: τ_low ≈ 0.05, τ_high ≈ 0.55, FNR ≈ 0.6% with Stage-2 rate ≈ 1.16% and ~9% extra FLOPs vs Stage 1 only.
  - 讲稿（中文要点）
      - 在 Stage 4 中，我们把 Stage 1 的输出看成一个经过温度缩放校准后的「假样本概率」p₁(x)。
      - 然后通过两个阈值 τ_low 和 τ_high 把样本空间划成三段：
          - 如果 p₁ ≤ τ_low，就直接由 Stage 1 判为 REAL，不调用二阶段；
          - 如果 p₁ ≥ τ_high，就直接判为 FAKE，也不调用二阶段；
          - 只有当 p₁ 落在中间区间时，才把样本升级到 Stage 2 做精细分析。
      - 在验证集上，我们对一系列 (τ_low, τ_high) 组合做网格搜索，目标是在给定 Stage‑2 调用率预算下，让系统级 FNR 最小。
      - 论文中报告的一个「安全优先」操作点是 τ_low 约 0.05、τ_high 约 0.55，此时系统 FNR 大约 0.6%，Stage‑2 只处理约 1.16% 的样本，平均 FLOPs 只比 Stage1‑only 多了约 9%。
      - 为了让这些阈值可以在不同数据集和设备之间迁移，我们还对 Stage 1 和 Stage 2 做了温度缩放校准，使概率更符合真实分布，这在后面的跨数据集和移动端实验中会用到。
  - Script (English key points)
      - In Stage 4, we interpret the Stage 1 output as a calibrated fake probability p₁(x) after temperature scaling.
      - We then define two thresholds, τ_low and τ_high, which partition the samples into three regions:
          - if p₁ ≤ τ_low, we predict REAL using Stage 1 only;
          - if p₁ ≥ τ_high, we predict FAKE using Stage 1 only;
          - and if τ_low < p₁ < τ_high, we escalate the sample to Stage 2.
      - On the combined validation set, we perform a grid search over (τ_low, τ_high) pairs and select operating points that minimise FNR under a constraint on the Stage‑2
        escalation rate.
      - A key “safety-first” operating point in the thesis uses τ_low around 0.05 and τ_high around 0.55, achieving an FNR of about 0.6% while sending only about 1.16% of
        samples to Stage 2 and adding roughly 9% extra FLOPs compared to Stage 1 only.
      - To make these thresholds transferable across datasets and devices, we use temperature scaling to calibrate the probabilities of both stages, which becomes
        important in the cross-dataset and mobile experiments.

        Slide 15 – Experimental Setup and Metrics

  - Slide 标题（英文）
      - Experimental Setup and Metrics
  - Slide 内容（英文 bullet）
      - Evaluation on combined validation over four datasets (CelebDF-v2, FF++, DFDC, DeeperForensics).
      - Cross-dataset evaluation on held-out Deepfake-Eval-2024 (val / test splits).
      - Frame-level metrics: AUC, F1, accuracy, precision, recall, FNR, FPR.
      - System-level metrics: Stage-2 escalation rate and average FLOPs per sample.
  - 讲稿（中文要点）
      - 在结果部分，我主要在两个层面进行评估：
          - 第一是四个训练/验证数据集的「合并验证集」，用来衡量 in-distribution 下单模型和级联的性能；
          - 第二是完全分布外的 Deepfake‑Eval‑2024 验证 / 测试集，用来评估在真实社交媒体数据上的泛化能力。
      - 指标方面，除了常规的帧级分类指标（AUC、F1、Accuracy、Precision、Recall、FNR、FPR），我特别关注系统级指标：
          - 比如 Stage‑2 escalation rate，也就是有多少比例的样本会被送到第二阶段；
          - 以及平均每个样本的 FLOPs，用来近似计算成本。
      - 后面的几张图表都是在这些设定下得到的。
  - Script (English key points)
      - In the experiments, I evaluate the system on two levels.
      - First, I use the combined validation split over the four training datasets to measure in-distribution performance of the single models and the cascade.
      - Second, I perform cross-dataset evaluation on the held-out Deepfake‑Eval‑2024 validation and test splits, which reflect real-world social media data.
      - At the frame level, I report standard classification metrics such as AUC, F1, accuracy, precision, recall, FNR, and FPR.
      - At the system level, I focus on the Stage‑2 escalation rate and the average FLOPs per sample, which directly capture the compute cost of the cascade.

  ———

  Slide 16 – Single-Model Baselines

  - Slide 标题（英文）
      - Single-Model Baselines on Combined Validation
  - Slide 内容（英文 bullet）
      - Stage 1 (MobileNetV4): AUC ≈ 0.9936 on the combined validation split.
      - Stage 2 (EfficientNetV2-B3): AUC ≈ 0.9633, F1 ≈ 0.8930, Acc ≈ 0.895.
      - Stage 2 FNR ≈ 9.43% at the chosen operating point (higher than Stage 1’s standalone FNR).
      - Error modes differ: Stage 2 captures manipulation patterns Stage 1 misses, motivating a cascade rather than picking a single model.
  - 讲稿（中文要点）
      - 首先看单模型基线：
          - 在四个数据集合并的验证集上，Stage 1 的 MobileNetV4 轻量模型可以达到大约 0.9936 的 AUC，这说明它作为过滤器的整体区分能力非常强；
          - Stage 2 的 EfficientNetV2‑B3 专家模型在同一 split 上的 AUC 约为 0.9633，F1 约 0.893，整体准确率约 0.895。
      - 一个看起来有点「反直觉」的结果是：在特定阈值下，Stage 2 单独的 FNR 约为 9.43%，比 Stage 1 的最佳 standalone FNR 更高。
      - 这并不是说 Stage 2 比 Stage 1 更差，而是说明两者的 error mode 不同：
          - Stage 1 在大部分易分样本上做得很好，但在一些细节纹理或复杂操控上会出错；
          - 而 Stage 2 在这些难例上更有优势，却在部分相对简单的样本上偏保守。
      - 正因为这种互补性，我们没有简单选择「只用 Stage 1」或「只用 Stage 2」，而是通过级联把两者的优势组合起来。
  - Script (English key points)
      - Let me first show the single-model baselines on the combined validation split.
      - Stage 1, the MobileNetV4 lightweight filter, achieves an AUC of about 0.9936, which is very strong for a compact model and confirms its effectiveness as a fast
        filter.
      - Stage 2, the EfficientNetV2‑B3 expert, achieves an AUC of roughly 0.9633, an F1 score of about 0.893, and an overall accuracy of about 0.895 at its chosen
        operating point.
      - Interestingly, the standalone FNR of Stage 2 at that operating point is around 9.43%, which is higher than Stage 1’s best standalone FNR.
      - This does not mean that Stage 2 is worse overall; rather, it indicates that the two models make different types of errors: Stage 2 tends to capture manipulation
        patterns that Stage 1 misses, while sometimes being more conservative on easier cases.
      - This complementary behaviour is precisely why we design a cascade instead of selecting a single model.

  ———

  Slide 17 – Cascade Results and Trade-Offs

  - Slide 标题（英文）
      - Cascade Results on Combined Validation
  - Slide 内容（英文 bullet）
      - Stage 1-only: strong AUC but FNR ≈ 3.1% at a reasonable operating point.
      - Two-stage cascade (τ_low ≈ 0.05, τ_high ≈ 0.55): FNR ≈ 0.6% on combined validation.
      - Stage-2 escalation rate ≈ 1.1–1.2%: only a small fraction of frames trigger the expert.
      - Average FLOPs increase by ≈ 9% relative to Stage 1-only, while reducing FNR by ≈ 5×.
      - Different (τ_low, τ_high) pairs trace a Pareto frontier between FNR and Stage-2 usage.
  - 讲稿（中文要点）
      - 把 Stage 1 和 Stage 2 级联之后，在合并验证集上可以看到比较清晰的收益：
          - 当只用 Stage 1 时，在一个合理阈值下，系统级 FNR 大约是 3.1%；
          - 使用两阶段级联并选择 τ_low ≈ 0.05、τ_high ≈ 0.55 这样的「安全优先」操作点时，系统级 FNR 可以降到大约 0.6%。
      - 与此同时，只有大约 1.1%–1.2% 的样本会被升级到 Stage 2，也就是说绝大多数样本只经过一次轻量推理。
      - 以 FLOPs 近似计算，平均每个样本的计算量只比 Stage 1‑only 增加大约 9%，但 FNR 从 3.1% 压到 0.6%，大概是 5 倍的相对改进。
      - 如果我们在 (τ_low, τ_high) 空间中画出不同组合的结果，可以看到一条「FNR vs Stage‑2 rate」的 Pareto 曲线：
          - 把 τ_low 降得更低，可以继续压 FNR，但 Stage‑2 rate 和计算成本会上升；
          - 把 τ_high 调得更高，则可以减少 Stage‑2 调用，但 FNR 会反弹。
      - 这条曲线为不同应用场景提供了清晰的选择：比如高安全优先 vs 计算成本优先。
  - Script (English key points)
      - When we combine Stage 1 and Stage 2 into a cascade, we obtain a clear system-level benefit on the combined validation split.
      - At a reasonable operating point, Stage 1-only yields a system FNR of about 3.1%.
      - With a two-stage cascade using a “safety-first” configuration, for example τ_low around 0.05 and τ_high around 0.55, the system-level FNR drops to roughly 0.6%.
      - At the same time, only about 1.1–1.2% of samples are escalated to Stage 2, so the vast majority of frames are handled by the lightweight Stage 1 model.
      - In terms of compute, the average FLOPs per sample increase by only about 9% compared to Stage 1-only, while delivering about a five-fold reduction in FNR.
      - If we plot different (τ_low, τ_high) pairs, we obtain a Pareto frontier between FNR and the Stage‑2 escalation rate: lowering τ_low further can reduce FNR but
        increases compute; raising τ_high can save compute but risks higher FNR.
      - This trade-off curve allows practitioners to select operating points according to their risk and compute budgets.

  ———

  Slide 18 – Cross-Dataset Results: Deepfake-Eval-2024

  - Slide 标题（英文）
      - Cross-Dataset Results: Deepfake-Eval-2024
  - Slide 内容（英文 bullet）
      - Calibrated cascade (Stage 1 + Stage 2 + in-distribution thresholds):
          - FNR ≈ 0.73–0.79, F1 ≈ 0.28–0.30, Stage-2 rate ≈ 0.51–0.52 on val/test.
      - Stage-2-only upper bound on Deepfake-Eval-2024:
          - FNR ≈ 0.15–0.18, F1 ≈ 0.45–0.60, Stage-2 rate ≈ 0.98–0.99, but FPR is extremely high (≈0.87–0.92).
      - Interpretation: strong distribution shift; thresholds tuned on curated datasets do not transfer.
      - Takeaway: cascade is effective in-distribution, but robust OOD performance requires additional domain-aware adaptation.
  - 讲稿（中文要点）
      - 在 Deepfake‑Eval‑2024 这个分布外数据集上，情况就明显不同了：
          - 如果直接使用在合并验证集上调好的「校准级联」（Stage 1 + Stage 2 + 固定 τ_low/τ_high），在 Deepfake‑Eval‑2024 上的 FNR 约在 0.73–0.79 之间，F1 只有 0.28–0.30 左
            右，说明大量真实的 deepfake 被漏检；Stage‑2 rate 仍然在 0.51–0.52 左右。
          - 作为上界比较，如果我们把 Stage 2 单独跑在 Deepfake‑Eval‑2024 上，也就是对所有样本都调用专家模型，FNR 可以降到大约 0.15–0.18，F1 在 0.45–0.60 之间，但 FPR 非常
            高，大约 0.87–0.92，Stage‑2 rate 接近 100%。
      - 这组结果反映了两个重要现象：
          - 第一，Deepfake‑Eval‑2024 与训练数据之间存在显著的分布差异，导致在 in-distribution 上调得很好的阈值迁移到 OOD 时会失效；
          - 第二，即便使用最强的 Stage 2 模型，在不重新调阈值、不做额外适配的情况下，也无法同时保证低 FNR 和低 FPR。
      - 因此论文在讨论部分明确强调：
          - 级联在 in-distribution 场景下非常有效，但在 OOD 场景中，必须结合域适配、重新校准甚至重新训练，才能获得更稳健的表现；
          - 本工作优先展示一种「可复现、可调参」的系统方法，并把更深入的 OOD 适配作为未来工作。
  - Script (English key points)
      - On the held-out Deepfake‑Eval‑2024 benchmark, the picture is quite different.
      - If we take the calibrated cascade tuned on the combined validation set and apply it directly to Deepfake‑Eval‑2024, the FNR is about 0.73–0.79, with F1 scores
        around 0.28–0.30 and a Stage‑2 rate of about 0.51–0.52. This means the cascade misses a large fraction of deepfakes under strong distribution shift.
      - As an upper bound, if we run Stage 2 alone on all samples, we can reduce FNR to around 0.15–0.18 and obtain F1 between roughly 0.45 and 0.60, but at the cost of
        extremely high FPR, about 0.87–0.92, and a Stage‑2 rate close to 1.0.
      - These results highlight two key points:
          - First, there is substantial distribution shift between the curated training datasets and Deepfake‑Eval‑2024, and thresholds tuned in-distribution do not
            transfer well.
          - Second, even the strongest single expert cannot simultaneously achieve low FNR and low FPR on this OOD benchmark without additional adaptation.
      - The thesis therefore emphasises that while the cascade is very effective in-distribution, robust OOD performance requires domain-aware calibration, potential re-
        tuning, or re-training, which we leave as an important direction for future work.


 Slide 19 – Robustness Evaluation Setup

  - Slide 标题（英文）
      - Robustness Evaluation: Setup
  - Slide 内容（英文 bullet）
      - Stress-test the calibrated cascade under common corruptions.
      - Perturbations applied to face crops before inference:
          - JPEG compression (qualities 95, 80, 60, 40, 20).
          - Gaussian noise (σ ∈ {2, 4, 8, 12}).
          - Motion blur (kernel sizes 3, 5, 9, 13).
          - Brightness changes (factors 0.7, 0.85, 1.15, 1.3).
      - Metrics: F1, accuracy, FNR, and Stage-2 rate for each perturbation level.
  - 讲稿（中文要点）
      - 为了评估系统面对「真实世界噪声」时的表现，我对校准后的级联系统做了一系列鲁棒性测试。
      - 具体做法是：在推理前对人脸 crop 施加不同类型的扰动，包括：
          - 不同质量等级的 JPEG 压缩，例如 95、80、60、40、20；
          - 不同强度的高斯噪声，σ 等于 2、4、8、12；
          - 不同长度的一维运动模糊卷积核，例如 3、5、9、13；
          - 不同亮度缩放因子，比如 0.7、0.85、1.15、1.3。
      - 在每一个扰动类型和强度下，我都固定级联阈值，单独统计对应的 F1、Accuracy、FNR 以及 Stage‑2 使用率。
      - 这些测试旨在模拟社交平台中常见的压缩、噪声、抖动和光照变化，而不是重新训练模型，因此可以看作是一种「后验鲁棒性」分析。
  - Script (English key points)
      - To understand how the cascade behaves under realistic noise, I perform a series of robustness experiments on the calibrated cascade.
      - Before inference, I apply different perturbations to the face crops, including:
          - JPEG compression at multiple quality levels (95, 80, 60, 40, 20);
          - Gaussian noise with σ values of 2, 4, 8, and 12;
          - Motion blur with kernel sizes 3, 5, 9, and 13;
          - And brightness changes with factors 0.7, 0.85, 1.15, and 1.3.
      - For each perturbation type and level, I keep the cascade thresholds fixed and measure F1, accuracy, FNR, and the Stage‑2 escalation rate.
      - These experiments approximate common distortions encountered on social media platforms and provide a post-hoc robustness analysis without retraining the models.

  ———

  Slide 20 – Robustness Results and Observations

  - Slide 标题（英文）
      - Robustness: Results and Observations
  - Slide 内容（英文 bullet）
      - Severe perturbations can substantially degrade F1 and increase FNR.
      - JPEG compression at very low quality and strong Gaussian noise are particularly harmful.
      - Stage-2 rate tends to increase under difficult conditions, as more samples become ambiguous for Stage 1.
      - Brightness shifts and motion blur expose specific failure modes (e.g., washed-out faces, smeared textures).
      - Takeaway: cascade improves efficiency, but robustness to heavy corruptions remains limited.
  - 讲稿（中文要点）
      - 从鲁棒性汇总表可以看到，在强扰动下系统性能会明显劣化：
          - 在很低质量的 JPEG 压缩、较大的高斯噪声、以及极端亮度变化等条件下，F1 会显著下降，FNR 则会大幅上升；
          - 这说明当前模型在输入质量被严重破坏时，已经难以可靠地区分真伪。
      - 同时，可以观察到 Stage‑2 使用率在困难场景下会普遍升高：
          - 这反映了级联系统的一个自然行为——当 Stage 1 的置信度变低时，会把更多样本升级到 Stage 2；
          - 某种程度上，这是级联试图在噪声环境下「自我保护」的机制，但也意味着整体计算成本会增加。
      - 对于亮度变化和运动模糊，错误样例分析表明：
          - 过暗或过亮时，人脸细节被洗掉，模型容易把真实样本当成伪造；
          - 剧烈的模糊会消除 deepfake 典型纹理，使得一些伪造样本变得更像真实样本。
      - 综合来看，级联架构主要改善的是「计算效率 vs FNR」的 trade‑off，对严重扰动下的鲁棒性提升有限，这一点在论文的讨论章节里被明确作为未来工作方向，例如更强的数据增广或域
        自适应方法。
  - Script (English key points)
      - The robustness table shows that under strong perturbations, system performance can degrade significantly.
      - For example, very low JPEG quality, strong Gaussian noise, and extreme brightness changes can dramatically reduce F1 and increase FNR, indicating that the models
        struggle when the input quality is heavily degraded.
      - At the same time, the Stage‑2 escalation rate tends to increase under these difficult conditions, because more samples fall into the ambiguous region where Stage 1
        is uncertain.
      - This behaviour reflects the cascade’s attempt to protect itself by delegating more samples to Stage 2, but it also means that the compute cost goes up.
      - Error analysis shows that brightness shifts and motion blur expose specific failure modes: washed-out or overly dark faces lose fine details, and heavy blur can
        remove the frequency patterns that deepfake detectors rely on.
      - Overall, the cascade primarily improves the efficiency–FNR trade-off; its robustness to heavy corruptions remains limited and motivates future work on stronger
        augmentation and domain adaptation.

  ———

  Slide 21 – Mobile Export Pipeline and Artifacts

  - Slide 标题（英文）
      - Mobile Export: Pipeline and Artifacts
  - Slide 内容（英文 bullet）
      - Export Stage 1 and Stage 2 from PyTorch to ONNX.
      - Apply post-training dynamic quantization to reduce model size.
      - Bundle models with thresholds, calibration parameters, and simple cascade logic.
      - On-device evaluation uses a compact mobile_eval subset (152 preprocessed face images).
      - Exported artifacts: MobileNetV4 ≈ 37.5 MB, EfficientNetV2-B3 ≈ 51.9 MB (total ≈ 89.3 MB).
  - 讲稿（中文要点）
      - 在移动端部署方面，我实现了一条从 PyTorch 到 ONNX 的导出流水线：
          - 首先把 Stage 1 和 Stage 2 的 PyTorch 模型导出为 ONNX 格式，方便在 Android 上使用通用推理引擎；
          - 然后对线性层做后训练动态量化，在不改变结构的前提下显著压缩模型体积。
      - 在部署时，不仅仅是模型权重被打包，还包括：
          - 对应的级联阈值 (τ_low, τ_high)、温度缩放参数，以及一个简单的 cascade 逻辑封装；
          - 这样手机端只需要加载这些工件，就可以复现和 PC 端一致的级联决策。
      - 为了避免把整套多 GB 的数据集搬到手机上，我构建了一个精简的 mobile_eval 子集，一共 152 张人脸图像，真实和伪造样本按目录区分，方便自动计算准确率。
      - 导出后的 ONNX 模型大小大约是：Stage 1 约 37.5 MB，Stage 2 约 51.9 MB，总体约 89.3 MB，对于一个研究原型来说已经可以接受；在实际产品中可以通过 QAT、蒸馏或更轻量架构
        进一步压缩。
  - Script (English key points)
      - For mobile deployment, I implement an export pipeline from PyTorch to ONNX.
      - Both Stage 1 and Stage 2 are exported as ONNX models, and I apply post-training dynamic quantization to reduce model sizes without changing the architecture.
      - The exported bundle includes not only the model weights but also the cascade thresholds, temperature scaling parameters, and a simple routing logic, so that the
        mobile app can reproduce the same decisions as the desktop pipeline.
      - To avoid copying multi-gigabyte datasets to the phone, I construct a compact mobile_eval subset with 152 preprocessed face images, organised into real and fake
        folders for automatic accuracy computation.
      - The resulting ONNX artifacts are about 37.5 MB for MobileNetV4 and 51.9 MB for EfficientNetV2‑B3, for a total of roughly 89.3 MB, which is acceptable for a
        research prototype and could be further reduced with more aggressive compression in future work.

  ———

  Slide 22 – On-Device Evaluation: Xiaomi 13

  - Slide 标题（英文）
      - On-Device Evaluation: Xiaomi 13
  - Slide 内容（英文 bullet）
      - Device: Xiaomi 13 smartphone (Snapdragon 8 Gen 2).
      - Dataset: mobile_eval subset (N = 152 images).
      - On-device cascade performance:
          - Accuracy ≈ 92.8%, FNR ≈ 2.5%, FPR ≈ 12.5%.
          - Stage-2 rate ≈ 7.2% (higher than ≈1.2% on combined validation).
          - Average latency ≈ 180 ms per image (≈3.4 ms preprocessing + ≈176 ms ONNX inference).
      - PC vs mobile: FNR stays <3% on both; higher FPR and Stage-2 rate on mobile reflect dataset differences rather than export degradation.
  - 讲稿（中文要点）
      - 在真实设备上，我选择了一台搭载 Snapdragon 8 Gen 2 的小米 13 手机，对 mobile_eval 子集做了 on-device 测试。
      - 在这 152 张图像上，导出的级联系统在手机端的表现大致是：
          - 准确率约 92.8%，FNR 约 2.5%，FPR 约 12.5%；
          - Stage‑2 使用率约 7.2%，明显高于 PC 合并验证集上的约 1.2%。
      - 延迟方面，单图像平均总时延约 180 毫秒，其中预处理大约 3.4 毫秒，ONNX 推理大约 176 毫秒，整体上对用户来说是接近实时的反馈。
      - 与 PC 端结果对比可以看到：
          - FNR 在两个平台上都保持在 3% 以下，说明在当前配置下，导出和量化过程并没有破坏系统「偏向低漏报」的特性；
          - 手机上的 FPR 和 Stage‑2 rate 都更高，这主要是因为 mobile_eval 子集的分布和 PC 合并验证集不一样，而不是导出本身的问题。
      - 总体上，这组实验证明：两阶段级联在消费级手机上是可行的，并且其行为与桌面环境基本一致，为未来更系统的 on-device benchmark 打下了基础。
  - Script (English key points)
      - For real-device testing, I evaluate the exported cascade on a Xiaomi 13 smartphone with a Snapdragon 8 Gen 2 chip, using the mobile_eval subset of 152 images.
      - On this subset, the on-device cascade achieves about 92.8% accuracy, an FNR of roughly 2.5%, and an FPR of about 12.5%.
      - The Stage‑2 escalation rate is around 7.2%, which is higher than the roughly 1.2% observed on the combined validation set.
      - The average per-image latency is approximately 180 ms, including about 3.4 ms for preprocessing and around 176 ms for ONNX inference, which is sufficiently fast
        for interactive single-image use.
      - Comparing PC and mobile, the FNR remains below 3% on both platforms, indicating that the cascade’s low-FNR behaviour survives the export and quantization steps.
      - The higher FPR and Stage‑2 rate on mobile are primarily due to distribution differences in the mobile_eval subset rather than degradation from the export process
        itself.
      - Overall, these results demonstrate that the two-stage cascade is practical on consumer smartphones and behaves consistently with the desktop pipeline.


  Slide 23 – System-Level Trade-Offs

  - Slide 标题（英文）
      - System-Level Trade-Offs
  - Slide 内容（英文 bullet）
      - Cascade exposes explicit dials: τ_low, τ_high, Stage-1 leakage π_leak, Stage-2 rate π₂.
      - Lower τ_low / higher τ_high → lower FNR but higher Stage-2 usage and compute.
      - Higher τ_low / lower τ_high → lower compute but higher FNR (more missed fakes).
      - Example operating points on combined validation:
          - Safety-first: FNR ≈ 0.6%, π₂ ≈ 1.16%.
          - Device-first: FNR ≈ 3.6%, π₂ ≈ 8.6%.
  - 讲稿（中文要点）
      - 在系统层面，级联最大的优势之一是把很多「隐含」的超参变成了清晰可调的旋钮：
          - 包括 Stage 1 的两个阈值 τ_low、τ_high，Stage‑1 leakage π_leak，以及 Stage‑2 escalation rate π₂ 等。
      - 从调参结果可以看到：
          - 当我们降低 τ_low、或适度提高 τ_high 时，系统 FNR 会明显下降，但需要接受更高的 Stage‑2 使用率和计算成本；
          - 相反，如果我们提高 τ_low、收紧 τ_high，可以大幅减少二阶段调用，但 FNR 会上升，意味着更多 deepfake 会被漏掉。
      - 例如在合并验证集上：
          - 「安全优先」的 operating point 可以把 FNR 压在 0.6% 左右，而 π₂ 只有大约 1.16%；
          - 更「设备优先」的设置则会把 FNR 放宽到 3.6% 左右，换取更低的平均计算量。
      - 这种显式的 trade‑off 让系统适用于不同部署场景，也方便在实际运行中进行长期监控和审计。
  - Script (English key points)
      - At the system level, one of the main benefits of the cascade is that it turns many implicit hyperparameters into explicit dials, such as τ_low, τ_high, the Stage‑1
        leakage π_leak, and the Stage‑2 escalation rate π₂.
      - Tuning results show clear patterns:
          - Lowering τ_low or raising τ_high can significantly reduce FNR, but at the cost of higher Stage‑2 usage and compute.
          - Conversely, increasing τ_low or tightening τ_high reduces compute but increases FNR, meaning more deepfakes are missed.
      - On the combined validation set, for example, a safety-first operating point keeps FNR around 0.6% with π₂ around 1.16%, while a more device-first configuration
        accepts FNR around 3.6% to reduce Stage‑2 usage.
      - These explicit trade-offs make the system easier to adapt to different deployment scenarios and to monitor over time than a single opaque model.

  ———

  Slide 24 – Limitations and Future Work

  - Slide 标题（英文）
      - Limitations and Future Work
  - Slide 内容（英文 bullet）
      - Cross-dataset gaps: cascade tuned in-distribution degrades on Deepfake-Eval-2024.
      - Robustness: strong corruptions (heavy compression, noise, blur) still cause high FNR.
      - No temporal modeling: frame-level detection ignores video consistency.
      - Fairness and bias: no stratified analysis across demographic groups or content domains.
      - Future directions:
          - Domain-aware calibration and adaptation.
          - Temporal aggregation and video-level modeling.
          - Stronger compression (distillation, NAS, pruning) for Stage 2.
          - Adversarial robustness and continuous monitoring in deployment.
  - 讲稿（中文要点）
      - 虽然级联在 in-distribution 上取得了不错的指标，但论文也明确承认了几个重要的局限：
          - 首先是跨数据集的性能缺口：在 Deepfake‑Eval‑2024 上，直接迁移 in-distribution 调好的阈值会导致 FNR 和 FPR 都变得不可接受；
          - 其次是鲁棒性有限：在强压缩、强噪声和重模糊等极端情况下，系统仍然会有很高的 FNR；
          - 此外，目前完全基于帧级分类，没有利用视频的时序一致性；
          - 也没有对不同人群、语言或内容类型做分组公平性分析，这在真实部署中是必须补上的一块。
      - 因此，我在未来工作中重点提出了几条方向：
          - 做域感知的校准和适配，让阈值和模型能针对具体平台和内容进行调整；
          - 引入时序建模和视频级聚合，以提高对局部操控和短时伪影的鲁棒性；
          - 对 Stage 2 探索更激进的压缩，例如蒸馏、结构搜索和剪枝，以进一步降低延迟和能耗；
          - 系统性研究对抗鲁棒性，并在部署中建立持续监控和定期更新机制。
  - Script (English key points)
      - Despite good in-distribution performance, the thesis explicitly acknowledges several important limitations.
      - First, there are cross-dataset gaps: when we transfer thresholds tuned in-distribution to Deepfake‑Eval‑2024, both FNR and FPR become problematic.
      - Second, robustness remains limited under strong corruptions such as heavy compression, strong noise, and severe blur.
      - Third, the current system is purely frame-level and does not exploit temporal consistency in videos.
      - Fourth, we do not perform stratified fairness analysis across demographic groups or content domains, which is critical in real deployments.
      - As future work, I highlight domain-aware calibration and adaptation, temporal aggregation and video-level modeling, stronger compression of Stage 2 via
        distillation, NAS, or pruning, and systematic work on adversarial robustness and continuous monitoring in deployed systems.

  ———

  Slide 25 – Ethics and Societal Impact

  - Slide 标题（英文）
      - Ethics and Societal Impact
  - Slide 内容（英文 bullet）
      - Positive impact: strengthen defenses against scams, disinformation, and non-consensual deepfakes.
      - On-device processing helps preserve user privacy (no raw media sent to servers).
      - Risk of misuse: detectors could be used to probe and evade defenses if thresholds are exposed.
      - Data governance: respect dataset licenses, NSFW content warnings, and gated access policies (e.g., Deepfake-Eval-2024).
      - Need for fairness audits: evaluate performance across demographics to avoid disparate impact.
  - 讲稿（中文要点）
      - 从伦理和社会影响的角度，这项工作既有积极的一面，也有需要谨慎对待的风险：
          - 正面来看，一个可部署的 deepfake 检测系统可以帮助平台、用户和监管机构更好地抵御诈骗、虚假信息和非自愿影像；
          - 将推理放在终端设备上，而不是把所有媒体上传到云端，有助于保护用户隐私，也更符合像 GDPR 这类法规对数据本地化的要求。
      - 同时，这类技术也存在被滥用的风险：
          - 如果攻击者可以反复查询检测系统，就可能用它来「探测」阈值和盲区，进而生成更难检测的伪造内容；
          - 因此在实际部署中，需要结合速率限制、API 访问控制，以及对阈值和模型细节的适当隐藏。
      - 在数据治理上，本工作遵守各个数据集的许可证要求，对 NSFW 内容保持谨慎；像 Deepfake‑Eval‑2024 这种 gated 数据集，只在允许的范围内做 OOD 评估。
      - 未来还需要开展更系统的公平性审计：
          - 例如按性别、肤色、语言等维度拆分 FNR / FPR，以避免系统对某些群体造成不成比例的误判风险。
  - Script (English key points)
      - From an ethical and societal perspective, this work has both positive impacts and potential risks.
      - On the positive side, a deployable deepfake detector can help platforms, users, and regulators defend against scams, disinformation, and non-consensual synthetic
        media.
      - On-device processing further helps protect user privacy by avoiding the need to upload raw media to cloud servers and aligns with regulations such as GDPR and data
        localization laws.
      - At the same time, detection technology can be misused: an attacker might probe the system to discover thresholds and blind spots and then craft deepfakes that
        evade detection.
      - Practical deployments therefore require rate limiting, access control, and careful handling of threshold details.
      - In terms of data governance, the work respects dataset licenses, treats NSFW content carefully, and uses gated datasets like Deepfake‑Eval‑2024 strictly for OOD
        evaluation.
      - Looking forward, systematic fairness audits—disaggregating FNR and FPR across demographic groups—will be important to avoid disparate impact.

  ———

  Slide 26 – Conclusion and Q&A

  - Slide 标题（英文）
      - Conclusion and Q&A
  - Slide 内容（英文 bullet）
      - Proposed MobileDeepfake: a cost-aware two-stage cascade for mobile deepfake detection.
      - Multi-dataset training, calibrated cascade routing, and explicit control over FNR and Stage-2 usage.
      - Robustness and OOD evaluations highlight both strengths and current limitations.
      - Mobile export and on-device validation demonstrate practical feasibility on consumer hardware.
      - Future work: domain adaptation, temporal modeling, stronger compression, and fairness/robustness audits.
      - Thank you for your attention — I am happy to take questions.
  - 讲稿（中文要点）
      - 最后做一个简要的总结：
          - 本论文提出了 MobileDeepfake，一个面向移动端的两阶段级联 deepfake 检测系统，通过显式的阈值调优，在可控的 Stage‑2 调用率下，把 FNR 压到了比较低的水平；
          - 整个系统基于多数据集训练和校准的级联路由，并通过鲁棒性和跨数据集实验展示了它在 in-distribution 场景中的优势以及在 OOD 场景中的局限；
          - 通过导出到 ONNX 并在真实手机上测试，我验证了这个方案在消费级硬件上的可行性。
      - 在未来工作方面，我希望进一步在几个方向深入：
          - 针对具体平台做更系统的域适配和重新校准；
          - 引入时序建模和视频级决策；
          - 对 Stage 2 做更强的压缩与加速；
          - 以及从公平性和对抗鲁棒性的角度，对整个系统进行更全面的审计。
      - 我的报告先到这里，非常感谢各位老师的聆听，接下来欢迎提问和指导。
  - Script (English key points)
      - To conclude, this thesis presents MobileDeepfake, a cost-aware two-stage cascade for mobile deepfake detection.
      - The system combines multi-dataset training, calibrated cascade routing, and explicit control over FNR and Stage‑2 usage, providing a practical way to balance
        accuracy and compute.
      - Robustness and cross-dataset evaluations highlight both the strengths of the cascade in-distribution and its limitations under strong distribution shift.
      - The mobile export pipeline and on-device validation on a consumer smartphone demonstrate that the approach is feasible beyond the lab.
      - Future work includes domain adaptation and recalibration for specific platforms, temporal modeling and video-level decisions, stronger compression of Stage 2, and
        comprehensive fairness and robustness audits.
      - Thank you very much for your attention, and I am happy to take any questions.
