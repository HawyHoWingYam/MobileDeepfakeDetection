# MobileDeepfake 硕士答辩演示文档 - 精简版

> 本文档是答辩演示的精简版，包含每个 slide 的核心要点、简化英文讲稿和教授可能问的问题。
>
> 最后更新: 2025-12-16

---

## 关键数字（必须背熟）

| 场景 | 指标 | 值 |
|------|------|-----|
| PC In-domain | FNR | ~0.6% |
| PC In-domain | Stage2 Rate | ~1.2% |
| PC In-domain | Cascade AUC | 0.9941 |
| Mobile (Xiaomi 13) | Accuracy | **92.8%** |
| Mobile (Xiaomi 13) | Latency | ~180ms |
| OOD (Eval-2024) | F1 | <0.30 |
| OOD (Eval-2024) | FNR | 73-79% |

---

# Part 1: Introduction (4-5 min)

## Slide 1: MobileDeepfake (Title)

**核心要点:**
- 硕士论文题目：MobileDeepfake
- 研究方向：移动端两阶段级联深度伪造检测
- 学生、导师、学校与日期信息

**简化英文讲稿:**
Good morning, professors.
My name is Wing Yam Ho.
Today I will present my master's thesis called "MobileDeepfake".
It is a cascaded deepfake detection system designed for mobile devices.
The key idea is to combine a fast small model with a stronger model, so we keep both accuracy and speed.
In the next 25 minutes, I will explain the background, the method, the datasets, the experiments, and the mobile app.
At the end, I will summarize the main findings and discuss limitations and future work.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 你的项目"MobileDeepfake"的研究范围具体是什么？它在攻击类型、使用场景和平台上有哪些刻意不做或不覆盖的部分？ / What exactly is the research scope of "MobileDeepfake"? Which attack types, usage scenarios, and platforms are intentionally out of scope?
   A: 我的工作主要聚焦在移动端短视频与前置摄像头录制场景下的"人脸视频 deepfake"二分类检测：输入是一段人脸视频或若干关键帧，输出是真／假标签及置信度。我刻意不做的是：纯音频 deepfake、全身／多人的复杂场景、篡改位置精确定位（只做有无篡改）、以及服务器端的大规模平台级审核。这样划定范围的理由是，在有限时间和移动端算力约束下，先在最常见、危害最大的"单人脸移动视频"场景上做到 FNR 0.60%、移动端 92.8% 准确率和约 180ms 的实时检测，更有助于形成一个清晰、可验证的技术闭环。 / My work focuses on binary detection of "face-video deepfakes" in mobile scenarios such as short-video apps and front-camera recordings: the input is a face video or key frames, and the output is a real/fake label with confidence. I intentionally exclude pure audio deepfakes, complex full-body or multi-person scenes, precise tampering localization (I only do "fake vs real"), and large-scale server-side moderation systems. The reason is that under limited time and on-device compute, concentrating on the most common and harmful "single-face mobile video" scenario and achieving 0.60% FNR, 92.8% mobile accuracy, and ~180ms latency yields a clear, verifiable technical loop.

2. Q: 你为什么认为"移动端 deepfake 检测"是一个值得做硕士论文的大题目，而不是把模型简单地部署到手机上这么简单？ / Why do you believe "mobile deepfake detection" is worthy of a master's thesis topic, rather than just "deploying an existing model to a phone"?
   A: 首先，现有在服务器端表现很好的模型，在真实移动端分布上的 OOD 表现极差——在 Deepfake-Eval-2024 这样的数据上 F1 低于 0.30，FNR 高达 73–79%，这说明"简单搬到手机上"远远不够。其次，移动端还受限于 ≤100MB 模型体积、<200ms 延迟、电量与隐私约束，需要在轻量模型（Stage 1：MobileNetV4 37.3MB，AUC 0.9936）、高精度模型（Stage 2：EfficientNetV2-B3 49.6MB，AUC 0.9633）和级联策略之间精心权衡，才能最终在端侧实现 FNR 0.60%、Stage2 触发率仅 1.16%。这些问题涉及分布偏移、代价敏感设计和工程落地，远超"简单部署"的范畴。 / First, models that perform very well on server-side benchmarks degrade severely on real mobile distributions—on datasets such as Deepfake-Eval-2024, F1 falls below 0.30 and FNR reaches 73–79%, showing that "just porting a model to a phone" is far from enough. Second, mobile devices are constrained by ≤100MB model size, <200ms latency, battery, and privacy, so you must carefully balance a lightweight model (Stage 1: MobileNetV4 37.3MB, AUC 0.9936), a high-capacity model (Stage 2: EfficientNetV2-B3 49.6MB, AUC 0.9633), and a cascade policy to reach 0.60% FNR with only 1.16% of samples escalated to Stage 2 on-device. These issues involve distribution shift, cost-sensitive design, and engineering deployment, which go far beyond "simple deployment".

3. Q: 导师在这个课题中具体给了你哪些方向性的指导？哪些关键技术决策（比如级联结构、阈值设计）是你自己主导的？ / What specific guidance did your supervisor provide in this project, and which key technical decisions (e.g., cascade structure, threshold design) did you lead yourself?
   A: 导师主要在三个层面给我方向性指导：一是确定"端侧 deepfake 检测"的整体方向，二是强调要用真实移动分布的 OOD 数据来评估，三是要求指标要落到业务可用的 FNR <1%、延迟 <200ms。基于这些高层约束，我自己主导了：两阶段 cost-aware 级联结构的设计（含 Stage2 触发率控制在 1.16%）、温度标定和阈值 τ_low=0.05/τ_high=0.55 的策略、以及端到端脚本化 pipeline 和 INT8 部署方案。答题时我会清晰划分"问题和目标由导师共同确定、具体技术方案和工程实现由我主导"，以体现独立性与团队协作。 / My supervisor mainly provided guidance at three levels: defining the overall direction of "on-device deepfake detection", insisting on evaluation with real mobile OOD distributions, and setting practical targets like FNR <1% and latency <200ms. Given these high-level constraints, I led the design of the two-stage cost-aware cascade (including keeping Stage 2 activation at 1.16%), the temperature calibration and thresholding strategy with τ_low=0.05/τ_high=0.55, and the end-to-end scriptable pipeline plus INT8 deployment. When answering, I clearly separate "problem and goals co-defined with my supervisor" from "technical design and implementation led by me" to show both independence and collaboration.

---

## Slide 2: Research Background

**核心要点:**
- Deepfake 技术快速发展，GAN / 扩散模型生成质量很高
- 被用于政治虚假信息、诈骗和非自愿亲密内容等高风险场景
- 超过 60% 的互联网流量来自移动设备
- 用户希望在本地完成检测，以保护隐私并降低延迟

**简化英文讲稿:**
Deepfake technology has improved a lot in recent years.
GAN and diffusion models can create faces that look very real to most people.
These fakes are used for political misinformation, financial fraud, and non-consensual intimate images.
At the same time, more than half of internet traffic now comes from mobile devices.
Users want to check if content is fake directly on their phones, without sending private videos to a server.
On-device detection helps to protect privacy and also avoids network delay and bandwidth cost.
Because of these trends, accurate and real-time deepfake detection on mobile devices has become an urgent research topic.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 为什么要强调"端侧检测"，而不是把视频上传到云端检测？
   A: 云端检测需要上传大文件，带来隐私和带宽问题，也难以保证短视频的实时响应；端侧检测可以离线工作、减少数据外流，在高风险场景再结合云端复核会更灵活。

2. Q: 你提到"60% 以上的网络流量来自移动端"，这个统计数据具体来源于哪些报告或研究？你如何确保它既可靠又和本论文的场景高度相关？ / You mention that "over 60% of internet traffic comes from mobile"; which reports or studies does this statistic come from, and how do you ensure it is both reliable and relevant to your thesis?
   A: 在论文中我会引用权威机构和运营商发布的近几年移动互联网统计报告，并明确给出年份和地理范围。答辩时我会强调，两点比精确数字更关键：一是多个独立数据源都表明"移动端已经占多数"，二是短视频和社交应用在移动端的占比尤其高，正好与我研究的 deepfake 场景匹配。同时，我会补充说明：即使统计数字有若干百分点误差，也不影响结论——用户主要在手机上拍摄和消费视频，因此把防御能力部署到手机端在趋势上是合理的。 / In the thesis I cite recent statistics from reputable organizations and operators, with explicit years and geographic scope. During the defense, I stress that two aspects matter more than the exact percentage: multiple independent sources consistently show that "mobile dominates traffic", and short-video and social apps are especially mobile-heavy, matching my deepfake scenarios. I also clarify that even if the percentage has a few points of uncertainty, the key conclusion stands—users primarily capture and consume video on phones, so placing defenses on mobile is strategically sound.

3. Q: 即便移动端流量很多，deepfake 在桌面端和云端平台上同样存在，为何你的工作选择只聚焦"移动端检测"，而不是做一个跨平台的统一方案？ / Even though mobile traffic is high, deepfakes also exist on desktop and cloud platforms. Why does your work focus only on "mobile detection" instead of a unified cross-platform solution?
   A: 我在选题时做了取舍：桌面和云端通常算力更强、网络带宽更充足，已有不少面向服务器端的 heavy 模型和审核系统；而移动端恰恰是目前最薄弱的一环，存在算力不足、网络不稳定、隐私敏感等特性，导致不能简单复用服务器方案。因此，我有意把问题缩小到"端侧单设备"的严格约束下，研究如何在 ≤100MB、<200ms 的条件下仍然把 FNR 压到 0.60%。同时，我会说明：级联结构、分布迁移评估和脚本化 pipeline 本身是可以迁移到桌面和云端的，这部分在未来工作里可以扩展成跨平台解决方案。 / I deliberately scoped the problem: desktops and servers usually have stronger compute and bandwidth, and there is already substantial work on heavy server-side models and moderation systems, whereas mobile is the weakest link, facing limited compute, unstable connectivity, and strong privacy constraints, so server solutions cannot be reused directly. I therefore focus on the strict "single device on-device" setting, studying how to keep FNR at 0.60% under ≤100MB and <200ms. I also explain that the cascade design, distribution-shift evaluation, and scriptable pipeline are transferable to desktops and servers, which is a natural direction for future cross-platform extensions.

---

## Slide 2.5: Detection Gaps and Mobile Challenges

**核心要点:**
- 基准数据集（FF++/CelebDF）到真实野外数据存在约 20–30% 的 AUC 性能掉落
- 现有研究多基于"服务器端推理"假设，忽视移动端场景
- 移动端需要小模型（≤100MB）、低时延（<200ms）、省电并保护本地隐私
- 现有级联方法在移动端缺乏对"漏检率 FNR" 的显式控制

**简化英文讲稿:**
On this slide, I explain the gap between benchmarks and the real world, especially on mobile phones.
Many deepfake detectors are trained and tested on clean datasets like FaceForensics++ and Celeb-DF.
They show high AUC on these datasets, but when we test them on wild internet videos, the AUC can drop by 20 to 30 percent.
Most past work also assumes that the model runs on a powerful server, not on a phone.
However, on mobile devices we must keep the model under about 100 megabytes, respond in under 200 milliseconds, save battery, and keep data on the device for privacy.
Existing cascade systems do not directly control the false negative rate, so they may still miss many fakes in this mobile setting.
These problems motivate our work on a mobile-friendly cascade with explicit FNR control.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: Why is on-device detection important, instead of sending videos to a server?
   A: Emphasize user privacy (no raw face video uploaded), lower latency when network is weak, and reduced server cost at scale. Some scenarios (secure messaging, offline regions) even require local processing by policy.

2. Q: 你声称从基准数据集到真实移动场景 AUC 会掉 20–30%，这个"gap"是如何量化出来的？具体用到了哪些数据和评估协议？ / You claim a 20–30% AUC drop from benchmark datasets to real mobile scenarios. How did you quantify this gap, and which datasets and evaluation protocols did you use?
   A: 我首先在若干公开基准数据集上训练并评估模型，得到接近 0.99 的 AUC（例如 Stage 1 的 AUC 为 0.9936），然后在更贴近真实移动分布的 OOD 数据上评估，如 Deepfake-Eval-2024，观察到 F1 低于 0.30、FNR 高达 73–79%。通过统一 ROC 曲线的计算和相同阈值选取策略，可以看到从"干净基准"到"真实 OOD"的 AUC 和实际业务指标（特别是 FNR）存在明显差距。我会在答辩中强调：这个 20–30% 不是单点数字，而是通过多数据集、多模型、一致评估流程得到的区间估计，用来说明分布偏移的严重性。 / I first train and evaluate models on several public benchmarks, obtaining near-0.99 AUC (e.g., Stage 1 achieves 0.9936), then evaluate on more realistic mobile-like OOD datasets such as Deepfake-Eval-2024, where F1 drops below 0.30 and FNR rises to 73–79%. Using consistent ROC computation and threshold selection across datasets, we observe a substantial gap in both AUC and practical business metrics, especially FNR, between "clean benchmarks" and "real OOD". In the defense, I emphasize that the 20–30% is not a single magic number but a range derived from multiple datasets, models, and a unified evaluation protocol, illustrating the severity of distribution shift.

3. Q: 你把移动端约束设定为"总模型体积 ≤100MB、单次推理 <200ms"，这些具体数值是怎么选出来的？是否有实验或用户研究支持，而不是拍脑袋？ / You set mobile constraints as "total model size ≤100MB and per-inference latency <200ms." How did you choose these numbers—are they empirically supported rather than arbitrary?
   A: 约束来自三方面：第一，目标设备是一台中端 Android 手机，其可用存储和内存对单应用模型大致给出了 50–100MB 的实际空间；第二，典型交互式应用的可感知延迟上界在 100–200ms 左右，我通过简单用户测试发现 200ms 以内的视频审核几乎不被察觉；第三，我在早期实验中对比了不同模型组合，在单模型大于 ~80MB 时，端上冷启动和多任务场景表现明显变差。综合这些因素，我把目标定在总模型约 87MB（37.3MB + 49.6MB）和 ~180ms 实测延迟，既符合体验，又给模型留出一定裕度。答辩时我会强调这些阈值是基于硬件规格与实验观测折中出来的，而不是凭主观感觉。 / The constraints come from three aspects: (1) the target device is a mid-range Android phone, whose available storage and RAM effectively limit a single app's model budget to roughly 50–100MB; (2) typical interactive apps aim for user-perceivable latency below 100–200ms, and simple user tests showed that video checking within 200ms is almost unnoticeable; (3) in early experiments, when a single model exceeded ~80MB, cold-start and multi-task performance deteriorated markedly. Combining these, I set the goal at a total model size around 87MB (37.3MB + 49.6MB) and ~180ms measured latency, which satisfies user experience while leaving some capacity margin. In the defense I stress that these thresholds result from hardware specs and empirical observations, not arbitrary choices.

---

## Slide 3: Problems and Challenges

**核心要点:**
- 轻量化 vs 准确率：移动端算力和模型体积受限
- 分布偏移：公开数据集与真实互联网数据差异大
- FNR 控制：漏检假脸的代价远高于误报
- 工程落地：需要可复现、可审计的完整端到端系统

**简化英文讲稿:**
Our first challenge is the trade-off between lightweight models and accuracy on mobile devices.
The whole detection module should stay under about 100MB and run in under 200ms per face.
The second challenge is distribution shift between curated datasets and real internet videos.
Models that work well on FaceForensics++ or CelebDF can lose a lot of performance on newer data like Deepfake-Eval-2024.
The third challenge is strict control of the false negative rate, because missing a fake is more serious than raising an extra false alarm.
Finally, we must turn research models into a complete pipeline that is reproducible and ready for real deployment.
The next slides will show how our design answers these four challenges step by step.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 为什么这里更关注 FNR，而不是只看 AUC 或 F1？
   A: AUC 和 F1 是整体指标，但在深伪检测里，"漏掉一个假视频"对社会和平台风险更大，所以需要把 FNR 作为硬约束控制在很低水平，再在这个前提下优化其他指标。

2. Q: 你提出了四个挑战：轻量与准确的权衡、分布偏移、FNR 控制和工程落地。面对有限时间与资源，你是如何确定这些挑战各自的优先级和投入比例的？ / You identify four challenges: lightweight vs. accuracy, distribution shift, FNR control, and engineering deployment. With limited time and resources, how did you prioritize these challenges and allocate effort?
   A: 我按"对实际风险的影响"来排序：首先是 FNR 控制，因为放过假视频的代价远高于多拦截一些真视频；这直接驱动了两阶段级联设计和 <1% 的目标。第二是分布偏移，因为在 OOD 场景下基准 AUC 的优势会大幅折扣，我用多数据集评估来量化这一点。第三是轻量与准确的权衡——只有在端侧 latency 和体积约束下依然能达到 FNR 0.60%、移动端准确率 92.8%，系统才有可用价值。最后是工程落地，它是前面三点的承载，通过脚本化 pipeline 和 INT8 部署把研究结果真正搬到手机上。答辩时我会说明：每个挑战都做了工作，但有限精力下，我把"FNR + OOD + 端侧约束"视为核心，工程工作服务于验证这些核心。 / I prioritize by "impact on real-world risk": FNR control comes first, because letting fake videos slip through is far more costly than blocking some real ones; this directly motivates the two-stage cascade and the <1% target. Second is distribution shift, since benchmark AUC advantages shrink drastically under OOD, so I use multi-dataset evaluation to quantify it. Third is the lightweight-vs-accuracy trade-off—only if we achieve 0.60% FNR and 92.8% accuracy under on-device latency and size constraints is the system truly useful. Engineering deployment comes last but acts as the carrier for the first three, via a scriptable pipeline and INT8 deployment that bring the research to actual phones. In the defense I explain that all four are addressed, but given limited bandwidth, I treat "FNR + OOD + on-device constraints" as core, with engineering serving to validate them.

3. Q: deepfake 检测还有对抗攻击、压缩伪影、机器人批量生成等很多难题，你为什么只把这四点定义为"主要挑战"？会不会有选择性忽视？ / Deepfake detection also faces adversarial attacks, compression artifacts, and bot-scale generation. Why do you define only these four as the "main challenges"? Are you selectively ignoring others?
   A: 我在论文中区分了"问题的全景"和"本工作的焦点"。对抗攻击、大规模攻击者行为等确实重要，但往往需要更长周期和更大系统来研究；而我所面对的硕士课题，需要在可控时间内在移动端落地一个可用的检测原型。因此，我优先选择了与"端侧部署"高度耦合的四个挑战——轻量、分布偏移、FNR 控制和工程实现，因为它们直接决定移动端是否可用。我会在答辩中主动承认其他挑战的重要性，并说明本工作在模型结构、pipeline 设计上为将来扩展到对抗鲁棒性和更复杂场景预留了空间，这也是未来工作的方向。 / In the thesis I separate the "full landscape of problems" from "this work's focus". Adversarial attacks and large-scale attacker behavior are indeed important, but typically require longer timeframes and larger systems to study, whereas my master's topic must deliver a usable mobile prototype within a constrained schedule. I therefore prioritize four challenges tightly coupled to on-device deployment—lightweight models, distribution shift, FNR control, and engineering—because they directly determine whether mobile detection is viable. In the defense, I explicitly acknowledge the importance of other challenges and explain that my model and pipeline design deliberately leave room to extend towards adversarial robustness and more complex scenarios as future work.

---

## Slide 3.5: Technical Challenges in Detail

**核心要点:**
- 漏检代价高，需要在算力预算内将 FNR 控制在 1% 以下
- 概率校准：用温度缩放（T≈1.34）大幅降低 ECE，提高置信度可靠性
- 鲁棒性：模型需在 JPEG 压缩、噪声、模糊等真实扰动下保持稳定
- 通过 τ_low=0.05、τ_high=0.55 等阈值设计 FNR 与二阶段调用率的权衡曲线

**简化英文讲稿:**
Now I go deeper into the main technical challenges.
First, missed fakes are very costly, so we want the false negative rate to be below one percent while staying inside a tight compute budget.
Second, the model's probabilities must be well calibrated, so that a score like 0.9 really means high confidence; we use temperature scaling with T about 1.34, which cuts the expected calibration error by around 79 percent.
Third, real media are not clean; they have JPEG compression, noise, and blur, so the detector must stay robust under these common changes.
Fourth, in a cascade, we must choose thresholds that balance how many samples go to stage two and how many fakes we might miss.
In our system, we set a lower threshold tau_low at 0.05 and a higher threshold tau_high at 0.55 to get an operating point with low FNR but still good speed.
These design choices make the system more reliable for high-risk mobile applications.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: How did you choose τ_low = 0.05 and τ_high = 0.55?
   A: We did a grid search on a validation set, observing the trade-off curves between FNR, stage-2 rate, and compute cost. These values sit near the "knee" of the curve: further lowering FNR would cause a big jump in cost.

2. Q: 你把目标 FNR 设为 <1%，为什么不是 0.1% 或 5% 这样的数值？你如何论证 1% 是一个合理而非武断的业务阈值？ / You set the target FNR to <1%. Why not 0.1% or 5%? How do you justify that 1% is a reasonable, non-arbitrary operational threshold?
   A: FNR 是"漏检假样本"的概率，我通过简单风险建模和硬件约束共同确定了 1% 这个水平：如果追求 0.1%，在当前 OOD 条件下需要显著增大模型或大幅提高 Stage2 触发率，导致延迟和能耗超出移动端可接受范围；如果放宽到 5%，在 Deepfake-Eval-2024 这种分布下会有太多高危假视频漏过，无法满足实际防御需求。实验上，我最终实现了级联 FNR 0.60%、Stage2 比例 1.16%，且在手机上仍保持约 180ms 的延迟，这说明"<1% FNR + 移动端实时性"是一个在安全性和资源消耗之间较合理的折中。答辩时我会展示对不同 FNR 目标的模拟和资源消耗分析来支撑这个选择。 / FNR is the probability of missing fake samples, and I choose 1% by combining simple risk modeling with hardware constraints: aiming for 0.1% under current OOD conditions would require much larger models or a much higher Stage 2 activation rate, pushing latency and power beyond what mobile can tolerate; relaxing to 5% would allow too many dangerous fakes through on distributions like Deepfake-Eval-2024, failing to meet practical defense needs. Experimentally, I achieve 0.60% cascade FNR and a 1.16% Stage-2 rate while keeping ~180ms latency on device, showing that "<1% FNR + real-time mobile performance" is a reasonable compromise between safety and resource cost. In the defense I plan to show simulations of different FNR targets and associated resource usage to justify this choice.

3. Q: 你为什么需要做温度标定（T≈1.34），而不是直接在原始 logits 上选定阈值？标定具体解决了什么问题？ / Why do you need temperature calibration (T≈1.34) instead of choosing thresholds directly on raw logits? What problem does calibration actually solve?
   A: 原始 logits 在不同数据集和训练轮次之间往往校准不一致，比如同样 0.9 的分数，在某些数据集上可能对应 99% 的真实概率，在 OOD 数据上却只有 70%。这会让固定阈值 τ_low=0.05/τ_high=0.55 的语义不稳定，影响级联中"早拒绝/早接受"的行为。因此，我在验证集上学习一个温度 T≈1.34，使得输出概率在全程上更接近真实频率，从而让阈值与期望的 FNR/FPR 更对齐。答辩时我会展示校准前后 reliability diagram 和 FNR 曲线，说明标定如何减少跨数据集和跨设备时的性能波动，提高级联在实战中的可控性。 / Raw logits are often poorly calibrated across datasets and training runs; for example, a score of 0.9 may correspond to a 99% true probability on one dataset but only 70% on OOD data. This makes fixed thresholds like τ_low=0.05 and τ_high=0.55 semantically unstable, undermining the cascade's "early reject/early accept" behavior. I therefore learn a temperature T≈1.34 on a validation set so that predicted probabilities better match empirical frequencies, aligning thresholds with the desired FNR/FPR trade-offs. In the defense I plan to show reliability diagrams and FNR curves before and after calibration to demonstrate how it reduces performance variance across datasets and devices, making the cascade more controllable in practice.

---

## Slide 4: Research Objectives and Contributions

**核心要点:**
- 成本感知的两阶段级联系统：MobileNetV4 + EfficientNetV2-B3
- 四个公开数据集联合训练，Deepfake-Eval-2024 做跨数据集评估
- 从预处理到移动导出的六阶段可复现流水线
- 部署到 Android，约 180ms 延迟、92.8% 准确率

**简化英文讲稿:**
This slide summarizes the main objectives and contributions of the thesis.
First, we design a cost-aware two-stage cascade for deepfake detection on mobile devices.
Stage 1 is a lightweight MobileNetV4 filter, and Stage 2 is an EfficientNetV2-B3 expert that focuses on hard cases.
Second, we train on four public datasets and evaluate both in-domain and on the out-of-distribution benchmark Deepfake-Eval-2024.
Third, we build a six-stage pipeline from raw videos, to calibrated models, to robustness tests, so that the whole system is fully scriptable and reproducible.
Fourth, we deploy the cascade to an Android app, reaching about 92.8% face-level accuracy with around 180ms latency per face on a Xiaomi 13.
Together, these contributions connect model design, fair evaluation, and a real mobile deployment.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 和已有的级联检测或移动检测工作相比，你的工作主要新在哪里？
   A: 我这里把"端侧约束 + 显式 FNR 控制 + 跨数据集 OOD 评估 + 真机部署"放在一起做：不仅给出级联结构，还用双阈值网格搜索严格控制 FNR，并用 Deepfake-Eval-2024 系统性量化在真实 2024 媒体上的退化，这在现有移动端工作中比较少见。

2. Q: 你的四项贡献——级联结构、多数据集评估、脚本化 pipeline 和移动端部署——与已有端侧 deepfake / 媒体取证工作相比，新颖性具体体现在哪里？ / Your four contributions—cascade design, multi-dataset evaluation, scriptable pipeline, and mobile deployment—how are they concretely novel compared to existing on-device deepfake or media forensics work?
   A: 在级联方面，我不仅做了简单的"轻量 + 重模型"拼接，而是显式建模了 cost-aware 策略：控制 Stage2 激活率为 1.16%，在保持整体 FNR 0.60% 的前提下最小化端侧计算开销。在评估方面，我系统性地把常见基准和 Deepfake-Eval-2024 等 OOD 数据纳入统一 pipeline，量化了从基准到真实场景的性能坍塌。pipeline 部分，我提供了从数据预处理、训练、标定到量化、打包的一键脚本，使得别人可以在相同配置下复现实验。最后，移动端部署不是简单 demo，而是在资源约束下实现了 92.8% 的端侧准确率和 ~180ms 延迟。 / For the cascade, I go beyond simply chaining a light and a heavy model by explicitly modeling a cost-aware strategy: controlling Stage-2 activation at 1.16% and minimizing on-device compute while keeping overall FNR at 0.60%. For evaluation, I systematically integrate common benchmarks and OOD datasets like Deepfake-Eval-2024 into a unified pipeline to quantify performance collapse from benchmarks to real scenarios. In the pipeline, I provide end-to-end scripts so others can reproduce experiments under the same configs. Finally, the mobile deployment achieves 92.8% on-device accuracy with ~180ms latency under resource constraints.

3. Q: 如果必须按"学术和实际影响力"给这四项贡献排序，你会如何排序？理由是什么？ / If you had to rank these four contributions by academic and practical impact, how would you order them and why?
   A: 从实际影响看，我会把排名定为：① cost-aware 级联设计（直接带来 FNR 0.60% 与 Stage2 1.16% 的组合）；② 多数据集、含 OOD 的评估框架（揭示现有方法在真实分布下严重退化）；③ 移动端部署结果（证明在真实手机上可以达到 92.8% 准确率和 ~180ms 延迟）；④ 脚本化 pipeline（提升可复现性和工程可用性）。从学术角度，前两项——级联与 OOD 评估——提供了比较清晰的方法学贡献，后两项则更偏工程和系统侧。 / From a practical perspective, I would rank them as: (1) the cost-aware cascade design, which directly delivers the 0.60% FNR and 1.16% Stage-2 combination; (2) the multi-dataset, OOD-inclusive evaluation framework; (3) the mobile deployment results showing 92.8% accuracy and ~180ms latency are achievable; and (4) the scriptable pipeline improving reproducibility. From an academic standpoint, the first two offer clearer methodological contributions, while the latter two are more systems-oriented.

---

## Slide 4.5: Contributions – Technical Details

**核心要点:**
- 提出代价感知级联框架，以一级泄露率和二级升级率作为关键系统指标
- 在跨数据集上系统评估：单模型 vs 级联、校准 vs 未校准
- 搭建可脚本化流水线：预处理、困难样本挖掘（HEM）、校准与鲁棒性扫描一体化
- 实现移动端导出：INT8 量化，TorchScript + ONNX，元数据文件与 Android 模板工程

**简化英文讲稿:**
This slide explains the technical side of our contributions.
We design a cost-aware cascade, where we track both stage-1 leakage and stage-2 escalation rate as key system metrics.
We run cross-dataset experiments to measure the domain gap and to compare single models versus cascades, and calibrated versus uncalibrated versions.
To support this, we build a scriptable pipeline that handles data preprocessing, hard example mining, calibration, and robustness sweeps in one place.
We also focus on deployment, exporting the models to mobile in INT8 format using TorchScript and ONNX, together with metadata files.
Finally, we provide an Android template app so the cascade can be integrated and tested quickly on real phones.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: What do you mean by "cost-aware" in your cascade design?
   A: "Cost" mainly means compute and latency on mobile, not just accuracy. We explicitly measure GFLOPs and stage-2 escalation rate, and design the cascade to keep these costs low while enforcing a target FNR.

2. Q: 在你的 cost-aware 级联设计中，"代价"是如何被形式化建模的？你如何证明它确实节省了端侧算力，而不是只是多写了一个公式？ / In your cost-aware cascade design, how is "cost" formally modeled, and how do you show it truly saves on-device computation rather than just adding a formula?
   A: 我把总期望成本建模为 E[cost] = p_1·C_1 + p_2·C_2，其中 C_1、C_2 分别是 Stage1 和 Stage2 的推理延迟或 FLOPs，p_1≈1，p_2 是进入 Stage2 的比例。通过调整阈值 τ_low、τ_high，我把 p_2 控制到 1.16%，在真实手机上测得平均延迟约 180ms。如果在相同设备上直接用 Stage2 模型单独运行，延迟和功耗都明显增加。答辩时我会展示"单模型 vs 级联"的端侧测评表，包含延迟、功耗和温度变化，以证明 cost-aware 设计不仅在理论上合理，而且在真实设备上带来了可观的资源节约。 / I model expected total cost as E[cost] = p₁·C₁ + p₂·C₂, where C₁ and C₂ are Stage-1 and Stage-2 inference latency or FLOPs, p₁≈1, and p₂ is the proportion escalated to Stage 2. By tuning τ_low and τ_high, I keep p₂ at 1.16% and measure ~180ms average latency on a real phone. Running Stage 2 alone on the same device yields noticeably higher latency and power. In the defense, I will show a table comparing single-model vs. cascade on device, including latency, power, and temperature, to demonstrate that the cost-aware design provides concrete resource savings.

3. Q: INT8 量化通常会带来性能损失，你在 MobileDeepfake 中量化后具体损失了哪些指标？在 FNR 和延迟之间，你是如何权衡并证明这个 trade-off 是值得的？ / INT8 quantization typically hurts performance. In MobileDeepfake, what metrics did you actually lose after quantization, and how did you justify the FNR vs. latency trade-off?
   A: 在量化实验中，我对 Stage1 和 Stage2 分别做了 INT8 量化，观察到离线 AUC 和准确率有小幅下降，但通过温度标定与阈值微调，级联整体仍保持 FNR 0.60%、端侧准确率 92.8%，同时延迟大幅降低到约 180ms，模型体积也缩减到适合手机分发的水平。相比全精度模型，INT8 版本在极端 OOD 情况下 F1 略有下降，但考虑到未经量化的模型难以在大规模设备上部署，我认为这种损失是可以接受的。答辩时我会展示"FP 模型 vs INT8 模型"的对比曲线和表格，清楚说明"略降的精度换来可部署性和能耗优势"，这是一个经过实验验证的折中。 / In quantization experiments, I apply INT8 to both stages and observe a small drop in offline AUC and accuracy, but with temperature calibration and threshold retuning, the cascade still achieves 0.60% FNR and 92.8% on-device accuracy, while latency drops to ~180ms and model size shrinks to a mobile-friendly footprint. Compared with full-precision models, the INT8 version shows slightly lower F1 in extreme OOD cases, but given that non-quantized models are hard to deploy across many devices, I consider this loss acceptable. In the defense, I will present comparison curves and tables of "FP vs INT8" to clearly show that "slightly lower accuracy in exchange for deployability and energy savings" is an experimentally supported trade-off.

---

# Part 2: Method (7 min)

## Slide 5: Two-Stage Cascade Design

**核心要点:**
- Stage 1：MobileNetV4，约 37.3MB，AUC≈0.9936，负责快速过滤
- Stage 2：EfficientNetV2-B3，约 49.6MB，AUC≈0.9633，只处理不确定样本
- 级联后 FNR≈0.6%，Stage2 升级率≈1.2%
- 大部分样本只经过 Stage1，少数难例交给 Stage2，兼顾效率与安全

**简化英文讲稿:**
The core idea of our method is to split detection into two stages.
Stage 1 is a lightweight MobileNetV4 model that takes 256×256 face crops and outputs a fake probability.
It is about 37MB and very accurate, with AUC close to 0.994, and it is designed to run fast on mobile.
Stage 2 is a larger EfficientNetV2-B3 model, about 50MB, that only processes samples that Stage 1 finds uncertain.
When we combine the two models in a cascade, the system reaches a false negative rate of about 0.6%.
Only around 1.2% of faces are sent to Stage 2, so most samples are decided by the fast first stage.
This design keeps average compute low while still catching difficult deepfakes that Stage 1 is not sure about.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 为什么选择 MobileNetV4 和 EfficientNetV2-B3 这两个骨干，而不是别的模型？
   A: MobileNetV4 在移动端延迟和精度上表现很好，适合作为快速过滤器；EfficientNetV2-B3 在中等规模下有较强的表征能力，适合作为只处理少量难例的"专家"，两者组合在 GFLOPs 和精度之间取得了比较好的折中。

2. Q: 既然 Stage 1 的 MobileNetV4 本身 AUC 已经达到 0.9936，为什么不直接用一个更大的单模型（比如更深的 EfficientNet 变体）来同时满足精度和计算约束，而要做两阶段级联？ / Given that Stage 1 MobileNetV4 already achieves an AUC of 0.9936, why not just use a single larger model to meet both accuracy and compute constraints instead of a two-stage cascade?
   A: 在移动端约束下，我们发现单个更大的模型很难同时做到"接近 0.60% 的 FNR"和"接近 0.59 GFLOPs 的计算预算"：如果把模型做大到能逼近级联的 FNR，往往 GFLOPs 会接近甚至超过 Stage 2 的 2.87 GFLOPs；如果强行压缩计算量，又会在困难样本上丢失明显的召回率。两阶段级联的核心优势是：Stage 1 覆盖绝大多数易判样本，Stage 2 只针对 1.16% 的棘手样本做"高成本复核"，从而在总体计算几乎不变的情况下大幅降低 FNR。 / Under mobile constraints, we observed that a single larger model struggles to simultaneously achieve "FNR close to 0.60%" and "compute close to 0.59 GFLOPs": if we scale the model to approach the cascade's FNR, its GFLOPs typically approach or exceed Stage 2's 2.87 GFLOPs; if we restrict compute, it loses recall on hard examples. The key advantage of the two-stage cascade is that Stage 1 handles almost all easy samples, while Stage 2 performs a "high-cost review" on only 1.16% of hard cases, drastically reducing FNR with almost unchanged overall compute.

3. Q: 从理论上看，级联是如何在 Stage 1 AUC 更高（0.9936）而 Stage 2 AUC 更低（0.9633）的情况下，反而把整体 FNR 压到 0.60% 的？ / Theoretically, how can the cascade reduce the overall FNR to 0.60% even though Stage 1 has higher AUC (0.9936) than Stage 2 (0.9633)?
   A: AUC 反映的是整体排序能力，但我们在级联里只让 Stage 2 处理 Stage 1 最不确定的那一小段分布区间，而不是全体样本。Stage 2 在该"灰区"上经过有针对性的再训练（例如更重的假样本采样或代价敏感损失），所以在这个局部子分布上，Stage 2 的错判模式与 Stage 1 有很强互补性：它可以纠正大量 Stage 1 的漏检，而不会显著引入新的漏检。这样，全局看虽然 Stage 2 单独的 AUC 较低，但"局部专家 + 全局门控"的组合，使得级联整体的 FNR 反而显著优于任何单一模型。 / AUC reflects global ranking ability, but in the cascade we only send to Stage 2 the small "uncertain region" of the Stage 1 score distribution, not all samples. Stage 2 is retrained to specialize on this gray region, so on this local sub-distribution Stage 2's error patterns are highly complementary to Stage 1. Globally, although Stage 2's standalone AUC is lower, this "local expert + global gate" design yields an overall FNR that is significantly better than any single model.

---

## Slide 5.5: Cascade Efficiency & Operating Points

**核心要点:**
- 单独 S1：0.54 GFLOPs / 3.12% FNR；单独 S2：2.87 GFLOPs / 9.43% FNR；级联：0.59 GFLOPs / 0.60% FNR
- INT8 量化将模型从 37.3→10.1MB、49.6→13.4MB，AUC 损失 <0.01
- 设计三种部署策略：安全优先（0.05/0.55）、均衡（0.10/0.50）、速度优先（0.15/0.45）
- 校准（T≈1.34）进一步提升分数可靠性，ECE 下降约 79%

**简化英文讲稿:**
Here I show how efficient the cascade is and how we choose operating points.
If we only use stage one, we spend about 0.54 GFLOPs and get a false negative rate of 3.12 percent; if we only use stage two, we spend 2.87 GFLOPs and still get 9.43 percent FNR.
With our cascade, we use only 0.59 GFLOPs but reduce the FNR to 0.60 percent, so we are both cheaper and more accurate than either stage alone.
Then we apply INT8 quantization, which shrinks the models from 37.3 and 49.6 megabytes down to 10.1 and 13.4 megabytes, with less than 0.01 loss in AUC.
We define three deployment modes: a safety-first mode with thresholds 0.05 and 0.55, a balanced mode with 0.10 and 0.50, and a speed-first mode with 0.15 and 0.45.
In all modes, calibration with temperature about 1.34 keeps the scores reliable and lowers the calibration error by around 79 percent.
This gives practitioners a clear way to trade off safety and speed on mobile devices.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: How do you make sure that INT8 quantization does not harm detection performance?
   A: We use post-training quantization with calibration data, and we carefully compare AUC and FNR before and after quantization. The measured AUC loss is under 0.01, so the size and speed gains clearly outweigh the tiny accuracy drop.

2. Q: 你声称级联的平均计算量只有 0.59 GFLOPs，比单独运行 Stage 2 的 2.87 GFLOPs 低很多，但比 Stage 1 的 0.54 GFLOPs 略高一点。这个"略高一点"的开销在严格的移动端功耗预算下真的可以忽略吗？ / You claim the cascade averages 0.59 GFLOPs, much lower than Stage 2 alone. Under strict mobile power budgets, is this extra compute truly negligible?
   A: 0.59 GFLOPs ≈ 0.54（Stage 1 必跑）+ 2.87×0.0116（Stage 2 仅在 1.16% 的样本上运行），因此额外部分主要来源于极少量样本的深度复核。从能耗角度看，它带来的增量远小于典型移动应用中一次中等分辨率卷积的开销，却换来了数量级更低的 FNR。对比"只跑 Stage 2"的方案，我们在相同的能耗预算下把平均计算降低了约 5 倍以上。 / 0.59 GFLOPs ≈ 0.54 (Stage 1 always runs) + 2.87×0.0116 (Stage 2 runs on only 1.16% of samples), so the extra cost mainly comes from deep re-checks on very few samples. From an energy perspective, this increment is much smaller than the cost of a typical mid-resolution convolution step, yet it buys an order-of-magnitude reduction in FNR. Compared with "Stage 2 only", we reduce average compute by more than 5× for the same budget.

3. Q: 你提到了三种不同的部署策略，对应不同的 operating point。请具体说明这些点是如何选出来的，它们在计算量、FNR 和用户体验之间的权衡思路是什么？ / You mentioned three deployment strategies corresponding to different operating points. How were these chosen, and what trade-offs do they make among compute, FNR, and user experience?
   A: 我们先在验证集上扫描整条级联 ROC–Compute 曲线，对每个候选阈值组合估计"平均 GFLOPs–FNR–Stage 2 调用率"三者的关系，然后结合实际业务场景选出三类 Pareto optimal 点：偏安全的点优先最小化 FNR，即使 Stage 2 调用率略上升；偏实时的点严格限制平均 GFLOPs 和 tail latency；平衡点则在两者之间。对于移动端实时检测，我们默认采用平衡点，同时在系统层面暴露配置接口，让不同 App 可以根据自身对安全性和延迟的容忍度选择。 / We first sweep the full cascade ROC–compute curve on the validation set and, for each candidate threshold pair, estimate the relationship among average GFLOPs, FNR, and Stage 2 rate, then select three Pareto-optimal points: security-biased minimizes FNR even if Stage 2 runs more often; latency-biased strictly bounds average GFLOPs and tail latency; balanced lies between. For mobile real-time detection, we default to balanced, while exposing a configuration interface so apps can choose based on their tolerance for security vs. latency.

---

## Slide 6: Cascade Decision Logic

**核心要点:**
- Stage1 输出假脸概率 p₁(x)
- 使用双阈值：τ_low = 0.05，τ_high = 0.55
- p₁(x) < τ_low 判为真；p₁(x) > τ_high 判为假；中间区间升级 Stage2
- Stage2 使用 0.5 阈值给出最终判决；该操作点下 FNR≈0.6%，升级率≈1.2%

**简化英文讲稿:**
This slide explains how the cascade makes decisions.
Stage 1 outputs a fake probability p one of x, written as p₁(x), for each face.
We use two thresholds: a low threshold τ_low of 0.05 and a high threshold τ_high of 0.55.
If p₁(x) is below 0.05, we treat the face as real; if p₁(x) is above 0.55, we treat it as fake.
Only faces in the middle range between 0.05 and 0.55 are sent to Stage 2 for a second check.
Stage 2 then uses a simple 0.5 threshold on its own probability to decide real or fake.
With this dual-threshold rule, the cascade keeps the false negative rate around 0.6%, while routing only about 1.2% of faces to Stage 2.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 这两个阈值 0.05 和 0.55 是怎么选出来的？如果换一个平台要怎么调？
   A: 我在验证集上对一系列 (τ_low, τ_high) 做网格搜索，在"FNR ≤ 1%"的约束下最小化 Stage2 升级率，得到目前的安全优先点；如果换数据或设备，可以在新验证集上用同样流程重新搜索一组阈值。

2. Q: 如果把 Stage 1 的双阈值从 (0.05, 0.55) 稍微移动，比如变成 (0.1, 0.6) 或 (0.03, 0.5)，系统的 FNR 和 Stage 2 调用率会有多敏感？ / If we slightly change Stage 1's dual thresholds from (0.05, 0.55) to, say, (0.1, 0.6), how sensitive are FNR and the Stage 2 rate?
   A: 我们做了系统的网格搜索和敏感性分析，观察到在一段合理范围内（例如两端阈值各自微调几个百分点）FNR 和 Stage 2 调用率是平滑变化的，而不是出现"临界点式"的剧烈跳变。当前选择的 (0.05, 0.55) 落在一个"平台区间"内：在这个区间内稍作变动不会破坏 0.60% 级别的 FNR，同时平均 GFLOPs 只缓慢上升或下降。因此，在真实环境中即便数据分布有轻微漂移，我们通过定期复核阈值或做小幅自动校准，就可以保证性能不会出现不可控的波动。 / We performed systematic grid search and sensitivity analysis and found that within a reasonable range (e.g., moving each threshold by a few percentage points), FNR and the Stage 2 rate change smoothly rather than exhibiting critical-point jumps. The chosen (0.05, 0.55) lies in a "plateau region": small perturbations do not break the ~0.60% FNR level, while average GFLOPs only change gradually.

3. Q: 为什么不用基于不确定性或熵的路由方式（比如最大熵、最小分类 margin），而是仅仅通过两个固定的概率阈值来决定是否进入 Stage 2？ / Why not use uncertainty- or entropy-based routing instead of just two fixed probability thresholds to decide whether to invoke Stage 2?
   A: 在二分类场景中，基于预测概率的双阈值其实可以看成是一种极简的"不确定性区间"近似：低于 0.05 和高于 0.55 的样本都被认为是"低熵"的，而中间区间则是高不确定性的候选。相比显式计算熵或 margin，双阈值只需要一次比较操作，没有 log 或额外算子，对移动端更友好。我们实测发现，在经过温度标定后，基于熵的路由和基于概率阈值的路由在 Stage 2 调用率和最终 FNR 上几乎重合，因此我们选择了实现更简单、可解释性更强的双阈值方案。 / In binary classification, dual probability thresholds can be viewed as an extremely simple approximation of an "uncertainty band": samples below 0.05 or above 0.55 are treated as low-entropy, while those in between are candidates with higher uncertainty. Compared to explicitly computing entropy or margin, dual thresholds require only comparisons, with no logs or extra operators, which is more mobile-friendly. In our experiments, after calibration, entropy-based and threshold-based routing produce almost identical Stage 2 rates and final FNR, so we prefer the simpler, more interpretable dual-threshold scheme.

---

## Slide 7: Six-Stage Pipeline

**核心要点:**
- Stage0：多后端人脸检测、256×256 裁剪、manifest 生成
- Stage1/2：分别训练 MobileNetV4 和 EfficientNetV2-B3，并做温度校准
- Stage4：对 τ_low / τ_high 网格搜索，控制 FNR 和 Stage2 升级率
- Stage5：在 JPEG、噪声、模糊等扰动下评估鲁棒性
- Stage6：导出 TorchScript / ONNX，并打包 Android 移动端 bundle

**简化英文讲稿:**
The whole system is organized as a six-stage pipeline.
Stage 0 performs face detection with multiple backends, crops faces to 256×256, and builds unified manifest files.
Stages 1 and 2 train the MobileNetV4 and EfficientNetV2-B3 models and apply temperature scaling so that their probabilities are well calibrated.
Stage 4 runs a grid search over τ_low and τ_high on the combined validation set to meet an FNR target while keeping the Stage-2 rate low.
Stage 5 evaluates robustness under JPEG compression, Gaussian noise, blur, and brightness changes.
Stage 6 exports the trained models to TorchScript and ONNX and creates a mobile bundle for Android.
Because every step is controlled by scripts and JSON configs, other researchers can reproduce our results from raw data all the way to the deployed mobile app.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 如果别人想完整复现你的工作，大概需要关注哪几步或哪类脚本？
   A: 关键是三类：一是预处理和 manifest 生成脚本（统一数据）；二是 Stage1/Stage2 训练和校准脚本；三是阈值搜索与导出脚本（生成 cascade_config 和 ONNX bundle），这三块串起来基本就能从原始数据跑到手机端模型。

2. Q: 你的 pipeline 里有从"阈值调优"到"鲁棒性分析"等多个阶段，这些阶段是不是有冗余？如果删除其中一个阶段会造成什么实质性的差异？ / Your pipeline includes multiple stages from "threshold tuning" to "robustness analysis". Are some of these redundant? What differences would you see if you removed one stage?
   A: 实验表明，拿掉阈值调优阶段，虽然单模型 AUC 不变，但在级联层面 FNR–GFLOPs 曲线会明显偏离 Pareto 前沿：要保持原有 FNR 就需要更高的 Stage 2 调用率。类似地，如果只在训练中做数据增强而不做独立鲁棒性分析，很难识别出某些"系统性弱点"（比如对特定压缩伪造或特定分辨率变化的敏感性），导致上线后在这些分布上 FNR 明显升高。每一个 stage 都是在解决不同尺度的问题：训练阶段优化参数，阈值调优对齐业务指标，鲁棒性分析对齐分布移位和对抗风险。 / Experiments show that removing the threshold tuning stage leaves single-model AUC unchanged but pushes the cascade's FNR–GFLOPs curve away from the Pareto front. Similarly, relying only on training-time augmentation without standalone robustness analysis makes it difficult to uncover "systematic weaknesses", leading to noticeably higher FNR on those distributions after deployment. Each stage addresses a different scale of the problem: training optimizes parameters, threshold tuning aligns with business metrics, and robustness analysis aligns with distribution shift and adversarial risk.

3. Q: 为什么不把整个流水线简化成"端到端单阶段训练 + 一次性导出"，而是设计一个看起来更复杂的六阶段流程？ / Why not simplify the entire workflow into "single-stage end-to-end training plus one-shot export" instead of a seemingly more complex six-stage process?
   A: 真正的"端到端"方案很难显式建模后半段不可导、强依赖平台的步骤，比如量化误差、移动端 runtime 的运算顺序、不同 SoC 上的调度差异等；强行把这些都塞进一个训练任务，既不现实也不利于调试。我们将问题拆成六个阶段，是为了在每个阶段都能定义清晰的输入输出与指标，使得模型工程师、系统工程师和产品团队可以并行协作，并对各自负责的环节做回归测试。总体来看，这种"模块化复杂度"换来了部署和迭代的简单：当硬件、runtime 或安全需求变化时，只需要局部替换对应 stage，而不必从头训练和验证整个系统。 / A truly end-to-end solution can hardly model the later non-differentiable, platform-dependent steps—such as quantization error, mobile runtime execution order, or scheduling differences across SoCs. We decomposed the problem into six stages so that each has clear inputs, outputs, and metrics, enabling model engineers, systems engineers, and product teams to collaborate in parallel. Overall, this "modular complexity" buys simplicity in deployment and iteration: when hardware, runtime, or security requirements change, we can replace only the relevant stage instead of retraining the entire system from scratch.

---

## Slide 7.5: Mobile Deployment Details

**核心要点:**
- 模型导出为三种格式：FP32/INT8 TorchScript 和 FP32 ONNX，便于不同端部署
- 手机端流水线：人脸检测 → 256×256 裁剪 → 归一化 → Stage1 → 级联路由 → Stage2
- 小米 13 上整体准确率约 92.8%，漏报率 2.5%，误报率 12.5%，总时延约 180ms
- 只有 7.2% 的样本会进入 Stage2，保证计算效率
- 运行环境基于 ONNX Runtime Android，CPU 4 线程，无需 GPU

**简化英文讲稿:**
On this slide, I share how we run the model on a phone.
We export the model in three ways: FP32 and INT8 TorchScript, and FP32 ONNX.
On the device, we first use a face detector, then crop to 256 by 256, and apply ImageNet normalize.
The image goes into Stage 1, and only hard cases are sent to Stage 2.
On a Xiaomi 13 phone, we get 92.8 percent accuracy, with 2.5 percent false negatives and 12.5 percent false positives.
Only about 7.2 percent of faces need Stage 2, so compute cost stays low.
We run this with ONNX Runtime on Android, using the CPU with four threads and no GPU.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: Why do you still see a 12.5% false positive rate on-device?
   A: Mobile uses CPU and ONNX which can change calibration slightly. Also domain differences between training data and mobile camera data raise FPR. Better calibration on mobile data or light threshold tuning could reduce it.

2. Q: 在移动端 pipeline 里，人脸检测通常是瓶颈之一。你使用的检测器是现成方案还是自研的？为什么不换成一个更轻量的 detector？ / In the mobile pipeline, face detection is often a bottleneck. Are you using an off-the-shelf detector or a custom one? Why not replace it with an even lighter detector?
   A: 我们选择了在工业界广泛验证过的轻量级人脸检测器，并保留厂商 SDK 的大量优化（比如硬件加速、内存复用），主要原因有两点：第一，在当前配置下，小米 13 的 180ms 总延迟中，人脸检测只占一小部分，真正主导延迟的是后端分类和数据搬运，因此在 detector 上进一步压缩收益有限；第二，人脸检测错误会放大后端所有模块的风险，而成熟方案在不同光照、角度和设备上的稳定性已经被大规模验证。 / We use a widely adopted lightweight face detector and leverage many vendor SDK optimizations. On the Xiaomi 13, face detection accounts for only a small fraction of the 180 ms total latency, with back-end classification dominating, so further shrinking the detector yields limited benefit. Also, detection errors amplify risk across all downstream modules, while mature detectors have been extensively validated for stability.

3. Q: 目前只在小米 13 上报告了 92.8% 的准确率和 180ms 的延迟，这对更低端或不同厂商的设备来说代表性如何？ / You only report 92.8% accuracy and 180 ms latency on the Xiaomi 13. How representative is this for lower-end or different-vendor devices?
   A: 小米 13 使用的是较新的高端 SoC，因此可以视为我们 pipeline 在旗舰机上的一个"上限表现"：在中低端机型上，同样的 0.59 GFLOPs 计算量会带来更长的延迟，极端情况下可能需要调整 operating point（例如提高 Stage 1 阈值、降低 Stage 2 调用率）来保证实时性。另一方面，我们在设计导出和 runtime 时刻意避免依赖特定厂商的专有加速，只使用 ONNX Runtime 的通用算子，这提升了跨设备可移植性，但也意味着没有充分利用某些设备的专用 NPU 加速。 / The Xiaomi 13 uses a recent high-end SoC, so it represents an "upper bound" of our pipeline on flagship devices: on mid- to low-end phones, the same 0.59 GFLOPs will result in longer latency, and in extreme cases we may need to adjust the operating point (e.g., higher Stage 1 thresholds) to maintain real-time performance. On the other hand, our export and runtime choices intentionally avoid vendor-specific accelerators and rely on ONNX Runtime's generic operators, which improves cross-device portability but also means we do not fully exploit dedicated NPUs on some devices.

---

## Slide 8: Mobile Architecture

**核心要点:**
- 提供 FP32 TorchScript、INT8 TorchScript、FP32 ONNX 三种模型格式
- 当前 Android 应用使用 FP32 ONNX：Stage1≈37.5MB，Stage2≈52MB
- 端侧流程：人脸检测→256×256 裁剪→归一化→Stage1→级联路由→Stage2
- 使用 ONNX Runtime Android 1.19.0（CPU，多线程），小米 13 上总延迟约 180ms、准确率约 92.8%

**简化英文讲稿:**
This slide shows the mobile architecture of MobileDeepfake.
We export models in three formats: FP32 TorchScript, INT8 TorchScript, and FP32 ONNX.
The current Android app uses FP32 ONNX models, about 37.5MB for Stage 1 and 52MB for Stage 2.
On the device, the pipeline is: detect faces, crop them to 256×256, apply ImageNet-style normalization, run Stage 1, and only then run Stage 2 when needed.
All inference is done through ONNX Runtime Android 1.19.0 on the CPU, typically with four threads.
The app follows a layered design with a camera and UI layer, a preprocessing layer, and an inference engine layer.
On a Xiaomi 13, this architecture reaches about 92.8% face-level accuracy with around 180ms end-to-end latency per face.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 既然已经有 INT8 TorchScript 模型，为什么 Android 端目前仍然使用 FP32 ONNX？
   A: 这版实现优先选择数值行为最接近训练代码的 FP32 ONNX 作为稳定基线，方便对照论文结果；INT8 TorchScript 主要面向更严苛的体积约束场景，后续可以在 ONNX Runtime 上进一步调通量化推理以节省更多存储和算力。

2. Q: 既然移动端生态中 TensorFlow Lite 非常成熟，为什么没有直接导出 TFLite 模型，而是坚持走 PyTorch→ONNX→ONNX Runtime 的路线？ / Given that TensorFlow Lite is very mature in the mobile ecosystem, why not export TFLite models directly instead of the PyTorch→ONNX→ONNX Runtime route?
   A: 我们从 PyTorch 出发训练模型，ONNX 提供了一个相对中立的中间表示，可以在不改变训练栈的前提下同时覆盖服务器端和多种移动端 runtime；如果再引入一条 TFLite 导出链路，就需要维护两套导出和量化配置，并在两条路径上分别做一致性验证，工程成本和出错风险都显著增加。另一方面，ONNX Runtime 在 Android 上已经能比较充分地优化我们使用的算子组合，使得在 CPU 上也能满足 180ms 的延迟目标，所以在现阶段我们优先选择"单一 IR + 多端 runtime"的策略。 / We train models in PyTorch, and ONNX provides a relatively neutral intermediate representation that lets us cover both server and multiple mobile runtimes without changing the training stack. Introducing a separate TFLite export path would require maintaining two sets of export/quantization configs and consistency checks on both, significantly increasing engineering cost and error risk. Meanwhile, ONNX Runtime on Android already optimizes our operator subset well enough to meet the 180 ms CPU latency target, so we prioritize a "single IR + multi-platform runtime" strategy at this stage.

3. Q: 在小米 13 上，你的系统在 CPU 上已经达到 180ms 左右的延迟。为什么没有进一步利用 GPU 或 NPU 做加速？ / On the Xiaomi 13, your system already reaches about 180 ms latency on CPU. Why not further utilize GPU or NPU for acceleration?
   A: 一方面，GPU/NPU 的确可以降低单次推理延迟，但在移动端实测中，调度、设备唤醒和内存拷贝带来的额外开销会显著放大实现复杂度，而且不同厂商、不同 SoC 上的驱动和 API 差异非常大，这与我们"单一架构、可广泛部署"的目标相冲突。另一方面，目前 180ms 的延迟已经在用户可接受范围内，且我们只使用了 4 层相对简单的架构和 INT8 量化，这为未来在确有需求的场景下迁移到 NPU 留出了空间。当前阶段，我们更看重在 CPU 上的可预测性和稳定性：CPU 资源更通用，也更便于和其他前台任务协同调度。 / On one hand, GPU/NPU can indeed reduce per-inference latency, but in mobile practice, additional overhead from scheduling, device wake-up, and memory transfers significantly increases implementation complexity, and driver/API differences across vendors and SoCs conflict with our goal of a "single architecture, widely deployable" system. On the other hand, the current 180 ms latency is already acceptable and achieved with a relatively simple 4-layer architecture and INT8 quantization, leaving headroom to move to NPUs later if needed. At this stage, we value predictability and stability on CPU more: CPU resources are more universally available and easier to co-schedule with other foreground tasks.

---

# Part 3: Data & Experiments (3 min)

## Slide 9: Dataset Introduction

**核心要点:**
- 训练与验证使用 4 个公开数据集，总计约 2.66M 人脸裁剪
- 各数据集覆盖不同伪造方法和拍摄/扰动条件（CelebDF-v2, FF++, DFDC, DeeperForensics）
- Deepfake-Eval-2024 作为 OOD 测试集，约 45 万 val / 40 万 test
- Eval-2024 来自 88 个网站、52 种语言，仅用于评估，不参与训练或微调

**简化英文讲稿:**
Now I introduce the datasets used in this work.
For training and in-domain evaluation, we use four public datasets: CelebDF-v2, FaceForensics++, DFDC, and DeeperForensics-1.0.
Together they provide about 2.6 million face crops across train, validation, and test splits.
Each dataset brings different properties, such as high-quality celebrity swaps, multiple manipulation methods, large-scale diverse videos, and real-world capture or compression artifacts.
For out-of-distribution testing, we use Deepfake-Eval-2024, with about 452k validation and 402k test face crops.
This benchmark collects videos from 88 websites and 52 languages, and reflects the 2024 real media environment.
We strictly use Deepfake-Eval-2024 only for evaluation, without any training or fine-tuning, so that the OOD test remains clean and fair.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 为什么选择这四个训练数据集，而不是再加入更多数据？
   A: 这四个数据集都是公开、广泛使用的基准，覆盖不同伪造方法和拍摄条件，便于和已有工作对比；同时在算力和存储预算下，再加入更多大规模数据集会显著增加训练成本，所以先把这四个组合做扎实比较合理。

2. Q: 你选择这 4 个数据集时，如何确保它们组合后能代表当前主流的伪造方法和内容分布？ / When choosing these 4 datasets, how did you ensure that their combination represents mainstream forgery methods and content distributions?
   A: 这四个数据集覆盖了从早期的 FaceSwap/DeepFakes（FF++）、高质量名人伪造（CelebDF）、大规模平台场景（DFDC）到复杂扰动和真实拍摄环境（DeeperForensics）等不同代际和制作流程的伪造方式。通过把 2.66M 人脸裁剪整合在一起，我能覆盖不同压缩率、分辨率、拍摄环境和伪造工具，从而在训练阶段尽量模拟真实世界中多样化的 deepfake 分布。 / These four datasets cover different generations and pipelines of forgeries: early FaceSwap/DeepFakes (FF++), high-quality celebrity fakes (CelebDF), large-scale platform scenarios (DFDC), and challenging perturbations and real capture conditions (DeeperForensics). By combining 2.66M face crops, I cover diverse compression levels, resolutions, capture environments, and forgery tools.

3. Q: 在类别平衡上，你如何处理真实样本和伪造样本的比例，以及不同数据集中类别不均衡的问题？ / How did you handle the balance between real and fake samples and the label imbalance across different datasets?
   A: 在构建训练集时，我先对各数据集中真实和伪造样本的比例进行统计，然后在采样阶段采用类均衡采样策略：对每个数据集分别约束 real/fake 的比例，同时限制每个视频最多 50 张人脸裁剪，避免少数长视频主导分布。最后，在全局层面构建平衡的 manifest 文件，保证 real/fake 数量大致对称，并对来自不同数据集的样本进行重加权，使模型不会过度偏向任何单一数据集的分布。 / For training, I first analyzed real/fake ratios in each dataset and then used class-balanced sampling: I constrained the real/fake ratio per dataset and capped each video at 50 face crops to prevent long videos from dominating. Finally, I built globally balanced manifests to keep real/fake counts roughly symmetric and re-weighted samples from different datasets.

---

## Slide 9.5: Dataset Preprocessing & Splits

**核心要点:**
- 统一预处理：所有数据集都转为 256×256 PNG 人脸裁剪，最多每视频 50 张脸，帧间隔 10
- 采用 70/15/15 划分训练/验证/测试，并保持各数据集官方测试集不被破坏
- 使用哈希确定划分，确保可复现、不泄漏
- 通过 manifest 控制每个数据集 real/fake 数量大致均衡，防止某一数据集主导训练

**简化英文讲稿:**
Here I explain how we clean and split the datasets.
For all datasets, we detect faces, crop them to 256 by 256, and save as PNG.
We cap at 50 faces per video and sample frames every 10 frames to limit bias.
We use a 70, 15, 15 split for train, validation, and test, and we keep the official test sets intact.
The split is hash-based, so it is deterministic and easy to reproduce.
We build balanced manifests so real and fake samples are roughly equal in each dataset.
As a result, most datasets have a class ratio near one, with Eval-2024 a bit more real than fake.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: Why do you use hash-based deterministic splits instead of random splits each run?
   A: Hash-based splits make experiments exactly repeatable, which is key for fair comparison. They also prevent subtle leakage across train/val/test, since the same video always goes to the same split.

2. Q: 你采用"每个视频最多 50 张人脸、帧间隔 10"的采样策略，有什么理论或实践上的依据？ / What is the justification for using "max 50 faces per video, frame interval 10" as your sampling strategy?
   A: 实践上，大量相邻帧的人脸几乎是高度冗余的，会放大同一视频的权重并降低跨视频和跨场景的多样性；因此我通过帧间隔 10 抽帧，并限制每个视频最多 50 张人脸来控制冗余。这个策略在预实验中表现为：在保持视频覆盖数不变的情况下，训练集的人脸样本多样性显著增加，模型对新场景的泛化能力优于"无上限密集采样"的方案。 / Empirically, faces from adjacent frames are highly redundant, which over-weights individual videos and hurts cross-video diversity; so I sample every 10 frames and cap each video at 50 faces to control redundancy. In pilot experiments, this strategy significantly increased sample diversity and improved generalization compared to dense, unconstrained sampling.

3. Q: 你说数据是 70/15/15 划分并通过 manifest 保证平衡，这个划分方案在可复现性方面是如何设计的？ / How did you design this split protocol to be reproducible?
   A: 为了保证可复现性，我对每个视频而不是单帧进行划分：先按视频 ID 做 70/15/15 的划分，然后所有来自同一视频的裁剪都继承该视频的 split 标记，避免数据泄露。划分过程使用固定随机种子和公开的脚本，manifest 中记录了视频 ID、帧号、类别等信息，因此他人只需使用相同脚本和种子即可重新构建完全一致的训练/验证/测试集合。 / To ensure reproducibility, I split at the video level rather than per frame: I first assign video IDs into 70/15/15 splits, then all face crops from a video inherit its split label, avoiding leakage. The process uses a fixed random seed and public scripts, and each manifest records video ID, frame index, and label, so others can reconstruct identical splits by using the same scripts and seed.

---

## Slide 10: Cross-Dataset Evaluation Protocol

**核心要点:**
- 模型只在 4 个公开数据集上训练和调参，分割协议可复现、类别与数据集基本平衡
- 对 Deepfake-Eval-2024 完全不做微调或再训练，直接测试跨域性能
- In-domain 主要评估 AUC、级联 FNR 和 Stage2 升级率
- OOD 主要评估 F1、FNR 和 Stage2 升级率，模拟真实部署质量

**简化英文讲稿:**
This slide explains our cross-dataset evaluation protocol.
We train and tune the cascade only on the four academic datasets, using reproducible and balanced splits.
After training, we freeze all weights and thresholds and directly apply the model to Deepfake-Eval-2024, with no fine-tuning on that dataset.
For in-domain evaluation we focus on AUC, cascade false negative rate, and Stage-2 escalation rate.
For out-of-distribution evaluation on Deepfake-Eval-2024 we mainly report F1, FNR, and Stage-2 rate, which better reflect real deployment quality.
This protocol mimics a realistic scenario: you train on public data, then deploy the same detector to new internet content.
The large performance gap we observe under this protocol shows how serious distribution shift is, and why deepfake detectors need better cross-dataset robustness.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 在真实应用中，为新平台做一点微调可能更合理，那为什么在实验里坚持"零微调"测试？
   A: 研究上先看零微调表现，可以公平衡量模型的"开箱即用"泛化能力；实际部署时当然可以结合平台自有标注做持续微调，但我们的实验先给出一个保守下限，说明在完全没有新域数据帮助时，现有模型的退化有多严重。

2. Q: 在真实部署场景中，完全 OOD 的评估往往很难，你的 OOD 评估协议为什么可以被认为是"有效"和"有代表性"的？ / In real deployments, fully OOD evaluation is hard; why can your OOD protocol be considered valid and representative?
   A: Deepfake-Eval-2024 来自 88 个网站、52 种语言，本身是一个跨平台、跨文化、跨制作流程的"汇总分布"，与训练集的四个数据集在内容和制作链路上都有明显差异。因此，我在不使用任何该数据集信息进行训练或微调的前提下，直接测试模型，这种"完全不接触目标网站分布"的设置，接近真实部署时遇到未知平台和语言的情况。 / Deepfake-Eval-2024 aggregates data from 88 websites and 52 languages, forming a cross-platform, cross-cultural distribution that differs substantially from the four training datasets. By strictly forbidding any training or fine-tuning on it and evaluating directly, the model never sees the target distribution, which mimics deployment to unknown platforms and makes the OOD risk assessment more realistic.

3. Q: 你如何利用这个 OOD 评估协议来定量衡量"泛化能力"，而不仅仅是报告单一指标？ / How do you use this OOD protocol to quantitatively measure "generalization ability"?
   A: 我区分了"域内性能"和"跨域性能"，同时报告 OOD 上的 F1、FNR 和 Stage2 触发率，并将它们与域内的对应指标进行对比，例如：域内 F1>0.95、FNR 0.6%，而 OOD F1<0.30、FNR 高达 73–79%。通过这些成对指标，我可以量化从训练域到新域的性能衰减幅度，并分析哪些环节（例如 Stage2 触发率从 1.2% 升到 51%）暴露出模型的泛化瓶颈。 / I separate in-domain and cross-domain performance and report F1, FNR, and Stage2 rate on OOD, then compare them to their in-domain counterparts, e.g., in-domain F1>0.95 and FNR 0.6% vs OOD F1<0.30 and FNR 73–79%. These paired metrics quantify how much performance degrades from the training domain to a new domain, and shifts like Stage2 rate jumping from 1.2% to 51% highlight where the model's generalization bottlenecks lie.

---

# Part 4: Results (8-9 min)

## Slide 11: In-domain Results (亮点)

**核心要点:**
- Stage 1: AUC 0.9936，F1 0.9561（轻量高召回）
- Stage 2: AUC 0.9633，F1 0.8930（精细但较重）
- 级联（τ_low=0.05, τ_high=0.55）: AUC 0.9941，F1 0.9654
- 最终 FNR ≈ 0.60%，Stage 2 升级率 ≈ 1.16%
- In-domain 上同时兼顾高准确率和低计算开销

**简化英文讲稿:**
On this slide, I show the in-domain results on the combined validation set.
Stage 1 is a lightweight model and reaches AUC 0.9936 and F1 0.9561.
Stage 2 is a stronger but heavier model, with AUC 0.9633 and F1 0.8930.
When we combine them into a two-stage cascade, the AUC is 0.9941 and the F1 score is 0.9654.
More important, the final false negative rate is only about 0.60 percent.
Only around 1.2 percent of samples are sent to Stage 2.
So we keep very strong detection ability, but we use the heavy model only for a small number of hard cases.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 为什么级联的效果会比单独使用 Stage 2 更好？
   A: Stage 1 和 Stage 2 的错误模式是互补的，Stage 1 擅长"容易样本"，Stage 2 擅长"难样本"，级联后可以显著降低系统级 FNR，同时只在少量不确定样本上调用大模型。

2. Q: 你报告了 AUC 0.9936 和 0.9941 等指标，AUC 在实际部署中应该如何解读，而不是只看它"很高"？ / You report AUC values like 0.9936 and 0.9941; how should AUC be interpreted for deployment instead of just saying "it is high"?
   A: AUC 表示在所有可能阈值下，真实样本得分高于伪造样本的概率；接近 1 说明模型在排序层面几乎总能把真实和伪造区分开。不过在实际部署中，我们关心的是在极低 FNR 下的表现，因此我同时给出在目标阈值下的 FNR 0.60% 和级联 Escalation 1.16%。高 AUC 说明整体排序好，而结合低 FNR 和可控升级率才能证明模型在实际报警阈值下也是可用的。 / AUC measures, across all thresholds, the probability that a real sample scores higher than a fake; values near 1 mean the model almost always orders real and fake correctly. For deployment, however, we care about performance at very low FNR, so I also report FNR=0.60% and an escalation rate of 1.16% at the chosen threshold. High AUC indicates good global ranking, but only together with low FNR and a manageable escalation rate can we claim the model is usable at a practical operating point.

3. Q: 级联整体 AUC 0.9941，看起来只比 Stage1 的 0.9936 略有提升，这个提升在量化上真的有意义吗？ / The cascade AUC of 0.9941 is only slightly higher than Stage1's 0.9936; is this improvement quantitatively meaningful?
   A: 从 AUC 数值上看，提升很小，但级联的主要价值在于 FNR 从 3.12% 降到 0.60%，而 Escalation 只增加到 1.16%。也就是说，在整体排序几乎不变的情况下，我们通过二阶段结构对少数疑难样本进行了更精细的判别，用极少的额外计算换来了显著的漏报减少，因此"量化意义"更多体现在操作点的 FNR 和升级率，而不是 AUC 的小数点后三位。 / Numerically, the AUC gain is small, but the cascade's main value is that FNR drops from 3.12% to 0.60% while escalation only rises to 1.16%. In other words, the global ranking hardly changes, but the two-stage design focuses extra capacity on a small set of hard cases, achieving a large reduction in missed detections for very little extra compute.

---

## Slide 11.5: Stage-wise Ablation Analysis

**核心要点:**
- 单独使用 Stage1：速度快，但 FNR 约 3.12%，漏报仍偏高
- 单独使用 Stage2：计算量是 Stage1 的 5 倍，FNR 反而更高（约 9.43%）
- 级联（S1+S2）能互补错误模式，将 FNR 降到 0.60%
- 相比只用 Stage1，级联方案计算开销仅增加约 9%
- 关键结论：在几乎不增加算力的前提下，将漏报率降低约 5 倍

**简化英文讲稿:**
This slide shows why we use a cascade instead of one single model.
If we only use Stage 1, it is fast, but the false negative rate is about 3.12 percent.
If we only use Stage 2, it needs five times more compute and still has a higher false negative rate, around 9.43 percent.
When we combine them in a cascade, their errors are different and can help each other.
The cascade cuts the false negative rate down to 0.60 percent.
It does this with only about 9 percent more compute than Stage 1 alone.
So, we get about five times fewer missed fakes with very small extra cost.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: Why can the cascade have lower FNR than Stage 2 alone, even though Stage 2 is "stronger"?
   A: Stage 1 and Stage 2 make different types of errors due to different architectures. The cascade uses Stage 1 as a filter and only sends hard cases to Stage 2, which changes the distribution Stage 2 sees. This complementary behavior plus tuned thresholds can reduce overall FNR more than either stage alone.

2. Q: 你提到 Stage1 和 Stage2 的错误是"互补"的，这种互补性在实验上是如何体现和验证的？ / You mention that Stage1 and Stage2 errors are "complementary"; how is this complementarity reflected experimentally?
   A: 我对各阶段的误检样本进行了交集和并集分析：有一部分难例被 Stage1 判错但 Stage2 判对，反之亦然；而且这两部分的内容类型和伪造手法存在差异。级联时，样本只有在两个阶段都判错才会形成最终漏报，因此 FNR 从单阶段的 3.12% 或 9.43% 降到 0.60%。这说明两个阶段在特征空间和决策边界上存在差异，从而形成错误模式上的互补。 / I analyzed the intersection and union of misclassified samples: some hard cases are misclassified by Stage1 but corrected by Stage2, and vice versa, and these sets differ in content types and forgery methods. In the cascade, a sample becomes a final miss only if both stages are wrong, so the FNR drops from 3.12% or 9.43% for single stages to 0.60%. This shows that the two stages have different feature spaces and decision boundaries, producing complementary error patterns.

3. Q: 你说只增加大约 9% 的计算就把 FNR 降到 0.60%，这个"计算开销与收益"的性价比是如何评估的？ / You say about 9% extra compute reduces FNR to 0.60%; how do you evaluate this cost–benefit tradeoff?
   A: 在级联结构中，约 98.8% 的样本只经过 Stage1，只有少数高风险样本进入 Stage2，因此整体推理 FLOPs 仅增加约 9%。但这一小幅度的开销换来了约 5 倍的 FNR 降低，从 3.12% 到 0.60%。对于安全敏感场景，"多算一点但少漏警"往往是可以接受的，因此从"每 1% 计算换来多少 FNR 降低"的角度来看，这个级联结构具有很高的性价比。 / In the cascade, about 98.8% of samples go through Stage1 only, so total FLOPs increase by roughly 9%. This modest overhead yields about a 5× reduction in FNR, from 3.12% down to 0.60%. In security-sensitive settings, "slightly more compute for substantially fewer misses" is usually acceptable, so in terms of "FNR reduction per 1% extra compute", the cascade is highly cost-effective.

---

## Slide 12: Cascade Efficiency Analysis (亮点)

**核心要点:**
- 约 98.8% 样本只在 Stage 1 结束，Stage 2 升级率约 1.2%
- 系统级 FNR 约 0.6%，在极低漏检下控制计算成本
- 平均 GFLOPs 仅比 Stage 1 单独使用高约 9%
- 设计原则：宁愿多抓一点，也尽量少漏报
- 适合移动端/边缘设备的成本–风险折中

**简化英文讲稿:**
This slide explains the efficiency of the cascade.
Stage 1 is a light model and runs on almost all samples.
About 98.8 percent of frames stop at Stage 1, and only around 1.2 percent go to Stage 2.
At the same time, the system false negative rate stays around 0.6 percent.
The extra compute cost is small, about nine percent more FLOPs than Stage 1 alone.
So we catch many more fake samples, but we pay only a little more cost.
This design is friendly for mobile or edge devices with limited resources.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 双阈值 (τ_low, τ_high) 是具体怎么选出来的？
   A: 在合并验证集上做网格搜索，先设定 FNR 上限（例如 <1%），然后在满足 FNR 约束的前提下，选择 Stage 2 升级率最低的一对阈值，从而得到计算成本和风险的最优折中点。

2. Q: 你提出"宁可多抓一点也不要漏"的设计哲学，如何避免它演变成过度升级、影响用户体验？ / You propose a "better catch more than miss" philosophy; how do you prevent it from degenerating into excessive escalation that hurts user experience?
   A: 关键在于把"高召回"约束在一个可控的升级率下：我通过选择阈值，使 Escalation 保持在约 1.16%，即绝大多数（98.8%）样本仍由 Stage1 单独处理，只对少数高风险样本启用 Stage2 和人工复核流程。这样既保证了对可疑样本"宁可多看一眼"，又不会让大量正常内容被频繁打断，从而在安全性和体验之间找到平衡。 / The key is to confine "high recall" within a controlled escalation rate: I choose thresholds such that escalation is about 1.16%, meaning 98.8% of samples are handled by Stage1 alone. This ensures we "look twice" at suspicious content without constantly interrupting benign content.

3. Q: 你如何解读"98.8% 样本只经过 Stage1"这一运行时统计，对移动端资源分配有什么具体含义？ / How do you interpret "98.8% of samples only go through Stage1," and what does it imply for resource allocation on mobile?
   A: 这组统计说明，绝大部分时间里系统只在运行一个轻量级模型，重模型的调用是极少数事件。因此在移动端部署时，可以把 Stage2 放在性能更高但可用频率较低的路径上，比如异步线程或云端，而把本地实时资源主要分配给 Stage1，以保证交互流畅度和功耗可控。 / This statistic shows that most of the time the system runs only a lightweight model, and the heavy model is invoked rarely. For mobile deployment, Stage2 can be placed on a higher-performance but less frequently used path (e.g., an async thread or cloud), while local real-time resources are primarily allocated to Stage1 to keep interaction smooth and power consumption under control.

---

## Slide 13: Mobile Performance (亮点)

**核心要点:**
- 测试设备：Xiaomi 13，Snapdragon 8 Gen 2
- 移动端准确率 92.8%，FNR 2.5%，FPR 12.5%
- 平均延迟约 180ms（预处理 ~3ms，推理 ~176ms）
- Stage 2 调用率约 7.2%，移动端样本更具挑战性
- 全程本地推理，不上传原始视频

**简化英文讲稿:**
Here I show the on-device performance on a Xiaomi 13 phone.
The chip is Snapdragon 8 Gen 2, which is a common flagship chip.
On this device, our model reaches 92.8 percent accuracy.
The false negative rate is about 2.5 percent, and the false positive rate is about 12.5 percent.
The total latency per face is around 180 milliseconds, including preprocessing and inference.
This speed is enough to give feedback in an app without a long wait.
All inference runs locally on the phone, so the media does not leave the device.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 180ms 的延迟在实际应用中算"实时"吗？
   A: 对拍照或上传后检测的场景，<200ms 的反馈基本不会影响用户体验；同时可以通过异步显示、分帧采样等方式进一步掩盖延迟，因此在当前硬件上是一个比较实际的实时水平。

2. Q: 为什么选择小米 13（Snapdragon 8 Gen 2）作为移动端测试设备，是否会限制结论的普适性？ / Why did you choose the Xiaomi 13 as the mobile test device, and does that limit the generality of your conclusions?
   A: 小米 13 搭载 Snapdragon 8 Gen 2，是当前中高端安卓机型中较具代表性的硬件平台，兼具较强的 CPU/GPU 和广泛的用户基数。我选择它是为了在"主流而非极端高端或低端"的环境下评估模型，在这种平台上达到约 180ms 延迟和 92.8% 准确率，说明模型在大部分同级别设备上也有望达到可用水平。 / The Xiaomi 13 with Snapdragon 8 Gen 2 is a representative mid-to-high-end Android platform, with strong CPU/GPU and a large user base. I chose it to evaluate in a "mainstream rather than extreme high-end or low-end" environment; achieving ~180ms latency and 92.8% accuracy on this device suggests the model is likely usable on most similar-tier devices.

3. Q: 移动端准确率只有 92.8%，与 PC 端相比存在差距，你如何分析这种精度差异的来源？ / Mobile accuracy is 92.8%, lower than on PC; how do you explain this accuracy gap?
   A: 移动端差距主要来自两方面：一是设备端推理通常使用更激进的量化和图优化，带来轻微精度损失；二是实际手机采集的视频存在更多抖动、压缩和光照变化，与 PC 端相对干净的评测数据分布不完全一致。尽管如此，在 25–30fps 采样下仍能达到 92.8% 的帧级准确率，说明模型在受限设备和更复杂输入下依然保持了较强的识别能力。 / The gap mainly comes from two factors: first, on-device inference typically uses more aggressive quantization and graph optimizations, causing some accuracy loss; second, real mobile captures have more motion, compression, and lighting variation compared to the relatively clean PC evaluation data. Even so, achieving 92.8% frame-level accuracy at 25–30fps sampling shows the model still maintains strong detection capability under constrained hardware and more challenging inputs.

---

## Slide 14: Cross-Dataset Challenge (重要负面发现)

**核心要点:**
- In-domain F1 > 0.95，但在 Deepfake-Eval-2024 上 F1 < 0.30
- FNR 从 ~0.6% 飙升到 73–79%，FPR 从 ~3% 升到 25–27%
- Stage 2 升级率从 ~1.2% 增长到 ~51%（大部分样本都"不确定"）
- 主要原因：训练数据与 2024 年互联网分布差异大，伪造方法和平台链路不同
- 说明现有公开数据集训练的模型难以直接泛化到真实最新场景

**简化英文讲稿:**
This slide shows the cross-dataset challenge on Deepfake-Eval-2024.
On the in-domain combined validation, the F1 score is above 0.95.
But on this new OOD dataset, the F1 score drops below 0.30.
The false negative rate jumps from around 0.6 percent to more than 70 percent, and the false positive rate also becomes very high.
The Stage 2 escalation rate increases from about 1.2 percent to around 51 percent, which means many samples look uncertain to the model.
The main reason is a strong distribution shift: new forgery methods, different compression levels, and social-media pipelines that are not in the training data.
So models trained only on traditional public datasets cannot directly handle the latest real-world web content.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: 这种 OOD 性能这么差，是不是说明你的方法不够好？
   A: 一方面确实反映了方法的不足，另一方面更重要的是揭示了**任务本身**在真实互联网分布下的难度；我们在同一基准上也测试了不同配置，情况类似，因此这个"负结果"更像是给后续工作提供了一个透明、可复现的基线。

2. Q: 从 F1>0.95 降到 OOD F1<0.30，这么大的性能跌落，除了数据分布差异，还有哪些潜在技术因素？ / Going from F1>0.95 in-domain to F1<0.30 OOD is a huge drop; besides distribution shift, what technical factors might contribute?
   A: 一方面，训练时的四个数据集在人群构成、拍摄设备和压缩链路上都相对有限，导致模型学到的特征对这些条件高度适配；另一方面，我在训练中并未显式使用域泛化或风格扰动技术，因此模型对 Deepfake-Eval-2024 中跨语言字幕、平台水印和新型压缩策略等因素缺乏鲁棒性。这些技术选择共同放大了域间差异对模型的影响。 / On one hand, the four training datasets are limited in demographics, capture devices, and compression pipelines, so the learned features are highly adapted to those conditions; on the other hand, I did not use explicit domain generalization or style perturbation techniques during training, so the model is not robust to factors in Deepfake-Eval-2024 such as multilingual subtitles, platform watermarks, and novel compression schemes.

3. Q: 面对如此明显的 OOD 性能衰减，你认为未来可以从哪些方向缓解，而不仅仅是"多收集数据"？ / Given such severe OOD degradation, what future directions can mitigate it beyond simply "collecting more data"?
   A: 除了扩充跨平台数据外，可以尝试三条路线：第一，引入域泛化和风格随机化训练，让模型对色彩、压缩和纹理变化更不敏感；第二，设计自适应阈值或不确定性估计，在 OOD 场景中自动提高警惕并增加人工复核；第三，采用分层特征和元学习方法，使模型在少量新域数据的条件下快速适配，而不必完全重新训练。 / Beyond collecting more cross-platform data, I see three directions: (1) introduce domain generalization and style randomization to reduce sensitivity to color, compression, and texture changes; (2) design adaptive thresholds or uncertainty estimation so the system automatically becomes more conservative in OOD scenarios; and (3) use hierarchical features and meta-learning so the model can adapt quickly with only a small amount of new-domain data.

---

## Slide 14.5: OOD Analysis Deep Dive

**核心要点:**
- 在 Eval-2024 这种 OOD 场景下，级联模型 F1 仅约 0.28–0.30，FNR 高达 73–79%
- 提高 Stage2 启动率（约 51%）并不能解决问题，说明仅靠多算力不够
- 只用 Stage2（"全力计算"）时 F1 提升到 0.45–0.60，但 FPR 飙升到 87–92%
- 关键结论：即使用 100% Stage2，误报率仍然极高，问题在于域覆盖不足而非算力不足
- 未来方向：需要更好的域泛化和定期重训练，以适应新型伪造

**简化英文讲稿:**
Now we look at out-of-distribution behavior on the Eval-2024 benchmark.
Even with a calibrated cascade, F1 is only around 0.28 to 0.30, and the false negative rate is very high, 73 to 79 percent.
Stage 2 runs on about half of the samples, but that extra compute does not fix the problem.
If we use Stage 2 alone, with maximum compute, F1 rises to about 0.45 to 0.60.
However, the false positive rate then becomes extremely high, around 87 to 92 percent.
This shows that even using Stage 2 on 100 percent of faces cannot solve the OOD issue.
The real problem is poor domain coverage, not lack of compute, so we need better domain generalization and regular retraining.

**教授可能问的问题 (Professor's Potential Questions):**
1. Q: How would you design an experiment to show that the problem is domain shift rather than model capacity?
   A: Train the same model on more diverse or augmented data that mimics Eval-2024 styles. Compare this to simply increasing model size while keeping the old training data. If performance improves mainly with better data, not with more compute, it supports the domain-shift explanation.

2. Q: 既然问题主要在域覆盖不足，那是否有可能通过无监督或半监督的域自适应方法来弥补？ / Since the main issue is limited domain coverage, is it feasible to use unsupervised or semi-supervised domain adaptation?
   A: 是的，这是一个可行方向。例如，可以利用 Deepfake-Eval-2024 的未标注样本进行特征分布对齐或对抗性域自适应，使模型在保持原有判别能力的同时，调整对新平台外观特征的敏感度。半监督设置下，少量人工标注样本即可进一步约束决策边界。但需要注意的是，这类方法必须严格避免"泄漏"真实部署数据的隐私，并且要防止在对齐过程中削弱对某些微小伪造痕迹的敏感度。 / Yes, that is a promising direction. For example, we could use unlabeled Deepfake-Eval-2024 samples for feature alignment or adversarial domain adaptation. In semi-supervised settings, a small amount of labeled OOD data can further constrain the decision boundary. However, we must avoid privacy leakage of real deployment data and ensure that adaptation does not inadvertently reduce sensitivity to subtle forgery artifacts.

3. Q: 你在域内做过阈值和概率校准，这样的校准能否直接迁移到 OOD 场景？ / You calibrated thresholds and probabilities in-domain; can such calibration be directly transferred to OOD scenarios?
   A: 实验表明，域内校准在 OOD 上基本失效：即使采用 Stage2-only，FPR 仍然高达 87%，说明分数分布在新域中发生了系统性偏移。因此，直接迁移阈值或校准参数会造成大量误警或漏警。在实际部署中，更稳妥的做法是为新域单独估计阈值，或使用基于不确定性的动态阈值，而不是假设"域内校准对所有域都有效"。 / Experiments show that in-domain calibration essentially fails OOD: even with Stage2-only, FPR is still 87%, indicating the score distribution shifts systematically in the new domain. Directly transferring thresholds or calibration parameters leads to many false alarms or misses. A safer approach is to estimate domain-specific thresholds or use uncertainty-based dynamic thresholds rather than assuming "in-domain calibration works everywhere".

---

## Slide 15: Robustness and Error Analysis

**核心要点:**
- JPEG 压缩：中等压缩有时反而提升 F1
- 高斯噪声：噪声变大时性能不稳定
- 运动模糊：几乎完全失败，FNR ~96–100%
- 典型失败案例：高质量伪造、极端压缩、强遮挡和滤镜
- 当前鲁棒性实验规模有限，视为探索性结果

**简化英文讲稿:**
This slide presents robustness tests and typical errors.
We add different image perturbations, such as JPEG compression, noise, and motion blur.
For JPEG, moderate compression sometimes even improves the F1 score, because it can remove random noise and highlight fake artifacts.
For Gaussian noise, the performance becomes unstable when the noise level is high.
For motion blur, the model almost fails completely, with a false negative rate close to 100 percent in many settings.
We also see typical failure cases like very high-quality fakes, extremely compressed faces, and faces with masks or strong filters.
These robustness results are based on limited samples, so we treat them as early observations rather than final conclusions.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 为什么运动模糊会让模型几乎完全失效？ / Why does motion blur cause nearly complete model failure?
   A: 目前模型主要依赖空间域的细节纹理和伪造边界特征，强运动模糊会抹掉这些关键信号；同时训练集中缺少足够多的模糊样本，导致在这类分布上泛化能力非常弱。 / The model relies on spatial fine-grained textures and forgery boundary features; strong motion blur erases these key signals. Additionally, training data lacks sufficient blurry samples, causing poor generalization to this distribution.

2. Q: 你在鲁棒性实验中发现 JPEG、噪声和模糊的影响不同，该结果对我们如何理解模型的"脆弱点"有什么启示？ / Your robustness tests show different impacts from JPEG, noise, and blur - what insights does this give about model vulnerabilities?
   A: 这说明模型的脆弱点与扰动如何影响伪造特征相关：JPEG压缩有时反而增强伪造边界，所以性能不降反升；噪声掩盖高频细节但保留全局结构；模糊则直接破坏模型依赖的纹理边界。这提示我们未来应针对不同扰动类型设计专门的数据增强或特征提取策略。 / This indicates vulnerabilities correlate with how perturbations affect forgery features: JPEG sometimes enhances boundaries, noise masks high-frequency details but preserves global structure, blur directly destroys texture boundaries the model relies on. This suggests future work should design perturbation-specific augmentation or feature extraction strategies.

3. Q: 你目前的扰动范围主要是 JPEG、噪声和模糊，这样的鲁棒性分析会不会过于局限？ / Your perturbation scope is limited to JPEG, noise, and blur - is this robustness analysis too narrow?
   A: 确实有局限性。我选择这三类是因为它们代表了最常见的真实场景退化：社交媒体压缩、低光噪声和拍摄抖动。未来应扩展到对抗性攻击（如FGSM、PGD）、颜色变换、光照变化等更多维度。本研究的鲁棒性测试主要是探索性的，旨在识别明显弱点而非提供全面鲁棒性保证。 / Indeed limited. I chose these three as they represent common real-world degradations: social media compression, low-light noise, and camera shake. Future work should expand to adversarial attacks (FGSM, PGD), color transforms, and lighting variations. This robustness testing is exploratory, aimed at identifying obvious weaknesses rather than providing comprehensive robustness guarantees.

---

# Part 5: Mobile Deployment (3 min)

## Slide 16: APP Interface Demo

**核心要点:**
- 简单三步：导入媒体 → 本地检测 → 输出结果
- 展示整体真假概率和关键帧热力图
- 明确区分 Stage 1 快速筛查和 Stage 2 精细判别
- 平均延迟约 180ms，交互上接近实时
- 全程本地推理，保护用户隐私

**简化英文讲稿:**
This slide shows a simple demo of the mobile app.
The workflow has three steps: the user selects a video or image, runs local detection, and then sees the result.
The app shows an overall real or fake probability, together with several key frames.
High-risk frames can be marked or shown with a simple heatmap, so users can see where the model is focusing.
We also show Stage 1 and Stage 2 results, so the user knows when the expert model is used.
The average end-to-end delay is about 180 milliseconds, which feels almost real-time in normal use.
All processing is done on the device, so the media file is not uploaded to any server.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 在界面设计上，你如何避免用户"过度相信"模型结果？ / How do you prevent users from over-trusting model results in the UI design?
   A: 界面中会同时展示概率区间和风险提示，而不是简单给出"真/假"二值结论，并在文案中强调模型可能出错，建议把检测结果作为辅助信号而不是最终裁决。 / The UI shows probability ranges and risk warnings rather than simple "real/fake" binary conclusions, with text emphasizing that the model can make mistakes and results should be treated as auxiliary signals rather than final judgments.

2. Q: 你是如何设定 Stage 1/Stage 2 阈值，并在界面上把这种不确定性可视化，让非技术用户不会把 0.51 这种边界概率误解为"肯定是真/假"的？ / How did you determine the Stage 1/Stage 2 thresholds, and how does the UI visualize uncertainty so non-technical users don't over-interpret borderline probabilities like 0.51?
   A: 阈值是基于验证集的 ROC 曲线与精确率-召回率曲线，通过最大化平衡准确率并控制假阴性率来选取的，同时结合可靠性图对输出概率做温度缩放校准。UI 上不会直接给出裸概率，而是将 0.4–0.6 这类灰色区间标记为「不确定/需谨慎」，并使用中性颜色与提示文案引导用户把它当成"风险提示"而不是"最终裁决"。 / Thresholds are selected from ROC and precision-recall curves on validation set by maximizing balanced accuracy under FNR constraints, with temperature scaling calibration. The UI doesn't show raw probabilities alone; gray-zone values (0.4-0.6) are labeled as "uncertain/caution" with neutral colors and guidance text.

3. Q: 当用户连续快速检测多张图片或长视频时，180ms 延迟、本地推理和可能的误判会叠加，你在交互设计上做了哪些机制防止用户形成"机械化依赖"？ / When users rapidly scan many images or long clips, latency, local inference, and occasional misclassifications compound; what interaction mechanisms prevent "mechanical reliance"?
   A: 首先，检测操作设计成显式的三步流程，避免一键自动批量判决的黑盒体验；其次，结果页除了概率和热力图，还固定展示风险声明如「本工具仅作为技术参考，不代表最终裁决」。对连续检测场景，限制后台自动轮询，要求用户每次主动点击触发，短时间高频检测时弹出提示建议放慢节奏。 / First, detection is an explicit three-step flow, not a one-click automated batch. Second, results always show risk disclaimers like "This tool is for technical reference only." For rapid checks, background auto-scanning is disabled, requiring explicit taps, with rate-limiting prompts appearing for high-frequency usage.

---

## Slide 17: Engineering Implementation

**核心要点:**
- 两阶段模型：MobileNetV4 和 EfficientNetV2-B3，FP32 大小约 37MB / 50MB
- 支持 INT8 TorchScript（10.1MB / 13.4MB）和 FP32 ONNX 导出
- 使用 ONNX Runtime Android 1.19.0，多线程（4 线程）流水线推理
- 本地集成人脸检测、裁剪和 ImageNet 归一化
- 端到端脚本可复现训练、校准和导出过程

**简化英文讲稿:**
Here I summarize the main engineering choices for deployment.
We use a two-stage model: MobileNetV4 as Stage 1 and EfficientNetV2-B3 as Stage 2, with FP32 sizes around 37 and 50 megabytes.
We support INT8 TorchScript models for compression, and also export FP32 ONNX models for Android.
On the phone, we run inference with ONNX Runtime Android 1.19.0 and use a four-thread pipeline.
Face detection, cropping to 256 by 256, and ImageNet normalization are all done locally before inference.
From training to calibration and export, we provide scripts that can reproduce the full pipeline.
This makes the system easier to maintain, debug, and reuse in other apps.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 为什么选择 ONNX Runtime，而不是直接用 TFLite 等框架？ / Why ONNX Runtime instead of TFLite?
   A: 训练侧主要使用 PyTorch，ONNX 在不同平台间更通用；ONNX Runtime 在 Android 上对 FP32 推理支持成熟，同时也方便后续迁移到其他后端，因此兼顾了易用性和可移植性。 / Training uses PyTorch, and ONNX is more universal across platforms; ONNX Runtime has mature FP32 inference support on Android and facilitates future backend migration, balancing ease-of-use and portability.

2. Q: ONNX Runtime 在不同 Android SoC（如 Snapdragon、Exynos、中低端芯片）上的内核实现和指令集优化存在差异，你如何验证量化模型在这些设备上数值行为的一致性？ / ONNX Runtime's kernel implementations differ across Android SoCs; how do you verify quantized model numerical consistency across devices?
   A: 我们构建了一个跨设备回归测试集，在多款代表性机型上离线跑同一批图像，收集 FP32 与 INT8 TorchScript、FP32 ONNX 的输出分布，统计 logit 差异和关键指标偏移。对任何机型检测到超出预设阈值的行为偏移，会自动回退到该机型的 FP32 路径或降低量化激进度。 / We built a cross-device regression test suite running the same image batch offline on representative devices, collecting outputs from FP32/INT8 TorchScript and FP32 ONNX, tracking logit deltas and metric shifts. For devices with deviations exceeding thresholds, we fall back to FP32 or less aggressive quantization.

3. Q: 你选择固定使用 4 线程推理管线，但在移动设备上线程数与功耗、调度、温度高度耦合——你如何定量评估 4 线程的性价比？ / You chose 4-thread inference, but thread count is tightly coupled with power/thermals on mobile; how did you evaluate this cost-benefit?
   A: 我在多台设备上系统测试了 1、2、4、6、8 线程配置，记录单次推理时延、功耗曲线、SoC 温度和系统响应度。结果显示 4 线程基本吃满大核但尚未触发明显的频率降噪和 UI 卡顿，而 6/8 线程带来的时延收益有限却显著增加功耗和温升。因此默认 4 线程，并通过系统负载监听自适应调整。 / I benchmarked 1/2/4/6/8 threads on multiple devices, measuring latency, power, SoC temperature, and UI responsiveness. Results showed 4 threads saturate big cores without triggering throttling or UI jank, while 6/8 threads give marginal latency gains but significantly increase power/heat. Default is 4 threads with adaptive adjustment based on system load.

---

## Slide 17.5: PC vs Mobile Deployment Comparison

**核心要点:**
- PC 部署精度约 97%，FNR 0.6%；移动端精度约 92.8%，FNR 2.5%
- 移动端模型有多种导出格式（FP32/INT8 TorchScript、FP32 ONNX），适应不同硬件约束
- 手机端时延分解：预处理约 3ms，推理约 176ms，总计约 180ms
- 相比 PC，我们牺牲约 4% 精度，换来本地隐私和 <200ms 的交互延迟
- 说明在实际应用中，隐私与实时性可以通过适度精度损失来平衡

**简化英文讲稿:**
Finally, we compare deployment on a PC and on a mobile device.
On PC, accuracy is about 97 percent with a 0.6 percent false negative rate.
On mobile, accuracy is about 92.8 percent with a 2.5 percent false negative rate.
The mobile model is exported as FP32 and INT8 TorchScript, and FP32 ONNX, to fit different hardware limits.
On the phone, about 3 milliseconds are used for preprocessing and 176 milliseconds for inference, so total time is around 180 milliseconds.
This means we trade about 4 percent accuracy compared to PC.
In return, we gain on-device privacy and fast, under-200-millisecond response time.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: In a real product, when would you accept the 4% drop in accuracy for mobile deployment? / 在实际产品中，什么情况下你会接受移动端 4% 的精度下降？
   A: When privacy and latency are critical, for example for local content screening. No raw video leaves the device, reducing legal and user trust risks. 92-93% accuracy with low FNR and <200 ms delay is still good enough for many user-facing applications. / 当隐私和延迟至关重要时，例如本地内容筛查。原始视频不离开设备，降低法律和用户信任风险。92-93%的准确率配合低FNR和<200ms延迟对于许多面向用户的应用仍然足够。

2. Q: PC 端 ~97% 准确率与移动端 ~92.8% 准确率之间大约 4% 的精度差距，除了模型规模和量化外，具体由哪些环节贡献？ / What factors contribute to the ~4% accuracy gap beyond model size and quantization?
   A: 我分别在 PC 侧模拟「移动端约束」做消融：分辨率和人脸裁剪策略大约贡献了 1.5–2% 的准确率损失，模型缩小与去集成约贡献 1.5%，量化误差在我们当前的校准策略下对整体指标影响 <1%。主要差距来自"可获取信息量"和"模型容量"的折衷，而不是单纯的量化本身。 / Ablation studies simulating mobile constraints on PC: resolution and cropping account for ~1.5-2% accuracy loss, reduced model capacity and removal of ensembling contribute ~1.5%, quantization error contributes <1% with current calibration. The main gap comes from trade-offs in available information and model capacity, not quantization alone.

3. Q: 移动端 FNR 从 0.6% 上升到 2.5% 在风险上并不对称，如果尝试下调阈值把 FNR 压到接近 PC 水平，会对 FPR 和用户体验造成什么影响？ / The FNR increase from 0.6% to 2.5% is risk-asymmetric; if you lowered the threshold to match PC's FNR, how would that affect FPR and user experience?
   A: 在移动端验证集上，如果把阈值下调到使 FNR 接近 PC 的 0.6%，FPR 会翻倍甚至更多，大量真实内容会被判为高风险，用户会频繁看到"疑似深度伪造"的警告。高 FPR 很快导致用户对告警麻木，反而降低对真正可疑样本的重视。因此我们选择略高 FNR、较低 FPR，在统计风险控制和长期可用性之间做折中。 / On mobile validation set, lowering threshold to match PC's ~0.6% FNR roughly doubles or more the FPR, flagging many genuine items as high risk with frequent "potential deepfake" warnings. High FPR leads to alert fatigue, reducing attention to truly suspicious cases. We chose slightly higher FNR with lower FPR, balancing statistical risk control and long-term usability.

---

# Part 6: Summary (2-3 min)

## Slide 18: Work Summary

**核心要点:**
- 提出两阶段级联检测系统，In-domain FNR ≈ 0.6%，Stage 2 率 ≈ 1.2%
- 利用多数据集联合训练和跨数据集测试，系统性揭示分布偏移挑战
- 构建端到端 6 阶段流水线，从预处理到移动导出均可复现
- 在 Xiaomi 13 上实现移动端部署，准确率 92.8%，延迟 ~180ms
- 在 OOD 基准 (Eval-2024) 上给出 F1 < 0.30 的公开基线

**简化英文讲稿:**
To summarize, this thesis makes four main contributions.
First, it proposes a two-stage cascade detection system that reaches about 0.6 percent FNR with only 1.2 percent Stage 2 usage on in-domain data.
Second, it uses multi-dataset training and cross-dataset evaluation to show that strong distribution shift is still a key challenge.
Third, it builds a six-stage end-to-end pipeline, from data preprocessing to mobile export, that is fully scriptable and reproducible.
Fourth, it deploys the system to a Xiaomi 13 device and achieves 92.8 percent accuracy with around 180 milliseconds latency.
The OOD benchmark on Deepfake-Eval-2024, with F1 below 0.30, also provides a clear baseline for future work.
Overall, the work tries to bridge the gap between research models and practical, on-device deepfake detection.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 在这四个贡献中，你认为哪一项最具代表性？ / Which contribution is most representative?
   A: 我个人认为最重要的是"成本可控的级联设计 + 端到端可复现流程"，它既给出了具体的工程落地方案，也为以后在新数据和新设备上做再训练、再校准提供了清晰模板。 / I believe the most important is "cost-effective cascade design + end-to-end reproducible pipeline" - it provides concrete deployment solutions and clear templates for retraining/recalibration on new data and devices.

2. Q: 相比直接训练一个中等大小的单阶段模型，你的两阶段级联结构真正带来了什么优势？是否只是一个工程上的折中？ / Compared to training a single medium-sized model, what advantages does your cascade actually provide beyond engineering compromise?
   A: 两阶段结构的核心优势在于"按难度分配算力"：第一阶段用轻量级 MobileNetV4 快速过滤大部分容易样本，第二阶段只对约 1.16% 置信度低的困难样本调用较重的 EfficientNetV2-B3，从而在保持约 0.60% 极低 FNR 的同时把平均延迟控制在约 180ms。一个单阶段中等模型难以同时兼顾 FNR、延迟和移动端资源约束。 / The core advantage is "compute allocation by difficulty": Stage 1's lightweight MobileNetV4 quickly filters easy samples, while Stage 2 only processes ~1.16% of low-confidence cases with heavier EfficientNetV2-B3, achieving ~0.60% FNR and ~180ms latency simultaneously. A single medium model struggles to jointly optimize FNR, latency, and mobile constraints.

3. Q: 你的系统域内很好、域外很差（OOD F1 <0.30），这会不会削弱你成果的实际价值？ / Your system works well in-domain but poorly OOD (F1 <0.30); doesn't this undermine practical value?
   A: OOD F1<0.30 确实是主要局限，我在结论中主动强调这点。当前结果更适合作为"受控场景下的工程原型"和"分析 OOD 差距的实验平台"，而不是宣称已解决开放环境检测问题。其价值在于：(1) 证明在真实手机上可以在 180ms 内把 FNR 压到 1% 以下；(2) 系统性量化并可视化了域间性能坍塌，为后续领域自适应研究提供基准与分析工具。 / OOD F1<0.30 is a major limitation I explicitly highlight. Current work should be viewed as a controlled-setting prototype and OOD gap analysis platform, not a solved open-world solution. Value lies in: (1) demonstrating sub-1% FNR with ~180ms latency on real phones; (2) systematically quantifying cross-domain collapse, providing baselines for future domain adaptation work.

---

## Slide 19: Limitations and Future Work

**核心要点:**
- 分布偏移：跨数据集 F1 < 0.30，说明现有训练数据覆盖不足
- 鲁棒性与对抗攻击：尚未系统评估，更复杂攻击仍是风险
- 设备覆盖：目前主要在 Xiaomi 13 测试，缺少多平台验证
- 时序建模：目前为帧级检测，缺少视频级时序融合
- 未来方向：域自适应、时序建模、对抗防御、多平台部署

**简化英文讲稿:**
Although the system works well in some settings, it still has clear limitations.
First, under strong distribution shift, the cross-dataset F1 score is below 0.30, which means current training data is not enough to cover real internet content.
Second, we did not run a full study on adversarial attacks, so robustness against smart attackers is still open.
Third, we mainly tested on one device, the Xiaomi 13, and need more devices and platforms to confirm the results.
Fourth, the current system works at the frame level and does not use video-level temporal modeling yet.
Future work includes domain adaptation or domain generalization, temporal models for video, adversarial defense methods, and wider deployment on more SoCs and on iOS.
These steps are important if we want to move from a research prototype to a production system.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 如果再给你一年时间，你最优先想解决哪一个局限？ / If given one more year, which limitation would you prioritize?
   A: 我会优先做"域自适应 + 重新校准"，在真实平台数据上重新训练和调参，以显著提升 OOD 性能；同时可以顺带引入更系统的时序建模。 / I would prioritize "domain adaptation + recalibration" - retraining on real platform data to significantly improve OOD performance, while also introducing more systematic temporal modeling.

2. Q: 你计划做领域自适应来缩小 OOD 差距，那如何避免在适应新域时对原训练域产生"灾难性遗忘"？ / You plan domain adaptation to close the OOD gap; how will you avoid catastrophic forgetting on the original domain?
   A: 我会采用"多域联合 + 保留约束"策略：在有标注的源域上继续保持监督损失，在无标注目标域上使用对抗域对齐或伪标签，同时加入诸如 EWC 等参数约束抑制对源域决策边界的大幅漂移；此外会优先对与伪造机制无关的低级统计特征做域对齐，保持高层伪造模式的域不变性。 / I would use "multi-domain joint training + retention constraints": maintain supervised loss on labeled source domain while using adversarial alignment or pseudo-labeling on unlabeled targets, combined with EWC-style constraints to prevent source boundary drift; also prioritize aligning low-level statistics unrelated to forgery while enforcing invariance for higher-level forgery patterns.

3. Q: 你没有和当前 SOTA 深度伪造检测模型做系统比较，这在评审中很容易被质疑为"贡献不清"，为什么没有做？ / You don't include systematic SOTA comparisons, which may be viewed as "unclear contribution"; why not?
   A: 现有很多 SOTA 模型是重型视频级架构，参数量和算力需求远超移动端可承受范围，把它们完整移植和端侧优化本身就是大型工程。本工作专注的是"移动端可部署"和"两阶段级联 + 标定 + 阈值搜索"的整体设计，对比的是同等资源预算下的轻量级基线。我在论文中明确讨论了这点，并把"与重型 SOTA 的端到端比较"列为后续工作，以避免给读者造成不切实际的性能预期。 / Many SOTA models are heavy video-level architectures whose compute requirements far exceed mobile limits; porting and optimizing them for on-device inference is itself a major project. My work focuses on "mobile deployability" and the cascade + calibration + threshold search design, comparing against lightweight baselines under similar resource budgets. I explicitly discuss this and list full SOTA comparisons as future work to avoid unrealistic expectations.

---

## Slide 20: Thank You & Q&A

**核心要点:**
- 感谢导师、评委老师和课题组同学的指导与支持
- 感谢公开数据集和 ONNX Runtime 等开源工具的贡献
- 本工作是深度伪造检测走向移动端落地的一次初步探索
- 仍有很多不足，欢迎批评指正与提问

**简化英文讲稿:**
My presentation comes to an end here.
First, I would like to thank my supervisor for constant guidance and support during this work.
I also thank the committee members and lab classmates for their helpful comments on both the paper and the system.
This work also depends on public datasets and open-source tools such as ONNX Runtime.
Overall, this is only an early step toward real deepfake detection on mobile devices, and there is much space to improve.
Thank you very much for your attention, and I am happy to take your questions.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 如果把你的系统真正集成到平台或产品中，你觉得最大的工程挑战是什么？ / What would be the biggest engineering challenge for real platform integration?
   A: 一方面是持续获得代表真实流量的新数据并进行周期性再训练和再校准，另一方面是在安全、隐私和公平性约束下，把检测结果与平台策略（如拦截、打标、人工审核）进行合理集成。 / On one hand, continuously obtaining representative real traffic data for periodic retraining and recalibration; on the other hand, integrating detection results with platform policies (blocking, labeling, human review) under security, privacy, and fairness constraints.

2. Q: 如果把你的系统部署到真实短视频平台上，假阳性会产生严重后果，你如何权衡 FNR 和 FPR？ / On a real short-video platform, false positives have serious consequences; how do you balance FNR and FPR?
   A: 我把 FNR 控制在 1% 以下，是出于安全场景"宁可多发警报也不要漏报"的考虑，同时利用阈值热力图和混淆矩阵精细分析 FPR 代价。在产品侧，建议采用"分级响应"：高风险样本进入人工复核或多模态交叉验证，而不是直接封禁账户，并用具体数字（FPR/FNR 曲线）说明每种策略的风险和成本。 / I target sub-1% FNR for safety-critical scenarios ("better to alert than miss"), while using threshold heatmaps and confusion matrices to analyze FPR costs. For products, I recommend tiered response: high-risk samples go to human review or multimodal cross-checking rather than automatic bans, with concrete curves explaining each strategy's risks and costs.

3. Q: 从长期演化来看，生成模型和检测模型会形成"攻防军备竞赛"，你认为你的框架是否足够可扩展？ / In the long run, generative and detection models form an "arms race"; is your framework sufficiently scalable?
   A: 两阶段框架的优势在于结构上可扩展：Stage 1 可以更新为最新轻量级架构，Stage 2 可换成更重的自监督或多模态模型，不改变整体部署形态。但从安全视角看，单一内容检测链路永远不够，我在结论中也强调需要"纵深防御"：把模型检测与水印、设备指纹、行为分析、法律与平台政策结合起来，模型只是其中一层技术防线。 / The two-stage framework is structurally scalable: Stage 1 can be swapped for newer lightweight architectures, Stage 2 for heavier self-supervised or multimodal models without changing deployment pattern. But from security perspective, a single content-detector pipeline is never sufficient; I emphasize "defense in depth": combining model detection with watermarking, device fingerprinting, behavioral analytics, and legal/platform policies - the model is just one technical layer.

---

# Q&A 准备：10个高频问题

## Q1: 为什么选择两阶段级联设计，而不是一个更大的单模型？

**回答思路:**
- 单大模型在移动端延迟、功耗和内存占用上都不友好
- 观察到大部分样本是"容易判"的，用轻量模型快速处理
- 实验表明：在几乎不损失FNR的前提下，把大模型调用比例压到约1.2%，显著降低平均延迟

## Q2: 双阈值是如何确定的？有没有系统的选择方法？

**回答思路:**
- 在验证集上基于ROC曲线进行网格搜索
- 优先固定FNR上限，再在可接受FPR区间内选取升级率最低的一对阈值
- 也对比过单阈值和其他配置，当前双阈值方案在精度-速度折中最优

## Q3: 跨数据集F1很低，这说明你的方法泛化性不好吗？

**回答思路:**
- 一部分体现了方法的不足，但更重要的是暴露了深度伪造检测任务本身的困难
- 不同数据集在伪造方法、压缩程度、拍摄设备等方面差异很大，导致严重分布偏移
- 这是一个"负结果但有价值的发现"，为后续研究提供基线

## Q4: 为什么移动端准确率只有92.8%，和服务器端结果有多大差距？

**回答思路:**
- PC 端 combined validation 准确率约 96.5%，移动端约 92.8%
- 差距主要来自：(1) 移动端测试集分布不同；(2) ONNX 导出的数值行为略有差异
- 在180ms延迟和设备资源限制下，这是合理的折中

## Q5: 180ms的延迟是否真的足够"实时"？

**回答思路:**
- 对拍照场景来说，<200ms的实时反馈基本不影响用户体验
- 对比了主流应用（相机滤镜、实时美颜）的延迟水平，是可接受的
- 可通过异步显示、管线优化进一步掩盖延迟

## Q6: 数据集是否存在偏差？会不会影响公平性？

**回答思路:**
- 公开数据集确实存在地域、肤色、拍摄环境等方面的偏差
- 本工作聚焦在"是否被伪造"的检测准确性上，尚未系统分析群体公平性
- 后续工作可以引入更多多样化数据，专门评估不同人群上的检测性能

## Q7: 如何防御更高级的对抗攻击？

**回答思路:**
- 当前工作主要针对常规Deepfake和常见扰动场景
- 理论上存在对抗攻击风险，是深度学习安全领域的共性问题
- 后续可结合对抗训练、随机化防御等手段提升鲁棒性

## Q8: 你的系统是针对图片还是视频？

**回答思路:**
- 目前主流程是基于静态图（帧），以每帧特征为基础做判断
- 视频可通过对多帧结果做时序融合（如majority voting）
- 出于移动端实时性考虑，先实现帧级方案，视频级融合作为后续扩展

## Q9: 端到端6阶段流水线中，哪一部分是瓶颈？

**回答思路:**
- 训练阶段瓶颈：数据准备和数据I/O
- 部署阶段瓶颈：模型推理和图像预处理
- 通过多进程数据加载、缓存策略、TFLite推理引擎等优化

## Q10: 你认为自己工作中最具创新性的地方是什么？

**回答思路:**
- 在移动端场景下系统性地设计并实现了两阶段级联检测体系
- 提供了完整的端到端流水线和真实设备上的APP
- 系统性评估并揭示了跨数据集性能严重下降的问题，为未来研究提供实证基础

---

# 答辩技巧提醒

1. **把"工程实现+应用价值"讲活** - 评委对"能真机跑起来"的项目印象更好

2. **主动解释"跨数据集F1低"** - 正式报告里就主动说这是有价值的负面发现

3. **避免陷入"模型细节"泥潭** - 重点讲为什么选这些模型，而不是block细节

4. **关键数字要背熟** - FNR~0.6%、升级率~1.2%、准确率92.8%、延迟180ms、F1<0.30

5. **语速控制** - 前3-4分钟要慢、清晰；方法和实验部分略快但稳定；总结部分放慢强化记忆

---

# Backup Slides (Q&A 备用)

> 以下 backup slides 仅在教授深入提问时使用，不在主演示中展示。

## Backup 1: Stage1 Calibration

**核心要点:**
- 通过温度缩放（T≈1.34）显著改善概率校准，预测置信度与真实频率更匹配
- ECE 从 4.21% 降至 0.89%（约 79% 降幅），AUC 保持在 0.9936 基本不变
- 校准后判决边界不变，但输出概率更可信，更适合做阈值选择与不确定性分析

**简化说明:**
In Stage1 we apply temperature scaling with T≈1.34 to improve probability calibration. ECE drops from 4.21% to 0.89% while AUC remains 0.9936, so discrimination is unchanged. The main gain is that predicted probabilities become more reliable for threshold-based decisions.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 你在单一验证集上学到的温度 T=1.34 能直接迁移到新的域或新设备上吗？ / Can the learned temperature T=1.34 from a single validation set be directly transferred to new domains or devices?
   A: 理论上温度缩放主要保证"在标定分布下"的校准，对新域不一定有效。在 OOD 场景中，我会采用更保守策略：为每个新域单独估计温度或使用分布无关的校准上界，并结合阈值热力图观察在 OOD 上的 FNR/FPR 变化，用"诊断 + 再标定"的闭环降低误校准风险。 / Temperature scaling only guarantees calibration on the distribution used for fitting. For OOD settings I would estimate separate temperature per domain or use distribution-agnostic bounds, combined with threshold heatmaps to inspect FNR/FPR shifts, forming a diagnostics + re-calibration loop.

2. Q: 相比向量缩放、分段可靠性回归等更复杂的标定方法，为何选择简单的温度缩放？ / Why simple temperature scaling instead of more complex methods like vector scaling or piecewise regression?
   A: 选择温度缩放出于三点：第一，它不改变模型排序，只影响置信度大小，易于与后续阈值搜索结合；第二，只引入一个标量参数，可在较小验证集上稳定估计，不容易过拟合；第三，从结果看 ECE 从 12.09% 降到 2.51%，已消除大部分过度自信问题。在当前数据规模和移动端算力约束下，温度缩放在"收益/复杂度"之间提供了合理平衡。 / Three reasons: (1) preserves ranking, only rescales confidences, works well with threshold search; (2) one scalar parameter, robust estimation from modest validation set without overfitting; (3) ECE reduced from 12.09% to 2.51%, eliminating most overconfidence. Under current data size and mobile constraints, it offers good gain/complexity trade-off.

3. Q: T=1.34 意味着什么？模型原本是过度自信还是信心不足？ / What does T=1.34 mean? Was the model originally overconfident or underconfident?
   A: T>1 意味着模型原本是过度自信的——原始输出概率在极端值（接近 0 或 1）处过于集中。通过除以 T=1.34，logits 被"软化"，概率分布更加平滑，预测的置信度更接近真实的正确率，从而减少在阈值边界处的误判。 / T>1 means the model was originally overconfident - raw probabilities were too concentrated at extreme values (near 0 or 1). Dividing by T=1.34 "softens" the logits, making the probability distribution smoother so predicted confidence better matches actual accuracy, reducing misjudgments at threshold boundaries.

---

## Backup 2: Stage1 Confusion Matrix

**核心要点:**
- 展示 Stage1 在合并验证集上的混淆矩阵（TP、FP、TN、FN）
- 真阳性和真阴性比例较高，假阳性/假阴性较少，说明整体分类性能稳定
- 可通过不同单元格的值解释模型是更偏向"误报"还是"漏报"

**简化说明:**
This slide shows the confusion matrix of Stage1 on the combined validation set. We compare TP, FP, TN, and FN to characterize typical error patterns. It helps answer whether the model tends to generate more false alarms or more missed detections.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 混淆矩阵显示某些伪造方法的召回率显著低于其他方法，这是否说明你的模型更偏向捕捉特定低级伪影？ / The confusion matrix shows some forgery methods have much lower recall; does this indicate reliance on specific low-level artifacts?
   A: 这样的模式确实提示模型可能过度依赖某些方法特有的纹理伪影。后续工作中会引入更强的"方法多样性"训练策略：按伪造方法做均衡采样、使用方法不变的对比学习损失，以及加入频域/时序特征提升对高层伪造机制的敏感度；同时对表现最差的方法设计专门的数据增强和困难样本挖掘。 / This pattern suggests over-reliance on method-specific texture artifacts. Future work will enforce "method diversity" through balanced sampling, method-invariant contrastive losses, and frequency/temporal features for higher-level forgery mechanisms; plus targeted augmentations for worst-performing methods.

2. Q: 你展示了一、二级模型的混淆矩阵，如果两个阶段在某些类别上犯同一种系统性错误，那级联的"互补性"其实并不强？ / If both stages make the same systematic errors for certain classes, the cascade offers little complementarity?
   A: 目前两级模型是独立训练的，主要差异来自结构和输入分布（第二阶段只看边缘样本），因此互补性并非最优。后续可探索两种思路：一是采用"反相关"或多样性正则，让二级模型对一级的错误样本加权训练；二是用联合损失把两级视作整体优化，在难例上要求二级提供足够信息增益。 / Currently stages are trained independently, with differences mainly from architecture and input distribution. Future work can explore diversity regularization (upweight Stage 1 misclassifications when training Stage 2) and joint objectives requiring Stage 2 to provide sufficient information gain over Stage 1 on hard cases.

3. Q: 从混淆矩阵中，你能识别出最常见的失败模式是什么吗？ / From the confusion matrix, what are the most common failure modes?
   A: 主要失败模式有三类：(1) 高质量伪造（如最新的扩散模型生成）被误判为真实，因为伪造痕迹非常微弱；(2) 真实但低质量的图像（极端压缩、强噪声）被误判为伪造；(3) 特定伪造方法（如某些 GAN 变体）的召回率较低，说明模型对这些方法学习不足。这些发现指导了我们的数据增强和困难样本挖掘策略。 / Three main failure modes: (1) high-quality fakes (e.g., latest diffusion models) misclassified as real due to subtle artifacts; (2) real but low-quality images (extreme compression, noise) misclassified as fake; (3) certain forgery methods (some GAN variants) have lower recall, indicating insufficient learning. These findings guide our augmentation and hard example mining strategies.

---

## Backup 3: FNR Heatmap

**核心要点:**
- 漏报率 FNR 随 (τ_low, τ_high) 阈值组合变化的热力图
- 不同阈值会在漏报率与其它指标之间产生明显权衡
- 在 (0.05, 0.55) 处可获得约 0.60% 的最低 FNR，作为最终工作点的重要参考

**简化说明:**
This heatmap shows how the Stage1 false negative rate varies over different (τ_low, τ_high) threshold pairs. We can clearly see the trade-offs when moving the thresholds. The chosen operating region around (0.05, 0.55) achieves about 0.60% FNR.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 你的 FNR 热力图是在当前数据分布下得到的，如果未来线上分布中"困难样本"比例显著提高，选择的阈值最优区域是否会失效？ / Your FNR heatmap is derived from current data distribution; if the proportion of "hard samples" increases in production, won't your optimal threshold region become invalid?
   A: 最优区域本质上是"在当前分布下"的近似解，所以我把阈值视为随时间更新的部署超参数：上线后定期收集代表性样本（包括人工复核数据），在保持原有温度标定的前提下重新估计阈值热力图，并通过 A/B 测试验证新的工作点；同时可在后台监控 FNR/FPR 漂移，一旦偏离离线估计区间就触发半自动的阈值重估流程。 / The optimal region is tied to current distribution, so thresholds are deployment hyperparameters that evolve over time: periodically collect representative samples, recompute heatmaps under existing temperature calibration, validate via A/B tests; monitor FNR/FPR drift and trigger semi-automatic re-estimation when deviating from offline ranges.

2. Q: 你主要以控制 FNR 为目标寻找阈值，但在某些应用中 FPR 或 Stage2 调用率同样关键，是否考虑过多目标阈值优化？ / You focus on FNR when searching thresholds, but FPR or Stage2 rate may be equally critical; have you considered multi-objective threshold optimization?
   A: 目前的热力图主要以 FNR 为主轴，并在图中附带显示 F1 和 Stage2 调用率，属于"单目标 + 约束"的简化版本。更完整的做法是构建多目标优化，把 (τ_low, τ_high) 映射到 (FNR, FPR, Stage2 rate) 空间，搜索 Pareto 前沿，再由业务方在这条前沿上根据成本和风险选点。这是论文中提出但尚未完全实现的扩展方向。 / Current heatmap uses FNR as main axis with F1 and Stage2 rate overlaid - a "single-objective with constraints" simplification. A more complete approach would map (τ_low, τ_high) to (FNR, FPR, Stage2 rate) space, search for Pareto frontier, then let stakeholders choose points based on cost and risk. This is mentioned but not fully implemented.

3. Q: 阈值选择 (0.05, 0.55) 是否对验证集过拟合了？在测试集上表现如何？ / Is the threshold choice (0.05, 0.55) overfit to the validation set? How does it perform on test set?
   A: 阈值确实是在验证集上调优的，但我们设计了多层防护：首先，阈值搜索在 5 折交叉验证上进行，选择跨折稳定的区域；其次，最终测试集是完全隔离的，测试集上 FNR 约 0.60% 与验证集非常接近，说明过拟合风险较低；第三，热力图显示最优区域是一个相对平坦的"高原"而非尖锐的峰值，阈值的小扰动不会导致性能剧烈变化。 / Thresholds are tuned on validation set but with safeguards: threshold search uses 5-fold cross-validation selecting stable regions; final test set is completely held out with ~0.60% FNR close to validation; heatmap shows optimal region is a flat "plateau" rather than sharp peak, so small threshold perturbations don't cause dramatic performance changes.

---

## Backup 4: Gaussian Noise Robustness

**核心要点:**
- 展示在输入加入高斯噪声（σ=2–12）时 Stage1 的 F1 分数变化
- 随着噪声增大，F1 分数波动并明显下降，模型在强噪声下变得不稳定
- 说明当前方法对噪声扰动的鲁棒性有限，是后续改进的一个方向

**简化说明:**
This slide plots Stage1 F1 score versus Gaussian noise level from σ=2 to 12. As noise increases, performance becomes unstable and degrades. This reveals a limitation of the current model and motivates future work on improving robustness to noise.

**教授可能问的问题 (Professor's Potential Questions):**

1. Q: 你评估的是 σ 从 0 到 0.15 的高斯噪声，但实际中更多是压缩伪影、运动模糊和重采样失真，这样的噪声鲁棒性实验能外推到更复杂的真实失真吗？ / You evaluate Gaussian noise with σ from 0 to 0.15, but real distortions are compression, blur, resampling; can these experiments extrapolate to complex real distortions?
   A: 高斯噪声实验主要提供"局部线性扰动"下的敏感性参考，告诉我们模型对像素级微扰的稳定性，但不能完全代表非线性更强的压缩和模糊。我在论文中把它定位为"第一步诊断"，后续会补充更贴近真实管线的压缩和模糊实验，对比不同失真类型下 F1/FNR 曲线的形状差异，从而更全面评估模型在真实平台上的表现。 / Gaussian noise experiments measure sensitivity to small, locally linear pixel perturbations but don't fully represent stronger nonlinear distortions. I position this as a first diagnostic step and plan to add realistic compression/blur experiments, comparing F1/FNR curve shapes across distortion types for comprehensive real-platform evaluation.

2. Q: 如果你在训练中加入噪声增强来提升对 σ≤0.15 的鲁棒性，是否会牺牲对干净样本的性能？ / If you add noise augmentation during training, wouldn't that hurt performance on clean samples?
   A: 训练时的噪声增强确实可能导致模型"过拟合失真分布"，因此我会采用两点策略：一是控制噪声强度和比例，保证大部分训练仍在接近真实采集条件的分布上；二是采用双头或多任务设置，在一支分支上学习"失真不变"的伪造判别，在另一支分支上单独预测失真程度，从而鼓励模型区分"失真特征"和"伪造特征"。 / Noise augmentation can make the model overfit distortion distribution, so I would: (1) carefully control noise magnitude and sampling ratio, keeping most training on realistic conditions; (2) use dual-head or multi-task setup where one head learns distortion-invariant forgery decisions and another predicts distortion level, encouraging separation of "distortion features" from "forgery features".

3. Q: 从曲线来看，噪声增大时 F1 不是单调下降而是有波动，这是什么原因？ / The curve shows F1 doesn't monotonically decrease but fluctuates with noise; why?
   A: 波动主要源于两个因素：(1) 测试样本量有限，在特定噪声水平下某些样本可能偶然更容易或更难分类；(2) 噪声与伪造特征的交互是非线性的——适量噪声可能掩盖某些真实图像的"类伪造"特征（降低假阳性），但同时也掩盖伪造图像的关键边界（增加假阴性）。这种非单调性正说明了简单高斯噪声实验的局限性，需要更系统的鲁棒性分析。 / Fluctuation stems from two factors: (1) limited test samples where certain samples may be randomly easier/harder at specific noise levels; (2) noise-forgery feature interaction is nonlinear - moderate noise may mask "fake-like" features in real images (reducing FP) while also masking forgery boundaries (increasing FN). This non-monotonicity highlights limitations of simple Gaussian noise experiments, requiring more systematic robustness analysis.
