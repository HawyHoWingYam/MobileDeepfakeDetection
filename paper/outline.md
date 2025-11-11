# 论文总体大纲（MobileDeepfakeDetection/paper）

目标：围绕移动端可落地的级联式深伪检测系统，完整呈现方法、数据、实验、结果与讨论，长度约一万字（可分阶段完善）。本大纲明确每一节需要覆盖的内容与落地素材来源（代码、输出目录、生成的 LaTeX 片段）。

1. Abstract（摘要）
- 问题与场景：移动端、资源受限、低漏报（FNR）要求。
- 方法概述：两级级联（Stage1: MobileNetV4，Stage2: EfficientNetV2），阈值网格搜索最小化 FNR 并控制二级调用率。
- 主要结果：Stage2 在合并验证上的 AUC/F1；级联在验证集上的低 FNR 与低升级率；导出到移动端的工件大小。
- 一句话结论与可复现性承诺。

2. Introduction（引言）
- 风险与挑战：分布外数据（Deepfake‑Eval‑2024）、移动端推理时延、模型体积、隐私与在端决策。
- 我们的系统化路线：Stage1 轻量、Stage2 专家、Stage4 阈值调优、Stage5 跨数据集评估、Stage6 导出与集成。
- 贡献点：
  - 级联检测系统，显式控制 FNR/二级升级率；
  - 可复现的多阶段训练/评估/导出流水线；
  - 移动端落地工件与测量方案；
  - 鲁棒性与权衡分析（FNR vs Stage‑2）。

3. Related Work（相关工作）
- 深伪检测综述与特征（空间/频域/多模态、Transformer 等）。
- 轻量化与压缩（MobileNet 家族、量化、蒸馏、早退/级联推理）。
- 移动端部署（TorchScript、PyTorch Mobile、torchao 潜力）。
- 我们的定位：侧重多数据集泛化、难例挖掘、显式阈值调优与在端可观测的系统权衡。

4. Datasets and Task（数据与任务）
- 任务定义：二分类（帧/图像级），视频按帧评估，标签 0=real，1=fake。
- 数据集：CelebDF‑v2、FaceForensics++、DeeperForensics、DFDC（训练/验证/测试，表格来自 manifests 行数）；Deepfake‑Eval‑2024 仅用于跨数据集验证/测试。
- 切分与采样：均衡采样（WeightedRandomSampler）、路径泄露检查、数据增强与归一化（Albumentations）。
- OOD 评估：报告校准/仅二级两种设置，说明分布差异带来的性能落差。

5. Method（方法）
- Stage1：MobileNetV4 轻量模型，BCE，均衡采样，关键超参（学习率、批大小、权重衰减、epoch 等）。
- Stage2（难例挖掘）：无增强打分，挑选误判和边界样本，生成难例清单。
- Stage3：EfficientNetV2‑b3 专家训练（正则化、AMP、早停），在合并验证上报告 AUC/F1 指标。
- Stage4（级联 + 阈值调优）：低/高阈值网格搜索 + 缓存 logits，目标是低 FNR + 高 F1 + 控制二级调用率。
- 校准：Stage2 温度标定与跨数据集阈值迁移。
- Stage5：跨数据集评估 + 鲁棒性（JPEG/噪声/模糊/亮度），自动生成表格/图。
- Stage6：TorchScript 导出 + 动态量化 + Android 集成。

6. Experimental Setup（实验设置）
- 环境与依赖（PyTorch、timm、Albumentations、scikit‑learn）。
- 关键指令：train_mobilenet / train_efficientnet / tune_cascade_system / eval_cascade / calibrate_temperature / export_torchscript。
- 超参：Stage1 与 Stage3 的默认值及必要变体；监控（梯度范数、早停、学习率调度）。
- 指标：AUC、F1、Accuracy、Precision/Recall、FNR/FPR、Stage‑2 升级率；部署时延与应用体积。
- 可复现性：outputs 目录结构、summary.json/CSV/图表、paper/generated 自动片段生成方式。

7. Results and Analysis（结果与分析）
- 单模型基线：Stage1 与 Stage2 ROC、校准、混淆矩阵（引用 outputs 对应图）。
- 级联阈值调优：最佳阈值组合（表格与可视化热力图）；FNR 与升级率权衡解读。
- 跨数据集（Deepfake‑Eval‑2024）：校准级联与二级仅使用的上限对比；对分布差异的讨论。
- 鲁棒性：四类扰动的 F1/Acc/FNR/S2 Rate 汇总表与曲线；阈值‑权衡图（trade‑off）。

8. Mobile（可暂时忽略）
- 暂不展开；保留导出工件表（paper/generated/mobile_tables.tex）。

9. Discussion（讨论）
- 系统权衡：阈值对 FNR 与 S2 升级率的影响、端上时延 VS 精度。
- 局限：分布外、扰动敏感、阈值与温度依赖、硬件差异。
- 展望：自适应阈值、蒸馏/架构搜索缩短二级时延、域适配与公平性评估。

10. Ethics and Societal Impact（伦理与影响）
- 防滥用：最小化在端日志、避免泄露敏感阈值、负责任披露与速率限制。
- 数据合规：遵循各数据集许可证（Deepfake‑Eval‑2024 的 gated 使用说明与 NSFW 提醒）。
- 公平性：数据多样性与潜在偏差、建议开展分组公平性审计。

11. Conclusion（结论）
- 方法与系统综述；关键指标回顾；在端落地与可复现性。
- 后续工作与开放问题：更强鲁棒性、蒸馏/量化、端侧加速与更大规模评测。

—
素材来源（可追溯）：
- 代码：`src/models/*`, `src/training/*`, `src/tools/*`, `src/utils/*`
- 配置与数据：`configs/*`, `manifests/*`, `Deepfake-Eval-2024/*`
- 输出：`outputs/stage*/run_*/`（图表、CSV、summary.json、best_config.json）
- 论文素材：`paper/generated/*.tex`（auto_tables、robustness、mobile 等）
