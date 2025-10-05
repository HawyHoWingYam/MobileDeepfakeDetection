 加入DFDC的最佳策略：Baseline三阶段渐进式方法                                                                 │
     │                                                                                                              │
     │ ✅ 方法确认：必须用Baseline，不用SupCon                                                                       │
     │                                                                                                              │
     │ 原因:                                                                                                        │
     │ - Baseline OOD AUC 0.732 > SupCon OOD AUC 0.665 (+6.7%)                                                      │
     │ - SupCon两阶段训练在跨数据集场景下已被证明无效                                                               │
     │ - Baseline训练更快（50 epochs vs 50+20 epochs）                                                              │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ 📋 三种策略对比                                                                                              │
     │                                                                                                              │
     │ 策略1️⃣: DFDC增强训练（推荐）⭐                                                                               │
     │                                                                                                              │
     │ 训练: CelebDF + FF++ + DFDC → 测试: DeeperForensics                                                          │
     │                                                                                                              │
     │ 时间成本: 1.5-2天                                                                                            │
     │ - Day 1: DFDC数据准备（manifest生成）                                                                        │
     │ - Day 1.5: Baseline训练（12-16小时）                                                                         │
     │ - Day 2: OOD评估                                                                                             │
     │                                                                                                              │
     │ 预期收益:                                                                                                    │
     │ - 如果DFDC与DeeperForensics相似 → AUC可能从0.73提升到0.80-0.85 ✅                                             │
     │ - 如果DFDC是不同域 → 可能无改善甚至下降 ❌                                                                    │
     │ - 成功率: 30-40%                                                                                             │
     │                                                                                                              │
     │ 适用场景: 相信更多训练数据能改善DeeperForensics OOD性能                                                      │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ 策略2️⃣: 完整4数据集LOD                                                                                      │
     │                                                                                                              │
     │ 训练4个配置:                                                                                                 │
     │ 1. CelebDF + FF++ + DFDC → 测试 DeeperForensics                                                              │
     │ 2. DeeperForensics + FF++ + DFDC → 测试 CelebDF                                                              │
     │ 3. CelebDF + DeeperForensics + DFDC → 测试 FF++                                                              │
     │ 4. CelebDF + FF++ + DeeperForensics → 测试 DFDC                                                              │
     │                                                                                                              │
     │ 时间成本: 3天                                                                                                │
     │ - 4个配置 × 12小时 = 48小时训练                                                                              │
     │ - 数据准备 + 评估 = 额外1天                                                                                  │
     │                                                                                                              │
     │ 预期收益:                                                                                                    │
     │ - 学术完整性最高，全面了解数据集间关系                                                                       │
     │ - 但如果所有LODO都失败，结论只是"LODO不可行"                                                                 │
     │ - 学术价值: 高，实用价值: 中                                                                                 │
     │                                                                                                              │
     │ 适用场景: 追求学术严谨性，有充足时间预算                                                                     │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ 策略3️⃣: DFDC快速OOD评估（最快）                                                                             │
     │                                                                                                              │
     │ 不改变训练: 用现有Baseline checkpoint → 测试: DFDC                                                           │
     │                                                                                                              │
     │ 时间成本: 1天                                                                                                │
     │ - DFDC数据准备（manifest）                                                                                   │
     │ - OOD评估（2小时）                                                                                           │
     │                                                                                                              │
     │ 预期收益:                                                                                                    │
     │ - 快速了解DFDC难度                                                                                           │
     │ - 为后续决策提供信息                                                                                         │
     │ - 风险: 如果DFDC也失败，没有解决方案                                                                         │
     │                                                                                                              │
     │ 适用场景: 先探路，再决定是否深度投入                                                                         │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ 🎯 推荐方案：渐进式三阶段执行                                                                                │
     │                                                                                                              │
     │ Phase 1: 快速验证（1天）⭐ 立即执行                                                                           │
     │                                                                                                              │
     │ 步骤1.1: 准备DFDC数据                                                                                        │
     │                                                                                                              │
     │ # 假设DFDC数据已下载到dataset/dfdc/                                                                          │
     │ python tools/data/generate_manifests.py dfdc \                                                               │
     │   --config configs/datasets.json \                                                                           │
     │   --data-dir dataset/dfdc/ \                                                                                 │
     │   --output-dir manifests/                                                                                    │
     │                                                                                                              │
     │ 步骤1.2: 快速OOD评估                                                                                         │
     │                                                                                                              │
     │ # 用现有Baseline checkpoint测试DFDC                                                                          │
     │ python src/stage_00/train_baseline.py \                                                                      │
     │   --eval-only \                                                                                              │
     │   --checkpoint experiments/baseline_lodo_df_comparison_20251004_194410_440eaa09/checkpoints/best_model.pth \ │
     │   --test-dataset dfdc \                                                                                      │
     │   --model tf_efficientnetv2_b0 \                                                                             │
     │   --batch-size 128                                                                                           │
     │                                                                                                              │
     │ 决策点1: 根据DFDC OOD结果决定Phase 2                                                                         │
     │ - DFDC AUC < 0.6 → DFDC极难，考虑策略1                                                                       │
     │ - DFDC AUC 0.6-0.8 → 与DeeperForensics类似，策略1可能有帮助                                                  │
     │ - DFDC AUC > 0.85 → DFDC简单，无需特别处理                                                                   │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ Phase 2: 条件训练（1.5天）根据Phase 1结果决定                                                                │
     │                                                                                                              │
     │ 选项2A: DFDC增强训练（如果Phase 1显示DFDC有价值）                                                            │
     │                                                                                                              │
     │ # 4数据集训练，测试DeeperForensics                                                                           │
     │ python src/stage_00/train_baseline.py \                                                                      │
     │   --model tf_efficientnetv2_b0 \                                                                             │
     │   --epochs 50 \                                                                                              │
     │   --batch-size 128 \                                                                                         │
     │   --multi-dataset \                                                                                          │
     │   --exclude-dataset deeperforensics_1_0 \                                                                    │
     │   --experiment-name baseline_4dataset_lodo_df                                                                │
     │                                                                                                              │
     │ # 注意：需要先在configs/datasets.json中添加DFDC配置                                                          │
     │                                                                                                              │
     │ 选项2B: 跳过DFDC，测试其他LODO（如果Phase 1显示DFDC无帮助）                                                  │
     │                                                                                                              │
     │ # LODO-CelebDF                                                                                               │
     │ python src/stage_00/train_baseline.py \                                                                      │
     │   --model tf_efficientnetv2_b0 \                                                                             │
     │   --epochs 50 \                                                                                              │
     │   --batch-size 128 \                                                                                         │
     │   --exclude-dataset celebdf_v2 \                                                                             │
     │   --experiment-name baseline_lodo_celebdf                                                                    │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ Phase 3: 完整评估（可选，2天）仅当时间充足                                                                   │
     │                                                                                                              │
     │ 如果Phase 2成功且有时间，运行完整4数据集LODO矩阵                                                             │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ 🔄 并行优化策略（推荐）                                                                                      │
     │                                                                                                              │
     │ 不要把所有希望放在DFDC上，同时进行：                                                                         │
     │                                                                                                              │
     │ 线A: DFDC探索（1-2.5天）                                                                                     │
     │                                                                                                              │
     │ - Phase 1: 快速OOD评估                                                                                       │
     │ - Phase 2: 条件训练                                                                                          │
     │                                                                                                              │
     │ 线B: 现有数据集LODO（2天，可并行）                                                                           │
     │                                                                                                              │
     │ # 同时启动LODO-CelebDF和LODO-FF++训练                                                                        │
     │ python src/stage_00/train_baseline.py \                                                                      │
     │   --model tf_efficientnetv2_b0 \                                                                             │
     │   --epochs 50 \                                                                                              │
     │   --batch-size 128 \                                                                                         │
     │   --exclude-dataset celebdf_v2 \                                                                             │
     │   --experiment-name baseline_lodo_celebdf &                                                                  │
     │                                                                                                              │
     │ python src/stage_00/train_baseline.py \                                                                      │
     │   --model tf_efficientnetv2_b0 \                                                                             │
     │   --epochs 50 \                                                                                              │
     │   --batch-size 128 \                                                                                         │
     │   --exclude-dataset faceforensics_plus_plus \                                                                │
     │   --experiment-name baseline_lodo_ff &                                                                       │
     │                                                                                                              │
     │ 决策矩阵                                                                                                     │
     │                                                                                                              │
     │ | 线A (DFDC)     | 线B (其他LODO) | 最终决策                     |                                           │
     │ |---------------|-------------|--------------------------|                                                   │
     │ | 成功 (AUC>0.85) | 成功          | ✅ Stage 01成功，进入Stage 02  |                                          │
     │ | 成功            | 失败          | ⚠️ 部分成功，文档化，进Stage 02    |                                     │
     │ | 失败            | 成功          | ⚠️ DeeperForensics特殊，可接受 |                                         │
     │ | 失败            | 失败          | ❌ LODO失败，全数据集训练          |                                      │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ ⏱️ 时间预算分配                                                                                              │
     │                                                                                                              │
     │ 总可用时间: 10.5天（1.5周）                                                                                  │
     │                                                                                                              │
     │ 保守方案:                                                                                                    │
     │ - Phase 1 (DFDC快速评估): 1天                                                                                │
     │ - Phase 2 (条件训练): 1.5天                                                                                  │
     │ - 线B (其他LODO): 2天（可并行）                                                                              │
     │ - 诊断分析 + 文档: 1天                                                                                       │
     │ - 总计: 3-4天                                                                                                │
     │ - 剩余: 6.5-7.5天用于Stage 02                                                                                │
     │                                                                                                              │
     │ 激进方案:                                                                                                    │
     │ - 完整4数据集LODO: 3天                                                                                       │
     │ - 诊断分析: 1天                                                                                              │
     │ - 总计: 4天                                                                                                  │
     │ - 剩余: 6.5天用于Stage 02                                                                                    │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ 📝 具体执行步骤（Phase 1 立即开始）                                                                          │
     │                                                                                                              │
     │ Day 1 上午: DFDC数据准备                                                                                     │
     │                                                                                                              │
     │ 1. 确认DFDC数据位置                                                                                          │
     │ 2. 生成train/val/test manifests                                                                              │
     │ 3. 验证数据完整性（抽样检查）                                                                                │
     │                                                                                                              │
     │ Day 1 下午: 快速OOD评估                                                                                      │
     │                                                                                                              │
     │ 1. 运行现有Baseline在DFDC上评估                                                                              │
     │ 2. 运行诊断工具分析结果                                                                                      │
     │ 3. 决策: 进入Phase 2A还是2B                                                                                  │
     │                                                                                                              │
     │ Day 2-3: Phase 2执行                                                                                         │
     │                                                                                                              │
     │ - 根据Day 1结果选择2A或2B                                                                                    │
     │ - 同时启动线B（其他LODO训练）                                                                                │
     │                                                                                                              │
     │ Day 4: 分析与决策                                                                                            │
     │                                                                                                              │
     │ - 汇总所有LODO结果                                                                                           │
     │ - 生成对比报告                                                                                               │
     │ - Stage Gate决策                                                                                             │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ ✅ 最终建议                                                                                                   │
     │                                                                                                              │
     │ 如果想加入DFDC：                                                                                             │
     │                                                                                                              │
     │ 1. ✅ 用Baseline，不用SupCon                                                                                  │
     │ 2. ✅ 采用渐进式策略：先Phase 1快速评估（1天），再决定是否Phase 2训练                                         │
     │ 3. ✅ 并行执行：同时运行其他LODO，不依赖单一路径                                                              │
     │ 4. ✅ 时间门槛：总投入控制在3-4天，为Stage 02保留≥6天                                                         │
     │                                                                                                              │
     │ 成功标准：                                                                                                   │
     │                                                                                                              │
     │ - 最小成功: 至少1个LODO配置AUC ≥ 0.85                                                                        │
     │ - 理想成功: 3/4个LODO配置AUC ≥ 0.85                                                                          │
     │ - 退出条件: 4天后仍无1个成功，立即停止，转全数据集训练                                                       │
     │                                                                                                              │
     │ ---                                                                                                          │
     │ 🚀 立即行动                                                                                                  │
     │                                                                                                              │
     │ Phase 1（今天）: 准备DFDC数据 + 快速OOD评估                                                                  │
     │ 明天: 根据结果启动Phase 2A或2B                                                                               │
     │ 后天: 等待训练完成，分析结果                                                                                 │
     │ 第4天: Stage Gate决策，进入Stage 02准备 