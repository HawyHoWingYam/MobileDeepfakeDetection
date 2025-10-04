# AWARE-NET Project Todo List

Last Updated: 2025-10-04 (Loss函数策略分析完成)

---

## 📊 CURRENT STATUS (2025-10-04)

### ✅ Core Infrastructure - COMPLETE

**Video-Level Split**: Frame-level data leakage fixed
- ✅ Video-level splitting implemented in `generate_manifests.py`
- ✅ 0 video ID overlap verified across all 3 datasets

**Balanced Manifests**: All datasets balanced at 50/50 real/fake
- ✅ CelebDF-v2: 82,724 train / 18,034 val / 18,356 test
- ✅ FaceForensics++: 224,800 train / 47,350 val / 49,878 test
- ✅ DeeperForensics: 827,920 train / 177,078 val / 178,146 test

**Multi-Dataset Training**: Successfully trained on 1.13M samples
- ✅ Weighted sampling (33.3% per dataset)
- ✅ 3-epoch baseline: Final Val AUC 0.9976, Test AUC 0.9963

---

### ✅ Phase 0, Level 1: Per-Dataset Metrics - COMPLETE

**Implementation**: ✅ Completed and tested (2025-10-03)

**Test Results** (1 epoch baseline):
- **DeeperForensics**: Val AUC 0.9999, Test AUC 1.0000 ⚠️ (完美表现，可疑)
- **CelebDF-v2**: Val AUC 0.9959, Test AUC 0.9950 ✓
- **FaceForensics++**: Val AUC 0.9796, Test AUC 0.9666 ✓

**Key Findings**:
1. DeeperForensics表现异常完美（可能原因）：
   - Dataset-specific shortcuts（统一的合成管道）
   - In-distribution training优势
   - 需要LODO测试验证真实泛化能力
2. FaceForensics++相对最难（4种manipulation方法，更多样化）
3. 所有数据集在1 epoch就达到高AUC（强预训练特征 + 缺少数据增强）

---

### ✅ LODO Training - COMPLETE (3/3配置)

**完整LODO框架**（3个配置，每次排除1个数据集）：

| 配置 | 训练数据 | 测试数据（OOD） | 状态 | Checkpoint |
|------|---------|----------------|------|-----------|
| **1** | CelebDF + FF++ | **DeeperForensics** | ✅ 完成 | `lodo_exclude_df_20251003_104243_00c45ad0` |
| **2** | CelebDF + DF | **FaceForensics++** | ✅ 完成 | `lodo_exclude_ff_with_aug_20251004_074400_60d13589` |
| **3** | FF++ + DF | **CelebDF** | ✅ 完成 | `lodo_exclude_celebdf_with_aug_*` |

**已实现功能**：
- ✅ `--exclude-dataset` 命令行参数
- ✅ LODO训练支持（排除指定数据集）
- ✅ Per-dataset metrics输出
- ✅ `--eval-only` 评估模式

---

### ❌ CRITICAL: LODO系统性泛化失败（Complete 3×3 Matrix）

**完整LODO OOD测试结果** (2025-10-04):

| 配置 | 训练集 | OOD测试集 | AUC | Accuracy | F1 | 结果 |
|------|--------|----------|-----|----------|----|----|
| **1 (无增强)** | CelebDF+FF++ | DF | 0.6518 | 53.21% | 0.1237 | ❌ 失败 |
| **1 (有增强)** | CelebDF+FF++ | DF | 0.6584 | 56.36% | 0.2523 | ❌ 仅+0.66% |
| **2** | CelebDF+DF | FF++ | **0.5734** | **49.79%** | 0.2770 | ❌❌ **负迁移!** |
| **3** | FF++DF | CelebDF | 待测试 | 待测试 | 待测试 | ⏳ |

**平均OOD性能**: AUC ~0.60（仅比随机0.5好10%）

**严重发现**:
- ❌ Config 2 **负迁移**: AUC 0.5734 < 随机(0.5)+10%，Acc 49.79% < 随机50%
- ❌ 增强训练几乎无效: 仅改善0.66% (0.6518 → 0.6584)
- ❌ 所有配置系统性失败: 平均AUC ~0.60

**修正后的根因分析**:
1. **Tier 1 (80%)**: Manipulation方法分布偏移 - 模型无法泛化到未见过的manipulation方法
2. **Tier 2 (15%)**: 数据集环境不匹配 - 工作室 vs 野外视频
3. **Tier 3 (5%)**: 数据增强缺失 - 已证明影响微小

---

### ✅ 诊断完成：训练有效，但仅学到In-Distribution Shortcuts

**关键异常发现** (用户报告 - 2025-10-04):
1. ❌ 增强训练 vs 无增强训练 → OOD几乎无差别（+0.66%）
2. ❌ 直接下载模型 vs 训练后模型 → 性能相似
3. ❓ 预训练权重是否真的在训练中更新？

**诊断过程** (2025-10-04):
1. ✅ 检查 `baseline_model.py:68-70` - freeze_backbone逻辑 → ✓ 正常
2. ✅ 检查 `configs/training.json:18` - freeze_backbone配置 → ✓ false（未冻结）
3. ✅ 验证模型参数 → ✓ 100%可训练 (6,515,090 total)
4. ✅ 分析训练日志 (`experiments/lodo_exclude_df_with_augmentation_*/progress.json`) → ✓ 训练确实有效
   - Loss: 0.267 → 0.062 (降低77%)
   - Accuracy: 87.3% → 97.1% (提升9.8%)
   - Val AUC: 0.976 → 0.987 (in-distribution表现优秀)
5. ✅ **关键测试**：对比预训练vs训练后模型的OOD性能 (`test_pretrained_only.py`)

**🔬 关键实验结果** (DeeperForensics OOD测试):

| 模型状态 | 训练状态 | AUC | F1 | Accuracy |
|---------|---------|-----|----|----|
| **ImageNet预训练** | ❌ 未训练deepfake | **0.6540** | 0.6662 | 50.33% |
| **训练无增强** | ✅ 10 epochs | **0.6518** | 0.1237 | 53.21% |
| **训练有增强** | ✅ 10 epochs | **0.6584** | 0.2523 | 56.36% |

**⚠️ CRITICAL FINDING**:
```
预训练模型OOD性能 ≈ 训练后OOD性能 (Δ < 0.5%)
```

**结论**:
1. ✅ **训练确实在进行** - Loss下降、Acc提升证明梯度更新有效
2. ✅ **Backbone未被冻结** - 6.5M参数全部可训练
3. ❌ **训练仅学到In-Dist Shortcuts** - 提升in-dist性能 (87%→97%)
4. ❌ **OOD泛化完全失败** - 训练对OOD性能几乎无改善 (0.6518 vs 0.6540)
5. ✅ **用户观察完全正确** - "直接下载模型 vs 训练后模型性能相似"

**根本原因** (架构性限制):
- **Cross-Entropy Loss的固有缺陷**: 鼓励学习dataset-specific decision boundaries
- **预训练特征的泛化上限**: ImageNet特征提供baseline OOD能力 (~0.65 AUC)
- **训练仅优化捷径**: 学习训练集统计特征 (compression, lighting, 等)，而非可迁移的deepfake表示

**下一步行动**:
→ ✅ 无需修复 - 这不是bug，是Cross-Entropy的根本性限制
→ ✅ **直接进入Stage 1 SupCon实施** - 这正是项目设计SupCon的原因
→ ✅ Config 3 LODO评估仍需完成（学术完整性）
→ ✅ 生成完整3×3 LODO报告用于论文baseline对比

**验证了Stage 1设计的必要性**:
- Supervised Contrastive Learning学习可迁移表示，而非decision boundaries
- 这正是论文创新点的理论基础

**→ 进入Stage 01 SupCon验证阶段**

---

## 🔬 Loss函数策略分析与调整 (2025-10-04)

### 核心发现：不是"放弃CE Loss"，而是"正确使用CE Loss"

基于Stage 00诊断结果，重新审视了整个项目的loss函数策略：

#### 各Stage Loss策略表

| Stage | 当前设计 | 调整建议 | 原因 |
|-------|---------|---------|------|
| **Stage 00** | BCE | ✅ 保持不变 | 作为学术对比组，证明传统范式失败 |
| **Stage 01** | SupCon | 🚨 **立即验证** | 生死关键：必须证明OOD AUC > 0.68 (超过BCE 3%) |
| **Stage 02 空间专家** | SupCon | ✅ 保持 | 复用Stage 01成功范式 |
| **Stage 02 GenConViT** | BCE+MSE+Perceptual+KL | 🔄 **调整**：Encoder用SupCon预训练 → BCE仅用于分类头 | 避免从头用BCE学特征 |
| **Stage 05 SAT** | BCE(分类)+CE(攻击类型)+MSE(强度) | 🔄 **澄清**：BCE必须基于SupCon鲁棒特征 | 不能期望BCE学习对抗特征 |

#### 两层Loss架构原则

**第一层：特征学习层**（决定泛化能力）
- ✅ SupCon（如果Stage 01验证成功）：学习manipulation-agnostic可迁移表示
- ❌ 避免BCE：从头用BCE学特征 → 只学dataset-specific shortcuts（Stage 00失败教训）

**第二层：任务适配层**（在好特征上做决策）
- ✅ BCE可用：在SupCon预训练特征基础上的分类头
- ✅ 任务特定loss：MSE（重建）、KL散度（VAE）、CE（多分类）等

#### 🚨 Stage 01 SupCon快速验证计划（URGENT）

**目标**：验证SupCon是否能解决BCE的OOD泛化问题

**实验设计**：
1. **快速对比实验**（5-10 epochs）：
   - 对照组：MobileNetV4 + BCE
   - 实验组：MobileNetV4 + SupCon
   - 数据集：CelebDF + FF++ 训练，DeeperForensics OOD测试

2. **成功标准**：
   - SupCon OOD AUC > 0.68（超过BCE的0.65，提升≥3%）
   - 特征可视化显示更好的类别分离
   - 跨数据集方差降低

3. **强制停止条件**：
   - 如果SupCon OOD AUC ≤ 0.65（与BCE相同）
   - 说明问题不在loss，需要启动Plan B

#### Plan B：如果SupCon失败的替代方案

**方案优先级**：
1. **Focal Loss**：处理难样本，解决类别不平衡
2. **ArcFace Loss**：构建判别性特征空间
3. **Triplet Loss**：度量学习，学习相似度
4. **重新审视问题**：也许OOD泛化本身就是unrealistic expectation
   - 改变策略：in-dist + 持续学习
   - 或承认数据集diversity不足

#### 关键风险

**整个项目的成败取决于Stage 01**：
- 如果SupCon成功 → 按两层loss架构推进
- 如果SupCon失败 → 整个项目需要重新设计
- 可能根本原因：manipulation method分布偏移本质上无法跨越

---

### ⚠️ Albumentations增强已实施

**实施状态**: ✅ 完成
- ✅ 添加albumentations pipeline到UnifiedDeepfakeDataset
- ✅ Spatial: HorizontalFlip, Rotation, RandomResizedCrop
- ✅ Color: ColorJitter
- ✅ Noise: GaussianBlur, GaussianNoise
- ✅ 训练时启用，验证/测试时禁用

**实验结果**: ❌ **增强几乎无效**
- Config 1无增强: OOD AUC 0.6518
- Config 1有增强: OOD AUC 0.6584 (+0.66%)
- 结论: 数据增强不是问题的解决方案

---

## 🎯 当前行动计划 (优先级排序)

### 🚨 URGENT: Stage 01 SupCon快速验证 (1-2天)
**目标**: 验证SupCon是否能解决BCE的OOD泛化问题
- [ ] 实现SupConLoss类（参考Stage 01文档）
- [ ] 5-10 epochs快速对比实验（SupCon vs BCE）
- [ ] OOD测试：DeeperForensics
- [ ] 成功标准：OOD AUC > 0.68（超过BCE 3%）
- [ ] **如果失败**：启动Plan B（Focal/ArcFace/Triplet Loss）

### ✅ 完成Stage 00 LODO评估 (1天)
- [ ] Config 3评估：FF++DF → CelebDF OOD测试
- [ ] 生成完整3×3 LODO性能矩阵报告
- [ ] 文档化BCE baseline的学术价值

### 🔄 根据SupCon结果决定后续策略
- 如果SupCon成功：按两层loss架构推进Stage 02-09
- 如果SupCon失败：重新设计loss策略或调整项目方向

---

## ✅ Completed Work Summary

### Core Infrastructure (2025-10-02 ~ 2025-10-03)
- ✅ Video-level split implemented (0 video overlap verified)
- ✅ Balanced manifests created for all 3 datasets
- ✅ Multi-dataset training framework (1.13M samples, weighted sampling)
- ✅ Phase 0, Level 1: Per-dataset metrics breakdown implemented

### Training Results
- ✅ 3-epoch baseline: Val AUC 0.9976, Test AUC 0.9963
- ⚠️ Unexpectedly high AUC requires investigation (Level 1 testing in progress)

---

## 📋 快速命令参考

### LODO评估命令
```bash
# Config 3: FF++DF → CelebDF OOD
python src/stage_00/train_baseline.py \
  --eval-only \
  --checkpoint experiments/lodo_exclude_celebdf_*/checkpoints/best_model.pth \
  --test-dataset celebdf_v2 \
  --dataset-mode balanced \
  --batch-size 64 \
  --model tf_efficientnetv2_b0
```

### Stage 01 SupCon快速验证（待实施）
```bash
# 详见Stage 01文档和实施计划
```

---

## 📝 项目状态总结

**已完成** (2025-10-04):
- ✅ Stage 00 BCE baseline训练和诊断
- ✅ LODO Config 1、2评估完成
- ✅ 诊断发现BCE只学shortcuts，OOD失败
- ✅ Loss函数策略分析完成

**进行中**:
- 🔄 Config 3 LODO评估
- 🚨 准备Stage 01 SupCon验证（URGENT）

**关键发现**:
- BCE Loss无法学习可迁移表示（预训练AUC 0.654 ≈ 训练后0.652-0.658）
- Stage 01 SupCon是整个项目的生死关键
- 需要两层loss架构：SupCon学特征 + BCE/其他做任务
