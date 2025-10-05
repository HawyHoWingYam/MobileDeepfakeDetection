# AWARE-NET Project Todo List

Last Updated: 2025-10-05 (Stage 01 LODO全面失败确认 + 诊断工具完成)

---

## 📊 CURRENT STATUS (2025-10-05)

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

## 🚨 CRITICAL: Stage 01 完全失败确认 (2025-10-05)

### ❌ LODO方法论：系统性泛化失败

**完整LODO Baseline评估结果**:

| 配置 | 训练集 | OOD测试集 | Epochs | Val AUC | OOD AUC | OOD Acc | 泛化差距 | 状态 |
|------|--------|----------|--------|---------|---------|---------|---------|------|
| **Exclude DF** | CelebDF + FF++ | DeeperForensics | 50 | 0.999 | **0.732** | 48.9% | -26.7% | ❌ 失败 |
| **Exclude CelebDF** | DF + FF++ | CelebDF | 5 | 0.997 | **0.720** | 48.8% | -27.7% | ❌ 失败 |
| **Exclude FF++** | CelebDF + DF | FF++ | 10 | 0.998 | **0.572** | 32.7% | -42.6% | ❌❌ 惨败 |

**LODO Baseline平均性能**:
- **平均OOD AUC**: **0.675** (远低于0.85目标)
- **平均泛化差距**: **-32.3%** (Val 0.998 → OOD 0.675)
- **最好配置**: Exclude DF (0.732)
- **最差配置**: Exclude FF++ (0.572) - 接近随机猜测

---

### ❌ SupCon方法：劣于Baseline

**SupCon两阶段训练结果** (Stage 1: 50 epochs, Stage 2: 20 epochs):

| 方法 | 训练集 | OOD测试集 | Val AUC | OOD AUC | 与Baseline对比 | 状态 |
|------|--------|----------|---------|---------|---------------|------|
| **SupCon** | CelebDF + FF++ | DeeperForensics | 0.969 | **0.665** | -6.7% | ❌ 失败 |
| **Baseline** | CelebDF + FF++ | DeeperForensics | 0.999 | **0.732** | 基准 | ❌ 失败 |

**关键发现**:
1. ❌ **SupCon < Baseline**: OOD AUC 0.665 vs 0.732，差距6.7%
2. ❌ **Stage 01假设失败**: SupCon并未改善跨数据集泛化
3. ❌ **两阶段训练问题**: 可能增加过拟合风险
4. ✅ **Baseline更可靠**: 训练更快（50 vs 70 epochs），性能更好

---

### ✅ 诊断工具完成 (2025-10-05)

**工具**: `tools/validation/model_diagnostics.py`

**功能**:
- ✅ ROC曲线分析 + 最佳阈值检测
- ✅ 混淆矩阵可视化（可调阈值）
- ✅ 阈值优化曲线（Accuracy/F1/Precision/Recall）
- ✅ 预测概率分布分析
- ✅ 支持Baseline和SupCon模型
- ✅ CLI接口 + Python API

**诊断结果 (Baseline on DeeperForensics)**:
```
默认阈值0.5: Acc 48.9%, Precision 99.7%, Recall 14.3%
最佳阈值0.0: Acc 66.3%, Precision 83.7%, Recall 53.9%
预测分布: Real mean=0.0012, Fake mean=0.1387 (分离度极差)
```

**结论**:
- ❌ **严重校准失败**: 模型预测概率集中在0附近
- ❌ **即使最佳阈值也差**: 最优Acc仅66.3%
- ❌ **不是阈值问题**: 是模型根本无法区分OOD样本

---

### ✅ 训练轮数分析：更多训练无帮助

**对比实验**:
| 配置 | Epochs | OOD AUC | 差距 |
|-----|--------|---------|------|
| Exclude DF | 50 | 0.732 | 基准 |
| Exclude CelebDF | 5 | 0.720 | -1.2% |
| Exclude FF++ | 10 | 0.572 | - |

**结论**:
- ✅ 5 epochs vs 50 epochs差距仅1.2%
- ✅ **更多训练不会改善OOD泛化**
- ✅ 问题在于域偏移，不是训练不足

---

## 🎯 Stage 01 最终决策

### ❌ 放弃LODO方法论
**理由**:
1. 3/3配置失败，平均OOD AUC仅0.675
2. FF++的0.572接近随机猜测
3. 单模型无法跨数据集泛化
4. 继续优化是沉没成本

### ❌ 放弃SupCon方法
**理由**:
1. SupCon (0.665) < Baseline (0.732)
2. 已被证明在跨数据集场景下无效
3. 训练时间更长，性能更差

### ✅ 调整Stage 01定位
**原定位**: SupCon快速过滤器 + OOD泛化能力(AUC 0.85+)
**新定位**: Baseline高性能过滤器 + In-distribution专精(AUC 0.95+)

**理由**:
1. 承认单模型跨域泛化的根本困难
2. Stage 01作为第一层粗过滤（高召回率）
3. 依赖Stage 02-04异构专家提升鲁棒性
4. 负面结果也是学术贡献

---

## 📋 当前行动计划 (2025-10-05 更新)

> ⚠️ `experiments/` 已清空，Stage 00 baseline 需全部重建后再推进 Stage 01/02。

### 🚀 Phase 1: 恢复 Stage 00 基线 + 全数据集训练 (1-1.5天)

**目标**: 重建 LODO Config 3、补齐 DFDC baseline，并恢复多数据集 in-distribution 过滤器**

#### Step 0: 重新生成 manifest / 校验配置
- 重新运行 CelebDF、FF++、DeeperForensics 的 manifest 与 `data_validator`
- DFDC 数据准备与 manifest 生成流程已集中整理在 `docs/OPERATIONS.md`
- 更新 `configs/datasets.json`（加入 dfdc、调整权重）及 `configs/training.json`

#### Step 1: 重跑 LODO Config 3（FF++ + DF → CelebDF）
```bash
python src/stage_00/train_baseline.py \
  --config configs/training.json \
  --model tf_efficientnetv2_b0 \
  --epochs 50 \
  --batch-size 128 \
  --multi-dataset \
  --exclude-dataset celebdf_v2 \
  --experiment-name baseline_lodo_config3_restart
```
- 生成新的 checkpoint / metrics，补齐 3×3 矩阵

#### Step 2: 3数据集全训练（基础回归）⭐
```bash
python src/stage_00/train_baseline.py \
  --config configs/training.json \
  --model tf_efficientnetv2_b0 \
  --epochs 50 \
  --batch-size 128 \
  --multi-dataset \
  --experiment-name baseline_full_3datasets_final
```
- 目标：In-dist AUC > 0.95

#### Step 3: 4数据集全训练（DFDC 加入后）
```bash
python src/stage_00/train_baseline.py \
  --config configs/training.json \
  --model tf_efficientnetv2_b0 \
  --epochs 50 \
  --batch-size 128 \
  --multi-dataset \
  --experiment-name baseline_full_4datasets_with_dfdc
```
- 目标：更广泛覆盖，记录 DFDC 的 in-dist 性能

#### Step 4: 更新文档与报告
- 将结果写入 `project_instruction/stage_00/stage_00_status_report.md`
- 在 `docs/stage_00/` 保存新版 3×3 LODO 矩阵（含 DFDC）

---

### 📝 Phase 2: 文档化与Stage Gate (1天)

#### 必须创建的文档

**1. LODO失败分析报告**
- 文件: `docs/stage_01/lodo_failure_analysis.md`
- 内容:
  - 3个LODO配置详细结果
  - 域偏移严重性量化
  - 单模型跨域泛化困难的原因
  - 对Stage 02的启示（异构专家的必要性）

**2. SupCon失败分析**
- 文件: `docs/stage_01/supcon_failure_analysis.md`
- 内容:
  - SupCon vs Baseline对比
  - 为什么SupCon在跨数据集场景下失败
  - 对比学习的局限性
  - 学术价值：负面结果

**3. Stage 01定位调整说明**
- 文件: `docs/stage_01/stage_01_revised_scope.md`
- 内容:
  - 原计划 vs 调整后计划
  - 新定位：In-distribution高性能过滤器
  - 不追求OOD泛化的理由
  - Stage 02-04如何弥补

**4. Stage Gate决策文档**
- 文件: `docs/stage_01/stage_gate_decision.md`
- 内容:
  - 技术Gate: ❌ OOD泛化未达标，✅ In-dist性能优秀
  - 学术Gate: ✅ 实验严谨，负面结果有价值
  - 系统Gate: ✅ 代码可复现，诊断工具完善
  - **决策**: Pivot - 调整Stage 01定位，进入Stage 02

---

### 🔬 Phase 3: 准备进入Stage 02 (剩余7-8天)

**Stage 02核心**: 异构专家系统

**关键组件**:
1. **空间专家**: 基于CNN的空域artifact检测
2. **频域专家**: FFT/DCT频谱分析
3. **时序专家**: 帧间不一致性检测
4. **融合模块**: LightGBM meta-learner

**Stage 01为Stage 02提供的价值**:
- ✅ 高质量in-distribution过滤器（第一层）
- ✅ 证明单模型局限性（异构专家的必要性）
- ✅ 完整的LODO baseline对比数据
- ✅ 可复用的诊断工具框架

---

## 📊 实验结果总结

### Baseline (BCE Loss)
| 指标 | In-Distribution | OOD (平均) | 差距 |
|-----|----------------|-----------|------|
| **AUC** | 0.998 | 0.675 | -32.3% |
| **Accuracy** | 99%+ | 43.5% | -55.5% |
| **F1** | 0.99+ | 0.38 | -61% |

### SupCon (两阶段)
| 指标 | In-Distribution | OOD (DF) | vs Baseline |
|-----|----------------|---------|------------|
| **AUC** | 0.969 | 0.665 | -6.7% |
| **Accuracy** | 91.3% | 58.7% | - |

### 诊断结果
- ❌ 预测概率严重偏向0（模型过于保守）
- ❌ 最佳阈值仅能达到66% Accuracy
- ❌ Real vs Fake预测分离度极差（0.14）

---

## ✅ Completed Work Summary (Updated 2025-10-05)

### Core Infrastructure
- ✅ Video-level split (0 video overlap)
- ✅ Balanced manifests (3 datasets)
- ✅ Multi-dataset training framework
- ✅ Per-dataset metrics breakdown

### Stage 00 Baseline
- ✅ 3-epoch baseline training
- ✅ Complete LODO evaluation (3 configs)
- ✅ Data augmentation testing
- ✅ Pretrained vs trained comparison

### Stage 01 SupCon
- ✅ Two-stage SupCon implementation
- ✅ SupCon vs Baseline comparison
- ✅ OOD evaluation on DeeperForensics
- ❌ **结论**: SupCon失败，不如Baseline

### Diagnostic Tools
- ✅ `tools/validation/model_diagnostics.py`
- ✅ ROC curve + optimal threshold
- ✅ Confusion matrix visualization
- ✅ Threshold optimization analysis
- ✅ Prediction distribution plots

### Documentation
- ✅ Training logs and analysis
- ⏳ **待完成**: 失败分析报告
- ⏳ **待完成**: Stage Gate文档

---

## 🚫 已废弃/不合时宜的内容

### ❌ 删除：Stage 01 SupCon快速验证计划
**原因**: 已完成完整SupCon训练（50+20 epochs），结果证明失败

### ❌ 删除：LODO优化计划
**原因**:
- 所有LODO配置已评估，全部失败
- 更多训练轮数无帮助（5 vs 50 epochs差距仅1.2%）
- 继续优化LODO是沉没成本

### ❌ 删除：数据增强作为主要优化方向
**原因**: 已验证增强仅改善0.66% OOD性能，不是瓶颈

### ❌ 删除：Plan B (Focal/ArcFace/Triplet Loss)
**原因**:
- Baseline已经是最简单有效的方法
- 问题不在loss函数，在于任务本质困难
- 直接进入Stage 02异构专家更合理

---

## 🎯 关键风险与应对

### 风险1: 全数据集训练可能也无法达到0.95 AUC
**应对**:
- 接受现实，调整期望到0.90-0.93
- 重点在于Stage 02的提升空间

### 风险2: Stage 02异构专家也无法解决跨域问题
**应对**:
- 调整项目定位：不追求universal detector
- 改为domain-adaptive detector
- 或接受需要持续学习的现实

### 风险3: 时间预算不足（剩余7-8天）
**应对**:
- Stage 02只实现核心专家（空间+频域）
- 简化融合模块
- Stage 03-09根据时间调整

---

## 📝 Quick Command Reference

### 全数据集训练
```bash
# 3数据集
python src/stage_00/train_baseline.py \
  --model tf_efficientnetv2_b0 \
  --epochs 50 \
  --batch-size 128 \
  --multi-dataset \
  --experiment-name baseline_full_3datasets_final
```

### 诊断工具使用
```bash
python -m tools.validation.model_diagnostics \
  --checkpoint experiments/[exp_name]/checkpoints/best_model.pth \
  --model-type baseline \
  --test-dataset deeperforensics_1_0 \
  --output-dir experiments/[exp_name]/diagnostics
```

### OOD评估
```bash
python src/stage_00/train_baseline.py \
  --eval-only \
  --checkpoint experiments/[exp_name]/checkpoints/best_model.pth \
  --test-dataset [dataset_name] \
  --model tf_efficientnetv2_b0
```

---

## 📌 项目当前状态

**当前阶段**: Stage 01收尾 → Stage 02准备

**已完成**:
- ✅ Stage 00 BCE baseline (完整LODO评估)
- ✅ Stage 01 SupCon验证 (失败确认)
- ✅ 诊断工具开发
- ✅ 失败原因分析

**进行中**:
- 🔄 全数据集训练（立即启动）
- 🔄 Stage 01文档化

**下一步**:
- 📝 Stage Gate评审
- 🚀 Stage 02异构专家系统设计与实现

**关键时间节点**:
- Day 1-2: 全数据集训练
- Day 3: 文档化 + Stage Gate
- Day 4-10: Stage 02实现

**成功指标**:
- ✅ Stage 01: In-dist AUC > 0.90（可接受）
- ⭐ Stage 02: 异构专家系统OOD AUC > 0.80（目标）
- 🎯 整体: 学术贡献（负面结果 + 创新方案）
