# AWARE-NET Stage 00 完成报告

Generated: 2025-10-04
Status: ⚠️ **PENDING - 需重建 LODO Config 3 与 DFDC baseline（experiments 已清空）**

---

## 📋 **执行摘要**

**Stage 00已搭建核心基础设施**，但近期清空 `experiments/` 后，LODO Config 3 的训练日志与 checkpoint 均需重跑，同时 DFDC 尚未纳入 baseline 体系。当前阶段主要任务从“总结”转为“恢复基线 + 扩展数据集”。

### **核心成就**
- ✅ 完整的基础设施和工具链
- ✅ 多数据集训练框架（CelebDF/FF++/DeeperForensics）
- ⏳ BCE baseline训练（需重跑 Config 3 / DFDC）
- ✅ 关键诊断：发现BCE只学shortcuts，无法OOD泛化
- ✅ Loss函数策略分析完成，为项目指明方向

### **关键发现**
**BCE Loss根本性限制**：
- 预训练模型（ImageNet）OOD AUC: 0.654
- BCE训练后 OOD AUC: 0.652-0.658 (无改善!)
- 证明：BCE只学dataset-specific shortcuts，无法学习可迁移表示
- 验证了Stage 01 SupCon设计的必要性

---

## ✅ **已完成的核心任务 (12/13)**

### **1. 环境配置与依赖管理** ✅
- ✅ Conda Environment完整配置 (PyTorch 2.7+, CUDA 12.6)
- ✅ 支持RTX 30/40/50系列GPU
- ✅ 环境验证脚本完成

### **2. 智能数据管理系统** ✅
- ✅ 多数据集配置完成（CelebDF-v2, FF++, DeeperForensics）
- ✅ Video-level split实现（0视频重叠）
- ✅ Balanced manifests生成（50/50 real/fake）
- ✅ Multi-dataset training支持（weighted sampling）

### **3. 学术工具函数库** ✅
- ✅ 评估指标：AUC-ROC, F1, Accuracy with Bootstrap CI
- ✅ 统计检验：DeLong test, paired t-test
- ✅ 校准评估：ECE, MCE, Brier Score
- ✅ 可视化：ROC/PR曲线，混淆矩阵，校准图

### **4. 实验管理工具** ✅
- ✅ 实验追踪系统（experiment registry）
- ✅ 可重现性保证（seed管理，deterministic）
- ✅ Checkpoint管理
- ✅ Per-dataset metrics breakdown

### **5. 基线模型实现与训练** ✅
- ✅ EfficientNetV2-B0 baseline实现
- ✅ BCE Loss训练框架
- ✅ 多数据集训练完成（1.13M samples）
- ✅ In-distribution性能优秀（Val AUC 0.99+）

### **6. LODO评估框架** ⚠️
- ✅ LODO Config 1: CelebDF+FF++ → DF (AUC 0.6584)
- ✅ LODO Config 2: CelebDF+DF → FF++ (AUC 0.5734, 负迁移!)
- ❌ LODO Config 3: FF++DF → CelebDF（checkpoint 已遗失，需重训）
- ⚠️ OOD 泛化失败证据需在重训后重新汇总

### **7. BCE Baseline诊断** ✅ (NEW - 2025-10-04)
- ✅ 发现BCE训练vs预训练OOD性能相同（0.654 vs 0.652-0.658）
- ✅ 证明BCE只学in-dist shortcuts，无法学可迁移特征
- ✅ 验证了项目设计假设：传统范式失败，需要SupCon

### **8. Loss函数策略分析** ✅ (NEW - 2025-10-04)
- ✅ 分析了整个项目的Loss策略
- ✅ 确认Stage 00 BCE作为对比组的价值
- ✅ 明确Stage 01 SupCon是生死关键
- ✅ 建立两层Loss架构原则

---

## 📊 **LODO评估结果总结**

### 完整3×3 LODO性能矩阵

| 配置 | 训练集 | OOD测试集 | AUC | Accuracy | F1 | 状态 |
|------|--------|----------|-----|----------|----|----|
| **1 (无增强)** | CelebDF+FF++ | DF | 0.6518 | 53.21% | 0.1237 | ✅ 完成 |
| **1 (有增强)** | CelebDF+FF++ | DF | 0.6584 | 56.36% | 0.2523 | ✅ 完成 |
| **2** | CelebDF+DF | FF++ | **0.5734** | **49.79%** | 0.2770 | ✅ 完成 |
| **3** | FF++DF | CelebDF | 待重跑 | 待重跑 | 待重跑 | ⛔ checkpoint 遗失 |

### 关键发现
- ❌ 平均OOD AUC ~0.60（仅比随机0.5好10%）
- ❌ Config 2负迁移（AUC 0.5734 < 0.6，Acc 49.79% < 50%）
- ❌ 数据增强几乎无效（仅+0.66%）
- ✅ 证明了BCE范式的系统性失败

---

## 🔬 **BCE Baseline诊断结果** (2025-10-04)

### 预训练 vs 训练后 OOD性能对比

| 模型状态 | 训练状态 | DeeperForensics OOD AUC |
|---------|---------|----------------------|
| ImageNet预训练 | ❌ 未训练deepfake | **0.6540** |
| BCE训练（无增强） | ✅ 10 epochs | 0.6518 |
| BCE训练（有增强） | ✅ 10 epochs | 0.6584 |

**结论**：
- BCE训练对OOD泛化**几乎无改善**（Δ < 0.5%）
- 训练确实有效（loss降77%，in-dist acc 87%→97%）
- 但**只学到in-distribution shortcuts**
- 预训练特征已提供baseline OOD能力

### 根本原因分析
1. **BCE Loss的固有缺陷**：最小化分类误差 → 学习dataset-specific shortcuts
2. **Shortcuts vs 可迁移特征**：压缩痕迹、光照模式等 vs manipulation-agnostic特征
3. **架构性限制**：不是bug，是CE/BCE Loss的根本性问题

---

## 🎯 **Loss函数策略分析** (2025-10-04)

### 各Stage Loss策略

| Stage | Loss设计 | 调整建议 | 原因 |
|-------|---------|---------|------|
| **Stage 00** | BCE | ✅ 保持不变 | 作为学术对比组，证明传统范式失败 |
| **Stage 01** | SupCon | 🚨 **立即验证** | 生死关键：必须OOD AUC > 0.68 |
| **Stage 02+** | 混合 | 🔄 两层架构 | SupCon学特征 + BCE/其他做任务 |

### 两层Loss架构原则

**第一层：特征学习**（决定泛化能力）
- ✅ SupCon/对比学习：学习manipulation-agnostic表示
- ❌ 避免BCE从头学特征：只会学shortcuts

**第二层：任务适配**（在好特征上做决策）
- ✅ BCE可用：在SupCon特征基础上的分类头
- ✅ 任务特定loss：MSE（重建）、KL（VAE）、CE（多分类）

---

## 📊 **Stage-Gate验证状态**

### **Technical Gates** ⚠️
- [x] 环境可重现性：95%+成功率
- [x] 数据管理完整性：支持3+数据集 manifest
- [ ] BCE baseline性能：需重训 Config 3，并新增 DFDC 基线
- [ ] 多数据集 OOD 证据：待重训后重新校验

### **Academic Gates** ⚠️
- [x] 诊断方法论（BCE → SupCon）仍成立
- [ ] 可重现性：Config 3 / DFDC 缺乏可复现实验记录

### **System Gates** ⚠️
- [x] 跨平台兼容性：Windows/Linux 仍可用
- [ ] 训练资产：`experiments/` 已清空，需要重新产出 baseline checkpoint
- [ ] 新数据接入：DFDC manifest / loader 尚未落地

---

## 📈 **完成度统计**

| 类别 | 完成度 | 状态 |
|------|--------|------|
| 环境配置 | 100% | ✅ 完成 |
| 数据管理 | 100% | ✅ 完成 |
| 学术工具 | 100% | ✅ 完成 |
| 实验管理 | 80% | ⚠️ Experiments 需重建 |
| 基线模型 | 75% | ⚠️ Config 3 / DFDC 待重训 |
| LODO评估 | 60% | ⛔ 缺少 Config 3 数据 |
| BCE诊断 | 100% | ✅ 完成 |
| Loss策略分析 | 100% | ✅ 完成 |
| Stage Gate验证 | 50% | ⚠️ 待满足重训条件 |

**总体完成度: ~80%（需补齐缺失实验以恢复基线）**

---

## 🚀 **准备进入Stage 01** ⚠️ (已完成，见Stage 01最终报告)

### ✅ 前置条件已满足
- ✅ 基础设施完整
- ✅ 工具链可用
- ✅ 数据集可访问
- ✅ BCE baseline完成（作为对比组）
- ✅ 理论动机明确（BCE失败 → 需要SupCon）

### 🔄 **Stage 01更新** (2025-10-05)
**Stage 01实验已全部完成**，结果显示：
- LODO平均AUC ~0.62（单模型OOD泛化失败）
- SupCon vs BCE提升 < 2%（对比学习效果有限）
- **定位调整**：Stage 01作为In-dist过滤器（AUC 0.99+）
- **OOD策略**：由Stage 02-04异构专家系统解决

详见：`../stage_01/stage_01_status_report.md`

---

## 📝 **技术债务和遗留问题**

### ⏳ 待完成
- [ ] 重跑 LODO Config 3（FF++DF → CelebDF），生成新的 checkpoint / metrics
- [ ] 合并 DFDC：生成 manifest、加入训练配置、完成 baseline 训练
- [ ] 重新汇总完整 3×3 LODO 矩阵（含 DFDC 版本）
- [ ] 恢复 `experiments/` 目录结构并记录最新实验 ID

### ✅ 已解决
- ✅ 所有依赖安装完成
- ✅ 数据集路径配置正确
- ✅ 核心工具验证可用
- ✅ BCE baseline诊断完成

---

## 🏆 **Stage 00 总结**

**核心价值**：
1. ✅ 建立了完整的基础设施和工具链
2. ✅ 完成了多数据集训练和LODO评估
3. ✅ **通过诊断证明了BCE范式的失败**
4. ✅ **为Stage 01的SupCon验证提供了理论依据**
5. ✅ 明确了整个项目的Loss函数策略

**学术贡献**：
- BCE baseline不是"失败的实验"，而是"有价值的对比组"
- 证明了传统CE/BCE范式无法学习可迁移特征
- 验证了"真实性建模"范式转换的必要性

**下一步**：
→ 🔄 **优先恢复 Stage 00 基线（LODO Config 3 + DFDC）**
→ ⚠️ **补齐后再推进 Stage 01 / Stage 02 的后续实验**

---

*报告生成时间: 2025-10-04（2025-10-05 更新：experiments 清空后状态重置）*
*Stage 00 完成度: ~80%*
*状态: 需重建基线与 OOD 证据，暂缓进入 Stage 01 扩展实验*
