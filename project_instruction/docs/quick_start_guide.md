# AWARE-NET 快速開始指南

## 🚀 項目簡介

AWARE-NET是一個先進的深度偽造檢測框架，採用10階段實施策略，目前已完成Stage 00-02的全面實施。

## 📋 當前項目狀態

- ✅ **Stage 00**: 基礎設施與基線 (100% 完成)
- ✅ **Stage 01**: SupCon快速過濾器 (99% 完成)
- ✅ **Stage 02**: 異構專家模型 (100% 完成)
- 🚀 **Stage 03**: 時序建模專家 (準備中)

## 🛠️ 環境設置

### 系統要求

- Python 3.8+
- CUDA 11.8+ (GPU支持)
- 16GB+ RAM
- 50GB+ 可用硬碟空間

### 快速安裝

1. **克隆項目**
```bash
git clone <project-url>
cd MobileDeepfakeDetection
```

2. **自動環境設置**
```bash
# 使用主安裝腳本（推薦）
python setup.py

# 或使用環境驗證腳本
python scripts/setup_environment.py
```

3. **手動Conda環境設置**
```bash
# 創建環境
conda env create -f environment.yml
conda activate aware_net_rtx50

# 驗證安裝
python scripts/setup_environment.py --validate-only
```

## 🏗️ 項目架構

### 目錄結構
```
MobileDeepfakeDetection/
├── src/                     # 源代碼
│   ├── stage_00/           # 基礎設施與基線
│   ├── stage_01/           # SupCon快速過濾器
│   ├── stage_02/           # 異構專家模型
│   └── utils/              # 通用工具
├── configs/                # 配置文件
├── docs/                   # 文檔
├── scripts/                # 實用腳本
├── environment.yml         # Conda環境
└── setup.py               # 主安裝腳本
```

### 核心組件

#### Stage 01: SupCon快速過濾器
- `src/stage_01/supcon_loss.py` - SupCon損失函數
- `src/stage_01/mobilenetv4_model.py` - MobileNetV4模型
- `src/stage_01/balanced_sampler.py` - 對比學習批次構建

#### Stage 02: 異構專家模型
- `src/stage_02/enhanced_spatial_expert.py` - 空間專家模型
- `src/stage_02/enhanced_genconvit.py` - 生成專家模型
- `src/stage_02/complementarity_analysis.py` - 互補性分析
- `src/stage_02/concurrent_testing_framework.py` - 並發測試框架

## 🚀 快速使用

### 1. 基本推理示例

```python
# 導入核心組件
from src.stage_02.enhanced_spatial_expert import create_enhanced_spatial_expert
from src.stage_02.enhanced_genconvit import create_enhanced_genconvit
from src.stage_02.complementarity_analysis import create_fusion_system

# 創建專家模型
spatial_expert = create_enhanced_spatial_expert(
    input_resolution=256,
    num_classes=1,
    use_focal_loss=True
)

generative_expert = create_enhanced_genconvit(
    input_resolution=256,
    fusion_strategy="cross_attention",
    reconstruction_mode="patch_based"
)

# 創建融合系統
fusion_system = create_fusion_system(
    hidden_dim=256,
    num_experts=2,
    uncertainty_aware=True
)

# 執行推理
import torch
input_tensor = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    spatial_output = spatial_expert(input_tensor)
    generative_output = generative_expert(input_tensor)

    # 融合專家輸出
    fusion_result = fusion_system.fuse_experts([spatial_output, generative_output])

print(f"最終預測: {fusion_result['prediction']}")
print(f"互補性分數: {fusion_result.get('complementarity_score', 'N/A')}")
```

### 2. 系統健康監控

```python
from src.stage_02.diagnostic_tools import SystemHealthMonitor

monitor = SystemHealthMonitor()
health = monitor.get_current_health()

print(f"CPU使用率: {health.cpu_usage:.1f}%")
print(f"內存使用率: {health.memory_usage:.1f}%")
print(f"GPU使用率: {health.gpu_usage:.1f}%")
```

### 3. Stage-Gate評估

```python
from src.stage_02.diagnostic_tools import create_diagnostic_system

# 創建診斷系統
evaluator = create_diagnostic_system()

# 運行綜合評估
gate_report = evaluator.evaluate_stage_2(
    spatial_expert=spatial_expert,
    generative_expert=generative_expert,
    test_dataloader=test_loader,
    complementarity_result=complementarity_analysis
)

print(f"Gate狀態: {gate_report.gate_status}")
print(f"總體分數: {gate_report.overall_score:.3f}")
```

## 🧪 測試運行

### 運行所有測試
```bash
python src/stage_02/test_suite.py
```

### 運行特定測試套件
```python
from src.stage_02.test_suite import run_specific_test_suite

run_specific_test_suite('spatial')      # 測試空間專家
run_specific_test_suite('genconvit')    # 測試GenConViT
run_specific_test_suite('integration') # 測試整合
```

### 性能基準測試
```python
from src.stage_02.concurrent_testing_framework import run_concurrent_tests

# 創建測試數據
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 運行並發性能測試
experts = {
    'spatial': spatial_expert,
    'generative': generative_expert
}

test_results = await run_concurrent_tests(
    experts=experts,
    fusion_system=fusion_system,
    dataloader=test_loader
)

print("性能報告:")
print(f"平均推理時間: {test_results['report']['performance_metrics']['avg_inference_time']:.3f}s")
print(f"平均準確率: {test_results['report']['performance_metrics']['avg_accuracy']:.3f}")
```

## 🔧 配置管理

### GPU配置
系統自動檢測GPU架構並優化：
- RTX 30/40系列：標準PyTorch 2.6+ + CUDA 12.4
- RTX 50系列：PyTorch nightly + CUDA 12.6
- 無GPU：自動CPU訓練回退

### 多GPU支持
```python
import torch.nn as nn

# 自動檢測多GPU
if torch.cuda.device_count() > 1:
    spatial_expert = nn.DataParallel(spatial_expert)
    generative_expert = nn.DataParallel(generative_expert)
```

## 📚 進階使用

### Stage 3整合準備
```python
from src.stage_02.stage3_integration_interface import create_integration_hub

# 創建整合中心
hub = create_integration_hub(
    integration_level="hybrid",
    temporal_mode="frame_sequence",
    max_sequence_length=16
)

# 創建Stage 2包裝器
stage2_wrapper = hub.create_stage2_wrapper(
    spatial_expert=spatial_expert,
    generative_expert=generative_expert,
    fusion_system=fusion_system
)
```

### 互補性分析
```python
from src.stage_02.complementarity_analysis import ComplementarityAnalyzer

analyzer = ComplementarityAnalyzer(config)
result = analyzer.analyze_complementarity(spatial_output, generative_output)

print(f"互補性分數: {result.overall_complementarity:.3f}")
print(f"決策多樣性: {result.decision_diversity:.3f}")
print("建議:", result.recommendations)
```

## 🔍 故障排除

### 常見問題

1. **CUDA內存不足**
   - 減少batch_size
   - 啟用梯度檢查點
   - 使用混合精度訓練

2. **導入錯誤**
   - 確保Python路徑包含src目錄
   - 檢查所有依賴項已安裝

3. **性能問題**
   - 啟用性能監控：`SystemHealthMonitor`
   - 檢查GPU利用率
   - 調整worker數量

### 調試模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用診斷工具進行詳細分析
from src.stage_02.diagnostic_tools import ModelValidator

validator = ModelValidator()
metrics = validator.validate_model_performance(model, dataloader, device)
print("調試指標:", metrics)
```

## 📈 下一步

1. **Stage 03開發**: 使用提供的集成接口開始時序專家開發
2. **性能優化**: 根據診斷工具建議優化內存使用
3. **模型訓練**: 執行小規模驗證後進行全面訓練
4. **學術驗證**: 使用stage-gate評估框架驗證創新

## 🤝 貢獻指南

1. 遵循現有代碼結構和命名慣例
2. 為新功能添加全面測試
3. 更新API變更的文檔
4. 提交前運行完整測試套件

## 📄 許可證

此實施是AWARE-NET學術研究項目的一部分。使用條款請參考主項目許可證。

---

*最後更新: 2025-09-23*
*版本: v0.3.0 (Stage 00-02 完成)*