# AWARE-NET Tools Directory

Tools目錄按功能分類組織，提供項目所需的各種實用工具。

## 目錄結構

```
tools/
├── data/                    # 數據處理工具
│   ├── generate_manifests.py    - 生成數據集清單文件
│   └── diagnose_path_leakage.py - 診斷數據集路徑洩漏問題
├── setup/                   # 環境設置工具
│   ├── environment_manager.py   - 智能PyTorch安裝器
│   └── setup_environment.py     - 全面環境驗證器
├── performance/             # 性能分析工具
│   └── profile_stage2_performance.py - Stage 2性能分析
├── validation/              # 驗證工具
│   ├── stage_gate_validator.py      - Stage Gate驗證器
│   └── verify_stage_0_completion.py - Stage 0完成驗證
└── tests/                   # 測試文件
    ├── test_baseline_model.py
    ├── test_dataset.py
    ├── test_metrics.py
    └── ...
```

## 工具使用指南

### 數據處理工具 (data/)

#### 1. 數據集清單生成器
```bash
python tools/data/generate_manifests.py celebdf
python tools/data/generate_manifests.py faceforensics
python tools/data/generate_manifests.py deeperforensics
```

#### 2. 路徑洩漏診斷器
```bash
python tools/data/diagnose_path_leakage.py
```

### 環境設置工具 (setup/)

#### 1. 智能PyTorch安裝器
```bash
# 檢查安裝計劃 (dry run)
python tools/setup/environment_manager.py --dry-run

# 安裝PyTorch
python tools/setup/environment_manager.py

# 僅驗證現有安裝
python tools/setup/environment_manager.py --verify-only

# 強制CPU版本
python tools/setup/environment_manager.py --force-cpu
```

#### 2. 全面環境驗證器
```bash
python tools/setup/setup_environment.py
```

### 性能分析工具 (performance/)

#### Stage 2性能分析
```bash
python tools/performance/profile_stage2_performance.py
```

### 驗證工具 (validation/)

#### 1. Stage Gate驗證器
```bash
python tools/validation/stage_gate_validator.py --stage 1
python tools/validation/stage_gate_validator.py --stage 2
```

#### 2. Stage 0完成驗證
```bash
python tools/validation/verify_stage_0_completion.py
```

## 模塊化使用

所有工具也可以作為Python模塊導入：

```python
# 數據工具
from tools.data import generate_manifests, diagnose_path_leakage

# 環境工具
from tools.setup import environment_manager, setup_environment

# 性能工具
from tools.performance import profile_stage2_performance

# 驗證工具
from tools.validation import stage_gate_validator, verify_stage_0_completion
```

## 開發與貢獻

- 每個新工具應放在合適的功能目錄下
- 更新對應的`__init__.py`文件以導出新功能
- 為工具添加適當的命令行接口
- 編寫測試文件到`tests/`目錄

## 測試

運行所有測試：
```bash
pytest tools/tests/
```

運行特定測試：
```bash
pytest tools/tests/test_dataset.py
pytest tools/tests/test_baseline_model.py
```