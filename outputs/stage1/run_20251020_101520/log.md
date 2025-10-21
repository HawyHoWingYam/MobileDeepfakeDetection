# Training Log

- **Experiment**: mobilenetv4_simple
- **Run ID**: 20251020_101520
- **Output Directory**: outputs/stage1/run_20251020_101520

---

2025-10-20 10:15:20,786 - INFO - utils.experiment_framework - Experiment initialized: outputs/stage1/run_20251020_101520
2025-10-20 10:15:20,786 - INFO - utils.experiment_framework - View results: tensorboard --logdir outputs/stage1
2025-10-20 10:15:20,908 - INFO - timm.models._builder - Loading pretrained weights from Hugging Face hub (timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k)
2025-10-20 10:15:21,140 - INFO - timm.models._hub - [timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
2025-10-20 10:15:21,179 - INFO - models.mobilenetv4_model - Loaded mobilenetv4_hybrid_medium with pretrained=True
2025-10-20 10:15:21,179 - INFO - models.mobilenetv4_model - === MOBILENETV4 INITIALIZATION VALIDATION ===
2025-10-20 10:15:21,179 - INFO - models.mobilenetv4_model - Expected feature dimension (backbone.num_features): 960
2025-10-20 10:15:21,179 - INFO - models.mobilenetv4_model - 🔍 Performing test forward pass to verify actual output dimensions...
2025-10-20 10:15:21,179 - INFO - models.mobilenetv4_model - ⚙️  Switching to eval mode for dimension validation (BatchNorm compatibility)
2025-10-20 10:15:21,182 - INFO - models.mobilenetv4_model -    ✅ Model mode: eval (training=False)
2025-10-20 10:15:21,184 - INFO - models.mobilenetv4_model -    📥 Dummy input shape: torch.Size([1, 3, 224, 224])
2025-10-20 10:15:21,184 - INFO - models.mobilenetv4_model -    🚀 Executing backbone forward pass...
2025-10-20 10:15:21,272 - INFO - models.mobilenetv4_model -    ✅ Test forward pass successful
2025-10-20 10:15:21,272 - INFO - models.mobilenetv4_model -    📤 Backbone output shape: torch.Size([1, 1280])
2025-10-20 10:15:21,272 - INFO - models.mobilenetv4_model -    📊 Actual feature dimension: 1280
2025-10-20 10:15:21,274 - INFO - models.mobilenetv4_model -    🔄 Restored training mode: True
2025-10-20 10:15:21,274 - INFO - models.mobilenetv4_model - 🎯 Dimension validation completed: Expected=960, Actual=1280
2025-10-20 10:15:21,274 - WARNING - models.mobilenetv4_model - ⚠️  DIMENSION MISMATCH DETECTED during initialization!
2025-10-20 10:15:21,274 - WARNING - models.mobilenetv4_model -    Expected: 960, Actual: 1280
2025-10-20 10:15:21,274 - WARNING - models.mobilenetv4_model -    Using actual dimension for classifier initialization
2025-10-20 10:15:21,274 - INFO - models.mobilenetv4_model - 🔥 Backbone unfrozen - full model will be trained
2025-10-20 10:15:21,274 - INFO - models.mobilenetv4_model - 🏗️  Creating classifier with validated dimensions...
2025-10-20 10:15:21,274 - INFO - models.mobilenetv4_model -    📐 Input dimension: 1280
2025-10-20 10:15:21,274 - INFO - models.mobilenetv4_model -    🎯 Output dimension: 1 (binary classification)
2025-10-20 10:15:21,274 - INFO - models.mobilenetv4_model -    🛡️  Dropout rate: 0.2
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -    ✅ Classifier layers created successfully
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -    🎲 Initializing classifier weights...
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -    ✅ Weight initialization completed
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model - 🔍 Classifier structure verification:
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -    📦 Total layers: 2
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -    Layer 0: Dropout
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -    Layer 1: Linear (1280 -> 1)
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -       Weight stats: mean=-0.0358, std=1.0135
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -       Weight range: [-3.1398, 3.1097]
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model -    Bias initialized: 0.0000
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model - ✅ Classifier creation and verification completed successfully!
2025-10-20 10:15:21,275 - INFO - models.mobilenetv4_model - 📋 Generating comprehensive model summary...
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model - 📊 COMPREHENSIVE MODEL SUMMARY:
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -    🏷️  Model Name: mobilenetv4_hybrid_medium
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -    📐 Feature Dimension: 1280
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -    🔧 Training Mode: True
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -    📈 PARAMETER COUNTS:
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Total Parameters: 9,794,929
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Trainable Parameters: 9,794,929
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Backbone Parameters: 9,793,648 (trainable: 9,793,648)
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Classifier Parameters: 1,281 (trainable: 1,281)
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -    💾 MEMORY ESTIMATES:
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Total Parameters Memory: 37.36 MB
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Classifier Memory: 0.00 MB
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -    📊 PARAMETER RATIOS:
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Classifier vs Total: 0.01%
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Trainable vs Total: 100.00%
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -    🎯 TRAINING COMPLEXITY:
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Primary Learning Components: Classifier (1,281 params)
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model -       Fine-tuning Components: Backbone (9,793,648 params)
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model - ✅ MODEL INITIALIZATION SUCCESSFULLY COMPLETED!
2025-10-20 10:15:21,279 - INFO - models.mobilenetv4_model - === INITIALIZATION VALIDATION FINISHED ===
2025-10-20 10:15:21,467 - INFO - train_mobilenet - === OPTIMIZER PARAMETER VALIDATION ===
2025-10-20 10:15:21,468 - INFO - train_mobilenet - 📊 Model Parameter Breakdown:
2025-10-20 10:15:21,468 - INFO - train_mobilenet -    Total parameters: 9,794,929
2025-10-20 10:15:21,469 - INFO - train_mobilenet -    Trainable parameters: 9,794,929
2025-10-20 10:15:21,469 - INFO - train_mobilenet -    Backbone trainable: 9,793,648
2025-10-20 10:15:21,469 - INFO - train_mobilenet -    Classifier trainable: 1,281
2025-10-20 10:15:21,469 - INFO - train_mobilenet - 🔍 ENHANCED OPTIMIZER PARAMETER ANALYSIS...
2025-10-20 10:15:21,469 - INFO - train_mobilenet -    📦 Optimizer group 0: 344 parameters
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Elements in group 0: 9,794,929
2025-10-20 10:15:21,469 - INFO - train_mobilenet -    📊 OPTIMIZER PARAMETER SUMMARY:
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Total parameter tensors: 344
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Total trainable elements: 9,794,929
2025-10-20 10:15:21,469 - INFO - train_mobilenet -    📈 PARAMETER SIZE DISTRIBUTION:
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Shape torch.Size([1, 1280]): 1 tensors, 1,280 elements
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Shape torch.Size([1024, 1, 5, 5]): 4 tensors, 102,400 elements
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Shape torch.Size([1024, 256, 1, 1]): 8 tensors, 2,097,152 elements
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Shape torch.Size([1024]): 24 tensors, 24,576 elements
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Shape torch.Size([128, 32, 3, 3]): 1 tensors, 36,864 elements
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Shape torch.Size([1280, 960, 1, 1]): 1 tensors, 1,228,800 elements
2025-10-20 10:15:21,469 - INFO - train_mobilenet -       Shape torch.Size([1280]): 2 tensors, 2,560 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([128]): 2 tensors, 256 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([160, 1, 3, 3]): 15 tensors, 21,600 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([160, 1, 5, 5]): 1 tensors, 4,000 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([160, 256, 1, 1]): 4 tensors, 163,840 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([160, 320, 1, 1]): 1 tensors, 51,200 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([160, 480, 1, 1]): 1 tensors, 76,800 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([160, 640, 1, 1]): 6 tensors, 614,400 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([160, 80, 1, 1]): 1 tensors, 12,800 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([160]): 70 tensors, 11,200 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([192, 1, 5, 5]): 1 tensors, 4,800 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([192, 48, 1, 1]): 1 tensors, 9,216 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([192]): 4 tensors, 768 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([1]): 1 tensors, 1 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([256, 1, 3, 3]): 4 tensors, 9,216 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([256, 1, 5, 5]): 4 tensors, 25,600 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([256, 1024, 1, 1]): 8 tensors, 2,097,152 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([256, 160, 1, 1]): 4 tensors, 163,840 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([256, 256, 1, 1]): 8 tensors, 524,288 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([256, 512, 1, 1]): 3 tensors, 393,216 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([256, 960, 1, 1]): 1 tensors, 245,760 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([256]): 64 tensors, 16,384 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([32, 3, 3, 3]): 1 tensors, 864 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([320, 160, 1, 1]): 1 tensors, 51,200 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([320]): 2 tensors, 640 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([32]): 2 tensors, 64 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([48, 1, 3, 3]): 1 tensors, 432 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([48, 128, 1, 1]): 1 tensors, 6,144 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([480, 1, 5, 5]): 1 tensors, 12,000 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([480, 80, 1, 1]): 1 tensors, 38,400 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([480]): 4 tensors, 1,920 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([48]): 4 tensors, 192 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([512, 1, 5, 5]): 1 tensors, 12,800 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([512, 256, 1, 1]): 3 tensors, 393,216 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([512]): 8 tensors, 4,096 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([64, 160, 1, 1]): 8 tensors, 81,920 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([64, 256, 1, 1]): 8 tensors, 131,072 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([640, 1, 3, 3]): 3 tensors, 17,280 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([640, 1, 5, 5]): 1 tensors, 16,000 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([640, 160, 1, 1]): 6 tensors, 614,400 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([640]): 20 tensors, 12,800 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([80, 1, 3, 3]): 2 tensors, 1,440 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([80, 160, 1, 1]): 1 tensors, 12,800 elements
2025-10-20 10:15:21,470 - INFO - train_mobilenet -       Shape torch.Size([80, 192, 1, 1]): 1 tensors, 15,360 elements
2025-10-20 10:15:21,471 - INFO - train_mobilenet -       Shape torch.Size([80]): 10 tensors, 800 elements
2025-10-20 10:15:21,471 - INFO - train_mobilenet -       Shape torch.Size([960, 1, 5, 5]): 1 tensors, 24,000 elements
2025-10-20 10:15:21,471 - INFO - train_mobilenet -       Shape torch.Size([960, 160, 1, 1]): 1 tensors, 153,600 elements
2025-10-20 10:15:21,471 - INFO - train_mobilenet -       Shape torch.Size([960, 256, 1, 1]): 1 tensors, 245,760 elements
2025-10-20 10:15:21,471 - INFO - train_mobilenet -       Shape torch.Size([960]): 6 tensors, 5,760 elements
2025-10-20 10:15:21,471 - INFO - train_mobilenet -    🔍 MODEL VS OPTIMIZER COMPARISON:
2025-10-20 10:15:21,471 - INFO - train_mobilenet -       Model: 344 tensors, 9,794,929 elements
2025-10-20 10:15:21,471 - INFO - train_mobilenet -       Optimizer: 344 tensors, 9,794,929 elements
2025-10-20 10:15:21,471 - INFO - train_mobilenet - ✅ PERFECT MATCH: Model and optimizer parameters are identical!
2025-10-20 10:15:21,472 - INFO - train_mobilenet -    Total optimizer parameters (corrected): 344
2025-10-20 10:15:21,472 - INFO - train_mobilenet - 🔍 ENHANCED CLASSIFIER PARAMETER DETECTION...
2025-10-20 10:15:21,472 - INFO - train_mobilenet -    🏷️  Model classifier param: classifier.1.weight | Shape: torch.Size([1, 1280]) | Count: 1280
2025-10-20 10:15:21,472 - INFO - train_mobilenet -    🏷️  Model classifier param: classifier.1.bias | Shape: torch.Size([1]) | Count: 1
2025-10-20 10:15:21,472 - INFO - train_mobilenet -    📊 Total classifier params in model: 2
2025-10-20 10:15:21,473 - INFO - train_mobilenet -    ✅ Direct match found: classifier.1.weight
2025-10-20 10:15:21,473 - INFO - train_mobilenet -    ✅ Direct match found: classifier.1.bias
2025-10-20 10:15:21,473 - INFO - train_mobilenet -    🔢 Detection results:
2025-10-20 10:15:21,473 - INFO - train_mobilenet -       Model classifier params: 2
2025-10-20 10:15:21,473 - INFO - train_mobilenet -       Shape-matched in optimizer: 2
2025-10-20 10:15:21,473 - INFO - train_mobilenet -       Direct object matches: 2
2025-10-20 10:15:21,473 - INFO - train_mobilenet -    📋 Detailed parameter verification:
2025-10-20 10:15:21,473 - INFO - train_mobilenet -       ✅ FOUND | classifier.1.weight | torch.Size([1, 1280])
2025-10-20 10:15:21,473 - INFO - train_mobilenet -       ✅ FOUND | classifier.1.bias | torch.Size([1])
2025-10-20 10:15:21,473 - INFO - train_mobilenet - ✅ SUCCESS: Classifier parameters properly tracked by optimizer!
2025-10-20 10:15:21,473 - INFO - train_mobilenet -    📈 Direct object matches: 2/2
2025-10-20 10:15:21,473 - INFO - train_mobilenet -    🎯 Training will correctly update classifier weights!
2025-10-20 10:15:21,473 - INFO - train_mobilenet - ✅ Optimizer tracks all model parameters correctly
2025-10-20 10:15:21,474 - INFO - train_mobilenet -    📊 Verified: 344 tensors in both model and optimizer
2025-10-20 10:15:21,474 - INFO - train_mobilenet - === OPTIMIZER VALIDATION COMPLETE ===
2025-10-20 10:15:21,474 - INFO - train_mobilenet - 🔧 Enhanced monitoring enabled:
2025-10-20 10:15:21,474 - INFO - train_mobilenet -    ✅ Gradient flow tracking
2025-10-20 10:15:21,474 - INFO - train_mobilenet -    ✅ Learning progress monitoring
2025-10-20 10:15:21,474 - INFO - train_mobilenet -    ✅ Parameter update verification
2025-10-20 10:15:21,474 - INFO - train_mobilenet - 🔧 Early stopping enabled:
2025-10-20 10:15:21,474 - INFO - train_mobilenet -    ⏱️  Patience: 5 epochs
2025-10-20 10:15:21,474 - INFO - train_mobilenet -    📊 Min Delta: 0.001
2025-10-20 10:15:21,474 - INFO - train_mobilenet -    🔄 Restore best weights: True
2025-10-20 10:15:21,474 - INFO - train_mobilenet - Setting up multi-dataset data loaders with 4 datasets...
2025-10-20 10:15:21,474 - INFO - train_mobilenet - Stage 01: Loading 4 datasets with equal weight balancing...
2025-10-20 10:15:21,474 - INFO - train_mobilenet - Loading celebdf_v2:
2025-10-20 10:15:21,474 - INFO - train_mobilenet -   Train: manifests/celebdf_v2_train_balanced.csv
2025-10-20 10:15:21,474 - INFO - train_mobilenet -   Val: manifests/celebdf_v2_val_balanced.csv
2025-10-20 10:15:21,474 - INFO - train_mobilenet -   Test: manifests/celebdf_v2_test_balanced.csv
2025-10-20 10:15:21,568 - INFO - training.dataset - Loaded manifest manifests/celebdf_v2_train_balanced.csv: total=83599 real=41674 fake=41925
2025-10-20 10:15:21,589 - INFO - training.dataset - Loaded manifest manifests/celebdf_v2_val_balanced.csv: total=17478 real=8765 fake=8713
2025-10-20 10:15:21,614 - INFO - training.dataset - Loaded manifest manifests/celebdf_v2_test_balanced.csv: total=18037 real=9118 fake=8919
2025-10-20 10:15:21,615 - INFO - train_mobilenet -   ✓ celebdf_v2: Train=83,599, Val=17,478, Test=18,037
2025-10-20 10:15:21,615 - INFO - train_mobilenet - Loading faceforensics:
2025-10-20 10:15:21,615 - INFO - train_mobilenet -   Train: manifests/faceforensics_train_balanced.csv
2025-10-20 10:15:21,615 - INFO - train_mobilenet -   Val: manifests/faceforensics_val_balanced.csv
2025-10-20 10:15:21,615 - INFO - train_mobilenet -   Test: manifests/faceforensics_test_balanced.csv
2025-10-20 10:15:21,894 - INFO - training.dataset - Loaded manifest manifests/faceforensics_train_balanced.csv: total=223653 real=111161 fake=112492
2025-10-20 10:15:21,967 - INFO - training.dataset - Loaded manifest manifests/faceforensics_val_balanced.csv: total=50945 real=26351 fake=24594
2025-10-20 10:15:22,028 - INFO - training.dataset - Loaded manifest manifests/faceforensics_test_balanced.csv: total=47430 real=23502 fake=23928
2025-10-20 10:15:22,029 - INFO - train_mobilenet -   ✓ faceforensics: Train=223,653, Val=50,945, Test=47,430
2025-10-20 10:15:22,029 - INFO - train_mobilenet - Loading deeperforensics:
2025-10-20 10:15:22,029 - INFO - train_mobilenet -   Train: manifests/deeperforensics_train_balanced.csv
2025-10-20 10:15:22,029 - INFO - train_mobilenet -   Val: manifests/deeperforensics_val_balanced.csv
2025-10-20 10:15:22,029 - INFO - train_mobilenet -   Test: manifests/deeperforensics_test_balanced.csv
2025-10-20 10:15:23,208 - INFO - training.dataset - Loaded manifest manifests/deeperforensics_train_balanced.csv: total=844396 real=414512 fake=429884
2025-10-20 10:15:23,454 - INFO - training.dataset - Loaded manifest manifests/deeperforensics_val_balanced.csv: total=165854 real=87807 fake=78047
2025-10-20 10:15:23,689 - INFO - training.dataset - Loaded manifest manifests/deeperforensics_test_balanced.csv: total=172894 real=89253 fake=83641
2025-10-20 10:15:23,689 - INFO - train_mobilenet -   ✓ deeperforensics: Train=844,396, Val=165,854, Test=172,894
2025-10-20 10:15:23,690 - INFO - train_mobilenet - Loading dfdc:
2025-10-20 10:15:23,690 - INFO - train_mobilenet -   Train: manifests/dfdc_train_balanced.csv
2025-10-20 10:15:23,690 - INFO - train_mobilenet -   Val: manifests/dfdc_val_balanced.csv
2025-10-20 10:15:23,690 - INFO - train_mobilenet -   Test: manifests/dfdc_test_balanced.csv
2025-10-20 10:15:24,472 - INFO - training.dataset - Loaded manifest manifests/dfdc_train_balanced.csv: total=721946 real=361229 fake=360717
2025-10-20 10:15:24,653 - INFO - training.dataset - Loaded manifest manifests/dfdc_val_balanced.csv: total=154919 real=77350 fake=77569
2025-10-20 10:15:24,831 - INFO - training.dataset - Loaded manifest manifests/dfdc_test_balanced.csv: total=154511 real=77109 fake=77402
2025-10-20 10:15:24,831 - INFO - train_mobilenet -   ✓ dfdc: Train=721,946, Val=154,919, Test=154,511
2025-10-20 10:15:24,832 - INFO - train_mobilenet - Successfully loaded 4 datasets for multi-dataset training
2025-10-20 10:15:24,832 - INFO - train_mobilenet - Dataset celebdf_v2: weight=0.250, samples=83,599, per_sample_weight=0.000003
2025-10-20 10:15:24,832 - INFO - train_mobilenet - Dataset faceforensics: weight=0.250, samples=223,653, per_sample_weight=0.000001
2025-10-20 10:15:24,832 - INFO - train_mobilenet - Dataset deeperforensics: weight=0.250, samples=844,396, per_sample_weight=0.000000
2025-10-20 10:15:24,835 - INFO - train_mobilenet - Dataset dfdc: weight=0.250, samples=721,946, per_sample_weight=0.000000
2025-10-20 10:15:24,909 - INFO - train_mobilenet - Combined dataset sizes:
2025-10-20 10:15:24,910 - INFO - train_mobilenet -   Train: 1,873,594 samples from 4 datasets
2025-10-20 10:15:24,910 - INFO - train_mobilenet -   Val: 389,196 samples
2025-10-20 10:15:24,910 - INFO - train_mobilenet -   Test: 392,872 samples
2025-10-20 10:15:24,910 - INFO - train_mobilenet - Equal weight balancing: Each dataset contributes 25.0% of training batches
2025-10-20 10:15:24,911 - INFO - train_mobilenet - Data loaders ready - Train: 7318 batches
2025-10-20 10:15:24,911 - INFO - train_mobilenet - === DEBUG: 4-Dataset Training Analysis ===
2025-10-20 10:15:28,128 - INFO - train_mobilenet - Batch 0: 256 samples - Real: 47.3%, Fake: 52.7%
2025-10-20 10:15:29,332 - INFO - train_mobilenet - First 256 samples - Real: 47.3%, Fake: 52.7%
2025-10-20 10:15:29,333 - INFO - train_mobilenet - Expected balance: 50% Real, 50% Fake (within margin)
2025-10-20 10:15:29,333 - INFO - train_mobilenet - === END DEBUG ANALYSIS ===
2025-10-20 10:15:29,333 - INFO - train_mobilenet - 🚀 Starting MobileNetV4 training...
2025-10-20 10:15:29,333 - INFO - train_mobilenet - 📋 Training Configuration:
2025-10-20 10:15:29,333 - INFO - train_mobilenet -    Model: mobilenetv4_hybrid_medium
2025-10-20 10:15:29,333 - INFO - train_mobilenet -    Epochs: 10
2025-10-20 10:15:29,333 - INFO - train_mobilenet -    Device: cuda
2025-10-20 10:15:29,333 - INFO - train_mobilenet -    Learning Rate: 0.0001
2025-10-20 10:15:29,333 - INFO - train_mobilenet -    Batch Size: 256
2025-10-20 10:15:29,333 - INFO - train_mobilenet -    Training samples: 1873594
2025-10-20 10:15:29,333 - INFO - train_mobilenet -    Validation samples: 389196
2025-10-20 10:15:29,337 - INFO - train_mobilenet -    Total Parameters: 9,794,929
2025-10-20 10:15:29,337 - INFO - train_mobilenet -    Trainable Parameters: 9,794,929
2025-10-20 10:15:29,337 - INFO - train_mobilenet - 🎯 Training Objectives:
2025-10-20 10:15:29,337 - INFO - train_mobilenet -    Stage 01: Simple MobileNetV4 + BCE classification
2025-10-20 10:15:29,337 - INFO - train_mobilenet -    Stage 00: Integrated experiment framework with TensorBoard
2025-10-20 10:15:29,337 - INFO - train_mobilenet -    📊 Enhanced monitoring and validation enabled
2025-10-20 10:15:29,337 - INFO - train_mobilenet - ============================================================
2025-10-20 10:15:29,338 - INFO - train_mobilenet - 🔍 Pre-training model validation...
2025-10-20 10:15:29,338 - INFO - train_mobilenet - ⚙️  Switching to eval mode for validation (BatchNorm compatibility)
2025-10-20 10:15:29,341 - INFO - train_mobilenet -    ✅ Model mode: eval (was True)
2025-10-20 10:15:29,343 - INFO - train_mobilenet -    📥 Validation input shape: torch.Size([1, 3, 224, 224])
2025-10-20 10:15:29,343 - INFO - train_mobilenet -    🚀 Executing validation forward pass...
2025-10-20 10:15:29,735 - INFO - models.mobilenetv4_model - 🚀 First forward pass - verifying dimensions...
2025-10-20 10:15:29,735 - INFO - models.mobilenetv4_model -    Input shape: torch.Size([1, 3, 224, 224])
2025-10-20 10:15:29,735 - INFO - models.mobilenetv4_model -    Features shape: torch.Size([1, 1280])
2025-10-20 10:15:29,735 - INFO - models.mobilenetv4_model -    Expected feature_dim: 1280
2025-10-20 10:15:29,735 - INFO - models.mobilenetv4_model -    Classifier input dim: 1280
2025-10-20 10:15:29,735 - INFO - models.mobilenetv4_model - ✅ Dimensions verified - proceeding with classification
2025-10-20 10:15:29,735 - INFO - models.mobilenetv4_model - 📈 Forward pass #1 - dimensions stable
2025-10-20 10:15:29,736 - INFO - train_mobilenet -    ✅ Model forward pass successful
2025-10-20 10:15:29,736 - INFO - train_mobilenet -    📤 Output shape: torch.Size([1])
2025-10-20 10:15:29,759 - INFO - train_mobilenet -    📊 Output range: [-62.6414, -62.6414]
2025-10-20 10:15:29,759 - INFO - train_mobilenet -    ✅ Binary classification output confirmed: -62.641396
2025-10-20 10:15:29,761 - INFO - train_mobilenet -    🔄 Restored training mode: True
2025-10-20 10:15:29,761 - INFO - train_mobilenet - 🎯 Pre-training validation completed successfully!
2025-10-20 10:18:07,937 - INFO - models.mobilenetv4_model - 📈 Forward pass #1001 - dimensions stable
2025-10-20 10:20:41,904 - INFO - models.mobilenetv4_model - 📈 Forward pass #2001 - dimensions stable
2025-10-20 10:23:15,860 - INFO - models.mobilenetv4_model - 📈 Forward pass #3001 - dimensions stable
2025-10-20 10:25:49,784 - INFO - models.mobilenetv4_model - 📈 Forward pass #4001 - dimensions stable
2025-10-20 10:28:23,800 - INFO - models.mobilenetv4_model - 📈 Forward pass #5001 - dimensions stable
2025-10-20 10:30:57,730 - INFO - models.mobilenetv4_model - 📈 Forward pass #6001 - dimensions stable
2025-10-20 10:33:31,653 - INFO - models.mobilenetv4_model - 📈 Forward pass #7001 - dimensions stable
2025-10-20 10:34:21,151 - INFO - train_mobilenet - 📈 Training Epoch 1 Results:
2025-10-20 10:34:21,151 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 10:34:21,152 - INFO - train_mobilenet -    Average batch loss: 2.061303
2025-10-20 10:34:21,152 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 10:34:21,152 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 10:34:21,153 - INFO - train_mobilenet -    Target distribution: Real=0.504, Fake=0.496
2025-10-20 10:34:21,177 - INFO - train_mobilenet - 📊 Gradient norms (Epoch 1):
2025-10-20 10:34:21,177 - INFO - train_mobilenet -    Classifier: 0.154403
2025-10-20 10:34:21,177 - INFO - train_mobilenet -    Backbone: 189.778609
2025-10-20 10:34:21,177 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 10:34:21,179 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 10:35:23,347 - INFO - models.mobilenetv4_model - 📈 Forward pass #8001 - dimensions stable
2025-10-20 10:36:53,313 - INFO - models.mobilenetv4_model - 📈 Forward pass #9001 - dimensions stable
2025-10-20 10:38:23,361 - INFO - models.mobilenetv4_model - 📈 Forward pass #10001 - dimensions stable
2025-10-20 10:39:50,485 - INFO - models.mobilenetv4_model - 📈 Forward pass #11001 - dimensions stable
2025-10-20 10:41:17,598 - INFO - models.mobilenetv4_model - 📈 Forward pass #12001 - dimensions stable
2025-10-20 10:42:44,395 - INFO - models.mobilenetv4_model - 📈 Forward pass #13001 - dimensions stable
2025-10-20 10:44:11,582 - INFO - models.mobilenetv4_model - 📈 Forward pass #14001 - dimensions stable
2025-10-20 10:45:09,685 - INFO - utils.evaluation - Train Loss: 0.3884
2025-10-20 10:45:09,685 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 10:45:09,685 - INFO - utils.evaluation - AUC: 0.9101
2025-10-20 10:45:09,685 - INFO - utils.evaluation - F1-Score: 0.8239
2025-10-20 10:45:09,685 - INFO - utils.evaluation - Accuracy: 0.8133
2025-10-20 10:45:09,685 - INFO - utils.evaluation - Precision: 0.7838
2025-10-20 10:45:09,685 - INFO - utils.evaluation - Recall: 0.8683
2025-10-20 10:45:09,685 - INFO - utils.evaluation - Specificity: 0.7576
2025-10-20 10:45:09,685 - INFO - utils.evaluation - False Negative Rate: 0.1317
2025-10-20 10:45:09,685 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 10:45:09,685 - INFO - train_mobilenet -    AUC: 0.9101
2025-10-20 10:45:09,685 - INFO - train_mobilenet -    Loss: 0.3884
2025-10-20 10:45:09,685 - INFO - train_mobilenet -    Accuracy: 0.8133
2025-10-20 10:45:09,685 - INFO - train_mobilenet -    F1-Score: 0.8239
2025-10-20 10:45:43,289 - INFO - models.mobilenetv4_model - 📈 Forward pass #15001 - dimensions stable
2025-10-20 10:47:12,353 - INFO - models.mobilenetv4_model - 📈 Forward pass #16001 - dimensions stable
2025-10-20 10:47:26,595 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 10:48:40,374 - INFO - models.mobilenetv4_model - 📈 Forward pass #17001 - dimensions stable
2025-10-20 10:49:39,792 - INFO - utils.evaluation - Validation Loss: 0.3625
2025-10-20 10:49:39,792 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 10:49:39,792 - INFO - utils.evaluation - AUC: 0.9174
2025-10-20 10:49:39,792 - INFO - utils.evaluation - F1-Score: 0.8207
2025-10-20 10:49:39,792 - INFO - utils.evaluation - Accuracy: 0.8191
2025-10-20 10:49:39,792 - INFO - utils.evaluation - Precision: 0.7909
2025-10-20 10:49:39,792 - INFO - utils.evaluation - Recall: 0.8528
2025-10-20 10:49:39,792 - INFO - utils.evaluation - Specificity: 0.7873
2025-10-20 10:49:39,792 - INFO - utils.evaluation - False Negative Rate: 0.1472
2025-10-20 10:49:39,805 - INFO - utils.experiment_framework - Epoch 1 [TRAIN] auc: 0.9101 | f1: 0.8239 | accuracy: 0.8133 | precision: 0.7838 | recall: 0.8683 | specificity: 0.7576 | fnr: 0.1317 | true_positives: 818254.0000 | false_positives: 225669.0000 | true_negatives: 705409.0000 | false_negatives: 124076.0000 | loss: 0.3884
2025-10-20 10:49:39,806 - INFO - utils.experiment_framework - Epoch 1 [VALIDATION] auc: 0.9174 | f1: 0.8207 | accuracy: 0.8191 | precision: 0.7909 | recall: 0.8528 | specificity: 0.7873 | fnr: 0.1472 | true_positives: 161114.0000 | false_positives: 42608.0000 | true_negatives: 157665.0000 | false_negatives: 27809.0000 | loss: 0.3625
2025-10-20 10:49:39,828 - INFO - train_mobilenet - ✅ New best validation AUC: 0.9174 (+0.9174)
2025-10-20 10:49:39,829 - INFO - train_mobilenet - 📍 Best epoch: 1, Patience counter reset
2025-10-20 10:49:39,995 - INFO - utils.experiment_framework - ✓ New best model saved! AUC: 0.9174
2025-10-20 10:49:39,995 - INFO - train_mobilenet - Epoch 1/10 - Train Loss: 0.3884, Val Loss: 0.3625, Val AUC: 0.9174, Best AUC: 0.9174
2025-10-20 10:50:31,979 - INFO - models.mobilenetv4_model - 📈 Forward pass #18001 - dimensions stable
2025-10-20 10:53:05,971 - INFO - models.mobilenetv4_model - 📈 Forward pass #19001 - dimensions stable
2025-10-20 10:55:39,934 - INFO - models.mobilenetv4_model - 📈 Forward pass #20001 - dimensions stable
2025-10-20 10:58:13,889 - INFO - models.mobilenetv4_model - 📈 Forward pass #21001 - dimensions stable
2025-10-20 11:00:47,928 - INFO - models.mobilenetv4_model - 📈 Forward pass #22001 - dimensions stable
2025-10-20 11:03:21,856 - INFO - models.mobilenetv4_model - 📈 Forward pass #23001 - dimensions stable
2025-10-20 11:05:55,775 - INFO - models.mobilenetv4_model - 📈 Forward pass #24001 - dimensions stable
2025-10-20 11:08:29,786 - INFO - train_mobilenet - 📈 Training Epoch 2 Results:
2025-10-20 11:08:29,786 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 11:08:29,786 - INFO - train_mobilenet -    Average batch loss: 0.459455
2025-10-20 11:08:29,786 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 11:08:29,787 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 11:08:29,788 - INFO - train_mobilenet -    Target distribution: Real=0.503, Fake=0.497
2025-10-20 11:08:29,788 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 11:08:29,790 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 11:08:32,512 - INFO - models.mobilenetv4_model - 📈 Forward pass #25001 - dimensions stable
2025-10-20 11:09:58,938 - INFO - models.mobilenetv4_model - 📈 Forward pass #26001 - dimensions stable
2025-10-20 11:11:26,208 - INFO - models.mobilenetv4_model - 📈 Forward pass #27001 - dimensions stable
2025-10-20 11:12:53,759 - INFO - models.mobilenetv4_model - 📈 Forward pass #28001 - dimensions stable
2025-10-20 11:14:20,460 - INFO - models.mobilenetv4_model - 📈 Forward pass #29001 - dimensions stable
2025-10-20 11:15:47,840 - INFO - models.mobilenetv4_model - 📈 Forward pass #30001 - dimensions stable
2025-10-20 11:17:15,459 - INFO - models.mobilenetv4_model - 📈 Forward pass #31001 - dimensions stable
2025-10-20 11:18:43,039 - INFO - models.mobilenetv4_model - 📈 Forward pass #32001 - dimensions stable
2025-10-20 11:19:12,454 - INFO - utils.evaluation - Train Loss: 0.5334
2025-10-20 11:19:12,454 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 11:19:12,454 - INFO - utils.evaluation - AUC: 0.8001
2025-10-20 11:19:12,454 - INFO - utils.evaluation - F1-Score: 0.6829
2025-10-20 11:19:12,454 - INFO - utils.evaluation - Accuracy: 0.7015
2025-10-20 11:19:12,454 - INFO - utils.evaluation - Precision: 0.7337
2025-10-20 11:19:12,454 - INFO - utils.evaluation - Recall: 0.6387
2025-10-20 11:19:12,454 - INFO - utils.evaluation - Specificity: 0.7651
2025-10-20 11:19:12,454 - INFO - utils.evaluation - False Negative Rate: 0.3613
2025-10-20 11:19:12,454 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 11:19:12,454 - INFO - train_mobilenet -    AUC: 0.8001
2025-10-20 11:19:12,454 - INFO - train_mobilenet -    Loss: 0.5334
2025-10-20 11:19:12,454 - INFO - train_mobilenet -    Accuracy: 0.7015
2025-10-20 11:19:12,454 - INFO - train_mobilenet -    F1-Score: 0.6829
2025-10-20 11:19:12,454 - WARNING - train_mobilenet - 📉 Learning degradation: AUC decreased by -0.1101
2025-10-20 11:20:13,245 - INFO - models.mobilenetv4_model - 📈 Forward pass #33001 - dimensions stable
2025-10-20 11:21:25,956 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 11:21:41,489 - INFO - models.mobilenetv4_model - 📈 Forward pass #34001 - dimensions stable
2025-10-20 11:23:08,144 - INFO - models.mobilenetv4_model - 📈 Forward pass #35001 - dimensions stable
2025-10-20 11:23:39,754 - INFO - utils.evaluation - Validation Loss: 0.4692
2025-10-20 11:23:39,754 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 11:23:39,754 - INFO - utils.evaluation - AUC: 0.8456
2025-10-20 11:23:39,754 - INFO - utils.evaluation - F1-Score: 0.7067
2025-10-20 11:23:39,754 - INFO - utils.evaluation - Accuracy: 0.7338
2025-10-20 11:23:39,754 - INFO - utils.evaluation - Precision: 0.7596
2025-10-20 11:23:39,754 - INFO - utils.evaluation - Recall: 0.6607
2025-10-20 11:23:39,754 - INFO - utils.evaluation - Specificity: 0.8028
2025-10-20 11:23:39,754 - INFO - utils.evaluation - False Negative Rate: 0.3393
2025-10-20 11:23:39,768 - INFO - utils.experiment_framework - Epoch 2 [TRAIN] auc: 0.8001 | f1: 0.6829 | accuracy: 0.7015 | precision: 0.7337 | recall: 0.6387 | specificity: 0.7651 | fnr: 0.3613 | true_positives: 602252.0000 | false_positives: 218549.0000 | true_negatives: 711883.0000 | false_negatives: 340724.0000 | loss: 0.5334
2025-10-20 11:23:39,769 - INFO - utils.experiment_framework - Epoch 2 [VALIDATION] auc: 0.8456 | f1: 0.7067 | accuracy: 0.7338 | precision: 0.7596 | recall: 0.6607 | specificity: 0.8028 | fnr: 0.3393 | true_positives: 124815.0000 | false_positives: 39491.0000 | true_negatives: 160782.0000 | false_negatives: 64108.0000 | loss: 0.4692
2025-10-20 11:23:39,769 - INFO - train_mobilenet - ⚠️  No improvement: Patience 1/5
2025-10-20 11:23:39,769 - INFO - train_mobilenet - Epoch 2/10 - Train Loss: 0.5334, Val Loss: 0.4692, Val AUC: 0.8456, Best AUC: 0.9174
2025-10-20 11:25:21,234 - INFO - models.mobilenetv4_model - 📈 Forward pass #36001 - dimensions stable
2025-10-20 11:27:55,238 - INFO - models.mobilenetv4_model - 📈 Forward pass #37001 - dimensions stable
2025-10-20 11:30:29,202 - INFO - models.mobilenetv4_model - 📈 Forward pass #38001 - dimensions stable
2025-10-20 11:33:03,316 - INFO - models.mobilenetv4_model - 📈 Forward pass #39001 - dimensions stable
2025-10-20 11:35:37,298 - INFO - models.mobilenetv4_model - 📈 Forward pass #40001 - dimensions stable
2025-10-20 11:38:11,263 - INFO - models.mobilenetv4_model - 📈 Forward pass #41001 - dimensions stable
2025-10-20 11:40:45,231 - INFO - models.mobilenetv4_model - 📈 Forward pass #42001 - dimensions stable
2025-10-20 11:42:29,571 - INFO - train_mobilenet - 📈 Training Epoch 3 Results:
2025-10-20 11:42:29,571 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 11:42:29,572 - INFO - train_mobilenet -    Average batch loss: 0.480656
2025-10-20 11:42:29,572 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 11:42:29,572 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 11:42:29,573 - INFO - train_mobilenet -    Target distribution: Real=0.503, Fake=0.497
2025-10-20 11:42:29,573 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 11:42:29,575 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 11:42:59,527 - INFO - models.mobilenetv4_model - 📈 Forward pass #43001 - dimensions stable
2025-10-20 11:44:26,744 - INFO - models.mobilenetv4_model - 📈 Forward pass #44001 - dimensions stable
2025-10-20 11:45:54,528 - INFO - models.mobilenetv4_model - 📈 Forward pass #45001 - dimensions stable
2025-10-20 11:47:22,126 - INFO - models.mobilenetv4_model - 📈 Forward pass #46001 - dimensions stable
2025-10-20 11:48:49,722 - INFO - models.mobilenetv4_model - 📈 Forward pass #47001 - dimensions stable
2025-10-20 11:50:17,341 - INFO - models.mobilenetv4_model - 📈 Forward pass #48001 - dimensions stable
2025-10-20 11:51:44,646 - INFO - models.mobilenetv4_model - 📈 Forward pass #49001 - dimensions stable
2025-10-20 11:53:13,104 - INFO - utils.evaluation - Train Loss: 0.3350
2025-10-20 11:53:13,104 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 11:53:13,104 - INFO - utils.evaluation - AUC: 0.9295
2025-10-20 11:53:13,104 - INFO - utils.evaluation - F1-Score: 0.8430
2025-10-20 11:53:13,104 - INFO - utils.evaluation - Accuracy: 0.8401
2025-10-20 11:53:13,105 - INFO - utils.evaluation - Precision: 0.8326
2025-10-20 11:53:13,105 - INFO - utils.evaluation - Recall: 0.8537
2025-10-20 11:53:13,105 - INFO - utils.evaluation - Specificity: 0.8263
2025-10-20 11:53:13,105 - INFO - utils.evaluation - False Negative Rate: 0.1463
2025-10-20 11:53:13,105 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 11:53:13,105 - INFO - train_mobilenet -    AUC: 0.9295
2025-10-20 11:53:13,105 - INFO - train_mobilenet -    Loss: 0.3350
2025-10-20 11:53:13,105 - INFO - train_mobilenet -    Accuracy: 0.8401
2025-10-20 11:53:13,105 - INFO - train_mobilenet -    F1-Score: 0.8430
2025-10-20 11:53:13,105 - INFO - train_mobilenet - 📈 Learning progress: AUC improved by +0.1294
2025-10-20 11:53:15,543 - INFO - models.mobilenetv4_model - 📈 Forward pass #50001 - dimensions stable
2025-10-20 11:54:41,937 - INFO - models.mobilenetv4_model - 📈 Forward pass #51001 - dimensions stable
2025-10-20 11:55:26,793 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 11:56:10,368 - INFO - models.mobilenetv4_model - 📈 Forward pass #52001 - dimensions stable
2025-10-20 11:57:37,128 - INFO - models.mobilenetv4_model - 📈 Forward pass #53001 - dimensions stable
2025-10-20 11:57:40,488 - INFO - utils.evaluation - Validation Loss: 0.3517
2025-10-20 11:57:40,489 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 11:57:40,489 - INFO - utils.evaluation - AUC: 0.9204
2025-10-20 11:57:40,489 - INFO - utils.evaluation - F1-Score: 0.8214
2025-10-20 11:57:40,489 - INFO - utils.evaluation - Accuracy: 0.8244
2025-10-20 11:57:40,489 - INFO - utils.evaluation - Precision: 0.8114
2025-10-20 11:57:40,489 - INFO - utils.evaluation - Recall: 0.8316
2025-10-20 11:57:40,489 - INFO - utils.evaluation - Specificity: 0.8177
2025-10-20 11:57:40,489 - INFO - utils.evaluation - False Negative Rate: 0.1684
2025-10-20 11:57:40,503 - INFO - utils.experiment_framework - Epoch 3 [TRAIN] auc: 0.9295 | f1: 0.8430 | accuracy: 0.8401 | precision: 0.8326 | recall: 0.8537 | specificity: 0.8263 | fnr: 0.1463 | true_positives: 804252.0000 | false_positives: 161747.0000 | true_negatives: 769624.0000 | false_negatives: 137785.0000 | loss: 0.3350
2025-10-20 11:57:40,504 - INFO - utils.experiment_framework - Epoch 3 [VALIDATION] auc: 0.9204 | f1: 0.8214 | accuracy: 0.8244 | precision: 0.8114 | recall: 0.8316 | specificity: 0.8177 | fnr: 0.1684 | true_positives: 157112.0000 | false_positives: 36519.0000 | true_negatives: 163754.0000 | false_negatives: 31811.0000 | loss: 0.3517
2025-10-20 11:57:40,527 - INFO - train_mobilenet - ✅ New best validation AUC: 0.9204 (+0.0030)
2025-10-20 11:57:40,527 - INFO - train_mobilenet - 📍 Best epoch: 3, Patience counter reset
2025-10-20 11:57:40,910 - INFO - utils.experiment_framework - ✓ New best model saved! AUC: 0.9204
2025-10-20 11:57:40,911 - INFO - train_mobilenet - Epoch 3/10 - Train Loss: 0.3350, Val Loss: 0.3517, Val AUC: 0.9204, Best AUC: 0.9204
2025-10-20 12:00:12,151 - INFO - models.mobilenetv4_model - 📈 Forward pass #54001 - dimensions stable
2025-10-20 12:02:46,151 - INFO - models.mobilenetv4_model - 📈 Forward pass #55001 - dimensions stable
2025-10-20 12:05:20,121 - INFO - models.mobilenetv4_model - 📈 Forward pass #56001 - dimensions stable
2025-10-20 12:07:54,230 - INFO - models.mobilenetv4_model - 📈 Forward pass #57001 - dimensions stable
2025-10-20 12:10:28,188 - INFO - models.mobilenetv4_model - 📈 Forward pass #58001 - dimensions stable
2025-10-20 12:13:02,163 - INFO - models.mobilenetv4_model - 📈 Forward pass #59001 - dimensions stable
2025-10-20 12:15:36,139 - INFO - models.mobilenetv4_model - 📈 Forward pass #60001 - dimensions stable
2025-10-20 12:16:30,885 - INFO - train_mobilenet - 📈 Training Epoch 4 Results:
2025-10-20 12:16:30,885 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 12:16:30,885 - INFO - train_mobilenet -    Average batch loss: 0.562116
2025-10-20 12:16:30,885 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 12:16:30,886 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 12:16:30,886 - INFO - train_mobilenet -    Target distribution: Real=0.504, Fake=0.496
2025-10-20 12:16:30,887 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 12:16:30,888 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 12:17:29,583 - INFO - models.mobilenetv4_model - 📈 Forward pass #61001 - dimensions stable
2025-10-20 12:18:56,852 - INFO - models.mobilenetv4_model - 📈 Forward pass #62001 - dimensions stable
2025-10-20 12:20:24,668 - INFO - models.mobilenetv4_model - 📈 Forward pass #63001 - dimensions stable
2025-10-20 12:21:52,216 - INFO - models.mobilenetv4_model - 📈 Forward pass #64001 - dimensions stable
2025-10-20 12:23:19,838 - INFO - models.mobilenetv4_model - 📈 Forward pass #65001 - dimensions stable
2025-10-20 12:24:47,576 - INFO - models.mobilenetv4_model - 📈 Forward pass #66001 - dimensions stable
2025-10-20 12:26:14,791 - INFO - models.mobilenetv4_model - 📈 Forward pass #67001 - dimensions stable
2025-10-20 12:27:15,347 - INFO - utils.evaluation - Train Loss: 0.4319
2025-10-20 12:27:15,347 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 12:27:15,347 - INFO - utils.evaluation - AUC: 0.8777
2025-10-20 12:27:15,347 - INFO - utils.evaluation - F1-Score: 0.7612
2025-10-20 12:27:15,347 - INFO - utils.evaluation - Accuracy: 0.7762
2025-10-20 12:27:15,347 - INFO - utils.evaluation - Precision: 0.8207
2025-10-20 12:27:15,347 - INFO - utils.evaluation - Recall: 0.7097
2025-10-20 12:27:15,347 - INFO - utils.evaluation - Specificity: 0.8434
2025-10-20 12:27:15,347 - INFO - utils.evaluation - False Negative Rate: 0.2903
2025-10-20 12:27:15,347 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 12:27:15,347 - INFO - train_mobilenet -    AUC: 0.8777
2025-10-20 12:27:15,347 - INFO - train_mobilenet -    Loss: 0.4319
2025-10-20 12:27:15,347 - INFO - train_mobilenet -    Accuracy: 0.7762
2025-10-20 12:27:15,347 - INFO - train_mobilenet -    F1-Score: 0.7612
2025-10-20 12:27:15,347 - WARNING - train_mobilenet - 📉 Learning degradation: AUC decreased by -0.0517
2025-10-20 12:27:45,780 - INFO - models.mobilenetv4_model - 📈 Forward pass #68001 - dimensions stable
2025-10-20 12:29:13,311 - INFO - models.mobilenetv4_model - 📈 Forward pass #69001 - dimensions stable
2025-10-20 12:29:30,026 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 12:30:41,818 - INFO - models.mobilenetv4_model - 📈 Forward pass #70001 - dimensions stable
2025-10-20 12:31:44,001 - INFO - utils.evaluation - Validation Loss: 0.3985
2025-10-20 12:31:44,001 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 12:31:44,001 - INFO - utils.evaluation - AUC: 0.8920
2025-10-20 12:31:44,001 - INFO - utils.evaluation - F1-Score: 0.7657
2025-10-20 12:31:44,001 - INFO - utils.evaluation - Accuracy: 0.7850
2025-10-20 12:31:44,001 - INFO - utils.evaluation - Precision: 0.8129
2025-10-20 12:31:44,001 - INFO - utils.evaluation - Recall: 0.7236
2025-10-20 12:31:44,001 - INFO - utils.evaluation - Specificity: 0.8429
2025-10-20 12:31:44,001 - INFO - utils.evaluation - False Negative Rate: 0.2764
2025-10-20 12:31:44,015 - INFO - utils.experiment_framework - Epoch 4 [TRAIN] auc: 0.8777 | f1: 0.7612 | accuracy: 0.7762 | precision: 0.8207 | recall: 0.7097 | specificity: 0.8434 | fnr: 0.2903 | true_positives: 668105.0000 | false_positives: 145953.0000 | true_negatives: 786053.0000 | false_negatives: 273297.0000 | loss: 0.4319
2025-10-20 12:31:44,016 - INFO - utils.experiment_framework - Epoch 4 [VALIDATION] auc: 0.8920 | f1: 0.7657 | accuracy: 0.7850 | precision: 0.8129 | recall: 0.7236 | specificity: 0.8429 | fnr: 0.2764 | true_positives: 136712.0000 | false_positives: 31465.0000 | true_negatives: 168808.0000 | false_negatives: 52211.0000 | loss: 0.3985
2025-10-20 12:31:44,016 - INFO - train_mobilenet - ⚠️  No improvement: Patience 1/5
2025-10-20 12:31:44,016 - INFO - train_mobilenet - Epoch 4/10 - Train Loss: 0.4319, Val Loss: 0.3985, Val AUC: 0.8920, Best AUC: 0.9204
2025-10-20 12:32:30,775 - INFO - models.mobilenetv4_model - 📈 Forward pass #71001 - dimensions stable
2025-10-20 12:35:04,742 - INFO - models.mobilenetv4_model - 📈 Forward pass #72001 - dimensions stable
2025-10-20 12:37:38,706 - INFO - models.mobilenetv4_model - 📈 Forward pass #73001 - dimensions stable
2025-10-20 12:40:12,655 - INFO - models.mobilenetv4_model - 📈 Forward pass #74001 - dimensions stable
2025-10-20 12:42:46,707 - INFO - models.mobilenetv4_model - 📈 Forward pass #75001 - dimensions stable
2025-10-20 12:45:20,670 - INFO - models.mobilenetv4_model - 📈 Forward pass #76001 - dimensions stable
2025-10-20 12:47:54,611 - INFO - models.mobilenetv4_model - 📈 Forward pass #77001 - dimensions stable
2025-10-20 12:50:28,662 - INFO - models.mobilenetv4_model - 📈 Forward pass #78001 - dimensions stable
2025-10-20 12:50:33,842 - INFO - train_mobilenet - 📈 Training Epoch 5 Results:
2025-10-20 12:50:33,842 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 12:50:33,842 - INFO - train_mobilenet -    Average batch loss: 0.541198
2025-10-20 12:50:33,842 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 12:50:33,842 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 12:50:33,843 - INFO - train_mobilenet -    Target distribution: Real=0.503, Fake=0.497
2025-10-20 12:50:33,844 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 12:50:33,845 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 12:52:01,232 - INFO - models.mobilenetv4_model - 📈 Forward pass #79001 - dimensions stable
2025-10-20 12:53:29,237 - INFO - models.mobilenetv4_model - 📈 Forward pass #80001 - dimensions stable
2025-10-20 12:54:57,264 - INFO - models.mobilenetv4_model - 📈 Forward pass #81001 - dimensions stable
2025-10-20 12:56:24,785 - INFO - models.mobilenetv4_model - 📈 Forward pass #82001 - dimensions stable
2025-10-20 12:57:52,158 - INFO - models.mobilenetv4_model - 📈 Forward pass #83001 - dimensions stable
2025-10-20 12:59:20,151 - INFO - models.mobilenetv4_model - 📈 Forward pass #84001 - dimensions stable
2025-10-20 13:00:47,636 - INFO - models.mobilenetv4_model - 📈 Forward pass #85001 - dimensions stable
2025-10-20 13:01:19,940 - INFO - utils.evaluation - Train Loss: 0.7195
2025-10-20 13:01:19,940 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 13:01:19,941 - INFO - utils.evaluation - AUC: 0.6053
2025-10-20 13:01:19,941 - INFO - utils.evaluation - F1-Score: 0.6874
2025-10-20 13:01:19,941 - INFO - utils.evaluation - Accuracy: 0.5692
2025-10-20 13:01:19,941 - INFO - utils.evaluation - Precision: 0.5417
2025-10-20 13:01:19,941 - INFO - utils.evaluation - Recall: 0.9401
2025-10-20 13:01:19,941 - INFO - utils.evaluation - Specificity: 0.1928
2025-10-20 13:01:19,941 - INFO - utils.evaluation - False Negative Rate: 0.0599
2025-10-20 13:01:19,941 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 13:01:19,941 - INFO - train_mobilenet -    AUC: 0.6053
2025-10-20 13:01:19,941 - INFO - train_mobilenet -    Loss: 0.7195
2025-10-20 13:01:19,941 - INFO - train_mobilenet -    Accuracy: 0.5692
2025-10-20 13:01:19,941 - INFO - train_mobilenet -    F1-Score: 0.6874
2025-10-20 13:01:19,941 - WARNING - train_mobilenet - 📉 Learning degradation: AUC decreased by -0.2724
2025-10-20 13:02:18,636 - INFO - models.mobilenetv4_model - 📈 Forward pass #86001 - dimensions stable
2025-10-20 13:03:35,254 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 13:03:48,186 - INFO - models.mobilenetv4_model - 📈 Forward pass #87001 - dimensions stable
2025-10-20 13:05:17,177 - INFO - models.mobilenetv4_model - 📈 Forward pass #88001 - dimensions stable
2025-10-20 13:05:52,274 - INFO - utils.evaluation - Validation Loss: 0.7097
2025-10-20 13:05:52,274 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 13:05:52,274 - INFO - utils.evaluation - AUC: 0.6501
2025-10-20 13:05:52,274 - INFO - utils.evaluation - F1-Score: 0.6896
2025-10-20 13:05:52,274 - INFO - utils.evaluation - Accuracy: 0.5907
2025-10-20 13:05:52,274 - INFO - utils.evaluation - Precision: 0.5456
2025-10-20 13:05:52,274 - INFO - utils.evaluation - Recall: 0.9369
2025-10-20 13:05:52,274 - INFO - utils.evaluation - Specificity: 0.2640
2025-10-20 13:05:52,274 - INFO - utils.evaluation - False Negative Rate: 0.0631
2025-10-20 13:05:52,289 - INFO - utils.experiment_framework - Epoch 5 [TRAIN] auc: 0.6053 | f1: 0.6874 | accuracy: 0.5692 | precision: 0.5417 | recall: 0.9401 | specificity: 0.1928 | fnr: 0.0599 | true_positives: 887163.0000 | false_positives: 750434.0000 | true_negatives: 179263.0000 | false_negatives: 56548.0000 | loss: 0.7195
2025-10-20 13:05:52,290 - INFO - utils.experiment_framework - Epoch 5 [VALIDATION] auc: 0.6501 | f1: 0.6896 | accuracy: 0.5907 | precision: 0.5456 | recall: 0.9369 | specificity: 0.2640 | fnr: 0.0631 | true_positives: 177010.0000 | false_positives: 147400.0000 | true_negatives: 52873.0000 | false_negatives: 11913.0000 | loss: 0.7097
2025-10-20 13:05:52,290 - INFO - train_mobilenet - ⚠️  No improvement: Patience 2/5
2025-10-20 13:05:52,290 - INFO - train_mobilenet - Epoch 5/10 - Train Loss: 0.7195, Val Loss: 0.7097, Val AUC: 0.6501, Best AUC: 0.9204
2025-10-20 13:07:28,549 - INFO - models.mobilenetv4_model - 📈 Forward pass #89001 - dimensions stable
2025-10-20 13:10:02,550 - INFO - models.mobilenetv4_model - 📈 Forward pass #90001 - dimensions stable
2025-10-20 13:12:36,529 - INFO - models.mobilenetv4_model - 📈 Forward pass #91001 - dimensions stable
2025-10-20 13:15:10,619 - INFO - models.mobilenetv4_model - 📈 Forward pass #92001 - dimensions stable
2025-10-20 13:17:44,576 - INFO - models.mobilenetv4_model - 📈 Forward pass #93001 - dimensions stable
2025-10-20 13:20:18,570 - INFO - models.mobilenetv4_model - 📈 Forward pass #94001 - dimensions stable
2025-10-20 13:22:52,545 - INFO - models.mobilenetv4_model - 📈 Forward pass #95001 - dimensions stable
2025-10-20 13:24:42,490 - INFO - train_mobilenet - 📈 Training Epoch 6 Results:
2025-10-20 13:24:42,491 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 13:24:42,491 - INFO - train_mobilenet -    Average batch loss: 0.491840
2025-10-20 13:24:42,491 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 13:24:42,492 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 13:24:42,494 - INFO - train_mobilenet -    Target distribution: Real=0.503, Fake=0.497
2025-10-20 13:24:42,508 - INFO - train_mobilenet - 📊 Gradient norms (Epoch 6):
2025-10-20 13:24:42,508 - INFO - train_mobilenet -    Classifier: 0.092295
2025-10-20 13:24:42,509 - INFO - train_mobilenet -    Backbone: 48.049330
2025-10-20 13:24:42,509 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 13:24:42,512 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 13:25:18,577 - INFO - models.mobilenetv4_model - 📈 Forward pass #96001 - dimensions stable
2025-10-20 13:26:50,940 - INFO - models.mobilenetv4_model - 📈 Forward pass #97001 - dimensions stable
2025-10-20 13:28:20,038 - INFO - models.mobilenetv4_model - 📈 Forward pass #98001 - dimensions stable
2025-10-20 13:29:49,133 - INFO - models.mobilenetv4_model - 📈 Forward pass #99001 - dimensions stable
2025-10-20 13:31:17,953 - INFO - models.mobilenetv4_model - 📈 Forward pass #100001 - dimensions stable
2025-10-20 13:32:46,289 - INFO - models.mobilenetv4_model - 📈 Forward pass #101001 - dimensions stable
2025-10-20 13:34:14,691 - INFO - models.mobilenetv4_model - 📈 Forward pass #102001 - dimensions stable
2025-10-20 13:35:42,844 - INFO - models.mobilenetv4_model - 📈 Forward pass #103001 - dimensions stable
2025-10-20 13:35:47,192 - INFO - utils.evaluation - Train Loss: 0.3796
2025-10-20 13:35:47,192 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 13:35:47,192 - INFO - utils.evaluation - AUC: 0.9067
2025-10-20 13:35:47,192 - INFO - utils.evaluation - F1-Score: 0.8072
2025-10-20 13:35:47,192 - INFO - utils.evaluation - Accuracy: 0.8120
2025-10-20 13:35:47,192 - INFO - utils.evaluation - Precision: 0.8346
2025-10-20 13:35:47,192 - INFO - utils.evaluation - Recall: 0.7816
2025-10-20 13:35:47,192 - INFO - utils.evaluation - Specificity: 0.8428
2025-10-20 13:35:47,192 - INFO - utils.evaluation - False Negative Rate: 0.2184
2025-10-20 13:35:47,192 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 13:35:47,192 - INFO - train_mobilenet -    AUC: 0.9067
2025-10-20 13:35:47,192 - INFO - train_mobilenet -    Loss: 0.3796
2025-10-20 13:35:47,192 - INFO - train_mobilenet -    Accuracy: 0.8120
2025-10-20 13:35:47,192 - INFO - train_mobilenet -    F1-Score: 0.8072
2025-10-20 13:35:47,192 - INFO - train_mobilenet - 📈 Learning progress: AUC improved by +0.3014
2025-10-20 13:37:15,092 - INFO - models.mobilenetv4_model - 📈 Forward pass #104001 - dimensions stable
2025-10-20 13:38:03,692 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 13:38:45,264 - INFO - models.mobilenetv4_model - 📈 Forward pass #105001 - dimensions stable
2025-10-20 13:40:13,891 - INFO - models.mobilenetv4_model - 📈 Forward pass #106001 - dimensions stable
2025-10-20 13:40:20,279 - INFO - utils.evaluation - Validation Loss: 0.3731
2025-10-20 13:40:20,279 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 13:40:20,279 - INFO - utils.evaluation - AUC: 0.9061
2025-10-20 13:40:20,279 - INFO - utils.evaluation - F1-Score: 0.7973
2025-10-20 13:40:20,279 - INFO - utils.evaluation - Accuracy: 0.8064
2025-10-20 13:40:20,279 - INFO - utils.evaluation - Precision: 0.8106
2025-10-20 13:40:20,280 - INFO - utils.evaluation - Recall: 0.7844
2025-10-20 13:40:20,280 - INFO - utils.evaluation - Specificity: 0.8271
2025-10-20 13:40:20,280 - INFO - utils.evaluation - False Negative Rate: 0.2156
2025-10-20 13:40:20,295 - INFO - utils.experiment_framework - Epoch 6 [TRAIN] auc: 0.9067 | f1: 0.8072 | accuracy: 0.8120 | precision: 0.8346 | recall: 0.7816 | specificity: 0.8428 | fnr: 0.2184 | true_positives: 737403.0000 | false_positives: 146177.0000 | true_negatives: 783823.0000 | false_negatives: 206005.0000 | loss: 0.3796
2025-10-20 13:40:20,296 - INFO - utils.experiment_framework - Epoch 6 [VALIDATION] auc: 0.9061 | f1: 0.7973 | accuracy: 0.8064 | precision: 0.8106 | recall: 0.7844 | specificity: 0.8271 | fnr: 0.2156 | true_positives: 148196.0000 | false_positives: 34632.0000 | true_negatives: 165641.0000 | false_negatives: 40727.0000 | loss: 0.3731
2025-10-20 13:40:20,296 - INFO - train_mobilenet - ⚠️  No improvement: Patience 3/5
2025-10-20 13:40:20,296 - INFO - train_mobilenet - Epoch 6/10 - Train Loss: 0.3796, Val Loss: 0.3731, Val AUC: 0.9061, Best AUC: 0.9204
2025-10-20 13:42:46,394 - INFO - models.mobilenetv4_model - 📈 Forward pass #107001 - dimensions stable
2025-10-20 13:45:20,397 - INFO - models.mobilenetv4_model - 📈 Forward pass #108001 - dimensions stable
2025-10-20 13:47:54,393 - INFO - models.mobilenetv4_model - 📈 Forward pass #109001 - dimensions stable
2025-10-20 13:50:28,545 - INFO - models.mobilenetv4_model - 📈 Forward pass #110001 - dimensions stable
2025-10-20 13:53:02,532 - INFO - models.mobilenetv4_model - 📈 Forward pass #111001 - dimensions stable
2025-10-20 13:55:36,504 - INFO - models.mobilenetv4_model - 📈 Forward pass #112001 - dimensions stable
2025-10-20 13:58:10,554 - INFO - models.mobilenetv4_model - 📈 Forward pass #113001 - dimensions stable
2025-10-20 13:59:10,541 - INFO - train_mobilenet - 📈 Training Epoch 7 Results:
2025-10-20 13:59:10,542 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 13:59:10,542 - INFO - train_mobilenet -    Average batch loss: 0.367853
2025-10-20 13:59:10,542 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 13:59:10,542 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 13:59:10,543 - INFO - train_mobilenet -    Target distribution: Real=0.503, Fake=0.497
2025-10-20 13:59:10,543 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 13:59:10,545 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 14:00:07,361 - INFO - models.mobilenetv4_model - 📈 Forward pass #114001 - dimensions stable
2025-10-20 14:01:36,121 - INFO - models.mobilenetv4_model - 📈 Forward pass #115001 - dimensions stable
2025-10-20 14:03:05,122 - INFO - models.mobilenetv4_model - 📈 Forward pass #116001 - dimensions stable
2025-10-20 14:04:33,429 - INFO - models.mobilenetv4_model - 📈 Forward pass #117001 - dimensions stable
2025-10-20 14:06:01,470 - INFO - models.mobilenetv4_model - 📈 Forward pass #118001 - dimensions stable
2025-10-20 14:07:29,244 - INFO - models.mobilenetv4_model - 📈 Forward pass #119001 - dimensions stable
2025-10-20 14:08:56,932 - INFO - models.mobilenetv4_model - 📈 Forward pass #120001 - dimensions stable
2025-10-20 14:10:00,650 - INFO - utils.evaluation - Train Loss: 0.2793
2025-10-20 14:10:00,650 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 14:10:00,650 - INFO - utils.evaluation - AUC: 0.9525
2025-10-20 14:10:00,650 - INFO - utils.evaluation - F1-Score: 0.8708
2025-10-20 14:10:00,650 - INFO - utils.evaluation - Accuracy: 0.8720
2025-10-20 14:10:00,650 - INFO - utils.evaluation - Precision: 0.8857
2025-10-20 14:10:00,650 - INFO - utils.evaluation - Recall: 0.8564
2025-10-20 14:10:00,650 - INFO - utils.evaluation - Specificity: 0.8879
2025-10-20 14:10:00,650 - INFO - utils.evaluation - False Negative Rate: 0.1436
2025-10-20 14:10:00,650 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 14:10:00,650 - INFO - train_mobilenet -    AUC: 0.9525
2025-10-20 14:10:00,650 - INFO - train_mobilenet -    Loss: 0.2793
2025-10-20 14:10:00,650 - INFO - train_mobilenet -    Accuracy: 0.8720
2025-10-20 14:10:00,650 - INFO - train_mobilenet -    F1-Score: 0.8708
2025-10-20 14:10:00,650 - INFO - train_mobilenet - 📈 Learning progress: AUC improved by +0.0458
2025-10-20 14:10:28,481 - INFO - models.mobilenetv4_model - 📈 Forward pass #121001 - dimensions stable
2025-10-20 14:11:57,459 - INFO - models.mobilenetv4_model - 📈 Forward pass #122001 - dimensions stable
2025-10-20 14:12:17,379 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 14:13:27,413 - INFO - models.mobilenetv4_model - 📈 Forward pass #123001 - dimensions stable
2025-10-20 14:14:33,829 - INFO - utils.evaluation - Validation Loss: 0.3214
2025-10-20 14:14:33,829 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 14:14:33,829 - INFO - utils.evaluation - AUC: 0.9328
2025-10-20 14:14:33,829 - INFO - utils.evaluation - F1-Score: 0.8368
2025-10-20 14:14:33,829 - INFO - utils.evaluation - Accuracy: 0.8413
2025-10-20 14:14:33,829 - INFO - utils.evaluation - Precision: 0.8354
2025-10-20 14:14:33,829 - INFO - utils.evaluation - Recall: 0.8382
2025-10-20 14:14:33,829 - INFO - utils.evaluation - Specificity: 0.8442
2025-10-20 14:14:33,829 - INFO - utils.evaluation - False Negative Rate: 0.1618
2025-10-20 14:14:33,845 - INFO - utils.experiment_framework - Epoch 7 [TRAIN] auc: 0.9525 | f1: 0.8708 | accuracy: 0.8720 | precision: 0.8857 | recall: 0.8564 | specificity: 0.8879 | fnr: 0.1436 | true_positives: 807924.0000 | false_positives: 104282.0000 | true_negatives: 825766.0000 | false_negatives: 135436.0000 | loss: 0.2793
2025-10-20 14:14:33,846 - INFO - utils.experiment_framework - Epoch 7 [VALIDATION] auc: 0.9328 | f1: 0.8368 | accuracy: 0.8413 | precision: 0.8354 | recall: 0.8382 | specificity: 0.8442 | fnr: 0.1618 | true_positives: 158356.0000 | false_positives: 31195.0000 | true_negatives: 169078.0000 | false_negatives: 30567.0000 | loss: 0.3214
2025-10-20 14:14:33,869 - INFO - train_mobilenet - ✅ New best validation AUC: 0.9328 (+0.0124)
2025-10-20 14:14:33,869 - INFO - train_mobilenet - 📍 Best epoch: 7, Patience counter reset
2025-10-20 14:14:34,071 - INFO - utils.experiment_framework - ✓ New best model saved! AUC: 0.9328
2025-10-20 14:14:34,072 - INFO - train_mobilenet - Epoch 7/10 - Train Loss: 0.2793, Val Loss: 0.3214, Val AUC: 0.9328, Best AUC: 0.9328
2025-10-20 14:15:15,428 - INFO - models.mobilenetv4_model - 📈 Forward pass #124001 - dimensions stable
2025-10-20 14:17:49,427 - INFO - models.mobilenetv4_model - 📈 Forward pass #125001 - dimensions stable
2025-10-20 14:20:23,407 - INFO - models.mobilenetv4_model - 📈 Forward pass #126001 - dimensions stable
2025-10-20 14:22:57,503 - INFO - models.mobilenetv4_model - 📈 Forward pass #127001 - dimensions stable
2025-10-20 14:25:31,482 - INFO - models.mobilenetv4_model - 📈 Forward pass #128001 - dimensions stable
2025-10-20 14:28:05,464 - INFO - models.mobilenetv4_model - 📈 Forward pass #129001 - dimensions stable
2025-10-20 14:30:39,542 - INFO - models.mobilenetv4_model - 📈 Forward pass #130001 - dimensions stable
2025-10-20 14:33:13,546 - INFO - models.mobilenetv4_model - 📈 Forward pass #131001 - dimensions stable
2025-10-20 14:33:23,938 - INFO - train_mobilenet - 📈 Training Epoch 8 Results:
2025-10-20 14:33:23,938 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 14:33:23,938 - INFO - train_mobilenet -    Average batch loss: 0.336146
2025-10-20 14:33:23,938 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 14:33:23,939 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 14:33:23,940 - INFO - train_mobilenet -    Target distribution: Real=0.503, Fake=0.497
2025-10-20 14:33:23,940 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 14:33:23,942 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 14:34:49,309 - INFO - models.mobilenetv4_model - 📈 Forward pass #132001 - dimensions stable
2025-10-20 14:36:18,271 - INFO - models.mobilenetv4_model - 📈 Forward pass #133001 - dimensions stable
2025-10-20 14:37:47,087 - INFO - models.mobilenetv4_model - 📈 Forward pass #134001 - dimensions stable
2025-10-20 14:39:15,241 - INFO - models.mobilenetv4_model - 📈 Forward pass #135001 - dimensions stable
2025-10-20 14:40:43,553 - INFO - models.mobilenetv4_model - 📈 Forward pass #136001 - dimensions stable
2025-10-20 14:42:11,734 - INFO - models.mobilenetv4_model - 📈 Forward pass #137001 - dimensions stable
2025-10-20 14:43:39,560 - INFO - models.mobilenetv4_model - 📈 Forward pass #138001 - dimensions stable
2025-10-20 14:44:15,327 - INFO - utils.evaluation - Train Loss: 0.2931
2025-10-20 14:44:15,328 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 14:44:15,328 - INFO - utils.evaluation - AUC: 0.9464
2025-10-20 14:44:15,328 - INFO - utils.evaluation - F1-Score: 0.8638
2025-10-20 14:44:15,328 - INFO - utils.evaluation - Accuracy: 0.8629
2025-10-20 14:44:15,328 - INFO - utils.evaluation - Precision: 0.8643
2025-10-20 14:44:15,328 - INFO - utils.evaluation - Recall: 0.8633
2025-10-20 14:44:15,328 - INFO - utils.evaluation - Specificity: 0.8625
2025-10-20 14:44:15,328 - INFO - utils.evaluation - False Negative Rate: 0.1367
2025-10-20 14:44:15,328 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 14:44:15,328 - INFO - train_mobilenet -    AUC: 0.9464
2025-10-20 14:44:15,328 - INFO - train_mobilenet -    Loss: 0.2931
2025-10-20 14:44:15,328 - INFO - train_mobilenet -    Accuracy: 0.8629
2025-10-20 14:44:15,328 - INFO - train_mobilenet -    F1-Score: 0.8638
2025-10-20 14:44:15,328 - WARNING - train_mobilenet - 📉 Learning degradation: AUC decreased by -0.0061
2025-10-20 14:45:11,389 - INFO - models.mobilenetv4_model - 📈 Forward pass #139001 - dimensions stable
2025-10-20 14:46:31,151 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 14:46:40,950 - INFO - models.mobilenetv4_model - 📈 Forward pass #140001 - dimensions stable
2025-10-20 14:48:09,596 - INFO - models.mobilenetv4_model - 📈 Forward pass #141001 - dimensions stable
2025-10-20 14:48:47,620 - INFO - utils.evaluation - Validation Loss: 0.3333
2025-10-20 14:48:47,620 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 14:48:47,620 - INFO - utils.evaluation - AUC: 0.9300
2025-10-20 14:48:47,620 - INFO - utils.evaluation - F1-Score: 0.8368
2025-10-20 14:48:47,620 - INFO - utils.evaluation - Accuracy: 0.8389
2025-10-20 14:48:47,620 - INFO - utils.evaluation - Precision: 0.8229
2025-10-20 14:48:47,620 - INFO - utils.evaluation - Recall: 0.8512
2025-10-20 14:48:47,620 - INFO - utils.evaluation - Specificity: 0.8273
2025-10-20 14:48:47,620 - INFO - utils.evaluation - False Negative Rate: 0.1488
2025-10-20 14:48:47,635 - INFO - utils.experiment_framework - Epoch 8 [TRAIN] auc: 0.9464 | f1: 0.8638 | accuracy: 0.8629 | precision: 0.8643 | recall: 0.8633 | specificity: 0.8625 | fnr: 0.1367 | true_positives: 814533.0000 | false_positives: 127896.0000 | true_negatives: 802035.0000 | false_negatives: 128944.0000 | loss: 0.2931
2025-10-20 14:48:47,636 - INFO - utils.experiment_framework - Epoch 8 [VALIDATION] auc: 0.9300 | f1: 0.8368 | accuracy: 0.8389 | precision: 0.8229 | recall: 0.8512 | specificity: 0.8273 | fnr: 0.1488 | true_positives: 160805.0000 | false_positives: 34597.0000 | true_negatives: 165676.0000 | false_negatives: 28118.0000 | loss: 0.3333
2025-10-20 14:48:47,636 - INFO - train_mobilenet - ⚠️  No improvement: Patience 1/5
2025-10-20 14:48:47,636 - INFO - train_mobilenet - Epoch 8/10 - Train Loss: 0.2931, Val Loss: 0.3333, Val AUC: 0.9300, Best AUC: 0.9328
2025-10-20 14:50:18,882 - INFO - models.mobilenetv4_model - 📈 Forward pass #142001 - dimensions stable
2025-10-20 14:52:52,902 - INFO - models.mobilenetv4_model - 📈 Forward pass #143001 - dimensions stable
2025-10-20 14:55:27,046 - INFO - models.mobilenetv4_model - 📈 Forward pass #144001 - dimensions stable
2025-10-20 14:58:01,024 - INFO - models.mobilenetv4_model - 📈 Forward pass #145001 - dimensions stable
2025-10-20 15:00:35,003 - INFO - models.mobilenetv4_model - 📈 Forward pass #146001 - dimensions stable
2025-10-20 15:03:08,962 - INFO - models.mobilenetv4_model - 📈 Forward pass #147001 - dimensions stable
2025-10-20 15:05:43,020 - INFO - models.mobilenetv4_model - 📈 Forward pass #148001 - dimensions stable
2025-10-20 15:07:37,801 - INFO - train_mobilenet - 📈 Training Epoch 9 Results:
2025-10-20 15:07:37,802 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 15:07:37,802 - INFO - train_mobilenet -    Average batch loss: 0.310658
2025-10-20 15:07:37,802 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 15:07:37,802 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 15:07:37,803 - INFO - train_mobilenet -    Target distribution: Real=0.503, Fake=0.497
2025-10-20 15:07:37,803 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 15:07:37,805 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 15:08:02,823 - INFO - models.mobilenetv4_model - 📈 Forward pass #149001 - dimensions stable
2025-10-20 15:09:31,962 - INFO - models.mobilenetv4_model - 📈 Forward pass #150001 - dimensions stable
2025-10-20 15:11:01,129 - INFO - models.mobilenetv4_model - 📈 Forward pass #151001 - dimensions stable
2025-10-20 15:12:29,889 - INFO - models.mobilenetv4_model - 📈 Forward pass #152001 - dimensions stable
2025-10-20 15:13:57,974 - INFO - models.mobilenetv4_model - 📈 Forward pass #153001 - dimensions stable
2025-10-20 15:15:26,243 - INFO - models.mobilenetv4_model - 📈 Forward pass #154001 - dimensions stable
2025-10-20 15:16:53,942 - INFO - models.mobilenetv4_model - 📈 Forward pass #155001 - dimensions stable
2025-10-20 15:18:22,252 - INFO - models.mobilenetv4_model - 📈 Forward pass #156001 - dimensions stable
2025-10-20 15:18:29,567 - INFO - utils.evaluation - Train Loss: 0.2724
2025-10-20 15:18:29,567 - INFO - utils.evaluation - === TRAIN Results ===
2025-10-20 15:18:29,567 - INFO - utils.evaluation - AUC: 0.9537
2025-10-20 15:18:29,567 - INFO - utils.evaluation - F1-Score: 0.8742
2025-10-20 15:18:29,567 - INFO - utils.evaluation - Accuracy: 0.8726
2025-10-20 15:18:29,567 - INFO - utils.evaluation - Precision: 0.8687
2025-10-20 15:18:29,567 - INFO - utils.evaluation - Recall: 0.8798
2025-10-20 15:18:29,567 - INFO - utils.evaluation - Specificity: 0.8652
2025-10-20 15:18:29,567 - INFO - utils.evaluation - False Negative Rate: 0.1202
2025-10-20 15:18:29,567 - INFO - train_mobilenet - 🎯 Training Performance Summary:
2025-10-20 15:18:29,567 - INFO - train_mobilenet -    AUC: 0.9537
2025-10-20 15:18:29,567 - INFO - train_mobilenet -    Loss: 0.2724
2025-10-20 15:18:29,567 - INFO - train_mobilenet -    Accuracy: 0.8726
2025-10-20 15:18:29,568 - INFO - train_mobilenet -    F1-Score: 0.8742
2025-10-20 15:18:29,568 - INFO - train_mobilenet - 📈 Learning progress: AUC improved by +0.0073
2025-10-20 15:19:54,073 - INFO - models.mobilenetv4_model - 📈 Forward pass #157001 - dimensions stable
2025-10-20 15:20:45,321 - INFO - utils.evaluation - Evaluating model on validation set...
2025-10-20 15:21:23,821 - INFO - models.mobilenetv4_model - 📈 Forward pass #158001 - dimensions stable
2025-10-20 15:22:51,544 - INFO - models.mobilenetv4_model - 📈 Forward pass #159001 - dimensions stable
2025-10-20 15:23:00,886 - INFO - utils.evaluation - Validation Loss: 0.3265
2025-10-20 15:23:00,886 - INFO - utils.evaluation - === VALIDATION Results ===
2025-10-20 15:23:00,886 - INFO - utils.evaluation - AUC: 0.9343
2025-10-20 15:23:00,886 - INFO - utils.evaluation - F1-Score: 0.8466
2025-10-20 15:23:00,886 - INFO - utils.evaluation - Accuracy: 0.8462
2025-10-20 15:23:00,886 - INFO - utils.evaluation - Precision: 0.8206
2025-10-20 15:23:00,886 - INFO - utils.evaluation - Recall: 0.8743
2025-10-20 15:23:00,886 - INFO - utils.evaluation - Specificity: 0.8196
2025-10-20 15:23:00,886 - INFO - utils.evaluation - False Negative Rate: 0.1257
2025-10-20 15:23:00,901 - INFO - utils.experiment_framework - Epoch 9 [TRAIN] auc: 0.9537 | f1: 0.8742 | accuracy: 0.8726 | precision: 0.8687 | recall: 0.8798 | specificity: 0.8652 | fnr: 0.1202 | true_positives: 829500.0000 | false_positives: 125423.0000 | true_negatives: 805150.0000 | false_negatives: 113335.0000 | loss: 0.2724
2025-10-20 15:23:00,902 - INFO - utils.experiment_framework - Epoch 9 [VALIDATION] auc: 0.9343 | f1: 0.8466 | accuracy: 0.8462 | precision: 0.8206 | recall: 0.8743 | specificity: 0.8196 | fnr: 0.1257 | true_positives: 165183.0000 | false_positives: 36120.0000 | true_negatives: 164153.0000 | false_negatives: 23740.0000 | loss: 0.3265
2025-10-20 15:23:00,924 - INFO - train_mobilenet - ✅ New best validation AUC: 0.9343 (+0.0015)
2025-10-20 15:23:00,925 - INFO - train_mobilenet - 📍 Best epoch: 9, Patience counter reset
2025-10-20 15:23:01,126 - INFO - utils.experiment_framework - ✓ New best model saved! AUC: 0.9343
2025-10-20 15:23:01,128 - INFO - train_mobilenet - Epoch 9/10 - Train Loss: 0.2724, Val Loss: 0.3265, Val AUC: 0.9343, Best AUC: 0.9343
2025-10-20 15:25:21,924 - INFO - models.mobilenetv4_model - 📈 Forward pass #160001 - dimensions stable
2025-10-20 15:27:55,925 - INFO - models.mobilenetv4_model - 📈 Forward pass #161001 - dimensions stable
2025-10-20 15:30:30,071 - INFO - models.mobilenetv4_model - 📈 Forward pass #162001 - dimensions stable
2025-10-20 15:33:04,068 - INFO - models.mobilenetv4_model - 📈 Forward pass #163001 - dimensions stable
2025-10-20 15:35:38,055 - INFO - models.mobilenetv4_model - 📈 Forward pass #164001 - dimensions stable
2025-10-20 15:38:12,158 - INFO - models.mobilenetv4_model - 📈 Forward pass #165001 - dimensions stable
2025-10-20 15:40:46,149 - INFO - models.mobilenetv4_model - 📈 Forward pass #166001 - dimensions stable
2025-10-20 15:41:51,452 - INFO - train_mobilenet - 📈 Training Epoch 10 Results:
2025-10-20 15:41:51,452 - INFO - train_mobilenet -    Total batches processed: 7318
2025-10-20 15:41:51,452 - INFO - train_mobilenet -    Average batch loss: 0.339690
2025-10-20 15:41:51,452 - INFO - train_mobilenet -    Total samples processed: 1873408
2025-10-20 15:41:51,453 - INFO - train_mobilenet -    Prediction range: [0.0000, 1.0000]
2025-10-20 15:41:51,454 - INFO - train_mobilenet -    Target distribution: Real=0.503, Fake=0.497
2025-10-20 15:41:51,454 - INFO - train_mobilenet - 🔍 Running comprehensive evaluation...
2025-10-20 15:41:51,456 - INFO - utils.evaluation - Evaluating model on train set...
2025-10-20 15:42:44,251 - INFO - models.mobilenetv4_model - 📈 Forward pass #167001 - dimensions stable
2025-10-20 15:44:11,881 - INFO - models.mobilenetv4_model - 📈 Forward pass #168001 - dimensions stable
2025-10-20 15:45:39,637 - INFO - models.mobilenetv4_model - 📈 Forward pass #169001 - dimensions stable
