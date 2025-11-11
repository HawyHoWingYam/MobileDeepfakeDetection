冻结配置

  - Stage 1: outputs/stage1/run_20251023_034316/best_model.pth（输入 256）
  - Stage 2: outputs/stage5/finetune_s2_b3_r4_pseudo/run_20251110_091416/best_model.pth（输入 512）
  - 阈值文件: outputs/stage4/run_manual_thresholds_r4_512/best_config.json
      - low=0.01, high=0.95, stage2_threshold=0.90（偏保守，显著降误报）
  - 温标: 1.0（避免异常超大 T=3320）
  - TTA: 移动端默认关闭；桌面评估可开启 --stage2-tta hflip 换取少量提升

  一步导出（Stage 6）

  - 命令（TorchScript + 动态量化 Linear；带阈值元数据）:
      - python -m src.tools.export_torchscript --stage both --stage1-ckpt outputs/stage1/run_20251023_034316/best_model.pth
        --stage2-ckpt outputs/stage5/finetune_s2_b3_r4_pseudo/run_20251110_091416/best_model.pth --stage1-model
        mobilenetv4_hybrid_medium --stage2-model tf_efficientnetv2_b3 --stage1-size 256 --stage2-size 512 --thresholds outputs/stage4/
        run_manual_thresholds_r4_512/best_config.json --temperature 1.0 --quantize-dynamic true --output-dir outputs/stage6/export_ts
  - 产物（交付）:
      - outputs/stage6/export_ts/stage1_mobilenetv4_ts.pt
      - outputs/stage6/export_ts/stage2_efficientnetv2_ts.pt
      - outputs/stage6/export_ts/bundle_meta.json（记录阈值文件、输入尺寸、温标）

  最终指标（可选，留档用）

  - 你已生成 val 的“手工阈值”评估：outputs/stage5/evals_r4_512_manual/summary.csv
  - 如需 test 也留档（不影响导出，时间较长）：
      - python src/tools/eval_cascade_across_datasets.py --stage1-ckpt outputs/stage1/run_20251023_034316/best_model.pth
        --stage2-ckpt outputs/stage5/finetune_s2_b3_r4_pseudo/run_20251110_091416/best_model.pth --thresholds outputs/stage4/
        run_manual_thresholds_r4_512/best_config.json --stage2-model tf_efficientnetv2_b3 --stage2-size 512 --stage2-temperature 1.0
        --datasets deepfake_eval_2024 --splits test --output-dir outputs/stage5/evals_r4_512_manual --resume

  交付清单

  - 模型包: outputs/stage6/export_ts/*
  - 阈值: outputs/stage4/run_manual_thresholds_r4_512/best_config.json
  - 指标: outputs/stage5/evals_r4_512_manual/summary.csv（已含 val；若跑了 test 也会写在同目录）

  后续提升（并行进行，不阻塞 Stage 6）

  - 如果后面要继续冲 0.8：
      - 训练 r5（12–16 epoch，512，AMP/累积/ckpt，80/20 伪标签），出新 best 后重复“评估 → 三维阈值复扫（可加 Precision 下限） → 应用阈
        值 → 重新导出”，只需替换 stage2_efficientnetv2_ts.pt 与 bundle_meta.json 即可。

  需要我现在帮你执行导出命令并回填导出目录与 meta 内容吗？