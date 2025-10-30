Run 1 – 128 / 6e-05 / neg=1.25

  - 目的：直接對比 run_20251028_065314（lr=4e-05）檢查學習率上調是否改善 AUC/特異度。

  python -m src.training.train_efficientnet \
    --train_manifest manifests/train_difficult_subset.csv \
    --batch_size 128 \
    --learning_rate 6e-05 \
    --neg_class_weight 1.25 \
    --label_smoothing 0.05 \
    --use_scheduler true \
    --patience 5 \
    --epochs 30

  Run 2 – 128 / 5e-05 / neg=1.25

  - 目的：若 6e-05 不穩，5e-05 作為中間值；同時觀察 scheduler 觸發頻率。

  ... --batch_size 128 --learning_rate 5e-05 --neg_class_weight 1.25 ...

  Run 3 – 128 / 4e-05 / neg=1.20

  - 目的：在原 lr 下降低 neg_class_weight，檢查 Label Smoothing 是否可自行抑制 FP。

  ... --batch_size 128 --learning_rate 4e-05 --neg_class_weight 1.2 ...

  Run 4 – 128 / 6e-05 / neg=1.20

  - 目的：結合較高 lr 與較低權重，追求更高 recall，同時看 FP 是否保持。

  ... --batch_size 128 --learning_rate 6e-05 --neg_class_weight 1.2 ...

  Run 5 – 128 / 6e-05 / neg=1.00

  - 目的：驗證完全移除額外權重時，Label Smoothing 是否足以控制假陽性。

  ... --batch_size 128 --learning_rate 6e-05 --neg_class_weight 1.0 ...

  Run 6 – 256 / 6e-05 / neg=1.25

  - 目的：轉換到大 batch，確認高 lr + 大 batch 是否穩定（比對舊的 run_20251028_065314）。

  ... --batch_size 256 --learning_rate 6e-05 --neg_class_weight 1.25 ...

  Run 7 – 256 / 5e-05 / neg=1.25

  - 目的：若 Run 6 過擬合，降低 lr 再試；也可觀察收斂速度差異。

  ... --batch_size 256 --learning_rate 5e-05 --neg_class_weight 1.25 ...

  Run 8 – 256 / 4e-05 / neg=1.20

  - 目的：在大 batch 下同時放鬆權重，測試是否能兼得 specificity 與 recall。

  ... --batch_size 256 --learning_rate 4e-05 --neg_class_weight 1.2 ...

  Run 9 – 256 / 6e-05 / neg=1.20

  - 目的：延續 Run 8，回升 lr 看是否提高 AUC/recall。

  ... --batch_size 256 --learning_rate 6e-05 --neg_class_weight 1.2 ...

  Run 10 – 256 / 4e-05 / neg=1.00

  - 目的：在大 batch 下完全取消權重，檢查最平衡的基線與 cascade 閾值對應。

  ... --batch_size 256 --learning_rate 4e-05 --neg_class_weight 1.0 ...