import sys
import json
import torch
from pathlib import Path
sys.path.insert(0, 'src')

from models.efficientnetv2_model import create_baseline_model
from training.dataset import CelebDFDataset
from utils.evaluation import ModelEvaluator
from torch.utils.data import DataLoader

print("=" * 80)
print("按数据集拆分评估（使用阈值 0.70）")
print("=" * 80)

# 加载模型
run_dir = Path("outputs/stage3/run_20251029_043420")
print(f"\n加载训练摘要...")
summary = json.loads((run_dir / "training_summary.json").read_text())

print(f"加载最佳模型 checkpoint...")
# 自动选择设备加载
device_for_load = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(run_dir / "best_model.pth", map_location=device_for_load)

# 创建模型
model = create_baseline_model(
    pretrained=False,
    dropout_rate=summary["hyperparameters"]["dropout_rate"],
    model_name=summary["hyperparameters"]["model_name"]
)

# 加载权重
model.load_state_dict(ckpt["model_state_dict"])
# 使用 GPU 如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
print(f"✓ 模型已加载到 {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 定义评估函数
def eval_manifest(manifest_name):
    """评估单个数据集"""
    manifest_path = f"manifests/{manifest_name}_val_balanced.csv"

    try:
        # 检查文件是否存在
        if not Path(manifest_path).exists():
            print(f"\n✗ {manifest_name}: 文件不存在 ({manifest_path})")
            return None

        # 创建数据集
        ds = CelebDFDataset(
            manifest_path=manifest_path,
            root_path=".",
            image_size=256,
            augmentation=False,
            normalize=True
        )

        # 创建数据加载器（num_workers=0 避免权限问题）
        # GPU 使用更大的 batch_size 以提高吞吐量
        batch_size = 256 if torch.cuda.is_available() else 128
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # 评估模型
        evaluator = ModelEvaluator(device=device, threshold=0.70)
        metrics = evaluator.evaluate_model(model, loader, mode=f"{manifest_name}_val")

        # 提取关键指标
        result = {
            "dataset": manifest_name,
            "auc": round(metrics.get("auc", 0.0), 4),
            "f1": round(metrics.get("f1", 0.0), 4),
            "precision": round(metrics.get("precision", 0.0), 4),
            "recall": round(metrics.get("recall", 0.0), 4),
            "accuracy": round(metrics.get("accuracy", 0.0), 4),
            "specificity": round(metrics.get("specificity", 0.0), 4),
            "samples": len(ds)
        }

        return result

    except Exception as e:
        print(f"\n✗ {manifest_name}: 评估失败")
        print(f"  错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 评估所有数据集
print("\n" + "-" * 80)
print("开始评估各验证集（这会花费 10-15 分钟）...")
print("-" * 80)

results = []
dataset_names = ["celebdf_v2", "faceforensics", "deeperforensics", "dfdc"]

for ds_name in dataset_names:
    print(f"\n📊 评估 {ds_name}...")
    result = eval_manifest(ds_name)
    if result:
        results.append(result)
        print(f"   ✓ 完成")
        print(f"     AUC={result['auc']}, F1={result['f1']}, "
              f"Recall={result['recall']}, Precision={result['precision']}")

# 汇总结果
print("\n" + "=" * 80)
print("【数据集拆分评估结果汇总】")
print("=" * 80)

if results:
    # 创建表格
    import pandas as pd
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))

    # 分析最佳和最弱性能
    print("\n" + "-" * 80)
    print("【性能分析】")
    print("-" * 80)

    best_auc = df_results.loc[df_results['auc'].idxmax()]
    worst_auc = df_results.loc[df_results['auc'].idxmin()]
    best_f1 = df_results.loc[df_results['f1'].idxmax()]
    worst_f1 = df_results.loc[df_results['f1'].idxmin()]

    print(f"\n最佳 AUC: {best_auc['dataset']:20s} ({best_auc['auc']:.4f})")
    print(f"最弱 AUC: {worst_auc['dataset']:20s} ({worst_auc['auc']:.4f})")
    print(f"AUC 方差: {df_results['auc'].std():.4f}")

    print(f"\n最佳 F1:  {best_f1['dataset']:20s} ({best_f1['f1']:.4f})")
    print(f"最弱 F1:  {worst_f1['dataset']:20s} ({worst_f1['f1']:.4f})")

    # 识别最弱的数据集
    print(f"\n⚠️ 最弱数据集分析:")
    print(f"   数据集: {worst_auc['dataset']}")
    print(f"   AUC: {worst_auc['auc']:.4f} (基线全集: 0.8431)")
    print(f"   F1: {worst_auc['f1']:.4f}")
    print(f"   Recall: {worst_auc['recall']:.4f}")
    print(f"   样本数: {worst_auc['samples']}")

    # 保存结果到 JSON
    results_file = Path("outputs/stage3/run_20251029_043420") / "per_dataset_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump({
            "threshold": 0.70,
            "results": results,
            "summary": {
                "best_auc": best_auc.to_dict(),
                "worst_auc": worst_auc.to_dict(),
                "best_f1": best_f1.to_dict(),
                "worst_f1": worst_f1.to_dict(),
                "auc_variance": float(df_results['auc'].std()),
                "f1_variance": float(df_results['f1'].std())
            }
        }, f, indent=2)
    print(f"\n✓ 结果已保存到: {results_file}")
else:
    print("❌ 评估失败：无法加载任何数据集")

print("\n" + "=" * 80)
