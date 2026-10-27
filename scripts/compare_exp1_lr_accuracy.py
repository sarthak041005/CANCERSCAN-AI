import json
from pathlib import Path

files = {
    "Baseline": Path("reports/breast_checkpoint_comparison.json"),
    "Experiment 1": Path(
        "reports/breast_resnet50_ivaf_exp1_lr_accuracy_validation.json"
    ),
}

for name, path in files.items():
    print(f"\n--- {name} ---")
    if not path.exists():
        print(f"File not found: {path}")
        continue

    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("validation_metrics_threshold_050", data)

    for key in (
        "accuracy", "balanced_accuracy", "malignant_precision",
        "malignant_recall", "specificity", "f1", "roc_auc", "loss"
    ):
        if key in metrics:
            value = metrics[key]
            print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")

    if "best_epoch" in data:
        print("best_epoch:", data["best_epoch"])
    if "best_stage" in data:
        print("best_stage:", data["best_stage"])
