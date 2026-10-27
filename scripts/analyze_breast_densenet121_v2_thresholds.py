from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
)

ROOT = Path.cwd()
INPUT = ROOT / "reports" / "breast_densenet121_v1_v2_validation_predictions.csv"
OUT_CSV = ROOT / "reports" / "breast_densenet121_v2_threshold_analysis.csv"
OUT_JSON = ROOT / "reports" / "breast_densenet121_v2_threshold_recommendations.json"

if not INPUT.exists():
    raise FileNotFoundError(f"Prediction file not found: {INPUT}")
for p in [OUT_CSV, OUT_JSON]:
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {p}")

df = pd.read_csv(INPUT)
prob_col = "DenseNet121_v2_malignant_probability"
if "actual_label" not in df.columns or prob_col not in df.columns:
    raise ValueError(
        f"Required columns not found. Available columns: {list(df.columns)}"
    )

label_map = {"BENIGN": 0, "MALIGNANT": 1}
y = df["actual_label"].astype(str).str.upper().map(label_map)
if y.isna().any():
    raise ValueError("Unexpected actual_label values in prediction CSV.")
y = y.to_numpy(dtype=int)
p = df[prob_col].to_numpy(dtype=float)

if len(np.unique(y)) != 2:
    raise ValueError("Both classes must be present.")
if not np.isfinite(p).all():
    raise ValueError("Probability column contains invalid values.")

rows = []
for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rows.append({
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "malignant_precision": float(precision_score(y, pred, zero_division=0)),
        "malignant_recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "malignant_f1": float(f1_score(y, pred, zero_division=0)),
        "malignant_f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    })

results = pd.DataFrame(rows)
auc = float(roc_auc_score(y, p))
results.to_csv(OUT_CSV, index=False)

# For each constraint, maximize recall; ties prefer specificity and F2.
recommendations = {}
for minimum_specificity in [0.60, 0.65, 0.70, 0.75, 0.80]:
    eligible = results[results["specificity"] >= minimum_specificity]
    if eligible.empty:
        recommendations[str(minimum_specificity)] = {
            "status": "No threshold meets this specificity constraint"
        }
        continue
    best = eligible.sort_values(
        ["malignant_recall_sensitivity", "specificity", "malignant_f2"],
        ascending=[False, False, False]
    ).iloc[0]
    recommendations[str(minimum_specificity)] = {
        "minimum_specificity_constraint": minimum_specificity,
        "threshold": float(best["threshold"]),
        "malignant_recall_sensitivity": float(best["malignant_recall_sensitivity"]),
        "specificity": float(best["specificity"]),
        "malignant_precision": float(best["malignant_precision"]),
        "malignant_f2": float(best["malignant_f2"]),
        "TN": int(best["TN"]), "FP": int(best["FP"]),
        "FN": int(best["FN"]), "TP": int(best["TP"]),
    }

payload = {
    "model": "DenseNet121 v2",
    "data_source": str(INPUT.relative_to(ROOT)),
    "validation_images": int(len(y)),
    "validation_roc_auc": auc,
    "analysis": "Thresholds 0.10 to 0.90 in increments of 0.01.",
    "recommendations_by_minimum_specificity": recommendations,
    "warning": (
        "These are validation-only operating-point estimates. Choose a threshold "
        "before final test evaluation; do not use the test set to tune it."
    ),
}
OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("=" * 70)
print("DenseNet121 v2 - Validation Threshold Analysis")
print("=" * 70)
print("Validation images:", len(y))
print(f"ROC-AUC (threshold-independent): {auc:.4f}")
print("\nBest recall under each minimum-specificity constraint:")
for constraint, item in recommendations.items():
    print(f"\nSpecificity >= {float(constraint):.0%}")
    if "threshold" not in item:
        print(item["status"])
        continue
    print(
        f"Threshold={item['threshold']:.2f} | "
        f"Recall={item['malignant_recall_sensitivity']:.4f} | "
        f"Specificity={item['specificity']:.4f} | "
        f"Precision={item['malignant_precision']:.4f} | "
        f"F2={item['malignant_f2']:.4f}"
    )
    print(
        f"TN={item['TN']}, FP={item['FP']}, "
        f"FN={item['FN']}, TP={item['TP']}"
    )

print("\nFull threshold table:", OUT_CSV)
print("Recommendations:", OUT_JSON)
print("No model or checkpoint was changed; test set was not used.")
