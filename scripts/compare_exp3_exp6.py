from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

reports = Path("reports")

exp3_file = reports / "breast_resnet50_ivaf_exp3_layer4_scheduler_predictions.csv"
exp6_file = reports / "breast_resnet50_ivaf_exp6_simclr_ivaf_predictions.csv"

exp3 = pd.read_csv(exp3_file)
exp6 = pd.read_csv(exp6_file)

# Identify each case using patient and paired image paths.
key_cols = ["patient_id", "cc_path", "mlo_path", "y_true"]
required = key_cols + ["malignant_probability"]

for name, df in [("Exp3", exp3), ("Exp6", exp6)]:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")

    if df.duplicated(key_cols).any():
        raise ValueError(f"{name} has duplicate case keys; stopping.")

# Merge one-to-one to ensure a fair, case-matched comparison.
merged = exp3[key_cols + ["malignant_probability"]].merge(
    exp6[key_cols + ["malignant_probability"]],
    on=key_cols,
    how="outer",
    suffixes=("_exp3", "_exp6"),
    indicator=True,
    validate="one_to_one"
)

print("\n=== CASE MATCHING AUDIT ===")
print("Exp3 cases:", len(exp3))
print("Exp6 cases:", len(exp6))
print("Matched cases:", int((merged["_merge"] == "both").sum()))
print("Exp3-only cases:", int((merged["_merge"] == "left_only").sum()))
print("Exp6-only cases:", int((merged["_merge"] == "right_only").sum()))

matched = merged[merged["_merge"] == "both"].copy()

if len(matched) == 0:
    raise ValueError("No matched cases found; cannot compare models.")

if len(matched) != len(exp3) or len(matched) != len(exp6):
    print("\nWARNING: CSVs do not contain exactly the same cases.")
    print("Metrics below use matched cases only.")

y = matched["y_true"].astype(int).to_numpy()
p3 = matched["malignant_probability_exp3"].astype(float).to_numpy()
p6 = matched["malignant_probability_exp6"].astype(float).to_numpy()

if not (np.isfinite(p3).all() and np.isfinite(p6).all()):
    raise ValueError("Non-finite probabilities detected.")
if not (np.isin(y, [0, 1]).all()):
    raise ValueError("Expected labels 0=Benign and 1=Malignant.")
if not ((p3 >= 0).all() and (p3 <= 1).all()
        and (p6 >= 0).all() and (p6 <= 1).all()):
    raise ValueError("Probabilities must be between 0 and 1.")

def evaluate(name, y_true, probs):
    pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "cases": len(y_true),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "malignant_precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "malignant_recall": recall_score(
            y_true, pred, zero_division=0
        ),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "malignant_f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }

results = pd.DataFrame([
    evaluate("Exp3", y, p3),
    evaluate("Exp6_SimCLR", y, p6),
])

out_csv = reports / "breast_exp3_vs_exp6_matched_metrics.csv"
out_json = reports / "breast_exp3_vs_exp6_matched_metrics.json"
out_preds = reports / "breast_exp3_vs_exp6_matched_predictions.csv"

results.to_csv(out_csv, index=False)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "threshold": 0.5,
        "label_mapping": {"0": "BENIGN", "1": "MALIGNANT"},
        "matched_cases": len(matched),
        "metrics": results.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "note": "Validation comparison only; no test set used."
    }, f, indent=2)

matched["exp3_probability"] = p3
matched["exp6_probability"] = p6
matched["exp3_prediction"] = (p3 >= 0.5).astype(int)
matched["exp6_prediction"] = (p6 >= 0.5).astype(int)
matched["predictions_agree"] = (
    matched["exp3_prediction"] == matched["exp6_prediction"]
)
matched.to_csv(out_preds, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

print("\n=== MATCHED VALIDATION METRICS (threshold=0.50) ===")
print(results.round(4).to_string(index=False))

print("\nPrediction agreement:",
      f"{matched['predictions_agree'].mean():.2%}")
print("\nSaved:")
print(out_csv.resolve())
print(out_json.resolve())
print(out_preds.resolve())
