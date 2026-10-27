from pathlib import Path
import pandas as pd
import numpy as np

saved = Path("reports/breast_resnet50_ivaf_imagenet_predictions.csv")
fresh = Path("reports/breast_imagenet_comparison_predictions.csv")

for path in [saved, fresh]:
    print("\nFile:", path)
    print("Exists:", path.exists())
    if path.exists():
        df = pd.read_csv(path)
        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())

if not saved.exists() or not fresh.exists():
    raise SystemExit("A prediction file is missing; no files were modified.")

a = pd.read_csv(saved)
b = pd.read_csv(fresh)

keys = ["patient_id", "cc_path", "mlo_path", "y_true"]
required = keys + ["malignant_probability", "y_pred_threshold_050"]

if not all(c in a.columns for c in required):
    raise SystemExit("Saved predictions have an unexpected schema.")
if not all(c in b.columns for c in required):
    raise SystemExit("Fresh predictions have an unexpected schema.")

merged = a[required].merge(
    b[required],
    on=keys,
    how="outer",
    suffixes=("_saved", "_fresh"),
    indicator=True,
)

print("\n" + "=" * 60)
print("SAVED VS FRESH CORRECTED CHECKPOINT PREDICTIONS")
print("=" * 60)
print("Saved rows:", len(a))
print("Fresh rows:", len(b))
print("Matched rows:", (merged["_merge"] == "both").sum())
print("Rows missing from either file:", (merged["_merge"] != "both").sum())

matched = merged[merged["_merge"] == "both"].copy()

if len(matched):
    matched["probability_difference"] = (
        matched["malignant_probability_saved"]
        - matched["malignant_probability_fresh"]
    ).abs()

    matched["prediction_changed"] = (
        matched["y_pred_threshold_050_saved"]
        != matched["y_pred_threshold_050_fresh"]
    )

    print("Changed class predictions:", int(matched["prediction_changed"].sum()))
    print(
        "Maximum probability difference:",
        float(matched["probability_difference"].max())
    )
    print(
        "Mean probability difference:",
        float(matched["probability_difference"].mean())
    )

    changed = matched[matched["prediction_changed"]]
    if len(changed):
        print("\nChanged predictions:")
        print(changed[
            keys + [
                "malignant_probability_saved",
                "malignant_probability_fresh",
                "y_pred_threshold_050_saved",
                "y_pred_threshold_050_fresh",
            ]
        ].to_string(index=False))

print("\nComparison complete. No files modified.")
