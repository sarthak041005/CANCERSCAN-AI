import pandas as pd
from pathlib import Path

p = Path("reports/breast_resnet50_ivaf_history.csv")

if not p.exists():
    print("History file not found:", p)
    raise SystemExit(1)

df = pd.read_csv(p)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- FIRST 5 EPOCHS ---")
print(df.head().to_string(index=False))

print("\n--- LAST 10 EPOCHS ---")
print(df.tail(10).to_string(index=False))

print("\n--- BEST VALIDATION VALUES ---")
for col in df.columns:
    if any(k in col.lower() for k in ["val_loss", "val_auc", "val_accuracy", "val_acc"]):
        try:
            idx = df[col].idxmin() if "loss" in col.lower() else df[col].idxmax()
            print(f"{col}: {df.loc[idx, col]} (row {idx})")
        except (TypeError, ValueError):
            pass

print("\n--- HISTORY SHAPE ---")
print("Rows:", len(df))
print("Epochs recorded:", df.shape[0])
