import pandas as pd
from pathlib import Path

root = Path(".")
files = ["datasets/train.csv", "datasets/val.csv"]

for filename in files:
    path = root / filename
    print("\n" + "=" * 65)
    print("FILE:", path)

    if not path.exists():
        print("ERROR: File not found")
        continue

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    for col in ["patient_id", "view", "breast", "abnormality_type",
                "pathology", "assessment", "subtlety", "label"]:
        if col in df.columns:
            print(f"\n{col} distribution:")
            print(df[col].fillna("MISSING").astype(str).value_counts().head(15))

    if "image_path" in df.columns:
        print("\nSample image paths:")
        print(df["image_path"].head(5).to_string(index=False))

    if "patient_id" in df.columns:
        print("\nUnique patients:", df["patient_id"].nunique())

print("\nAudit preview complete.")
