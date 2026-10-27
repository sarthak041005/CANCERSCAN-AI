from pathlib import Path
import pandas as pd

files = {
    "train": Path("datasets/train.csv"),
    "val": Path("datasets/val.csv"),
    "test": Path("datasets/test.csv"),
}

data = {}

for name, path in files.items():
    print("\n" + "=" * 55)
    print(name.upper(), "-", path)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)
    data[name] = df

    print("Rows:", len(df))
    print("Unique patients:", df["patient_id"].nunique())
    print("Labels:")
    print(df["label"].value_counts(dropna=False).sort_index().to_dict())

    print("Views:")
    print(df["view"].value_counts(dropna=False).to_dict())

print("\n" + "=" * 55)
print("PATIENT OVERLAP CHECK")

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    if a not in data or b not in data:
        continue

    patients_a = set(data[a]["patient_id"].dropna().astype(str))
    patients_b = set(data[b]["patient_id"].dropna().astype(str))
    overlap = patients_a & patients_b

    print(f"{a} vs {b}: {len(overlap)} overlapping patients")
    if overlap:
        print("Examples:", sorted(overlap)[:10])

print("\nAudit complete. No files modified.")
