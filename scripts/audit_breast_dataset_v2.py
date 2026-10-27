import pandas as pd
from pathlib import Path

train = pd.read_csv("datasets/train.csv")
val = pd.read_csv("datasets/val.csv")

print("=" * 65)
print("1. PATIENT OVERLAP")
print("=" * 65)

train_patients = set(train["patient_id"].dropna().astype(str))
val_patients = set(val["patient_id"].dropna().astype(str))
overlap = train_patients & val_patients

print("Train unique patients:", len(train_patients))
print("Validation unique patients:", len(val_patients))
print("Overlapping patients:", len(overlap))
if overlap:
    print("Overlap examples:", sorted(overlap)[:20])

print("\n" + "=" * 65)
print("2. DUPLICATE RECORDS")
print("=" * 65)

for name, df in [("TRAIN", train), ("VALIDATION", val)]:
    print(f"\n{name}")
    print("Duplicate complete rows:", df.duplicated().sum())
    print("Duplicate image_path rows:", df["image_path"].duplicated().sum())

    key_cols = [
        "patient_id", "breast", "abnormality_type",
        "pathology", "assessment", "subtlety", "label"
    ]
    print("Duplicate clinical keys:", df.duplicated(key_cols).sum())

print("\n" + "=" * 65)
print("3. IMAGE PATH CHECK")
print("=" * 65)

for name, df in [("TRAIN", train), ("VALIDATION", val)]:
    found = 0
    missing = []
    for raw in df["image_path"].dropna().astype(str):
        p = Path(raw.replace("\\", "/"))
        candidates = [p, Path(".") / p]

        # Also try the filename inside the project's JPEG directory.
        candidates.append(Path("datasets/jpeg") / p.name)

        if any(candidate.exists() for candidate in candidates):
            found += 1
        else:
            missing.append(raw)

    print(f"\n{name}:")
    print("Paths checked:", df["image_path"].notna().sum())
    print("Paths found by direct checks:", found)
    print("Paths not found:", len(missing))
    for item in missing[:3]:
        print("  Missing example:", item)

print("\n" + "=" * 65)
print("4. CC-MLO PAIRING AUDIT")
print("=" * 65)

pair_keys = [
    "patient_id", "breast", "abnormality_type",
    "pathology", "assessment", "subtlety", "label"
]

for name, df in [("TRAIN", train), ("VALIDATION", val)]:
    print(f"\n{name}")

    cc = df[df["view"].astype(str).str.upper().eq("CC")]
    mlo = df[df["view"].astype(str).str.upper().eq("MLO")]

    cc_keys = set(map(tuple, cc[pair_keys].fillna("<NA>").astype(str).to_numpy()))
    mlo_keys = set(map(tuple, mlo[pair_keys].fillna("<NA>").astype(str).to_numpy()))

    matched = cc_keys & mlo_keys
    print("CC rows:", len(cc))
    print("MLO rows:", len(mlo))
    print("Distinct CC matching keys:", len(cc_keys))
    print("Distinct MLO matching keys:", len(mlo_keys))
    print("Matching clinical keys:", len(matched))
    print("CC keys without MLO match:", len(cc_keys - mlo_keys))
    print("MLO keys without CC match:", len(mlo_keys - cc_keys))

print("\nAudit complete.")
