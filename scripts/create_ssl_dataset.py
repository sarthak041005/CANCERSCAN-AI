from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "mnt" / "data" / "CancerScan_AI" / "datasets"

META_CSV = DATASETS / "CBIS-DDSM" / "mass_case_description_test_set.csv"
TRAIN_CSV = DATASETS / "train.csv"
VAL_CSV = DATASETS / "val.csv"
TEST_CSV = DATASETS / "test.csv"

PREPROCESSED_DIR = DATASETS / "breast_preprocessed"
OUTPUT_CSV = DATASETS / "ssl_manifest.csv"

meta = pd.read_csv(META_CSV)
train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)
test = pd.read_csv(TEST_CSV)

train_patients = set(train["patient_id"].astype(str))
val_patients = set(val["patient_id"].astype(str))
test_patients = set(test["patient_id"].astype(str))

meta["patient_id"] = meta["PatientID"].astype(str).str.extract(
    r"(P_\d{5})",
    expand=False
)

meta["relative_path"] = (
    meta["image_path"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
    .str.replace("CBIS-DDSM/jpeg/", "", regex=False)
)

meta["preprocessed_path"] = meta["relative_path"].apply(
    lambda x: str(PREPROCESSED_DIR / Path(x))
)

meta["uid"] = meta["relative_path"].apply(
    lambda x: Path(x).parent.name
)

meta["split_group"] = meta["patient_id"].apply(
    lambda x: (
        "supervised_train"
        if x in train_patients
        else "validation"
        if x in val_patients
        else "test"
        if x in test_patients
        else "other"
    )
)

excluded_patients = val_patients | test_patients

ssl_data = meta[
    ~meta["patient_id"].isin(excluded_patients)
].copy()

duplicate_paths = ssl_data["preprocessed_path"].duplicated().sum()

if duplicate_paths > 0:
    raise RuntimeError(
        f"Found {duplicate_paths} duplicate image paths in the SSL dataset."
    )

missing_files = [
    path
    for path in ssl_data["preprocessed_path"]
    if not Path(path).exists()
]

if missing_files:
    print(f"Missing preprocessed images: {len(missing_files)}")
    for path in missing_files[:10]:
        print(path)
    raise FileNotFoundError("Some preprocessed images were not found.")

ssl_data = ssl_data[
    [
        "patient_id",
        "uid",
        "preprocessed_path",
        "split_group"
    ]
].copy()

ssl_data.to_csv(OUTPUT_CSV, index=False)

print("=" * 60)
print("SSL DATASET CREATED")
print("=" * 60)
print(f"Total metadata images: {len(meta)}")
print(f"SSL images: {len(ssl_data)}")
print(f"SSL patients: {ssl_data['patient_id'].nunique()}")
print()
print("Images by source group:")
print(ssl_data["split_group"].value_counts())
print()
print(f"Output file: {OUTPUT_CSV}")
print("=" * 60)