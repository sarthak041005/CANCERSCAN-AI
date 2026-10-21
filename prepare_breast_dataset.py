from pathlib import Path
import pandas as pd
import shutil
import re
import random

# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

CSV_DIR = BASE / "datasets" / "CBIS-DDSM"
JPEG_DIR = BASE / "datasets" / "jpeg"
OUTPUT_DIR = BASE / "datasets" / "breast"

MASS_CSV = CSV_DIR / "mass_case_description_train_set.csv"
CALC_CSV = CSV_DIR / "calc_case_description_train_set.csv"

VAL_RATIO = 0.20
SEED = 42

random.seed(SEED)


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):
    label = str(label).strip().upper()

    if label in {"MALIGNANT"}:
        return "MALIGNANT"

    if label in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
        return "BENIGN"

    return None


# ============================================================
# EXTRACT UID NUMBERS FROM PATH
# ============================================================

def extract_uids(path_string):
    """
    Extract UID-like directory names from a CBIS-DDSM path.
    """

    if pd.isna(path_string):
        return []

    text = str(path_string)

    # UID folders in CBIS-DDSM look like:
    # 1.3.6.1.4.1.9590.100.1.2.xxxxxxxxx
    matches = re.findall(
        r"1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+",
        text
    )

    return matches


# ============================================================
# INDEX JPEG FILES
# ============================================================

print("\nIndexing JPEG files...")
print("This may take a little while...\n")

jpeg_files = list(JPEG_DIR.rglob("*.jpg"))

print(f"JPEG files found: {len(jpeg_files)}")

if len(jpeg_files) == 0:
    print("ERROR: No .jpg files found.")
    raise SystemExit(1)


# Create an index:
# UID -> list of JPEG paths

jpeg_index = {}

for img in jpeg_files:
    relative_path = str(img.relative_to(JPEG_DIR))

    uids = extract_uids(relative_path)

    for uid in uids:
        jpeg_index.setdefault(uid, []).append(img)


print(f"Unique UID keys indexed: {len(jpeg_index)}")


# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nReading CSV files...")

mass_df = pd.read_csv(MASS_CSV)
calc_df = pd.read_csv(CALC_CSV)

print(f"Mass rows: {len(mass_df)}")
print(f"Calcification rows: {len(calc_df)}")


# Keep common columns

mass_df = mass_df[
    [
        "patient_id",
        "pathology",
        "image file path",
        "cropped image file path",
        "ROI mask file path",
    ]
].copy()

calc_df = calc_df[
    [
        "patient_id",
        "pathology",
        "image file path",
        "cropped image file path",
        "ROI mask file path",
    ]
].copy()


# Combine

df = pd.concat([mass_df, calc_df], ignore_index=True)

# Normalize labels

df["label"] = df["pathology"].apply(normalize_label)

# Remove unknown labels

df = df[df["label"].notna()].copy()

print("\nLabels:")
print(df["label"].value_counts())


# ============================================================
# FIND JPEG FOR EACH CSV ROW
# ============================================================

def find_matching_jpeg(row):
    """
    Find the best JPEG corresponding to the CSV image path.

    We primarily match using UID values contained in the
    original CBIS-DDSM path.
    """

    # Prefer original image path
    image_path = row["image file path"]

    uids = extract_uids(image_path)

    if not uids:
        return None

    # Use the last UID because it is normally the most
    # specific image/series identifier.
    target_uid = uids[-1]

    candidates = jpeg_index.get(target_uid, [])

    if len(candidates) == 1:
        return candidates[0]

    # If multiple candidates exist, try matching more UIDs.
    if len(candidates) > 1:

        for candidate in candidates:

            candidate_text = str(candidate)

            if all(uid in candidate_text for uid in uids):
                return candidate

        return candidates[0] if candidates else None

    return None


print("\nMatching CSV records to JPEG files...")

df["jpeg_path"] = df.apply(find_matching_jpeg, axis=1)

matched = df["jpeg_path"].notna().sum()
unmatched = df["jpeg_path"].isna().sum()

print(f"Matched:   {matched}")
print(f"Unmatched: {unmatched}")

if matched == 0:
    print("\nERROR: No JPEG files could be matched.")
    print("We need to inspect the exact JPEG naming structure.")
    raise SystemExit(1)


# ============================================================
# REMOVE DUPLICATE IMAGES
# ============================================================

df = df[df["jpeg_path"].notna()].copy()

df["jpeg_path"] = df["jpeg_path"].astype(str)

before = len(df)

df = df.drop_duplicates(
    subset=["jpeg_path"]
).reset_index(drop=True)

after = len(df)

print(f"\nDuplicate images removed: {before - after}")
print(f"Final usable images: {after}")


# ============================================================
# PATIENT-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

patients = df["patient_id"].astype(str).unique()

random.shuffle(patients)

split_index = int(len(patients) * (1 - VAL_RATIO))

train_patients = set(patients[:split_index])
val_patients = set(patients[split_index:])


df["split"] = df["patient_id"].astype(str).apply(
    lambda x: "train" if x in train_patients else "val"
)


print("\nSplit:")
print(df["split"].value_counts())

print("\nTrain labels:")
print(df[df["split"] == "train"]["label"].value_counts())

print("\nValidation labels:")
print(df[df["split"] == "val"]["label"].value_counts())


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val"]:
    for label in ["BENIGN", "MALIGNANT"]:

        folder = OUTPUT_DIR / split / label

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY FILES
# ============================================================

print("\nCopying images...")

successful = 0

for index, row in df.iterrows():

    source = Path(row["jpeg_path"])

    split = row["split"]
    label = row["label"]

    destination_dir = OUTPUT_DIR / split / label

    destination_name = (
        f"{row['patient_id']}_"
        f"{index}_"
        f"{source.name}"
    )

    destination = destination_dir / destination_name

    try:

        shutil.copy2(
            source,
            destination
        )

        successful += 1

    except Exception as e:

        print(
            f"Failed: {source}\n"
            f"Reason: {e}"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n========================================")
print("DATASET PREPARATION COMPLETE")
print("========================================")

print(f"Successfully copied: {successful}")

for split in ["train", "val"]:

    for label in ["BENIGN", "MALIGNANT"]:

        folder = OUTPUT_DIR / split / label

        count = len(list(folder.glob("*.jpg")))

        print(
            f"{split:5s} | "
            f"{label:10s} | "
            f"{count} images"
        )

print("\nDataset location:")
print(OUTPUT_DIR)

print("\nYou can now run:")
print("python train.py --cancer breast")