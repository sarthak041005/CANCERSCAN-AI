import os
import glob
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DICOM_CSV = "mnt/data/CancerScan_AI/datasets/CBIS-DDSM/dicom_info.csv"
JPEG_ROOT = "mnt/data/CancerScan_AI/datasets/jpeg"
OUTPUT_CSV = "mnt/data/CancerScan_AI/datasets/roi_dataset.csv"


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(DICOM_CSV)

print("Original metadata rows:", len(df))


# ============================================================
# FIND JPEG FILES
# ============================================================

jpeg_files = (
    glob.glob(
        os.path.join(JPEG_ROOT, "**", "*.jpg"),
        recursive=True
    )
    +
    glob.glob(
        os.path.join(JPEG_ROOT, "**", "*.jpeg"),
        recursive=True
    )
)

print("JPEG files:", len(jpeg_files))


# ============================================================
# CREATE UID → JPEG PATH MAPPING
# ============================================================

uid_to_jpeg = {}

for file_path in jpeg_files:

    parent_uid = os.path.basename(
        os.path.dirname(file_path)
    )

    # Keep first occurrence
    if parent_uid not in uid_to_jpeg:
        uid_to_jpeg[parent_uid] = file_path


print("Unique JPEG UIDs:", len(uid_to_jpeg))


# ============================================================
# FIND CROPPED IMAGE
# ============================================================

def find_cropped_image(cropped_path):

    if pd.isna(cropped_path):
        return None

    cropped_path = str(cropped_path)

    # Extract all UID-like directory names
    parts = cropped_path.replace("\\", "/").split("/")

    # Search from the deepest directory backwards
    for part in reversed(parts):

        if part in uid_to_jpeg:
            return uid_to_jpeg[part]

    return None


df["image_path"] = df["cropped image file path"].apply(
    find_cropped_image
)


# ============================================================
# REMOVE UNMATCHED IMAGES
# ============================================================

before = len(df)

df = df[df["image_path"].notna()].copy()

print(
    "Matched cropped images:",
    len(df),
    "/",
    before
)


# ============================================================
# REMOVE DUPLICATE IMAGES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["image_path"]
).copy()

print(
    "After removing duplicate images:",
    len(df),
)


# ============================================================
# BINARY LABEL
# ============================================================

df["label"] = (
    df["pathology"] == "MALIGNANT"
).astype(int)


# ============================================================
# SELECT IMPORTANT COLUMNS
# ============================================================

output = df[
    [
        "patient_id",
        "image_path",
        "pathology",
        "label",
        "abnormality type",
        "left or right breast",
        "image view",
        "assessment",
        "subtlety",
    ]
].copy()


# Rename columns
output = output.rename(
    columns={
        "abnormality type": "abnormality_type",
        "left or right breast": "breast",
        "image view": "view",
    }
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_CSV),
    exist_ok=True
)

output.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\nROI DATASET")
print("==============================")

print("Total images:", len(output))

print(
    "Unique patients:",
    output["patient_id"].nunique()
)

print("\nPathology:")
print(
    output["pathology"].value_counts()
)

print("\nBinary labels:")
print(
    output["label"].value_counts()
)

print("\nAbnormality type:")
print(
    output["abnormality_type"].value_counts()
)

print("\nViews:")
print(
    output["view"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_CSV)