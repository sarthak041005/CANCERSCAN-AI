from pathlib import Path
import pandas as pd
import shutil
import re

# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent

DATASETS = BASE / "mnt" / "data" / "CancerScan_AI" / "datasets"

JPEG_DIR = DATASETS / "jpeg"
OUTPUT_DIR = DATASETS / "breast"

TRAIN_CSV = DATASETS / "train.csv"
VAL_CSV = DATASETS / "val.csv"
TEST_CSV = DATASETS / "test.csv"


# ============================================================
# SETTINGS
# ============================================================

SPLITS = {
    "train": TRAIN_CSV,
    "val": VAL_CSV,
    "test": TEST_CSV,
}

LABELS = {
    "BENIGN": "BENIGN",
    "BENIGN_WITHOUT_CALLBACK": "BENIGN",
    "MALIGNANT": "MALIGNANT",
}


# ============================================================
# EXTRACT UID FROM IMAGE PATH
# ============================================================

def extract_uid(path_string):
    """
    Extract the CBIS-DDSM UID from an image path.

    Example:
    .../1.3.6.1.4.1.9590.100.1.2.406.../1-192.jpg

    returns:
    1.3.6.1.4.1.9590.100.1.2.406...
    """

    if pd.isna(path_string):
        return None

    text = str(path_string)

    match = re.search(
        r"1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+",
        text
    )

    if match:
        return match.group(0)

    return None


# ============================================================
# INDEX ALL JPEG FILES BY UID
# ============================================================

print("\n========================================")
print("INDEXING JPEG FILES")
print("========================================")

print(f"JPEG directory: {JPEG_DIR}")

jpeg_files = list(JPEG_DIR.rglob("*.jpg"))

print(f"JPEG files found: {len(jpeg_files)}")

if len(jpeg_files) == 0:
    raise SystemExit(
        "\nERROR: No JPEG files found."
    )


jpeg_index = {}

for image_path in jpeg_files:

    # UID is the parent directory name
    uid = image_path.parent.name

    # Only index CBIS-DDSM UID directories
    if uid.startswith("1.3.6.1.4.1.9590.100.1.2."):

        jpeg_index.setdefault(uid, []).append(image_path)


print(f"Unique UID directories indexed: {len(jpeg_index)}")


# ============================================================
# CLEAR PREVIOUS PREPARED DATASET
# ============================================================

print("\n========================================")
print("PREPARING OUTPUT DIRECTORIES")
print("========================================")

for split in SPLITS:

    for label in ["BENIGN", "MALIGNANT"]:

        folder = OUTPUT_DIR / split / label

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

summary = {}

for split, csv_path in SPLITS.items():

    print("\n========================================")
    print(f"PROCESSING: {split.upper()}")
    print("========================================")

    if not csv_path.exists():

        print(f"ERROR: CSV not found: {csv_path}")

        continue

    df = pd.read_csv(csv_path)

    print(f"CSV records: {len(df)}")

    successful = 0
    unmatched = 0
    ambiguous = 0

    label_counts = {
        "BENIGN": 0,
        "MALIGNANT": 0,
    }

    for index, row in df.iterrows():

        pathology = str(
            row["pathology"]
        ).strip().upper()

        label = LABELS.get(pathology)

        if label is None:

            print(
                f"Skipping unknown pathology: {pathology}"
            )

            continue

        uid = extract_uid(
            row["image_path"]
        )

        if uid is None:

            print(
                f"Could not extract UID: "
                f"{row['image_path']}"
            )

            unmatched += 1

            continue

        candidates = jpeg_index.get(uid, [])

        if len(candidates) == 0:

            print(
                f"JPEG not found for UID:\n{uid}"
            )

            unmatched += 1

            continue

        if len(candidates) > 1:

            print(
                f"WARNING: Multiple JPEGs for UID: "
                f"{uid}"
            )

            ambiguous += 1

        source = candidates[0]

        patient_id = str(
            row["patient_id"]
        )

        destination_dir = (
            OUTPUT_DIR /
            split /
            label
        )

        destination_name = (
            f"{patient_id}_"
            f"{index}_"
            f"{source.name}"
        )

        destination = (
            destination_dir /
            destination_name
        )

        try:

            shutil.copy2(
                source,
                destination
            )

            successful += 1
            label_counts[label] += 1

        except Exception as e:

            print(
                f"Failed to copy:\n"
                f"{source}\n"
                f"Reason: {e}"
            )

    summary[split] = {
        "csv": len(df),
        "copied": successful,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "benign": label_counts["BENIGN"],
        "malignant": label_counts["MALIGNANT"],
    }

    print("\nResult:")
    print(f"CSV records : {len(df)}")
    print(f"Copied      : {successful}")
    print(f"Unmatched   : {unmatched}")
    print(f"Ambiguous   : {ambiguous}")
    print(f"BENIGN      : {label_counts['BENIGN']}")
    print(f"MALIGNANT   : {label_counts['MALIGNANT']}")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n\n========================================")
print("FINAL DATASET REPORT")
print("========================================")

for split in SPLITS:

    result = summary.get(split)

    if result is None:
        continue

    print(
        f"\n{split.upper()}"
    )

    print(
        f"CSV records : {result['csv']}"
    )

    print(
        f"Copied      : {result['copied']}"
    )

    print(
        f"Unmatched   : {result['unmatched']}"
    )

    print(
        f"Ambiguous   : {result['ambiguous']}"
    )

    print(
        f"BENIGN      : {result['benign']}"
    )

    print(
        f"MALIGNANT   : {result['malignant']}"
    )


print("\n========================================")
print("DATASET LOCATION")
print("========================================")

print(OUTPUT_DIR)

print("\nPreparation finished.")