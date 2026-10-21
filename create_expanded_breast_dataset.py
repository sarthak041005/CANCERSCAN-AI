from pathlib import Path
import re
import shutil
import random
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

CSV_DIR = BASE / "datasets" / "CBIS-DDSM"
JPEG_DIR = BASE / "datasets" / "jpeg"
OUTPUT_DIR = BASE / "datasets" / "breast_expanded"

VAL_RATIO = 0.20
SEED = 42


# ============================================================
# PATH COLUMNS
# ============================================================

PATH_COLUMNS = [
    "image file path",
    "file_path",
    "image_path",
]


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value == "BENIGN_WITHOUT_CALLBACK":
        return "BENIGN"

    if value == "BENIGN":
        return "BENIGN"

    if value == "MALIGNANT":
        return "MALIGNANT"

    return None


# ============================================================
# EXTRACT DICOM UID
# ============================================================

def extract_uids(path_text):
    if pd.isna(path_text):
        return []

    text = str(path_text)

    pattern = r"1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+"

    return re.findall(pattern, text)


# ============================================================
# LOAD PATHOLOGY DATA
# ============================================================

print("=" * 70)
print("CREATING EXPANDED BREAST DATASET")
print("=" * 70)

csv_files = [
    "calc_case_description_train_set.csv",
    "mass_case_description_train_set.csv",
    "meta.csv",
    "dicom_info.csv",
]

records = []

for filename in csv_files:

    csv_path = CSV_DIR / filename

    print(f"\nReading: {filename}")

    df = pd.read_csv(csv_path)

    print(f"Rows: {len(df)}")

    # Find pathology column
    pathology_column = None

    for column in df.columns:
        if str(column).strip().lower() == "pathology":
            pathology_column = column
            break

    if pathology_column is None:
        print("No pathology column found.")
        continue

    # Find image path column
    image_column = None

    for column in PATH_COLUMNS:
        if column in df.columns:
            image_column = column
            break

    if image_column is None:
        print("No image path column found.")
        continue

    for _, row in df.iterrows():

        label = normalize_label(row[pathology_column])

        if label is None:
            continue

        patient_id = None

        if "patient_id" in df.columns:
            patient_id = str(row["patient_id"]).strip()

        elif "PatientID" in df.columns:
            patient_id = str(row["PatientID"]).strip()

        if not patient_id or patient_id == "nan":
            continue

        image_path = str(row[image_column]).strip()

        if not image_path or image_path == "nan":
            continue

        records.append({
            "patient_id": patient_id,
            "label": label,
            "image_path": image_path,
            "source": filename,
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

print("\n" + "=" * 70)
print("RAW COMBINED DATA")
print("=" * 70)

print(f"Total records: {len(df)}")
print(f"Unique image paths: {df['image_path'].nunique()}")
print(f"Unique patients: {df['patient_id'].nunique()}")


# ============================================================
# FIND CONFLICTING PATIENTS
# ============================================================

patient_labels = (
    df.groupby("patient_id")["label"]
    .nunique()
)

conflicting_patients = set(
    patient_labels[patient_labels > 1].index
)

print(f"\nConflicting patients: {len(conflicting_patients)}")

# Remove all images belonging to conflicting patients
df = df[
    ~df["patient_id"].isin(conflicting_patients)
].copy()


# ============================================================
# REMOVE DUPLICATE IMAGE PATHS
# ============================================================

df = df.drop_duplicates(
    subset=["image_path"]
).copy()

print(f"Clean labelled records: {len(df)}")


# ============================================================
# INDEX JPEG FILES
# ============================================================

print("\nIndexing JPEG files...")

jpeg_files = list(
    JPEG_DIR.rglob("*.jpg")
)

print(f"JPEG files found: {len(jpeg_files)}")


jpeg_index = {}

uid_pattern = (
    r"1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+"
)

for image in jpeg_files:

    relative_path = str(
        image.relative_to(JPEG_DIR)
    )

    uids = re.findall(
        uid_pattern,
        relative_path
    )

    for uid in uids:

        if uid not in jpeg_index:
            jpeg_index[uid] = []

        jpeg_index[uid].append(image)


print(
    f"Unique UID keys indexed: "
    f"{len(jpeg_index)}"
)


# ============================================================
# MATCH CSV RECORDS TO JPEG FILES
# ============================================================

matched_records = []

unmatched = 0
ambiguous = 0

for _, row in df.iterrows():

    uids = extract_uids(
        row["image_path"]
    )

    if not uids:
        unmatched += 1
        continue

    target_uid = uids[-1]

    candidates = jpeg_index.get(
        target_uid,
        []
    )

    # If exactly one candidate exists
    if len(candidates) == 1:

        matched_records.append({
            "patient_id": row["patient_id"],
            "label": row["label"],
            "source_path": candidates[0],
        })

        continue

    # If multiple candidates exist,
    # try all UIDs from the metadata path
    found = []

    for uid in uids:

        for candidate in jpeg_index.get(
            uid,
            []
        ):
            found.append(candidate)

    found = list(dict.fromkeys(found))

    if len(found) == 1:

        matched_records.append({
            "patient_id": row["patient_id"],
            "label": row["label"],
            "source_path": found[0],
        })

    elif len(found) == 0:

        unmatched += 1

    else:

        ambiguous += 1


print("\n" + "=" * 70)
print("MATCHING RESULTS")
print("=" * 70)

print(
    f"Clean labelled records: {len(df)}"
)

print(
    f"Matched JPEG images: "
    f"{len(matched_records)}"
)

print(
    f"Unmatched records: {unmatched}"
)

print(
    f"Ambiguous records: {ambiguous}"
)


# ============================================================
# CREATE MATCHED DATAFRAME
# ============================================================

matched_df = pd.DataFrame(
    matched_records
)

if len(matched_df) == 0:

    raise RuntimeError(
        "No images were matched."
    )


# Remove any accidental duplicate JPEG
matched_df = matched_df.drop_duplicates(
    subset=["source_path"]
).copy()


print(
    f"Unique matched JPEG images: "
    f"{len(matched_df)}"
)


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

print("\nCreating patient-level split...")

patients = list(
    matched_df["patient_id"].unique()
)

random.seed(SEED)

random.shuffle(patients)

val_patient_count = round(
    len(patients) * VAL_RATIO
)

val_patients = set(
    patients[:val_patient_count]
)

train_patients = set(
    patients[val_patient_count:]
)


matched_df["split"] = matched_df[
    "patient_id"
].apply(
    lambda x:
        "val"
        if x in val_patients
        else "train"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

if OUTPUT_DIR.exists():

    print(
        "\nRemoving existing "
        "breast_expanded dataset..."
    )

    shutil.rmtree(OUTPUT_DIR)


for split in ["train", "val"]:

    for label in [
        "BENIGN",
        "MALIGNANT"
    ]:

        (
            OUTPUT_DIR
            / split
            / label
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY IMAGES
# ============================================================

print("\nCopying images...")

copied = 0

for _, row in matched_df.iterrows():

    source = Path(
        row["source_path"]
    )

    destination = (
        OUTPUT_DIR
        / row["split"]
        / row["label"]
        / source.name
    )

    # Avoid filename collision
    if destination.exists():

        stem = source.stem
        suffix = source.suffix

        counter = 1

        while destination.exists():

            destination = (
                OUTPUT_DIR
                / row["split"]
                / row["label"]
                / f"{stem}_{counter}{suffix}"
            )

            counter += 1

    shutil.copy2(
        source,
        destination
    )

    copied += 1

    if copied % 250 == 0:

        print(
            f"Copied: {copied}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"Total images: {len(matched_df)}"
)

print(
    f"Total patients: "
    f"{matched_df['patient_id'].nunique()}"
)

print("\nTRAIN")

train_df = matched_df[
    matched_df["split"] == "train"
]

print(
    f"Images: {len(train_df)}"
)

print(
    train_df["label"].value_counts()
)

print(
    f"Patients: "
    f"{train_df['patient_id'].nunique()}"
)

print("\nVALIDATION")

val_df = matched_df[
    matched_df["split"] == "val"
]

print(
    f"Images: {len(val_df)}"
)

print(
    val_df["label"].value_counts()
)

print(
    f"Patients: "
    f"{val_df['patient_id'].nunique()}"
)

print("\n" + "=" * 70)
print("DATASET CREATION COMPLETE")
print("=" * 70)

print(
    f"\nDataset location:\n{OUTPUT_DIR}"
)