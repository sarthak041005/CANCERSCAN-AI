from pathlib import Path
import re
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV_DIR = BASE / "datasets" / "CBIS-DDSM"
JPEG_DIR = BASE / "datasets" / "jpeg"

FILES = [
    "calc_case_description_train_set.csv",
    "mass_case_description_train_set.csv",
    "meta.csv",
    "dicom_info.csv",
]

print("=" * 70)
print("BREAST DATASET FINAL VERIFICATION")
print("=" * 70)

# ------------------------------------------------------------
# Load all pathology-labelled metadata
# ------------------------------------------------------------

frames = []

for filename in FILES:
    path = CSV_DIR / filename

    df = pd.read_csv(path)

    df = df[
        ["patient_id", "pathology", "image file path"]
    ].copy()

    df["source"] = filename

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

data = data.dropna(
    subset=[
        "patient_id",
        "pathology",
        "image file path",
    ]
)

# ------------------------------------------------------------
# Normalize pathology labels
# ------------------------------------------------------------

data["label"] = data["pathology"].replace(
    {
        "BENIGN_WITHOUT_CALLBACK": "BENIGN"
    }
)

# ------------------------------------------------------------
# Find patients with conflicting labels
# ------------------------------------------------------------

patient_labels = (
    data.groupby("patient_id")["label"]
    .nunique()
)

conflicting_patients = set(
    patient_labels[
        patient_labels > 1
    ].index
)

print()
print("Conflicting patients:", len(conflicting_patients))

# Remove conflicting patients completely

data = data[
    ~data["patient_id"].isin(conflicting_patients)
].copy()

# Remove duplicate metadata references

data = data.drop_duplicates(
    subset=["image file path"]
).reset_index(drop=True)

print("Clean labelled records:", len(data))

# ------------------------------------------------------------
# Extract CBIS-DDSM UID
# ------------------------------------------------------------

UID_PATTERN = (
    r"1\.3\.6\.1\.4\.1\.9590\.100\.1\.2\.\d+"
)


def extract_uids(path_string):
    return re.findall(
        UID_PATTERN,
        str(path_string)
    )


# ------------------------------------------------------------
# Index JPEG files using the SAME logic as the working script
# ------------------------------------------------------------

print()
print("Indexing JPEG files...")

jpeg_files = list(
    JPEG_DIR.rglob("*.jpg")
)

print("JPEG files found:", len(jpeg_files))

jpeg_index = {}

for image in jpeg_files:

    relative_path = str(
        image.relative_to(JPEG_DIR)
    )

    uids = extract_uids(relative_path)

    for uid in uids:

        jpeg_index.setdefault(
            uid,
            []
        ).append(image)

print(
    "Unique UID keys indexed:",
    len(jpeg_index)
)

# ------------------------------------------------------------
# Match every labelled record
# ------------------------------------------------------------

matched_paths = set()
unmatched = 0
ambiguous = 0

for _, row in data.iterrows():

    uids = extract_uids(
        row["image file path"]
    )

    if not uids:
        unmatched += 1
        continue

    target_uid = uids[-1]

    candidates = jpeg_index.get(
        target_uid,
        []
    )

    if len(candidates) == 1:

        matched_paths.add(
            str(candidates[0])
        )

    elif len(candidates) > 1:

        found = None

        for candidate in candidates:

            candidate_text = str(
                candidate
            )

            if all(
                uid in candidate_text
                for uid in uids
            ):
                found = candidate
                break

        if found is not None:

            matched_paths.add(
                str(found)
            )

        else:

            ambiguous += 1

    else:

        unmatched += 1

# ------------------------------------------------------------
# Final results
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    "Clean labelled records:",
    len(data)
)

print(
    "Unique matched JPEG images:",
    len(matched_paths)
)

print(
    "Unmatched records:",
    unmatched
)

print(
    "Ambiguous records:",
    ambiguous
)

print()
print("LABEL COUNTS")
print("-" * 40)

print(
    data["label"].value_counts()
)

print()
print("UNIQUE PATIENTS")
print("-" * 40)

print(
    data["patient_id"].nunique()
)

print()
print("=" * 70)