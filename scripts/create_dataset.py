import pandas as pd
import glob
import os


CSV_PATH = "mnt/data/CancerScan_AI/datasets/CBIS-DDSM/dicom_info.csv"
JPEG_ROOT = "mnt/data/CancerScan_AI/datasets/jpeg"
OUTPUT_PATH = "mnt/data/CancerScan_AI/datasets/training_data.csv"


# --------------------------------------------------
# 1. Load metadata
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("CSV rows:", len(df))


# --------------------------------------------------
# 2. Find all JPEG files
# --------------------------------------------------

jpeg_files = (
    glob.glob(os.path.join(JPEG_ROOT, "**", "*.jpg"), recursive=True)
    + glob.glob(os.path.join(JPEG_ROOT, "**", "*.jpeg"), recursive=True)
)

print("JPEG files:", len(jpeg_files))


# --------------------------------------------------
# 3. Create UID -> JPEG mapping
# --------------------------------------------------

jpeg_map = {}

for file_path in jpeg_files:
    uid = os.path.basename(os.path.dirname(file_path))

    # Keep first image if multiple JPEGs exist
    # for the same UID.
    if uid not in jpeg_map:
        jpeg_map[uid] = file_path


# --------------------------------------------------
# 4. Match CSV records to JPEGs
# --------------------------------------------------

records = []

for _, row in df.iterrows():

    image_path = str(row["image file path"])

    # Extract the UID immediately before 000000.dcm
    parts = image_path.replace("\\", "/").split("/")

    uid = None

    if len(parts) >= 2:
        uid = parts[-2]

    if uid in jpeg_map:

        records.append({
            "patient_id": row["patient_id"],
            "breast": row["left or right breast"],
            "view": row["image view"],
            "abnormality_type": row["abnormality type"],
            "assessment": row["assessment"],
            "pathology": row["pathology"],
            "subtlety": row["subtlety"],
            "image_path": jpeg_map[uid],
            "uid": uid
        })


# --------------------------------------------------
# 5. Create dataframe
# --------------------------------------------------

output_df = pd.DataFrame(records)

print("Matched images:", len(output_df))

print("\nPathology:")
print(output_df["pathology"].value_counts())


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

output_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:")
print(OUTPUT_PATH)
