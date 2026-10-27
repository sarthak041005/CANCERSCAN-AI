import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "mnt/data/CancerScan_AI/datasets/roi_dataset.csv"

OUTPUT_DIR = "mnt/data/CancerScan_AI/datasets"

TRAIN_CSV = os.path.join(OUTPUT_DIR, "roi_train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "roi_val.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "roi_test.csv")

SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Total ROI images:", len(df))
print("Total patients:", df["patient_id"].nunique())


# ============================================================
# CREATE PATIENT-LEVEL TABLE
# ============================================================

patient_labels = (
    df.groupby("patient_id")["label"]
    .max()
    .reset_index()
)


print("\nPatients by label:")
print(
    patient_labels["label"].value_counts()
)


# ============================================================
# TRAIN / TEMP SPLIT
# ============================================================

train_patients, temp_patients = train_test_split(
    patient_labels,
    test_size=0.30,
    random_state=SEED,
    stratify=patient_labels["label"]
)


# ============================================================
# VALIDATION / TEST SPLIT
# ============================================================

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_patients["label"]
)


# ============================================================
# PATIENT ID SETS
# ============================================================

train_ids = set(train_patients["patient_id"])
val_ids = set(val_patients["patient_id"])
test_ids = set(test_patients["patient_id"])


# ============================================================
# CREATE IMAGE SPLITS
# ============================================================

train_df = df[
    df["patient_id"].isin(train_ids)
].copy()

val_df = df[
    df["patient_id"].isin(val_ids)
].copy()

test_df = df[
    df["patient_id"].isin(test_ids)
].copy()


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_CSV,
    index=False
)

val_df.to_csv(
    VAL_CSV,
    index=False
)

test_df.to_csv(
    TEST_CSV,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\nTRAIN")
print("Images:", len(train_df))
print("Patients:", train_df["patient_id"].nunique())
print(train_df["label"].value_counts())


print("\nVALIDATION")
print("Images:", len(val_df))
print("Patients:", val_df["patient_id"].nunique())
print(val_df["label"].value_counts())


print("\nTEST")
print("Images:", len(test_df))
print("Patients:", test_df["patient_id"].nunique())
print(test_df["label"].value_counts())


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\nPatient leakage check:")

print(
    "Train/Val overlap:",
    len(train_ids & val_ids)
)

print(
    "Train/Test overlap:",
    len(train_ids & test_ids)
)

print(
    "Val/Test overlap:",
    len(val_ids & test_ids)
)


# ============================================================
# TOTAL CHECK
# ============================================================

print("\nTotal split images:")

print(
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print("Original images:", len(df))
