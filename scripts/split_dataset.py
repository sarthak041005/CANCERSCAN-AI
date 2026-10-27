import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "mnt/data/CancerScan_AI/datasets/training_data_clean.csv"

TRAIN_PATH = "mnt/data/CancerScan_AI/datasets/train.csv"
VAL_PATH = "mnt/data/CancerScan_AI/datasets/val.csv"
TEST_PATH = "mnt/data/CancerScan_AI/datasets/test.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("Total images:", len(df))
print("Total patients:", df["patient_id"].nunique())


# --------------------------------------------------
# Create patient-level dataframe
# --------------------------------------------------

patients = df[
    ["patient_id", "label"]
].drop_duplicates("patient_id")


print("\nPatients by label:")
print(patients["label"].value_counts())


# --------------------------------------------------
# 70% TRAIN / 30% TEMP
# --------------------------------------------------

train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.30,
    random_state=42,
    stratify=patients["label"]
)


# --------------------------------------------------
# 15% VALIDATION / 15% TEST
# --------------------------------------------------

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42,
    stratify=temp_patients["label"]
)


# --------------------------------------------------
# Get patient IDs
# --------------------------------------------------

train_ids = set(train_patients["patient_id"])
val_ids = set(val_patients["patient_id"])
test_ids = set(test_patients["patient_id"])


# --------------------------------------------------
# Create image datasets
# --------------------------------------------------

train_df = df[df["patient_id"].isin(train_ids)].copy()
val_df = df[df["patient_id"].isin(val_ids)].copy()
test_df = df[df["patient_id"].isin(test_ids)].copy()


# --------------------------------------------------
# Save
# --------------------------------------------------

train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)


# --------------------------------------------------
# Print results
# --------------------------------------------------

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


# --------------------------------------------------
# Verify no patient leakage
# --------------------------------------------------

train_test_overlap = train_ids & test_ids
train_val_overlap = train_ids & val_ids
val_test_overlap = val_ids & test_ids


print("\nPatient leakage check:")
print("Train/Test overlap:", len(train_test_overlap))
print("Train/Val overlap:", len(train_val_overlap))
print("Val/Test overlap:", len(val_test_overlap))