import pandas as pd


INPUT_PATH = "mnt/data/CancerScan_AI/datasets/training_data.csv"
OUTPUT_PATH = "mnt/data/CancerScan_AI/datasets/training_data_clean.csv"


# Load dataset
df = pd.read_csv(INPUT_PATH)

print("Original rows:", len(df))

# Remove duplicate image paths


df = df.drop_duplicates(subset="image_path").copy()

print("After removing duplicate images:", len(df))


# --------------------------------------------------
# Create binary labels
#
# 0 = benign / non-malignant
# 1 = malignant
# --------------------------------------------------

df["label"] = df["pathology"].apply(
    lambda x: 1 if x == "MALIGNANT" else 0
)


# --------------------------------------------------
# Keep only useful columns
# --------------------------------------------------

df = df[
    [
        "patient_id",
        "image_path",
        "pathology",
        "label",
        "abnormality_type",
        "breast",
        "view",
        "assessment",
        "subtlety",
    ]
]


# --------------------------------------------------
# Print statistics
# --------------------------------------------------

print("\nFinal dataset:")
print(df["pathology"].value_counts())

print("\nBinary labels:")
print(df["label"].value_counts())

print("\nUnique patients:", df["patient_id"].nunique())

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved to:")
print(OUTPUT_PATH)
