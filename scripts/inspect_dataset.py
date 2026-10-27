import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


CSV = "mnt/data/CancerScan_AI/datasets/training_data_clean.csv"

df = pd.read_csv(CSV)

# Binary labels
malignant = df[df["label"] == 1].drop_duplicates("image_path")
non_malignant = df[df["label"] == 0].drop_duplicates("image_path")


print("Malignant images:", len(malignant))
print("Non-malignant images:", len(non_malignant))


# Select examples
malignant_samples = malignant.sample(
    min(6, len(malignant)),
    random_state=42
)

non_malignant_samples = non_malignant.sample(
    min(6, len(non_malignant)),
    random_state=42
)


fig, axes = plt.subplots(
    3,
    4,
    figsize=(16, 12)
)


samples = list(malignant_samples.itertuples()) + list(
    non_malignant_samples.itertuples()
)


for ax, row in zip(axes.flat, samples):

    image = Image.open(row.image_path)

    ax.imshow(
        image,
        cmap="gray"
    )

    label = "MALIGNANT" if row.label == 1 else "NON-MALIGNANT"

    ax.set_title(
        f"{label}\n"
        f"Patient: {row.patient_id}\n"
        f"Size: {image.size}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "mnt/data/CancerScan_AI/dataset_preview.png",
    dpi=150
)

plt.show()

print("\nSaved:")
print("mnt/data/CancerScan_AI/dataset_preview.png") 