
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_ROOT = Path(
    r"C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\lung_xray_clean_512"
)

TEST_DIR = DATA_ROOT / "test"

MODEL_PATH = Path(
    r"C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\chest_xray_lung\lung_cnn_clean_output\lung_resnet18_clean_best.pth"
)

# Save heatmaps inside the CancerScan_AI project
OUTPUT_DIR = Path("results/lung")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16

# --------------------------------------------------
# IMAGE TRANSFORMATIONS
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# LOAD TEST DATASET
# --------------------------------------------------

if not TEST_DIR.is_dir():
    raise FileNotFoundError(
        f"Test dataset directory not found: {TEST_DIR}"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model checkpoint not found: {MODEL_PATH}"
    )

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=transform
)

if len(test_dataset) == 0:
    raise ValueError("The test dataset contains no images.")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes
num_classes = len(class_names)

if num_classes != 2:
    raise ValueError(
        f"Expected 2 classes, but found {num_classes}: {class_names}"
    )

print("=" * 65)
print("LUNG CANCER MODEL - TEST EVALUATION")
print("=" * 65)
print(f"Device: {DEVICE}")
print(f"Classes: {class_names}")
print(f"Test images: {len(test_dataset)}")

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Support the checkpoint format used by this project
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("Trained model loaded successfully.")

# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

accuracy = accuracy_score(all_labels, all_predictions)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# Preserve the original script's class-0 metrics
precision = precision_score(
    all_labels,
    all_predictions,
    labels=[0],
    average="micro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    labels=[0],
    average="micro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    labels=[0],
    average="micro",
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)

print("\nConfusion Matrix:")
print(cm)

print("=" * 65)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(
    f"{class_names[0]} Precision: "
    f"{precision:.4f} ({precision * 100:.2f}%)"
)
print(
    f"{class_names[0]} Recall: "
    f"{recall:.4f} ({recall * 100:.2f}%)"
)
print(
    f"{class_names[0]} F1-score: "
    f"{f1:.4f} ({f1 * 100:.2f}%)"
)
print("=" * 65)

# --------------------------------------------------
# HEATMAP 1: RAW CONFUSION MATRIX
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor="white",
    cbar=True
)

plt.title("Lung Cancer - Confusion Matrix (Test Set)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

count_path = OUTPUT_DIR / "lung_confusion_matrix_counts.png"

plt.savefig(
    count_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# HEATMAP 2: NORMALIZED CONFUSION MATRIX
# --------------------------------------------------

cm_normalized = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    normalize="true"
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2%",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor="white",
    cbar=True
)

plt.title("Lung Cancer - Normalized Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

percentage_path = OUTPUT_DIR / "lung_confusion_matrix_percentages.png"

plt.savefig(
    percentage_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# SAVE PREDICTIONS FOR FUTURE ANALYSIS
# --------------------------------------------------

predictions_path = OUTPUT_DIR / "lung_predictions.csv"

pd_available = True
try:
    import pandas as pd
except ImportError:
    pd_available = False

if pd_available:
    import numpy as np

    prediction_df = pd.DataFrame({
        "y_true": [class_names[i] for i in all_labels],
        "y_pred": [class_names[i] for i in all_predictions]
    })

    prediction_df.to_csv(predictions_path, index=False)
    print(f"Predictions CSV: {predictions_path.resolve()}")

print("\nHEATMAPS GENERATED SUCCESSFULLY")
print(f"Raw counts: {count_path.resolve()}")
print(f"Percentages: {percentage_path.resolve()}")
print("=" * 65)
