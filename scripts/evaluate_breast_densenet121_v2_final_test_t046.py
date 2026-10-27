from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, classification_report
)

ROOT = Path.cwd()
CSV_PATH = ROOT / "datasets" / "test.csv"
IMAGE_ROOT = ROOT / "datasets" / "breast_preprocessed_224"
CHECKPOINT = ROOT / "saved_models" / "breast_densenet121_224_v2_best.pth"

OUT_JSON = ROOT / "reports" / "breast_densenet121_v2_final_test_t046.json"
OUT_TXT = ROOT / "reports" / "breast_densenet121_v2_final_test_t046.txt"
OUT_CSV = ROOT / "reports" / "breast_densenet121_v2_final_test_predictions_t046.csv"

THRESHOLD = 0.46
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

for path in [CSV_PATH, IMAGE_ROOT, CHECKPOINT]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

for path in [OUT_JSON, OUT_TXT, OUT_CSV]:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. "
            "Change output filenames in this script to rerun."
        )

df = pd.read_csv(CSV_PATH).copy()
if not {"image_path", "pathology"}.issubset(df.columns):
    raise ValueError(f"Required columns missing. Found: {list(df.columns)}")

df["pathology"] = df["pathology"].astype(str).str.strip().str.upper()
unknown = sorted(set(df["pathology"]) - set(LABEL_MAP))
if unknown:
    raise ValueError(f"Unknown pathology labels: {unknown}")

df["label"] = df["pathology"].map(LABEL_MAP).astype(int)

def resolve_path(raw):
    parts = [p for p in str(raw).replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot resolve image path: {raw}")
    return IMAGE_ROOT / parts[-2] / parts[-1]

df["resolved_path"] = df["image_path"].apply(resolve_path)
missing = df.loc[~df["resolved_path"].apply(lambda p: p.is_file()), "resolved_path"]
if len(missing):
    raise FileNotFoundError(
        f"{len(missing)} test images missing:\n" +
        "\n".join(map(str, missing.head(10)))
    )

if df["label"].nunique() != 2:
    raise ValueError("Test set must contain both classes.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class TestDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row["resolved_path"]) as img:
            x = transform(img.convert("RGB"))
        return x, int(row["label"]), str(row["image_path"])

loader = DataLoader(
    TestDataset(df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

model = models.densenet121(weights=None)
n_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(n_features, 2),
)

try:
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
except TypeError:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
    raise KeyError("Checkpoint does not contain model_state_dict.")

arch = checkpoint.get("architecture")
if arch and arch.lower() != "densenet121":
    raise ValueError(f"Unexpected architecture: {arch}")

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(DEVICE).eval()

y_true, probabilities, image_paths = [], [], []
with torch.inference_mode():
    for images, labels, paths in loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(labels.numpy().tolist())
        probabilities.extend(probs.cpu().numpy().tolist())
        image_paths.extend(paths)

y_true = np.asarray(y_true, dtype=int)
probs = np.asarray(probabilities, dtype=float)
y_pred = (probs >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

metrics = {
    "model": "DenseNet121 v2",
    "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
    "test_images": int(len(y_true)),
    "threshold": THRESHOLD,
    "positive_class": "MALIGNANT",
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    "malignant_precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "malignant_recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
    "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    "malignant_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "malignant_f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_true, probs)),
    "confusion_matrix": {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    },
    "classification_report": classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    ),
    "threshold_selection": (
        "Fixed at 0.46 using validation-only analysis before final test evaluation."
    ),
    "warning": (
        "This is a research evaluation on a limited held-out dataset, not "
        "clinical validation. Do not adjust the threshold based on these test results."
    ),
}

pred_df = pd.DataFrame({
    "image_path": image_paths,
    "actual_label": [CLASS_NAMES[i] for i in y_true],
    "predicted_label": [CLASS_NAMES[i] for i in y_pred],
    "malignant_probability": probs,
    "threshold": THRESHOLD,
})
pred_df.to_csv(OUT_CSV, index=False)

(ROOT / "reports").mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

lines = [
    "CancerScan AI - DenseNet121 v2 Final Test Evaluation",
    "=" * 60,
    f"Test images: {len(y_true)}",
    f"Threshold: {THRESHOLD}",
    f"Device: {DEVICE}",
    "",
]
for key in [
    "accuracy", "balanced_accuracy", "malignant_precision",
    "malignant_recall_sensitivity", "specificity",
    "malignant_f1", "malignant_f2", "roc_auc"
]:
    lines.append(f"{key}: {metrics[key]:.4f}")
lines += [
    "",
    f"Confusion matrix: {metrics['confusion_matrix']}",
    "",
    classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=CLASS_NAMES, zero_division=0
    ),
    "",
    metrics["warning"],
]
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("=" * 60)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 60)
print("Model: DenseNet121 v2")
print("Test images:", len(y_true))
print("Fixed threshold:", THRESHOLD)
for key in [
    "accuracy", "balanced_accuracy", "malignant_precision",
    "malignant_recall_sensitivity", "specificity",
    "malignant_f1", "malignant_f2", "roc_auc"
]:
    print(f"{key}: {metrics[key]:.4f}")
print("Confusion matrix:", metrics["confusion_matrix"])
print("\nClassification report:")
print(classification_report(
    y_true, y_pred, labels=[0, 1],
    target_names=CLASS_NAMES, zero_division=0
))
print("\nSaved:")
print(OUT_JSON)
print(OUT_TXT)
print(OUT_CSV)
print("\nCheckpoint unchanged. Do not tune the threshold using these test results.")
