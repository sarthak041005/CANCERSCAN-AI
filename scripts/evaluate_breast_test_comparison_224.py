from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
ROOT = Path.cwd()
CSV_PATH = ROOT / "datasets" / "test.csv"
IMAGE_ROOT = ROOT / "datasets" / "breast_preprocessed_224"

CHECKPOINTS = {
    "ResNet18": ROOT / "saved_models" / "breast_resnet18_224_baseline.pth",
    "DenseNet121": ROOT / "saved_models" / "breast_densenet121_224_best.pth",
}

REPORT_JSON = ROOT / "reports" / "breast_test_comparison_224.json"
REPORT_TXT = ROOT / "reports" / "breast_test_comparison_224.txt"
PREDICTIONS_CSV = ROOT / "reports" / "breast_test_predictions_224.csv"

BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

# --------------------------------------------------
# 2. Safety checks
# --------------------------------------------------
for path in [CSV_PATH, IMAGE_ROOT, *CHECKPOINTS.values()]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

for path in [REPORT_JSON, REPORT_TXT, PREDICTIONS_CSV]:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {path}\n"
            "Rename/archive the old report or edit the output filename in this script."
        )

(ROOT / "reports").mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CancerScan AI - Breast Cancer Final Test Evaluation")
print("=" * 65)
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# 3. Load test CSV and resolve image paths
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)

required_columns = {"image_path", "pathology"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        f"Missing CSV columns: {missing_columns}. "
        f"Available columns: {list(df.columns)}"
    )

df = df.copy()
df["pathology"] = df["pathology"].astype(str).str.strip().str.upper()
unknown_labels = sorted(set(df["pathology"]) - set(LABEL_MAP))
if unknown_labels:
    raise ValueError(f"Unrecognized pathology labels: {unknown_labels}")

def resolve_image_path(raw_path):
    normalized = str(raw_path).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Cannot resolve image path: {raw_path}")
    return IMAGE_ROOT / parts[-2] / parts[-1]

df["resolved_path"] = df["image_path"].apply(resolve_image_path)
df["label"] = df["pathology"].map(LABEL_MAP).astype(int)

missing_images = df.loc[
    ~df["resolved_path"].apply(lambda p: p.is_file()),
    "resolved_path"
]
if not missing_images.empty:
    raise FileNotFoundError(
        f"{len(missing_images)} test images are missing. Examples:\n"
        + "\n".join(str(p) for p in missing_images.head(10))
    )

if df["label"].nunique() != 2:
    raise ValueError("The test set must contain both benign and malignant cases.")

print("\nTest images:", len(df))
print("Label counts:")
print(df["label"].map({0: "BENIGN", 1: "MALIGNANT"}).value_counts())
print("All test image paths verified.")

# --------------------------------------------------
# 4. Dataset and evaluation preprocessing
# --------------------------------------------------
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class BreastTestDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["resolved_path"]) as image:
            image = image.convert("RGB")
            image = eval_transform(image)
        return image, int(row["label"]), str(row["image_path"])

test_loader = DataLoader(
    BreastTestDataset(df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# --------------------------------------------------
# 5. Build models matching the saved checkpoints
# --------------------------------------------------
def build_model(architecture):
    if architecture == "ResNet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 2),
        )
    elif architecture == "DenseNet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 2),
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model

def load_checkpoint(path, architecture):
    # These checkpoints were created by the user's own training scripts.
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")

    state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError(
            f"'model_state_dict' not found in {path}. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    saved_arch = checkpoint.get("architecture")
    if saved_arch and saved_arch.lower() != architecture.lower():
        raise ValueError(
            f"Architecture mismatch in {path}: {saved_arch} vs {architecture}"
        )

    model = build_model(architecture)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, checkpoint

# --------------------------------------------------
# 6. Evaluate metrics
# --------------------------------------------------
def evaluate_model(name, checkpoint_path):
    architecture = name
    model, checkpoint = load_checkpoint(checkpoint_path, architecture)

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []

    with torch.inference_mode():
        for images, labels, paths in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = (probabilities >= 0.5).long()

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_paths.extend(paths)

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_predictions, dtype=int)
    y_prob = np.asarray(all_probabilities, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    metrics = {
        "model": name,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "malignant_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "malignant_recall_sensitivity": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "benign_recall": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "specificity": specificity,
        "malignant_f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }

    predictions_df = pd.DataFrame({
        "image_path": all_paths,
        "actual_label": [
            CLASS_NAMES[i] for i in y_true
        ],
        f"{name}_predicted_label": [
            CLASS_NAMES[i] for i in y_pred
        ],
        f"{name}_malignant_probability": y_prob,
    })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, predictions_df

# --------------------------------------------------
# 7. Run both evaluations and save new reports
# --------------------------------------------------
results = {}
prediction_frames = []

for name, checkpoint_path in CHECKPOINTS.items():
    print(f"\nEvaluating {name}...")
    metrics, predictions = evaluate_model(name, checkpoint_path)
    results[name] = metrics
    prediction_frames.append(predictions)

    print(f"\n{name} results")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "malignant_precision",
        "malignant_recall_sensitivity",
        "specificity",
        "malignant_f1",
        "roc_auc",
    ]:
        print(f"{key}: {metrics[key]:.4f}")
    print("Confusion matrix:", metrics["confusion_matrix"])
    print(classification_report(
        df["label"].to_numpy(),
        predictions[f"{name}_predicted_label"].map(
            {"BENIGN": 0, "MALIGNANT": 1}
        ).to_numpy(),
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

# Combine model predictions row-by-row for direct comparison.
combined_predictions = prediction_frames[0].copy()
for frame in prediction_frames[1:]:
    for column in frame.columns:
        if column != "image_path":
            combined_predictions[column] = frame[column].values

combined_predictions.to_csv(PREDICTIONS_CSV, index=False)

report = {
    "title": "CancerScan AI Breast Cancer Test-Set Comparison",
    "image_size": "224x224",
    "positive_class": "MALIGNANT",
    "negative_class": "BENIGN (BENIGN and BENIGN_WITHOUT_CALLBACK)",
    "test_csv": str(CSV_PATH.relative_to(ROOT)),
    "test_images": int(len(df)),
    "device": str(DEVICE),
    "warning": (
        "Test-set results are for final reporting. Do not tune models or "
        "thresholds based on these results; use validation data for model selection."
    ),
    "results": results,
}

REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

lines = [
    "CancerScan AI - Breast Cancer Test-Set Comparison",
    "=" * 58,
    f"Test images: {len(df)}",
    f"Device: {DEVICE}",
    "",
]
for name, metrics in results.items():
    lines.extend([
        name,
        "-" * 35,
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"Malignant precision: {metrics['malignant_precision']:.4f}",
        f"Malignant recall/sensitivity: {metrics['malignant_recall_sensitivity']:.4f}",
        f"Specificity: {metrics['specificity']:.4f}",
        f"Malignant F1: {metrics['malignant_f1']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"Confusion matrix (TN, FP, FN, TP): {metrics['confusion_matrix']}",
        "",
        "Classification report:",
        json.dumps(metrics["classification_report"], indent=2),
        "",
    ])
lines.append(
    "Important: Do not tune models or thresholds using this final test set."
)
REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)
print("JSON report:", REPORT_JSON)
print("Text report:", REPORT_TXT)
print("Predictions CSV:", PREDICTIONS_CSV)
print("Existing model checkpoints were not modified.")
