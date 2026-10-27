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
IMAGE_ROOT = ROOT / "datasets" / "breast_preprocessed_224"
VAL_CSV = ROOT / "datasets" / "val.csv"

CHECKPOINTS = {
    "DenseNet121_v1": ROOT / "saved_models" / "breast_densenet121_224_best.pth",
    "DenseNet121_v2": ROOT / "saved_models" / "breast_densenet121_224_v2_best.pth",
}

OUT_JSON = ROOT / "reports" / "breast_densenet121_v1_v2_validation_comparison.json"
OUT_CSV = ROOT / "reports" / "breast_densenet121_v1_v2_validation_predictions.csv"
OUT_TXT = ROOT / "reports" / "breast_densenet121_v1_v2_validation_comparison.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
CLASS_NAMES = ["BENIGN", "MALIGNANT"]
LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}

# Refuse to overwrite any prior outputs.
for path in [VAL_CSV, IMAGE_ROOT, *CHECKPOINTS.values()]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

for path in [OUT_JSON, OUT_CSV, OUT_TXT]:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {path}. Change output filenames in this script."
        )

def resolve_image(raw_path):
    parts = [p for p in str(raw_path).replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot resolve image path: {raw_path}")
    return IMAGE_ROOT / parts[-2] / parts[-1]

df = pd.read_csv(VAL_CSV).copy()
if not {"image_path", "pathology"}.issubset(df.columns):
    raise ValueError(f"Expected image_path and pathology columns: {list(df.columns)}")

df["pathology"] = df["pathology"].astype(str).str.strip().str.upper()
unknown = sorted(set(df["pathology"]) - set(LABEL_MAP))
if unknown:
    raise ValueError(f"Unknown labels: {unknown}")

df["label"] = df["pathology"].map(LABEL_MAP).astype(int)
df["resolved_path"] = df["image_path"].apply(resolve_image)

missing = df.loc[~df["resolved_path"].apply(lambda p: p.is_file()), "resolved_path"]
if len(missing):
    raise FileNotFoundError(
        f"{len(missing)} validation images missing:\n" +
        "\n".join(map(str, missing.head(10)))
    )

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class ValidationDataset(Dataset):
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
    ValidationDataset(df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

def build_model():
    model = models.densenet121(weights=None)
    features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(features, 2),
    )
    return model

def load_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise KeyError(
            f"Expected model_state_dict in {path}; keys: "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    arch = ckpt.get("architecture")
    if arch and arch.lower() != "densenet121":
        raise ValueError(f"Unexpected architecture in {path}: {arch}")

    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model, ckpt

def metrics_at_threshold(y_true, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "malignant_precision": float(precision_score(y_true, pred, zero_division=0)),
        "malignant_recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "malignant_f1": float(f1_score(y_true, pred, zero_division=0)),
        "malignant_f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        },
        "classification_report": classification_report(
            y_true, pred, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        ),
    }

y_true = df["label"].to_numpy(dtype=int)
if len(np.unique(y_true)) != 2:
    raise ValueError("Validation split must contain both classes.")

results = {}
predictions_df = pd.DataFrame({
    "image_path": df["image_path"].tolist(),
    "actual_label": [CLASS_NAMES[y] for y in y_true],
})
rows_for_text = []

print("=" * 65)
print("DenseNet121 v1 vs v2 - SAME VALIDATION SET")
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Validation images:", len(df))
print("Label counts:", pd.Series(y_true).value_counts().sort_index().to_dict())
print("=" * 65)

for name, path in CHECKPOINTS.items():
    print(f"\nEvaluating {name}...")
    model, ckpt = load_checkpoint(path)
    probabilities = []

    with torch.inference_mode():
        for images, labels, paths in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probabilities.extend(
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            )

    p = np.asarray(probabilities, dtype=float)
    if len(p) != len(y_true):
        raise RuntimeError("Prediction count does not match validation count.")

    try:
        auc = float(roc_auc_score(y_true, p))
    except ValueError:
        auc = None

    # Compare both checkpoints at identical thresholds.
    thresholds = [0.50, 0.46]
    model_result = {
        "checkpoint": str(path.relative_to(ROOT)),
        "best_epoch_in_checkpoint": ckpt.get("best_epoch"),
        "roc_auc": auc,
        "metrics_by_threshold": {},
    }

    predictions_df[f"{name}_malignant_probability"] = p

    for threshold in thresholds:
        m = metrics_at_threshold(y_true, p, threshold)
        model_result["metrics_by_threshold"][f"{threshold:.2f}"] = m
        print(
            f"{name} | threshold={threshold:.2f} | "
            f"recall={m['malignant_recall']:.4f} | "
            f"specificity={m['specificity']:.4f} | "
            f"F2={m['malignant_f2']:.4f} | "
            f"FN={m['confusion_matrix']['FN']} | "
            f"FP={m['confusion_matrix']['FP']}"
        )
        rows_for_text.append({
            "model": name,
            "threshold": threshold,
            **{k: v for k, v in m.items()
               if k not in ["classification_report", "confusion_matrix"]},
            **m["confusion_matrix"],
        })

    results[name] = model_result
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

predictions_df.to_csv(OUT_CSV, index=False)

report = {
    "title": "DenseNet121 v1 vs v2 validation comparison",
    "validation_csv": str(VAL_CSV.relative_to(ROOT)),
    "validation_images": int(len(df)),
    "label_mapping": LABEL_MAP,
    "device": str(DEVICE),
    "results": results,
    "note": (
        "Both models evaluated on identical validation images with identical "
        "preprocessing. Thresholds are 0.50 and 0.46. Do not use the final test "
        "set to select the model or threshold."
    ),
}
OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
pd.DataFrame(rows_for_text).to_csv(
    ROOT / "reports" / "breast_densenet121_v1_v2_validation_metrics.csv",
    index=False
)

text_lines = [
    "DenseNet121 v1 vs v2 - Validation comparison",
    "=" * 55,
    f"Validation images: {len(df)}",
    "Same images and preprocessing for both models.",
    "",
]
for name, result in results.items():
    text_lines += [
        name,
        f"ROC-AUC: {result['roc_auc']}",
    ]
    for threshold, m in result["metrics_by_threshold"].items():
        text_lines += [
            f"Threshold: {threshold}",
            f"Accuracy: {m['accuracy']:.4f}",
            f"Balanced accuracy: {m['balanced_accuracy']:.4f}",
            f"Malignant precision: {m['malignant_precision']:.4f}",
            f"Malignant recall: {m['malignant_recall']:.4f}",
            f"Specificity: {m['specificity']:.4f}",
            f"Malignant F1: {m['malignant_f1']:.4f}",
            f"Malignant F2: {m['malignant_f2']:.4f}",
            f"Confusion matrix: {m['confusion_matrix']}",
            "",
        ]
OUT_TXT.write_text("\n".join(text_lines), encoding="utf-8")

print("\nCOMPARISON COMPLETED")
print("JSON:", OUT_JSON)
print("Text:", OUT_TXT)
print("Metrics CSV:", ROOT / "reports" / "breast_densenet121_v1_v2_validation_metrics.csv")
print("Predictions CSV:", OUT_CSV)
print("No checkpoint or test data was modified or used.")
