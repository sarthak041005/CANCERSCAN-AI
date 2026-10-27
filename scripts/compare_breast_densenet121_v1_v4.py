from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
IMAGE_DIR = DATA_DIR / "breast_preprocessed_224"
VAL_CSV = DATA_DIR / "val.csv"
REPORT_DIR = ROOT / "reports" / "breast_densenet121_v1_v4_comparison"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "V1_DenseNet121_Baseline":
        ROOT / "saved_models" / "breast_densenet121_224_best.pth",
    "V2_DenseNet121_WeightedLoss":
        ROOT / "saved_models" / "breast_densenet121_224_v2_best.pth",
    "V3_DenseNet121_PartialFineTune":
        ROOT / "saved_models" / "breast_densenet121_224_v3_best.pth",
    "V4_DenseNet121_FrozenBackbone":
        ROOT / "saved_models" / "breast_densenet121_224_v4_frozen_best.pth",
}

BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_SIZE = 224
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.02), 2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class index 0 = BENIGN, class index 1 = MALIGNANT
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class BreastValidationDataset(Dataset):
    def __init__(self, csv_path, image_root):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)

        if "image_path" not in self.df.columns:
            raise ValueError(
                f"'image_path' column missing from {csv_path}. "
                f"Columns found: {list(self.df.columns)}"
            )

        # Support common label column names.
        label_col = next(
            (c for c in ["label", "class", "target", "diagnosis"]
             if c in self.df.columns),
            None
        )
        if label_col is None:
            raise ValueError(
                "Could not find label column. Expected one of: "
                "label, class, target, diagnosis. "
                f"Columns found: {list(self.df.columns)}"
            )

        self.label_col = label_col
        self.paths = []
        self.labels = []

        for _, row in self.df.iterrows():
            raw_path = str(row["image_path"])
            parts = Path(raw_path.replace("\\", "/")).parts

            # Existing dataset convention: last two path components
            # are UID folder and image filename.
            if len(parts) >= 2:
                candidate = self.image_root / parts[-2] / parts[-1]
            else:
                candidate = self.image_root / parts[-1]

            if not candidate.exists():
                # Try filename directly under image root as a fallback.
                fallback = self.image_root / parts[-1]
                if fallback.exists():
                    candidate = fallback
                else:
                    raise FileNotFoundError(
                        f"Cannot resolve image: {raw_path}\n"
                        f"Tried: {candidate}\nTried: {fallback}"
                    )

            label_value = row[label_col]
            if isinstance(label_value, str):
                normalized = label_value.strip().upper()
                if normalized in ("MALIGNANT", "MALIGNANT_WITHOUT_CALLBACK", "1"):
                    label = 1
                elif normalized in (
                    "BENIGN", "BENIGN_WITHOUT_CALLBACK", "BENIGN_WITH_CALLBACK", "0"
                ):
                    label = 0
                else:
                    raise ValueError(
                        f"Unknown label {label_value!r} in column {label_col!r}"
                    )
            else:
                label = int(label_value)
                if label not in (0, 1):
                    raise ValueError(f"Unexpected numeric label: {label_value}")

            self.paths.append(candidate)
            self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, self.labels[idx], str(self.paths[idx])


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint)
        )
    else:
        raise ValueError(f"Unexpected checkpoint format: {path}")

    # Remove DataParallel prefix if present.
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }

    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features

    # All four inspected checkpoints have classifier.1.weight/bias.
    # Dropout is inactive in eval mode, so p does not affect predictions.
    if "classifier.1.weight" in state:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 2),
        )
    elif "classifier.weight" in state:
        model.classifier = nn.Linear(in_features, 2)
    else:
        raise KeyError(
            f"Unrecognized classifier keys in {path}: "
            f"{[k for k in state if k.startswith('classifier')]}"
        )

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------
def get_predictions(model, loader):
    all_labels, all_probs, all_paths = [], [], []

    with torch.inference_mode():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probabilities.cpu().numpy().tolist())
            all_paths.extend(list(paths))

    return (
        np.asarray(all_labels, dtype=int),
        np.asarray(all_probs, dtype=float),
        all_paths,
    )


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def calculate_metrics(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "malignant_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "malignant_recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "malignant_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "malignant_f2": float(
            fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix(y_true, probabilities, threshold, model_name):
    y_pred = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{model_name}\nConfusion Matrix (threshold={threshold:.2f})")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
    )

    cutoff = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cutoff else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / f"{model_name}_confusion_matrix_t050.png",
        dpi=180,
    )
    plt.close(fig)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("=" * 72)
    print("BREAST CANCER DENSENET121 V1-V4 VALIDATION COMPARISON")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Validation CSV: {VAL_CSV}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Output directory: {REPORT_DIR}")
    print("Test data will NOT be loaded or evaluated.")

    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Validation CSV not found: {VAL_CSV}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    dataset = BreastValidationDataset(VAL_CSV, IMAGE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    y_true = np.asarray(dataset.labels, dtype=int)
    print(f"Validation images: {len(dataset)}")
    print(f"Benign: {(y_true == 0).sum()}, Malignant: {(y_true == 1).sum()}")

    all_predictions = {}
    summary_rows = []
    threshold_rows = []
    predictions_df = pd.DataFrame({
        "image_path": [str(p) for p in dataset.paths],
        "true_label": y_true,
        "true_class": [CLASS_NAMES[y] for y in y_true],
    })

    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nLoading {model_name}...")
        model = load_checkpoint(checkpoint_path)
        labels, probs, paths = get_predictions(model, loader)

        if not np.array_equal(labels, y_true):
            raise RuntimeError("Label order mismatch between model and dataset.")

        all_predictions[model_name] = probs
        predictions_df[f"{model_name}_malignant_probability"] = probs

        # Common threshold for direct comparison.
        metrics_050 = calculate_metrics(labels, probs, 0.50)
        metrics_050["model"] = model_name
        summary_rows.append(metrics_050)

        save_confusion_matrix(labels, probs, 0.50, model_name)

        # Threshold sweeps are exploratory and based on validation only.
        for threshold in THRESHOLDS:
            row = calculate_metrics(labels, probs, float(threshold))
            row["model"] = model_name
            threshold_rows.append(row)

        print(
            f"  AUC={metrics_050['roc_auc']:.4f} | "
            f"Acc={metrics_050['accuracy']:.4f} | "
            f"Recall={metrics_050['malignant_recall_sensitivity']:.4f} | "
            f"Specificity={metrics_050['specificity']:.4f} | "
            f"F2={metrics_050['malignant_f2']:.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save tabular outputs.
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        ["model", "threshold", "accuracy", "balanced_accuracy",
         "malignant_precision", "malignant_recall_sensitivity",
         "specificity", "malignant_f1", "malignant_f2", "roc_auc",
         "tn", "fp", "fn", "tp"]
    ]
    summary_df.to_csv(REPORT_DIR / "metrics_at_threshold_050.csv", index=False)
    predictions_df.to_csv(REPORT_DIR / "validation_predictions_all_models.csv", index=False)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(REPORT_DIR / "threshold_sweep_validation.csv", index=False)

    # ROC curves.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in all_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_value = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set(
        xlabel="False Positive Rate (1 - Specificity)",
        ylabel="True Positive Rate (Sensitivity)",
        title="DenseNet121 V1-V4 ROC Curves — Validation Set",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves_validation.png", dpi=200)
    plt.close(fig)

    # Precision-recall curves.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in all_predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set(
        xlabel="Malignant Recall (Sensitivity)",
        ylabel="Malignant Precision",
        title="DenseNet121 V1-V4 Precision-Recall Curves — Validation Set",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "precision_recall_curves_validation.png", dpi=200)
    plt.close(fig)

    # Machine-readable summary.
    result = {
        "dataset": "validation only",
        "validation_csv": str(VAL_CSV),
        "image_count": int(len(dataset)),
        "class_mapping": {"BENIGN": 0, "MALIGNANT": 1},
        "threshold_050_metrics": summary_rows,
        "notes": [
            "No test.csv data was loaded or evaluated by this script.",
            "Threshold sweep results use the validation set and are exploratory.",
            "A threshold selected on validation data is not an independent clinical validation.",
            "High sensitivity can come with lower specificity and more false positives.",
            "This model is for research/educational use, not clinical diagnosis."
        ],
    }

    with open(REPORT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(REPORT_DIR / "comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("BREAST DENSENET121 V1-V4 VALIDATION COMPARISON\n")
        f.write("=" * 58 + "\n\n")
        f.write(f"Validation images: {len(dataset)}\n")
        f.write("No test.csv data was loaded.\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nThreshold sweeps are exploratory; do not treat them as independent evaluation.\n")

    print("\n" + "=" * 72)
    print("COMPARISON COMPLETE")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print("\nSaved outputs:")
    for file in sorted(REPORT_DIR.iterdir()):
        print(f"  {file}")
    print("\nReminder: validation results are for model comparison, not clinical diagnosis.")


if __name__ == "__main__":
    main()
