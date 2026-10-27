from pathlib import Path
import copy
import json
import random
import sys

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

# ============================================================
# CancerScan AI: DenseNet121 v2
# Weighted loss + staged fine-tuning
# Uses TRAIN and VALIDATION only. Does not load test.csv.
# ============================================================

ROOT = Path.cwd()
TRAIN_CSV = ROOT / "datasets" / "train.csv"
VAL_CSV = ROOT / "datasets" / "val.csv"
IMAGE_ROOT = ROOT / "datasets" / "breast_preprocessed_224"
INITIAL_CHECKPOINT = ROOT / "saved_models" / "breast_densenet121_224_best.pth"

MODEL_OUT = ROOT / "saved_models" / "breast_densenet121_224_v2_best.pth"
HISTORY_OUT = ROOT / "reports" / "breast_densenet121_224_v2_history.json"
VALIDATION_OUT = ROOT / "reports" / "breast_densenet121_224_v2_validation.json"
THRESHOLD_OUT = ROOT / "reports" / "breast_densenet121_224_v2_thresholds.csv"

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
WARMUP_EPOCHS = 3
MAX_EPOCHS = 30
PATIENCE = 8
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
MIN_SPECIFICITY_FOR_THRESHOLD = 0.60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

# -------------------- Reproducibility --------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------- Safety checks --------------------
for path in [TRAIN_CSV, VAL_CSV, IMAGE_ROOT, INITIAL_CHECKPOINT]:
    if not path.exists():
        raise FileNotFoundError(f"Required path missing: {path}")

for path in [MODEL_OUT, HISTORY_OUT, VALIDATION_OUT, THRESHOLD_OUT]:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}\n"
            "Choose new output filenames before running again."
        )

(ROOT / "saved_models").mkdir(parents=True, exist_ok=True)
(ROOT / "reports").mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("CancerScan AI - DenseNet121 v2")
print("=" * 68)
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------- Load and validate CSVs --------------------
def load_split(csv_path):
    frame = pd.read_csv(csv_path).copy()
    required = {"image_path", "pathology"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    frame["pathology"] = frame["pathology"].astype(str).str.strip().str.upper()
    unknown = sorted(set(frame["pathology"]) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Unknown pathology labels in {csv_path.name}: {unknown}")

    def resolve(raw_path):
        parts = [p for p in str(raw_path).replace("\\", "/").split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"Cannot resolve image path: {raw_path}")
        return IMAGE_ROOT / parts[-2] / parts[-1]

    frame["resolved_path"] = frame["image_path"].apply(resolve)
    frame["label"] = frame["pathology"].map(LABEL_MAP).astype(int)

    missing_images = frame.loc[
        ~frame["resolved_path"].apply(lambda p: p.is_file()), "resolved_path"
    ]
    if not missing_images.empty:
        raise FileNotFoundError(
            f"{len(missing_images)} images missing in {csv_path.name}. Examples:\n"
            + "\n".join(map(str, missing_images.head(10)))
        )
    return frame.reset_index(drop=True)

train_df = load_split(TRAIN_CSV)
val_df = load_split(VAL_CSV)

# If a patient identifier exists, verify that train and validation are disjoint.
patient_col = next(
    (c for c in ["patient_id", "patient_ID", "patient", "subject_id"]
     if c in train_df.columns and c in val_df.columns),
    None
)
if patient_col:
    train_patients = set(train_df[patient_col].astype(str))
    val_patients = set(val_df[patient_col].astype(str))
    overlap = train_patients.intersection(val_patients)
    print("Patient ID column:", patient_col)
    print("Train/validation patient overlap:", len(overlap))
    if overlap:
        raise ValueError(
            "Patient overlap detected between train and validation. "
            "Stop and fix the split before training."
        )
else:
    print("Note: no recognized patient-ID column; using the existing CSV split.")

if train_df["label"].nunique() != 2 or val_df["label"].nunique() != 2:
    raise ValueError("Both splits must contain benign and malignant cases.")

print("\nTrain images:", len(train_df))
print("Train labels:", train_df["label"].value_counts().sort_index().to_dict(),
      "(0=benign, 1=malignant)")
print("Validation images:", len(val_df))
print("Validation labels:", val_df["label"].value_counts().sort_index().to_dict())

# -------------------- Transforms --------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(
        degrees=0, translate=(0.02, 0.02), scale=(0.97, 1.03)
    ),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

class BreastDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["resolved_path"]) as img:
            image = img.convert("RGB")
            image = self.transform(image)
        return image, int(row["label"])

train_ds = BreastDataset(train_df, train_transform)
val_ds = BreastDataset(val_df, eval_transform)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available()
)

# -------------------- Build matching DenseNet121 --------------------
model = models.densenet121(weights=None)
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, 2)
)

try:
    checkpoint = torch.load(
        INITIAL_CHECKPOINT, map_location="cpu", weights_only=True
    )
except TypeError:
    checkpoint = torch.load(INITIAL_CHECKPOINT, map_location="cpu")

if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
    raise KeyError(
        "Expected 'model_state_dict' in initial checkpoint. "
        f"Found: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
    )

saved_arch = checkpoint.get("architecture")
if saved_arch and saved_arch.lower() != "densenet121":
    raise ValueError(f"Expected DenseNet121 checkpoint, got: {saved_arch}")

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model = model.to(DEVICE)
print("\nInitialized from existing checkpoint:", INITIAL_CHECKPOINT)
print("Initial checkpoint has not been modified.")

# -------------------- Weighted loss --------------------
train_counts = train_df["label"].value_counts().reindex([0, 1], fill_value=0)
if (train_counts == 0).any():
    raise ValueError("Training split must include both classes.")

# Inverse-frequency weights, normalized to have mean 1.
class_weights = len(train_df) / (2.0 * train_counts.to_numpy(dtype=np.float64))
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.tensor(
    class_weights, dtype=torch.float32, device=DEVICE
)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print("\nClass weights (BENIGN, MALIGNANT):",
      [round(float(x), 4) for x in class_weights])
print("Weighted cross-entropy enabled.")

# -------------------- Metrics --------------------
def calculate_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "malignant_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "malignant_recall": sensitivity,
        "specificity": specificity,
        "malignant_f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "malignant_f2": float(
            fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        },
    }

def run_validation():
    model.eval()
    losses = []
    labels_all = []
    probs_all = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            losses.append(float(loss.item()))
            labels_all.extend(labels.cpu().numpy().tolist())
            probs_all.extend(probabilities.cpu().numpy().tolist())

    metrics = calculate_metrics(labels_all, probs_all, threshold=0.5)
    metrics["loss"] = float(np.mean(losses))
    return metrics, np.asarray(labels_all), np.asarray(probs_all)

# -------------------- Optimizers: staged fine-tuning --------------------
def freeze_backbone():
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_all():
    for param in model.features.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

def make_optimizer():
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.features.parameters() if p.requires_grad],
                "lr": BACKBONE_LR,
            },
            {
                "params": [p for p in model.classifier.parameters() if p.requires_grad],
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

# -------------------- Training loop --------------------
history = []
best_score = -1.0
best_epoch = None
best_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    if epoch <= WARMUP_EPOCHS:
        freeze_backbone()
        stage = "classifier_warmup"
    else:
        unfreeze_all()
        stage = "full_finetuning"

    # Rebuild optimizer when moving from frozen to unfrozen backbone.
    if epoch == 1 or epoch == WARMUP_EPOCHS + 1:
        optimizer = make_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
        )

    model.train()
    total_loss = 0.0
    total_examples = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_n = labels.size(0)
        total_loss += float(loss.item()) * batch_n
        total_examples += batch_n
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    train_loss = total_loss / max(total_examples, 1)
    train_accuracy = correct / max(total_examples, 1)
    val_metrics, _, _ = run_validation()

    # Select the checkpoint using validation F2 at threshold 0.5.
    # Threshold selection itself is performed after restoring this checkpoint.
    selection_score = val_metrics["malignant_f2"]
    scheduler.step(selection_score)

    record = {
        "epoch": epoch,
        "stage": stage,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation": val_metrics,
        "selection_metric": "validation malignant F2 at threshold 0.5",
        "selection_score": selection_score,
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | {stage} | "
        f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_recall={val_metrics['malignant_recall']:.4f} | "
        f"val_specificity={val_metrics['specificity']:.4f} | "
        f"val_F2={selection_score:.4f} | val_AUC={val_metrics['roc_auc']:.4f}"
    )

    if selection_score > best_score + 1e-6:
        best_score = selection_score
        best_epoch = epoch
        best_state = copy.deepcopy({
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        })
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": best_state,
            "architecture": "densenet121",
            "image_size": IMAGE_SIZE,
            "labels": {"BENIGN": 0, "MALIGNANT": 1},
            "best_epoch": best_epoch,
            "best_validation_f2_at_0_5": best_score,
            "initial_checkpoint": str(INITIAL_CHECKPOINT.relative_to(ROOT)),
            "class_weights": class_weights.tolist(),
            "selection_metric": "validation malignant F2 at threshold 0.5",
            "seed": SEED,
        }, MODEL_OUT)
        print("  Saved new best checkpoint:", MODEL_OUT)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping after {epoch} epochs.")
        break

if best_state is None:
    raise RuntimeError("No valid checkpoint was saved.")

# -------------------- Restore best checkpoint --------------------
model.load_state_dict(best_state, strict=True)
model.eval()
best_val_metrics, y_val, p_val = run_validation()

# -------------------- Validation-only threshold analysis --------------------
threshold_rows = []
for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
    metrics = calculate_metrics(y_val, p_val, threshold=float(threshold))
    threshold_rows.append(metrics)

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(THRESHOLD_OUT, index=False)

eligible = threshold_df[
    threshold_df["specificity"] >= MIN_SPECIFICITY_FOR_THRESHOLD
]
if eligible.empty:
    # Explicit fallback if no threshold meets the predeclared constraint.
    eligible = threshold_df

# Maximize malignant recall; ties prefer higher specificity, then higher F2.
chosen = eligible.sort_values(
    ["malignant_recall", "specificity", "malignant_f2"],
    ascending=[False, False, False]
).iloc[0]
chosen_threshold = float(chosen["threshold"])

val_predictions = (p_val >= chosen_threshold).astype(int)
final_validation_metrics = calculate_metrics(
    y_val, p_val, threshold=chosen_threshold
)
final_validation_metrics["best_checkpoint_epoch"] = int(best_epoch)
final_validation_metrics["checkpoint_selection_score_f2_at_0_5"] = float(best_score)
final_validation_metrics["threshold_selection_rule"] = (
    "Maximize malignant recall on validation data subject to specificity >= "
    f"{MIN_SPECIFICITY_FOR_THRESHOLD:.2f}; ties prefer higher specificity then F2. "
    "If no threshold meets the constraint, maximize recall over the tested thresholds."
)
final_validation_metrics["classification_report"] = classification_report(
    y_val, val_predictions, labels=[0, 1], target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)
final_validation_metrics["class_weights"] = {
    "BENIGN": float(class_weights[0]),
    "MALIGNANT": float(class_weights[1]),
}
final_validation_metrics["training_images"] = int(len(train_df))
final_validation_metrics["validation_images"] = int(len(val_df))
final_validation_metrics["initial_checkpoint"] = str(
    INITIAL_CHECKPOINT.relative_to(ROOT)
)
final_validation_metrics["new_checkpoint"] = str(MODEL_OUT.relative_to(ROOT))

history_payload = {
    "experiment": "DenseNet121 v2 weighted loss and staged fine-tuning",
    "seed": SEED,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "warmup_epochs": WARMUP_EPOCHS,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "backbone_lr": BACKBONE_LR,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "class_weights": {
        "BENIGN": float(class_weights[0]),
        "MALIGNANT": float(class_weights[1]),
    },
    "best_epoch": int(best_epoch),
    "best_validation_f2_at_0_5": float(best_score),
    "epochs_completed": len(history),
    "history": history,
}
HISTORY_OUT.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
VALIDATION_OUT.write_text(
    json.dumps(final_validation_metrics, indent=2), encoding="utf-8"
)

print("\n" + "=" * 68)
print("DENSENET121 V2 TRAINING COMPLETED")
print("=" * 68)
print("Best epoch:", best_epoch)
print(f"Best validation F2 at threshold 0.50: {best_score:.4f}")
print(f"Validation-selected threshold: {chosen_threshold:.2f}")
for key in [
    "accuracy", "balanced_accuracy", "malignant_precision",
    "malignant_recall", "specificity", "malignant_f1",
    "malignant_f2", "roc_auc"
]:
    print(f"{key}: {final_validation_metrics[key]:.4f}")
print("Confusion matrix:", final_validation_metrics["confusion_matrix"])
print("\nNew checkpoint:", MODEL_OUT)
print("Training history:", HISTORY_OUT)
print("Validation metrics:", VALIDATION_OUT)
print("Threshold sweep:", THRESHOLD_OUT)
print("\nThe test set was not loaded or evaluated.")
print("Original checkpoint and previous reports were not modified.")
