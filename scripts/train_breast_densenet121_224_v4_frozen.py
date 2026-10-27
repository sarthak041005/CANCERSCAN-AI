from pathlib import Path
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

# CancerScan AI - DenseNet121 v4
# Entire feature extractor remains frozen. Only the classifier is trained.
# Uses train.csv and val.csv only. Never loads test.csv.

ROOT = Path.cwd()
TRAIN_CSV = ROOT / "datasets" / "train.csv"
VAL_CSV = ROOT / "datasets" / "val.csv"
IMAGE_ROOT = ROOT / "datasets" / "breast_preprocessed_224"
INITIAL = ROOT / "saved_models" / "breast_densenet121_224_best.pth"

MODEL_OUT = ROOT / "saved_models" / "breast_densenet121_224_v4_frozen_best.pth"
HISTORY_JSON = ROOT / "reports" / "breast_densenet121_224_v4_frozen_history.json"
HISTORY_CSV = ROOT / "reports" / "breast_densenet121_224_v4_frozen_history.csv"
VAL_JSON = ROOT / "reports" / "breast_densenet121_224_v4_frozen_validation.json"

SEED = 43
BATCH_SIZE = 8
MAX_EPOCHS = 40
PATIENCE = 8
HEAD_LR = 1e-4
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

for p in [TRAIN_CSV, VAL_CSV, IMAGE_ROOT, INITIAL]:
    if not p.exists():
        raise FileNotFoundError(f"Required path not found: {p}")

for p in [MODEL_OUT, HISTORY_JSON, HISTORY_CSV, VAL_JSON]:
    if p.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {p}. "
            "Change output names in this script before rerunning."
        )

(ROOT / "reports").mkdir(parents=True, exist_ok=True)
(ROOT / "saved_models").mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("CancerScan AI - DenseNet121 v4 (fully frozen backbone)")
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def load_split(csv_path):
    df = pd.read_csv(csv_path).copy()
    required = {"image_path", "pathology", "patient_id"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{csv_path.name} missing required columns: {sorted(missing_cols)}"
        )

    df["pathology"] = df["pathology"].astype(str).str.strip().str.upper()
    unknown = sorted(set(df["pathology"]) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Unknown labels in {csv_path.name}: {unknown}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    if df["patient_id"].isin(["", "nan", "None"]).any():
        raise ValueError(f"Missing patient IDs in {csv_path.name}")

    def resolve(raw):
        parts = [x for x in str(raw).replace("\\", "/").split("/") if x]
        if len(parts) < 2:
            raise ValueError(f"Cannot resolve image path: {raw}")
        return IMAGE_ROOT / parts[-2] / parts[-1]

    df["resolved_path"] = df["image_path"].apply(resolve)
    df["label"] = df["pathology"].map(LABEL_MAP).astype(int)
    missing = df.loc[
        ~df["resolved_path"].apply(lambda x: x.is_file()), "resolved_path"
    ]
    if len(missing):
        raise FileNotFoundError(
            f"{len(missing)} images missing in {csv_path.name}:\n"
            + "\n".join(str(x) for x in missing.head(10))
        )
    return df.reset_index(drop=True)

train_df = load_split(TRAIN_CSV)
val_df = load_split(VAL_CSV)

overlap = set(train_df["patient_id"]) & set(val_df["patient_id"])
print("Train patients:", train_df["patient_id"].nunique())
print("Validation patients:", val_df["patient_id"].nunique())
print("Patient overlap:", len(overlap))
if overlap:
    raise ValueError("Patient overlap found; training aborted.")
if train_df["label"].nunique() != 2 or val_df["label"].nunique() != 2:
    raise ValueError("Both splits must contain benign and malignant examples.")

print("Train images:", len(train_df),
      "class counts:", train_df["label"].value_counts().sort_index().to_dict())
print("Validation images:", len(val_df),
      "class counts:", val_df["label"].value_counts().sort_index().to_dict())

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(
        degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)
    ),
    transforms.ColorJitter(brightness=0.05, contrast=0.07),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class BreastDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row["resolved_path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(row["label"])

train_loader = DataLoader(
    BreastDataset(train_df, train_tf), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    BreastDataset(val_df, eval_tf), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
)

model = models.densenet121(weights=None)
n_features = model.classifier.in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(n_features, 2))

try:
    ckpt = torch.load(INITIAL, map_location="cpu", weights_only=True)
except TypeError:
    ckpt = torch.load(INITIAL, map_location="cpu")

if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
    raise KeyError("Initial checkpoint has no model_state_dict.")
arch = ckpt.get("architecture")
if arch and arch.lower() != "densenet121":
    raise ValueError(f"Unexpected checkpoint architecture: {arch}")
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.to(DEVICE)
print("Initialized from:", INITIAL)
print("Original checkpoint will not be modified.")

# Freeze all DenseNet feature layers; only classifier parameters can update.
for p in model.features.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

counts = train_df["label"].value_counts().reindex([0, 1], fill_value=0)
if (counts == 0).any():
    raise ValueError("Training data must contain both classes.")
weights_np = len(train_df) / (2.0 * counts.to_numpy(dtype=np.float64))
weights_np = weights_np / weights_np.mean()
class_weights = torch.tensor(weights_np, dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(
    model.classifier.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
)
print("Class weights [BENIGN, MALIGNANT]:", weights_np.round(4).tolist())
print("Trainable parameters: classifier only")

def get_metrics(y, probs):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "malignant_precision": float(precision_score(y, pred, zero_division=0)),
        "malignant_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "malignant_f1": float(f1_score(y, pred, zero_division=0)),
        "malignant_f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        },
    }

def validate():
    model.eval()
    loss_sum, total = 0.0, 0
    ys, ps = [], []
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            prob = torch.softmax(logits, dim=1)[:, 1]
            n = labels.size(0)
            loss_sum += float(loss.item()) * n
            total += n
            ys.extend(labels.cpu().numpy().tolist())
            ps.extend(prob.cpu().numpy().tolist())
    metrics = get_metrics(ys, ps)
    metrics["loss"] = loss_sum / max(total, 1)
    return metrics, np.asarray(ys), np.asarray(ps)

history = []
best_bal_acc = -1.0
best_recall = -1.0
best_epoch = None
best_state = None
stale = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # Feature extractor stays in eval mode to prevent BN statistics changing.
    model.train()
    model.features.eval()
    train_loss_sum, total, correct = 0.0, 0, 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=5.0
        )
        optimizer.step()
        n = labels.size(0)
        train_loss_sum += float(loss.item()) * n
        total += n
        correct += int((logits.argmax(1) == labels).sum().item())

    train_loss = train_loss_sum / max(total, 1)
    train_acc = correct / max(total, 1)
    val_metrics, _, _ = validate()
    score = val_metrics["balanced_accuracy"]
    recall_now = val_metrics["malignant_recall"]
    scheduler.step(score)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "validation": val_metrics,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    })

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_acc={val_metrics['accuracy']:.4f} | "
        f"val_bal_acc={score:.4f} | val_recall={recall_now:.4f} | "
        f"val_specificity={val_metrics['specificity']:.4f} | "
        f"val_auc={val_metrics['roc_auc']:.4f}"
    )

    improved = (
        score > best_bal_acc + 1e-6
        or (abs(score - best_bal_acc) <= 1e-6 and recall_now > best_recall + 1e-6)
    )
    if improved:
        best_bal_acc = score
        best_recall = recall_now
        best_epoch = epoch
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        torch.save({
            "model_state_dict": best_state,
            "architecture": "densenet121",
            "image_size": 224,
            "labels": {"BENIGN": 0, "MALIGNANT": 1},
            "best_epoch": best_epoch,
            "best_validation_balanced_accuracy": best_bal_acc,
            "best_validation_malignant_recall": best_recall,
            "initial_checkpoint": str(INITIAL.relative_to(ROOT)),
            "class_weights": weights_np.tolist(),
            "selection_metric": (
                "validation balanced accuracy at threshold 0.50; "
                "malignant recall tie-break"
            ),
            "seed": SEED,
            "training_strategy": "fully frozen features; classifier-only training",
        }, MODEL_OUT)
        stale = 0
        print("  Saved new best checkpoint:", MODEL_OUT)
    else:
        stale += 1

    if stale >= PATIENCE:
        print(f"Early stopping after {epoch} epochs.")
        break

if best_state is None:
    raise RuntimeError("No best checkpoint was saved.")

model.load_state_dict(best_state, strict=True)
best_val, y_val, p_val = validate()
pred_val = (p_val >= 0.5).astype(int)
best_val["threshold"] = 0.5
best_val["best_epoch"] = int(best_epoch)
best_val["train_images"] = int(len(train_df))
best_val["validation_images"] = int(len(val_df))
best_val["patient_overlap"] = 0
best_val["initial_checkpoint"] = str(INITIAL.relative_to(ROOT))
best_val["new_checkpoint"] = str(MODEL_OUT.relative_to(ROOT))
best_val["classification_report"] = classification_report(
    y_val, pred_val, labels=[0, 1], target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)
best_val["note"] = (
    "Validation-only model selection. The test set was not loaded or used. "
    "External patient-level validation is still required."
)

payload = {
    "experiment": "DenseNet121 v4 fully frozen backbone, weighted cross-entropy",
    "seed": SEED,
    "image_size": 224,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "dropout": 0.5,
    "class_weights": {"BENIGN": float(weights_np[0]), "MALIGNANT": float(weights_np[1])},
    "best_epoch": int(best_epoch),
    "best_validation_balanced_accuracy": float(best_bal_acc),
    "best_validation_malignant_recall": float(best_recall),
    "epochs_completed": len(history),
    "history": history,
}
HISTORY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
pd.DataFrame([
    {
        "epoch": h["epoch"],
        "train_loss": h["train_loss"],
        "train_accuracy": h["train_accuracy"],
        "validation_loss": h["validation"]["loss"],
        "validation_accuracy": h["validation"]["accuracy"],
        "validation_balanced_accuracy": h["validation"]["balanced_accuracy"],
        "validation_malignant_recall": h["validation"]["malignant_recall"],
        "validation_specificity": h["validation"]["specificity"],
        "validation_f1": h["validation"]["malignant_f1"],
        "validation_f2": h["validation"]["malignant_f2"],
        "validation_roc_auc": h["validation"]["roc_auc"],
        "learning_rate": h["learning_rate"],
    }
    for h in history
]).to_csv(HISTORY_CSV, index=False)
VAL_JSON.write_text(json.dumps(best_val, indent=2), encoding="utf-8")

print("\n" + "=" * 68)
print("DENSENET121 V4 TRAINING COMPLETED")
print("=" * 68)
print("Best epoch:", best_epoch)
print(f"Validation accuracy: {best_val['accuracy']:.4f}")
print(f"Validation balanced accuracy: {best_val['balanced_accuracy']:.4f}")
print(f"Validation malignant recall: {best_val['malignant_recall']:.4f}")
print(f"Validation specificity: {best_val['specificity']:.4f}")
print(f"Validation ROC-AUC: {best_val['roc_auc']:.4f}")
print("Confusion matrix:", best_val["confusion_matrix"])
print("\nNew checkpoint:", MODEL_OUT)
print("History JSON:", HISTORY_JSON)
print("History CSV:", HISTORY_CSV)
print("Validation report:", VAL_JSON)
print("\nThe test set was not loaded or evaluated.")
print("Existing checkpoints and reports were not modified.")
