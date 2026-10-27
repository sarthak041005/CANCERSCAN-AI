from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
)

# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
IMAGE_DIR = DATA_DIR / "breast_preprocessed_224"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "saved_models"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

CHECKPOINT_PATH = MODEL_DIR / "breast_efficientnet_b0_224_best.pth"
HISTORY_JSON = REPORT_DIR / "breast_efficientnet_b0_224_history.json"
HISTORY_CSV = REPORT_DIR / "breast_efficientnet_b0_224_history.csv"
VAL_JSON = REPORT_DIR / "breast_efficientnet_b0_224_validation.json"
VAL_PREDICTIONS = REPORT_DIR / "breast_efficientnet_b0_224_validation_predictions.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 20
WARMUP_EPOCHS = 3
PATIENCE = 6
NUM_WORKERS = 0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# DATASET
# =========================================================
def normalize_label(value):
    if pd.isna(value):
        raise ValueError("Encountered a missing label.")

    if isinstance(value, str):
        value = value.strip().upper()
        if value in {"BENIGN", "BENIGN_WITHOUT_CALLBACK", "BENIGN_WITH_CALLBACK", "0"}:
            return 0
        if value in {"MALIGNANT", "MALIGNANT_WITHOUT_CALLBACK", "1"}:
            return 1
        raise ValueError(f"Unknown label: {value}")

    value = int(value)
    if value not in (0, 1):
        raise ValueError(f"Unknown numeric label: {value}")
    return value


def resolve_image(raw_path):
    parts = Path(str(raw_path).replace("\\", "/")).parts
    candidates = []

    if len(parts) >= 2:
        candidates.append(IMAGE_DIR / parts[-2] / parts[-1])

    candidates.append(IMAGE_DIR / parts[-1])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find image for CSV path: {raw_path}")


class BreastDataset(Dataset):
    def __init__(self, csv_path, train=False):
        self.df = pd.read_csv(csv_path)

        required = {"image_path", "label", "patient_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        self.paths = []
        self.labels = []
        self.patient_ids = []

        for _, row in self.df.iterrows():
            self.paths.append(resolve_image(row["image_path"]))
            self.labels.append(normalize_label(row["label"]))
            self.patient_ids.append(str(row["patient_id"]).strip())

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
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

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            image = im.convert("RGB")
            image = self.transform(image)

        return (
            image,
            self.labels[index],
            str(self.paths[index]),
        )


# =========================================================
# MODEL
# =========================================================
def build_model():
    # Download pretrained weights if they are not cached locally.
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 2),
    )
    return model


def freeze_backbone(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_block(model):
    # Fine-tune only the last EfficientNet feature block and classifier.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


# =========================================================
# METRICS / EVALUATION
# =========================================================
def calculate_metrics(labels, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if tn + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "malignant_precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "malignant_recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "malignant_f1": float(f1_score(labels, predictions, zero_division=0)),
        "malignant_f2": float(
            fbeta_score(labels, predictions, beta=2, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    # Keep frozen feature layers in eval mode, including BatchNorm,
    # so small batches do not alter frozen pretrained running statistics.
    if training:
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_paths = []

    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for images, labels, paths in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, dtype=torch.long, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probabilities.detach().cpu().numpy().tolist())
            all_paths.extend(list(paths))

    labels_np = np.asarray(all_labels, dtype=int)
    probs_np = np.asarray(all_probs, dtype=float)
    metrics = calculate_metrics(labels_np, probs_np, threshold=0.5)
    metrics["loss"] = float(total_loss / len(loader.dataset))

    return metrics, labels_np, probs_np, all_paths


# =========================================================
# MAIN TRAINING
# =========================================================
def main():
    print("=" * 70)
    print("BREAST CANCER CLASSIFICATION — EFFICIENTNET-B0")
    print("=" * 70)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Train CSV:", TRAIN_CSV)
    print("Validation CSV:", VAL_CSV)
    print("Test CSV will not be loaded.")

    train_ds = BreastDataset(TRAIN_CSV, train=True)
    val_ds = BreastDataset(VAL_CSV, train=False)

    # Verify patient-level separation using actual CSV patient IDs.
    train_patients = set(train_ds.patient_ids)
    val_patients = set(val_ds.patient_ids)
    overlap = train_patients & val_patients

    print(f"Train images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Patient overlap: {len(overlap)}")

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: {len(overlap)} shared patient IDs."
        )

    print("Train label counts:", {
        CLASS_NAMES[c]: train_ds.labels.count(c) for c in (0, 1)
    })
    print("Validation label counts:", {
        CLASS_NAMES[c]: val_ds.labels.count(c) for c in (0, 1)
    })

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model().to(DEVICE)

    # Weighted loss gives the minority malignant class greater influence.
    counts = np.bincount(train_ds.labels, minlength=2)
    class_weights = len(train_ds) / (2.0 * counts)
    class_weights = torch.tensor(
        class_weights, dtype=torch.float32, device=DEVICE
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # At the start of the fine-tuning phase, unfreeze only the last block.
        if epoch == WARMUP_EPOCHS + 1:
            print("\nUnfreezing final EfficientNet feature block.")
            unfreeze_last_block(model)

            optimizer = torch.optim.AdamW([
                {
                    "params": [
                        p for p in model.features[-1].parameters()
                        if p.requires_grad
                    ],
                    "lr": 1e-5,
                },
                {
                    "params": model.classifier.parameters(),
                    "lr": 1e-4,
                },
            ], weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

        train_metrics, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_metrics, _, _, _ = run_epoch(
            model, val_loader, criterion, optimizer=None
        )

        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "phase": "frozen" if epoch <= WARMUP_EPOCHS else "partial_finetune",
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_malignant_recall": val_metrics["mal
```powershell
@'
from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
)

# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
IMAGE_DIR = DATA_DIR / "breast_preprocessed_224"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "saved_models"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

CHECKPOINT_PATH = MODEL_DIR / "breast_efficientnet_b0_224_best.pth"
HISTORY_JSON = REPORT_DIR / "breast_efficientnet_b0_224_history.json"
HISTORY_CSV = REPORT_DIR / "breast_efficientnet_b0_224_history.csv"
VAL_JSON = REPORT_DIR / "breast_efficientnet_b0_224_validation.json"
VAL_PREDICTIONS = REPORT_DIR / "breast_efficientnet_b0_224_validation_predictions.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 20
WARMUP_EPOCHS = 3
PATIENCE = 6
NUM_WORKERS = 0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# DATASET
# =========================================================
def normalize_label(value):
    if pd.isna(value):
        raise ValueError("Encountered a missing label.")

    if isinstance(value, str):
        value = value.strip().upper()
        if value in {"BENIGN", "BENIGN_WITHOUT_CALLBACK", "BENIGN_WITH_CALLBACK", "0"}:
            return 0
        if value in {"MALIGNANT", "MALIGNANT_WITHOUT_CALLBACK", "1"}:
            return 1
        raise ValueError(f"Unknown label: {value}")

    value = int(value)
    if value not in (0, 1):
        raise ValueError(f"Unknown numeric label: {value}")
    return value


def resolve_image(raw_path):
    parts = Path(str(raw_path).replace("\\", "/")).parts
    candidates = []

    if len(parts) >= 2:
        candidates.append(IMAGE_DIR / parts[-2] / parts[-1])

    candidates.append(IMAGE_DIR / parts[-1])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find image for CSV path: {raw_path}")


class BreastDataset(Dataset):
    def __init__(self, csv_path, train=False):
        self.df = pd.read_csv(csv_path)

        required = {"image_path", "label", "patient_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        self.paths = []
        self.labels = []
        self.patient_ids = []

        for _, row in self.df.iterrows():
            self.paths.append(resolve_image(row["image_path"]))
            self.labels.append(normalize_label(row["label"]))
            self.patient_ids.append(str(row["patient_id"]).strip())

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
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

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            image = im.convert("RGB")
            image = self.transform(image)

        return (
            image,
            self.labels[index],
            str(self.paths[index]),
        )


# =========================================================
# MODEL
# =========================================================
def build_model():
    # Download pretrained weights if they are not cached locally.
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 2),
    )
    return model


def freeze_backbone(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_block(model):
    # Fine-tune only the last EfficientNet feature block and classifier.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


# =========================================================
# METRICS / EVALUATION
# =========================================================
def calculate_metrics(labels, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if tn + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "malignant_precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "malignant_recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "malignant_f1": float(f1_score(labels, predictions, zero_division=0)),
        "malignant_f2": float(
            fbeta_score(labels, predictions, beta=2, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    # Keep frozen feature layers in eval mode, including BatchNorm,
    # so small batches do not alter frozen pretrained running statistics.
    if training:
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_paths = []

    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for images, labels, paths in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, dtype=torch.long, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probabilities.detach().cpu().numpy().tolist())
            all_paths.extend(list(paths))

    labels_np = np.asarray(all_labels, dtype=int)
    probs_np = np.asarray(all_probs, dtype=float)
    metrics = calculate_metrics(labels_np, probs_np, threshold=0.5)
    metrics["loss"] = float(total_loss / len(loader.dataset))

    return metrics, labels_np, probs_np, all_paths


# =========================================================
# MAIN TRAINING
# =========================================================
def main():
    print("=" * 70)
    print("BREAST CANCER CLASSIFICATION — EFFICIENTNET-B0")
    print("=" * 70)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Train CSV:", TRAIN_CSV)
    print("Validation CSV:", VAL_CSV)
    print("Test CSV will not be loaded.")

    train_ds = BreastDataset(TRAIN_CSV, train=True)
    val_ds = BreastDataset(VAL_CSV, train=False)

    # Verify patient-level separation using actual CSV patient IDs.
    train_patients = set(train_ds.patient_ids)
    val_patients = set(val_ds.patient_ids)
    overlap = train_patients & val_patients

    print(f"Train images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Patient overlap: {len(overlap)}")

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: {len(overlap)} shared patient IDs."
        )

    print("Train label counts:", {
        CLASS_NAMES[c]: train_ds.labels.count(c) for c in (0, 1)
    })
    print("Validation label counts:", {
        CLASS_NAMES[c]: val_ds.labels.count(c) for c in (0, 1)
    })

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model().to(DEVICE)

    # Weighted loss gives the minority malignant class greater influence.
    counts = np.bincount(train_ds.labels, minlength=2)
    class_weights = len(train_ds) / (2.0 * counts)
    class_weights = torch.tensor(
        class_weights, dtype=torch.float32, device=DEVICE
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # At the start of the fine-tuning phase, unfreeze only the last block.
        if epoch == WARMUP_EPOCHS + 1:
            print("\nUnfreezing final EfficientNet feature block.")
            unfreeze_last_block(model)

            optimizer = torch.optim.AdamW([
                {
                    "params": [
                        p for p in model.features[-1].parameters()
                        if p.requires_grad
                    ],
                    "lr": 1e-5,
                },
                {
                    "params": model.classifier.parameters(),
                    "lr": 1e-4,
                },
            ], weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

        train_metrics, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_metrics, _, _, _ = run_epoch(
            model, val_loader, criterion, optimizer=None
        )

        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "phase": "frozen" if epoch <= WARMUP_EPOCHS else "partial_finetune",
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_malignant_recall": val_metrics["malignant_recall_sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["malignant_f1"],
            "val_f2": val_metrics["malignant_f2"],
            "val_auc": val_metrics["roc_auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"phase={row['phase']} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_recall={row['val_malignant_recall']:.4f} | "
            f"val_spec={row['val_specificity']:.4f} | "
            f"val_auc={row['val_auc']:.4f}"
        )

        # Save the best validation-loss checkpoint only.
        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "model_name": "EfficientNet-B0",
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "class_mapping": {"BENIGN": 0, "MALIGNANT": 1},
                "image_size": IMAGE_SIZE,
                "class_weights": class_weights.detach().cpu().tolist(),
            }, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping after epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Reload the best validation-loss checkpoint for final validation reporting.
    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_metrics, y_true, probabilities, paths = run_epoch(
        model, val_loader, criterion, optimizer=None
    )

    predictions = (probabilities >= 0.5).astype(int)
    prediction_df = pd.DataFrame({
        "image_path": paths,
        "true_label": y_true,
        "true_class": [CLASS_NAMES[x] for x in y_true],
        "malignant_probability": probabilities,
        "predicted_label_t050": predictions,
        "predicted_class_t050": [CLASS_NAMES[x] for x in predictions],
    })
    prediction_df.to_csv(VAL_PREDICTIONS, index=False)

    report = {
        "model": "EfficientNet-B0",
        "checkpoint": str(CHECKPOINT_PATH),
        "best_epoch_by_validation_loss": best_epoch,
        "best_validation_loss": best_val_loss,
        "validation_images": len(val_ds),
        "train_patient_count": len(train_patients),
        "validation_patient_count": len(val_patients),
        "patient_overlap": len(overlap),
        "validation_metrics_threshold_050": val_metrics,
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "elapsed_minutes": (time.time() - start_time) / 60,
        "test_set_used": False,
        "note": (
            "Research prototype only. Validation results are not clinical "
            "evidence. The existing test set has been examined previously."
        ),
    }

    with open(VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("EFFICIENTNET-B0 TRAINING COMPLETE")
    print("=" * 70)
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    for key, value in val_metrics.items():
        print(f"{key}: {value}")
    print("Confusion matrix [BENIGN, MALIGNANT]:")
    print(report["confusion_matrix"])
    print("\nCheckpoint:", CHECKPOINT_PATH)
    print("History:", HISTORY_CSV)
    print("Validation report:", VAL_JSON)
    print("Validation predictions:", VAL_PREDICTIONS)
    print("Test CSV was not loaded or evaluated.")


if __name__ == "__main__":
    main()
