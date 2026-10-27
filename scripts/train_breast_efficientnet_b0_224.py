
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, roc_auc_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
IMAGE_ROOT = DATA / "breast_preprocessed_224"
REPORTS = ROOT / "reports"
MODELS = ROOT / "saved_models"

REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

TRAIN_CSV = DATA / "train.csv"
VAL_CSV = DATA / "val.csv"
CHECKPOINT = MODELS / "breast_efficientnet_b0_224_best.pth"
HISTORY = REPORTS / "breast_efficientnet_b0_224_history.csv"
VAL_REPORT = REPORTS / "breast_efficientnet_b0_224_validation.json"
VAL_PREDS = REPORTS / "breast_efficientnet_b0_224_validation_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
WARMUP_EPOCHS = 3
PATIENCE = 6
CLASSES = ["BENIGN", "MALIGNANT"]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def label_to_int(value):
    if isinstance(value, str):
        value = value.strip().upper()
        if value in {"BENIGN", "BENIGN_WITHOUT_CALLBACK", "BENIGN_WITH_CALLBACK", "0"}:
            return 0
        if value in {"MALIGNANT", "MALIGNANT_WITHOUT_CALLBACK", "1"}:
            return 1
        raise ValueError(f"Unknown label: {value}")
    value = int(value)
    if value not in (0, 1):
        raise ValueError(f"Unknown label: {value}")
    return value


def resolve_path(raw):
    parts = Path(str(raw).replace("\\", "/")).parts
    candidates = []
    if len(parts) >= 2:
        candidates.append(IMAGE_ROOT / parts[-2] / parts[-1])
    candidates.append(IMAGE_ROOT / parts[-1])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Image not found: {raw}")


class BreastDataset(Dataset):
    def __init__(self, csv_path, training=False):
        df = pd.read_csv(csv_path)
        required = {"image_path", "label", "patient_id"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} must contain {required}")

        self.paths = [resolve_path(p) for p in df["image_path"]]
        self.labels = [label_to_int(x) for x in df["label"]]
        self.patients = df["patient_id"].astype(str).str.strip().tolist()

        operations = [transforms.Resize((224, 224))]
        if training:
            operations += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
        operations += [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            image = self.transform(im.convert("RGB"))
        return image, self.labels[index], str(self.paths[index])


def make_model():
    model = models.efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )
    features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(features, 2),
    )
    return model


def metrics(y, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "malignant_precision": float(precision_score(y, pred, zero_division=0)),
        "malignant_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    # Keep frozen BatchNorm layers from updating their running statistics.
    if training:
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()

    total_loss = 0.0
    all_y, all_probs, all_paths = [], [], []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.long)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * len(labels)
            all_y.extend(labels.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())
            all_paths.extend(paths)

    y = np.asarray(all_y, dtype=int)
    probs = np.asarray(all_probs, dtype=float)
    result = metrics(y, probs)
    result["loss"] = total_loss / len(loader.dataset)
    return result, y, probs, all_paths


def main():
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = BreastDataset(TRAIN_CSV, training=True)
    val_ds = BreastDataset(VAL_CSV, training=False)

    train_patients = set(train_ds.patients)
    val_patients = set(val_ds.patients)
    overlap = train_patients & val_patients

    print("Train images:", len(train_ds))
    print("Validation images:", len(val_ds))
    print("Train patients:", len(train_patients))
    print("Validation patients:", len(val_patients))
    print("Patient overlap:", len(overlap))

    if overlap:
        raise RuntimeError("Patient overlap detected; training stopped.")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = make_model().to(DEVICE)

    counts = np.bincount(train_ds.labels, minlength=2)
    weights = len(train_ds) / (2.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )

    # Initial phase: train classifier only.
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            print("Unfreezing final EfficientNet feature block.")
            for p in model.features.parameters():
                p.requires_grad = False
            for p in model.features[-1].parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW([
                {
                    "params": [
                        p for p in model.features[-1].parameters()
                        if p.requires_grad
                    ],
                    "lr": 1e-5,
                },
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ], weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

        train_m, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_m, _, _, _ = run_epoch(
            model, val_loader, criterion
        )
        scheduler.step(val_m["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_recall": val_m["malignant_recall"],
            "val_specificity": val_m["specificity"],
            "val_f1": val_m["f1"],
            "val_f2": val_m["f2"],
            "val_auc": val_m["roc_auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_m['loss']:.4f} | "
            f"train_acc={train_m['accuracy']:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"val_acc={val_m['accuracy']:.4f} | "
            f"recall={val_m['malignant_recall']:.4f} | "
            f"specificity={val_m['specificity']:.4f} | "
            f"AUC={val_m['roc_auc']:.4f}"
        )

        if val_m["loss"] < best_loss - 1e-4:
            best_loss = val_m["loss"]
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "model_name": "EfficientNet-B0",
                "epoch": epoch,
                "best_val_loss": best_loss,
                "class_mapping": {"BENIGN": 0, "MALIGNANT": 1},
                "image_size": 224,
            }, CHECKPOINT)
        else:
            stale += 1

        if stale >= PATIENCE:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(HISTORY, index=False)

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_m, y, probs, paths = run_epoch(model, val_loader, criterion)
    pred = (probs >= 0.5).astype(int)

    pd.DataFrame({
        "image_path": paths,
        "true_label": y,
        "true_class": [CLASSES[i] for i in y],
        "malignant_probability": probs,
        "predicted_label_t050": pred,
        "predicted_class_t050": [CLASSES[i] for i in pred],
    }).to_csv(VAL_PREDS, index=False)

    report = {
        "model": "EfficientNet-B0",
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "validation_metrics_threshold_050": val_m,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "patient_overlap": len(overlap),
        "test_set_used": False,
        "clinical_use": "Research prototype only; not for diagnosis.",
    }
    VAL_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTRAINING COMPLETE")
    print("Best epoch:", best_epoch)
    print(json.dumps(report, indent=2))
    print("Checkpoint:", CHECKPOINT)
    print("History:", HISTORY)
    print("Validation report:", VAL_REPORT)
    print("Validation predictions:", VAL_PREDS)


if __name__ == "__main__":
    main()