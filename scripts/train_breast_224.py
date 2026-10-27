from pathlib import Path
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
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
IMAGE_ROOT = DATA / "breast_preprocessed_224"
MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "breast_resnet18_224_baseline.pth"
HISTORY_PATH = REPORT_DIR / "breast_resnet18_224_baseline.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
SEED = 42
NUM_WORKERS = 0

def main():
    # Refuse to overwrite previous results.
    if MODEL_PATH.exists() or HISTORY_PATH.exists():
        raise FileExistsError(
            "Output already exists. No files were changed:\n"
            f"{MODEL_PATH}\n{HISTORY_PATH}\n"
            "Choose new output filenames before running again."
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ImageNet normalization is used with ImageNet-pretrained ResNet18.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    class BreastDataset(Dataset):
        def __init__(self, csv_name, transform):
            self.csv_path = DATA / csv_name
            self.df = pd.read_csv(self.csv_path)
            self.transform = transform
            self.records = []

            required = {"patient_id", "image_path", "pathology"}
            missing_columns = required - set(self.df.columns)
            if missing_columns:
                raise ValueError(
                    f"{csv_name} is missing columns: {sorted(missing_columns)}"
                )

            for _, row in self.df.iterrows():
                pathology = str(row["pathology"]).strip().upper()

                if pathology == "MALIGNANT":
                    label = 1
                elif pathology in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
                    label = 0
                else:
                    raise ValueError(
                        f"Unknown pathology in {csv_name}: {pathology}"
                    )

                parts = str(row["image_path"]).replace("\\", "/").split("/")
                if len(parts) < 2:
                    raise ValueError(
                        f"Unexpected image path in {csv_name}: "
                        f"{row['image_path']}"
                    )

                image_path = IMAGE_ROOT / parts[-2] / parts[-1]
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found for {csv_name}: {image_path}"
                    )

                self.records.append({
                    "path": image_path,
                    "label": label,
                    "patient_id": str(row["patient_id"]),
                })

            if not self.records:
                raise ValueError(f"No usable images found in {csv_name}")

            print(
                f"{csv_name}: {len(self.records)} images; "
                f"{len(set(r['patient_id'] for r in self.records))} patients; "
                f"labels={pd.Series([r['label'] for r in self.records]).value_counts().to_dict()}"
            )

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            record = self.records[idx]
            with Image.open(record["path"]) as im:
                image = im.convert("RGB")
            image = self.transform(image)
            return image, record["label"]

    train_ds = BreastDataset("train.csv", train_transform)
    val_ds = BreastDataset("val.csv", eval_transform)

    # Check patient separation before training.
    train_patients = {r["patient_id"] for r in train_ds.records}
    val_patients = {r["patient_id"] for r in val_ds.records}
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(
            f"Patient overlap detected between train and validation: "
            f"{len(overlap)} patients. Training stopped."
        )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    # ImageNet-pretrained baseline; this experiment does not load SSL weights.
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {
        "experiment": "ImageNet-pretrained ResNet18 baseline",
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "labels": {"BENIGN": 0, "MALIGNANT": 1},
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "patient_overlap_train_val": 0,
        "epochs": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_true, train_pred = [], []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_true.extend(labels.detach().cpu().tolist())
            train_pred.extend(logits.argmax(1).detach().cpu().tolist())

        train_loss = train_loss_sum / len(train_ds)
        train_acc = accuracy_score(train_true, train_pred)

        model.eval()
        val_loss_sum = 0.0
        val_true, val_pred, val_probs = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 1]

                val_loss_sum += loss.item() * labels.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())
                val_probs.extend(probs.cpu().tolist())

        val_loss = val_loss_sum / len(val_ds)
        val_acc = accuracy_score(val_true, val_pred)
        val_precision = precision_score(
            val_true, val_pred, zero_division=0
        )
        val_recall = recall_score(val_true, val_pred, zero_division=0)
        val_f1 = f1_score(val_true, val_pred, zero_division=0)
        val_auc = roc_auc_score(val_true, val_probs)

        scheduler.step(val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": float(train_acc),
            "val_loss": val_loss,
            "val_accuracy": float(val_acc),
            "val_precision_malignant": float(val_precision),
            "val_recall_malignant": float(val_recall),
            "val_f1_malignant": float(val_f1),
            "val_roc_auc": float(val_auc),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history["epochs"].append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"malignant recall {val_recall:.4f} F1 {val_f1:.4f} "
            f"AUC {val_auc:.4f}"
        )

        # Save the best checkpoint based on validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "resnet18",
                "pretraining": "ImageNet",
                "image_size": IMAGE_SIZE,
                "labels": {"BENIGN": 0, "MALIGNANT": 1},
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_accuracy": float(val_acc),
                "val_roc_auc": float(val_auc),
                "seed": SEED,
            }, MODEL_PATH)

        # Persist progress after every epoch.
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        history["elapsed_seconds"] = time.time() - start_time
        HISTORY_PATH.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 5))
    print("Checkpoint:", MODEL_PATH)
    print("History:", HISTORY_PATH)
    print("Test data was not loaded or evaluated.")
    print("Use a separate, one-time test evaluation after model selection.")

if __name__ == "__main__":
    main()
