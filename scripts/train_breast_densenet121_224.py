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
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
IMAGE_ROOT = DATA / "breast_preprocessed_224"
MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "breast_densenet121_224_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_densenet121_224_history.json"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25
FROZEN_EPOCHS = 3
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
SEED = 42
PATIENCE = 7
NUM_WORKERS = 0

def main():
    if MODEL_PATH.exists() or HISTORY_PATH.exists():
        raise FileExistsError(
            "Output already exists; refusing to overwrite:\n"
            f"{MODEL_PATH}\n{HISTORY_PATH}"
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

    for name in ["train.csv", "val.csv", "test.csv"]:
        if not (DATA / name).is_file():
            raise FileNotFoundError(f"Required split is missing: {DATA / name}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(5),
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
            df = pd.read_csv(DATA / csv_name)
            required = {"patient_id", "image_path", "pathology"}
            if not required.issubset(df.columns):
                raise ValueError(
                    f"{csv_name} must contain columns {sorted(required)}"
                )

            self.records = []
            self.transform = transform

            for _, row in df.iterrows():
                pathology = str(row["pathology"]).strip().upper()
                if pathology == "MALIGNANT":
                    label = 1
                elif pathology in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
                    label = 0
                else:
                    raise ValueError(f"Unexpected pathology: {pathology}")

                parts = str(row["image_path"]).replace("\\", "/").split("/")
                if len(parts) < 2:
                    raise ValueError(f"Invalid image path: {row['image_path']}")

                path = IMAGE_ROOT / parts[-2] / parts[-1]
                if not path.is_file():
                    raise FileNotFoundError(path)

                self.records.append({
                    "path": path,
                    "label": label,
                    "patient": str(row["patient_id"])
                })

            if not self.records:
                raise ValueError(f"No records found in {csv_name}")

            print(
                f"{csv_name}: {len(self.records)} images, "
                f"{len(set(r['patient'] for r in self.records))} patients, "
                f"labels={pd.Series([r['label'] for r in self.records]).value_counts().to_dict()}"
            )

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            r = self.records[index]
            with Image.open(r["path"]) as image:
                image = image.convert("RGB")
            return self.transform(image), r["label"]

    train_ds = BreastDataset("train.csv", train_transform)
    val_ds = BreastDataset("val.csv", eval_transform)

    # Verify patient separation across train, validation, and test.
    def patient_ids(csv_name):
        df = pd.read_csv(DATA / csv_name)
        return set(df["patient_id"].dropna().astype(str))

    train_patients = patient_ids("train.csv")
    val_patients = patient_ids("val.csv")
    test_patients = patient_ids("test.csv")

    for name, overlap in [
        ("train/validation", train_patients & val_patients),
        ("train/test", train_patients & test_patients),
        ("validation/test", val_patients & test_patients),
    ]:
        print(f"Patient overlap {name}: {len(overlap)}")
        if overlap:
            raise RuntimeError(
                f"Patient leakage detected in {name}; training stopped."
            )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    print("Loading ImageNet-pretrained DenseNet121...")
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 2)
    )
    model = model.to(device)

    # First train only the new classification head.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=HEAD_LR, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {
        "experiment": "ImageNet-pretrained DenseNet121",
        "architecture": "densenet121",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "frozen_epochs": FROZEN_EPOCHS,
        "labels": {"BENIGN": 0, "MALIGNANT": 1},
        "seed": SEED,
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "test_evaluated": False,
        "epochs": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze for fine-tuning after the classifier warm-up.
        if epoch == FROZEN_EPOCHS + 1:
            print("\nUnfreezing DenseNet121 features for fine-tuning.")
            for parameter in model.features.parameters():
                parameter.requires_grad = True

            optimizer = torch.optim.AdamW([
                {"params": model.features.parameters(), "lr": FINE_TUNE_LR},
                {"params": model.classifier.parameters(), "lr": HEAD_LR / 10},
            ], weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

        model.train()
        # Keep frozen pretrained BatchNorm statistics stable during warm-up.
        if epoch <= FROZEN_EPOCHS:
            model.features.eval()
            model.classifier.train()

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
            train_true.extend(labels.cpu().tolist())
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
                probabilities = torch.softmax(logits, dim=1)[:, 1]

                val_loss_sum += loss.item() * labels.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())
                val_probs.extend(probabilities.cpu().tolist())

        val_loss = val_loss_sum / len(val_ds)
        val_acc = accuracy_score(val_true, val_pred)
        val_precision = precision_score(val_true, val_pred, zero_division=0)
        val_recall = recall_score(val_true, val_pred, zero_division=0)
        val_f1 = f1_score(val_true, val_pred, zero_division=0)
        val_auc = roc_auc_score(val_true, val_probs)

        scheduler.step(val_loss)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": float(train_acc),
            "val_loss": val_loss,
            "val_accuracy": float(val_acc),
            "val_precision_malignant": float(val_precision),
            "val_recall_malignant": float(val_recall),
            "val_f1_malignant": float(val_f1),
            "val_roc_auc": float(val_auc)
        }
        history["epochs"].append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"malignant recall {val_recall:.4f} "
            f"F1 {val_f1:.4f} AUC {val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "densenet121",
                "pretraining": "ImageNet",
                "image_size": IMAGE_SIZE,
                "labels": {"BENIGN": 0, "MALIGNANT": 1},
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_accuracy": float(val_acc),
                "val_roc_auc": float(val_auc),
                "seed": SEED
            }, MODEL_PATH)
        else:
            epochs_without_improvement += 1

        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        history["elapsed_seconds"] = time.time() - start_time
        HISTORY_PATH.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping: no validation-loss improvement for {PATIENCE} epochs.")
            break

    print("\nDenseNet121 training finished.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 5))
    print("Checkpoint:", MODEL_PATH)
    print("History:", HISTORY_PATH)
    print("Independent test set was not evaluated.")

if __name__ == "__main__":
    main()
