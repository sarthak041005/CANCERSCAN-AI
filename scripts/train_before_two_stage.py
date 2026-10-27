import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from config import (
    CONFIGS,
    DEVICE,
    IMAGE_SIZE,
    NUM_WORKERS,
    REPORT_DIR
)

from models.cnn import CancerCNN


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Train breast/lung cancer CNN model"
)

parser.add_argument(
    "--cancer",
    choices=["breast", "lung"],
    required=True,
    help="Cancer dataset to train on"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="Number of training epochs"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="Training batch size"
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate"
)

parser.add_argument(
    "--ssl",
    action="store_true",
    help="Use breast SimSiam SSL initialization"
)

parser.add_argument(
    "--freeze-backbone",
    action="store_true",
    help="Freeze the ResNet18 backbone and train only the classifier"
)

args = parser.parse_args()


# ============================================================
# BASIC SETTINGS
# ============================================================

cancer = args.cancer

cfg = CONFIGS[cancer]

dataset_dir = Path(cfg["dataset_dir"])

train_dir = dataset_dir / "train"
val_dir = dataset_dir / "val"

if args.ssl and cancer != "breast":
    raise ValueError(
        "The --ssl option is currently available only for breast cancer."
    )

use_ssl = args.ssl

print("=" * 60)

if use_ssl:
    print("BREAST CANCER MODEL - SSL FINE-TUNING")
else:
    print("BREAST CANCER MODEL TRAINING")

print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Learning rate: {args.lr}")

if use_ssl:
    print("Initialization: Breast SimSiam SSL")
else:
    print("Initialization: ImageNet ResNet18")

if args.freeze_backbone:
    print("Backbone: FROZEN")
    print("Classifier: TRAINABLE")


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=val_transform
)

print()
print(f"Classes: {train_dataset.classes}")
print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_dataset.targets,
    minlength=len(train_dataset.classes)
).astype(np.float32)

class_weights = (
    class_counts.sum() /
    (len(class_counts) * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print(f"Class counts: {class_counts.tolist()}")
print(f"Class weights: {class_weights.detach().cpu().tolist()}")


# ============================================================
# MODEL
# ============================================================

model = CancerCNN(
    len(train_dataset.classes),
    pretrained=not use_ssl
).to(DEVICE)


# ============================================================
# SSL ENCODER LOADING
# ============================================================

if use_ssl:

    ssl_path = (
        Path(__file__).resolve().parent
        / "models"
        / "breast_simsiam.pth"
    )

    if not ssl_path.exists():
        raise FileNotFoundError(
            f"SSL checkpoint not found:\n{ssl_path}"
        )

    checkpoint = torch.load(
        ssl_path,
        map_location=DEVICE
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "SSL checkpoint is not a dictionary."
        )

    if "model_state" not in checkpoint:
        raise KeyError(
            "SSL checkpoint does not contain 'model_state'."
        )

    ssl_state = checkpoint["model_state"]

    encoder_state = {}

    for key, value in ssl_state.items():

        if key.startswith("encoder."):

            new_key = key.replace(
                "encoder.",
                "backbone.",
                1
            )

            encoder_state[new_key] = value

    load_result = model.load_state_dict(
        encoder_state,
        strict=False
    )

    print()
    print("=" * 60)
    print("SSL ENCODER LOADED")
    print("=" * 60)

    print(f"Checkpoint: {ssl_path}")

    if "epoch" in checkpoint:
        print(
            f"SSL epoch: {checkpoint['epoch']}"
        )

    if "loss" in checkpoint:
        print(
            f"SSL loss: {checkpoint['loss']:.4f}"
        )

    print(
        f"Encoder tensors loaded: "
        f"{len(encoder_state)}"
    )

    print(
        f"Missing keys: "
        f"{len(load_result.missing_keys)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(load_result.unexpected_keys)}"
    )

    print("=" * 60)


# ============================================================
# FREEZE BACKBONE
# ============================================================

if args.freeze_backbone:

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    print()
    print("Backbone: FROZEN")
    print("Classifier: TRAINABLE")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=args.lr,
    weight_decay=5e-4
)


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None
):

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_training
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

            if is_training:
                loss.backward()

                optimizer.step()

        total_loss += (
            loss.item() *
            images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        if probabilities.shape[1] == 2:

            all_probabilities.extend(
                probabilities[:, 1]
                .detach()
                .cpu()
                .numpy()
            )

    avg_loss = (
        total_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    precision = precision_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    if len(np.unique(all_targets)) == 2:

        auc = roc_auc_score(
            all_targets,
            all_probabilities
        )

    else:
        auc = float("nan")

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc)
    }


# ============================================================
# OUTPUT PATHS
# ============================================================

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "saved_models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if use_ssl:

    model_path = (
        MODEL_DIR /
        f"{cancer}_cnn_ssl.pth"
    )

    history_path = (
        REPORT_DIR /
        f"{cancer}_training_history_ssl.json"
    )

else:

    model_path = (
        Path(cfg["model_path"])
    )

    history_path = (
        REPORT_DIR /
        f"{cancer}_training_history.json"
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": [],
    "val_auc": [],
    "learning_rate": []
}


# ============================================================
# EARLY STOPPING
# ============================================================

best_auc = -float("inf")

patience = 5

epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(1, args.epochs + 1):

    current_lr = optimizer.param_groups[0]["lr"]

    train_metrics = run_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_metrics = run_epoch(
        model,
        val_loader,
        criterion,
        optimizer=None
    )

    scheduler.step(
        val_metrics["auc"]
    )

    history["train_loss"].append(
        train_metrics["loss"]
    )

    history["val_loss"].append(
        val_metrics["loss"]
    )

    history["train_accuracy"].append(
        train_metrics["accuracy"]
    )

    history["val_accuracy"].append(
        val_metrics["accuracy"]
    )

    history["val_precision"].append(
        val_metrics["precision"]
    )

    history["val_recall"].append(
        val_metrics["recall"]
    )

    history["val_f1"].append(
        val_metrics["f1"]
    )

    history["val_auc"].append(
        val_metrics["auc"]
    )

    history["learning_rate"].append(
        current_lr
    )

    print(
        f"Epoch {epoch}/{args.epochs} | "
        f"LR {current_lr:.2e} | "
        f"train loss {train_metrics['loss']:.4f} "
        f"acc {train_metrics['accuracy']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} "
        f"acc {val_metrics['accuracy']:.4f} "
        f"recall {val_metrics['recall']:.4f} "
        f"F1 {val_metrics['f1']:.4f} "
        f"AUC {val_metrics['auc']:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_metrics["auc"] > best_auc:

        best_auc = val_metrics["auc"]

        torch.save(
            model.state_dict(),
            model_path
        )

        epochs_without_improvement = 0

        print(
            f"Saved best model: {model_path}"
        )

    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= patience:

        print(
            f"\nEarly stopping at epoch "
            f"{epoch}"
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# TRAINING PLOTS
# ============================================================

plot_suffix = "_ssl" if use_ssl else ""

loss_plot = (
    REPORT_DIR /
    f"{cancer}_loss{plot_suffix}.png"
)

accuracy_plot = (
    REPORT_DIR /
    f"{cancer}_accuracy{plot_suffix}.png"
)

auc_plot = (
    REPORT_DIR /
    f"{cancer}_auc{plot_suffix}.png"
)


# ------------------------------------------------------------
# Loss plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    f"{cancer.capitalize()} Training Loss"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    loss_plot,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Accuracy plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    f"{cancer.capitalize()} Training Accuracy"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    accuracy_plot,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# AUC plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history["val_auc"],
    label="Validation AUC"
)

plt.xlabel("Epoch")
plt.ylabel("ROC-AUC")
plt.title(
    f"{cancer.capitalize()} Validation AUC"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    auc_plot,
    dpi=150
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation AUC: {best_auc:.4f}"
)

print(
    f"Best model: {model_path}"
)

print(
    f"History: {history_path}"
)

print("=" * 60)
