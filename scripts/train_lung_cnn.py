import os
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\chest_xray_lung")

TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

OUTPUT_DIR = DATA_ROOT / "lung_cnn_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "lung_resnet18_best.pth"
RESULTS_PATH = OUTPUT_DIR / "training_results.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

VALIDATION_SIZE = 0.10
RANDOM_STATE = 42

PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("LUNG CANCER CNN TRAINING")
print("=" * 60)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

print("=" * 60)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# ============================================================
# 3. IMAGE TRANSFORMS
# ============================================================

# ImageNet normalization because we are using a pretrained ResNet18.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Conservative augmentation for medical images.
    transforms.RandomRotation(degrees=7),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

base_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR)
)

print(f"Classes: {base_dataset.classes}")
print(f"Total training images: {len(base_dataset)}")

# Expected:
# Cancer -> class 0
# NORMAL -> class 1

targets = np.array(base_dataset.targets)

indices = np.arange(len(base_dataset))

train_indices, val_indices = train_test_split(
    indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=targets
)

print(f"\nTraining images: {len(train_indices)}")
print(f"Validation images: {len(val_indices)}")


# ============================================================
# 5. CREATE SEPARATE TRAIN / VALIDATION DATASETS
# ============================================================

train_full = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform
)

val_full = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=val_test_transform
)

train_dataset = Subset(
    train_full,
    train_indices
)

val_dataset = Subset(
    val_full,
    val_indices
)


# ============================================================
# 6. TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=val_test_transform
)

print(f"Test images: {len(test_dataset)}")


# ============================================================
# 7. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 8. CLASS DISTRIBUTION / CLASS WEIGHTS
# ============================================================

train_targets = targets[train_indices]

class_counts = np.bincount(
    train_targets,
    minlength=len(base_dataset.classes)
)

print("\nTraining class distribution:")

for class_id, class_name in enumerate(base_dataset.classes):
    print(
        f"{class_name}: "
        f"{class_counts[class_id]}"
    )

# Balanced class weights:
# total / (number_of_classes * class_count)

total_train = len(train_targets)
num_classes = len(base_dataset.classes)

class_weights = total_train / (
    num_classes * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass weights:")
print(class_weights)


# ============================================================
# 9. LOAD PRETRAINED RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Replace final classification layer.
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(DEVICE)

print("ResNet18 loaded successfully.")


# ============================================================
# 10. LOSS FUNCTION + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 11. TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# 12. TRAINING LOOP
# ============================================================

best_val_loss = float("inf")
epochs_without_improvement = 0

history = []

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_acc, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_acc:.4f} "
        f"| LR: {current_lr:.6f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": current_lr
    })


    # Save best model.
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": base_dataset.classes,
                "image_size": IMAGE_SIZE,
                "best_val_loss": best_val_loss
            },
            MODEL_PATH
        )

        print("  -> Best model saved.")

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break


# ============================================================
# 13. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating on untouched test set...")

test_loss, test_accuracy, y_true, y_pred = evaluate(
    model,
    test_loader,
    criterion
)


# ============================================================
# 15. CONFUSION MATRIX VALUES
# ============================================================

# Class 0 = Cancer
# Class 1 = NORMAL

tp = np.sum(
    (y_true == 0) & (y_pred == 0)
)

fn = np.sum(
    (y_true == 0) & (y_pred == 1)
)

tn = np.sum(
    (y_true == 1) & (y_pred == 1)
)

fp = np.sum(
    (y_true == 1) & (y_pred == 0)
)


def safe_divide(a, b):
    return a / b if b != 0 else 0.0


precision = safe_divide(
    tp,
    tp + fp
)

recall = safe_divide(
    tp,
    tp + fn
)

specificity = safe_divide(
    tn,
    tn + fp
)

f1 = safe_divide(
    2 * precision * recall,
    precision + recall
)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss       : {test_loss:.4f}")
print(f"Accuracy        : {test_accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall/Sens.    : {recall:.4f}")
print(f"Specificity     : {specificity:.4f}")
print(f"F1 Score        : {f1:.4f}")

print("\nConfusion Matrix:")
print(f"True Positive   : {tp}")
print(f"False Negative  : {fn}")
print(f"True Negative   : {tn}")
print(f"False Positive  : {fp}")

print("=" * 60)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results = {
    "device": str(DEVICE),
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),

    "classes": base_dataset.classes,

    "dataset": {
        "train_original": len(base_dataset),
        "train_used": len(train_dataset),
        "validation": len(val_dataset),
        "test": len(test_dataset)
    },

    "metrics": {
        "test_loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1)
    },

    "confusion_matrix": {
        "TP": int(tp),
        "FN": int(fn),
        "TN": int(tn),
        "FP": int(fp)
    },

    "history": history
}

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(f"\nModel saved to:")
print(MODEL_PATH)

print(f"\nResults saved to:")
print(RESULTS_PATH)

print("\nTraining complete.")