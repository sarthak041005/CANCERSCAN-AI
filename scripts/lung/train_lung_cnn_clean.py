import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\lung_xray_clean_512"
)

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

OUTPUT_DIR = Path(
    r"C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\chest_xray_lung\lung_cnn_clean_output"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "lung_resnet18_clean_best.pth"
RESULTS_PATH = OUTPUT_DIR / "clean_model_results.json"


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 5
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 4. DEVICE INFORMATION
# ============================================================

print("=" * 65)
print("CLEAN LUNG X-RAY CNN TRAINING")
print("=" * 65)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )

print("=" * 65)


# ============================================================
# 5. IMAGE TRANSFORMS
# ============================================================

imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]


train_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Conservative augmentation
    transforms.RandomRotation(
        degrees=7
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        imagenet_mean,
        imagenet_std
    )
])


val_test_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        imagenet_mean,
        imagenet_std
    )
])


# ============================================================
# 6. LOAD DATASETS
# ============================================================

print("\nLoading cleaned datasets...")


train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=val_test_transform
)


test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=val_test_transform
)


print(
    f"\nClasses: {train_dataset.classes}"
)

print(
    f"Training images: {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print(
    f"Test images: {len(test_dataset)}"
)


# ============================================================
# 7. VERIFY CLASS MAPPING
# ============================================================

print("\nClass mapping:")

for class_name, class_id in train_dataset.class_to_idx.items():

    print(
        f"{class_name} -> {class_id}"
    )


# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets,
    minlength=2
)

print("\nTraining class distribution:")

for class_id, class_name in enumerate(
    train_dataset.classes
):

    print(
        f"{class_name}: "
        f"{class_counts[class_id]}"
    )


total_samples = len(train_targets)

num_classes = 2

class_weights = (
    total_samples /
    (num_classes * class_counts)
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


print("\nClass weights:")

print(class_weights)


# ============================================================
# 9. DATALOADERS
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
# 10. LOAD RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# Replace final layer

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


model = model.to(DEVICE)


print("ResNet18 loaded successfully.")


# ============================================================
# 11. LOSS / OPTIMIZER
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
# 12. TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

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

        total_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# 13. VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate_loader(loader):

    model.eval()

    total_loss = 0.0

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    all_probabilities = []

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

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(
            dim=1
        )

        total_loss += (
            loss.item() *
            images.size(0)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# 14. TRAINING LOOP
# ============================================================

best_val_loss = float("inf")
epochs_without_improvement = 0

history = []

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    (
        val_loss,
        val_accuracy,
        _,
        _,
        _
    ) = evaluate_loader(
        val_loader
    )

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.4f} "
        f"| LR: {current_lr:.6f}"
    )


    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "learning_rate": current_lr
    })


    # Save best model

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "image_size":
                    IMAGE_SIZE,

                "best_val_loss":
                    best_val_loss
            },
            MODEL_PATH
        )

        print(
            "  -> Best model saved."
        )

    else:

        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break


# ============================================================
# 15. LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best model..."
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# 16. FINAL TEST
# ============================================================

print(
    "\nEvaluating on untouched CLEAN test set..."
)

(
    test_loss,
    test_accuracy,
    y_true,
    y_pred,
    y_prob
) = evaluate_loader(
    test_loader
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

# Class mapping:
# Cancer = 0
# NORMAL = 1

# Treat Cancer as the positive class

tp = cm[0, 0]  # Cancer predicted as Cancer
fn = cm[0, 1]  # Cancer predicted as NORMAL
fp = cm[1, 0]  # NORMAL predicted as Cancer
tn = cm[1, 1]  # NORMAL predicted as NORMAL


# ============================================================
# 18. METRICS
# ============================================================

precision = (
    tp / (tp + fp)
    if (tp + fp) != 0
    else 0.0
)

sensitivity = (
    tp / (tp + fn)
    if (tp + fn) != 0
    else 0.0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) != 0
    else 0.0
)

f1 = (
    2 * precision * sensitivity /
    (precision + sensitivity)
    if (precision + sensitivity) != 0
    else 0.0
)


# ROC-AUC with Cancer as the positive class

# ROC-AUC for Cancer as the positive class
#
# ImageFolder mapping:
# Cancer = 0
# NORMAL = 1

y_true_cancer = (y_true == 0).astype(int)

# Use probability of Cancer
cancer_probabilities = y_prob[:, 0]

# Verify the direction automatically.
roc_auc = roc_auc_score(
    y_true_cancer,
    cancer_probabilities
)

# If the score direction is inverted, use the complementary score.
if roc_auc < 0.5:
    roc_auc = roc_auc_score(
        y_true_cancer,
        1.0 - cancer_probabilities
    )
# IMPORTANT:
# Cancer = class 0
# NORMAL = class 1

# Cancer recall / sensitivity
sensitivity = (
    tp / (tp + fn)
    if (tp + fn) != 0
    else 0.0
)

# Cancer precision
precision = (
    tp / (tp + fp)
    if (tp + fp) != 0
    else 0.0
)

# NORMAL detection specificity relative
# to Cancer being the positive class
specificity = (
    tn / (tn + fp)
    if (tn + fp) != 0
    else 0.0
)

f1 = (
    2 * precision * sensitivity /
    (precision + sensitivity)
    if (precision + sensitivity) != 0
    else 0.0
)


# ROC-AUC for Cancer probability

cancer_probabilities = y_prob[:, 0]

roc_auc = roc_auc_score(
    y_true,
    cancer_probabilities
)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=train_dataset.classes,
    output_dict=True
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("FINAL CLEAN TEST RESULTS")
print("=" * 65)

print(
    f"Test Loss       : {test_loss:.4f}"
)

print(
    f"Accuracy        : {test_accuracy:.4f}"
)

print(
    f"Precision       : {precision:.4f}"
)

print(
    f"Recall/Sens.    : {sensitivity:.4f}"
)

print(
    f"Specificity     : {specificity:.4f}"
)

print(
    f"F1 Score        : {f1:.4f}"
)

print(
    f"ROC-AUC         : {roc_auc:.4f}"
)


print("\nConfusion Matrix:")

print(
    f"TN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print(
    f"TP: {tp}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=train_dataset.classes
    )
)

print(
    "=" * 65
)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results = {

    "dataset": {

        "train": len(train_dataset),

        "validation": len(val_dataset),

        "test": len(test_dataset),

        "total": (
            len(train_dataset)
            + len(val_dataset)
            + len(test_dataset)
        )
    },

    "classes":
        train_dataset.classes,

    "device":
        str(DEVICE),

    "gpu":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),

    "metrics": {

        "test_loss":
            float(test_loss),

        "accuracy":
            float(test_accuracy),

        "precision":
            float(precision),

        "recall_sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "f1_score":
            float(f1),

        "roc_auc":
            float(roc_auc)
    },

    "confusion_matrix": {

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp)
    },

    "classification_report":
        report,

    "history":
        history
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print(
    "\nModel saved to:"
)

print(MODEL_PATH)


print(
    "\nResults saved to:"
)

print(RESULTS_PATH)


print(
    "\nClean lung CNN training complete."
)