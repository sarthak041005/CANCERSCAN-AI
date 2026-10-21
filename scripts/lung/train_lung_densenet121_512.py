from pathlib import Path
import copy
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
DATA_DIR = Path(
    r"C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\lung_xray_clean_512"
)

OUTPUT_DIR = Path("saved_models/lung")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.0001
NUM_WORKERS = 0  # Windows-friendly setting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = OUTPUT_DIR / "densenet121_512_best.pth"

print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------
weights = DenseNet121_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=7),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# --------------------------------------------------
# 3. Load existing train/val/test folders
# --------------------------------------------------
train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
test_dir = DATA_DIR / "test"

for folder in (train_dir, val_dir, test_dir):
    if not folder.exists():
        raise FileNotFoundError(
            f"Missing folder: {folder}\n"
            "Check that your dataset contains train, val, and test folders."
        )

train_dataset = datasets.ImageFolder(
    train_dir, transform=train_transform
)
val_dataset = datasets.ImageFolder(
    val_dir, transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    test_dir, transform=eval_transform
)

classes = train_dataset.classes
print("Classes:", classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

if val_dataset.classes != classes or test_dataset.classes != classes:
    raise ValueError("Class folders differ between dataset splits.")

if "Cancer" not in classes:
    raise ValueError("Expected a class folder named 'Cancer'.")

cancer_idx = classes.index("Cancer")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

# --------------------------------------------------
# 4. Create pretrained DenseNet121
# --------------------------------------------------
model = models.densenet121(weights=weights)

input_features = model.classifier.in_features
model.classifier = nn.Linear(input_features, len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001,
)

# Mixed precision reduces GPU memory usage on supported GPUs.
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# --------------------------------------------------
# 5. Training and evaluation
# --------------------------------------------------
def run_epoch(loader, training=False):
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(predictions.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds,
        pos_label=cancer_idx, zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds,
        pos_label=cancer_idx, zero_division=0
    )
    f1 = f1_score(
        all_labels, all_preds,
        pos_label=cancer_idx, zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision_cancer": precision,
        "recall_cancer": recall,
        "f1_cancer": f1,
        "labels": all_labels,
        "predictions": all_preds,
    }


# --------------------------------------------------
# 6. Train; select checkpoint by validation F1
# --------------------------------------------------
best_val_f1 = -1.0
best_state = None

for epoch in range(EPOCHS):
    start = time.time()

    train_metrics = run_epoch(train_loader, training=True)
    val_metrics = run_epoch(val_loader, training=False)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(
        f"Train loss: {train_metrics['loss']:.4f} | "
        f"Train accuracy: {train_metrics['accuracy']:.4f}"
    )
    print(
        f"Val loss: {val_metrics['loss']:.4f} | "
        f"Val accuracy: {val_metrics['accuracy']:.4f}"
    )
    print(
        f"Val Cancer precision: {val_metrics['precision_cancer']:.4f} | "
        f"recall: {val_metrics['recall_cancer']:.4f} | "
        f"F1: {val_metrics['f1_cancer']:.4f}"
    )
    print(f"Epoch time: {time.time() - start:.1f}s")

    if val_metrics["f1_cancer"] > best_val_f1:
        best_val_f1 = val_metrics["f1_cancer"]
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "architecture": "densenet121",
            "best_val_f1_cancer": best_val_f1,
        }, BEST_MODEL_PATH)

        print("Saved best checkpoint:", BEST_MODEL_PATH)

# --------------------------------------------------
# 7. Final evaluation on the held-out test set
# --------------------------------------------------
checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = run_epoch(test_loader, training=False)
cm = confusion_matrix(
    test_metrics["labels"],
    test_metrics["predictions"],
    labels=list(range(len(classes))),
)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Test loss: {test_metrics['loss']:.4f}")
print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
print(f"Cancer precision: {test_metrics['precision_cancer']:.4f}")
print(f"Cancer recall/sensitivity: {test_metrics['recall_cancer']:.4f}")
print(f"Cancer F1-score: {test_metrics['f1_cancer']:.4f}")
print("\nClass order:", classes)
print("Confusion matrix (rows=true, columns=predicted):")
print(cm)

results = {
    "architecture": "DenseNet121",
    "image_size": IMAGE_SIZE,
    "classes": classes,
    "test_loss": test_metrics["loss"],
    "test_accuracy": test_metrics["accuracy"],
    "cancer_precision": test_metrics["precision_cancer"],
    "cancer_recall_sensitivity": test_metrics["recall_cancer"],
    "cancer_f1": test_metrics["f1_cancer"],
    "confusion_matrix": cm.tolist(),
}

results_path = OUTPUT_DIR / "densenet121_512_test_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved test metrics:", results_path)
print("Training and evaluation complete.")

