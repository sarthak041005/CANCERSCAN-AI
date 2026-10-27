from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm


BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "mnt" / "data" / "CancerScan_AI" / "datasets"
MODEL_DIR = BASE / "mnt" / "data" / "CancerScan_AI" / "models"

TRAIN_CSV = DATASETS / "train.csv"
VAL_CSV = DATASETS / "val.csv"
SSL_CHECKPOINT = MODEL_DIR / "breast_simsiam.pth"
OUTPUT_CHECKPOINT = MODEL_DIR / "breast_ssl_finetuned.pth"

PREPROCESSED_DIR = DATASETS / "breast_preprocessed"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


class BreastDataset(Dataset):
    def __init__(self, csv_path, training=False):
        self.data = pd.read_csv(csv_path)
        self.training = training

        if training:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(8),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.12
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(
                    p=0.15,
                    scale=(0.01, 0.08),
                    ratio=(0.5, 2.0)
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.data)

    def get_image_path(self, image_path):
        raw_path = str(image_path).replace("\\", "/")

        marker = "CBIS-DDSM/jpeg/"

        if marker in raw_path:
            relative_path = raw_path.split(marker, 1)[1]
            return PREPROCESSED_DIR / Path(relative_path)

        marker = "jpeg/"

        if marker in raw_path:
            relative_path = raw_path.split(marker, 1)[1]
            return PREPROCESSED_DIR / Path(relative_path)

        direct_path = Path(raw_path)

        if direct_path.exists():
            return direct_path

        filename = Path(raw_path).name

        matches = list(PREPROCESSED_DIR.rglob(filename))

        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            f"Could not find preprocessed image for: {image_path}"
        )

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = self.get_image_path(row["image_path"])

        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, label


class SSLBreastModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.encoder = resnet18(weights=None)

        feature_dim = self.encoder.fc.in_features

        self.encoder.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.50),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


def load_ssl_encoder(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    ssl_state = checkpoint["model_state"]

    encoder_state = {}

    for key, value in ssl_state.items():
        if key.startswith("encoder."):
            new_key = key.replace("encoder.", "", 1)
            encoder_state[new_key] = value

    missing, unexpected = model.encoder.load_state_dict(
        encoder_state,
        strict=False
    )

    print(f"SSL checkpoint epoch: {checkpoint['epoch']}")
    print(f"SSL checkpoint loss: {checkpoint['loss']:.4f}")
    print(f"Encoder parameters loaded: {len(encoder_state)}")

    if missing:
        print(f"Missing encoder parameters: {len(missing)}")

    if unexpected:
        print(f"Unexpected encoder parameters: {len(unexpected)}")


def calculate_class_weights(dataset):
    labels = dataset.data["label"].astype(int).values

    counts = np.bincount(labels, minlength=2)

    weights = len(labels) / (
        2.0 * np.maximum(counts, 1)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def evaluate(model, loader, criterion):
    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1
    )


def main():
    print("=" * 60)
    print("SSL-PRETRAINED BREAST CANCER FINE-TUNING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Validation CSV: {VAL_CSV}")
    print(f"SSL checkpoint: {SSL_CHECKPOINT}")
    print()

    train_dataset = BreastDataset(
        TRAIN_CSV,
        training=True
    )

    val_dataset = BreastDataset(
        VAL_CSV,
        training=False
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    print()
    print("Training class distribution:")
    print(
        train_dataset.data["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Validation class distribution:")
    print(
        val_dataset.data["label"]
        .value_counts()
        .sort_index()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = SSLBreastModel(
        num_classes=2
    )

    load_ssl_encoder(
        model,
        SSL_CHECKPOINT
    )

    model = model.to(DEVICE)

    class_weights = calculate_class_weights(
        train_dataset
    ).to(DEVICE)

    print()
    print(
        "Class weights:",
        class_weights.detach().cpu().numpy()
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
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

    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0

    print()
    print("=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}"
            )

        train_loss = (
            running_loss /
            len(train_loader)
        )

        train_accuracy = correct / total

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        print(
            f"Train Loss: {train_loss:.4f}"
        )
        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f}"
        )
        print(
            f"Val Accuracy: {val_accuracy:.4f}"
        )
        print(
            f"Val Precision: {val_precision:.4f}"
        )
        print(
            f"Val Recall: {val_recall:.4f}"
        )
        print(
            f"Val F1: {val_f1:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "classes": [
                        "BENIGN",
                        "MALIGNANT"
                    ],
                    "image_size": IMAGE_SIZE
                },
                OUTPUT_CHECKPOINT
            )

            print(
                f"Saved best model: {OUTPUT_CHECKPOINT}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print()
            print(
                "Early stopping triggered."
            )
            break

    print()
    print("=" * 60)
    print("SSL FINE-TUNING COMPLETED")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {OUTPUT_CHECKPOINT}")
    print("=" * 60)


if __name__ == "__main__":
    main()