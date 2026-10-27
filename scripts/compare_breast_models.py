from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "mnt" / "data" / "CancerScan_AI" / "datasets"
MODELS = BASE / "mnt" / "data" / "CancerScan_AI" / "models"
SAVED_MODELS = BASE / "mnt" / "data" / "CancerScan_AI" / "saved_models"

TEST_CSV = DATASETS / "test.csv"
BASELINE_CHECKPOINT = SAVED_MODELS / "breast_cnn.pth"
SSL_CHECKPOINT = MODELS / "breast_ssl_finetuned.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16

CLASS_NAMES = ["BENIGN", "MALIGNANT"]


class BreastDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = Path(row["image_path"])

        if not image_path.exists():
            relative_path = (
                str(row["image_path"])
                .replace("\\", "/")
                .replace("CBIS-DDSM/jpeg/", "")
            )

            image_path = (
                DATASETS / "breast_preprocessed" / relative_path
            )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])

        return image, label


class BaselineModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = resnet18(weights=None)

        n = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.50),
            nn.Linear(n, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class SSLModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = resnet18(weights=None)

        n = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.50),
            nn.Linear(n, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def load_baseline_model():
    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = BaselineModel(num_classes=2)

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model

def load_ssl_model():
    checkpoint = torch.load(
        SSL_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = SSLModel(num_classes=2)

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("encoder."):
            new_key = "backbone." + key[len("encoder."):]
            new_state_dict[new_key] = value

        elif key.startswith("classifier."):
            new_key = "backbone.fc." + key[len("classifier."):]
            new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def evaluate_model(model, loader):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def print_results(name, results):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Precision:    {results['precision']:.4f}")
    print(f"Sensitivity:  {results['recall']:.4f}")
    print(f"Specificity:  {results['specificity']:.4f}")
    print(f"F1 Score:     {results['f1']:.4f}")
    print(f"ROC-AUC:      {results['roc_auc']:.4f}")

    print()
    print("Confusion Matrix:")
    print("[[TN, FP],")
    print(" [FN, TP]]")
    print(
        f"[[{results['tn']}, {results['fp']}],"
        f" [{results['fn']}, {results['tp']}]]"
    )


def main():
    print("=" * 60)
    print("BREAST CANCER MODEL COMPARISON")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    test_df = pd.read_csv(TEST_CSV)

    print(f"Test records: {len(test_df)}")

    if "label" not in test_df.columns:
        raise ValueError(
            "test.csv must contain a 'label' column."
        )

    print()
    print("Test class distribution:")
    print(test_df["label"].value_counts().sort_index())

    test_dataset = BreastDataset(test_df)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print()
    print("Loading baseline model...")

    baseline_model = load_baseline_model()

    print("Evaluating baseline model...")

    baseline_results = evaluate_model(
        baseline_model,
        test_loader
    )

    print_results(
        "BASELINE: ImageNet-Pretrained ResNet18",
        baseline_results
    )

    print()
    print("Loading SSL fine-tuned model...")

    ssl_model = load_ssl_model()

    print("Evaluating SSL model...")

    ssl_results = evaluate_model(
        ssl_model,
        test_loader
    )

    print_results(
        "SSL: SimSiam Pretraining + Fine-Tuning",
        ssl_results
    )

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1 Score",
            "ROC-AUC",
        ],
        "Baseline": [
            baseline_results["accuracy"],
            baseline_results["precision"],
            baseline_results["recall"],
            baseline_results["specificity"],
            baseline_results["f1"],
            baseline_results["roc_auc"],
        ],
        "SSL": [
            ssl_results["accuracy"],
            ssl_results["precision"],
            ssl_results["recall"],
            ssl_results["specificity"],
            ssl_results["f1"],
            ssl_results["roc_auc"],
        ],
    })

    comparison["SSL - Baseline"] = (
        comparison["SSL"] - comparison["Baseline"]
    )

    print(comparison.to_string(index=False))

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()