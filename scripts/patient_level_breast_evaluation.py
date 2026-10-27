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

            image_path = DATASETS / "breast_preprocessed" / relative_path

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, int(row["label"])


class BreastModel(nn.Module):
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

    model = BreastModel()

    state_dict = (
        checkpoint["model_state"]
        if "model_state" in checkpoint
        else checkpoint
    )

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

    model = BreastModel()

    state_dict = (
        checkpoint["model_state"]
        if "model_state" in checkpoint
        else checkpoint
    )

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


def get_image_predictions(model, dataframe):
    dataset = BreastDataset(dataframe)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    probabilities = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            probabilities.extend(
                probs[:, 1].cpu().numpy()
            )

    result = dataframe.copy().reset_index(drop=True)

    result["malignant_probability"] = probabilities

    return result


def aggregate_patient_predictions(image_results):
    patient_results = []

    for patient_id, group in image_results.groupby("patient_id"):
        malignant_probability = group[
            "malignant_probability"
        ].mean()

        label_values = group["label"].unique()

        if len(label_values) != 1:
            raise ValueError(
                f"Patient {patient_id} has multiple labels: "
                f"{label_values}"
            )

        true_label = int(label_values[0])

        prediction = int(malignant_probability >= 0.5)

        patient_results.append({
            "patient_id": patient_id,
            "true_label": true_label,
            "malignant_probability": malignant_probability,
            "prediction": prediction,
            "number_of_images": len(group),
        })

    return pd.DataFrame(patient_results)


def calculate_metrics(patient_results):
    y_true = patient_results["true_label"].to_numpy()
    y_pred = patient_results["prediction"].to_numpy()
    y_prob = patient_results["malignant_probability"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
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
    print(f"Sensitivity:  {results['sensitivity']:.4f}")
    print(f"Specificity:  {results['specificity']:.4f}")
    print(f"F1 Score:     {results['f1']:.4f}")
    print(f"ROC-AUC:      {results['roc_auc']:.4f}")

    print()
    print("Confusion Matrix:")
    print("[[TN, FP],")
    print(" [FN, TP]]")
    print(
        f"[[{results['tn']}, {results['fp']}], "
        f"[{results['fn']}, {results['tp']}]]"
    )


def main():
    print("=" * 60)
    print("PATIENT-LEVEL BREAST CANCER EVALUATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    test_df = pd.read_csv(TEST_CSV)

    print(f"Test image records: {len(test_df)}")
    print(f"Unique test patients: {test_df['patient_id'].nunique()}")

    baseline_model = load_baseline_model()

    print()
    print("Generating baseline image predictions...")

    baseline_images = get_image_predictions(
        baseline_model,
        test_df
    )

    baseline_patients = aggregate_patient_predictions(
        baseline_images
    )

    baseline_results = calculate_metrics(
        baseline_patients
    )

    print_results(
        "BASELINE - PATIENT LEVEL",
        baseline_results
    )

    ssl_model = load_ssl_model()

    print()
    print("Generating SSL image predictions...")

    ssl_images = get_image_predictions(
        ssl_model,
        test_df
    )

    ssl_patients = aggregate_patient_predictions(
        ssl_images
    )

    ssl_results = calculate_metrics(
        ssl_patients
    )

    print_results(
        "SSL - PATIENT LEVEL",
        ssl_results
    )

    print()
    print("=" * 60)
    print("PATIENT-LEVEL COMPARISON")
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
            baseline_results["sensitivity"],
            baseline_results["specificity"],
            baseline_results["f1"],
            baseline_results["roc_auc"],
        ],
        "SSL": [
            ssl_results["accuracy"],
            ssl_results["precision"],
            ssl_results["sensitivity"],
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
    print("PATIENT IMAGE DISTRIBUTION")
    print("=" * 60)

    print(
        baseline_patients["number_of_images"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()