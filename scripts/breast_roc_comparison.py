from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent.parent

DATASETS = BASE / "mnt" / "data" / "CancerScan_AI" / "datasets"
MODELS = BASE / "mnt" / "data" / "CancerScan_AI" / "models"
SAVED_MODELS = BASE / "mnt" / "data" / "CancerScan_AI" / "saved_models"

TEST_CSV = DATASETS / "test.csv"
BASELINE_CHECKPOINT = SAVED_MODELS / "breast_cnn.pth"
SSL_CHECKPOINT = MODELS / "breast_ssl_finetuned.pth"

OUTPUT_DIR = BASE / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

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


def load_model(checkpoint_path, ssl=False):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = BreastModel()

    state_dict = (
        checkpoint["model_state"]
        if "model_state" in checkpoint
        else checkpoint
    )

    if ssl:
        new_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("encoder."):
                new_key = "backbone." + key[len("encoder."):]
                new_state_dict[new_key] = value

            elif key.startswith("classifier."):
                new_key = "backbone.fc." + key[len("classifier."):]
                new_state_dict[new_key] = value

        state_dict = new_state_dict

    model.load_state_dict(state_dict)

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

    return np.array(probabilities)


def aggregate_patient_predictions(dataframe, probabilities):

    result = dataframe.copy().reset_index(drop=True)

    result["malignant_probability"] = probabilities

    patient_results = []

    for patient_id, group in result.groupby("patient_id"):

        patient_probability = group[
            "malignant_probability"
        ].mean()

        true_label = int(
            group["label"].iloc[0]
        )

        patient_results.append({
            "patient_id": patient_id,
            "true_label": true_label,
            "malignant_probability": patient_probability
        })

    return pd.DataFrame(patient_results)


def main():

    print("=" * 60)
    print("BREAST CANCER PATIENT-LEVEL ROC COMPARISON")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    print()
    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_CSV)

    print(f"Test image records: {len(test_df)}")
    print(
        f"Unique test patients: "
        f"{test_df['patient_id'].nunique()}"
    )

    print()
    print("Loading baseline model...")

    baseline_model = load_model(
        BASELINE_CHECKPOINT,
        ssl=False
    )

    print("Generating baseline predictions...")

    baseline_image_probs = get_image_predictions(
        baseline_model,
        test_df
    )

    baseline_patients = aggregate_patient_predictions(
        test_df,
        baseline_image_probs
    )

    print()
    print("Loading SSL model...")

    ssl_model = load_model(
        SSL_CHECKPOINT,
        ssl=True
    )

    print("Generating SSL predictions...")

    ssl_image_probs = get_image_predictions(
        ssl_model,
        test_df
    )

    ssl_patients = aggregate_patient_predictions(
        test_df,
        ssl_image_probs
    )

    y_true = baseline_patients["true_label"].to_numpy()

    baseline_probs = (
        baseline_patients["malignant_probability"].to_numpy()
    )

    ssl_probs = (
        ssl_patients["malignant_probability"].to_numpy()
    )

    baseline_auc = roc_auc_score(
        y_true,
        baseline_probs
    )

    ssl_auc = roc_auc_score(
        y_true,
        ssl_probs
    )

    print()
    print("=" * 60)
    print("PATIENT-LEVEL ROC-AUC")
    print("=" * 60)

    print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
    print(f"SSL ROC-AUC:      {ssl_auc:.4f}")
    print(
        f"Difference:       "
        f"{ssl_auc - baseline_auc:+.4f}"
    )

    baseline_fpr, baseline_tpr, _ = roc_curve(
        y_true,
        baseline_probs
    )

    ssl_fpr, ssl_tpr, _ = roc_curve(
        y_true,
        ssl_probs
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        baseline_fpr,
        baseline_tpr,
        label=f"Baseline (AUC = {baseline_auc:.4f})"
    )

    plt.plot(
        ssl_fpr,
        ssl_tpr,
        label=f"SSL (AUC = {ssl_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "Patient-Level Breast Cancer ROC Comparison"
    )

    plt.legend()
    plt.grid(True)

    output_path = (
        OUTPUT_DIR /
        "breast_patient_level_roc_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("ROC curve saved to:")
    print(output_path)

    print()
    print("=" * 60)
    print("ROC COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()