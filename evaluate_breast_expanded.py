from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from config import CONFIGS, DEVICE, IMAGE_SIZE, NUM_WORKERS
from models.cnn import CancerCNN
from utils import eval_transforms


def main():

    cfg = CONFIGS["breast"]

    root = Path(cfg["dataset_dir"])
    model_path = Path(
        r".\saved_models\breast_cnn_expanded.pth"
    )

    # =========================================================
    # DATASET
    # =========================================================

    val_dataset = ImageFolder(
        root / "val",
        transform=eval_transforms(IMAGE_SIZE)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    classes = val_dataset.classes

    print("=================================================================")
    print("BREAST CANCER MODEL - EXPANDED MODEL CLASSIFICATION REPORT")
    print("=================================================================")
    print("Device:", DEVICE)
    print("Classes:", classes)
    print("Validation images:", len(val_dataset))
    print("Loading:", model_path)

    # =========================================================
    # MODEL
    # =========================================================

    model = CancerCNN(
        num_classes=len(classes),
        pretrained=False
    ).to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    # =========================================================
    # PREDICTIONS
    # =========================================================

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)

            outputs = model(x)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = outputs.argmax(
                dim=1
            )

            y_true.extend(
                y.tolist()
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

            # Probability of MALIGNANT
            malignant_index = classes.index(
                "MALIGNANT"
            )

            y_prob.extend(
                probabilities[
                    :,
                    malignant_index
                ].cpu().tolist()
            )

    # =========================================================
    # METRICS
    # =========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    # =========================================================
    # OUTPUT
    # =========================================================

    print("\n=================================================================")
    print("RESULTS")
    print("=================================================================")

    print(
        f"Accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Malignant Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Malignant Recall/Sensitivity: {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"Malignant F1-score: {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"ROC-AUC: {auc:.4f} "
        f"({auc * 100:.2f}%)"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=classes,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print("\nInterpretation:")
    print(
        "Rows = Actual classes"
    )
    print(
        "Columns = Predicted classes"
    )


if __name__ == "__main__":
    main()
