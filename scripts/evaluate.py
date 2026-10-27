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


def main(cancer):

    cfg = CONFIGS[cancer]

    root = Path(cfg["dataset_dir"])
    model_path = Path(cfg["model_path"])

    # Validation dataset
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

    print("Classes:", val_dataset.classes)
    print("Validation images:", len(val_dataset))
    print("Loading:", model_path)

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    classes = checkpoint["classes"]

    # Create model
    model = CancerCNN(
        len(classes),
        True
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)

            outputs = model(x)

            probabilities = torch.softmax(outputs, dim=1)

            predictions = outputs.argmax(dim=1)

            y_true.extend(y.tolist())
            y_pred.extend(predictions.cpu().tolist())

            # Probability of MALIGNANT
            if "MALIGNANT" in classes:
                malignant_index = classes.index("MALIGNANT")
                y_prob.extend(
                    probabilities[:, malignant_index]
                    .cpu()
                    .tolist()
                )

    # =========================================================
    # METRICS
    # =========================================================

    accuracy = accuracy_score(y_true, y_pred)

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

    print("\n========================================")
    print(f"{cancer.upper()} MODEL EVALUATION")
    print("========================================")

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    # ROC-AUC
    if len(set(y_true)) == 2:

        auc = roc_auc_score(
            y_true,
            y_prob
        )

        print(f"ROC-AUC     : {auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=classes,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    print("\nInterpretation:")
    print(
        "Rows = Actual classes\n"
        "Columns = Predicted classes"
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cancer",
        choices=["breast", "lung"],
        required=True
    )

    args = parser.parse_args()

    main(args.cancer)