
"""
Confusion Matrix Heatmap Generator
Project: CancerScan_AI

Supports:
    1. Breast cancer: BENIGN, MALIGNANT
    2. Lung cancer: Cancer, NORMAL

Input CSV must contain two columns:
    y_true, y_pred

The labels can be class names or integer class indices.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


CLASS_NAMES = {
    "breast": ["BENIGN", "MALIGNANT"],
    "lung": ["Cancer", "NORMAL"],
}


def plot_confusion_heatmap(y_true, y_pred, class_names, output_dir):
    """Generate count and normalized confusion matrix heatmaps."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate raw confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    # Calculate row-normalized confusion matrix
    cm_normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        normalize="true"
    )

    # Plot raw counts
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        cbar=True
    )

    plt.title("Confusion Matrix - Raw Counts")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    count_path = output_dir / "confusion_matrix_counts.png"
    plt.savefig(count_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Plot percentages
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar=True
    )

    plt.title("Normalized Confusion Matrix (%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    percentage_path = output_dir / "confusion_matrix_percentages.png"
    plt.savefig(percentage_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )
    )

    print(f"\nCount heatmap saved: {count_path}")
    print(f"Percentage heatmap saved: {percentage_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cancer model confusion matrix heatmaps."
    )

    parser.add_argument(
        "--cancer",
        choices=["breast", "lung"],
        required=True,
        help="Choose breast or lung cancer model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV containing y_true and y_pred columns."
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saved heatmaps."
    )

    args = parser.parse_args()

    csv_path = Path(args.input)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"y_true", "y_pred"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns named 'y_true' and 'y_pred'."
        )

    df = df.dropna(subset=["y_true", "y_pred"])

    if df.empty:
        raise ValueError("The CSV contains no valid prediction rows.")

    class_names = CLASS_NAMES[args.cancer]

    # Convert textual class labels to indices when necessary.
    # Integer labels must follow the class order shown above.
    label_to_index = {
        name: index for index, name in enumerate(class_names)
    }

    def convert_labels(series):
        converted = []

        for value in series:
            if isinstance(value, str):
                value = value.strip()

                if value in label_to_index:
                    converted.append(label_to_index[value])
                    continue

            try:
                index = int(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Unknown label '{value}'. Expected class names "
                    f"{class_names} or integer indices."
                )

            if index < 0 or index >= len(class_names):
                raise ValueError(
                    f"Class index {index} is invalid for {class_names}."
                )

            converted.append(index)

        return converted

    y_true = convert_labels(df["y_true"])
    y_pred = convert_labels(df["y_pred"])

    output_dir = args.output_dir or f"results/{args.cancer}"

    plot_confusion_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
