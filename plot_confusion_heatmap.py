
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


CLASS_NAMES = {
    "lung": ["Cancer", "NORMAL"],
    "breast": ["BENIGN", "MALIGNANT"],
}


def generate_heatmaps(y_true, y_pred, class_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = confusion_matrix(
        y_true, y_pred, labels=labels, normalize="true"
    )

    # Heatmap 1: Raw counts
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Raw Counts")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix_counts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Heatmap 2: Percentages
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1%",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix (%)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix_percentages.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    print(f"\nHeatmaps saved in: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancer", choices=["lung", "breast"], required=True
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV not found: {csv_path.resolve()}"
        )

    df = pd.read_csv(csv_path)

    if not {"y_true", "y_pred"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns named y_true and y_pred."
        )

    df = df.dropna(subset=["y_true", "y_pred"])
    if df.empty:
        raise ValueError("No valid predictions found in the CSV.")

    names = CLASS_NAMES[args.cancer]
    label_map = {name: i for i, name in enumerate(names)}

    def convert(series):
        result = []
        for value in series:
            if isinstance(value, str):
                value = value.strip()
                if value in label_map:
                    result.append(label_map[value])
                    continue

            try:
                index = int(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid label {value!r}; expected {names} "
                    "or integer class indices."
                )

            if index not in range(len(names)):
                raise ValueError(f"Invalid class index: {index}")
            result.append(index)

        return result

    y_true = convert(df["y_true"])
    y_pred = convert(df["y_pred"])

    output_dir = args.output_dir or f"results/{args.cancer}"
    generate_heatmaps(y_true, y_pred, names, output_dir)


if __name__ == "__main__":
    main()
