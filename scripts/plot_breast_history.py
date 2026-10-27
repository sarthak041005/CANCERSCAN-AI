import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("reports/breast_resnet50_ivaf_history.csv")
out = Path("reports")
out.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

plots = [
    ("train_loss", "val_loss", "Loss"),
    ("train_accuracy", "val_accuracy", "Accuracy"),
    ("train_roc_auc", "val_roc_auc", "ROC-AUC"),
]

for ax, (train_col, val_col, title) in zip(axes, plots):
    ax.plot(df["epoch"], df[train_col], label="Train")
    ax.plot(df["epoch"], df[val_col], label="Validation")
    ax.axvline(15.5, linestyle="--", label="Fine-tuning starts")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.tight_layout()
fig.savefig(out / "breast_resnet50_ivaf_training_curves.png", dpi=200)
print("Saved: reports/breast_resnet50_ivaf_training_curves.png")
plt.show()
