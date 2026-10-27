import pandas as pd

files = {
    "Original baseline": "reports/breast_resnet50_ivaf_history.csv",
    "Experiment 1": "reports/breast_resnet50_ivaf_exp1_lr_accuracy_history.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(f"\n--- {name} ---")

    for metric in ["val_accuracy", "val_roc_auc", "val_loss"]:
        if metric not in df.columns:
            continue

        idx = df[metric].idxmin() if metric == "val_loss" else df[metric].idxmax()
        row = df.loc[idx]

        print(
            f"{metric}: {row[metric]:.4f} | "
            f"epoch={int(row['epoch'])} | stage={row['stage']}"
        )
