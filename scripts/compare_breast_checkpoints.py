from os import name
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import train_breast_resnet50_ivaf as training


ROOT = Path(".")
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "baseline": ROOT / "saved_models" / "breast_resnet50_ivaf_best.pth",
    "exp3_layer4_scheduler": ROOT / "saved_models" / "breast_resnet50_ivaf_exp3_layer4_scheduler_best.pth",
    "imagenet": ROOT / "saved_models" / "breast_resnet50_ivaf_imagenet_best.pth",
}

DEVICE = training.DEVICE
BATCH_SIZE = 8
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ValidationDataset(Dataset):
    def __init__(self, pairs, use_imagenet_normalization):
        self.pairs = pairs.reset_index(drop=True)
        self.use_imagenet_normalization = use_imagenet_normalization

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs.iloc[index]

        with Image.open(row["cc_path"]) as im:
            cc = im.convert("RGB")
        with Image.open(row["mlo_path"]) as im:
            mlo = im.convert("RGB")

        cc = TF.resize(
            cc, [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )
        mlo = TF.resize(
            mlo, [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )

        cc = TF.to_tensor(cc)
        mlo = TF.to_tensor(mlo)

        if self.use_imagenet_normalization:
            cc = TF.normalize(cc, MEAN, STD)
            mlo = TF.normalize(mlo, MEAN, STD)

        return (
            cc,
            mlo,
            int(row["label"]),
            str(row["patient_id"]),
            str(row["cc_path"]),
            str(row["mlo_path"]),
        )


def evaluate_checkpoint(name, checkpoint_path, pairs):
    print("\n" + "=" * 72)
    print("Evaluating:", name)
    print("Checkpoint:", checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    # Checkpoint architecture is shared, so avoid downloading new weights.
    model = training.ResNet50IVAF(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    use_normalization = name in ("baseline", "imagenet")
    dataset = ValidationDataset(pairs, use_normalization)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    y_true = []
    y_prob = []
    predictions = []
    rows = []


    with torch.inference_mode():
        for cc, mlo, labels, patient_ids, cc_paths, mlo_paths in loader:
            cc = cc.to(DEVICE)
            mlo = mlo.to(DEVICE)

            # Evaluate in full precision to avoid mixed-precision instability.
            logits = model(cc.float(), mlo.float())

            logits = logits.float()

            if not torch.isfinite(logits).all():
                bad_count = (~torch.isfinite(logits)).sum().item()
                raise RuntimeError(
                    f"{name}: model produced {bad_count} NaN/Inf logit values. "
                    "Check checkpoint weights and input preprocessing."
                )

            probabilities = torch.softmax(logits, dim=1)[:, 1]

            if not torch.isfinite(probabilities).all():
                raise RuntimeError(
                    f"{name}: probabilities contain NaN/Inf after softmax."
                )

            predicted = (probabilities >= 0.5).long()

            batch_labels = labels.numpy().tolist()
            batch_probs = probabilities.cpu().numpy().tolist()
            batch_preds = predicted.cpu().numpy().tolist()

            y_true.extend(batch_labels)
            y_prob.extend(batch_probs)
            predictions.extend(batch_preds)

            for i in range(len(batch_labels)):
                rows.append({
                    "model": name,
                    "patient_id": patient_ids[i],
                    "cc_path": cc_paths[i],
                    "mlo_path": mlo_paths[i],
                    "y_true": batch_labels[i],
                    "malignant_probability": batch_probs[i],
                    "y_pred_threshold_050": batch_preds[i],
                })


    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "validation_pairs": len(y_true),
        "normalization": (
            "ImageNet mean/std" if use_normalization
            else "ToTensor only: [0,1]"
        ),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "malignant_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "malignant_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "specificity": float(
            recall_score(y_true, predictions, pos_label=0, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
        "test_set_used": False,
    }

    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "MALIGNANT"],
        zero_division=0,
    ))

    predictions_path = REPORT_DIR / f"breast_{name}_comparison_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def main():
    print("Device:", DEVICE)
    print("Building validation pairs from datasets/val.csv")

    val_rows = training.prepare_rows(training.VAL_CSV, "validation")
    val_pairs, audit = training.build_pairs(val_rows, "validation")

    if len(val_pairs) == 0:
        raise RuntimeError("No validation pairs were created.")

    print("Validation pairs:", len(val_pairs))
    print("Pair audit:", json.dumps(audit, indent=2, default=str))

    # Keep the same pair order for both checkpoints.
    results = []
    for name, path in CHECKPOINTS.items():
        results.append(evaluate_checkpoint(name, path, val_pairs))

    comparison = pd.DataFrame(results)
    csv_path = REPORT_DIR / "breast_checkpoint_comparison.csv"
    json_path = REPORT_DIR / "breast_checkpoint_comparison.json"

    comparison.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE VALIDATION COMPARISON")
    columns = [
        "model", "normalization", "accuracy", "balanced_accuracy",
        "malignant_precision", "malignant_recall", "specificity",
        "f1", "roc_auc", "tn", "fp", "fn", "tp",
    ]
    print(comparison[columns].to_string(index=False))

    print("\nSaved reports:")
    print(csv_path)
    print(json_path)
    print("\nTest set was not used. No model training was performed.")


if __name__ == "__main__":
    main()
