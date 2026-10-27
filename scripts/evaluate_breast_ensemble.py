from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import train_breast_resnet50_ivaf as tr

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_models():
    # 1. ResNet50 IVAF
    r50_path = MODEL_DIR / "breast_resnet50_ivaf_stage3_improved_best.pth"
    if not r50_path.is_file():
        r50_path = MODEL_DIR / "breast_resnet50_ivaf_best.pth"

    print(f"Loading ResNet50 IVAF from: {r50_path.name}")
    cp_r50 = torch.load(r50_path, map_location=DEVICE, weights_only=False)
    model_r50 = tr.ResNet50IVAF(pretrained=False).to(DEVICE)
    model_r50.load_state_dict(cp_r50["model_state_dict"])
    model_r50.eval()

    # 2. DenseNet121
    dense_path = MODEL_DIR / "breast_densenet121_224_best.pth"
    print(f"Loading DenseNet121 from: {dense_path.name}")
    cp_dense = torch.load(dense_path, map_location=DEVICE, weights_only=False)
    model_dense = models.densenet121(weights=None)
    num_features = model_dense.classifier.in_features
    model_dense.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 2),
    )
    dense_sd = cp_dense["model_state_dict"] if "model_state_dict" in cp_dense else cp_dense
    model_dense.load_state_dict(dense_sd)
    model_dense = model_dense.to(DEVICE)
    model_dense.eval()

    return model_r50, model_dense


def predict_pair(model_r50, model_dense, cc_img, mlo_img, use_tta=True):
    # ResNet50 transforms: [0, 1] ToTensor
    cc_base = TF.to_tensor(TF.resize(cc_img, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)).unsqueeze(0).to(DEVICE)
    mlo_base = TF.to_tensor(TF.resize(mlo_img, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)).unsqueeze(0).to(DEVICE)

    # DenseNet inputs: ImageNet normalized
    cc_dense = TF.normalize(cc_base, MEAN, STD)
    mlo_dense = TF.normalize(mlo_base, MEAN, STD)

    with torch.no_grad():
        # DenseNet probability: average of CC and MLO view predictions
        p_dense_cc = torch.softmax(model_dense(cc_dense), dim=1)[0, 1].item()
        p_dense_mlo = torch.softmax(model_dense(mlo_dense), dim=1)[0, 1].item()
        p_dense = 0.5 * (p_dense_cc + p_dense_mlo)

        if not use_tta:
            p_r50 = torch.softmax(model_r50(cc_base, mlo_base), dim=1)[0, 1].item()
        else:
            probs = []
            # 1. Normal
            probs.append(torch.softmax(model_r50(cc_base, mlo_base), dim=1)[0, 1].item())
            # 2. Scale 1.03
            cc_z = TF.affine(cc_base, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
            mlo_z = TF.affine(mlo_base, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
            probs.append(torch.softmax(model_r50(cc_z, mlo_z), dim=1)[0, 1].item())
            # 3. Brightness adjustment
            cc_b = TF.adjust_brightness(cc_base, 1.04)
            mlo_b = TF.adjust_brightness(mlo_base, 1.04)
            probs.append(torch.softmax(model_r50(cc_b, mlo_b), dim=1)[0, 1].item())
            p_r50 = float(np.mean(probs))

    return p_r50, p_dense


def evaluate_split(split_name, model_r50, model_dense, optimal_threshold=None, weight_r50=0.55):
    rows = tr.prepare_rows(DATA_DIR / f"{split_name}.csv", split_name)
    pairs, _ = tr.build_pairs(rows, split_name)

    y_true = []
    r50_probs = []
    dense_probs = []
    ens_probs = []
    metadata = []

    for idx in range(len(pairs)):
        r = pairs.iloc[idx]
        with Image.open(r["cc_path"]) as im:
            cc = im.convert("RGB")
        with Image.open(r["mlo_path"]) as im:
            mlo = im.convert("RGB")

        p_r, p_d = predict_pair(model_r50, model_dense, cc, mlo, use_tta=True)
        p_e = weight_r50 * p_r + (1.0 - weight_r50) * p_d

        y = int(r["label"])
        y_true.append(y)
        r50_probs.append(p_r)
        dense_probs.append(p_d)
        ens_probs.append(p_e)

        metadata.append({
            "patient_id": r["patient_id"],
            "cc_path": r["cc_path"],
            "mlo_path": r["mlo_path"],
            "y_true": y,
            "r50_prob": p_r,
            "dense_prob": p_d,
            "ensemble_prob": p_e,
        })

    y_true = np.array(y_true)
    r50_probs = np.array(r50_probs)
    dense_probs = np.array(dense_probs)
    ens_probs = np.array(ens_probs)

    # Compute optimal threshold on ensemble probabilities if not provided
    if optimal_threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, ens_probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = float(np.clip(thresholds[best_idx], 0.25, 0.75))

    def get_metrics(probs, thresh):
        preds = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        return {
            "threshold": float(thresh),
            "accuracy": float(accuracy_score(y_true, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall_sensitivity": float(recall_score(y_true, preds, zero_division=0)),
            "specificity": float(recall_score(y_true, preds, pos_label=0, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    results = {
        "split": split_name,
        "cases": len(y_true),
        "optimal_threshold": optimal_threshold,
        "resnet50_t050": get_metrics(r50_probs, 0.50),
        "densenet_t050": get_metrics(dense_probs, 0.50),
        "ensemble_t050": get_metrics(ens_probs, 0.50),
        "ensemble_t_opt": get_metrics(ens_probs, optimal_threshold),
    }

    df_meta = pd.DataFrame(metadata)
    df_meta[f"pred_t{int(optimal_threshold*100):02d}"] = (ens_probs >= optimal_threshold).astype(int)
    df_meta.to_csv(REPORT_DIR / f"breast_ensemble_{split_name}_predictions.csv", index=False)

    return results, optimal_threshold


def main():
    print("=" * 72)
    print("CancerScan AI | Multi-Model Ensemble Evaluation (ResNet50 + DenseNet121)")
    print("=" * 72)

    model_r50, model_dense = load_models()

    print("\n--- Evaluating Validation Set ---")
    val_results, opt_thresh = evaluate_split("val", model_r50, model_dense)
    print(f"Optimal Threshold (Youden's J on Val): {opt_thresh:.3f}")

    print("\n--- Evaluating Held-Out Test Set ---")
    test_results, _ = evaluate_split("test", model_r50, model_dense, optimal_threshold=opt_thresh)

    summary = {
        "validation": val_results,
        "test": test_results,
    }

    out_json = REPORT_DIR / "breast_ensemble_benchmark.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE PERFORMANCE COMPARISON")
    print("=" * 72)
    for split, res in [("VALIDATION", val_results), ("TEST", test_results)]:
        print(f"\n--- {split} (N={res['cases']}) ---")
        fmt = "{:<25} {:<10} {:<10} {:<10} {:<10} {:<10}"
        print(fmt.format("Configuration", "Acc", "Bal Acc", "Sensitivity", "Spec", "AUC"))
        for k, label in [
            ("resnet50_t050", "ResNet50 IVAF (0.50)"),
            ("densenet_t050", "DenseNet121 (0.50)"),
            ("ensemble_t050", "Ensemble (0.50)"),
            ("ensemble_t_opt", f"Ensemble (T*={opt_thresh:.2f})"),
        ]:
            m = res[k]
            print(fmt.format(
                label,
                f"{m['accuracy']*100:.1f}%",
                f"{m['balanced_accuracy']*100:.1f}%",
                f"{m['recall_sensitivity']*100:.1f}%",
                f"{m['specificity']*100:.1f}%",
                f"{m['roc_auc']:.4f}",
            ))

    print(f"\nReport saved to: {out_json}")


if __name__ == "__main__":
    main()
