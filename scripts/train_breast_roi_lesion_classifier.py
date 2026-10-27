"""
CancerScan AI: High-Resolution Cropped Tumor Lesion Classifier
=============================================================
Trains a specialized Focal Tumor Classifier directly on the zoomed-in
lesion crops (1,104 train, 208 val, 233 held-out test), strictly isolated
by patient ID to ensure zero data leakage.

Initialized with our SimCLR Mammography Foundation Encoder weights.
Evaluated with 8-view Test-Time Augmentation (TTA) on unseen test patients.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision import transforms as T

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

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "roi_train_strict.csv"
VAL_CSV = DATA_DIR / "roi_val_strict.csv"
TEST_CSV = DATA_DIR / "roi_test_strict.csv"
FOUNDATION_PATH = MODEL_DIR / "breast_densenet121_simclr_foundation.pth"

CHECKPOINT_PATH = MODEL_DIR / "breast_roi_lesion_classifier_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_roi_lesion_history.csv"
TEST_JSON_PATH = REPORT_DIR / "breast_roi_lesion_test.json"
TEST_PREDS_PATH = REPORT_DIR / "breast_roi_lesion_test_predictions.csv"
CM_PATH = REPORT_DIR / "breast_roi_lesion_test_cm.png"
ROC_PATH = REPORT_DIR / "breast_roi_lesion_test_roc.png"

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 20
LR = 1.5e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def resolve_image_path(raw_path: str) -> Path:
    raw = str(raw_path).replace("\\", "/").strip()
    raw = raw.replace("mnt/data/CancerScan_AI/", "")
    p = Path(raw)
    candidates = [
        ROOT / p,
        DATA_DIR / p,
        DATA_DIR / "jpeg" / p.name,
    ]
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if len(parts) >= 2:
        candidates.append(DATA_DIR / "jpeg" / parts[-2] / parts[-1])
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Missing ROI image: {raw_path}")


class LesionPatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool, size: int = 256):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.size = size

        if self.training:
            self.transform = T.Compose([
                T.Resize((size, size), interpolation=InterpolationMode.BILINEAR, antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15, fill=0),
                T.ColorJitter(brightness=0.12, contrast=0.12),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((size, size), interpolation=InterpolationMode.BILINEAR, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        resolved = resolve_image_path(row["image_path"])
        with Image.open(resolved) as im:
            img = im.convert("RGB")
        tensor = self.transform(img)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label, str(row["patient_id"]), str(resolved)


class FocalLesionDenseNet(nn.Module):
    def __init__(self, foundation_path: Path = None):
        super().__init__()
        base = models.densenet121(weights=None)
        self.features = base.features  # 1024 channels

        if foundation_path and foundation_path.is_file():
            print(f"Loading SimCLR Foundation weights: {foundation_path.name}")
            ckpt = torch.load(foundation_path, map_location="cpu")
            state_dict = ckpt.get("backbone_state_dict", ckpt)
            self.features.load_state_dict(state_dict, strict=False)
            print("Successfully loaded domain-adapted Mammography Foundation backbone!")
        else:
            print("Loading standard ImageNet weights.")
            base_pretrained = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.features = base_pretrained.features

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        # GAP (1024) + GMP (1024) = 2048 dims
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        feat = self.features(x)
        avg_p = self.gap(feat).flatten(1)
        max_p = self.gmp(feat).flatten(1)
        descriptor = torch.cat([avg_p, max_p], dim=1)
        return self.classifier(descriptor)


def evaluate_lesion_model(model, loader, threshold: float = 0.5, use_tta: bool = True):
    model.eval()
    device = next(model.parameters()).device
    y_true, y_prob, metadata = [], [], []

    with torch.no_grad():
        for images, labels, p_ids, paths in loader:
            images = images.to(device, non_blocking=True)

            if not use_tta:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                p_list = []
                # 1. Base
                p_list.append(torch.softmax(model(images), dim=1)[:, 1])
                # 2. H-flip
                p_list.append(torch.softmax(model(TF.hflip(images)), dim=1)[:, 1])
                # 3. V-flip
                p_list.append(torch.softmax(model(TF.vflip(images)), dim=1)[:, 1])
                # 4. Zoom in 4%
                p_list.append(torch.softmax(model(TF.affine(images, 0, [0, 0], 1.04, [0, 0], interpolation=InterpolationMode.BILINEAR)), dim=1)[:, 1])
                # 5. Zoom out 4%
                p_list.append(torch.softmax(model(TF.affine(images, 0, [0, 0], 0.96, [0, 0], interpolation=InterpolationMode.BILINEAR)), dim=1)[:, 1])
                # 6. Rotate +10 deg
                p_list.append(torch.softmax(model(TF.rotate(images, 10, interpolation=InterpolationMode.BILINEAR, fill=0)), dim=1)[:, 1])
                # 7. Rotate -10 deg
                p_list.append(torch.softmax(model(TF.rotate(images, -10, interpolation=InterpolationMode.BILINEAR, fill=0)), dim=1)[:, 1])
                # 8. Contrast up
                p_list.append(torch.softmax(model(TF.adjust_contrast(images, 1.06)), dim=1)[:, 1])

                probs = torch.stack(p_list, dim=0).mean(dim=0)

            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

            for i in range(len(p_ids)):
                metadata.append({
                    "patient_id": str(p_ids[i]),
                    "image_path": str(paths[i]),
                    "probability": float(probs[i].item()),
                    "y_true": int(labels[i].item()),
                })

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "threshold": float(threshold),
        "use_tta": bool(use_tta),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    df_meta = pd.DataFrame(metadata)
    df_meta[f"pred_t{int(threshold*100):02d}"] = y_pred
    return metrics, y_true, y_prob, y_pred, df_meta


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("CancerScan AI | High-Resolution Focal Lesion Classifier")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print("=" * 76)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"Train Lesions (Strictly Isolated)    : {len(train_df)} (Benign={int((train_df['label']==0).sum())}, Malignant={int((train_df['label']==1).sum())})")
    print(f"Val Lesions (Strictly Isolated)      : {len(val_df)} (Benign={int((val_df['label']==0).sum())}, Malignant={int((val_df['label']==1).sum())})")
    print(f"Held-Out Test Lesions                : {len(test_df)} (Benign={int((test_df['label']==0).sum())}, Malignant={int((test_df['label']==1).sum())})")

    train_ds = LesionPatchDataset(train_df, training=True, size=IMAGE_SIZE)
    val_ds = LesionPatchDataset(val_df, training=False, size=IMAGE_SIZE)
    test_ds = LesionPatchDataset(test_df, training=False, size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = FocalLesionDenseNet(FOUNDATION_PATH).to(device)

    # Differential learning rate
    backbone_params = list(model.features.parameters())
    classifier_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR * 0.2},
        {"params": classifier_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6)

    # Class weighting
    n_b = int((train_df["label"] == 0).sum())
    n_m = int((train_df["label"] == 1).sum())
    w0 = (n_b + n_m) / (2.0 * n_b)
    w1 = (n_b + n_m) / (2.0 * n_m)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([w0, w1], dtype=torch.float32, device=device),
        label_smoothing=LABEL_SMOOTHING
    )
    scaler = GradScaler()

    best_composite = 0.0
    best_threshold = 0.5
    best_weights = None
    history = []

    print("\n--- Training Focal Lesion Specialist Network ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        start_t = time.time()

        for images, labels, _, _ in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.2)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        # Evaluate on Validation
        val_m, val_yt, val_yp, _, _ = evaluate_lesion_model(model, val_loader, threshold=0.5, use_tta=True)

        # Optimize threshold on validation ROC curve
        fpr, tpr, threshs = roc_curve(val_yt, val_yp)
        j_scores = tpr - fpr
        opt_t = float(np.clip(threshs[np.argmax(j_scores)], 0.25, 0.75))
        val_m_opt, _, _, _, _ = evaluate_lesion_model(model, val_loader, threshold=opt_t, use_tta=True)

        composite = (val_m_opt["balanced_accuracy"] + val_m_opt["roc_auc"]) / 2.0
        is_best = composite > best_composite

        if is_best:
            best_composite = composite
            best_threshold = opt_t
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "epoch": epoch,
                "threshold": opt_t,
                "val_metrics": val_m_opt,
            }, CHECKPOINT_PATH)
            star = "  <-- BEST"
        else:
            star = ""

        elapsed = time.time() - start_t
        print(f"[Epoch {epoch:02d}/{EPOCHS}] Loss={avg_loss:.4f} | "
              f"Val Acc={val_m_opt['accuracy']*100:.1f}% Sens={val_m_opt['recall_sensitivity']*100:.1f}% "
              f"Spec={val_m_opt['specificity']*100:.1f}% AUC={val_m_opt['roc_auc']:.4f} T*={opt_t:.3f} ({elapsed:.1f}s){star}")

        history.append({
            "epoch": epoch, "loss": avg_loss,
            "val_acc": val_m_opt["accuracy"], "val_bal_acc": val_m_opt["balanced_accuracy"],
            "val_sens": val_m_opt["recall_sensitivity"], "val_spec": val_m_opt["specificity"],
            "val_auc": val_m_opt["roc_auc"], "opt_t": opt_t,
        })

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    # -------------------------------------------------------------
    # FINAL EVALUATION ON COMPLETELY HELD-OUT TEST SET
    # -------------------------------------------------------------
    print("\n" + "=" * 76)
    print("FINAL EVALUATION ON STRICTLY HELD-OUT TEST LESIONS (N=233)")
    print("=" * 76)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    opt_t = ckpt["threshold"]
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (Optimal T* = {opt_t:.3f})")

    # 1. Standard 0.50 threshold
    test_m_50, yt, yp, ypred_50, _ = evaluate_lesion_model(model, test_loader, threshold=0.50, use_tta=True)
    # 2. Optimal validation threshold
    test_m_opt, yt, yp, ypred_opt, df_preds = evaluate_lesion_model(model, test_loader, threshold=opt_t, use_tta=True)

    df_preds.to_csv(TEST_PREDS_PATH, index=False)

    summary = {
        "model": "FocalLesionDenseNet",
        "best_epoch": ckpt["epoch"],
        "optimal_threshold": float(opt_t),
        "test_standard_050": test_m_50,
        "test_optimal_threshold": test_m_opt,
    }
    TEST_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nClassification Report (Standard Threshold = 0.50):")
    print(classification_report(yt, ypred_50, target_names=CLASS_NAMES, digits=4))

    print(f"\nClassification Report (Optimal Threshold = {opt_t:.3f}):")
    print(classification_report(yt, ypred_opt, target_names=CLASS_NAMES, digits=4))

    print("\nSummary Metrics (Optimal Threshold):")
    print(f"  Accuracy           : {test_m_opt['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy  : {test_m_opt['balanced_accuracy']*100:.2f}%")
    print(f"  Recall/Sensitivity : {test_m_opt['recall_sensitivity']*100:.2f}%")
    print(f"  Specificity        : {test_m_opt['specificity']*100:.2f}%")
    print(f"  Precision          : {test_m_opt['precision']*100:.2f}%")
    print(f"  F1-Score           : {test_m_opt['f1']:.4f}")
    print(f"  ROC-AUC            : {test_m_opt['roc_auc']:.4f}")
    print(f"  Confusion Matrix   : TN={test_m_opt['tn']}, FP={test_m_opt['fp']}, FN={test_m_opt['fn']}, TP={test_m_opt['tp']}")

    # Plot Confusion Matrix
    cm = confusion_matrix(yt, ypred_opt, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Focal Lesion Test CM (T* = {opt_t:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontweight="bold", fontsize=13)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=160)
    plt.close(fig)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(yt, yp)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {test_m_opt['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Focal Lesion Test ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=160)
    plt.close(fig)

    print("\n" + "=" * 76)
    print("FOCAL LESION CLASSIFIER EVALUATION COMPLETE")
    print(f"Best checkpoint : {CHECKPOINT_PATH}")
    print(f"Test results    : {TEST_JSON_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
