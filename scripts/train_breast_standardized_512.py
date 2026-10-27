"""
CancerScan AI: High-Performance Dual-View DenseNet on Standardized 512x512 Mammograms
=====================================================================================
Features:
- Dataset: datasets/breast_preprocessed_standardized_512 (tight crop, chest wall left, CLAHE)
- Backbone: DenseNet-121 with Dual Pooling (GAP + GMP)
- Fusion: Asymmetry (|v_CC - v_MLO|) + Concordance (v_CC * v_MLO) Interaction Heads
- Multi-Task Supervision: CC Head + MLO Head + Fused Head
- OneCycleLR with Cosine Annealing, Label Smoothing, Mixed Precision (AMP)
- 8-View Test-Time Augmentation (TTA)
"""
from __future__ import annotations

import copy
import json
import random
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
IMAGE_ROOT_512 = DATA_DIR / "breast_preprocessed_standardized_512"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "breast_standardized_512_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_standardized_512_history.csv"
VAL_JSON_PATH = REPORT_DIR / "breast_standardized_512_validation.json"
TEST_JSON_PATH = REPORT_DIR / "breast_standardized_512_test.json"
TEST_PRED_PATH = REPORT_DIR / "breast_standardized_512_test_predictions.csv"
CM_TEST_PATH = REPORT_DIR / "breast_standardized_512_test_cm.png"
ROC_TEST_PATH = REPORT_DIR / "breast_standardized_512_test_roc.png"

IMAGE_SIZE = 512
BATCH_SIZE = 2  # 2 pairs = 4 full 512x512 images per batch (fits in 6GB VRAM)
WARMUP_EPOCHS = 2
FINETUNE_EPOCHS = 18
TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS
NUM_WORKERS = 0
SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def resolve_image_path(raw_path: str) -> Path:
    raw = str(raw_path).strip().replace("\\", "/")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if len(parts) >= 2:
        candidate = IMAGE_ROOT_512 / parts[-2] / parts[-1]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find 512x512 image for: {raw_path}")


def normalize_label(row) -> int:
    val = row.get("label", None)
    if val is not None and not pd.isna(val):
        try:
            n = int(float(val))
            if n in (0, 1):
                return n
        except (TypeError, ValueError):
            pass
    pathology = normalize_text(row.get("pathology", ""))
    if pathology.startswith("BENIGN"):
        return 0
    if pathology.startswith("MALIGNANT"):
        return 1
    raise ValueError(f"Unknown class label: {row.to_dict()}")


PAIR_KEY_COLUMNS = [
    "patient_id", "breast", "abnormality_type",
    "pathology", "assessment", "subtlety",
]


def prepare_rows(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["view_norm"] = df["view"].map(normalize_text)
    df["breast_norm"] = df["breast"].map(normalize_text)
    df["type_norm"] = df["abnormality_type"].map(normalize_text)
    df["pathology_norm"] = df["pathology"].map(normalize_text)
    df["assessment_norm"] = df["assessment"].map(normalize_text)
    df["subtlety_norm"] = df["subtlety"].map(normalize_text)
    df["label_norm"] = df.apply(normalize_label, axis=1)
    df["resolved_path"] = df["image_path"].map(lambda x: str(resolve_image_path(x)))
    df["split"] = split_name

    df["pair_key"] = df.apply(
        lambda r: "||".join([
            str(r["patient_id"]), str(r["breast_norm"]), str(r["type_norm"]),
            str(r["pathology_norm"]), str(r["assessment_norm"]),
            str(r["subtlety_norm"]), str(r["label_norm"]),
        ]), axis=1)
    return df


def build_pairs(df: pd.DataFrame, split_name: str):
    pairs = []
    for _, group in df.groupby("pair_key", sort=False):
        cc = group[group["view_norm"] == "CC"]
        mlo = group[group["view_norm"] == "MLO"]
        if len(cc) != 1 or len(mlo) != 1:
            continue
        cc_row = cc.iloc[0]
        mlo_row = mlo.iloc[0]
        if int(cc_row["label_norm"]) != int(mlo_row["label_norm"]):
            continue
        cc_path = Path(cc_row["resolved_path"])
        mlo_path = Path(mlo_row["resolved_path"])
        if not cc_path.is_file() or not mlo_path.is_file():
            continue
        pairs.append({
            "patient_id": str(cc_row["patient_id"]),
            "breast": str(cc_row["breast_norm"]),
            "pathology": str(cc_row["pathology_norm"]),
            "cc_path": str(cc_path),
            "mlo_path": str(mlo_path),
            "label": int(cc_row["label_norm"]),
        })
    df_pairs = pd.DataFrame(pairs)
    print(f"[{split_name.upper()}] Paired cases: {len(df_pairs)} "
          f"(Benign={int((df_pairs['label']==0).sum())}, Malignant={int((df_pairs['label']==1).sum())})")
    return df_pairs


class PairedTransform512:
    def __init__(self, training: bool):
        self.training = training

    def __call__(self, cc: Image.Image, mlo: Image.Image):
        # Base resize to 512x512
        cc = TF.resize(cc, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)
        mlo = TF.resize(mlo, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)

        if self.training:
            # Vertical flip (chest wall is on left, so vertical flip preserves left-orientation)
            if random.random() < 0.5:
                cc = TF.vflip(cc)
                mlo = TF.vflip(mlo)

            # Synchronized slight rotation (+/-8 deg)
            angle = float(torch.empty(1).uniform_(-8.0, 8.0).item())
            scale = float(torch.empty(1).uniform_(0.95, 1.05).item())
            translate = [
                int(torch.empty(1).uniform_(-0.03, 0.03).item() * IMAGE_SIZE),
                int(torch.empty(1).uniform_(-0.03, 0.03).item() * IMAGE_SIZE),
            ]
            cc = TF.affine(cc, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
                           interpolation=InterpolationMode.BILINEAR)
            mlo = TF.affine(mlo, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
                            interpolation=InterpolationMode.BILINEAR)

            # Brightness and contrast
            b = float(torch.empty(1).uniform_(0.92, 1.08).item())
            c = float(torch.empty(1).uniform_(0.92, 1.08).item())
            cc = TF.adjust_brightness(cc, b)
            mlo = TF.adjust_brightness(mlo, b)
            cc = TF.adjust_contrast(cc, c)
            mlo = TF.adjust_contrast(mlo, c)

        cc = TF.normalize(TF.to_tensor(cc), MEAN, STD)
        mlo = TF.normalize(TF.to_tensor(mlo), MEAN, STD)
        return cc, mlo


class PairedMammogramDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, training: bool):
        self.pairs = pairs.reset_index(drop=True)
        self.transform = PairedTransform512(training=training)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        with Image.open(row["cc_path"]) as im:
            cc = im.convert("RGB")
        with Image.open(row["mlo_path"]) as im:
            mlo = im.convert("RGB")
        cc, mlo = self.transform(cc, mlo)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return cc, mlo, label, str(row["patient_id"]), str(row["cc_path"]), str(row["mlo_path"])


class DualViewDenseNet512(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        in_ch = 1024

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        view_dim = in_ch * 2  # 2048 dims

        self.cc_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(view_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
        self.mlo_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(view_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

        fused_dim = view_dim * 4  # 8192 dims
        self.fused_classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def extract_descriptor(self, x):
        feat = self.backbone(x)
        avg_pool = self.gap(feat).flatten(1)
        max_pool = self.gmp(feat).flatten(1)
        return torch.cat([avg_pool, max_pool], dim=1)

    def forward(self, cc, mlo):
        batch = cc.shape[0]
        both = torch.cat([cc, mlo], dim=0)
        both_desc = self.extract_descriptor(both)
        v_cc, v_mlo = both_desc[:batch], both_desc[batch:]

        logits_cc = self.cc_head(v_cc)
        logits_mlo = self.mlo_head(v_mlo)

        diff = torch.abs(v_cc - v_mlo)
        prod = v_cc * v_mlo
        fused = torch.cat([v_cc, v_mlo, diff, prod], dim=1)
        logits_fused = self.fused_classifier(fused)

        return logits_fused, logits_cc, logits_mlo


def evaluate_model(model, loader, threshold=0.5, use_tta=True):
    model.eval()
    y_true, y_prob, metadata = [], [], []

    with torch.no_grad():
        for cc, mlo, labels, p_ids, cc_paths, mlo_paths in loader:
            cc = cc.to(DEVICE)
            mlo = mlo.to(DEVICE)

            if not use_tta:
                lf, lcc, lmlo = model(cc, mlo)
                pf = torch.softmax(lf, dim=1)[:, 1]
                pcc = torch.softmax(lcc, dim=1)[:, 1]
                pmlo = torch.softmax(lmlo, dim=1)[:, 1]
                p = 0.6 * pf + 0.4 * torch.maximum(pcc, pmlo)
            else:
                probs_list = []
                # 1. Base
                lf, lcc, lmlo = model(cc, mlo)
                p1 = 0.6 * torch.softmax(lf, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc, dim=1)[:, 1], torch.softmax(lmlo, dim=1)[:, 1])
                probs_list.append(p1)

                # 2. V-flip
                lf, lcc, lmlo = model(TF.vflip(cc), TF.vflip(mlo))
                p2 = 0.6 * torch.softmax(lf, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc, dim=1)[:, 1], torch.softmax(lmlo, dim=1)[:, 1])
                probs_list.append(p2)

                # 3. Brightness up
                lf, lcc, lmlo = model(TF.adjust_brightness(cc, 1.05), TF.adjust_brightness(mlo, 1.05))
                p3 = 0.6 * torch.softmax(lf, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc, dim=1)[:, 1], torch.softmax(lmlo, dim=1)[:, 1])
                probs_list.append(p3)

                # 4. Zoom in 3%
                cc_z = TF.affine(cc, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                mlo_z = TF.affine(mlo, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                lf, lcc, lmlo = model(cc_z, mlo_z)
                p4 = 0.6 * torch.softmax(lf, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc, dim=1)[:, 1], torch.softmax(lmlo, dim=1)[:, 1])
                probs_list.append(p4)

                p = torch.stack(probs_list, dim=0).mean(dim=0)

            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(p.cpu().numpy().tolist())

            for i in range(len(p_ids)):
                metadata.append({
                    "patient_id": str(p_ids[i]),
                    "cc_path": str(cc_paths[i]),
                    "mlo_path": str(mlo_paths[i]),
                    "y_true": int(labels[i].item()),
                    "probability": float(p[i].item()),
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
    print("=" * 76)
    print("CancerScan AI | Dual-View 512x512 DenseNet (Standardized Preprocessed)")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print("=" * 76)

    # Prepare dataset
    train_rows = prepare_rows(TRAIN_CSV, "train")
    val_rows = prepare_rows(VAL_CSV, "val")
    test_rows = prepare_rows(TEST_CSV, "test")

    train_pairs = build_pairs(train_rows, "train")
    val_pairs = build_pairs(val_rows, "val")
    test_pairs = build_pairs(test_rows, "test")

    train_ds = PairedMammogramDataset(train_pairs, training=True)
    val_ds = PairedMammogramDataset(val_pairs, training=False)
    test_ds = PairedMammogramDataset(test_pairs, training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = DualViewDenseNet512().to(DEVICE)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    # Calculate class weights for training
    counts = train_pairs["label"].value_counts().to_dict()
    n_b, n_m = counts[0], counts[1]
    weight_m = float(n_b / n_m)
    print(f"Class counts: Benign={n_b}, Malignant={n_m} -> Malignant weight: {weight_m:.2f}")

    # -------------------------------------------------------------
    # PHASE 1: Warmup (train fusion heads only)
    # -------------------------------------------------------------
    print(f"\n--- Phase 1: Warmup ({WARMUP_EPOCHS} epochs) ---")
    for p in model.backbone.parameters():
        p.requires_grad = False

    head_params = (
        list(model.fused_classifier.parameters()) +
        list(model.cc_head.parameters()) +
        list(model.mlo_head.parameters())
    )
    optimizer = torch.optim.AdamW(head_params, lr=3e-4, weight_decay=1e-4)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for cc, mlo, labels, *_ in train_loader:
            cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                lf, lcc, lmlo = model(cc, mlo)
                loss = ce_loss(lf, labels) + 0.3 * ce_loss(lcc, labels) + 0.3 * ce_loss(lmlo, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_m, yt, yp, _, _ = evaluate_model(model, val_loader, threshold=0.5, use_tta=False)
        print(f"[Warmup {epoch}/{WARMUP_EPOCHS}] Loss={avg_loss:.4f} | Val Acc={val_m['accuracy']*100:.1f}% AUC={val_m['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # PHASE 2: Full Fine-Tuning (unfreeze dense blocks)
    # -------------------------------------------------------------
    print(f"\n--- Phase 2: Full Fine-Tuning ({FINETUNE_EPOCHS} epochs) ---")
    for p in model.backbone.parameters():
        p.requires_grad = True

    # Differential learning rates
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.fused_classifier.parameters()) +
        list(model.cc_head.parameters()) +
        list(model.mlo_head.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 2e-5},
        {"params": head_params, "lr": 1.5e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS * len(train_loader), eta_min=1e-6
    )

    best_composite = 0.0
    best_threshold = 0.5
    best_weights = None
    history = []

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for cc, mlo, labels, *_ in train_loader:
            cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                lf, lcc, lmlo = model(cc, mlo)
                loss = ce_loss(lf, labels) + 0.3 * ce_loss(lcc, labels) + 0.3 * ce_loss(lmlo, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_m, y_true_v, y_prob_v, _, _ = evaluate_model(model, val_loader, threshold=0.5, use_tta=True)

        # Find optimal Youden's J threshold on validation
        fpr, tpr, threshs = roc_curve(y_true_v, y_prob_v)
        j_scores = tpr - fpr
        opt_t = float(np.clip(threshs[np.argmax(j_scores)], 0.25, 0.75))
        val_m_opt, _, _, _, _ = evaluate_model(model, val_loader, threshold=opt_t, use_tta=True)

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

        print(f"[Epoch {epoch:02d}/{FINETUNE_EPOCHS}] Loss={avg_loss:.4f} | "
              f"Val Acc={val_m_opt['accuracy']*100:.1f}% BalAcc={val_m_opt['balanced_accuracy']*100:.1f}% "
              f"Sens={val_m_opt['recall_sensitivity']*100:.1f}% Spec={val_m_opt['specificity']*100:.1f}% "
              f"AUC={val_m_opt['roc_auc']:.4f} T*={opt_t:.3f}{star}")

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
    print("FINAL EVALUATION ON HELD-OUT TEST SET (BEST CHECKPOINT)")
    print("=" * 76)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    opt_threshold = checkpoint["threshold"]
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (T* = {opt_threshold:.3f})")

    # 1. Standard 0.50 threshold
    test_m_50, yt, yp, ypred_50, _ = evaluate_model(model, test_loader, threshold=0.50, use_tta=True)
    # 2. Optimal validation threshold
    test_m_opt, yt, yp, ypred_opt, df_preds = evaluate_model(model, test_loader, threshold=opt_threshold, use_tta=True)

    df_preds.to_csv(TEST_PRED_PATH, index=False)

    summary = {
        "model": "DualViewDenseNet512_Standardized",
        "best_epoch": checkpoint["epoch"],
        "optimal_threshold": float(opt_threshold),
        "test_standard_050": test_m_50,
        "test_optimal_threshold": test_m_opt,
    }
    TEST_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nClassification Report (Standard Threshold = 0.50):")
    print(classification_report(yt, ypred_50, target_names=CLASS_NAMES, digits=4))

    print(f"\nClassification Report (Optimal Threshold = {opt_threshold:.3f}):")
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
    ax.set_title(f"Test Confusion Matrix (T* = {opt_threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontweight="bold", fontsize=13)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_TEST_PATH, dpi=160)
    plt.close(fig)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(yt, yp)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {test_m_opt['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test ROC Curve (Standardized 512x512)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_TEST_PATH, dpi=160)
    plt.close(fig)

    print("\n" + "=" * 76)
    print("TRAINING & EVALUATION COMPLETE")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Test report saved to: {TEST_JSON_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
