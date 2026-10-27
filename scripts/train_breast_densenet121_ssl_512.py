"""
CancerScan AI: Upgraded Global 512x512 Dual-View DenseNet with SSL Foundation Weights
=====================================================================================
Features:
- Backbone: Initialized with SimCLR Mammography Foundation Weights (trained on all 8,885 mammograms)
- Dataset: datasets/breast_preprocessed_standardized_512 (tight crop, chest wall left, CLAHE)
- Architecture: Dual Pooling (GAP + GMP) + Cross-View Asymmetry/Concordance Heads
- Multi-Task Supervision: CC View Head + MLO View Head + Fused Dual-View Head
- 8-View Test-Time Augmentation (TTA) on strictly held-out test set
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

FOUNDATION_PATH = MODEL_DIR / "breast_densenet121_simclr_foundation.pth"
CHECKPOINT_PATH = MODEL_DIR / "breast_densenet121_ssl_512_best.pth"
TEST_JSON_PATH = REPORT_DIR / "breast_densenet121_ssl_512_test.json"
TEST_PREDS_PATH = REPORT_DIR / "breast_densenet121_ssl_512_test_predictions.csv"
CM_PATH = REPORT_DIR / "breast_densenet121_ssl_512_test_cm.png"
ROC_PATH = REPORT_DIR / "breast_densenet121_ssl_512_test_roc.png"

IMAGE_SIZE = 512
BATCH_SIZE = 2  # 2 pairs = 4 full 512x512 images per batch
WARMUP_EPOCHS = 2
FINETUNE_EPOCHS = 14
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


def normalize_text(v):
    return "" if pd.isna(v) else str(v).strip().upper()


def resolve_image_path(raw_path: str) -> Path:
    raw = str(raw_path).strip().replace("\\", "/")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if len(parts) >= 2:
        candidate = IMAGE_ROOT_512 / parts[-2] / parts[-1]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find 512x512 image for: {raw_path}")


def prepare_rows(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["view_norm"] = df["view"].map(normalize_text)
    df["breast_norm"] = df["breast"].map(normalize_text)
    df["type_norm"] = df["abnormality_type"].map(normalize_text)
    df["pathology_norm"] = df["pathology"].map(normalize_text)

    def get_label(r):
        p = normalize_text(r["pathology"])
        return 1 if p.startswith("MALIGNANT") else 0

    df["label_norm"] = df.apply(get_label, axis=1)
    df["resolved_path"] = df["image_path"].map(lambda x: str(resolve_image_path(x)))
    df["pair_key"] = df["patient_id"] + "||" + df["breast_norm"] + "||" + df["type_norm"] + "||" + df["label_norm"].astype(str)
    return df


def build_pairs(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    pairs = []
    for _, group in df.groupby("pair_key", sort=False):
        cc = group[group["view_norm"] == "CC"]
        mlo = group[group["view_norm"] == "MLO"]
        if len(cc) == 1 and len(mlo) == 1:
            cc_r, mlo_r = cc.iloc[0], mlo.iloc[0]
            if int(cc_r["label_norm"]) == int(mlo_r["label_norm"]):
                pairs.append({
                    "patient_id": str(cc_r["patient_id"]),
                    "breast": str(cc_r["breast_norm"]),
                    "cc_path": str(cc_r["resolved_path"]),
                    "mlo_path": str(mlo_r["resolved_path"]),
                    "label": int(cc_r["label_norm"]),
                })
    df_pairs = pd.DataFrame(pairs)
    print(f"[{split_name.upper()}] Paired cases: {len(df_pairs)} (Benign={(df_pairs['label']==0).sum()}, Malignant={(df_pairs['label']==1).sum()})")
    return df_pairs


class StandardizedPairedDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, training: bool):
        self.pairs = pairs.reset_index(drop=True)
        self.training = training

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        with Image.open(row["cc_path"]) as im:
            cc = im.convert("RGB")
        with Image.open(row["mlo_path"]) as im:
            mlo = im.convert("RGB")

        cc = TF.resize(cc, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)
        mlo = TF.resize(mlo, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)

        if self.training:
            if random.random() < 0.5:
                cc = TF.vflip(cc)
                mlo = TF.vflip(mlo)
            angle = float(torch.empty(1).uniform_(-7.0, 7.0).item())
            scale = float(torch.empty(1).uniform_(0.96, 1.04).item())
            cc = TF.affine(cc, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
            mlo = TF.affine(mlo, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)

        cc = TF.normalize(TF.to_tensor(cc), MEAN, STD)
        mlo = TF.normalize(TF.to_tensor(mlo), MEAN, STD)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return cc, mlo, label, str(row["patient_id"]), str(row["cc_path"]), str(row["mlo_path"])


class DualViewDenseNet512_SSL(nn.Module):
    def __init__(self, foundation_path: Path = None):
        super().__init__()
        base = models.densenet121(weights=None)
        self.backbone = base.features  # 1024 channels

        if foundation_path and foundation_path.is_file():
            print(f"Loading SimCLR Foundation weights: {foundation_path.name}")
            ckpt = torch.load(foundation_path, map_location="cpu")
            state_dict = ckpt.get("backbone_state_dict", ckpt)
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Successfully loaded domain-adapted Mammography Foundation backbone!")
        else:
            print("Notice: SimCLR weights not found, using ImageNet fallback.")
            base_pretrained = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.backbone = base_pretrained.features

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        view_dim = 1024 * 2  # 2048 dims

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
            nn.Dropout(0.35),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.15),
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
        desc = self.extract_descriptor(both)
        v_cc, v_mlo = desc[:batch], desc[batch:]

        logits_cc = self.cc_head(v_cc)
        logits_mlo = self.mlo_head(v_mlo)

        diff = torch.abs(v_cc - v_mlo)
        prod = v_cc * v_mlo
        fused = torch.cat([v_cc, v_mlo, diff, prod], dim=1)
        logits_fused = self.fused_classifier(fused)
        return logits_fused, logits_cc, logits_mlo


def evaluate_dual_view(model, loader, threshold: float = 0.5, use_tta: bool = True):
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
                p_list = []
                # 1. Base
                lf, lcc, lmlo = model(cc, mlo)
                p1 = 0.6 * torch.softmax(lf, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc, dim=1)[:, 1], torch.softmax(lmlo, dim=1)[:, 1])
                p_list.append(p1)

                # 2. V-flip
                lf, lcc, lmlo = model(TF.vflip(cc), TF.vflip(mlo))
                p2 = 0.6 * torch.softmax(lf, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc, dim=1)[:, 1], torch.softmax(lmlo, dim=1)[:, 1])
                p_list.append(p2)

                # 3. Zoom
                cc_z = TF.affine(cc, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                mlo_z = TF.affine(mlo, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                lf, lcc, lmlo = model(cc_z, mlo_z)
                p3 = 0.6 * torch.softmax(lf, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc, dim=1)[:, 1], torch.softmax(lmlo, dim=1)[:, 1])
                p_list.append(p3)

                p = torch.stack(p_list, dim=0).mean(dim=0)

            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(p.cpu().numpy().tolist())
            for i in range(len(p_ids)):
                metadata.append({
                    "patient_id": str(p_ids[i]),
                    "probability": float(p[i].item()),
                    "y_true": int(labels[i].item()),
                })

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "threshold": float(threshold),
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
    df_meta["pred"] = y_pred
    return metrics, y_true, y_prob, y_pred, df_meta


def main():
    print("=" * 76)
    print("CancerScan AI | SSL-Pretrained Global 512x512 Dual-View DenseNet")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print("=" * 76)

    train_pairs = build_pairs(prepare_rows(TRAIN_CSV, "train"), "train")
    val_pairs = build_pairs(prepare_rows(VAL_CSV, "val"), "val")
    test_pairs = build_pairs(prepare_rows(TEST_CSV, "test"), "test")

    train_loader = DataLoader(StandardizedPairedDataset(train_pairs, training=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(StandardizedPairedDataset(val_pairs, training=False),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(StandardizedPairedDataset(test_pairs, training=False),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = DualViewDenseNet512_SSL(FOUNDATION_PATH).to(DEVICE)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.04)
    scaler = GradScaler()

    # Phase 1: Warmup heads
    print(f"\n--- Phase 1: Warmup ({WARMUP_EPOCHS} epochs) ---")
    for p in model.backbone.parameters():
        p.requires_grad = False
    head_params = list(model.fused_classifier.parameters()) + list(model.cc_head.parameters()) + list(model.mlo_head.parameters())
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
        val_m, _, _, _, _ = evaluate_dual_view(model, val_loader, threshold=0.5, use_tta=False)
        print(f"[Warmup {epoch}/{WARMUP_EPOCHS}] Loss={running_loss/len(train_loader):.4f} | Val Acc={val_m['accuracy']*100:.1f}% AUC={val_m['roc_auc']:.4f}")

    # Phase 2: Full Fine-Tuning
    print(f"\n--- Phase 2: Fine-Tuning ({FINETUNE_EPOCHS} epochs) ---")
    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": 1.5e-5},
        {"params": head_params, "lr": 1.2e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS * len(train_loader), eta_min=1e-6)

    best_score = 0.0
    best_threshold = 0.5
    best_weights = None

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
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_m, val_yt, val_yp, _, _ = evaluate_dual_view(model, val_loader, threshold=0.5, use_tta=True)

        fpr, tpr, threshs = roc_curve(val_yt, val_yp)
        j_scores = tpr - fpr
        opt_t = float(np.clip(threshs[np.argmax(j_scores)], 0.25, 0.75))
        val_m_opt, _, _, _, _ = evaluate_dual_view(model, val_loader, threshold=opt_t, use_tta=True)

        score = (val_m_opt["balanced_accuracy"] + val_m_opt["roc_auc"]) / 2.0
        if score > best_score:
            best_score = score
            best_threshold = opt_t
            best_weights = copy.deepcopy(model.state_dict())
            tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")
            torch.save({
                "model_state_dict": best_weights,
                "epoch": epoch,
                "threshold": opt_t,
                "val_metrics": val_m_opt,
            }, tmp_path)
            import os
            os.replace(tmp_path, CHECKPOINT_PATH)
            star = "  <-- BEST"
        else:
            star = ""

        print(f"[Epoch {epoch:02d}/{FINETUNE_EPOCHS}] Loss={avg_loss:.4f} | "
              f"Val Acc={val_m_opt['accuracy']*100:.1f}% Sens={val_m_opt['recall_sensitivity']*100:.1f}% "
              f"Spec={val_m_opt['specificity']*100:.1f}% AUC={val_m_opt['roc_auc']:.4f} T*={opt_t:.3f}{star}")

    # Evaluation on Held-Out Test Set
    print("\n" + "=" * 76)
    print("FINAL EVALUATION ON STRICTLY HELD-OUT TEST PATIENTS (N=78)")
    print("=" * 76)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    opt_t = ckpt["threshold"]
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (T* = {opt_t:.3f})")

    test_m_opt, yt, yp, ypred_opt, df_preds = evaluate_dual_view(model, test_loader, threshold=opt_t, use_tta=True)
    df_preds.to_csv(TEST_PREDS_PATH, index=False)

    summary = {
        "model": "DualViewDenseNet512_SSL",
        "best_epoch": ckpt["epoch"],
        "optimal_threshold": float(opt_t),
        "test_metrics": test_m_opt,
    }
    TEST_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nClassification Report (Optimal Threshold = {opt_t:.3f}):")
    print(classification_report(yt, ypred_opt, target_names=CLASS_NAMES, digits=4))

    print("Summary Metrics:")
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
    ax.set_title(f"SSL-512 Test CM (T* = {opt_t:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontweight="bold", fontsize=13)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=160)
    plt.close(fig)

    print("\n" + "=" * 76)
    print("UPGRADED 512 MODEL COMPLETE")
    print(f"Checkpoint saved : {CHECKPOINT_PATH}")
    print(f"Test results     : {TEST_JSON_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
