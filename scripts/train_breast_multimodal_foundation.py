"""
CancerScan AI: Fine-Tuning Domain-Adapted Foundation Model
==========================================================
Downstream fine-tuning of the SimCLR Mammography Foundation Encoder
(trained on all 8,885 images) combined with:
1. Standardized 512x512 Dual-View Mammograms (CC + MLO)
2. High-Capacity Dual Pooling (GAP + GMP)
3. Cross-View Asymmetry (|v_CC - v_MLO|) & Concordance (v_CC * v_MLO) Heads
4. Patient Clinical Tabular Conditioning (BI-RADS assessment + subtlety embeddings)
5. Trained on Train + Validation combined pairs (482 pairs)
6. Evaluated on the strictly Held-Out Test Set (83 pairs)
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

FOUNDATION_WEIGHTS = MODEL_DIR / "breast_densenet121_simclr_foundation.pth"
CHECKPOINT_PATH = MODEL_DIR / "breast_multimodal_foundation_best.pth"
TEST_JSON_PATH = REPORT_DIR / "breast_multimodal_foundation_test.json"
TEST_PRED_PATH = REPORT_DIR / "breast_multimodal_foundation_test_preds.csv"
CM_PATH = REPORT_DIR / "breast_multimodal_foundation_test_cm.png"
ROC_PATH = REPORT_DIR / "breast_multimodal_foundation_test_roc.png"

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 15
LR_BACKBONE = 1.5e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.04
SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
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
    raise FileNotFoundError(f"Missing image: {raw_path}")


def prepare_pairs_with_metadata(csv_path: Path):
    df = pd.read_csv(csv_path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["view_norm"] = df["view"].map(normalize_text)
    df["breast_norm"] = df["breast"].map(normalize_text)
    df["type_norm"] = df["abnormality_type"].map(normalize_text)
    df["pathology_norm"] = df["pathology"].map(normalize_text)
    df["resolved_path"] = df["image_path"].map(lambda x: str(resolve_image_path(x)))

    def get_label(r):
        p = normalize_text(r["pathology"])
        return 1 if p.startswith("MALIGNANT") else 0

    df["label_norm"] = df.apply(get_label, axis=1)
    df["pair_key"] = df["patient_id"] + "||" + df["breast_norm"] + "||" + df["type_norm"] + "||" + df["label_norm"].astype(str)

    type_to_idx = {"CALCIFICATION": 1, "MASS": 2}
    pairs = []
    for _, group in df.groupby("pair_key", sort=False):
        cc = group[group["view_norm"] == "CC"]
        mlo = group[group["view_norm"] == "MLO"]
        if len(cc) == 1 and len(mlo) == 1:
            cc_r = cc.iloc[0]
            mlo_r = mlo.iloc[0]
            if int(cc_r["label_norm"]) == int(mlo_r["label_norm"]):
                ass = int(cc_r["assessment"]) if pd.notna(cc_r.get("assessment")) and str(cc_r["assessment"]).isdigit() else 4
                ass = min(max(ass, 0), 5)
                sub = int(cc_r["subtlety"]) if pd.notna(cc_r.get("subtlety")) and str(cc_r["subtlety"]).isdigit() else 3
                sub = min(max(sub, 1), 5)
                abn = type_to_idx.get(normalize_text(cc_r.get("abnormality_type")), 0)

                pairs.append({
                    "patient_id": str(cc_r["patient_id"]),
                    "cc_path": str(cc_r["resolved_path"]),
                    "mlo_path": str(mlo_r["resolved_path"]),
                    "assessment": ass,
                    "subtlety": sub,
                    "abnormality_type_idx": abn,
                    "label": int(cc_r["label_norm"]),
                })
    return pd.DataFrame(pairs)


class MultimodalStandardizedDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, training: bool):
        self.pairs = pairs.reset_index(drop=True)
        self.training = training

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        with Image.open(row["cc_path"]) as im:
            cc = im.convert("RGB")
        with Image.open(row["mlo_path"]) as im:
            mlo = im.convert("RGB")

        cc = TF.resize(cc, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)
        mlo = TF.resize(mlo, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)

        if self.training:
            # Symmetrical vertical flip (preserves chest-wall on left)
            if random.random() < 0.5:
                cc = TF.vflip(cc)
                mlo = TF.vflip(mlo)
            angle = float(torch.empty(1).uniform_(-8.0, 8.0).item())
            scale = float(torch.empty(1).uniform_(0.96, 1.04).item())
            cc = TF.affine(cc, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
            mlo = TF.affine(mlo, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)

        cc = TF.normalize(TF.to_tensor(cc), MEAN, STD)
        mlo = TF.normalize(TF.to_tensor(mlo), MEAN, STD)
        clinical = torch.tensor([row["assessment"], row["subtlety"], row["abnormality_type_idx"]], dtype=torch.long)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return cc, mlo, clinical, label, str(row["patient_id"]), str(row["cc_path"]), str(row["mlo_path"])


class VisionTabularFoundationNet(nn.Module):
    def __init__(self, foundation_weights_path: Path = None):
        super().__init__()
        base = models.densenet121(weights=None)
        self.vision_backbone = base.features  # 1024 channels

        if foundation_weights_path and foundation_weights_path.is_file():
            print(f"Loading SimCLR Foundation weights from: {foundation_weights_path.name}")
            ckpt = torch.load(foundation_weights_path, map_location="cpu")
            state_dict = ckpt.get("backbone_state_dict", ckpt)
            self.vision_backbone.load_state_dict(state_dict, strict=False)
            print("Successfully loaded domain-adapted Mammography Foundation backbone!")
        else:
            print("Notice: Foundation weights not found, using ImageNet fallback.")
            base_pretrained = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.vision_backbone = base_pretrained.features

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # View descriptor = GAP (1024) + GMP (1024) = 2048 dims
        # Fused view = [CC, MLO, |CC - MLO|, CC * MLO] = 2048 * 4 = 8192 dims
        self.vis_proj = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(8192, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
        )

        # Tabular clinical embeddings
        self.embed_assessment = nn.Embedding(6, 16)      # BI-RADS 0 to 5
        self.embed_subtlety = nn.Embedding(6, 8)         # 1 to 5
        self.embed_type = nn.Embedding(3, 8)             # unknown, calcification, mass
        self.tab_proj = nn.Sequential(
            nn.Linear(16 + 8 + 8, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # Fused classification head: 128 (vision) + 32 (clinical) = 160 dims
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def extract_view(self, x):
        feat = self.vision_backbone(x)
        avg_pool = self.gap(feat).flatten(1)
        max_pool = self.gmp(feat).flatten(1)
        return torch.cat([avg_pool, max_pool], dim=1)

    def forward(self, cc, mlo, clinical):
        batch = cc.shape[0]
        both = torch.cat([cc, mlo], dim=0)
        desc = self.extract_view(both)
        v_cc, v_mlo = desc[:batch], desc[batch:]

        diff = torch.abs(v_cc - v_mlo)
        prod = v_cc * v_mlo
        fused_vis = torch.cat([v_cc, v_mlo, diff, prod], dim=1)
        vis_emb = self.vis_proj(fused_vis)

        e_ass = self.embed_assessment(clinical[:, 0])
        e_sub = self.embed_subtlety(clinical[:, 1])
        e_typ = self.embed_type(clinical[:, 2])
        tab_raw = torch.cat([e_ass, e_sub, e_typ], dim=1)
        tab_emb = self.tab_proj(tab_raw)

        joint = torch.cat([vis_emb, tab_emb], dim=1)
        logits = self.classifier(joint)
        return logits


def evaluate_test(model, loader, threshold=0.5, use_tta=True):
    model.eval()
    y_true, y_prob, meta = [], [], []

    with torch.no_grad():
        for cc, mlo, clinical, labels, p_ids, cc_paths, mlo_paths in loader:
            cc, mlo, clinical = cc.cuda(), mlo.cuda(), clinical.cuda()

            if not use_tta:
                logits = model(cc, mlo, clinical)
                p = torch.softmax(logits, dim=1)[:, 1]
            else:
                p_list = []
                # 1. Base
                p_list.append(torch.softmax(model(cc, mlo, clinical), dim=1)[:, 1])
                # 2. V-flip
                p_list.append(torch.softmax(model(TF.vflip(cc), TF.vflip(mlo), clinical), dim=1)[:, 1])
                # 3. Zoom
                cc_z = TF.affine(cc, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                mlo_z = TF.affine(mlo, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                p_list.append(torch.softmax(model(cc_z, mlo_z, clinical), dim=1)[:, 1])
                p = torch.stack(p_list, dim=0).mean(dim=0)

            y_true.extend(labels.numpy().tolist())
            y_prob.extend(p.cpu().numpy().tolist())
            for i in range(len(p_ids)):
                meta.append({
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
    df_meta = pd.DataFrame(meta)
    df_meta["y_pred"] = y_pred
    return metrics, y_true, y_prob, y_pred, df_meta


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("CancerScan AI | Fine-Tuning SimCLR Mammography Foundation Model")
    print("=" * 76)

    train_pairs = prepare_pairs_with_metadata(TRAIN_CSV)
    val_pairs = prepare_pairs_with_metadata(VAL_CSV)
    test_pairs = prepare_pairs_with_metadata(TEST_CSV)

    # Train on Train + Validation combined pairs (482 pairs)
    combined_train_pairs = pd.concat([train_pairs, val_pairs], ignore_index=True)
    print(f"Training Cases (Train + Val combined) : {len(combined_train_pairs)} pairs "
          f"(Benign={int((combined_train_pairs['label']==0).sum())}, Malignant={int((combined_train_pairs['label']==1).sum())})")
    print(f"Held-Out Test Cases                  : {len(test_pairs)} pairs "
          f"(Benign={int((test_pairs['label']==0).sum())}, Malignant={int((test_pairs['label']==1).sum())})")

    train_loader = DataLoader(
        MultimodalStandardizedDataset(combined_train_pairs, training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        MultimodalStandardizedDataset(val_pairs, training=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        MultimodalStandardizedDataset(test_pairs, training=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    model = VisionTabularFoundationNet(FOUNDATION_WEIGHTS).to(device)

    # Fine-tune denseblock3, denseblock4, norm5 of backbone + all head layers
    for param in model.vision_backbone.parameters():
        param.requires_grad = False
    for name, param in model.vision_backbone.named_parameters():
        if "denseblock4" in name or "denseblock3" in name or "norm5" in name:
            param.requires_grad = True

    backbone_params = [p for p in model.vision_backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.vis_proj.parameters()) +
        list(model.embed_assessment.parameters()) +
        list(model.embed_subtlety.parameters()) +
        list(model.embed_type.parameters()) +
        list(model.tab_proj.parameters()) +
        list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-7)

    # Balanced class weights
    total = len(combined_train_pairs)
    w0 = total / (2.0 * max((combined_train_pairs["label"] == 0).sum(), 1))
    w1 = total / (2.0 * max((combined_train_pairs["label"] == 1).sum(), 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([w0, w1], dtype=torch.float32, device=device),
        label_smoothing=LABEL_SMOOTHING
    )
    scaler = GradScaler()

    best_val_score = 0.0
    best_threshold = 0.5
    best_weights = None

    print("\n--- Training Foundation Model on Standardized Combined Dataset ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for cc, mlo, clinical, labels, *_ in train_loader:
            cc, mlo, clinical, labels = cc.to(device), mlo.to(device), clinical.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(cc, mlo, clinical)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_m, val_y, val_p, _, _ = evaluate_test(model, val_loader, threshold=0.5, use_tta=True)

        fpr, tpr, threshs = roc_curve(val_y, val_p)
        j_scores = tpr - fpr
        opt_t = float(np.clip(threshs[np.argmax(j_scores)], 0.25, 0.75))
        val_m_opt, _, _, _, _ = evaluate_test(model, val_loader, threshold=opt_t, use_tta=True)

        score = (val_m_opt["balanced_accuracy"] + val_m_opt["roc_auc"]) / 2.0
        is_best = score > best_val_score
        if is_best:
            best_val_score = score
            best_threshold = opt_t
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "epoch": epoch,
                "threshold": opt_t,
                "val_score": score,
            }, CHECKPOINT_PATH)
            star = "  <-- BEST"
        else:
            star = ""

        print(f"[Epoch {epoch:02d}/{EPOCHS}] Loss={avg_loss:.4f} | "
              f"Val Acc={val_m_opt['accuracy']*100:.1f}% Sens={val_m_opt['recall_sensitivity']*100:.1f}% "
              f"Spec={val_m_opt['specificity']*100:.1f}% AUC={val_m_opt['roc_auc']:.4f} T*={opt_t:.3f}{star}")

    # Final Evaluation on the Held-Out Test Set
    print("\n" + "=" * 76)
    print("FINAL EVALUATION ON COMPLETELY HELD-OUT TEST SET")
    print("=" * 76)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    opt_t = ckpt["threshold"]
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (Optimal T* = {opt_t:.3f})")

    # 1. Standard 0.50 threshold
    test_m_50, yt, yp, ypred_50, _ = evaluate_test(model, test_loader, threshold=0.50, use_tta=True)
    # 2. Optimal threshold
    test_m_opt, yt, yp, ypred_opt, df_preds = evaluate_test(model, test_loader, threshold=opt_t, use_tta=True)

    df_preds.to_csv(TEST_PRED_PATH, index=False)

    summary = {
        "model": "VisionTabularFoundationNet",
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
    ax.set_title(f"Test Confusion Matrix (T* = {opt_t:.2f})")
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
    ax.set_title("Test ROC Curve (Foundation Multimodal)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=160)
    plt.close(fig)

    print("\n" + "=" * 76)
    print("FOUNDATION MODEL EVALUATION COMPLETE")
    print(f"Best checkpoint : {CHECKPOINT_PATH}")
    print(f"Test results    : {TEST_JSON_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
