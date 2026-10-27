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

# ==============================================================================
# CancerScan AI | High-Performance 512x512 Multi-View DenseNet-121
# Architecture: Dual-View Feature Extractor + Mixed GAP/GMP + Asymmetry Fusion
# Multi-Task View Supervision + Cosine Annealing + Test-Time Augmentation (TTA)
# ==============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
IMAGE_ROOT_512 = DATA_DIR / "breast_preprocessed"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "breast_densenet121_multiview_512_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_densenet121_multiview_512_history.csv"
VAL_JSON_PATH = REPORT_DIR / "breast_densenet121_multiview_512_validation.json"
TEST_JSON_PATH = REPORT_DIR / "breast_densenet121_multiview_512_test.json"
VAL_PRED_PATH = REPORT_DIR / "breast_densenet121_multiview_512_val_predictions.csv"
TEST_PRED_PATH = REPORT_DIR / "breast_densenet121_multiview_512_test_predictions.csv"
ROC_VAL_PATH = REPORT_DIR / "breast_densenet121_multiview_512_roc_curve.png"
CM_VAL_PATH = REPORT_DIR / "breast_densenet121_multiview_512_val_cm.png"
CM_TEST_PATH = REPORT_DIR / "breast_densenet121_multiview_512_test_cm.png"

IMAGE_SIZE = 512
BATCH_SIZE = 2  # 2 pairs = 4 full 512x512 images per gradient step
WARMUP_EPOCHS = 2
FINETUNE_EPOCHS = 10
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
        except (ValueError, TypeError):
            pass
    pathology = normalize_text(row.get("pathology", ""))
    if pathology.startswith("BENIGN"):
        return 0
    if pathology.startswith("MALIGNANT"):
        return 1
    raise ValueError(f"Unknown pathology: {row}")


PAIR_KEY_COLUMNS = [
    "patient_id",
    "breast",
    "abnormality_type",
    "pathology",
    "assessment",
    "subtlety",
]


def prepare_and_build_pairs(csv_path: Path, split_name: str):
    df = pd.read_csv(csv_path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["view_norm"] = df["view"].map(normalize_text)
    df["breast_norm"] = df["breast"].map(normalize_text)
    df["type_norm"] = df["abnormality_type"].map(normalize_text)
    df["pathology_norm"] = df["pathology"].map(normalize_text)
    df["assessment_norm"] = df["assessment"].map(normalize_text)
    df["subtlety_norm"] = df["subtlety"].map(normalize_text)
    df["label_norm"] = df.apply(normalize_label, axis=1)
    df["resolved_path"] = df["image_path"].map(lambda x: str(resolve_image_path(x)))

    df["pair_key"] = df.apply(
        lambda r: "||".join([
            str(r["patient_id"]),
            str(r["breast_norm"]),
            str(r["type_norm"]),
            str(r["pathology_norm"]),
            str(r["assessment_norm"]),
            str(r["subtlety_norm"]),
            str(r["label_norm"]),
        ]),
        axis=1,
    )

    pairs = []
    for _, group in df.groupby("pair_key", sort=False):
        cc = group[group["view_norm"] == "CC"]
        mlo = group[group["view_norm"] == "MLO"]
        if len(cc) == 1 and len(mlo) == 1:
            cc_row = cc.iloc[0]
            mlo_row = mlo.iloc[0]
            if int(cc_row["label_norm"]) == int(mlo_row["label_norm"]):
                pairs.append({
                    "patient_id": str(cc_row["patient_id"]),
                    "breast": str(cc_row["breast_norm"]),
                    "cc_path": str(cc_row["resolved_path"]),
                    "mlo_path": str(mlo_row["resolved_path"]),
                    "label": int(cc_row["label_norm"]),
                })

    return pd.DataFrame(pairs)


class PairedTransform512:
    def __init__(self, training: bool):
        self.training = training

    def __call__(self, cc, mlo):
        cc = TF.resize(cc, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)
        mlo = TF.resize(mlo, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)

        if self.training:
            # Synchronized geometric transforms for paired mammogram projections
            angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
            translate = [
                int(torch.empty(1).uniform_(-0.04, 0.04).item() * IMAGE_SIZE),
                int(torch.empty(1).uniform_(-0.04, 0.04).item() * IMAGE_SIZE),
            ]
            scale = float(torch.empty(1).uniform_(0.96, 1.04).item())

            cc = TF.affine(cc, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR)
            mlo = TF.affine(mlo, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR)

            brightness = float(torch.empty(1).uniform_(0.94, 1.06).item())
            contrast = float(torch.empty(1).uniform_(0.94, 1.06).item())
            cc = TF.adjust_brightness(cc, brightness)
            mlo = TF.adjust_brightness(mlo, brightness)
            cc = TF.adjust_contrast(cc, contrast)
            mlo = TF.adjust_contrast(mlo, contrast)

        cc = TF.to_tensor(cc)
        mlo = TF.to_tensor(mlo)
        cc = TF.normalize(cc, mean=MEAN, std=STD)
        mlo = TF.normalize(mlo, mean=MEAN, std=STD)
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
    """
    State-of-the-art Dual-View DenseNet architecture:
    - Shared DenseNet-121 backbone
    - Dual Pooling: Global Average Pooling (parenchymal context) + Global Max Pooling (focal lesion signal)
    - Cross-View Interaction: Feature Concat + Asymmetry Difference + Concordance Product
    - Multi-Task View Supervision: Fused Head + CC View Head + MLO View Head
    """
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        in_ch = 1024

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # Each view descriptor has GAP (1024) + GMP (1024) = 2048 dims
        view_dim = in_ch * 2  # 2048

        # Auxiliary single-view classifier heads
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

        # Fused dual-view head: [v_cc, v_mlo, |v_cc - v_mlo|, v_cc * v_mlo] = 2048 * 4 = 8192 dims
        fused_dim = view_dim * 4
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

    def extract_view_descriptor(self, x):
        feat = self.backbone(x)  # B x 1024 x 16 x 16
        avg_pool = self.gap(feat).flatten(1)  # B x 1024
        max_pool = self.gmp(feat).flatten(1)  # B x 1024
        return torch.cat([avg_pool, max_pool], dim=1)  # B x 2048

    def forward(self, cc, mlo):
        batch = cc.shape[0]
        both = torch.cat([cc, mlo], dim=0)
        both_descriptors = self.extract_view_descriptor(both)
        v_cc, v_mlo = both_descriptors[:batch], both_descriptors[batch:]

        logits_cc = self.cc_head(v_cc)
        logits_mlo = self.mlo_head(v_mlo)

        # Multi-view interaction features
        diff = torch.abs(v_cc - v_mlo)
        prod = v_cc * v_mlo
        fused_rep = torch.cat([v_cc, v_mlo, diff, prod], dim=1)
        logits_fused = self.fused_classifier(fused_rep)

        return logits_fused, logits_cc, logits_mlo


def evaluate_512(model, loader, threshold=0.5, use_tta=True):
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
                # Combined case probability: 0.6 fused + 0.4 max(CC, MLO)
                p = 0.6 * pf + 0.4 * torch.maximum(pcc, pmlo)
            else:
                prob_estimates = []
                # Pass 1: Standard
                lf, lcc, lmlo = model(cc, mlo)
                p1 = 0.6 * torch.softmax(lf, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc, dim=1)[:, 1], torch.softmax(lmlo, dim=1)[:, 1])
                prob_estimates.append(p1)

                # Pass 2: Scale 1.03
                cc_z = TF.affine(cc, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                mlo_z = TF.affine(mlo, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                lf2, lcc2, lmlo2 = model(cc_z, mlo_z)
                p2 = 0.6 * torch.softmax(lf2, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc2, dim=1)[:, 1], torch.softmax(lmlo2, dim=1)[:, 1])
                prob_estimates.append(p2)

                # Pass 3: Slight brightness adjustment
                cc_b = TF.adjust_brightness(cc, 1.04)
                mlo_b = TF.adjust_brightness(mlo, 1.04)
                lf3, lcc3, lmlo3 = model(cc_b, mlo_b)
                p3 = 0.6 * torch.softmax(lf3, dim=1)[:, 1] + 0.4 * torch.maximum(torch.softmax(lcc3, dim=1)[:, 1], torch.softmax(lmlo3, dim=1)[:, 1])
                prob_estimates.append(p3)

                p = torch.stack(prob_estimates, dim=0).mean(dim=0)

            batch_n = labels.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(p.cpu().numpy().tolist())

            for i in range(batch_n):
                metadata.append({
                    "patient_id": p_ids[i],
                    "cc_path": cc_paths[i],
                    "mlo_path": mlo_paths[i],
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


def plot_cm(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_roc_fig(y_true, y_prob, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="navy", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main():
    print("=" * 80)
    print("CancerScan AI | High-Performance 512x512 Multi-View DenseNet-121")
    print("=" * 80)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_pairs = prepare_and_build_pairs(TRAIN_CSV, "train")
    val_pairs = prepare_and_build_pairs(VAL_CSV, "validation")
    test_pairs = prepare_and_build_pairs(TEST_CSV, "test")

    print(f"Train Pairs (512x512): {len(train_pairs)} (Benign: {(train_pairs['label']==0).sum()}, Malignant: {(train_pairs['label']==1).sum()})")
    print(f"Val Pairs (512x512):   {len(val_pairs)} (Benign: {(val_pairs['label']==0).sum()}, Malignant: {(val_pairs['label']==1).sum()})")
    print(f"Test Pairs (512x512):  {len(test_pairs)} (Benign: {(test_pairs['label']==0).sum()}, Malignant: {(test_pairs['label']==1).sum()})")

    train_loader = DataLoader(
        PairedMammogramDataset(train_pairs, training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        PairedMammogramDataset(val_pairs, training=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        PairedMammogramDataset(test_pairs, training=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = DualViewDenseNet512().to(DEVICE)

    # Calculate class weights
    total = len(train_pairs)
    w0 = total / (2.0 * max((train_pairs["label"] == 0).sum(), 1))
    w1 = total / (2.0 * max((train_pairs["label"] == 1).sum(), 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # =========================================================================
    # STAGE A: Warmup (Freeze Backbone, Train Heads)
    # =========================================================================
    print("\n--- STAGE A: Warmup (Heads Only) ---")
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = list(model.cc_head.parameters()) + list(model.mlo_head.parameters()) + list(model.fused_classifier.parameters())
    optimizer_warmup = torch.optim.AdamW(head_params, lr=2e-4, weight_decay=1e-4)

    for ep in range(1, WARMUP_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for cc, mlo, labels, _, _, _ in train_loader:
            cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.to(DEVICE)
            optimizer_warmup.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                lf, lcc, lmlo = model(cc, mlo)
                loss = criterion(lf, labels) + 0.35 * criterion(lcc, labels) + 0.35 * criterion(lmlo, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer_warmup)
            scaler.update()
            total_loss += float(loss.detach().item()) * labels.size(0)

        val_m, _, _, _, _ = evaluate_512(model, val_loader, threshold=0.5, use_tta=False)
        print(f"Warmup Ep {ep}/{WARMUP_EPOCHS} | Train Loss: {total_loss/len(train_pairs):.4f} | Val Acc: {val_m['accuracy']*100:.2f}% | Val AUC: {val_m['roc_auc']:.4f}")

    # =========================================================================
    # STAGE B: Full Fine-Tuning (Unfreeze denseblock3 + denseblock4)
    # =========================================================================
    print("\n--- STAGE B: Multi-View Fine-Tuning (512x512) ---")
    # Unfreeze denseblock4 and denseblock3
    for name, param in model.backbone.named_parameters():
        if "denseblock4" in name or "denseblock3" in name or "norm5" in name:
            param.requires_grad = True

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-5},
        {"params": head_params, "lr": 6e-5},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-7)

    best_score = 0.0
    best_state = None
    best_val_metrics = None
    history = []

    for ep in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        # Keep BatchNorm statistics frozen for stability
        for module in model.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        total_loss = 0.0
        for cc, mlo, labels, _, _, _ in train_loader:
            cc, mlo, labels = cc.to(DEVICE), mlo.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                lf, lcc, lmlo = model(cc, mlo)
                loss = criterion(lf, labels) + 0.35 * criterion(lcc, labels) + 0.35 * criterion(lmlo, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * labels.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_pairs)

        val_m, _, val_probs, _, _ = evaluate_512(model, val_loader, threshold=0.5, use_tta=False)
        composite_score = 0.5 * val_m["balanced_accuracy"] + 0.5 * val_m["roc_auc"]

        record = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_sensitivity": val_m["recall_sensitivity"],
            "val_specificity": val_m["specificity"],
            "val_auc": val_m["roc_auc"],
            "composite_score": composite_score,
            "lr_backbone": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        print(
            f"Epoch {ep:02d}/{FINETUNE_EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_m['accuracy']*100:.2f}% | "
            f"Bal Acc: {val_m['balanced_accuracy']*100:.2f}% | "
            f"AUC: {val_m['roc_auc']:.4f} | "
            f"Score: {composite_score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

        if composite_score > best_score:
            best_score = composite_score
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_m
            print(f"  --> [NEW BEST] Composite score: {best_score:.4f}! Saving checkpoint.")
            torch.save({
                "model_name": "Dual-View DenseNet121 (512x512)",
                "model_state_dict": best_state,
                "epoch": ep,
                "validation_metrics": best_val_metrics,
                "image_size": IMAGE_SIZE,
            }, CHECKPOINT_PATH)

    # Load best checkpoint
    model.load_state_dict(best_state)

    # Validation Evaluation
    val_m_std, val_y_true, val_y_prob, val_y_pred_std, val_df = evaluate_512(model, val_loader, threshold=0.5, use_tta=True)
    fpr, tpr, thresholds = roc_curve(val_y_true, val_y_prob)
    best_idx = np.argmax(tpr - fpr)
    opt_thresh = float(np.clip(thresholds[best_idx], 0.25, 0.75))

    val_m_opt, _, _, val_y_pred_opt, val_df_opt = evaluate_512(model, val_loader, threshold=opt_thresh, use_tta=True)
    val_df_opt.to_csv(VAL_PRED_PATH, index=False)
    plot_cm(val_y_true, val_y_pred_opt, CM_VAL_PATH, f"Dual-View DenseNet 512x512 Val (T*={opt_thresh:.2f})")
    plot_roc_fig(val_y_true, val_y_prob, ROC_VAL_PATH, "Dual-View DenseNet 512x512 Val ROC")

    VAL_JSON_PATH.write_text(json.dumps({
        "optimal_threshold": opt_thresh,
        "standard_t050": val_m_std,
        "optimal_threshold_metrics": val_m_opt,
    }, indent=2), encoding="utf-8")

    # Held-Out Test Evaluation
    test_m_std, test_y_true, test_y_prob, test_y_pred_std, test_df = evaluate_512(model, test_loader, threshold=0.5, use_tta=True)
    test_m_opt, _, _, test_y_pred_opt, test_df_opt = evaluate_512(model, test_loader, threshold=opt_thresh, use_tta=True)
    test_df_opt.to_csv(TEST_PRED_PATH, index=False)
    plot_cm(test_y_true, test_y_pred_opt, CM_TEST_PATH, f"Dual-View DenseNet 512x512 Test (T*={opt_thresh:.2f})")

    TEST_JSON_PATH.write_text(json.dumps({
        "optimal_threshold_from_val": opt_thresh,
        "standard_t050": test_m_std,
        "optimal_threshold_metrics": test_m_opt,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL 512x512 DUAL-VIEW DENSENET PERFORMANCE")
    print("=" * 80)
    print(f"Optimal Threshold T*: {opt_thresh:.3f}\n")
    print(f"--- VALIDATION (N={len(val_pairs)}) ---")
    print(f"  T=0.50 + TTA:  Acc = {val_m_std['accuracy']*100:.2f}% | Bal Acc = {val_m_std['balanced_accuracy']*100:.2f}% | Sens = {val_m_std['recall_sensitivity']*100:.2f}% | Spec = {val_m_std['specificity']*100:.2f}% | AUC = {val_m_std['roc_auc']:.4f}")
    print(f"  T={opt_thresh:.2f} + TTA:  Acc = {val_m_opt['accuracy']*100:.2f}% | Bal Acc = {val_m_opt['balanced_accuracy']*100:.2f}% | Sens = {val_m_opt['recall_sensitivity']*100:.2f}% | Spec = {val_m_opt['specificity']*100:.2f}% | AUC = {val_m_opt['roc_auc']:.4f}")
    print(f"\n--- HELD-OUT TEST (N={len(test_pairs)}) ---")
    print(f"  T=0.50 + TTA:  Acc = {test_m_std['accuracy']*100:.2f}% | Bal Acc = {test_m_std['balanced_accuracy']*100:.2f}% | Sens = {test_m_std['recall_sensitivity']*100:.2f}% | Spec = {test_m_std['specificity']*100:.2f}% | AUC = {test_m_std['roc_auc']:.4f}")
    print(f"  T={opt_thresh:.2f} + TTA:  Acc = {test_m_opt['accuracy']*100:.2f}% | Bal Acc = {test_m_opt['balanced_accuracy']*100:.2f}% | Sens = {test_m_opt['recall_sensitivity']*100:.2f}% | Spec = {test_m_opt['specificity']*100:.2f}% | AUC = {test_m_opt['roc_auc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
