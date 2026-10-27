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

# ============================================================
# CancerScan AI: ResNet50 + Inter-View Attention Fusion (IVAF)
# Stage 3 Improved Fine-Tuning with AUC/Balanced-Accuracy Tracking,
# Proper Learning Rates, Label Smoothing, and Test-Time Augmentation
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
IMAGE_ROOT = DATA_DIR / "breast_preprocessed_224"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "breast_resnet50_ivaf_stage3_improved_best.pth"
BASELINE_CHECKPOINT_PATH = MODEL_DIR / "breast_resnet50_ivaf_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_resnet50_ivaf_stage3_improved_history.csv"
VAL_JSON_PATH = REPORT_DIR / "breast_resnet50_ivaf_stage3_improved_validation.json"
TEST_JSON_PATH = REPORT_DIR / "breast_resnet50_ivaf_stage3_improved_test.json"
VAL_PRED_PATH = REPORT_DIR / "breast_resnet50_ivaf_stage3_improved_val_predictions.csv"
TEST_PRED_PATH = REPORT_DIR / "breast_resnet50_ivaf_stage3_improved_test_predictions.csv"
CM_VAL_PATH = REPORT_DIR / "breast_resnet50_ivaf_stage3_improved_val_confusion_matrix.png"
CM_TEST_PATH = REPORT_DIR / "breast_resnet50_ivaf_stage3_improved_test_confusion_matrix.png"
ROC_PATH = REPORT_DIR / "breast_resnet50_ivaf_stage3_improved_roc_curve.png"

IMAGE_SIZE = 224
BATCH_SIZE = 4
MAX_EPOCHS = 15
PATIENCE = 7
NUM_WORKERS = 0
SEED = 42

BACKBONE_LR = 1e-5
IVAF_LR = 3e-5
CLASSIFIER_LR = 5e-5
WEIGHT_DECAY = 1e-4

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
    p = Path(raw)

    candidates = [
        p if p.is_absolute() else ROOT / p,
        IMAGE_ROOT / p,
    ]

    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if len(parts) >= 2:
        candidates.append(IMAGE_ROOT / parts[-2] / parts[-1])
    if parts:
        candidates.append(IMAGE_ROOT / parts[-1])

    candidates.append(DATA_DIR / p)

    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Cannot find image for CSV path: {raw_path}")


def normalize_label(row) -> int:
    value = row.get("label", None)
    if value is not None and not pd.isna(value):
        try:
            n = int(float(value))
            if n in (0, 1):
                return n
        except (TypeError, ValueError):
            pass

    pathology = normalize_text(row.get("pathology", ""))
    if pathology.startswith("BENIGN"):
        return 0
    if pathology.startswith("MALIGNANT"):
        return 1
    raise ValueError(f"Unknown class label/pathology: {row.to_dict()}")


PAIR_KEY_COLUMNS = [
    "patient_id",
    "breast",
    "abnormality_type",
    "pathology",
    "assessment",
    "subtlety",
]


def prepare_rows(csv_path: Path, split_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"patient_id", "image_path", "view"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    for col in PAIR_KEY_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"{csv_path.name} is missing '{col}'.")

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
    return df


def build_pairs(df: pd.DataFrame, split_name: str):
    pairs = []
    audit = {
        "split": split_name,
        "input_image_rows": int(len(df)),
        "paired_cases": 0,
        "cc_without_mlo": 0,
        "mlo_without_cc": 0,
        "ambiguous_groups_skipped": 0,
        "missing_image_files": 0,
    }

    for _, group in df.groupby("pair_key", sort=False):
        cc = group[group["view_norm"] == "CC"]
        mlo = group[group["view_norm"] == "MLO"]

        if len(cc) == 0 and len(mlo) == 0:
            continue
        if len(cc) == 0:
            audit["cc_without_mlo"] += len(mlo)
            continue
        if len(mlo) == 0:
            audit["mlo_without_cc"] += len(cc)
            continue
        if len(cc) != 1 or len(mlo) != 1:
            audit["ambiguous_groups_skipped"] += 1
            continue

        cc_row = cc.iloc[0]
        mlo_row = mlo.iloc[0]
        if int(cc_row["label_norm"]) != int(mlo_row["label_norm"]):
            audit["ambiguous_groups_skipped"] += 1
            continue

        cc_path = Path(cc_row["resolved_path"])
        mlo_path = Path(mlo_row["resolved_path"])
        if not cc_path.is_file() or not mlo_path.is_file():
            audit["missing_image_files"] += 1
            continue

        pairs.append({
            "patient_id": str(cc_row["patient_id"]),
            "breast": str(cc_row["breast_norm"]),
            "abnormality_type": str(cc_row["type_norm"]),
            "pathology": str(cc_row["pathology_norm"]),
            "assessment": str(cc_row["assessment_norm"]),
            "subtlety": str(cc_row["subtlety_norm"]),
            "cc_path": str(cc_path),
            "mlo_path": str(mlo_path),
            "label": int(cc_row["label_norm"]),
        })

    audit["paired_cases"] = len(pairs)
    return pd.DataFrame(pairs), audit


class PairedTransform:
    def __init__(self, training: bool):
        self.training = training

    def __call__(self, cc, mlo):
        cc = TF.resize(cc, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)
        mlo = TF.resize(mlo, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)

        if self.training:
            angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
            translate = [
                int(torch.empty(1).uniform_(-0.05, 0.05).item() * IMAGE_SIZE),
                int(torch.empty(1).uniform_(-0.05, 0.05).item() * IMAGE_SIZE),
            ]
            scale = float(torch.empty(1).uniform_(0.95, 1.05).item())

            cc = TF.affine(
                cc, angle=angle, translate=translate, scale=scale,
                shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR,
            )
            mlo = TF.affine(
                mlo, angle=angle, translate=translate, scale=scale,
                shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR,
            )

            brightness = float(torch.empty(1).uniform_(0.92, 1.08).item())
            contrast = float(torch.empty(1).uniform_(0.92, 1.08).item())
            cc = TF.adjust_brightness(cc, brightness)
            mlo = TF.adjust_brightness(mlo, brightness)
            cc = TF.adjust_contrast(cc, contrast)
            mlo = TF.adjust_contrast(mlo, contrast)

        cc = TF.to_tensor(cc)
        mlo = TF.to_tensor(mlo)
        return cc, mlo


class PairedMammogramDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, training: bool):
        self.pairs = pairs.reset_index(drop=True)
        self.transform = PairedTransform(training=training)

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
        return (
            cc, mlo, label, str(row["patient_id"]),
            str(row["cc_path"]), str(row["mlo_path"]),
        )


class IVAF(nn.Module):
    def __init__(self, channels=2048):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, 64, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, cc_features, mlo_features):
        joined = torch.cat([cc_features, mlo_features], dim=1)
        gate = self.attention(joined)
        fused = gate * cc_features + (1.0 - gate) * mlo_features
        return fused, gate


class ResNet50IVAF(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone_model = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
        self.ivaf = IVAF(channels=2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
        )

    def forward(self, cc, mlo, return_attention=False):
        batch = cc.shape[0]
        both = torch.cat([cc, mlo], dim=0)
        features = self.backbone(both)
        cc_features, mlo_features = features[:batch], features[batch:]
        fused, gate = self.ivaf(cc_features, mlo_features)
        logits = self.classifier(self.pool(fused))
        if return_attention:
            return logits, gate
        return logits


def set_trainable_parameters(model: ResNet50IVAF):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    # Unfreeze layer4 of ResNet50 (index 7 in Sequential)
    for name, parameter in model.backbone.named_parameters():
        if name.startswith("7."):
            parameter.requires_grad = True

    for parameter in model.ivaf.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def make_optimizer(model):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    params = [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": model.ivaf.parameters(), "lr": IVAF_LR},
        {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR},
    ]
    return torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)


def calculate_class_weights(pairs: pd.DataFrame):
    counts = pairs["label"].value_counts().to_dict()
    total = len(pairs)
    weights = []
    for label in (0, 1):
        count = max(int(counts.get(label, 0)), 1)
        weights.append(total / (2.0 * count))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])
    return max(0.20, min(0.80, best_threshold))


def evaluate_model(model, loader, threshold=0.5, use_tta=False):
    model.eval()
    y_true, y_prob, rows = [], [], []

    with torch.no_grad():
        for cc, mlo, labels, patient_ids, cc_paths, mlo_paths in loader:
            cc = cc.to(DEVICE, non_blocking=True)
            mlo = mlo.to(DEVICE, non_blocking=True)

            if use_tta:
                probs_list = []
                p1 = torch.softmax(model(cc, mlo), dim=1)[:, 1]
                probs_list.append(p1)

                cc_z = TF.affine(cc, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                mlo_z = TF.affine(mlo, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                p2 = torch.softmax(model(cc_z, mlo_z), dim=1)[:, 1]
                probs_list.append(p2)

                cc_b = TF.adjust_brightness(cc, 1.04)
                mlo_b = TF.adjust_brightness(mlo, 1.04)
                p3 = torch.softmax(model(cc_b, mlo_b), dim=1)[:, 1]
                probs_list.append(p3)

                probabilities = torch.stack(probs_list, dim=0).mean(dim=0)
            else:
                logits = model(cc, mlo)
                probabilities = torch.softmax(logits, dim=1)[:, 1]

            batch_n = labels.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())

            for i in range(batch_n):
                rows.append({
                    "patient_id": str(patient_ids[i]),
                    "cc_path": str(cc_paths[i]),
                    "mlo_path": str(mlo_paths[i]),
                    "y_true": int(labels[i].item()),
                    "malignant_probability": float(probabilities[i].item()),
                })

    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "use_tta": bool(use_tta),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "malignant_precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "malignant_recall": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "specificity": float(recall_score(y_true_np, y_pred_np, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_np, y_prob_np)) if len(set(y_true_np)) == 2 else 0.5,
    }

    df_rows = pd.DataFrame(rows)
    df_rows[f"y_pred_t{int(threshold*100):02d}"] = y_pred_np
    return metrics, y_true_np, y_prob_np, y_pred_np, df_rows


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


def plot_roc_curve_fig(y_true, y_prob, save_path, title):
    if len(set(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main():
    print("=" * 76)
    print("CancerScan AI | Stage 3 Improved Fine-Tuning: ResNet50 + IVAF")
    print("=" * 76)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_rows = prepare_rows(TRAIN_CSV, "train")
    val_rows = prepare_rows(VAL_CSV, "validation")
    test_rows = prepare_rows(TEST_CSV, "test")

    train_pairs, _ = build_pairs(train_rows, "train")
    val_pairs, _ = build_pairs(val_rows, "validation")
    test_pairs, _ = build_pairs(test_rows, "test")

    print(f"Training paired cases: {len(train_pairs)} (Benign: {(train_pairs['label']==0).sum()}, Malignant: {(train_pairs['label']==1).sum()})")
    print(f"Validation paired cases: {len(val_pairs)} (Benign: {(val_pairs['label']==0).sum()}, Malignant: {(val_pairs['label']==1).sum()})")
    print(f"Test paired cases: {len(test_pairs)} (Benign: {(test_pairs['label']==0).sum()}, Malignant: {(test_pairs['label']==1).sum()})")

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

    if not BASELINE_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CHECKPOINT_PATH}")

    model = ResNet50IVAF(pretrained=False).to(DEVICE)
    baseline_cp = torch.load(BASELINE_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(baseline_cp["model_state_dict"], strict=True)
    print(f"Loaded baseline checkpoint from {BASELINE_CHECKPOINT_PATH}")

    set_trainable_parameters(model)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-7)
    class_weights = calculate_class_weights(train_pairs)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # Initial baseline evaluation
    init_val_metrics, _, init_val_probs, _, _ = evaluate_model(model, val_loader, threshold=0.5, use_tta=False)
    init_score = 0.5 * init_val_metrics["balanced_accuracy"] + 0.5 * init_val_metrics["roc_auc"]
    print(f"\nBaseline Validation Performance:")
    print(f"  Accuracy: {init_val_metrics['accuracy']*100:.2f}% | Bal Acc: {init_val_metrics['balanced_accuracy']*100:.2f}% | AUC: {init_val_metrics['roc_auc']:.4f}")
    print(f"  Composite Score (Bal Acc + AUC)/2: {init_score:.4f}")

    best_score = init_score
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = int(baseline_cp.get("epoch", 28))
    best_val_metrics = init_val_metrics
    patience_counter = 0
    history = []

    print("\n--- Starting Stage 3 Improved Fine-Tuning ---")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        # Keep BatchNorm frozen in backbone for stability
        for module in model.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        train_loss = 0.0
        train_items = 0

        for cc, mlo, labels, _, _, _ in train_loader:
            cc = cc.to(DEVICE, non_blocking=True)
            mlo = mlo.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(cc, mlo)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.detach().item()) * labels.size(0)
            train_items += labels.size(0)

        scheduler.step()
        train_loss_epoch = train_loss / max(train_items, 1)

        val_metrics, _, val_probs, _, _ = evaluate_model(model, val_loader, threshold=0.5, use_tta=False)
        val_score = 0.5 * val_metrics["balanced_accuracy"] + 0.5 * val_metrics["roc_auc"]

        record = {
            "epoch": best_epoch + epoch,
            "train_loss": train_loss_epoch,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_malignant_recall": val_metrics["malignant_recall"],
            "val_specificity": val_metrics["specificity"],
            "val_auc": val_metrics["roc_auc"],
            "val_composite_score": val_score,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_ivaf": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        print(
            f"Epoch {best_epoch + epoch:02d} | "
            f"Train Loss: {train_loss_epoch:.4f} | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Bal Acc: {val_metrics['balanced_accuracy']*100:.2f}% | "
            f"AUC: {val_metrics['roc_auc']:.4f} | "
            f"Score: {val_score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

        if val_score > best_score + 1e-4:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            patience_counter = 0
            print(f"  --> [NEW BEST] Composite score improved to {best_score:.4f}! Saving checkpoint.")
            torch.save({
                "model_name": "ResNet50 + IVAF Stage 3 Improved",
                "model_state_dict": best_state,
                "epoch": best_epoch + epoch,
                "stage": "stage3_layer4_ivaf_finetune",
                "validation_metrics": best_val_metrics,
                "composite_score": best_score,
                "image_size": IMAGE_SIZE,
                "preprocessing": "ToTensor only [0, 1]",
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load best checkpoint for full validation and test evaluation
    model.load_state_dict(best_state)

    # 1. Validation evaluation with optimal threshold search & TTA
    val_m_std, val_y_true, val_y_prob, val_y_pred_std, val_df_std = evaluate_model(model, val_loader, threshold=0.5, use_tta=False)
    opt_threshold = find_optimal_threshold(val_y_true, val_y_prob)
    val_m_opt, _, _, val_y_pred_opt, val_df_opt = evaluate_model(model, val_loader, threshold=opt_threshold, use_tta=False)
    val_m_tta, _, val_y_prob_tta, val_y_pred_tta, val_df_tta = evaluate_model(model, val_loader, threshold=opt_threshold, use_tta=True)

    val_df_tta.to_csv(VAL_PRED_PATH, index=False)
    plot_cm(val_y_true, val_y_pred_tta, CM_VAL_PATH, "Stage 3 Improved ResNet50 + IVAF (Validation)")
    plot_roc_curve_fig(val_y_true, val_y_prob_tta, ROC_PATH, "Stage 3 Improved ResNet50 + IVAF ROC")

    val_summary = {
        "model": "ResNet50 + IVAF Stage 3 Improved",
        "checkpoint": str(CHECKPOINT_PATH),
        "optimal_threshold": opt_threshold,
        "standard_threshold_050": val_m_std,
        "optimal_threshold_metrics": val_m_opt,
        "optimal_threshold_with_tta_metrics": val_m_tta,
    }
    VAL_JSON_PATH.write_text(json.dumps(val_summary, indent=2), encoding="utf-8")

    # 2. Test Set Evaluation
    test_m_std, test_y_true, test_y_prob, test_y_pred_std, test_df_std = evaluate_model(model, test_loader, threshold=0.5, use_tta=False)
    test_m_opt, _, _, test_y_pred_opt, test_df_opt = evaluate_model(model, test_loader, threshold=opt_threshold, use_tta=False)
    test_m_tta, _, test_y_prob_tta, test_y_pred_tta, test_df_tta = evaluate_model(model, test_loader, threshold=opt_threshold, use_tta=True)

    test_df_tta.to_csv(TEST_PRED_PATH, index=False)
    plot_cm(test_y_true, test_y_pred_tta, CM_TEST_PATH, "Stage 3 Improved ResNet50 + IVAF (Test)")

    test_summary = {
        "model": "ResNet50 + IVAF Stage 3 Improved",
        "checkpoint": str(CHECKPOINT_PATH),
        "optimal_threshold_from_val": opt_threshold,
        "standard_threshold_050": test_m_std,
        "optimal_threshold_metrics": test_m_opt,
        "optimal_threshold_with_tta_metrics": test_m_tta,
    }
    TEST_JSON_PATH.write_text(json.dumps(test_summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("FINAL EVALUATION RESULTS SUMMARY")
    print("=" * 76)
    print(f"Optimal Threshold (T* computed on Val ROC): {opt_threshold:.3f}\n")
    print(f"--- VALIDATION (N={len(val_pairs)}) ---")
    print(f"  Threshold 0.50:            Acc = {val_m_std['accuracy']*100:.2f}% | Bal Acc = {val_m_std['balanced_accuracy']*100:.2f}% | AUC = {val_m_std['roc_auc']:.4f}")
    print(f"  Optimal Threshold ({opt_threshold:.2f}):  Acc = {val_m_opt['accuracy']*100:.2f}% | Bal Acc = {val_m_opt['balanced_accuracy']*100:.2f}% | AUC = {val_m_opt['roc_auc']:.4f}")
    print(f"  Optimal Threshold + TTA:   Acc = {val_m_tta['accuracy']*100:.2f}% | Bal Acc = {val_m_tta['balanced_accuracy']*100:.2f}% | AUC = {val_m_tta['roc_auc']:.4f}")
    print(f"\n--- HELD-OUT TEST (N={len(test_pairs)}) ---")
    print(f"  Threshold 0.50:            Acc = {test_m_std['accuracy']*100:.2f}% | Bal Acc = {test_m_std['balanced_accuracy']*100:.2f}% | AUC = {test_m_std['roc_auc']:.4f}")
    print(f"  Optimal Threshold ({opt_threshold:.2f}):  Acc = {test_m_opt['accuracy']*100:.2f}% | Bal Acc = {test_m_opt['balanced_accuracy']*100:.2f}% | AUC = {test_m_opt['roc_auc']:.4f}")
    print(f"  Optimal Threshold + TTA:   Acc = {test_m_tta['accuracy']*100:.2f}% | Bal Acc = {test_m_tta['balanced_accuracy']*100:.2f}% | AUC = {test_m_tta['roc_auc']:.4f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
