from __future__ import annotations

import copy
import json
import random
import warnings
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
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

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
# Adapted for the project's train.csv / val.csv format.
# This is a research prototype, not a diagnostic device.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
IMAGE_ROOT = DATA_DIR / "breast_preprocessed_224"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "breast_resnet50_ivaf_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_resnet50_ivaf_history.csv"
VAL_JSON_PATH = REPORT_DIR / "breast_resnet50_ivaf_validation.json"
VAL_PRED_PATH = REPORT_DIR / "breast_resnet50_ivaf_validation_predictions.csv"
PAIR_AUDIT_PATH = REPORT_DIR / "breast_resnet50_ivaf_pair_audit.json"
CM_PATH = REPORT_DIR / "breast_resnet50_ivaf_confusion_matrix.png"
ROC_PATH = REPORT_DIR / "breast_resnet50_ivaf_roc_curve.png"

IMAGE_SIZE = 224
BATCH_SIZE = 4  # two ResNet50 passes per case; conservative for RTX 3050 6 GB
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 15
PATIENCE = 5
NUM_WORKERS = 0  # Windows-friendly
SEED = 42
USE_IMAGENET_WEIGHTS = True

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
    """Resolve project CSV paths against the preprocessed image folders."""
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

    # Also allow an exact relative path under datasets.
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

    raise FileNotFoundError(
        f"Cannot find image for CSV path: {raw_path}\n"
        f"Checked IMAGE_ROOT: {IMAGE_ROOT}\n"
        "Confirm that breast_preprocessed_224 contains the expected images."
    )


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

    if "label" not in df.columns and "pathology" not in df.columns:
        raise ValueError(f"{csv_path.name} must contain 'label' or 'pathology'.")

    for col in PAIR_KEY_COLUMNS:
        if col not in df.columns:
            # Missing pairing metadata means we cannot safely pair this CSV.
            raise ValueError(
                f"{csv_path.name} is missing '{col}'. "
                "Reliable CC-MLO lesion pairing cannot be guaranteed."
            )

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

    # Stable grouping key. Duplicate keys are not arbitrarily paired.
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
        "other_views_skipped": 0,
        "missing_image_files": 0,
        "notes": (
            "Pairing uses patient_id, breast, abnormality_type, pathology, "
            "assessment, subtlety and label. Groups with duplicate CC or MLO "
            "records are skipped rather than arbitrarily matched."
        ),
    }

    for _, group in df.groupby("pair_key", sort=False):
        cc = group[group["view_norm"] == "CC"]
        mlo = group[group["view_norm"] == "MLO"]

        if len(cc) == 0 and len(mlo) == 0:
            audit["other_views_skipped"] += len(group)
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


def get_transforms(training: bool):
    if training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.10, 0.10),
                    scale=(0.90, 1.10),
                )
            ], p=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.ToTensor(),  # scales pixel values to [0, 1], as paper describes
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


class PairedMammogramDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, training: bool):
        self.pairs = pairs.reset_index(drop=True)
        self.transform = get_transforms(training)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        with Image.open(row["cc_path"]) as im:
            cc = im.convert("RGB")
        with Image.open(row["mlo_path"]) as im:
            mlo = im.convert("RGB")

        cc = self.transform(cc)
        mlo = self.transform(mlo)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return cc, mlo, label, str(row["patient_id"]), str(row["cc_path"]), str(row["mlo_path"])


class IVAF(nn.Module):
    """
    Paper-style IVAF:
      cat(CC, MLO) -> 1x1 conv 4096->64 -> ReLU
      -> 1x1 conv 64->2048 -> sigmoid attention
      -> A*CC + (1-A)*MLO
    """
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
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                backbone_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            except Exception as exc:
                warnings.warn(
                    f"Could not load ImageNet weights ({exc}). "
                    "Falling back to randomly initialized ResNet50."
                )
                backbone_model = models.resnet50(weights=None)
        else:
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
        # One concatenated backbone call shares weights and is more efficient.
        batch = cc.shape[0]
        both = torch.cat([cc, mlo], dim=0)
        features = self.backbone(both)
        cc_features, mlo_features = features[:batch], features[batch:]
        fused, gate = self.ivaf(cc_features, mlo_features)
        logits = self.classifier(self.pool(fused))
        if return_attention:
            return logits, gate
        return logits


def set_stage(model: ResNet50IVAF, stage: int):
    # Freeze all backbone layers in Stage I.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    if stage == 2:
        # ResNet50 final two residual stages: layer3 and layer4.
        for name, parameter in model.backbone.named_parameters():
            if name.startswith("7.") or name.startswith("6."):
                parameter.requires_grad = True

    for parameter in model.ivaf.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def make_optimizer(model, stage: int):
    if stage == 1:
        params = [
            {"params": model.ivaf.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ]
    else:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        params = [
            {"params": backbone_params, "lr": 1e-5},
            {"params": model.ivaf.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-5},
        ]
    return torch.optim.Adam(params, betas=(0.9, 0.999), eps=1e-7)


def calculate_class_weights(pairs: pd.DataFrame):
    counts = pairs["label"].value_counts().to_dict()
    total = len(pairs)
    weights = []
    for label in (0, 1):
        count = max(int(counts.get(label, 0)), 1)
        weights.append(total / (2.0 * count))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    y_true, y_prob, y_pred = [], [], []
    rows = []

    for cc, mlo, labels, patient_ids, cc_paths, mlo_paths in loader:
        cc = cc.to(DEVICE, non_blocking=True)
        mlo = mlo.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(cc, mlo)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_n = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_n
        total_items += batch_n
        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        predictions = (probabilities >= 0.5).long()

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(probabilities.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())

        for i in range(batch_n):
            rows.append({
                "patient_id": str(patient_ids[i]),
                "cc_path": str(cc_paths[i]),
                "mlo_path": str(mlo_paths[i]),
                "y_true": int(labels[i].detach().cpu().item()),
                "malignant_probability": float(probabilities[i].cpu().item()),
                "y_pred_threshold_050": int(predictions[i].cpu().item()),
            })

    metrics = {
        "loss": total_loss / max(total_items, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "malignant_precision": precision_score(y_true, y_pred, zero_division=0),
        "malignant_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else None,
    }
    return metrics, np.array(y_true), np.array(y_prob), np.array(y_pred), pd.DataFrame(rows)


def save_checkpoint(model, epoch, stage, val_metrics):
    torch.save({
        "model_name": "ResNet50 + IVAF",
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "validation_metrics": val_metrics,
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "preprocessing": "Resize 224x224; RGB; ToTensor scaling to [0,1]",
        "architecture": "Shared ResNet50 feature extractor + IVAF + GAP + Dropout(0.5) + Dense(1024) + 2-class logits",
    }, CHECKPOINT_PATH)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ResNet50 + IVAF — Validation Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=160)
    plt.close(fig)


def plot_roc(y_true, y_prob):
    if len(set(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ResNet50 + IVAF — Validation ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=160)
    plt.close(fig)


def main():
    print("=" * 72)
    print("CancerScan AI | ResNet50 + IVAF training")
    print("=" * 72)
    print(f"Project root: {ROOT}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image root: {IMAGE_ROOT}")
    print("Note: this script trains on unambiguous CC-MLO pairs only.")

    train_rows = prepare_rows(TRAIN_CSV, "train")
    val_rows = prepare_rows(VAL_CSV, "validation")

    train_patients = set(train_rows["patient_id"])
    val_patients = set(val_rows["patient_id"])
    overlap = sorted(train_patients & val_patients)
    print(f"Patient overlap between CSVs: {len(overlap)}")
    if overlap:
        raise RuntimeError(
            "Patient leakage detected between train.csv and val.csv. "
            f"Example overlapping patient IDs: {overlap[:10]}"
        )

    train_pairs, train_audit = build_pairs(train_rows, "train")
    val_pairs, val_audit = build_pairs(val_rows, "validation")
    audit = {
        "train": train_audit,
        "validation": val_audit,
        "patient_overlap_count": len(overlap),
        "patient_overlap_examples": overlap[:10],
    }
    PAIR_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nPairing audit:")
    print(json.dumps(audit, indent=2))
    if len(train_pairs) < 8 or len(val_pairs) < 4:
        raise RuntimeError(
            "Too few unambiguous CC-MLO pairs were found for a meaningful run. "
            "Read reports/breast_resnet50_ivaf_pair_audit.json. "
            "Use the original CBIS-DDSM metadata including abnormality ID and ROI paths "
            "to create a stricter paired manifest; do not pair images arbitrarily."
        )

    print(f"\nTraining paired cases: {len(train_pairs)}")
    print(f"Validation paired cases: {len(val_pairs)}")
    print("Training class counts:", train_pairs["label"].value_counts().sort_index().to_dict())
    print("Validation class counts:", val_pairs["label"].value_counts().sort_index().to_dict())

    train_dataset = PairedMammogramDataset(train_pairs, training=True)
    val_dataset = PairedMammogramDataset(val_pairs, training=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )

    model = ResNet50IVAF(pretrained=USE_IMAGENET_WEIGHTS).to(DEVICE)
    class_weights = calculate_class_weights(train_pairs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_loss = float("inf")
    best_epoch = -1
    best_stage = None
    best_state = None
    history = []

    stages = [
        (1, STAGE1_EPOCHS, "frozen"),
        (2, STAGE2_EPOCHS, "partial_finetune"),
    ]

    global_epoch = 0
    for stage, max_epochs, stage_name in stages:
        set_stage(model, stage)
        optimizer = make_optimizer(model, stage)
        print(f"\n--- Stage {stage}: {stage_name} | max epochs={max_epochs} ---")
        patience_counter = 0
        stage_best_loss = float("inf")

        for epoch_in_stage in range(1, max_epochs + 1):
            global_epoch += 1
            train_metrics, _, _, _, _ = run_epoch(
                model, train_loader, criterion, optimizer=optimizer, scaler=scaler
            )
            val_metrics, y_true, y_prob, y_pred, val_predictions = run_epoch(
                model, val_loader, criterion, optimizer=None, scaler=None
            )

            record = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": epoch_in_stage,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(record)
            pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

            auc_text = "NA" if val_metrics["roc_auc"] is None else f'{val_metrics["roc_auc"]:.4f}'
            print(
                f"Epoch {global_epoch:02d} [{stage_name}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"malignant_recall={val_metrics['malignant_recall']:.4f} "
                f"specificity={val_metrics['specificity']:.4f} "
                f"AUC={auc_text}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = global_epoch
                best_stage = stage_name
                best_state = copy.deepcopy(model.state_dict())
                save_checkpoint(model, global_epoch, stage_name, val_metrics)
                val_predictions.to_csv(VAL_PRED_PATH, index=False)
                patience_counter = 0
            else:
                patience_counter += 1

            if val_metrics["loss"] < stage_best_loss - 1e-4:
                stage_best_loss = val_metrics["loss"]

            if patience_counter >= PATIENCE:
                print(f"Early stopping stage {stage_name} after {epoch_in_stage} epochs.")
                break

        # Start fine-tuning from the best overall checkpoint, not a worse last epoch.
        if best_state is not None:
            model.load_state_dict(best_state)

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    model.load_state_dict(best_state)
    final_metrics, y_true, y_prob, y_pred, val_predictions = run_epoch(
        model, val_loader, criterion, optimizer=None, scaler=None
    )
    val_predictions.to_csv(VAL_PRED_PATH, index=False)

    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    output = {
        "model": "ResNet50 + IVAF",
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_validation_loss": best_val_loss,
        "validation_metrics_threshold_050": final_metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "paired_train_cases": int(len(train_pairs)),
        "paired_validation_cases": int(len(val_pairs)),
        "patient_overlap": int(len(overlap)),
        "test_set_used": False,
        "clinical_use": "Research prototype only; not for diagnosis.",
        "checkpoint": str(CHECKPOINT_PATH),
        "history_csv": str(HISTORY_PATH),
        "validation_predictions_csv": str(VAL_PRED_PATH),
        "pair_audit_json": str(PAIR_AUDIT_PATH),
    }
    VAL_JSON_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc(y_true, y_prob)

    print("\nTRAINING COMPLETE")
    print(json.dumps(output, indent=2))
    print("\nSaved files:")
    for path in [
        CHECKPOINT_PATH, HISTORY_PATH, VAL_JSON_PATH, VAL_PRED_PATH,
        PAIR_AUDIT_PATH, CM_PATH, ROC_PATH
    ]:
        print(path)


if __name__ == "__main__":
    main()
