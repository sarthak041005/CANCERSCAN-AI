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


CHECKPOINT_PATH = MODEL_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_history.csv"
VAL_JSON_PATH = REPORT_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_validation.json"
VAL_PRED_PATH = REPORT_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_predictions.csv"
PAIR_AUDIT_PATH = REPORT_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_pair_audit.json"
CM_PATH = REPORT_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_confusion_matrix.png"
ROC_PATH = REPORT_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_roc_curve.png"


IMAGE_SIZE = 224
BATCH_SIZE = 4  # two ResNet50 passes per case; conservative for RTX 3050 6 GB
STAGE1_EPOCHS = 0  # Resume the original baseline checkpoint; no frozen warm-up stage
STAGE2_EPOCHS = 8
PATIENCE = 3
NUM_WORKERS = 0  # Windows-friendly
SEED = 42
USE_IMAGENET_WEIGHTS = False  # Load the saved baseline checkpoint below
BASELINE_CHECKPOINT_PATH = MODEL_DIR / "breast_resnet50_ivaf_exp3_layer4_scheduler_best.pth"
BACKBONE_LR_EXP2 = 2e-7
IVAF_LR_EXP2 = 1e-6
CLASSIFIER_LR_EXP2 = 2e-6

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


class PairedTransform:
    """
    Resize paired CC/MLO mammograms and apply the baseline preprocessing.
    Baseline checkpoint inputs are RGB tensors scaled to [0, 1], without ImageNet normalization.
    """

    def __init__(self, training: bool):
        self.training = training

    def __call__(self, cc, mlo):
        cc = TF.resize(
            cc, [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )
        mlo = TF.resize(
            mlo, [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )

        if self.training:
            # Use identical geometric parameters for the paired views.
            angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
            translate = [
                int(torch.empty(1).uniform_(-0.05, 0.05).item() * IMAGE_SIZE),
                int(torch.empty(1).uniform_(-0.05, 0.05).item() * IMAGE_SIZE),
            ]
            scale = float(torch.empty(1).uniform_(0.95, 1.05).item())

            cc = TF.affine(
                cc, angle=angle, translate=translate, scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            mlo = TF.affine(
                mlo, angle=angle, translate=translate, scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )

            # Matching intensity changes for both views.
            brightness = float(torch.empty(1).uniform_(0.9, 1.1).item())
            contrast = float(torch.empty(1).uniform_(0.9, 1.1).item())
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
            # Fail loudly if weights cannot be loaded; never silently use random weights.
            backbone_model = models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2
            )
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
        # ResNet50 Sequential children: index 7 is layer4.
        # Start conservatively on this small paired dataset; test layer3 only
        # as a separate experiment if validation results justify it.
        for name, parameter in model.backbone.named_parameters():
            if name.startswith("7."):
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
            {"params": backbone_params, "lr": BACKBONE_LR_EXP2},
            {"params": model.ivaf.parameters(), "lr": IVAF_LR_EXP2},
            {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR_EXP2},
        ]
    return torch.optim.AdamW(
        params, betas=(0.9, 0.999), eps=1e-7, weight_decay=1e-4
    )


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
    if training:
        # Small batches (4 paired cases) make BatchNorm running-stat updates noisy.
        # Keep pretrained backbone BN statistics fixed while allowing gradients
        # for the explicitly unfrozen layer4 parameters.
        for module in model.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

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
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
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
        "preprocessing": {
            "image_size": IMAGE_SIZE,
            "color_mode": "RGB",
            "normalization": "None; RGB ToTensor scales pixels to [0, 1]",
            "training_augmentation": (
                "Shared affine, brightness, and contrast parameters for paired CC/MLO views; "
                "no horizontal flip"
            ),
        },
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
    print("CancerScan AI | Experiment 4: conservative layer4 fine-tuning + scheduler")
    print("=" * 72)
    print(f"Project root: {ROOT}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image root: {IMAGE_ROOT}")
    print("Note: this script trains on unambiguous CC-MLO pairs only.")
    print("Experiment 4: load ORIGINAL baseline; fine-tune layer4 + IVAF + classifier.")
    print(f"Baseline checkpoint: {BASELINE_CHECKPOINT_PATH}")
    print(f"Stage 1 epochs: {STAGE1_EPOCHS} (must be 0); Stage 2 epochs: {STAGE2_EPOCHS}")

    if STAGE1_EPOCHS != 0:
        raise RuntimeError("Experiment 2 must skip Stage 1; set STAGE1_EPOCHS = 0.")

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

    if not BASELINE_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Original baseline checkpoint not found: {BASELINE_CHECKPOINT_PATH}")

    # Construct the same architecture without downloading/initializing ImageNet weights,
    # then load the original trained baseline weights explicitly.
    model = ResNet50IVAF(pretrained=False).to(DEVICE)
    baseline_checkpoint = torch.load(
        BASELINE_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
    )
    if "model_state_dict" not in baseline_checkpoint:
        raise KeyError("Baseline checkpoint has no 'model_state_dict' entry.")
    model.load_state_dict(baseline_checkpoint["model_state_dict"], strict=True)
    print(f"Loaded baseline weights from: {BASELINE_CHECKPOINT_PATH}")
    print(f"Baseline checkpoint epoch: {baseline_checkpoint.get('epoch', 'unknown')}")
    print("Experiment 4 preprocessing: RGB tensors scaled to [0, 1], no ImageNet normalization.")
    print("Baseline checkpoint is read-only; this experiment writes only exp3-prefixed outputs.")

    class_weights = calculate_class_weights(train_pairs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # Evaluate the baseline checkpoint with the exact preprocessing used for this run.
    # This is the starting point for Experiment 2; never overwrite the baseline file.
    initial_metrics, initial_y_true, initial_y_prob, initial_y_pred, initial_predictions = run_epoch(
        model, val_loader, criterion, optimizer=None, scaler=None
    )
    best_val_loss = float(initial_metrics["loss"])
    best_epoch = int(baseline_checkpoint.get("epoch", 0))
    best_stage = "exp3_initialization"
    best_state = copy.deepcopy(model.state_dict())
    save_checkpoint(model, best_epoch, best_stage, initial_metrics)
    initial_predictions.to_csv(VAL_PRED_PATH, index=False)
    print(f"Baseline checkpoint validation loss under baseline preprocessing: {best_val_loss:.4f}")

    history = []
    stages = [(2, STAGE2_EPOCHS, "exp4_layer4_conservative_scheduler")]
    # Continue epoch numbering from the original baseline checkpoint.
    global_epoch = int(baseline_checkpoint.get("epoch", 0))
    for stage, max_epochs, stage_name in stages:
        set_stage(model, stage)
        optimizer = make_optimizer(model, stage)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, threshold=1e-4,
            min_lr=[1e-8, 1e-8, 1e-8]
        )
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
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_ivaf": optimizer.param_groups[1]["lr"],
                "lr_classifier": optimizer.param_groups[2]["lr"],
            }
            history.append(record)
            pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

            scheduler.step(val_metrics["loss"])
            auc_text = "NA" if val_metrics["roc_auc"] is None else f'{val_metrics["roc_auc"]:.4f}'
            print(
                f"Epoch {global_epoch:02d} [{stage_name}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"malignant_recall={val_metrics['malignant_recall']:.4f} "
                f"specificity={val_metrics['specificity']:.4f} "
                f"AUC={auc_text} "
                f"LRs=({optimizer.param_groups[0]['lr']:.1e},"
                f"{optimizer.param_groups[1]['lr']:.1e},"
                f"{optimizer.param_groups[2]['lr']:.1e})"
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
        "model": "ResNet50 + IVAF — Experiment 4 conservative layer4 fine-tuning with ReduceLROnPlateau",
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
        "checkpoint_selection": "lowest validation loss; baseline checkpoint retained as candidate",
        "scheduler": "ReduceLROnPlateau on validation loss; factor=0.5, patience=1",
        "early_stopping_patience": PATIENCE,
        "starting_checkpoint": str(BASELINE_CHECKPOINT_PATH),
        "experiment_stage": "layer4_only_conservative_finetune_with_scheduler",
        "backbone_layers_unfrozen": ["layer4"],
        "backbone_batchnorm_running_stats_frozen": True,
        "optimizer": "AdamW + ReduceLROnPlateau(mode=min, factor=0.5, patience=1)",
        "weight_decay": 1e-4,
        "learning_rates": {
            "backbone": BACKBONE_LR_EXP2,
            "ivaf": IVAF_LR_EXP2,
            "classifier": CLASSIFIER_LR_EXP2,
        },
        "input_preprocessing": "RGB ToTensor scaling to [0, 1]; no ImageNet normalization",
        "clinical_use": "Research prototype only; not for diagnosis.",
        "checkpoint": str(CHECKPOINT_PATH),
        "history_csv": str(HISTORY_PATH),
        "validation_predictions_csv": str(VAL_PRED_PATH),
        "pair_audit_json": str(PAIR_AUDIT_PATH),
    }
    VAL_JSON_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc(y_true, y_prob)
    if history:
        hist = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(hist["epoch"], hist["train_loss"], marker="o", label="Train loss")
        axes[0].plot(hist["epoch"], hist["val_loss"], marker="o", label="Validation loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Experiment 4 loss curves"); axes[0].legend(); axes[0].grid(True, alpha=0.25)
        axes[1].plot(hist["epoch"], hist["val_accuracy"], marker="o", label="Accuracy")
        axes[1].plot(hist["epoch"], hist["val_balanced_accuracy"], marker="o", label="Balanced accuracy")
        axes[1].plot(hist["epoch"], hist["val_roc_auc"], marker="o", label="ROC-AUC")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_ylim(0, 1)
        axes[1].set_title("Experiment 4 validation metrics"); axes[1].legend(); axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        curve_path = REPORT_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_learning_curves.png"
        fig.savefig(curve_path, dpi=160); plt.close(fig)

    print("\nTRAINING COMPLETE")
    print(json.dumps(output, indent=2))
    print("\nSaved files:")
    for path in [
        CHECKPOINT_PATH, HISTORY_PATH, VAL_JSON_PATH, VAL_PRED_PATH,
        PAIR_AUDIT_PATH, CM_PATH, ROC_PATH,
        REPORT_DIR / "breast_resnet50_ivaf_exp4_layer4_scheduler_learning_curves.png"
    ]:
        print(path)


if __name__ == "__main__":
    main()
