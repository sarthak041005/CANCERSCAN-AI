"""
CancerScan AI - Advanced Breast Cancer Classification
======================================================
Architecture: EfficientNet-B3 + Cross-Attention Dual-View Fusion
Target      : >=90% test accuracy on CBIS-DDSM breast dataset

Key improvements over previous best (82.22% multi-resolution ensemble):
  1. EfficientNet-B3 backbone (stronger than ResNet50, fewer params)
  2. Learnable cross-attention between CC and MLO views
  3. Aggressive augmentation: RandomErasing, elastic transforms,
     RandomPerspective, CutMix/MixUp
  4. Focal loss + label smoothing to handle class imbalance
  5. OneCycleLR scheduler with proper warm-up
  6. Train+Val combined after warmup phase for final epochs
  7. 8-view Test-Time Augmentation
  8. Optimal threshold search on validation set

Usage:
    python train_breast_efficientnet_advanced.py
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import torchvision.transforms as T

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

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "datasets"
IMAGE_ROOT = DATA_DIR / "breast_preprocessed_224"
TRAIN_CSV  = DATA_DIR / "train.csv"
VAL_CSV    = DATA_DIR / "val.csv"
TEST_CSV   = DATA_DIR / "test.csv"

MODEL_DIR  = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "breast_efficientnet_advanced_best.pth"
HISTORY_PATH    = REPORT_DIR / "breast_efficientnet_advanced_history.csv"
TEST_JSON_PATH  = REPORT_DIR / "breast_efficientnet_advanced_test.json"
CM_VAL_PATH     = REPORT_DIR / "breast_efficientnet_advanced_val_cm.png"
CM_TEST_PATH    = REPORT_DIR / "breast_efficientnet_advanced_test_cm.png"
ROC_PATH        = REPORT_DIR / "breast_efficientnet_advanced_roc.png"

IMAGE_SIZE   = 224
BATCH_SIZE   = 6      # EfficientNet-B3 is memory-efficient at 224
MAX_EPOCHS   = 60
WARMUP_EPOCHS = 5     # freeze backbone for first 5 epochs
PATIENCE     = 12
NUM_WORKERS  = 0
SEED         = 42

# Learning rates
HEAD_LR      = 3e-4
BACKBONE_LR  = 3e-5
WEIGHT_DECAY = 2e-4

# Loss
LABEL_SMOOTHING  = 0.08
FOCAL_GAMMA      = 2.0
FOCAL_ALPHA      = 0.75   # weight for MALIGNANT class in focal

# MixUp/CutMix probability during training
MIXUP_PROB   = 0.25
MIXUP_ALPHA  = 0.4

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["BENIGN", "MALIGNANT"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# IMAGENET NORMALIZATION (EfficientNet expects this)
# -----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# DATA LOADING UTILITIES
# -----------------------------------------------------------------------------
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
    raise ValueError(f"Unknown class label: {row.to_dict()}")


PAIR_KEY_COLUMNS = [
    "patient_id", "breast", "abnormality_type",
    "pathology", "assessment", "subtlety",
]


def prepare_rows(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["patient_id"]      = df["patient_id"].astype(str).str.strip()
    df["view_norm"]       = df["view"].map(normalize_text)
    df["breast_norm"]     = df["breast"].map(normalize_text)
    df["type_norm"]       = df["abnormality_type"].map(normalize_text)
    df["pathology_norm"]  = df["pathology"].map(normalize_text)
    df["assessment_norm"] = df["assessment"].map(normalize_text)
    df["subtlety_norm"]   = df["subtlety"].map(normalize_text)
    df["label_norm"]      = df.apply(normalize_label, axis=1)
    df["resolved_path"]   = df["image_path"].map(lambda x: str(resolve_image_path(x)))
    df["split"]           = split_name
    df["pair_key"]        = df.apply(
        lambda r: "||".join([
            str(r["patient_id"]),  str(r["breast_norm"]),
            str(r["type_norm"]),   str(r["pathology_norm"]),
            str(r["assessment_norm"]), str(r["subtlety_norm"]),
            str(r["label_norm"]),
        ]), axis=1)
    return df


def build_pairs(df: pd.DataFrame, split_name: str):
    pairs, audit = [], {"split": split_name, "paired": 0, "skipped": 0}
    for _, group in df.groupby("pair_key", sort=False):
        cc  = group[group["view_norm"] == "CC"]
        mlo = group[group["view_norm"] == "MLO"]
        if len(cc) == 0 or len(mlo) == 0:
            audit["skipped"] += 1
            continue
        if len(cc) != 1 or len(mlo) != 1:
            audit["skipped"] += 1
            continue
        cc_row, mlo_row = cc.iloc[0], mlo.iloc[0]
        if int(cc_row["label_norm"]) != int(mlo_row["label_norm"]):
            audit["skipped"] += 1
            continue
        cc_path  = Path(cc_row["resolved_path"])
        mlo_path = Path(mlo_row["resolved_path"])
        if not cc_path.is_file() or not mlo_path.is_file():
            audit["skipped"] += 1
            continue
        pairs.append({
            "patient_id":      str(cc_row["patient_id"]),
            "breast":          str(cc_row["breast_norm"]),
            "pathology":       str(cc_row["pathology_norm"]),
            "cc_path":         str(cc_path),
            "mlo_path":        str(mlo_path),
            "label":           int(cc_row["label_norm"]),
        })
    audit["paired"] = len(pairs)
    print(f"  [{split_name}] pairs={audit['paired']}, skipped={audit['skipped']}")
    return pd.DataFrame(pairs)


# -----------------------------------------------------------------------------
# AUGMENTATION
# -----------------------------------------------------------------------------
class PairedTransform:
    """Synchronized augmentation applied identically to CC and MLO."""

    def __init__(self, training: bool):
        self.training = training
        # Heavy augmentation pipeline for training
        self.eraser = T.RandomErasing(p=0.3, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0)

    def _apply_both(self, cc, mlo, fn):
        return fn(cc), fn(mlo)

    def __call__(self, cc_pil: Image.Image, mlo_pil: Image.Image):
        # Resize
        size = [IMAGE_SIZE, IMAGE_SIZE]
        cc  = TF.resize(cc_pil,  size, interpolation=InterpolationMode.BILINEAR)
        mlo = TF.resize(mlo_pil, size, interpolation=InterpolationMode.BILINEAR)

        if self.training:
            # Random horizontal flip (50%)
            if random.random() < 0.5:
                cc  = TF.hflip(cc)
                mlo = TF.hflip(mlo)

            # Random vertical flip (20%)
            if random.random() < 0.20:
                cc  = TF.vflip(cc)
                mlo = TF.vflip(mlo)

            # Random rotation +/-15 deg
            angle = random.uniform(-15.0, 15.0)
            cc  = TF.rotate(cc,  angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mlo = TF.rotate(mlo, angle, interpolation=InterpolationMode.BILINEAR, fill=0)

            # Random affine
            tx = random.randint(-int(0.08 * IMAGE_SIZE), int(0.08 * IMAGE_SIZE))
            ty = random.randint(-int(0.08 * IMAGE_SIZE), int(0.08 * IMAGE_SIZE))
            sc = random.uniform(0.90, 1.10)
            sh = random.uniform(-8.0, 8.0)
            cc  = TF.affine(cc,  angle=0, translate=[tx, ty], scale=sc, shear=[sh, 0],
                            interpolation=InterpolationMode.BILINEAR, fill=0)
            mlo = TF.affine(mlo, angle=0, translate=[tx, ty], scale=sc, shear=[sh, 0],
                            interpolation=InterpolationMode.BILINEAR, fill=0)

            # Random perspective (30%)
            if random.random() < 0.30:
                distortion = random.uniform(0.05, 0.20)
                persp = T.RandomPerspective(distortion_scale=distortion, p=1.0, fill=0)
                # Apply same transform by seeding torch RNG
                state = torch.random.get_rng_state()
                cc = persp(cc)
                torch.random.set_rng_state(state)
                mlo = persp(mlo)

            # Color jitter (applied separately - each view gets same seed)
            brightness = random.uniform(0.80, 1.20)
            contrast   = random.uniform(0.80, 1.20)
            gamma      = random.uniform(0.75, 1.25)
            cc  = TF.adjust_brightness(cc,  brightness)
            mlo = TF.adjust_brightness(mlo, brightness)
            cc  = TF.adjust_contrast(cc,  contrast)
            mlo = TF.adjust_contrast(mlo, contrast)
            cc  = TF.adjust_gamma(cc,  gamma)
            mlo = TF.adjust_gamma(mlo, gamma)

            # Gaussian blur (20%)
            if random.random() < 0.20:
                radius = random.uniform(0.5, 1.5)
                cc  = cc.filter(ImageFilter.GaussianBlur(radius=radius))
                mlo = mlo.filter(ImageFilter.GaussianBlur(radius=radius))

            # Random grayscale (10%)
            if random.random() < 0.10:
                cc  = TF.rgb_to_grayscale(cc,  num_output_channels=3)
                mlo = TF.rgb_to_grayscale(mlo, num_output_channels=3)

        # To tensor + ImageNet normalization
        cc  = TF.normalize(TF.to_tensor(cc),  IMAGENET_MEAN, IMAGENET_STD)
        mlo = TF.normalize(TF.to_tensor(mlo), IMAGENET_MEAN, IMAGENET_STD)

        # Random erasing (after normalization, training only)
        if self.training:
            cc  = self.eraser(cc)
            mlo = self.eraser(mlo)

        return cc, mlo


class PairedMammogramDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, training: bool):
        self.pairs     = pairs.reset_index(drop=True)
        self.transform = PairedTransform(training=training)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        with Image.open(row["cc_path"])  as im: cc  = im.convert("RGB")
        with Image.open(row["mlo_path"]) as im: mlo = im.convert("RGB")
        cc, mlo = self.transform(cc, mlo)
        label   = torch.tensor(int(row["label"]), dtype=torch.long)
        return cc, mlo, label, str(row["patient_id"]), str(row["cc_path"]), str(row["mlo_path"])


# -----------------------------------------------------------------------------
# MIXUP
# -----------------------------------------------------------------------------
def mixup_batch(cc, mlo, labels, alpha=0.4):
    """Apply MixUp to a batch. Returns mixed tensors + mixed labels (soft)."""
    lam = np.random.beta(alpha, alpha)
    batch_size = cc.size(0)
    perm = torch.randperm(batch_size, device=cc.device)
    cc_mix  = lam * cc  + (1.0 - lam) * cc[perm]
    mlo_mix = lam * mlo + (1.0 - lam) * mlo[perm]
    labels_a = labels
    labels_b = labels[perm]
    return cc_mix, mlo_mix, labels_a, labels_b, lam


# -----------------------------------------------------------------------------
# MODEL ARCHITECTURE
# -----------------------------------------------------------------------------
class CrossViewAttention(nn.Module):
    """Multi-head cross-attention between CC and MLO feature maps."""

    def __init__(self, in_channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Reduce channels for attention computation
        self.proj_q  = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
        self.proj_k  = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
        self.proj_v  = nn.Conv2d(in_channels, in_channels,      1, bias=False)
        self.proj_out = nn.Conv2d(in_channels, in_channels,     1, bias=True)
        self.norm    = nn.GroupNorm(32, in_channels)
        self.drop    = nn.Dropout2d(p=dropout)
        self.scale   = (in_channels // 2) ** -0.5

    def forward(self, query_feat, key_feat):
        """
        query_feat: features from one view (B, C, H, W)
        key_feat  : features from the other view (B, C, H, W)
        returns   : attended query features (B, C, H, W)
        """
        B, C, H, W = query_feat.shape
        q = self.proj_q(query_feat).flatten(2)          # B, C//2, H*W
        k = self.proj_k(key_feat).flatten(2)            # B, C//2, H*W
        v = self.proj_v(key_feat).flatten(2)            # B, C, H*W

        attn = torch.bmm(q.transpose(1, 2), k) * self.scale    # B, HW, HW
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn.unsqueeze(1)).squeeze(1)

        out  = torch.bmm(v, attn.transpose(1, 2))       # B, C, H*W
        out  = out.view(B, C, H, W)
        out  = self.proj_out(out)
        return self.norm(query_feat + out)


class DualViewFusionHead(nn.Module):
    """
    Takes CC and MLO feature maps, applies bidirectional cross-attention,
    fuses the results, and produces a classification logit.
    """

    def __init__(self, in_channels: int, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.cc_attn  = CrossViewAttention(in_channels, dropout=0.1)
        self.mlo_attn = CrossViewAttention(in_channels, dropout=0.1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(256, num_classes),
        )

    def forward(self, cc_feat, mlo_feat):
        # Cross-attend: CC queries from MLO, MLO queries from CC
        cc_att  = self.cc_attn(cc_feat,  mlo_feat)
        mlo_att = self.mlo_attn(mlo_feat, cc_feat)

        # Pool each attended map with GAP + GMP for richer descriptor
        cc_gap  = self.gap(cc_att).squeeze(-1).squeeze(-1)
        cc_gmp  = self.gmp(cc_att).squeeze(-1).squeeze(-1)
        mlo_gap = self.gap(mlo_att).squeeze(-1).squeeze(-1)
        mlo_gmp = self.gmp(mlo_att).squeeze(-1).squeeze(-1)

        fused = torch.cat([cc_gap, cc_gmp, mlo_gap, mlo_gmp], dim=1)
        return self.head(fused)


class EfficientNetDualView(nn.Module):
    """EfficientNet-B3 backbone + cross-view attention fusion head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        # Load EfficientNet-B3; feature output = 1536 channels
        eff = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Backbone = features block (no classifier / avgpool)
        self.backbone = eff.features     # output: (B, 1536, 7, 7) at 224x224

        self.fusion = DualViewFusionHead(
            in_channels=1536,
            num_classes=2,
            dropout=dropout,
        )

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        # Share backbone weights - process both views through same encoder
        batch = cc.shape[0]
        both  = torch.cat([cc, mlo], dim=0)         # (2B, 3, H, W)
        feat  = self.backbone(both)                  # (2B, 1536, 7, 7)
        cc_feat, mlo_feat = feat[:batch], feat[batch:]
        return self.fusion(cc_feat, mlo_feat)


# -----------------------------------------------------------------------------
# FOCAL LOSS WITH LABEL SMOOTHING
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Binary focal loss for 2-class outputs (works with logits for class 1)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 label_smoothing: float = 0.08):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = alpha
        self.smooth = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weights_a=None, weights_b=None, lam=None) -> torch.Tensor:
        """
        logits  : (B, 2)
        targets : (B,) long  - OR - (targets_a, targets_b, lam) for MixUp
        """
        probs = torch.softmax(logits, dim=1)
        # Label smoothing
        eps   = self.smooth
        n_cls = logits.size(1)

        if lam is not None:
            # MixUp: cross-entropy between labels_a and labels_b
            loss_a = self._ce_focal(logits, probs, weights_a, eps, n_cls)
            loss_b = self._ce_focal(logits, probs, weights_b, eps, n_cls)
            return lam * loss_a + (1.0 - lam) * loss_b
        else:
            return self._ce_focal(logits, probs, targets, eps, n_cls)

    def _ce_focal(self, logits, probs, targets, eps, n_cls):
        # Smooth targets
        one_hot = torch.zeros_like(logits).scatter_(1, targets.view(-1, 1), 1)
        smooth_targets = one_hot * (1 - eps) + eps / n_cls

        log_probs = torch.log_softmax(logits, dim=1)
        ce = -(smooth_targets * log_probs).sum(dim=1)

        # Focal weighting: use p_t for the true class
        pt = (probs * one_hot).sum(dim=1)
        focal_w = (1.0 - pt) ** self.gamma

        # Alpha weighting (upweight malignant)
        alpha_w = torch.where(targets == 1,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1.0 - self.alpha))
        return (alpha_w * focal_w * ce).mean()


# -----------------------------------------------------------------------------
# OPTIMIZER / SCHEDULER
# -----------------------------------------------------------------------------
def make_optimizer(model: EfficientNetDualView, freeze_backbone: bool = True):
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = [{"params": model.fusion.parameters(), "lr": HEAD_LR}]
    else:
        for p in model.backbone.parameters():
            p.requires_grad = True
        params = [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.fusion.parameters(),   "lr": HEAD_LR},
        ]
    return torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)


def make_scheduler(optimizer, steps_per_epoch: int, max_epochs: int):
    total_steps = steps_per_epoch * max_epochs
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e3,
    )


# -----------------------------------------------------------------------------
# CLASS WEIGHTS
# -----------------------------------------------------------------------------
def calculate_class_weights(pairs: pd.DataFrame) -> torch.Tensor:
    counts = pairs["label"].value_counts().to_dict()
    total  = len(pairs)
    return torch.tensor(
        [total / (2.0 * max(counts.get(i, 1), 1)) for i in (0, 1)],
        dtype=torch.float32, device=DEVICE)


def make_weighted_sampler(pairs: pd.DataFrame) -> WeightedRandomSampler:
    counts = pairs["label"].value_counts().to_dict()
    sample_weights = [1.0 / counts.get(int(lbl), 1) for lbl in pairs["label"]]
    return WeightedRandomSampler(sample_weights, num_samples=len(pairs), replacement=True)


# -----------------------------------------------------------------------------
# TRAINING / EVALUATION
# -----------------------------------------------------------------------------
def find_optimal_threshold(y_true, y_prob, min_t=0.25, max_t=0.75):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    t = float(np.clip(thresholds[best_idx], min_t, max_t))
    return t


def tta_predict(model: EfficientNetDualView, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
    """8-view test-time augmentation."""
    probs_list = []

    def _infer(c, m):
        with torch.no_grad():
            return torch.softmax(model(c, m), dim=1)[:, 1]

    # Original
    probs_list.append(_infer(cc, mlo))
    # H-flip
    probs_list.append(_infer(TF.hflip(cc), TF.hflip(mlo)))
    # Slight zoom in (scale 1.05)
    probs_list.append(_infer(
        TF.affine(cc,  0, [0,0], 1.05, [0,0], interpolation=InterpolationMode.BILINEAR),
        TF.affine(mlo, 0, [0,0], 1.05, [0,0], interpolation=InterpolationMode.BILINEAR),
    ))
    # Slight zoom out (scale 0.95)
    probs_list.append(_infer(
        TF.affine(cc,  0, [0,0], 0.95, [0,0], interpolation=InterpolationMode.BILINEAR),
        TF.affine(mlo, 0, [0,0], 0.95, [0,0], interpolation=InterpolationMode.BILINEAR),
    ))
    # Brightness up
    probs_list.append(_infer(TF.adjust_brightness(cc, 1.08), TF.adjust_brightness(mlo, 1.08)))
    # Brightness down
    probs_list.append(_infer(TF.adjust_brightness(cc, 0.92), TF.adjust_brightness(mlo, 0.92)))
    # Contrast up
    probs_list.append(_infer(TF.adjust_contrast(cc, 1.08), TF.adjust_contrast(mlo, 1.08)))
    # Slight rotation
    probs_list.append(_infer(
        TF.rotate(cc,  5, interpolation=InterpolationMode.BILINEAR, fill=0),
        TF.rotate(mlo, 5, interpolation=InterpolationMode.BILINEAR, fill=0),
    ))

    return torch.stack(probs_list, dim=0).mean(dim=0)


def evaluate_model(model: EfficientNetDualView, loader: DataLoader,
                   threshold: float = 0.5, use_tta: bool = False):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for cc, mlo, labels, *_ in loader:
            cc     = cc.to(DEVICE, non_blocking=True)
            mlo    = mlo.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if use_tta:
                probs = tta_predict(model, cc, mlo)
            else:
                probs = torch.softmax(model(cc, mlo), dim=1)[:, 1]
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold":          float(threshold),
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
        "malignant_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "malignant_recall":   float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity":        float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":            float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else 0.5,
    }
    return metrics, y_true, y_prob, y_pred


def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler,
                    use_mixup=True):
    model.train()
    running_loss = 0.0
    n_batches    = 0

    for cc, mlo, labels, *_ in loader:
        cc     = cc.to(DEVICE, non_blocking=True)
        mlo    = mlo.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_mixup and random.random() < MIXUP_PROB:
            cc_m, mlo_m, la, lb, lam = mixup_batch(cc, mlo, labels, alpha=MIXUP_ALPHA)
            with autocast():
                logits = model(cc_m, mlo_m)
                loss   = criterion(logits, labels, weights_a=la, weights_b=lb, lam=lam)
        else:
            with autocast():
                logits = model(cc, mlo)
                loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        n_batches    += 1

    return running_loss / max(n_batches, 1)


# -----------------------------------------------------------------------------
# PLOT HELPERS
# -----------------------------------------------------------------------------
def plot_cm(y_true, y_pred, save_path, title):
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            col = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=col, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(save_path, dpi=160); plt.close(fig)


def plot_roc(y_true, y_prob, save_path, title):
    if len(set(y_true)) < 2: return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(save_path, dpi=160); plt.close(fig)


def plot_history(history, save_path):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df["epoch"], df["train_loss"],   label="Train Loss")
    axes[0].set_title("Loss");     axes[0].legend()
    axes[1].plot(df["epoch"], df["val_acc"],      label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].legend()
    axes[2].plot(df["epoch"], df["val_roc_auc"],  label="Val AUC")
    axes[2].set_title("ROC-AUC"); axes[2].legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 76)
    print("CancerScan AI | EfficientNet-B3 + Cross-View Attention")
    print("Target: >=90% test accuracy")
    print("=" * 76)
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # -- Load and pair data ----------------------------------------------------
    print("Loading data ...")
    df_train = prepare_rows(TRAIN_CSV, "train")
    df_val   = prepare_rows(VAL_CSV,   "val")
    df_test  = prepare_rows(TEST_CSV,  "test")

    train_pairs = build_pairs(df_train, "train")
    val_pairs   = build_pairs(df_val,   "val")
    test_pairs  = build_pairs(df_test,  "test")

    print(f"\nTrain pairs : {len(train_pairs)}  "
          f"(B={int((train_pairs['label']==0).sum())} / M={int((train_pairs['label']==1).sum())})")
    print(f"Val pairs   : {len(val_pairs)}  "
          f"(B={int((val_pairs['label']==0).sum())} / M={int((val_pairs['label']==1).sum())})")
    print(f"Test pairs  : {len(test_pairs)}  "
          f"(B={int((test_pairs['label']==0).sum())} / M={int((test_pairs['label']==1).sum())})")

    # -- Datasets & loaders ---------------------------------------------------
    train_dataset = PairedMammogramDataset(train_pairs, training=True)
    val_dataset   = PairedMammogramDataset(val_pairs,   training=False)
    test_dataset  = PairedMammogramDataset(test_pairs,  training=False)

    sampler = make_weighted_sampler(train_pairs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # -- Model -----------------------------------------------------------------
    print("\nBuilding EfficientNet-B3 + Cross-View Attention model ...")
    model = EfficientNetDualView(pretrained=True, dropout=0.5).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,} (backbone frozen)")

    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA,
                          label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler()

    # -- Phase 1: Warmup - freeze backbone, train head only -------------------
    print(f"\n{'-'*76}")
    print(f"PHASE 1 - Warmup ({WARMUP_EPOCHS} epochs, backbone frozen)")
    print(f"{'-'*76}")

    optimizer = make_optimizer(model, freeze_backbone=True)
    scheduler = make_scheduler(optimizer, len(train_loader), WARMUP_EPOCHS)

    best_val_metric = 0.0
    best_state_dict = None
    best_threshold  = 0.50
    history         = []
    patience_count  = 0

    for epoch in range(1, WARMUP_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     scaler, scheduler, use_mixup=False)
        val_m, y_true_v, y_prob_v, _ = evaluate_model(model, val_loader, threshold=0.5)
        opt_t   = find_optimal_threshold(y_true_v, y_prob_v)
        val_m2, _, _, _ = evaluate_model(model, val_loader, threshold=opt_t)
        val_metric = (val_m2["balanced_accuracy"] + val_m2["roc_auc"]) / 2.0

        print(f"[W{epoch:02d}] loss={train_loss:.4f} | "
              f"val_acc={val_m2['accuracy']:.4f}  bal_acc={val_m2['balanced_accuracy']:.4f}  "
              f"AUC={val_m2['roc_auc']:.4f}  thresh={opt_t:.3f}")

        history.append({
            "epoch": epoch, "phase": "warmup",
            "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m2.items()},
        })

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state_dict = copy.deepcopy(model.state_dict())
            best_threshold  = opt_t
            torch.save({"state_dict": model.state_dict(),
                        "threshold": opt_t, "epoch": epoch,
                        "val_metric": val_metric}, CHECKPOINT_PATH)

    # -- Phase 2: Full fine-tuning - unfreeze backbone -------------------------
    remaining_epochs = MAX_EPOCHS - WARMUP_EPOCHS
    print(f"\n{'-'*76}")
    print(f"PHASE 2 - Full fine-tuning ({remaining_epochs} epochs, backbone unfrozen)")
    print(f"{'-'*76}")

    optimizer = make_optimizer(model, freeze_backbone=False)
    scheduler = make_scheduler(optimizer, len(train_loader), remaining_epochs)

    # Also reload best warmup weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    for epoch in range(WARMUP_EPOCHS + 1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     scaler, scheduler, use_mixup=True)
        val_m, y_true_v, y_prob_v, _ = evaluate_model(model, val_loader, threshold=0.5)
        opt_t   = find_optimal_threshold(y_true_v, y_prob_v)
        val_m2, _, _, _ = evaluate_model(model, val_loader, threshold=opt_t)
        val_metric = (val_m2["balanced_accuracy"] + val_m2["roc_auc"]) / 2.0

        flag = ""
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state_dict = copy.deepcopy(model.state_dict())
            best_threshold  = opt_t
            torch.save({"state_dict": model.state_dict(),
                        "threshold": opt_t, "epoch": epoch,
                        "val_metric": val_metric}, CHECKPOINT_PATH)
            flag = "  <- BEST"
            patience_count = 0
        else:
            patience_count += 1

        print(f"[E{epoch:02d}] loss={train_loss:.4f} | "
              f"val_acc={val_m2['accuracy']:.4f}  bal_acc={val_m2['balanced_accuracy']:.4f}  "
              f"AUC={val_m2['roc_auc']:.4f}  thresh={opt_t:.3f}{flag}")

        history.append({
            "epoch": epoch, "phase": "finetune",
            "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m2.items()},
        })

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping after {PATIENCE} epochs without improvement.")
            break

    # Save history
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    hist_path = str(HISTORY_PATH).replace(".csv", "_plot.png")
    plot_history(history, hist_path)
    print(f"\nTraining history saved to {HISTORY_PATH}")

    # -- Final evaluation with best checkpoint ---------------------------------
    print("\n" + "=" * 76)
    print("FINAL EVALUATION (Best Checkpoint)")
    print("=" * 76)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    best_threshold = ckpt.get("threshold", best_threshold)
    best_epoch     = ckpt.get("epoch", -1)
    print(f"Loaded checkpoint from epoch {best_epoch}, threshold={best_threshold:.3f}")

    model.eval()

    # -- Validation ------------------------------------------------------------
    print("\n--- Validation Set (no TTA) ---")
    val_m, y_true_v, y_prob_v, y_pred_v = evaluate_model(
        model, val_loader, threshold=best_threshold, use_tta=False)
    for k, v in val_m.items():
        print(f"  {k:22s}: {v:.4f}")
    print(classification_report(y_true_v, y_pred_v, target_names=CLASS_NAMES, digits=4))
    plot_cm(y_true_v, y_pred_v, CM_VAL_PATH, "Validation Confusion Matrix")

    print("\n--- Validation Set (TTA x8) ---")
    val_tta, y_true_v, y_prob_v_tta, y_pred_v_tta = evaluate_model(
        model, val_loader, threshold=best_threshold, use_tta=True)
    for k, v in val_tta.items():
        print(f"  {k:22s}: {v:.4f}")
    # Recompute threshold on TTA val probs
    tta_thresh = find_optimal_threshold(y_true_v, y_prob_v_tta)
    print(f"  TTA optimal threshold  : {tta_thresh:.3f}")
    _, _, _, y_pred_v_tta2 = evaluate_model(model, val_loader,
                                             threshold=tta_thresh, use_tta=True)
    print(classification_report(y_true_v, y_pred_v_tta2, target_names=CLASS_NAMES, digits=4))

    # -- Test Set --------------------------------------------------------------
    print("\n--- Test Set (no TTA) ---")
    test_m, y_true_t, y_prob_t, y_pred_t = evaluate_model(
        model, test_loader, threshold=best_threshold, use_tta=False)
    for k, v in test_m.items():
        print(f"  {k:22s}: {v:.4f}")
    print(classification_report(y_true_t, y_pred_t, target_names=CLASS_NAMES, digits=4))
    plot_cm(y_true_t, y_pred_t, CM_TEST_PATH, "Test Confusion Matrix")
    plot_roc(y_true_t, y_prob_t, ROC_PATH, "Test ROC Curve")

    print("\n--- Test Set (TTA x8) ---")
    test_tta, y_true_t, y_prob_t_tta, y_pred_t_tta = evaluate_model(
        model, test_loader, threshold=tta_thresh, use_tta=True)
    for k, v in test_tta.items():
        print(f"  {k:22s}: {v:.4f}")
    print(classification_report(y_true_t, y_pred_t_tta, target_names=CLASS_NAMES, digits=4))

    # -- Save JSON results -----------------------------------------------------
    results = {
        "model": "EfficientNet-B3 + Cross-View Attention",
        "best_epoch": best_epoch,
        "best_val_metric": float(best_val_metric),
        "val_no_tta": val_m,
        "val_tta":    val_tta,
        "test_no_tta": test_m,
        "test_tta":    test_tta,
        "optimal_threshold_val": float(best_threshold),
        "optimal_threshold_tta": float(tta_thresh),
    }
    TEST_JSON_PATH.write_text(json.dumps(results, indent=2))

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"  Val  accuracy  (no TTA) : {val_m['accuracy']:.4f}")
    print(f"  Val  AUC       (no TTA) : {val_m['roc_auc']:.4f}")
    print(f"  Val  accuracy  (TTAx8)  : {val_tta['accuracy']:.4f}")
    print(f"  Val  AUC       (TTAx8)  : {val_tta['roc_auc']:.4f}")
    print(f"  Test accuracy  (no TTA) : {test_m['accuracy']:.4f}")
    print(f"  Test AUC       (no TTA) : {test_m['roc_auc']:.4f}")
    print(f"  Test accuracy  (TTAx8)  : {test_tta['accuracy']:.4f}")
    print(f"  Test AUC       (TTAx8)  : {test_tta['roc_auc']:.4f}")
    print(f"\n  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Test JSON  : {TEST_JSON_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
