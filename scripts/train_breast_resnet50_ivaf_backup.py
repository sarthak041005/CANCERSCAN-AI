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


CHECKPOINT_PATH = MODEL_DIR / "breast_resnet50_ivaf_imagenet_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_resnet50_ivaf_imagenet_history.csv"
VAL_JSON_PATH = REPORT_DIR / "breast_resnet50_ivaf_imagenet_validation.json"
VAL_PRED_PATH = REPORT_DIR / "breast_resnet50_ivaf_imagenet_predictions.csv"
PAIR_AUDIT_PATH = REPORT_DIR / "breast_resnet50_ivaf_imagenet_pair_audit.json"
CM_PATH = REPORT_DIR / "breast_resnet50_ivaf_imagenet_confusion_matrix.png"
ROC_PATH = REPORT_DIR / "breast_resnet50_ivaf_imagenet_roc_curve.png"


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
