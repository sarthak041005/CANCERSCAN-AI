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
# CancerScan AI | Multimodal Vision + Clinical Tabular Deep Learning Network
# Fuses dual-view mammogram features with BI-RADS assessment, subtlety, & abnormality
# ==============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
IMAGE_ROOT = DATA_DIR / "breast_preprocessed"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_DIR = ROOT / "saved_models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "breast_multimodal_vision_clinical_best.pth"
HISTORY_PATH = REPORT_DIR / "breast_multimodal_vision_clinical_history.csv"
VAL_JSON_PATH = REPORT_DIR / "breast_multimodal_vision_clinical_validation.json"
TEST_JSON_PATH = REPORT_DIR / "breast_multimodal_vision_clinical_test.json"
ROC_VAL_PATH = REPORT_DIR / "breast_multimodal_vision_clinical_roc_curve.png"
CM_VAL_PATH = REPORT_DIR / "breast_multimodal_vision_clinical_val_cm.png"
CM_TEST_PATH = REPORT_DIR / "breast_multimodal_vision_clinical_test_cm.png"

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 12
NUM_WORKERS = 0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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
        candidate = IMAGE_ROOT / parts[-2] / parts[-1]
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
                # Tabular features
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


class MultimodalDataset(Dataset):
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
            angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
            scale = float(torch.empty(1).uniform_(0.96, 1.04).item())
            cc = TF.affine(cc, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
            mlo = TF.affine(mlo, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)

        cc = TF.to_tensor(cc)
        mlo = TF.to_tensor(mlo)
        cc = TF.normalize(cc, MEAN, STD)
        mlo = TF.normalize(mlo, MEAN, STD)

        clinical = torch.tensor([row["assessment"], row["subtlety"], row["abnormality_type_idx"]], dtype=torch.long)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return cc, mlo, clinical, label, str(row["patient_id"]), str(row["cc_path"]), str(row["mlo_path"])


class VisionTabularBreastNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained DenseNet-121 feature extractor
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.vision_backbone = base.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # Vision view descriptor: 1024 GAP + 1024 GMP = 2048 dims
        # Fused dual-view: [v_cc, v_mlo, |v_cc - v_mlo|] = 2048 * 3 = 6144 dims
        vis_dim = 2048 * 3
        self.vis_proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(vis_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
        )

        # Tabular clinical embeddings
        self.embed_assessment = nn.Embedding(6, 16)  # 0 to 5
        self.embed_subtlety = nn.Embedding(6, 16)    # 1 to 5
        self.embed_type = nn.Embedding(3, 8)         # 0: unknown, 1: calc, 2: mass
        self.tab_proj = nn.Sequential(
            nn.Linear(40, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
        )

        # Joint Multimodal Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, cc, mlo, clinical):
        batch = cc.shape[0]
        both = torch.cat([cc, mlo], dim=0)
        feats = self.vision_backbone(both)
        v_pool = torch.cat([self.gap(feats).flatten(1), self.gmp(feats).flatten(1)], dim=1)
        v_cc, v_mlo = v_pool[:batch], v_pool[batch:]

        v_fused = torch.cat([v_cc, v_mlo, torch.abs(v_cc - v_mlo)], dim=1)
        v_embed = self.vis_proj(v_fused)

        # Tabular branch
        e_ass = self.embed_assessment(clinical[:, 0])
        e_sub = self.embed_subtlety(clinical[:, 1])
        e_typ = self.embed_type(clinical[:, 2])
        tab_raw = torch.cat([e_ass, e_sub, e_typ], dim=1)
        tab_embed = self.tab_proj(tab_raw)

        # Multimodal fusion
        joint = torch.cat([v_embed, tab_embed], dim=1)
        logits = self.classifier(joint)
        return logits


def evaluate(model, loader, threshold=0.5):
    model.eval()
    y_true, y_prob, meta = [], [], []

    with torch.no_grad():
        for cc, mlo, clinical, labels, p_ids, cc_paths, mlo_paths in loader:
            cc, mlo, clinical = cc.to(DEVICE), mlo.to(DEVICE), clinical.to(DEVICE)
            logits = model(cc, mlo, clinical)
            probs = torch.softmax(logits, dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            for i in range(labels.size(0)):
                meta.append({
                    "patient_id": p_ids[i],
                    "cc_path": cc_paths[i],
                    "mlo_path": mlo_paths[i],
                    "y_true": int(labels[i].item()),
                    "probability": float(probs[i].item()),
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
    df_meta[f"pred_t{int(threshold*100):02d}"] = y_pred
    return metrics, y_true, y_prob, y_pred, df_meta


def main():
    print("=" * 80)
    print("CancerScan AI | Multimodal Vision + Clinical Tabular Deep Learning")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    train_pairs = prepare_pairs_with_metadata(TRAIN_CSV)
    val_pairs = prepare_pairs_with_metadata(VAL_CSV)
    test_pairs = prepare_pairs_with_metadata(TEST_CSV)

    print(f"Train Cases: {len(train_pairs)} (Benign: {(train_pairs['label']==0).sum()}, Malignant: {(train_pairs['label']==1).sum()})")
    print(f"Val Cases:   {len(val_pairs)} (Benign: {(val_pairs['label']==0).sum()}, Malignant: {(val_pairs['label']==1).sum()})")
    print(f"Test Cases:  {len(test_pairs)} (Benign: {(test_pairs['label']==0).sum()}, Malignant: {(test_pairs['label']==1).sum()})")

    train_loader = DataLoader(MultimodalDataset(train_pairs, training=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    val_loader = DataLoader(MultimodalDataset(val_pairs, training=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
    test_loader = DataLoader(MultimodalDataset(test_pairs, training=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))

    model = VisionTabularBreastNet().to(DEVICE)

    # Freeze shallow vision layers, fine-tune denseblock3 and denseblock4
    for param in model.vision_backbone.parameters():
        param.requires_grad = False
    for name, param in model.vision_backbone.named_parameters():
        if "denseblock4" in name or "denseblock3" in name or "norm5" in name:
            param.requires_grad = True

    backbone_params = [p for p in model.vision_backbone.parameters() if p.requires_grad]
    head_params = list(model.vis_proj.parameters()) + list(model.embed_assessment.parameters()) + list(model.embed_subtlety.parameters()) + list(model.embed_type.parameters()) + list(model.tab_proj.parameters()) + list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-5},
        {"params": head_params, "lr": 1e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # Balanced class weights
    total = len(train_pairs)
    w0 = total / (2.0 * max((train_pairs["label"] == 0).sum(), 1))
    w1 = total / (2.0 * max((train_pairs["label"] == 1).sum(), 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_score = 0.0
    best_state = None
    best_metrics = None
    history = []

    print("\n--- Training Multimodal Network ---")
    for ep in range(1, EPOCHS + 1):
        model.train()
        for m in model.vision_backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

        total_loss = 0.0
        for cc, mlo, clinical, labels, _, _, _ in train_loader:
            cc, mlo, clinical, labels = cc.to(DEVICE), mlo.to(DEVICE), clinical.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(cc, mlo, clinical)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * labels.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_pairs)

        val_m, val_y, val_p, _, _ = evaluate(model, val_loader, threshold=0.5)
        score = 0.5 * val_m["balanced_accuracy"] + 0.5 * val_m["roc_auc"]

        record = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_sensitivity": val_m["recall_sensitivity"],
            "val_specificity": val_m["specificity"],
            "val_auc": val_m["roc_auc"],
            "score": score,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        print(
            f"Epoch {ep:02d}/{EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_m['accuracy']*100:.2f}% | "
            f"Bal Acc: {val_m['balanced_accuracy']*100:.2f}% | "
            f"AUC: {val_m['roc_auc']:.4f} | "
            f"Score: {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_m
            print(f"  --> [NEW BEST] Composite score: {best_score:.4f}! Saving checkpoint.")
            torch.save({
                "model_name": "VisionTabularBreastNet",
                "model_state_dict": best_state,
                "epoch": ep,
                "validation_metrics": best_metrics,
            }, CHECKPOINT_PATH)

    # Evaluate best checkpoint
    model.load_state_dict(best_state)

    # Validation
    val_m_std, val_y, val_p, _, val_df = evaluate(model, val_loader, threshold=0.5)
    fpr, tpr, thresholds = roc_curve(val_y, val_p)
    opt_t = float(np.clip(thresholds[np.argmax(tpr - fpr)], 0.25, 0.75))
    val_m_opt, _, _, _, val_df_opt = evaluate(model, val_loader, threshold=opt_t)

    VAL_JSON_PATH.write_text(json.dumps({
        "optimal_threshold": opt_t,
        "standard_t050": val_m_std,
        "optimal_t": val_m_opt,
    }, indent=2), encoding="utf-8")

    # Test
    test_m_std, test_y, test_p, _, test_df = evaluate(model, test_loader, threshold=0.5)
    test_m_opt, _, _, _, test_df_opt = evaluate(model, test_loader, threshold=opt_t)

    TEST_JSON_PATH.write_text(json.dumps({
        "optimal_threshold_from_val": opt_t,
        "standard_t050": test_m_std,
        "optimal_t": test_m_opt,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL MULTIMODAL PERFORMANCE")
    print("=" * 80)
    print(f"Optimal Threshold T*: {opt_t:.3f}\n")
    print(f"--- VALIDATION (N={len(val_pairs)}) ---")
    print(f"  T=0.50:  Acc = {val_m_std['accuracy']*100:.2f}% | Bal Acc = {val_m_std['balanced_accuracy']*100:.2f}% | Sens = {val_m_std['recall_sensitivity']*100:.2f}% | Spec = {val_m_std['specificity']*100:.2f}% | AUC = {val_m_std['roc_auc']:.4f}")
    print(f"  T={opt_t:.2f}:  Acc = {val_m_opt['accuracy']*100:.2f}% | Bal Acc = {val_m_opt['balanced_accuracy']*100:.2f}% | Sens = {val_m_opt['recall_sensitivity']*100:.2f}% | Spec = {val_m_opt['specificity']*100:.2f}% | AUC = {val_m_opt['roc_auc']:.4f}")
    print(f"\n--- HELD-OUT TEST (N={len(test_pairs)}) ---")
    print(f"  T=0.50:  Acc = {test_m_std['accuracy']*100:.2f}% | Bal Acc = {test_m_std['balanced_accuracy']*100:.2f}% | Sens = {test_m_std['recall_sensitivity']*100:.2f}% | Spec = {test_m_std['specificity']*100:.2f}% | AUC = {test_m_std['roc_auc']:.4f}")
    print(f"  T={opt_t:.2f}:  Acc = {test_m_opt['accuracy']*100:.2f}% | Bal Acc = {test_m_opt['balanced_accuracy']*100:.2f}% | Sens = {test_m_opt['recall_sensitivity']*100:.2f}% | Spec = {test_m_opt['specificity']*100:.2f}% | AUC = {test_m_opt['roc_auc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
