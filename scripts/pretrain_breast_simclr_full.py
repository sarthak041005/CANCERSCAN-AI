"""
CancerScan AI: Full-Scale SimCLR Self-Supervised Pretraining
=============================================================
Pretrains DenseNet-121 on all 8,885 CBIS-DDSM mammograms in ssl_manifest.csv
(with strict held-out validation and test patient exclusion).

Generates a domain-adapted Mammography Foundation Encoder
prior to supervised downstream fine-tuning.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
MANIFEST_PATH = DATA_DIR / "ssl_manifest.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
OUTPUT_DIR = ROOT / "saved_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "breast_densenet121_simclr_foundation.pth"
HISTORY_PATH = ROOT / "reports" / "breast_simclr_pretrain_history.csv"
ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.15
NUM_WORKERS = 0
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SimCLRAugmentation:
    """Medical-specific SimCLR dual-view transform."""
    def __init__(self, size: int):
        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10, fill=0),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])

    def __call__(self, img_pil: Image.Image):
        # Generate two augmented views of the same mammogram
        x1 = self.transform(img_pil)
        x2 = self.transform(img_pil)
        return x1, x2


class SSLMammogramDataset(Dataset):
    def __init__(self, image_paths: list[str], size: int):
        self.paths = image_paths
        self.augment = SimCLRAugmentation(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
        x1, x2 = self.augment(img)
        return x1, x2


class SimCLRDenseNet121(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        in_features = 1024  # DenseNet-121 feature map channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Non-linear projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        z = self.projector(pooled)
        return F.normalize(z, dim=1)


def nt_xent_loss(z1, z2, temperature: float = 0.15):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) loss."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D

    sim_matrix = torch.matmul(z, z.T) / temperature
    diag = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(diag, float("-inf"))

    targets = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])
    loss = F.cross_entropy(sim_matrix, targets)
    return loss


def load_eligible_paths():
    manifest = pd.read_csv(MANIFEST_PATH)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)

    heldout = set(val["patient_id"].dropna().astype(str)) | set(test["patient_id"].dropna().astype(str))
    manifest["patient_id"] = manifest["patient_id"].astype(str)

    # Filter out held-out patients
    filtered = manifest[~manifest["patient_id"].isin(heldout)].drop_duplicates("preprocessed_path")
    valid_paths = [str(p) for p in filtered["preprocessed_path"].dropna() if Path(str(p)).is_file()]
    random.shuffle(valid_paths)
    return valid_paths


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("CancerScan AI | SimCLR Self-Supervised Foundation Pretraining")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print("=" * 76)

    paths = load_eligible_paths()
    print(f"Eligible mammograms for pretraining : {len(paths)}")
    print(f"Image Resolution                    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size                          : {BATCH_SIZE}")
    print(f"Total Epochs                        : {EPOCHS}")

    dataset = SSLMammogramDataset(paths, size=IMAGE_SIZE)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    model = SimCLRDenseNet121().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(loader), eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    history = []
    print("\n--- Starting SimCLR Contrastive Pretraining ---")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, (x1, x2) in enumerate(loader, start=1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if step % 100 == 0 or step == len(loader):
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch:02d}/{EPOCHS}] Step {step}/{len(loader)} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - start_time
        print(f"==> Epoch {epoch:02d} Complete | Average NT-Xent Loss: {avg_loss:.4f} | Duration: {epoch_time:.1f}s")

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "duration_sec": epoch_time,
        })

        # Save checkpoint each epoch
        torch.save({
            "epoch": epoch,
            "backbone_state_dict": model.backbone.state_dict(),
            "model_state_dict": model.state_dict(),
            "loss": avg_loss,
            "image_size": IMAGE_SIZE,
            "architecture": "densenet121",
        }, CHECKPOINT_PATH)

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    print("\n" + "=" * 76)
    print("SIMCLR PRETRAINING COMPLETE")
    print(f"Foundation weights saved to: {CHECKPOINT_PATH}")
    print(f"Training history saved to  : {HISTORY_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
