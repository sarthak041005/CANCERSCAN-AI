from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "mnt" / "data" / "CancerScan_AI" / "datasets"
MANIFEST = DATASETS / "ssl_manifest.csv"
MODEL_DIR = BASE / "mnt" / "data" / "CancerScan_AI" / "models"
CHECKPOINT = MODEL_DIR / "breast_simsiam.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


class SSLDataset(Dataset):
    def __init__(self, manifest_path):
        self.data = pd.read_csv(manifest_path)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.7, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.20
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index]["preprocessed_path"]

        image = Image.open(image_path).convert("RGB")

        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(
                output_dim,
                affine=False
            )
        )

    def forward(self, x):
        return self.network(x)


class PredictionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class SimSiam(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.encoder = backbone
        self.projector = ProjectionHead(
            input_dim=feature_dim
        )
        self.predictor = PredictionHead()

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()


def negative_cosine_similarity(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)

    return -(p * z).sum(dim=1).mean()


def main():
    print("=" * 60)
    print("BREAST MAMMOGRAPHY SELF-SUPERVISED PRETRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Images: {len(pd.read_csv(MANIFEST))}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )
    else:
        print("GPU not available. Training will use CPU.")

    print("=" * 60)

    dataset = SSLDataset(MANIFEST)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = SimSiam().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for view1, view2 in progress:
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)

            p1, p2, z1, z2 = model(
                view1,
                view2
            )

            loss1 = negative_cosine_similarity(
                p1,
                z2
            )

            loss2 = negative_cosine_similarity(
                p2,
                z1
            )

            loss = 0.5 * (loss1 + loss2)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = running_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- SSL Loss: {epoch_loss:.4f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "image_size": IMAGE_SIZE
                },
                CHECKPOINT
            )

            print(
                f"Saved best SSL model: {CHECKPOINT}"
            )

    print()
    print("=" * 60)
    print("SSL PRETRAINING COMPLETED")
    print("=" * 60)
    print(f"Best SSL loss: {best_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
