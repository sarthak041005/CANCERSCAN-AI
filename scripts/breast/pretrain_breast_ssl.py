
import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class MammogramDataset(Dataset):
    def __init__(self, paths, size=512):
        self.paths = paths
        self.resize = transforms.Resize((size, size), antialias=True)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

    def view(self, image):
        image = self.resize(image)
        if random.random() < 0.8:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.9, 1.1)
            )
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.95, 1.05)
            )
        return self.normalize(self.to_tensor(image))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.view(image), self.view(image)


class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )
        feature_count = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_count, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        return F.normalize(self.projector(self.encoder(x)), dim=1)



def contrastive_loss(z1, z2, temperature=0.2):
    # Compute loss in float32 for numerical stability with AMP.
    z1 = z1.float()
    z2 = z2.float()

    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    logits = (z @ z.T) / temperature

    diagonal = torch.eye(
        2 * batch, device=z.device, dtype=torch.bool
    )
    logits = logits.masked_fill(diagonal, float("-inf"))

    targets = torch.cat([
        torch.arange(batch, 2 * batch, device=z.device),
        torch.arange(0, batch, device=z.device)
    ])
    return F.cross_entropy(logits, targets)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    root = Path.cwd()
    manifest_file = root / "datasets" / "ssl_manifest.csv"
    val_file = root / "datasets" / "val.csv"
    test_file = root / "datasets" / "test.csv"
    output_dir = root / "saved_models" / "breast"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in (manifest_file, val_file, test_file):
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    manifest = pd.read_csv(manifest_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)

    required = {"patient_id", "preprocessed_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(
            "ssl_manifest.csv needs patient_id and preprocessed_path columns"
        )

    heldout = set(val.patient_id.dropna().astype(str))
    heldout |= set(test.patient_id.dropna().astype(str))

    manifest["patient_id"] = manifest["patient_id"].astype(str)
    manifest = manifest[
        ~manifest["patient_id"].isin(heldout)
    ].drop_duplicates("preprocessed_path")

    paths = [
        str(p) for p in manifest["preprocessed_path"].dropna()
        if Path(str(p)).is_file()
    ]
    random.seed(42)
    random.shuffle(paths)

    if args.max_images > 0:
        paths = paths[:args.max_images]
    if len(paths) < 2:
        raise RuntimeError(
            f"Only {len(paths)} eligible images found. Check manifest paths."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("SSL images:", len(paths))
    print("Input size:", args.image_size, "x", args.image_size)
    print("Batch size:", args.batch_size)

    dataset = MammogramDataset(paths, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    model = SSLModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for step, (view1, view2) in enumerate(loader, start=1):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                z1 = model(view1)
                z2 = model(view2)
                loss = contrastive_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

            if step % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs} "
                    f"step {step}/{len(loader)} "
                    f"loss={loss.item():.4f}"
                )

        if steps == 0:
            raise RuntimeError("No batches processed. Check batch size.")

        average_loss = total_loss / steps
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"average_loss={average_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "encoder_state": model.encoder.state_dict(),
            "model_state": model.state_dict(),
            "image_size": args.image_size,
            "architecture": "densenet121",
            "ssl_method": "simclr",
            "loss": average_loss,
        }
        torch.save(
            checkpoint, output_dir / "breast_ssl_latest.pth"
        )

    print("SSL smoke test completed.")
    print("Checkpoint folder:", output_dir)


if __name__ == "__main__":
    main()
