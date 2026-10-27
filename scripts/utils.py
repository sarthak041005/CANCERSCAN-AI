import random
import numpy as np
import torch

from torchvision import transforms


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_transforms(size=512):
    return transforms.Compose([
        transforms.Resize(
            (size, size)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            5
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.485, 0.485],
            [0.229, 0.229, 0.229]
        ),
    ])


def eval_transforms(size=512):
    return transforms.Compose([
        transforms.Resize(
            (size, size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.485, 0.485],
            [0.229, 0.229, 0.229]
        )
    ])


def preprocess_pil(image, size=512):
    return eval_transforms(size)(
        image.convert("RGB")
    ).unsqueeze(0)