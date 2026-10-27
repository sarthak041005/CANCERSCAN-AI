from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_lung_models(model_dir: Path, device: torch.device):
    """
    Loads lung cancer detection models (DenseNet-121 @ 512px and ResNet-18 @ 224px).
    """
    lung_dir = model_dir / "lung"

    # 1. Champion DenseNet-121 (512x512, 97.49% Test Accuracy)
    densenet_path = lung_dir / "densenet121_512_best.pth"
    densenet_model = None
    if densenet_path.is_file():
        ckpt = torch.load(densenet_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(1024, 2)
        m.load_state_dict(sd)
        m = m.to(device)
        m.eval()
        densenet_model = m

    # 2. ResNet-18 Clean Baseline (224x224)
    resnet_path = lung_dir / "lung_resnet18_clean_best.pth"
    resnet_model = None
    if resnet_path.is_file():
        ckpt = torch.load(resnet_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(sd)
        m = m.to(device)
        m.eval()
        resnet_model = m

    return {
        "densenet": densenet_model,
        "resnet": resnet_model,
    }


def predict_lung(
    image: Image.Image,
    models_dict: dict,
    device: torch.device,
    model_choice: str = "DenseNet-121 (512px - 97.5% Acc)",
    use_tta: bool = True,
) -> Dict:
    """
    Runs lung cancer inference on Chest X-Ray or CT slice.
    Classes: ['Cancer' (0), 'NORMAL' (1)]
    """
    model = None
    size = 512

    if "DenseNet" in model_choice and models_dict.get("densenet") is not None:
        model = models_dict["densenet"]
        size = 512
    elif models_dict.get("resnet") is not None:
        model = models_dict["resnet"]
        size = 224
    elif models_dict.get("densenet") is not None:
        model = models_dict["densenet"]
        size = 512

    if model is None:
        raise ValueError("No lung cancer detection model loaded.")

    # Image preprocessing
    img_rgb = image.convert("RGB")
    t = TF.to_tensor(TF.resize(img_rgb, [size, size], interpolation=InterpolationMode.BILINEAR))
    t_norm = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)

    with torch.no_grad():
        if not use_tta:
            logits = model(t_norm)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        else:
            p1 = torch.softmax(model(t_norm), dim=1)[0].cpu().numpy()
            t_flip = TF.hflip(t_norm)
            p2 = torch.softmax(model(t_flip), dim=1)[0].cpu().numpy()
            t_b = TF.adjust_brightness(t_norm, 1.04)
            p3 = torch.softmax(model(t_b), dim=1)[0].cpu().numpy()
            probs = (p1 + p2 + p3) / 3.0

    # Index 0 = Cancer, Index 1 = Normal
    cancer_prob = float(probs[0])
    normal_prob = float(probs[1])

    is_cancer = cancer_prob >= 0.50
    prediction = "CANCER DETECTED" if is_cancer else "NORMAL"

    if is_cancer:
        confidence = cancer_prob * 100.0
        risk = "HIGH SUSPICION (Malignant Pulmonary Nodule)"
        recommendation = "Suspicious pulmonary consolidation/mass detected. Urgent contrast-enhanced CT of chest and pulmonology referral advised."
        badge_color = "red"
    else:
        confidence = normal_prob * 100.0
        risk = "NORMAL / LOW RISK"
        recommendation = "Clear lung fields without focal consolidation, suspicious nodules, or pleural effusion. Routine health checkup."
        badge_color = "green"

    return {
        "prediction": prediction,
        "is_cancer": is_cancer,
        "cancer_prob": cancer_prob,
        "normal_prob": normal_prob,
        "confidence": float(confidence),
        "risk_level": risk,
        "badge_color": badge_color,
        "recommendation": recommendation,
        "model_used": model_choice,
        "classes": ["Cancer", "NORMAL"],
        "probs": [cancer_prob, normal_prob],
    }
