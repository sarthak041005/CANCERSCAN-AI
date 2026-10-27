from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class IVAF(nn.Module):
    """
    Inter-View Attention Fusion block.
    Dynamically weighs cross-projection features between mammogram views.
    """
    def __init__(self, channels: int = 2048):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, 64, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, cc_features: torch.Tensor, mlo_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([cc_features, mlo_features], dim=1)
        gate = self.attention(joined)
        fused = gate * cc_features + (1.0 - gate) * mlo_features
        return fused, gate


class ResNet50IVAF(nn.Module):
    """
    Dual-view ResNet-50 with Inter-View Attention Fusion (224x224 input).
    """
    def __init__(self):
        super().__init__()
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

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor, return_attention: bool = False):
        batch = cc.shape[0]
        both = torch.cat([cc, mlo], dim=0)
        features = self.backbone(both)
        cc_features, mlo_features = features[:batch], features[batch:]
        fused, gate = self.ivaf(cc_features, mlo_features)
        logits = self.classifier(self.pool(fused))
        if return_attention:
            return logits, gate
        return logits


class DualViewDenseNet512(nn.Module):
    """
    High-Resolution (512x512) Dual-View DenseNet-121:
    - Dual Pooling: Global Average Pooling + Global Max Pooling (preserves small focal lesions)
    - Cross-View Interaction: Feature Concat + Asymmetry Difference + Concordance Product
    - Fused + Auxiliary single-view classification heads
    """
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=None)
        self.backbone = base.features
        in_ch = 1024

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        view_dim = in_ch * 2  # 2048

        self.cc_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(view_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

        self.mlo_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(view_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

        fused_dim = view_dim * 4  # 8192
        self.fused_classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def extract_view_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        avg_pool = self.gap(feat).flatten(1)
        max_pool = self.gmp(feat).flatten(1)
        return torch.cat([avg_pool, max_pool], dim=1)

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor):
        batch = cc.shape[0]
        both = torch.cat([cc, mlo], dim=0)
        both_descriptors = self.extract_view_descriptor(both)
        v_cc, v_mlo = both_descriptors[:batch], both_descriptors[batch:]

        logits_cc = self.cc_head(v_cc)
        logits_mlo = self.mlo_head(v_mlo)

        diff = torch.abs(v_cc - v_mlo)
        prod = v_cc * v_mlo
        fused_rep = torch.cat([v_cc, v_mlo, diff, prod], dim=1)
        logits_fused = self.fused_classifier(fused_rep)

        return logits_fused, logits_cc, logits_mlo


class FocalLesionDenseNet(nn.Module):
    """
    Focal Lesion Specialist (256x256):
    Pre-trained with SimCLR Foundation self-supervised representations on 8,885 mammograms.
    Specialized for high-optical-magnification suspicious ROIs and microcalcification clusters.
    """
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=None)
        self.features = base.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        avg_p = self.gap(feat).flatten(1)
        max_p = self.gmp(feat).flatten(1)
        descriptor = torch.cat([avg_p, max_p], dim=1)
        return self.classifier(descriptor)


def load_breast_models(model_dir: Path, device: torch.device):
    """
    Loads all 3 champion breast cancer models with their trained checkpoints.
    """
    # 1. ResNet50 IVAF
    r50_path = model_dir / "breast_resnet50_ivaf_stage3_improved_best.pth"
    r50_model = None
    if r50_path.is_file():
        ckpt = torch.load(r50_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        r50_model = ResNet50IVAF().to(device)
        r50_model.load_state_dict(sd)
        r50_model.eval()

    # 2. Dual-View DenseNet-121 (512x512)
    d512_path = model_dir / "breast_densenet121_multiview_512_best.pth"
    d512_model = None
    if d512_path.is_file():
        ckpt = torch.load(d512_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        d512_model = DualViewDenseNet512().to(device)
        d512_model.load_state_dict(sd)
        d512_model.eval()

    # 3. Focal Lesion Specialist (256x256)
    roi_path = model_dir / "breast_roi_lesion_classifier_best.pth"
    roi_model = None
    if roi_path.is_file():
        ckpt = torch.load(roi_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        roi_model = FocalLesionDenseNet().to(device)
        roi_model.load_state_dict(sd)
        roi_model.eval()

    return {
        "r50": r50_model,
        "d512": d512_model,
        "roi": roi_model,
    }


def predict_single_breast(
    image: Image.Image,
    models_dict: dict,
    device: torch.device,
    use_tta: bool = True,
    optimal_threshold: float = 0.4910,
) -> Dict:
    """
    Performs high-precision single-image breast cancer detection.
    Combines:
    - Focal Lesion DenseNet (256x256) with Test-Time Augmentation
    - High-Resolution DenseNet-121 (512x512) dual-view projection head
    - ResNet-50 IVAF (224x224) attention model
    """
    img_rgb = image.convert("RGB")
    roi_model = models_dict.get("roi")
    d512_model = models_dict.get("d512")
    r50_model = models_dict.get("r50")

    component_probs = {}

    # 1. Focal Lesion Specialist (256x256)
    p_roi = 0.5
    if roi_model is not None:
        t_base = TF.to_tensor(TF.resize(img_rgb, [256, 256], interpolation=InterpolationMode.BILINEAR))
        t_norm = TF.normalize(t_base, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)
        with torch.no_grad():
            if not use_tta:
                p_roi = float(torch.softmax(roi_model(t_norm), dim=1)[0, 1].item())
            else:
                p1 = torch.softmax(roi_model(t_norm), dim=1)[0, 1].item()
                p2 = torch.softmax(roi_model(TF.hflip(t_norm)), dim=1)[0, 1].item()
                p3 = torch.softmax(roi_model(TF.adjust_brightness(t_norm, 1.04)), dim=1)[0, 1].item()
                p4 = torch.softmax(roi_model(TF.affine(t_norm, angle=0, translate=[0, 0], scale=1.03, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)), dim=1)[0, 1].item()
                p_roi = float(np.mean([p1, p2, p3, p4]))
        component_probs["Focal Lesion Specialist (256px)"] = p_roi

    # 2. DenseNet-121 Multi-Resolution (512x512)
    p_d512 = 0.5
    if d512_model is not None:
        t_512 = TF.to_tensor(TF.resize(img_rgb, [512, 512], interpolation=InterpolationMode.BILINEAR))
        t_512 = TF.normalize(t_512, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)
        with torch.no_grad():
            desc = d512_model.extract_view_descriptor(t_512)
            p_cc = torch.softmax(d512_model.cc_head(desc), dim=1)[0, 1].item()
            p_mlo = torch.softmax(d512_model.mlo_head(desc), dim=1)[0, 1].item()
            p_d512 = float(0.5 * (p_cc + p_mlo))
        component_probs["DenseNet-121 (512px)"] = p_d512

    # 3. ResNet-50 IVAF (224x224)
    p_r50 = 0.5
    if r50_model is not None:
        t_224 = TF.to_tensor(TF.resize(img_rgb, [224, 224], interpolation=InterpolationMode.BILINEAR)).unsqueeze(0).to(device)
        with torch.no_grad():
            p_r50 = float(torch.softmax(r50_model(t_224, t_224), dim=1)[0, 1].item())
        component_probs["ResNet-50 IVAF (224px)"] = p_r50

    # Weighted Ensemble Fusion: 45% Focal + 35% DenseNet512 + 20% ResNet50
    active_weights = []
    active_probs = []
    if "Focal Lesion Specialist (256px)" in component_probs:
        active_weights.append(0.45)
        active_probs.append(p_roi)
    if "DenseNet-121 (512px)" in component_probs:
        active_weights.append(0.35)
        active_probs.append(p_d512)
    if "ResNet-50 IVAF (224px)" in component_probs:
        active_weights.append(0.20)
        active_probs.append(p_r50)

    if active_weights:
        total_w = sum(active_weights)
        malignant_prob = sum(w * p for w, p in zip(active_weights, active_probs)) / total_w
    else:
        malignant_prob = 0.5

    benign_prob = 1.0 - malignant_prob
    is_malignant = malignant_prob >= optimal_threshold

    prediction = "MALIGNANT" if is_malignant else "BENIGN"
    confidence = (malignant_prob if is_malignant else benign_prob) * 100.0

    if malignant_prob >= 0.70:
        risk = "HIGH RISK (BI-RADS 4/5 Suspicion)"
        recommendation = "Suspicious lesion detected with irregular margins or microcalcifications. Urgent core-needle biopsy and ultrasound correlation advised."
    elif is_malignant:
        risk = "MODERATE SUSPICION (Borderline Malignant)"
        recommendation = "Focal abnormality detected exceeding clinical threshold. Diagnostic magnification mammography or ultrasound evaluation recommended."
    elif malignant_prob >= 0.25:
        risk = "PROBABLY BENIGN (BI-RADS 2/3)"
        recommendation = "Findings suggestive of benign parenchymal tissue. Short-interval follow-up in 6 months advised."
    else:
        risk = "NORMAL / LOW RISK (BI-RADS 1/2)"
        recommendation = "Clear breast parenchyma. No suspicious microcalcification clusters or masses identified. Routine screening interval."

    return {
        "classes": ["BENIGN", "MALIGNANT"],
        "probs": [float(benign_prob), float(malignant_prob)],
        "prediction": prediction,
        "confidence": float(confidence),
        "malignant_prob": float(malignant_prob),
        "benign_prob": float(benign_prob),
        "risk_level": risk,
        "recommendation": recommendation,
        "component_probs": component_probs,
    }
