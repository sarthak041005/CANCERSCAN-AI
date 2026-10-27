import torch
from pathlib import Path

p = Path("saved_models/breast_resnet50_ivaf_best.pth")
ckpt = torch.load(p, map_location="cpu", weights_only=True)

print("Epoch:", ckpt.get("epoch"))
print("Stage:", ckpt.get("stage"))
print("Validation metrics:")
for k, v in ckpt.get("validation_metrics", {}).items():
    print(f"  {k}: {v}")
