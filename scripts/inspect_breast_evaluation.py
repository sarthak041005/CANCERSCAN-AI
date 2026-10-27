from pathlib import Path
import torch

paths = [
    Path("saved_models/breast_resnet50_ivaf_best.pth"),
    Path("saved_models/breast_resnet50_ivaf_imagenet_best.pth"),
]

for path in paths:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    print("\n" + "=" * 65)
    print(path.name)
    for key in ["epoch", "stage", "validation_metrics",
                "class_names", "preprocessing", "architecture"]:
        print(f"\n{key}:")
        print(repr(ckpt.get(key, "<not stored>")))

source_path = Path("train_breast_resnet50_ivaf.py")
lines = source_path.read_text(encoding="utf-8-sig").splitlines()

ranges = [
    (265, 402),  # transforms, dataset and model
    (445, 505),  # evaluation loop
    (565, 752),  # training setup and validation pipeline
]

for start, end in ranges:
    print("\n" + "=" * 65)
    print(f"SOURCE LINES {start}-{min(end, len(lines))}")
    for i in range(start - 1, min(end, len(lines))):
        print(f"{i + 1:04d}: {lines[i]}")
