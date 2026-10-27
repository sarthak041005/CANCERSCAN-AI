import torch
from pathlib import Path

checkpoint_path = Path("saved_models/breast_resnet50_ivaf_best.pth")
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("\n--- CHECKPOINT ---")
print("Exists:", checkpoint_path.exists())
print("Type:", type(checkpoint).__name__)

if isinstance(checkpoint, dict):
    print("Keys:", list(checkpoint.keys()))
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"{key}: dictionary with {len(value)} entries")
        elif isinstance(value, (str, int, float, bool, type(None))):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {type(value).__name__}")

print("\n--- MODEL LOADING LOCATIONS ---")
script = Path("train_breast_resnet50_ivaf_exp2.py")
lines = script.read_text(encoding="utf-8-sig").splitlines()

terms = ("load_state_dict", "torch.load", "USE_IMAGENET_WEIGHTS",
         "optimizer", "stage2", "partial_finetune", "CHECKPOINT_PATH")

for i, line in enumerate(lines):
    if any(term.lower() in line.lower() for term in terms):
        start, end = max(0, i - 2), min(len(lines), i + 4)
        print(f"\n--- Lines {start + 1}-{end} ---")
        for j in range(start, end):
            print(f"{j + 1}: {lines[j]}")
