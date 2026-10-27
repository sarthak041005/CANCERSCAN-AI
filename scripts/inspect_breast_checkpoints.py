from pathlib import Path
import ast
import torch

checkpoints = [
    Path("saved_models/breast_resnet50_ivaf_best.pth"),
    Path("saved_models/breast_resnet50_ivaf_imagenet_best.pth"),
]

print("=" * 65)
print("CHECKPOINT INSPECTION")
print("=" * 65)

for path in checkpoints:
    print(f"\nFILE: {path}")

    if not path.exists():
        print("NOT FOUND")
        continue

    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        print("Safe checkpoint loading failed:", repr(exc))
        print("No changes made to the checkpoint.")
        continue

    if isinstance(checkpoint, dict):
        print("Top-level keys:", list(checkpoint.keys())[:40])

        for key, value in checkpoint.items():
            if isinstance(value, dict):
                tensor_items = [
                    (k, tuple(v.shape))
                    for k, v in value.items()
                    if isinstance(v, torch.Tensor)
                ]
                if tensor_items:
                    print(f"\nTensor dictionary: {key}")
                    print("Tensor count:", len(tensor_items))
                    print("First tensor names/shapes:")
                    for item in tensor_items[:8]:
                        print(" ", item)

            elif isinstance(value, (int, float, str, bool)):
                print(f"{key}: {value}")
    else:
        print("Checkpoint type:", type(checkpoint).__name__)

print("\n" + "=" * 65)
print("TRAINING SCRIPT FUNCTIONS AND CLASSES")
print("=" * 65)

script = Path("train_breast_resnet50_ivaf.py")

if script.exists():
    source = script.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"FUNCTION: {node.name} (line {node.lineno})")
        elif isinstance(node, ast.ClassDef):
            print(f"CLASS: {node.name} (line {node.lineno})")
else:
    print("Training script not found.")

print("\nInspection complete. No files modified.")
