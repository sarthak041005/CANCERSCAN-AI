from pathlib import Path

source = Path("train_breast_resnet50_ivaf_exp1.py")
target = Path("train_breast_resnet50_ivaf_exp1_lr_accuracy.py")

if not source.is_file():
    raise SystemExit(f"Source script not found: {source}")

if target.exists():
    raise SystemExit(
        f"{target} already exists. No changes made; choose a new filename."
    )

text = source.read_text(encoding="utf-8")

old_block = '''    else:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        params = [
            {"params": backbone_params, "lr": 1e-5},
            {"params": model.ivaf.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-5},
        ]
    return torch.optim.Adam(params, betas=(0.9, 0.999), eps=1e-7)
'''

new_block = '''    else:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        params = [
            {"params": backbone_params, "lr": 5e-6},
            {"params": model.ivaf.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 2e-5},
        ]
    return torch.optim.Adam(params, betas=(0.9, 0.999), eps=1e-7)
'''

if text.count(old_block) != 1:
    raise SystemExit(
        "Expected Stage 2 optimizer block was not found exactly once. "
        "No experiment script created."
    )

text = text.replace(old_block, new_block, 1)

old_prefix = "breast_resnet50_ivaf_exp1"
new_prefix = "breast_resnet50_ivaf_exp1_lr_accuracy"

if old_prefix not in text:
    raise SystemExit("Expected output prefix not found. No script created.")

text = text.replace(old_prefix, new_prefix)

target.write_text(text, encoding="utf-8")
print(f"Created: {target}")
print("Stage 1 learning rates and all other training settings remain unchanged.")
print("Stage 2 learning rates: backbone=5e-6, IVAF=1e-5, classifier=2e-5")
