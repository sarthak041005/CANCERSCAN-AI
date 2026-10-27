from pathlib import Path

p = Path("train_breast_resnet50_ivaf_exp1.py")
text = p.read_text(encoding="utf-8")

old = "breast_resnet50_ivaf_imagenet"
new = "breast_resnet50_ivaf_exp1"

if old not in text:
    raise SystemExit("Expected output filename prefix not found. No changes made.")

text = text.replace(old, new)
p.write_text(text, encoding="utf-8")

print("Created separate experiment script:", p)
print("Experiment outputs use prefix:", new)
