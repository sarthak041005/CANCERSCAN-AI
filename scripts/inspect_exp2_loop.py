from pathlib import Path

path = Path("train_breast_resnet50_ivaf_exp2.py")
lines = path.read_text(encoding="utf-8-sig").splitlines()

ranges = [(410, 445), (610, 710)]

for start, end in ranges:
    print(f"\n{'=' * 20} LINES {start}-{end} {'=' * 20}")
    for i in range(start - 1, min(end, len(lines))):
        print(f"{i + 1}: {lines[i]}")
