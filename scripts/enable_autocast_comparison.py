from pathlib import Path

path = Path("compare_breast_checkpoints.py")
source = path.read_text(encoding="utf-8-sig")

old = '''            logits = model(cc, mlo)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
'''

new = '''            with torch.autocast(
                device_type="cuda",
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(cc, mlo)

            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
'''

if old not in source:
    raise SystemExit(
        "Expected inference block not found. File was not modified."
    )

path.write_text(source.replace(old, new, 1), encoding="utf-8")
print("Updated comparison inference to use CUDA autocast.")
print("No checkpoint or dataset files modified.")
