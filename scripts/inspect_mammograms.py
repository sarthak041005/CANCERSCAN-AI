from PIL import Image
from pathlib import Path
from collections import Counter

ROOT = Path(r"datasets\jpeg")

files = list(ROOT.rglob("*.jpg"))

print("=" * 60)
print("CBIS-DDSM MAMMOGRAPHY DATASET INSPECTION")
print("=" * 60)

print(f"\nTotal JPG images: {len(files)}")

dimensions = Counter()
modes = Counter()
bad_images = []

for i, file in enumerate(files, 1):
    try:
        with Image.open(file) as img:
            dimensions[img.size] += 1
            modes[img.mode] += 1

    except Exception as e:
        bad_images.append((file.name, str(e)))

    if i % 1000 == 0:
        print(f"Checked {i}/{len(files)} images...")

print("\n" + "=" * 60)
print("IMAGE DIMENSIONS")
print("=" * 60)

for size, count in dimensions.most_common():
    print(f"{size}: {count} images")

print("\n" + "=" * 60)
print("IMAGE MODES")
print("=" * 60)

for mode, count in modes.most_common():
    print(f"{mode}: {count} images")

print("\n" + "=" * 60)
print("CORRUPTED / UNREADABLE IMAGES")
print("=" * 60)

print(f"Bad images: {len(bad_images)}")

for name, error in bad_images[:20]:
    print(name, "->", error)

print("\nInspection complete.")