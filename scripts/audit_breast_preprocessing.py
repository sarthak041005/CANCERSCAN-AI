
from pathlib import Path
from collections import Counter
import csv

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "datasets" / "breast_preprocessed_224"
REPORT_DIR = ROOT / "reports" / "breast_preprocessing_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

files = sorted(
    p for p in FOLDER.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)

dimension_counts = Counter()
format_counts = Counter()
errors = []
dark_images = []
extreme_images = []
samples = []

print(f"Folder: {FOLDER}")
print(f"Images found: {len(files)}")

for i, path in enumerate(files, start=1):
    try:
        with Image.open(path) as im:
            im.verify()

        with Image.open(path) as im:
            width, height = im.size
            image_format = im.format or path.suffix.lower()
            grayscale = im.convert("L")
            stats = ImageStat.Stat(grayscale)
            mean_brightness = stats.mean[0]
            std_brightness = stats.stddev[0]

        dimension_counts[(width, height)] += 1
        format_counts[image_format] += 1

        if mean_brightness < 5:
            dark_images.append(str(path.relative_to(ROOT)))

        if std_brightness < 3:
            extreme_images.append(str(path.relative_to(ROOT)))

        if len(samples) < 20:
            samples.append({
                "path": str(path.relative_to(ROOT)),
                "width": width,
                "height": height,
                "format": image_format,
                "mean_brightness": round(mean_brightness, 2),
                "brightness_std": round(std_brightness, 2),
            })

    except Exception as exc:
        errors.append({
            "path": str(path.relative_to(ROOT)),
            "error": str(exc),
        })

    if i % 1000 == 0:
        print(f"Checked {i}/{len(files)}")

with open(REPORT_DIR / "image_audit.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "path", "width", "height", "format",
            "mean_brightness", "brightness_std"
        ],
    )
    writer.writeheader()
    writer.writerows(samples)

with open(REPORT_DIR / "corrupt_images.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "error"])
    writer.writeheader()
    writer.writerows(errors)

with open(REPORT_DIR / "extreme_brightness_candidates.txt", "w", encoding="utf-8") as f:
    f.write("Very dark images (mean brightness < 5):\n")
    f.write("\n".join(dark_images))
    f.write("\n\nVery low contrast images (brightness std < 3):\n")
    f.write("\n".join(extreme_images))

print("\n========== AUDIT RESULTS ==========")
print("Total images:", len(files))
print("Readable images:", len(files) - len(errors))
print("Corrupt/unreadable images:", len(errors))
print("\nImage dimensions:")
for dimensions, count in dimension_counts.most_common():
    print(f"  {dimensions}: {count}")

print("\nImage formats:")
for fmt, count in format_counts.most_common():
    print(f"  {fmt}: {count}")

print("\nVery dark candidates:", len(dark_images))
print("Very low contrast candidates:", len(extreme_images))
print("Report folder:", REPORT_DIR)
print("\nNote: brightness candidates are not automatically defective.")
print("This script does not detect or remove text and does not modify images.")
