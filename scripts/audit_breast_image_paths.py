from pathlib import Path
from collections import Counter
import pandas as pd

root = Path(".")
preprocessed = root / "datasets" / "breast_preprocessed_224"
csv_files = ["datasets/train.csv", "datasets/val.csv"]

print("Preprocessed directory exists:", preprocessed.exists())

if not preprocessed.exists():
    raise SystemExit("ERROR: Preprocessed directory not found.")

images = [
    p for p in preprocessed.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

print("Preprocessed image count:", len(images))

# Index files by filename to diagnose matches.
by_name = {}
for p in images:
    by_name.setdefault(p.name.lower(), []).append(p)

print("Unique image filenames:", len(by_name))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    exact = 0
    basename_found = 0
    basename_ambiguous = 0
    missing = []

    for raw in df["image_path"].dropna().astype(str):
        normalized = raw.replace("\\", "/")
        marker = "datasets/jpeg/"

        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidate = preprocessed / Path(relative)
        else:
            candidate = None

        if candidate is not None and candidate.is_file():
            exact += 1
            continue

        matches = by_name.get(Path(normalized).name.lower(), [])

        if len(matches) == 1:
            basename_found += 1
        elif len(matches) > 1:
            basename_ambiguous += 1
        else:
            missing.append(raw)

    print("\n" + "=" * 55)
    print("CSV:", csv_file)
    print("Rows:", len(df))
    print("Exact relative-path matches:", exact)
    print("Unique filename matches:", basename_found)
    print("Ambiguous filename matches:", basename_ambiguous)
    print("No filename match:", len(missing))

    for item in missing[:3]:
        print("Missing example:", item)

print("\nPath audit finished. No files were modified.")
