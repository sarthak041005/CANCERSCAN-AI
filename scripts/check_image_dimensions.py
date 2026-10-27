
from pathlib import Path
import cv2
import csv

# Update these paths to match your project
DATASETS = {
    "Breast": Path(
        r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\datasets\breast_cleaned_v2"
    ),
    "Lung": Path(
        r"C:\Users\sarth\OneDrive\Desktop\Lung cancer dataset\chest_xray_lung\train\Cancer"
    ),
}

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def inspect_dataset(dataset_name, folder):
    if not folder.exists():
        print(f"\n{dataset_name}: Folder not found: {folder}")
        return

    records = []

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            continue

        height, width = image.shape[:2]

        records.append({
            "filename": str(path),
            "width": width,
            "height": height,
            "pixels": width * height,
        })

    if not records:
        print(f"\n{dataset_name}: No readable images found.")
        return

    # Extreme images by total pixel count
    smallest = min(records, key=lambda x: x["pixels"])
    largest = max(records, key=lambda x: x["pixels"])

    # Extreme images by individual dimensions
    narrowest = min(records, key=lambda x: x["width"])
    widest = max(records, key=lambda x: x["width"])
    shortest = min(records, key=lambda x: x["height"])
    tallest = max(records, key=lambda x: x["height"])

    print(f"\n{'=' * 55}")
    print(f"{dataset_name.upper()} DATASET")
    print(f"{'=' * 55}")
    print(f"Readable images: {len(records)}")

    def show(label, item):
        print(f"\n{label}")
        print(f"  File: {item['filename']}")
        print(
            f"  Dimensions: {item['width']} x "
            f"{item['height']} pixels"
        )
        print(f"  Total pixels: {item['pixels']:,}")

    show("SMALLEST BY TOTAL PIXELS", smallest)
    show("LARGEST BY TOTAL PIXELS", largest)
    show("NARROWEST IMAGE", narrowest)
    show("WIDEST IMAGE", widest)
    show("SHORTEST IMAGE", shortest)
    show("TALLEST IMAGE", tallest)

    # Save all image dimensions to CSV  
    output_csv = Path(f"{dataset_name.lower()}_dimensions.csv")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "width", "height", "pixels"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nFull dimensions report saved to: {output_csv.resolve()}")


for name, folder in DATASETS.items():
    inspect_dataset(name, folder)
