from pathlib import Path
import cv2
import csv

DATASETS = {
    "Breast": Path(
        r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\datasets\breast_cleaned_v2"
    ),
    "Lung": Path(
        r"C:\Users\sarth\OneDrive\Desktop\Lung_Cancer_Detection_AI - Copy\datasets\processed\lung_xray_clean_512\train"
    ),
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def inspect_dataset(dataset_name, folder):
    if not folder.exists():
        print(f"\n{dataset_name}: Folder not found: {folder}")
        return

    records = []

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
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

    print(f"\n{'=' * 55}")
    print(f"{dataset_name.upper()} DATASET")
    print(f"{'=' * 55}")
    print(f"Readable images: {len(records)}")

    extremes = [
        ("SMALLEST BY TOTAL PIXELS", min(records, key=lambda x: x["pixels"])),
        ("LARGEST BY TOTAL PIXELS", max(records, key=lambda x: x["pixels"])),
        ("NARROWEST IMAGE", min(records, key=lambda x: x["width"])),
        ("WIDEST IMAGE", max(records, key=lambda x: x["width"])),
        ("SHORTEST IMAGE", min(records, key=lambda x: x["height"])),
        ("TALLEST IMAGE", max(records, key=lambda x: x["height"])),
    ]

    for label, item in extremes:
        print(f"\n{label}")
        print(f"  File: {item['filename']}")
        print(f"  Dimensions: {item['width']} x {item['height']} pixels")
        print(f"  Total pixels: {item['pixels']:,}")

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{dataset_name.lower()}_dimensions.csv"

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "width", "height", "pixels"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nCSV saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    for name, folder in DATASETS.items():
        inspect_dataset(name, folder)