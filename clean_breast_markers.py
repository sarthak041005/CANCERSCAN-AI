import csv
import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "mammo_text_detection" / "text_marker_detection.csv"

SOURCE_ROOT = PROJECT_ROOT / "datasets" / "jpeg"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "breast_cleaned"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_MARKERS = {
    "L", "R", "LCC", "RCC",
    "LMLO", "RMLO", "MLO",
    "CC", "LAT", "AP", "PA"
}


def parse_boxes(box_string):
    boxes = []

    if not box_string:
        return boxes

    for item in box_string.split(" | "):
        try:
            label, coords = item.split("@", 1)
            coords = coords.strip("()")
            x, y, w, h = map(int, coords.split(","))

            if w > 0 and h > 0:
                boxes.append((label.strip(), x, y, w, h))

        except Exception:
            continue

    return boxes


def mask_box(image, x, y, w, h):
    height, width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x1 >= x2 or y1 >= y2:
        return image

    pad = max(10, int(max(w, h) * 0.10))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    region = image[y1:y2, x1:x2]

    if region.size > 0:
        mask = cv2.inRange(region, 0, 255)

        cleaned = cv2.inpaint(
            region,
            mask,
            7,
            cv2.INPAINT_TELEA
        )

        image[y1:y2, x1:x2] = cleaned

    return image


with open(CSV_PATH, "r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


total_images = 0
cleaned_images = 0
copied_images = 0
failed_images = 0


for row in rows:

    csv_path = Path(row["image"])

    # CSV path is datasets/jpeg/...
    relative_to_jpeg = Path(*csv_path.parts[2:])

    source_path = SOURCE_ROOT / relative_to_jpeg
    output_path = OUTPUT_ROOT / relative_to_jpeg

    if not source_path.exists():
        print(f"WARNING: Missing image: {source_path}")
        failed_images += 1
        continue

    output_path.parent.mkdir(parents=True, exist_ok=True)

    marker_text = row.get("target_marker", "").strip()
    boxes = parse_boxes(row.get("boxes", ""))

    confirmed_boxes = [
        (label, x, y, w, h)
        for label, x, y, w, h in boxes
        if label in TARGET_MARKERS
    ]

    if marker_text and confirmed_boxes:

        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"WARNING: Could not read: {source_path}")
            failed_images += 1
            continue

        for label, x, y, w, h in confirmed_boxes:
            image = mask_box(image, x, y, w, h)

        if cv2.imwrite(str(output_path), image):
            cleaned_images += 1
        else:
            failed_images += 1

    else:
        shutil.copy2(source_path, output_path)
        copied_images += 1

    total_images += 1

    if total_images % 500 == 0:
        print(f"Processed {total_images}/{len(rows)}")


print()
print("=" * 60)
print("BREAST MAMMOGRAM CLEANING COMPLETED")
print("=" * 60)
print(f"Total images processed : {total_images}")
print(f"Images with markers    : {cleaned_images}")
print(f"Images copied unchanged: {copied_images}")
print(f"Failed images          : {failed_images}")
print(f"Output folder          : {OUTPUT_ROOT}")
print("=" * 60)
