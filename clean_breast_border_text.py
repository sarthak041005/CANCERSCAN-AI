import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "datasets" / "jpeg"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "breast_cleaned_v2"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def clean_border(image):
    height, width = image.shape[:2]

    border_x = int(width * 0.12)
    border_y = int(height * 0.12)

    mask = np.zeros((height, width), dtype=np.uint8)

    mask[:border_y, :] = 255
    mask[height - border_y:, :] = 255
    mask[:, :border_x] = 255
    mask[:, width - border_x:] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    threshold = cv2.bitwise_and(threshold, mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        threshold,
        connectivity=8
    )

    text_mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, num_labels):

        x, y, w, h, area = stats[i]

        if area < 8:
            continue

        if area > width * height * 0.02:
            continue

        if w <= width * 0.20 and h <= height * 0.12:

            component = (labels == i).astype(np.uint8) * 255
            text_mask = cv2.bitwise_or(text_mask, component)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    text_mask = cv2.dilate(
        text_mask,
        kernel,
        iterations=2
    )

    cleaned = cv2.inpaint(
        image,
        text_mask,
        5,
        cv2.INPAINT_TELEA
    )

    return cleaned


total = 0
processed = 0
failed = 0

image_extensions = {".jpg", ".jpeg", ".png"}

for source_path in SOURCE_ROOT.rglob("*"):

    if source_path.suffix.lower() not in image_extensions:
        continue

    relative_path = source_path.relative_to(SOURCE_ROOT)
    output_path = OUTPUT_ROOT / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image = cv2.imread(str(source_path))

    if image is None:
        failed += 1
        continue

    cleaned = clean_border(image)

    if cv2.imwrite(str(output_path), cleaned):
        processed += 1
    else:
        failed += 1

    total += 1

    if total % 500 == 0:
        print(f"Processed {total} images")


print()
print("=" * 60)
print("BREAST BORDER TEXT CLEANING COMPLETED")
print("=" * 60)
print(f"Total images : {total}")
print(f"Processed    : {processed}")
print(f"Failed       : {failed}")
print(f"Output       : {OUTPUT_ROOT}")
print("=" * 60)