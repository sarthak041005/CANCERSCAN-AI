from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Project paths
BASE = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE
    / "mnt"
    / "data"
    / "CancerScan_AI"
    / "datasets"
    / "jpeg"
)

OUTPUT_DIR = (
    BASE
    / "mnt"
    / "data"
    / "CancerScan_AI"
    / "datasets"
    / "breast_preprocessed"
)

# Preprocessing settings
IMAGE_SIZE = 512

VALID_EXTENSIONS = {".jpg", ".jpeg"}

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all JPEG images
images = [
    p
    for p in INPUT_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in VALID_EXTENSIONS
]

print("\n========================================")
print("BREAST IMAGE PREPROCESSING")
print("========================================")
print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Images found: {len(images)}")

if len(images) == 0:
    print("\nERROR: No JPEG images found.")
    raise SystemExit(1)


def preprocess_image(image_path):
    # Read image as grayscale
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    # Remove completely black outer borders
    non_zero = cv2.findNonZero(image)

    if non_zero is not None:
        x, y, w, h = cv2.boundingRect(non_zero)
        image = image[y:y + h, x:x + w]

    # Resize while preserving aspect ratio
    height, width = image.shape

    scale = min(
        IMAGE_SIZE / width,
        IMAGE_SIZE / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Create square canvas
    canvas = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    # Calculate padding
    x_offset = (
        IMAGE_SIZE - new_width
    ) // 2

    y_offset = (
        IMAGE_SIZE - new_height
    ) // 2

    # Place image on canvas
    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = image

    # Normalize pixel intensity
    canvas = cv2.normalize(
        canvas,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return canvas


# Process all images
successful = 0
failed = 0

print("\nStarting preprocessing...\n")

for image_path in tqdm(
    images,
    desc="Processing mammograms"
):
    try:
        processed = preprocess_image(image_path)

        if processed is None:
            failed += 1
            continue

        # Preserve original folder structure
        relative_path = image_path.relative_to(INPUT_DIR)

        output_path = (
            OUTPUT_DIR
            / relative_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save processed image
        success = cv2.imwrite(
            str(output_path),
            processed
        )

        if success:
            successful += 1
        else:
            failed += 1

    except Exception as e:
        failed += 1
        print(f"\nFailed: {image_path}")
        print(f"Reason: {e}")

# Display final results
print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")
print(f"Original images: {len(images)}")
print(f"Successfully processed: {successful}")
print(f"Failed: {failed}")
print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Processed dataset: {OUTPUT_DIR}") 