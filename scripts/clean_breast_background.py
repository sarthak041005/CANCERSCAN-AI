"""
CancerScan AI: conservative mammogram background-text cleaning (224x224)
- Never overwrites source images.
- First run with PREVIEW_ONLY = True to inspect a small sample.
- Removes small bright connected components ONLY outside a dilated breast/tissue mask.
- This is a heuristic; inspect results before processing the full dataset.
Requires: pip install opencv-python pillow numpy
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

# ---- EDIT THESE PATHS IF NEEDED ----
INPUT_DIR = Path(r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\datasets\breast_preprocessed_224")
OUTPUT_DIR = Path(r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\datasets\breast_preprocessed_224_clean")
PREVIEW_DIR = Path(r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI\datasets\breast_cleaning_preview")

# Start conservatively. Preview only a limited sample before full processing.
PREVIEW_ONLY = True
MAX_PREVIEW_IMAGES = 40

# Thresholds for 8-bit grayscale images.
TISSUE_THRESHOLD = 22       # separates tissue from near-black background
BRIGHT_TEXT_THRESHOLD = 95 # candidate bright pixels outside tissue
DILATE_PIXELS = 5           # safety margin around tissue
MAX_COMPONENT_AREA = 220    # ignore large external structures; tune after preview
MIN_COMPONENT_AREA = 2


def clean_image(pil_image):
    """Return (cleaned PIL image, diagnostic PIL image, count_removed_pixels)."""
    original = ImageOps.exif_transpose(pil_image).convert("L")
    gray = np.array(original)
    h, w = gray.shape

    # Initial tissue/foreground mask; keep the largest non-background component.
    foreground = (gray > TISSUE_THRESHOLD).astype(np.uint8)
    # Close tiny gaps, then retain largest connected component as a conservative
    # estimate of the breast/foreground region.
    kernel3 = np.ones((3, 3), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel3, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)

    if n <= 1:
        return original, original.copy(), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    tissue_mask = (labels == largest_label).astype(np.uint8)

    # Dilate the mask so pixels near the breast edge are protected.
    ksize = DILATE_PIXELS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    protected = cv2.dilate(tissue_mask, kernel, iterations=1).astype(bool)

    # Candidate pixels must be bright and safely outside the protected region.
    candidates = ((gray >= BRIGHT_TEXT_THRESHOLD) & (~protected)).astype(np.uint8)
    n2, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)

    remove_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in range(1, n2):
        area = int(comp_stats[idx, cv2.CC_STAT_AREA])
        x = int(comp_stats[idx, cv2.CC_STAT_LEFT])
        y = int(comp_stats[idx, cv2.CC_STAT_TOP])
        cw = int(comp_stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(comp_stats[idx, cv2.CC_STAT_HEIGHT])

        if MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA:
            # Keep only components plausibly text/markers: small enough, and not
            # a large bright image feature. Restrict removal to exterior only.
            remove_mask[comp_labels == idx] = 1

    cleaned = gray.copy()
    cleaned[remove_mask.astype(bool)] = 0

    # Diagnostic image: removed pixels shown in white on black.
    diagnostic = (remove_mask * 255).astype(np.uint8)
    return Image.fromarray(cleaned), Image.fromarray(diagnostic), int(remove_mask.sum())


def make_comparison(before, after, diagnostic, title, dest):
    tile_w, tile_h = 320, 360
    canvas = Image.new("RGB", (tile_w * 3, tile_h + 35), "white")
    draw = ImageDraw.Draw(canvas)
    for x, im, label in [
        (0, before, "BEFORE"),
        (tile_w, after, "AFTER"),
        (tile_w * 2, diagnostic, "REMOVED PIXELS"),
    ]:
        im = im.convert("RGB")
        im.thumbnail((tile_w - 16, tile_h - 35))
        canvas.paste(im, (x + (tile_w - im.width) // 2, 28 + (tile_h - 35 - im.height) // 2))
        draw.text((x + 10, 8), label, fill="black")
    canvas.save(dest)


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input folder not found:\n{INPUT_DIR}\n"
            "Edit INPUT_DIR at the top of this script to match your computer."
        )

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext)
    if not files:
        raise RuntimeError(f"No images found under {INPUT_DIR}")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    selected = files[:MAX_PREVIEW_IMAGES] if PREVIEW_ONLY else files
    processed = 0
    failed = 0
    removed_total = 0

    for i, src in enumerate(selected, start=1):
        try:
            with Image.open(src) as im:
                before = ImageOps.exif_transpose(im).convert("L")
                cleaned, diagnostic, removed = clean_image(before)

            if PREVIEW_ONLY:
                # Keep original relative paths in preview output.
                rel = src.relative_to(INPUT_DIR)
                out = PREVIEW_DIR / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                cleaned.save(out.with_suffix(".png"))
                if i <= 12:
                    comparison = PREVIEW_DIR / f"comparison_{i:02d}_{src.stem}.jpg"
                    make_comparison(before, cleaned, diagnostic, src.name, comparison)
            else:
                rel = src.relative_to(INPUT_DIR)
                out = OUTPUT_DIR / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                # Save cleaned grayscale as PNG to avoid extra JPEG compression.
                cleaned.save(out.with_suffix(".png"))

            processed += 1
            removed_total += removed
            if i % 100 == 0 or i == len(selected):
                print(f"[{i}/{len(selected)}] processed; removed candidate pixels so far: {removed_total}")
        except Exception as exc:
            failed += 1
            print(f"FAILED: {src} :: {exc}")

    print("\nFinished.")
    print(f"Images selected: {len(selected)}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Candidate pixels set to black: {removed_total}")
    if PREVIEW_ONLY:
        print(f"PREVIEW ONLY. Inspect comparisons here:\n{PREVIEW_DIR}")
        print("If breast tissue is altered, do NOT switch to full processing; adjust thresholds first.")
        print("To process all images after validation: set PREVIEW_ONLY = False.")
        print(f"Full cleaned output will go to:\n{OUTPUT_DIR}")
    else:
        print(f"Cleaned images saved to:\n{OUTPUT_DIR}")
        print("Review samples and verify labels/splits before training.")


if __name__ == "__main__":
    main()
