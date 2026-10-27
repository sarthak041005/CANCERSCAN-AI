"""
CancerScan AI — segmentation-based breast mask preview.

This script uses a conservative foreground segmentation approach (Otsu thresholding,
morphology, largest connected component) to estimate the breast region. It does NOT
claim to be a validated mammography segmentation model. It only previews proposed
external annotation removal and never modifies source files.

Install:
    python -m pip install opencv-python pillow numpy

Run:
    python .\clean_breast_background_v3.py

Review comparison images carefully before considering any batch cleaning.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

PROJECT = Path(r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI")
INPUT = PROJECT / "datasets" / "breast_preprocessed_224"
PREVIEW = PROJECT / "datasets" / "breast_cleaning_preview_v3"

PREVIEW_ONLY = True
MAX_IMAGES = 40
COMPARISONS = 20

# Mask and annotation parameters. Conservative defaults.
MASK_DILATION = 9          # safety margin around estimated breast/foreground
MIN_TEXT_AREA = 2
MAX_TEXT_AREA = 1000
MIN_TEXT_BRIGHTNESS = 65
MAX_TEXT_WIDTH_FRAC = 0.40
MAX_TEXT_HEIGHT_FRAC = 0.30


def load_gray(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("L")


def largest_component(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros(binary.shape, dtype=np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def estimate_breast_protection(gray):
    """
    Produce a protected mask around foreground. Otsu threshold + morphological
    closing + largest component is a rough heuristic and can fail on unusual views.
    """
    arr = np.asarray(gray)
    h, w = arr.shape

    # Ignore near-black background, then estimate a foreground threshold.
    nonblack = arr[arr > 8]
    if nonblack.size < 30:
        return np.ones((h, w), dtype=bool)

    otsu_input = arr.copy()
    _, otsu = cv2.threshold(otsu_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = ((arr > max(8, int(_ * 0.65))).astype(np.uint8))

    # Connect tissue texture without excessive smoothing.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    component = largest_component(binary)

    # If foreground mask is implausibly tiny/huge, fail safe: protect everything.
    fraction = float(component.mean())
    if fraction < 0.04 or fraction > 0.92:
        return np.ones((h, w), dtype=bool)

    k = MASK_DILATION * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    protected = cv2.dilate(component, kernel, iterations=1).astype(bool)

    # Always protect thin border bands and any nonblack pixel close to mask boundary.
    edge = max(2, int(min(h, w) * 0.01))
    protected[:edge, :] = True
    protected[-edge:, :] = True
    protected[:, :edge] = True
    protected[:, -edge:] = True
    return protected


def remove_external_labels(gray):
    arr = np.asarray(gray).copy()
    h, w = arr.shape
    protected = estimate_breast_protection(gray)

    # Bright connected components in background only.
    candidates = ((arr >= MIN_TEXT_BRIGHTNESS) & (~protected)).astype(np.uint8)

    # Close tiny gaps between characters; never expand into protected tissue.
    join = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, join, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    remove = np.zeros((h, w), dtype=np.uint8)
    removed_components = 0

    for idx in range(1, n):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if not (MIN_TEXT_AREA <= area <= MAX_TEXT_AREA):
            continue
        if cw > int(w * MAX_TEXT_WIDTH_FRAC) or ch > int(h * MAX_TEXT_HEIGHT_FRAC):
            continue

        comp = (labels == idx) & (~protected)
        if np.any(comp):
            remove[comp] = 1
            removed_components += 1

    out = arr.copy()
    out[remove.astype(bool)] = 0
    return Image.fromarray(out), Image.fromarray(remove * 255), int(remove.sum()), removed_components, Image.fromarray(protected.astype(np.uint8) * 255)


def make_comparison(before, after, removed, protected, path):
    width, height = 320, 360
    canvas = Image.new("RGB", (width * 4, height + 40), "white")
    draw = ImageDraw.Draw(canvas)
    for x, im, title in [
        (0, before, "BEFORE"),
        (width, after, "AFTER"),
        (width * 2, removed, "REMOVED"),
        (width * 3, protected, "PROTECTED MASK")
    ]:
        view = im.convert("RGB")
        view.thumbnail((width - 16, height - 40))
        canvas.paste(view, (x + (width - view.width) // 2, 32 + (height - 40 - view.height) // 2))
        draw.text((x + 8, 10), title, fill="black")
    canvas.save(path, quality=95)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input dataset not found:\n{INPUT}\nEdit PROJECT path in this script.")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in INPUT.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise RuntimeError(f"No images found under {INPUT}")

    PREVIEW.mkdir(parents=True, exist_ok=True)
    selected = files[:MAX_IMAGES] if PREVIEW_ONLY else files
    total_pixels = total_components = failures = 0

    for i, src in enumerate(selected, 1):
        try:
            before = load_gray(src)
            after, removed, count, components, protected = remove_external_labels(before)
            total_pixels += count
            total_components += components
            rel = src.relative_to(INPUT)
            preview_out = PREVIEW / rel.parent / f"{src.stem}_preview.png"
            preview_out.parent.mkdir(parents=True, exist_ok=True)
            after.save(preview_out)
            if i <= COMPARISONS:
                make_comparison(before, after, removed, protected,
                                PREVIEW / f"comparison_{i:02d}_{src.stem}.jpg")
            print(f"[{i}/{len(selected)}] {src.name} | removed pixels={count}")
        except Exception as e:
            failures += 1
            print(f"FAILED {src}: {e}")

    print("\nFinished preview.")
    print(f"Images: {len(selected)} | failures: {failures}")
    print(f"Candidate components removed: {total_components}")
    print(f"Candidate pixels changed to black: {total_pixels}")
    print(f"Preview folder: {PREVIEW}")
    print("This is a heuristic mask, not a validated medical segmentation model.")
    print("Do not process the full dataset until the mask and cleaned images are reviewed.")


if __name__ == "__main__":
    main()
