"""
CancerScan AI — breast foreground segmentation audit (NO image cleaning).

Purpose:
- Create preview panels showing the original mammogram and a rough breast/foreground mask.
- Do NOT erase text, alter images, or create a training dataset.
- Helps assess whether a classical threshold-based mask is adequate before choosing
  a dedicated segmentation model.

Install:
    python -m pip install opencv-python pillow numpy

Run from the CancerScan_AI project directory:
    python .\audit_breast_mask.py

Output:
    datasets\breast_mask_audit
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

PROJECT = Path(r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI")
INPUT = PROJECT / "datasets" / "breast_preprocessed_224"
OUTPUT = PROJECT / "datasets" / "breast_mask_audit"
MAX_IMAGES = 40
COMPARISONS = 20


def load_gray(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("L")


def make_mask(gray):
    arr = np.asarray(gray)
    h, w = arr.shape

    # Estimate background using Otsu threshold, but fail safely on odd masks.
    threshold, _ = cv2.threshold(
        arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = (arr > max(8, int(threshold * 0.65))).astype(np.uint8)

    # Close small gaps; do not remove or modify any source pixels.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros((h, w), dtype=np.uint8), "empty mask"

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    ratio = float(mask.mean())

    # Flag suspicious masks instead of attempting to correct them automatically.
    if ratio < 0.04:
        return mask, f"WARNING: tiny mask ({ratio:.1%})"
    if ratio > 0.92:
        return mask, f"WARNING: mask covers nearly whole image ({ratio:.1%})"
    return mask, f"estimated foreground {ratio:.1%}"


def comparison(before, mask, note, dest):
    w, h = 320, 360
    canvas = Image.new("RGB", (w * 3, h + 45), "white")
    draw = ImageDraw.Draw(canvas)

    overlay = np.asarray(before).copy()
    m = mask.astype(bool)
    # Mark estimated mask boundary in white; original pixels are otherwise unchanged.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    overlay[boundary > 0] = 255

    panels = [
        (0, before.convert("RGB"), "ORIGINAL"),
        (w, Image.fromarray(mask * 255).convert("RGB"), "ESTIMATED MASK"),
        (w * 2, Image.fromarray(overlay).convert("RGB"), "BOUNDARY OVERLAY"),
    ]
    for x, im, title in panels:
        im.thumbnail((w - 16, h - 40))
        canvas.paste(im, (x + (w - im.width) // 2, 32 + (h - 40 - im.height) // 2))
        draw.text((x + 8, 8), title, fill="black")
    draw.text((8, h + 18), note, fill="black")
    canvas.save(dest, quality=95)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in INPUT.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if not files:
        raise RuntimeError(f"No images found in {INPUT}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = files[:MAX_IMAGES]
    for i, path in enumerate(selected, 1):
        try:
            image = load_gray(path)
            mask, note = make_mask(image)
            if i <= COMPARISONS:
                comparison(image, mask, note, OUTPUT / f"mask_audit_{i:02d}_{path.stem}.jpg")
            print(f"[{i}/{len(selected)}] {path.name}: {note}")
        except Exception as exc:
            print(f"FAILED {path}: {exc}")

    print(f"\nAudit complete. Open: {OUTPUT}")
    print("No source images were modified and no text was erased.")
    print("This is a rough mask audit, not a clinically validated segmentation model.")


if __name__ == "__main__":
    main()
