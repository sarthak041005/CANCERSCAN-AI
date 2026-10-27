"""
CancerScan AI — diverse breast-mask audit (NO cleaning / NO source modifications).

Samples images spread across the dataset rather than taking only the first files.
Creates ORIGINAL / ESTIMATED MASK / BOUNDARY OVERLAY panels for review.

Install:
    python -m pip install opencv-python pillow numpy

Run from the CancerScan_AI project directory:
    python .\audit_breast_mask_diverse.py
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

PROJECT = Path(r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI")
INPUT = PROJECT / "datasets" / "breast_preprocessed_224"
OUTPUT = PROJECT / "datasets" / "breast_mask_audit_diverse"
MAX_IMAGES = 60
COMPARISONS = 60


def load_gray(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("L")


def make_mask(gray):
    arr = np.asarray(gray)
    h, w = arr.shape
    threshold, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (arr > max(8, int(threshold * 0.65))).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), dtype=np.uint8), "WARNING: empty mask"
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    ratio = float(mask.mean())
    if ratio < 0.04:
        note = f"WARNING: tiny mask {ratio:.1%}"
    elif ratio > 0.92:
        note = f"WARNING: huge mask {ratio:.1%}"
    else:
        note = f"foreground {ratio:.1%}"
    return mask, note


def make_comparison(before, mask, note, dest):
    panel_w, panel_h = 320, 360
    canvas = Image.new("RGB", (panel_w * 3, panel_h + 48), "white")
    draw = ImageDraw.Draw(canvas)
    boundary = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    overlay = np.asarray(before).copy()
    overlay[boundary > 0] = 255

    panels = [
        (0, before.convert("RGB"), "ORIGINAL"),
        (panel_w, Image.fromarray(mask * 255).convert("RGB"), "ESTIMATED MASK"),
        (panel_w * 2, Image.fromarray(overlay).convert("RGB"), "BOUNDARY OVERLAY"),
    ]
    for x, im, title in panels:
        im.thumbnail((panel_w - 16, panel_h - 40))
        canvas.paste(im, (x + (panel_w - im.width) // 2, 32 + (panel_h - 40 - im.height) // 2))
        draw.text((x + 8, 8), title, fill="black")
    draw.text((8, panel_h + 18), note, fill="black")
    canvas.save(dest, quality=95)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in INPUT.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise RuntimeError(f"No images found in {INPUT}")

    # Evenly sample across the sorted list to cover more filenames/views.
    count = min(MAX_IMAGES, len(files))
    indices = np.linspace(0, len(files) - 1, num=count, dtype=int)
    selected = [files[i] for i in indices]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for i, path in enumerate(selected, 1):
        try:
            image = load_gray(path)
            mask, note = make_mask(image)
            make_comparison(image, mask, note, OUTPUT / f"mask_audit_{i:02d}_{path.stem}.jpg")
            print(f"[{i}/{len(selected)}] {path.name}: {note}")
        except Exception as exc:
            failures += 1
            print(f"FAILED {path}: {exc}")

    print(f"\nAudit finished. Selected: {len(selected)} | Failures: {failures}")
    print(f"Output folder: {OUTPUT}")
    print("This script only creates visual audits; it does not clean or modify source images.")


if __name__ == "__main__":
    main()
