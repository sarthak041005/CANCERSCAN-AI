"""
CancerScan AI — optional pretrained breast-region segmentation baseline.

This script downloads no weights and trains no model. It tests a classical OpenCV
GrabCut segmentation proposal on a small image sample and saves masks/overlays for
manual review. GrabCut is not a validated mammography segmentation model; do not use
its output for automatic cleaning without careful review.

Install:
    python -m pip install opencv-python pillow numpy

Run from the CancerScan_AI project folder:
    python .\try_grabcut_breast_masks.py

Output:
    datasets\grabcut_mask_auditGet-Content .\scripts\breast\pretrain_breast_ssl.py

No source images are modified.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

PROJECT = Path(r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI")
INPUT = PROJECT / "datasets" / "breast_preprocessed_224"
OUTPUT = PROJECT / "datasets" / "grabcut_mask_audit"
MAX_IMAGES = 20


def load_gray(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("L")


def grabcut_mask(gray):
    arr = np.asarray(gray)
    h, w = arr.shape
    # Use a center rectangle as probable foreground; border is probable background.
    # This is only a proposal and can fail for breast tissue near image edges.
    bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    margin_x = max(3, int(w * 0.06))
    margin_y = max(3, int(h * 0.04))
    rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))
    mask[margin_y:h-margin_y, margin_x:w-margin_x] = cv2.GC_PR_FGD

    # Strong near-black pixels are probable background; bright tissue pixels are
    # probable foreground. These heuristics are not a clinical segmentation model.
    mask[arr < 8] = cv2.GC_BGD
    mask[arr > 45] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), dtype=np.uint8), "GrabCut failed"

    result = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        result = (labels == largest).astype(np.uint8)
    ratio = float(result.mean())
    warning = ""
    if ratio < 0.04 or ratio > 0.92:
        warning = " WARNING: implausible area"
    return result, f"mask area {ratio:.1%}{warning}"


def save_panel(image, mask, note, path):
    width, height = 320, 360
    canvas = Image.new("RGB", (width * 3, height + 48), "white")
    draw = ImageDraw.Draw(canvas)
    boundary = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    overlay = np.asarray(image).copy()
    overlay[boundary > 0] = 255
    panels = [
        (0, image.convert("RGB"), "ORIGINAL"),
        (width, Image.fromarray(mask * 255).convert("RGB"), "GRABCUT MASK"),
        (width * 2, Image.fromarray(overlay).convert("RGB"), "BOUNDARY OVERLAY"),
    ]
    for x, im, title in panels:
        im.thumbnail((width - 16, height - 40))
        canvas.paste(im, (x + (width - im.width) // 2, 32 + (height - 40 - im.height) // 2))
        draw.text((x + 8, 8), title, fill="black")
    draw.text((8, height + 18), note, fill="black")
    canvas.save(path, quality=95)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in INPUT.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise RuntimeError(f"No images found in {INPUT}")
    indices = np.linspace(0, len(files) - 1, num=min(MAX_IMAGES, len(files)), dtype=int)
    selected = [files[i] for i in indices]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for i, path in enumerate(selected, 1):
        try:
            image = load_gray(path)
            mask, note = grabcut_mask(image)
            save_panel(image, mask, note, OUTPUT / f"grabcut_{i:02d}_{path.stem}.jpg")
            print(f"[{i}/{len(selected)}] {path.name}: {note}")
        except Exception as exc:
            failures += 1
            print(f"FAILED {path}: {exc}")
    print(f"\nFinished. Images: {len(selected)} | failures: {failures}")
    print(f"Output: {OUTPUT}")
    print("This is an experimental mask proposal only. No source images were modified.")


if __name__ == "__main__":
    main()
