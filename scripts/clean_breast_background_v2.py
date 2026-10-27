"""
CancerScan AI — improved preview-only external mammogram annotation cleaner.
This is a heuristic, not clinically validated. It never overwrites inputs.
Install: python -m pip install opencv-python pillow numpy
Run: python clean_breast_background_v2.py
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

PROJECT_DIR = Path(r"C:\Users\sarth\OneDrive\Desktop\CancerScan_AI_complete\mnt\data\CancerScan_AI")
INPUT_DIR = PROJECT_DIR / "datasets" / "breast_preprocessed_224"
PREVIEW_DIR = PROJECT_DIR / "datasets" / "breast_cleaning_preview_v2"
OUTPUT_DIR = PROJECT_DIR / "datasets" / "breast_preprocessed_224_clean_v2"

PREVIEW_ONLY = True
MAX_PREVIEW_IMAGES = 40
COMPARISONS_TO_SAVE = 20
BACKGROUND_THRESHOLD = 18
BRIGHT_TEXT_THRESHOLD = 70
TISSUE_SAFETY_MARGIN = 7
MIN_COMPONENT_AREA = 2
MAX_COMPONENT_AREA = 900
MAX_COMPONENT_WIDTH_FRACTION = 0.32
MAX_COMPONENT_HEIGHT_FRACTION = 0.24

def load_gray(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("L")

def estimate_protected_mask(gray):
    arr = np.asarray(gray)
    h, w = arr.shape
    foreground = (arr > BACKGROUND_THRESHOLD).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, k3, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if n <= 1:
        return np.ones((h, w), dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = (labels == largest).astype(np.uint8)
    ksize = TISSUE_SAFETY_MARGIN * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    protected = cv2.dilate(main, kernel, iterations=1).astype(bool)
    edge = max(2, int(round(min(h, w) * 0.015)))
    protected[:edge, :] = True
    protected[-edge:, :] = True
    protected[:, :edge] = True
    protected[:, -edge:] = True
    return protected

def remove_external_components(gray):
    arr = np.asarray(gray).copy()
    h, w = arr.shape
    protected = estimate_protected_mask(gray)
    candidates = ((arr >= BRIGHT_TEXT_THRESHOLD) & (~protected)).astype(np.uint8)
    join_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    joined = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, join_kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    remove = np.zeros((h, w), dtype=np.uint8)
    removed_components = 0
    for idx in range(1, n):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if not (MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA):
            continue
        if cw > int(w * MAX_COMPONENT_WIDTH_FRACTION) or ch > int(h * MAX_COMPONENT_HEIGHT_FRACTION):
            continue
        component = (labels == idx) & (~protected)
        if np.any(component):
            remove[component] = 1
            removed_components += 1
    cleaned = arr.copy()
    cleaned[remove.astype(bool)] = 0
    return Image.fromarray(cleaned), Image.fromarray(remove * 255), int(remove.sum()), removed_components

def make_comparison(before, after, removed, dest):
    tile_w, tile_h = 320, 360
    canvas = Image.new("RGB", (tile_w * 3, tile_h + 38), "white")
    draw = ImageDraw.Draw(canvas)
    for x, im, label in [(0, before, "BEFORE"), (tile_w, after, "AFTER"), (tile_w * 2, removed, "REMOVED PIXELS")]:
        view = im.convert("RGB")
        view.thumbnail((tile_w - 18, tile_h - 40))
        canvas.paste(view, (x + (tile_w - view.width) // 2, 30 + (tile_h - 40 - view.height) // 2))
        draw.text((x + 10, 8), label, fill="black")
    canvas.save(dest, quality=95)

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found:\n{INPUT_DIR}\nEdit PROJECT_DIR at the top if needed.")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if not files:
        raise RuntimeError(f"No images found under {INPUT_DIR}")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    selected = files[:MAX_PREVIEW_IMAGES] if PREVIEW_ONLY else files
    succeeded = failed = total_pixels = total_components = 0
    for i, src in enumerate(selected, 1):
        try:
            before = load_gray(src)
            after, removed, pixel_count, component_count = remove_external_components(before)
            rel = src.relative_to(INPUT_DIR)
            if PREVIEW_ONLY:
                out = PREVIEW_DIR / rel.parent / f"{rel.stem}_preview.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                after.save(out)
                if i <= COMPARISONS_TO_SAVE:
                    make_comparison(before, after, removed, PREVIEW_DIR / f"comparison_{i:02d}_{src.stem}.jpg")
            else:
                out = OUTPUT_DIR / rel.parent / f"{rel.stem}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                after.save(out)
            succeeded += 1
            total_pixels += pixel_count
            total_components += component_count
            if i % 10 == 0 or i == len(selected):
                print(f"[{i}/{len(selected)}] processed")
        except Exception as exc:
            failed += 1
            print(f"FAILED: {src} :: {exc}")
    print("\nFinished.")
    print(f"Selected: {len(selected)} | Successful: {succeeded} | Failed: {failed}")
    print(f"Candidate components removed: {total_components}")
    print(f"Candidate pixels set to black: {total_pixels}")
    if PREVIEW_ONLY:
        print(f"\nPREVIEW ONLY. Inspect:\n{PREVIEW_DIR}")
        print("Review BEFORE/AFTER/REMOVED PIXELS; keep PREVIEW_ONLY=True until validated.")
        print(f"Full output directory if later approved:\n{OUTPUT_DIR}")
    else:
        print(f"\nCleaned output saved to:\n{OUTPUT_DIR}")
        print("Review outputs and confirm labels/patient-level splits before training.")

if __name__ == "__main__":
    main()
