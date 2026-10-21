from pathlib import Path

import cv2
import pytesseract
import pandas as pd
import re

# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_DIR = Path(r"datasets\jpeg")

OUTPUT_DIR = Path("mammo_text_detection")
PREVIEW_DIR = OUTPUT_DIR / "previews"

OUTPUT_DIR.mkdir(exist_ok=True)
PREVIEW_DIR.mkdir(exist_ok=True)

# Tell Python exactly where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# --------------------------------------------------
# TEXT / MARKERS WE ARE PARTICULARLY INTERESTED IN
# --------------------------------------------------

TARGET_WORDS = {
    "L",
    "R",
    "LCC",
    "RCC",
    "LMLO",
    "RMLO",
    "MLO",
    "CC",
    "LAT",
    "AP",
    "PA"
}

# --------------------------------------------------
# STRICT OCR SETTINGS
# --------------------------------------------------

MIN_OCR_CONFIDENCE = 70

# Text markers are usually relatively small compared
# with the complete mammogram.
MIN_BOX_WIDTH = 3
MIN_BOX_HEIGHT = 3
MAX_BOX_WIDTH_RATIO = 0.20
MAX_BOX_HEIGHT_RATIO = 0.20

# We mainly care about text close to the image border.
BORDER_RATIO = 0.25

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalize_text(text):
    """Clean OCR text for comparison."""

    text = text.upper().strip()

    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("_", "")
    text = text.replace(".", "")
    text = text.replace("|", "")
    text = text.replace("'", "")

    return text


def is_near_border(x, y, w, h, image_width, image_height):
    """Check whether the OCR box is reasonably close to an image border."""

    left_distance = x
    right_distance = image_width - (x + w)
    top_distance = y
    bottom_distance = image_height - (y + h)

    border_limit_x = int(image_width * BORDER_RATIO)
    border_limit_y = int(image_height * BORDER_RATIO)

    return (
        left_distance <= border_limit_x
        or right_distance <= border_limit_x
        or top_distance <= border_limit_y
        or bottom_distance <= border_limit_y
    )


def is_reasonable_box(x, y, w, h, image_width, image_height):
    """Reject boxes that are clearly too small or too large."""

    if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
        return False

    if w > image_width * MAX_BOX_WIDTH_RATIO:
        return False

    if h > image_height * MAX_BOX_HEIGHT_RATIO:
        return False

    return True


def looks_like_marker(text):
    """Check whether OCR text resembles a mammogram marker."""

    normalized = normalize_text(text)

    if normalized in TARGET_WORDS:
        return True

    # Allow combinations such as LCC/RCC/LMLO/RMLO.
    if re.fullmatch(r"[LR](CC|MLO)", normalized):
        return True

    return False


def looks_like_short_text(text):
    """Allow only short OCR results."""

    normalized = normalize_text(text)

    if not normalized:
        return False

    # Avoid long strings that are usually tissue/false OCR.
    if len(normalized) > 5:
        return False

    # Only allow alphabetic marker-like text.
    if not re.fullmatch(r"[A-Z]+", normalized):
        return False

    return True


# --------------------------------------------------
# FIND ALL MAMMOGRAMS
# --------------------------------------------------

images = list(INPUT_DIR.rglob("*.jpg"))

print(f"Found {len(images)} mammography images.")
print("Starting STRICT text/marker detection...\n")

results = []

# --------------------------------------------------
# PROCESS EACH IMAGE
# --------------------------------------------------

for index, image_path in enumerate(images, start=1):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        results.append({
            "image": str(image_path),
            "status": "UNREADABLE",
            "detected_text": "",
            "target_marker": "",
            "boxes": ""
        })

        continue

    height, width = image.shape

    # --------------------------------------------------
    # UPSCALE IMAGE FOR OCR
    # --------------------------------------------------

    scale = 2

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------
    # THRESHOLDING
    # --------------------------------------------------

    _, thresh = cv2.threshold(
        resized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------
    # OCR
    # --------------------------------------------------

    # PSM 11 searches for sparse text.
    config = "--oem 3 --psm 11"

    data = pytesseract.image_to_data(
        thresh,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    detected_text = []
    target_marker = []
    boxes = []

    # --------------------------------------------------
    # COLLECT AND FILTER OCR RESULTS
    # --------------------------------------------------

    for i, raw_text in enumerate(data["text"]):

        raw_text = raw_text.strip()

        if not raw_text:
            continue

        # --------------------------------------------------
        # OCR CONFIDENCE
        # --------------------------------------------------

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        # Strict confidence requirement.
        if confidence < MIN_OCR_CONFIDENCE:
            continue

        # --------------------------------------------------
        # GET BOX
        # --------------------------------------------------

        x = int(data["left"][i] / scale)
        y = int(data["top"][i] / scale)
        w = int(data["width"][i] / scale)
        h = int(data["height"][i] / scale)

        # --------------------------------------------------
        # BOX SIZE FILTER
        # --------------------------------------------------

        if not is_reasonable_box(
            x,
            y,
            w,
            h,
            width,
            height
        ):
            continue

        # --------------------------------------------------
        # NORMALIZE OCR TEXT
        # --------------------------------------------------

        normalized = normalize_text(raw_text)

        if not looks_like_short_text(normalized):
            continue

        # --------------------------------------------------
        # BORDER FILTER
        # --------------------------------------------------

        near_border = is_near_border(
            x,
            y,
            w,
            h,
            width,
            height
        )

        # --------------------------------------------------
        # MARKER FILTER
        # --------------------------------------------------

        marker = looks_like_marker(normalized)

        # Only accept a detection if:
        #
        # 1. It looks like a known mammogram marker
        #    AND is near an image border
        #
        # OR
        #
        # 2. It is very high-confidence short text
        #    near the image border.
        #
        # This makes the detector conservative.

        if not near_border:
            continue

        if not marker and confidence < 85:
            continue

        # --------------------------------------------------
        # SAVE ACCEPTED DETECTION
        # --------------------------------------------------

        detected_text.append(normalized)

        if marker:
            target_marker.append(normalized)

        boxes.append(
            f"{normalized}@({x},{y},{w},{h})"
        )

    # --------------------------------------------------
    # REMOVE DUPLICATE DETECTIONS
    # --------------------------------------------------

    detected_text = list(dict.fromkeys(detected_text))
    target_marker = list(dict.fromkeys(target_marker))
    boxes = list(dict.fromkeys(boxes))

    # --------------------------------------------------
    # SAVE PREVIEW ONLY FOR ACCEPTED DETECTIONS
    # --------------------------------------------------

    if boxes:

        preview = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

        for box in boxes:

            try:

                text, coordinates = box.split("@")

                x, y, w, h = map(
                    int,
                    coordinates.strip("()").split(",")
                )

                # Red bounding box.
                cv2.rectangle(
                    preview,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

                # Text label.
                cv2.putText(
                    preview,
                    text,
                    (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

            except Exception:
                pass

        preview_name = (
            f"{index:05d}_{image_path.stem}.jpg"
        )

        cv2.imwrite(
            str(PREVIEW_DIR / preview_name),
            preview
        )

    # --------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------

    results.append({

        "image": str(image_path),

        "status": "OK",

        "detected_text": " | ".join(
            detected_text
        ),

        "target_marker": " | ".join(
            target_marker
        ),

        "boxes": " | ".join(
            boxes
        )
    })

    # --------------------------------------------------
    # PROGRESS
    # --------------------------------------------------

    if index % 100 == 0:

        print(
            f"Processed {index}/{len(images)}"
        )


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

df = pd.DataFrame(results)

csv_path = (
    OUTPUT_DIR /
    "text_marker_detection.csv"
)

df.to_csv(
    csv_path,
    index=False
)

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

detections = sum(
    bool(row["boxes"])
    for row in results
)

markers = sum(
    bool(row["target_marker"])
    for row in results
)

print("\n----------------------------------------")
print("STRICT DETECTION COMPLETED")
print("----------------------------------------")
print(f"Images processed: {len(images)}")
print(f"Images with accepted text: {detections}")
print(f"Images with target markers: {markers}")
print(f"CSV report: {csv_path}")
print(f"Preview folder: {PREVIEW_DIR}")
print("----------------------------------------")