
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import random

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "datasets" / "breast_preprocessed_224"
OUT = ROOT / "reports" / "breast_preprocessing_audit"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(
    p for p in FOLDER.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

random.seed(42)
samples = random.sample(files, min(20, len(files)))

thumb_size = 224
label_height = 34
columns = 4
rows = (len(samples) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * thumb_size, rows * (thumb_size + label_height)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(samples):
    image = Image.open(path).convert("RGB")
    image.thumbnail((thumb_size, thumb_size))

    x = (i % columns) * thumb_size
    y = (i // columns) * (thumb_size + label_height)

    sheet.paste(image, (x, y))
    label = f"{i+1}. {path.parent.name[:18]}"
    draw.text((x + 3, y + thumb_size + 5), label, fill="black")

sheet_path = OUT / "breast_random_contact_sheet.jpg"
sheet.save(sheet_path, quality=95)

print("Contact sheet saved:", sheet_path)
print("\nSample image paths:")
for path in samples:
    print(path.relative_to(ROOT))

sheet.show()
