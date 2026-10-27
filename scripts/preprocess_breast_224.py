from pathlib import Path
from PIL import Image
import json

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "datasets" / "breast_preprocessed"
OUTPUT = PROJECT / "datasets" / "breast_preprocessed_224"
SIZE = (224, 224)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    images = sorted(
        p for p in SOURCE.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No images found in {SOURCE}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    failed = []
    processed = 0

    print(f"Images found: {len(images)}")
    print(f"Output folder: {OUTPUT}")

    for i, path in enumerate(images, 1):
        try:
            relative = path.relative_to(SOURCE)
            destination = OUTPUT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(path) as image:
                if image.mode in ("1", "L", "I", "F", "I;16"):
                    image = image.convert("L")
                else:
                    image = image.convert("RGB")

                image = image.resize(SIZE, Image.Resampling.LANCZOS)

                kwargs = {}
                if destination.suffix.lower() in (".jpg", ".jpeg"):
                    kwargs = {"quality": 95, "optimize": True}

                image.save(destination, **kwargs)

            processed += 1

            if i % 500 == 0 or i == len(images):
                print(f"Progress: {i}/{len(images)}")

        except Exception as exc:
            failed.append({"file": str(path), "error": str(exc)})
            print(f"FAILED: {path} - {exc}")

    report = {
        "target_size": list(SIZE),
        "source_count": len(images),
        "processed": processed,
        "failed_count": len(failed),
        "failed_files": failed
    }

    (OUTPUT / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\n--- SUMMARY ---")
    print(f"Found: {len(images)}")
    print(f"Processed: {processed}")
    print(f"Failed: {len(failed)}")
    print(f"Output: {OUTPUT}")

if __name__ == "__main__":
    main()
