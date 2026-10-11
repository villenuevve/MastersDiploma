import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "new_images"
OUTPUT_DIR = BASE_DIR / "results" / "batch_predictions"

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def safe_stem(path):
    import re
    return re.sub(r"[^\w-]+", "_", path.stem, flags=re.UNICODE).strip("_")


def main():
    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    rows = []
    class_pixel_counts = Counter()
    total_pixels = 0
    missing = []

    for path in images:
        stem = safe_stem(path)
        mask_path = OUTPUT_DIR / f"{stem}_class_ids.png"
        overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"

        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception as exc:
            missing.append({"filename": path.name, "reason": f"image read error: {exc}"})
            continue

        if not mask_path.exists():
            missing.append({"filename": path.name, "reason": "class_ids mask missing"})
            continue

        with Image.open(mask_path) as im:
            mask = im.convert("L")
            counts = Counter(mask.getdata())

        pixel_count = sum(counts.values())
        total_pixels += pixel_count

        class_counts = {
            str(class_id): int(count)
            for class_id, count in sorted(counts.items())
        }
        for class_id, count in counts.items():
            class_pixel_counts[class_id] += count

        rows.append({
            "filename": path.name,
            "width": width,
            "height": height,
            "pixels": pixel_count,
            "class_count": len(counts),
            "classes_present": "; ".join(map(str, sorted(counts))),
            "mask_file": mask_path.name,
            "overlay_exists": overlay_path.exists(),
            "pixel_counts_by_class_id_json": json.dumps(
                class_counts, ensure_ascii=False
            ),
        })

    csv_path = OUTPUT_DIR / "batch_analysis_per_image.csv"
    fields = [
        "filename", "width", "height", "pixels", "class_count",
        "classes_present", "mask_file", "overlay_exists",
        "pixel_counts_by_class_id_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_directory": str(INPUT_DIR),
        "results_directory": str(OUTPUT_DIR),
        "input_images": len(images),
        "images_with_saved_masks": len(rows),
        "images_without_saved_masks": len(missing),
        "total_pixels_in_saved_masks": total_pixels,
        "class_pixel_counts_by_id": {
            str(k): int(v) for k, v in sorted(class_pixel_counts.items())
        },
        "class_pixel_percentages_by_id": {
            str(k): round(v / total_pixels * 100, 4) if total_pixels else 0
            for k, v in sorted(class_pixel_counts.items())
        },
        "missing_results": missing,
        "note": (
            "This report describes model predictions only. "
            "Without ground-truth masks, it does not calculate mIoU "
            "or pixel accuracy."
        ),
    }

    summary_path = OUTPUT_DIR / "batch_analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== BATCH ANALYSIS ===")
    print(f"Input images: {len(images)}")
    print(f"Images with saved masks: {len(rows)}")
    print(f"Missing masks: {len(missing)}")
    print(f"Total mask pixels analyzed: {total_pixels:,}")
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")

    if missing:
        print("\nImages needing inference:")
        for item in missing:
            print(f"- {item['filename']}: {item['reason']}")


if __name__ == "__main__":
    main()
