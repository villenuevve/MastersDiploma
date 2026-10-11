import csv
import re
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.segmentation import predict_mask, colorize_mask
from backend.visualization import create_overlay, save_class_ids

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "new_images"
OUTPUT_DIR = BASE_DIR / "results" / "batch_predictions"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_SIDE = 1024


def safe_name(name):
    name = re.sub(r"[^\w-]+", "_", name, flags=re.UNICODE)
    return name.strip("_") or "image"


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    rows = []
    print(f"Images found: {len(image_paths)}")
    print(f"Maximum inference side: {MAX_SIDE}px")
    print(f"Output folder: {OUTPUT_DIR}")

    for index, image_path in enumerate(image_paths, start=1):
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            original_width, original_height = image.size

            # Reduce only the inference copy; preserve the original for output.
            inference_image = image.copy()
            inference_image.thumbnail(
                (MAX_SIDE, MAX_SIDE),
                Image.Resampling.LANCZOS,
            )

            mask, stats = predict_mask(inference_image, return_stats=True)

            # Restore the mask to the original image dimensions.
            mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
            mask_image = mask_image.resize(
                (original_width, original_height),
                resample=Image.Resampling.NEAREST,
            )
            mask = np.asarray(mask_image, dtype=np.uint8)

            colored_mask = colorize_mask(mask)
            overlay = create_overlay(image, colored_mask)

            stem = safe_name(image_path.stem)
            mask_path = OUTPUT_DIR / f"{stem}_mask.png"
            overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"
            ids_path = OUTPUT_DIR / f"{stem}_class_ids.png"

            Image.fromarray(colored_mask).save(mask_path)
            overlay.save(overlay_path)
            save_class_ids(mask, str(ids_path))

            class_summary = "; ".join(
                f"{item['name']}: {item['percentage']}%"
                for item in stats["detected_classes"]
            )

            rows.append({
                "filename": image_path.name,
                "width": original_width,
                "height": original_height,
                "inference_width": inference_image.width,
                "inference_height": inference_image.height,
                "model": stats["model"],
                "device": stats["device"],
                "inference_time_ms": stats["inference_time_ms"],
                "detected_class_count": len(stats["detected_classes"]),
                "detected_classes": class_summary,
                "mask_file": mask_path.name,
                "overlay_file": overlay_path.name,
                "class_ids_file": ids_path.name,
                "status": "success",
            })

            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: "
                f"{original_width}x{original_height} -> "
                f"{inference_image.width}x{inference_image.height}; "
                f"{stats['inference_time_ms']:.1f} ms"
            )

        except Exception as exc:
            print(f"[{index}/{len(image_paths)}] ERROR {image_path.name}: {exc}")
            rows.append({
                "filename": image_path.name,
                "status": f"error: {exc}",
            })
            traceback.print_exc()

    fieldnames = [
        "filename", "width", "height",
        "inference_width", "inference_height",
        "model", "device", "inference_time_ms",
        "detected_class_count", "detected_classes",
        "mask_file", "overlay_file", "class_ids_file", "status",
    ]

    csv_path = OUTPUT_DIR / "batch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(row.get("status") == "success" for row in rows)
    print("\nBatch processing complete.")
    print(f"Successfully processed: {successful}/{len(image_paths)}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
