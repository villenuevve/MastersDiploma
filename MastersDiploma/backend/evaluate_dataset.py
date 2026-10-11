from pathlib import Path
from datetime import datetime
from time import perf_counter
import csv
import json
import math

import numpy as np
from PIL import Image

from backend.metrics import evaluate_masks
from backend.segmentation import predict_mask


ROOT = Path("data/evaluation/cityscapes8")
OUTPUT_DIR = Path("results")
NUM_CLASSES = 19
IGNORE_INDEX = 255

LABEL_ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}


def json_safe(value):
    """Convert NumPy values and non-finite floats into valid JSON values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def convert_ground_truth(mask):
    converted = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        converted[mask == label_id] = train_id
    return converted


def evaluate_split(split):
    image_dir = ROOT / "images" / split
    mask_dir = ROOT / "masks" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = sorted(image_dir.glob("*.png"))
    predictions = []
    targets = []
    image_results = []
    evaluated_files = []

    for image_path in image_paths:
        mask_path = mask_dir / image_path.name

        if not mask_path.is_file():
            print(f"Skipping {image_path.name}: mask missing")
            continue

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            width, height = image.size

            with Image.open(mask_path) as mask_image:
                target = convert_ground_truth(
                    np.asarray(mask_image.convert("L"))
                )

            if not np.any(target != IGNORE_INDEX):
                print(f"Skipping {image_path.name}: no valid labels")
                continue

            # Time only the model prediction, not file loading or metric calculation.
            start = perf_counter()
            prediction = np.asarray(predict_mask(image)).astype(np.uint8)
            inference_seconds = perf_counter() - start

        if prediction.shape != target.shape:
            raise ValueError(
                f"Size mismatch for {image_path.name}: "
                f"prediction={prediction.shape}, target={target.shape}"
            )

        image_metrics = evaluate_masks(
            prediction,
            target,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
        )

        valid_pixels = int(np.count_nonzero(target != IGNORE_INDEX))
        image_result = {
            "split": split,
            "file": image_path.name,
            "width": width,
            "height": height,
            "total_pixels": width * height,
            "evaluated_pixels": valid_pixels,
            "inference_seconds": inference_seconds,
            "inference_ms": inference_seconds * 1000,
            "fps": 1.0 / inference_seconds if inference_seconds > 0 else None,
            "mean_iou": image_metrics["mean_iou"],
            "pixel_accuracy": image_metrics["pixel_accuracy"],
            "per_class_iou": image_metrics["per_class_iou"],
            "confusion_matrix": image_metrics["confusion_matrix"],
        }

        image_results.append(json_safe(image_result))
        predictions.append(prediction)
        targets.append(target)
        evaluated_files.append(image_path.name)

        print(
            f"[{split}] {image_path.name} | "
            f"{width}x{height} | "
            f"mIoU={image_metrics['mean_iou'] * 100:.2f}% | "
            f"Pixel Acc={image_metrics['pixel_accuracy'] * 100:.2f}% | "
            f"Inference={inference_seconds * 1000:.2f} ms"
        )

    if not predictions:
        raise RuntimeError(f"No valid image-mask pairs in {split}")

    metrics = evaluate_masks(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
    )
    metrics["split"] = split
    metrics["images_evaluated"] = len(evaluated_files)
    metrics["files"] = evaluated_files

    inference_times = [r["inference_seconds"] for r in image_results]
    metrics["inference_timing"] = {
        "mean_ms": float(np.mean(inference_times) * 1000),
        "median_ms": float(np.median(inference_times) * 1000),
        "min_ms": float(np.min(inference_times) * 1000),
        "max_ms": float(np.max(inference_times) * 1000),
        "total_seconds": float(np.sum(inference_times)),
        "mean_fps": (
            float(1.0 / np.mean(inference_times))
            if np.mean(inference_times) > 0 else None
        ),
        "note": (
            "Timing covers predict_mask only; excludes image loading, "
            "ground-truth loading, and metric calculation. First-call "
            "initialization may affect the first measurement."
        ),
    }

    return json_safe(metrics), image_results, predictions, targets


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_metrics = {}
    all_image_results = []
    cached_predictions = {}
    cached_targets = {}

    for split in ("train", "val"):
        metrics, image_results, predictions, targets = evaluate_split(split)
        split_metrics[split] = metrics
        all_image_results.extend(image_results)

        for filename, prediction, target in zip(
            metrics["files"], predictions, targets
        ):
            key = f"{split}/{filename}"
            cached_predictions[key] = prediction
            cached_targets[key] = target

    # Combined metrics reuse the predictions above; no second inference pass.
    all_keys = sorted(cached_predictions)
    all_predictions = [cached_predictions[k] for k in all_keys]
    all_targets = [cached_targets[k] for k in all_keys]

    combined = evaluate_masks(
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_targets, axis=0),
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
    )
    combined["split"] = "all"
    combined["images_evaluated"] = len(all_keys)
    combined["files"] = all_keys

    combined_times = [
        r["inference_seconds"] for r in all_image_results
    ]
    combined["inference_timing"] = {
        "mean_ms": float(np.mean(combined_times) * 1000),
        "median_ms": float(np.median(combined_times) * 1000),
        "min_ms": float(np.min(combined_times) * 1000),
        "max_ms": float(np.max(combined_times) * 1000),
        "total_seconds": float(np.sum(combined_times)),
        "mean_fps": (
            float(1.0 / np.mean(combined_times))
            if np.mean(combined_times) > 0 else None
        ),
        "note": (
            "Combined train+val timing is descriptive, not an independent "
            "test benchmark."
        ),
    }

    results = {
        "train": split_metrics["train"],
        "val": split_metrics["val"],
        "all": json_safe(combined),
    }

    # Preserve existing JSON outputs used by the application and report.
    (OUTPUT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(results["val"], indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "evaluation_metrics_all.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    # New per-image diagnostic dataset.
    per_image_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "images": all_image_results,
    }
    (OUTPUT_DIR / "evaluation_per_image.json").write_text(
        json.dumps(
            json_safe(per_image_payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # Flat CSV suitable for spreadsheet analysis.
    csv_path = OUTPUT_DIR / "evaluation_per_image.csv"
    fields = [
        "split", "file", "width", "height", "total_pixels",
        "evaluated_pixels", "mean_iou", "pixel_accuracy",
        "inference_ms", "fps",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in all_image_results:
            writer.writerow({field: result.get(field) for field in fields})

    for split, result in results.items():
        timing = result["inference_timing"]
        print(f"\n=== {split.upper()} ===")
        print(f"Images: {result['images_evaluated']}")
        print(f"mIoU: {result['mean_iou'] * 100:.2f}%")
        print(f"Pixel Accuracy: {result['pixel_accuracy'] * 100:.2f}%")
        print(f"Evaluated pixels: {result['evaluated_pixels']:,}")
        print(f"Mean inference: {timing['mean_ms']:.2f} ms")
        print(f"Median inference: {timing['median_ms']:.2f} ms")
        print(f"Mean FPS: {timing['mean_fps']:.2f}")

    print("\nSaved:")
    print(OUTPUT_DIR / "evaluation_metrics.json")
    print(OUTPUT_DIR / "evaluation_metrics_all.json")
    print(OUTPUT_DIR / "evaluation_per_image.json")
    print(csv_path)


if __name__ == "__main__":
    main()
