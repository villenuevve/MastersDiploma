import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.segmentation import predict_mask


ROOT = Path("data/evaluation/cityscapes_full")
IMAGE_DIR = ROOT / "leftImg8bit" / "val"
MASK_DIR = ROOT / "gtFine" / "val"
OUTPUT_PATH = Path("results/cityscapes_full_val_metrics.json")

NUM_CLASSES = 19
IGNORE_INDEX = 255

LABEL_ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def convert_ground_truth(mask):
    raw = np.asarray(mask.convert("L"), dtype=np.uint8)
    converted = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)

    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        converted[raw == label_id] = train_id

    return converted


def update_confusion_matrix(confusion, prediction, target):
    valid = (
        (target != IGNORE_INDEX)
        & (target >= 0)
        & (target < NUM_CLASSES)
        & (prediction >= 0)
        & (prediction < NUM_CLASSES)
    )

    encoded = NUM_CLASSES * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)

    confusion += np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_metrics(confusion):
    true_positive = np.diag(confusion)
    ground_truth_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    union = ground_truth_count + predicted_count - true_positive

    iou = np.divide(
        true_positive,
        union,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=union > 0
    )

    present_classes = union > 0
    mean_iou = float(iou[present_classes].mean())

    total_pixels = int(confusion.sum())
    pixel_accuracy = (
        float(true_positive.sum() / total_pixels)
        if total_pixels else 0.0
    )

    per_class = {
        CLASS_NAMES[i]: {
            "iou": round(float(iou[i]), 6),
            "iou_percent": round(float(iou[i] * 100), 2),
            "pixels": int(ground_truth_count[i]),
        }
        for i in range(NUM_CLASSES)
    }

    return {
        "mIoU": round(mean_iou, 6),
        "mIoU_percent": round(mean_iou * 100, 2),
        "pixel_accuracy": round(pixel_accuracy, 6),
        "pixel_accuracy_percent": round(pixel_accuracy * 100, 2),
        "evaluated_pixels": total_pixels,
        "classes_present": int(present_classes.sum()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def main():
    images = sorted(IMAGE_DIR.rglob("*_leftImg8bit.png"))

    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    inference_times = []
    processed = 0
    start_total = time.perf_counter()

    print(f"Validation images: {len(images)}")
    print("Starting full Cityscapes evaluation...")
    print("The model configuration will remain unchanged.")

    for index, image_path in enumerate(images, start=1):
        mask_name = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelIds.png"
        )
        mask_path = MASK_DIR / image_path.parent.name / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as mask:
            target = convert_ground_truth(mask)

        inference_start = time.perf_counter()
        prediction = predict_mask(image)
        inference_ms = (time.perf_counter() - inference_start) * 1000

        prediction = np.asarray(prediction, dtype=np.uint8)

        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {image_path.name}: "
                f"prediction={prediction.shape}, target={target.shape}"
            )

        update_confusion_matrix(confusion, prediction, target)

        inference_times.append(inference_ms)
        processed += 1

        if index % 25 == 0 or index == len(images):
            elapsed = time.perf_counter() - start_total
            print(
                f"[{index}/{len(images)}] "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"Last inference: {inference_ms:.1f} ms"
            )

    total_time = time.perf_counter() - start_total
    metrics = calculate_metrics(confusion)

    result = {
        "dataset": "Cityscapes",
        "split": "val",
        "model": "SegFormer-B2 Cityscapes",
        "images_evaluated": processed,
        "total_evaluation_time_seconds": round(total_time, 2),
        "inference_time_ms": {
            "mean": round(float(np.mean(inference_times)), 2),
            "median": round(float(np.median(inference_times)), 2),
            "min": round(float(np.min(inference_times)), 2),
            "max": round(float(np.max(inference_times)), 2),
        },
        **metrics,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n=== CITYSCAPES FULL VALIDATION RESULTS ===")
    print(f"Images evaluated: {processed}")
    print(f"mIoU: {metrics['mIoU_percent']}%")
    print(f"Pixel accuracy: {metrics['pixel_accuracy_percent']}%")
    print(f"Mean inference time: {result['inference_time_ms']['mean']} ms")
    print(f"Total time: {total_time / 60:.1f} min")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\n=== PER-CLASS IoU ===")
    for name, values in metrics["per_class"].items():
        print(f"{name:15s}: {values['iou_percent']:6.2f}%")


if __name__ == "__main__":
    main()
