import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import backend.segmentation as segmentation
from backend.evaluate_cityscapes_full import (
    IMAGE_DIR,
    MASK_DIR,
    NUM_CLASSES,
    convert_ground_truth,
    update_confusion_matrix,
    calculate_metrics,
)

RESOLUTIONS = [1280, 1600, 1920]
OUTPUT_PATH = Path("results/resolution_comparison.json")


def evaluate_resolution(images, resolution):
    segmentation.MAX_INPUT_SIDE = resolution

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    inference_times = []
    start_total = time.perf_counter()

    print(f"\n=== MAX INPUT SIDE: {resolution} ===")

    for index, image_path in enumerate(images, start=1):
        mask_name = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelIds.png"
        )
        mask_path = MASK_DIR / image_path.parent.name / mask_name

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as mask:
            target = convert_ground_truth(mask)

        start = time.perf_counter()
        prediction = segmentation.predict_mask(image)
        inference_ms = (time.perf_counter() - start) * 1000

        prediction = np.asarray(prediction, dtype=np.uint8)

        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: {image_path.name}"
            )

        update_confusion_matrix(confusion, prediction, target)
        inference_times.append(inference_ms)

        if index % 25 == 0 or index == len(images):
            elapsed = (time.perf_counter() - start_total) / 60
            print(
                f"[{index}/{len(images)}] "
                f"Elapsed: {elapsed:.1f} min | "
                f"Inference: {inference_ms:.1f} ms"
            )

    metrics = calculate_metrics(confusion)
    total_time = time.perf_counter() - start_total

    return {
        "max_input_side": resolution,
        "images_evaluated": len(images),
        "mIoU_percent": metrics["mIoU_percent"],
        "pixel_accuracy_percent": metrics["pixel_accuracy_percent"],
        "inference_time_ms": {
            "mean": round(float(np.mean(inference_times)), 2),
            "median": round(float(np.median(inference_times)), 2),
            "min": round(float(np.min(inference_times)), 2),
            "max": round(float(np.max(inference_times)), 2),
        },
        "total_time_seconds": round(total_time, 2),
        "per_class": metrics["per_class"],
    }


def main():
    images = sorted(IMAGE_DIR.rglob("*_leftImg8bit.png"))

    if len(images) != 500:
        raise ValueError(
            f"Expected 500 validation images, found {len(images)}"
        )

    results = []

    print("Model:", segmentation.MODEL_NAME)
    print("Device:", segmentation.DEVICE)
    print("Validation images:", len(images))
    print("Resolutions:", RESOLUTIONS)

    for resolution in RESOLUTIONS:
        try:
            result = evaluate_resolution(images, resolution)
            results.append(result)

            print(
                f"\nRESULT {resolution}: "
                f"mIoU={result['mIoU_percent']}%, "
                f"Pixel Accuracy={result['pixel_accuracy_percent']}%, "
                f"Mean inference={result['inference_time_ms']['mean']} ms"
            )

        except Exception as error:
            print(f"\nERROR at resolution {resolution}: {error}")
            results.append({
                "max_input_side": resolution,
                "error": str(error),
            })

            if segmentation.DEVICE.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        finally:
            gc.collect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": segmentation.MODEL_NAME,
        "device": str(segmentation.DEVICE),
        "dataset": "Cityscapes",
        "split": "val",
        "images": len(images),
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\n=== COMPARISON SUMMARY ===")

    for result in results:
        if "error" in result:
            print(
                f"{result['max_input_side']} px: "
                f"FAILED — {result['error']}"
            )
        else:
            print(
                f"{result['max_input_side']} px | "
                f"mIoU: {result['mIoU_percent']}% | "
                f"Pixel Accuracy: {result['pixel_accuracy_percent']}% | "
                f"Mean inference: {result['inference_time_ms']['mean']} ms"
            )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
