import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

IMAGE_DIR = Path("data/evaluation/cityscapes_full/leftImg8bit/val")
MASK_DIR = Path("data/evaluation/cityscapes_trainIds/val")

MODEL_NAME = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
OUTPUT_PATH = Path("results/cityscapes_b3_1600_metrics.json")

MAX_INPUT_SIDE = 1600
NUM_CLASSES = 19
IGNORE_INDEX = 255

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def update_confusion_matrix(confusion, prediction, target):
    valid = (
        (target != IGNORE_INDEX)
        & (target < NUM_CLASSES)
        & (prediction < NUM_CLASSES)
    )

    encoded = (
        NUM_CLASSES * target[valid].astype(np.int64)
        + prediction[valid].astype(np.int64)
    )

    confusion += np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_metrics(confusion):
    tp = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - tp

    total_pixels = int(confusion.sum())

    iou = np.divide(
        tp,
        union,
        out=np.full(NUM_CLASSES, np.nan),
        where=union > 0
    )

    recall = np.divide(
        tp,
        gt_count,
        out=np.full(NUM_CLASSES, np.nan),
        where=gt_count > 0
    )

    precision = np.divide(
        tp,
        pred_count,
        out=np.full(NUM_CLASSES, np.nan),
        where=pred_count > 0
    )

    present_gt = gt_count > 0
    present_union = union > 0

    pixel_accuracy = (
        float(tp.sum() / total_pixels)
        if total_pixels > 0 else 0.0
    )

    mean_accuracy = (
        float(np.nanmean(recall[present_gt]))
        if np.any(present_gt) else 0.0
    )

    miou = (
        float(np.nanmean(iou[present_union]))
        if np.any(present_union) else 0.0
    )

    frequencies = gt_count / total_pixels if total_pixels else gt_count

    fwiou = (
        float(np.sum(frequencies[present_gt] * iou[present_gt]))
        if np.any(present_gt) else 0.0
    )

    def class_values(values):
        result = {}
        for index, name in enumerate(CLASS_NAMES):
            value = values[index]
            result[name] = (
                round(float(value) * 100, 2)
                if np.isfinite(value) else None
            )
        return result

    return {
        "mIoU_percent": round(miou * 100, 2),
        "pixel_accuracy_percent": round(pixel_accuracy * 100, 2),
        "mean_accuracy_percent": round(mean_accuracy * 100, 2),
        "frequency_weighted_IoU_percent": round(fwiou * 100, 2),
        "valid_pixels": total_pixels,
        "per_class": {
            "IoU_percent": class_values(iou),
            "precision_percent": class_values(precision),
            "recall_percent": class_values(recall),
        },
        "confusion_matrix": confusion.tolist(),
    }


def predict(image, model, processor, device):
    original_width, original_height = image.size

    # Preprocessing: resize and processor normalization.
    preprocessing_start = time.perf_counter()

    scale = min(
        1.0,
        MAX_INPUT_SIDE / max(original_width, original_height)
    )

    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))

    resized = image.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    inputs = processor(images=resized, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    preprocessing_ms = (
        time.perf_counter() - preprocessing_start
    ) * 1000

    # Model forward. Synchronization is required for accurate MPS timing.
    synchronize(device)
    inference_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    synchronize(device)

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    # Postprocessing: interpolate logits, argmax and restore original size.
    postprocessing_start = time.perf_counter()

    with torch.no_grad():
        logits = F.interpolate(
            outputs.logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )

        prediction = (
            logits.argmax(dim=1)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    mask = Image.fromarray(prediction)
    mask = mask.resize(
        (original_width, original_height),
        Image.Resampling.NEAREST
    )

    prediction = np.asarray(mask, dtype=np.uint8)

    postprocessing_ms = (
        time.perf_counter() - postprocessing_start
    ) * 1000

    return prediction, {
        "preprocessing_ms": preprocessing_ms,
        "inference_ms": inference_ms,
        "postprocessing_ms": postprocessing_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Number of validation images"
    )
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    images = sorted(IMAGE_DIR.rglob("*_leftImg8bit.png"))

    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    if args.max_images < 1 or args.max_images > len(images):
        raise ValueError(
            f"max-images must be between 1 and {len(images)}"
        )

    images = images[:args.max_images]

    print("=== SEGFORMER-B3 EVALUATION ===")
    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Maximum input side:", MAX_INPUT_SIDE)
    print("Images requested:", len(images))
    print("Ground truth:", MASK_DIR)

    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    processor.do_resize = False
    processor.do_pad = False

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    preprocessing_times = []
    inference_times = []
    postprocessing_times = []

    total_pixels = 0
    ignored_pixels = 0

    started = time.perf_counter()

    for index, image_path in enumerate(images, start=1):
        relative = image_path.relative_to(IMAGE_DIR)

        mask_name = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelTrainIds.png"
        )

        mask_path = MASK_DIR / relative.parent / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"TrainIds mask not found: {mask_path}"
            )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as mask:
            target = np.asarray(mask.convert("L"), dtype=np.uint8)

        if target.shape != (image.height, image.width):
            raise ValueError(
                f"Image/mask dimensions mismatch: {image_path.name}"
            )

        total_pixels += target.size
        ignored_pixels += int(np.count_nonzero(target == IGNORE_INDEX))

        prediction, timings = predict(
            image, model, processor, device
        )

        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {image_path.name}"
            )

        update_confusion_matrix(confusion, prediction, target)

        preprocessing_times.append(timings["preprocessing_ms"])
        inference_times.append(timings["inference_ms"])
        postprocessing_times.append(timings["postprocessing_ms"])

        print(
            f"[{index}/{len(images)}] "
            f"Pre: {timings['preprocessing_ms']:.1f} ms | "
            f"Forward: {timings['inference_ms']:.1f} ms | "
            f"Post: {timings['postprocessing_ms']:.1f} ms",
            flush=True
        )

    total_time = time.perf_counter() - started
    metrics = calculate_metrics(confusion)

    mean_pre = float(np.mean(preprocessing_times))
    mean_inf = float(np.mean(inference_times))
    mean_post = float(np.mean(postprocessing_times))
    mean_end_to_end = mean_pre + mean_inf + mean_post

    print("\n=== RESULTS ===")
    print("Images:", len(images))
    print("mIoU:", metrics["mIoU_percent"], "%")
    print("Pixel Accuracy:", metrics["pixel_accuracy_percent"], "%")
    print("Mean Accuracy:", metrics["mean_accuracy_percent"], "%")
    print(
        "Frequency Weighted IoU:",
        metrics["frequency_weighted_IoU_percent"], "%"
    )
    print("Valid pixels:", metrics["valid_pixels"])
    print("Ignored pixels:", ignored_pixels)
    print("Ignored pixels (%):", round(ignored_pixels / total_pixels * 100, 3))

    print("\n=== TIMING ===")
    print("Mean preprocessing:", round(mean_pre, 2), "ms")
    print("Mean model forward:", round(mean_inf, 2), "ms")
    print("Mean postprocessing:", round(mean_post, 2), "ms")
    print("Mean end-to-end:", round(mean_end_to_end, 2), "ms")
    print("Total elapsed:", round(total_time / 60, 2), "min")

    print("\n=== PER-CLASS METRICS ===")
    for name in CLASS_NAMES:
        iou = metrics["per_class"]["IoU_percent"][name]
        precision = metrics["per_class"]["precision_percent"][name]
        recall = metrics["per_class"]["recall_percent"][name]

        print(
            f"{name:15s} "
            f"IoU: {str(iou):>6s}% | "
            f"Precision: {str(precision):>6s}% | "
            f"Recall: {str(recall):>6s}%"
        )

    if len(images) == 500:
        result = {
            "model": MODEL_NAME,
            "dataset": "Cityscapes",
            "split": "val",
            "images_evaluated": len(images),
            "max_input_side": MAX_INPUT_SIDE,
            "device": str(device),
            "timing_ms": {
                "mean_preprocessing": round(mean_pre, 2),
                "mean_model_forward": round(mean_inf, 2),
                "mean_postprocessing": round(mean_post, 2),
                "mean_end_to_end": round(mean_end_to_end, 2),
            },
            "total_time_minutes": round(total_time / 60, 2),
            "total_pixels": total_pixels,
            "ignored_pixels": ignored_pixels,
            "ignored_pixels_percent": round(
                ignored_pixels / total_pixels * 100, 3
            ),
            **metrics,
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("\nSaved:", OUTPUT_PATH)
    else:
        print("\nTest run only. Full evaluation JSON was not saved.")


if __name__ == "__main__":
    main()
