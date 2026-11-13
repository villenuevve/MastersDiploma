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

ROOT = Path("data/evaluation/cityscapes_full")
IMAGE_DIR = ROOT / "leftImg8bit" / "val"
MASK_DIR = ROOT / "gtFine" / "val"

BASELINE_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
FINETUNED_MODEL = Path(
    "archive/experimental_models/segformer-b2-cityscapes-ft"
)

OUTPUT_PATH = Path("results/b2_checkpoint_comparison_1920.json")

MAX_INPUT_SIDE = 1920
NUM_CLASSES = 19
IGNORE_INDEX = 255

LABEL_MAPPING = {
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
    target = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)

    for label_id, train_id in LABEL_MAPPING.items():
        target[raw == label_id] = train_id

    return target


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
    tp = np.diag(confusion)
    gt_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    union = gt_count + pred_count - tp

    iou = np.divide(
        tp,
        union,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=union > 0
    )

    present = union > 0
    total_pixels = int(confusion.sum())

    return {
        "mIoU_percent": round(float(iou[present].mean() * 100), 4),
        "pixel_accuracy_percent": round(
            float(tp.sum() / total_pixels * 100), 4
        ),
        "evaluated_pixels": total_pixels,
        "per_class_iou": {
            CLASS_NAMES[i]: round(float(iou[i] * 100), 4)
            for i in range(NUM_CLASSES)
        },
    }


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_checkpoint(model_name, device):
    print(f"\nLoading checkpoint: {model_name}", flush=True)

    processor = SegformerImageProcessor.from_pretrained(model_name)
    processor.do_resize = False
    processor.do_pad = False

    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    if model.config.num_labels != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"got {model.config.num_labels}"
        )

    return model, processor


def predict(image, model, processor, device):
    original_width, original_height = image.size

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

    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)

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

    return np.asarray(mask, dtype=np.uint8)


def evaluate(model_name, images, device):
    model, processor = load_checkpoint(model_name, device)

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )
    inference_times = []

    started = time.perf_counter()

    try:
        for index, (image_path, target) in enumerate(images, start=1):
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            synchronize(device)
            inference_start = time.perf_counter()

            prediction = predict(image, model, processor, device)

            synchronize(device)
            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            if prediction.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {image_path.name}: "
                    f"{prediction.shape} != {target.shape}"
                )

            update_confusion_matrix(confusion, prediction, target)
            inference_times.append(inference_ms)

            if index % 25 == 0 or index == len(images):
                print(
                    f"[{index}/{len(images)}] "
                    f"Last inference: {inference_ms:.1f} ms",
                    flush=True
                )

        total_time = time.perf_counter() - started
        metrics = calculate_metrics(confusion)

        return {
            "model": str(model_name),
            "images_evaluated": len(images),
            "max_input_side": MAX_INPUT_SIDE,
            "mean_inference_ms": round(
                float(np.mean(inference_times)), 2
            ),
            "median_inference_ms": round(
                float(np.median(inference_times)), 2
            ),
            "total_time_minutes": round(total_time / 60, 2),
            **metrics,
        }

    finally:
        del model
        del processor

        if device.type == "mps":
            torch.mps.empty_cache()


def main():
    if not FINETUNED_MODEL.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {FINETUNED_MODEL}"
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("=== SEGFORMER B2 CHECKPOINT COMPARISON ===")
    print("Device:", device)
    print("Maximum input side:", MAX_INPUT_SIDE)

    image_paths = sorted(IMAGE_DIR.rglob("*_leftImg8bit.png"))

    if len(image_paths) < 500:
        raise RuntimeError(
            f"Expected at least 500 validation images, "
            f"found {len(image_paths)}"
        )

    image_paths = image_paths[:500]
    images = []

    print("\nPreparing Cityscapes validation data...", flush=True)

    for image_path in image_paths:
        relative = image_path.relative_to(IMAGE_DIR)

        mask_name = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelIds.png"
        )
        mask_path = MASK_DIR / relative.parent / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        with Image.open(mask_path) as mask:
            target = convert_ground_truth(mask)

        images.append((image_path, target))

    print("Validation images:", len(images))

    results = {}

    print("\n===== BASELINE B2 =====")
    results["baseline_b2"] = evaluate(
        BASELINE_MODEL,
        images,
        device
    )

    print("\n===== FINE-TUNED B2 =====")
    results["finetuned_b2"] = evaluate(
        str(FINETUNED_MODEL),
        images,
        device
    )

    baseline = results["baseline_b2"]
    finetuned = results["finetuned_b2"]

    results["comparison"] = {
        "mIoU_difference_percentage_points": round(
            finetuned["mIoU_percent"] - baseline["mIoU_percent"], 4
        ),
        "pixel_accuracy_difference_percentage_points": round(
            finetuned["pixel_accuracy_percent"]
            - baseline["pixel_accuracy_percent"], 4
        ),
        "inference_time_difference_ms": round(
            finetuned["mean_inference_ms"]
            - baseline["mean_inference_ms"], 2
        ),
    }

    results["dataset"] = "Cityscapes"
    results["split"] = "val"
    results["evaluation_images"] = len(images)
    results["device"] = str(device)
    results["preprocessing"] = (
        "Aspect-preserving resize, maximum side 1920; "
        "bilinear image interpolation; nearest-neighbor mask resize"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\n===== FINAL COMPARISON =====")
    print(
        f"Baseline B2: "
        f"mIoU {baseline['mIoU_percent']}% | "
        f"Pixel Accuracy {baseline['pixel_accuracy_percent']}%"
    )
    print(
        f"Fine-tuned B2: "
        f"mIoU {finetuned['mIoU_percent']}% | "
        f"Pixel Accuracy {finetuned['pixel_accuracy_percent']}%"
    )
    print(
        "mIoU difference:",
        results["comparison"]["mIoU_difference_percentage_points"],
        "percentage points"
    )
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
