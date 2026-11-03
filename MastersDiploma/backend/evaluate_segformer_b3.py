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

ROOT = Path("data/evaluation/cityscapes_full")
IMAGE_DIR = ROOT / "leftImg8bit" / "val"
MASK_DIR = ROOT / "gtFine" / "val"

MODEL_NAME = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
OUTPUT_PATH = Path("results/cityscapes_b3_1600_metrics.json")

MAX_INPUT_SIDE = 1600
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
    converted = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)

    for label_id, train_id in LABEL_MAPPING.items():
        converted[raw == label_id] = train_id

    return converted


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
        "mIoU_percent": round(float(iou[present].mean() * 100), 2),
        "pixel_accuracy_percent": round(
            float(tp.sum() / total_pixels * 100), 2
        ),
        "per_class": {
            CLASS_NAMES[i]: round(float(iou[i] * 100), 2)
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": confusion.tolist(),
    }


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

    with torch.no_grad():
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Number of validation images; use 1 for memory test"
    )
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("=== SEGFORMER-B3 EVALUATION ===")
    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Maximum input side:", MAX_INPUT_SIDE)
    print("Images requested:", args.max_images)

    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    processor.do_resize = False
    processor.do_pad = False

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    images = sorted(IMAGE_DIR.rglob("*_leftImg8bit.png"))

    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    if args.max_images < 1 or args.max_images > len(images):
        raise ValueError("max-images must be between 1 and 500")

    images = images[:args.max_images]

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    inference_times = []
    started = time.perf_counter()

    for index, image_path in enumerate(images, start=1):
        relative = image_path.relative_to(IMAGE_DIR)

        mask_name = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelIds.png"
        )
        mask_path = MASK_DIR / relative.parent / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as mask:
            target = convert_ground_truth(mask)

        inference_start = time.perf_counter()
        prediction = predict(image, model, processor, device)
        inference_times.append(
            (time.perf_counter() - inference_start) * 1000
        )

        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {image_path.name}")

        update_confusion_matrix(confusion, prediction, target)

        print(
            f"[{index}/{len(images)}] "
            f"Inference: {inference_times[-1]:.1f} ms",
            flush=True
        )

    total_time = time.perf_counter() - started
    metrics = calculate_metrics(confusion)

    print("\n=== RESULTS ===")
    print("Images:", len(images))
    print("mIoU:", metrics["mIoU_percent"], "%")
    print("Pixel Accuracy:", metrics["pixel_accuracy_percent"], "%")
    print(
        "Mean inference:",
        round(float(np.mean(inference_times)), 2),
        "ms"
    )
    print("Elapsed:", round(total_time / 60, 2), "min")

    if len(images) == 500:
        result = {
            "model": MODEL_NAME,
            "dataset": "Cityscapes",
            "split": "val",
            "images_evaluated": len(images),
            "max_input_side": MAX_INPUT_SIDE,
            "mean_inference_ms": round(
                float(np.mean(inference_times)), 2
            ),
            "total_time_minutes": round(total_time / 60, 2),
            **metrics,
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("\n=== PER-CLASS IoU ===")
        for name, value in metrics["per_class"].items():
            print(f"{name:15s}: {value:6.2f}%")

        print("\nSaved:", OUTPUT_PATH)
    else:
        print("\nSingle-image test only. No full evaluation JSON saved.")


if __name__ == "__main__":
    main()
