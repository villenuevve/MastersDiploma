import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ROOT = Path("data/evaluation/cityscapes_full")
IMAGE_DIR = ROOT / "leftImg8bit" / "val"
MASK_DIR = ROOT / "gtFine" / "val"
OUTPUT = Path("results/segformer_b2_b3_comparison_1600.json")

MODELS = {
    "B2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "B3": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
}
MAX_SIDE = 1600
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


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def convert_mask(mask):
    raw = np.asarray(mask.convert("L"), dtype=np.uint8)
    target = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    for label_id, train_id in LABEL_MAPPING.items():
        target[raw == label_id] = train_id
    return target


def update_confusion(confusion, prediction, target):
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
        encoded, minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(confusion):
    tp = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - tp
    iou = np.divide(
        tp, union, out=np.zeros(NUM_CLASSES, dtype=float), where=union > 0
    )
    present = union > 0
    return {
        "mIoU_percent": round(float(iou[present].mean() * 100), 4),
        "pixel_accuracy_percent": round(
            float(tp.sum() / confusion.sum() * 100), 4
        ),
        "evaluated_pixels": int(confusion.sum()),
        "per_class_iou_percent": {
            CLASS_NAMES[i]: round(float(iou[i] * 100), 4)
            for i in range(NUM_CLASSES)
        },
    }


def predict(image, model, processor, device):
    original_width, original_height = image.size
    scale = min(1.0, MAX_SIDE / max(original_width, original_height))
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))

    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    inputs = processor(images=resized, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(
            outputs.logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    restored = Image.fromarray(prediction).resize(
        (original_width, original_height), Image.Resampling.NEAREST
    )
    return np.asarray(restored, dtype=np.uint8)


def evaluate(model_name, images, device):
    print(f"\nLoading {model_name}...", flush=True)
    processor = SegformerImageProcessor.from_pretrained(model_name, local_files_only=True)
    processor.do_resize = False
    processor.do_pad = False

    model = SegformerForSemanticSegmentation.from_pretrained(model_name, local_files_only=True)
    if model.config.num_labels != NUM_CLASSES:
        raise ValueError(f"Expected 19 classes, got {model.config.num_labels}")
    model.to(device)
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    times = []
    started = time.perf_counter()

    try:
        for index, (image_path, image, target) in enumerate(images, 1):
            sync(device)
            t0 = time.perf_counter()
            prediction = predict(image, model, processor, device)
            sync(device)
            elapsed = (time.perf_counter() - t0) * 1000

            if prediction.shape != target.shape:
                raise ValueError(f"Shape mismatch: {image_path.name}")

            update_confusion(confusion, prediction, target)
            times.append(elapsed)

            if index % 25 == 0 or index == len(images):
                print(
                    f"[{index}/{len(images)}] {elapsed:.1f} ms",
                    flush=True,
                )

        return {
            "model": model_name,
            "images_evaluated": len(images),
            "max_input_side": MAX_SIDE,
            "mean_inference_ms": round(float(np.mean(times)), 2),
            "median_inference_ms": round(float(np.median(times)), 2),
            "total_time_minutes": round(
                (time.perf_counter() - started) / 60, 2
            ),
            **metrics(confusion),
        }
    finally:
        del model, processor
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("=== SEGFORMER B2 vs B3 ===")
    print("Device:", device)
    print("Maximum input side:", MAX_SIDE)

    paths = sorted(IMAGE_DIR.rglob("*_leftImg8bit.png"))[:500]
    if len(paths) != 500:
        raise RuntimeError(f"Expected 500 images, found {len(paths)}")

    images = []
    print("Preparing shared validation set...", flush=True)

    for path in paths:
        relative = path.relative_to(IMAGE_DIR)
        mask_name = path.name.replace(
            "_leftImg8bit.png", "_gtFine_labelIds.png"
        )
        mask_path = MASK_DIR / relative.parent / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        with Image.open(path) as img:
            image = img.convert("RGB").copy()
        with Image.open(mask_path) as mask:
            target = convert_mask(mask)

        images.append((path, image, target))

    print("Validation images:", len(images))

    results = {
        "dataset": "Cityscapes",
        "split": "val",
        "evaluation_images": len(images),
        "device": str(device),
        "max_input_side": MAX_SIDE,
        "status": "in_progress",
    }

    for label, model_name in MODELS.items():
        print(f"\n===== SEGFORMER {label} =====")
        results[label] = evaluate(model_name, images, device)

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Intermediate results saved: {OUTPUT}", flush=True)

    b2, b3 = results["B2"], results["B3"]
    results["comparison_B3_minus_B2"] = {
        "mIoU_difference_percentage_points": round(
            b3["mIoU_percent"] - b2["mIoU_percent"], 4
        ),
        "pixel_accuracy_difference_percentage_points": round(
            b3["pixel_accuracy_percent"] - b2["pixel_accuracy_percent"], 4
        ),
        "mean_inference_time_difference_ms": round(
            b3["mean_inference_ms"] - b2["mean_inference_ms"], 2
        ),
    }
    results["status"] = "completed"
    results["dataset"] = "Cityscapes"
    results["split"] = "val"
    results["evaluation_images"] = len(images)
    results["device"] = str(device)
    results["preprocessing"] = (
        "Identical aspect-preserving resize, maximum side 1600; "
        "bilinear image resize; processor resize/pad disabled; "
        "bilinear logit interpolation; nearest-neighbor mask restoration"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n===== FINAL COMPARISON =====")
    for label in ("B2", "B3"):
        item = results[label]
        print(
            f"{label}: mIoU {item['mIoU_percent']}% | "
            f"Pixel Accuracy {item['pixel_accuracy_percent']}% | "
            f"Mean inference {item['mean_inference_ms']} ms"
        )
    print("B3 − B2:", results["comparison_B3_minus_B2"])
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
