from pathlib import Path
import json

import numpy as np
from PIL import Image

from backend.metrics import evaluate_masks
from backend.segmentation import predict_mask

ROOT = Path("data/evaluation/cityscapes8")
OUTPUT_DIR = Path("results")

LABEL_ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
IGNORE_INDEX = 255


def convert_ground_truth(mask):
    converted = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        converted[mask == label_id] = train_id
    return converted


def evaluate_split(split):
    image_dir = ROOT / "images" / split
    mask_dir = ROOT / "masks" / split
    image_paths = sorted(image_dir.glob("*.png"))

    predictions = []
    targets = []
    evaluated_files = []

    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            print(f"Skipping {image_path.name}: mask missing")
            continue

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            prediction = np.asarray(predict_mask(image))

        with Image.open(mask_path) as im:
            target = convert_ground_truth(np.asarray(im.convert("L")))

        if prediction.shape != target.shape:
            raise ValueError(
                f"Size mismatch for {image_path.name}: "
                f"{prediction.shape} vs {target.shape}"
            )

        if not np.any(target != IGNORE_INDEX):
            print(f"Skipping {image_path.name}: no valid labels")
            continue

        predictions.append(prediction.astype(np.uint8))
        targets.append(target)
        evaluated_files.append(image_path.name)
        print(f"[{split}] {image_path.name}: {image.width}x{image.height}")

    if not predictions:
        raise RuntimeError(f"No valid image-mask pairs in {split}")

    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    metrics = evaluate_masks(
        pred, target, num_classes=19, ignore_index=IGNORE_INDEX
    )
    metrics["split"] = split
    metrics["images_evaluated"] = len(evaluated_files)
    metrics["files"] = evaluated_files
    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for split in ("train", "val"):
        results[split] = evaluate_split(split)

    all_files = sorted(
        (ROOT / "images" / split / name)
        for split in ("train", "val")
        for name in results[split]["files"]
    )

    predictions = []
    targets = []
    for image_path in all_files:
        split = image_path.parent.name
        mask_path = ROOT / "masks" / split / image_path.name

        with Image.open(image_path) as im:
            prediction = np.asarray(predict_mask(im.convert("RGB"))).astype(np.uint8)
        with Image.open(mask_path) as im:
            target = convert_ground_truth(np.asarray(im.convert("L")))

        predictions.append(prediction)
        targets.append(target)

    metrics = evaluate_masks(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        num_classes=19,
        ignore_index=IGNORE_INDEX,
    )
    metrics["images_evaluated"] = len(all_files)
    metrics["files"] = [str(p.relative_to(ROOT)) for p in all_files]
    results["all"] = metrics

    (OUTPUT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(results["val"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "evaluation_metrics_all.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for split, result in results.items():
        print(f"\n=== {split.upper()} ===")
        print(f"Images: {result['images_evaluated']}")
        print(f"mIoU: {result['mean_iou'] * 100:.2f}%")
        print(f"Pixel Accuracy: {result['pixel_accuracy'] * 100:.2f}%")
        print(f"Evaluated pixels: {result['evaluated_pixels']:,}")

    print("\nValidation metrics saved to results/evaluation_metrics.json")
    print("All split metrics saved to results/evaluation_metrics_all.json")


if __name__ == "__main__":
    main()
