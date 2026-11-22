import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.segmentation import predict_mask, load_model, CITYSCAPES_CLASSES

DATA = ROOT / "data/evaluation/cityscapes_full"
OUTPUT = ROOT / "data/evaluation/results"

ID_MAPPING = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}

NUM_CLASSES = 19
IGNORE = 255

def convert_gt(label_ids):
    result = np.full(label_ids.shape, IGNORE, dtype=np.uint8)
    for source_id, train_id in ID_MAPPING.items():
        result[label_ids == source_id] = train_id
    return result

def main():
    image_root = DATA / "leftImg8bit/val"
    mask_root = DATA / "gtFine/val"

    images = {
        p.name.replace("_leftImg8bit.png", ""): p
        for p in image_root.rglob("*_leftImg8bit.png")
    }
    masks = {
        p.name.replace("_gtFine_labelIds.png", ""): p
        for p in mask_root.rglob("*_gtFine_labelIds.png")
    }

    common = sorted(images.keys() & masks.keys())
    if len(common) != 500:
        raise RuntimeError(
            f"Expected 500 matched pairs, found {len(common)}"
        )

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_time = 0.0

    print("Loading SegFormer-B3...")
    load_model()
    print(f"Starting evaluation: {len(common)} images")

    for index, key in enumerate(common, 1):
        with Image.open(images[key]) as image:
            image = image.convert("RGB")
            prediction = predict_mask(image)

        with Image.open(masks[key]) as gt_image:
            gt = convert_gt(np.array(gt_image, dtype=np.uint8))

        if prediction.shape != gt.shape:
            raise ValueError(
                f"Dimension mismatch: {key}: "
                f"prediction={prediction.shape}, GT={gt.shape}"
            )

        valid = (gt != IGNORE) & (prediction < NUM_CLASSES)
        encoded = (
            gt[valid].astype(np.int64) * NUM_CLASSES
            + prediction[valid].astype(np.int64)
        )
        confusion += np.bincount(
            encoded, minlength=NUM_CLASSES ** 2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        total_time += 1

        if index % 10 == 0 or index == len(common):
            print(f"Processed {index}/{len(common)}")

    true_positive = np.diag(confusion)
    ground_truth_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    union = ground_truth_count + predicted_count - true_positive

    iou = np.divide(
        true_positive,
        union,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=union > 0,
    )

    pixel_accuracy = (
        true_positive.sum() / confusion.sum()
        if confusion.sum() else 0.0
    )
    mean_iou = float(np.nanmean(iou))

    results = {
        "model": "SegFormer-B3",
        "dataset": "Cityscapes",
        "split": "validation",
        "images_evaluated": len(common),
        "pixel_accuracy": round(float(pixel_accuracy), 6),
        "mean_iou": round(mean_iou, 6),
        "per_class_iou": {
            name: (
                round(float(iou[i]), 6)
                if np.isfinite(iou[i]) else None
            )
            for i, name in enumerate(CITYSCAPES_CLASSES)
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT / "cityscapes_val_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savetxt(
        OUTPUT / "cityscapes_val_confusion_matrix.csv",
        confusion,
        delimiter=",",
        fmt="%d",
    )

    print("\n===== EVALUATION RESULTS =====")
    print(f"Images: {len(common)}")
    print(f"Pixel Accuracy: {pixel_accuracy * 100:.2f}%")
    print(f"Mean IoU: {mean_iou * 100:.2f}%")
    print("\nPer-class IoU:")

    for name, value in results["per_class_iou"].items():
        display = f"{value * 100:.2f}%" if value is not None else "N/A"
        print(f"{name}: {display}")

    print("\nSaved:")
    print(OUTPUT / "cityscapes_val_metrics.json")
    print(OUTPUT / "cityscapes_val_confusion_matrix.csv")

if __name__ == "__main__":
    main()
