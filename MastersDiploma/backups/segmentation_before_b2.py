import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

CITYSCAPES_COLORS = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
], dtype=np.uint8)


def predict_mask(image: Image.Image, return_stats=False):
    """
    Predict Cityscapes semantic classes for an RGB image.

    Returns a class-ID mask with the original image dimensions.
    If return_stats=True, also returns inference statistics.
    """
    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

        logits = F.interpolate(
            outputs.logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )

        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    elapsed_ms = (time.perf_counter() - start) * 1000

    if not return_stats:
        return mask

    total_pixels = int(mask.size)
    detected_classes = []

    for class_id in np.unique(mask):
        pixel_count = int(np.count_nonzero(mask == class_id))

        if 0 <= class_id < len(CITYSCAPES_CLASSES):
            detected_classes.append({
                "id": int(class_id),
                "name": CITYSCAPES_CLASSES[int(class_id)],
                "pixel_count": pixel_count,
                "percentage": round(
                    100 * pixel_count / total_pixels, 2
                ),
            })

    stats = {
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "inference_time_ms": round(elapsed_ms, 2),
        "image_width": image.width,
        "image_height": image.height,
        "detected_classes": detected_classes,
    }

    return mask, stats


def colorize_mask(mask: np.ndarray):
    colored = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8,
    )

    valid = (mask >= 0) & (mask < len(CITYSCAPES_COLORS))
    colored[valid] = CITYSCAPES_COLORS[mask[valid]]

    return colored


def get_detected_classes(mask: np.ndarray):
    detected = []

    for class_id in np.unique(mask):
        class_id = int(class_id)

        if 0 <= class_id < len(CITYSCAPES_CLASSES):
            detected.append({
                "id": class_id,
                "name": CITYSCAPES_CLASSES[class_id],
            })

    return detected
