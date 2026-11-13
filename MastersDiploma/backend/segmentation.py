import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

MODEL_NAME = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"

# Limit the input size to avoid excessive memory consumption.
MAX_INPUT_SIDE = 1600

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Model and processor are loaded only when inference is requested.
processor = None
model = None


def load_model():
    global processor, model

    if model is not None and processor is not None:
        return

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    loaded_processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    loaded_processor.do_resize = False
    loaded_processor.do_pad = False

    loaded_model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME
    )
    loaded_model.to(DEVICE)
    loaded_model.eval()

    processor = loaded_processor
    model = loaded_model

    print("Model loaded successfully.")


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


def predict_mask(image: Image.Image, return_stats=False, return_confidence=False):
    load_model()

    image = image.convert("RGB")
    original_width, original_height = image.size

    tile_size = 1024
    overlap = 256
    stride = tile_size - overlap

    def get_positions(length):
        if length <= tile_size:
            return [0]

        positions = list(range(0, length - tile_size + 1, stride))
        last_position = length - tile_size

        if positions[-1] != last_position:
            positions.append(last_position)

        return positions

    x_positions = get_positions(original_width)
    y_positions = get_positions(original_height)

    total_tiles = len(x_positions) * len(y_positions)

    mask = np.zeros(
        (original_height, original_width),
        dtype=np.uint8,
    )

    best_score = np.full(
        (original_height, original_width),
        -1.0,
        dtype=np.float32,
    )

    confidence_map = np.zeros(
        (original_height, original_width),
        dtype=np.float32,
    )

    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for y0 in y_positions:
        for x0 in x_positions:
            x1 = min(x0 + tile_size, original_width)
            y1 = min(y0 + tile_size, original_height)

            tile = image.crop((x0, y0, x1, y1))
            tile_width, tile_height = tile.size

            inputs = processor(images=tile, return_tensors="pt")
            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(**inputs)

                logits = F.interpolate(
                    outputs.logits,
                    size=(tile_height, tile_width),
                    mode="bilinear",
                    align_corners=False,
                )

                probabilities = torch.softmax(logits, dim=1)
                tile_confidence = probabilities.max(dim=1).values[0]
                tile_mask = logits.argmax(dim=1)[0]

                tile_mask = tile_mask.cpu().numpy().astype(np.uint8)
                tile_confidence = (
                    tile_confidence.cpu().numpy().astype(np.float32)
                )

            # Reduce the influence of predictions near artificial tile edges.
            # Real image boundaries are not penalized.
            ramp = min(overlap // 2, tile_width // 4, tile_height // 4)

            weight_x = np.ones(tile_width, dtype=np.float32)
            weight_y = np.ones(tile_height, dtype=np.float32)

            if ramp > 0:
                ramp_values = np.linspace(
                    0.5, 1.0, ramp, dtype=np.float32
                )

                if x0 > 0:
                    weight_x[:ramp] = ramp_values

                if x1 < original_width:
                    weight_x[-ramp:] = ramp_values[::-1]

                if y0 > 0:
                    weight_y[:ramp] = ramp_values

                if y1 < original_height:
                    weight_y[-ramp:] = ramp_values[::-1]

            tile_weight = np.outer(weight_y, weight_x)
            tile_score = tile_confidence * tile_weight

            region_score = best_score[y0:y1, x0:x1]
            update = tile_score > region_score

            region_mask = mask[y0:y1, x0:x1]
            region_confidence = confidence_map[y0:y1, x0:x1]

            region_mask[update] = tile_mask[update]
            region_confidence[update] = tile_confidence[update]
            region_score[update] = tile_score[update]

            del inputs, outputs, logits, probabilities
            del tile_mask, tile_confidence, tile

    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not return_stats and not return_confidence:
        return mask

    total_pixels = int(mask.size)
    detected_classes = []

    for class_id in np.unique(mask):
        class_id = int(class_id)
        pixel_count = int(np.count_nonzero(mask == class_id))

        if 0 <= class_id < len(CITYSCAPES_CLASSES):
            detected_classes.append({
                "id": class_id,
                "name": CITYSCAPES_CLASSES[class_id],
                "pixel_count": pixel_count,
                "percentage": round(
                    100 * pixel_count / total_pixels, 2
                ),
            })

    stats = {
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "inference_time_ms": round(elapsed_ms, 2),
        "image_width": original_width,
        "image_height": original_height,
        "inference_width": min(tile_size, original_width),
        "inference_height": min(tile_size, original_height),
        "inference_mode": "tiled",
        "tile_size": tile_size,
        "tile_overlap": overlap,
        "tile_count": total_tiles,
        "detected_classes": detected_classes,
    }

    if return_confidence:
        return mask, stats, np.clip(confidence_map, 0.0, 1.0)

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


def create_traversability_map(
    mask: np.ndarray,
    confidence_map: np.ndarray = None,
    obstacle_radius: int = 12
):
    """
    Estimate traversability using semantic classes, model confidence
    and obstacle inflation.

    Values:
        255 - high traversability
        150 - medium traversability
         70 - low traversability
          0 - obstacle or unknown surface

    Areas near obstacles receive an additional risk penalty.
    """

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Segmentation mask must be a 2D array.")

    if obstacle_radius < 1:
        raise ValueError("obstacle_radius must be at least 1.")

    class_weights = {
        0: 255,  # road
        1: 220,  # sidewalk
        8: 70,   # vegetation
        9: 150,  # terrain
    }

    traversability = np.zeros(mask.shape, dtype=np.float32)

    for class_id, weight in class_weights.items():
        traversability[mask == class_id] = weight

    if confidence_map is not None:
        confidence_map = np.asarray(
            confidence_map,
            dtype=np.float32
        )

        if confidence_map.shape != mask.shape:
            raise ValueError(
                "Confidence map and segmentation mask "
                "must have the same dimensions."
            )

        confidence_map = np.nan_to_num(
            confidence_map,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )
        confidence_map = np.clip(confidence_map, 0.0, 1.0)

        traversability *= confidence_map

    # Physical and dynamic obstacles in Cityscapes.
    obstacle_classes = [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
    obstacle_mask = np.isin(mask, obstacle_classes)

    if np.any(obstacle_mask):
        # Distance from every free pixel to the nearest obstacle.
        free_mask = (~obstacle_mask).astype(np.uint8)

        distances = cv2.distanceTransform(
            free_mask,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE
        )

        # Risk gradually decreases with distance from obstacles.
        near_obstacles = (
            (distances > 0)
            & (distances < obstacle_radius)
            & (traversability > 0)
        )

        risk_factor = (
            0.35
            + 0.65 * distances / obstacle_radius
        )

        traversability[near_obstacles] *= (
            risk_factor[near_obstacles]
        )

    # Obstacles must always remain non-traversable.
    traversability[obstacle_mask] = 0

    return np.clip(
        np.rint(traversability),
        0,
        255
    ).astype(np.uint8)


def colorize_traversability_map(traversability: np.ndarray):
    """
    Convert traversability values into a color visualization:
    green - high, yellow - medium, orange - low, red - not recommended.
    """
    colored = np.zeros(
        (*traversability.shape, 3),
        dtype=np.uint8,
    )

    colored[traversability == 0] = [220, 65, 65]
    colored[
        (traversability > 0) & (traversability < 100)
    ] = [240, 145, 55]
    colored[
        (traversability >= 100) & (traversability < 200)
    ] = [235, 205, 65]
    colored[traversability >= 200] = [65, 180, 110]

    return colored
