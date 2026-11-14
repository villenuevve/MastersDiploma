import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    DepthAnythingForDepthEstimation,
)


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

processor = None
model = None
_model_lock = threading.Lock()


def load_depth_model():
    """Load Depth Anything V2 lazily."""
    global processor, model

    if processor is not None and model is not None:
        return

    with _model_lock:
        if processor is not None and model is not None:
            return

        loaded_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        loaded_model = DepthAnythingForDepthEstimation.from_pretrained(
            MODEL_ID
        )

        loaded_model.to(DEVICE)
        loaded_model.eval()

        processor = loaded_processor
        model = loaded_model


def predict_depth(image, return_stats=False):
    """
    Predict relative depth from a PIL image.

    Returns:
        depth_map: float32 array with the original image dimensions.
        stats: optional inference statistics.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    if not hasattr(image, "convert"):
        raise TypeError("Input image must be a PIL image.")

    image = image.convert("RGB")
    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Input image has invalid dimensions.")

    load_depth_model()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

        # Restore the prediction to the original image dimensions.
        predicted_depth = F.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    depth_map = (
        predicted_depth.squeeze(0)
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if not np.isfinite(depth_map).all():
        raise RuntimeError("Depth model produced NaN or infinite values.")

    if return_stats:
        stats = {
            "model": MODEL_ID,
            "device": str(DEVICE),
            "inference_time_ms": round(inference_time_ms, 2),
            "width": original_width,
            "height": original_height,
            "min": float(np.min(depth_map)),
            "max": float(np.max(depth_map)),
            "mean": float(np.mean(depth_map)),
            "p05": float(np.percentile(depth_map, 5)),
            "p95": float(np.percentile(depth_map, 95)),
            "depth_type": "relative",
        }
        return depth_map, stats

    return depth_map


def normalize_depth_for_visualization(depth_map):
    """
    Convert relative depth into an 8-bit grayscale visualization.

    This normalization is only for display. It does not convert
    relative depth into metric distance.
    """
    depth_map = np.asarray(depth_map, dtype=np.float32)

    if depth_map.ndim != 2:
        raise ValueError("Depth map must be a 2D array.")

    if not np.isfinite(depth_map).all():
        raise ValueError("Depth map contains non-finite values.")

    low = float(np.percentile(depth_map, 2))
    high = float(np.percentile(depth_map, 98))

    if high <= low:
        return np.zeros(depth_map.shape, dtype=np.uint8)

    normalized = (depth_map - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    return (normalized * 255).astype(np.uint8)
