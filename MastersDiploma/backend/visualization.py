
from PIL import Image
import numpy as np

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


def save_class_ids(mask: np.ndarray, path: str):
    """Save the class ID of every pixel for interactive inspection."""
    class_ids = np.asarray(mask, dtype=np.uint8)
    Image.fromarray(class_ids, mode="L").save(path)


def save_mask(mask: np.ndarray, path: str):
    """Save a colorized semantic segmentation mask."""
    colored = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8
    )

    for idx, color in enumerate(CITYSCAPES_COLORS):
        colored[mask == idx] = color

    Image.fromarray(colored).save(path)


def create_overlay(
    image: Image.Image,
    colored_mask: np.ndarray,
    alpha: float = 0.55
):
    """Blend the original image with the segmentation mask."""
    image_np = np.array(image.convert("RGB"), dtype=np.uint8)

    if image_np.shape != colored_mask.shape:
        raise ValueError(
            "The original image and segmentation mask must "
            "have the same dimensions."
        )

    overlay = (
        image_np.astype(np.float32) * (1 - alpha)
        + colored_mask.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
