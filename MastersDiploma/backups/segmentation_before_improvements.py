from PIL import Image
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()


# Cityscapes classes
CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)


def predict_mask(image: Image.Image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    mask = (
        upsampled
        .argmax(dim=1)[0]
        .cpu()
        .numpy()
    )

    return mask


def colorize_mask(mask: np.ndarray):

    colored = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8
    )

    for idx, color in enumerate(CITYSCAPES_COLORS):
        colored[mask == idx] = color

    return colored


def get_detected_classes(mask: np.ndarray):

    unique_ids = sorted(
        set(mask.flatten().tolist())
    )

    detected = []

    for class_id in unique_ids:

        if 0 <= class_id < len(CITYSCAPES_CLASSES):

            detected.append({
                "id": class_id,
                "name": CITYSCAPES_CLASSES[class_id]
            })

    return detected