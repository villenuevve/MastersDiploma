import numpy as np


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


def evaluate_masks(
    prediction,
    target,
    num_classes=19,
    ignore_index=255,
):
    """
    Calculate semantic segmentation metrics.

    prediction: predicted class IDs, shape (H, W)
    target: ground-truth class IDs, shape (H, W)
    Valid class IDs: 0..num_classes-1
    Ignore label: 255
    """

    prediction = np.asarray(prediction)
    target = np.asarray(target)

    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("Prediction and target must be 2D class-ID masks.")

    if prediction.shape != target.shape:
        raise ValueError(
            f"Mask dimensions differ: prediction={prediction.shape}, "
            f"target={target.shape}"
        )

    valid = (
        (target != ignore_index)
        & (target >= 0)
        & (target < num_classes)
        & (prediction >= 0)
        & (prediction < num_classes)
    )

    if not np.any(valid):
        raise ValueError("No valid pixels available for evaluation.")

    pred = prediction[valid].astype(np.int64)
    gt = target[valid].astype(np.int64)

    confusion = np.bincount(
        num_classes * gt + pred,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth_count = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)

    union = ground_truth_count + predicted_count - true_positive

    iou = np.divide(
        true_positive,
        union,
        out=np.full(num_classes, np.nan),
        where=union != 0,
    )

    present_classes = union > 0
    mean_iou = float(np.mean(iou[present_classes]))

    pixel_accuracy = float(
        true_positive.sum() / confusion.sum()
    )

    per_class = {}
    for class_id, class_iou in enumerate(iou):
        name = (
            CITYSCAPES_CLASSES[class_id]
            if class_id < len(CITYSCAPES_CLASSES)
            else f"class_{class_id}"
        )

        per_class[name] = (
            None if np.isnan(class_iou) else round(float(class_iou), 4)
        )

    return {
        "mean_iou": round(mean_iou, 4),
        "pixel_accuracy": round(pixel_accuracy, 4),
        "evaluated_pixels": int(confusion.sum()),
        "per_class_iou": per_class,
        "confusion_matrix": confusion.tolist(),
    }
