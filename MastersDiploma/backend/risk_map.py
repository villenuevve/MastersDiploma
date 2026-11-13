import numpy as np


# Relative navigation risk for Cityscapes train IDs.
# Values range from 0 (lowest risk) to 100 (highest risk).
CLASS_RISK = {
    0: 5,    # road
    1: 15,   # sidewalk
    2: 95,   # building
    3: 90,   # wall
    4: 90,   # fence
    5: 80,   # pole
    6: 70,   # traffic light
    7: 60,   # traffic sign
    8: 40,   # vegetation
    9: 30,   # terrain
    10: 100, # sky
    11: 75,  # person
    12: 85,  # rider
    13: 95,  # car
    14: 95,  # truck
    15: 95,  # bus
    16: 95,  # train
    17: 90,  # motorcycle
    18: 85,  # bicycle
}

CLASS_NAMES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


def analyze_risk(class_mask, confidence_map=None):
    """
    Build a pixel-wise relative navigation risk map.

    Risk is based on semantic class and model uncertainty.
    Output values are in the range 0-100.
    """
    mask = np.asarray(class_mask)

    if mask.ndim != 2:
        raise ValueError("class_mask must be a 2D array.")

    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("class_mask must contain integer class IDs.")

    height, width = mask.shape

    if confidence_map is not None:
        confidence = np.asarray(confidence_map, dtype=np.float32)

        if confidence.shape != mask.shape:
            raise ValueError(
                "confidence_map dimensions must match class_mask."
            )

        if not np.all(np.isfinite(confidence)):
            raise ValueError("confidence_map contains invalid values.")

        if confidence.max(initial=0) > 1.0:
            confidence = confidence / 100.0

        confidence = np.clip(confidence, 0.0, 1.0)
    else:
        confidence = np.ones(mask.shape, dtype=np.float32)

    base_risk = np.zeros(mask.shape, dtype=np.float32)

    for class_id, risk_value in CLASS_RISK.items():
        base_risk[mask == class_id] = risk_value

    # Unknown class IDs are treated conservatively.
    known_classes = np.isin(mask, list(CLASS_RISK.keys()))
    base_risk[~known_classes] = 100.0

    # Low model confidence adds up to 20 risk points.
    uncertainty_risk = (1.0 - confidence) * 20.0

    risk_map = np.clip(
        base_risk + uncertainty_risk,
        0.0,
        100.0
    )

    risk_map = np.rint(risk_map).astype(np.uint8)

    total_pixels = int(risk_map.size)

    low_mask = risk_map < 30
    medium_mask = (risk_map >= 30) & (risk_map < 60)
    high_mask = risk_map >= 60

    class_statistics = []

    for class_id in np.unique(mask):
        class_pixels = mask == class_id
        pixel_count = int(np.count_nonzero(class_pixels))

        class_statistics.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES.get(
                int(class_id),
                "unknown"
            ),
            "pixel_count": pixel_count,
            "percentage": round(
                pixel_count / total_pixels * 100.0,
                2
            ),
            "mean_risk": round(
                float(np.mean(risk_map[class_pixels])),
                2
            ),
        })

    class_statistics.sort(
        key=lambda item: item["mean_risk"],
        reverse=True
    )

    return {
        "risk_map": risk_map,
        "statistics": {
            "width": int(width),
            "height": int(height),
            "mean_risk": round(float(np.mean(risk_map)), 2),
            "max_risk": int(np.max(risk_map)),
            "low_risk_percentage": round(
                float(np.mean(low_mask) * 100.0), 2
            ),
            "medium_risk_percentage": round(
                float(np.mean(medium_mask) * 100.0), 2
            ),
            "high_risk_percentage": round(
                float(np.mean(high_mask) * 100.0), 2
            ),
            "class_statistics": class_statistics,
        },
    }
