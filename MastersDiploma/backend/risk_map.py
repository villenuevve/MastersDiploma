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

# Classes treated as potential navigation obstacles.
POSITION_RISK_CLASSES = {
    5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18
}

# Additional relative risk based on vertical image position.
POSITION_RISK_TOP = 0
POSITION_RISK_MIDDLE = 5
POSITION_RISK_BOTTOM = 10


def analyze_risk(class_mask, confidence_map=None):
    """
    Build a pixel-wise relative navigation risk map.

    Risk combines:
    - semantic class risk;
    - model uncertainty;
    - vertical image position for potential obstacles.

    Position is an image-space heuristic, not metric distance.
    Output values are in the range 0-100.
    """
    mask = np.asarray(class_mask)

    if mask.ndim != 2:
        raise ValueError("class_mask must be a 2D array.")

    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("class_mask must contain integer class IDs.")

    height, width = mask.shape

    if height == 0 or width == 0:
        raise ValueError("class_mask must not be empty.")

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

    # Vectorized lookup for Cityscapes class IDs.
    # Unknown, negative and out-of-range IDs retain risk 100.
    risk_lut = np.full(19, 100.0, dtype=np.float32)
    for class_id, risk_value in CLASS_RISK.items():
        risk_lut[class_id] = risk_value

    valid_ids = (mask >= 0) & (mask < 19)
    base_risk = np.full(mask.shape, 100.0, dtype=np.float32)
    base_risk[valid_ids] = risk_lut[mask[valid_ids]]


    # Low model confidence adds up to 20 risk points.
    uncertainty_risk = (1.0 - confidence) * 20.0

    # Position risk: lower image regions receive a larger addition.
    row_positions = np.arange(height, dtype=np.float32)[:, np.newaxis]

    position_by_row = np.where(
        row_positions < height / 3.0,
        POSITION_RISK_TOP,
        np.where(
            row_positions < 2.0 * height / 3.0,
            POSITION_RISK_MIDDLE,
            POSITION_RISK_BOTTOM,
        ),
    )

    position_lut = np.zeros(19, dtype=bool)
    position_lut[list(POSITION_RISK_CLASSES)] = True

    potential_obstacles = np.zeros(mask.shape, dtype=bool)
    potential_obstacles[valid_ids] = position_lut[mask[valid_ids]]

    position_risk = np.where(
        potential_obstacles,
        position_by_row,
        0.0
    ).astype(np.float32, copy=False)

    # Combine all risk components and keep the result in [0, 100].
    risk_map = np.clip(
        base_risk + uncertainty_risk + position_risk,
        0.0,
        100.0
    )

    risk_map = np.rint(risk_map).astype(np.uint8)

    total_pixels = int(risk_map.size)

    low_mask = risk_map < 30
    medium_mask = (risk_map >= 30) & (risk_map < 60)
    high_mask = risk_map >= 60

    unique_classes, inverse, counts = np.unique(
        mask,
        return_inverse=True,
        return_counts=True
    )

    risk_sums = np.bincount(
        inverse.ravel(),
        weights=risk_map.ravel()
    )

    class_statistics = []

    for index, class_id in enumerate(unique_classes):
        pixel_count = int(counts[index])

        class_statistics.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES.get(int(class_id), "unknown"),
            "pixel_count": pixel_count,
            "percentage": round(
                pixel_count / total_pixels * 100.0,
                2
            ),
            "mean_risk": round(
                float(risk_sums[index] / pixel_count),
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
