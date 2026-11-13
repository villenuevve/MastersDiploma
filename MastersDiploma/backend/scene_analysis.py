import cv2
import numpy as np


CITYSCAPES_CLASSES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall",
    4: "fence", 5: "pole", 6: "traffic light",
    7: "traffic sign", 8: "vegetation", 9: "terrain",
    10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train",
    17: "motorcycle", 18: "bicycle",
}

OBJECT_CLASSES = {5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18}


def _horizontal_position(x, width):
    ratio = x / width
    if ratio < 1 / 3:
        return "left"
    if ratio < 2 / 3:
        return "center"
    return "right"


def _vertical_position(y, height):
    ratio = y / height
    if ratio < 1 / 3:
        return "top"
    if ratio < 2 / 3:
        return "middle"
    return "bottom"


def analyze_scene(class_mask, confidence_map=None,
                  min_component_area=20, max_objects=100):
    mask = np.asarray(class_mask)

    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("class_mask must be a 2D integer array")

    height, width = mask.shape
    total_pixels = height * width

    if total_pixels == 0:
        raise ValueError("class_mask cannot be empty")

    if confidence_map is not None:
        confidence = np.asarray(confidence_map, dtype=np.float32)
        if confidence.shape != mask.shape:
            raise ValueError("confidence_map dimensions must match class_mask")
        confidence = np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0)
        if confidence.max(initial=0) > 1:
            confidence = confidence / 100.0
        confidence = np.clip(confidence, 0, 1)
    else:
        confidence = None

    minimum_area = max(int(min_component_area), int(total_pixels * 0.0001))
    objects = []

    for class_id in OBJECT_CLASSES:
        binary = (mask == class_id).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        for component_id in range(1, count):
            x, y, w, h, area = stats[component_id]

            if area < minimum_area:
                continue

            center_x, center_y = centroids[component_id]
            component = labels == component_id

            mean_confidence = (
                float(confidence[component].mean())
                if confidence is not None else None
            )

            objects.append({
                "class_id": int(class_id),
                "class_name": CITYSCAPES_CLASSES[class_id],
                "area_pixels": int(area),
                "bbox": {
                    "x": int(x), "y": int(y),
                    "width": int(w), "height": int(h),
                },
                "center": {
                    "x": round(float(center_x), 2),
                    "y": round(float(center_y), 2),
                },
                "normalized_center": {
                    "x": round(float(center_x / width), 4),
                    "y": round(float(center_y / height), 4),
                },
                "horizontal_position": _horizontal_position(center_x, width),
                "vertical_position": _vertical_position(center_y, height),
                "position": (
                    f"{_vertical_position(center_y, height)}-"
                    f"{_horizontal_position(center_x, width)}"
                ),
                "mean_confidence": (
                    round(mean_confidence, 4)
                    if mean_confidence is not None else None
                ),
                "is_potential_obstacle": True,
            })

    objects.sort(key=lambda item: item["area_pixels"], reverse=True)
    objects = objects[:max_objects]

    class_distribution = {}
    for class_id, class_name in CITYSCAPES_CLASSES.items():
        pixels = int(np.count_nonzero(mask == class_id))
        if pixels:
            class_distribution[class_name] = {
                "pixels": pixels,
                "percentage": round(pixels / total_pixels * 100, 2),
            }

    road_pixels = int(np.count_nonzero(mask == 0))
    sidewalk_pixels = int(np.count_nonzero(mask == 1))

    return {
        "image_width": width,
        "image_height": height,
        "object_count": len(objects),
        "obstacle_count": len(objects),
        "objects": objects,
        "obstacles": objects,
        "class_distribution": class_distribution,
        "navigation_surface": {
            "road_pixels": road_pixels,
            "road_percentage": round(road_pixels / total_pixels * 100, 2),
            "sidewalk_pixels": sidewalk_pixels,
            "sidewalk_percentage": round(
                sidewalk_pixels / total_pixels * 100, 2
            ),
        },
    }
