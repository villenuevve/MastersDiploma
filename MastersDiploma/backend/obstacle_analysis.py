import math

import numpy as np
from scipy import ndimage


CLASS_NAMES = {
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

DYNAMIC_CLASSES = {11, 12, 13, 14, 15, 16, 17, 18}
STATIC_CLASSES = {3, 4, 5, 6, 7}
ENVIRONMENT_CLASSES = {2}

CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)
MIN_AREA_FRACTION = 0.0001
MIN_AREA_PIXELS = 32


def _get_position(x, width):
    if x < width / 3:
        return "left"
    if x < 2 * width / 3:
        return "center"
    return "right"


def _extract_components(mask, selected_classes, category, min_area):
    height, width = mask.shape
    total_pixels = mask.size
    objects = []

    for class_id in selected_classes:
        binary_mask = mask == class_id

        if not np.any(binary_mask):
            continue

        labels, component_count = ndimage.label(
            binary_mask,
            structure=CONNECTIVITY,
        )

        component_sizes = np.bincount(labels.ravel())
        component_slices = ndimage.find_objects(labels)

        valid_components = [
            component_id
            for component_id in range(1, component_count + 1)
            if component_sizes[component_id] >= min_area
        ]

        if not valid_components:
            continue

        # Calculate component centroids using weighted bincount.
        # This is equivalent to ndimage.center_of_mass for integer labels.
        flat_labels = labels.ravel()
        foreground = np.flatnonzero(flat_labels)
        component_ids = flat_labels[foreground]

        sum_y = np.bincount(
            component_ids,
            weights=foreground // width,
            minlength=component_count + 1,
        )
        sum_x = np.bincount(
            component_ids,
            weights=foreground % width,
            minlength=component_count + 1,
        )

        centers = [
            (
                sum_y[component_id] / component_sizes[component_id],
                sum_x[component_id] / component_sizes[component_id],
            )
            for component_id in valid_components
        ]

        for component_id, center in zip(valid_components, centers):
            area = int(component_sizes[component_id])

            component_slice = component_slices[component_id - 1]

            y_min = component_slice[0].start
            y_max = component_slice[0].stop - 1
            x_min = component_slice[1].start
            x_max = component_slice[1].stop - 1

            center_y, center_x = center

            objects.append({
                "id": 0,
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "category": category,
                "area_pixels": area,
                "area_percentage": round(
                    area / total_pixels * 100,
                    3,
                ),
                "position": _get_position(center_x, width),
                "center": {
                    "x": round(float(center_x), 1),
                    "y": round(float(center_y), 1),
                },
                "bounding_box": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                },
            })

    objects.sort(key=lambda item: item["area_pixels"], reverse=True)

    for index, item in enumerate(objects, start=1):
        item["id"] = index

    return objects


def analyze_obstacles(mask):
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Expected a 2D semantic segmentation mask.")

    if mask.size == 0:
        raise ValueError("Segmentation mask is empty.")

    min_area = max(
        MIN_AREA_PIXELS,
        math.ceil(mask.size * MIN_AREA_FRACTION),
    )

    dynamic_obstacles = _extract_components(
        mask,
        DYNAMIC_CLASSES,
        "dynamic",
        min_area,
    )

    static_obstacles = _extract_components(
        mask,
        STATIC_CLASSES,
        "static",
        min_area,
    )

    environment_objects = _extract_components(
        mask,
        ENVIRONMENT_CLASSES,
        "environment",
        min_area,
    )

    obstacles = dynamic_obstacles + static_obstacles
    obstacles.sort(
        key=lambda item: item["area_pixels"],
        reverse=True,
    )

    for index, item in enumerate(obstacles, start=1):
        item["id"] = index

    return {
        "obstacle_summary": {
            "total": len(obstacles),
            "dynamic": len(dynamic_obstacles),
            "static": len(static_obstacles),
            "min_area_pixels": min_area,
        },
        "obstacles": obstacles,
        "environment_objects": environment_objects,
    }
