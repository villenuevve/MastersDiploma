import cv2
import numpy as np


def analyze_safe_corridor(
    traversability: np.ndarray,
    risk_map: np.ndarray,
    min_traversability: int = 100,
    max_risk: int = 60,
):
    """
    Detect a connected safe corridor starting from the lower-central
    region of the image.

    This is a 2D image-space heuristic. It does not estimate metric
    distance, robot dimensions, or a physically feasible trajectory.
    """

    traversability = np.asarray(traversability)
    risk_map = np.asarray(risk_map)

    if traversability.ndim != 2 or risk_map.ndim != 2:
        raise ValueError("Traversability and risk maps must be 2D arrays.")

    if traversability.shape != risk_map.shape:
        raise ValueError("Traversability and risk map dimensions do not match.")

    if traversability.size == 0:
        raise ValueError("Input maps must not be empty.")

    height, width = traversability.shape

    # Candidate pixels must be traversable and have risk below the threshold.
    candidate = (
        (traversability >= min_traversability)
        & (risk_map < max_risk)
    ).astype(np.uint8)

    # Find connected candidate regions.
    component_count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(candidate, connectivity=8)
    )

    # The assumed robot position is a lower-central image region.
    seed_y_start = int(height * 0.85)
    seed_x_start = int(width * 0.30)
    seed_x_end = max(seed_x_start + 1, int(width * 0.70))

    seed_labels = labels[
        seed_y_start:height,
        seed_x_start:seed_x_end
    ]

    seed_labels = seed_labels[seed_labels > 0]

    selected_label = None

    if seed_labels.size:
        counts = np.bincount(seed_labels, minlength=component_count)
        counts[0] = 0

        # Prefer the component with the largest overlap with the seed region.
        selected_label = int(np.argmax(counts))

    # Create an RGB diagnostic image.
    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    visualization[:] = (35, 35, 35)

    # Candidate safe pixels are shown in blue.
    visualization[candidate > 0] = (65, 90, 115)

    # High-risk pixels are shown in red.
    visualization[risk_map >= max_risk] = (90, 55, 55)

    # A component must extend into the upper half of the image.
    # This rejects regions that touch the assumed robot position
    # but are blocked from progressing forward.
    if selected_label is not None:
        component_y = np.where(labels == selected_label)[0]

        if (
            component_y.size == 0
            or int(component_y.min()) > int(height * 0.5)
        ):
            selected_label = None

    found = selected_label is not None
    corridor_mask = np.zeros((height, width), dtype=np.uint8)

    statistics = {
        "found": False,
        "corridor_pixels": 0,
        "corridor_percentage": 0.0,
        "mean_traversability": None,
        "mean_risk": None,
        "center_x_percent": None,
        "direction": "unavailable",
        "min_traversability": int(min_traversability),
        "max_risk": int(max_risk),
        "seed_region": {
            "x_start": seed_x_start,
            "x_end": seed_x_end,
            "y_start": seed_y_start,
        },
    }

    if found:
        corridor_mask[labels == selected_label] = 1
        visualization[corridor_mask > 0] = (35, 170, 75)

        corridor_pixels = int(np.count_nonzero(corridor_mask))
        ys, xs = np.where(corridor_mask > 0)

        center_x = float(np.mean(xs))
        center_x_percent = center_x / max(width - 1, 1) * 100

        mean_traversability = float(
            np.mean(traversability[corridor_mask > 0])
        )
        mean_risk = float(np.mean(risk_map[corridor_mask > 0]))

        if center_x_percent < 45:
            direction = "left"
        elif center_x_percent > 55:
            direction = "right"
        else:
            direction = "center"

        # Draw a row-wise centerline through the selected connected region.
        centerline = []

        row_step = max(1, height // 300)

        for y in range(int(ys.min()), int(ys.max()) + 1, row_step):
            row_x = np.where(corridor_mask[y] > 0)[0]

            if row_x.size:
                centerline.append(
                    (int(np.median(row_x)), int(y))
                )

        if len(centerline) >= 2:
            points = np.asarray(centerline, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                visualization,
                [points],
                isClosed=False,
                color=(0, 255, 255),
                thickness=max(2, width // 500),
                lineType=cv2.LINE_AA,
            )

        # Mark the assumed starting region.
        cv2.rectangle(
            visualization,
            (seed_x_start, seed_y_start),
            (seed_x_end - 1, height - 1),
            (255, 255, 255),
            1,
        )

        statistics.update({
            "found": True,
            "corridor_pixels": corridor_pixels,
            "corridor_percentage": round(
                corridor_pixels / traversability.size * 100, 2
            ),
            "mean_traversability": round(mean_traversability, 2),
            "mean_risk": round(mean_risk, 2),
            "center_x_percent": round(center_x_percent, 2),
            "direction": direction,
        })

    else:
        # Show the seed area even when no connected corridor is available.
        cv2.rectangle(
            visualization,
            (seed_x_start, seed_y_start),
            (seed_x_end - 1, height - 1),
            (255, 255, 255),
            1,
        )

    return {
        "visualization": visualization,
        "corridor_mask": corridor_mask,
        "statistics": statistics,
    }
