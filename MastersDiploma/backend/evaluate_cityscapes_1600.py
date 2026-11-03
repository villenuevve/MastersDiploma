from pathlib import Path

from backend import segmentation
from backend import evaluate_cityscapes_full as evaluator


def main():
    # Change only the maximum input image side for this experiment.
    segmentation.MAX_INPUT_SIDE = 1600

    # Preserve the original baseline results.
    evaluator.OUTPUT_PATH = Path(
        "results/cityscapes_1600_metrics.json"
    )

    print("=== RESOLUTION EXPERIMENT ===")
    print("Model:", segmentation.MODEL_NAME)
    print("Device:", segmentation.DEVICE)
    print("Maximum input side:", segmentation.MAX_INPUT_SIDE)
    print("Dataset: Cityscapes validation (500 images)")
    print()

    evaluator.main()


if __name__ == "__main__":
    main()
