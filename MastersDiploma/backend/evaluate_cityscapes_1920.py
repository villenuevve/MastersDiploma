from pathlib import Path

from backend import segmentation
from backend import evaluate_cityscapes_full as evaluator


def main():
    # Experimental resolution only. Does not modify the application.
    segmentation.MAX_INPUT_SIDE = 1920

    # Save results separately.
    evaluator.OUTPUT_PATH = Path(
        "results/cityscapes_1920_metrics.json"
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
