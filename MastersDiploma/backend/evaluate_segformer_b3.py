import gc
import json
from pathlib import Path

import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from backend.train_segformer_b3 import (
    MODEL_NAME,
    IGNORE_INDEX,
    InfraParisZipSource,
    has_trainable_pixels,
    evaluate,
    synchronize,
)


CHECKPOINT = Path(
    "models/segformer_b3_infraparis_1000_512_e3_filtered"
)

IMAGES_ZIP = Path("~/Downloads/Left.zip").expanduser()
MASKS_ZIP = Path(
    "~/Downloads/semantic_segmentation_truth.zip"
).expanduser()

MAX_SIDE = 512

RESULT_PATH = Path(
    "results/segformer_b3_infraparis_1000_512_e3_test_metrics.json"
)


def main():
    if not CHECKPOINT.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {CHECKPOINT}"
        )

    if RESULT_PATH.exists():
        raise FileExistsError(
            f"Results file already exists: {RESULT_PATH}"
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("=== SEGFORMER-B3 INDEPENDENT TEST EVALUATION ===")
    print("Device:", device)
    print("Fine-tuned checkpoint:", CHECKPOINT)
    print("Maximum input side:", MAX_SIDE)

    zip_source = InfraParisZipSource(
        IMAGES_ZIP,
        MASKS_ZIP
    )

    try:
        splits = zip_source.load_records()
        test_candidates = splits["Test"]

        print("\nChecking Test masks...", flush=True)

        test_records = []
        excluded_records = []

        for index, record in enumerate(test_candidates, start=1):
            if has_trainable_pixels(
                record,
                MAX_SIDE,
                zip_source
            ):
                test_records.append(record)
            else:
                excluded_records.append(record)

            if index % 100 == 0 or index == len(test_candidates):
                print(
                    f"Test filtering: {index}/{len(test_candidates)} | "
                    f"Usable: {len(test_records)} | "
                    f"Excluded: {len(excluded_records)}",
                    flush=True
                )

        if not test_records:
            raise RuntimeError("No usable Test images found.")

        print("\nTest images available:", len(test_candidates))
        print("Test images evaluated:", len(test_records))
        print("Fully unlabeled Test images excluded:", len(excluded_records))

        processor = SegformerImageProcessor.from_pretrained(
            CHECKPOINT,
            local_files_only=True
        )
        processor.do_resize = False
        processor.do_pad = False

        # 1. Evaluate the original pretrained model.
        print("\n=== ORIGINAL MODEL: TEST ===")

        baseline_model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            local_files_only=True
        )
        baseline_model.config.semantic_loss_ignore_index = IGNORE_INDEX
        baseline_model.to(device)

        baseline_metrics = evaluate(
            baseline_model,
            processor,
            test_records,
            device,
            MAX_SIDE,
            zip_source
        )

        synchronize(device)

        print(
            f"\nOriginal Test mIoU: "
            f"{baseline_metrics['mIoU']:.5f}"
        )
        print(
            f"Original Test Pixel Accuracy: "
            f"{baseline_metrics['pixel_accuracy']:.5f}"
        )

        del baseline_model
        gc.collect()

        # 2. Evaluate the fine-tuned model on exactly the same images.
        print("\n=== FINE-TUNED MODEL: TEST ===")

        finetuned_model = SegformerForSemanticSegmentation.from_pretrained(
            CHECKPOINT,
            local_files_only=True
        )
        finetuned_model.config.semantic_loss_ignore_index = IGNORE_INDEX
        finetuned_model.to(device)

        finetuned_metrics = evaluate(
            finetuned_model,
            processor,
            test_records,
            device,
            MAX_SIDE,
            zip_source
        )

        synchronize(device)

        print(
            f"\nFine-tuned Test mIoU: "
            f"{finetuned_metrics['mIoU']:.5f}"
        )
        print(
            f"Fine-tuned Test Pixel Accuracy: "
            f"{finetuned_metrics['pixel_accuracy']:.5f}"
        )

        # 3. Calculate differences between models.
        miou_change = (
            finetuned_metrics["mIoU"]
            - baseline_metrics["mIoU"]
        ) * 100

        accuracy_change = (
            finetuned_metrics["pixel_accuracy"]
            - baseline_metrics["pixel_accuracy"]
        ) * 100

        print("\n=== TEST COMPARISON ===")
        print(
            f"mIoU: "
            f"{baseline_metrics['mIoU']:.5%} -> "
            f"{finetuned_metrics['mIoU']:.5%} "
            f"({miou_change:+.2f} pp)"
        )
        print(
            f"Pixel Accuracy: "
            f"{baseline_metrics['pixel_accuracy']:.5%} -> "
            f"{finetuned_metrics['pixel_accuracy']:.5%} "
            f"({accuracy_change:+.2f} pp)"
        )

        print("\n=== CLASS IoU COMPARISON ===")
        print(
            f"{'Class':<18} {'Original':>12} "
            f"{'Fine-tuned':>12} {'Change':>12}"
        )

        class_changes = {}

        for class_name, before_value in baseline_metrics["class_iou"].items():
            after_value = finetuned_metrics["class_iou"].get(class_name)

            if after_value is None:
                print(
                    f"{class_name:<18} "
                    f"{before_value:>11.2%} "
                    f"{'N/A':>12} {'N/A':>12}"
                )
                continue

            change = (after_value - before_value) * 100
            class_changes[class_name] = change

            print(
                f"{class_name:<18} "
                f"{before_value:>11.2%} "
                f"{after_value:>11.2%} "
                f"{change:>+11.2f} pp"
            )

        # 4. Save the complete evaluation results separately.
        result = {
            "dataset": "InfraParis",
            "split": "Test",
            "checkpoint": str(CHECKPOINT),
            "base_model": MODEL_NAME,
            "max_input_side": MAX_SIDE,
            "device": str(device),
            "test_candidates": len(test_candidates),
            "test_images_evaluated": len(test_records),
            "excluded_fully_unlabeled_images": [
                {
                    "id": record["id"],
                    "filename": record["filename"],
                }
                for record in excluded_records
            ],
            "original_model": baseline_metrics,
            "fine_tuned_model": finetuned_metrics,
            "change_percentage_points": {
                "mIoU": miou_change,
                "pixel_accuracy": accuracy_change,
                "class_iou": class_changes,
            },
            "test_image_ids": [
                record["id"] for record in test_records
            ],
        }

        RESULT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with RESULT_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2
            )

        print("\n=== EVALUATION COMPLETED ===")
        print("Results saved to:", RESULT_PATH)
        print("Test images evaluated:", len(test_records))
        print("Training was not performed.")
        print("Checkpoint was not modified.")

        del finetuned_model
        gc.collect()

    finally:
        zip_source.close()


if __name__ == "__main__":
    main()
