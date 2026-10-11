import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODELS = {
    "B0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "B2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
}

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]


def sync_device(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    if len(sys.argv) != 2:
        print('Usage: python tools/compare_models.py "path/to/image.jpg"')
        sys.exit(1)

    image_path = Path(sys.argv[1]).expanduser()

    if not image_path.is_file():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 65)
    print("SEMANTIC SEGMENTATION MODEL COMPARISON")
    print("=" * 65)
    print(f"Image: {image_path}")
    print(f"Image size: {image.width} x {image.height}")
    print(f"Device: {device}")
    print("Timing: 1 warm-up + 5 measured inference runs")
    print("Note: timings exclude model loading and image preprocessing.")

    results = {}
    output_dir = Path("results/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, model_name in MODELS.items():
        print(f"\nLoading {label}: {model_name}")

        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval()

        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Warm-up
        with torch.inference_mode():
            _ = model(**inputs)
        sync_device(device)

        timings = []
        with torch.inference_mode():
            for _ in range(5):
                sync_device(device)
                start = time.perf_counter()

                outputs = model(**inputs)
                logits = F.interpolate(
                    outputs.logits,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

                sync_device(device)
                timings.append((time.perf_counter() - start) * 1000)

        del model, processor, inputs, outputs, logits

        mask_path = output_dir / f"{label}_mask.png"
        Image.fromarray(mask).save(mask_path)

        results[label] = {
            "mask": mask,
            "times": timings,
        }

        print(f"Mean inference time:   {np.mean(timings):.2f} ms")
        print(f"Median inference time: {np.median(timings):.2f} ms")
        print(f"Std. deviation:       {np.std(timings):.2f} ms")
        print(f"Detected classes:     {len(np.unique(mask))}/19")
        print(f"Saved mask: {mask_path}")

        print("Class pixel distribution:")
        unique_ids, counts = np.unique(mask, return_counts=True)
        total = mask.size

        for class_id, count in zip(unique_ids, counts):
            name = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
            print(f"  {name:16s} {count / total * 100:6.2f}%")

    mask0 = results["B0"]["mask"]
    mask2 = results["B2"]["mask"]

    same = np.mean(mask0 == mask2) * 100
    changed = 100 - same

    print("\n" + "=" * 65)
    print("COMPARISON")
    print("=" * 65)
    print(f"B0 mean inference: {np.mean(results['B0']['times']):.2f} ms")
    print(f"B2 mean inference: {np.mean(results['B2']['times']):.2f} ms")
    print(f"Pixel agreement:   {same:.2f}%")
    print(f"Pixels changed:    {changed:.2f}%")
    print("\nPer-class pixel percentage difference (B2 - B0):")

    for class_id, name in enumerate(CLASSES):
        pct0 = np.mean(mask0 == class_id) * 100
        pct2 = np.mean(mask2 == class_id) * 100
        print(f"  {name:16s} {pct2 - pct0:+7.2f} percentage points")

    print("\nImportant: agreement is NOT accuracy.")
    print("True mIoU and Pixel Accuracy require ground-truth masks.")
    print(f"Masks saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
