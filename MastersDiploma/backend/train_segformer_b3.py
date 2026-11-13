import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


MODEL_NAME = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"

IMAGE_DIR = Path("data/evaluation/cityscapes_full/leftImg8bit/train")
MASK_DIR = Path("data/evaluation/cityscapes_trainIds/train")

NUM_CLASSES = 19
IGNORE_INDEX = 255

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def get_mask_path(image_path):
    relative = image_path.relative_to(IMAGE_DIR)
    mask_name = image_path.name.replace(
        "_leftImg8bit.png",
        "_gtFine_labelTrainIds.png"
    )
    return MASK_DIR / relative.parent / mask_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/segformer_b3_cityscapes_finetuned_pilot")
    )
    args = parser.parse_args()

    if args.max_images < 1:
        raise ValueError("--max-images must be positive")
    if args.max_side < 256:
        raise ValueError("--max-side must be at least 256")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.output.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {args.output}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    images = sorted(IMAGE_DIR.rglob("*_leftImg8bit.png"))

    if not images:
        raise FileNotFoundError(f"No training images in {IMAGE_DIR}")

    if args.max_images > len(images):
        raise ValueError(
            f"Requested {args.max_images} images, available: {len(images)}"
        )

    # Random sampling across all cities, not only the first sorted files.
    selected_images = random.sample(images, args.max_images)

    for image_path in selected_images:
        mask_path = get_mask_path(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

    print("=== SEGFORMER-B3 FINE-TUNING ===")
    print("Initial checkpoint:", MODEL_NAME)
    print("Device:", device)
    print("Training images:", len(selected_images))
    print("Maximum input side:", args.max_side)
    print("Epochs:", args.epochs)
    print("Batch size: 1")
    print("Learning rate:", args.lr)
    print("Ignore index:", IGNORE_INDEX)
    print("Output:", args.output)

    processor = SegformerImageProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )
    processor.do_resize = False
    processor.do_pad = False

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    if model.config.num_labels != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {model.config.num_labels}"
        )

    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    epoch_losses = []
    started = time.perf_counter()

    for epoch in range(args.epochs):
        losses = []
        epoch_started = time.perf_counter()

        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===")

        for index, image_path in enumerate(selected_images, start=1):
            mask_path = get_mask_path(image_path)

            with Image.open(image_path) as source:
                image = source.convert("RGB")

            with Image.open(mask_path) as source:
                mask = source.convert("L")

            if image.size != mask.size:
                raise ValueError(
                    f"Dimension mismatch: {image_path.name}"
                )

            original_width, original_height = image.size

            scale = min(
                1.0,
                args.max_side / max(original_width, original_height)
            )

            width = max(1, round(original_width * scale))
            height = max(1, round(original_height * scale))

            image = image.resize(
                (width, height),
                Image.Resampling.BILINEAR
            )
            mask = mask.resize(
                (width, height),
                Image.Resampling.NEAREST
            )

            target_array = np.array(mask, dtype=np.uint8)

            invalid = (
                (target_array != IGNORE_INDEX)
                & (target_array >= NUM_CLASSES)
            )
            if np.any(invalid):
                raise ValueError(
                    f"Invalid TrainIds in {mask_path.name}: "
                    f"{np.unique(target_array[invalid]).tolist()}"
                )

            if not np.any(target_array != IGNORE_INDEX):
                raise ValueError(
                    f"Mask contains only ignored pixels: {mask_path}"
                )

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            pixel_values = inputs["pixel_values"].to(device)

            labels = torch.from_numpy(
                target_array.astype(np.int64)
            ).unsqueeze(0).to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(pixel_values=pixel_values)

            logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            loss = F.cross_entropy(
                logits,
                labels,
                ignore_index=IGNORE_INDEX
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at image {image_path.name}"
                )

            loss.backward()
            optimizer.step()

            synchronize(device)

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)

            if index % 10 == 0 or index == len(selected_images):
                print(
                    f"[{index}/{len(selected_images)}] "
                    f"Loss: {loss_value:.5f} | "
                    f"Mean loss: {np.mean(losses):.5f}",
                    flush=True
                )

        epoch_loss = float(np.mean(losses))
        epoch_losses.append(epoch_loss)

        print(
            f"Epoch {epoch + 1} completed | "
            f"Mean loss: {epoch_loss:.5f} | "
            f"Time: {(time.perf_counter() - epoch_started) / 60:.2f} min"
        )

    total_time = time.perf_counter() - started

    print("\n=== SAVING CHECKPOINT ===")

    args.output.mkdir(parents=True, exist_ok=False)

    model.to("cpu")
    model.save_pretrained(
        args.output,
        safe_serialization=True
    )
    processor.save_pretrained(args.output)

    metadata = {
        "base_model": MODEL_NAME,
        "dataset": "Cityscapes",
        "training_split": "train",
        "training_images": len(selected_images),
        "epochs": args.epochs,
        "max_input_side": args.max_side,
        "batch_size": 1,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "seed": args.seed,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "class_names": CLASS_NAMES,
        "device": str(device),
        "epoch_mean_losses": epoch_losses,
        "training_time_minutes": round(total_time / 60, 2),
        "selected_images": [
            str(path.relative_to(IMAGE_DIR))
            for path in selected_images
        ],
    }

    with open(
        args.output / "training_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("Checkpoint saved:", args.output)
    print("Training time:", round(total_time / 60, 2), "min")
    print("Epoch losses:", epoch_losses)
    print("Training metadata: training_config.json")


if __name__ == "__main__":
    main()
