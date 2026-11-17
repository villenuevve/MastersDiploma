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

from backend.database import BASE_DIR, SessionLocal
from backend.models import Dataset, Image as DBImage


MODEL_NAME = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"

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


def resolve_path(relative_path):
    path = Path(relative_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_dataset(dataset_id):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found.")

        if dataset.is_system:
            raise ValueError(
                "The system dataset 'Uploaded images' cannot be used "
                "for managed training splits. Create a separate dataset."
            )

        records = db.query(DBImage).filter(
            DBImage.dataset_id == dataset_id,
            DBImage.has_ground_truth.is_(True),
            DBImage.ground_truth_path.isnot(None),
            DBImage.split_id.isnot(None),
        ).all()

        result = {
            "Train": [],
            "Val": [],
            "Test": [],
        }

        for record in records:
            split_name = record.split.name if record.split else None

            if split_name not in result:
                continue

            image_path = resolve_path(record.file_path)
            mask_path = resolve_path(record.ground_truth_path)

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image {record.id} not found: {image_path}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Ground Truth for image {record.id} not found: "
                    f"{mask_path}"
                )

            result[split_name].append({
                "id": record.id,
                "filename": record.filename,
                "image": image_path,
                "mask": mask_path,
            })

        return dataset.name, result

    finally:
        db.close()


def load_pair(record, max_side):
    with Image.open(record["image"]) as source:
        image = source.convert("RGB")

    with Image.open(record["mask"]) as source:
        mask = source.convert("L")

    if image.size != mask.size:
        raise ValueError(
            f"Image/GT dimension mismatch for image ID {record['id']}: "
            f"{image.size} != {mask.size}"
        )

    width, height = image.size
    scale = min(1.0, max_side / max(width, height))

    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )

    image = image.resize(
        new_size,
        Image.Resampling.BILINEAR
    )
    mask = mask.resize(
        new_size,
        Image.Resampling.NEAREST
    )

    target = np.asarray(mask, dtype=np.uint8)

    invalid = (
        (target != IGNORE_INDEX)
        & (target >= NUM_CLASSES)
    )

    if np.any(invalid):
        raise ValueError(
            f"Invalid class IDs in {record['mask']}: "
            f"{np.unique(target[invalid]).tolist()}"
        )

    if not np.any(target != IGNORE_INDEX):
        raise ValueError(
            f"Ground Truth contains only ignored pixels: {record['mask']}"
        )

    return image, target


def calculate_metrics(confusion):
    true_positive = np.diag(confusion).astype(np.float64)
    ground_truth_count = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)

    union = ground_truth_count + predicted_count - true_positive
    valid_classes = union > 0

    iou = np.zeros(NUM_CLASSES, dtype=np.float64)
    iou[valid_classes] = (
        true_positive[valid_classes] / union[valid_classes]
    )

    total = confusion.sum()
    pixel_accuracy = (
        float(true_positive.sum() / total)
        if total > 0 else 0.0
    )

    mean_iou = (
        float(iou[valid_classes].mean())
        if np.any(valid_classes) else 0.0
    )

    return {
        "mIoU": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "valid_classes": int(valid_classes.sum()),
        "class_iou": {
            CLASS_NAMES[index]: float(iou[index])
            for index in range(NUM_CLASSES)
            if valid_classes[index]
        },
        "valid_pixels": int(total),
    }


def evaluate(model, processor, records, device, max_side):
    model.eval()
    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            image, target = load_pair(record, max_side)

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            pixel_values = inputs["pixel_values"].to(device)

            outputs = model(pixel_values=pixel_values)

            logits = F.interpolate(
                outputs.logits,
                size=target.shape,
                mode="bilinear",
                align_corners=False
            )

            prediction = (
                logits.argmax(dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            valid = target != IGNORE_INDEX

            encoded = (
                target[valid].astype(np.int64) * NUM_CLASSES
                + prediction[valid].astype(np.int64)
            )

            confusion += np.bincount(
                encoded,
                minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            if index % 10 == 0 or index == len(records):
                print(
                    f"Validation: {index}/{len(records)}",
                    flush=True
                )

    return calculate_metrics(confusion)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SegFormer-B3 using a database dataset."
    )

    parser.add_argument(
        "--dataset-id",
        type=int,
        required=True
    )
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/segformer_b3_user_finetuned")
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

    dataset_name, splits = load_dataset(args.dataset_id)

    for split_name in ("Train", "Val", "Test"):
        if not splits[split_name]:
            raise ValueError(
                f"Dataset '{dataset_name}' has no labeled images "
                f"assigned to {split_name}."
            )

    if args.max_images > len(splits["Train"]):
        raise ValueError(
            f"Requested {args.max_images} training images, "
            f"available: {len(splits['Train'])}"
        )

    train_records = random.sample(
        splits["Train"],
        args.max_images
    )
    val_records = splits["Val"]

    print("=== SEGFORMER-B3 USER DATASET FINE-TUNING ===")
    print("Dataset:", dataset_name)
    print("Dataset ID:", args.dataset_id)
    print("Initial checkpoint:", MODEL_NAME)
    print("Device:", device)
    print("Train images:", len(train_records))
    print("Validation images:", len(val_records))
    print("Test images reserved:", len(splits["Test"]))
    print("Maximum input side:", args.max_side)
    print("Epochs:", args.epochs)
    print("Batch size: 1")
    print("Learning rate:", args.lr)
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
            f"Expected {NUM_CLASSES} classes, "
            f"got {model.config.num_labels}"
        )

    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    epoch_losses = []
    validation_history = []
    best_miou = -1.0
    started = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        losses = []
        epoch_started = time.perf_counter()

        random.shuffle(train_records)

        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===")

        for index, record in enumerate(train_records, start=1):
            image, target = load_pair(record, args.max_side)

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            pixel_values = inputs["pixel_values"].to(device)

            labels = torch.from_numpy(
                target.astype(np.int64)
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
                    f"Non-finite loss for image ID {record['id']}"
                )

            loss.backward()
            optimizer.step()

            synchronize(device)

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)

            if index % 10 == 0 or index == len(train_records):
                print(
                    f"Train [{index}/{len(train_records)}] "
                    f"Loss: {loss_value:.5f} | "
                    f"Mean: {np.mean(losses):.5f}",
                    flush=True
                )

        epoch_loss = float(np.mean(losses))
        epoch_losses.append(epoch_loss)

        print("Running validation...")

        val_metrics = evaluate(
            model,
            processor,
            val_records,
            device,
            args.max_side
        )

        validation_history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            **val_metrics,
        })

        print(
            f"Epoch {epoch + 1} | "
            f"Train loss: {epoch_loss:.5f} | "
            f"Val mIoU: {val_metrics['mIoU']:.5f} | "
            f"Val pixel accuracy: "
            f"{val_metrics['pixel_accuracy']:.5f} | "
            f"Time: "
            f"{(time.perf_counter() - epoch_started) / 60:.2f} min"
        )

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]

            args.output.mkdir(
                parents=True,
                exist_ok=True
            )

            model.to("cpu")
            model.save_pretrained(
                args.output,
                safe_serialization=True
            )
            processor.save_pretrained(args.output)
            model.to(device)

            print(
                f"Best checkpoint updated. "
                f"Val mIoU: {best_miou:.5f}"
            )

    total_time = time.perf_counter() - started

    metadata = {
        "base_model": MODEL_NAME,
        "dataset": dataset_name,
        "dataset_id": args.dataset_id,
        "training_split": "Train",
        "validation_split": "Val",
        "test_split": "Test",
        "training_images": [
            {
                "id": record["id"],
                "filename": record["filename"],
            }
            for record in train_records
        ],
        "validation_images": [
            {
                "id": record["id"],
                "filename": record["filename"],
            }
            for record in val_records
        ],
        "reserved_test_images": len(splits["Test"]),
        "epochs": args.epochs,
        "max_input_side": args.max_side,
        "batch_size": 1,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "seed": args.seed,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "device": str(device),
        "epoch_mean_losses": epoch_losses,
        "validation_history": validation_history,
        "best_validation_miou": best_miou,
        "training_time_minutes": round(total_time / 60, 2),
        "checkpoint_selection": "Best validation mIoU",
    }

    with open(
        args.output / "training_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n=== TRAINING COMPLETED ===")
    print("Best validation mIoU:", round(best_miou, 5))
    print("Checkpoint:", args.output)
    print("Training time:", round(total_time / 60, 2), "min")
    print("Metadata: training_config.json")
    print("Test split was not used during training or validation.")


if __name__ == "__main__":
    main()
