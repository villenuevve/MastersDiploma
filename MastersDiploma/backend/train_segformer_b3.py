import argparse
import json
import random
import time
import zipfile
from io import BytesIO
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

SPLITS = {
    "train": "Train",
    "val": "Val",
    "test": "Test",
}


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def resolve_path(relative_path):
    path = Path(relative_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_database_dataset(dataset_id):
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
                "for managed training splits."
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


class InfraParisZipSource:
    def __init__(self, images_path, masks_path):
        self.images_path = Path(images_path).expanduser().resolve()
        self.masks_path = Path(masks_path).expanduser().resolve()

        if not self.images_path.is_file():
            raise FileNotFoundError(
                f"Images ZIP not found: {self.images_path}"
            )

        if not self.masks_path.is_file():
            raise FileNotFoundError(
                f"Ground Truth ZIP not found: {self.masks_path}"
            )

        self.images_zip = zipfile.ZipFile(self.images_path, "r")
        self.masks_zip = zipfile.ZipFile(self.masks_path, "r")

    def close(self):
        self.images_zip.close()
        self.masks_zip.close()

    def load_records(self):
        image_files = {}
        mask_files = {}

        valid_splits = {"train", "val", "test"}

        for name in self.images_zip.namelist():
            parts = name.split("/")
            if len(parts) < 4 or not name.lower().endswith(".png"):
                continue

            split_index = next(
                (i for i, part in enumerate(parts)
                 if part.lower() in valid_splits),
                None
            )
            if split_index is None:
                continue

            split = parts[split_index].lower()
            relative_parts = parts[split_index + 1:]
            stem = Path(relative_parts[-1]).stem
            location = tuple(relative_parts[:-1])
            key = (split, location, stem)

            if key in image_files:
                raise ValueError(
                    f"Duplicate InfraParis image key: {key}"
                )

            image_files[key] = name

        for name in self.masks_zip.namelist():
            parts = name.split("/")
            if len(parts) < 4 or not parts[-1].endswith("_labelIds.png"):
                continue

            split_index = next(
                (i for i, part in enumerate(parts)
                 if part.lower() in valid_splits),
                None
            )
            if split_index is None:
                continue

            split = parts[split_index].lower()
            relative_parts = parts[split_index + 1:]
            stem = relative_parts[-1][:-len("_labelIds.png")]
            location = tuple(relative_parts[:-1])
            key = (split, location, stem)

            if key in mask_files:
                raise ValueError(
                    f"Duplicate InfraParis mask key: {key}"
                )

            mask_files[key] = name

        image_keys = set(image_files)
        mask_keys = set(mask_files)

        missing_masks = image_keys - mask_keys
        missing_images = mask_keys - image_keys

        if missing_masks or missing_images:
            raise ValueError(
                "InfraParis image/mask pairing failed. "
                f"Images without masks: {len(missing_masks)}. "
                f"Masks without images: {len(missing_images)}. "
                f"Image examples: {list(sorted(missing_masks))[:3]}. "
                f"Mask examples: {list(sorted(missing_images))[:3]}"
            )

        if not image_keys:
            raise ValueError(
                "No InfraParis image/mask pairs found in ZIP archives."
            )

        result = {
            "Train": [],
            "Val": [],
            "Test": [],
        }

        for split, location, stem in sorted(image_keys):
            result[SPLITS[split]].append({
                "id": f"{split}:{'/'.join(location)}:{stem}",
                "filename": f"{stem}.png",
                "image": image_files[(split, location, stem)],
                "mask": mask_files[(split, location, stem)],
                "source": "infraparis",
            })

        print("InfraParis dataset loaded:")
        for split_name, records in result.items():
            print(f"  {split_name}: {len(records)} image/mask pairs")

        return result


def load_pair(record, max_side, zip_source=None):
    if record.get("source") == "infraparis":
        if zip_source is None:
            raise ValueError("InfraParis ZIP source was not provided.")

        image_bytes = zip_source.images_zip.read(record["image"])
        mask_bytes = zip_source.masks_zip.read(record["mask"])

        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        with Image.open(BytesIO(mask_bytes)) as source:
            mask = source.convert("L")

    else:
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")

        with Image.open(record["mask"]) as source:
            mask = source.convert("L")

    if image.size != mask.size:
        raise ValueError(
            f"Image/GT dimension mismatch for {record['id']}: "
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

    target = np.asarray(mask, dtype=np.uint8).copy()

    # InfraParis ID 19 is Unlabeled and is excluded
    # from the 19-class Cityscapes training targets.
    if record.get("source") == "infraparis":
        target[target == 19] = IGNORE_INDEX

    invalid = (
        (target != IGNORE_INDEX)
        & (target >= NUM_CLASSES)
    )

    if np.any(invalid):
        raise ValueError(
            f"Invalid class IDs in {record['id']}: "
            f"{np.unique(target[invalid]).tolist()}"
        )

    if not np.any(target != IGNORE_INDEX):
        raise ValueError(
            f"Ground Truth contains only ignored pixels: {record['id']}"
        )

    return image, target



def has_trainable_pixels(record, max_side, zip_source):
    """
    Check whether a mask contains at least one valid target pixel
    after applying the same resizing and label mapping as load_pair().
    Only the mask is read; the RGB image is not loaded.
    """
    if record.get("source") != "infraparis":
        return True

    if zip_source is None:
        raise ValueError("InfraParis ZIP source was not provided.")

    mask_bytes = zip_source.masks_zip.read(record["mask"])

    with Image.open(BytesIO(mask_bytes)) as source:
        mask = source.convert("L")

    width, height = mask.size
    scale = min(1.0, max_side / max(width, height))

    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )

    mask = mask.resize(
        new_size,
        Image.Resampling.NEAREST
    )

    target = np.asarray(mask, dtype=np.uint8).copy()
    target[target == 19] = IGNORE_INDEX

    invalid = (
        (target != IGNORE_INDEX)
        & (target >= NUM_CLASSES)
    )

    if np.any(invalid):
        raise ValueError(
            f"Invalid class IDs in {record['id']}: "
            f"{np.unique(target[invalid]).tolist()}"
        )

    return bool(np.any(target != IGNORE_INDEX))


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


def evaluate(model, processor, records, device, max_side, zip_source):
    model.eval()

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            image, target = load_pair(
                record,
                max_side,
                zip_source
            )

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
        description="Fine-tune SegFormer-B3 on a database dataset "
                    "or InfraParis ZIP archives."
    )

    parser.add_argument(
        "--source",
        choices=["database", "infraparis"],
        default="database"
    )
    parser.add_argument("--dataset-id", type=int, default=None)

    parser.add_argument(
        "--images-zip",
        type=Path,
        default=Path("~/Downloads/Left.zip")
    )
    parser.add_argument(
        "--masks-zip",
        type=Path,
        default=Path("~/Downloads/semantic_segmentation_truth.zip")
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

    if args.source == "database" and args.dataset_id is None:
        parser.error("--dataset-id is required when --source=database")

    if args.max_images < 1:
        parser.error("--max-images must be positive")
    if args.max_side < 256:
        parser.error("--max-side must be at least 256")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")

    args.output = args.output.expanduser()

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

    zip_source = None

    if args.source == "database":
        dataset_name, splits = load_database_dataset(args.dataset_id)

    else:
        zip_source = InfraParisZipSource(
            args.images_zip,
            args.masks_zip
        )
        dataset_name = "InfraParis"
        splits = zip_source.load_records()

    for split_name in ("Train", "Val", "Test"):
        if not splits[split_name]:
            raise ValueError(
                f"Dataset '{dataset_name}' has no labeled images "
                f"assigned to {split_name}."
            )

    train_candidates = splits["Train"]
    excluded_unlabeled_training_records = []

    if args.source == "infraparis":
        usable_train_records = []

        print("Checking InfraParis training masks...", flush=True)

        for index, record in enumerate(train_candidates, start=1):
            if has_trainable_pixels(
                record,
                args.max_side,
                zip_source
            ):
                usable_train_records.append(record)
            else:
                excluded_unlabeled_training_records.append(record)

            if index % 500 == 0 or index == len(train_candidates):
                print(
                    f"Mask filtering: {index}/{len(train_candidates)} | "
                    f"Usable: {len(usable_train_records)} | "
                    f"Excluded: "
                    f"{len(excluded_unlabeled_training_records)}",
                    flush=True
                )

        train_candidates = usable_train_records

    if args.max_images > len(train_candidates):
        raise ValueError(
            f"Requested {args.max_images} training images, "
            f"available after filtering: {len(train_candidates)}"
        )

    train_records = random.sample(
        train_candidates,
        args.max_images
    )
    val_records = splits["Val"]

    print("=== SEGFORMER-B3 FINE-TUNING ===")
    print("Source:", args.source)
    print("Dataset:", dataset_name)
    print("Dataset ID:", args.dataset_id)
    print("Initial checkpoint:", MODEL_NAME)
    print("Device:", device)
    print("Train images selected:", len(train_records))
    print("Usable Train candidates:", len(train_candidates))
    print(
        "Excluded fully unlabeled Train images:",
        len(excluded_unlabeled_training_records)
    )
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

    print("\n=== BASELINE EVALUATION (BEFORE FINE-TUNING) ===")
    baseline_metrics = evaluate(
        model,
        processor,
        val_records,
        device,
        args.max_side,
        zip_source
    )

    print(
        f"Baseline Val mIoU: {baseline_metrics['mIoU']:.5f} | "
        f"Baseline Val pixel accuracy: "
        f"{baseline_metrics['pixel_accuracy']:.5f}"
    )

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
            image, target = load_pair(
                record,
                args.max_side,
                zip_source
            )

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
            args.max_side,
            zip_source
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
        "source": args.source,
        "dataset_id": args.dataset_id,
        "source_archives": (
            {
                "images": str(zip_source.images_path),
                "masks": str(zip_source.masks_path),
            }
            if zip_source else None
        ),
        "source_class_mapping": (
            {
                "0-18": "Mapped directly to the 19 Cityscapes classes",
                "19": IGNORE_INDEX,
            }
            if args.source == "infraparis" else None
        ),
        "training_split": "Train",
        "validation_split": "Val",
        "test_split": "Test",
        "split_sizes": {
            name: len(records)
            for name, records in splits.items()
        },
        "training_candidates_original": len(splits["Train"]),
        "training_candidates_usable": len(train_candidates),
        "excluded_unlabeled_training_count": (
            len(excluded_unlabeled_training_records)
        ),
        "excluded_unlabeled_training_images": [
            {
                "id": record["id"],
                "filename": record["filename"],
            }
            for record in excluded_unlabeled_training_records
        ],
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
        "baseline_validation": baseline_metrics,
        "epoch_mean_losses": epoch_losses,
        "validation_history": validation_history,
        "best_validation_miou": best_miou,
        "training_time_minutes": round(total_time / 60, 2),
        "checkpoint_selection": "Best validation mIoU",
        "test_evaluated": False,
    }

    with open(
        args.output / "training_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    if zip_source is not None:
        zip_source.close()

    print("\n=== TRAINING COMPLETED ===")
    print("Best validation mIoU:", round(best_miou, 5))
    print("Checkpoint:", args.output)
    print("Training time:", round(total_time / 60, 2), "min")
    print("Metadata: training_config.json")
    print("Test split was not used during training or validation.")


if __name__ == "__main__":
    main()
