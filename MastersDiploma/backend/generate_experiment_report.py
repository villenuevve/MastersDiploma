import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


CONFIG_PATH = Path(
    "models/segformer_b3_infraparis_1000_512_e3_filtered/"
    "training_config.json"
)

TEST_PATH = Path(
    "results/segformer_b3_infraparis_1000_512_e3_test_metrics.json"
)

OUTPUT_DIR = Path("experiments/infraparis_1000")
TABLES_DIR = OUTPUT_DIR / "tables"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
REPORTS_DIR = OUTPUT_DIR / "reports"


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def main():
    for directory in (TABLES_DIR, GRAPHS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)

    if not TEST_PATH.is_file():
        raise FileNotFoundError(TEST_PATH)

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    with TEST_PATH.open(encoding="utf-8") as file:
        test = json.load(file)

    baseline_val = config["baseline_validation"]
    history = config["validation_history"]
    original_test = test["original_model"]
    finetuned_test = test["fine_tuned_model"]

    # ---------------------------------------------------------
    # 1. Experiment configuration
    # ---------------------------------------------------------

    config_rows = [
        ("Dataset", config["dataset"]),
        ("Base model", config["base_model"]),
        ("Training split", config["training_split"]),
        ("Validation split", config["validation_split"]),
        ("Test split", config["test_split"]),
        ("Original Train candidates", config["training_candidates_original"]),
        ("Usable Train candidates", config["training_candidates_usable"]),
        ("Excluded fully unlabeled images", config["excluded_unlabeled_training_count"]),
        ("Images used for training", len(config["training_images"])),
        ("Validation images", len(config["validation_images"])),
        ("Test images evaluated", test["test_images_evaluated"]),
        ("Epochs", config["epochs"]),
        ("Maximum input side", config["max_input_side"]),
        ("Batch size", config["batch_size"]),
        ("Optimizer", config["optimizer"]),
        ("Learning rate", config["learning_rate"]),
        ("Weight decay", config["weight_decay"]),
        ("Random seed", config["seed"]),
        ("Number of classes", config["num_classes"]),
        ("Ignore index", config["ignore_index"]),
        ("Device", config["device"]),
        ("Training time (minutes)", config["training_time_minutes"]),
        ("Checkpoint selection", config["checkpoint_selection"]),
    ]

    write_csv(
        TABLES_DIR / "experiment_config.csv",
        ["Parameter", "Value"],
        config_rows
    )

    # ---------------------------------------------------------
    # 2. Validation history
    # ---------------------------------------------------------

    validation_rows = [
        [
            "Baseline",
            "",
            baseline_val["mIoU"] * 100,
            baseline_val["pixel_accuracy"] * 100,
        ]
    ]

    for epoch in history:
        validation_rows.append([
            epoch["epoch"],
            epoch["train_loss"],
            epoch["mIoU"] * 100,
            epoch["pixel_accuracy"] * 100,
        ])

    write_csv(
        TABLES_DIR / "validation_history.csv",
        ["Stage", "Train Loss", "Validation mIoU (%)",
         "Validation Pixel Accuracy (%)"],
        validation_rows
    )

    # ---------------------------------------------------------
    # 3. Test metrics comparison
    # ---------------------------------------------------------

    test_rows = [
        [
            "mIoU",
            original_test["mIoU"] * 100,
            finetuned_test["mIoU"] * 100,
            test["change_percentage_points"]["mIoU"],
        ],
        [
            "Pixel Accuracy",
            original_test["pixel_accuracy"] * 100,
            finetuned_test["pixel_accuracy"] * 100,
            test["change_percentage_points"]["pixel_accuracy"],
        ],
    ]

    write_csv(
        TABLES_DIR / "test_metrics_comparison.csv",
        ["Metric", "Original (%)", "Fine-tuned (%)",
         "Change (percentage points)"],
        test_rows
    )

    # ---------------------------------------------------------
    # 4. Per-class IoU
    # ---------------------------------------------------------

    original_classes = original_test["class_iou"]
    finetuned_classes = finetuned_test["class_iou"]

    class_names = config["class_names"]
    class_rows = []

    for name in class_names:
        before = original_classes.get(name)
        after = finetuned_classes.get(name)

        class_rows.append([
            name,
            before * 100 if before is not None else "",
            after * 100 if after is not None else "",
            (after - before) * 100
            if before is not None and after is not None else "",
        ])

    write_csv(
        TABLES_DIR / "per_class_iou.csv",
        ["Class", "Original IoU (%)", "Fine-tuned IoU (%)",
         "Change (percentage points)"],
        class_rows
    )

    # ---------------------------------------------------------
    # 5. Training image list
    # ---------------------------------------------------------

    write_csv(
        TABLES_DIR / "training_images.csv",
        ["Image ID", "Filename"],
        [
            [item["id"], item["filename"]]
            for item in config["training_images"]
        ]
    )

    # ---------------------------------------------------------
    # 6. Training Loss graph
    # ---------------------------------------------------------

    epochs = [item["epoch"] for item in history]
    losses = [item["train_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o", linewidth=2)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Training Loss")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    save_plot("training_loss.png")

    # ---------------------------------------------------------
    # 7. Validation mIoU graph
    # ---------------------------------------------------------

    miou_values = [baseline_val["mIoU"] * 100]
    miou_values.extend([item["mIoU"] * 100 for item in history])

    stages = ["Baseline"] + [f"Epoch {e}" for e in epochs]

    plt.figure(figsize=(8, 5))
    plt.plot(stages, miou_values, marker="o", linewidth=2)
    plt.title("Validation mIoU")
    plt.xlabel("Training stage")
    plt.ylabel("mIoU (%)")
    plt.grid(True, alpha=0.3)
    save_plot("validation_miou.png")

    # ---------------------------------------------------------
    # 8. Validation Pixel Accuracy graph
    # ---------------------------------------------------------

    accuracy_values = [baseline_val["pixel_accuracy"] * 100]
    accuracy_values.extend([
        item["pixel_accuracy"] * 100 for item in history
    ])

    plt.figure(figsize=(8, 5))
    plt.plot(stages, accuracy_values, marker="o", linewidth=2)
    plt.title("Validation Pixel Accuracy")
    plt.xlabel("Training stage")
    plt.ylabel("Pixel Accuracy (%)")
    plt.grid(True, alpha=0.3)
    save_plot("validation_pixel_accuracy.png")

    # ---------------------------------------------------------
    # 9. Test metrics comparison graph
    # ---------------------------------------------------------

    metric_names = ["mIoU", "Pixel Accuracy"]
    original_values = [
        original_test["mIoU"] * 100,
        original_test["pixel_accuracy"] * 100,
    ]
    finetuned_values = [
        finetuned_test["mIoU"] * 100,
        finetuned_test["pixel_accuracy"] * 100,
    ]

    positions = list(range(len(metric_names)))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        [p - width / 2 for p in positions],
        original_values,
        width,
        label="Original"
    )
    plt.bar(
        [p + width / 2 for p in positions],
        finetuned_values,
        width,
        label="Fine-tuned"
    )
    plt.xticks(positions, metric_names)
    plt.ylabel("Score (%)")
    plt.title("Independent Test: Original vs Fine-tuned")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    save_plot("test_metrics_comparison.png")

    # ---------------------------------------------------------
    # 10. Per-class IoU graph
    # ---------------------------------------------------------

    x = list(range(len(class_names)))
    width = 0.38

    before_values = [
        original_classes.get(name, 0) * 100
        for name in class_names
    ]
    after_values = [
        finetuned_classes.get(name, 0) * 100
        for name in class_names
    ]

    plt.figure(figsize=(15, 7))
    plt.bar(
        [p - width / 2 for p in x],
        before_values,
        width,
        label="Original"
    )
    plt.bar(
        [p + width / 2 for p in x],
        after_values,
        width,
        label="Fine-tuned"
    )
    plt.xticks(x, class_names, rotation=55, ha="right")
    plt.ylabel("IoU (%)")
    plt.title("Per-class IoU on Independent Test Set")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    save_plot("per_class_iou.png")

    # ---------------------------------------------------------
    # 11. Markdown report
    # ---------------------------------------------------------

    miou_change = test["change_percentage_points"]["mIoU"]
    accuracy_change = test["change_percentage_points"]["pixel_accuracy"]

    improved_classes = sum(
        1 for name in class_names
        if (
            name in original_classes
            and name in finetuned_classes
            and finetuned_classes[name] > original_classes[name]
        )
    )

    report = f"""# SegFormer-B3 Fine-tuning Experiment

## 1. Objective

Evaluate domain adaptation of the pretrained SegFormer-B3 model
to the InfraParis dataset for semantic segmentation.

## 2. Dataset

- Dataset: InfraParis
- Original Train candidates: {config['training_candidates_original']}
- Usable Train images: {config['training_candidates_usable']}
- Images used in this experiment: {len(config['training_images'])}
- Excluded fully unlabeled Train images: {config['excluded_unlabeled_training_count']}
- Validation images: {len(config['validation_images'])}
- Independent Test images: {test['test_images_evaluated']}
- Number of semantic classes: {config['num_classes']}
- Unlabeled target pixels mapped to ignore index: {config['ignore_index']}

## 3. Model and Training Configuration

- Base model: `{config['base_model']}`
- Optimizer: {config['optimizer']}
- Learning rate: {config['learning_rate']}
- Weight decay: {config['weight_decay']}
- Batch size: {config['batch_size']}
- Epochs: {config['epochs']}
- Maximum input side: {config['max_input_side']} px
- Random seed: {config['seed']}
- Device: {config['device']}
- Training time: {config['training_time_minutes']} minutes
- Checkpoint selection: {config['checkpoint_selection']}

## 4. Validation Results

| Stage | Training Loss | mIoU (%) | Pixel Accuracy (%) |
|---|---:|---:|---:|
| Baseline | — | {baseline_val['mIoU'] * 100:.2f} | {baseline_val['pixel_accuracy'] * 100:.2f} |
"""

    for item in history:
        report += (
            f"| Epoch {item['epoch']} | {item['train_loss']:.5f} "
            f"| {item['mIoU'] * 100:.2f} "
            f"| {item['pixel_accuracy'] * 100:.2f} |\n"
        )

    report += f"""
## 5. Independent Test Results

| Metric | Original model (%) | Fine-tuned model (%) | Change (pp) |
|---|---:|---:|---:|
| mIoU | {original_test['mIoU'] * 100:.2f} | {finetuned_test['mIoU'] * 100:.2f} | {miou_change:+.2f} |
| Pixel Accuracy | {original_test['pixel_accuracy'] * 100:.2f} | {finetuned_test['pixel_accuracy'] * 100:.2f} | {accuracy_change:+.2f} |

The fine-tuned model improved IoU in {improved_classes} of {len(class_names)} classes.

## 6. Interpretation

The fine-tuned model achieved higher mIoU and Pixel Accuracy than
the original pretrained model on the same independent Test split.

The Test split was not used for gradient updates or checkpoint selection.
The best checkpoint was selected using Validation mIoU.

The results indicate improved adaptation to the InfraParis domain.
The experiment used 1,000 of the 6,566 usable Train images, so it does
not represent training on the complete available training split.

## 7. Reproducibility

- Training configuration: `training_config.json`
- Test metrics: `{TEST_PATH}`
- Checkpoint: `{test['checkpoint']}`
- Training image list: `tables/training_images.csv`

## 8. Generated Materials

See the `tables/` and `graphs/` directories for CSV tables and PNG figures.
"""

    report_path = REPORTS_DIR / "experiment_report.md"
    report_path.write_text(report, encoding="utf-8")

    # ---------------------------------------------------------
    # 12. Summary
    # ---------------------------------------------------------

    print("\n=== EXPERIMENT REPORT GENERATED ===")
    print("Output directory:", OUTPUT_DIR.resolve())
    print("\nTables:")
    for path in sorted(TABLES_DIR.glob("*.csv")):
        print(" ", path)

    print("\nGraphs:")
    for path in sorted(GRAPHS_DIR.glob("*.png")):
        print(" ", path)

    print("\nReport:")
    print(" ", report_path)

    print("\nTest mIoU improvement: {:+.2f} pp".format(miou_change))
    print("Test Pixel Accuracy improvement: {:+.2f} pp".format(
        accuracy_change
    ))


if __name__ == "__main__":
    main()
