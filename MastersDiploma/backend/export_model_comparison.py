import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
EXPERIMENT_DIR = RESULTS / "experiment"
FIGURES_DIR = EXPERIMENT_DIR / "figures"

EXPERIMENTS = {
    "1000": RESULTS / "segformer_b3_infraparis_1000_512_e3_test_metrics.json",
    "full": RESULTS / "segformer_b3_infraparis_full_512_e3_test_metrics.json",
}

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


data_1000 = load_json(EXPERIMENTS["1000"])
data_full = load_json(EXPERIMENTS["full"])

# Проверяем, что сравниваются одинаковые тестовые изображения.
assert data_1000["test_images_evaluated"] == data_full["test_images_evaluated"]
assert data_1000["test_image_ids"] == data_full["test_image_ids"]

# Проверяем совпадение исходной модели в двух экспериментах.
for metric in ("mIoU", "pixel_accuracy"):
    assert abs(
        data_1000["original_model"][metric]
        - data_full["original_model"][metric]
    ) < 1e-10


original = data_full["original_model"]
fine_1000 = data_1000["fine_tuned_model"]
fine_full = data_full["fine_tuned_model"]

models = [
    ("Original SegFormer-B3", original),
    ("Fine-tuned, 1,000 images", fine_1000),
    ("Fine-tuned, 6,566 images", fine_full),
]

# 1. Сводная таблица моделей.
comparison_path = EXPERIMENT_DIR / "model_comparison.csv"

with comparison_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Model",
        "Training images",
        "Test images",
        "mIoU (%)",
        "Pixel Accuracy (%)",
    ])

    writer.writerow([
        models[0][0], 0, data_full["test_images_evaluated"],
        original["mIoU"] * 100,
        original["pixel_accuracy"] * 100,
    ])

    writer.writerow([
        models[1][0], 1000, data_1000["test_images_evaluated"],
        fine_1000["mIoU"] * 100,
        fine_1000["pixel_accuracy"] * 100,
    ])

    writer.writerow([
        models[2][0], 6566, data_full["test_images_evaluated"],
        fine_full["mIoU"] * 100,
        fine_full["pixel_accuracy"] * 100,
    ])


# 2. Сравнение IoU по каждому классу.
class_path = EXPERIMENT_DIR / "class_iou_comparison.csv"

with class_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Class",
        "Original IoU (%)",
        "Fine-tuned IoU (%)",
        "Improvement (pp)",
    ])

    for class_name, original_iou in original["class_iou"].items():
        fine_iou = fine_full["class_iou"][class_name]

        writer.writerow([
            class_name,
            original_iou * 100,
            fine_iou * 100,
            (fine_iou - original_iou) * 100,
        ])


# 3. График сравнения моделей.
labels = ["Original", "Fine-tuned\n1,000 images", "Fine-tuned\n6,566 images"]
miou_values = [model[1]["mIoU"] * 100 for model in models]
accuracy_values = [
    model[1]["pixel_accuracy"] * 100 for model in models
]

x = list(range(len(labels)))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    [i - width / 2 for i in x],
    miou_values,
    width,
    label="mIoU",
)

bars2 = ax.bar(
    [i + width / 2 for i in x],
    accuracy_values,
    width,
    label="Pixel Accuracy",
)

ax.set_title("SegFormer-B3: Test Performance Comparison")
ax.set_ylabel("Metric (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.25)

ax.bar_label(bars1, fmt="%.2f%%", padding=3)
ax.bar_label(bars2, fmt="%.2f%%", padding=3)

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "model_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# 4. График IoU по классам: исходная и дообученная модели.
class_names = list(original["class_iou"].keys())

original_values = [
    original["class_iou"][name] * 100 for name in class_names
]
fine_values = [
    fine_full["class_iou"][name] * 100 for name in class_names
]

positions = list(range(len(class_names)))
height = 0.38

fig, ax = plt.subplots(figsize=(12, 9))

ax.barh(
    [p - height / 2 for p in positions],
    original_values,
    height,
    label="Original",
)

ax.barh(
    [p + height / 2 for p in positions],
    fine_values,
    height,
    label="Fine-tuned",
)

ax.set_yticks(positions)
ax.set_yticklabels(class_names)
ax.invert_yaxis()
ax.set_xlabel("IoU (%)")
ax.set_title("SegFormer-B3: Test IoU by Class")
ax.set_xlim(0, 105)
ax.legend()
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "class_iou_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


print("Model comparison exported successfully.")
print(f"Test images: {data_full['test_images_evaluated']}")
print(f"Original mIoU: {original['mIoU'] * 100:.2f}%")
print(f"Fine-tuned mIoU (1000): {fine_1000['mIoU'] * 100:.2f}%")
print(f"Fine-tuned mIoU (full): {fine_full['mIoU'] * 100:.2f}%")
print(f"mIoU improvement: {(fine_full['mIoU'] - original['mIoU']) * 100:+.2f} pp")
print()
print("Created files:")
print(comparison_path)
print(class_path)
print(FIGURES_DIR / "model_comparison.png")
print(FIGURES_DIR / "class_iou_comparison.png")
