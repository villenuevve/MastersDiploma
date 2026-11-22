
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "results" / (
    "segformer_b3_infraparis_full_512_e3_test_metrics.json"
)

OUTPUT = ROOT / "results" / "experiment"
FIGURES = OUTPUT / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

with SOURCE.open("r", encoding="utf-8") as f:
    data = json.load(f)

original = data["original_model"]
fine_tuned = data["fine_tuned_model"]
changes = data["change_percentage_points"]


# 1. Общие метрики
metrics = [
    ("mIoU", original["mIoU"] * 100,
     fine_tuned["mIoU"] * 100,
     changes["mIoU"]),

    ("Pixel Accuracy", original["pixel_accuracy"] * 100,
     fine_tuned["pixel_accuracy"] * 100,
     changes["pixel_accuracy"]),
]

with (OUTPUT / "test_metrics.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "Metric", "Original (%)",
        "Fine-tuned (%)", "Change (pp)"
    ])
    for row in metrics:
        writer.writerow([row[0], *[round(v, 4) for v in row[1:]]])


# 2. IoU по классам
classes = list(original["class_iou"].keys())

class_rows = []

for name in classes:
    before = original["class_iou"][name] * 100
    after = fine_tuned["class_iou"][name] * 100
    delta = changes["class_iou"][name]

    class_rows.append((name, before, after, delta))

with (OUTPUT / "test_iou_by_class.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "Class", "Original IoU (%)",
        "Fine-tuned IoU (%)", "Change (pp)"
    ])
    for row in class_rows:
        writer.writerow([row[0], *[round(v, 4) for v in row[1:]]])


# 3. График общих метрик
labels = [row[0] for row in metrics]
before = [row[1] for row in metrics]
after = [row[2] for row in metrics]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(
    [i - width / 2 for i in x],
    before, width, label="Original"
)
bars2 = ax.bar(
    [i + width / 2 for i in x],
    after, width, label="Fine-tuned"
)

ax.set_title("SegFormer-B3: Test metrics comparison")
ax.set_ylabel("Score (%)")
ax.set_xticks(list(x), labels)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.25)

ax.bar_label(bars1, fmt="%.2f%%", padding=3)
ax.bar_label(bars2, fmt="%.2f%%", padding=3)

fig.tight_layout()
fig.savefig(
    FIGURES / "test_metrics_comparison.png",
    dpi=200, bbox_inches="tight"
)
plt.close(fig)


# 4. График IoU по 19 классам
names = [row[0] for row in class_rows]
before = [row[1] for row in class_rows]
after = [row[2] for row in class_rows]

y = range(len(names))
height = 0.38

fig, ax = plt.subplots(figsize=(12, 9))

ax.barh(
    [i - height / 2 for i in y],
    before, height, label="Original"
)
ax.barh(
    [i + height / 2 for i in y],
    after, height, label="Fine-tuned"
)

ax.set_yticks(list(y), names)
ax.invert_yaxis()
ax.set_xlabel("IoU (%)")
ax.set_title("SegFormer-B3: Test IoU by class")
ax.set_xlim(0, 105)
ax.legend()
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIGURES / "test_iou_by_class.png",
    dpi=200, bbox_inches="tight"
)
plt.close(fig)


print("EXPERIMENT RESULTS EXPORTED")
print("Output:", OUTPUT)
print("Test images:", data["test_images_evaluated"])
print("mIoU:", round(fine_tuned["mIoU"] * 100, 2), "%")
print("Pixel Accuracy:",
      round(fine_tuned["pixel_accuracy"] * 100, 2), "%")
print("Files:")
for path in sorted(OUTPUT.rglob("*")):
    if path.is_file():
        print(path.relative_to(ROOT))
