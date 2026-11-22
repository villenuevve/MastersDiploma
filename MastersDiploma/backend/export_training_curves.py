import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


CONFIG_PATH = Path(
    "models/segformer_b3_infraparis_full_512_e3/training_config.json"
)
OUTPUT_DIR = Path("results/experiment")
FIGURES_DIR = OUTPUT_DIR / "figures"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = json.load(file)

history = config["validation_history"]
baseline = config["baseline_validation"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# CSV с динамикой обучения
csv_path = OUTPUT_DIR / "training_history.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch",
        "train_loss",
        "validation_mIoU",
        "validation_pixel_accuracy",
    ])

    for item in history:
        writer.writerow([
            item["epoch"],
            item["train_loss"],
            item["mIoU"],
            item["pixel_accuracy"],
        ])

# Данные
epochs = [item["epoch"] for item in history]
losses = [item["train_loss"] for item in history]
miou = [item["mIoU"] * 100 for item in history]
accuracy = [item["pixel_accuracy"] * 100 for item in history]

baseline_miou = baseline["mIoU"] * 100
baseline_accuracy = baseline["pixel_accuracy"] * 100

# График Train Loss
plt.figure(figsize=(9, 5))
plt.plot(epochs, losses, marker="o", linewidth=2)
plt.xticks(epochs)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("SegFormer-B3: Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "training_loss.png", dpi=200)
plt.close()

# График Validation Metrics
plt.figure(figsize=(9, 5))
plt.plot(
    [0] + epochs,
    [baseline_miou] + miou,
    marker="o",
    linewidth=2,
    label="Validation mIoU",
)
plt.plot(
    [0] + epochs,
    [baseline_accuracy] + accuracy,
    marker="s",
    linewidth=2,
    label="Validation Pixel Accuracy",
)
plt.xticks([0] + epochs, ["Baseline"] + [str(e) for e in epochs])
plt.xlabel("Training Stage")
plt.ylabel("Metric (%)")
plt.title("SegFormer-B3: Validation Performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "validation_metrics.png", dpi=200)
plt.close()

print("Training curves exported successfully.")
print(f"CSV: {csv_path}")
print(f"Loss graph: {FIGURES_DIR / 'training_loss.png'}")
print(f"Validation graph: {FIGURES_DIR / 'validation_metrics.png'}")
