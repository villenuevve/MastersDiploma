from pathlib import Path
import csv
import json

import numpy as np
from PIL import Image, ImageOps, ImageDraw

from backend.metrics import CITYSCAPES_CLASSES
from backend.segmentation import predict_mask
from backend.evaluate_dataset import convert_ground_truth, ROOT, IGNORE_INDEX

OUT = Path("results")
VIS = OUT / "visual_comparisons"
PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70),
    (102, 102, 156), (190, 153, 153), (153, 153, 153),
    (250, 170, 30), (220, 220, 0), (107, 142, 35),
    (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]

def colorize(mask):
    arr = np.asarray(mask)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        rgb[arr == idx] = color
    rgb[arr == IGNORE_INDEX] = (0, 0, 0)
    return Image.fromarray(rgb)

def save_confusion_matrix(matrix, path):
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    cell = 36
    left = 150
    top = 120
    canvas = Image.new("RGB", (left + n * cell + 20, top + n * cell + 20), "white")
    draw = ImageDraw.Draw(canvas)
    maximum = max(float(matrix.max()), 1.0)

    for i, name in enumerate(CITYSCAPES_CLASSES):
        draw.text((5, top + i * cell + 10), name, fill="black")
        draw.text((left + i * cell + 3, 5), str(i), fill="black")

    for row in range(n):
        for col in range(n):
            value = matrix[row, col]
            intensity = int(255 * (1 - value / maximum))
            x0, y0 = left + col * cell, top + row * cell
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1),
                           fill=(255, intensity, intensity),
                           outline=(210, 210, 210))
            if value:
                draw.text((x0 + 3, y0 + 10), str(int(value)), fill="black")
    canvas.save(path)

def main():
    OUT.mkdir(exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)

    metrics_path = OUT / "evaluation_metrics_all.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Не найден results/evaluation_metrics_all.json. "
            "Сначала запусти: python -m backend.evaluate_dataset"
        )

    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Экспорт метрик и таблиц для каждого доступного split.
    summary_rows = []
    per_class_rows = []
    for split, metrics in data.items():
        summary_rows.append({
            "split": split,
            "images_evaluated": metrics.get("images_evaluated", ""),
            "evaluated_pixels": metrics.get("evaluated_pixels", ""),
            "mIoU_percent": round(metrics.get("mean_iou", 0) * 100, 2),
            "pixel_accuracy_percent": round(metrics.get("pixel_accuracy", 0) * 100, 2),
        })
        for class_name, score in metrics.get("per_class_iou", {}).items():
            per_class_rows.append({
                "split": split,
                "class": class_name,
                "IoU_percent": "" if score is None else round(score * 100, 2),
            })

        matrix = metrics.get("confusion_matrix")
        if matrix:
            csv_path = OUT / f"confusion_matrix_{split}.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["actual\\predicted"] + CITYSCAPES_CLASSES)
                for name, row in zip(CITYSCAPES_CLASSES, matrix):
                    writer.writerow([name] + row)
            save_confusion_matrix(matrix, OUT / f"confusion_matrix_{split}.png")

    for filename, rows, fields in [
        ("evaluation_summary_all_splits.csv", summary_rows,
         ["split", "images_evaluated", "evaluated_pixels", "mIoU_percent", "pixel_accuracy_percent"]),
        ("evaluation_per_class_all_splits.csv", per_class_rows,
         ["split", "class", "IoU_percent"]),
    ]:
        with (OUT / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # Пересоздаём визуальные сравнения для каждого изображения,
    # перечисленного в уже рассчитанных метриках.
    comparison_rows = []
    for split in ("train", "val"):
        metrics = data.get(split)
        if not metrics:
            continue

        for filename in metrics.get("files", []):
            image_path = ROOT / "images" / split / filename
            mask_path = ROOT / "masks" / split / filename
            if not image_path.is_file() or not mask_path.is_file():
                print(f"Пропущен отсутствующий файл: {image_path}")
                continue

            with Image.open(image_path) as im:
                original = im.convert("RGB")
                pred = np.asarray(predict_mask(original)).astype(np.uint8)
            with Image.open(mask_path) as im:
                target = convert_ground_truth(np.asarray(im.convert("L")))

            if pred.shape != target.shape:
                print(f"Пропущено из-за несовпадения размеров: {filename}")
                continue

            gt_rgb = colorize(target)
            pred_rgb = colorize(pred)
            overlay = Image.blend(original, pred_rgb, 0.45)

            split_dir = VIS / split
            split_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(filename).stem

            original.save(split_dir / f"{stem}_original.png")
            gt_rgb.save(split_dir / f"{stem}_ground_truth.png")
            pred_rgb.save(split_dir / f"{stem}_prediction.png")
            overlay.save(split_dir / f"{stem}_overlay.png")

            # Компактное изображение: оригинал, ground truth, prediction, overlay.
            panels = [original, gt_rgb, pred_rgb, overlay]
            thumb_width = 480
            thumb_height = max(1, round(original.height * thumb_width / original.width))
            labels = ["Original", "Ground truth", "Prediction", "Overlay"]
            sheet = Image.new("RGB", (thumb_width * 2, (thumb_height + 28) * 2), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (panel, label) in enumerate(zip(panels, labels)):
                panel = panel.resize((thumb_width, thumb_height))
                x = (i % 2) * thumb_width
                y = (i // 2) * (thumb_height + 28)
                sheet.paste(panel, (x, y + 28))
                draw.text((x + 8, y + 7), label, fill="black")
            sheet.save(split_dir / f"{stem}_comparison.png")

            comparison_rows.append({
                "split": split,
                "filename": filename,
                "original": str(split_dir / f"{stem}_original.png"),
                "ground_truth": str(split_dir / f"{stem}_ground_truth.png"),
                "prediction": str(split_dir / f"{stem}_prediction.png"),
                "overlay": str(split_dir / f"{stem}_overlay.png"),
                "comparison": str(split_dir / f"{stem}_comparison.png"),
            })
            print(f"[{split}] Экспортировано: {filename}")

    with (OUT / "visual_comparisons_index.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = ["split", "filename", "original", "ground_truth",
                  "prediction", "overlay", "comparison"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    report = [
        "# Полный отчёт оценки семантической сегментации",
        "",
        "> `all` объединяет train и validation; это не независимый тестовый набор.",
        "",
        "| Split | Изображения | Пиксели | mIoU | Pixel Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['split']} | {row['images_evaluated']} | "
            f"{row['evaluated_pixels']} | {row['mIoU_percent']:.2f}% | "
            f"{row['pixel_accuracy_percent']:.2f}% |"
        )
    report += [
        "",
        "## Сгенерированные файлы",
        "",
        "- `evaluation_summary_all_splits.csv` — сводные метрики.",
        "- `evaluation_per_class_all_splits.csv` — IoU по классам для каждого split.",
        "- `confusion_matrix_<split>.csv` — матрицы ошибок.",
        "- `confusion_matrix_<split>.png` — визуализации матриц ошибок.",
        "- `visual_comparisons/` — исходные изображения, ground truth, предсказания, overlay и сравнительные панели.",
        "- `visual_comparisons_index.csv` — список визуальных файлов.",
    ]
    (OUT / "evaluation_full_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("\nЭкспорт завершён. Файлы сохранены в results/")
    print(f"Визуальных сравнений: {len(comparison_rows)}")
    print(f"Открой папку командой: open {OUT.resolve()}")

if __name__ == "__main__":
    main()
