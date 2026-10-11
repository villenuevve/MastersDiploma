from pathlib import Path
from collections import Counter
import csv
import html
import math

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
]

COLORS = [
    "#804080", "#f4a460", "#696969", "#8b6b61", "#c9a66b",
    "#b0b0b0", "#ff0000", "#ff8c00", "#008000", "#9acd32",
    "#4682b4", "#ff00ff", "#800080", "#0000ff", "#000080",
    "#00ffff", "#800000", "#dc143c", "#d6bd00"
]

mask_files = sorted(RESULTS.glob("*_class_ids.png"))

if not mask_files:
    raise SystemExit(
        f"Не знайдено масок *_class_ids.png у {RESULTS}. "
        "Перевір шлях до результатів."
    )

per_image = []
total_counts = np.zeros(len(CLASSES), dtype=np.int64)
class_image_frequency = np.zeros(len(CLASSES), dtype=np.int64)
total_valid_pixels = 0
total_ignored_pixels = 0
total_pixels = 0

for path in mask_files:
    with Image.open(path) as im:
        arr = np.asarray(im)

    # ID-маска має містити числові ID класів, а не кольорову візуалізацію.
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        print(f"Пропущено маску з непідтримуваним форматом: {path.name}")
        continue

    arr = arr.astype(np.int32, copy=False)
    pixels = arr.size
    counts = np.bincount(arr.ravel(), minlength=256)
    counts = counts[:256]

    class_counts = np.array([
        int(counts[i]) if i < len(counts) else 0
        for i in range(len(CLASSES))
    ], dtype=np.int64)

    valid_pixels = int(class_counts.sum())
    ignored_pixels = pixels - valid_pixels

    if valid_pixels == 0:
        print(f"Пропущено маску без ID класів 0–18: {path.name}")
        continue

    percentages = class_counts / valid_pixels * 100
    present = class_counts > 0
    probabilities = class_counts[present] / valid_pixels
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    max_entropy = math.log2(len(CLASSES))
    normalized_entropy = entropy / max_entropy if max_entropy else 0.0

    dominant_id = int(np.argmax(class_counts))
    row = {
        "image": path.name.removesuffix("_class_ids.png"),
        "mask_file": path.name,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "total_pixels": int(pixels),
        "valid_pixels": valid_pixels,
        "ignored_pixels": int(ignored_pixels),
        "class_count": int(present.sum()),
        "dominant_class": CLASSES[dominant_id],
        "dominant_class_percent": round(float(percentages[dominant_id]), 4),
        "entropy_bits": round(entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }

    for i, name in enumerate(CLASSES):
        row[f"{name}_pixels"] = int(class_counts[i])
        row[f"{name}_percent"] = round(float(percentages[i]), 4)

    per_image.append(row)
    total_counts += class_counts
    class_image_frequency += present.astype(np.int64)
    total_valid_pixels += valid_pixels
    total_ignored_pixels += ignored_pixels
    total_pixels += pixels

if not per_image:
    raise SystemExit(
        "Маски знайдені, але жодна не містить коректних ID класів 0–18."
    )

per_image.sort(key=lambda r: r["image"].lower())

# CSV: повна статистика кожного зображення.
per_image_path = RESULTS / "prediction_analysis_per_image.csv"
with per_image_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
    writer.writeheader()
    writer.writerows(per_image)

# CSV: агрегована статистика класів.
class_rows = []
for i, name in enumerate(CLASSES):
    class_rows.append({
        "class_id": i,
        "class_name": name,
        "total_pixels": int(total_counts[i]),
        "percent_of_valid_pixels": round(
            float(total_counts[i] / total_valid_pixels * 100), 6
        ),
        "images_containing_class": int(class_image_frequency[i]),
        "percent_of_images_containing_class": round(
            float(class_image_frequency[i] / len(per_image) * 100), 4
        ),
    })

class_path = RESULTS / "prediction_analysis_class_statistics.csv"
with class_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
    writer.writeheader()
    writer.writerows(class_rows)

# HTML-звіт із графіком розподілу класів.
max_percent = max(r["percent_of_valid_pixels"] for r in class_rows) or 1
bars = []
for r in class_rows:
    i = r["class_id"]
    pct = r["percent_of_valid_pixels"]
    width = pct / max_percent * 100
    bars.append(
        f'<div class="barrow"><span class="name">{html.escape(r["class_name"])}</span>'
        f'<div class="track"><div class="bar" style="width:{width:.2f}%;'
        f'background:{COLORS[i]}"></div></div>'
        f'<span class="num">{pct:.2f}%</span></div>'
    )

top_rows = sorted(
    per_image, key=lambda r: r["normalized_entropy"], reverse=True
)[:20]

table_rows = []
for r in top_rows:
    table_rows.append(
        "<tr>"
        f"<td>{html.escape(r['image'])}</td>"
        f"<td>{r['width']} × {r['height']}</td>"
        f"<td>{r['class_count']}/19</td>"
        f"<td>{html.escape(r['dominant_class'])} "
        f"({r['dominant_class_percent']:.1f}%)</td>"
        f"<td>{r['normalized_entropy']:.3f}</td>"
        f"<td>{r['valid_pixels']:,}</td>"
        "</tr>"
    )

mean_classes = float(np.mean([r["class_count"] for r in per_image]))
mean_dominant = float(np.mean([r["dominant_class_percent"] for r in per_image]))
mean_entropy = float(np.mean([r["normalized_entropy"] for r in per_image]))
mean_coverage = total_valid_pixels / total_pixels * 100 if total_pixels else 0

report = f"""<!doctype html>
<html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Аналіз наявних прогнозів сегментації</title>
<style>
body{{font:15px Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 18px;color:#202938}}
h1{{font-size:27px}}h2{{margin-top:32px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.card{{background:#f1f5f9;border-radius:8px;padding:16px}}
.value{{font-size:25px;font-weight:bold;margin-top:8px}}
.note{{background:#fff7ed;padding:14px;border-left:4px solid #f59e0b;line-height:1.5}}
.barrow{{display:grid;grid-template-columns:130px minmax(80px,1fr) 85px;gap:10px;align-items:center;margin:8px 0}}
.name{{font-size:13px}}.track{{height:15px;background:#e5e7eb;border-radius:3px;overflow:hidden}}
.bar{{height:100%}}.num{{text-align:right;font-variant-numeric:tabular-nums}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}
th{{background:#f1f5f9}}.scroll{{overflow-x:auto}}
</style></head><body>
<h1>Аналіз прогнозів семантичної сегментації</h1>
<p>Звіт побудовано з уже збережених ID-масок. Повторного запуску моделі не було.</p>
<div class="cards">
<div class="card">Оброблені маски<div class="value">{len(per_image)}</div></div>
<div class="card">Пікселі з ID класів 0–18<div class="value">{total_valid_pixels:,}</div></div>
<div class="card">Середня кількість класів на фото<div class="value">{mean_classes:.2f}/19</div></div>
<div class="card">Середня частка домінантного класу<div class="value">{mean_dominant:.2f}%</div></div>
<div class="card">Середня нормалізована ентропія<div class="value">{mean_entropy:.3f}</div></div>
<div class="card">Пікселі ID 0–18 від усіх пікселів<div class="value">{mean_coverage:.2f}%</div></div>
</div>
<div class="note"><b>Обмеження інтерпретації.</b>
Ці показники описують розподіл класів у прогнозах, а не точність моделі.
Вони не є mIoU, Pixel Accuracy або доказом правильного розпізнавання.
Нормалізована ентропія показує різноманітність розподілу класів, а не впевненість моделі.
Пікселі, що не мають ID 0–18, не входять до відсотків класів.</div>
<h2>Розподіл класів за кількістю пікселів</h2>
{''.join(bars)}
<h2>20 фото з найбільш різноманітним розподілом класів</h2>
<p>Відсортовано за нормалізованою ентропією прогнозованих класів.</p>
<div class="scroll"><table><thead><tr>
<th>Фото</th><th>Розмір</th><th>Класів</th><th>Домінантний клас</th>
<th>Ентропія</th><th>Валідні пікселі</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p>CSV зі статистикою по кожному фото: prediction_analysis_per_image.csv</p>
<p>CSV зі статистикою класів: prediction_analysis_class_statistics.csv</p>
</body></html>"""

report_path = RESULTS / "prediction_analysis_report.html"
report_path.write_text(report, encoding="utf-8")

print("\nАНАЛІЗ ЗАВЕРШЕНО — модель повторно не запускалася")
print(f"Масок проаналізовано: {len(per_image)}")
print(f"Пікселів загалом: {total_pixels:,}")
print(f"Пікселів з ID класів 0–18: {total_valid_pixels:,}")
print(f"Пікселів поза ID 0–18: {total_ignored_pixels:,}")
print(f"Середня кількість класів на фото: {mean_classes:.2f}/19")
print(f"Середня частка домінантного класу: {mean_dominant:.2f}%")
print("\nСтворено файли:")
print(per_image_path)
print(class_path)
print(report_path)
