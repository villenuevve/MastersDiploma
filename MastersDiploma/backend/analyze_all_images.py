import csv
import html
import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "new_images"
OUT = ROOT / "results"
API_URL = "http://127.0.0.1:8000/predict"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

OUT.mkdir(parents=True, exist_ok=True)
images = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)

if not images:
    raise SystemExit(f"Изображения не найдены: {IMAGE_DIR}")

rows = []
class_pixels = defaultdict(int)
class_images = defaultdict(int)
class_percentages = defaultdict(list)
errors = []
total_start = time.perf_counter()

print(f"Изображений найдено: {len(images)}")

for i, path in enumerate(images, 1):
    start = time.perf_counter()
    try:
        boundary = "----SegmentationBoundary7MA4YWxk"
        data = path.read_bytes()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

        request = Request(
            API_URL,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))

        elapsed = (time.perf_counter() - start) * 1000
        detected = result.get("detected_classes", [])
        total_pixels = sum(int(c.get("pixel_count", 0)) for c in detected)

        image_classes = []
        for c in detected:
            name = str(c.get("name", "unknown"))
            pixels = int(c.get("pixel_count", 0))
            percentage = float(c.get("percentage", 0))
            class_pixels[name] += pixels
            class_images[name] += 1
            class_percentages[name].append(percentage)
            image_classes.append({
                "name": name,
                "pixels": pixels,
                "percentage": percentage,
            })

        rows.append({
            "image": str(path.relative_to(ROOT)),
            "status": "ok",
            "width": result.get("width", ""),
            "height": result.get("height", ""),
            "classes_count": result.get("class_count", len(detected)),
            "pixels_counted": total_pixels,
            "processing_ms": round(elapsed, 2),
            "classes": image_classes,
        })
        print(f"[{i}/{len(images)}] OK {path.name}: "
              f"{len(image_classes)} classes, {elapsed:.0f} ms")

    except Exception as exc:
        errors.append({"image": str(path), "error": str(exc)})
        rows.append({
            "image": str(path.relative_to(ROOT)),
            "status": "error",
            "width": "",
            "height": "",
            "classes_count": "",
            "pixels_counted": "",
            "processing_ms": "",
            "classes": [],
        })
        print(f"[{i}/{len(images)}] ERROR {path.name}: {exc}")

successful = [r for r in rows if r["status"] == "ok"]
total_seconds = time.perf_counter() - total_start
class_names = sorted(class_pixels, key=lambda n: class_pixels[n], reverse=True)

# Per-image CSV
with (OUT / "all_images_analysis.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image", "status", "width", "height", "classes_count",
        "pixels_counted", "processing_ms", "detected_classes"
    ])
    for r in rows:
        details = "; ".join(
            f'{c["name"]}: {c["percentage"]:.2f}%'
            for c in r["classes"]
        )
        writer.writerow([
            r["image"], r["status"], r["width"], r["height"],
            r["classes_count"], r["pixels_counted"],
            r["processing_ms"], details
        ])

# Aggregated class CSV
with (OUT / "all_images_class_statistics.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "class", "total_predicted_pixels", "images_containing_class",
        "image_presence_percent", "mean_area_percent_when_present"
    ])
    for name in class_names:
        percentages = class_percentages[name]
        writer.writerow([
            name,
            class_pixels[name],
            class_images[name],
            round(class_images[name] / max(len(successful), 1) * 100, 2),
            round(sum(percentages) / len(percentages), 2),
        ])

def esc(value):
    return html.escape(str(value))

mean_ms = (
    sum(r["processing_ms"] for r in successful) / len(successful)
    if successful else 0
)
class_rows = ""
for name in class_names:
    percentages = class_percentages[name]
    presence = class_images[name] / max(len(successful), 1) * 100
    mean_area = sum(percentages) / len(percentages)
    class_rows += (
        f"<tr><td>{esc(name)}</td>"
        f"<td>{class_pixels[name]:,}</td>"
        f"<td>{class_images[name]} / {len(successful)}</td>"
        f"<td>{presence:.1f}%</td>"
        f"<td>{mean_area:.2f}%</td></tr>"
    )

image_rows = ""
for r in rows:
    details = ", ".join(
        f'{c["name"]} ({c["percentage"]:.1f}%)'
        for c in sorted(r["classes"], key=lambda c: c["percentage"], reverse=True)
    )
    image_rows += (
        f"<tr><td>{esc(Path(r['image']).name)}</td>"
        f"<td>{esc(r['status'])}</td>"
        f"<td>{esc(r['classes_count'])}</td>"
        f"<td>{esc(r['processing_ms'])}</td>"
        f"<td>{esc(details) if details else '—'}</td></tr>"
    )

report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>All Images Segmentation Analysis</title>
<style>
body{{font:15px -apple-system,BlinkMacSystemFont,sans-serif;margin:32px auto;
max-width:1200px;padding:0 20px;color:#222}}
h1,h2{{margin-top:30px}} .cards{{display:flex;flex-wrap:wrap;gap:12px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:16px;min-width:150px}}
.value{{font-size:25px;font-weight:650;margin-top:8px}}
table{{border-collapse:collapse;width:100%;margin:16px 0 36px}}
th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}
th{{background:#f4f4f4;position:sticky;top:0}}
td:last-child{{overflow-wrap:anywhere}} .note{{color:#555}}
</style></head><body>
<h1>Segmentation analysis — all images</h1>
<p class="note">Model predictions from the application's /predict endpoint.
These are predicted class distributions, not ground-truth accuracy measurements.</p>
<div class="cards">
<div class="card">Images found<div class="value">{len(images)}</div></div>
<div class="card">Processed successfully<div class="value">{len(successful)}</div></div>
<div class="card">Errors<div class="value">{len(errors)}</div></div>
<div class="card">Mean processing time<div class="value">{mean_ms:.0f} ms</div></div>
<div class="card">Total elapsed time<div class="value">{total_seconds / 60:.1f} min</div></div>
</div>
<h2>Class statistics across the collection</h2>
<p class="note">Pixel totals are summed from each image's predicted class counts.
Image presence is the percentage of successfully processed images in which the class appeared.
Mean area is averaged over images where the class appeared.</p>
<table><thead><tr><th>Class</th><th>Predicted pixels</th>
<th>Images containing class</th><th>Image presence</th>
<th>Mean area when present</th></tr></thead><tbody>{class_rows}</tbody></table>
<h2>Per-image results</h2>
<table><thead><tr><th>Image</th><th>Status</th><th>Classes</th>
<th>Processing (ms)</th><th>Predicted classes and area</th></tr></thead>
<tbody>{image_rows}</tbody></table>
<p>CSV exports: all_images_analysis.csv and all_images_class_statistics.csv.</p>
</body></html>"""

report_path = OUT / "all_images_analysis.html"
report_path.write_text(report, encoding="utf-8")

(OUT / "all_images_analysis_errors.json").write_text(
    json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("\nFinished.")
print(f"HTML report: {report_path}")
print(f"Per-image CSV: {OUT / 'all_images_analysis.csv'}")
print(f"Class statistics CSV: {OUT / 'all_images_class_statistics.csv'}")
print(f"Successful: {len(successful)}/{len(images)}")
