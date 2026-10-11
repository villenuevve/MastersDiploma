import csv
import json
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def load_json(name):
    p = OUT / name
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)

validation = load_json("evaluation_metrics.json")
combined = load_json("evaluation_metrics_all.json")

if not validation:
    raise SystemExit("ERROR: results/evaluation_metrics.json not found")

classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle"
]

def metric_row(label, data):
    return {
        "Split": label,
        "mIoU_percent": round(float(data["mean_iou"]) * 100, 2)
            if data.get("mean_iou") is not None else "",
        "Pixel_Accuracy_percent": round(float(data["pixel_accuracy"]) * 100, 2)
            if data.get("pixel_accuracy") is not None else "",
        "Images": data.get("images_evaluated", ""),
        "Valid_Pixels": data.get("evaluated_pixels", "")
    }

splits = []
for key, label in [("train", "Train"), ("val", "Validation"), ("all", "All images")]:
    data = combined.get(key)
    if isinstance(data, dict) and "mean_iou" in data:
        splits.append((label, data))

if not any(label == "Validation" for label, _ in splits):
    splits.append(("Validation", validation))

#Summary metrics
with (OUT / "evaluation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    fields = ["Split", "mIoU_percent", "Pixel_Accuracy_percent", "Images", "Valid_Pixels"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for label, data in splits:
        w.writerow(metric_row(label, data))

#Per-class IoU
per_class = validation.get("per_class_iou", {})
class_rows = []
for name in classes:
    score = per_class.get(name)
    class_rows.append({
        "Class": name.title(),
        "IoU": score if score is not None else "",
        "IoU_percent": round(float(score) * 100, 2) if score is not None else ""
    })

with (OUT / "evaluation_per_class_iou.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["Class", "IoU", "IoU_percent"])
    w.writeheader()
    w.writerows(class_rows)

# 3. Confusion matrix CSV
matrix = validation.get("confusion_matrix", [])
if matrix and len(matrix) == len(classes):
    with (OUT / "confusion_matrix_validation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Actual / Predicted"] + [c.title() for c in classes])
        for name, row in zip(classes, matrix):
            w.writerow([name.title()] + row)
else:
    print("WARNING: validation confusion matrix missing or not 19x19")

#HTML report with all tables and colored confusion matrix
def fmt_percent(value):
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"

summary_html = "".join(
    "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in metric_row(label, data).values()) + "</tr>"
    for label, data in splits
)
class_html = "".join(
    f"<tr><td>{html.escape(r['Class'])}</td><td>{fmt_percent(r['IoU']) if r['IoU'] != '' else 'N/A'}</td></tr>"
    for r in class_rows
)

matrix_html = ""
if matrix and len(matrix) == len(classes):
    max_value = max(max(map(int, row)) for row in matrix)
    headers = "".join(f"<th>{html.escape(c.title())}</th>" for c in classes)
    rows = []
    for i, row in enumerate(matrix):
        cells = []
        for j, raw in enumerate(row):
            value = int(raw)
            strength = (value / max_value) ** 0.5 if max_value else 0
            if i == j:
                bg = f"rgba(30,130,76,{0.12 + 0.78 * strength:.3f})"
            else:
                bg = f"rgba(190,65,65,{0.04 + 0.72 * strength:.3f})"
            cells.append(
                f'<td title="Actual: {html.escape(classes[i])}; Predicted: {html.escape(classes[j])}; Pixels: {value:,}" '
                f'style="background:{bg}">{value:,}' if value else
                f'<td title="Actual: {html.escape(classes[i])}; Predicted: {html.escape(classes[j])}; Pixels: 0">·'
            )
            cells[-1] += "</td>"
        rows.append(f"<tr><th>{html.escape(classes[i].title())}</th>{''.join(cells)}</tr>")
    matrix_html = f"""
    <h2>Confusion Matrix — Validation</h2>
    <p>Rows are actual classes; columns are predicted classes.
    Green diagonal cells indicate correct predictions; red off-diagonal cells indicate errors.
    Hover over a cell to see its pixel count.</p>
    <div class="matrix"><table><thead><tr><th>Actual / Predicted</th>{headers}</tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    """

split_table = f"""
<table><thead><tr>
<th>Split</th><th>mIoU</th><th>Pixel Accuracy</th><th>Images</th><th>Valid pixels</th>
</tr></thead><tbody>{''.join(
    f"<tr><td>{html.escape(label)}</td><td>{fmt_percent(data.get('mean_iou'))}</td>"
    f"<td>{fmt_percent(data.get('pixel_accuracy'))}</td>"
    f"<td>{data.get('images_evaluated', 'N/A')}</td><td>{data.get('evaluated_pixels', 'N/A'):,}</td></tr>"
    for label, data in splits
)}</tbody></table>
"""

report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Segmentation Evaluation</title>
<style>
body{{font:15px -apple-system,BlinkMacSystemFont,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#222}}
h1,h2{{margin-top:30px}} table{{border-collapse:collapse;width:100%;margin:14px 0 28px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
th{{background:#f3f4f6}} .matrix{{overflow:auto;max-height:75vh}}
.matrix table{{width:max-content;font-size:11px}} .matrix th,.matrix td{{padding:5px;min-width:48px;text-align:center;white-space:nowrap}}
.note{{background:#f5f5f5;padding:12px;border-radius:8px}} a{{color:#245caa}}
</style></head><body>
<h1>Semantic Segmentation Evaluation</h1>
<p>Model evaluation against Cityscapes ground-truth masks.</p>
<div class="note">
<strong>Validation mIoU:</strong> {fmt_percent(validation.get('mean_iou'))}<br>
<strong>Pixel Accuracy:</strong> {fmt_percent(validation.get('pixel_accuracy'))}<br>
<strong>Images evaluated:</strong> {validation.get('images_evaluated', 'N/A')}<br>
<strong>Valid pixels:</strong> {validation.get('evaluated_pixels', 0):,}
</div>
<h2>Dataset Split Comparison</h2>{split_table}
<h2>Validation IoU by Class</h2>
<table><thead><tr><th>Class</th><th>IoU</th></tr></thead><tbody>{class_html}</tbody></table>
{matrix_html}
<h2>Limitations</h2>
<p>Evaluation uses a small subset of four training and four validation images.
Combined results include both splits and are not an independent test result.
N/A indicates that a class has no valid ground-truth pixels in the evaluated data.</p>
</body></html>"""

(OUT / "evaluation_report.html").write_text(report, encoding="utf-8")

md = [
    "# Semantic Segmentation Evaluation", "",
    "## Dataset split comparison", "",
    "| Split | mIoU | Pixel Accuracy | Images | Valid pixels |",
    "|---|---:|---:|---:|---:|"
]
for label, data in splits:
    md.append(
        f"| {label} | {fmt_percent(data.get('mean_iou'))} | "
        f"{fmt_percent(data.get('pixel_accuracy'))} | "
        f"{data.get('images_evaluated', 'N/A')} | {data.get('evaluated_pixels', 'N/A')} |"
    )
md += ["", "## Validation IoU by class", "", "| Class | IoU |", "|---|---:|"]
for r in class_rows:
    md.append(f"| {r['Class']} | {fmt_percent(r['IoU']) if r['IoU'] != '' else 'N/A'} |")
md += [
    "", "## Confusion Matrix", "",
    "See `confusion_matrix_validation.csv` or `evaluation_report.html`.",
    "", "## Limitations", "",
    "Small subset: four training and four validation images. Combined results are not an independent test.",
]
(OUT / "evaluation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("\nExport complete. Files in results/:")
for name in [
    "evaluation_report.html",
    "evaluation_report.md",
    "evaluation_summary.csv",
    "evaluation_per_class_iou.csv",
    "confusion_matrix_validation.csv"
]:
    p = OUT / name
    print(f"{'OK' if p.exists() else 'MISSING'}  {p.name}")
