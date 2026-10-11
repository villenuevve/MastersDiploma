from pathlib import Path
import json
import html
import numpy as np
from PIL import Image, ImageDraw
from backend.segmentation import predict_mask

ROOT = Path(".")
DATA = ROOT / "data/evaluation/cityscapes8"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]
COLORS = [
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),
    (190,153,153),(153,153,153),(250,170,30),(220,220,0),
    (107,142,35),(152,251,152),(70,130,180),(220,20,60),
    (255,0,0),(0,0,142),(0,0,70),(0,60,100),
    (0,80,100),(0,0,230),(119,11,32)
]
LABEL_MAP = {
    7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7,
    21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14,
    28:15, 31:16, 32:17, 33:18
}

metrics_path = OUT / "evaluation_metrics_all.json"
if not metrics_path.exists():
    raise FileNotFoundError(
        "Не найден results/evaluation_metrics_all.json. "
        "Сначала запусти существующий evaluate_dataset.py."
    )

results = json.loads(metrics_path.read_text(encoding="utf-8"))
all_metrics = results["all"]
val_metrics = results["val"]

cm = np.asarray(all_metrics["confusion_matrix"], dtype=np.float64)
if cm.shape != (19, 19):
    raise ValueError(f"Ожидалась матрица 19x19, получена {cm.shape}")

# Строки — истинный класс, столбцы — предсказанный.
tp = np.diag(cm)
true_count = cm.sum(axis=1)
pred_count = cm.sum(axis=0)
union = true_count + pred_count - tp
iou = np.divide(tp, union, out=np.full(19, np.nan), where=union > 0)
precision = np.divide(tp, pred_count, out=np.full(19, np.nan), where=pred_count > 0)
recall = np.divide(tp, true_count, out=np.full(19, np.nan), where=true_count > 0)
f1 = np.divide(
    2 * precision * recall, precision + recall,
    out=np.full(19, np.nan),
    where=np.isfinite(precision + recall) & ((precision + recall) > 0)
)

def pct(v):
    return "—" if not np.isfinite(v) else f"{v * 100:.2f}%"

def svg_bars(values, title, color="#4263eb"):
    valid = [(i, v) for i, v in enumerate(values) if np.isfinite(v)]
    width, height = 1000, 410
    left, top, bottom = 165, 45, 35
    plot_h = height - top - bottom
    bar_h = min(14, plot_h / max(len(valid), 1) * 0.68)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        f'<text x="10" y="24" font-size="18" font-weight="bold">{html.escape(title)}</text>'
    ]
    for j, (i, v) in enumerate(valid):
        y = top + j * (plot_h / max(len(valid), 1))
        w = max(0, min(100, v * 100)) * 5.7
        parts.append(
            f'<text x="{left-10}" y="{y+bar_h}" text-anchor="end" '
            f'font-size="11">{html.escape(NAMES[i])}</text>'
            f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" '
            f'fill="{color}" rx="2"/>'
            f'<text x="{left+w+5}" y="{y+bar_h}" font-size="10">{v*100:.1f}%</text>'
        )
    parts.append("</svg>")
    return "".join(parts)

# Confusion matrix heatmap. Each cell is normalized by the true-class count.
row_sums = cm.sum(axis=1, keepdims=True)
norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
cell = 29
margin = 160
size = margin + 19 * cell + 15
parts = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="Confusion matrix">']
parts.append('<text x="160" y="20" font-size="15" font-weight="bold">Predicted class →</text>')
parts.append('<text x="10" y="45" font-size="12">True class ↓</text>')
for i, name in enumerate(NAMES):
    x = margin + i * cell + cell / 2
    y = margin + i * cell + cell / 2
    parts.append(
        f'<text x="{x}" y="{margin-5}" font-size="8" text-anchor="start" '
        f'transform="rotate(-55 {x} {margin-5})">{html.escape(name)}</text>'
    )
    parts.append(
        f'<text x="{margin-7}" y="{y+3}" font-size="8" text-anchor="end">'
        f'{html.escape(name)}</text>'
    )
    for j in range(19):
        v = norm[i, j]
        shade = int(255 * (1 - min(v, 1)))
        fill = f"rgb({shade},{shade},255)"
        parts.append(
            f'<rect x="{margin+j*cell}" y="{margin+i*cell}" '
            f'width="{cell-1}" height="{cell-1}" fill="{fill}">'
            f'<title>True: {html.escape(name)}; predicted: '
            f'{html.escape(NAMES[j])}; count: {int(cm[i,j]):,}; '
            f'row-normalized: {v*100:.1f}%</title></rect>'
        )
        if cm[i, j] > 0:
            parts.append(
                f'<text x="{margin+j*cell+cell/2}" y="{margin+i*cell+cell/2+3}" '
                f'text-anchor="middle" font-size="6" fill="#111">'
                f'{int(cm[i,j])}</text>'
            )
parts.append("</svg>")
confusion_svg = "".join(parts)

def colorize(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(COLORS):
        rgb[mask == idx] = color
    rgb[mask == 255] = (0, 0, 0)
    return Image.fromarray(rgb)

def make_overlay(image, mask):
    image = image.convert("RGB")
    color = colorize(mask).resize(image.size, Image.Resampling.NEAREST)
    return Image.blend(image, color, 0.5)

def save_examples():
    cards = []
    image_dir = DATA / "images" / "val"
    mask_dir = DATA / "masks" / "val"
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            continue
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        with Image.open(mask_path) as gt_file:
            gt_raw = np.asarray(gt_file.convert("L"))
        gt = np.full(gt_raw.shape, 255, dtype=np.uint8)
        for label_id, train_id in LABEL_MAP.items():
            gt[gt_raw == label_id] = train_id

        pred = np.asarray(predict_mask(image)).astype(np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f"Размеры масок не совпадают: {image_path.name}")

        valid = gt != 255
        error = valid & (pred != gt)
        error_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
        error_rgb[valid & ~error] = (50, 160, 80)
        error_rgb[error] = (230, 45, 45)

        panels = [
            ("Original image", image),
            ("Ground truth", colorize(gt)),
            ("Model prediction", colorize(pred)),
            ("Error map: red = incorrect", Image.fromarray(error_rgb))
        ]
        panel_html = []
        for title, panel in panels:
            panel.thumbnail((440, 280))
            name = f"analysis_{image_path.stem}_{len(panel_html)}.jpg"
            panel.save(OUT / name, quality=88)
            panel_html.append(
                f'<div><p>{html.escape(title)}</p>'
                f'<img src="{html.escape(name)}" alt="{html.escape(title)}"></div>'
            )
        err_rate = error.sum() / max(valid.sum(), 1) * 100
        cards.append(
            f'<section><h3>{html.escape(image_path.name)}</h3>'
            f'<p>Pixel error on valid ground-truth pixels: {err_rate:.2f}%</p>'
            f'<div class="panels">{"".join(panel_html)}</div></section>'
        )
    return "".join(cards)

table_rows = ""
for i, name in enumerate(NAMES):
    table_rows += (
        f"<tr><td>{html.escape(name)}</td><td>{pct(iou[i])}</td>"
        f"<td>{pct(precision[i])}</td><td>{pct(recall[i])}</td>"
        f"<td>{pct(f1[i])}</td><td>{int(true_count[i]):,}</td></tr>"
    )

accuracy = all_metrics.get("pixel_accuracy", 0)
miou = all_metrics.get("mean_iou", 0)
val_accuracy = val_metrics.get("pixel_accuracy", 0)
val_miou = val_metrics.get("mean_iou", 0)
timing = all_metrics.get("inference_timing", {})
examples = save_examples()

report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Segmentation — Evaluation Report</title>
<style>
body{{font:14px -apple-system,BlinkMacSystemFont,sans-serif;max-width:1400px;
margin:30px auto;padding:0 22px;color:#202124}}
h1{{font-size:30px}}h2{{margin-top:40px;border-bottom:1px solid #ddd;padding-bottom:8px}}
.note{{color:#555;line-height:1.6}}.cards{{display:flex;flex-wrap:wrap;gap:14px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:16px;min-width:170px}}
.value{{font-size:27px;font-weight:650;margin-top:8px}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}
th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#f4f4f4}}
svg{{max-width:100%;height:auto}}.panels{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.panels img{{width:100%;height:auto;border:1px solid #ddd}}section{{margin:28px 0}}
@media(max-width:700px){{.panels{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Semantic Segmentation — Evaluation Report</h1>
<p class="note">Quantitative metrics are calculated against Cityscapes ground-truth masks.
Confusion matrix rows are true classes and columns are predicted classes.
The combined result includes the train and validation subsets; validation metrics
are reported separately. The dataset contains only eight labelled images, so
these measurements should be treated as a small-sample evaluation.</p>
<div class="cards">
<div class="card">Combined mIoU<div class="value">{miou*100:.2f}%</div></div>
<div class="card">Combined Pixel Accuracy<div class="value">{accuracy*100:.2f}%</div></div>
<div class="card">Validation mIoU<div class="value">{val_miou*100:.2f}%</div></div>
<div class="card">Validation Pixel Accuracy<div class="value">{val_accuracy*100:.2f}%</div></div>
<div class="card">Mean inference time<div class="value">{timing.get("mean_ms", 0):.1f} ms</div></div>
<div class="card">Evaluated images<div class="value">{all_metrics.get("images_evaluated", 0)}</div></div>
</div>
<h2>1. Confusion matrix</h2>
<p class="note">Darker cells indicate a larger fraction of the true class assigned
to that predicted class. Diagonal cells are correct predictions. Hover over cells
to see class names and counts.</p>
{confusion_svg}
<h2>2. Per-class quality metrics</h2>
<p class="note">IoU = TP / (TP + FP + FN). Precision = TP / (TP + FP).
Recall = TP / (TP + FN). F1 is the harmonic mean of precision and recall.
Classes without ground-truth pixels or predicted pixels may have undefined metrics (N/A); a zero IoU means no overlap was achieved when the class was evaluable.</p>
{svg_bars(iou, "Intersection over Union (IoU)")}
{svg_bars(precision, "Precision", "#0f9d58")}
{svg_bars(recall, "Recall", "#f29900")}
{svg_bars(f1, "F1-score", "#8e63ce")}
<table><thead><tr><th>Class</th><th>IoU</th><th>Precision</th>
<th>Recall</th><th>F1-score</th><th>Ground-truth pixels</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<h2>3. Qualitative analysis</h2>
<p class="note">Red pixels in the error map are incorrect predictions; green pixels
are correct predictions on valid ground-truth pixels. Black regions are ignored
labels. These examples are from the validation subset.</p>
{examples}
<h2>4. Interpretation</h2>
<ul>
<li>Pixel Accuracy measures the fraction of correctly classified valid pixels;
large classes such as road and building can dominate this score.</li>
<li>mIoU averages class IoU and gives a more balanced view of segmentation quality.</li>
<li>Per-class Precision and Recall reveal different failure modes: false-positive
class assignments versus missed ground-truth objects.</li>
<li>The small labelled sample is not sufficient to establish generalization to
all street scenes. The separate 120-image collection can be used for descriptive
class-distribution and inference-time analysis, not ground-truth accuracy.</li>
</ul>
</body></html>"""

path = OUT / "segmentation_quality_report.html"
path.write_text(report, encoding="utf-8")
print(f"Report created: {path.resolve()}")
print("Open with: open results/segmentation_quality_report.html")
