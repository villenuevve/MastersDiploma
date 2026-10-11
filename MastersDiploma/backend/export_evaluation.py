from pathlib import Path
from datetime import datetime
import csv
import json
import html
import math

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle"
]

def load_json(name):
    path = OUT / name
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)

def pct(value):
    return None if value is None else round(float(value) * 100, 2)

def fmt(value, suffix=""):
    return "N/A" if value is None else f"{value:,.2f}{suffix}"

def calculate_metrics(data):
    matrix = data.get("confusion_matrix")
    if not matrix:
        return []

    n = len(CLASSES)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        return []

    tp = [float(matrix[i][i]) for i in range(n)]
    actual = [sum(map(float, matrix[i])) for i in range(n)]
    predicted = [sum(float(matrix[i][j]) for i in range(n)) for j in range(n)]
    total = sum(actual)
    total_tp = sum(tp)

    rows = []
    for i, name in enumerate(CLASSES):
        union = actual[i] + predicted[i] - tp[i]
        precision = tp[i] / predicted[i] if predicted[i] else None
        recall = tp[i] / actual[i] if actual[i] else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None
            and precision + recall > 0 else
            (0.0 if precision is not None and recall is not None else None)
        )
        iou = tp[i] / union if union else None

        rows.append({
            "class": name,
            "tp": int(tp[i]),
            "ground_truth_pixels": int(actual[i]),
            "predicted_pixels": int(predicted[i]),
            "false_positive": int(predicted[i] - tp[i]),
            "false_negative": int(actual[i] - tp[i]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "iou_percent": pct(iou),
            "support_percent": 100 * actual[i] / total if total else None,
        })

    return rows

def confusion_html(matrix, normalized=False):
    if not matrix:
        return "<p>Confusion matrix unavailable.</p>"

    n = len(CLASSES)
    values = [[float(v) for v in row] for row in matrix]
    if normalized:
        for i in range(n):
            row_sum = sum(values[i])
            values[i] = [
                v / row_sum if row_sum else 0 for v in values[i]
            ]

    maximum = max((max(row) for row in values), default=0)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in CLASSES)
    rows = []

    for i, row in enumerate(values):
        cells = []
        for j, value in enumerate(row):
            strength = value / maximum if maximum else 0
            if i == j:
                bg = f"rgba(22,163,74,{0.08 + 0.82 * strength:.3f})"
            else:
                bg = f"rgba(220,38,38,{0.03 + 0.75 * strength:.3f})"

            shown = f"{value:.1%}" if normalized else f"{int(value):,}"
            original = int(matrix[i][j])
            title = (
                f"Actual: {CLASSES[i]} | Predicted: {CLASSES[j]} | "
                f"Pixels: {original:,}"
            )
            cells.append(
                f'<td title="{html.escape(title)}" style="background:{bg}">'
                f'{shown}</td>'
            )
        rows.append(
            f"<tr><th>{html.escape(CLASSES[i])}</th>{''.join(cells)}</tr>"
        )

    return (
        '<div class="matrix-wrap"><table class="matrix-table"><thead><tr>'
        '<th>Actual ↓ / Predicted →</th>' + head +
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    )

def main():
    validation = load_json("evaluation_metrics.json")
    combined = load_json("evaluation_metrics_all.json")

    if not validation and not combined:
        raise SystemExit(
            "No evaluation metrics found. Run: python -m backend.evaluate_dataset"
        )

    splits = []
    for key in ("train", "val", "all"):
        data = combined.get(key)
        if isinstance(data, dict):
            splits.append((key, data))

    if not any(key == "val" for key, _ in splits) and validation:
        splits.append(("val", validation))

    if not splits:
        raise SystemExit("No usable evaluation splits found.")

    selected = next((d for k, d in splits if k == "val"), splits[0][1])
    class_rows = calculate_metrics(selected)
    valid_rows = [r for r in class_rows if r["iou"] is not None]
    matrix = selected.get("confusion_matrix", [])

    summary = []
    for split, data in splits:
        cm = data.get("confusion_matrix", [])
        rows = calculate_metrics(data)
        supported = [r for r in rows if r["iou"] is not None]
        fw_iou = None
        total_support = sum(r["ground_truth_pixels"] for r in supported)
        if total_support:
            fw_iou = sum(
                r["iou"] * r["ground_truth_pixels"] for r in supported
            ) / total_support

        class_accuracy_values = [
            r["recall"] for r in rows if r["recall"] is not None
        ]
        mean_class_accuracy = (
            sum(class_accuracy_values) / len(class_accuracy_values)
            if class_accuracy_values else None
        )

        summary.append({
            "split": split,
            "images": data.get("images_evaluated"),
            "pixels": data.get("evaluated_pixels"),
            "miou": data.get("mean_iou"),
            "pixel_accuracy": data.get("pixel_accuracy"),
            "mean_class_accuracy": mean_class_accuracy,
            "frequency_weighted_iou": fw_iou,
            "classes_with_union": len(supported),
        })

    # Export complete per-class metrics to CSV.
    with (OUT / "evaluation_per_class_detailed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "class", "tp", "ground_truth_pixels", "predicted_pixels",
            "false_positive", "false_negative", "precision", "recall",
            "f1", "iou", "iou_percent", "support_percent"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(class_rows)

    with (OUT / "evaluation_summary_detailed.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # Export both raw and row-normalized validation confusion matrices.
    if matrix and len(matrix) == len(CLASSES):
        for normalized in (False, True):
            filename = (
                "confusion_matrix_validation_normalized.csv"
                if normalized else "confusion_matrix_validation.csv"
            )
            with (OUT / filename).open(
                "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["Actual / Predicted"] + CLASSES)
                for i, row in enumerate(matrix):
                    if normalized:
                        total = sum(row)
                        values = [
                            round(v / total, 6) if total else 0 for v in row
                        ]
                    else:
                        values = row
                    writer.writerow([CLASSES[i]] + values)

    def split_row(s):
        return (
            "<tr>" +
            f"<td>{html.escape(s['split'].upper())}</td>" +
            f"<td>{s['images'] if s['images'] is not None else 'N/A'}</td>" +
            f"<td>{s['pixels']:,}</td>" +
            f"<td>{fmt(pct(s['miou']), '%')}</td>" +
            f"<td>{fmt(pct(s['pixel_accuracy']), '%')}</td>" +
            f"<td>{fmt(pct(s['mean_class_accuracy']), '%')}</td>" +
            f"<td>{fmt(pct(s['frequency_weighted_iou']), '%')}</td>" +
            f"<td>{s['classes_with_union']}/19</td></tr>"
        )

    class_table_rows = []
    for r in sorted(
        class_rows,
        key=lambda item: -1 if item["iou"] is None else item["iou"],
        reverse=True
    ):
        score = r["iou_percent"]
        bar = 0 if score is None else score
        class_table_rows.append(
            f'<tr data-iou="{"" if score is None else score}">'
            f'<td>{html.escape(r["class"].title())}</td>'
            f'<td>{fmt(score, "%")}</td>'
            f'<td>{fmt(pct(r["precision"]), "%")}</td>'
            f'<td>{fmt(pct(r["recall"]), "%")}</td>'
            f'<td>{fmt(pct(r["f1"]), "%")}</td>'
            f'<td>{r["ground_truth_pixels"]:,}</td>'
            f'<td>{r["predicted_pixels"]:,}</td>'
            f'<td>{r["false_positive"]:,}</td>'
            f'<td>{r["false_negative"]:,}</td>'
            f'<td><div class="bar"><span style="width:{bar}%"></span></div></td>'
            '</tr>'
        )

    chart_data = [
        {"name": r["class"], "iou": r["iou_percent"]}
        for r in class_rows
    ]
    chart_json = json.dumps(chart_data).replace("</", "<\\/")
    summary_json = json.dumps(summary).replace("</", "<\\/")

    errors = sorted(
        [r for r in class_rows if r["iou"] is not None],
        key=lambda r: r["iou"]
    )
    best = sorted(
        [r for r in class_rows if r["iou"] is not None],
        key=lambda r: r["iou"], reverse=True
    )
    best_html = "".join(
        f"<li><span>{html.escape(r['class'].title())}</span>"
        f"<strong>{r['iou_percent']:.2f}%</strong></li>"
        for r in best[:5]
    )
    worst_html = "".join(
        f"<li><span>{html.escape(r['class'].title())}</span>"
        f"<strong>{r['iou_percent']:.2f}%</strong></li>"
        for r in errors[:5]
    )

    absent = [r["class"] for r in class_rows if r["iou"] is None]
    absent_text = (
        ", ".join(absent) if absent else "No classes without a defined IoU."
    )
    val = next((s for s in summary if s["split"] == "val"), summary[0])
    evaluation_date = datetime.now().astimezone().strftime("%d %B %Y, %H:%M %Z")

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Segmentation | Model Evaluation Report</title>
<style>
:root {{
 --bg:#f4f6fa;--panel:#fff;--ink:#172033;--muted:#687386;
 --line:#e1e6ef;--accent:#3659d9;--green:#16835d;--red:#c24141;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1450px;margin:auto;padding:34px 28px 64px}}
header{{background:#151f36;color:white;border-radius:20px;padding:34px;
 display:flex;justify-content:space-between;gap:24px;align-items:flex-start}}
.eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:11px;
 color:#aab8d7;font-weight:700}}
h1{{font-size:clamp(27px,4vw,42px);line-height:1.12;margin:12px 0}}
header p{{color:#d0d8e8;max-width:700px;margin:0}}
.header-meta{{font-size:12px;color:#c0cbe0;min-width:170px}}
.actions{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 28px}}
button,select,input{{font:inherit}}
button{{border:1px solid var(--line);background:white;border-radius:9px;
 padding:9px 13px;cursor:pointer;color:var(--ink)}}
button:hover{{border-color:var(--accent)}}
section{{margin:30px 0}}
.section-head{{display:flex;justify-content:space-between;align-items:end;
 gap:12px;flex-wrap:wrap;margin-bottom:14px}}
h2{{font-size:21px;margin:0}} h3{{font-size:15px;margin:0 0 12px}}
.sub{{color:var(--muted);margin:4px 0 0;font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}
.card,.panel{{background:var(--panel);border:1px solid var(--line);
 border-radius:14px;padding:19px;min-width:0}}
.kpi-label{{color:var(--muted);font-size:12px;font-weight:600}}
.kpi-value{{font-size:29px;font-weight:750;letter-spacing:-.04em;margin:8px 0 3px}}
.kpi-note{{font-size:11px;color:var(--muted)}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.three-col{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;
 white-space:nowrap}}
th{{background:#f7f8fb;color:#536075;font-size:11px;position:sticky;top:0;z-index:1}}
th:first-child,td:first-child{{text-align:left}}
tbody tr:hover{{background:#f6f8ff}}
.bar{{width:95px;height:7px;background:#e9edf5;border-radius:9px;overflow:hidden}}
.bar span{{display:block;height:100%;background:var(--accent);border-radius:9px}}
canvas{{display:block;width:100%;height:320px}}
.matrix-wrap{{overflow:auto;max-height:680px;border:1px solid var(--line);border-radius:12px}}
.matrix-table{{width:max-content;font-size:10px}}
.matrix-table th,.matrix-table td{{padding:7px;min-width:72px;text-align:center}}
.matrix-table thead th{{position:sticky;top:0}}
.matrix-table th:first-child{{position:sticky;left:0;z-index:3;background:#f1f4f9}}
.matrix-table td{{font-variant-numeric:tabular-nums}}
select,input{{border:1px solid var(--line);border-radius:8px;padding:8px;background:white}}
.list{{list-style:none;padding:0;margin:0}}
.list li{{display:flex;justify-content:space-between;gap:12px;padding:9px 0;
 border-bottom:1px solid var(--line)}}
.list li:last-child{{border:0}}
.callout{{padding:14px 16px;border-radius:10px;background:#eef3ff;
 border:1px solid #dbe5ff;color:#263b70;margin:14px 0}}
.warning{{background:#fff6e8;border-color:#f5dfb4;color:#7d4b0b}}
.muted{{color:var(--muted)}}
footer{{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);
 font-size:11px;margin-top:40px}}
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}
 .two-col,.three-col{{grid-template-columns:1fr}}header{{display:block}}
 .header-meta{{margin-top:20px}}main{{padding:18px 12px 42px}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}
 header{{background:#151f36!important;print-color-adjust:exact}}
 .actions,.no-print{{display:none!important}}.card,.panel{{break-inside:avoid}}
 .matrix-wrap{{max-height:none;overflow:visible}}.matrix-table{{font-size:7px}}
 .matrix-table th,.matrix-table td{{min-width:0;padding:3px}}
 section{{break-inside:avoid}}}}
</style>
</head>
<body>
<main>
<header>
 <div>
  <div class="eyebrow">Computer Vision · Model Validation</div>
  <h1>Semantic Segmentation<br>Evaluation Report</h1>
  <p>Quantitative evaluation of pixel-level scene understanding using Cityscapes
  semantic classes. Metrics are computed from the stored ground-truth confusion matrix.</p>
 </div>
 <div class="header-meta">
  <strong>REPORT GENERATED</strong><br>{html.escape(evaluation_date)}<br><br>
  <strong>CLASS TAXONOMY</strong><br>Cityscapes · 19 classes<br><br>
  <strong>VALIDATION IMAGES</strong><br>{val['images'] if val['images'] is not None else 'N/A'}
 </div>
</header>

<div class="actions no-print">
 <button onclick="window.print()">Export / Save as PDF</button>
 <button onclick="downloadCSV()">Download class metrics CSV</button>
 <button onclick="downloadSummary()">Download split summary CSV</button>
 <button onclick="location.reload()">Refresh report</button>
</div>

<div class="callout warning">
 <strong>Evaluation scope:</strong> this report currently uses a small subset
 (four training and four validation images). The combined split includes training
 images and is not an independent test result. Treat the results as preliminary,
 not as a reliable estimate of deployment performance.
</div>

<section>
 <div class="section-head"><div><h2>Executive summary</h2>
 <p class="sub">Primary validation metrics and sample coverage</p></div></div>
 <div class="grid">
  <div class="card"><div class="kpi-label">Mean Intersection over Union</div>
   <div class="kpi-value">{fmt(pct(val['miou']), '%')}</div>
   <div class="kpi-note">Mean IoU reported by evaluator</div></div>
  <div class="card"><div class="kpi-label">Pixel Accuracy</div>
   <div class="kpi-value">{fmt(pct(val['pixel_accuracy']), '%')}</div>
   <div class="kpi-note">Correct pixels / evaluated pixels</div></div>
  <div class="card"><div class="kpi-label">Evaluated images</div>
   <div class="kpi-value">{val['images'] if val['images'] is not None else 'N/A'}</div>
   <div class="kpi-note">Validation subset</div></div>
  <div class="card"><div class="kpi-label">Evaluated pixels</div>
   <div class="kpi-value">{(val['pixels'] or 0):,}</div>
   <div class="kpi-note">Pixels included in confusion matrix</div></div>
 </div>
</section>

<section>
 <div class="section-head"><div><h2>Dataset split comparison</h2>
 <p class="sub">Train, validation and combined metrics. Combined is descriptive only.</p>
 </div></div>
 <div class="table-wrap"><table id="summaryTable">
 <thead><tr><th>Split</th><th>Images</th><th>Valid pixels</th><th>mIoU</th>
 <th>Pixel accuracy</th><th>Mean class accuracy</th><th>Frequency-weighted IoU</th>
 <th>Classes present</th></tr></thead>
 <tbody>{''.join(split_row(s) for s in summary)}</tbody>
 </table></div>
 <div class="panel" style="margin-top:16px"><h3>mIoU by split</h3><canvas id="splitChart"></canvas></div>
</section>

<section>
 <div class="section-head"><div><h2>Per-class performance</h2>
 <p class="sub">IoU, precision, recall, F1, class support and error counts.</p></div>
 <input id="classSearch" placeholder="Filter classes..." aria-label="Filter classes">
 </div>
 <div class="table-wrap"><table id="classTable">
 <thead><tr>
 <th><button onclick="sortClass(0)">Class ↕</button></th>
 <th><button onclick="sortClass(1)">IoU ↕</button></th>
 <th>Precision</th><th>Recall</th><th>F1-score</th>
 <th>GT pixels</th><th>Predicted pixels</th><th>False positives</th>
 <th>False negatives</th><th>IoU scale</th>
 </tr></thead><tbody>{''.join(class_table_rows)}</tbody></table></div>
 <p class="sub">Precision = TP/(TP+FP); Recall = TP/(TP+FN);
 F1 = harmonic mean of precision and recall; IoU = TP/(TP+FP+FN).
 N/A means the class has no valid union in this evaluated subset.</p>
</section>

<section class="two-col">
 <div class="panel"><h3>Strongest classes by IoU</h3><ul class="list">{best_html}</ul></div>
 <div class="panel"><h3>Weakest classes by IoU</h3><ul class="list">{worst_html}</ul></div>
</section>

<section>
 <div class="section-head"><div><h2>IoU by semantic class</h2>
 <p class="sub">Class-level overlap quality; missing classes are not plotted as zero.</p>
 </div></div>
 <div class="panel"><canvas id="classChart"></canvas></div>
</section>

<section>
 <div class="section-head"><div><h2>Confusion matrix</h2>
 <p class="sub">Rows represent ground truth; columns represent predicted class.
 Hover over a cell for the raw pixel count.</p></div>
 <label>Display
 <select id="matrixMode">
  <option value="raw">Absolute pixel counts</option>
  <option value="normalized">Row-normalized (%)</option>
 </select></label>
 </div>
 <div id="matrixContainer">{confusion_html(matrix)}</div>
</section>

<section>
 <div class="section-head"><div><h2>Data coverage and limitations</h2></div></div>
 <div class="panel">
 <h3>Classes without a defined IoU</h3>
 <p>{html.escape(absent_text)}</p>
 <h3 style="margin-top:20px">Interpretation notes</h3>
 <ul>
  <li>Pixel accuracy can be dominated by large regions such as road, buildings and sky.</li>
  <li>mIoU gives each class with a nonzero union equal weight, making it more informative for class balance.</li>
  <li>Precision and recall are derived from the confusion matrix; absent ground-truth classes may have undefined recall or IoU.</li>
  <li>Training and validation subsets contain only a few images. Results can change substantially with scene composition.</li>
  <li>The combined split overlaps with training and validation data and must not be described as a held-out test set.</li>
  <li>Inference latency, memory usage and FPS are not included because they are not stored in the evaluation metrics JSON.</li>
 </ul>
 </div>
</section>

<footer>
Generated from results/evaluation_metrics_all.json and
results/evaluation_metrics.json. Model scores reflect the saved evaluation run;
they are not recalculated by opening this HTML file.
</footer>
</main>

<script>
const CLASS_DATA = {chart_json};
const SPLIT_DATA = {summary_json};
const MATRIX_RAW = {json.dumps(matrix).replace("</", "<\\/")};
const CLASS_NAMES = {json.dumps(CLASSES)};

function drawBars(canvasId, data, labelKey, valueKey) {{
 const canvas = document.getElementById(canvasId);
 const ctx = canvas.getContext('2d');
 const dpr = window.devicePixelRatio || 1;
 const width = canvas.clientWidth || 600;
 const height = canvas.clientHeight || 320;
 canvas.width = width * dpr; canvas.height = height * dpr;
 ctx.scale(dpr,dpr); ctx.clearRect(0,0,width,height);
 const left = 118, right = 48, top = 12, bottom = 30;
 const plotW = width-left-right, plotH = height-top-bottom;
 const valid = data.filter(d => d[valueKey] !== null && d[valueKey] !== undefined);
 if (!valid.length) return;
 const rowH = plotH/valid.length;
 ctx.font = '11px -apple-system, sans-serif';
 valid.forEach((d,i) => {{
  const y = top+i*rowH;
  const v = Number(d[valueKey]);
  ctx.fillStyle = '#526075'; ctx.textAlign = 'right';
  ctx.fillText(String(d[labelKey]).slice(0,18),left-9,y+rowH*.65);
  ctx.fillStyle = '#e9edf5'; ctx.fillRect(left,y+rowH*.22,plotW,rowH*.52);
  ctx.fillStyle = v < 40 ? '#c65b5b' : v < 70 ? '#d6a23e' : '#3659d9';
  ctx.fillRect(left,y+rowH*.22,plotW*Math.max(0,Math.min(100,v))/100,rowH*.52);
  ctx.fillStyle = '#273248'; ctx.textAlign = 'left';
  ctx.fillText(v.toFixed(1)+'%',left+plotW+6,y+rowH*.65);
 }});
}}

function drawSplitChart() {{
 const data = SPLIT_DATA.filter(s => s.miou !== null);
 drawBars('splitChart', data.map(s => ({{name:s.split.toUpperCase(),value:s.miou*100}})), 'name','value');
}}
function drawClassChart() {{
 drawBars('classChart', CLASS_DATA, 'name','iou');
}}

function matrixHtml(normalized) {{
 const n = CLASS_NAMES.length;
 let values = MATRIX_RAW.map(row => row.map(Number));
 if (normalized) values = values.map(row => {{
  const total = row.reduce((a,b)=>a+b,0);
  return row.map(v=>total ? v/total : 0);
 }});
 const max = Math.max(1,...values.flat());
 let out = '<div class="matrix-wrap"><table class="matrix-table"><thead><tr><th>Actual ↓ / Predicted →</th>';
 out += CLASS_NAMES.map(c=>'<th>'+esc(c)+'</th>').join('');
 out += '</tr></thead><tbody>';
 values.forEach((row,i) => {{
  out += '<tr><th>'+esc(CLASS_NAMES[i])+'</th>';
  row.forEach((v,j) => {{
   const strength = v/max;
   const alpha = (i===j ? .08+.82*strength : .03+.75*strength).toFixed(3);
   const color = i===j ? '22,163,74' : '220,38,38';
   const raw = Number(MATRIX_RAW[i][j]);
   const shown = normalized ? (v*100).toFixed(1)+'%' : Math.round(v).toLocaleString();
   out += '<td title="Actual: '+esc(CLASS_NAMES[i])+' | Predicted: '+esc(CLASS_NAMES[j])+' | Pixels: '+raw.toLocaleString()+'" style="background:rgba('+color+','+alpha+')">'+shown+'</td>';
  }});
  out += '</tr>';
 }});
 return out+'</tbody></table></div>';
}}
function esc(s) {{
 return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
document.getElementById('matrixMode').addEventListener('change', e => {{
 document.getElementById('matrixContainer').innerHTML = matrixHtml(e.target.value==='normalized');
}});
document.getElementById('classSearch').addEventListener('input', e => {{
 const q=e.target.value.toLowerCase();
 document.querySelectorAll('#classTable tbody tr').forEach(row => {{
  row.style.display=row.cells[0].textContent.toLowerCase().includes(q)?'':'none';
 }});
}});
let sortAsc = false;
function sortClass(column) {{
 const tbody=document.querySelector('#classTable tbody');
 const rows=[...tbody.rows];
 sortAsc=!sortAsc;
 rows.sort((a,b)=> {{
  if(column===0) return sortAsc?a.cells[0].textContent.localeCompare(b.cells[0].textContent):b.cells[0].textContent.localeCompare(a.cells[0].textContent);
  const av=parseFloat(a.cells[column].textContent);
  const bv=parseFloat(b.cells[column].textContent);
  const aa=Number.isNaN(av)?-1:av, bb=Number.isNaN(bv)?-1:bv;
  return sortAsc?aa-bb:bb-aa;
 }});
 rows.forEach(r=>tbody.appendChild(r));
}}
function csvDownload(filename, rows) {{
  const csv=rows.map(row=>row.map(v=>'"'+String(v??'').split('"').join('""')+'"').join(',')).join(String.fromCharCode(13,10));
 const blob=new Blob(['\\ufeff'+csv],{{type:'text/csv;charset=utf-8;'}});
 const url=URL.createObjectURL(blob);
 const a=document.createElement('a');a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url);
}}
function downloadCSV() {{
 const table=document.getElementById('classTable');
 const rows=[...table.rows].map(row=>[...row.cells].slice(0,9).map(c=>c.innerText));
 csvDownload('evaluation_per_class_detailed.csv',rows);
}}
function downloadSummary() {{
 const table=document.getElementById('summaryTable');
 csvDownload('evaluation_summary_detailed.csv',[...table.rows].map(row=>[...row.cells].map(c=>c.innerText)));
}}
drawClassChart(); drawSplitChart();
window.addEventListener('resize',()=>{{drawClassChart();drawSplitChart();}});
</script>
</body></html>"""

    (OUT / "evaluation_report.html").write_text(report, encoding="utf-8")
    print("Professional evaluation report generated:")
    print(OUT / "evaluation_report.html")
    print(OUT / "evaluation_per_class_detailed.csv")
    print(OUT / "evaluation_summary_detailed.csv")
    print("Open report with: open results/evaluation_report.html")

if __name__ == "__main__":
    main()
