from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import load_image_from_bytes
from backend.segmentation import predict_mask, colorize_mask, CITYSCAPES_COLORS
from backend.visualization import save_mask, save_class_ids, create_overlay

from PIL import Image, UnidentifiedImageError
from pathlib import Path
from io import BytesIO
from time import perf_counter
import base64
import numpy as np


app = FastAPI(title="Semantic Segmentation Robot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Semantic Segmentation Robot"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 20 MB."
        )

    try:
        image, tensor, original_size = load_image_from_bytes(content)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload a valid JPG or PNG."
        )

    start_time = perf_counter()
    mask, segmentation_stats = predict_mask(image, return_stats=True)
    inference_time_ms = (perf_counter() - start_time) * 1000

    colored_mask = colorize_mask(mask)
    overlay = create_overlay(image, colored_mask)

    # Use only the filename stem; never trust a client-supplied path.
    original_name = Path(file.filename or "image").stem
    safe_stem = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in original_name
    ).strip("_") or "image"

    mask_path = RESULTS_DIR / f"{safe_stem}_mask.png"
    overlay_path = RESULTS_DIR / f"{safe_stem}_overlay.png"
    preview_path = RESULTS_DIR / f"{safe_stem}_preview.jpg"
    class_ids_path = RESULTS_DIR / f"{safe_stem}_class_ids.png"

    save_mask(mask, str(mask_path))
    save_class_ids(mask, str(class_ids_path))
    overlay.save(overlay_path)

    preview = overlay.convert("RGB")
    preview.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
    preview.save(preview_path, "JPEG", quality=85, optimize=True)

    #Encode the full class-ID image for pixel inspection in the browser
    ids_buffer = BytesIO()
    ids_image = Image.fromarray(
        np.asarray(mask, dtype=np.uint8),
        mode="L"
    )
    ids_image.save(ids_buffer, format="PNG")
    class_ids_base64 = base64.b64encode(
        ids_buffer.getvalue()
    ).decode("ascii")

    total_pixels = int(mask.size)
    unique_classes, pixel_counts = np.unique(mask, return_counts=True)

    detected_classes = []
    for class_id, pixel_count in zip(unique_classes, pixel_counts):
        class_id = int(class_id)
        pixel_count = int(pixel_count)

        detected_classes.append({
            "id": class_id,
            "name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "percentage": round(pixel_count / total_pixels * 100, 2),
            "pixels": pixel_count,
            "color": (
                "#{:02x}{:02x}{:02x}".format(
                    *[int(channel) for channel in CITYSCAPES_COLORS[class_id]]
                )
                if 0 <= class_id < len(CITYSCAPES_COLORS)
                else "#607d8b"
            ),
        })

    return JSONResponse({
        "filename": file.filename or "image",
        "size_bytes": len(content),
        "original_width": original_size[0],
        "original_height": original_size[1],
        "inference_width": segmentation_stats["inference_width"],
        "inference_height": segmentation_stats["inference_height"],
        "model": segmentation_stats["model"],
        "device": segmentation_stats["device"],
        "tensor_shape": list(tensor.shape),
        "mask_shape": list(mask.shape),
        "unique_classes": [int(x) for x in unique_classes],
        "detected_classes": detected_classes,
        "class_count": len(detected_classes),
        "inference_time_ms": round(inference_time_ms, 2),
        "mask_file": str(mask_path),
        "overlay_file": str(overlay_path),
        "preview_file": str(preview_path),
        "preview_url": f"/results/{preview_path.name}",
        "class_ids_file": str(class_ids_path),
        "class_ids_base64": class_ids_base64,
        "message": "Segmentation completed successfully",
    })


@app.get("/results/{filename}")
def get_result(filename: str):
    #results directory
    safe_name = Path(filename).name
    file_path = RESULTS_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")

    return FileResponse(file_path)


@app.get("/evaluate")
def get_model_evaluation():
    """Return metrics from the latest dataset evaluation."""
    import json
    from pathlib import Path
    from fastapi import HTTPException

    metrics_path = Path(__file__).resolve().parent.parent / "results" / "evaluation_metrics.json"

    if not metrics_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run backend.evaluate_dataset first."
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/evaluate/all")
def get_all_evaluation_results():
    """Return train, validation and combined dataset evaluation metrics."""
    import json

    metrics_path = RESULTS_DIR / "evaluation_metrics_all.json"

    if not metrics_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Full evaluation results not found. Run backend.evaluate_dataset first."
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/evaluation/sample/{split}/{filename}")
def get_evaluation_sample(split: str, filename: str):
    if split not in {"train", "val"}:
        raise HTTPException(status_code=400, detail="Invalid dataset split.")

    safe_name = Path(filename).name
    if "/" in filename or "\\" in filename or not safe_name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    image_path = BASE_DIR / "data" / "evaluation" / "cityscapes8" / "images" / split / safe_name
    mask_path = BASE_DIR / "data" / "evaluation" / "cityscapes8" / "masks" / split / safe_name

    if not image_path.is_file() or not mask_path.is_file():
        raise HTTPException(status_code=404, detail="Image or ground-truth mask not found.")

    return {
        "filename": safe_name,
        "image_url": f"/evaluation/file/{split}/images/{safe_name}",
        "mask_url": f"/evaluation/file/{split}/masks/{safe_name}",
    }


@app.get("/evaluation/file/{split}/{kind}/{filename}")
def get_evaluation_file(split: str, kind: str, filename: str):
    if split not in {"train", "val"} or kind not in {"images", "masks"}:
        raise HTTPException(status_code=400, detail="Invalid dataset path.")

    safe_name = Path(filename).name
    if "/" in filename or "\\" in filename or not safe_name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = BASE_DIR / "data" / "evaluation" / "cityscapes8" / kind / split / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    return FileResponse(file_path)


@app.get("/evaluation/prediction/{split}/{filename}")
def get_evaluation_prediction(split: str, filename: str):
    from PIL import Image
    from backend.segmentation import predict_mask, colorize_mask

    if split not in {"train", "val"}:
        raise HTTPException(status_code=400, detail="Invalid dataset split.")

    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    image_path = (
        BASE_DIR / "data" / "evaluation" / "cityscapes8"
        / "images" / split / safe_name
    )

    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Dataset image not found.")

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            prediction = predict_mask(image)

        colored = colorize_mask(prediction)

        output_dir = RESULTS_DIR / "evaluation_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split}_{safe_name}"

        if isinstance(colored, Image.Image):
            colored.save(output_path)
        else:
            Image.fromarray(np.asarray(colored).astype(np.uint8)).save(output_path)

        return FileResponse(output_path, media_type="image/png")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc


@app.get("/evaluation/ground-truth/{split}/{filename}")
def get_evaluation_ground_truth_color(split: str, filename: str):
    from PIL import Image
    from backend.segmentation import colorize_mask

    if split not in {"train", "val"}:
        raise HTTPException(status_code=400, detail="Invalid dataset split.")

    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    mask_path = (
        BASE_DIR / "data" / "evaluation" / "cityscapes8"
        / "masks" / split / safe_name
    )

    if not mask_path.is_file():
        raise HTTPException(status_code=404, detail="Ground-truth mask not found.")

    raw_to_train = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
    }

    with Image.open(mask_path) as source:
        raw_mask = np.asarray(source, dtype=np.uint8)

    train_mask = np.full(raw_mask.shape, 255, dtype=np.uint8)
    for raw_id, train_id in raw_to_train.items():
        train_mask[raw_mask == raw_id] = train_id

    colored = colorize_mask(train_mask)

    output_dir = RESULTS_DIR / "evaluation_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gt_{split}_{safe_name}"

    Image.fromarray(colored).save(output_path)
    return FileResponse(output_path, media_type="image/png")
