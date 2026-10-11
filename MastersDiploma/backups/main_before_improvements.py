from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import load_image_from_bytes
from backend.segmentation import predict_mask, colorize_mask
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
    mask = predict_mask(image)
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
    class_ids_path = RESULTS_DIR / f"{safe_stem}_class_ids.png"

    save_mask(mask, str(mask_path))
    save_class_ids(mask, str(class_ids_path))
    overlay.save(overlay_path)

    # Encode the full class-ID image for pixel inspection in the browser.
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
        })

    return JSONResponse({
        "filename": file.filename or "image",
        "size_bytes": len(content),
        "original_width": original_size[0],
        "original_height": original_size[1],
        "tensor_shape": list(tensor.shape),
        "mask_shape": list(mask.shape),
        "unique_classes": [int(x) for x in unique_classes],
        "detected_classes": detected_classes,
        "class_count": len(detected_classes),
        "inference_time_ms": round(inference_time_ms, 2),
        "mask_file": str(mask_path),
        "overlay_file": str(overlay_path),
        "class_ids_file": str(class_ids_path),
        "class_ids_base64": class_ids_base64,
        "message": "Segmentation completed successfully",
    })


@app.get("/results/{filename}")
def get_result(filename: str):
    # Only serve files directly inside the results directory.
    safe_name = Path(filename).name
    file_path = RESULTS_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")

    return FileResponse(file_path)
