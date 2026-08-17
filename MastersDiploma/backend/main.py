from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import load_image_from_bytes
from backend.segmentation import predict_mask, colorize_mask
from backend.visualization import save_mask, create_overlay

import os
import numpy as np


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


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

    image, tensor, original_size = load_image_from_bytes(content)

    # Prediction
    mask = predict_mask(image)

    # Colored mask
    colored_mask = colorize_mask(mask)

    # Overlay
    overlay = create_overlay(
        image,
        colored_mask
    )

    # File names
    base_name = os.path.splitext(
        file.filename
    )[0]

    mask_path = os.path.join(
        RESULTS_DIR,
        f"{base_name}_mask.png"
    )

    overlay_path = os.path.join(
        RESULTS_DIR,
        f"{base_name}_overlay.png"
    )

    # Save results
    save_mask(
        mask,
        mask_path
    )

    overlay.save(
        overlay_path
    )

    # ==========================================
    # DETECTED CLASSES
    # ==========================================

    total_pixels = mask.size

    unique_classes = sorted(
        set(
            mask.flatten().tolist()
        )
    )

    detected_classes = []

    for class_id in unique_classes:

        # Number of pixels belonging to class
        pixel_count = int(
            np.sum(mask == class_id)
        )

        # Percentage
        percentage = (
            pixel_count /
            total_pixels *
            100
        )

        detected_classes.append({

            "id": int(class_id),

            "name": CLASS_NAMES.get(
                int(class_id),
                f"class_{class_id}"
            ),

            "percentage": round(
                percentage,
                2
            ),

            "pixels": pixel_count

        })


    # ==========================================
    # RESPONSE
    # ==========================================

    return JSONResponse({

        "filename": file.filename,

        "size_bytes": len(content),

        "original_width":
            original_size[0],

        "original_height":
            original_size[1],

        "tensor_shape":
            list(tensor.shape),

        "mask_shape":
            list(mask.shape),

        "unique_classes":
            unique_classes,

        "detected_classes":
            detected_classes,

        "class_count":
            len(detected_classes),

        "mask_file":
            mask_path,

        "overlay_file":
            overlay_path,

        "message":
            "segmentation saved successfully"

    })


@app.get("/results/{filename}")
def get_result(filename: str):

    file_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    return FileResponse(
        file_path
    )