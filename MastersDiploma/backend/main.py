
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import load_image_from_bytes
from backend.segmentation import (
    predict_mask,
    colorize_mask,
    create_traversability_map,
    colorize_traversability_map,
    CITYSCAPES_COLORS,
)
from backend.visualization import save_mask, save_class_ids, create_overlay
from backend.obstacle_analysis import analyze_obstacles
from backend.scene_analysis import analyze_scene
from backend.risk_map import analyze_risk
from backend.safe_corridor import analyze_safe_corridor

from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import Optional
from io import BytesIO
from time import perf_counter
import base64
import numpy as np
import json
import uuid
import zipfile
import re

from backend.database import SessionLocal
from backend.auth import router as auth_router, get_current_user, require_roles
from backend.models import Dataset, DatasetSplit, Image as DBImage, Prediction, DetectedClass, User


app = FastAPI(title="Semantic Segmentation Robot")

app.include_router(auth_router)

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

UPLOADS_DIR = BASE_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 12_000
UPLOAD_CHUNK_SIZE = 1024 * 1024

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


def _get_upload_id(stem: str) -> str:
    metadata_path = RESULTS_DIR / f"{stem}_results.json"

    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Result metadata not found.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        upload_id = metadata.get("upload_id")
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Could not read result metadata.")

    if not isinstance(upload_id, str) or not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(status_code=404, detail="Invalid result metadata.")

    return upload_id


def check_upload_access(upload_id: str, current_user: User, db):
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(status_code=404, detail="Image record not found.")

    allowed_paths = [
        f"data/uploads/{upload_id}{extension}"
        for extension in (".jpg", ".jpeg", ".png", ".img")
    ]

    image = (
        db.query(DBImage)
        .filter(DBImage.file_path.in_(allowed_paths))
        .first()
    )

    if image is None:
        raise HTTPException(status_code=404, detail="Image record not found.")

    if current_user.role.name in ("Administrator", "Technician"):
        return image

    if image.uploaded_by != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this result."
        )

    return image


def check_result_access(filename: str, current_user: User, db):
    suffixes = (
        "_safe_corridor.png",
        "_traversability.png",
        "_risk_map.png",
        "_class_ids.png",
        "_confidence.npy",
        "_confidence.png",
        "_overlay.png",
        "_preview.jpg",
        "_mask.png",
    )

    stem = next(
        (filename[:-len(suffix)] for suffix in suffixes
         if filename.endswith(suffix)),
        None
    )

    if stem is None:
        raise HTTPException(status_code=404, detail="Result not found.")

    metadata_path = RESULTS_DIR / f"{stem}_results.json"

    # New-format results: resolve ownership through the upload UUID.
    if metadata_path.is_file():
        upload_id = _get_upload_id(stem)

        if not stem.endswith(f"_{upload_id[:8]}"):
            raise HTTPException(
                status_code=404,
                detail="Result metadata does not match the image."
            )

        return check_upload_access(upload_id, current_user, db)

    # Legacy results: resolve the image through its stored artifact paths.
    mask_rel = f"results/{stem}_mask.png"
    overlay_rel = f"results/{stem}_overlay.png"

    image = (
        db.query(DBImage)
        .outerjoin(Prediction, Prediction.image_id == DBImage.id)
        .filter(
            (DBImage.mask_path == mask_rel)
            | (Prediction.result_path == overlay_rel)
        )
        .first()
    )

    if image is None:
        raise HTTPException(status_code=404, detail="Image record not found.")

    if current_user.role.name in ("Administrator", "Technician"):
        return image

    if image.uploaded_by != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this result."
        )

    return image


def _load_confidence_map(confidence_path: Path) -> np.ndarray:
    """
    Safely load a saved confidence map.
    Validate file size and array dimensions before allocating a full array.
    """
    if not confidence_path.is_file():
        raise ValueError("Confidence data file not found.")

    max_npy_bytes = MAX_IMAGE_PIXELS * np.dtype(np.float32).itemsize + 1024

    if confidence_path.stat().st_size > max_npy_bytes:
        raise ValueError("Confidence data file exceeds the allowed size.")

    try:
        mapped = np.load(
            confidence_path,
            allow_pickle=False,
            mmap_mode="r"
        )
    except (OSError, ValueError, EOFError) as error:
        raise ValueError("Confidence data file is invalid.") from error

    if mapped.ndim != 2 or mapped.size == 0:
        raise ValueError("Confidence data must be a non-empty 2D array.")

    if mapped.size > MAX_IMAGE_PIXELS:
        raise ValueError("Confidence map exceeds the allowed pixel count.")

    if mapped.dtype != np.float32:
        raise ValueError("Confidence map must contain float32 values.")

    if not np.isfinite(mapped).all():
        raise ValueError("Confidence data contains invalid values.")

    return np.asarray(mapped, dtype=np.float32)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    target_dataset_id: Optional[int] = None,
):
    if target_dataset_id is not None:
        if current_user.role.name not in ("Technician", "Administrator"):
            raise HTTPException(
                status_code=403,
                detail="Only Technicians and Administrators can upload to managed datasets."
            )

    content_buffer = bytearray()

    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)

        if not chunk:
            break

        if len(content_buffer) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum size is 20 MB."
            )

        content_buffer.extend(chunk)

    if not content_buffer:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    content = bytes(content_buffer)

    try:
        with Image.open(BytesIO(content)) as probe:
            if probe.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image format. Upload JPG or PNG."
                )

            width, height = probe.size

            if (
                width <= 0
                or height <= 0
                or width > MAX_IMAGE_SIDE
                or height > MAX_IMAGE_SIDE
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the allowed limit."
                )

        image, original_size = load_image_from_bytes(content)

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload a valid JPG or PNG."
        )

    upload_id = uuid.uuid4().hex

    # Run semantic segmentation
    start_time = perf_counter()
    mask, segmentation_stats, confidence_map = predict_mask(
        image, return_stats=True, return_confidence=True
    )
    inference_time_ms = (perf_counter() - start_time) * 1000

    # Generate segmentation visualization
    colored_mask = colorize_mask(mask)
    overlay = create_overlay(image, colored_mask)

    # Sanitize uploaded filename
    original_name = Path(file.filename or "image").stem
    safe_stem = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in original_name
    ).strip("_") or "image"
    safe_stem = f"{safe_stem}_{upload_id[:8]}"

    # Generate traversability map
    traversability = create_traversability_map(mask, confidence_map)

    # Generate Risk Map before saving its output file
    risk_analysis = analyze_risk(mask, confidence_map)
    traversability_colored = colorize_traversability_map(traversability)

    confidence_uint8 = np.clip(confidence_map * 255, 0, 255).astype(np.uint8)
    confidence_colored = np.zeros((*confidence_uint8.shape, 3), dtype=np.uint8)
    confidence_colored[..., 0] = 255 - confidence_uint8
    confidence_colored[..., 1] = confidence_uint8
    confidence_colored[..., 2] = 50

    mean_confidence = float(np.mean(confidence_map))
    low_confidence_percentage = float(np.mean(confidence_map < 0.5) * 100)

    # Output paths
    mask_path = RESULTS_DIR / f"{safe_stem}_mask.png"
    overlay_path = RESULTS_DIR / f"{safe_stem}_overlay.png"
    preview_path = RESULTS_DIR / f"{safe_stem}_preview.jpg"
    class_ids_path = RESULTS_DIR / f"{safe_stem}_class_ids.png"
    traversability_path = RESULTS_DIR / f"{safe_stem}_traversability.png"
    confidence_path = RESULTS_DIR / f"{safe_stem}_confidence.png"
    confidence_npy_path = RESULTS_DIR / f"{safe_stem}_confidence.npy"
    risk_map_path = RESULTS_DIR / f"{safe_stem}_risk_map.png"
    safe_corridor_path = RESULTS_DIR / f"{safe_stem}_safe_corridor.png"

    safe_corridor_analysis = analyze_safe_corridor(
        traversability,
        risk_analysis["risk_map"]
    )

    created_paths = []
    db = None
    try:
        created_paths.extend([
            mask_path,
            overlay_path,
            preview_path,
            class_ids_path,
            traversability_path,
            confidence_path,
            confidence_npy_path,
            risk_map_path,
            safe_corridor_path,
        ])

        # Save segmentation results
        save_mask(mask, str(mask_path))
        save_class_ids(mask, str(class_ids_path))
        overlay.save(overlay_path)

        # Save traversability map
        Image.fromarray(traversability_colored).save(traversability_path)
        Image.fromarray(confidence_colored).save(confidence_path)
        np.save(confidence_npy_path, confidence_map.astype(np.float32))
        Image.fromarray(risk_analysis['risk_map'], mode='L').save(risk_map_path)
        Image.fromarray(
            safe_corridor_analysis["visualization"]
        ).save(safe_corridor_path)

        # Save optimized preview
        preview = overlay.convert("RGB")
        preview.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
        preview.save(
            preview_path,
            "JPEG",
            quality=85,
            optimize=True
        )

        # Encode class-ID image for pixel inspection in the browser
        ids_buffer = BytesIO()
        ids_image = Image.fromarray(
            np.asarray(mask, dtype=np.uint8),
            mode="L"
        )
        ids_image.save(ids_buffer, format="PNG")

        class_ids_base64 = base64.b64encode(
            ids_buffer.getvalue()
        ).decode("ascii")

        # Calculate class distribution
        total_pixels = int(mask.size)
        unique_classes, pixel_counts = np.unique(
            mask,
            return_counts=True
        )

        detected_classes = []

        for class_id, pixel_count in zip(unique_classes, pixel_counts):
            class_id = int(class_id)
            pixel_count = int(pixel_count)

            detected_classes.append({
                "id": class_id,
                "name": CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                ),
                "percentage": round(
                    pixel_count / total_pixels * 100,
                    2
                ),
                "pixels": pixel_count,
                "color": (
                    "#{:02x}{:02x}{:02x}".format(
                        *[
                            int(channel)
                            for channel in CITYSCAPES_COLORS[class_id]
                        ]
                    )
                    if 0 <= class_id < len(CITYSCAPES_COLORS)
                    else "#607d8b"
                ),
            })

        obstacle_analysis = analyze_obstacles(mask)
        scene_analysis = analyze_scene(mask, confidence_map)

        # Save uploaded image and segmentation metadata to SQLite
        original_extension = Path(file.filename or "").suffix.lower()
        if original_extension not in {".jpg", ".jpeg", ".png"}:
            original_extension = ".img"

        uploaded_path = UPLOADS_DIR / f"{upload_id}{original_extension}"
        created_paths.append(uploaded_path)
        uploaded_path.write_bytes(content)

        db = SessionLocal()
        try:
            if target_dataset_id is not None:
                if current_user.role.name not in ("Technician", "Administrator"):
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to upload into a managed dataset."
                    )

                dataset = db.query(Dataset).filter(
                    Dataset.id == target_dataset_id
                ).first()

                if dataset is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Target dataset not found."
                    )
            else:
                dataset = (
                    db.query(Dataset)
                    .filter(Dataset.name == "Uploaded images")
                    .first()
                )

                if dataset is None:
                    dataset = Dataset(
                        name="Uploaded images",
                        description="Images uploaded through the web application."
                    )
                    db.add(dataset)
                    db.flush()

            db_image = DBImage(
                dataset_id=dataset.id,
                filename=file.filename or "image",
                uploaded_by=current_user.id,
                file_path=str(uploaded_path.relative_to(BASE_DIR)),
                width=original_size[0],
                height=original_size[1],
                file_size=len(content),
                has_mask=True,
                mask_path=str(mask_path.relative_to(BASE_DIR))
            )
            db.add(db_image)
            db.flush()

            prediction = Prediction(
                image_id=db_image.id,
                model_name=segmentation_stats["model"],
                result_path=str(overlay_path.relative_to(BASE_DIR)),
                inference_time_ms=round(inference_time_ms, 2),
                mean_confidence=round(mean_confidence * 100, 2),
                low_confidence_percentage=round(low_confidence_percentage, 2)
            )
            db.add(prediction)
            db.flush()

            for item in detected_classes:
                db.add(DetectedClass(
                    prediction_id=prediction.id,
                    class_id=item["id"],
                    class_name=item["name"],
                    pixel_count=item["pixels"],
                    percentage=item["percentage"]
                ))

            db_image_id = db_image.id
            prediction_id = prediction.id

        except Exception:
            db.rollback()
            raise

        response_data = {
            "filename": file.filename or "image",
            "image_id": db_image_id,
            "prediction_id": prediction_id,
            "size_bytes": len(content),

            "original_width": original_size[0],
            "original_height": original_size[1],

            "inference_width": segmentation_stats["inference_width"],
            "inference_height": segmentation_stats["inference_height"],

            "model": segmentation_stats["model"],
            "device": segmentation_stats["device"],

            "mask_shape": list(mask.shape),

            "unique_classes": [
                int(x) for x in unique_classes
            ],
            "detected_classes": detected_classes,
            "obstacle_summary": obstacle_analysis["obstacle_summary"],
            "obstacles": obstacle_analysis["obstacles"],
            "environment_objects": obstacle_analysis["environment_objects"],
            "scene_analysis": scene_analysis,
            "risk_map_file": str(risk_map_path),
            "risk_map_url": f"/results/{risk_map_path.name}",
            "risk_statistics": risk_analysis["statistics"],
            "safe_corridor_file": str(safe_corridor_path),
            "safe_corridor_url": f"/results/{safe_corridor_path.name}",
            "safe_corridor_statistics": safe_corridor_analysis["statistics"],
            "class_count": len(detected_classes),

            "inference_time_ms": round(inference_time_ms, 2),

            "mask_file": str(mask_path),
            "overlay_file": str(overlay_path),
            "preview_file": str(preview_path),
            "preview_url": f"/results/{preview_path.name}",

            "traversability_file": str(traversability_path),
            "traversability_url": f"/results/{traversability_path.name}",

            "confidence_file": confidence_npy_path.name,
            "confidence_url": f"/results/{confidence_path.name}",
            "mean_confidence": round(mean_confidence * 100, 2),
            "low_confidence_percentage": round(low_confidence_percentage, 2),

            "class_ids_file": str(class_ids_path),
            "class_ids_filename": class_ids_path.name,
            "class_ids_base64": class_ids_base64,

            "message": "Segmentation completed successfully",
            "upload_id": upload_id,
        }

        metadata_path = RESULTS_DIR / f"{safe_stem}_results.json"
        created_paths.append(metadata_path)

        metadata = {
            key: value
            for key, value in response_data.items()
            if key != "class_ids_base64"
        }

        metadata["artifacts"] = {
            "original_image": uploaded_path.name,
            "overlay": overlay_path.name,
            "mask": mask_path.name,
            "class_ids": class_ids_path.name,
            "traversability": traversability_path.name,
            "confidence": confidence_path.name,
            "confidence_raw": confidence_npy_path.name,
            "risk_map": risk_map_path.name,
            "safe_corridor": safe_corridor_path.name,
        }

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        db.commit()
        return JSONResponse(response_data)


    except Exception:
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
        for artifact_path in reversed(created_paths):
            try:
                artifact_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        if db is not None:
            db.close()

@app.get("/results/archive/{class_ids_filename}")
def download_results_archive(
    class_ids_filename: str,
    upload_id: str,
    current_user: User = Depends(get_current_user)
):
    if (
        Path(class_ids_filename).name != class_ids_filename
        or not class_ids_filename.endswith("_class_ids.png")
        or not re.fullmatch(r"[0-9a-f]{32}", upload_id)
    ):
        raise HTTPException(status_code=400, detail="Invalid archive parameters.")

    stem = class_ids_filename[:-len("_class_ids.png")]

    if not stem.endswith(upload_id[:8]):
        raise HTTPException(status_code=400, detail="Image ID does not match.")

    metadata_path = RESULTS_DIR / f"{stem}_results.json"

    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Results metadata not found.")

    metadata_upload_id = _get_upload_id(stem)

    if metadata_upload_id != upload_id:
        raise HTTPException(
            status_code=400,
            detail="Archive metadata does not match the requested image."
        )

    db = SessionLocal()
    try:
        check_upload_access(upload_id, current_user, db)
    finally:
        db.close()

    original_candidates = list(UPLOADS_DIR.glob(f"{upload_id}.*"))

    if len(original_candidates) != 1 or not original_candidates[0].is_file():
        raise HTTPException(status_code=404, detail="Original image not found.")

    artifact_suffixes = [
        "_mask.png",
        "_overlay.png",
        "_preview.jpg",
        "_class_ids.png",
        "_traversability.png",
        "_confidence.png",
        "_confidence.npy",
        "_risk_map.png",
        "_safe_corridor.png",
    ]

    archive_buffer = BytesIO()

    try:
        with zipfile.ZipFile(
            archive_buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:

            original_path = original_candidates[0]
            archive.write(
                original_path,
                arcname=f"original/{original_path.name}"
            )

            for suffix in artifact_suffixes:
                artifact = RESULTS_DIR / f"{stem}{suffix}"

                if artifact.is_file():
                    archive.write(
                        artifact,
                        arcname=f"results/{artifact.name}"
                    )

            archive.write(
                metadata_path,
                arcname="results.json"
            )

    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create archive: {error}"
        )

    archive_buffer.seek(0)

    return StreamingResponse(
        archive_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition":
                f'attachment; filename="segmentation_{stem}.zip"'
        }
    )


@app.get("/results/{filename}")
def get_result(
    filename: str,
    download: bool = False,
    current_user: User = Depends(get_current_user)
):
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = RESULTS_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")

    db = SessionLocal()
    try:
        check_result_access(safe_name, current_user, db)
    finally:
        db.close()

    return FileResponse(
        file_path,
        filename=safe_name if download else None
    )




class DatasetCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: Optional[str] = Field(default=None, max_length=500)


class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    description: Optional[str] = Field(default=None, max_length=500)


class ImageSplitRequest(BaseModel):
    split_id: Optional[int] = None


class ConfidenceRequest(BaseModel):
    confidence_filename: str
    low_threshold: int = Field(default=85, ge=0, le=100)
    high_threshold: int = Field(default=99, ge=0, le=100)


class TraversabilityRequest(BaseModel):
    class_ids_filename: str
    obstacle_radius: int = Field(default=12, ge=4, le=30)


@app.post("/confidence/recalculate")
def recalculate_confidence(
    request: ConfidenceRequest,
    current_user: User = Depends(get_current_user)
):
    """Recalculate confidence categories from saved confidence data."""

    filename = request.confidence_filename

    if (
        Path(filename).name != filename
        or not filename.endswith("_confidence.npy")
    ):
        raise HTTPException(status_code=400, detail="Invalid confidence filename.")

    if request.low_threshold >= request.high_threshold:
        raise HTTPException(
            status_code=400,
            detail="Low threshold must be lower than High threshold."
        )

    db = SessionLocal()
    try:
        check_result_access(filename, current_user, db)
    finally:
        db.close()

    confidence_path = RESULTS_DIR / filename

    if not confidence_path.is_file():
        raise HTTPException(status_code=404, detail="Confidence data file not found.")

    try:
        confidence_map = _load_confidence_map(confidence_path)
        confidence_map = np.clip(confidence_map, 0.0, 1.0)

        low_limit = request.low_threshold / 100.0
        high_limit = request.high_threshold / 100.0

        categories = {
            "low": confidence_map < low_limit,
            "medium": (
                (confidence_map >= low_limit)
                & (confidence_map < high_limit)
            ),
            "high": confidence_map >= high_limit,
        }

        colors = {
            "low": (231, 76, 60),
            "medium": (241, 196, 15),
            "high": (46, 204, 113),
        }

        colored = np.zeros((*confidence_map.shape, 3), dtype=np.uint8)
        statistics = {}
        total_pixels = int(confidence_map.size)

        for name, category_mask in categories.items():
            colored[category_mask] = colors[name]
            pixels = int(np.count_nonzero(category_mask))

            statistics[name] = {
                "pixels": pixels,
                "percentage": round(pixels / total_pixels * 100, 2)
            }

        output_name = filename[:-4] + ".png"
        output_path = RESULTS_DIR / output_name

        Image.fromarray(colored).save(output_path)

        return JSONResponse({
            "message": "Confidence map recalculated successfully.",
            "confidence_url": f"/results/{output_name}",
            "low_threshold": request.low_threshold,
            "high_threshold": request.high_threshold,
            "mean_confidence": round(
                float(np.mean(confidence_map)) * 100, 2
            ),
            "width": int(confidence_map.shape[1]),
            "height": int(confidence_map.shape[0]),
            "total_pixels": total_pixels,
            "statistics": statistics
        })

    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Confidence map recalculation failed: {error}"
        )


@app.post("/traversability/recalculate")
def recalculate_traversability(
    request: TraversabilityRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Recalculate traversability from stored segmentation artifacts.
    The segmentation model is not executed again.
    """

    filename = request.class_ids_filename

    # Only accept filenames generated by the prediction endpoint.
    if (
        Path(filename).name != filename
        or not filename.endswith("_class_ids.png")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid class-ID filename."
        )

    stem = filename[:-len("_class_ids.png")]

    db = SessionLocal()
    try:
        check_result_access(filename, current_user, db)
    finally:
        db.close()

    class_ids_path = RESULTS_DIR / filename
    confidence_path = RESULTS_DIR / f"{stem}_confidence.npy"
    output_path = RESULTS_DIR / f"{stem}_traversability.png"

    if not class_ids_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Class-ID mask not found."
        )

    if not confidence_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Original confidence map not found. "
                   "Run segmentation again to generate it."
        )

    try:
        with Image.open(class_ids_path) as image:
            width, height = image.size

            if (
                width <= 0
                or height <= 0
                or width > MAX_IMAGE_SIDE
                or height > MAX_IMAGE_SIDE
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ValueError("Class-ID mask exceeds the allowed dimensions.")

            mask = np.asarray(image.convert("L"), dtype=np.uint8)

        confidence_map = _load_confidence_map(confidence_path)

        if confidence_map.shape != mask.shape:
            raise ValueError("Confidence and mask dimensions do not match.")

        traversability = create_traversability_map(
            mask,
            confidence_map,
            obstacle_radius=request.obstacle_radius
        )

        colored = colorize_traversability_map(traversability)
        Image.fromarray(colored).save(output_path)

        # Regenerate the safe corridor using the updated traversability map.
        risk_map_path = RESULTS_DIR / f"{stem}_risk_map.png"

        if risk_map_path.is_file():
            with Image.open(risk_map_path) as risk_image:
                risk_map = np.asarray(
                    risk_image.convert("L"), dtype=np.uint8
                )
        else:
            risk_analysis = analyze_risk(mask, confidence_map)
            risk_map = risk_analysis["risk_map"]
            Image.fromarray(risk_map, mode="L").save(risk_map_path)

        safe_corridor_analysis = analyze_safe_corridor(
            traversability,
            risk_map
        )

        safe_corridor_path = (
            RESULTS_DIR / f"{stem}_safe_corridor.png"
        )
        Image.fromarray(
            safe_corridor_analysis["visualization"]
        ).save(safe_corridor_path)

    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not recalculate traversability: {error}"
        )

    total_pixels = int(traversability.size)

    categories = {
        "high": traversability >= 200,
        "medium": (traversability >= 100) & (traversability < 200),
        "low": (traversability > 0) & (traversability < 100),
        "blocked": traversability == 0,
    }

    statistics = {}

    for name, category_mask in categories.items():
        pixels = int(np.count_nonzero(category_mask))

        statistics[name] = {
            "pixels": pixels,
            "percentage": round(pixels / total_pixels * 100, 2)
        }

    return JSONResponse({
        "message": "Traversability map recalculated successfully.",
        "traversability_url": f"/results/{output_path.name}",
        "safe_corridor_url": f"/results/{safe_corridor_path.name}",
        "safe_corridor_statistics": safe_corridor_analysis["statistics"],
        "obstacle_radius": request.obstacle_radius,
        "width": int(traversability.shape[1]),
        "height": int(traversability.shape[0]),
        "total_pixels": total_pixels,
        "statistics": statistics
    })


@app.get("/evaluate")
def get_model_evaluation(
    current_user: User = Depends(get_current_user)
):
    metrics_path = RESULTS_DIR / "evaluation_metrics.json"

    if not metrics_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run backend.evaluate_dataset first."
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/evaluate/all")
def get_all_evaluation_results(
    current_user: User = Depends(get_current_user)
):
    metrics_path = RESULTS_DIR / "evaluation_metrics_all.json"

    if not metrics_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Full evaluation results not found. Run backend.evaluate_dataset first."
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/evaluation/sample/{split}/{filename}")
def get_evaluation_sample(
    split: str,
    filename: str,
    current_user: User = Depends(get_current_user)
):
    if split not in {"train", "val"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset split."
        )

    safe_name = Path(filename).name

    if "/" in filename or "\\" in filename or not safe_name.endswith(".png"):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    image_path = (
        BASE_DIR / "data" / "evaluation" / "cityscapes8"
        / "images" / split / safe_name
    )
    mask_path = (
        BASE_DIR / "data" / "evaluation" / "cityscapes8"
        / "masks" / split / safe_name
    )

    if not image_path.is_file() or not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image or ground-truth mask not found."
        )

    return {
        "filename": safe_name,
        "image_url": f"/evaluation/file/{split}/images/{safe_name}",
        "mask_url": f"/evaluation/file/{split}/masks/{safe_name}",
    }


@app.get("/evaluation/file/{split}/{kind}/{filename}")
def get_evaluation_file(
    split: str,
    kind: str,
    filename: str,
    current_user: User = Depends(get_current_user)
):
    if split not in {"train", "val"} or kind not in {"images", "masks"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset path."
        )

    safe_name = Path(filename).name

    if "/" in filename or "\\" in filename or not safe_name.endswith(".png"):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    file_path = (
        BASE_DIR / "data" / "evaluation" / "cityscapes8"
        / kind / split / safe_name
    )

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found."
        )

    return FileResponse(file_path)


@app.get("/evaluation/prediction/{split}/{filename}")
def get_evaluation_prediction(
    split: str,
    filename: str,
    current_user: User = Depends(get_current_user)
):
    if split not in {"train", "val"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset split."
        )

    safe_name = Path(filename).name

    if safe_name != filename or not safe_name.endswith(".png"):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    image_path = (
        BASE_DIR / "data" / "evaluation" / "cityscapes8"
        / "images" / split / safe_name
    )

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset image not found."
        )

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
            Image.fromarray(
                np.asarray(colored).astype(np.uint8)
            ).save(output_path)

        return FileResponse(
            output_path,
            media_type="image/png"
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc


@app.get("/evaluation/ground-truth/{split}/{filename}")
def get_evaluation_ground_truth_color(
    split: str,
    filename: str,
    current_user: User = Depends(get_current_user)
):
    if split not in {"train", "val"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset split."
        )

    safe_name = Path(filename).name

    if safe_name != filename or not safe_name.endswith(".png"):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    mask_path = (
        BASE_DIR / "data" / "evaluation" / "cityscapes8"
        / "masks" / split / safe_name
    )

    if not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Ground-truth mask not found."
        )

    raw_to_train = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
    }

    with Image.open(mask_path) as source:
        raw_mask = np.asarray(source, dtype=np.uint8)

    train_mask = np.full(
        raw_mask.shape,
        255,
        dtype=np.uint8
    )

    for raw_id, train_id in raw_to_train.items():
        train_mask[raw_mask == raw_id] = train_id

    colored = colorize_mask(train_mask)

    output_dir = RESULTS_DIR / "evaluation_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"gt_{split}_{safe_name}"

    Image.fromarray(colored).save(output_path)

    return FileResponse(
        output_path,
        media_type="image/png"
    )


@app.get("/batch-analytics")
def batch_analytics(
    current_user: User = Depends(get_current_user)
):
    """Return aggregate analytics for the 120-image batch."""

    summary_path = (
        BASE_DIR
        / "archive"
        / "batch_reports"
        / "batch_analysis_summary.json"
    )

    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Batch analytics not found. Run backend.analyze_batch_results first."
        )

    try:
        with summary_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read batch analytics: {exc}"
        )

    return data




@app.post("/datasets")
def create_dataset(
    request: DatasetCreateRequest,
    current_user=Depends(require_roles("Technician", "Administrator")),
):
    db = SessionLocal()
    try:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="Dataset name cannot be empty.")

        description = request.description
        if description is not None:
            description = description.strip() or None

        dataset = Dataset(
            name=name,
            description=description,
            owner_id=current_user.id,
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "owner_id": dataset.owner_id,
            "created_at": dataset.created_at,
            "image_count": 0,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.patch("/datasets/{dataset_id}")
def update_dataset(
    dataset_id: int,
    request: DatasetUpdateRequest,
    current_user=Depends(require_roles("Technician", "Administrator")),
):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        if dataset.name == "Uploaded images":
            raise HTTPException(
                status_code=403,
                detail="The system dataset cannot be modified.",
            )

        updates = request.dict(exclude_unset=True)

        if "name" in updates:
            name = updates["name"].strip()
            if not name:
                raise HTTPException(
                    status_code=422,
                    detail="Dataset name cannot be empty.",
                )
            dataset.name = name

        if "description" in updates:
            description = updates["description"]
            dataset.description = (
                description.strip() or None
                if description is not None
                else None
            )

        db.commit()
        db.refresh(dataset)

        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "owner_id": dataset.owner_id,
            "created_at": dataset.created_at,
            "image_count": len(dataset.images),
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.delete("/datasets/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    current_user=Depends(require_roles("Administrator")),
):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        if dataset.name == "Uploaded images":
            raise HTTPException(
                status_code=403,
                detail="The system dataset cannot be deleted.",
            )

        if dataset.images or dataset.splits:
            raise HTTPException(
                status_code=409,
                detail="Dataset must be empty and have no splits before deletion.",
            )

        db.delete(dataset)
        db.commit()

        return {
            "message": "Dataset deleted successfully.",
            "dataset_id": dataset_id,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()




@app.get("/datasets/{dataset_id}/splits")
def get_dataset_splits(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found."
            )

        splits = db.query(DatasetSplit).filter(
            DatasetSplit.dataset_id == dataset_id
        ).order_by(DatasetSplit.id).all()

        return {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "splits": [
                {
                    "id": split.id,
                    "name": split.name,
                    "image_count": len(split.images),
                }
                for split in splits
            ],
        }
    finally:
        db.close()


@app.post("/datasets/{dataset_id}/splits")
def create_dataset_splits(
    dataset_id: int,
    current_user: User = Depends(require_roles("Technician", "Administrator")),
):
    """Create the standard Train, Val and Test splits for a dataset."""
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        if dataset.name == "Uploaded images":
            raise HTTPException(
                status_code=403,
                detail="The system dataset cannot have managed splits."
            )

        split_names = ("Train", "Val", "Test")
        existing = {
            split.name: split
            for split in db.query(DatasetSplit).filter(
                DatasetSplit.dataset_id == dataset_id
            ).all()
        }

        for name in split_names:
            if name not in existing:
                split = DatasetSplit(dataset_id=dataset_id, name=name)
                db.add(split)
                existing[name] = split

        db.commit()

        return {
            "dataset_id": dataset_id,
            "splits": [
                {
                    "id": existing[name].id,
                    "name": name,
                    "image_count": len(existing[name].images),
                }
                for name in split_names
            ],
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.patch("/datasets/{dataset_id}/images/{image_id}/split")
def assign_image_split(
    dataset_id: int,
    image_id: int,
    request: ImageSplitRequest,
    current_user: User = Depends(require_roles("Technician", "Administrator")),
):
    """Assign an image to Train, Val or Test; null split_id removes assignment."""
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        if dataset.name == "Uploaded images":
            raise HTTPException(
                status_code=403,
                detail="The system dataset cannot have managed splits."
            )

        image = db.query(DBImage).filter(
            DBImage.id == image_id,
            DBImage.dataset_id == dataset_id,
        ).first()

        if image is None:
            raise HTTPException(
                status_code=404,
                detail="Image not found in this dataset."
            )

        if request.split_id is None:
            image.split_id = None
        else:
            split = db.query(DatasetSplit).filter(
                DatasetSplit.id == request.split_id,
                DatasetSplit.dataset_id == dataset_id,
            ).first()

            if split is None:
                raise HTTPException(
                    status_code=404,
                    detail="Split not found in this dataset."
                )

            image.split_id = split.id

        db.commit()

        return {
            "message": "Image split assignment updated.",
            "dataset_id": dataset_id,
            "image_id": image_id,
            "split_id": image.split_id,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.delete("/datasets/{dataset_id}/images/{image_id}")
def delete_dataset_image(
    dataset_id: int,
    image_id: int,
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    ),
):
    """Delete an image, its database records and its own result files."""
    db = SessionLocal()
    files_to_delete = set()

    try:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found."
            )

        if dataset.name == "Uploaded images":
            raise HTTPException(
                status_code=403,
                detail="Images in the system dataset cannot be deleted."
            )

        image = db.query(DBImage).filter(
            DBImage.id == image_id,
            DBImage.dataset_id == dataset_id,
        ).first()

        if image is None:
            raise HTTPException(
                status_code=404,
                detail="Image not found in this dataset."
            )

        uploads_root = UPLOADS_DIR.resolve()
        results_root = RESULTS_DIR.resolve()
        ground_truth_root = GROUND_TRUTH_DIR.resolve()

        def add_safe_file(raw_path, expected_root, relative_to_base=False):
            if not isinstance(raw_path, str) or not raw_path:
                return

            candidate = Path(raw_path)

            if relative_to_base and not candidate.is_absolute():
                candidate = BASE_DIR / candidate
            elif not candidate.is_absolute():
                candidate = expected_root / candidate

            try:
                resolved = candidate.resolve()
                if resolved.parent != expected_root:
                    return
                if resolved.is_file():
                    files_to_delete.add(resolved)
            except (OSError, RuntimeError):
                return

        # Original image.
        add_safe_file(
            image.file_path,
            uploads_root,
            relative_to_base=True
        )

        # Database-linked prediction mask and overlays.
        add_safe_file(
            image.mask_path,
            results_root,
            relative_to_base=True
        )

        # Independent Ground Truth annotation.
        add_safe_file(
            image.ground_truth_path,
            ground_truth_root,
            relative_to_base=True
        )

        predictions = db.query(Prediction).filter(
            Prediction.image_id == image_id
        ).all()

        prediction_count = len(predictions)

        for prediction in predictions:
            add_safe_file(
                prediction.result_path,
                results_root,
                relative_to_base=True
            )

        # Identify this upload using its original UUID filename.
        upload_id = Path(image.file_path or "").stem

        if re.fullmatch(r"[0-9a-f]{32}", upload_id):
            metadata_candidates = list(
                RESULTS_DIR.glob(
                    "*_" + upload_id[:8] + "_results.json"
                )
            )

            for metadata_path in metadata_candidates:
                try:
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue

                if metadata.get("upload_id") != upload_id:
                    continue

                # Metadata itself.
                add_safe_file(metadata_path, results_root)

                # Additional artifacts recorded by predict().
                artifacts = metadata.get("artifacts", {})
                if isinstance(artifacts, dict):
                    add_safe_file(
                        artifacts.get("original_image"),
                        uploads_root
                    )

                    for key in (
                        "overlay",
                        "mask",
                        "class_ids",
                        "traversability",
                        "confidence",
                        "confidence_raw",
                        "risk_map",
                        "safe_corridor",
                    ):
                        add_safe_file(
                            artifacts.get(key),
                            results_root
                        )

                # Paths stored in the response metadata.
                for key in (
                    "preview_file",
                    "traversability_file",
                    "class_ids_file",
                    "risk_map_file",
                    "safe_corridor_file",
                ):
                    add_safe_file(
                        metadata.get(key),
                        results_root,
                        relative_to_base=True
                    )

                confidence_file = metadata.get("confidence_file")
                add_safe_file(confidence_file, results_root)

        for prediction in predictions:
            db.query(DetectedClass).filter(
                DetectedClass.prediction_id == prediction.id
            ).delete(synchronize_session=False)
            db.delete(prediction)

        db.delete(image)
        db.commit()

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    deleted_files = 0
    cleanup_failures = 0

    # Physical deletion happens only after the database commit.
    for file_path in files_to_delete:
        try:
            file_path.unlink(missing_ok=True)
            deleted_files += 1
        except OSError:
            cleanup_failures += 1

    return {
        "message": "Image and related records deleted.",
        "dataset_id": dataset_id,
        "image_id": image_id,
        "deleted_predictions": prediction_count,
        "deleted_files": deleted_files,
        "cleanup_failures": cleanup_failures,
    }


@app.patch("/images/{image_id}/dataset")
def move_image_to_dataset(
    image_id: int,
    target_dataset_id: int,
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    ),
):
    """Move an image to another managed dataset."""
    db = SessionLocal()
    try:
        image = db.query(DBImage).filter(
            DBImage.id == image_id
        ).first()

        if image is None:
            raise HTTPException(
                status_code=404,
                detail="Image not found."
            )

        target_dataset = db.query(Dataset).filter(
            Dataset.id == target_dataset_id
        ).first()

        if target_dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Target dataset not found."
            )

        if target_dataset.name == "Uploaded images":
            raise HTTPException(
                status_code=403,
                detail="Images cannot be moved into the system dataset."
            )

        source_dataset_id = image.dataset_id

        if source_dataset_id == target_dataset_id:
            raise HTTPException(
                status_code=400,
                detail="Image already belongs to this dataset."
            )

        image.dataset_id = target_dataset_id
        image.split_id = None

        db.commit()

        return {
            "message": "Image moved successfully.",
            "image_id": image.id,
            "source_dataset_id": source_dataset_id,
            "target_dataset_id": target_dataset_id,
            "split_id": None,
        }

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.get("/datasets")
def get_datasets(
    current_user: User = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        datasets = db.query(Dataset).order_by(Dataset.id.desc()).all()

        return [
            {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "created_at": dataset.created_at.isoformat()
                if dataset.created_at else None,
                "image_count": len(dataset.images),
            }
            for dataset in datasets
        ]
    finally:
        db.close()


@app.get("/datasets/{dataset_id}/images")
def get_dataset_images(
    dataset_id: int,
    split_id: Optional[int] = None,
    current_user: User = Depends(require_roles("Technician", "Administrator"))
):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found."
            )

        image_query = db.query(DBImage).filter(
            DBImage.dataset_id == dataset_id
        )

        if split_id is not None:
            image_query = image_query.filter(
                DBImage.split_id == split_id
            )

        images = image_query.order_by(DBImage.id.desc()).all()

        return {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "image_count": len(images),
            "images": [
                {
                    "id": image.id,
                    "filename": image.filename,
                    "width": image.width,
                    "height": image.height,
                    "file_size": image.file_size,
                    "has_mask": image.has_mask,
                    "mask_path": image.mask_path,
                    "has_ground_truth": image.has_ground_truth,
                    "ground_truth_format": image.ground_truth_format,
                    "split_id": image.split_id,
                    "split_name": image.split.name if image.split else None,
                    "created_at": image.created_at.isoformat()
                    if image.created_at else None,
                }
                for image in images
            ],
        }
    finally:
        db.close()


@app.post("/datasets/{dataset_id}/images/batch")
async def batch_predict(
    dataset_id: int,
    files: list[UploadFile] = File(...),
    current_user: User = Depends(require_roles("Technician", "Administrator"))
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were uploaded."
        )

    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found."
            )

        if dataset.name == "Uploaded images":
            raise HTTPException(
                status_code=403,
                detail="Batch uploads to the system dataset are not allowed."
            )
    finally:
        db.close()

    results = []

    for file in files:
        try:
            response = await predict(
                file,
                current_user,
                target_dataset_id=dataset_id
            )
            result = json.loads(response.body)

            results.append({
                "filename": file.filename,
                "status": "success",
                "result": result
            })

        except HTTPException as exc:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": exc.detail,
                "status_code": exc.status_code
            })

        except Exception as exc:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(exc)
            })

    successful = sum(
        1 for item in results
        if item["status"] == "success"
    )

    return {
        "dataset_id": dataset_id,
        "total_files": len(files),
        "successful": successful,
        "failed": len(files) - successful,
        "results": results
    }



# GROUND_TRUTH_API

GROUND_TRUTH_DIR = BASE_DIR / "data" / "ground_truth"
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_GT_VALUES = set(range(19)) | {255}
MAX_GT_UPLOAD_BYTES = 20 * 1024 * 1024


def get_ground_truth_image(db, dataset_id: int, image_id: int):
    image = (
        db.query(DBImage)
        .filter(
            DBImage.id == image_id,
            DBImage.dataset_id == dataset_id
        )
        .first()
    )

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Image not found in this dataset."
        )

    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id
    ).first()
    if dataset is not None and dataset.name == "Uploaded images":
        raise HTTPException(
            status_code=403,
            detail="Ground Truth cannot be managed in the system dataset."
        )

    return image


def resolve_ground_truth_path(relative_path: str) -> Path:
    root = GROUND_TRUTH_DIR.resolve()
    path = (BASE_DIR / relative_path).resolve()

    if path.parent != root or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Ground Truth file not found."
        )

    return path


@app.post(
    "/datasets/{dataset_id}/images/{image_id}/ground-truth"
)
async def upload_ground_truth(
    dataset_id: int,
    image_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    )
):
    if not file.filename or not file.filename.lower().endswith(".png"):
        raise HTTPException(
            status_code=400,
            detail="Ground Truth must be a PNG file."
        )

    content = await file.read(MAX_GT_UPLOAD_BYTES + 1)
    await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(content) > MAX_GT_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Ground Truth file exceeds 20 MB."
        )

    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.format != "PNG":
                raise HTTPException(
                    status_code=400,
                    detail="Only PNG format is supported."
                )

            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail="Ground Truth image exceeds pixel limit."
                )

            if opened.mode not in (
                "L", "P", "I", "I;16", "I;16B", "I;16L"
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Ground Truth must be a single-channel "
                        "class-ID mask, not an RGB color image."
                    )
                )

            mask = np.asarray(opened).copy()

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted PNG file."
        )

    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise HTTPException(
            status_code=400,
            detail="Ground Truth must contain integer class IDs."
        )

    if mask.shape != (opened.height, opened.width):
        raise HTTPException(
            status_code=400,
            detail="Invalid Ground Truth dimensions."
        )

    unique_values = set(np.unique(mask).tolist())
    invalid_values = unique_values - ALLOWED_GT_VALUES

    if invalid_values:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid class IDs. Allowed values: 0-18 and 255. "
                f"Found: {sorted(invalid_values)[:20]}"
            )
        )

    if not any(value in range(19) for value in unique_values):
        raise HTTPException(
            status_code=400,
            detail="Ground Truth must contain at least one valid class."
        )

    db = SessionLocal()
    new_path = None

    try:
        image = get_ground_truth_image(db, dataset_id, image_id)

        if image.width != opened.width or image.height != opened.height:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Ground Truth dimensions ({opened.width}x{opened.height}) "
                    f"do not match source image ({image.width}x{image.height})."
                )
            )

        filename = f"image_{image_id}_{uuid.uuid4().hex}.png"
        new_path = GROUND_TRUTH_DIR / filename
        temp_path = GROUND_TRUTH_DIR / f".{filename}.tmp.png"

        Image.fromarray(mask.astype(np.uint8)).save(
            temp_path,
            format="PNG"
        )
        temp_path.replace(new_path)

        old_relative_path = image.ground_truth_path

        image.has_ground_truth = True
        image.ground_truth_path = str(
            new_path.relative_to(BASE_DIR)
        )
        image.ground_truth_format = "trainIds"

        db.commit()

        if old_relative_path:
            try:
                old_path = resolve_ground_truth_path(old_relative_path)
                old_path.unlink(missing_ok=True)
            except (HTTPException, OSError):
                pass

        return {
            "status": "success",
            "message": "Ground Truth uploaded successfully.",
            "image_id": image.id,
            "has_ground_truth": True,
            "format": "trainIds",
            "width": opened.width,
            "height": opened.height,
            "classes": sorted(
                value for value in unique_values if value != 255
            ),
            "ignored_pixels": int(np.count_nonzero(mask == 255))
        }

    except Exception:
        db.rollback()

        if new_path and new_path.exists():
            new_path.unlink(missing_ok=True)

        raise
    finally:
        db.close()


@app.get(
    "/datasets/{dataset_id}/images/{image_id}/ground-truth"
)
def get_ground_truth(
    dataset_id: int,
    image_id: int,
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    )
):
    db = SessionLocal()

    try:
        image = get_ground_truth_image(db, dataset_id, image_id)

        if not image.has_ground_truth or not image.ground_truth_path:
            raise HTTPException(
                status_code=404,
                detail="Ground Truth is not available for this image."
            )

        path = resolve_ground_truth_path(image.ground_truth_path)

        return FileResponse(
            path=str(path),
            media_type="image/png",
            filename=f"ground_truth_{image_id}.png"
        )
    finally:
        db.close()


@app.delete(
    "/datasets/{dataset_id}/images/{image_id}/ground-truth"
)
def delete_ground_truth(
    dataset_id: int,
    image_id: int,
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    )
):
    db = SessionLocal()

    try:
        image = get_ground_truth_image(db, dataset_id, image_id)

        old_relative_path = image.ground_truth_path

        image.has_ground_truth = False
        image.ground_truth_path = None
        image.ground_truth_format = None

        db.commit()

        if old_relative_path:
            try:
                old_path = resolve_ground_truth_path(old_relative_path)
                old_path.unlink(missing_ok=True)
            except (HTTPException, OSError):
                pass

        return {
            "status": "success",
            "message": "Ground Truth deleted.",
            "image_id": image_id,
            "has_ground_truth": False
        }
    finally:
        db.close()
