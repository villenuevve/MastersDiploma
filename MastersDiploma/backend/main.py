
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import load_image_from_bytes
from backend.depth_estimation import (
    predict_depth,
    normalize_depth_for_visualization,
)
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
from datetime import datetime
from typing import Optional
from io import BytesIO, StringIO
from time import perf_counter
import base64
import numpy as np
import json
import cv2
import uuid
import zipfile
import re
from threading import Lock, Thread
import random
import csv

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

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_SIDE = 12_000
UPLOAD_CHUNK_SIZE = 1024 * 1024

VIDEO_RESULTS_DIR = RESULTS_DIR / "videos"
VIDEO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_VIDEO_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 120
MAX_VIDEO_FRAME_PIXELS = 8_000_000

# VIDEO_PROGRESS_IMPLEMENTATION
VIDEO_JOBS = {}
VIDEO_JOBS_LOCK = Lock()


def _video_job_update(video_id, **changes):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(video_id)
        if job is not None:
            job.update(changes)


def _get_video_job_for_user(video_id, current_user):
    if not re.fullmatch(r"[0-9a-f]{32}", video_id):
        raise HTTPException(status_code=404, detail="Video job not found.")

    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(video_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Video job not found. It may have been cleared after a server restart."
            )
        result = dict(job)

    role_name = getattr(getattr(current_user, "role", None), "name", None)
    if (
        role_name not in ("Administrator", "Technician")
        and result.get("owner_id") != current_user.id
    ):
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this video job."
        )

    return result


def _process_video_job(video_id, source_path, output_path, metadata_path,
                       owner_id, original_filename, frame_interval):
    started_at = perf_counter()
    capture = None
    writer = None
    frame_index = 0
    processed_frames = 0
    inference_time_ms = 0.0
    flow_frames = 0
    last_mask = None
    previous_gray = None
    cancelled = False

    FLOW_SCALE = 0.5
    INFERENCE_MAX_SIDE = 1024
    MOTION_THRESHOLD = 3.5

    _video_job_update(
        video_id,
        status="processing",
        message="Initializing video decoder and segmentation model.",
        started_at=started_at
    )

    try:
        capture = cv2.VideoCapture(str(source_path))

        if not capture.isOpened():
            raise ValueError("Could not decode the uploaded video.")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video frame rate.")
        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions.")
        if width * height > MAX_VIDEO_FRAME_PIXELS:
            raise ValueError("Video resolution exceeds the 8-megapixel limit.")
        if (
            source_frame_count > 0
            and source_frame_count / fps > MAX_VIDEO_DURATION_SECONDS
        ):
            raise ValueError("Video duration exceeds the 120-second limit.")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise ValueError("Could not initialize the MP4 encoder.")

        _video_job_update(
            video_id,
            message="Segmentation with Optical Flow started.",
            width=width,
            height=height,
            source_fps=round(fps, 3),
            source_frames=source_frame_count,
            total_frames_to_process=source_frame_count
        )

        flow_width = max(1, int(width * FLOW_SCALE))
        flow_height = max(1, int(height * FLOW_SCALE))

        while True:
            with VIDEO_JOBS_LOCK:
                job = VIDEO_JOBS.get(video_id, {})
                cancel_requested = job.get("cancel_requested", False)

            if cancel_requested:
                cancelled = True
                break

            success, frame_bgr = capture.read()
            if not success:
                break

            if frame_index / fps >= MAX_VIDEO_DURATION_SECONDS:
                raise ValueError("Video duration exceeds the 120-second limit.")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(
                cv2.resize(
                    frame_bgr,
                    (flow_width, flow_height),
                    interpolation=cv2.INTER_AREA
                ),
                cv2.COLOR_BGR2GRAY
            )

            motion = 0.0
            warped_mask = None

            if last_mask is not None and previous_gray is not None:
                # Current-to-previous optical flow for mask remapping.
                flow = cv2.calcOpticalFlowFarneback(
                    gray,
                    previous_gray,
                    None,
                    0.5,
                    3,
                    15,
                    2,
                    5,
                    1.1,
                    0
                )

                magnitude = np.sqrt(
                    flow[..., 0] ** 2 + flow[..., 1] ** 2
                )
                motion = float(np.mean(magnitude))

                flow = cv2.resize(
                    flow,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR
                )
                flow /= FLOW_SCALE

                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32)
                )

                warped_mask = cv2.remap(
                    last_mask,
                    grid_x + flow[..., 0],
                    grid_y + flow[..., 1],
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REPLICATE
                )

                flow_frames += 1

            periodic_inference = (
                last_mask is None
                or frame_index % max(1, frame_interval) == 0
            )

            motion_inference = (
                last_mask is not None
                and motion >= MOTION_THRESHOLD
            )

            run_inference = periodic_inference or motion_inference

            if run_inference:
                scale = min(
                    1.0,
                    INFERENCE_MAX_SIDE / max(width, height)
                )

                infer_width = max(1, int(width * scale))
                infer_height = max(1, int(height * scale))

                infer_rgb = cv2.resize(
                    frame_rgb,
                    (infer_width, infer_height),
                    interpolation=cv2.INTER_AREA
                )

                inference_start = perf_counter()
                predicted_mask = predict_mask(Image.fromarray(infer_rgb))
                inference_time_ms += (
                    perf_counter() - inference_start
                ) * 1000

                if predicted_mask.shape[:2] != (height, width):
                    predicted_mask = cv2.resize(
                        predicted_mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )

                last_mask = predicted_mask.astype(np.uint8)
                processed_frames += 1
                reason = "motion" if motion_inference and not periodic_inference else "scheduled"

            else:
                if warped_mask is None:
                    raise ValueError("Optical Flow mask propagation failed.")

                last_mask = warped_mask.astype(np.uint8)
                reason = "optical_flow"

            colored_mask = colorize_mask(last_mask)

            if colored_mask.shape[:2] != (height, width):
                colored_mask = cv2.resize(
                    colored_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            overlay = cv2.addWeighted(
                frame_rgb, 0.45,
                colored_mask, 0.55,
                0
            )

            writer.write(
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )

            previous_gray = gray
            frame_index += 1

            if frame_index % 10 == 0 or frame_index == 1 or run_inference:
                elapsed = max(perf_counter() - started_at, 0.001)
                speed = frame_index / elapsed

                progress = (
                    min(99.9, frame_index / source_frame_count * 100)
                    if source_frame_count > 0 else 0
                )

                eta = (
                    (source_frame_count - frame_index) / speed
                    if source_frame_count > 0 and speed > 0
                    else None
                )

                _video_job_update(
                    video_id,
                    processed_frames=processed_frames,
                    rendered_frames=frame_index,
                    current_frame=frame_index,
                    progress_percent=round(progress, 1),
                    speed_fps=round(speed, 2),
                    elapsed_seconds=round(elapsed, 1),
                    eta_seconds=round(eta, 1) if eta is not None else None,
                    inference_time_ms=round(inference_time_ms, 1),
                    flow_frames=flow_frames,
                    last_keyframe_reason=reason,
                    message=(
                        f"Rendered {frame_index} frames; "
                        f"SegFormer inference: {processed_frames} frames."
                    )
                )

        capture.release()
        capture = None
        writer.release()
        writer = None

        if cancelled:
            output_path.unlink(missing_ok=True)
            _video_job_update(
                video_id,
                status="cancelled",
                message="Video processing was cancelled.",
                progress_percent=round(
                    frame_index / source_frame_count * 100, 1
                ) if source_frame_count > 0 else 0,
                elapsed_seconds=round(perf_counter() - started_at, 1),
                eta_seconds=0
            )
            return

        if frame_index == 0 or processed_frames == 0:
            raise ValueError("The uploaded video contains no readable frames.")

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise ValueError("Video rendering failed.")

        elapsed = perf_counter() - started_at

        metadata = {
            "video_id": video_id,
            "owner_id": owner_id,
            "original_filename": original_filename,
            "output_filename": output_path.name,
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "source_frames": frame_index,
            "processed_frames": processed_frames,
            "rendered_frames": frame_index,
            "flow_frames": flow_frames,
            "frame_interval": frame_interval,
            "duration_seconds": round(frame_index / fps, 2),
            "inference_time_ms": round(inference_time_ms, 2),
            "processing_time_seconds": round(elapsed, 2),
            "average_inference_ms": round(
                inference_time_ms / processed_frames, 2
            ),
            "model": "SegFormer-B3",
            "video_method": "SegFormer + Optical Flow",
            "inference_max_side": INFERENCE_MAX_SIDE
        }

        temporary_metadata = metadata_path.with_suffix(".tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        temporary_metadata.replace(metadata_path)

        _video_job_update(
            video_id,
            status="completed",
            message="Video segmentation completed.",
            progress_percent=100.0,
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=0,
            speed_fps=round(frame_index / max(elapsed, 0.001), 2),
            metadata=metadata,
            video_url=f"/video/results/{video_id}"
        )

    except Exception as error:
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        _video_job_update(
            video_id,
            status="failed",
            message=str(error),
            error=str(error),
            elapsed_seconds=round(perf_counter() - started_at, 1),
            eta_seconds=None
        )

    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        source_path.unlink(missing_ok=True)



VIDEO_UPLOAD_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


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


def ensure_dataset_access(
    dataset: Dataset,
    current_user: User,
    allow_system_read: bool = False,
):
    # Read an already-loaded role without triggering lazy loading.
    role = current_user.__dict__.get("role")

    if role is not None:
        role_name = role.name
    else:
        role_id = current_user.__dict__.get("role_id")

        if role_id is None:
            raise HTTPException(
                status_code=403,
                detail="Unable to determine user role."
            )

        role_model = User.__mapper__.relationships["role"].mapper.class_
        role_db = SessionLocal()

        try:
            role_record = role_db.query(role_model).filter(
                role_model.id == role_id
            ).first()
            role_name = role_record.name if role_record else None
        finally:
            role_db.close()

    if role_name == "Administrator":
        return

    if role_name != "Technician":
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access managed datasets."
        )

    if dataset.is_system:
        if allow_system_read:
            return

        raise HTTPException(
            status_code=403,
            detail="The system dataset is read-only."
        )

    if dataset.owner_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this dataset."
        )
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
        if not isinstance(metadata, dict):
            raise HTTPException(
                status_code=404,
                detail="Invalid result metadata."
            )
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
        "_depth.npy",
        "_depth.png",
        "_uncertainty.npy",
        "_uncertainty.png",
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

    # Legacy results: resolve all images linked to the artifact paths.
    mask_rel = f"results/{stem}_mask.png"
    overlay_rel = f"results/{stem}_overlay.png"

    matching_images = (
        db.query(DBImage)
        .outerjoin(Prediction, Prediction.image_id == DBImage.id)
        .filter(
            (DBImage.mask_path == mask_rel)
            | (Prediction.result_path == overlay_rel)
        )
        .distinct()
        .all()
    )

    if not matching_images:
        raise HTTPException(status_code=404, detail="Image record not found.")

    if current_user.role.name in ("Administrator", "Technician"):
        return matching_images[0]

    # A legacy artifact shared by multiple image records has ambiguous ownership.
    if len(matching_images) != 1:
        raise HTTPException(
            status_code=403,
            detail="Access to this legacy result is ambiguous."
        )

    image = matching_images[0]

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

    if float(mapped.min()) < 0.0 or float(mapped.max()) > 1.0:
        raise ValueError("Confidence values must be within [0, 1].")

    return np.asarray(mapped, dtype=np.float32)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    target_dataset_id: Optional[int] = None,
):
    request_start = perf_counter()
    performance = {}
    if target_dataset_id is not None:
        if current_user.role.name not in ("Technician", "Administrator"):
            raise HTTPException(
                status_code=403,
                detail="Only Technicians and Administrators can upload to managed datasets."
            )

    upload_start = perf_counter()
    content_buffer = bytearray()

    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)

        if not chunk:
            break

        if len(content_buffer) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum size is 50 MB."
            )

        content_buffer.extend(chunk)

    if not content_buffer:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    content = bytes(content_buffer)
    performance["upload_read_ms"] = round(
        (perf_counter() - upload_start) * 1000, 2
    )

    image_decode_start = perf_counter()
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
        performance["image_decode_ms"] = round(
            (perf_counter() - image_decode_start) * 1000, 2
        )

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
    (
        mask,
        segmentation_stats,
        confidence_map,
        uncertainty_map,
    ) = predict_mask(
        image,
        return_stats=True,
        return_uncertainty=True,
        profile=True,
    )
    inference_time_ms = (perf_counter() - start_time) * 1000
    performance["segmentation_total_ms"] = round(
        inference_time_ms, 2
    )
    performance["segmentation_profile"] = segmentation_stats.get(
        "profile", {}
    )

    # Estimate relative depth independently of semantic segmentation.
    depth_start = perf_counter()
    depth_map, depth_stats = predict_depth(
        image,
        return_stats=True
    )
    performance["depth_total_ms"] = round(
        (perf_counter() - depth_start) * 1000, 2
    )

    postprocessing_start = perf_counter()
    postprocessing_profile = {}

    stage_start = perf_counter()
    depth_visualization = normalize_depth_for_visualization(depth_map)
    postprocessing_profile["depth_visualization_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

    # Generate segmentation visualization
    stage_start = perf_counter()
    colored_mask = colorize_mask(mask)
    postprocessing_profile["colorize_mask_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

    stage_start = perf_counter()
    overlay = create_overlay(image, colored_mask)
    postprocessing_profile["create_overlay_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

    # Sanitize uploaded filename
    original_name = Path(file.filename or "image").stem
    safe_stem = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in original_name
    ).strip("_") or "image"
    safe_stem = f"{safe_stem}_{upload_id[:8]}"

    # Generate traversability map
    stage_start = perf_counter()
    traversability = create_traversability_map(mask, confidence_map)
    postprocessing_profile["traversability_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

    # Generate Risk Map before saving its output file
    stage_start = perf_counter()
    risk_analysis = analyze_risk(mask, confidence_map)
    postprocessing_profile["risk_analysis_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

    stage_start = perf_counter()
    traversability_colored = colorize_traversability_map(traversability)
    postprocessing_profile["traversability_visualization_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

    stage_start = perf_counter()
    confidence_uint8 = np.clip(confidence_map * 255, 0, 255).astype(np.uint8)
    confidence_colored = np.zeros((*confidence_uint8.shape, 3), dtype=np.uint8)
    confidence_colored[..., 0] = 255 - confidence_uint8
    confidence_colored[..., 1] = confidence_uint8
    confidence_colored[..., 2] = 50

    mean_confidence = float(np.mean(confidence_map))
    low_confidence_percentage = float(np.mean(confidence_map < 0.5) * 100)
    postprocessing_profile["confidence_processing_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

    # Normalized predictive entropy: 0 = certain, 1 = maximally uncertain.
    stage_start = perf_counter()
    uncertainty_visualization = (
        np.clip(uncertainty_map, 0.0, 1.0) * 255
    ).astype(np.uint8)

    uncertainty_stats = {
        "method": "normalized_predictive_entropy",
        "min": round(float(np.min(uncertainty_map)), 4),
        "max": round(float(np.max(uncertainty_map)), 4),
        "mean": round(float(np.mean(uncertainty_map)), 4),
        "median": round(float(np.median(uncertainty_map)), 4),
        "p95": round(float(np.percentile(uncertainty_map, 95)), 4),
    }
    postprocessing_profile["uncertainty_processing_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

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
    depth_path = RESULTS_DIR / f"{safe_stem}_depth.png"
    depth_npy_path = RESULTS_DIR / f"{safe_stem}_depth.npy"
    uncertainty_path = RESULTS_DIR / f"{safe_stem}_uncertainty.png"
    uncertainty_npy_path = RESULTS_DIR / f"{safe_stem}_uncertainty.npy"

    stage_start = perf_counter()
    safe_corridor_analysis = analyze_safe_corridor(
        traversability,
        risk_analysis["risk_map"]
    )
    postprocessing_profile["safe_corridor_ms"] = round(
        (perf_counter() - stage_start) * 1000, 2
    )

    performance["postprocessing_profile"] = postprocessing_profile
    performance["postprocessing_ms"] = round(
        (perf_counter() - postprocessing_start) * 1000, 2
    )

    created_paths = []
    db = None
    persistence_start = perf_counter()
    persistence_profile = {}
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
            depth_path,
            depth_npy_path,
            uncertainty_path,
            uncertainty_npy_path,
        ])

        # Save segmentation results
        artifact_write_start = perf_counter()
        save_mask(mask, str(mask_path))
        save_class_ids(mask, str(class_ids_path))
        overlay.save(overlay_path)

        # Save traversability map
        Image.fromarray(traversability_colored).save(traversability_path)
        Image.fromarray(confidence_colored).save(confidence_path)
        np.save(confidence_npy_path, confidence_map)
        Image.fromarray(risk_analysis['risk_map'], mode='L').save(risk_map_path)
        Image.fromarray(
            safe_corridor_analysis["visualization"]
        ).save(safe_corridor_path)

        # Save relative depth data and its display-only visualization.
        np.save(depth_npy_path, depth_map)
        Image.fromarray(depth_visualization, mode="L").save(depth_path)

        # Save normalized predictive entropy as visualization and raw data.
        np.save(
            uncertainty_npy_path,
            uncertainty_map
        )
        Image.fromarray(
            uncertainty_visualization,
            mode="L"
        ).save(uncertainty_path)

        # Save optimized preview
        preview = overlay.convert("RGB")
        preview.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
        preview.save(
            preview_path,
            "JPEG",
            quality=85,
            optimize=False
        )

        # Reuse the saved class-ID PNG for browser pixel inspection.
        class_ids_base64 = base64.b64encode(
            class_ids_path.read_bytes()
        ).decode("ascii")

        persistence_profile["artifact_write_ms"] = round(
            (perf_counter() - artifact_write_start) * 1000, 2
        )

        # Calculate class distribution
        class_statistics_start = perf_counter()
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

        persistence_profile["class_statistics_ms"] = round(
            (perf_counter() - class_statistics_start) * 1000, 2
        )

        scene_analysis_start = perf_counter()

        obstacle_analysis_start = perf_counter()
        obstacle_analysis = analyze_obstacles(mask)
        persistence_profile["obstacle_analysis_ms"] = round(
            (perf_counter() - obstacle_analysis_start) * 1000, 2
        )

        scene_details_start = perf_counter()
        scene_analysis = analyze_scene(mask, confidence_map)
        persistence_profile["scene_details_ms"] = round(
            (perf_counter() - scene_details_start) * 1000, 2
        )

        persistence_profile["scene_analysis_ms"] = round(
            (perf_counter() - scene_analysis_start) * 1000, 2
        )

        # Save uploaded image and segmentation metadata to SQLite
        original_extension = Path(file.filename or "").suffix.lower()
        if original_extension not in {".jpg", ".jpeg", ".png"}:
            original_extension = ".img"

        uploaded_path = UPLOADS_DIR / f"{upload_id}{original_extension}"
        created_paths.append(uploaded_path)
        upload_write_start = perf_counter()
        uploaded_path.write_bytes(content)
        persistence_profile["upload_write_ms"] = round(
            (perf_counter() - upload_write_start) * 1000, 2
        )

        database_start = perf_counter()
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

                ensure_dataset_access(dataset, current_user)
            else:
                dataset = (
                    db.query(Dataset)
                    .filter(Dataset.is_system.is_(True))
                    .first()
                )

                if dataset is None:
                    dataset = Dataset(
                        name="Uploaded images",
                        description="Images uploaded through the web application.",
                        is_system=True,
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
            db_dataset_id = dataset.id
            prediction_id = prediction.id

        except Exception:
            db.rollback()
            raise

        persistence_profile["database_ms"] = round(
            (perf_counter() - database_start) * 1000, 2
        )
        performance["persistence_profile"] = persistence_profile

        response_data = {
            "performance": performance,
            "filename": file.filename or "image",
            "image_id": db_image_id,
            "dataset_id": db_dataset_id,
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

            "depth_model": depth_stats["model"],
            "depth_device": depth_stats["device"],
            "depth_type": depth_stats["depth_type"],
            "depth_inference_time_ms": depth_stats["inference_time_ms"],
            "depth_statistics": depth_stats,
            "depth_file": depth_path.name,
            "depth_url": f"/results/{depth_path.name}",
            "depth_raw_file": depth_npy_path.name,

            "uncertainty_method": uncertainty_stats["method"],
            "uncertainty_statistics": uncertainty_stats,
            "uncertainty_file": uncertainty_path.name,
            "uncertainty_url": f"/results/{uncertainty_path.name}",
            "uncertainty_raw_file": uncertainty_npy_path.name,

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
            "depth": depth_path.name,
            "depth_raw": depth_npy_path.name,
            "uncertainty": uncertainty_path.name,
            "uncertainty_raw": uncertainty_npy_path.name,
        }

        metadata_write_start = perf_counter()
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        persistence_profile["metadata_write_ms"] = round(
            (perf_counter() - metadata_write_start) * 1000, 2
        )

        database_commit_start = perf_counter()
        db.commit()
        persistence_profile["db_commit_ms"] = round(
            (perf_counter() - database_commit_start) * 1000, 2
        )

        performance["persistence_ms"] = round(
            (perf_counter() - persistence_start) * 1000, 2
        )
        performance["total_request_ms"] = round(
            (perf_counter() - request_start) * 1000, 2
        )

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
        "_depth.png",
        "_depth.npy",
        "_uncertainty.png",
        "_uncertainty.npy",
    ]

    archive_buffer = BytesIO()

    try:
        with zipfile.ZipFile(
            archive_buffer,
            mode="w",
            compression=zipfile.ZIP_STORED
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






# ================= VIDEO SEMANTIC SEGMENTATION =================

@app.post("/video/predict")
def predict_video(
    file: UploadFile = File(...),
    frame_interval: int = Form(default=5, ge=1, le=30),
    current_user: User = Depends(get_current_user),
):
    video_id = uuid.uuid4().hex
    extension = Path(file.filename or "").suffix.lower()

    if extension not in VIDEO_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Upload MP4, MOV, AVI or MKV."
        )

    with VIDEO_JOBS_LOCK:
        if any(
            job.get("status") in ("queued", "processing")
            for job in VIDEO_JOBS.values()
        ):
            raise HTTPException(
                status_code=409,
                detail="Another video is currently being processed. Wait until it finishes."
            )

    source_path = VIDEO_RESULTS_DIR / f".{video_id}_source{extension}"
    output_path = VIDEO_RESULTS_DIR / f"{video_id}.mp4"
    metadata_path = VIDEO_RESULTS_DIR / f"{video_id}.json"

    try:
        total_bytes = 0
        with source_path.open("wb") as destination:
            while True:
                chunk = file.file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 200 MB upload limit."
                    )
                destination.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty."
            )

        job = {
            "video_id": video_id,
            "owner_id": current_user.id,
            "status": "queued",
            "message": "Video uploaded. Waiting for segmentation.",
            "progress_percent": 0.0,
            "processed_frames": 0,
            "total_frames_to_process": 0,
            "current_frame": 0,
            "speed_fps": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "cancel_requested": False,
            "created_at": datetime.utcnow().isoformat()
        }

        with VIDEO_JOBS_LOCK:
            VIDEO_JOBS[video_id] = job

        worker = Thread(
            target=_process_video_job,
            args=(
                video_id,
                source_path,
                output_path,
                metadata_path,
                current_user.id,
                Path(file.filename or "video").name,
                frame_interval
            ),
            daemon=True
        )
        worker.start()

        return {
            "success": True,
            "video_id": video_id,
            "status": "queued",
            "status_url": f"/video/status/{video_id}"
        }

    except HTTPException:
        source_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        source_path.unlink(missing_ok=True)
        with VIDEO_JOBS_LOCK:
            VIDEO_JOBS.pop(video_id, None)
        raise HTTPException(
            status_code=500,
            detail=f"Could not start video processing: {str(error)}"
        )


@app.get("/video/status/{video_id}")
def get_video_status(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    job = _get_video_job_for_user(video_id, current_user)
    started_at = job.get("started_at")
    if started_at is not None and job.get("status") == "processing":
        job["elapsed_seconds"] = round(perf_counter() - started_at, 1)

    job.pop("cancel_requested", None)
    job.pop("started_at", None)
    job.pop("owner_id", None)
    return job


@app.post("/video/cancel/{video_id}")
def cancel_video(
    video_id: str,
    current_user: User = Depends(get_current_user)
):
    _get_video_job_for_user(video_id, current_user)

    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(video_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video job not found.")
        if job.get("status") not in ("queued", "processing"):
            return {
                "success": False,
                "status": job.get("status"),
                "message": "This video job is no longer active."
            }
        job["cancel_requested"] = True
        job["message"] = "Cancellation requested. Waiting for the current frame to finish."

    return {
        "success": True,
        "status": "cancelling",
        "message": "Video cancellation requested."
    }


@app.get("/video/results/{video_id}")
def get_video_result(
    video_id: str,
    download: bool = False,
    current_user: User = Depends(get_current_user),
):
    if not re.fullmatch(r"[0-9a-f]{32}", video_id):
        raise HTTPException(
            status_code=404,
            detail="Video result not found."
        )

    metadata_path = VIDEO_RESULTS_DIR / f"{video_id}.json"

    if not metadata_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video result not found."
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Could not read video metadata."
        )

    if metadata.get("video_id") != video_id:
        raise HTTPException(
            status_code=404,
            detail="Invalid video metadata."
        )

    if (
        current_user.role.name not in ("Administrator", "Technician")
        and metadata.get("owner_id") != current_user.id
    ):
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this video."
        )

    output_filename = metadata.get("output_filename")

    if output_filename != f"{video_id}.mp4":
        raise HTTPException(
            status_code=404,
            detail="Invalid video result filename."
        )

    output_path = VIDEO_RESULTS_DIR / output_filename

    if not output_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video file not found."
        )

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=output_filename if download else None
    )




class DatasetCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: Optional[str] = Field(default=None, max_length=500)


class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    description: Optional[str] = Field(default=None, max_length=500)


class DatasetSplitGenerateRequest(BaseModel):
    train_percent: int = Field(default=70, ge=0, le=100)
    val_percent: int = Field(default=15, ge=0, le=100)
    test_percent: int = Field(default=15, ge=0, le=100)
    seed: int = Field(default=42, ge=0, le=4294967295)


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

    risk_map_path = RESULTS_DIR / f"{stem}_risk_map.png"
    safe_corridor_path = (
        RESULTS_DIR / f"{stem}_safe_corridor.png"
    )

    temp_paths = {}
    original_contents = {}

    try:
        # Load and validate the original class-ID mask.
        with Image.open(class_ids_path) as image:
            width, height = image.size

            if (
                width <= 0
                or height <= 0
                or width > MAX_IMAGE_SIDE
                or height > MAX_IMAGE_SIDE
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ValueError(
                    "Class-ID mask exceeds the allowed dimensions."
                )

            mask = np.asarray(
                image.convert("L"),
                dtype=np.uint8
            )

        # Load and validate the original confidence map.
        confidence_map = _load_confidence_map(confidence_path)

        if confidence_map.shape != mask.shape:
            raise ValueError(
                "Confidence and mask dimensions do not match."
            )

        # Calculate traversability without saving intermediate results.
        traversability = create_traversability_map(
            mask,
            confidence_map,
            obstacle_radius=request.obstacle_radius
        )

        colored = colorize_traversability_map(traversability)

        # Recalculate Risk Map to apply the current risk model.
        risk_analysis = analyze_risk(mask, confidence_map)
        risk_map = risk_analysis["risk_map"]

        if risk_map.shape != mask.shape:
            raise ValueError(
                "Generated Risk Map dimensions do not match the mask."
            )

        # Calculate Safe Corridor before writing any final files.
        safe_corridor_analysis = analyze_safe_corridor(
            traversability,
            risk_map
        )

        # Prepare all output files in the same directory.
        outputs = {
            output_path: lambda p: Image.fromarray(colored).save(p),
            risk_map_path: lambda p: Image.fromarray(
                risk_map.astype(np.uint8),
                mode="L"
            ).save(p),
            safe_corridor_path: lambda p: Image.fromarray(
                safe_corridor_analysis["visualization"]
            ).save(p),
        }

        # Preserve existing files so they can be restored if replacement fails.
        for final_path in outputs:
            original_contents[final_path] = (
                final_path.read_bytes()
                if final_path.is_file()
                else None
            )

        # Write all new files to temporary paths first.
        for final_path, save_function in outputs.items():
            temp_path = final_path.with_name(
                f".{final_path.stem}.{uuid.uuid4().hex}.tmp"
                f"{final_path.suffix}"
            )

            temp_paths[final_path] = temp_path
            save_function(temp_path)

        # Replace final files only after all calculations and writes succeed.
        replaced_paths = []

        try:
            for final_path, temp_path in temp_paths.items():
                temp_path.replace(final_path)
                replaced_paths.append(final_path)

        except Exception:
            # Restore the original state if any replacement fails.
            for final_path in reversed(replaced_paths):
                original = original_contents[final_path]

                try:
                    if original is None:
                        if final_path.exists():
                            final_path.unlink()
                    else:
                        rollback_path = final_path.with_name(
                            f".{final_path.stem}."
                            f"{uuid.uuid4().hex}.rollback"
                        )

                        rollback_path.write_bytes(original)
                        rollback_path.replace(final_path)

                except OSError:
                    pass

            raise

    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not recalculate traversability: {error}"
        )

    finally:
        # Remove temporary files regardless of success or failure.
        for temp_path in temp_paths.values():
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

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

        if name == "Uploaded images":
            raise HTTPException(
                status_code=422,
                detail="This dataset name is reserved."
            )

        description = request.description
        if description is not None:
            description = description.strip() or None

        dataset = Dataset(
            name=name,
            description=description,
            owner_id=current_user.id,
            is_system=False,
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "is_system": dataset.is_system,
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

        ensure_dataset_access(dataset, current_user)

        if dataset.is_system:
            raise HTTPException(
                status_code=403,
                detail="The system dataset cannot be modified.",
            )

        updates = request.dict(exclude_unset=True)

        if "name" in updates:
            raw_name = updates["name"]
            if raw_name is None:
                raise HTTPException(
                    status_code=422,
                    detail="Dataset name cannot be null.",
                )
            name = raw_name.strip()
            if not name:
                raise HTTPException(
                    status_code=422,
                    detail="Dataset name cannot be empty.",
                )
            if name == "Uploaded images":
                raise HTTPException(
                    status_code=422,
                    detail="This dataset name is reserved.",
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
            "is_system": dataset.is_system,
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

        if dataset.is_system:
            raise HTTPException(
                status_code=403,
                detail="The system dataset cannot be deleted."
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

        images = db.query(DBImage).filter(
            DBImage.dataset_id == dataset_id
        ).all()

        image_ids = [image.id for image in images]

        for image in images:
            add_safe_file(
                image.file_path,
                uploads_root,
                relative_to_base=True
            )
            add_safe_file(
                image.mask_path,
                results_root,
                relative_to_base=True
            )
            add_safe_file(
                image.ground_truth_path,
                ground_truth_root,
                relative_to_base=True
            )

            upload_id = Path(image.file_path or "").stem

            if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
                continue

            metadata_candidates = RESULTS_DIR.glob(
                "*_" + upload_id[:8] + "_results.json"
            )

            for metadata_path in metadata_candidates:
                try:
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue

                if not isinstance(metadata, dict):
                    continue

                if metadata.get("upload_id") != upload_id:
                    continue

                add_safe_file(str(metadata_path), results_root)

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
                        "depth",
                        "depth_raw",
                        "uncertainty",
                        "uncertainty_raw",
                    ):
                        add_safe_file(
                            artifacts.get(key),
                            results_root
                        )

                for key in (
                    "preview_file",
                    "traversability_file",
                    "class_ids_file",
                    "risk_map_file",
                    "safe_corridor_file",
                    "confidence_file",
                    "depth_file",
                    "depth_raw_file",
                    "uncertainty_file",
                    "uncertainty_raw_file",
                ):
                    add_safe_file(
                        metadata.get(key),
                        results_root
                    )

        predictions = []

        if image_ids:
            predictions = db.query(Prediction).filter(
                Prediction.image_id.in_(image_ids)
            ).all()

        prediction_ids = [prediction.id for prediction in predictions]

        for prediction in predictions:
            add_safe_file(
                prediction.result_path,
                results_root,
                relative_to_base=True
            )

        if prediction_ids:
            db.query(DetectedClass).filter(
                DetectedClass.prediction_id.in_(prediction_ids)
            ).delete(synchronize_session=False)

            db.query(Prediction).filter(
                Prediction.id.in_(prediction_ids)
            ).delete(synchronize_session=False)

        if image_ids:
            db.query(DBImage).filter(
                DBImage.id.in_(image_ids)
            ).delete(synchronize_session=False)

        db.query(DatasetSplit).filter(
            DatasetSplit.dataset_id == dataset_id
        ).delete(synchronize_session=False)

        db.delete(dataset)
        db.commit()

        deleted_images = len(images)
        deleted_predictions = len(predictions)

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

    remaining_db = SessionLocal()

    try:
        referenced_paths = set()

        def add_reference(raw_path):
            if not isinstance(raw_path, str) or not raw_path:
                return

            candidate = Path(raw_path)

            if not candidate.is_absolute():
                candidate = BASE_DIR / candidate

            try:
                referenced_paths.add(candidate.resolve())
            except (OSError, RuntimeError):
                pass

        remaining_images = remaining_db.query(
            DBImage.file_path,
            DBImage.mask_path,
            DBImage.ground_truth_path,
        ).all()

        for image_paths in remaining_images:
            for raw_path in image_paths:
                add_reference(raw_path)

        remaining_predictions = remaining_db.query(
            Prediction.result_path
        ).all()

        for (raw_path,) in remaining_predictions:
            add_reference(raw_path)

        # Preserve artifacts referenced by metadata of uploads still in the database.
        remaining_upload_ids = set()
        for image_paths in remaining_images:
            original_path = image_paths[0]
            if isinstance(original_path, str) and original_path:
                upload_id = Path(original_path).stem
                if re.fullmatch(r"[0-9a-f]{32}", upload_id):
                    remaining_upload_ids.add(upload_id)

        def add_metadata_reference(raw_path, expected_root):
            if not isinstance(raw_path, str) or not raw_path:
                return

            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = expected_root / candidate

            try:
                referenced_paths.add(candidate.resolve())
            except (OSError, RuntimeError):
                pass

        for metadata_path in RESULTS_DIR.glob("*_results.json"):
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(metadata, dict):
                continue

            if metadata.get("upload_id") not in remaining_upload_ids:
                continue

            add_reference(str(metadata_path))

            artifacts = metadata.get("artifacts", {})
            if isinstance(artifacts, dict):
                add_metadata_reference(
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
                    "depth",
                    "depth_raw",
                    "uncertainty",
                    "uncertainty_raw",
                ):
                    add_metadata_reference(
                        artifacts.get(key),
                        results_root
                    )

            for key in (
                "preview_file",
                "traversability_file",
                "class_ids_file",
                "risk_map_file",
                "safe_corridor_file",
                "confidence_file",
                "depth_file",
                "depth_raw_file",
                "uncertainty_file",
                "uncertainty_raw_file",
            ):
                add_metadata_reference(
                    metadata.get(key),
                    results_root
                )

        files_to_delete = {
            file_path
            for file_path in files_to_delete
            if file_path not in referenced_paths
        }

    finally:
        remaining_db.close()

    deleted_files = 0
    cleanup_failures = 0

    for file_path in files_to_delete:
        try:
            file_path.unlink(missing_ok=True)
            deleted_files += 1
        except OSError:
            cleanup_failures += 1

    return {
        "message": "Dataset and related records deleted successfully.",
        "dataset_id": dataset_id,
        "deleted_images": deleted_images,
        "deleted_predictions": deleted_predictions,
        "deleted_files": deleted_files,
        "cleanup_failures": cleanup_failures,
    }


@app.get("/datasets/{dataset_id}/splits")
def get_dataset_splits(
    dataset_id: int,
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    ),
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

        ensure_dataset_access(
            dataset,
            current_user,
            allow_system_read=True
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
    request: DatasetSplitGenerateRequest,
    current_user: User = Depends(require_roles("Technician", "Administrator")),
):
    """Generate reproducible proportional Train / Val / Test splits."""
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        ensure_dataset_access(dataset, current_user)

        if dataset.is_system:
            raise HTTPException(
                status_code=403,
                detail="The system dataset cannot have managed splits."
            )

        percentages = [
            request.train_percent,
            request.val_percent,
            request.test_percent,
        ]

        if sum(percentages) != 100:
            raise HTTPException(
                status_code=422,
                detail="Train, Validation and Test percentages must sum to 100."
            )

        names = ("Train", "Val", "Test")

        existing = {
            split.name: split
            for split in db.query(DatasetSplit).filter(
                DatasetSplit.dataset_id == dataset_id
            ).all()
        }

        for name in names:
            if name not in existing:
                split = DatasetSplit(dataset_id=dataset_id, name=name)
                db.add(split)
                existing[name] = split

        db.flush()

        images = db.query(DBImage).filter(
            DBImage.dataset_id == dataset_id
        ).order_by(DBImage.id).all()

        random.Random(request.seed).shuffle(images)

        total = len(images)
        raw = [total * value / 100 for value in percentages]
        counts = [int(value) for value in raw]

        remainder = total - sum(counts)
        order = sorted(
            range(3),
            key=lambda i: (-(raw[i] - counts[i]), i)
        )

        for i in order[:remainder]:
            counts[i] += 1

        cursor = 0
        for name, count in zip(names, counts):
            for image in images[cursor:cursor + count]:
                image.split_id = existing[name].id
            cursor += count

        db.commit()

        return {
            "dataset_id": dataset_id,
            "seed": request.seed,
            "total_images": total,
            "splits": [
                {
                    "id": existing[name].id,
                    "name": name,
                    "image_count": counts[i],
                    "percentage": percentages[i],
                }
                for i, name in enumerate(names)
            ],
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

        ensure_dataset_access(dataset, current_user)

        if dataset.is_system:
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

                if not isinstance(metadata, dict):
                    continue

                if metadata.get("upload_id") != upload_id:
                    continue

                # Metadata itself.
                add_safe_file(str(metadata_path), results_root)

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
                        "depth",
                        "depth_raw",
                        "uncertainty",
                        "uncertainty_raw",
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
                    "depth_file",
                    "depth_raw_file",
                    "uncertainty_file",
                    "uncertainty_raw_file",
                ):
                    add_safe_file(
                        metadata.get(key),
                        results_root
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

    # Keep files that are still referenced by other database records.
    remaining_db = SessionLocal()
    try:
        referenced_paths = set()

        def add_reference(raw_path):
            if not isinstance(raw_path, str) or not raw_path:
                return

            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = BASE_DIR / candidate

            try:
                referenced_paths.add(candidate.resolve())
            except (OSError, RuntimeError):
                pass

        remaining_images = remaining_db.query(
            DBImage.file_path,
            DBImage.mask_path,
            DBImage.ground_truth_path,
        ).all()

        for image_paths in remaining_images:
            for raw_path in image_paths:
                add_reference(raw_path)

        remaining_predictions = remaining_db.query(
            Prediction.result_path
        ).all()

        for (raw_path,) in remaining_predictions:
            add_reference(raw_path)

        # Preserve artifacts referenced by metadata of uploads still in the database.
        remaining_upload_ids = set()
        for image_paths in remaining_images:
            original_path = image_paths[0]
            if isinstance(original_path, str) and original_path:
                upload_id = Path(original_path).stem
                if re.fullmatch(r"[0-9a-f]{32}", upload_id):
                    remaining_upload_ids.add(upload_id)

        def add_metadata_reference(raw_path, expected_root):
            if not isinstance(raw_path, str) or not raw_path:
                return

            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = expected_root / candidate

            try:
                referenced_paths.add(candidate.resolve())
            except (OSError, RuntimeError):
                pass

        for metadata_path in RESULTS_DIR.glob("*_results.json"):
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(metadata, dict):
                continue

            if metadata.get("upload_id") not in remaining_upload_ids:
                continue

            add_reference(str(metadata_path))

            artifacts = metadata.get("artifacts", {})
            if isinstance(artifacts, dict):
                add_metadata_reference(
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
                    "depth",
                    "depth_raw",
                    "uncertainty",
                    "uncertainty_raw",
                ):
                    add_metadata_reference(
                        artifacts.get(key),
                        results_root
                    )

            for key in (
                "preview_file",
                "traversability_file",
                "class_ids_file",
                "risk_map_file",
                "safe_corridor_file",
                "confidence_file",
                "depth_file",
                "depth_raw_file",
                "uncertainty_file",
                "uncertainty_raw_file",
            ):
                add_metadata_reference(
                    metadata.get(key),
                    results_root
                )

        files_to_delete = {
            file_path
            for file_path in files_to_delete
            if file_path not in referenced_paths
        }
    finally:
        remaining_db.close()

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

        if target_dataset.is_system:
            raise HTTPException(
                status_code=403,
                detail="Images cannot be moved into the system dataset."
            )

        source_dataset_id = image.dataset_id

        source_dataset = db.query(Dataset).filter(
            Dataset.id == source_dataset_id
        ).first()

        if source_dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Source dataset not found."
            )

        ensure_dataset_access(
            source_dataset,
            current_user,
            allow_system_read=True
        )
        ensure_dataset_access(target_dataset, current_user)

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
    current_user: User = Depends(require_roles("Technician", "Administrator"))
):
    db = SessionLocal()
    try:
        role_name = current_user.role.name if current_user.role else None

        if role_name == "Administrator":
            datasets = db.query(Dataset).order_by(
                Dataset.id.desc()
            ).all()
        elif role_name == "Technician":
            datasets = db.query(Dataset).filter(
                (Dataset.owner_id == current_user.id)
                | (Dataset.is_system.is_(True))
            ).order_by(Dataset.id.desc()).all()
        else:
            return []

        return [
            {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "is_system": dataset.is_system,
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

        ensure_dataset_access(
            dataset,
            current_user,
            allow_system_read=True
        )

        if split_id is not None:
            split = db.query(DatasetSplit).filter(
                DatasetSplit.id == split_id,
                DatasetSplit.dataset_id == dataset_id,
            ).first()

            if split is None:
                raise HTTPException(
                    status_code=404,
                    detail="Split not found in this dataset."
                )

        # Automatically assign images that do not yet belong to a split.
        # Existing assignments are preserved.
        if not dataset.is_system:
            unassigned_images = db.query(DBImage).filter(
                DBImage.dataset_id == dataset_id,
                DBImage.split_id.is_(None),
            ).order_by(DBImage.id).all()

            if unassigned_images:
                split_names = ("Train", "Val", "Test")
                split_percentages = (70, 15, 15)

                existing_splits = {
                    split.name: split
                    for split in db.query(DatasetSplit).filter(
                        DatasetSplit.dataset_id == dataset_id
                    ).all()
                }

                for name in split_names:
                    if name not in existing_splits:
                        split = DatasetSplit(
                            dataset_id=dataset_id,
                            name=name
                        )
                        db.add(split)
                        existing_splits[name] = split

                db.flush()

                random.Random(42).shuffle(unassigned_images)

                total = len(unassigned_images)
                raw_counts = [
                    total * percentage / 100
                    for percentage in split_percentages
                ]
                counts = [int(value) for value in raw_counts]

                remainder = total - sum(counts)
                order = sorted(
                    range(3),
                    key=lambda i: (-(raw_counts[i] - counts[i]), i)
                )

                for i in order[:remainder]:
                    counts[i] += 1

                cursor = 0
                for name, count in zip(split_names, counts):
                    for image in unassigned_images[cursor:cursor + count]:
                        image.split_id = existing_splits[name].id
                    cursor += count

                db.commit()

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




@app.get("/datasets/{dataset_id}/images/{image_id}/result")
def get_dataset_image_result(
    dataset_id: int,
    image_id: int,
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    ),
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

        ensure_dataset_access(
            dataset,
            current_user,
            allow_system_read=True
        )

        image = db.query(DBImage).filter(
            DBImage.id == image_id,
            DBImage.dataset_id == dataset_id,
        ).first()

        if image is None:
            raise HTTPException(
                status_code=404,
                detail="Image not found."
            )

        prediction = (
            db.query(Prediction)
            .filter(Prediction.image_id == image_id)
            .order_by(
                Prediction.created_at.desc(),
                Prediction.id.desc()
            )
            .first()
        )

        if prediction is None or not image.has_mask:
            raise HTTPException(
                status_code=404,
                detail="Segmentation result not found."
            )

        upload_match = re.fullmatch(
            r"data/uploads/([0-9a-f]{32})\.(?:jpg|jpeg|png|img)",
            image.file_path or "",
        )

        if not upload_match:
            raise HTTPException(
                status_code=404,
                detail="Original image reference is invalid."
            )

        upload_id = upload_match.group(1)

        original_path = (BASE_DIR / image.file_path).resolve()

        try:
            original_path.relative_to(UPLOADS_DIR.resolve())
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail="Original image reference is invalid."
            )

        if not original_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Original image file not found."
            )

        metadata_candidates = list(
            RESULTS_DIR.glob(f"*_{upload_id[:8]}_results.json")
        )

        if len(metadata_candidates) > 1:
            raise HTTPException(
                status_code=404,
                detail="Result metadata is ambiguous."
            )

        # Existing metadata: validate it against the database.
        if len(metadata_candidates) == 1:
            metadata_path = metadata_candidates[0]
            metadata_stem = metadata_path.name[
                :-len("_results.json")
            ]

            if _get_upload_id(metadata_stem) != upload_id:
                raise HTTPException(
                    status_code=404,
                    detail="Result metadata does not match the image."
                )

            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

            if (
                metadata.get("image_id") != image.id
                or metadata.get("prediction_id") != prediction.id
                or (
                    metadata.get("dataset_id") is not None
                    and metadata.get("dataset_id") != dataset.id
                )
            ):
                raise HTTPException(
                    status_code=404,
                    detail="Result metadata does not match the database record."
                )

            return metadata

        # Legacy record without JSON: reconstruct only when the
        # original, mask and overlay have the same unique upload UUID.
        mask_match = re.fullmatch(
            rf"results/(.+)_({upload_id[:8]})_mask\.png",
            image.mask_path or "",
        )

        overlay_match = re.fullmatch(
            rf"results/(.+)_({upload_id[:8]})_overlay\.png",
            prediction.result_path or "",
        )

        if (
            not mask_match
            or not overlay_match
            or mask_match.group(1) != overlay_match.group(1)
            or mask_match.group(2) != overlay_match.group(2)
        ):
            raise HTTPException(
                status_code=404,
                detail="Legacy result cannot be matched safely."
            )

        result_stem = mask_match.group(1) + "_" + upload_id[:8]

        def safe_result_file(relative_path):
            if not relative_path:
                return None

            candidate = (BASE_DIR / relative_path).resolve()

            try:
                candidate.relative_to(RESULTS_DIR.resolve())
            except ValueError:
                return None

            return candidate if candidate.is_file() else None

        mask_path = safe_result_file(image.mask_path)
        overlay_path = safe_result_file(prediction.result_path)

        if mask_path is None or overlay_path is None:
            raise HTTPException(
                status_code=404,
                detail="Segmentation mask or overlay file not found."
            )

        with Image.open(original_path) as original_image:
            original_width, original_height = original_image.size

        with Image.open(mask_path) as mask_image:
            mask_width, mask_height = mask_image.size

        detected_classes = (
            db.query(DetectedClass)
            .filter(
                DetectedClass.prediction_id == prediction.id
            )
            .order_by(DetectedClass.class_id.asc())
            .all()
        )

        metadata = {
            "filename": image.filename,
            "image_id": image.id,
            "dataset_id": dataset.id,
            "prediction_id": prediction.id,
            "upload_id": upload_id,
            "model": prediction.model_name,
            "original_width": original_width,
            "original_height": original_height,
            "inference_width": mask_width,
            "inference_height": mask_height,
            "mask_shape": [mask_height, mask_width],
            "mask_file": str(mask_path),
            "overlay_file": str(overlay_path),
            "inference_time_ms": prediction.inference_time_ms,
            "mean_confidence": prediction.mean_confidence,
            "low_confidence_percentage": (
                prediction.low_confidence_percentage
            ),
            "detected_classes": [
                {
                    "id": item.class_id,
                    "name": item.class_name,
                    "pixels": item.pixel_count,
                    "percentage": item.percentage,
                }
                for item in detected_classes
            ],
        }

        # Attach optional artifacts only when the exact UUID-based
        # file exists. Missing historical artifacts remain optional.
        optional_artifacts = {
            "preview": ("preview_file", "preview_url", "_preview.jpg"),
            "traversability": (
                "traversability_file",
                "traversability_url",
                "_traversability.png",
            ),
            "confidence": (
                None,
                "confidence_url",
                "_confidence.png",
            ),
            "risk_map": (
                "risk_map_file",
                "risk_map_url",
                "_risk_map.png",
            ),
            "safe_corridor": (
                "safe_corridor_file",
                "safe_corridor_url",
                "_safe_corridor.png",
            ),
            "depth": ("depth_file", "depth_url", "_depth.png"),
            "uncertainty": (
                "uncertainty_file",
                "uncertainty_url",
                "_uncertainty.png",
            ),
        }

        for artifact_name, (file_key, url_key, suffix) in optional_artifacts.items():
            candidate = safe_result_file(
                f"results/{result_stem}{suffix}"
            )

            if candidate is None:
                continue

            if artifact_name == "confidence":
                metadata["confidence_file"] = (
                    f"{result_stem}_confidence.npy"
                    if safe_result_file(
                        f"results/{result_stem}_confidence.npy"
                    )
                    else None
                )
                if metadata["confidence_file"] is None:
                    metadata.pop("confidence_file", None)
            elif file_key:
                metadata[file_key] = str(candidate)

            metadata[url_key] = f"/results/{candidate.name}"

        class_ids_path = safe_result_file(
            f"results/{result_stem}_class_ids.png"
        )

        if class_ids_path:
            metadata["class_ids_file"] = str(class_ids_path)
            metadata["class_ids_filename"] = class_ids_path.name

        return metadata

    except (OSError, json.JSONDecodeError, UnidentifiedImageError):
        raise HTTPException(
            status_code=500,
            detail="Could not read segmentation result metadata."
        )
    finally:
        db.close()


@app.get("/datasets/{dataset_id}/images/{image_id}/original")
def get_dataset_image_original(
    dataset_id: int,
    image_id: int,
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    ),
):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        ensure_dataset_access(
            dataset,
            current_user,
            allow_system_read=True
        )

        image = db.query(DBImage).filter(
            DBImage.id == image_id,
            DBImage.dataset_id == dataset_id,
        ).first()

        if image is None:
            raise HTTPException(status_code=404, detail="Image not found.")

        upload_match = re.fullmatch(
            r"data/uploads/([0-9a-f]{32})\.(jpg|jpeg|png|img)",
            image.file_path or "",
        )

        if not upload_match:
            raise HTTPException(
                status_code=404,
                detail="Original image reference is invalid."
            )

        upload_id, extension = upload_match.groups()

        check_upload_access(upload_id, current_user, db)

        original_path = UPLOADS_DIR / f"{upload_id}.{extension}"

        if not original_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Original image file not found."
            )

        return FileResponse(
            original_path,
            filename=Path(image.filename).name,
        )
    finally:
        db.close()


@app.post("/datasets/import")
async def import_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(
        require_roles("Technician", "Administrator")
    ),
):
    """Import a dataset ZIP previously created by the export endpoint."""

    import tempfile
    from pathlib import PurePosixPath

    MAX_ARCHIVE_SIZE = 500 * 1024 * 1024
    MAX_TOTAL_SIZE = 2 * 1024 * 1024 * 1024
    MAX_FILE_SIZE = 50 * 1024 * 1024
    MAX_FILES = 10000
    ALLOWED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
    }

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a ZIP archive."
        )

    db = SessionLocal()
    created_paths = []

    try:
        # Store the uploaded archive in a spooled temporary file.
        with tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024,
            mode="w+b"
        ) as archive_buffer:

            uploaded_size = 0

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                uploaded_size += len(chunk)

                if uploaded_size > MAX_ARCHIVE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="ZIP archive exceeds the 500 MB limit."
                    )

                archive_buffer.write(chunk)

            if uploaded_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded archive is empty."
                )

            archive_buffer.seek(0)

            try:
                archive = zipfile.ZipFile(archive_buffer, "r")
            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid ZIP archive."
                )

            with archive:
                entries = archive.infolist()

                if len(entries) > MAX_FILES + 20:
                    raise HTTPException(
                        status_code=413,
                        detail="Archive contains too many entries."
                    )

                # Validate every archive path before reading any file.
                entry_map = {}
                root_folder = None
                total_uncompressed = 0

                for info in entries:
                    raw_name = info.filename

                    if (
                        not raw_name
                        or "\\" in raw_name
                        or "\x00" in raw_name
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail="Archive contains an invalid path."
                        )

                    path = PurePosixPath(raw_name)

                    if (
                        path.is_absolute()
                        or any(part in ("..", ".") for part in path.parts)
                        or not path.parts
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail="Archive contains an unsafe path."
                        )

                    if len(path.parts) < 2:
                        raise HTTPException(
                            status_code=400,
                            detail="Unexpected archive structure."
                        )

                    if root_folder is None:
                        root_folder = path.parts[0]
                    elif path.parts[0] != root_folder:
                        raise HTTPException(
                            status_code=400,
                            detail="Archive must contain one dataset root folder."
                        )

                    if info.filename in entry_map:
                        raise HTTPException(
                            status_code=400,
                            detail="Archive contains duplicate paths."
                        )

                    # Reject symbolic links.
                    unix_mode = info.external_attr >> 16

                    if (unix_mode & 0o170000) == 0o120000:
                        raise HTTPException(
                            status_code=400,
                            detail="Symbolic links are not allowed."
                        )

                    if info.flag_bits & 0x1:
                        raise HTTPException(
                            status_code=400,
                            detail="Encrypted ZIP entries are not supported."
                        )

                    if info.is_dir():
                        continue

                    if info.file_size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="An archive file exceeds the 50 MB limit."
                        )

                    total_uncompressed += info.file_size

                    if total_uncompressed > MAX_TOTAL_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="Uncompressed archive exceeds 2 GB."
                        )

                    if (
                        info.file_size > 1024 * 1024
                        and info.compress_size > 0
                        and info.file_size / info.compress_size > 200
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail="Suspicious compression ratio detected."
                        )

                    entry_map[info.filename] = info

                if not root_folder:
                    raise HTTPException(
                        status_code=400,
                        detail="Empty archive."
                    )

                metadata_path = (
                    f"{root_folder}/metadata/dataset.json"
                )
                csv_path = (
                    f"{root_folder}/metadata/images.csv"
                )

                if metadata_path not in entry_map or csv_path not in entry_map:
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset metadata files are missing."
                    )

                try:
                    metadata = json.loads(
                        archive.read(metadata_path).decode("utf-8")
                    )

                    if not isinstance(metadata, dict):
                        raise ValueError("Metadata must be an object.")

                    csv_text = archive.read(csv_path).decode("utf-8-sig")
                    reader = csv.DictReader(StringIO(csv_text))

                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset metadata is invalid."
                    )

                required_columns = {
                    "filename",
                    "archive_image_path",
                    "width",
                    "height",
                    "file_size",
                    "has_ground_truth",
                    "ground_truth_format",
                    "archive_ground_truth_path",
                    "split",
                    "created_at",
                }

                if not reader.fieldnames or not required_columns.issubset(
                    set(reader.fieldnames)
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset CSV has an invalid structure."
                    )

                dataset_name = metadata.get("name")
                description = metadata.get("description")

                if (
                    not isinstance(dataset_name, str)
                    or not dataset_name.strip()
                    or len(dataset_name.strip()) > 150
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid dataset name in metadata."
                    )

                if description is not None:
                    if not isinstance(description, str) or len(description) > 500:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid dataset description."
                        )

                rows = list(reader)

                if len(rows) > MAX_FILES:
                    raise HTTPException(
                        status_code=413,
                        detail="Dataset contains too many images."
                    )

                if metadata.get("image_count") != len(rows):
                    raise HTTPException(
                        status_code=400,
                        detail="Image count does not match dataset metadata."
                    )

                if not rows:
                    raise HTTPException(
                        status_code=400,
                        detail="Cannot import an empty dataset."
                    )

                referenced_files = set()
                validated_rows = []
                ground_truth_count = 0
                split_counts = {
                    "train": 0,
                    "val": 0,
                    "test": 0,
                    "unassigned": 0,
                }

                def validate_image_file(archive_path, expected_split, is_gt=False):
                    if not isinstance(archive_path, str) or not archive_path:
                        raise HTTPException(
                            status_code=400,
                            detail="An image archive path is missing."
                        )

                    path = PurePosixPath(archive_path)

                    if (
                        path.is_absolute()
                        or "\\" in archive_path
                        or any(part in ("..", ".") for part in path.parts)
                        or len(path.parts) != 4
                        or path.parts[0] != root_folder
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid image path in CSV."
                        )

                    expected_folder = "ground_truth" if is_gt else "images"

                    if (
                        path.parts[1] != expected_split
                        or path.parts[2] != expected_folder
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail="Image path does not match its split."
                        )

                    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                        raise HTTPException(
                            status_code=400,
                            detail="Unsupported image file extension."
                        )

                    normalized_path = path.as_posix()

                    if normalized_path not in entry_map:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Referenced archive file is missing: {normalized_path}"
                        )

                    if normalized_path in referenced_files:
                        raise HTTPException(
                            status_code=400,
                            detail="An archive file is referenced more than once."
                        )

                    referenced_files.add(normalized_path)

                    try:
                        content = archive.read(normalized_path)

                        with Image.open(BytesIO(content)) as img:
                            img.verify()

                        with Image.open(BytesIO(content)) as img:
                            width, height = img.size

                            if (
                                width <= 0
                                or height <= 0
                                or width > MAX_IMAGE_SIDE
                                or height > MAX_IMAGE_SIDE
                                or width * height > MAX_IMAGE_PIXELS
                            ):
                                raise ValueError("Invalid image dimensions.")

                    except (
                        OSError,
                        ValueError,
                        UnidentifiedImageError,
                        zipfile.BadZipFile,
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid image file: {normalized_path}"
                        )

                    return width, height

                for row in rows:
                    filename = (row.get("filename") or "").strip()

                    if not filename or len(filename) > 255:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid image filename in CSV."
                        )

                    split_name = (row.get("split") or "").strip().lower()

                    if split_name not in split_counts:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported split name: {split_name}"
                        )

                    split_counts[split_name] += 1

                    original_archive_path = row.get("archive_image_path")
                    width, height = validate_image_file(
                        original_archive_path,
                        split_name,
                    )

                    try:
                        expected_width = int(row["width"])
                        expected_height = int(row["height"])
                    except (TypeError, ValueError):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid image dimensions in CSV."
                        )

                    if (
                        expected_width <= 0
                        or expected_height <= 0
                        or expected_width != width
                        or expected_height != height
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image dimensions do not match CSV: {filename}"
                        )

                    try:
                        expected_file_size = int(row["file_size"])
                    except (TypeError, ValueError):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid image file size in CSV."
                        )

                    actual_file_size = entry_map[original_archive_path].file_size

                    if (
                        expected_file_size <= 0
                        or expected_file_size != actual_file_size
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image file size does not match CSV: {filename}"
                        )

                    gt_value = (row.get("has_ground_truth") or "").strip().lower()

                    if gt_value not in ("true", "false", "1", "0"):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid Ground Truth flag in CSV."
                        )

                    has_gt = gt_value in ("true", "1")
                    gt_content = None
                    gt_extension = None

                    if has_gt:
                        gt_archive_path = row.get(
                            "archive_ground_truth_path"
                        )

                        gt_width, gt_height = validate_image_file(
                            gt_archive_path,
                            split_name,
                            is_gt=True,
                        )

                        if (gt_width, gt_height) != (width, height):
                            raise HTTPException(
                                status_code=400,
                                detail=f"Ground Truth dimensions do not match image: {filename}"
                            )

                        gt_extension = PurePosixPath(
                            gt_archive_path
                        ).suffix.lower()

                        ground_truth_count += 1

                    elif (row.get("archive_ground_truth_path") or "").strip():
                        raise HTTPException(
                            status_code=400,
                            detail="Ground Truth path exists but its flag is false."
                        )

                    ground_truth_format = (
                        row.get("ground_truth_format") or ""
                    ).strip() or None

                    if (
                        ground_truth_format is not None
                        and len(ground_truth_format) > 30
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail="Ground Truth format is too long."
                        )

                    created_at = None
                    created_at_text = (row.get("created_at") or "").strip()

                    if created_at_text:
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_text.replace("Z", "+00:00")
                            )

                            if created_at.tzinfo is not None:
                                created_at = created_at.replace(tzinfo=None)

                        except ValueError:
                            raise HTTPException(
                                status_code=400,
                                detail="Invalid image creation date in CSV."
                            )

                    validated_rows.append({
                        "filename": filename,
                        "archive_image_path": original_archive_path,
                        "width": width,
                        "height": height,
                        "split": split_name,
                        "has_gt": has_gt,
                        "archive_gt_path": (
                            row.get("archive_ground_truth_path")
                            if has_gt else None
                        ),
                        "gt_extension": gt_extension,
                        "gt_format": ground_truth_format,
                        "created_at": created_at,
                    })

                expected_gt_count = metadata.get("ground_truth_count")

                if expected_gt_count != ground_truth_count:
                    raise HTTPException(
                        status_code=400,
                        detail="Ground Truth count does not match metadata."
                    )

                expected_split_counts = metadata.get("split_counts")

                if (
                    not isinstance(expected_split_counts, dict)
                    or set(expected_split_counts.keys()) != set(split_counts.keys())
                    or any(
                        type(expected_split_counts[name]) is not int
                        or expected_split_counts[name] < 0
                        or expected_split_counts[name] != split_counts[name]
                        for name in split_counts
                    )
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Split counts do not match dataset metadata."
                    )

                # Reject unreferenced files other than documented metadata/readme.
                allowed_unreferenced = {
                    metadata_path,
                    csv_path,
                    f"{root_folder}/README.txt",
                }

                unreferenced = (
                    set(entry_map.keys())
                    - referenced_files
                    - allowed_unreferenced
                )

                if unreferenced:
                    raise HTTPException(
                        status_code=400,
                        detail="Archive contains unexpected files."
                    )

                # All archive contents have now been validated.
                dataset = Dataset(
                    name=dataset_name.strip(),
                    description=description.strip() if description else None,
                    owner_id=current_user.id,
                    is_system=False,
                )

                db.add(dataset)
                db.flush()

                split_records = {}

                for split_name, display_name in (
                    ("train", "Train"),
                    ("val", "Val"),
                    ("test", "Test"),
                ):
                    split_record = DatasetSplit(
                        dataset_id=dataset.id,
                        name=display_name,
                    )
                    db.add(split_record)
                    split_records[split_name] = split_record

                db.flush()

                UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

                imported_count = 0

                for item in validated_rows:
                    image_extension = Path(item["filename"]).suffix.lower()

                    if image_extension not in ALLOWED_EXTENSIONS:
                        raise HTTPException(
                            status_code=400,
                            detail="Unsupported original image extension."
                        )

                    image_id_prefix = uuid.uuid4().hex

                    original_path = (
                        UPLOADS_DIR / f"{image_id_prefix}{image_extension}"
                    )

                    original_content = archive.read(
                        item["archive_image_path"]
                    )
                    created_paths.append(original_path)
                    original_path.write_bytes(original_content)

                    gt_path = None

                    if item["has_gt"]:
                        gt_path = (
                            GROUND_TRUTH_DIR
                            / f"{image_id_prefix}_gt{item['gt_extension']}"
                        )

                        gt_content = archive.read(item["archive_gt_path"])
                        created_paths.append(gt_path)
                        gt_path.write_bytes(gt_content)

                    split_id = None

                    if item["split"] in split_records:
                        split_id = split_records[item["split"]].id

                    image_record = DBImage(
                        dataset_id=dataset.id,
                        split_id=split_id,
                        uploaded_by=current_user.id,
                        filename=item["filename"],
                        file_path=str(original_path.relative_to(BASE_DIR)),
                        width=item["width"],
                        height=item["height"],
                        file_size=len(original_content),
                        has_mask=False,
                        mask_path=None,
                        has_ground_truth=item["has_gt"],
                        ground_truth_path=(
                            str(gt_path.relative_to(BASE_DIR))
                            if gt_path else None
                        ),
                        ground_truth_format=item["gt_format"],
                        created_at=item["created_at"],
                    )

                    db.add(image_record)
                    imported_count += 1

                db.commit()

                return {
                    "message": "Dataset imported successfully.",
                    "dataset_id": dataset.id,
                    "name": dataset.name,
                    "image_count": imported_count,
                    "ground_truth_count": ground_truth_count,
                    "split_counts": split_counts,
                }

    except Exception:
        db.rollback()

        for created_path in reversed(created_paths):
            try:
                created_path.unlink(missing_ok=True)
            except OSError:
                pass

        raise

    finally:
        db.close()
        await file.close()


@app.get("/datasets/{dataset_id}/export")
def export_dataset(
    dataset_id: int,
    include_predictions: bool = False,
    current_user: User = Depends(get_current_user)
):
    db = SessionLocal()

    try:
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        ensure_dataset_access(
            dataset,
            current_user,
            allow_system_read=True
        )

        images = db.query(DBImage).filter(
            DBImage.dataset_id == dataset_id
        ).order_by(DBImage.id.asc()).all()

        def resolve_prediction_file(stored_path: str):
            if not stored_path:
                return None

            try:
                raw_path = str(stored_path).replace("\\", "/")

                if raw_path.startswith("/results/"):
                    candidate = RESULTS_DIR / raw_path[len("/results/"):]
                else:
                    candidate = Path(raw_path)

                    if not candidate.is_absolute():
                        if candidate.parts and candidate.parts[0] == "results":
                            candidate = BASE_DIR / candidate
                        else:
                            candidate = RESULTS_DIR / candidate

                candidate = candidate.resolve()
                candidate.relative_to(RESULTS_DIR.resolve())

                if candidate.is_file():
                    return candidate

            except (ValueError, OSError, TypeError):
                pass

            return None

        def resolve_dataset_file(stored_path: str, description: str) -> Path:
            if not stored_path:
                raise HTTPException(
                    status_code=409,
                    detail=f"Missing file path for {description}."
                )

            file_path = (BASE_DIR / stored_path).resolve()

            try:
                file_path.relative_to(BASE_DIR.resolve())
            except ValueError:
                raise HTTPException(
                    status_code=409,
                    detail=f"Invalid file path for {description}."
                )

            if not file_path.is_file():
                raise HTTPException(
                    status_code=409,
                    detail=f"File not found for {description}: {stored_path}"
                )

            return file_path

        safe_name = re.sub(
            r"[^A-Za-z0-9_-]+", "_", dataset.name
        ).strip("_") or f"dataset_{dataset.id}"

        root_folder = f"{safe_name}_{dataset.id}"

        split_names = {
            split.id: split.name
            for split in dataset.splits
        }

        def get_split_folder(split_id):
            split_name = split_names.get(split_id, "")

            normalized = str(split_name).strip().lower()

            if normalized == "train":
                return "train"
            if normalized in ("val", "validation"):
                return "val"
            if normalized == "test":
                return "test"

            return "unassigned"

        archive_buffer = BytesIO()
        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)

        csv_writer.writerow([
            "image_id",
            "filename",
            "archive_image_path",
            "width",
            "height",
            "file_size",
            "has_ground_truth",
            "ground_truth_format",
            "archive_ground_truth_path",
            "split",
            "uploaded_by",
            "created_at",
            "prediction_id",
            "prediction_model",
            "archive_overlay_path",
            "archive_mask_path",
            "archive_class_ids_path",
            "archive_traversability_path",
            "archive_confidence_path",
            "archive_confidence_raw_path",
            "archive_preview_path",
            "archive_depth_path",
            "archive_depth_raw_path",
            "archive_uncertainty_path",
            "archive_uncertainty_raw_path",
            "archive_risk_map_path",
            "archive_safe_corridor_path",
            "archive_prediction_metadata_path"
        ])

        split_counts = {
            "train": 0,
            "val": 0,
            "test": 0,
            "unassigned": 0
        }

        ground_truth_count = 0
        prediction_count = 0
        prediction_artifact_counts = {
            "overlay": 0,
            "mask": 0,
            "class_ids": 0,
            "traversability": 0,
            "confidence": 0,
            "confidence_raw": 0,
            "preview": 0,
            "depth": 0,
            "depth_raw": 0,
            "uncertainty": 0,
            "uncertainty_raw": 0,
            "risk_map": 0,
            "safe_corridor": 0
        }

        with zipfile.ZipFile(
            archive_buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:

            for image in images:
                split_folder = get_split_folder(image.split_id)
                split_counts[split_folder] += 1

                original_path = resolve_dataset_file(
                    image.file_path,
                    f"original image {image.id}"
                )

                original_name = Path(
                    str(image.filename).replace("\\", "/")
                ).name

                image_archive_path = (
                    f"{root_folder}/{split_folder}/images/"
                    f"{image.id}_{original_name}"
                )

                archive.write(original_path, image_archive_path)

                gt_archive_path = ""

                if image.has_ground_truth:
                    gt_path = resolve_dataset_file(
                        image.ground_truth_path,
                        f"Ground Truth mask for image {image.id}"
                    )

                    extension = gt_path.suffix or ".png"

                    gt_archive_path = (
                        f"{root_folder}/{split_folder}/ground_truth/"
                        f"{image.id}_mask{extension}"
                    )

                    archive.write(gt_path, gt_archive_path)
                    ground_truth_count += 1

                prediction_id = ""
                prediction_model = ""
                prediction_archive_paths = {
                    "overlay": "",
                    "mask": "",
                    "class_ids": "",
                    "traversability": "",
                    "confidence": "",
                    "confidence_raw": "",
                    "preview": "",
                    "depth": "",
                    "depth_raw": "",
                    "uncertainty": "",
                    "uncertainty_raw": "",
                    "risk_map": "",
                    "safe_corridor": "",
                    "metadata": ""
                }

                if include_predictions:
                    prediction = (
                        db.query(Prediction)
                        .filter(Prediction.image_id == image.id)
                        .order_by(
                            Prediction.created_at.desc(),
                            Prediction.id.desc()
                        )
                        .first()
                    )

                    if prediction is not None:
                        prediction_id = prediction.id
                        prediction_model = prediction.model_name or ""
                        prediction_count += 1

                        overlay_path = resolve_prediction_file(
                            prediction.result_path
                        )

                        result_metadata = {}

                        if overlay_path is not None:
                            metadata_path = (
                                RESULTS_DIR /
                                f"{overlay_path.stem[:-8] if overlay_path.stem.endswith('_overlay') else overlay_path.stem}_results.json"
                            )

                            try:
                                metadata_path = metadata_path.resolve()
                                metadata_path.relative_to(
                                    RESULTS_DIR.resolve()
                                )

                                if metadata_path.is_file():
                                    result_metadata = json.loads(
                                        metadata_path.read_text(
                                            encoding="utf-8"
                                        )
                                    )

                                    if not isinstance(result_metadata, dict):
                                        result_metadata = {}

                            except (OSError, ValueError, json.JSONDecodeError):
                                result_metadata = {}

                        artifact_sources = {
                            "overlay": prediction.result_path,
                            "mask": result_metadata.get("mask_file"),
                            "class_ids": result_metadata.get("class_ids_file"),
                            "traversability": (
                                result_metadata.get("traversability_file")
                                or result_metadata.get("traversability_url")
                            ),
                            "preview": (
                                result_metadata.get("preview_file")
                                or result_metadata.get("preview_url")
                            ),
                            "depth": (
                                result_metadata.get("depth_file")
                                or result_metadata.get("depth_url")
                            ),
                            "depth_raw": result_metadata.get("depth_raw_file"),
                            "uncertainty": (
                                result_metadata.get("uncertainty_file")
                                or result_metadata.get("uncertainty_url")
                            ),
                            "uncertainty_raw": result_metadata.get(
                                "uncertainty_raw_file"
                            ),
                            "risk_map": (
                                result_metadata.get("risk_map_file")
                                or result_metadata.get("risk_map_url")
                            ),
                            "safe_corridor": (
                                result_metadata.get("safe_corridor_file")
                                or result_metadata.get("safe_corridor_url")
                            )
                        }

                        confidence_url = result_metadata.get(
                            "confidence_url", ""
                        )

                        if isinstance(confidence_url, str) and confidence_url:
                            artifact_sources["confidence"] = confidence_url

                        confidence_raw_file = result_metadata.get(
                            "confidence_file"
                        )

                        if (
                            isinstance(confidence_raw_file, str)
                            and confidence_raw_file.endswith(".npy")
                        ):
                            artifact_sources["confidence_raw"] = (
                                confidence_raw_file
                            )

                        for artifact_name, stored_path in artifact_sources.items():
                            artifact_path = resolve_prediction_file(
                                stored_path
                            )

                            if artifact_path is None:
                                continue

                            archive_path = (
                                f"{root_folder}/{split_folder}/predictions/"
                                f"{artifact_name}/"
                                f"{image.id}_{artifact_path.name}"
                            )

                            archive.write(artifact_path, archive_path)

                            prediction_archive_paths[artifact_name] = (
                                archive_path
                            )
                            prediction_artifact_counts[artifact_name] += 1

                        prediction_metadata = {
                            "image_id": image.id,
                            "prediction_id": prediction.id,
                            "model_name": prediction.model_name,
                            "created_at": (
                                prediction.created_at.isoformat()
                                if prediction.created_at else None
                            ),
                            "inference_time_ms": prediction.inference_time_ms,
                            "mean_confidence": prediction.mean_confidence,
                            "low_confidence_percentage": (
                                prediction.low_confidence_percentage
                            ),
                            "detected_classes": [
                                {
                                    "class_id": item.class_id,
                                    "class_name": item.class_name,
                                    "pixel_count": item.pixel_count,
                                    "percentage": item.percentage
                                }
                                for item in prediction.detected_classes
                            ],
                            "artifacts": {
                                key: value
                                for key, value in prediction_archive_paths.items()
                                if value
                            }
                        }

                        prediction_metadata_path = (
                            f"{root_folder}/{split_folder}/predictions/"
                            f"metadata/{image.id}_prediction.json"
                        )

                        archive.writestr(
                            prediction_metadata_path,
                            json.dumps(
                                prediction_metadata,
                                ensure_ascii=False,
                                indent=2
                            )
                        )

                        prediction_archive_paths["metadata"] = (
                            prediction_metadata_path
                        )

                csv_writer.writerow([
                    image.id,
                    image.filename,
                    image_archive_path,
                    image.width,
                    image.height,
                    image.file_size,
                    image.has_ground_truth,
                    image.ground_truth_format,
                    gt_archive_path,
                    split_folder,
                    image.uploaded_by,
                    image.created_at.isoformat() if image.created_at else "",
                    prediction_id,
                    prediction_model,
                    prediction_archive_paths["overlay"],
                    prediction_archive_paths["mask"],
                    prediction_archive_paths["class_ids"],
                    prediction_archive_paths["traversability"],
                    prediction_archive_paths["confidence"],
                    prediction_archive_paths["confidence_raw"],
                    prediction_archive_paths["preview"],
                    prediction_archive_paths["depth"],
                    prediction_archive_paths["depth_raw"],
                    prediction_archive_paths["uncertainty"],
                    prediction_archive_paths["uncertainty_raw"],
                    prediction_archive_paths["risk_map"],
                    prediction_archive_paths["safe_corridor"],
                    prediction_archive_paths["metadata"]
                ])

            metadata = {
                "dataset_id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "is_system": dataset.is_system,
                "created_at": (
                    dataset.created_at.isoformat()
                    if dataset.created_at else None
                ),
                "exported_at": datetime.utcnow().isoformat() + "Z",
                "image_count": len(images),
                "ground_truth_count": ground_truth_count,
                "include_predictions": include_predictions,
                "prediction_count": prediction_count,
                "prediction_artifact_counts": prediction_artifact_counts,
                "split_counts": split_counts,
                "splits": [
                    {"id": split.id, "name": split.name}
                    for split in dataset.splits
                ]
            }

            archive.writestr(
                f"{root_folder}/metadata/dataset.json",
                json.dumps(metadata, ensure_ascii=False, indent=2)
            )

            archive.writestr(
                f"{root_folder}/metadata/images.csv",
                csv_buffer.getvalue()
            )

            prediction_artifact_summary = (
                "Prediction artifacts included:" + chr(10)
                + "- Overlay: " + str(prediction_artifact_counts["overlay"]) + chr(10)
                + "- Predicted masks: " + str(prediction_artifact_counts["mask"]) + chr(10)
                + "- Class IDs: " + str(prediction_artifact_counts["class_ids"]) + chr(10)
                + "- Traversability: " + str(prediction_artifact_counts["traversability"]) + chr(10)
                + "- Confidence maps: " + str(prediction_artifact_counts["confidence"]) + chr(10)
                + "- Confidence raw arrays: " + str(prediction_artifact_counts["confidence_raw"]) + chr(10)
                + "- Preview images: " + str(prediction_artifact_counts["preview"]) + chr(10)
                + "- Depth maps: " + str(prediction_artifact_counts["depth"]) + chr(10)
                + "- Depth raw arrays: " + str(prediction_artifact_counts["depth_raw"]) + chr(10)
                + "- Uncertainty maps: " + str(prediction_artifact_counts["uncertainty"]) + chr(10)
                + "- Uncertainty raw arrays: " + str(prediction_artifact_counts["uncertainty_raw"]) + chr(10)
                + "- Risk maps: " + str(prediction_artifact_counts["risk_map"]) + chr(10)
                + "- Safe corridors: " + str(prediction_artifact_counts["safe_corridor"])
                if include_predictions
                else "Model predictions are not included in this export."
            )

            readme = f"""Dataset: {dataset.name}

Contents:
- train/images/: training images
- train/ground_truth/: training Ground Truth masks
- val/images/: validation images
- val/ground_truth/: validation Ground Truth masks
- test/images/: test images
- test/ground_truth/: test Ground Truth masks
- unassigned/: images without an assigned split
- metadata/dataset.json: dataset metadata and split counts
- metadata/images.csv: image metadata and archive paths

Image counts:
- Train: {split_counts["train"]}
- Val: {split_counts["val"]}
- Test: {split_counts["test"]}
- Unassigned: {split_counts["unassigned"]}

Ground Truth masks included: {ground_truth_count}

Predictions included: {prediction_count}

{prediction_artifact_summary}

Ground Truth and Predicted Masks are stored separately.
Images without saved predictions are included without prediction artifacts.
The export does not modify the database or source files.
"""

            archive.writestr(
                f"{root_folder}/README.txt",
                readme
            )

        archive_buffer.seek(0)

        return StreamingResponse(
            archive_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{safe_name}_{dataset.id}.zip"'
                )
            }
        )

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

        ensure_dataset_access(dataset, current_user)

        if dataset.is_system:
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

GROUND_TRUTH_DIR = BASE_DIR / "data" / "ground_truth"
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
