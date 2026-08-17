from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from backend.inference import load_image_from_bytes
from backend.segmentation import predict_mask, colorize_mask
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image, tensor, original_size = load_image_from_bytes(content)
    mask = predict_mask(image)
    colored_mask = colorize_mask(mask)

    mask_image = Image.fromarray(colored_mask)
    buffer = BytesIO()
    mask_image.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return JSONResponse({
        "filename": file.filename,
        "size_bytes": len(content),
        "original_width": original_size[0],
        "original_height": original_size[1],
        "tensor_shape": list(tensor.shape),
        "mask_shape": list(mask.shape),
        "mask_preview_base64": mask_base64[:120] + "...",
        "message": "segmentation with visualization done"
    })