from io import BytesIO
from PIL import Image, ImageOps


def load_image_from_bytes(content: bytes):
    with Image.open(BytesIO(content)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    original_size = image.size
    return image, original_size
