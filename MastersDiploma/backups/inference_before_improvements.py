from io import BytesIO
from PIL import Image
from torchvision import transforms

IMG_SIZE = 512

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image_from_bytes(content: bytes):
    image = Image.open(BytesIO(content)).convert("RGB")
    original_size = image.size
    tensor = preprocess(image).unsqueeze(0)
    return image, tensor, original_size
