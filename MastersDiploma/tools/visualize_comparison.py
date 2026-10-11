from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from backend.segmentation import colorize_mask

root = Path("results/model_comparison")
image_path = Path("data/images-3.jpg")
output_path = root / "visual_comparison.png"

image = Image.open(image_path).convert("RGB")
b0 = np.asarray(Image.open(root / "B0_mask.png"))
b2 = np.asarray(Image.open(root / "B2_mask.png"))

mask_b0 = colorize_mask(b0)
mask_b2 = colorize_mask(b2)

overlay_b0 = (
    0.55 * np.asarray(image.resize((b0.shape[1], b0.shape[0]))).astype(np.float32)
    + 0.45 * mask_b0.astype(np.float32)
).clip(0, 255).astype(np.uint8)

overlay_b2 = (
    0.55 * np.asarray(image.resize((b2.shape[1], b2.shape[0]))).astype(np.float32)
    + 0.45 * mask_b2.astype(np.float32)
).clip(0, 255).astype(np.uint8)

agreement = np.mean(b0 == b2) * 100
difference = (b0 != b2).astype(np.uint8) * 255

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
panels = [
    (np.asarray(image), "Original image"),
    (mask_b0, "SegFormer-B0 mask"),
    (mask_b2, "SegFormer-B2 mask"),
    (overlay_b0, "B0 overlay"),
    (overlay_b2, "B2 overlay"),
    (difference, f"Prediction differences | Agreement: {agreement:.2f}%"),
]

for ax, (img, title) in zip(axes.flat, panels):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(title)
    ax.axis("off")

fig.tight_layout()
fig.savefig(output_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved comparison: {output_path.resolve()}")
print(f"Pixel agreement: {agreement:.2f}%")
print(f"Changed pixels: {100 - agreement:.2f}%")
