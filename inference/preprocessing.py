from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(input_dir):
    input_dir = Path(input_dir)

    image_paths = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def pad_to_multiple(img, multiple=32):
    h, w = img.shape

    new_h = h if h % multiple == 0 else (h // multiple + 1) * multiple
    new_w = w if w % multiple == 0 else (w // multiple + 1) * multiple

    padded = np.zeros((new_h, new_w), dtype=np.float32)
    padded[:h, :w] = img

    return padded, h, w


def load_frame(path, mean, std, pad_multiple=32):
    img = Image.open(path).convert("I")
    img = np.array(img, dtype=np.float32)

    padded, original_h, original_w = pad_to_multiple(img, pad_multiple)

    normalized = (padded - mean) / std

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    # shape: [1, 1, H, W]

    return tensor, img, original_h, original_w