from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def prob_to_binary(prob_mask, threshold=0.5):
    return (prob_mask >= threshold).astype(np.uint8)


def extract_objects(binary_mask, min_area=2):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    objects = []

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]

        if area < min_area:
            continue

        cx, cy = centroids[label_id]

        objects.append({
            "x_center": float(cx),
            "y_center": float(cy),
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": int(area)
        })

    return objects


def save_mask(binary_mask, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_img.save(save_path)


def save_overlay(original_img, binary_mask, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    original = original_img.astype(np.float32)

    if original.max() > original.min():
        original = (original - original.min()) / (original.max() - original.min())
    else:
        original = original * 0.0

    original = (original * 255).astype(np.uint8)

    rgb = np.stack([original, original, original], axis=-1)

    # выделяем найденные пиксели красным
    rgb[binary_mask > 0] = [255, 0, 0]

    Image.fromarray(rgb).save(save_path)