import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from model.RFR_framework import RFR


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


class RFRAppNet(nn.Module):
    """
    Wrapper around original RFR.

    The original training code often saves state_dict with keys like:
        model.feat_extract.weight

    Therefore we keep RFR inside self.model.
    """

    def __init__(self, head_name="ResUNet", mid_channels=16):
        super().__init__()
        self.model = RFR(mid_channels=mid_channels, head_name=head_name)

    def forward_test(self, img, feat_prop):
        return self.model.forward_test(img, feat_prop)


def list_images(input_dir):
    input_dir = Path(input_dir)

    image_paths = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def pad_to_multiple(img, multiple):
    h, w = img.shape

    new_h = h if h % multiple == 0 else (h // multiple + 1) * multiple
    new_w = w if w % multiple == 0 else (w // multiple + 1) * multiple

    padded = np.zeros((new_h, new_w), dtype=np.float32)
    padded[:h, :w] = img

    return padded, h, w


def load_frame(path, mean, std, pad_multiple):
    img = Image.open(path).convert("I")
    img = np.array(img, dtype=np.float32)

    normalized = (img - mean) / std
    padded, original_h, original_w = pad_to_multiple(normalized, pad_multiple)

    tensor = torch.from_numpy(np.ascontiguousarray(padded))
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor, img, original_h, original_w


def prob_to_binary(prob_mask, threshold):
    return (prob_mask >= threshold).astype(np.uint8)


def extract_objects(binary_mask, min_area):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    objects = []

    for label_id in range(1, num_labels):
        x, y, width, height, area = stats[label_id]

        if area < min_area:
            continue

        cx, cy = centroids[label_id]

        objects.append({
            "x": int(x),
            "y": int(y),
            "x_center": float(cx),
            "y_center": float(cy),
            "width": int(width),
            "height": int(height),
            "area": int(area),
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

    img = original_img.astype(np.float32)

    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = img * 0.0

    img = (img * 255).astype(np.uint8)

    rgb = np.stack([img, img, img], axis=-1)
    rgb[binary_mask > 0] = [255, 0, 0]

    Image.fromarray(rgb).save(save_path)


class RFRDetector:
    def __init__(
        self,
        config_path,
        model_name=None,
        checkpoint_path=None,
        device=None,
    ):
        self.config_path = Path(config_path)
        self.full_config = self._load_json(self.config_path)

        self.model_name = model_name or self.full_config.get("default_model", "ResUNet_RFR")
        self.common_config = self.full_config.get("common", {})

        self.model_config = self._get_model_config(self.model_name)

        self.checkpoint_path = checkpoint_path or self.model_config.get("checkpoint")
        if self.checkpoint_path is None:
            raise ValueError(f"Checkpoint path is not specified for model {self.model_name}")

        self.checkpoint_path = Path(self.checkpoint_path)

        self.device = torch.device(device) if device is not None else self._select_device()

        self.net = self._load_model()

    @staticmethod
    def _load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_model_config(self, model_name):
        models = self.full_config.get("models")

        if models is None:
            return self.full_config

        if model_name not in models:
            available = ", ".join(models.keys())
            raise ValueError(
                f"Unknown model_name: {model_name}. "
                f"Available models: {available}"
            )

        return models[model_name]

    @staticmethod
    def _select_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _load_checkpoint(path, device):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "net"):
                if key in checkpoint:
                    return checkpoint[key]

        return checkpoint

    @staticmethod
    def _strip_module_prefix(state_dict):
        clean_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            clean_state_dict[key] = value

        return clean_state_dict

    @staticmethod
    def _add_model_prefix_if_needed(state_dict):
        if not state_dict:
            return state_dict

        first_key = next(iter(state_dict.keys()))

        if first_key.startswith("model."):
            return state_dict

        return {
            f"model.{key}": value
            for key, value in state_dict.items()
        }

    def _load_model(self):
        head_name = self.model_config.get("head_name", "ResUNet")
        mid_channels = int(self.common_config.get("mid_channels", 16))

        net = RFRAppNet(
            head_name=head_name,
            mid_channels=mid_channels,
        )

        checkpoint = self._load_checkpoint(self.checkpoint_path, self.device)
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)

        try:
            net.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            state_dict = self._add_model_prefix_if_needed(state_dict)
            net.load_state_dict(state_dict, strict=True)

        net.to(self.device)
        net.eval()

        print("Model:", self.model_name)
        print("Head:", head_name)
        print("Checkpoint:", self.checkpoint_path)
        print("Device:", self.device)

        return net

    @torch.no_grad()
    def predict_folder(
        self,
        input_dir,
        output_dir,
        should_stop=None,
        progress_callback=None,
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        masks_dir = output_dir / "masks"
        overlays_dir = output_dir / "overlays"

        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_images(input_dir)

        if not image_paths:
            raise RuntimeError(f"No images found in: {input_dir}")

        mean = float(self.common_config.get("mean", 72.1040267944336))
        std = float(self.common_config.get("std", 12.302865028381348))
        threshold = float(self.common_config.get("threshold", 0.5))
        min_area = int(self.common_config.get("min_area", 2))
        pad_multiple = int(self.common_config.get("pad_multiple", 32))

        all_results = []
        feat_prop = None

        for frame_idx, image_path in enumerate(image_paths):
            if should_stop is not None and should_stop():
                print("Inference stopped by user.")
                break

            img_tensor, original_img, original_h, original_w = load_frame(
                image_path,
                mean=mean,
                std=std,
                pad_multiple=pad_multiple,
            )

            img_tensor = img_tensor.to(self.device)

            pred, feat_prop = self.net.forward_test(img_tensor, feat_prop)

            pred = pred[:, :, :original_h, :original_w]

            prob_mask = pred[0, 0].detach().cpu().numpy()
            binary_mask = prob_to_binary(prob_mask, threshold)

            objects = extract_objects(binary_mask, min_area)

            mask_path = masks_dir / f"{image_path.stem}_mask.png"
            overlay_path = overlays_dir / f"{image_path.stem}_overlay.png"

            save_mask(binary_mask, mask_path)
            save_overlay(original_img, binary_mask, overlay_path)

            for object_id, obj in enumerate(objects):
                all_results.append({
                    "frame_idx": frame_idx,
                    "frame_name": image_path.name,
                    "object_id": object_id,
                    "x_center": obj["x_center"],
                    "y_center": obj["y_center"],
                    "width": obj["width"],
                    "height": obj["height"],
                    "area": obj["area"],
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "model_name": self.model_name,
                })

            print(
                f"{frame_idx + 1}/{len(image_paths)} "
                f"{image_path.name}: objects={len(objects)}"
            )

            if progress_callback is not None:
                progress_callback(
                    frame_idx + 1,
                    len(image_paths),
                    image_path.name,
                    len(objects),
                )

        return all_results