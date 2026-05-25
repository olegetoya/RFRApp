import json
from pathlib import Path

import torch
from torch import nn

from model.RFR_framework import RFR
from inference.preprocessing import load_frame, list_images
from inference.postprocessing import (
    prob_to_binary,
    extract_objects,
    save_mask,
    save_overlay,
)


class DetectorNet(nn.Module):
    def __init__(self, head_name="ResUNet", mid_channels=16):
        super().__init__()
        self.model = RFR(mid_channels=mid_channels, head_name=head_name)

    def forward_test(self, img, feat_prop):
        pred, feat_prop = self.model.forward_test(img, feat_prop)
        return pred, feat_prop


class RFRDetector:
    def __init__(self, checkpoint_path, config_path):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)

        self.config = self._load_config()
        self.device = self._select_device()
        self.model = self._load_model()

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _select_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        model = DetectorNet(
            head_name=self.config["head_name"],
            mid_channels=self.config["mid_channels"],
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        clean_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            clean_state_dict[key] = value

        model.load_state_dict(clean_state_dict, strict=True)
        model.to(self.device)
        model.eval()

        print("Device:", self.device)
        print("Model loaded:", self.checkpoint_path)

        return model

    @torch.no_grad()
    def predict_folder(self, input_dir, output_dir):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        masks_dir = output_dir / "masks"
        overlays_dir = output_dir / "overlays"

        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_images(input_dir)

        if not image_paths:
            raise RuntimeError(f"No images found in {input_dir}")

        threshold = float(self.config["threshold"])
        min_area = int(self.config["min_area"])
        mean = float(self.config["mean"])
        std = float(self.config["std"])
        pad_multiple = int(self.config["pad_multiple"])

        all_results = []
        feat_prop = None

        for frame_idx, image_path in enumerate(image_paths):
            img_tensor, original_img, original_h, original_w = load_frame(
                image_path,
                mean=mean,
                std=std,
                pad_multiple=pad_multiple,
            )

            img_tensor = img_tensor.to(self.device)

            pred, feat_prop = self.model.forward_test(img_tensor, feat_prop)

            pred = pred[:, :, :original_h, :original_w]
            prob_mask = pred[0, 0].detach().cpu().numpy()

            binary_mask = prob_to_binary(prob_mask, threshold=threshold)
            objects = extract_objects(binary_mask, min_area=min_area)

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
                    "x": obj["x"],
                    "y": obj["y"],
                    "width": obj["width"],
                    "height": obj["height"],
                    "area": obj["area"],
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                })

            print(f"{frame_idx + 1}/{len(image_paths)} {image_path.name}: objects={len(objects)}")

        return all_results