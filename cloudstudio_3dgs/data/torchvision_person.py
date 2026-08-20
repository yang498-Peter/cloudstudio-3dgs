"""Explicit, hash-locked TorchVision person-segmentation runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ARCHITECTURE = "maskrcnn_resnet50_fpn_v2"
EXPECTED_WEIGHTS = "MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_person_model_lock(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "runtime",
        "runtime_version",
        "torch_version",
        "architecture",
        "weights",
        "weights_url",
        "weights_sha256",
        "person_class_index",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"person model lock is missing {sorted(missing)}")
    if value["schema_version"] != 1:
        raise ValueError("unsupported person model lock schema")
    if value["runtime"] != "torchvision":
        raise ValueError("person model lock must use torchvision")
    if value["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError("unexpected person model architecture")
    if value["weights"] != EXPECTED_WEIGHTS:
        raise ValueError("unexpected person model weights identity")
    if value["person_class_index"] != 1:
        raise ValueError("COCO person class index must be 1")
    digest = str(value["weights_sha256"]).lower()
    if len(digest) != 64:
        raise ValueError("person model lock SHA256 must contain 64 characters")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError("person model lock SHA256 is not hexadecimal") from exc
    value["weights_sha256"] = digest
    return value


class TorchVisionPersonSegmenter:
    """Mask R-CNN adapter that returns only COCO person instances."""

    def __init__(
        self,
        lock_path: Path,
        weights_path: Path,
        *,
        device: str,
        score_threshold: float = 0.65,
        inference_max_dimension: int = 800,
    ) -> None:
        lock = load_person_model_lock(lock_path)
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing person model weights: {weights_path}")
        actual_sha256 = _sha256_file(weights_path)
        if actual_sha256 != lock["weights_sha256"]:
            raise ValueError(
                "person model weights SHA256 mismatch: "
                f"expected {lock['weights_sha256']}, computed {actual_sha256}"
            )

        import torch
        import torchvision
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

        torchvision_version = importlib.metadata.version("torchvision")
        torch_version = str(torch.__version__)
        if torchvision_version != str(lock["runtime_version"]):
            raise ValueError(
                "torchvision version mismatch: "
                f"expected {lock['runtime_version']}, found {torchvision_version}"
            )
        if torch_version != str(lock["torch_version"]):
            raise ValueError(
                f"torch version mismatch: expected {lock['torch_version']}, found {torch_version}"
            )
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA person-mask inference requested but CUDA is unavailable")

        model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self._torch = torch
        self._model = model.eval().to(resolved_device)
        self._device = resolved_device
        self._person_class_index = int(lock["person_class_index"])
        if not 0.0 < score_threshold <= 1.0:
            raise ValueError("person score threshold must be in (0, 1]")
        if inference_max_dimension < 800:
            raise ValueError("person inference max dimension must be at least 800")
        self._score_threshold = float(score_threshold)
        self._inference_max_dimension = int(inference_max_dimension)
        self.model_identity = {
            "runtime": "torchvision",
            "version": torchvision_version,
            "torch_version": torch_version,
            "architecture": str(lock["architecture"]),
            "weights": str(lock["weights"]),
            "weights_url": str(lock["weights_url"]),
            "weights_sha256": actual_sha256,
            "person_class_index": self._person_class_index,
            "device": str(resolved_device),
            "inference_max_dimension": self._inference_max_dimension,
        }

    def segment(self, image: np.ndarray) -> list[dict[str, Any]]:
        pixels = np.asarray(image, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("person segmenter expects an RGB uint8 image")
        original_height, original_width = pixels.shape[:2]
        maximum = max(original_height, original_width)
        scale = min(1.0, self._inference_max_dimension / maximum)
        if scale < 1.0:
            from PIL import Image

            inference_width = max(1, int(round(original_width * scale)))
            inference_height = max(1, int(round(original_height * scale)))
            pixels = np.asarray(
                Image.fromarray(pixels).resize(
                    (inference_width, inference_height), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
        else:
            inference_height, inference_width = original_height, original_width
        tensor = (
            self._torch.from_numpy(np.array(pixels, dtype=np.uint8, copy=True, order="C"))
            .permute(2, 0, 1)
            .to(device=self._device, dtype=self._torch.float32)
            .div_(255.0)
        )
        with self._torch.inference_mode():
            prediction = self._model([tensor])[0]
        keep = (prediction["labels"] == self._person_class_index) & (
            prediction["scores"] >= self._score_threshold
        )
        scores = prediction["scores"][keep].detach().cpu().numpy()
        selected_boxes = prediction["boxes"][keep]
        selected_masks = prediction["masks"][keep]
        if (inference_height, inference_width) != (original_height, original_width):
            selected_masks = self._torch.nn.functional.interpolate(
                selected_masks,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            selected_boxes = selected_boxes.clone()
            selected_boxes[:, (0, 2)] *= original_width / inference_width
            selected_boxes[:, (1, 3)] *= original_height / inference_height
        boxes = selected_boxes.detach().cpu().numpy()
        masks = selected_masks[:, 0].detach().cpu().numpy()
        return [
            {
                "score": float(scores[index]),
                "box_xyxy": boxes[index].astype(np.float64).tolist(),
                "mask": masks[index].astype(np.float32),
            }
            for index in range(len(scores))
        ]
