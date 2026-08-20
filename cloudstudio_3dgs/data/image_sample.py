"""Per-image multimodal loading with a single spatial transform contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_FACTORS = (1, 2, 4)


@dataclass(frozen=True)
class CropWindow:
    """Full-resolution crop window shared by every image modality."""

    x: int
    y: int
    width: int
    height: int

    def validate(self, image_width: int, image_height: int, factor: int) -> None:
        if min(self.x, self.y) < 0 or min(self.width, self.height) <= 0:
            raise ValueError("crop coordinates must be non-negative with positive size")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError("crop window exceeds the source image")
        if self.width % factor or self.height % factor:
            raise ValueError("crop width and height must be divisible by factor")


@dataclass(frozen=True)
class ImageSample:
    image: np.ndarray
    valid_mask: np.ndarray
    static_mask: np.ndarray
    depth_valid_mask: np.ndarray | None
    mask: np.ndarray
    depth: np.ndarray | None
    confidence: np.ndarray | None
    factor: int
    crop: CropWindow


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _read_float_array(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            actual_key = "range_m" if key == "depth" and "range_m" in archive else key
            if actual_key not in archive:
                raise KeyError(f"{path} does not contain array {key!r}")
            value = archive[actual_key]
            if value.ndim == 1 and {"pixel_index", "shape"} <= set(archive.files):
                shape = np.asarray(archive["shape"], dtype=np.int64)
                indexes = np.asarray(archive["pixel_index"], dtype=np.int64)
                if shape.shape != (2,) or len(indexes) != len(value):
                    raise ValueError(f"invalid sparse depth layout in {path}")
                dense = np.zeros((int(shape[0]), int(shape[1])), dtype=np.float32)
                if np.any(indexes < 0) or np.any(indexes >= dense.size):
                    raise ValueError(f"sparse depth pixel index is outside shape in {path}")
                dense.flat[indexes] = value
                value = dense
    else:
        raise ValueError(f"expected .npy or .npz for {key}, got {path}")
    return np.asarray(value, dtype=np.float32)


def _validate_spatial_shape(name: str, array: np.ndarray, shape: tuple[int, int]) -> None:
    if array.shape != shape:
        raise ValueError(
            f"{name} shape {array.shape} does not match image shape {shape}"
        )


def _resize_image(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(array).resize(size, Image.Resampling.LANCZOS))


def _resize_mask(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8) * 255)
    return np.asarray(image.resize(size, Image.Resampling.NEAREST), dtype=np.uint8) > 0


def _resize_float(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32))
    return np.asarray(image.resize(size, Image.Resampling.NEAREST), dtype=np.float32)


def prepare_image_sample(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    static_mask: np.ndarray | None = None,
    depth_valid_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    factor: int = 1,
    crop: CropWindow | None = None,
) -> ImageSample:
    """Apply one crop/downsample transform and compose the training mask."""
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("image must have shape [H, W, 3] or [H, W, 4]")
    if pixels.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(f"factor must be one of {SUPPORTED_FACTORS}")

    height, width = pixels.shape[:2]
    shape = (height, width)
    valid = np.asarray(valid_mask, dtype=bool)
    _validate_spatial_shape("valid_mask", valid, shape)
    static = np.ones(shape, dtype=bool) if static_mask is None else np.asarray(static_mask, dtype=bool)
    _validate_spatial_shape("static_mask", static, shape)

    depth_array = None if depth is None else np.asarray(depth, dtype=np.float32)
    confidence_array = None if confidence is None else np.asarray(confidence, dtype=np.float32)
    explicit_depth_valid = (
        None if depth_valid_mask is None else np.asarray(depth_valid_mask, dtype=bool)
    )
    for name, array in (
        ("depth", depth_array),
        ("confidence", confidence_array),
        ("depth_valid_mask", explicit_depth_valid),
    ):
        if array is not None:
            _validate_spatial_shape(name, array, shape)
    if confidence_array is not None and depth_array is None:
        raise ValueError("confidence requires depth")
    if explicit_depth_valid is not None and depth_array is None:
        raise ValueError("depth_valid_mask requires depth")

    depth_valid = None
    if depth_array is not None:
        depth_valid = np.isfinite(depth_array) & (depth_array > 0.0)
        if confidence_array is not None:
            depth_valid &= np.isfinite(confidence_array) & (confidence_array > 0.0)
        if explicit_depth_valid is not None:
            depth_valid &= explicit_depth_valid

    window = crop or CropWindow(0, 0, width, height)
    window.validate(width, height, factor)
    rows = slice(window.y, window.y + window.height)
    columns = slice(window.x, window.x + window.width)
    pixels = pixels[rows, columns]
    valid = valid[rows, columns]
    static = static[rows, columns]
    if depth_valid is not None:
        depth_valid = depth_valid[rows, columns]
    if depth_array is not None:
        depth_array = depth_array[rows, columns]
    if confidence_array is not None:
        confidence_array = confidence_array[rows, columns]

    if factor > 1:
        output_size = (window.width // factor, window.height // factor)
        pixels = _resize_image(pixels, output_size)
        valid = _resize_mask(valid, output_size)
        static = _resize_mask(static, output_size)
        if depth_valid is not None:
            depth_valid = _resize_mask(depth_valid, output_size)
        if depth_array is not None:
            depth_array = _resize_float(depth_array, output_size)
        if confidence_array is not None:
            confidence_array = _resize_float(confidence_array, output_size)

    combined = valid & static
    if depth_valid is not None:
        combined &= depth_valid
    return ImageSample(
        image=pixels,
        valid_mask=valid,
        static_mask=static,
        depth_valid_mask=depth_valid,
        mask=combined,
        depth=depth_array,
        confidence=confidence_array,
        factor=factor,
        crop=window,
    )


def load_image_sample(
    image_path: Path,
    valid_mask_path: Path,
    *,
    static_mask_path: Path | None = None,
    depth_valid_mask_path: Path | None = None,
    depth_path: Path | None = None,
    confidence_path: Path | None = None,
    factor: int = 1,
    crop: CropWindow | None = None,
    depth_key: str = "depth",
    confidence_key: str = "confidence",
) -> ImageSample:
    """Read one image's own artifacts; no camera-level mask fallback is used."""
    for name, path in (("image", image_path), ("valid mask", valid_mask_path)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    optional_paths = (
        ("static mask", static_mask_path),
        ("depth-valid mask", depth_valid_mask_path),
        ("depth", depth_path),
        ("confidence", confidence_path),
    )
    for name, path in optional_paths:
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    valid = _read_mask(valid_mask_path)
    static = None if static_mask_path is None else _read_mask(static_mask_path)
    depth_valid = (
        None if depth_valid_mask_path is None else _read_mask(depth_valid_mask_path)
    )
    depth = None if depth_path is None else _read_float_array(depth_path, depth_key)
    confidence = (
        None
        if confidence_path is None
        else _read_float_array(confidence_path, confidence_key)
    )
    return prepare_image_sample(
        image,
        valid,
        static_mask=static,
        depth_valid_mask=depth_valid,
        depth=depth,
        confidence=confidence,
        factor=factor,
        crop=crop,
    )
