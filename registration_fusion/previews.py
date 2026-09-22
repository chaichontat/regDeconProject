from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_max_projection_png(path: str | Path, stack_zcyx: np.ndarray) -> Path:
    """Write a Z max-projection PNG for a ZCYX stack."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    projection = max_projection_image(stack_zcyx)
    Image.fromarray(projection).save(path)
    return path


def max_projection_image(stack_zcyx: np.ndarray) -> np.ndarray:
    data = np.asarray(stack_zcyx, dtype=np.float32)
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]
    if data.ndim != 4:
        raise ValueError(f"Expected ZYX or ZCYX stack, got shape {data.shape}")

    projected_cyx = np.max(data, axis=0)
    if projected_cyx.shape[0] == 1:
        return _scale_uint8(projected_cyx[0])
    if projected_cyx.shape[0] == 3:
        channels = [_scale_uint8(projected_cyx[channel]) for channel in range(3)]
        return np.stack(channels, axis=-1)
    return _scale_uint8(np.max(projected_cyx, axis=0))


def _scale_uint8(image: np.ndarray) -> np.ndarray:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low, high = np.percentile(finite, [0.1, 99.9])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (np.asarray(image, dtype=np.float32) - np.float32(low)) / np.float32(high - low)
    return np.clip(scaled * np.float32(255.0), 0, 255).astype(np.uint8)
