from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def read_tiff_stack(path: str | Path) -> np.ndarray:
    """Read a multi-page TIFF stack as a ``(z, y, x)`` NumPy array."""
    stack = tifffile.imread(Path(path))
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF stack, got shape {stack.shape} from {path}")
    return np.asarray(stack)


def write_tiff_stack(
    path: str | Path,
    stack: np.ndarray,
    *,
    bit_depth: int = 16,
    compression: int | str | None = None,
) -> None:
    """Write a ``(z, y, x)`` stack as an ImageJ/MATLAB-compatible TIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {stack.shape}")

    if bit_depth == 16:
        data = np.clip(stack, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    elif bit_depth == 32:
        data = stack.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported bit depth {bit_depth}; expected 16 or 32")

    tifffile.imwrite(path, data, photometric="minisblack", compression=compression)


def read_tiff_zcyx(path: str | Path, *, channels: int | None = None) -> np.ndarray:
    """Read a 3D TIFF or page-major ZCYX TIFF as ``(z, c, y, x)``."""
    stack = tifffile.imread(Path(path))
    if stack.ndim == 3:
        if channels is None:
            return stack[:, np.newaxis, :, :]
        if channels <= 0:
            raise ValueError("channels must be positive")
        if stack.shape[0] % channels:
            raise ValueError(f"{path} has {stack.shape[0]} planes, not divisible by {channels} channels")
        z_size = stack.shape[0] // channels
        return stack.reshape(z_size, channels, stack.shape[1], stack.shape[2])
    if stack.ndim == 4:
        if channels is not None and stack.shape[1] != channels:
            raise ValueError(f"{path} has {stack.shape[1]} channels, not {channels}")
        return np.asarray(stack)
    raise ValueError(f"Expected a 3D or 4D TIFF stack, got shape {stack.shape} from {path}")


def write_tiff_zcyx(
    path: str | Path,
    stack: np.ndarray,
    *,
    bit_depth: int = 16,
    compression: int | str | None = None,
) -> None:
    """Write ``(z, c, y, x)`` as page-major ``(z*c, y, x)`` TIFF."""
    if stack.ndim != 4:
        raise ValueError(f"Expected a ZCYX stack, got shape {stack.shape}")
    z_size, channels, y_size, x_size = stack.shape
    write_tiff_stack(
        path,
        stack.reshape(z_size * channels, y_size, x_size),
        bit_depth=bit_depth,
        compression=compression,
    )
