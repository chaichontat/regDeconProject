from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import tifffile
import zarr
from PIL import Image

from .deconvolution import WbProjectorConfig, make_wb_projectors_gpu
from .previews import _scale_uint8

try:
    profile
except NameError:  # pragma: no cover - only defined by kernprof
    profile = lambda func: func


@dataclass(frozen=True)
class PreviewConfig:
    chunk_depth: int = 64
    halo_z: int = 20
    iterations: int = 1
    max_chunks: int | None = None


@profile
def write_deconvolved_max_projection_png(
    dataset_dir: Path,
    *,
    output_png: Path | None = None,
    view_a_path: Path | None = None,
    view_b_path: Path | None = None,
    config: PreviewConfig = PreviewConfig(),
) -> Path:
    dataset_dir = Path(dataset_dir)
    output_dir = dataset_dir / "png"
    output_dir.mkdir(exist_ok=True)
    output_png = output_png or output_dir / "deconvolved_max_projection.png"
    output_json = output_png.with_suffix(".json")

    metadata = json.loads((dataset_dir / "simulation.json").read_text())
    z_size, y_size, x_size, channels = [int(value) for value in metadata["shape_zyxc"]]
    background = float(metadata.get("background", 0.0))
    psf_a = _read_normalized_psf(dataset_dir / "psf_cropped_view_a.tif")
    psf_b = _read_normalized_psf(dataset_dir / "psf_cropped_view_b.tif")
    min_slab_depth = max(int(psf_a.shape[0]), int(psf_b.shape[0]))
    if z_size < min_slab_depth:
        raise ValueError(
            "z_size must be at least the largest PSF z-depth "
            f"({min_slab_depth}) so chunks can embed the projector"
        )
    forward_a, back_a = [arr[:, 0].astype(cp.float32) for arr in make_wb_projectors_gpu(psf_a, WbProjectorConfig())]
    forward_b, back_b = [arr[:, 0].astype(cp.float32) for arr in make_wb_projectors_gpu(psf_b, WbProjectorConfig())]

    mip = np.zeros((channels, y_size, x_size), dtype=np.float32)
    view_a_path = view_a_path or dataset_dir / "view_a_object_grid.tif"
    view_b_path = view_b_path or dataset_dir / "view_b_object_grid.tif"
    if not view_a_path.exists():
        view_a_path = dataset_dir / "view_a.tif"
    if not view_b_path.exists():
        view_b_path = dataset_dir / "view_b.tif"
    with tifffile.TiffFile(view_a_path) as tif_a, tifffile.TiffFile(view_b_path) as tif_b:
        for channel in range(channels):
            _process_channel(
                mip[channel],
                tif_a,
                tif_b,
                channel=channel,
                channels=channels,
                z_size=z_size,
                y_size=y_size,
                x_size=x_size,
                background=background,
                forward_a=forward_a,
                back_a=back_a,
                forward_b=forward_b,
                back_b=back_b,
                config=config,
                min_slab_depth=min_slab_depth,
            )

    rgb = np.stack([_scale_uint8(mip[channel]) for channel in range(min(3, channels))], axis=-1)
    Image.fromarray(rgb if rgb.shape[-1] > 1 else rgb[..., 0]).save(output_png)
    output_json.write_text(
        json.dumps(
            {
                "source": str(dataset_dir),
                "view_a": str(view_a_path),
                "view_b": str(view_b_path),
                "output_png": str(output_png),
                "method": "chunked dual-view WB Richardson-Lucy max projection",
                "convolution": "MATLAB-style FFT: ifftn(fftn(volume) * fftn(ifftshift(centered_projector)))",
                "background_subtracted": background,
                "chunk_depth": config.chunk_depth,
                "halo_z": config.halo_z,
                "iterations": config.iterations,
                "channels": channels,
                "shape_zyxc": [z_size, y_size, x_size, channels],
            },
            indent=2,
        )
    )
    return output_png


@profile
def write_zarr_deconvolved_max_projection_png(
    view_a_path: Path,
    view_b_path: Path,
    *,
    psf_a_path: Path,
    psf_b_path: Path,
    output_png: Path,
    background: float = 0.0,
    config: PreviewConfig = PreviewConfig(),
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json = output_png.with_suffix(".json")
    view_a = zarr.open_array(view_a_path, mode="r")
    view_b = zarr.open_array(view_b_path, mode="r")
    if view_a.shape != view_b.shape:
        raise ValueError(f"view A and view B Zarr arrays must have the same shape, got {view_a.shape} and {view_b.shape}")
    if view_a.ndim != 4:
        raise ValueError(f"Expected Zarr arrays shaped (Z, C, Y, X), got {view_a.shape}")
    z_size, channels, y_size, x_size = [int(value) for value in view_a.shape]
    psf_a = _read_normalized_psf(psf_a_path)
    psf_b = _read_normalized_psf(psf_b_path)
    min_slab_depth = max(int(psf_a.shape[0]), int(psf_b.shape[0]))
    if z_size < min_slab_depth:
        raise ValueError(
            "z_size must be at least the largest PSF z-depth "
            f"({min_slab_depth}) so chunks can embed the projector"
        )
    forward_a, back_a = [arr[:, 0].astype(cp.float32) for arr in make_wb_projectors_gpu(psf_a, WbProjectorConfig())]
    forward_b, back_b = [arr[:, 0].astype(cp.float32) for arr in make_wb_projectors_gpu(psf_b, WbProjectorConfig())]

    mip = np.zeros((channels, y_size, x_size), dtype=np.float32)
    for channel in range(channels):
        _process_zarr_channel(
            mip[channel],
            view_a,
            view_b,
            channel=channel,
            z_size=z_size,
            background=background,
            forward_a=forward_a,
            back_a=back_a,
            forward_b=forward_b,
            back_b=back_b,
            config=config,
            min_slab_depth=min_slab_depth,
        )

    rgb = np.stack([_scale_uint8(mip[channel]) for channel in range(min(3, channels))], axis=-1)
    Image.fromarray(rgb if rgb.shape[-1] > 1 else rgb[..., 0]).save(output_png)
    output_json.write_text(
        json.dumps(
            {
                "view_a": str(view_a_path),
                "view_b": str(view_b_path),
                "psf_a": str(psf_a_path),
                "psf_b": str(psf_b_path),
                "output_png": str(output_png),
                "method": "chunked Zarr dual-view WB Richardson-Lucy max projection",
                "background_subtracted": float(background),
                "chunk_depth": config.chunk_depth,
                "halo_z": config.halo_z,
                "iterations": config.iterations,
                "channels": channels,
                "shape_zcyx": [z_size, channels, y_size, x_size],
            },
            indent=2,
        )
    )
    return output_png


@profile
def _process_channel(
    mip_channel: np.ndarray,
    tif_a: tifffile.TiffFile,
    tif_b: tifffile.TiffFile,
    *,
    channel: int,
    channels: int,
    z_size: int,
    y_size: int,
    x_size: int,
    background: float,
    forward_a: cp.ndarray,
    back_a: cp.ndarray,
    forward_b: cp.ndarray,
    back_b: cp.ndarray,
    config: PreviewConfig,
    min_slab_depth: int,
) -> None:
    chunks_done = 0
    for z0 in range(0, z_size, config.chunk_depth):
        if config.max_chunks is not None and chunks_done >= config.max_chunks:
            break
        z1 = min(z_size, z0 + config.chunk_depth)
        hz0 = max(0, z0 - config.halo_z)
        hz1 = min(z_size, z1 + config.halo_z)
        hz0, hz1 = _expand_z_window(hz0, hz1, z_size, min_slab_depth)
        c0 = z0 - hz0
        c1 = c0 + (z1 - z0)
        image_a = cp.asarray(
            _read_slab(tif_a, channel, channels, hz0, hz1, y_size, x_size, background),
            dtype=cp.float32,
        )
        image_a = cp.maximum(image_a, np.float32(1e-6))
        image_b = cp.asarray(
            _read_slab(tif_b, channel, channels, hz0, hz1, y_size, x_size, background),
            dtype=cp.float32,
        )
        image_b = cp.maximum(image_b, np.float32(1e-6))
        estimate = cp.maximum((image_a + image_b) * np.float32(0.5), np.float32(1e-6))

        for _ in range(config.iterations):
            estimate = _apply_view_update(estimate, image_a, forward_a, back_a)
            estimate = _apply_view_update(estimate, image_b, forward_b, back_b)

        central_mip = cp.asnumpy(cp.max(estimate[c0:c1], axis=0))
        np.maximum(mip_channel, central_mip, out=mip_channel)
        del image_a, image_b, estimate, central_mip
        cp.get_default_memory_pool().free_all_blocks()
        chunks_done += 1


@profile
def _process_zarr_channel(
    mip_channel: np.ndarray,
    view_a: zarr.Array,
    view_b: zarr.Array,
    *,
    channel: int,
    z_size: int,
    background: float,
    forward_a: cp.ndarray,
    back_a: cp.ndarray,
    forward_b: cp.ndarray,
    back_b: cp.ndarray,
    config: PreviewConfig,
    min_slab_depth: int,
) -> None:
    chunks_done = 0
    for z0 in range(0, z_size, config.chunk_depth):
        if config.max_chunks is not None and chunks_done >= config.max_chunks:
            break
        z1 = min(z_size, z0 + config.chunk_depth)
        hz0 = max(0, z0 - config.halo_z)
        hz1 = min(z_size, z1 + config.halo_z)
        hz0, hz1 = _expand_z_window(hz0, hz1, z_size, min_slab_depth)
        c0 = z0 - hz0
        c1 = c0 + (z1 - z0)
        image_a = cp.asarray(_read_zarr_slab(view_a, channel, hz0, hz1, background), dtype=cp.float32)
        image_a = cp.maximum(image_a, np.float32(1e-6))
        image_b = cp.asarray(_read_zarr_slab(view_b, channel, hz0, hz1, background), dtype=cp.float32)
        image_b = cp.maximum(image_b, np.float32(1e-6))
        estimate = cp.maximum((image_a + image_b) * np.float32(0.5), np.float32(1e-6))

        for _ in range(config.iterations):
            estimate = _apply_view_update(estimate, image_a, forward_a, back_a)
            estimate = _apply_view_update(estimate, image_b, forward_b, back_b)

        central_mip = cp.asnumpy(cp.max(estimate[c0:c1], axis=0))
        np.maximum(mip_channel, central_mip, out=mip_channel)
        del image_a, image_b, estimate, central_mip
        cp.get_default_memory_pool().free_all_blocks()
        chunks_done += 1


def _expand_z_window(z0: int, z1: int, z_size: int, min_depth: int) -> tuple[int, int]:
    missing = min_depth - (z1 - z0)
    if missing <= 0:
        return z0, z1
    grow_before = min(z0, (missing + 1) // 2)
    z0 -= grow_before
    missing -= grow_before
    grow_after = min(z_size - z1, missing)
    z1 += grow_after
    missing -= grow_after
    if missing > 0:
        z0 -= min(z0, missing)
    return z0, z1


@profile
def _read_slab(
    tif: tifffile.TiffFile,
    channel: int,
    channels: int,
    z0: int,
    z1: int,
    y_size: int,
    x_size: int,
    background: float,
) -> np.ndarray:
    page_indices = [z * channels + channel for z in range(z0, z1)]
    slab = tif.asarray(key=page_indices, maxworkers=min(8, len(page_indices))).reshape(z1 - z0, y_size, x_size)
    return slab.astype(np.float32) - np.float32(background)


def _read_zarr_slab(
    array: zarr.Array,
    channel: int,
    z0: int,
    z1: int,
    background: float,
) -> np.ndarray:
    return np.asarray(array[z0:z1, channel, :, :], dtype=np.float32) - np.float32(background)


def _apply_view_update(
    estimate: cp.ndarray,
    image: cp.ndarray,
    forward_projector: cp.ndarray,
    back_projector: cp.ndarray,
) -> cp.ndarray:
    blurred = cp.maximum(_fft_convolve_same(estimate, forward_projector), np.float32(1e-6))
    ratio = image / blurred
    del blurred
    correction = _fft_convolve_same(ratio, back_projector)
    del ratio
    updated = cp.maximum(estimate * correction, np.float32(1e-6))
    del correction, estimate
    cp.get_default_memory_pool().free_all_blocks()
    return updated


@profile
def _fft_convolve_same(
    volume: cp.ndarray,
    projector: cp.ndarray,
) -> cp.ndarray:
    shape = tuple(int(value) for value in volume.shape)
    otf = _projector_otf(projector, shape)
    transformed = cp.fft.rfftn(volume)
    result = cp.fft.irfftn(transformed * otf, s=shape).astype(cp.float32)
    del transformed, otf
    return result


def _projector_otf(projector: cp.ndarray, shape: tuple[int, int, int]) -> cp.ndarray:
    padded = cp.zeros(shape, dtype=cp.float32)
    starts = [(shape[axis] - int(projector.shape[axis])) // 2 for axis in range(3)]
    slices = tuple(slice(start, start + int(projector.shape[axis])) for axis, start in enumerate(starts))
    padded[slices] = projector
    return cp.fft.rfftn(cp.fft.ifftshift(padded))


def _read_normalized_psf(path: Path) -> np.ndarray:
    psf = tifffile.imread(path).astype(np.float32)
    total = np.float32(psf.sum())
    if total <= 0:
        raise ValueError(f"{path} has non-positive sum")
    return psf / total
