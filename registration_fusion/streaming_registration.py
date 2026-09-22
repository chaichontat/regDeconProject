from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import tifffile
import zarr
from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform

from .previews import _scale_uint8
from .registration_gpu import (
    _fit_transform,
    apply_output_to_input_transform_gpu,
    estimate_translation_gpu,
    require_cuda_gpu,
)


@dataclass(frozen=True)
class TiffZcyxInfo:
    z_size: int
    channels: int
    y_size: int
    x_size: int
    dtype: np.dtype

    @property
    def shape_zcyx(self) -> tuple[int, int, int, int]:
        return self.z_size, self.channels, self.y_size, self.x_size


@dataclass(frozen=True)
class ZarrZcyxInfo:
    z_size: int
    channels: int
    y_size: int
    x_size: int
    dtype: np.dtype
    chunks_zcyx: tuple[int, int, int, int]
    array_path: str | None = None
    axes: tuple[str, ...] = ("z", "c", "y", "x")
    time_index: int | None = None

    @property
    def shape_zcyx(self) -> tuple[int, int, int, int]:
        return self.z_size, self.channels, self.y_size, self.x_size


@dataclass(frozen=True)
class ZarrZcyxSource:
    array: zarr.Array
    info: ZarrZcyxInfo
    indices_tzcyx: tuple[int | None, int, int, int, int]


@dataclass(frozen=True)
class StreamingRegistrationResult:
    output_to_input_matrix_zyx: np.ndarray
    output_to_input_offset_zyx: np.ndarray
    initial_shift_zyx: tuple[float, float, float]
    stage_modes: tuple[str, ...]
    reference_correlation: float
    reference_downsample: int
    fixed_array_path: str | None = None
    moving_array_path: str | None = None
    fixed_reference_array_path: str | None = None
    moving_reference_array_path: str | None = None
    reference_shape_zyx: tuple[int, int, int] | None = None


def inspect_tiff_zcyx(path: Path, *, channels: int | None = None) -> TiffZcyxInfo:
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) == 0:
            raise ValueError(f"{path} has no TIFF pages")
        page_shape = tif.pages[0].shape
        if len(page_shape) != 2:
            raise ValueError(f"Expected 2D TIFF pages at {path}, got page shape {page_shape}")
        page_count = len(tif.pages)
        dtype = np.dtype(tif.pages[0].dtype)
    if channels is None:
        channels = 1
    if channels <= 0:
        raise ValueError("channels must be positive")
    if page_count % channels:
        raise ValueError(f"{path} has {page_count} pages, not divisible by {channels} channels")
    return TiffZcyxInfo(page_count // channels, channels, page_shape[0], page_shape[1], dtype)


def inspect_zarr_zcyx(
    path: Path,
    *,
    array_path: str | None = None,
    time_index: int = 0,
) -> ZarrZcyxInfo:
    return open_zarr_zcyx(path, array_path=array_path, time_index=time_index).info


def open_zarr_zcyx(
    path: Path,
    *,
    array_path: str | None = None,
    time_index: int = 0,
) -> ZarrZcyxSource:
    array, resolved_array_path, axes = _open_zarr_array_with_axes(path, array_path=array_path)
    indices = _zcyx_axis_indices(axes, path, resolved_array_path)
    t_index, z_index, c_index, y_index, x_index = indices
    if t_index is not None:
        if not 0 <= time_index < int(array.shape[t_index]):
            raise ValueError(f"time_index {time_index} is outside OME-Zarr T range [0, {array.shape[t_index]})")
    if len({z_index, c_index, y_index, x_index}) != 4:
        raise ValueError(f"Zarr axes must contain distinct Z/C/Y/X dimensions at {path}")
    z_size = int(array.shape[z_index])
    channels = int(array.shape[c_index])
    y_size = int(array.shape[y_index])
    x_size = int(array.shape[x_index])
    chunks = (
        int(array.chunks[z_index]),
        int(array.chunks[c_index]),
        int(array.chunks[y_index]),
        int(array.chunks[x_index]),
    )
    info = ZarrZcyxInfo(
        z_size,
        channels,
        y_size,
        x_size,
        np.dtype(array.dtype),
        chunks,
        array_path=resolved_array_path,
        axes=tuple(axes),
        time_index=time_index if t_index is not None else None,
    )
    return ZarrZcyxSource(array=array, info=info, indices_tzcyx=(t_index, z_index, c_index, y_index, x_index))


def select_zarr_reference_array_path(
    path: Path,
    *,
    reference_downsample: int,
    array_path: str | None = None,
) -> str | None:
    if reference_downsample <= 0:
        raise ValueError("reference_downsample must be positive")
    group, datasets, axes = _open_ome_zarr_group(path)
    if group is None:
        return array_path
    full_path = array_path or datasets[0]
    full = open_zarr_zcyx(path, array_path=full_path).info
    best_path = full_path
    best_score = float("inf")
    for dataset_path in datasets:
        candidate = group[dataset_path]
        if not isinstance(candidate, zarr.Array):
            continue
        try:
            _, z_index, _c_index, y_index, x_index = _zcyx_axis_indices(axes, path, dataset_path)
        except ValueError:
            continue
        scale_y = full.y_size / max(float(candidate.shape[y_index]), 1.0)
        scale_x = full.x_size / max(float(candidate.shape[x_index]), 1.0)
        scale_z = full.z_size / max(float(candidate.shape[z_index]), 1.0)
        scale = max(scale_z, scale_y, scale_x)
        score = abs(np.log2(max(scale, 1e-6) / float(reference_downsample)))
        if score < best_score:
            best_score = score
            best_path = dataset_path
    return best_path


def seeded_random_12dof_output_to_input(
    shape_zyx: tuple[int, int, int],
    *,
    seed: int,
    max_translation_zyx: tuple[float, float, float] = (4.0, 24.0, 24.0),
    max_rotation_deg: float = 1.0,
    max_scale_delta: float = 0.01,
    max_shear: float = 0.003,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    angles = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg, size=3)).astype(np.float32)
    scales = (1.0 + rng.uniform(-max_scale_delta, max_scale_delta, size=3)).astype(np.float32)
    shear = rng.uniform(-max_shear, max_shear, size=(3, 3)).astype(np.float32)
    np.fill_diagonal(shear, 0.0)
    translation = np.array(
        [rng.uniform(-limit, limit) for limit in max_translation_zyx],
        dtype=np.float32,
    )
    matrix = ((_rotation_zyx(angles) @ np.diag(scales)) + shear).astype(np.float32)
    center = (np.asarray(shape_zyx, dtype=np.float32) - np.float32(1.0)) / np.float32(2.0)
    offset = (center - matrix @ center + translation).astype(np.float32)
    metadata = {
        "seed": int(seed),
        "matrix_zyx": matrix.tolist(),
        "offset_zyx": offset.tolist(),
        "angles_deg_zyx": np.rad2deg(angles).tolist(),
        "scales_zyx": scales.tolist(),
        "translation_zyx": translation.tolist(),
        "max_shear": float(max_shear),
    }
    return matrix, offset, metadata


def build_downsampled_channel_max_reference(
    path: Path,
    *,
    channels: int,
    downsample: int,
) -> np.ndarray:
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    info = inspect_tiff_zcyx(path, channels=channels)
    z_indices = list(range(0, info.z_size, downsample))
    ref = np.empty(
        (
            len(z_indices),
            (info.y_size + downsample - 1) // downsample,
            (info.x_size + downsample - 1) // downsample,
        ),
        dtype=np.float32,
    )
    with tifffile.TiffFile(path) as tif:
        for out_z, z in enumerate(z_indices):
            plane_max = None
            for channel in range(channels):
                plane = tif.pages[z * channels + channel].asarray()[::downsample, ::downsample].astype(np.float32)
                plane_max = plane if plane_max is None else np.maximum(plane_max, plane)
            ref[out_z] = plane_max
    return ref


def build_downsampled_channel_max_reference_zarr(
    path: Path,
    *,
    downsample: int,
    array_path: str | None = None,
    reference_array_path: str | None = None,
    time_index: int = 0,
) -> np.ndarray:
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    selected_reference_array = reference_array_path
    if selected_reference_array is None:
        selected_reference_array = select_zarr_reference_array_path(
            path,
            reference_downsample=downsample,
            array_path=array_path,
        )
    source = open_zarr_zcyx(path, array_path=selected_reference_array, time_index=time_index)
    info = source.info
    full_array_path = array_path
    if full_array_path is None:
        full_array_path = select_zarr_reference_array_path(path, reference_downsample=1, array_path=array_path)
    use_strided_root_reference = selected_reference_array == full_array_path
    stride = downsample if use_strided_root_reference else 1
    z_indices = list(range(0, info.z_size, stride))
    ref = np.empty(
        (
            len(z_indices),
            (info.y_size + stride - 1) // stride,
            (info.x_size + stride - 1) // stride,
        ),
        dtype=np.float32,
    )
    for out_z, z in enumerate(z_indices):
        planes = _read_zarr_z_plane_all_channels(source, z, downsample=stride)
        ref[out_z] = planes.max(axis=0)
    return ref


def estimate_streaming_registration(
    fixed_path: Path,
    moving_path: Path,
    *,
    channels: int,
    mode: str,
    reference_downsample: int,
    ftol: float,
    max_iterations: int,
) -> StreamingRegistrationResult:
    require_cuda_gpu()
    fixed_ref = build_downsampled_channel_max_reference(fixed_path, channels=channels, downsample=reference_downsample)
    moving_ref = build_downsampled_channel_max_reference(moving_path, channels=channels, downsample=reference_downsample)
    initial_shift_ds = estimate_translation_gpu(cp.asarray(fixed_ref), cp.asarray(moving_ref))
    matrix_ds, offset_ds, stage_modes = _fit_transform(
        fixed_ref,
        moving_ref,
        mode=mode,
        initial_shift_zyx=initial_shift_ds,
        ftol=ftol,
        max_iterations=max_iterations,
        downsample=1,
    )
    registered_ref = apply_output_to_input_transform_gpu(moving_ref[:, np.newaxis], matrix_ds, offset_ds)[:, 0]
    return StreamingRegistrationResult(
        output_to_input_matrix_zyx=matrix_ds,
        output_to_input_offset_zyx=(offset_ds * np.float32(reference_downsample)).astype(np.float32),
        initial_shift_zyx=tuple(float(value * reference_downsample) for value in initial_shift_ds),
        stage_modes=stage_modes,
        reference_correlation=_correlation(fixed_ref, registered_ref),
        reference_downsample=reference_downsample,
    )


def estimate_zarr_streaming_registration(
    fixed_path: Path,
    moving_path: Path,
    *,
    mode: str,
    reference_downsample: int,
    ftol: float,
    max_iterations: int,
    fixed_array_path: str | None = None,
    moving_array_path: str | None = None,
    fixed_reference_array_path: str | None = None,
    moving_reference_array_path: str | None = None,
    time_index: int = 0,
) -> StreamingRegistrationResult:
    require_cuda_gpu()
    fixed_full = inspect_zarr_zcyx(fixed_path, array_path=fixed_array_path, time_index=time_index)
    moving_full = inspect_zarr_zcyx(moving_path, array_path=moving_array_path, time_index=time_index)
    if fixed_full.shape_zcyx != moving_full.shape_zcyx:
        raise ValueError(f"Expected matching full-resolution ZCYX shapes, got fixed={fixed_full.shape_zcyx}, moving={moving_full.shape_zcyx}")
    fixed_reference_array_path = fixed_reference_array_path or select_zarr_reference_array_path(
        fixed_path,
        reference_downsample=reference_downsample,
        array_path=fixed_array_path,
    )
    moving_reference_array_path = moving_reference_array_path or select_zarr_reference_array_path(
        moving_path,
        reference_downsample=reference_downsample,
        array_path=moving_array_path,
    )
    fixed_reference = inspect_zarr_zcyx(fixed_path, array_path=fixed_reference_array_path, time_index=time_index)
    moving_reference = inspect_zarr_zcyx(moving_path, array_path=moving_reference_array_path, time_index=time_index)
    if fixed_reference.shape_zcyx != moving_reference.shape_zcyx:
        raise ValueError(
            "Expected matching reference ZCYX shapes, "
            f"got fixed={fixed_reference.shape_zcyx}, moving={moving_reference.shape_zcyx}"
        )
    fixed_ref = build_downsampled_channel_max_reference_zarr(
        fixed_path,
        downsample=reference_downsample,
        array_path=fixed_array_path,
        reference_array_path=fixed_reference_array_path,
        time_index=time_index,
    )
    moving_ref = build_downsampled_channel_max_reference_zarr(
        moving_path,
        downsample=reference_downsample,
        array_path=moving_array_path,
        reference_array_path=moving_reference_array_path,
        time_index=time_index,
    )
    initial_shift_ds = estimate_translation_gpu(cp.asarray(fixed_ref), cp.asarray(moving_ref))
    matrix_ds, offset_ds, stage_modes = _fit_transform(
        fixed_ref,
        moving_ref,
        mode=mode,
        initial_shift_zyx=initial_shift_ds,
        ftol=ftol,
        max_iterations=max_iterations,
        downsample=1,
    )
    registered_ref = apply_output_to_input_transform_gpu(moving_ref[:, np.newaxis], matrix_ds, offset_ds)[:, 0]
    matrix_full, offset_full = _scale_reference_transform_to_full_resolution(
        matrix_ds,
        offset_ds,
        reference_shape_zyx=fixed_ref.shape,
        full_shape_zyx=(fixed_full.z_size, fixed_full.y_size, fixed_full.x_size),
    )
    return StreamingRegistrationResult(
        output_to_input_matrix_zyx=matrix_full,
        output_to_input_offset_zyx=offset_full,
        initial_shift_zyx=tuple(
            float(value)
            for value in _scale_reference_vector_to_full_resolution(
                np.asarray(initial_shift_ds, dtype=np.float32),
                reference_shape_zyx=fixed_ref.shape,
                full_shape_zyx=(fixed_full.z_size, fixed_full.y_size, fixed_full.x_size),
            )
        ),
        stage_modes=stage_modes,
        reference_correlation=_correlation(fixed_ref, registered_ref),
        reference_downsample=reference_downsample,
        fixed_array_path=fixed_full.array_path,
        moving_array_path=moving_full.array_path,
        fixed_reference_array_path=fixed_reference.array_path,
        moving_reference_array_path=moving_reference.array_path,
        reference_shape_zyx=tuple(int(value) for value in fixed_ref.shape),
    )


def write_affine_transformed_zcyx_tiff(
    input_path: Path,
    output_path: Path,
    *,
    channels: int,
    matrix_zyx: np.ndarray,
    offset_zyx: np.ndarray,
    chunk_depth: int,
    bit_depth: int,
    compression: int | str | None,
    png_path: Path | None = None,
) -> None:
    require_cuda_gpu()
    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")
    info = inspect_tiff_zcyx(input_path, channels=channels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix_zyx, dtype=np.float32)
    offset = np.asarray(offset_zyx, dtype=np.float32)
    mip = np.zeros((channels, info.y_size, info.x_size), dtype=np.float32) if png_path is not None else None

    with tifffile.TiffFile(input_path) as src, tifffile.TiffWriter(output_path, bigtiff=True) as dst:
        for z0 in range(0, info.z_size, chunk_depth):
            z1 = min(info.z_size, z0 + chunk_depth)
            read_z0, read_z1 = _input_z_bounds(matrix, offset, (z0, z1), info)
            chunk = np.empty(
                (z1 - z0, channels, info.y_size, info.x_size),
                dtype=_output_dtype(bit_depth),
            )
            for channel in range(channels):
                slab = _read_channel_slab(src, channel, channels, read_z0, read_z1, info)
                if slab.size == 0:
                    chunk[:, channel] = 0
                    continue
                local_offset = matrix @ np.array([z0, 0, 0], dtype=np.float32) + offset
                local_offset -= np.array([read_z0, 0, 0], dtype=np.float32)
                transformed = gpu_affine_transform(
                    cp.asarray(slab, dtype=cp.float32),
                    cp.asarray(matrix, dtype=cp.float32),
                    cp.asarray(local_offset, dtype=cp.float32),
                    output_shape=(z1 - z0, info.y_size, info.x_size),
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
                transformed_cpu = cp.asnumpy(transformed)
                if mip is not None:
                    np.maximum(mip[channel], transformed_cpu.max(axis=0), out=mip[channel])
                chunk[:, channel] = _convert_output(transformed_cpu, bit_depth)
                del transformed
                del transformed_cpu
                del slab
                cp.get_default_memory_pool().free_all_blocks()
            _write_zcyx_chunk(dst, chunk, bit_depth=bit_depth, compression=compression)
            del chunk
            gc.collect()

    if png_path is not None and mip is not None:
        _write_mip_png(png_path, mip)


def write_affine_transformed_zcyx_zarr(
    input_path: Path,
    output_path: Path,
    *,
    matrix_zyx: np.ndarray,
    offset_zyx: np.ndarray,
    chunk_depth: int,
    bit_depth: int,
    png_path: Path | None = None,
    array_path: str | None = None,
    time_index: int = 0,
) -> None:
    require_cuda_gpu()
    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")
    source = open_zarr_zcyx(input_path, array_path=array_path, time_index=time_index)
    info = source.info
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open_array(
        output_path,
        mode="w",
        shape=info.shape_zcyx,
        chunks=_output_zarr_chunks(info, chunk_depth),
        dtype=_output_dtype(bit_depth),
    )
    matrix = np.asarray(matrix_zyx, dtype=np.float32)
    offset = np.asarray(offset_zyx, dtype=np.float32)
    mip = np.zeros((info.channels, info.y_size, info.x_size), dtype=np.float32) if png_path is not None else None

    for z0 in range(0, info.z_size, chunk_depth):
        z1 = min(info.z_size, z0 + chunk_depth)
        read_z0, read_z1 = _input_z_bounds(matrix, offset, (z0, z1), info)
        chunk = np.empty(
            (z1 - z0, info.channels, info.y_size, info.x_size),
            dtype=_output_dtype(bit_depth),
        )
        for channel in range(info.channels):
            slab = _read_zarr_channel_slab(source, channel, read_z0, read_z1)
            if slab.size == 0:
                chunk[:, channel] = 0
                continue
            local_offset = matrix @ np.array([z0, 0, 0], dtype=np.float32) + offset
            local_offset -= np.array([read_z0, 0, 0], dtype=np.float32)
            transformed = gpu_affine_transform(
                cp.asarray(slab, dtype=cp.float32),
                cp.asarray(matrix, dtype=cp.float32),
                cp.asarray(local_offset, dtype=cp.float32),
                output_shape=(z1 - z0, info.y_size, info.x_size),
                order=1,
                mode="constant",
                cval=0.0,
            )
            transformed_cpu = cp.asnumpy(transformed)
            if mip is not None:
                np.maximum(mip[channel], transformed_cpu.max(axis=0), out=mip[channel])
            chunk[:, channel] = _convert_output(transformed_cpu, bit_depth)
            del transformed
            del transformed_cpu
            del slab
            cp.get_default_memory_pool().free_all_blocks()
        dst[z0:z1, :, :, :] = chunk
        del chunk
        gc.collect()

    if png_path is not None and mip is not None:
        _write_mip_png(png_path, mip)


def _open_zarr_array_with_axes(path: Path, *, array_path: str | None) -> tuple[zarr.Array, str | None, tuple[str, ...]]:
    if array_path is None:
        try:
            array = zarr.open_array(path, mode="r")
            return array, None, _array_axes(array, default=("z", "c", "y", "x"))
        except (FileNotFoundError, KeyError, ValueError, zarr.errors.ContainsGroupError):
            pass
    group, datasets, axes = _open_ome_zarr_group(path)
    if group is None:
        array = zarr.open_array(path if array_path is None else path / array_path, mode="r")
        return array, array_path, _array_axes(array, default=("z", "c", "y", "x"))
    resolved_array_path = array_path or datasets[0]
    array = group[resolved_array_path]
    if not isinstance(array, zarr.Array):
        raise ValueError(f"Expected Zarr array at {path}/{resolved_array_path}")
    return array, resolved_array_path, axes


def _open_ome_zarr_group(path: Path) -> tuple[zarr.Group | None, tuple[str, ...], tuple[str, ...]]:
    try:
        group = zarr.open_group(path, mode="r")
    except (FileNotFoundError, ValueError, zarr.errors.ContainsArrayError):
        return None, (), ()
    multiscales = group.attrs.get("multiscales")
    if not multiscales:
        return group, tuple(str(key) for key in group.keys()), ("z", "c", "y", "x")
    first = multiscales[0]
    datasets = tuple(str(dataset["path"]) for dataset in first.get("datasets", ()))
    if not datasets:
        raise ValueError(f"OME-Zarr multiscales metadata at {path} has no datasets")
    axes = _parse_ome_axes(first.get("axes"))
    return group, datasets, axes


def _parse_ome_axes(raw_axes: object) -> tuple[str, ...]:
    if not isinstance(raw_axes, list):
        return ("t", "c", "z", "y", "x")
    axes = []
    for axis in raw_axes:
        if isinstance(axis, dict):
            name = axis.get("name")
        else:
            name = axis
        if not isinstance(name, str):
            raise ValueError(f"Unsupported OME-Zarr axis metadata: {raw_axes!r}")
        axes.append(name.lower())
    return tuple(axes)


def _array_axes(array: zarr.Array, *, default: tuple[str, ...]) -> tuple[str, ...]:
    axes = array.attrs.get("_ARRAY_DIMENSIONS")
    if isinstance(axes, list) and len(axes) == array.ndim:
        return tuple(str(axis).lower() for axis in axes)
    dimension_names = getattr(array, "dimension_names", None)
    if dimension_names is not None and len(dimension_names) == array.ndim:
        return tuple(str(axis).lower() for axis in dimension_names)
    if array.ndim == len(default):
        return default
    if array.ndim == 5:
        return ("t", "c", "z", "y", "x")
    raise ValueError(f"Expected a 4D or 5D Zarr array, got shape {array.shape}")


def _zcyx_axis_indices(
    axes: tuple[str, ...],
    path: Path,
    array_path: str | None,
) -> tuple[int | None, int, int, int, int]:
    normalized = tuple(axis.lower() for axis in axes)
    label = f"{path}/{array_path}" if array_path else str(path)
    try:
        z_index = normalized.index("z")
        c_index = normalized.index("c")
        y_index = normalized.index("y")
        x_index = normalized.index("x")
    except ValueError as exc:
        raise ValueError(f"Expected Zarr axes to include Z, C, Y, and X at {label}; got {axes}") from exc
    t_index = normalized.index("t") if "t" in normalized else None
    return t_index, z_index, c_index, y_index, x_index


def _scale_reference_transform_to_full_resolution(
    matrix_reference: np.ndarray,
    offset_reference: np.ndarray,
    *,
    reference_shape_zyx: tuple[int, int, int],
    full_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    reference_to_full = _reference_to_full_scale(reference_shape_zyx, full_shape_zyx)
    full_to_reference = np.linalg.inv(reference_to_full).astype(np.float32)
    matrix_full = reference_to_full @ matrix_reference @ full_to_reference
    offset_full = reference_to_full @ offset_reference
    return matrix_full.astype(np.float32), offset_full.astype(np.float32)


def _scale_reference_vector_to_full_resolution(
    vector_reference_zyx: np.ndarray,
    *,
    reference_shape_zyx: tuple[int, int, int],
    full_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    return (_reference_to_full_scale(reference_shape_zyx, full_shape_zyx) @ vector_reference_zyx).astype(np.float32)


def _reference_to_full_scale(
    reference_shape_zyx: tuple[int, int, int],
    full_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    reference_shape = np.asarray(reference_shape_zyx, dtype=np.float32)
    full_shape = np.asarray(full_shape_zyx, dtype=np.float32)
    if np.any(reference_shape <= 0) or np.any(full_shape <= 0):
        raise ValueError("reference and full-resolution shapes must be positive")
    return np.diag(full_shape / reference_shape).astype(np.float32)


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def _input_z_bounds(
    matrix: np.ndarray,
    offset: np.ndarray,
    z_range: tuple[int, int],
    info: TiffZcyxInfo | ZarrZcyxInfo,
) -> tuple[int, int]:
    z0, z1 = z_range
    corners = np.array(
        [
            [z, y, x]
            for z in (z0, z1 - 1)
            for y in (0, info.y_size - 1)
            for x in (0, info.x_size - 1)
        ],
        dtype=np.float32,
    )
    input_z = (corners @ matrix.T + offset)[..., 0]
    read_z0 = max(0, int(np.floor(float(input_z.min()))) - 2)
    read_z1 = min(info.z_size, int(np.ceil(float(input_z.max()))) + 3)
    if read_z1 <= read_z0:
        return 0, 0
    return read_z0, read_z1


def _read_channel_slab(
    tif: tifffile.TiffFile,
    channel: int,
    channels: int,
    z0: int,
    z1: int,
    info: TiffZcyxInfo,
) -> np.ndarray:
    if z1 <= z0:
        return np.empty((0, info.y_size, info.x_size), dtype=np.float32)
    page_indices = [z * channels + channel for z in range(z0, z1)]
    slab = tif.asarray(key=page_indices, maxworkers=min(8, len(page_indices))).reshape(z1 - z0, info.y_size, info.x_size)
    return slab.astype(np.float32, copy=False)


def _read_zarr_channel_slab(
    source: ZarrZcyxSource,
    channel: int,
    z0: int,
    z1: int,
) -> np.ndarray:
    info = source.info
    if z1 <= z0:
        return np.empty((0, info.y_size, info.x_size), dtype=np.float32)
    t_index, z_index, c_index, y_index, x_index = source.indices_tzcyx
    selection: list[int | slice] = [slice(None)] * source.array.ndim
    if t_index is not None:
        if info.time_index is None:
            raise ValueError("time_index is required for T-axis OME-Zarr arrays")
        selection[t_index] = info.time_index
    selection[z_index] = slice(z0, z1)
    selection[c_index] = channel
    selection[y_index] = slice(None)
    selection[x_index] = slice(None)
    data = np.asarray(source.array[tuple(selection)], dtype=np.float32)
    axes_after = _remaining_axes_after_integer_selection(source.array.ndim, selection)
    return _move_selected_axes(data, axes_after, (z_index, y_index, x_index))


def _read_zarr_z_plane_all_channels(
    source: ZarrZcyxSource,
    z: int,
    *,
    downsample: int,
) -> np.ndarray:
    info = source.info
    t_index, z_index, c_index, y_index, x_index = source.indices_tzcyx
    selection: list[int | slice] = [slice(None)] * source.array.ndim
    if t_index is not None:
        if info.time_index is None:
            raise ValueError("time_index is required for T-axis OME-Zarr arrays")
        selection[t_index] = info.time_index
    selection[z_index] = z
    selection[c_index] = slice(None)
    selection[y_index] = slice(None, None, downsample)
    selection[x_index] = slice(None, None, downsample)
    data = np.asarray(source.array[tuple(selection)], dtype=np.float32)
    axes_after = _remaining_axes_after_integer_selection(source.array.ndim, selection)
    return _move_selected_axes(data, axes_after, (c_index, y_index, x_index))


def _remaining_axes_after_integer_selection(ndim: int, selection: list[int | slice]) -> tuple[int, ...]:
    return tuple(axis for axis in range(ndim) if not isinstance(selection[axis], int))


def _move_selected_axes(data: np.ndarray, axes_after: tuple[int, ...], target_axes: tuple[int, ...]) -> np.ndarray:
    current_positions = [axes_after.index(axis) for axis in target_axes]
    return np.moveaxis(data, current_positions, range(len(target_axes)))


def _write_zcyx_chunk(
    tif: tifffile.TiffWriter,
    chunk_zcyx: np.ndarray,
    *,
    bit_depth: int,
    compression: int | str | None,
) -> None:
    data = _convert_output(chunk_zcyx, bit_depth)
    for z_index in range(data.shape[0]):
        for channel in range(data.shape[1]):
            tif.write(data[z_index, channel], photometric="minisblack", compression=compression)


def _output_zarr_chunks(info: ZarrZcyxInfo, chunk_depth: int) -> tuple[int, int, int, int]:
    return (
        max(1, min(chunk_depth, info.z_size)),
        max(1, min(1, info.channels)),
        max(1, min(info.chunks_zcyx[2], info.y_size)),
        max(1, min(info.chunks_zcyx[3], info.x_size)),
    )


def _output_dtype(bit_depth: int) -> np.dtype:
    if bit_depth == 16:
        return np.dtype(np.uint16)
    if bit_depth == 32:
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported bit depth {bit_depth}; expected 16 or 32")


def _convert_output(data: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 16:
        if data.dtype == np.uint16:
            return data
        return np.clip(data, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    if bit_depth == 32:
        if data.dtype == np.float32:
            return data
        return data.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported bit depth {bit_depth}; expected 16 or 32")


def _write_mip_png(path: Path, mip_cyx: np.ndarray) -> None:
    from PIL import Image

    if mip_cyx.shape[0] == 1:
        image = _scale_uint8(mip_cyx[0])
    elif mip_cyx.shape[0] == 3:
        image = np.stack([_scale_uint8(mip_cyx[channel]) for channel in range(3)], axis=-1)
    else:
        image = _scale_uint8(np.max(mip_cyx, axis=0))
    Image.fromarray(image).save(path)


def _rotation_zyx(angles: np.ndarray) -> np.ndarray:
    rz, ry, rx = [float(value) for value in angles]
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    rot_z = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_x = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a.astype(np.float64, copy=False) - float(np.mean(a))
    b0 = b.astype(np.float64, copy=False) - float(np.mean(b))
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0:
        return 0.0
    return float(np.sum(a0 * b0) / denom)
