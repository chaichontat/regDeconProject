from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from .flatfield import FlatfieldProfile
from .previews import write_max_projection_png
from .stitching import Index3, StitchTile, fuse_tiles, prepare_fusion_flatfield


@dataclass(frozen=True)
class MosaicTileGeometry:
    index_zyx: Index3
    shape_zyx: Index3


@dataclass(frozen=True)
class StreamedMosaicStats:
    output_zarr: str | None
    progress_path: str | None
    preview_path: str | None
    output_shape_zcyx: tuple[int, int, int, int]
    output_origin_zyx: tuple[int, int, int]
    output_dtype: str
    z_start: int
    z_stop: int
    chunk_depth: int
    max_chunks: int | None
    preview_downsample: int
    total_chunks: int
    chunks_written: int
    chunks_skipped: int
    read_seconds: float
    fuse_seconds: float
    write_seconds: float
    total_seconds: float
    benchmark_only: bool


def write_streamed_mosaic_zarr(
    output_zarr: Path | None,
    *,
    source_label: str,
    tile_geometries: tuple[MosaicTileGeometry, ...],
    z_size: int,
    output_channels: tuple[int, ...],
    origins_zyx: dict[Index3, tuple[float, float, float]],
    overlap_zyx: Index3,
    read_tiles: Callable[[int, tuple[int, ...]], list[StitchTile]],
    z_start: int,
    z_stop: int | None,
    chunk_depth: int,
    bit_depth: int,
    zarr_attrs: dict[str, object],
    attr_keys_to_validate: tuple[str, ...],
    max_chunks: int | None = None,
    metadata_path: Path | None = None,
    benchmark_only: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    progress_path: Path | None = None,
    preview_path: Path | None = None,
    preview_downsample: int = 4,
    flatfield_profile: FlatfieldProfile | None = None,
    fuse_tiles_fn: Callable[..., tuple[np.ndarray, np.ndarray]] = fuse_tiles,
    log: Callable[[str], None] | None = None,
) -> StreamedMosaicStats:
    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")
    if bit_depth not in (16, 32):
        raise ValueError("bit_depth must be 16 or 32")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive when provided")
    if benchmark_only and resume:
        raise ValueError("resume is only valid when writing an output Zarr")
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot be combined")
    if preview_downsample <= 0:
        raise ValueError("preview_downsample must be positive")
    if not output_channels:
        raise ValueError("output_channels must not be empty")
    _validate_unique_channels(output_channels)
    _validate_zero_z_origins(origins_zyx, source_label=source_label)

    stop = validate_z_range(z_start=z_start, z_stop=z_stop, z_size=z_size)
    z_count = stop - z_start
    dtype = np.uint16 if bit_depth == 16 else np.float32
    output_origin_zyx, output_shape_yx = output_bounds_from_origins(tile_geometries, origins_zyx)
    output_shape_zcyx = (z_count, len(output_channels), output_shape_yx[0], output_shape_yx[1])
    output_bytes = array_nbytes(output_shape_zcyx, dtype)
    completed_chunks: set[tuple[int, int]] = set()

    if log is not None:
        log(
            f"{source_label}: output_shape_zcyx={output_shape_zcyx} "
            f"dtype={np.dtype(dtype)} chunk_depth={chunk_depth} benchmark_only={benchmark_only}"
        )

    if benchmark_only:
        output = None
        if progress_path is None and metadata_path is not None:
            progress_path = metadata_path.with_suffix(".progress.jsonl")
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text("")
    else:
        if output_zarr is None:
            raise ValueError("output_zarr is required unless benchmark_only is enabled")
        output_zarr.parent.mkdir(parents=True, exist_ok=True)
        if progress_path is None:
            progress_path = output_zarr.with_suffix(".progress.jsonl")
        if resume:
            if not output_zarr.exists():
                raise ValueError(f"Cannot resume {output_zarr}; output Zarr is missing")
            require_resume_progress(output_zarr, progress_path)
            output = zarr.open_array(output_zarr, mode="r+")
            if tuple(int(v) for v in output.shape) != output_shape_zcyx:
                raise ValueError(f"Existing output shape {output.shape} does not match expected {output_shape_zcyx}")
            if np.dtype(output.dtype) != np.dtype(dtype):
                raise ValueError(f"Existing output dtype {output.dtype} does not match expected {np.dtype(dtype)}")
            validate_zarr_attrs(output.attrs.asdict(), zarr_attrs, attr_keys_to_validate)
            completed_chunks = read_completed_chunks(
                progress_path,
                output_shape_zcyx=output_shape_zcyx,
                output_channels=output_channels,
                chunk_depth=chunk_depth,
            )
        else:
            if output_zarr.exists() and not overwrite:
                raise ValueError(f"{output_zarr} already exists; pass overwrite=True or resume=True")
            check_output_disk_space(output_zarr, output_bytes)
            output = zarr.open_array(
                output_zarr,
                mode="w",
                shape=output_shape_zcyx,
                chunks=(min(chunk_depth, z_count), 1, min(512, output_shape_yx[0]), min(512, output_shape_yx[1])),
                dtype=dtype,
            )
            output.attrs.update(zarr_attrs)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text("")

    read_seconds = 0.0
    fuse_seconds = 0.0
    write_seconds = 0.0
    chunks_written = 0
    chunks_skipped = 0
    preview_max_cyx: np.ndarray | None = None
    total_chunks = math.ceil(z_count / chunk_depth)
    tile_shape_zyx = _shared_tile_shape(tile_geometries)
    fusion_flatfield = prepare_fusion_flatfield(
        None if flatfield_profile is None else flatfield_profile.flatfield_yx,
        tile_shape_zyx=tile_shape_zyx,
    )
    total_start = time.perf_counter()
    for chunk_z0 in range(z_start, stop, chunk_depth):
        if max_chunks is not None and chunks_written >= max_chunks:
            break
        chunk_z1 = min(chunk_z0 + chunk_depth, stop)
        if (chunk_z0, chunk_z1) in completed_chunks:
            chunks_skipped += 1
            if log is not None:
                log(f"{source_label}: skip completed z=[{chunk_z0}, {chunk_z1})")
            continue
        z_indices = tuple(range(chunk_z0, chunk_z1))
        out_z0 = chunk_z0 - z_start
        if log is not None:
            log(f"{source_label}: chunk z=[{chunk_z0}, {chunk_z1}) channels={output_channels}")

        for output_channel_index, output_channel in enumerate(output_channels):
            start = time.perf_counter()
            tiles = read_tiles(output_channel, z_indices)
            read_elapsed = time.perf_counter() - start
            read_seconds += read_elapsed

            start = time.perf_counter()
            if fusion_flatfield is None:
                fused, _ = fuse_tiles_fn(tiles, origins_zyx=origins_zyx, overlap_zyx=overlap_zyx)
            else:
                fused, _ = fuse_tiles_fn(
                    tiles,
                    origins_zyx=origins_zyx,
                    overlap_zyx=overlap_zyx,
                    flatfield_yx=fusion_flatfield,
                )
            fuse_elapsed = time.perf_counter() - start
            fuse_seconds += fuse_elapsed
            if fused.shape[0] != len(z_indices):
                raise ValueError(f"Expected fused chunk depth {len(z_indices)}, got {fused.shape[0]}")
            if fused.shape[2:] != output_shape_yx:
                raise ValueError(f"Expected fused YX shape {output_shape_yx}, got {fused.shape[2:]}")
            if preview_path is not None:
                chunk_preview = fused[:, 0, ::preview_downsample, ::preview_downsample].max(axis=0)
                if preview_max_cyx is None:
                    preview_max_cyx = np.zeros((len(output_channels), *chunk_preview.shape), dtype=np.float32)
                preview_max_cyx[output_channel_index] = np.maximum(
                    preview_max_cyx[output_channel_index],
                    chunk_preview,
                )
            if bit_depth == 16:
                data = np.clip(fused, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
            else:
                data = fused.astype(np.float32, copy=False)

            if output is not None:
                start = time.perf_counter()
                output[out_z0 : out_z0 + data.shape[0], output_channel_index : output_channel_index + 1] = data
                write_elapsed = time.perf_counter() - start
                write_seconds += write_elapsed
            else:
                write_elapsed = 0.0
            if log is not None:
                log(
                    f"{source_label}: channel={output_channel} z=[{chunk_z0}, {chunk_z1}) "
                    f"read={read_elapsed:.2f}s fuse={fuse_elapsed:.2f}s write={write_elapsed:.2f}s"
                )

        chunks_written += 1
        if progress_path is not None:
            append_completed_chunk(
                progress_path,
                z_start=chunk_z0,
                z_stop=chunk_z1,
                output_shape_zcyx=output_shape_zcyx,
                output_channels=output_channels,
                chunk_depth=chunk_depth,
                wrote_output=output is not None,
            )
    if preview_path is not None and preview_max_cyx is not None:
        write_max_projection_png(preview_path, preview_max_cyx[np.newaxis])

    return StreamedMosaicStats(
        output_zarr=str(output_zarr) if output_zarr is not None else None,
        progress_path=str(progress_path) if progress_path is not None else None,
        preview_path=str(preview_path) if preview_path is not None else None,
        output_shape_zcyx=tuple(int(v) for v in output_shape_zcyx),
        output_origin_zyx=output_origin_zyx,
        output_dtype=str(np.dtype(dtype)),
        z_start=z_start,
        z_stop=stop,
        chunk_depth=chunk_depth,
        max_chunks=max_chunks,
        preview_downsample=preview_downsample,
        total_chunks=total_chunks,
        chunks_written=chunks_written,
        chunks_skipped=chunks_skipped,
        read_seconds=read_seconds,
        fuse_seconds=fuse_seconds,
        write_seconds=write_seconds,
        total_seconds=time.perf_counter() - total_start,
        benchmark_only=benchmark_only,
    )


def output_bounds_from_origins(
    tiles: tuple[MosaicTileGeometry, ...],
    origins_zyx: dict[Index3, tuple[float, float, float]],
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    origin_values = []
    extents = []
    for tile in tiles:
        origin = origins_zyx[tile.index_zyx]
        origin_values.append(origin)
        extents.append(tuple(origin[axis] + tile.shape_zyx[axis] for axis in range(3)))
    min_origin = np.floor(np.asarray(origin_values, dtype=np.float32).min(axis=0)).astype(np.int64)
    max_extent = np.ceil(np.asarray(extents, dtype=np.float32).max(axis=0)).astype(np.int64)
    output_shape = max_extent - min_origin
    return (
        tuple(int(v) for v in min_origin),
        (int(output_shape[1]), int(output_shape[2])),
    )


def validate_z_range(*, z_start: int, z_stop: int | None, z_size: int) -> int:
    stop = z_size if z_stop is None else z_stop
    if z_start < 0:
        raise ValueError(f"z_start {z_start} is outside available Z range [0, {z_size})")
    if stop > z_size:
        raise ValueError(f"z_stop {stop} is outside available Z range [0, {z_size}]")
    if stop <= z_start:
        raise ValueError("Requested z range is empty")
    return stop


def validate_zarr_attrs(actual: dict[str, object], expected: dict[str, object], checked_keys: tuple[str, ...]) -> None:
    for key in checked_keys:
        if actual.get(key) != expected[key]:
            raise ValueError(f"Existing output Zarr attribute {key!r} does not match this stitch run")


def read_completed_chunks(
    path: Path,
    *,
    output_shape_zcyx: tuple[int, int, int, int],
    output_channels: tuple[int, ...],
    chunk_depth: int,
) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if tuple(record["output_shape_zcyx"]) != output_shape_zcyx:
            raise ValueError(f"Progress record {line_number} has a different output shape")
        if tuple(record["output_channels"]) != output_channels:
            raise ValueError(f"Progress record {line_number} has different output channels")
        if int(record["chunk_depth"]) != chunk_depth:
            raise ValueError(f"Progress record {line_number} has a different chunk depth")
        if not bool(record.get("wrote_output", True)):
            raise ValueError(f"Progress record {line_number} was written by a benchmark-only run")
        completed.add((int(record["z_start"]), int(record["z_stop"])))
    return completed


def require_resume_progress(output_zarr: Path, progress_path: Path) -> None:
    if not progress_path.exists():
        raise ValueError(f"Cannot resume {output_zarr}; progress log is missing: {progress_path}")


def append_completed_chunk(
    path: Path,
    *,
    z_start: int,
    z_stop: int,
    output_shape_zcyx: tuple[int, int, int, int],
    output_channels: tuple[int, ...],
    chunk_depth: int,
    wrote_output: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "z_start": int(z_start),
        "z_stop": int(z_stop),
        "output_shape_zcyx": list(output_shape_zcyx),
        "output_channels": list(output_channels),
        "chunk_depth": int(chunk_depth),
        "wrote_output": bool(wrote_output),
    }
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def array_nbytes(shape: tuple[int, ...], dtype: type[np.generic] | np.dtype) -> int:
    return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)


def check_output_disk_space(output_zarr: Path, required_bytes: int) -> None:
    check_path = output_zarr.parent if output_zarr.parent.exists() else output_zarr.parent.parent
    free_bytes = shutil.disk_usage(check_path).free
    if free_bytes < required_bytes:
        raise ValueError(
            f"Insufficient free space for {output_zarr}: "
            f"need at least {required_bytes} bytes, available {free_bytes} bytes"
        )


def _shared_tile_shape(tiles: tuple[MosaicTileGeometry, ...]) -> Index3:
    shapes = {tile.shape_zyx for tile in tiles}
    if len(shapes) != 1:
        raise ValueError(f"Streamed fusion requires uniform tile shapes, got {sorted(shapes)}")
    return next(iter(shapes))


def _validate_unique_channels(channels: tuple[int, ...]) -> None:
    if len(set(channels)) != len(channels):
        raise ValueError(f"output_channels must be unique, got {list(channels)}")


def _validate_zero_z_origins(origins_zyx: dict[Index3, tuple[float, float, float]], *, source_label: str) -> None:
    bad_indices = [index for index, origin in sorted(origins_zyx.items()) if not np.isclose(origin[0], 0.0)]
    if bad_indices:
        raise ValueError(f"{source_label} chunked writer requires zero Z origins; nonzero Z origins for tiles: {bad_indices}")
