from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from aicspylibczi import CziFile

from .flatfield import FlatfieldProfile, flatfield_metadata
from .mosaic_streaming import (
    MosaicTileGeometry,
    append_completed_chunk,
    array_nbytes,
    check_output_disk_space,
    output_bounds_from_origins,
    read_completed_chunks,
    require_resume_progress,
    validate_z_range,
    validate_zarr_attrs,
    write_streamed_mosaic_zarr,
)
from .stitching import PairShift, StitchTile, estimate_tile_placement, fuse_tiles


@dataclass(frozen=True)
class CziMosaicTile:
    index_zyx: tuple[int, int, int]
    mosaic_index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class CziStitchPlan:
    czi_path: str
    channel: int
    z_size: int
    tile_shape_zyx: tuple[int, int, int]
    output_shape_yx: tuple[int, int]
    overlap_zyx: tuple[int, int, int]
    tile_count: int
    tiles: tuple[CziMosaicTile, ...]
    nominal_origins_zyx: dict[tuple[int, int, int], tuple[float, float, float]]


@dataclass(frozen=True)
class CziPlacement:
    plan: CziStitchPlan
    reference_z_indices: tuple[int, ...]
    origins_zyx: dict[tuple[int, int, int], tuple[float, float, float]]
    pair_shifts: tuple[PairShift, ...]
    fallback_min_ncc: float
    estimate_seconds: float
    coarse_max_size: int = 1024
    fine_upsample_factor: int | None = None


@dataclass(frozen=True)
class CziStitchRun:
    source_czi: str
    output_zarr: str | None
    metadata_path: str | None
    progress_path: str | None
    preview_path: str | None
    placement: CziPlacement
    placement_estimated_this_run: bool
    output_channels: tuple[int, ...]
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
    flatfield_profile: FlatfieldProfile | None = None


def build_czi_stitch_plan(
    czi_path: Path,
    *,
    channel: int,
    overlap_zyx: tuple[int, int, int],
    z_sample: int = 0,
) -> CziStitchPlan:
    czi = CziFile(czi_path)
    dims = czi.get_dims_shape()[0]
    z_size = int(dims["Z"][1] - dims["Z"][0])
    _validate_czi_channel_indices((channel,), dims=dims)
    _validate_czi_z_index(z_sample, z_size=z_size, name="z_sample")
    boxes = czi.get_all_mosaic_tile_bounding_boxes(C=channel, Z=z_sample, T=0)
    if not boxes:
        raise ValueError("CZI does not expose mosaic tile bounding boxes")

    xs = sorted({int(box.x) for box in boxes.values()})
    ys = sorted({int(box.y) for box in boxes.values()})
    min_x = min(xs)
    min_y = min(ys)
    tiles = []
    nominal_origins = {}
    for info, box in sorted(boxes.items(), key=lambda item: (item[1].y, item[1].x)):
        mosaic_index = int(info.dimension_coordinates["M"])
        index = (0, ys.index(int(box.y)), xs.index(int(box.x)))
        tile = CziMosaicTile(
            index_zyx=index,
            mosaic_index=mosaic_index,
            x=int(box.x),
            y=int(box.y),
            width=int(box.w),
            height=int(box.h),
        )
        tiles.append(tile)
        nominal_origins[index] = (0.0, float(tile.y - min_y), float(tile.x - min_x))

    tile_widths = {tile.width for tile in tiles}
    tile_heights = {tile.height for tile in tiles}
    if len(tile_widths) != 1 or len(tile_heights) != 1:
        raise ValueError("All mosaic tiles must have matching X/Y shape")
    tile_shape = (1, next(iter(tile_heights)), next(iter(tile_widths)))
    output_shape_yx = (
        max(tile.y - min_y + tile.height for tile in tiles),
        max(tile.x - min_x + tile.width for tile in tiles),
    )
    return CziStitchPlan(
        czi_path=str(czi_path),
        channel=channel,
        z_size=z_size,
        tile_shape_zyx=tile_shape,
        output_shape_yx=tuple(int(v) for v in output_shape_yx),
        overlap_zyx=overlap_zyx,
        tile_count=len(tiles),
        tiles=tuple(tiles),
        nominal_origins_zyx=nominal_origins,
    )


def estimate_czi_stitch_placement(
    czi_path: Path,
    *,
    channel: int,
    reference_z_start: int,
    reference_z_count: int,
    overlap_zyx: tuple[int, int, int],
    fallback_min_ncc: float,
    coarse_max_size: int = 1024,
    fine_upsample_factor: int = 1,
    force_zero_z_origin: bool = True,
) -> CziPlacement:
    if reference_z_count <= 0:
        raise ValueError("reference_z_count must be positive")
    if coarse_max_size <= 0:
        raise ValueError("coarse_max_size must be positive")
    plan = build_czi_stitch_plan(czi_path, channel=channel, overlap_zyx=overlap_zyx, z_sample=reference_z_start)
    reference_z_stop = _validate_czi_z_range(
        z_start=reference_z_start,
        z_stop=reference_z_start + reference_z_count,
        z_size=plan.z_size,
    )
    z_indices = tuple(range(reference_z_start, reference_z_stop))

    czi = CziFile(czi_path)
    start = time.perf_counter()
    tiles = _read_czi_tiles(czi, plan.tiles, channel=channel, z_indices=z_indices)
    placement = estimate_tile_placement(
        tiles,
        overlap_zyx=overlap_zyx,
        reference_channel=0,
        coarse_max_size=coarse_max_size,
        fine_upsample_factor=fine_upsample_factor,
        nominal_origins_zyx=plan.nominal_origins_zyx,
        nominal_fallback_min_ncc=fallback_min_ncc,
    )
    origins = placement.origins_zyx
    if force_zero_z_origin:
        origins = {index: (0.0, origin[1], origin[2]) for index, origin in origins.items()}
    return CziPlacement(
        plan=plan,
        reference_z_indices=z_indices,
        origins_zyx=origins,
        pair_shifts=placement.pair_shifts,
        fallback_min_ncc=fallback_min_ncc,
        coarse_max_size=coarse_max_size,
        fine_upsample_factor=fine_upsample_factor,
        estimate_seconds=time.perf_counter() - start,
    )


def write_czi_stitched_zarr(
    czi_path: Path,
    output_zarr: Path | None,
    *,
    channel: int,
    output_channels: tuple[int, ...] | None = None,
    placement: CziPlacement | None = None,
    placement_output_path: Path | None = None,
    z_start: int,
    z_stop: int | None,
    chunk_depth: int,
    reference_z_start: int,
    reference_z_count: int,
    overlap_zyx: tuple[int, int, int],
    fallback_min_ncc: float,
    coarse_max_size: int = 1024,
    fine_upsample_factor: int = 1,
    bit_depth: int = 16,
    max_chunks: int | None = None,
    metadata_path: Path | None = None,
    benchmark_only: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    progress_path: Path | None = None,
    preview_path: Path | None = None,
    preview_downsample: int = 4,
    flatfield_profile: FlatfieldProfile | None = None,
) -> CziStitchRun:
    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")
    if bit_depth not in (16, 32):
        raise ValueError("bit_depth must be 16 or 32")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive when provided")
    requested_output_channels = None if output_channels is None else tuple(int(value) for value in output_channels)
    if requested_output_channels is not None:
        if not requested_output_channels:
            raise ValueError("output_channels must not be empty")
        _validate_unique_channels(requested_output_channels)
        validate_czi_output_channels(czi_path, requested_output_channels)
    if benchmark_only and resume:
        raise ValueError("resume is only valid when writing an output Zarr")
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot be combined")
    if preview_downsample <= 0:
        raise ValueError("preview_downsample must be positive")
    placement_estimated_this_run = placement is None
    if placement is None:
        placement = estimate_czi_stitch_placement(
            czi_path,
            channel=channel,
            reference_z_start=reference_z_start,
            reference_z_count=reference_z_count,
            overlap_zyx=overlap_zyx,
            fallback_min_ncc=fallback_min_ncc,
            coarse_max_size=coarse_max_size,
            fine_upsample_factor=fine_upsample_factor,
            force_zero_z_origin=True,
        )
    else:
        validate_czi_placement_for_file(czi_path, placement)
    _validate_zero_z_origins(placement)
    channels = _resolve_czi_output_channels(
        output_channels=requested_output_channels,
        placement=placement,
    )
    _validate_unique_channels(channels)
    validate_czi_output_channels(czi_path, channels)
    validate_czi_output_channel_mosaic_geometry(czi_path, placement, channels)
    if placement_output_path is not None:
        save_czi_placement(placement_output_path, placement)

    plan = placement.plan
    output_origin_zyx, _output_shape_yx = _output_bounds_from_origins(plan.tiles, placement.origins_zyx)
    stop = _validate_czi_z_range(z_start=z_start, z_stop=z_stop, z_size=plan.z_size)
    czi: CziFile | None = None

    def read_chunk(output_channel: int, z_indices: tuple[int, ...]) -> list[StitchTile]:
        nonlocal czi
        if czi is None:
            czi = CziFile(czi_path)
        return _read_czi_tiles(
            czi,
            plan.tiles,
            channel=output_channel,
            z_indices=z_indices,
        )

    stats = write_streamed_mosaic_zarr(
        output_zarr,
        source_label="CZI stitch",
        tile_geometries=_czi_tile_geometries(plan.tiles),
        z_size=plan.z_size,
        output_channels=channels,
        origins_zyx=placement.origins_zyx,
        overlap_zyx=plan.overlap_zyx,
        read_tiles=read_chunk,
        z_start=z_start,
        z_stop=z_stop,
        chunk_depth=chunk_depth,
        bit_depth=bit_depth,
        zarr_attrs=_czi_zarr_attrs(
            placement,
            channels,
            output_origin_zyx,
            source_czi=czi_path,
            z_start=z_start,
            z_stop=stop,
            flatfield_profile=flatfield_profile,
        ),
        attr_keys_to_validate=_czi_zarr_attr_keys(),
        max_chunks=max_chunks,
        metadata_path=metadata_path,
        benchmark_only=benchmark_only,
        resume=resume,
        overwrite=overwrite,
        progress_path=progress_path,
        preview_path=preview_path,
        preview_downsample=preview_downsample,
        flatfield_profile=flatfield_profile,
        fuse_tiles_fn=fuse_tiles,
    )

    run = CziStitchRun(
        source_czi=str(czi_path),
        output_zarr=stats.output_zarr,
        metadata_path=str(metadata_path) if metadata_path is not None else None,
        progress_path=stats.progress_path,
        preview_path=stats.preview_path,
        placement=placement,
        placement_estimated_this_run=placement_estimated_this_run,
        output_channels=channels,
        output_shape_zcyx=stats.output_shape_zcyx,
        output_origin_zyx=stats.output_origin_zyx,
        output_dtype=stats.output_dtype,
        z_start=stats.z_start,
        z_stop=stats.z_stop,
        chunk_depth=stats.chunk_depth,
        max_chunks=stats.max_chunks,
        preview_downsample=stats.preview_downsample,
        total_chunks=stats.total_chunks,
        chunks_written=stats.chunks_written,
        chunks_skipped=stats.chunks_skipped,
        read_seconds=stats.read_seconds,
        fuse_seconds=stats.fuse_seconds,
        write_seconds=stats.write_seconds,
        total_seconds=stats.total_seconds,
        benchmark_only=stats.benchmark_only,
        flatfield_profile=flatfield_profile,
    )
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(_jsonable_run(run), indent=2))
    return run


def save_czi_placement(path: Path | str, placement: CziPlacement) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable_placement(placement), indent=2))


def load_czi_placement(path: Path | str) -> CziPlacement:
    path = Path(path)
    return _placement_from_jsonable(json.loads(path.read_text()))


def summarize_czi_placement(placement: CziPlacement) -> dict[str, object]:
    nccs = [pair.ncc for pair in placement.pair_shifts]
    return {
        "tile_count": int(placement.plan.tile_count),
        "pair_count": len(placement.pair_shifts),
        "fallback_pair_count": int(sum(pair.ncc < placement.fallback_min_ncc for pair in placement.pair_shifts)),
        "fallback_min_ncc": float(placement.fallback_min_ncc),
        "coarse_max_size": int(placement.coarse_max_size),
        "fine_upsample_factor": placement.fine_upsample_factor,
        "ncc_min": float(min(nccs)) if nccs else None,
        "ncc_median": float(np.median(nccs)) if nccs else None,
        "ncc_max": float(max(nccs)) if nccs else None,
        "estimate_seconds": float(placement.estimate_seconds),
        "reference_z_indices": list(placement.reference_z_indices),
    }


def validate_czi_output_channels(czi_path: Path, channels: tuple[int, ...]) -> None:
    dims = CziFile(czi_path).get_dims_shape()[0]
    _validate_czi_channel_indices(channels, dims=dims)


def validate_czi_output_channel_count(czi_path: Path, *, start_channel: int, channel_count: int) -> None:
    if channel_count <= 0:
        raise ValueError("channels must be positive")
    channels = tuple(range(start_channel, start_channel + channel_count))
    validate_czi_output_channels(czi_path, channels)


def validate_czi_output_channel_mosaic_geometry(
    czi_path: Path,
    placement: CziPlacement,
    channels: tuple[int, ...],
) -> None:
    reference_z = placement.reference_z_indices[0] if placement.reference_z_indices else 0
    for channel in channels:
        current_plan = build_czi_stitch_plan(
            czi_path,
            channel=channel,
            overlap_zyx=placement.plan.overlap_zyx,
            z_sample=reference_z,
        )
        _validate_matching_czi_mosaic_geometry(placement.plan, current_plan)


def _validate_czi_z_index(z_index: int, *, z_size: int, name: str) -> None:
    if z_index < 0 or z_index >= z_size:
        raise ValueError(f"{name} {z_index} is outside available Z range [0, {z_size})")


def _validate_czi_z_range(*, z_start: int, z_stop: int | None, z_size: int) -> int:
    return validate_z_range(z_start=z_start, z_stop=z_stop, z_size=z_size)


def _resolve_czi_output_channels(
    *,
    output_channels: tuple[int, ...] | None,
    placement: CziPlacement,
) -> tuple[int, ...]:
    if output_channels is not None:
        if not output_channels:
            raise ValueError("output_channels must not be empty")
        return tuple(int(value) for value in output_channels)
    return (int(placement.plan.channel),)


def _validate_czi_channel_indices(channels: tuple[int, ...], *, dims: dict[str, tuple[int, int]]) -> None:
    start, stop = dims.get("C", (0, 1))
    for channel in channels:
        if channel < start or channel >= stop:
            raise ValueError(f"CZI channel {channel} is outside available channel range [{start}, {stop})")


def _validate_unique_channels(channels: tuple[int, ...]) -> None:
    if len(set(channels)) != len(channels):
        raise ValueError(f"output_channels must be unique, got {list(channels)}")


def _array_nbytes(shape: tuple[int, ...], dtype: type[np.generic] | np.dtype) -> int:
    return array_nbytes(shape, dtype)


def _check_output_disk_space(output_zarr: Path, required_bytes: int) -> None:
    check_output_disk_space(output_zarr, required_bytes)


def _validate_zero_z_origins(placement: CziPlacement) -> None:
    bad_indices = [index for index, origin in sorted(placement.origins_zyx.items()) if not np.isclose(origin[0], 0.0)]
    if bad_indices:
        raise ValueError(f"CZI chunked writer requires zero Z origins; nonzero Z origins for tiles: {bad_indices}")


def summarize_czi_stitch_plan(plan: CziStitchPlan, *, channels: int = 1, bit_depth: int = 16) -> dict[str, object]:
    if channels <= 0:
        raise ValueError("channels must be positive")
    if bit_depth not in (16, 32):
        raise ValueError("bit_depth must be 16 or 32")
    dtype = np.uint16 if bit_depth == 16 else np.float32
    shape = (plan.z_size, channels, plan.output_shape_yx[0], plan.output_shape_yx[1])
    return {
        "czi_path": plan.czi_path,
        "channel": int(plan.channel),
        "tile_count": int(plan.tile_count),
        "z_size": int(plan.z_size),
        "tile_shape_zyx": list(plan.tile_shape_zyx),
        "nominal_output_shape_zcyx": list(shape),
        "nominal_output_bytes": int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize),
        "overlap_zyx": list(plan.overlap_zyx),
        "grid_indices_zyx": [list(tile.index_zyx) for tile in plan.tiles],
    }


def save_czi_placement_origins_csv(path: Path | str, placement: CziPlacement) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index_z", "index_y", "index_x", "origin_z", "origin_y", "origin_x"])
        for index, origin in sorted(placement.origins_zyx.items()):
            writer.writerow([*index, *origin])


def load_czi_placement_origins_csv(path: Path | str) -> dict[tuple[int, int, int], tuple[float, float, float]]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"index_z", "index_y", "index_x", "origin_z", "origin_y", "origin_x"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Origin CSV is missing required columns: {sorted(missing)}")
        origins = {}
        for row_number, row in enumerate(reader, start=2):
            index = (int(row["index_z"]), int(row["index_y"]), int(row["index_x"]))
            if index in origins:
                raise ValueError(f"Origin CSV row {row_number} duplicates tile index {index}")
            origins[index] = (float(row["origin_z"]), float(row["origin_y"]), float(row["origin_x"]))
    if not origins:
        raise ValueError("Origin CSV must contain at least one row")
    return origins


def replace_czi_placement_origins(
    placement: CziPlacement,
    origins_zyx: dict[tuple[int, int, int], tuple[float, float, float]],
) -> CziPlacement:
    expected = set(placement.origins_zyx)
    provided = set(origins_zyx)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise ValueError(f"Origin CSV indices do not match placement; missing={missing}, extra={extra}")
    return replace(placement, origins_zyx={index: tuple(origin) for index, origin in origins_zyx.items()})


def validate_czi_placement_for_file(czi_path: Path, placement: CziPlacement) -> None:
    reference_z = placement.reference_z_indices[0] if placement.reference_z_indices else 0
    current_plan = build_czi_stitch_plan(
        czi_path,
        channel=placement.plan.channel,
        overlap_zyx=placement.plan.overlap_zyx,
        z_sample=reference_z,
    )
    _validate_matching_czi_plans(placement.plan, current_plan)


def _validate_matching_czi_plans(expected: CziStitchPlan, actual: CziStitchPlan) -> None:
    if _czi_plan_signature(expected) != _czi_plan_signature(actual):
        raise ValueError("CZI placement does not match the target CZI mosaic geometry")


def _validate_matching_czi_mosaic_geometry(expected: CziStitchPlan, actual: CziStitchPlan) -> None:
    if _czi_plan_geometry_signature(expected) != _czi_plan_geometry_signature(actual):
        raise ValueError("CZI output channel mosaic geometry does not match the placement channel")


def _czi_plan_signature(plan: CziStitchPlan) -> tuple[object, ...]:
    return (
        int(plan.channel),
        *_czi_plan_geometry_signature(plan),
    )


def _czi_plan_geometry_signature(plan: CziStitchPlan) -> tuple[object, ...]:
    tiles = tuple(
        (tile.index_zyx, tile.mosaic_index, tile.x, tile.y, tile.width, tile.height)
        for tile in sorted(plan.tiles, key=lambda value: value.index_zyx)
    )
    return (
        int(plan.z_size),
        tuple(plan.tile_shape_zyx),
        tuple(plan.overlap_zyx),
        int(plan.tile_count),
        tiles,
    )


def _czi_zarr_attrs(
    placement: CziPlacement,
    output_channels: tuple[int, ...],
    output_origin_zyx: tuple[int, int, int],
    *,
    source_czi: Path | str | None = None,
    z_start: int,
    z_stop: int,
    flatfield_profile: FlatfieldProfile | None = None,
) -> dict[str, object]:
    return {
        "registration_fusion_schema": "czi_stitched_zcyx_v1",
        "axes": ["z", "c", "y", "x"],
        "source_czi": str(source_czi) if source_czi is not None else placement.plan.czi_path,
        "placement_channel": int(placement.plan.channel),
        "output_channels": list(output_channels),
        "source_z_start": int(z_start),
        "source_z_stop": int(z_stop),
        "output_origin_zyx": list(output_origin_zyx),
        "overlap_zyx": list(placement.plan.overlap_zyx),
        "tile_count": int(placement.plan.tile_count),
        "placement_reference_z_indices": list(placement.reference_z_indices),
        "placement_origin_records": _origin_records(placement.origins_zyx),
        "placement_summary": summarize_czi_placement(placement),
        "flatfield_correction": flatfield_metadata(flatfield_profile),
    }


def _validate_czi_zarr_attrs(actual: dict[str, object], expected: dict[str, object]) -> None:
    validate_zarr_attrs(actual, expected, _czi_zarr_attr_keys())


def _czi_zarr_attr_keys() -> tuple[str, ...]:
    return (
        "registration_fusion_schema",
        "axes",
        "source_czi",
        "placement_channel",
        "output_channels",
        "source_z_start",
        "source_z_stop",
        "output_origin_zyx",
        "overlap_zyx",
        "tile_count",
        "placement_reference_z_indices",
        "placement_origin_records",
        "flatfield_correction",
    )


def _read_completed_chunks(
    path: Path,
    *,
    output_shape_zcyx: tuple[int, int, int, int],
    output_channels: tuple[int, ...],
    chunk_depth: int,
) -> set[tuple[int, int]]:
    return read_completed_chunks(
        path,
        output_shape_zcyx=output_shape_zcyx,
        output_channels=output_channels,
        chunk_depth=chunk_depth,
    )


def _require_resume_progress(output_zarr: Path, progress_path: Path) -> None:
    require_resume_progress(output_zarr, progress_path)


def _append_completed_chunk(
    path: Path,
    *,
    z_start: int,
    z_stop: int,
    output_shape_zcyx: tuple[int, int, int, int],
    output_channels: tuple[int, ...],
    chunk_depth: int,
    wrote_output: bool = True,
) -> None:
    append_completed_chunk(
        path,
        z_start=z_start,
        z_stop=z_stop,
        output_shape_zcyx=output_shape_zcyx,
        output_channels=output_channels,
        chunk_depth=chunk_depth,
        wrote_output=wrote_output,
    )


def _read_czi_tiles(
    czi: CziFile,
    tiles: tuple[CziMosaicTile, ...],
    *,
    channel: int,
    z_indices: tuple[int, ...],
) -> list[StitchTile]:
    stitch_tiles_out = []
    for tile in tiles:
        planes = []
        for z in z_indices:
            plane = np.squeeze(czi.read_image(C=channel, Z=z, M=tile.mosaic_index)[0])
            if plane.ndim != 2:
                raise ValueError(f"Expected 2D plane for M={tile.mosaic_index} Z={z}, got shape {plane.shape}")
            planes.append(plane.astype(np.float32, copy=False))
        stitch_tiles_out.append(StitchTile(tile.index_zyx, np.stack(planes, axis=0)[:, np.newaxis]))
    return stitch_tiles_out


def _output_bounds_from_origins(
    tiles: tuple[CziMosaicTile, ...],
    origins_zyx: dict[tuple[int, int, int], tuple[float, float, float]],
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    return output_bounds_from_origins(_czi_tile_geometries(tiles), origins_zyx)


def _czi_tile_geometries(tiles: tuple[CziMosaicTile, ...]) -> tuple[MosaicTileGeometry, ...]:
    return tuple(MosaicTileGeometry(tile.index_zyx, (1, tile.height, tile.width)) for tile in tiles)


def _jsonable_run(run: CziStitchRun) -> dict[str, object]:
    placement = run.placement
    placement_summary = summarize_czi_placement(placement)
    output_bytes = int(np.prod(run.output_shape_zcyx, dtype=np.int64) * np.dtype(run.output_dtype).itemsize)
    placement_seconds_this_run = placement.estimate_seconds if run.placement_estimated_this_run else 0.0
    estimated = None
    if 0 < run.chunks_written < run.total_chunks:
        seconds_per_chunk = run.total_seconds / run.chunks_written
        estimated_processing_loop = seconds_per_chunk * run.total_chunks
        estimated = {
            "chunks_measured": int(run.chunks_written),
            "seconds_per_chunk": float(seconds_per_chunk),
            "estimated_processing_loop_seconds": float(estimated_processing_loop),
            "estimated_total_seconds": float(placement_seconds_this_run + estimated_processing_loop),
        }
    return {
        "source_czi": run.source_czi,
        "output_zarr": run.output_zarr,
        "progress_path": run.progress_path,
        "preview_path": run.preview_path,
        "output_channels": list(run.output_channels),
        "output_channel_mosaic_geometry_validation": {
            "status": "passed",
            "placement_channel": int(placement.plan.channel),
            "validated_output_channels": list(run.output_channels),
            "matched_fields": [
                "z_size",
                "tile_shape_zyx",
                "overlap_zyx",
                "tile_count",
                "tile_indices",
                "mosaic_indices",
                "tile_bounding_boxes",
            ],
        },
        "output_shape_zcyx": list(run.output_shape_zcyx),
        "output_origin_zyx": list(run.output_origin_zyx),
        "output_dtype": run.output_dtype,
        "benchmark_only": bool(run.benchmark_only),
        "flatfield_correction": flatfield_metadata(run.flatfield_profile),
        "estimated_output_bytes": output_bytes,
        "z_start": int(run.z_start),
        "z_stop": int(run.z_stop),
        "chunk_depth": int(run.chunk_depth),
        "max_chunks": None if run.max_chunks is None else int(run.max_chunks),
        "preview_downsample": int(run.preview_downsample),
        "total_chunks": int(run.total_chunks),
        "chunks_written": int(run.chunks_written),
        "chunks_skipped": int(run.chunks_skipped),
        "estimated_full_run": estimated,
        "timing_seconds": {
            "placement_estimate_original": float(placement.estimate_seconds),
            "placement_estimate_this_run": float(placement_seconds_this_run),
            "read": float(run.read_seconds),
            "fuse": float(run.fuse_seconds),
            "write": float(run.write_seconds),
            "total_processing_loop": float(run.total_seconds),
        },
        "plan": {
            **asdict(placement.plan),
            "tiles": [asdict(tile) for tile in placement.plan.tiles],
            "nominal_origins_zyx": {str(k): list(v) for k, v in placement.plan.nominal_origins_zyx.items()},
        },
        "placement": {
            "reference_z_indices": list(placement.reference_z_indices),
            "fallback_min_ncc": float(placement.fallback_min_ncc),
            "coarse_max_size": int(placement.coarse_max_size),
            "fine_upsample_factor": placement.fine_upsample_factor,
            "origins_zyx": {str(k): list(v) for k, v in placement.origins_zyx.items()},
            "origin_records": _origin_records(placement.origins_zyx),
            "pair_count": placement_summary["pair_count"],
            "fallback_pair_count": placement_summary["fallback_pair_count"],
            "ncc_min": placement_summary["ncc_min"],
            "ncc_median": placement_summary["ncc_median"],
            "ncc_max": placement_summary["ncc_max"],
            "pair_shifts": [
                {
                    "fixed": list(pair.fixed_index_zyx),
                    "moving": list(pair.moving_index_zyx),
                    "axis": int(pair.axis),
                    "origin_delta_zyx": list(pair.origin_delta_zyx),
                    "ncc": float(pair.ncc),
                }
                for pair in placement.pair_shifts
            ],
        },
    }


def _jsonable_placement(placement: CziPlacement) -> dict[str, object]:
    return {
        "version": 1,
        "plan": {
            "czi_path": placement.plan.czi_path,
            "channel": int(placement.plan.channel),
            "z_size": int(placement.plan.z_size),
            "tile_shape_zyx": list(placement.plan.tile_shape_zyx),
            "output_shape_yx": list(placement.plan.output_shape_yx),
            "overlap_zyx": list(placement.plan.overlap_zyx),
            "tile_count": int(placement.plan.tile_count),
            "tiles": [
                {
                    "index_zyx": list(tile.index_zyx),
                    "mosaic_index": int(tile.mosaic_index),
                    "x": int(tile.x),
                    "y": int(tile.y),
                    "width": int(tile.width),
                    "height": int(tile.height),
                }
                for tile in placement.plan.tiles
            ],
            "nominal_origin_records": _origin_records(placement.plan.nominal_origins_zyx),
        },
        "placement": {
            "reference_z_indices": list(placement.reference_z_indices),
            "origins": _origin_records(placement.origins_zyx),
            "pair_shifts": [
                {
                    "fixed_index_zyx": list(pair.fixed_index_zyx),
                    "moving_index_zyx": list(pair.moving_index_zyx),
                    "axis": int(pair.axis),
                    "origin_delta_zyx": list(pair.origin_delta_zyx),
                    "ncc": float(pair.ncc),
                }
                for pair in placement.pair_shifts
            ],
            "fallback_min_ncc": float(placement.fallback_min_ncc),
            "coarse_max_size": int(placement.coarse_max_size),
            "fine_upsample_factor": placement.fine_upsample_factor,
            "estimate_seconds": float(placement.estimate_seconds),
        },
    }


def _placement_from_jsonable(payload: dict[str, Any]) -> CziPlacement:
    plan_data = payload["plan"]
    placement_data = payload["placement"]
    tiles = tuple(
        CziMosaicTile(
            index_zyx=_int3(tile["index_zyx"]),
            mosaic_index=int(tile["mosaic_index"]),
            x=int(tile["x"]),
            y=int(tile["y"]),
            width=int(tile["width"]),
            height=int(tile["height"]),
        )
        for tile in plan_data["tiles"]
    )
    nominal_origins = _origins_from_records(plan_data["nominal_origin_records"])
    plan = CziStitchPlan(
        czi_path=str(plan_data["czi_path"]),
        channel=int(plan_data["channel"]),
        z_size=int(plan_data["z_size"]),
        tile_shape_zyx=_int3(plan_data["tile_shape_zyx"]),
        output_shape_yx=(int(plan_data["output_shape_yx"][0]), int(plan_data["output_shape_yx"][1])),
        overlap_zyx=_int3(plan_data["overlap_zyx"]),
        tile_count=int(plan_data["tile_count"]),
        tiles=tiles,
        nominal_origins_zyx=nominal_origins,
    )
    pair_shifts = tuple(
        PairShift(
            fixed_index_zyx=_int3(pair["fixed_index_zyx"]),
            moving_index_zyx=_int3(pair["moving_index_zyx"]),
            axis=int(pair["axis"]),
            origin_delta_zyx=_float3(pair["origin_delta_zyx"]),
            ncc=float(pair["ncc"]),
        )
        for pair in placement_data["pair_shifts"]
    )
    return CziPlacement(
        plan=plan,
        reference_z_indices=tuple(int(value) for value in placement_data["reference_z_indices"]),
        origins_zyx=_origins_from_records(placement_data["origins"]),
        pair_shifts=pair_shifts,
        fallback_min_ncc=float(placement_data["fallback_min_ncc"]),
        coarse_max_size=int(placement_data.get("coarse_max_size", 1024)),
        fine_upsample_factor=(
            None
            if placement_data.get("fine_upsample_factor") is None
            else int(placement_data["fine_upsample_factor"])
        ),
        estimate_seconds=float(placement_data["estimate_seconds"]),
    )


def _origin_records(origins: dict[tuple[int, int, int], tuple[float, float, float]]) -> list[dict[str, object]]:
    return [
        {"index_zyx": list(index), "origin_zyx": list(origin)}
        for index, origin in sorted(origins.items())
    ]


def _origins_from_records(records: list[dict[str, object]]) -> dict[tuple[int, int, int], tuple[float, float, float]]:
    return {_int3(record["index_zyx"]): _float3(record["origin_zyx"]) for record in records}


def _int3(values: object) -> tuple[int, int, int]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"Expected a 3-value sequence, got {values!r}")
    items = list(values)
    if len(items) != 3:
        raise ValueError(f"Expected 3 values, got {items}")
    return int(items[0]), int(items[1]), int(items[2])


def _float3(values: object) -> tuple[float, float, float]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"Expected a 3-value sequence, got {values!r}")
    items = list(values)
    if len(items) != 3:
        raise ValueError(f"Expected 3 values, got {items}")
    return float(items[0]), float(items[1]), float(items[2])
