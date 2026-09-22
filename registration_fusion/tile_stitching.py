from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
import zarr

from .flatfield import FlatfieldProfile, flatfield_metadata
from .io import read_tiff_zcyx
from .mosaic_streaming import (
    MosaicTileGeometry,
    array_nbytes,
    check_output_disk_space,
    output_bounds_from_origins,
    validate_z_range,
    write_streamed_mosaic_zarr,
)
from .stitching import PairShift, StitchTile, StitchingResult, estimate_tile_placement, fuse_tiles, stitch_tiles

DEFAULT_OME_FOLDER_PATTERN = "20x-EdUTest-514-L3.*.ome.tif"


def _log(message: str) -> None:
    print(f"[registration-fusion] {message}", flush=True)


@dataclass(frozen=True)
class TileManifestEntry:
    path: Path
    index_zyx: tuple[int, int, int]
    nominal_origin_zyx: tuple[float, float, float] | None


@dataclass(frozen=True)
class OmeFolderTile:
    path: Path
    index_zyx: tuple[int, int, int]
    nominal_origin_zyx: tuple[float, float, float]
    position_yx: tuple[float, float]
    physical_size_yx: tuple[float, float]
    pixel_origin_yx: tuple[float, float]
    tile_shape_zcyx: tuple[int, int, int, int]


@dataclass(frozen=True)
class OmeFolderPlan:
    folder: Path
    pattern: str
    tiles: tuple[OmeFolderTile, ...]
    tile_shape_zcyx: tuple[int, int, int, int]
    overlap_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class OmeFolderPlacement:
    plan: OmeFolderPlan
    reference_z_indices: tuple[int, ...]
    origins_zyx: dict[tuple[int, int, int], tuple[float, float, float]]
    pair_shifts: tuple[PairShift, ...]
    fallback_min_ncc: float | None
    estimate_seconds: float
    reference_channel: int = 0
    coarse_max_size: int = 1024
    fine_upsample_factor: int = 1


@dataclass(frozen=True)
class TileStitchRun:
    output_zarr: str
    metadata_path: str | None
    manifest_path: str
    tile_count: int
    output_shape_zcyx: tuple[int, int, int, int]
    output_dtype: str
    overlap_zyx: tuple[int, int, int]
    reference_channel: int
    fallback_min_ncc: float | None
    coarse_max_size: int
    fine_upsample_factor: int
    result: StitchingResult
    flatfield_profile: FlatfieldProfile | None = None


@dataclass(frozen=True)
class OmeFolderStitchRun:
    output_zarr: str | None
    metadata_path: str | None
    progress_path: str | None
    preview_path: str | None
    folder: str
    pattern: str
    tile_count: int
    output_shape_zcyx: tuple[int, int, int, int]
    output_origin_zyx: tuple[int, int, int]
    output_dtype: str
    overlap_zyx: tuple[int, int, int]
    reference_channel: int
    fallback_min_ncc: float | None
    coarse_max_size: int
    fine_upsample_factor: int
    placement: OmeFolderPlacement
    placement_estimated_this_run: bool
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


def stitch_tiff_tile_manifest(
    manifest_path: Path,
    output_zarr: Path,
    *,
    channels: int | None,
    overlap_zyx: tuple[int, int, int],
    reference_channel: int = 0,
    fallback_min_ncc: float | None = None,
    coarse_max_size: int = 1024,
    fine_upsample_factor: int = 1,
    bit_depth: int = 16,
    metadata_path: Path | None = None,
    overwrite: bool = False,
    flatfield_profile: FlatfieldProfile | None = None,
) -> TileStitchRun:
    if bit_depth not in (16, 32):
        raise ValueError("bit_depth must be 16 or 32")
    entries = read_tile_manifest(manifest_path)
    tiles = [
        StitchTile(entry.index_zyx, read_tiff_zcyx(entry.path, channels=channels).astype(np.float32, copy=False))
        for entry in entries
    ]
    nominal_origins = {
        entry.index_zyx: entry.nominal_origin_zyx
        for entry in entries
        if entry.nominal_origin_zyx is not None
    }
    if nominal_origins and len(nominal_origins) != len(entries):
        raise ValueError("Either every manifest row must provide nominal origins, or none may provide them")
    if fallback_min_ncc is not None and not nominal_origins:
        raise ValueError("fallback_min_ncc requires origin_z, origin_y, and origin_x columns in the manifest")
    result = stitch_tiles(
        tiles,
        overlap_zyx=overlap_zyx,
        reference_channel=reference_channel,
        coarse_max_size=coarse_max_size,
        fine_upsample_factor=fine_upsample_factor,
        nominal_origins_zyx=nominal_origins or None,
        nominal_fallback_min_ncc=fallback_min_ncc,
        flatfield_yx=None if flatfield_profile is None else flatfield_profile.flatfield_yx,
    )
    dtype = np.uint16 if bit_depth == 16 else np.float32
    output_bytes = _array_nbytes(result.fused_zcyx.shape, dtype)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    if output_zarr.exists() and not overwrite:
        raise ValueError(f"{output_zarr} already exists; pass overwrite=True to replace it")
    _check_output_disk_space(output_zarr, output_bytes)
    output = zarr.open_array(
        output_zarr,
        mode="w",
        shape=result.fused_zcyx.shape,
        chunks=(min(result.fused_zcyx.shape[0], 16), 1, min(result.fused_zcyx.shape[2], 512), min(result.fused_zcyx.shape[3], 512)),
        dtype=dtype,
    )
    output.attrs.update(
        _tile_zarr_attrs(
            manifest_path=manifest_path,
            entries=entries,
            overlap_zyx=overlap_zyx,
            reference_channel=reference_channel,
            fallback_min_ncc=fallback_min_ncc,
            coarse_max_size=coarse_max_size,
            fine_upsample_factor=fine_upsample_factor,
            flatfield_profile=flatfield_profile,
        )
    )
    if bit_depth == 16:
        output[:] = np.clip(result.fused_zcyx, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    elif bit_depth == 32:
        output[:] = result.fused_zcyx.astype(np.float32, copy=False)

    run = TileStitchRun(
        output_zarr=str(output_zarr),
        metadata_path=str(metadata_path) if metadata_path is not None else None,
        manifest_path=str(manifest_path),
        tile_count=len(entries),
        output_shape_zcyx=tuple(int(v) for v in result.fused_zcyx.shape),
        output_dtype=str(np.dtype(dtype)),
        overlap_zyx=overlap_zyx,
        reference_channel=reference_channel,
        fallback_min_ncc=fallback_min_ncc,
        coarse_max_size=coarse_max_size,
        fine_upsample_factor=fine_upsample_factor,
        flatfield_profile=flatfield_profile,
        result=result,
    )
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(_jsonable_tile_run(run), indent=2))
    return run


def stitch_ome_tiff_folder(
    folder: Path,
    output_zarr: Path | None,
    *,
    pattern: str = DEFAULT_OME_FOLDER_PATTERN,
    overlap_z: int | None = None,
    overlap_y: int | None = None,
    overlap_x: int | None = None,
    placement: OmeFolderPlacement | None = None,
    placement_output_path: Path | None = None,
    z_start: int = 0,
    z_stop: int | None = None,
    chunk_depth: int = 4,
    reference_z_start: int = 0,
    reference_z_count: int = 2,
    reference_channel: int = 0,
    fallback_min_ncc: float | None = 0.8,
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
) -> OmeFolderStitchRun:
    if bit_depth not in (16, 32):
        raise ValueError("bit_depth must be 16 or 32")
    run_start = time.monotonic()
    _log(f"stitch-ome-folder start: folder={folder} pattern={pattern}")
    if flatfield_profile is None:
        _log("BaSiC flatfield: none")
    else:
        _log(
            "BaSiC flatfield: "
            f"{flatfield_profile.source_path} shape={tuple(flatfield_profile.flatfield_yx.shape)} "
            "mode=fusion-only inverse flatfield; darkfield ignored"
        )
    if benchmark_only and output_zarr is None:
        _log("benchmark mode: no output Zarr will be written")
    placement_estimated_this_run = placement is None
    if placement is None:
        placement = estimate_ome_folder_stitch_placement(
            folder,
            pattern=pattern,
            overlap_z=overlap_z,
            overlap_y=overlap_y,
            overlap_x=overlap_x,
            reference_channel=reference_channel,
            reference_z_start=reference_z_start,
            reference_z_count=reference_z_count,
            fallback_min_ncc=fallback_min_ncc,
            coarse_max_size=coarse_max_size,
            fine_upsample_factor=fine_upsample_factor,
        )
    else:
        validate_ome_folder_placement(folder, placement, pattern=pattern)
    if placement_output_path is not None:
        save_ome_folder_placement(placement_output_path, placement)

    plan = placement.plan
    _validate_ome_output_channels((reference_channel,), plan=plan)
    stop = validate_z_range(z_start=z_start, z_stop=z_stop, z_size=plan.tile_shape_zcyx[0])
    output_origin_zyx, _output_shape_yx = _ome_output_bounds_from_origins(plan, placement.origins_zyx)
    _log(
        f"streaming fusion: z=[{z_start}, {stop}) chunk_depth={chunk_depth}; "
        f"tiles={len(plan.tiles)}; pair_count={len(placement.pair_shifts)}"
    )
    stats = write_streamed_mosaic_zarr(
        output_zarr,
        source_label="OME folder stitch",
        tile_geometries=_ome_tile_geometries(plan),
        z_size=plan.tile_shape_zcyx[0],
        output_channels=(reference_channel,),
        origins_zyx=placement.origins_zyx,
        overlap_zyx=plan.overlap_zyx,
        read_tiles=lambda output_channel, z_indices: _read_ome_folder_tiles(
            plan,
            channel=output_channel,
            z_indices=z_indices,
        ),
        z_start=z_start,
        z_stop=z_stop,
        chunk_depth=chunk_depth,
        bit_depth=bit_depth,
        zarr_attrs=_ome_folder_zarr_attrs(
            placement=placement,
            output_channels=(reference_channel,),
            output_origin_zyx=output_origin_zyx,
            z_start=z_start,
            z_stop=stop,
            flatfield_profile=flatfield_profile,
        ),
        attr_keys_to_validate=_ome_folder_zarr_attr_keys(),
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
        log=_log,
    )

    run = OmeFolderStitchRun(
        output_zarr=stats.output_zarr,
        metadata_path=str(metadata_path) if metadata_path is not None else None,
        progress_path=stats.progress_path,
        preview_path=stats.preview_path,
        folder=str(folder),
        pattern=pattern,
        tile_count=len(plan.tiles),
        output_shape_zcyx=stats.output_shape_zcyx,
        output_origin_zyx=stats.output_origin_zyx,
        output_dtype=stats.output_dtype,
        overlap_zyx=plan.overlap_zyx,
        reference_channel=reference_channel,
        fallback_min_ncc=placement.fallback_min_ncc,
        coarse_max_size=placement.coarse_max_size,
        fine_upsample_factor=placement.fine_upsample_factor,
        placement=placement,
        placement_estimated_this_run=placement_estimated_this_run,
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
        _log(f"writing metadata json={metadata_path}")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(_jsonable_ome_folder_run(run), indent=2))
    _log(f"stitch-ome-folder complete in {time.monotonic() - run_start:.1f}s")
    return run


def estimate_ome_folder_stitch_placement(
    folder: Path,
    *,
    pattern: str = DEFAULT_OME_FOLDER_PATTERN,
    overlap_z: int | None = None,
    overlap_y: int | None = None,
    overlap_x: int | None = None,
    reference_channel: int = 0,
    reference_z_start: int = 0,
    reference_z_count: int = 2,
    fallback_min_ncc: float | None = None,
    coarse_max_size: int = 1024,
    fine_upsample_factor: int = 1,
) -> OmeFolderPlacement:
    if reference_z_count < 2:
        raise ValueError("reference_z_count must be at least 2 for OME placement")
    if coarse_max_size <= 0:
        raise ValueError("coarse_max_size must be positive")
    plan_start = time.monotonic()
    plan = build_ome_folder_plan(
        folder,
        pattern=pattern,
        overlap_z=overlap_z,
        overlap_y=overlap_y,
        overlap_x=overlap_x,
    )
    _validate_ome_output_channels((reference_channel,), plan=plan)
    z_stop = validate_z_range(
        z_start=reference_z_start,
        z_stop=reference_z_start + reference_z_count,
        z_size=plan.tile_shape_zcyx[0],
    )
    z_indices = tuple(range(reference_z_start, z_stop))
    placement_overlap_zyx = (
        min(plan.overlap_zyx[0], len(z_indices) - 1),
        plan.overlap_zyx[1],
        plan.overlap_zyx[2],
    )
    _log(
        f"planned {len(plan.tiles)} OME tiles in {time.monotonic() - plan_start:.1f}s; "
        f"tile_shape_zcyx={plan.tile_shape_zcyx}; overlap_zyx={plan.overlap_zyx}"
    )
    start = time.perf_counter()
    _log(
        f"estimating OME placement from z={list(z_indices)}; "
        f"reference_channel={reference_channel}; coarse_max_size={coarse_max_size}; "
        f"fine_upsample_factor={fine_upsample_factor}"
    )
    tiles = _read_ome_folder_tiles(plan, channel=reference_channel, z_indices=z_indices)
    placement = estimate_tile_placement(
        tiles,
        overlap_zyx=placement_overlap_zyx,
        reference_channel=0,
        coarse_max_size=coarse_max_size,
        fine_upsample_factor=fine_upsample_factor,
        nominal_origins_zyx={tile.index_zyx: tile.nominal_origin_zyx for tile in plan.tiles},
        nominal_fallback_min_ncc=fallback_min_ncc,
    )
    origins = {index: (0.0, origin[1], origin[2]) for index, origin in placement.origins_zyx.items()}
    _log(
        f"OME placement estimated in {time.perf_counter() - start:.1f}s; "
        f"pairs={len(placement.pair_shifts)}"
    )
    return OmeFolderPlacement(
        plan=plan,
        reference_z_indices=z_indices,
        origins_zyx=origins,
        pair_shifts=placement.pair_shifts,
        fallback_min_ncc=fallback_min_ncc,
        estimate_seconds=time.perf_counter() - start,
        reference_channel=reference_channel,
        coarse_max_size=coarse_max_size,
        fine_upsample_factor=fine_upsample_factor,
    )


def save_ome_folder_placement(path: Path | str, placement: OmeFolderPlacement) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable_ome_folder_placement(placement), indent=2))


def load_ome_folder_placement(path: Path | str) -> OmeFolderPlacement:
    path = Path(path)
    return _ome_folder_placement_from_jsonable(json.loads(path.read_text()))


def validate_ome_folder_placement(
    folder: Path,
    placement: OmeFolderPlacement,
    *,
    pattern: str = DEFAULT_OME_FOLDER_PATTERN,
) -> None:
    current = build_ome_folder_plan(
        folder,
        pattern=pattern,
        overlap_z=placement.plan.overlap_zyx[0],
        overlap_y=placement.plan.overlap_zyx[1],
        overlap_x=placement.plan.overlap_zyx[2],
    )
    if _ome_plan_signature(current) != _ome_plan_signature(placement.plan):
        raise ValueError("OME placement does not match the target OME folder geometry")


def build_ome_folder_plan(
    folder: Path,
    *,
    pattern: str = DEFAULT_OME_FOLDER_PATTERN,
    overlap_z: int | None = None,
    overlap_y: int | None = None,
    overlap_x: int | None = None,
) -> OmeFolderPlan:
    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"OME-TIFF folder does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"OME-TIFF input must be a folder: {folder}")
    paths = sorted(path for path in folder.glob(pattern) if path.stat().st_size > 16)
    if not paths:
        raise ValueError(f"No OME-TIFF tiles matching {pattern!r} found in {folder}")
    _log(f"discovered {len(paths)} matching OME-TIFF tiles after ignoring <=16-byte stubs")

    parsed = []
    for tile_number, path in enumerate(paths, start=1):
        metadata_start = time.monotonic()
        tile = _read_ome_folder_tile_metadata(path)
        parsed.append(tile)
        _log(
            f"metadata tile {tile_number}/{len(paths)} path={path.name} "
            f"position_yx={tile.position_yx} physical_size_yx={tile.physical_size_yx} "
            f"pixel_origin_yx={tile.pixel_origin_yx} shape_zcyx={tile.tile_shape_zcyx} "
            f"in {time.monotonic() - metadata_start:.1f}s"
        )
    shapes = {tile.tile_shape_zcyx for tile in parsed}
    if len(shapes) != 1:
        raise ValueError(f"All OME-TIFF tiles must have matching ZCYX shapes, got {sorted(shapes)}")
    physical_sizes = {tile.physical_size_yx for tile in parsed}
    if len(physical_sizes) != 1:
        raise ValueError(f"All OME-TIFF tiles must have matching PhysicalSizeY/X, got {sorted(physical_sizes)}")

    unique_y = sorted({tile.pixel_origin_yx[0] for tile in parsed})
    unique_x = sorted({tile.pixel_origin_yx[1] for tile in parsed})
    y_to_index = {value: index for index, value in enumerate(unique_y)}
    x_to_index = {value: index for index, value in enumerate(unique_x)}
    seen_indices: set[tuple[int, int, int]] = set()
    indexed_tiles = []
    for tile in parsed:
        index = (0, y_to_index[tile.pixel_origin_yx[0]], x_to_index[tile.pixel_origin_yx[1]])
        if index in seen_indices:
            raise ValueError(f"Duplicate OME tile position for grid index {index}")
        seen_indices.add(index)
        indexed_tiles.append(
            OmeFolderTile(
                path=tile.path,
                index_zyx=index,
                nominal_origin_zyx=(0.0, tile.pixel_origin_yx[0], tile.pixel_origin_yx[1]),
                position_yx=tile.position_yx,
                physical_size_yx=tile.physical_size_yx,
                pixel_origin_yx=tile.pixel_origin_yx,
                tile_shape_zcyx=tile.tile_shape_zcyx,
            )
        )

    tile_shape = indexed_tiles[0].tile_shape_zcyx
    inferred_overlap = _infer_ome_folder_overlap(tile_shape, unique_y, unique_x)
    overlap = (
        inferred_overlap[0] if overlap_z is None else overlap_z,
        inferred_overlap[1] if overlap_y is None else overlap_y,
        inferred_overlap[2] if overlap_x is None else overlap_x,
    )
    return OmeFolderPlan(
        folder=folder,
        pattern=pattern,
        tiles=tuple(sorted(indexed_tiles, key=lambda tile: tile.index_zyx)),
        tile_shape_zcyx=tile_shape,
        overlap_zyx=tuple(int(value) for value in overlap),
    )


def summarize_tiff_tile_manifest(
    manifest_path: Path,
    *,
    channels: int | None,
    overlap_zyx: tuple[int, int, int],
    bit_depth: int = 16,
) -> dict[str, object]:
    if bit_depth not in (16, 32):
        raise ValueError("bit_depth must be 16 or 32")
    entries = read_tile_manifest(manifest_path)
    shapes = []
    dtypes = []
    for entry in entries:
        shape, dtype = inspect_tiff_zcyx_shape(entry.path, channels=channels)
        shapes.append(shape)
        dtypes.append(str(dtype))
    if len(set(shapes)) != 1:
        raise ValueError(f"All tile TIFFs must have matching ZCYX shapes, got {sorted(set(shapes))}")
    tile_shape_zcyx = shapes[0]
    output_shape_zcyx = _estimate_manifest_output_shape(entries, tile_shape_zcyx, overlap_zyx)
    dtype = np.uint16 if bit_depth == 16 else np.float32
    return {
        "manifest_path": str(manifest_path),
        "tile_count": len(entries),
        "tile_shape_zcyx": list(tile_shape_zcyx),
        "tile_dtypes": sorted(set(dtypes)),
        "estimated_output_shape_zcyx": list(output_shape_zcyx),
        "estimated_output_bytes": int(np.prod(output_shape_zcyx, dtype=np.int64) * np.dtype(dtype).itemsize),
        "overlap_zyx": list(overlap_zyx),
        "tile_indices_zyx": [list(entry.index_zyx) for entry in entries],
        "has_nominal_origins": all(entry.nominal_origin_zyx is not None for entry in entries),
    }


def read_tile_manifest(manifest_path: Path) -> list[TileManifestEntry]:
    entries = []
    seen_indices = set()
    base_dir = manifest_path.parent
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "index_z", "index_y", "index_x"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Tile manifest is missing required columns: {sorted(missing)}")
        has_nominal = {"origin_z", "origin_y", "origin_x"}.issubset(set(reader.fieldnames or []))
        for row_number, row in enumerate(reader, start=2):
            tile_path = Path(row["path"])
            if not tile_path.is_absolute():
                tile_path = base_dir / tile_path
            index = (int(row["index_z"]), int(row["index_y"]), int(row["index_x"]))
            if index in seen_indices:
                raise ValueError(f"Tile manifest row {row_number} duplicates tile index {index}")
            seen_indices.add(index)
            nominal = None
            if has_nominal and row["origin_z"] and row["origin_y"] and row["origin_x"]:
                nominal = (float(row["origin_z"]), float(row["origin_y"]), float(row["origin_x"]))
            if not tile_path.exists():
                raise ValueError(f"Tile manifest row {row_number} points to a missing file: {tile_path}")
            entries.append(TileManifestEntry(tile_path, index, nominal))
    if not entries:
        raise ValueError("Tile manifest must contain at least one tile")
    return entries


def inspect_tiff_zcyx_shape(path: Path, *, channels: int | None) -> tuple[tuple[int, int, int, int], np.dtype]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path} does not contain any TIFF series")
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
    if len(shape) == 3:
        if channels is None:
            return (shape[0], 1, shape[1], shape[2]), dtype
        if channels <= 0:
            raise ValueError("channels must be positive")
        if shape[0] % channels:
            raise ValueError(f"{path} has {shape[0]} planes, not divisible by {channels} channels")
        return (shape[0] // channels, channels, shape[1], shape[2]), dtype
    if len(shape) == 4:
        if channels is not None and shape[1] != channels:
            raise ValueError(f"{path} has {shape[1]} channels, not {channels}")
        return shape, dtype
    raise ValueError(f"Expected a 3D or 4D TIFF stack, got shape {shape} from {path}")


def _jsonable_tile_run(run: TileStitchRun) -> dict[str, object]:
    nccs = [pair.ncc for pair in run.result.pair_shifts]
    return {
        "output_zarr": run.output_zarr,
        "manifest_path": run.manifest_path,
        "tile_count": int(run.tile_count),
        "output_shape_zcyx": list(run.output_shape_zcyx),
        "output_dtype": run.output_dtype,
        "estimated_output_bytes": _array_nbytes(run.output_shape_zcyx, np.dtype(run.output_dtype)),
        "overlap_zyx": list(run.overlap_zyx),
        "reference_channel": int(run.reference_channel),
        "fallback_min_ncc": None if run.fallback_min_ncc is None else float(run.fallback_min_ncc),
        "coarse_max_size": int(run.coarse_max_size),
        "fine_upsample_factor": int(run.fine_upsample_factor),
        "flatfield_correction": flatfield_metadata(run.flatfield_profile),
        "origins_zyx": {str(index): list(origin) for index, origin in run.result.origins_zyx.items()},
        "pair_count": len(run.result.pair_shifts),
        "ncc_min": float(min(nccs)) if nccs else None,
        "ncc_median": float(np.median(nccs)) if nccs else None,
        "ncc_max": float(max(nccs)) if nccs else None,
        "pair_shifts": [
            {
                "fixed": list(pair.fixed_index_zyx),
                "moving": list(pair.moving_index_zyx),
                "axis": int(pair.axis),
                "origin_delta_zyx": list(pair.origin_delta_zyx),
                "ncc": float(pair.ncc),
            }
            for pair in run.result.pair_shifts
        ],
    }


def _jsonable_ome_folder_run(run: OmeFolderStitchRun) -> dict[str, object]:
    placement = run.placement
    nccs = [pair.ncc for pair in placement.pair_shifts]
    estimated = None
    if 0 < run.chunks_written < run.total_chunks:
        seconds_per_chunk = run.total_seconds / run.chunks_written
        estimated = {
            "chunks_measured": int(run.chunks_written),
            "seconds_per_chunk": float(seconds_per_chunk),
            "estimated_processing_loop_seconds": float(seconds_per_chunk * run.total_chunks),
        }
    return {
        "output_zarr": run.output_zarr,
        "progress_path": run.progress_path,
        "preview_path": run.preview_path,
        "folder": run.folder,
        "pattern": run.pattern,
        "tile_count": int(run.tile_count),
        "tile_shape_zcyx": list(placement.plan.tile_shape_zcyx),
        "output_shape_zcyx": list(run.output_shape_zcyx),
        "output_origin_zyx": list(run.output_origin_zyx),
        "output_dtype": run.output_dtype,
        "estimated_output_bytes": _array_nbytes(run.output_shape_zcyx, np.dtype(run.output_dtype)),
        "benchmark_only": bool(run.benchmark_only),
        "overlap_zyx": list(run.overlap_zyx),
        "output_channels": [int(run.reference_channel)],
        "placement_reference_channel": int(placement.reference_channel),
        "fallback_min_ncc": None if run.fallback_min_ncc is None else float(run.fallback_min_ncc),
        "coarse_max_size": int(run.coarse_max_size),
        "fine_upsample_factor": int(run.fine_upsample_factor),
        "flatfield_correction": flatfield_metadata(run.flatfield_profile),
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
            "placement_estimate_this_run": float(placement.estimate_seconds if run.placement_estimated_this_run else 0.0),
            "read": float(run.read_seconds),
            "fuse": float(run.fuse_seconds),
            "write": float(run.write_seconds),
            "total_processing_loop": float(run.total_seconds),
        },
        "tiles": [_jsonable_ome_folder_tile(tile) for tile in placement.plan.tiles],
        "origins_zyx": {str(index): list(origin) for index, origin in placement.origins_zyx.items()},
        "origin_records": _origin_records(placement.origins_zyx),
        "pair_count": len(placement.pair_shifts),
        "ncc_min": float(min(nccs)) if nccs else None,
        "ncc_median": float(np.median(nccs)) if nccs else None,
        "ncc_max": float(max(nccs)) if nccs else None,
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
    }


def _tile_zarr_attrs(
    *,
    manifest_path: Path,
    entries: list[TileManifestEntry],
    overlap_zyx: tuple[int, int, int],
    reference_channel: int,
    fallback_min_ncc: float | None,
    coarse_max_size: int,
    fine_upsample_factor: int,
    flatfield_profile: FlatfieldProfile | None,
) -> dict[str, object]:
    return {
        "registration_fusion_schema": "tiff_tiles_stitched_zcyx_v1",
        "axes": ["z", "c", "y", "x"],
        "source_manifest": str(manifest_path),
        "tile_count": len(entries),
        "tile_indices_zyx": [list(entry.index_zyx) for entry in entries],
        "overlap_zyx": list(overlap_zyx),
        "reference_channel": int(reference_channel),
        "fallback_min_ncc": None if fallback_min_ncc is None else float(fallback_min_ncc),
        "coarse_max_size": int(coarse_max_size),
        "fine_upsample_factor": int(fine_upsample_factor),
        "flatfield_correction": flatfield_metadata(flatfield_profile),
    }


def _ome_folder_zarr_attrs(
    *,
    placement: OmeFolderPlacement,
    output_channels: tuple[int, ...],
    output_origin_zyx: tuple[int, int, int],
    z_start: int,
    z_stop: int,
    flatfield_profile: FlatfieldProfile | None,
) -> dict[str, object]:
    plan = placement.plan
    return {
        "registration_fusion_schema": "ome_tiff_folder_stitched_zcyx_v1",
        "axes": ["z", "c", "y", "x"],
        "source_folder": str(plan.folder),
        "source_pattern": plan.pattern,
        "output_channels": list(output_channels),
        "source_z_start": int(z_start),
        "source_z_stop": int(z_stop),
        "output_origin_zyx": list(output_origin_zyx),
        "tile_count": len(plan.tiles),
        "tile_indices_zyx": [list(tile.index_zyx) for tile in plan.tiles],
        "tile_pixel_origins_zyx": [list(tile.nominal_origin_zyx) for tile in plan.tiles],
        "overlap_zyx": list(plan.overlap_zyx),
        "placement_reference_channel": int(placement.reference_channel),
        "placement_reference_z_indices": list(placement.reference_z_indices),
        "placement_origin_records": _origin_records(placement.origins_zyx),
        "fallback_min_ncc": None if placement.fallback_min_ncc is None else float(placement.fallback_min_ncc),
        "coarse_max_size": int(placement.coarse_max_size),
        "fine_upsample_factor": int(placement.fine_upsample_factor),
        "flatfield_correction": flatfield_metadata(flatfield_profile),
    }


def _ome_folder_zarr_attr_keys() -> tuple[str, ...]:
    return (
        "registration_fusion_schema",
        "axes",
        "source_folder",
        "source_pattern",
        "output_channels",
        "source_z_start",
        "source_z_stop",
        "output_origin_zyx",
        "tile_count",
        "tile_indices_zyx",
        "tile_pixel_origins_zyx",
        "overlap_zyx",
        "placement_reference_channel",
        "placement_reference_z_indices",
        "placement_origin_records",
        "flatfield_correction",
    )


def _jsonable_ome_folder_placement(placement: OmeFolderPlacement) -> dict[str, object]:
    plan = placement.plan
    return {
        "version": 1,
        "plan": {
            "folder": str(plan.folder),
            "pattern": plan.pattern,
            "tile_shape_zcyx": list(plan.tile_shape_zcyx),
            "overlap_zyx": list(plan.overlap_zyx),
            "tiles": [_jsonable_ome_folder_tile(tile) for tile in plan.tiles],
        },
        "placement": {
            "reference_channel": int(placement.reference_channel),
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
            "fallback_min_ncc": None if placement.fallback_min_ncc is None else float(placement.fallback_min_ncc),
            "coarse_max_size": int(placement.coarse_max_size),
            "fine_upsample_factor": int(placement.fine_upsample_factor),
            "estimate_seconds": float(placement.estimate_seconds),
        },
    }


def _ome_folder_placement_from_jsonable(payload: dict[str, object]) -> OmeFolderPlacement:
    plan_data = payload["plan"]
    placement_data = payload["placement"]
    if not isinstance(plan_data, dict) or not isinstance(placement_data, dict):
        raise ValueError("OME placement JSON is missing plan or placement sections")
    tiles = tuple(_ome_folder_tile_from_jsonable(tile) for tile in plan_data["tiles"])
    plan = OmeFolderPlan(
        folder=Path(str(plan_data["folder"])),
        pattern=str(plan_data["pattern"]),
        tiles=tiles,
        tile_shape_zcyx=_int4(plan_data["tile_shape_zcyx"]),
        overlap_zyx=_int3(plan_data["overlap_zyx"]),
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
    fallback = placement_data.get("fallback_min_ncc")
    return OmeFolderPlacement(
        plan=plan,
        reference_z_indices=tuple(int(value) for value in placement_data["reference_z_indices"]),
        origins_zyx=_origins_from_records(placement_data["origins"]),
        pair_shifts=pair_shifts,
        fallback_min_ncc=None if fallback is None else float(fallback),
        reference_channel=int(placement_data.get("reference_channel", 0)),
        coarse_max_size=int(placement_data.get("coarse_max_size", 1024)),
        fine_upsample_factor=int(placement_data.get("fine_upsample_factor", 1)),
        estimate_seconds=float(placement_data["estimate_seconds"]),
    )


def _ome_folder_tile_from_jsonable(payload: object) -> OmeFolderTile:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected OME tile record, got {payload!r}")
    return OmeFolderTile(
        path=Path(str(payload["path"])),
        index_zyx=_int3(payload["index_zyx"]),
        nominal_origin_zyx=_float3(payload["nominal_origin_zyx"]),
        position_yx=_float2(payload["position_yx"]),
        physical_size_yx=_float2(payload["physical_size_yx"]),
        pixel_origin_yx=_float2(payload["pixel_origin_yx"]),
        tile_shape_zcyx=_int4(payload["tile_shape_zcyx"]),
    )


def _ome_plan_signature(plan: OmeFolderPlan) -> tuple[object, ...]:
    return (
        str(plan.folder),
        plan.pattern,
        tuple(plan.tile_shape_zcyx),
        tuple(plan.overlap_zyx),
        tuple(
            (
                tile.index_zyx,
                str(tile.path),
                tile.nominal_origin_zyx,
                tile.physical_size_yx,
                tile.tile_shape_zcyx,
            )
            for tile in sorted(plan.tiles, key=lambda value: value.index_zyx)
        ),
    )


def _ome_tile_geometries(plan: OmeFolderPlan) -> tuple[MosaicTileGeometry, ...]:
    tile_shape = (plan.tile_shape_zcyx[0], plan.tile_shape_zcyx[2], plan.tile_shape_zcyx[3])
    return tuple(MosaicTileGeometry(tile.index_zyx, tile_shape) for tile in plan.tiles)


def _ome_output_bounds_from_origins(
    plan: OmeFolderPlan,
    origins_zyx: dict[tuple[int, int, int], tuple[float, float, float]],
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    return output_bounds_from_origins(_ome_tile_geometries(plan), origins_zyx)


def _validate_ome_output_channels(channels: tuple[int, ...], *, plan: OmeFolderPlan) -> None:
    channel_count = plan.tile_shape_zcyx[1]
    for channel in channels:
        if channel < 0 or channel >= channel_count:
            raise ValueError(f"OME channel {channel} is outside available channel range [0, {channel_count})")


def _read_ome_folder_tiles(
    plan: OmeFolderPlan,
    *,
    channel: int,
    z_indices: tuple[int, ...],
) -> list[StitchTile]:
    _validate_ome_output_channels((channel,), plan=plan)
    tiles = []
    for tile in plan.tiles:
        data = _read_ome_tiff_zcyx_slab(tile.path, channel=channel, z_indices=z_indices)
        tiles.append(StitchTile(tile.index_zyx, data.astype(np.float32, copy=False)))
    return tiles


def _read_ome_tiff_zcyx_slab(path: Path, *, channel: int, z_indices: tuple[int, ...]) -> np.ndarray:
    if not z_indices:
        raise ValueError("z_indices must not be empty")
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path} does not contain any TIFF series")
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        if len(shape) == 3:
            if channel != 0:
                raise ValueError(f"{path} has one channel; requested channel {channel}")
            pages = series.pages
            planes = []
            for z in z_indices:
                if z < 0 or z >= shape[0]:
                    raise ValueError(f"z index {z} is outside available range [0, {shape[0]}) for {path}")
                planes.append(np.asarray(pages[z].asarray()))
            return np.stack(planes, axis=0)[:, np.newaxis]
        if len(shape) == 4:
            raise ValueError(
                f"{path} has 4D OME-TIFF shape {shape}; streaming 4D OME slab reads are not implemented"
            )
    raise ValueError(f"Expected a 3D or 4D OME-TIFF stack, got shape {shape} from {path}")


def _array_nbytes(shape: tuple[int, ...], dtype: type[np.generic] | np.dtype) -> int:
    return array_nbytes(shape, dtype)


def _check_output_disk_space(output_zarr: Path, required_bytes: int) -> None:
    check_output_disk_space(output_zarr, required_bytes)


def _estimate_manifest_output_shape(
    entries: list[TileManifestEntry],
    tile_shape_zcyx: tuple[int, int, int, int],
    overlap_zyx: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    tile_shape_zyx = (tile_shape_zcyx[0], tile_shape_zcyx[2], tile_shape_zcyx[3])
    channels = tile_shape_zcyx[1]
    origins = [entry.nominal_origin_zyx for entry in entries]
    if all(origin is not None for origin in origins):
        origin_array = np.asarray(origins, dtype=np.float32)
    elif any(origin is not None for origin in origins):
        raise ValueError("Either every manifest row must provide nominal origins, or none may provide them")
    else:
        stride = tuple(tile_shape_zyx[axis] - overlap_zyx[axis] for axis in range(3))
        if any(value <= 0 for value in stride):
            raise ValueError("overlap must be smaller than tile shape on every axis")
        origin_array = np.asarray(
            [
                tuple(entry.index_zyx[axis] * stride[axis] for axis in range(3))
                for entry in entries
            ],
            dtype=np.float32,
        )
    min_origin = np.floor(origin_array.min(axis=0)).astype(np.int64)
    max_extent = np.ceil(origin_array + np.asarray(tile_shape_zyx, dtype=np.float32)).max(axis=0).astype(np.int64)
    output_zyx = tuple(int(value) for value in (max_extent - min_origin))
    return output_zyx[0], channels, output_zyx[1], output_zyx[2]


def _read_ome_folder_tile_metadata(path: Path) -> OmeFolderTile:
    with tifffile.TiffFile(path) as tif:
        ome_xml = tif.ome_metadata
    if not ome_xml:
        raise ValueError(f"{path} is missing OME metadata")
    shape, _dtype = inspect_tiff_zcyx_shape(path, channels=None)
    position_yx, physical_size_yx = _parse_ome_tile_position(ome_xml, path)
    origin_y = position_yx[0] / physical_size_yx[0]
    origin_x = position_yx[1] / physical_size_yx[1]
    return OmeFolderTile(
        path=path,
        index_zyx=(0, 0, 0),
        nominal_origin_zyx=(0.0, origin_y, origin_x),
        position_yx=position_yx,
        physical_size_yx=physical_size_yx,
        pixel_origin_yx=(origin_y, origin_x),
        tile_shape_zcyx=shape,
    )


def _parse_ome_tile_position(ome_xml: str, path: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    root = ET.fromstring(ome_xml)
    pixels = next((element for element in root.iter() if _local_name(element.tag) == "Pixels"), None)
    if pixels is None:
        raise ValueError(f"{path} OME metadata is missing Pixels")
    try:
        physical_size_y = float(pixels.attrib["PhysicalSizeY"])
        physical_size_x = float(pixels.attrib["PhysicalSizeX"])
    except KeyError as exc:
        raise ValueError(f"{path} OME metadata is missing {exc.args[0]}") from exc
    if physical_size_y <= 0 or physical_size_x <= 0:
        raise ValueError(f"{path} OME PhysicalSizeY/X must be positive")

    planes = [element for element in root.iter() if _local_name(element.tag) == "Plane"]
    plane = next(
        (element for element in planes if "PositionY" in element.attrib and "PositionX" in element.attrib),
        None,
    )
    if plane is None:
        raise ValueError(f"{path} OME metadata is missing Plane PositionY/PositionX")
    return (
        (float(plane.attrib["PositionY"]), float(plane.attrib["PositionX"])),
        (physical_size_y, physical_size_x),
    )


def _infer_ome_folder_overlap(
    tile_shape_zcyx: tuple[int, int, int, int],
    unique_y_px: list[float],
    unique_x_px: list[float],
) -> tuple[int, int, int]:
    z_size, _channels, tile_y, tile_x = tile_shape_zcyx
    overlap_z = min(4, z_size - 1) if z_size > 0 else 0
    overlap_y = 0 if len(unique_y_px) < 2 else int(round(tile_y - float(np.median(np.diff(unique_y_px)))))
    overlap_x = 0 if len(unique_x_px) < 2 else int(round(tile_x - float(np.median(np.diff(unique_x_px)))))
    return overlap_z, overlap_y, overlap_x


def _jsonable_ome_folder_tile(tile: OmeFolderTile) -> dict[str, object]:
    return {
        "path": str(tile.path),
        "index_zyx": list(tile.index_zyx),
        "position_yx": list(tile.position_yx),
        "physical_size_yx": list(tile.physical_size_yx),
        "pixel_origin_yx": list(tile.pixel_origin_yx),
        "nominal_origin_zyx": list(tile.nominal_origin_zyx),
        "tile_shape_zcyx": list(tile.tile_shape_zcyx),
    }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _origin_records(origins: dict[tuple[int, int, int], tuple[float, float, float]]) -> list[dict[str, object]]:
    return [
        {"index_zyx": list(index), "origin_zyx": list(origin)}
        for index, origin in sorted(origins.items())
    ]


def _origins_from_records(records: object) -> dict[tuple[int, int, int], tuple[float, float, float]]:
    if not isinstance(records, list):
        raise ValueError(f"Expected origin record list, got {records!r}")
    return {_int3(record["index_zyx"]): _float3(record["origin_zyx"]) for record in records}


def _int3(values: object) -> tuple[int, int, int]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"Expected a 3-value sequence, got {values!r}")
    items = list(values)
    if len(items) != 3:
        raise ValueError(f"Expected 3 values, got {items}")
    return int(items[0]), int(items[1]), int(items[2])


def _int4(values: object) -> tuple[int, int, int, int]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"Expected a 4-value sequence, got {values!r}")
    items = list(values)
    if len(items) != 4:
        raise ValueError(f"Expected 4 values, got {items}")
    return int(items[0]), int(items[1]), int(items[2]), int(items[3])


def _float2(values: object) -> tuple[float, float]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"Expected a 2-value sequence, got {values!r}")
    items = list(values)
    if len(items) != 2:
        raise ValueError(f"Expected 2 values, got {items}")
    return float(items[0]), float(items[1])


def _float3(values: object) -> tuple[float, float, float]:
    if not isinstance(values, list | tuple):
        raise ValueError(f"Expected a 3-value sequence, got {values!r}")
    items = list(values)
    if len(items) != 3:
        raise ValueError(f"Expected 3 values, got {items}")
    return float(items[0]), float(items[1]), float(items[2])
