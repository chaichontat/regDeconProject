from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .flatfield import validate_flatfield

EPS = np.float32(1e-6)


Index3 = tuple[int, int, int]


@dataclass(frozen=True)
class StitchTile:
    index_zyx: Index3
    data_zcyx: np.ndarray


@dataclass(frozen=True)
class FusionFlatfield:
    inv_flat_gpu: Any
    shape_yx: tuple[int, int]


@dataclass(frozen=True)
class PairShift:
    fixed_index_zyx: Index3
    moving_index_zyx: Index3
    axis: int
    origin_delta_zyx: tuple[float, float, float]
    ncc: float


@dataclass(frozen=True)
class StitchingResult:
    fused_zcyx: np.ndarray
    origins_zyx: dict[Index3, tuple[float, float, float]]
    pair_shifts: tuple[PairShift, ...]
    weight_sum_zyx: np.ndarray


@dataclass(frozen=True)
class StitchPlacement:
    origins_zyx: dict[Index3, tuple[float, float, float]]
    pair_shifts: tuple[PairShift, ...]


def make_overlapping_tiles(
    volume_zcyx: np.ndarray,
    *,
    tile_shape_zyx: Index3,
    overlap_zyx: Index3,
) -> list[StitchTile]:
    """Slice a ZCYX volume into a regular overlapped 3D tile grid."""
    volume = _as_zcyx(volume_zcyx, "volume_zcyx")
    tile_shape = _as_positive_int3(tile_shape_zyx, "tile_shape_zyx")
    overlap = _as_nonnegative_int3(overlap_zyx, "overlap_zyx")
    stride = _stride_from_tile_overlap(tile_shape, overlap)
    grid_shape = _regular_grid_shape(volume.shape[0:1] + volume.shape[2:4], tile_shape, stride)

    tiles: list[StitchTile] = []
    for index in product(*(range(size) for size in grid_shape)):
        start = tuple(index[axis] * stride[axis] for axis in range(3))
        z0, y0, x0 = start
        z1, y1, x1 = (start[axis] + tile_shape[axis] for axis in range(3))
        tiles.append(StitchTile(index, volume[z0:z1, :, y0:y1, x0:x1].copy()))
    return tiles


def estimate_adjacent_shift(
    fixed_zcyx: np.ndarray,
    moving_zcyx: np.ndarray,
    *,
    axis: int,
    overlap_zyx: Index3,
    reference_channel: int = 0,
    use_log: bool = True,
    coarse_max_size: int = 1024,
    fine_upsample_factor: int = 1,
) -> tuple[tuple[float, float, float], float]:
    """Estimate the origin of an adjacent moving tile relative to a fixed tile.

    The nominal displacement along ``axis`` comes from the tile size and expected
    overlap. Phase correlation is run on the expected overlap slabs to recover
    residual 3D shift, then the 2^3 Fourier wrap candidates are scored by NCC on
    the resulting full-tile overlap.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    fixed = _as_zcyx(fixed_zcyx, "fixed_zcyx")
    moving = _as_zcyx(moving_zcyx, "moving_zcyx")
    if fixed.shape != moving.shape:
        raise ValueError(f"Expected matching tile shapes, got {fixed.shape} and {moving.shape}")
    if not 0 <= reference_channel < fixed.shape[1]:
        raise ValueError(f"reference_channel {reference_channel} is outside channel count {fixed.shape[1]}")
    if coarse_max_size < 1:
        raise ValueError("coarse_max_size must be at least 1")
    if fine_upsample_factor < 1:
        raise ValueError("fine_upsample_factor must be at least 1")
    _require_cuda()

    tile_shape = fixed.shape[0:1] + fixed.shape[2:4]
    overlap = _as_nonnegative_int3(overlap_zyx, "overlap_zyx")
    for dim, (overlap_size, tile_size) in enumerate(zip(overlap, tile_shape, strict=True)):
        if overlap_size >= tile_size:
            raise ValueError(f"overlap_zyx[{dim}] must be smaller than tile size {tile_size}")
        if dim == axis and overlap_size <= 0:
            raise ValueError(f"overlap_zyx[{dim}] must be between 1 and tile size - 1")

    fixed_ref = fixed[:, reference_channel].astype(np.float32, copy=False)
    moving_ref = moving[:, reference_channel].astype(np.float32, copy=False)
    fixed_ref_work, moving_ref_work = _gpu_arrays(fixed_ref, moving_ref)
    fixed_overlap = _expected_overlap_slab(fixed_ref_work, axis=axis, overlap=overlap, trailing=True)
    moving_overlap = _expected_overlap_slab(moving_ref_work, axis=axis, overlap=overlap, trailing=False)
    coarse_fixed, coarse_moving, coarse_scale = _coarse_phase_inputs_gpu(
        fixed_overlap,
        moving_overlap,
        max_size=coarse_max_size,
    )
    residual_peak = _phase_correlation_peak_gpu(coarse_fixed, coarse_moving, use_log=use_log)
    base_delta = np.zeros(3, dtype=np.float32)
    base_delta[axis] = tile_shape[axis] - overlap[axis]

    best_delta: np.ndarray | None = None
    best_ncc = -np.inf
    scale = np.asarray(coarse_scale, dtype=np.float32)
    for residual in _wrap_candidates(residual_peak, coarse_fixed.shape):
        candidate = base_delta + np.asarray(residual, dtype=np.float32) * scale
        ncc = _overlap_ncc_for_origin_delta_gpu(fixed_ref_work, moving_ref_work, candidate)
        if ncc > best_ncc:
            best_ncc = ncc
            best_delta = candidate

    if best_delta is None or not np.isfinite(best_ncc):
        raise ValueError("Could not find a valid adjacent-tile overlap for NCC scoring")
    if fine_upsample_factor > 1:
        best_delta = _refine_delta_subpixel(
            fixed_ref_work,
            moving_ref_work,
            best_delta,
            upsample_factor=fine_upsample_factor,
            use_log=use_log,
        )
        best_ncc = _overlap_ncc_for_origin_delta_gpu(fixed_ref_work, moving_ref_work, best_delta)
    return tuple(float(v) for v in best_delta), float(best_ncc)


def stitch_tiles(
    tiles: list[StitchTile],
    *,
    overlap_zyx: Index3,
    reference_channel: int = 0,
    use_log: bool = True,
    coarse_max_size: int = 1024,
    fine_upsample_factor: int = 1,
    nominal_origins_zyx: dict[Index3, tuple[float, float, float]] | None = None,
    nominal_fallback_min_ncc: float | None = None,
    flatfield_yx: np.ndarray | FusionFlatfield | None = None,
) -> StitchingResult:
    placement = estimate_tile_placement(
        tiles,
        overlap_zyx=overlap_zyx,
        reference_channel=reference_channel,
        use_log=use_log,
        coarse_max_size=coarse_max_size,
        fine_upsample_factor=fine_upsample_factor,
        nominal_origins_zyx=nominal_origins_zyx,
        nominal_fallback_min_ncc=nominal_fallback_min_ncc,
    )
    fused, weight_sum = fuse_tiles(
        tiles,
        origins_zyx=placement.origins_zyx,
        overlap_zyx=overlap_zyx,
        flatfield_yx=flatfield_yx,
    )
    return StitchingResult(
        fused_zcyx=fused,
        origins_zyx=placement.origins_zyx,
        pair_shifts=placement.pair_shifts,
        weight_sum_zyx=weight_sum,
    )


def estimate_tile_placement(
    tiles: list[StitchTile],
    *,
    overlap_zyx: Index3,
    reference_channel: int = 0,
    use_log: bool = True,
    coarse_max_size: int = 1024,
    fine_upsample_factor: int = 1,
    nominal_origins_zyx: dict[Index3, tuple[float, float, float]] | None = None,
    nominal_fallback_min_ncc: float | None = None,
) -> StitchPlacement:
    """Estimate tile origins without materializing a fused mosaic."""
    if not tiles:
        raise ValueError("tiles must not be empty")
    tile_by_index = {tile.index_zyx: tile for tile in tiles}
    if len(tile_by_index) != len(tiles):
        raise ValueError("tile indices must be unique")
    channel_count = _shared_channel_count(tiles)
    if not 0 <= reference_channel < channel_count:
        raise ValueError(f"reference_channel {reference_channel} is outside channel count {channel_count}")

    overlap = _as_nonnegative_int3(overlap_zyx, "overlap_zyx")
    _require_cuda()

    pair_shifts = _estimate_grid_pair_shifts(
        tile_by_index,
        overlap_zyx=overlap,
        reference_channel=reference_channel,
        use_log=use_log,
        coarse_max_size=coarse_max_size,
        fine_upsample_factor=fine_upsample_factor,
    )
    if nominal_origins_zyx is not None and nominal_fallback_min_ncc is not None:
        pair_shifts = _apply_nominal_pair_fallback(pair_shifts, nominal_origins_zyx, min_ncc=nominal_fallback_min_ncc)
    origins = _solve_origins(pair_shifts, root=min(tile_by_index))
    missing_origins = sorted(set(tile_by_index) - set(origins))
    if missing_origins:
        raise ValueError(f"Tile graph is disconnected; missing origins for {missing_origins}")
    return StitchPlacement(origins_zyx=origins, pair_shifts=tuple(pair_shifts))


def fuse_tiles(
    tiles: list[StitchTile],
    *,
    origins_zyx: dict[Index3, tuple[float, float, float]],
    overlap_zyx: Index3,
    flatfield_yx: np.ndarray | FusionFlatfield | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not tiles:
        raise ValueError("tiles must not be empty")
    tile_shape = _shared_tile_shape(tiles)
    channels = tiles[0].data_zcyx.shape[1]
    overlap = _as_nonnegative_int3(overlap_zyx, "overlap_zyx")
    tile_indices = {tile.index_zyx for tile in tiles}
    _validate_overlap_smaller_than_neighbor_axis(overlap, tile_shape, tile_indices)
    _require_cuda()
    missing_origins = sorted(tile_indices - set(origins_zyx))
    if missing_origins:
        raise ValueError(f"Missing origins for tiles: {missing_origins}")

    origin_values = np.asarray([origins_zyx[tile.index_zyx] for tile in tiles], dtype=np.float32)
    min_origin = np.floor(origin_values.min(axis=0)).astype(np.int64)
    max_extent = np.ceil(origin_values + np.asarray(tile_shape, dtype=np.float32)).max(axis=0).astype(np.int64)
    output_shape = tuple(int(v) for v in (max_extent - min_origin))
    fusion_flatfield = prepare_fusion_flatfield(flatfield_yx, tile_shape_zyx=tile_shape)
    return _fuse_tiles_gpu(
        tiles,
        origins_zyx,
        tile_indices,
        overlap,
        tile_shape,
        channels,
        min_origin,
        output_shape,
        fusion_flatfield=fusion_flatfield,
    )


def prepare_fusion_flatfield(
    flatfield_yx: np.ndarray | FusionFlatfield | None,
    *,
    tile_shape_zyx: Index3,
) -> FusionFlatfield | None:
    if flatfield_yx is None:
        return None
    if isinstance(flatfield_yx, FusionFlatfield):
        if flatfield_yx.shape_yx != tile_shape_zyx[1:3]:
            raise ValueError(
                f"flatfield shape {flatfield_yx.shape_yx} does not match tile YX shape {tile_shape_zyx[1:3]}"
            )
        return flatfield_yx

    flatfield = np.asarray(flatfield_yx, dtype=np.float32)
    validate_flatfield(flatfield)
    expected_shape_yx = tile_shape_zyx[1:3]
    if flatfield.shape != expected_shape_yx:
        raise ValueError(f"flatfield shape {flatfield.shape} does not match tile YX shape {expected_shape_yx}")
    cp = _cupy()
    inv_flat_gpu = 1.0 / cp.asarray(flatfield, dtype=cp.float32)
    return FusionFlatfield(inv_flat_gpu=inv_flat_gpu, shape_yx=expected_shape_yx)


def _require_cuda() -> None:
    try:
        cp = _cupy()
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError("CUDA stitching requires a CUDA-capable GPU") from exc
    if device_count < 1:
        raise RuntimeError("CUDA stitching requires a CUDA-capable GPU")


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CUDA stitching requires CuPy") from exc
    return cp


def _gpu_arrays(fixed: np.ndarray, moving: np.ndarray):
    cp = _cupy()
    return cp.asarray(fixed, dtype=cp.float32), cp.asarray(moving, dtype=cp.float32)


def _fuse_tiles_gpu(
    tiles: list[StitchTile],
    origins_zyx: dict[Index3, tuple[float, float, float]],
    tile_indices: set[Index3],
    overlap: Index3,
    tile_shape: Index3,
    channels: int,
    min_origin: np.ndarray,
    output_shape: tuple[int, int, int],
    *,
    fusion_flatfield: FusionFlatfield | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cp = _cupy()
    accum = cp.zeros((output_shape[0], channels, output_shape[1], output_shape[2]), dtype=cp.float32)
    weight_sum = cp.zeros(output_shape, dtype=cp.float32)

    for tile in tiles:
        origin = np.asarray(origins_zyx[tile.index_zyx], dtype=np.float32) - min_origin
        int_origin = np.floor(origin).astype(np.int64)
        frac = origin - int_origin
        weight = cp.asarray(_tile_weight(tile_shape, tile.index_zyx, tile_indices, overlap), dtype=cp.float32)
        shifted_weight = _shift_zyx_gpu(weight, frac)
        tile_gpu = cp.asarray(tile.data_zcyx, dtype=cp.float32)
        if fusion_flatfield is not None:
            tile_gpu *= fusion_flatfield.inv_flat_gpu[cp.newaxis, cp.newaxis, :, :]
        shifted_data = cp.empty(tile_gpu.shape, dtype=cp.float32)
        for channel in range(channels):
            shifted_data[:, channel] = _shift_zyx_gpu(tile_gpu[:, channel], frac)

        z0, y0, x0 = (int(v) for v in int_origin)
        z1, y1, x1 = z0 + tile_shape[0], y0 + tile_shape[1], x0 + tile_shape[2]
        accum[z0:z1, :, y0:y1, x0:x1] += shifted_data * shifted_weight[:, cp.newaxis]
        weight_sum[z0:z1, y0:y1, x0:x1] += shifted_weight

    accum /= cp.maximum(weight_sum[:, cp.newaxis, :, :], EPS)
    return cp.asnumpy(accum), cp.asnumpy(weight_sum)


def _estimate_grid_pair_shifts(
    tile_by_index: dict[Index3, StitchTile],
    *,
    overlap_zyx: Index3,
    reference_channel: int,
    use_log: bool,
    coarse_max_size: int,
    fine_upsample_factor: int,
) -> list[PairShift]:
    pair_shifts: list[PairShift] = []
    for index in sorted(tile_by_index):
        fixed = tile_by_index[index]
        for axis in range(3):
            neighbor = list(index)
            neighbor[axis] += 1
            moving_index = tuple(neighbor)
            if moving_index not in tile_by_index:
                continue
            delta, ncc = estimate_adjacent_shift(
                fixed.data_zcyx,
                tile_by_index[moving_index].data_zcyx,
                axis=axis,
                overlap_zyx=overlap_zyx,
                reference_channel=reference_channel,
                use_log=use_log,
                coarse_max_size=coarse_max_size,
                fine_upsample_factor=fine_upsample_factor,
            )
            pair_shifts.append(PairShift(index, moving_index, axis, delta, ncc))
    return pair_shifts


def _apply_nominal_pair_fallback(
    pair_shifts: list[PairShift],
    nominal_origins_zyx: dict[Index3, tuple[float, float, float]],
    *,
    min_ncc: float,
) -> list[PairShift]:
    adjusted = []
    for pair in pair_shifts:
        if pair.ncc >= min_ncc:
            adjusted.append(pair)
            continue
        try:
            fixed_origin = np.asarray(nominal_origins_zyx[pair.fixed_index_zyx], dtype=np.float32)
            moving_origin = np.asarray(nominal_origins_zyx[pair.moving_index_zyx], dtype=np.float32)
        except KeyError as exc:
            raise ValueError("nominal_origins_zyx must contain every low-NCC pair endpoint") from exc
        adjusted.append(
            PairShift(
                fixed_index_zyx=pair.fixed_index_zyx,
                moving_index_zyx=pair.moving_index_zyx,
                axis=pair.axis,
                origin_delta_zyx=tuple(float(v) for v in moving_origin - fixed_origin),
                ncc=pair.ncc,
            )
        )
    return adjusted


def _solve_origins(pair_shifts: list[PairShift], *, root: Index3) -> dict[Index3, tuple[float, float, float]]:
    indices = sorted({root} | {pair.fixed_index_zyx for pair in pair_shifts} | {pair.moving_index_zyx for pair in pair_shifts})
    index_to_row = {index: row for row, index in enumerate(indices)}
    adjacency: dict[Index3, list[Index3]] = {index: [] for index in indices}
    for pair in pair_shifts:
        adjacency[pair.fixed_index_zyx].append(pair.moving_index_zyx)
        adjacency[pair.moving_index_zyx].append(pair.fixed_index_zyx)
    connected = _connected_indices(adjacency, root)
    if len(connected) != len(indices):
        missing = sorted(set(indices) - connected)
        raise ValueError(f"Tile graph is disconnected; missing origins for {missing}")

    rows = []
    values = []
    for pair in pair_shifts:
        row = np.zeros(len(indices), dtype=np.float32)
        row[index_to_row[pair.moving_index_zyx]] = 1.0
        row[index_to_row[pair.fixed_index_zyx]] = -1.0
        rows.append(row)
        values.append(pair.origin_delta_zyx)

    root_row = np.zeros(len(indices), dtype=np.float32)
    root_row[index_to_row[root]] = 1.0
    rows.append(root_row)
    values.append((0.0, 0.0, 0.0))

    design = np.stack(rows)
    targets = np.asarray(values, dtype=np.float32)
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
    origins = {index: solution[row] for index, row in index_to_row.items()}
    min_origin = np.min(np.stack(list(origins.values())), axis=0)
    return {index: tuple(float(v) for v in origin - min_origin) for index, origin in origins.items()}


def _connected_indices(adjacency: dict[Index3, list[Index3]], root: Index3) -> set[Index3]:
    connected = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        for neighbor in adjacency.get(current, []):
            if neighbor in connected:
                continue
            connected.add(neighbor)
            stack.append(neighbor)
    return connected


def _phase_correlation_peak_gpu(fixed, moving, *, use_log: bool) -> tuple[int, int, int]:
    cp = _cupy()
    fixed_float = _phase_input_gpu(fixed, use_log=use_log)
    moving_float = _phase_input_gpu(moving, use_log=use_log)
    product_spectrum = cp.fft.fftn(fixed_float) * cp.conj(cp.fft.fftn(moving_float))
    product_spectrum /= cp.maximum(cp.abs(product_spectrum), EPS)
    phase = cp.abs(cp.fft.ifftn(product_spectrum))
    peak = cp.unravel_index(cp.argmax(phase), phase.shape)
    return tuple(int(cp.asnumpy(value)) for value in peak)


def _coarse_phase_inputs_gpu(fixed, moving, *, max_size: int):
    if max_size < 1:
        raise ValueError("max_size must be at least 1")
    if fixed.shape != moving.shape:
        raise ValueError(f"Expected matching coarse phase input shapes, got {fixed.shape} and {moving.shape}")
    steps = tuple(max(1, int(np.ceil(size / max_size))) for size in fixed.shape)
    slices = tuple(slice(None, None, step) for step in steps)
    return fixed[slices], moving[slices], steps


def _phase_input_gpu(stack, *, use_log: bool):
    cp = _cupy()
    data = stack.astype(cp.float32, copy=False)
    if use_log:
        data = data - cp.min(data)
        data = cp.log(cp.maximum(data, EPS))
    return data - cp.mean(data)


def _wrap_candidates(peak: tuple[int, int, int], shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    per_axis = []
    for value, size in zip(peak, shape, strict=True):
        if value == 0:
            per_axis.append((0,))
        else:
            per_axis.append((int(value), int(value - size)))
    return [tuple(candidate) for candidate in product(*per_axis)]


def _overlap_ncc_for_origin_delta_gpu(fixed, moving, origin_delta: np.ndarray) -> float:
    delta = np.rint(origin_delta).astype(np.int64)
    fixed_slices, moving_slices = _overlap_slices_for_delta(fixed.shape, moving.shape, delta)
    if fixed_slices is None or moving_slices is None:
        return -np.inf
    return _normalized_cross_correlation_gpu(fixed[fixed_slices], moving[moving_slices])


def _normalized_cross_correlation_gpu(a, b) -> float:
    cp = _cupy()
    if a.shape != b.shape:
        raise ValueError(f"Expected matching shapes for NCC, got {a.shape} and {b.shape}")
    a_float = a.astype(cp.float32, copy=False)
    b_float = b.astype(cp.float32, copy=False)
    a_centered = a_float - cp.mean(a_float)
    b_centered = b_float - cp.mean(b_float)
    denom = cp.linalg.norm(a_centered) * cp.linalg.norm(b_centered)
    if float(cp.asnumpy(denom)) == 0.0:
        return 0.0
    return float(cp.asnumpy(cp.sum(a_centered * b_centered) / denom))


def _refine_delta_subpixel(
    fixed,
    moving,
    origin_delta: np.ndarray,
    *,
    upsample_factor: int,
    use_log: bool,
) -> np.ndarray:
    integer_delta = np.rint(origin_delta).astype(np.int64)
    fixed_slices, moving_slices = _overlap_slices_for_delta(fixed.shape, moving.shape, integer_delta)
    if fixed_slices is None or moving_slices is None:
        return origin_delta
    return _refine_delta_subpixel_gpu(
        fixed[fixed_slices],
        moving[moving_slices],
        origin_delta,
        max_iterations=upsample_factor * 2,
    )


def _refine_delta_subpixel_gpu(
    fixed_overlap,
    moving_overlap,
    origin_delta: np.ndarray,
    *,
    max_iterations: int,
) -> np.ndarray:
    from .registration_gpu import register_stack_pair_gpu

    cp = _cupy()
    fixed_cpu = cp.asnumpy(fixed_overlap)[..., np.newaxis, :, :]
    moving_cpu = cp.asnumpy(moving_overlap)[..., np.newaxis, :, :]
    result = register_stack_pair_gpu(
        fixed_cpu,
        moving_cpu,
        mode="translation",
        max_iterations=max_iterations,
        downsample=1,
    )
    residual_shift = -result.output_to_input_offset_zyx.astype(np.float32, copy=False)
    return origin_delta + residual_shift


def _overlap_slices_for_delta(
    fixed_shape: tuple[int, int, int],
    moving_shape: tuple[int, int, int],
    delta: np.ndarray,
) -> tuple[tuple[slice, slice, slice] | None, tuple[slice, slice, slice] | None]:
    fixed_slices = []
    moving_slices = []
    for axis in range(3):
        fixed_start = max(0, int(delta[axis]))
        fixed_stop = min(fixed_shape[axis], int(delta[axis]) + moving_shape[axis])
        if fixed_stop <= fixed_start:
            return None, None
        moving_start = fixed_start - int(delta[axis])
        moving_stop = fixed_stop - int(delta[axis])
        fixed_slices.append(slice(fixed_start, fixed_stop))
        moving_slices.append(slice(moving_start, moving_stop))
    return tuple(fixed_slices), tuple(moving_slices)


def _expected_overlap_slab(
    stack_zyx: np.ndarray,
    *,
    axis: int,
    overlap: Index3,
    trailing: bool,
) -> np.ndarray:
    slices = []
    for dim in range(3):
        if dim == axis:
            slices.append(slice(-overlap[dim], None) if trailing else slice(0, overlap[dim]))
        else:
            slices.append(slice(None))
    return stack_zyx[tuple(slices)]


def _tile_weight(tile_shape: Index3, index: Index3, tile_indices: set[Index3], overlap: Index3) -> np.ndarray:
    weight = np.ones(tile_shape, dtype=np.float32)
    for axis in range(3):
        line = np.ones(tile_shape[axis], dtype=np.float32)
        overlap_size = overlap[axis]
        lower = list(index)
        lower[axis] -= 1
        upper = list(index)
        upper[axis] += 1
        if overlap_size > 0 and tuple(lower) in tile_indices:
            line[:overlap_size] *= np.linspace(0.0, 1.0, overlap_size, dtype=np.float32)
        if overlap_size > 0 and tuple(upper) in tile_indices:
            line[-overlap_size:] *= np.linspace(1.0, 0.0, overlap_size, dtype=np.float32)
        shape = [1, 1, 1]
        shape[axis] = tile_shape[axis]
        weight *= line.reshape(shape)
    return weight


def _shift_zyx_gpu(stack, shift_zyx: np.ndarray):
    if np.allclose(shift_zyx, 0.0):
        return stack.astype(_cupy().float32, copy=False)
    from cupyx.scipy.ndimage import shift as gpu_shift

    return gpu_shift(
        stack.astype(_cupy().float32, copy=False),
        shift=tuple(float(v) for v in shift_zyx),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _shared_tile_shape(tiles: list[StitchTile]) -> Index3:
    first = _as_zcyx(tiles[0].data_zcyx, "tile.data_zcyx")
    shape = first.shape[0:1] + first.shape[2:4]
    channels = first.shape[1]
    for tile in tiles:
        data = _as_zcyx(tile.data_zcyx, "tile.data_zcyx")
        if data.shape[1] != channels or data.shape[0:1] + data.shape[2:4] != shape:
            raise ValueError("All tiles must have matching ZCYX shape")
    return tuple(int(v) for v in shape)


def _shared_channel_count(tiles: list[StitchTile]) -> int:
    first = _as_zcyx(tiles[0].data_zcyx, "tile.data_zcyx")
    channels = first.shape[1]
    for tile in tiles:
        data = _as_zcyx(tile.data_zcyx, "tile.data_zcyx")
        if data.shape[1] != channels:
            raise ValueError("All tiles must have matching channel counts")
    return int(channels)


def _regular_grid_shape(volume_shape: Index3, tile_shape: Index3, stride: Index3) -> Index3:
    grid_shape = []
    for axis, (volume_size, tile_size, step) in enumerate(zip(volume_shape, tile_shape, stride, strict=True)):
        remaining = volume_size - tile_size
        if remaining < 0:
            raise ValueError(f"tile_shape_zyx[{axis}] is larger than the volume")
        if remaining % step:
            raise ValueError("tile shape and overlap do not exactly cover the volume with a regular grid")
        grid_shape.append(remaining // step + 1)
    return tuple(grid_shape)


def _stride_from_tile_overlap(tile_shape: Index3, overlap: Index3) -> Index3:
    stride = tuple(tile_shape[axis] - overlap[axis] for axis in range(3))
    if any(value <= 0 for value in stride):
        raise ValueError("overlap must be smaller than tile shape on every axis")
    return stride


def _validate_overlap_smaller_than_neighbor_axis(
    overlap: Index3,
    tile_shape: Index3,
    tile_indices: set[Index3],
) -> None:
    for axis, (overlap_size, tile_size) in enumerate(zip(overlap, tile_shape, strict=True)):
        has_axis_neighbor = any(_has_neighbor_along_axis(index, axis, tile_indices) for index in tile_indices)
        if has_axis_neighbor and overlap_size >= tile_size:
            raise ValueError(f"overlap_zyx[{axis}] must be smaller than tile size {tile_size}")


def _has_neighbor_along_axis(index: Index3, axis: int, tile_indices: set[Index3]) -> bool:
    lower = list(index)
    lower[axis] -= 1
    upper = list(index)
    upper[axis] += 1
    return tuple(lower) in tile_indices or tuple(upper) in tile_indices


def _as_zcyx(array: np.ndarray, name: str) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim != 4:
        raise ValueError(f"{name} must be a ZCYX array, got shape {data.shape}")
    return data


def _as_positive_int3(value: Index3, name: str) -> Index3:
    parsed = tuple(int(v) for v in value)
    if len(parsed) != 3 or any(v <= 0 for v in parsed):
        raise ValueError(f"{name} must contain three positive integers")
    return parsed


def _as_nonnegative_int3(value: Index3, name: str) -> Index3:
    parsed = tuple(int(v) for v in value)
    if len(parsed) != 3 or any(v < 0 for v in parsed):
        raise ValueError(f"{name} must contain three non-negative integers")
    return parsed
