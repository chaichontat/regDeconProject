from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import cupy as cp
import zarr
from click.testing import CliRunner
from PIL import Image
from scipy import ndimage

from registration_fusion.cli import main as click_main
from registration_fusion.core import register_stack_pair
from registration_fusion.czi_stitching import (
    CziMosaicTile,
    CziPlacement,
    CziStitchRun,
    CziStitchPlan,
    _append_completed_chunk,
    _array_nbytes,
    _check_output_disk_space,
    _czi_zarr_attrs,
    _jsonable_run,
    _output_bounds_from_origins,
    _read_completed_chunks,
    _require_resume_progress,
    _resolve_czi_output_channels,
    _validate_czi_channel_indices,
    _validate_czi_z_index,
    _validate_czi_z_range,
    _validate_czi_zarr_attrs,
    _validate_matching_czi_mosaic_geometry,
    _validate_matching_czi_plans,
    _validate_unique_channels,
    _validate_zero_z_origins,
    estimate_czi_stitch_placement,
    load_czi_placement,
    load_czi_placement_origins_csv,
    replace_czi_placement_origins,
    save_czi_placement,
    save_czi_placement_origins_csv,
    summarize_czi_stitch_plan,
    validate_czi_output_channel_count,
    write_czi_stitched_zarr,
)
from registration_fusion.tile_stitching import (
    _array_nbytes as _tile_array_nbytes,
    _check_output_disk_space as _check_tile_output_disk_space,
    DEFAULT_OME_FOLDER_PATTERN,
    OmeFolderPlacement,
    _read_ome_folder_tiles,
    _read_ome_tiff_zcyx_slab,
    build_ome_folder_plan,
    inspect_tiff_zcyx_shape,
    read_tile_manifest,
    summarize_tiff_tile_manifest,
)
from registration_fusion.deconvolution import (
    deconvolve_dual_view_gpu,
    default_dual_view_psfs,
    make_wb_projectors_gpu,
    register_translation_gpu,
)
from registration_fusion.flatfield import load_flatfield_profile
from registration_fusion.io import read_tiff_stack, read_tiff_zcyx, write_tiff_stack, write_tiff_zcyx
from registration_fusion.previews import max_projection_image
from registration_fusion.registration_gpu import register_stack_pair_gpu
from registration_fusion.streaming_registration import (
    _read_zarr_channel_slab,
    build_downsampled_channel_max_reference_zarr,
    inspect_zarr_zcyx,
    open_zarr_zcyx,
    select_zarr_reference_array_path,
)
from registration_fusion.register import main
from registration_fusion import video as video_module
from registration_fusion.video import write_zarr_slice_video
from registration_fusion.simulate_dual_view import (
    DEFAULT_PSF_PATH,
    PsfCropConfig,
    SimulationConfig,
    crop_psf_like_fishtools,
    generate_dataset,
    rotate_psf_for_view,
    transpose_psf_for_orthogonal_view,
)
from registration_fusion.stitching import (
    PairShift,
    StitchTile,
    _coarse_phase_inputs_gpu,
    _refine_delta_subpixel_gpu,
    estimate_tile_placement,
    estimate_adjacent_shift,
    fuse_tiles,
    make_overlapping_tiles,
    stitch_tiles,
)


def test_tiff_stack_round_trip_preserves_shape_and_dtype(tmp_path):
    stack = (np.arange(5 * 7 * 9).reshape(5, 7, 9) % 2048).astype(np.uint16)
    path = tmp_path / "stack.tif"

    write_tiff_stack(path, stack)

    restored = read_tiff_stack(path)
    assert restored.shape == stack.shape
    assert restored.dtype == np.uint16
    np.testing.assert_array_equal(restored, stack)


def test_rgb_max_projection_png_image_preserves_channel_layout():
    stack = np.zeros((4, 3, 6, 5), dtype=np.float32)
    stack[1, 0, 2, 3] = 10
    stack[2, 1, 1, 4] = 20
    stack[3, 2, 5, 0] = 30

    preview = max_projection_image(stack)

    assert preview.shape == (6, 5, 3)
    assert preview.dtype == np.uint8
    assert preview[2, 3, 0] == 255
    assert preview[1, 4, 1] == 255
    assert preview[5, 0, 2] == 255


def test_zarr_slice_video_streams_resized_frames(tmp_path, monkeypatch):
    data = np.zeros((6, 2, 20, 10), dtype=np.uint16)
    data[:, 0] = np.arange(6, dtype=np.uint16)[:, np.newaxis, np.newaxis]
    data[:, 1] = data[:, 0] + 10
    zarr_path = tmp_path / "input.zarr"
    _write_test_zarr(zarr_path, data)
    output_path = tmp_path / "out.mp4"
    processes = _capture_ffmpeg_processes(monkeypatch)

    result = write_zarr_slice_video(
        zarr_path,
        output_path,
        height=8,
        fps=2.0,
        max_frames=3,
        sample_frames=2,
        sample_stride=4,
        crf=22,
        preset="fast",
    )

    assert result.output_path == str(output_path)
    assert result.shape_zcyx == data.shape
    assert result.channels == (0, 1)
    assert result.frame_count == 3
    assert result.output_width == 4
    assert result.output_height == 8
    assert processes[0].args["width"] == 4
    assert processes[0].args["height"] == 8
    assert processes[0].args["input_pix_fmt"] == "gray"
    assert len(processes[0].stdin.data) == 3 * 4 * 8
    assert processes[0].stdin.closed


def test_zarr_slice_video_draws_scale_bar_as_rgb(tmp_path, monkeypatch):
    data = np.zeros((2, 1, 80, 160), dtype=np.uint16)
    zarr_path = tmp_path / "input.zarr"
    _write_test_zarr(zarr_path, data)
    output_path = tmp_path / "out.mp4"
    processes = _capture_ffmpeg_processes(monkeypatch)

    write_zarr_slice_video(
        zarr_path,
        output_path,
        height=80,
        width=160,
        max_frames=1,
        sample_frames=1,
        scale_bar_um=20,
        pixel_size_um=1.0,
    )

    encoded = bytes(processes[0].stdin.data)
    frame = np.frombuffer(encoded, dtype=np.uint8).reshape(80, 160, 3)
    assert processes[0].args["input_pix_fmt"] == "rgb24"
    assert len(encoded) == 80 * 160 * 3
    assert frame.max() >= 200
    assert processes[0].stdin.closed


def test_zarr_slice_video_infers_scale_bar_pixel_size_from_source_czi(tmp_path, monkeypatch):
    data = np.zeros((2, 1, 80, 160), dtype=np.uint16)
    zarr_path = tmp_path / "input.zarr"
    _write_test_zarr(zarr_path, data)
    zarr.open_array(zarr_path, mode="a").attrs["source_czi"] = "/data/source.czi"
    processes = _capture_ffmpeg_processes(monkeypatch)
    inferred_paths = []

    def fake_czi_x_pixel_size_um(path):
        inferred_paths.append(path)
        return 1.0

    monkeypatch.setattr(video_module, "_czi_x_pixel_size_um", fake_czi_x_pixel_size_um)

    write_zarr_slice_video(
        zarr_path,
        tmp_path / "out.mp4",
        height=80,
        width=160,
        max_frames=1,
        sample_frames=1,
        scale_bar_um=20,
    )

    assert inferred_paths == [Path("/data/source.czi")]
    assert processes[0].args["input_pix_fmt"] == "rgb24"


def test_zarr_slice_video_requires_pixel_size_for_scale_bar(tmp_path):
    data = np.zeros((2, 1, 80, 160), dtype=np.uint16)
    zarr_path = tmp_path / "input.zarr"
    _write_test_zarr(zarr_path, data)

    with pytest.raises(ValueError, match="scale_bar_um requires pixel_size_um"):
        write_zarr_slice_video(
            zarr_path,
            tmp_path / "out.mp4",
            height=80,
            width=160,
            max_frames=1,
            sample_frames=1,
            scale_bar_um=20,
        )


def test_translation_registration_improves_correlation():
    fixed = _synthetic_stack()
    moving = ndimage.shift(fixed, shift=(2, -3, 4), order=1, mode="constant", cval=0)
    before = _corr(fixed, moving)

    result = register_stack_pair(
        fixed,
        moving,
        mode="translation",
        max_iterations=0,
    )

    assert result.registered.shape == fixed.shape
    assert result.final_metric_value > before + 0.25
    assert result.final_metric_value > 0.9
    np.testing.assert_allclose(result.initial_shift_zyx, (-2, 3, -4), atol=1)


def test_cli_registers_single_pair_without_timepoints(tmp_path):
    fixed = _synthetic_stack().astype(np.uint16)
    moving = ndimage.shift(fixed, shift=(1, -2, 3), order=1, mode="constant", cval=0).astype(np.uint16)
    tifffile.imwrite(tmp_path / "StackA.tif", fixed)
    tifffile.imwrite(tmp_path / "StackB.tif", moving)
    output_path = tmp_path / "registered.tif"

    status = main(
        [
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--mode",
            "translation",
            "--max-iterations",
            "0",
        ]
    )

    assert status == 0
    output = tifffile.imread(output_path)
    assert output.shape == fixed.shape
    assert output.dtype == np.uint16
    assert _corr(fixed, output) > _corr(fixed, moving)


def test_invalid_registration_mode_has_clear_error():
    fixed = _synthetic_stack()
    with pytest.raises(ValueError, match="Unsupported registration mode"):
        register_stack_pair(fixed, fixed, mode="bad-mode")


def test_fishtools_psf_crop_matches_expected_shape_and_normalization():
    psf = crop_psf_like_fishtools(DEFAULT_PSF_PATH)

    assert psf.shape == (7, 31, 31)
    np.testing.assert_allclose(float(psf.sum()), 1.0, rtol=1e-6)


def test_view_angle_rotates_second_view_psf():
    psf = crop_psf_like_fishtools(DEFAULT_PSF_PATH, PsfCropConfig(max_z=3, size=9))

    rotated = rotate_psf_for_view(psf, 45.0)
    transposed = transpose_psf_for_orthogonal_view(psf)

    assert rotated.ndim == 3
    assert rotated.shape[0] > psf.shape[0]
    assert rotated.shape[1] > psf.shape[0]
    np.testing.assert_allclose(float(rotated.sum()), 1.0, rtol=1e-6)
    assert transposed.shape == (psf.shape[1], psf.shape[0], psf.shape[2])
    np.testing.assert_allclose(transposed, np.swapaxes(psf, 0, 1))


def test_dual_view_simulator_is_deterministic_and_page_major(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(12, 32, 40, 2),
        spots_per_channel=5,
        seed=123,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_dataset(first, config)
    generate_dataset(second, config)

    view_a = tifffile.imread(first / "view_a.tif")
    view_b = tifffile.imread(first / "view_b.tif")
    view_b_object_grid = tifffile.imread(first / "view_b_object_grid.tif")
    assert view_a.shape == (24, 32, 40)
    assert view_b.shape == (64, 12, 40)
    assert view_b_object_grid.shape == (24, 32, 40)
    np.testing.assert_array_equal(view_a, tifffile.imread(second / "view_a.tif"))
    np.testing.assert_array_equal(view_b, tifffile.imread(second / "view_b.tif"))
    np.testing.assert_array_equal(view_b_object_grid, tifffile.imread(second / "view_b_object_grid.tif"))
    assert (first / "spots.csv").read_text() == (second / "spots.csv").read_text()
    assert (first / "ground_truth.tif").exists()
    assert Image.open(first / "ground_truth.png").size == (40, 32)
    assert Image.open(first / "view_a.png").size == (40, 32)
    assert Image.open(first / "view_b.png").size == (40, 12)
    assert Image.open(first / "view_b_object_grid.png").size == (40, 32)
    np.testing.assert_allclose(
        tifffile.imread(first / "psf_cropped_view_b.tif"),
        np.swapaxes(tifffile.imread(first / "psf_cropped_view_a.tif"), 0, 1),
    )
    np.testing.assert_allclose(
        tifffile.imread(first / "psf_cropped_view_b_native.tif"),
        tifffile.imread(first / "psf_cropped_view_a.tif"),
    )

    with tifffile.TiffFile(first / "view_a.tif") as tif:
        metadata = tif.shaped_metadata[0]
    assert metadata["simulation"]["shape_zyxc"] == [12, 32, 40, 2]
    assert metadata["simulation"]["view_angle_deg"] == 90.0
    assert metadata["coordinate_system"] == "view_a_objective_native"
    assert metadata["channels"] == ["channel_0", "channel_1"]
    assert metadata["plane_order"] == "z-major, channel-minor; reshape pages to (z, c, y, x)"
    with tifffile.TiffFile(first / "view_b.tif") as tif:
        metadata_b = tif.shaped_metadata[0]
    assert metadata_b["coordinate_system"] == "view_b_objective_native"
    assert metadata_b["object_axis_mapping"] == {"z": "object_y", "y": "object_z", "x": "object_x"}


def test_stitching_tiles_reconstruct_simulated_volume_with_3d_overlap(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(24, 48, 56, 2),
        spots_per_channel=80,
        seed=19,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, config)
    volume = read_tiff_zcyx(dataset / "view_a.tif", channels=2).astype(np.float32)
    tile_shape = (16, 30, 34)
    overlap = (8, 12, 12)
    tiles = make_overlapping_tiles(volume, tile_shape_zyx=tile_shape, overlap_zyx=overlap)

    result = stitch_tiles(tiles, overlap_zyx=overlap, reference_channel=0, fine_upsample_factor=10)

    assert result.fused_zcyx.shape == volume.shape
    assert len(result.pair_shifts) == 12
    assert np.min(result.weight_sum_zyx) >= 0.0
    assert np.count_nonzero(result.weight_sum_zyx) == np.prod(volume.shape[0:1] + volume.shape[2:4])
    assert _corr(volume, result.fused_zcyx) > 0.995


def test_adjacent_shift_uses_ncc_to_select_expected_3d_tile_offset(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(24, 48, 56, 1),
        spots_per_channel=80,
        seed=23,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, config)
    volume = read_tiff_zcyx(dataset / "view_a.tif", channels=1).astype(np.float32)
    tile_shape = (16, 30, 34)
    overlap = (8, 12, 12)
    tiles = make_overlapping_tiles(volume, tile_shape_zyx=tile_shape, overlap_zyx=overlap)
    tile_by_index = {tile.index_zyx: tile for tile in tiles}

    delta, ncc = estimate_adjacent_shift(
        tile_by_index[(0, 0, 0)].data_zcyx,
        tile_by_index[(0, 0, 1)].data_zcyx,
        axis=2,
        overlap_zyx=overlap,
        reference_channel=0,
    )

    np.testing.assert_allclose(delta, (0, 0, 22), atol=1)
    assert ncc > 0.95


def test_adjacent_shift_downsamples_coarse_phase_inputs_before_ncc():
    rng = np.random.default_rng(41)
    volume = rng.normal(size=(8, 1, 32, 88)).astype(np.float32)
    volume = ndimage.gaussian_filter(volume, sigma=(0.5, 0.0, 1.0, 1.0)).astype(np.float32)
    fixed = volume[:, :, :, 0:48]
    moving = volume[:, :, :, 20:68]

    delta, ncc = estimate_adjacent_shift(
        fixed,
        moving,
        axis=2,
        overlap_zyx=(4, 16, 32),
        reference_channel=0,
        coarse_max_size=8,
    )

    np.testing.assert_allclose(delta, (0, 0, 20), atol=1)
    assert ncc > 0.95


def test_coarse_phase_inputs_use_gpu_strided_downsampling():
    fixed = cp.zeros((8, 32, 48), dtype=cp.float32)
    moving = cp.zeros_like(fixed)

    coarse_fixed, coarse_moving, steps = _coarse_phase_inputs_gpu(fixed, moving, max_size=16)

    assert coarse_fixed.shape == (8, 16, 16)
    assert coarse_moving.shape == (8, 16, 16)
    assert steps == (1, 2, 3)


def test_fine_refinement_passes_iteration_budget_to_gpu_registration(monkeypatch):
    calls = []

    class FakeRegistrationResult:
        output_to_input_offset_zyx = np.asarray([0.0, 0.0, -0.25], dtype=np.float32)

    def fake_register_stack_pair_gpu(_fixed, _moving, *, mode, max_iterations, downsample):
        calls.append((mode, max_iterations, downsample))
        return FakeRegistrationResult()

    monkeypatch.setattr(
        "registration_fusion.registration_gpu.register_stack_pair_gpu",
        fake_register_stack_pair_gpu,
    )
    origin = np.asarray([0.0, 0.0, 20.0], dtype=np.float32)

    refined = _refine_delta_subpixel_gpu(
        cp.zeros((4, 8, 8), dtype=cp.float32),
        cp.zeros((4, 8, 8), dtype=cp.float32),
        origin,
        max_iterations=12,
    )

    assert calls == [("translation", 12, 1)]
    np.testing.assert_allclose(refined, [0.0, 0.0, 20.25])


def test_stitching_validates_reference_channel_even_for_single_tile():
    tile = StitchTile((0, 0, 0), np.zeros((4, 1, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="reference_channel 1 is outside channel count 1"):
        stitch_tiles([tile], overlap_zyx=(1, 2, 2), reference_channel=1)


def test_estimate_tile_placement_does_not_fuse_tiles(monkeypatch):
    tiles = [
        StitchTile((0, 0, 0), np.zeros((2, 1, 4, 4), dtype=np.float32)),
        StitchTile((0, 0, 1), np.zeros((2, 1, 4, 4), dtype=np.float32)),
    ]

    monkeypatch.setattr("registration_fusion.stitching._require_cuda", lambda: None)
    monkeypatch.setattr(
        "registration_fusion.stitching._estimate_grid_pair_shifts",
        lambda *_args, **_kwargs: [PairShift((0, 0, 0), (0, 0, 1), 2, (0.0, 0.0, 3.0), 0.95)],
    )
    monkeypatch.setattr(
        "registration_fusion.stitching.fuse_tiles",
        lambda *_args, **_kwargs: pytest.fail("placement estimation must not fuse tiles"),
    )

    placement = estimate_tile_placement(tiles, overlap_zyx=(1, 1, 1), reference_channel=0)

    assert placement.origins_zyx == {(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 3.0)}
    assert len(placement.pair_shifts) == 1


def test_estimate_tile_placement_allows_zero_overlap_on_non_neighbor_axes(monkeypatch):
    tiles = [
        StitchTile((0, 0, 0), np.zeros((2, 1, 4, 4), dtype=np.float32)),
        StitchTile((0, 0, 1), np.zeros((2, 1, 4, 4), dtype=np.float32)),
    ]

    monkeypatch.setattr("registration_fusion.stitching._require_cuda", lambda: None)
    monkeypatch.setattr(
        "registration_fusion.stitching.estimate_adjacent_shift",
        lambda *_args, **_kwargs: ((0.0, 0.0, 2.0), 0.9),
    )

    placement = estimate_tile_placement(tiles, overlap_zyx=(1, 0, 2), reference_channel=0)

    assert placement.origins_zyx[(0, 0, 1)] == (0.0, 0.0, 2.0)


def test_estimate_tile_placement_rejects_disconnected_tile_graph(monkeypatch):
    tiles = [
        StitchTile((0, 0, 0), np.zeros((2, 1, 4, 4), dtype=np.float32)),
        StitchTile((0, 0, 2), np.zeros((2, 1, 4, 4), dtype=np.float32)),
    ]

    monkeypatch.setattr("registration_fusion.stitching._require_cuda", lambda: None)

    with pytest.raises(ValueError, match="Tile graph is disconnected"):
        estimate_tile_placement(tiles, overlap_zyx=(1, 1, 1), reference_channel=0)


def test_fuse_tiles_allows_non_neighbor_axis_overlap_at_least_tile_shape():
    tile = StitchTile((0, 0, 0), np.zeros((4, 1, 8, 8), dtype=np.float32))

    fused, weight_sum = fuse_tiles(
        [tile],
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0)},
        overlap_zyx=(4, 2, 2),
    )

    assert fused.shape == tile.data_zcyx.shape
    assert weight_sum.shape == tile.data_zcyx.shape[0:1] + tile.data_zcyx.shape[2:4]


def test_fuse_tiles_applies_flatfield_inverse_on_gpu():
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device is not available")

    flatfield = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    tile = StitchTile((0, 0, 0), flatfield[np.newaxis, np.newaxis].copy())

    fused, weight_sum = fuse_tiles(
        [tile],
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0)},
        overlap_zyx=(1, 1, 1),
        flatfield_yx=flatfield,
    )

    np.testing.assert_allclose(fused, np.ones_like(fused), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(weight_sum, np.ones_like(weight_sum), rtol=1e-6, atol=1e-6)


def test_load_flatfield_profile_reads_basic_pickle_without_darkfield(tmp_path):
    path = tmp_path / "basic.pkl"
    flatfield = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    payload = {"basic": SimpleNamespace(flatfield=flatfield, darkfield=np.full((2, 2), 999.0))}
    path.write_bytes(pickle.dumps(payload))

    profile = load_flatfield_profile(path)

    assert profile.source_path == str(path)
    np.testing.assert_array_equal(profile.flatfield_yx, flatfield)


def test_fuse_tiles_rejects_neighbor_axis_overlap_at_least_tile_shape_before_gpu_allocation():
    tiles = [
        StitchTile((0, 0, 0), np.zeros((4, 1, 8, 8), dtype=np.float32)),
        StitchTile((1, 0, 0), np.zeros((4, 1, 8, 8), dtype=np.float32)),
    ]

    with pytest.raises(ValueError, match=r"overlap_zyx\[0\] must be smaller than tile size 4"):
        fuse_tiles(
            tiles,
            origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (1, 0, 0): (4.0, 0.0, 0.0)},
            overlap_zyx=(4, 2, 2),
        )


def test_stitching_uses_cuda_backend(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(24, 48, 56, 1),
        spots_per_channel=20,
        seed=29,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, config)
    volume = read_tiff_zcyx(dataset / "view_a.tif", channels=1).astype(np.float32)
    tiles = make_overlapping_tiles(volume, tile_shape_zyx=(16, 30, 34), overlap_zyx=(8, 12, 12))

    assert cp.cuda.runtime.getDeviceCount() > 0
    result = stitch_tiles(tiles, overlap_zyx=(8, 12, 12))
    assert _corr(volume, result.fused_zcyx) > 0.995


def test_stitching_supports_connected_irregular_tile_grid(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(24, 48, 56, 1),
        spots_per_channel=80,
        seed=31,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, config)
    volume = read_tiff_zcyx(dataset / "view_a.tif", channels=1).astype(np.float32)
    full_tiles = make_overlapping_tiles(volume, tile_shape_zyx=(16, 30, 34), overlap_zyx=(8, 12, 12))
    irregular_tiles = [tile for tile in full_tiles if tile.index_zyx != (0, 0, 0)]

    result = stitch_tiles(irregular_tiles, overlap_zyx=(8, 12, 12), reference_channel=0)

    assert set(result.origins_zyx) == {tile.index_zyx for tile in irregular_tiles}
    assert len(result.pair_shifts) == 9
    assert result.fused_zcyx.shape[1] == 1
    assert np.count_nonzero(result.weight_sum_zyx) > 0


def test_click_stitch_tiles_cli_stitches_tiff_manifest(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(24, 48, 56, 1),
        spots_per_channel=80,
        seed=37,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, config)
    volume = read_tiff_zcyx(dataset / "view_a.tif", channels=1).astype(np.float32)
    tiles = make_overlapping_tiles(volume, tile_shape_zyx=(16, 30, 34), overlap_zyx=(8, 12, 12))
    manifest = tmp_path / "tiles.csv"
    with manifest.open("w") as handle:
        handle.write("path,index_z,index_y,index_x\n")
        for tile in tiles:
            z, y, x = tile.index_zyx
            path = tmp_path / f"tile_{z}_{y}_{x}.tif"
            write_tiff_zcyx(path, tile.data_zcyx, bit_depth=32)
            handle.write(f"{path.name},{z},{y},{x}\n")
    output = tmp_path / "stitched.zarr"
    metadata = tmp_path / "stitched.json"

    result = CliRunner().invoke(
        click_main,
        [
            "stitch-tiles",
            str(manifest),
            "--output",
            str(output),
            "--metadata-output",
            str(metadata),
            "--channels",
            "1",
            "--overlap-z",
            "8",
            "--overlap-y",
            "12",
            "--overlap-x",
            "12",
            "--coarse-max-size",
            "512",
            "--bit-depth",
            "32",
        ],
    )

    assert result.exit_code == 0, result.output
    stitched_array = zarr.open_array(output, mode="r")
    stitched = stitched_array[:]
    assert stitched.shape == volume.shape
    assert _corr(volume, stitched) > 0.995
    attrs = stitched_array.attrs.asdict()
    assert attrs["registration_fusion_schema"] == "tiff_tiles_stitched_zcyx_v1"
    assert attrs["axes"] == ["z", "c", "y", "x"]
    assert attrs["source_manifest"] == str(manifest)
    assert attrs["overlap_zyx"] == [8, 12, 12]
    assert attrs["reference_channel"] == 0
    assert attrs["coarse_max_size"] == 512
    metadata_text = metadata.read_text()
    assert '"pair_count": 12' in metadata_text
    assert '"coarse_max_size": 512' in metadata_text
    assert '"estimated_output_bytes": 258048' in metadata_text


def test_click_inspect_tile_manifest_estimates_output_size(tmp_path):
    volume = np.zeros((8, 1, 12, 14), dtype=np.float32)
    tiles = make_overlapping_tiles(volume, tile_shape_zyx=(6, 8, 9), overlap_zyx=(4, 4, 4))
    manifest = tmp_path / "tiles.csv"
    with manifest.open("w") as handle:
        handle.write("path,index_z,index_y,index_x\n")
        for tile in tiles:
            z, y, x = tile.index_zyx
            path = tmp_path / f"tile_{z}_{y}_{x}.tif"
            write_tiff_zcyx(path, tile.data_zcyx, bit_depth=32)
            handle.write(f"{path.name},{z},{y},{x}\n")
    metadata = tmp_path / "inspect.json"

    result = CliRunner().invoke(
        click_main,
        [
            "inspect-tile-manifest",
            str(manifest),
            "--metadata-output",
            str(metadata),
            "--channels",
            "1",
            "--overlap-z",
            "4",
            "--overlap-y",
            "4",
            "--overlap-x",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = summarize_tiff_tile_manifest(manifest, channels=1, overlap_zyx=(4, 4, 4))
    assert summary["estimated_output_shape_zcyx"] == [8, 1, 12, 14]
    assert '"estimated_output_bytes": 2688' in metadata.read_text()


def _write_positioned_ome_tile(
    path: Path,
    data_zyx: np.ndarray,
    *,
    position_y: float | None,
    position_x: float | None,
    physical_size_y: float = 0.5,
    physical_size_x: float = 0.25,
) -> None:
    z_size, y_size, x_size = data_zyx.shape
    plane_attrs = ""
    if position_y is not None and position_x is not None:
        plane_attrs = f' PositionY="{position_y}" PositionX="{position_x}"'
    ome_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0">'
        f'<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeX="{x_size}" SizeY="{y_size}" '
        f'SizeZ="{z_size}" SizeC="1" SizeT="1" PhysicalSizeY="{physical_size_y}" '
        f'PhysicalSizeX="{physical_size_x}">'
        '<Channel ID="Channel:0:0" SamplesPerPixel="1"/>'
        "<TiffData/>"
        f'<Plane TheZ="0" TheC="0" TheT="0"{plane_attrs}/>'
        "</Pixels></Image></OME>"
    )
    tifffile.imwrite(
        path,
        data_zyx.astype(np.uint16, copy=False),
        description=ome_xml,
        photometric="minisblack",
    )


def test_ome_folder_plan_infers_indices_origins_and_overlap(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    data = np.zeros((3, 8, 10), dtype=np.uint16)
    for row in range(2):
        for col in range(2):
            _write_positioned_ome_tile(
                folder / f"20x-EdUTest-514-L3.{row}{col}.ome.tif",
                data + row * 10 + col,
                position_y=row * 6 * 0.5,
                position_x=col * 7 * 0.25,
            )
    (folder / "20x-EdUTest-514-L3.stub.ome.tif").write_bytes(b"0" * 16)
    _write_positioned_ome_tile(
        folder / "other.ome.tif",
        data,
        position_y=999.0,
        position_x=999.0,
    )

    plan = build_ome_folder_plan(folder)

    assert [tile.index_zyx for tile in plan.tiles] == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    assert [tile.nominal_origin_zyx for tile in plan.tiles] == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 7.0),
        (0.0, 6.0, 0.0),
        (0.0, 6.0, 7.0),
    ]
    assert plan.tile_shape_zcyx == (3, 1, 8, 10)
    assert plan.overlap_zyx == (2, 2, 3)
    assert plan.pattern == DEFAULT_OME_FOLDER_PATTERN


def test_ome_folder_plan_rejects_missing_and_duplicate_positions(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    _write_positioned_ome_tile(
        missing / "20x-EdUTest-514-L3.000.ome.tif",
        np.zeros((2, 4, 5), dtype=np.uint16),
        position_y=None,
        position_x=None,
    )
    with pytest.raises(ValueError, match="PositionY/PositionX"):
        build_ome_folder_plan(missing)

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    for index in range(2):
        _write_positioned_ome_tile(
            duplicate / f"20x-EdUTest-514-L3.00{index}.ome.tif",
            np.zeros((2, 4, 5), dtype=np.uint16),
            position_y=1.0,
            position_x=2.0,
        )
    with pytest.raises(ValueError, match="Duplicate OME tile position"):
        build_ome_folder_plan(duplicate)


def test_ome_folder_tile_reader_reads_requested_z_slab(tmp_path, monkeypatch):
    folder = tmp_path / "tiles"
    folder.mkdir()
    data = np.arange(4 * 3 * 5, dtype=np.uint16).reshape(4, 3, 5)
    _write_positioned_ome_tile(
        folder / "20x-EdUTest-514-L3.000.ome.tif",
        data,
        position_y=0.0,
        position_x=0.0,
    )
    plan = build_ome_folder_plan(folder)
    requested = []

    original = tifffile.TiffPage.asarray

    def record_asarray(self, *args, **kwargs):
        requested.append(self.index)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffPage, "asarray", record_asarray)

    tiles = _read_ome_folder_tiles(plan, channel=0, z_indices=(1, 3))

    assert requested == [1, 3]
    assert tiles[0].data_zcyx.shape == (2, 1, 3, 5)
    np.testing.assert_array_equal(tiles[0].data_zcyx[:, 0], data[[1, 3]])


def test_ome_4d_slab_reader_fails_instead_of_materializing_full_tile(tmp_path):
    path = tmp_path / "tile_4d.ome.tif"
    tifffile.imwrite(path, np.zeros((3, 2, 4, 5), dtype=np.uint16))

    with pytest.raises(ValueError, match="streaming 4D OME slab reads are not implemented"):
        _read_ome_tiff_zcyx_slab(path, channel=0, z_indices=(0,))


def test_click_stitch_ome_folder_writes_metadata_and_passes_flatfield_to_fusion(tmp_path, monkeypatch):
    folder = tmp_path / "ome_tiles"
    folder.mkdir()
    _write_positioned_ome_tile(
        folder / "20x-EdUTest-514-L3.000.ome.tif",
        np.full((2, 4, 5), 10, dtype=np.uint16),
        position_y=0.0,
        position_x=0.0,
    )
    _write_positioned_ome_tile(
        folder / "20x-EdUTest-514-L3.001.ome.tif",
        np.full((2, 4, 5), 20, dtype=np.uint16),
        position_y=0.0,
        position_x=3 * 0.25,
    )
    flatfield = np.full((4, 5), 2.0, dtype=np.float32)
    profile = tmp_path / "basic.pkl"
    profile.write_bytes(pickle.dumps({"basic": SimpleNamespace(flatfield=flatfield, darkfield=np.full((4, 5), 999.0))}))
    output = tmp_path / "stitched.zarr"
    metadata = tmp_path / "stitched.json"
    captured = {}

    def fake_estimate_ome_folder_stitch_placement(folder_arg, **kwargs):
        plan = build_ome_folder_plan(
            folder_arg,
            pattern=kwargs["pattern"],
            overlap_z=kwargs["overlap_z"],
            overlap_y=kwargs["overlap_y"],
            overlap_x=kwargs["overlap_x"],
        )
        return OmeFolderPlacement(
            plan=plan,
            reference_z_indices=(0,),
            origins_zyx={tile.index_zyx: (0.0, 0.0, 0.0) for tile in plan.tiles},
            pair_shifts=(),
            fallback_min_ncc=kwargs["fallback_min_ncc"],
            estimate_seconds=0.0,
            reference_channel=kwargs["reference_channel"],
            coarse_max_size=kwargs["coarse_max_size"],
            fine_upsample_factor=kwargs["fine_upsample_factor"],
        )

    def fake_prepare_fusion_flatfield(flatfield_yx, *, tile_shape_zyx):
        captured["prepared_flatfield"] = flatfield_yx
        captured["prepared_tile_shape"] = tile_shape_zyx
        return flatfield_yx

    def fake_fuse_tiles(tiles, *, origins_zyx, overlap_zyx, flatfield_yx=None):
        captured["flatfield_yx"] = flatfield_yx
        captured["tile_values"] = [float(tile.data_zcyx[0, 0, 0, 0]) for tile in tiles]
        corrected = tiles[0].data_zcyx * (1.0 / flatfield_yx[np.newaxis, np.newaxis])
        return corrected.astype(np.float32, copy=False), np.ones(corrected.shape[0:1] + corrected.shape[2:4], dtype=np.float32)

    monkeypatch.setattr(
        "registration_fusion.tile_stitching.estimate_ome_folder_stitch_placement",
        fake_estimate_ome_folder_stitch_placement,
    )
    monkeypatch.setattr("registration_fusion.mosaic_streaming.prepare_fusion_flatfield", fake_prepare_fusion_flatfield)
    monkeypatch.setattr("registration_fusion.tile_stitching.fuse_tiles", fake_fuse_tiles)

    result = CliRunner().invoke(
        click_main,
        [
            "stitch-ome-folder",
            str(folder),
            "--output",
            str(output),
            "--metadata-output",
            str(metadata),
            "--basic-profile",
            str(profile),
            "--bit-depth",
            "32",
        ],
    )

    assert result.exit_code == 0, result.output
    np.testing.assert_array_equal(captured["flatfield_yx"], flatfield)
    assert captured["tile_values"] == [10.0, 20.0]
    np.testing.assert_array_equal(zarr.open_array(output, mode="r")[:], np.full((2, 1, 4, 5), 5.0, dtype=np.float32))
    written = json.loads(metadata.read_text())
    assert written["folder"] == str(folder)
    assert written["pattern"] == DEFAULT_OME_FOLDER_PATTERN
    assert written["tile_count"] == 2
    assert written["overlap_zyx"] == [1, 0, 2]
    assert written["flatfield_correction"]["source_path"] == str(profile)
    assert written["flatfield_correction"]["shape_yx"] == [4, 5]
    assert written["output_channels"] == [0]
    assert written["placement_reference_channel"] == 0


def test_ome_folder_reused_placement_metadata_uses_placement_parameters(tmp_path, monkeypatch):
    folder = tmp_path / "ome_tiles"
    folder.mkdir()
    _write_positioned_ome_tile(
        folder / "20x-EdUTest-514-L3.000.ome.tif",
        np.full((2, 4, 5), 10, dtype=np.uint16),
        position_y=0.0,
        position_x=0.0,
    )
    plan = build_ome_folder_plan(folder)
    placement = OmeFolderPlacement(
        plan=plan,
        reference_z_indices=(0, 1),
        origins_zyx={plan.tiles[0].index_zyx: (0.0, 0.0, 0.0)},
        pair_shifts=(),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
        reference_channel=0,
        coarse_max_size=2048,
        fine_upsample_factor=10,
    )
    placement_path = tmp_path / "placement.json"
    metadata = tmp_path / "stitched.json"
    output = tmp_path / "stitched.zarr"
    captured = {}

    from registration_fusion.tile_stitching import save_ome_folder_placement

    save_ome_folder_placement(placement_path, placement)

    def fake_fuse_tiles(tiles, *, origins_zyx, overlap_zyx):
        captured["tile_values"] = [float(tile.data_zcyx[0, 0, 0, 0]) for tile in tiles]
        return tiles[0].data_zcyx.astype(np.float32, copy=False), np.ones((1, 4, 5), dtype=np.float32)

    monkeypatch.setattr("registration_fusion.tile_stitching.fuse_tiles", fake_fuse_tiles)

    result = CliRunner().invoke(
        click_main,
        [
            "stitch-ome-folder",
            str(folder),
            "--output",
            str(output),
            "--metadata-output",
            str(metadata),
            "--placement-input",
            str(placement_path),
            "--coarse-max-size",
            "1024",
            "--fine-upsample-factor",
            "1",
            "--bit-depth",
            "32",
            "--chunk-depth",
            "1",
            "--z-stop",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["tile_values"] == [10.0]
    written = json.loads(metadata.read_text())
    assert written["fallback_min_ncc"] == 0.8
    assert written["coarse_max_size"] == 2048
    assert written["fine_upsample_factor"] == 10
    assert written["placement_reference_channel"] == 0


def test_tile_output_disk_space_check_rejects_obviously_too_large_output(tmp_path, monkeypatch):
    class DiskUsage:
        free = 99

    monkeypatch.setattr("registration_fusion.mosaic_streaming.shutil.disk_usage", lambda _path: DiskUsage())

    assert _tile_array_nbytes((10, 2, 20, 35), np.uint16) == 10 * 2 * 20 * 35 * 2
    with pytest.raises(ValueError, match="Insufficient free space"):
        _check_tile_output_disk_space(tmp_path / "out.zarr", 100)


def test_inspect_tiff_zcyx_shape_uses_metadata_for_page_major_channels(tmp_path):
    path = tmp_path / "tile.tif"
    write_tiff_zcyx(path, np.zeros((5, 2, 7, 9), dtype=np.float32), bit_depth=32)

    shape, dtype = inspect_tiff_zcyx_shape(path, channels=2)

    assert shape == (5, 2, 7, 9)
    assert dtype == np.dtype(np.float32)


def test_4d_tiff_channel_count_must_match_requested_channels(tmp_path):
    path = tmp_path / "tile_4d.tif"
    tifffile.imwrite(path, np.zeros((5, 2, 7, 9), dtype=np.float32))

    assert inspect_tiff_zcyx_shape(path, channels=2)[0] == (5, 2, 7, 9)
    assert read_tiff_zcyx(path, channels=2).shape == (5, 2, 7, 9)
    with pytest.raises(ValueError, match="has 2 channels, not 3"):
        inspect_tiff_zcyx_shape(path, channels=3)
    with pytest.raises(ValueError, match="has 2 channels, not 3"):
        read_tiff_zcyx(path, channels=3)


def test_click_stitch_tiles_cli_requires_overwrite_for_existing_output(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(24, 48, 56, 1),
        spots_per_channel=80,
        seed=41,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, config)
    volume = read_tiff_zcyx(dataset / "view_a.tif", channels=1).astype(np.float32)
    tiles = make_overlapping_tiles(volume, tile_shape_zyx=(16, 30, 34), overlap_zyx=(8, 12, 12))
    manifest = tmp_path / "tiles.csv"
    with manifest.open("w") as handle:
        handle.write("path,index_z,index_y,index_x\n")
        for tile in tiles:
            z, y, x = tile.index_zyx
            path = tmp_path / f"tile_{z}_{y}_{x}.tif"
            write_tiff_zcyx(path, tile.data_zcyx, bit_depth=32)
            handle.write(f"{path.name},{z},{y},{x}\n")
    output = tmp_path / "stitched.zarr"
    args = [
        "stitch-tiles",
        str(manifest),
        "--output",
        str(output),
        "--channels",
        "1",
        "--overlap-z",
        "8",
        "--overlap-y",
        "12",
        "--overlap-x",
        "12",
        "--bit-depth",
        "32",
    ]

    first = CliRunner().invoke(click_main, args)
    second = CliRunner().invoke(click_main, args)
    third = CliRunner().invoke(click_main, [*args, "--overwrite"])

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert "already exists" in second.output
    assert third.exit_code == 0, third.output


def test_tile_manifest_rejects_duplicate_tile_indices(tmp_path):
    tile_path = tmp_path / "tile.tif"
    write_tiff_zcyx(tile_path, np.zeros((2, 1, 4, 4), dtype=np.float32), bit_depth=32)
    manifest = tmp_path / "tiles.csv"
    manifest.write_text(
        "path,index_z,index_y,index_x\n"
        "tile.tif,0,0,0\n"
        "tile.tif,0,0,0\n"
    )

    with pytest.raises(ValueError, match="duplicates tile index"):
        read_tile_manifest(manifest)


def test_czi_placement_json_round_trip_preserves_full_mosaic_layout(tmp_path):
    tiles = (
        CziMosaicTile((0, 0, 1), mosaic_index=3, x=100, y=0, width=1920, height=1920),
        CziMosaicTile((0, 1, 0), mosaic_index=4, x=0, y=1630, width=1920, height=1920),
    )
    plan = CziStitchPlan(
        czi_path="/data/acquisition.czi",
        channel=0,
        z_size=548,
        tile_shape_zyx=(1, 1920, 1920),
        output_shape_yx=(3550, 3940),
        overlap_zyx=(4, 290, 288),
        tile_count=2,
        tiles=tiles,
        nominal_origins_zyx={
            (0, 0, 1): (0.0, 0.0, 100.0),
            (0, 1, 0): (0.0, 1630.0, 0.0),
        },
    )
    placement = CziPlacement(
        plan=plan,
        reference_z_indices=(244, 245, 246, 247),
        origins_zyx={
            (0, 0, 1): (0.0, 0.25, 99.5),
            (0, 1, 0): (0.0, 1629.75, 0.0),
        },
        pair_shifts=(
            PairShift(
                (0, 0, 1),
                (0, 1, 0),
                1,
                (0.0, 1629.5, -99.5),
                0.91,
            ),
        ),
        fallback_min_ncc=0.8,
        coarse_max_size=512,
        fine_upsample_factor=10,
        estimate_seconds=12.5,
    )

    path = tmp_path / "placement.json"
    save_czi_placement(path, placement)
    loaded = load_czi_placement(path)

    assert loaded.plan.z_size == 548
    assert loaded.plan.tiles == tiles
    assert loaded.plan.nominal_origins_zyx == plan.nominal_origins_zyx
    assert loaded.reference_z_indices == placement.reference_z_indices
    assert loaded.origins_zyx == placement.origins_zyx
    assert loaded.pair_shifts == placement.pair_shifts
    assert loaded.coarse_max_size == 512
    assert loaded.fine_upsample_factor == 10
    payload = json.loads(path.read_text())
    assert payload["placement"]["fine_upsample_factor"] == 10


def test_czi_placement_json_marks_missing_legacy_fine_factor_as_unknown(tmp_path):
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=0,
            z_size=10,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 20),
            overlap_zyx=(1, 5, 5),
            tile_count=1,
            tiles=(CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0)},
        pair_shifts=(),
        fallback_min_ncc=0.8,
        coarse_max_size=512,
        estimate_seconds=1.0,
    )
    path = tmp_path / "legacy_placement.json"
    save_czi_placement(path, placement)
    payload = json.loads(path.read_text())
    del payload["placement"]["fine_upsample_factor"]
    path.write_text(json.dumps(payload))

    loaded = load_czi_placement(path)

    assert loaded.coarse_max_size == 512
    assert loaded.fine_upsample_factor is None


def test_czi_placement_origin_csv_round_trip_supports_manual_edits(tmp_path):
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=0,
            z_size=10,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(35, 35),
            overlap_zyx=(1, 5, 5),
            tile_count=2,
            tiles=(
                CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
                CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
            ),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.25, 14.75)},
        pair_shifts=(PairShift((0, 0, 0), (0, 0, 1), 2, (0.0, 0.25, 14.75), 0.95),),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )
    csv_path = tmp_path / "origins.csv"

    save_czi_placement_origins_csv(csv_path, placement)
    text = csv_path.read_text()
    csv_path.write_text(text.replace("0.25,14.75", "0.5,14.25"))
    origins = load_czi_placement_origins_csv(csv_path)
    updated = replace_czi_placement_origins(placement, origins)

    assert updated.origins_zyx[(0, 0, 1)] == (0.0, 0.5, 14.25)
    assert updated.pair_shifts == placement.pair_shifts
    with pytest.raises(ValueError, match="indices do not match"):
        replace_czi_placement_origins(placement, {(0, 0, 0): (0.0, 0.0, 0.0)})


def test_czi_plan_validation_rejects_wrong_mosaic_geometry():
    plan = CziStitchPlan(
        czi_path="/data/acquisition.czi",
        channel=0,
        z_size=10,
        tile_shape_zyx=(1, 20, 20),
        output_shape_yx=(20, 35),
        overlap_zyx=(1, 5, 5),
        tile_count=2,
        tiles=(
            CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
            CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
        ),
        nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
    )
    shifted_plan = CziStitchPlan(
        czi_path="/data/other.czi",
        channel=0,
        z_size=10,
        tile_shape_zyx=(1, 20, 20),
        output_shape_yx=(20, 36),
        overlap_zyx=(1, 5, 5),
        tile_count=2,
        tiles=(
            CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
            CziMosaicTile((0, 0, 1), mosaic_index=1, x=16, y=0, width=20, height=20),
        ),
        nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 16.0)},
    )

    _validate_matching_czi_plans(plan, plan)
    with pytest.raises(ValueError, match="does not match"):
        _validate_matching_czi_plans(plan, shifted_plan)


def test_czi_output_channel_geometry_validation_ignores_channel_only():
    plan = CziStitchPlan(
        czi_path="/data/acquisition.czi",
        channel=0,
        z_size=10,
        tile_shape_zyx=(1, 20, 20),
        output_shape_yx=(20, 35),
        overlap_zyx=(1, 5, 5),
        tile_count=2,
        tiles=(
            CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
            CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
        ),
        nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
    )
    same_geometry_other_channel = CziStitchPlan(
        czi_path="/data/acquisition.czi",
        channel=1,
        z_size=10,
        tile_shape_zyx=(1, 20, 20),
        output_shape_yx=(20, 35),
        overlap_zyx=(1, 5, 5),
        tile_count=2,
        tiles=plan.tiles,
        nominal_origins_zyx=plan.nominal_origins_zyx,
    )
    shifted_geometry = CziStitchPlan(
        czi_path="/data/acquisition.czi",
        channel=1,
        z_size=10,
        tile_shape_zyx=(1, 20, 20),
        output_shape_yx=(20, 36),
        overlap_zyx=(1, 5, 5),
        tile_count=2,
        tiles=(
            CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
            CziMosaicTile((0, 0, 1), mosaic_index=1, x=16, y=0, width=20, height=20),
        ),
        nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 16.0)},
    )

    _validate_matching_czi_mosaic_geometry(plan, same_geometry_other_channel)
    with pytest.raises(ValueError, match="output channel mosaic geometry"):
        _validate_matching_czi_mosaic_geometry(plan, shifted_geometry)


def test_czi_channel_validation_rejects_out_of_range_channels():
    dims = {"C": (0, 2)}

    _validate_czi_channel_indices((0, 1), dims=dims)
    with pytest.raises(ValueError, match=r"outside available channel range \[0, 2\)"):
        _validate_czi_channel_indices((2,), dims=dims)


def test_czi_z_index_validation_rejects_out_of_range_reference_slices():
    _validate_czi_z_index(0, z_size=5, name="z_sample")
    _validate_czi_z_index(4, z_size=5, name="z_sample")
    with pytest.raises(ValueError, match=r"outside available Z range \[0, 5\)"):
        _validate_czi_z_index(-1, z_size=5, name="z_sample")
    with pytest.raises(ValueError, match=r"outside available Z range \[0, 5\)"):
        _validate_czi_z_index(5, z_size=5, name="z_sample")


def test_czi_z_range_validation_rejects_out_of_range_writer_requests():
    assert _validate_czi_z_range(z_start=0, z_stop=None, z_size=5) == 5
    assert _validate_czi_z_range(z_start=1, z_stop=5, z_size=5) == 5
    with pytest.raises(ValueError, match=r"z_start -1 is outside available Z range \[0, 5\)"):
        _validate_czi_z_range(z_start=-1, z_stop=1, z_size=5)
    with pytest.raises(ValueError, match=r"z_stop 6 is outside available Z range \[0, 5\]"):
        _validate_czi_z_range(z_start=0, z_stop=6, z_size=5)
    with pytest.raises(ValueError, match="Requested z range is empty"):
        _validate_czi_z_range(z_start=3, z_stop=3, z_size=5)


def test_czi_placement_estimation_rejects_reference_z_count_overrun(monkeypatch):
    plan = CziStitchPlan(
        czi_path="/data/acquisition.czi",
        channel=0,
        z_size=5,
        tile_shape_zyx=(1, 20, 20),
        output_shape_yx=(20, 35),
        overlap_zyx=(1, 5, 5),
        tile_count=2,
        tiles=(
            CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
            CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
        ),
        nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
    )
    monkeypatch.setattr("registration_fusion.czi_stitching.build_czi_stitch_plan", lambda *_args, **_kwargs: plan)

    with pytest.raises(ValueError, match=r"z_stop 6 is outside available Z range \[0, 5\]"):
        estimate_czi_stitch_placement(
            Path("/data/acquisition.czi"),
            channel=0,
            reference_z_start=4,
            reference_z_count=2,
            overlap_zyx=(1, 5, 5),
            fallback_min_ncc=0.8,
        )


def test_czi_output_channel_count_validation_rejects_unavailable_inspect_channels(monkeypatch):
    class FakeCziFile:
        def __init__(self, _path):
            pass

        def get_dims_shape(self):
            return [{"C": (0, 2)}]

    monkeypatch.setattr("registration_fusion.czi_stitching.CziFile", FakeCziFile)

    validate_czi_output_channel_count(Path("sample.czi"), start_channel=0, channel_count=2)
    with pytest.raises(ValueError, match=r"outside available channel range \[0, 2\)"):
        validate_czi_output_channel_count(Path("sample.czi"), start_channel=0, channel_count=3)


def test_czi_output_channels_must_be_unique():
    _validate_unique_channels((0, 1))
    with pytest.raises(ValueError, match="must be unique"):
        _validate_unique_channels((0, 0))


def test_czi_output_channels_default_to_reused_placement_channel():
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=1,
            z_size=10,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 20),
            overlap_zyx=(1, 5, 5),
            tile_count=1,
            tiles=(CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0)},
        pair_shifts=(),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )

    assert _resolve_czi_output_channels(output_channels=None, placement=placement) == (1,)
    assert _resolve_czi_output_channels(output_channels=(0, 1), placement=placement) == (0, 1)
    with pytest.raises(ValueError, match="must not be empty"):
        _resolve_czi_output_channels(output_channels=(), placement=placement)


def test_czi_output_disk_space_check_rejects_obviously_too_large_output(tmp_path, monkeypatch):
    class DiskUsage:
        free = 99

    monkeypatch.setattr("registration_fusion.mosaic_streaming.shutil.disk_usage", lambda _path: DiskUsage())

    assert _array_nbytes((10, 2, 20, 35), np.uint16) == 10 * 2 * 20 * 35 * 2
    with pytest.raises(ValueError, match="Insufficient free space"):
        _check_output_disk_space(tmp_path / "out.zarr", 100)


def test_click_stitch_czi_rejects_zero_max_chunks(tmp_path):
    czi_path = tmp_path / "placeholder.czi"
    czi_path.write_bytes(b"")
    result = CliRunner().invoke(
        click_main,
        [
            "stitch-czi",
            str(czi_path),
            "--benchmark-only",
            "--max-chunks",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "max_chunks must be positive" in result.output


def test_click_stitch_czi_resume_missing_output_reports_error_without_traceback(tmp_path, monkeypatch):
    czi_path = tmp_path / "placeholder.czi"
    czi_path.write_bytes(b"")
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path=str(czi_path),
            channel=0,
            z_size=4,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 35),
            overlap_zyx=(1, 5, 5),
            tile_count=2,
            tiles=(
                CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
                CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
            ),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        pair_shifts=(PairShift((0, 0, 0), (0, 0, 1), 2, (0.0, 0.0, 15.0), 0.95),),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )
    placement_path = tmp_path / "placement.json"
    save_czi_placement(placement_path, placement)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_placement_for_file", lambda *_args: None)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_output_channels", lambda *_args: None)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_output_channel_mosaic_geometry", lambda *_args: None)

    result = CliRunner().invoke(
        click_main,
        [
            "stitch-czi",
            str(czi_path),
            "--output",
            str(tmp_path / "missing.zarr"),
            "--placement-input",
            str(placement_path),
            "--resume",
        ],
    )

    assert result.exit_code != 0
    assert "output Zarr is missing" in result.output
    assert "Traceback" not in result.output


def test_click_estimate_czi_placement_reports_validation_error_without_traceback(tmp_path):
    czi_path = tmp_path / "placeholder.czi"
    czi_path.write_bytes(b"")

    result = CliRunner().invoke(
        click_main,
        [
            "estimate-czi-placement",
            str(czi_path),
            "--output",
            str(tmp_path / "placement.json"),
            "--reference-z-count",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "reference_z_count must be positive" in result.output
    assert "Traceback" not in result.output


def test_czi_stitch_plan_summary_estimates_output_bytes():
    plan = CziStitchPlan(
        czi_path="/data/acquisition.czi",
        channel=0,
        z_size=10,
        tile_shape_zyx=(1, 20, 20),
        output_shape_yx=(20, 35),
        overlap_zyx=(1, 5, 5),
        tile_count=2,
        tiles=(
            CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
            CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
        ),
        nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
    )

    summary = summarize_czi_stitch_plan(plan, channels=2, bit_depth=16)

    assert summary["nominal_output_shape_zcyx"] == [10, 2, 20, 35]
    assert summary["nominal_output_bytes"] == 10 * 2 * 20 * 35 * 2


def test_czi_zarr_attrs_capture_and_validate_stitch_identity():
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=0,
            z_size=10,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 35),
            overlap_zyx=(1, 5, 5),
            tile_count=2,
            tiles=(
                CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
                CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
            ),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.25, 14.75)},
        pair_shifts=(PairShift((0, 0, 0), (0, 0, 1), 2, (0.0, 0.25, 14.75), 0.95),),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )
    attrs = _czi_zarr_attrs(
        placement,
        (0, 1),
        (0, 0, 0),
        source_czi="/data/current_acquisition.czi",
        z_start=2,
        z_stop=8,
    )

    assert attrs["axes"] == ["z", "c", "y", "x"]
    assert attrs["source_czi"] == "/data/current_acquisition.czi"
    assert attrs["output_channels"] == [0, 1]
    assert attrs["source_z_start"] == 2
    assert attrs["source_z_stop"] == 8
    assert attrs["placement_reference_z_indices"] == [0, 1]
    assert attrs["placement_origin_records"] == [
        {"index_zyx": [0, 0, 0], "origin_zyx": [0.0, 0.0, 0.0]},
        {"index_zyx": [0, 0, 1], "origin_zyx": [0.0, 0.25, 14.75]},
    ]
    _validate_czi_zarr_attrs(attrs, attrs)
    bad_attrs = {**attrs, "output_channels": [1, 0]}
    with pytest.raises(ValueError, match="output_channels"):
        _validate_czi_zarr_attrs(bad_attrs, attrs)
    bad_z_attrs = {**attrs, "source_z_start": 0}
    with pytest.raises(ValueError, match="source_z_start"):
        _validate_czi_zarr_attrs(bad_z_attrs, attrs)
    bad_source_attrs = {**attrs, "source_czi": "/data/other.czi"}
    with pytest.raises(ValueError, match="source_czi"):
        _validate_czi_zarr_attrs(bad_source_attrs, attrs)
    bad_origin_attrs = {
        **attrs,
        "placement_origin_records": [
            {"index_zyx": [0, 0, 0], "origin_zyx": [0.0, 0.0, 0.0]},
            {"index_zyx": [0, 0, 1], "origin_zyx": [0.0, 0.0, 14.75]},
        ],
    }
    with pytest.raises(ValueError, match="placement_origin_records"):
        _validate_czi_zarr_attrs(bad_origin_attrs, attrs)


def test_czi_run_metadata_records_output_channel_geometry_validation():
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=1,
            z_size=10,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 35),
            overlap_zyx=(1, 5, 5),
            tile_count=2,
            tiles=(
                CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
                CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
            ),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.25, 14.75)},
        pair_shifts=(PairShift((0, 0, 0), (0, 0, 1), 2, (0.0, 0.25, 14.75), 0.95),),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )
    run = CziStitchRun(
        source_czi="/data/current_acquisition.czi",
        output_zarr="/data/stitched.zarr",
        metadata_path="/data/stitched.json",
        progress_path="/data/stitched.progress.jsonl",
        preview_path=None,
        placement=placement,
        placement_estimated_this_run=False,
        output_channels=(0, 1),
        output_shape_zcyx=(8, 2, 20, 35),
        output_origin_zyx=(0, 0, 0),
        output_dtype="uint16",
        z_start=2,
        z_stop=10,
        chunk_depth=2,
        max_chunks=1,
        preview_downsample=4,
        total_chunks=4,
        chunks_written=1,
        chunks_skipped=0,
        read_seconds=1.0,
        fuse_seconds=2.0,
        write_seconds=3.0,
        total_seconds=4.0,
        benchmark_only=True,
    )

    metadata = _jsonable_run(run)

    assert metadata["output_channel_mosaic_geometry_validation"] == {
        "status": "passed",
        "placement_channel": 1,
        "validated_output_channels": [0, 1],
        "matched_fields": [
            "z_size",
            "tile_shape_zyx",
            "overlap_zyx",
            "tile_count",
            "tile_indices",
            "mosaic_indices",
            "tile_bounding_boxes",
        ],
    }
    assert metadata["max_chunks"] == 1
    assert metadata["preview_downsample"] == 4


def test_czi_benchmark_only_defaults_progress_log_to_metadata_sidecar(tmp_path, monkeypatch):
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=0,
            z_size=4,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 35),
            overlap_zyx=(1, 5, 5),
            tile_count=2,
            tiles=(
                CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
                CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
            ),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        pair_shifts=(PairShift((0, 0, 0), (0, 0, 1), 2, (0.0, 0.0, 15.0), 0.95),),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )

    monkeypatch.setattr("registration_fusion.czi_stitching.CziFile", lambda _path: object())
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_placement_for_file", lambda *_args: None)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_output_channels", lambda *_args: None)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_output_channel_mosaic_geometry", lambda *_args: None)
    monkeypatch.setattr(
        "registration_fusion.czi_stitching._read_czi_tiles",
        lambda *_args, **kwargs: [{"depth": len(kwargs["z_indices"])}],
    )

    def fake_fuse_tiles(tiles, *, origins_zyx, overlap_zyx):
        return np.zeros((tiles[0]["depth"], 1, 20, 35), dtype=np.float32), None

    monkeypatch.setattr("registration_fusion.czi_stitching.fuse_tiles", fake_fuse_tiles)

    metadata_path = tmp_path / "benchmark.json"
    run = write_czi_stitched_zarr(
        tmp_path / "acquisition.czi",
        None,
        channel=0,
        output_channels=(0, 1),
        placement=placement,
        z_start=0,
        z_stop=None,
        chunk_depth=2,
        reference_z_start=0,
        reference_z_count=2,
        overlap_zyx=(1, 5, 5),
        fallback_min_ncc=0.8,
        metadata_path=metadata_path,
        benchmark_only=True,
    )

    progress_path = tmp_path / "benchmark.progress.jsonl"
    records = [json.loads(line) for line in progress_path.read_text().splitlines()]
    metadata = json.loads(metadata_path.read_text())

    assert run.progress_path == str(progress_path)
    assert metadata["progress_path"] == str(progress_path)
    assert records == [
        {
            "z_start": 0,
            "z_stop": 2,
            "output_shape_zcyx": [4, 2, 20, 35],
            "output_channels": [0, 1],
            "chunk_depth": 2,
            "wrote_output": False,
        },
        {
            "z_start": 2,
            "z_stop": 4,
            "output_shape_zcyx": [4, 2, 20, 35],
            "output_channels": [0, 1],
            "chunk_depth": 2,
            "wrote_output": False,
        },
    ]
    with pytest.raises(ValueError, match="benchmark-only"):
        _read_completed_chunks(
            progress_path,
            output_shape_zcyx=(4, 2, 20, 35),
            output_channels=(0, 1),
            chunk_depth=2,
        )


def test_czi_benchmark_only_writes_preview_and_metadata_path(tmp_path, monkeypatch):
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=0,
            z_size=2,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 35),
            overlap_zyx=(1, 5, 5),
            tile_count=2,
            tiles=(
                CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
                CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
            ),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        pair_shifts=(PairShift((0, 0, 0), (0, 0, 1), 2, (0.0, 0.0, 15.0), 0.95),),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )
    monkeypatch.setattr("registration_fusion.czi_stitching.CziFile", lambda _path: object())
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_placement_for_file", lambda *_args: None)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_output_channels", lambda *_args: None)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_output_channel_mosaic_geometry", lambda *_args: None)
    monkeypatch.setattr(
        "registration_fusion.czi_stitching._read_czi_tiles",
        lambda *_args, **kwargs: [{"depth": len(kwargs["z_indices"])}],
    )

    def fake_fuse_tiles(tiles, *, origins_zyx, overlap_zyx):
        return np.ones((tiles[0]["depth"], 1, 20, 35), dtype=np.float32), None

    monkeypatch.setattr("registration_fusion.czi_stitching.fuse_tiles", fake_fuse_tiles)

    metadata_path = tmp_path / "benchmark.json"
    preview_path = tmp_path / "preview.png"
    run = write_czi_stitched_zarr(
        tmp_path / "acquisition.czi",
        None,
        channel=0,
        output_channels=(0, 1),
        placement=placement,
        z_start=0,
        z_stop=None,
        chunk_depth=2,
        reference_z_start=0,
        reference_z_count=2,
        overlap_zyx=(1, 5, 5),
        fallback_min_ncc=0.8,
        metadata_path=metadata_path,
        benchmark_only=True,
        preview_path=preview_path,
        preview_downsample=5,
    )

    metadata = json.loads(metadata_path.read_text())
    preview = Image.open(preview_path)

    assert run.preview_path == str(preview_path)
    assert metadata["preview_path"] == str(preview_path)
    assert preview.size == (7, 4)


def test_czi_chunked_writer_requires_zero_z_origins():
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=0,
            z_size=10,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 35),
            overlap_zyx=(1, 5, 5),
            tile_count=2,
            tiles=(
                CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
                CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
            ),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.5, 0.25, 14.75)},
        pair_shifts=(PairShift((0, 0, 0), (0, 0, 1), 2, (0.5, 0.25, 14.75), 0.95),),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )

    with pytest.raises(ValueError, match="zero Z origins"):
        _validate_zero_z_origins(placement)


def test_czi_progress_log_tracks_completed_resume_chunks(tmp_path):
    path = tmp_path / "stitch.progress.jsonl"
    output_shape = (548, 2, 10102, 6824)
    channels = (0, 1)

    _append_completed_chunk(
        path,
        z_start=0,
        z_stop=2,
        output_shape_zcyx=output_shape,
        output_channels=channels,
        chunk_depth=2,
    )
    _append_completed_chunk(
        path,
        z_start=2,
        z_stop=4,
        output_shape_zcyx=output_shape,
        output_channels=channels,
        chunk_depth=2,
    )

    completed = _read_completed_chunks(
        path,
        output_shape_zcyx=output_shape,
        output_channels=channels,
        chunk_depth=2,
    )

    assert completed == {(0, 2), (2, 4)}
    with pytest.raises(ValueError, match="different output channels"):
        _read_completed_chunks(
            path,
            output_shape_zcyx=output_shape,
            output_channels=(0,),
            chunk_depth=2,
        )


def test_czi_resume_progress_log_must_exist(tmp_path):
    missing = tmp_path / "missing.progress.jsonl"

    with pytest.raises(ValueError, match="progress log is missing"):
        _require_resume_progress(tmp_path / "output.zarr", missing)


def test_czi_resume_requires_existing_output_zarr(tmp_path, monkeypatch):
    placement = CziPlacement(
        plan=CziStitchPlan(
            czi_path="/data/acquisition.czi",
            channel=0,
            z_size=4,
            tile_shape_zyx=(1, 20, 20),
            output_shape_yx=(20, 35),
            overlap_zyx=(1, 5, 5),
            tile_count=2,
            tiles=(
                CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=20, height=20),
                CziMosaicTile((0, 0, 1), mosaic_index=1, x=15, y=0, width=20, height=20),
            ),
            nominal_origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        ),
        reference_z_indices=(0, 1),
        origins_zyx={(0, 0, 0): (0.0, 0.0, 0.0), (0, 0, 1): (0.0, 0.0, 15.0)},
        pair_shifts=(PairShift((0, 0, 0), (0, 0, 1), 2, (0.0, 0.0, 15.0), 0.95),),
        fallback_min_ncc=0.8,
        estimate_seconds=1.0,
    )
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_placement_for_file", lambda *_args: None)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_output_channels", lambda *_args: None)
    monkeypatch.setattr("registration_fusion.czi_stitching.validate_czi_output_channel_mosaic_geometry", lambda *_args: None)

    with pytest.raises(ValueError, match="output Zarr is missing"):
        write_czi_stitched_zarr(
            tmp_path / "acquisition.czi",
            tmp_path / "missing.zarr",
            channel=0,
            output_channels=(0,),
            placement=placement,
            z_start=0,
            z_stop=2,
            chunk_depth=2,
            reference_z_start=0,
            reference_z_count=2,
            overlap_zyx=(1, 5, 5),
            fallback_min_ncc=0.8,
            resume=True,
        )


def test_czi_output_bounds_include_negative_solved_origins():
    tiles = (
        CziMosaicTile((0, 0, 0), mosaic_index=0, x=0, y=0, width=50, height=100),
        CziMosaicTile((0, 1, 0), mosaic_index=1, x=0, y=80, width=50, height=100),
    )
    origins = {
        (0, 0, 0): (0.0, -0.25, 3.5),
        (0, 1, 0): (0.0, 80.2, -1.2),
    }

    origin, shape_yx = _output_bounds_from_origins(tiles, origins)

    assert origin == (0, -1, -2)
    assert shape_yx == (182, 56)


def test_gpu_translation_registration_uses_shared_shift_for_channels():
    fixed = np.stack([_synthetic_stack(), _synthetic_stack() * 0.5], axis=1)
    moving = ndimage.shift(fixed, shift=(2, 0, -3, 4), order=1, mode="constant", cval=0)

    result = register_translation_gpu(fixed, moving)

    np.testing.assert_allclose(result.shift_zyx, (-2, 3, -4), atol=1)
    assert result.registered.shape == fixed.shape
    assert _corr(fixed, result.registered) > _corr(fixed, moving)


def test_gpu_rigid_registration_runs_affine_resampling_path():
    fixed = _synthetic_stack()[:16, np.newaxis, :32, :32]
    moving = ndimage.shift(fixed, shift=(1, 0, -2, 3), order=1, mode="constant", cval=0)

    result = register_stack_pair_gpu(fixed, moving, mode="rigid", max_iterations=3, downsample=1)

    assert result.mode == "rigid"
    assert result.output_to_input_matrix_zyx.shape == (3, 3)
    assert result.output_to_input_offset_zyx.shape == (3,)
    assert _corr(fixed, result.registered) > _corr(fixed, moving)


def test_gpu_affine_12dof_uses_matlab_progressive_stages():
    fixed = _synthetic_stack()[:16, np.newaxis, :32, :32]
    moving = ndimage.shift(fixed, shift=(1, 0, -2, 3), order=1, mode="constant", cval=0)

    result = register_stack_pair_gpu(fixed, moving, mode="affine-12dof", max_iterations=1, downsample=1)

    assert result.stage_modes == ("translation", "rigid", "scale-9dof", "affine-12dof")
    assert result.output_to_input_matrix_zyx.shape == (3, 3)
    assert result.output_to_input_offset_zyx.shape == (3,)
    assert _corr(fixed, result.registered) > _corr(fixed, moving)


def test_gpu_dual_view_deconvolution_returns_finite_zcyx_stack():
    view_a = _synthetic_stack()[4:16, np.newaxis, :32, :32]
    view_b = ndimage.gaussian_filter(view_a, sigma=(1.5, 0, 0.4, 0.4))
    psf_a, psf_b = default_dual_view_psfs(DEFAULT_PSF_PATH, PsfCropConfig(max_z=3, size=9))

    deconvolved = deconvolve_dual_view_gpu(view_a, view_b, psf_a, psf_b, iterations=1)

    assert deconvolved.shape == view_a.shape
    assert deconvolved.dtype == np.float32
    assert np.all(np.isfinite(deconvolved))
    assert float(deconvolved.max()) > float(deconvolved.min())


def test_wb_back_projector_is_centered_like_matlab():
    psf = _anisotropic_gaussian_psf()

    _, back = make_wb_projectors_gpu(psf)

    back_zyx = cp.asnumpy(back[:, 0])
    assert np.unravel_index(np.argmax(back_zyx), back_zyx.shape) == tuple(size // 2 for size in back_zyx.shape)


def test_gpu_dual_view_deconvolution_improves_reconstruction_metrics():
    ground_truth, view_a, view_b, psf_a, psf_b, spot_coords = _spot_deconvolution_fixture()
    raw_fused = (view_a + view_b) * 0.5

    deconvolved = deconvolve_dual_view_gpu(view_a, view_b, psf_a, psf_b, iterations=1)

    assert _corr(ground_truth, deconvolved) > _corr(ground_truth, raw_fused) + 0.15
    assert _mean_fwhm_sum(deconvolved, spot_coords) < 0.75 * _mean_fwhm_sum(raw_fused, spot_coords)
    assert max(_spot_localization_errors(deconvolved, spot_coords, radius=2)) <= 1.0


def test_click_dual_view_cli_writes_deconvolved_and_metadata(tmp_path):
    fixed = _synthetic_stack()[:12, :32, :32].astype(np.uint16)
    moving = ndimage.shift(fixed, shift=(1, -2, 3), order=1, mode="constant", cval=0).astype(np.uint16)
    fixed_path = tmp_path / "view_a.tif"
    moving_path = tmp_path / "view_b.tif"
    output_path = tmp_path / "decon.tif"
    registered_path = tmp_path / "registered_b.tif"
    metadata_path = tmp_path / "run.json"
    tifffile.imwrite(fixed_path, fixed)
    tifffile.imwrite(moving_path, moving)

    result = CliRunner().invoke(
        click_main,
        [
            "dual-view",
            "--fixed",
            str(fixed_path),
            "--moving",
            str(moving_path),
            "--output",
            str(output_path),
            "--registered-output",
            str(registered_path),
            "--metadata-output",
            str(metadata_path),
            "--iterations",
            "1",
            "--psf-max-z",
            "3",
            "--psf-size",
            "9",
            "--bit-depth",
            "32",
            "--view-angle-deg",
            "45",
        ],
    )

    assert result.exit_code == 0, result.output
    assert tifffile.imread(output_path).shape == (12, 32, 32)
    assert tifffile.imread(registered_path).shape == (12, 32, 32)
    for suffix in ["view_a", "view_b", "deconvolved"]:
        png = output_path.with_name(f"{output_path.stem}__{suffix}_max_projection.png")
        assert Image.open(png).size == (32, 32)
    metadata = metadata_path.read_text()
    assert '"mode": "gpu-wb-dual-view-rld"' in metadata
    assert '"view_angle_deg": 45.0' in metadata
    assert '"deconvolved_max_projection"' in metadata


def test_click_register_cli_writes_registered_stack_and_transform_metadata(tmp_path):
    fixed = _synthetic_stack()[:12, :32, :32].astype(np.uint16)
    moving = ndimage.shift(fixed, shift=(1, -2, 3), order=1, mode="constant", cval=0).astype(np.uint16)
    fixed_path = tmp_path / "view_a.tif"
    moving_path = tmp_path / "view_b.tif"
    output_path = tmp_path / "registered.tif"
    metadata_path = tmp_path / "registration.json"
    tifffile.imwrite(fixed_path, fixed)
    tifffile.imwrite(moving_path, moving)

    result = CliRunner().invoke(
        click_main,
        [
            "register",
            "--fixed",
            str(fixed_path),
            "--moving",
            str(moving_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_path),
            "--mode",
            "translation",
            "--max-iterations",
            "0",
            "--bit-depth",
            "16",
        ],
    )

    assert result.exit_code == 0, result.output
    registered = tifffile.imread(output_path)
    assert registered.shape == fixed.shape
    assert _corr(fixed, registered) > _corr(fixed, moving)
    for suffix in ["fixed", "moving", "registered"]:
        png = output_path.with_name(f"{output_path.stem}__{suffix}_max_projection.png")
        assert Image.open(png).size == (32, 32)
    metadata = metadata_path.read_text()
    assert '"mode": "gpu-translation"' in metadata
    assert '"output_to_input_matrix_zyx"' in metadata
    assert '"registered_max_projection"' in metadata


def test_click_streaming_perturb_and_register_runs_without_full_stack_load(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(12, 32, 40, 1),
        spots_per_channel=12,
        seed=7,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, config)
    perturbed = tmp_path / "view_b_perturbed.tif"
    registered = tmp_path / "view_b_registered.tif"
    metadata_path = tmp_path / "stream_registration.json"

    perturb = CliRunner().invoke(
        click_main,
        [
            "perturb-view",
            "--input",
            str(dataset / "view_b_object_grid.tif"),
            "--output",
            str(perturbed),
            "--channels",
            "1",
            "--seed",
            "5",
            "--chunk-depth",
            "4",
        ],
    )
    assert perturb.exit_code == 0, perturb.output

    register = CliRunner().invoke(
        click_main,
        [
            "register-stream",
            "--fixed",
            str(dataset / "view_a.tif"),
            "--moving",
            str(perturbed),
            "--output",
            str(registered),
            "--metadata-output",
            str(metadata_path),
            "--channels",
            "1",
            "--mode",
            "affine-12dof",
            "--reference-downsample",
            "2",
            "--max-iterations",
            "1",
            "--chunk-depth",
            "4",
        ],
    )

    assert register.exit_code == 0, register.output
    with tifffile.TiffFile(registered) as tif:
        registered_shape = (len(tif.pages), *tif.pages[0].shape)
    with tifffile.TiffFile(dataset / "view_a.tif") as tif:
        fixed_shape = (len(tif.pages), *tif.pages[0].shape)
    assert registered_shape == fixed_shape
    metadata = metadata_path.read_text()
    assert '"mode": "gpu-stream-affine-12dof"' in metadata
    assert '"stage_modes"' in metadata


def test_click_zarr_perturb_and_register_runs_without_full_stack_load(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(12, 32, 40, 1),
        spots_per_channel=12,
        seed=7,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, config)
    fixed_zarr = tmp_path / "view_a.zarr"
    moving_zarr = tmp_path / "view_b_object_grid.zarr"
    perturbed = tmp_path / "view_b_perturbed.zarr"
    registered = tmp_path / "view_b_registered.zarr"
    metadata_path = tmp_path / "zarr_registration.json"
    deconvolved_png = tmp_path / "deconvolved_zarr.png"
    _write_test_zarr(fixed_zarr, tifffile.imread(dataset / "view_a.tif").reshape(12, 1, 32, 40))
    _write_test_zarr(moving_zarr, tifffile.imread(dataset / "view_b_object_grid.tif").reshape(12, 1, 32, 40))

    perturb = CliRunner().invoke(
        click_main,
        [
            "perturb-zarr",
            "--input",
            str(moving_zarr),
            "--output",
            str(perturbed),
            "--seed",
            "5",
            "--chunk-depth",
            "4",
        ],
    )
    assert perturb.exit_code == 0, perturb.output

    register = CliRunner().invoke(
        click_main,
        [
            "register-zarr",
            "--fixed",
            str(fixed_zarr),
            "--moving",
            str(perturbed),
            "--output",
            str(registered),
            "--metadata-output",
            str(metadata_path),
            "--mode",
            "affine-12dof",
            "--reference-downsample",
            "2",
            "--max-iterations",
            "1",
            "--chunk-depth",
            "4",
        ],
    )

    assert register.exit_code == 0, register.output
    output = zarr.open_array(registered, mode="r")
    assert output.shape == (12, 1, 32, 40)
    assert output.dtype == np.uint16
    metadata = metadata_path.read_text()
    assert '"mode": "gpu-zarr-affine-12dof"' in metadata
    assert '"stage_modes"' in metadata

    preview = CliRunner().invoke(
        click_main,
        [
            "deconvolved-preview-zarr",
            "--view-a",
            str(fixed_zarr),
            "--view-b",
            str(registered),
            "--psf-a",
            str(dataset / "psf_cropped_view_a.tif"),
            "--psf-b",
            str(dataset / "psf_cropped_view_b.tif"),
            "--output",
            str(deconvolved_png),
            "--background",
            "10",
            "--chunk-depth",
            "5",
            "--halo-z",
            "3",
        ],
    )
    assert preview.exit_code == 0, preview.output
    assert Image.open(deconvolved_png).size == (40, 32)
    assert deconvolved_png.with_suffix(".json").exists()


def test_ome_zarr_reference_uses_pyramid_level_and_axis_metadata(tmp_path):
    root = tmp_path / "ome.zarr"
    group = zarr.open_group(root, mode="w")
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": axes,
            "datasets": [{"path": "0"}, {"path": "1"}],
        }
    ]
    full = np.arange(1 * 2 * 8 * 16 * 20, dtype=np.uint16).reshape(1, 2, 8, 16, 20)
    level = full[:, :, ::2, ::4, ::4]
    group.create_array("0", data=full, chunks=(1, 1, 2, 8, 10))
    group.create_array("1", data=level, chunks=(1, 1, 2, 4, 5))

    assert select_zarr_reference_array_path(root, reference_downsample=4) == "1"
    assert inspect_zarr_zcyx(root).shape_zcyx == (8, 2, 16, 20)

    ref = build_downsampled_channel_max_reference_zarr(root, downsample=4)
    np.testing.assert_array_equal(ref, level[0].max(axis=0).astype(np.float32))

    source = open_zarr_zcyx(root)
    slab = _read_zarr_channel_slab(source, channel=1, z0=1, z1=3)
    np.testing.assert_array_equal(slab, full[0, 1, 1:3].astype(np.float32))


def test_root_zarr_reference_still_strides_without_pyramid(tmp_path):
    path = tmp_path / "root.zarr"
    data = np.arange(8 * 2 * 12 * 10, dtype=np.uint16).reshape(8, 2, 12, 10)
    _write_test_zarr(path, data)

    ref = build_downsampled_channel_max_reference_zarr(path, downsample=2)

    np.testing.assert_array_equal(ref, data[::2, :, ::2, ::2].max(axis=1).astype(np.float32))


def test_click_deconvolved_preview_writes_png_for_simulated_dataset(tmp_path):
    config = SimulationConfig(
        shape_zyxc=(12, 32, 32, 1),
        spots_per_channel=3,
        seed=11,
        background=10,
        psf_crop=PsfCropConfig(max_z=3, size=9),
    )
    dataset = tmp_path / "dataset"
    output_png = tmp_path / "deconvolved_preview.png"
    generate_dataset(dataset, config)

    result = CliRunner().invoke(
        click_main,
        [
            "deconvolved-preview",
            str(dataset),
            "--output",
            str(output_png),
            "--chunk-depth",
            "5",
            "--halo-z",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert Image.open(output_png).size == (32, 32)
    assert output_png.with_suffix(".json").exists()


def _synthetic_stack() -> np.ndarray:
    stack = np.zeros((24, 32, 40), dtype=np.float32)
    stack[4:11, 5:13, 7:18] = 800
    stack[12:19, 16:25, 20:35] = 1300
    stack[7:18, 24:29, 4:10] = 500
    stack = ndimage.gaussian_filter(stack, sigma=1.0)
    return stack


def _write_test_zarr(path, data: np.ndarray) -> None:
    array = zarr.open_array(path, mode="w", shape=data.shape, chunks=(4, 1, 16, 20), dtype=data.dtype)
    array[:] = data


def _capture_ffmpeg_processes(monkeypatch) -> list["_FakeFfmpegProcess"]:
    processes: list[_FakeFfmpegProcess] = []

    def fake_start_ffmpeg(output_video, *, width, height, fps, crf, preset, input_pix_fmt):
        process = _FakeFfmpegProcess()
        process.args = {
            "output_video": output_video,
            "width": width,
            "height": height,
            "fps": fps,
            "crf": crf,
            "preset": preset,
            "input_pix_fmt": input_pix_fmt,
        }
        processes.append(process)
        return process

    monkeypatch.setattr(video_module, "_start_ffmpeg", fake_start_ffmpeg)
    return processes


class _FakeFfmpegStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes | bytearray | memoryview) -> int:
        chunk = bytes(data)
        self.data.extend(chunk)
        return len(chunk)

    def close(self) -> None:
        self.closed = True


class _FakeFfmpegProcess:
    def __init__(self) -> None:
        self.stdin = _FakeFfmpegStdin()
        self.args: dict[str, object] = {}

    def wait(self) -> int:
        return 0

    def kill(self) -> None:
        pass


def _spot_deconvolution_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    ground_truth = np.zeros((32, 1, 64, 64), dtype=np.float32)
    spot_coords = [(8, 18, 18), (14, 42, 24), (22, 28, 45)]
    for z, y, x in spot_coords:
        ground_truth[z, 0, y, x] = 1000.0

    psf_a = _anisotropic_gaussian_psf()
    psf_b = np.swapaxes(psf_a, 0, 1).copy()
    view_a = ndimage.convolve(ground_truth[:, 0], psf_a, mode="constant")[:, np.newaxis]
    view_b = ndimage.convolve(ground_truth[:, 0], psf_b, mode="constant")[:, np.newaxis]
    return ground_truth, view_a.astype(np.float32), view_b.astype(np.float32), psf_a, psf_b, spot_coords


def _anisotropic_gaussian_psf() -> np.ndarray:
    zz, yy, xx = np.meshgrid(np.arange(9) - 4, np.arange(9) - 4, np.arange(9) - 4, indexing="ij")
    psf = np.exp(-(zz**2 / (2 * 2.4**2) + yy**2 / (2 * 0.9**2) + xx**2 / (2 * 0.9**2)))
    psf = psf.astype(np.float32)
    return psf / np.float32(psf.sum())


def _mean_fwhm_sum(stack: np.ndarray, spot_coords: list[tuple[int, int, int]]) -> float:
    widths = []
    for z, y, x in spot_coords:
        widths.append(
            _fwhm_width(stack[:, 0, y, x])
            + _fwhm_width(stack[z, 0, :, x])
            + _fwhm_width(stack[z, 0, y, :])
        )
    return float(np.mean(widths))


def _fwhm_width(line: np.ndarray) -> int:
    half_max = float(np.max(line)) * 0.5
    above = np.flatnonzero(line >= half_max)
    if above.size == 0:
        return line.size
    return int(above[-1] - above[0] + 1)


def _spot_localization_errors(
    stack: np.ndarray,
    spot_coords: list[tuple[int, int, int]],
    *,
    radius: int,
) -> list[float]:
    errors = []
    for z, y, x in spot_coords:
        window = stack[z - radius : z + radius + 1, 0, y - radius : y + radius + 1, x - radius : x + radius + 1]
        local_peak = np.asarray(np.unravel_index(np.argmax(window), window.shape), dtype=np.float32)
        peak_zyx = local_peak + np.asarray([z - radius, y - radius, x - radius], dtype=np.float32)
        errors.append(float(np.linalg.norm(peak_zyx - np.asarray([z, y, x], dtype=np.float32))))
    return errors


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a.astype(np.float64) - float(np.mean(a))
    b0 = b.astype(np.float64) - float(np.mean(b))
    return float(np.sum(a0 * b0) / (np.linalg.norm(a0) * np.linalg.norm(b0)))
