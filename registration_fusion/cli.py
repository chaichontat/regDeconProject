from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from .deconvolution import (
    WbProjectorConfig,
    deconvolve_dual_view_gpu,
    default_dual_view_psfs,
)
from .deconvolution_preview import PreviewConfig, write_deconvolved_max_projection_png, write_zarr_deconvolved_max_projection_png
from .flatfield import load_flatfield_profile
from .czi_stitching import (
    build_czi_stitch_plan,
    estimate_czi_stitch_placement,
    load_czi_placement,
    load_czi_placement_origins_csv,
    replace_czi_placement_origins,
    save_czi_placement,
    save_czi_placement_origins_csv,
    summarize_czi_stitch_plan,
    summarize_czi_placement,
    validate_czi_output_channel_count,
    write_czi_stitched_zarr,
)
from .io import read_tiff_zcyx, write_tiff_zcyx
from .previews import write_max_projection_png
from .registration_gpu import VALID_GPU_REGISTRATION_MODES, register_stack_pair_gpu
from .simulate_dual_view import DEFAULT_PSF_PATH, PsfCropConfig
from .streaming_registration import (
    estimate_streaming_registration,
    estimate_zarr_streaming_registration,
    inspect_tiff_zcyx,
    inspect_zarr_zcyx,
    seeded_random_12dof_output_to_input,
    write_affine_transformed_zcyx_tiff,
    write_affine_transformed_zcyx_zarr,
    write_metadata,
)
from .tile_stitching import (
    DEFAULT_OME_FOLDER_PATTERN,
    load_ome_folder_placement,
    stitch_ome_tiff_folder,
    stitch_tiff_tile_manifest,
    summarize_tiff_tile_manifest,
)
from .video import write_zarr_slice_video


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """GPU-accelerated dual-view registration and deconvolution."""


@main.command("dual-view")
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--registered-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--png-dir", type=click.Path(path_type=Path, file_okay=False), help="Directory for max-projection PNG previews.")
@click.option("--channels", type=int, help="Channel count for page-major TIFFs shaped as (Z*C, Y, X).")
@click.option("--registration-mode", type=click.Choice(sorted(VALID_GPU_REGISTRATION_MODES)), default="translation", show_default=True)
@click.option("--registration-ftol", type=float, default=1e-4, show_default=True)
@click.option("--registration-max-iterations", type=int, default=50, show_default=True)
@click.option("--registration-downsample", type=int, default=2, show_default=True)
@click.option("--psf", "psf_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=DEFAULT_PSF_PATH)
@click.option("--iterations", type=int, default=1, show_default=True)
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="32", show_default=True)
@click.option("--compression", callback=lambda _ctx, _param, value: _parse_compression(value), help="TIFF compression name or code.")
@click.option("--psf-step", type=int, default=6, show_default=True)
@click.option("--psf-max-z", type=int, default=7, show_default=True)
@click.option("--psf-size", type=int, default=31, show_default=True)
@click.option("--psf-center", type=int, default=50, show_default=True)
@click.option(
    "--view-angle-deg",
    type=float,
    default=90.0,
    show_default=True,
    help="Angle between view-A and view-B detection axes for the simple dual-view PSF model.",
)
@click.option("--wb-alpha", type=float, default=0.02, show_default=True)
@click.option("--wb-beta", type=float, default=0.02, show_default=True)
@click.option("--wb-n", type=int, default=10, show_default=True)
@click.option("--wb-sigma-g", type=float, default=1.7, show_default=True)
def dual_view(
    fixed_path: Path,
    moving_path: Path,
    output_path: Path,
    registered_output: Path | None,
    metadata_output: Path | None,
    png_dir: Path | None,
    channels: int | None,
    registration_mode: str,
    registration_ftol: float,
    registration_max_iterations: int,
    registration_downsample: int,
    psf_path: Path,
    iterations: int,
    bit_depth: str,
    compression: int | str | None,
    psf_step: int,
    psf_max_z: int,
    psf_size: int,
    psf_center: int,
    view_angle_deg: float,
    wb_alpha: float,
    wb_beta: float,
    wb_n: int,
    wb_sigma_g: float,
) -> None:
    """Register view B onto view A, then run GPU WB joint deconvolution."""
    fixed = read_tiff_zcyx(fixed_path, channels=channels)
    moving = read_tiff_zcyx(moving_path, channels=channels)
    registration = register_stack_pair_gpu(
        fixed,
        moving,
        mode=registration_mode,
        ftol=registration_ftol,
        max_iterations=registration_max_iterations,
        downsample=registration_downsample,
    )
    psf_a, psf_b = default_dual_view_psfs(
        psf_path,
        PsfCropConfig(step=psf_step, max_z=psf_max_z, size=psf_size, center=psf_center),
        view_angle_deg=view_angle_deg,
    )
    deconvolved = deconvolve_dual_view_gpu(
        fixed,
        registration.registered,
        psf_a,
        psf_b,
        iterations=iterations,
        projector_config=WbProjectorConfig(alpha=wb_alpha, beta=wb_beta, n=wb_n, sigma_g=wb_sigma_g),
    )

    write_tiff_zcyx(output_path, deconvolved, bit_depth=int(bit_depth), compression=compression)
    if registered_output is not None:
        write_tiff_zcyx(registered_output, registration.registered, bit_depth=int(bit_depth), compression=compression)
    preview_paths = _write_dual_view_previews(
        output_path,
        png_dir,
        fixed=fixed,
        moving=moving,
        deconvolved=deconvolved,
    )

    metadata = {
        "fixed": str(fixed_path),
        "moving": str(moving_path),
        "output": str(output_path),
        "registered_output": str(registered_output) if registered_output else None,
        "png_previews": {name: str(path) for name, path in preview_paths.items()},
        "shape_zcyx": list(map(int, fixed.shape)),
        "registration": {
            "mode": f"gpu-{registration.mode}",
            "stage_modes": list(registration.stage_modes),
            "initial_shift_zyx": list(map(float, registration.initial_shift_zyx)),
            "output_to_input_matrix_zyx": registration.output_to_input_matrix_zyx.tolist(),
            "output_to_input_offset_zyx": registration.output_to_input_offset_zyx.tolist(),
            "correlation": float(registration.correlation),
        },
        "deconvolution": {
            "mode": "gpu-wb-dual-view-rld",
            "iterations": int(iterations),
            "psf": str(psf_path),
            "psf_crop": {
                "step": int(psf_step),
                "max_z": int(psf_max_z),
                "size": int(psf_size),
                "center": int(psf_center),
            },
            "view_angle_deg": float(view_angle_deg),
            "wb": {
                "alpha": float(wb_alpha),
                "beta": float(wb_beta),
                "n": int(wb_n),
                "sigma_g": float(wb_sigma_g),
            },
        },
    }
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2))

    dz, dy, dx = registration.initial_shift_zyx
    click.echo(
        f"wrote {output_path} using GPU {registration.mode} registration "
        f"(initial shift z/y/x={dz:.1f}/{dy:.1f}/{dx:.1f}); "
        f"registered corr={registration.correlation:.4f}"
    )


@main.command("register")
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--png-dir", type=click.Path(path_type=Path, file_okay=False), help="Directory for max-projection PNG previews.")
@click.option("--channels", type=int, help="Channel count for page-major TIFFs shaped as (Z*C, Y, X).")
@click.option("--mode", type=click.Choice(sorted(VALID_GPU_REGISTRATION_MODES)), default="translation", show_default=True)
@click.option("--ftol", type=float, default=1e-4, show_default=True)
@click.option("--max-iterations", type=int, default=50, show_default=True)
@click.option("--downsample", type=int, default=2, show_default=True)
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
@click.option("--compression", callback=lambda _ctx, _param, value: _parse_compression(value), help="TIFF compression name or code.")
def register(
    fixed_path: Path,
    moving_path: Path,
    output_path: Path,
    metadata_output: Path | None,
    png_dir: Path | None,
    channels: int | None,
    mode: str,
    ftol: float,
    max_iterations: int,
    downsample: int,
    bit_depth: str,
    compression: int | str | None,
) -> None:
    """Register view B onto view A with the migrated GPU registration path."""
    fixed = read_tiff_zcyx(fixed_path, channels=channels)
    moving = read_tiff_zcyx(moving_path, channels=channels)
    result = register_stack_pair_gpu(
        fixed,
        moving,
        mode=mode,
        ftol=ftol,
        max_iterations=max_iterations,
        downsample=downsample,
    )
    write_tiff_zcyx(output_path, result.registered, bit_depth=int(bit_depth), compression=compression)
    preview_paths = _write_registration_previews(
        output_path,
        png_dir,
        fixed=fixed,
        moving=moving,
        registered=result.registered,
    )

    metadata = {
        "fixed": str(fixed_path),
        "moving": str(moving_path),
        "output": str(output_path),
        "png_previews": {name: str(path) for name, path in preview_paths.items()},
        "shape_zcyx": list(map(int, fixed.shape)),
        "registration": {
            "mode": f"gpu-{result.mode}",
            "stage_modes": list(result.stage_modes),
            "initial_shift_zyx": list(map(float, result.initial_shift_zyx)),
            "output_to_input_matrix_zyx": result.output_to_input_matrix_zyx.tolist(),
            "output_to_input_offset_zyx": result.output_to_input_offset_zyx.tolist(),
            "correlation": float(result.correlation),
        },
    }
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2))

    dz, dy, dx = result.initial_shift_zyx
    click.echo(
        f"wrote {output_path} using GPU {result.mode} registration "
        f"(initial shift z/y/x={dz:.1f}/{dy:.1f}/{dx:.1f}); corr={result.correlation:.4f}"
    )


@main.command("deconvolved-preview")
@click.argument("dataset_dir", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--output", "output_png", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--view-a", "view_a_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--view-b", "view_b_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--chunk-depth", type=int, default=64, show_default=True)
@click.option("--halo-z", type=int, default=20, show_default=True)
@click.option("--iterations", type=int, default=1, show_default=True)
@click.option("--max-chunks", type=int, help="Limit chunks per channel for profiling/debugging.")
def deconvolved_preview(
    dataset_dir: Path,
    output_png: Path | None,
    view_a_path: Path | None,
    view_b_path: Path | None,
    chunk_depth: int,
    halo_z: int,
    iterations: int,
    max_chunks: int | None,
) -> None:
    """Write a chunked GPU deconvolved max-projection PNG for a simulated dataset."""
    output = write_deconvolved_max_projection_png(
        dataset_dir,
        output_png=output_png,
        view_a_path=view_a_path,
        view_b_path=view_b_path,
        config=PreviewConfig(
            chunk_depth=chunk_depth,
            halo_z=halo_z,
            iterations=iterations,
            max_chunks=max_chunks,
        ),
    )
    click.echo(f"wrote {output}")


@main.command("deconvolved-preview-zarr")
@click.option("--view-a", "view_a_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--view-b", "view_b_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--psf-a", "psf_a_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--psf-b", "psf_b_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output", "output_png", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--background", type=float, default=0.0, show_default=True)
@click.option("--chunk-depth", type=int, default=64, show_default=True)
@click.option("--halo-z", type=int, default=20, show_default=True)
@click.option("--iterations", type=int, default=1, show_default=True)
@click.option("--max-chunks", type=int, help="Limit chunks per channel for profiling/debugging.")
def deconvolved_preview_zarr(
    view_a_path: Path,
    view_b_path: Path,
    psf_a_path: Path,
    psf_b_path: Path,
    output_png: Path,
    background: float,
    chunk_depth: int,
    halo_z: int,
    iterations: int,
    max_chunks: int | None,
) -> None:
    """Write a chunked GPU deconvolved max-projection PNG from ZCYX Zarr arrays."""
    output = write_zarr_deconvolved_max_projection_png(
        view_a_path,
        view_b_path,
        psf_a_path=psf_a_path,
        psf_b_path=psf_b_path,
        output_png=output_png,
        background=background,
        config=PreviewConfig(
            chunk_depth=chunk_depth,
            halo_z=halo_z,
            iterations=iterations,
            max_chunks=max_chunks,
        ),
    )
    click.echo(f"wrote {output}")


@main.command("zarr-video")
@click.argument("input_zarr", type=click.Path(path_type=Path, exists=True))
@click.option("--output", "output_video", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--channel", "channels", type=int, multiple=True, help="Channel to include. Repeat to max-project selected channels; defaults to all channels.")
@click.option("--height", type=int, default=1080, show_default=True, help="Output video height in pixels; width preserves source aspect.")
@click.option("--width", type=int, help="Override output width in pixels.")
@click.option("--fps", type=float, default=24.0, show_default=True)
@click.option("--z-step", type=int, default=1, show_default=True, help="Render every Nth Z slice.")
@click.option("--max-frames", type=int, help="Limit rendered frames for smoke tests.")
@click.option("--sample-frames", type=int, default=64, show_default=True, help="Frames sampled for global contrast scaling.")
@click.option("--sample-stride", type=int, help="Spatial stride for contrast sampling.")
@click.option("--crf", type=int, default=18, show_default=True, help="x264 quality; lower is larger/better.")
@click.option("--preset", default="medium", show_default=True, help="x264 encoding preset.")
@click.option("--scale-bar-um", type=float, help="Draw a bottom-right scale bar with this length in micrometers.")
@click.option("--pixel-size-um", type=float, help="Source XY pixel size in micrometers; overrides source_czi metadata.")
def zarr_video(
    input_zarr: Path,
    output_video: Path,
    channels: tuple[int, ...],
    height: int,
    width: int | None,
    fps: float,
    z_step: int,
    max_frames: int | None,
    sample_frames: int,
    sample_stride: int | None,
    crf: int,
    preset: str,
    scale_bar_um: float | None,
    pixel_size_um: float | None,
) -> None:
    """Render a ZCYX Zarr as a streaming 8-bit MP4 through Z."""
    result = write_zarr_slice_video(
        input_zarr,
        output_video,
        channels=channels or None,
        height=height,
        width=width,
        fps=fps,
        z_step=z_step,
        max_frames=max_frames,
        sample_frames=sample_frames,
        sample_stride=sample_stride,
        crf=crf,
        preset=preset,
        scale_bar_um=scale_bar_um,
        pixel_size_um=pixel_size_um,
    )
    click.echo(
        f"wrote {result.output_path}; frames={result.frame_count}; "
        f"size={result.output_width}x{result.output_height}; fps={result.fps:g}; "
        f"channels={list(result.channels)}"
    )


@main.command("perturb-view")
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--png-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--channels", type=int, required=True)
@click.option("--seed", type=int, default=123, show_default=True)
@click.option("--chunk-depth", type=int, default=8, show_default=True)
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
@click.option("--compression", callback=lambda _ctx, _param, value: _parse_compression(value), help="TIFF compression name or code.")
def perturb_view(
    input_path: Path,
    output_path: Path,
    metadata_output: Path | None,
    png_output: Path | None,
    channels: int,
    seed: int,
    chunk_depth: int,
    bit_depth: str,
    compression: int | str | None,
) -> None:
    """Write a seeded 12-DOF affine perturbation of a page-major ZCYX TIFF in chunks."""
    info = inspect_tiff_zcyx(input_path, channels=channels)
    matrix, offset, transform_metadata = seeded_random_12dof_output_to_input(info.shape_zcyx[0:1] + info.shape_zcyx[2:4], seed=seed)
    write_affine_transformed_zcyx_tiff(
        input_path,
        output_path,
        channels=channels,
        matrix_zyx=matrix,
        offset_zyx=offset,
        chunk_depth=chunk_depth,
        bit_depth=int(bit_depth),
        compression=compression,
        png_path=png_output,
    )
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "shape_zcyx": list(map(int, info.shape_zcyx)),
        "transform": transform_metadata,
        "chunk_depth": int(chunk_depth),
    }
    if metadata_output is not None:
        write_metadata(metadata_output, metadata)
    click.echo(f"wrote perturbed view {output_path}")


@main.command("perturb-zarr")
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--png-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--seed", type=int, default=123, show_default=True)
@click.option("--chunk-depth", type=int, default=2, show_default=True)
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
def perturb_zarr(
    input_path: Path,
    output_path: Path,
    metadata_output: Path | None,
    png_output: Path | None,
    seed: int,
    chunk_depth: int,
    bit_depth: str,
) -> None:
    """Write a seeded 12-DOF affine perturbation of a ZCYX Zarr array in chunks."""
    info = inspect_zarr_zcyx(input_path)
    matrix, offset, transform_metadata = seeded_random_12dof_output_to_input(info.shape_zcyx[0:1] + info.shape_zcyx[2:4], seed=seed)
    write_affine_transformed_zcyx_zarr(
        input_path,
        output_path,
        matrix_zyx=matrix,
        offset_zyx=offset,
        chunk_depth=chunk_depth,
        bit_depth=int(bit_depth),
        png_path=png_output,
    )
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "shape_zcyx": list(map(int, info.shape_zcyx)),
        "chunks_zcyx": list(map(int, info.chunks_zcyx)),
        "transform": transform_metadata,
        "chunk_depth": int(chunk_depth),
    }
    if metadata_output is not None:
        write_metadata(metadata_output, metadata)
    click.echo(f"wrote perturbed Zarr view {output_path}")


@main.command("register-stream")
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--png-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--channels", type=int, required=True)
@click.option("--mode", type=click.Choice(sorted(VALID_GPU_REGISTRATION_MODES)), default="affine-12dof", show_default=True)
@click.option("--reference-downsample", type=int, default=8, show_default=True)
@click.option("--ftol", type=float, default=1e-4, show_default=True)
@click.option("--max-iterations", type=int, default=20, show_default=True)
@click.option("--chunk-depth", type=int, default=8, show_default=True)
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
@click.option("--compression", callback=lambda _ctx, _param, value: _parse_compression(value), help="TIFF compression name or code.")
def register_stream(
    fixed_path: Path,
    moving_path: Path,
    output_path: Path,
    metadata_output: Path | None,
    png_output: Path | None,
    channels: int,
    mode: str,
    reference_downsample: int,
    ftol: float,
    max_iterations: int,
    chunk_depth: int,
    bit_depth: str,
    compression: int | str | None,
) -> None:
    """Register page-major ZCYX TIFFs using streamed downsampled refs and chunked output resampling."""
    result = estimate_streaming_registration(
        fixed_path,
        moving_path,
        channels=channels,
        mode=mode,
        reference_downsample=reference_downsample,
        ftol=ftol,
        max_iterations=max_iterations,
    )
    write_affine_transformed_zcyx_tiff(
        moving_path,
        output_path,
        channels=channels,
        matrix_zyx=result.output_to_input_matrix_zyx,
        offset_zyx=result.output_to_input_offset_zyx,
        chunk_depth=chunk_depth,
        bit_depth=int(bit_depth),
        compression=compression,
        png_path=png_output,
    )
    fixed_info = inspect_tiff_zcyx(fixed_path, channels=channels)
    metadata = {
        "fixed": str(fixed_path),
        "moving": str(moving_path),
        "output": str(output_path),
        "shape_zcyx": list(map(int, fixed_info.shape_zcyx)),
        "registration": {
            "mode": f"gpu-stream-{mode}",
            "stage_modes": list(result.stage_modes),
            "reference_downsample": int(reference_downsample),
            "initial_shift_zyx": list(map(float, result.initial_shift_zyx)),
            "output_to_input_matrix_zyx": result.output_to_input_matrix_zyx.tolist(),
            "output_to_input_offset_zyx": result.output_to_input_offset_zyx.tolist(),
            "reference_correlation": float(result.reference_correlation),
        },
    }
    if metadata_output is not None:
        write_metadata(metadata_output, metadata)
    click.echo(
        f"wrote streamed registration {output_path}; "
        f"reference corr={result.reference_correlation:.4f}"
    )


@main.command("register-zarr")
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--png-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--mode", type=click.Choice(sorted(VALID_GPU_REGISTRATION_MODES)), default="affine-12dof", show_default=True)
@click.option("--reference-downsample", type=int, default=16, show_default=True, help="Target pyramid downsample for registration references.")
@click.option("--fixed-array", type=str, help="Array path inside fixed OME-Zarr; defaults to multiscale level 0.")
@click.option("--moving-array", type=str, help="Array path inside moving OME-Zarr; defaults to multiscale level 0.")
@click.option("--fixed-reference-array", type=str, help="Fixed pyramid array for registration reference; defaults to level closest to --reference-downsample.")
@click.option("--moving-reference-array", type=str, help="Moving pyramid array for registration reference; defaults to level closest to --reference-downsample.")
@click.option("--time-index", type=int, default=0, show_default=True, help="T index for OME-Zarr arrays with a time axis.")
@click.option("--ftol", type=float, default=1e-4, show_default=True)
@click.option("--max-iterations", type=int, default=20, show_default=True)
@click.option("--chunk-depth", type=int, default=2, show_default=True)
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
def register_zarr(
    fixed_path: Path,
    moving_path: Path,
    output_path: Path,
    metadata_output: Path | None,
    png_output: Path | None,
    mode: str,
    reference_downsample: int,
    fixed_array: str | None,
    moving_array: str | None,
    fixed_reference_array: str | None,
    moving_reference_array: str | None,
    time_index: int,
    ftol: float,
    max_iterations: int,
    chunk_depth: int,
    bit_depth: str,
) -> None:
    """Register ZCYX or OME-Zarr arrays using pyramid refs and chunked output resampling."""
    result = estimate_zarr_streaming_registration(
        fixed_path,
        moving_path,
        mode=mode,
        reference_downsample=reference_downsample,
        ftol=ftol,
        max_iterations=max_iterations,
        fixed_array_path=fixed_array,
        moving_array_path=moving_array,
        fixed_reference_array_path=fixed_reference_array,
        moving_reference_array_path=moving_reference_array,
        time_index=time_index,
    )
    write_affine_transformed_zcyx_zarr(
        moving_path,
        output_path,
        matrix_zyx=result.output_to_input_matrix_zyx,
        offset_zyx=result.output_to_input_offset_zyx,
        chunk_depth=chunk_depth,
        bit_depth=int(bit_depth),
        png_path=png_output,
        array_path=moving_array,
        time_index=time_index,
    )
    fixed_info = inspect_zarr_zcyx(fixed_path, array_path=fixed_array, time_index=time_index)
    metadata = {
        "fixed": str(fixed_path),
        "moving": str(moving_path),
        "output": str(output_path),
        "fixed_array": result.fixed_array_path,
        "moving_array": result.moving_array_path,
        "shape_zcyx": list(map(int, fixed_info.shape_zcyx)),
        "chunks_zcyx": list(map(int, fixed_info.chunks_zcyx)),
        "registration": {
            "mode": f"gpu-zarr-{mode}",
            "stage_modes": list(result.stage_modes),
            "reference_downsample": int(reference_downsample),
            "fixed_reference_array": result.fixed_reference_array_path,
            "moving_reference_array": result.moving_reference_array_path,
            "reference_shape_zyx": list(map(int, result.reference_shape_zyx or ())),
            "initial_shift_zyx": list(map(float, result.initial_shift_zyx)),
            "output_to_input_matrix_zyx": result.output_to_input_matrix_zyx.tolist(),
            "output_to_input_offset_zyx": result.output_to_input_offset_zyx.tolist(),
            "reference_correlation": float(result.reference_correlation),
        },
    }
    if metadata_output is not None:
        write_metadata(metadata_output, metadata)
    click.echo(
        f"wrote Zarr registration {output_path}; "
        f"reference corr={result.reference_correlation:.4f}"
    )


@main.command("stitch-czi")
@click.argument("czi_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--output", "output_zarr", type=click.Path(path_type=Path))
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--progress-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--placement-input", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--placement-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--channel", type=int, default=0, show_default=True)
@click.option(
    "--apply-channel",
    "apply_channels",
    type=int,
    multiple=True,
    help="Output channel to stitch with the placement estimated from --channel. Repeat for multicolor output.",
)
@click.option("--z-start", type=int, default=0, show_default=True)
@click.option("--z-stop", type=int)
@click.option("--chunk-depth", type=int, default=4, show_default=True)
@click.option("--reference-z-start", type=int, default=244, show_default=True)
@click.option("--reference-z-count", type=int, default=8, show_default=True)
@click.option("--overlap-z", type=int, default=4, show_default=True)
@click.option("--overlap-y", type=int, default=290, show_default=True)
@click.option("--overlap-x", type=int, default=288, show_default=True)
@click.option("--fallback-min-ncc", type=float, default=0.8, show_default=True)
@click.option("--coarse-max-size", type=int, default=1024, show_default=True, help="Maximum coarse phase-correlation size per Z/Y/X axis.")
@click.option("--fine-upsample-factor", type=int, default=10, show_default=True, help="Set to 1 to disable fine refinement; values above 1 set the fine optimizer budget.")
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
@click.option("--max-chunks", type=int, help="Limit output chunks for benchmarking/smoke tests.")
@click.option("--benchmark-only", is_flag=True, help="Run read/fuse timing without writing an output Zarr.")
@click.option("--resume", is_flag=True, help="Resume a previous output Zarr using the chunk progress log.")
@click.option("--overwrite", is_flag=True, help="Replace an existing output Zarr instead of failing.")
@click.option("--preview-output", type=click.Path(path_type=Path, dir_okay=False), help="Write a downsampled max-projection PNG for processed chunks.")
@click.option("--preview-downsample", type=int, default=4, show_default=True)
@click.option(
    "--basic-profile",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="BaSiC pickle whose flatfield is applied during fusion only.",
)
def stitch_czi(
    czi_path: Path,
    output_zarr: Path | None,
    metadata_output: Path | None,
    progress_output: Path | None,
    placement_input: Path | None,
    placement_output: Path | None,
    channel: int,
    apply_channels: tuple[int, ...],
    z_start: int,
    z_stop: int | None,
    chunk_depth: int,
    reference_z_start: int,
    reference_z_count: int,
    overlap_z: int,
    overlap_y: int,
    overlap_x: int,
    fallback_min_ncc: float,
    coarse_max_size: int,
    fine_upsample_factor: int,
    bit_depth: str,
    max_chunks: int | None,
    benchmark_only: bool,
    resume: bool,
    overwrite: bool,
    preview_output: Path | None,
    preview_downsample: int,
    basic_profile: Path | None,
) -> None:
    """Stitch a CZI mosaic channel to a chunked ZCYX Zarr array with CUDA."""
    if output_zarr is None and not benchmark_only:
        raise click.UsageError("--output is required unless --benchmark-only is set")
    if resume and benchmark_only:
        raise click.UsageError("--resume cannot be combined with --benchmark-only")
    if resume and overwrite:
        raise click.UsageError("--resume cannot be combined with --overwrite")
    placement = load_czi_placement(placement_input) if placement_input is not None else None
    flatfield_profile = load_flatfield_profile(basic_profile) if basic_profile is not None else None
    try:
        run = write_czi_stitched_zarr(
            czi_path,
            output_zarr,
            channel=channel,
            output_channels=apply_channels or None,
            placement=placement,
            placement_output_path=placement_output,
            z_start=z_start,
            z_stop=z_stop,
            chunk_depth=chunk_depth,
            reference_z_start=reference_z_start,
            reference_z_count=reference_z_count,
            overlap_zyx=(overlap_z, overlap_y, overlap_x),
            fallback_min_ncc=fallback_min_ncc,
            coarse_max_size=coarse_max_size,
            fine_upsample_factor=fine_upsample_factor,
            bit_depth=int(bit_depth),
            max_chunks=max_chunks,
            metadata_path=metadata_output,
            benchmark_only=benchmark_only,
            resume=resume,
            overwrite=overwrite,
            progress_path=progress_output,
            preview_path=preview_output,
            preview_downsample=preview_downsample,
            flatfield_profile=flatfield_profile,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    action = "benchmarked" if benchmark_only else "wrote"
    output_label = output_zarr if output_zarr is not None else czi_path
    message = (
        f"{action} {output_label}; chunks={run.chunks_written}; "
        f"skipped={run.chunks_skipped}; "
        f"read={run.read_seconds:.2f}s fuse={run.fuse_seconds:.2f}s write={run.write_seconds:.2f}s"
    )
    if 0 < run.chunks_written < run.total_chunks:
        estimated_processing_loop = run.total_seconds / run.chunks_written * run.total_chunks
        loop_label = "processing loop" if benchmark_only else "write loop"
        message += f"; projected full {loop_label}={estimated_processing_loop:.2f}s"
    click.echo(message)


@main.command("inspect-czi-stitch")
@click.argument("czi_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--channel", type=int, default=0, show_default=True)
@click.option("--channels", type=int, default=1, show_default=True, help="Number of output channels to estimate.")
@click.option("--z-sample", type=int, default=0, show_default=True)
@click.option("--overlap-z", type=int, default=4, show_default=True)
@click.option("--overlap-y", type=int, default=290, show_default=True)
@click.option("--overlap-x", type=int, default=288, show_default=True)
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
def inspect_czi_stitch_command(
    czi_path: Path,
    metadata_output: Path | None,
    channel: int,
    channels: int,
    z_sample: int,
    overlap_z: int,
    overlap_y: int,
    overlap_x: int,
    bit_depth: str,
) -> None:
    """Inspect CZI mosaic geometry and estimate stitched output size."""
    try:
        validate_czi_output_channel_count(czi_path, start_channel=channel, channel_count=channels)
        plan = build_czi_stitch_plan(
            czi_path,
            channel=channel,
            overlap_zyx=(overlap_z, overlap_y, overlap_x),
            z_sample=z_sample,
        )
        summary = summarize_czi_stitch_plan(plan, channels=channels, bit_depth=int(bit_depth))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(summary, indent=2))
    click.echo(
        f"tiles={summary['tile_count']}; z={summary['z_size']}; "
        f"nominal shape={summary['nominal_output_shape_zcyx']}; "
        f"bytes={summary['nominal_output_bytes']}"
    )


@main.command("estimate-czi-placement")
@click.argument("czi_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--channel", type=int, default=0, show_default=True)
@click.option("--reference-z-start", type=int, default=244, show_default=True)
@click.option("--reference-z-count", type=int, default=8, show_default=True)
@click.option("--overlap-z", type=int, default=4, show_default=True)
@click.option("--overlap-y", type=int, default=290, show_default=True)
@click.option("--overlap-x", type=int, default=288, show_default=True)
@click.option("--fallback-min-ncc", type=float, default=0.8, show_default=True)
@click.option("--coarse-max-size", type=int, default=1024, show_default=True, help="Maximum coarse phase-correlation size per Z/Y/X axis.")
@click.option("--fine-upsample-factor", type=int, default=10, show_default=True, help="Set to 1 to disable fine refinement; values above 1 set the fine optimizer budget.")
def estimate_czi_placement_command(
    czi_path: Path,
    output_path: Path,
    metadata_output: Path | None,
    channel: int,
    reference_z_start: int,
    reference_z_count: int,
    overlap_z: int,
    overlap_y: int,
    overlap_x: int,
    fallback_min_ncc: float,
    coarse_max_size: int,
    fine_upsample_factor: int,
) -> None:
    """Estimate and save reusable CZI mosaic placement with CUDA."""
    try:
        placement = estimate_czi_stitch_placement(
            czi_path,
            channel=channel,
            reference_z_start=reference_z_start,
            reference_z_count=reference_z_count,
            overlap_zyx=(overlap_z, overlap_y, overlap_x),
            fallback_min_ncc=fallback_min_ncc,
            coarse_max_size=coarse_max_size,
            fine_upsample_factor=fine_upsample_factor,
            force_zero_z_origin=True,
        )
        save_czi_placement(output_path, placement)
        summary = summarize_czi_placement(placement)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(summary, indent=2))
    ncc_median = summary["ncc_median"]
    ncc_text = f"{ncc_median:.4f}" if isinstance(ncc_median, float) else "n/a"
    click.echo(
        f"wrote {output_path}; tiles={summary['tile_count']}; pairs={summary['pair_count']}; "
        f"fallbacks={summary['fallback_pair_count']}; "
        f"ncc median={ncc_text}; estimate={summary['estimate_seconds']:.2f}s"
    )


@main.command("export-czi-placement-origins")
@click.argument("placement_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
def export_czi_placement_origins_command(placement_path: Path, output_path: Path) -> None:
    """Export CZI placement origins to an editable CSV."""
    try:
        placement = load_czi_placement(placement_path)
        save_czi_placement_origins_csv(output_path, placement)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote {output_path}; origins={len(placement.origins_zyx)}")


@main.command("apply-czi-placement-origins")
@click.argument("placement_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--origins", "origins_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
def apply_czi_placement_origins_command(placement_path: Path, origins_path: Path, output_path: Path) -> None:
    """Apply edited origin CSV values to a CZI placement JSON."""
    try:
        placement = load_czi_placement(placement_path)
        origins = load_czi_placement_origins_csv(origins_path)
        updated = replace_czi_placement_origins(placement, origins)
        save_czi_placement(output_path, updated)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote {output_path}; origins={len(updated.origins_zyx)}")


@main.command("stitch-tiles")
@click.argument("manifest_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--output", "output_zarr", type=click.Path(path_type=Path), required=True)
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--channels", type=int, help="Channel count for page-major TIFF tiles shaped as (Z*C, Y, X).")
@click.option("--reference-channel", type=int, default=0, show_default=True)
@click.option("--overlap-z", type=int, required=True)
@click.option("--overlap-y", type=int, required=True)
@click.option("--overlap-x", type=int, required=True)
@click.option("--fallback-min-ncc", type=float, help="Use manifest nominal origins for pairs below this NCC.")
@click.option("--coarse-max-size", type=int, default=1024, show_default=True, help="Maximum coarse phase-correlation size per Z/Y/X axis.")
@click.option("--fine-upsample-factor", type=int, default=1, show_default=True, help="Set to 1 to disable fine refinement; values above 1 set the fine optimizer budget.")
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
@click.option("--overwrite", is_flag=True, help="Replace an existing output Zarr instead of failing.")
@click.option(
    "--basic-profile",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="BaSiC pickle whose flatfield is applied during fusion only.",
)
def stitch_tiles_command(
    manifest_path: Path,
    output_zarr: Path,
    metadata_output: Path | None,
    channels: int | None,
    reference_channel: int,
    overlap_z: int,
    overlap_y: int,
    overlap_x: int,
    fallback_min_ncc: float | None,
    coarse_max_size: int,
    fine_upsample_factor: int,
    bit_depth: str,
    overwrite: bool,
    basic_profile: Path | None,
) -> None:
    """Stitch a CSV manifest of TIFF tiles with the CUDA stitching core."""
    flatfield_profile = load_flatfield_profile(basic_profile) if basic_profile is not None else None
    try:
        run = stitch_tiff_tile_manifest(
            manifest_path,
            output_zarr,
            channels=channels,
            overlap_zyx=(overlap_z, overlap_y, overlap_x),
            reference_channel=reference_channel,
            fallback_min_ncc=fallback_min_ncc,
            coarse_max_size=coarse_max_size,
            fine_upsample_factor=fine_upsample_factor,
            bit_depth=int(bit_depth),
            metadata_path=metadata_output,
            overwrite=overwrite,
            flatfield_profile=flatfield_profile,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"wrote {output_zarr}; tiles={run.tile_count}; "
        f"shape={run.output_shape_zcyx}; dtype={run.output_dtype}"
    )


@main.command("stitch-ome-folder")
@click.argument("folder", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--output", "output_zarr", type=click.Path(path_type=Path))
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--progress-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--placement-input", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--placement-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--pattern", default=DEFAULT_OME_FOLDER_PATTERN, show_default=True)
@click.option("--reference-channel", type=int, default=0, show_default=True)
@click.option("--z-start", type=int, default=0, show_default=True)
@click.option("--z-stop", type=int)
@click.option("--chunk-depth", type=int, default=4, show_default=True)
@click.option("--reference-z-start", type=int, default=0, show_default=True)
@click.option("--reference-z-count", type=int, default=2, show_default=True)
@click.option("--overlap-z", type=int, help="Override inferred Z overlap.")
@click.option("--overlap-y", type=int, help="Override inferred Y overlap.")
@click.option("--overlap-x", type=int, help="Override inferred X overlap.")
@click.option("--fallback-min-ncc", type=float, default=0.8, show_default=True, help="Use OME metadata origins for pairs below this NCC.")
@click.option("--coarse-max-size", type=int, default=1024, show_default=True, help="Maximum coarse phase-correlation size per Z/Y/X axis.")
@click.option("--fine-upsample-factor", type=int, default=1, show_default=True, help="Set to 1 to disable fine refinement; values above 1 set the fine optimizer budget.")
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
@click.option("--max-chunks", type=int, help="Limit output chunks for benchmarking/smoke tests.")
@click.option("--benchmark-only", is_flag=True, help="Run read/fuse timing without writing an output Zarr.")
@click.option("--resume", is_flag=True, help="Resume a previous output Zarr using the chunk progress log.")
@click.option("--overwrite", is_flag=True, help="Replace an existing output Zarr instead of failing.")
@click.option("--preview-output", type=click.Path(path_type=Path, dir_okay=False), help="Write a downsampled max-projection PNG for processed chunks.")
@click.option("--preview-downsample", type=int, default=4, show_default=True)
@click.option(
    "--basic-profile",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="BaSiC pickle whose flatfield is applied during fusion only.",
)
def stitch_ome_folder_command(
    folder: Path,
    output_zarr: Path,
    metadata_output: Path | None,
    progress_output: Path | None,
    placement_input: Path | None,
    placement_output: Path | None,
    pattern: str,
    reference_channel: int,
    z_start: int,
    z_stop: int | None,
    chunk_depth: int,
    reference_z_start: int,
    reference_z_count: int,
    overlap_z: int | None,
    overlap_y: int | None,
    overlap_x: int | None,
    fallback_min_ncc: float | None,
    coarse_max_size: int,
    fine_upsample_factor: int,
    bit_depth: str,
    max_chunks: int | None,
    benchmark_only: bool,
    resume: bool,
    overwrite: bool,
    preview_output: Path | None,
    preview_downsample: int,
    basic_profile: Path | None,
) -> None:
    """Stitch folder-discovered OME-TIFF tiles using OME Plane positions."""
    if output_zarr is None and not benchmark_only:
        raise click.UsageError("--output is required unless --benchmark-only is set")
    if resume and benchmark_only:
        raise click.UsageError("--resume cannot be combined with --benchmark-only")
    if resume and overwrite:
        raise click.UsageError("--resume cannot be combined with --overwrite")
    placement = load_ome_folder_placement(placement_input) if placement_input is not None else None
    flatfield_profile = load_flatfield_profile(basic_profile) if basic_profile is not None else None
    try:
        run = stitch_ome_tiff_folder(
            folder,
            output_zarr,
            pattern=pattern,
            placement=placement,
            placement_output_path=placement_output,
            z_start=z_start,
            z_stop=z_stop,
            chunk_depth=chunk_depth,
            reference_z_start=reference_z_start,
            reference_z_count=reference_z_count,
            overlap_z=overlap_z,
            overlap_y=overlap_y,
            overlap_x=overlap_x,
            reference_channel=reference_channel,
            fallback_min_ncc=fallback_min_ncc,
            coarse_max_size=coarse_max_size,
            fine_upsample_factor=fine_upsample_factor,
            bit_depth=int(bit_depth),
            max_chunks=max_chunks,
            metadata_path=metadata_output,
            benchmark_only=benchmark_only,
            resume=resume,
            overwrite=overwrite,
            progress_path=progress_output,
            preview_path=preview_output,
            preview_downsample=preview_downsample,
            flatfield_profile=flatfield_profile,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    action = "benchmarked" if benchmark_only else "wrote"
    output_label = output_zarr if output_zarr is not None else folder
    message = (
        f"{action} {output_label}; tiles={run.tile_count}; chunks={run.chunks_written}; "
        f"skipped={run.chunks_skipped}; overlap={run.overlap_zyx}; "
        f"shape={run.output_shape_zcyx}; dtype={run.output_dtype}; "
        f"read={run.read_seconds:.2f}s fuse={run.fuse_seconds:.2f}s write={run.write_seconds:.2f}s"
    )
    if 0 < run.chunks_written < run.total_chunks:
        message += f"; projected full write loop={run.total_seconds / run.chunks_written * run.total_chunks:.2f}s"
    click.echo(message)


@main.command("inspect-tile-manifest")
@click.argument("manifest_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--metadata-output", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--channels", type=int, help="Channel count for page-major TIFF tiles shaped as (Z*C, Y, X).")
@click.option("--overlap-z", type=int, required=True)
@click.option("--overlap-y", type=int, required=True)
@click.option("--overlap-x", type=int, required=True)
@click.option("--bit-depth", type=click.Choice(["16", "32"]), default="16", show_default=True)
def inspect_tile_manifest_command(
    manifest_path: Path,
    metadata_output: Path | None,
    channels: int | None,
    overlap_z: int,
    overlap_y: int,
    overlap_x: int,
    bit_depth: str,
) -> None:
    """Inspect a TIFF tile manifest and estimate stitched output size."""
    try:
        summary = summarize_tiff_tile_manifest(
            manifest_path,
            channels=channels,
            overlap_zyx=(overlap_z, overlap_y, overlap_x),
            bit_depth=int(bit_depth),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(summary, indent=2))
    click.echo(
        f"tiles={summary['tile_count']}; tile shape={summary['tile_shape_zcyx']}; "
        f"estimated shape={summary['estimated_output_shape_zcyx']}; "
        f"bytes={summary['estimated_output_bytes']}"
    )


def _parse_compression(value: str | None) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _write_dual_view_previews(
    output_path: Path,
    png_dir: Path | None,
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    deconvolved: np.ndarray,
) -> dict[str, Path]:
    directory = png_dir or output_path.parent
    stem = output_path.stem
    paths = {
        "view_a_max_projection": directory / f"{stem}__view_a_max_projection.png",
        "view_b_max_projection": directory / f"{stem}__view_b_max_projection.png",
        "deconvolved_max_projection": directory / f"{stem}__deconvolved_max_projection.png",
    }
    write_max_projection_png(paths["view_a_max_projection"], fixed)
    write_max_projection_png(paths["view_b_max_projection"], moving)
    write_max_projection_png(paths["deconvolved_max_projection"], deconvolved)
    return paths


def _write_registration_previews(
    output_path: Path,
    png_dir: Path | None,
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    registered: np.ndarray,
) -> dict[str, Path]:
    directory = png_dir or output_path.parent
    stem = output_path.stem
    paths = {
        "fixed_max_projection": directory / f"{stem}__fixed_max_projection.png",
        "moving_max_projection": directory / f"{stem}__moving_max_projection.png",
        "registered_max_projection": directory / f"{stem}__registered_max_projection.png",
    }
    write_max_projection_png(paths["fixed_max_projection"], fixed)
    write_max_projection_png(paths["moving_max_projection"], moving)
    write_max_projection_png(paths["registered_max_projection"], registered)
    return paths


if __name__ == "__main__":
    main()
