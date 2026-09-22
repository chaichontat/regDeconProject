from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import cupy as cp
import numpy as np
from aicspylibczi import CziFile
from PIL import Image
from scipy import ndimage

from registration_fusion.previews import _scale_uint8
from registration_fusion.registration_gpu import (
    _fit_transform,
    apply_output_to_input_transform_gpu,
    estimate_translation_gpu,
    require_cuda_gpu,
)


@dataclass(frozen=True)
class ReferenceConfig:
    czi_path: str
    fixed_channel: int
    moving_channel: int
    z_step: int
    scale_factor: float
    preprocess: str
    fixed_clip_percentile: float
    moving_clip_percentile: float


@click.command()
@click.argument("czi_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--fixed-channel", type=int, default=0, show_default=True)
@click.option("--moving-channel", type=int, default=1, show_default=True)
@click.option("--z-step", type=int, default=4, show_default=True)
@click.option("--scale-factor", type=float, default=0.03125, show_default=True)
@click.option("--preprocess", type=click.Choice(["raw", "sqrt", "log", "dog"]), default="sqrt", show_default=True)
@click.option("--fixed-clip-percentile", type=float, default=99.9, show_default=True)
@click.option("--moving-clip-percentile", type=float, default=99.9, show_default=True)
@click.option("--mode", multiple=True, default=("translation", "rigid", "affine-12dof"), show_default=True)
@click.option("--max-iterations", type=int, default=20, show_default=True)
@click.option("--ftol", type=float, default=1e-4, show_default=True)
@click.option("--force-rebuild", is_flag=True)
def main(
    czi_path: Path,
    output_dir: Path,
    fixed_channel: int,
    moving_channel: int,
    z_step: int,
    scale_factor: float,
    preprocess: str,
    fixed_clip_percentile: float,
    moving_clip_percentile: float,
    mode: tuple[str, ...],
    max_iterations: int,
    ftol: float,
    force_rebuild: bool,
) -> None:
    require_cuda_gpu()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ReferenceConfig(
        czi_path=str(czi_path),
        fixed_channel=fixed_channel,
        moving_channel=moving_channel,
        z_step=z_step,
        scale_factor=scale_factor,
        preprocess=preprocess,
        fixed_clip_percentile=fixed_clip_percentile,
        moving_clip_percentile=moving_clip_percentile,
    )
    fixed, moving, z_indices = _load_or_build_references(output_dir, czi_path, config, force_rebuild=force_rebuild)
    metrics = []
    for registration_mode in mode:
        metrics.append(
            _run_mode(
                output_dir,
                fixed,
                moving,
                z_indices=z_indices,
                config=config,
                mode=registration_mode,
                max_iterations=max_iterations,
                ftol=ftol,
            )
        )
    summary_path = output_dir / "registration_summary.json"
    summary_path.write_text(json.dumps({"config": asdict(config), "runs": metrics}, indent=2))
    click.echo(f"wrote {summary_path}")


def _load_or_build_references(
    output_dir: Path,
    czi_path: Path,
    config: ReferenceConfig,
    *,
    force_rebuild: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    fixed_path = output_dir / "fixed_reference.npy"
    moving_path = output_dir / "moving_reference.npy"
    metadata_path = output_dir / "reference_metadata.json"
    if not force_rebuild and fixed_path.exists() and moving_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config") == asdict(config):
            return np.load(fixed_path), np.load(moving_path), [int(value) for value in metadata["z_indices"]]

    czi = CziFile(czi_path)
    dims = czi.get_dims_shape()[0]
    z_size = int(dims["Z"][1] - dims["Z"][0])
    z_indices = list(range(0, z_size, config.z_step))
    fixed = _read_channel_reference(czi, config.fixed_channel, z_indices, config.scale_factor)
    moving = _read_channel_reference(czi, config.moving_channel, z_indices, config.scale_factor)
    fixed = _preprocess_reference(fixed, config.preprocess, config.fixed_clip_percentile)
    moving = _preprocess_reference(moving, config.preprocess, config.moving_clip_percentile)
    np.save(fixed_path, fixed)
    np.save(moving_path, moving)
    metadata_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "z_indices": z_indices,
                "reference_shape_zyx": list(map(int, fixed.shape)),
            },
            indent=2,
        )
    )
    _write_reference_pngs(output_dir, fixed, moving)
    return fixed, moving, z_indices


def _read_channel_reference(
    czi: CziFile,
    channel: int,
    z_indices: list[int],
    scale_factor: float,
) -> np.ndarray:
    planes = []
    for z in z_indices:
        plane = czi.read_mosaic(C=channel, Z=z, scale_factor=scale_factor)
        plane = np.squeeze(plane)
        if plane.ndim != 2:
            raise ValueError(f"Expected 2D mosaic plane after squeeze, got shape {plane.shape}")
        planes.append(plane.astype(np.float32, copy=False))
    return np.stack(planes, axis=0)


def _preprocess_reference(stack: np.ndarray, method: str, clip_percentile: float) -> np.ndarray:
    baseline = np.percentile(stack, 1.0)
    high = np.percentile(stack, clip_percentile)
    normalized = np.clip((stack - baseline) / max(high - baseline, 1.0), 0.0, 1.0).astype(np.float32)
    if method == "raw":
        return normalized
    if method == "sqrt":
        return np.sqrt(normalized, dtype=np.float32)
    if method == "log":
        return (np.log1p(normalized * np.float32(20.0)) / np.log1p(np.float32(20.0))).astype(np.float32)
    if method == "dog":
        fine = ndimage.gaussian_filter(normalized, sigma=(0.5, 0.8, 0.8))
        coarse = ndimage.gaussian_filter(normalized, sigma=(2.0, 4.0, 4.0))
        dog = np.maximum(fine - coarse, 0.0)
        high_dog = np.percentile(dog, 99.9)
        return np.clip(dog / max(high_dog, 1e-6), 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported preprocess mode {method}")


def _run_mode(
    output_dir: Path,
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    z_indices: list[int],
    config: ReferenceConfig,
    mode: str,
    max_iterations: int,
    ftol: float,
) -> dict[str, object]:
    initial_shift = estimate_translation_gpu(cp.asarray(fixed), cp.asarray(moving))
    matrix, offset, stage_modes = _fit_transform(
        fixed,
        moving,
        mode=mode,
        initial_shift_zyx=initial_shift,
        ftol=ftol,
        max_iterations=max_iterations,
        downsample=1,
    )
    registered = apply_output_to_input_transform_gpu(moving[:, np.newaxis], matrix, offset)[:, 0]
    before = _correlation(fixed, moving)
    after = _correlation(fixed, registered)
    ncc_gain = after - before
    run_dir = output_dir / mode
    run_dir.mkdir(exist_ok=True)
    _write_registration_pngs(run_dir, fixed, moving, registered)
    full_matrix, full_offset = _scale_reference_transform_to_full_resolution(
        matrix,
        offset,
        z_step=config.z_step,
        scale_factor=config.scale_factor,
    )
    metadata = {
        "mode": mode,
        "stage_modes": list(stage_modes),
        "reference_shape_zyx": list(map(int, fixed.shape)),
        "z_indices": z_indices,
        "initial_shift_reference_zyx": list(map(float, initial_shift)),
        "output_to_input_matrix_reference_zyx": matrix.tolist(),
        "output_to_input_offset_reference_zyx": offset.tolist(),
        "output_to_input_matrix_full_zyx": full_matrix.tolist(),
        "output_to_input_offset_full_zyx": full_offset.tolist(),
        "correlation_before": before,
        "correlation_after": after,
        "correlation_gain": ncc_gain,
    }
    (run_dir / "registration.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _scale_reference_transform_to_full_resolution(
    matrix_reference: np.ndarray,
    offset_reference: np.ndarray,
    *,
    z_step: int,
    scale_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    full_to_ref = np.diag([1.0 / z_step, scale_factor, scale_factor]).astype(np.float32)
    ref_to_full = np.diag([float(z_step), 1.0 / scale_factor, 1.0 / scale_factor]).astype(np.float32)
    matrix_full = ref_to_full @ matrix_reference @ full_to_ref
    offset_full = ref_to_full @ offset_reference
    return matrix_full.astype(np.float32), offset_full.astype(np.float32)


def _write_reference_pngs(output_dir: Path, fixed: np.ndarray, moving: np.ndarray) -> None:
    Image.fromarray(_scale_uint8(fixed.max(axis=0))).save(output_dir / "fixed_reference_max_projection.png")
    Image.fromarray(_scale_uint8(moving.max(axis=0))).save(output_dir / "moving_reference_max_projection.png")
    overlay = np.stack([_scale_uint8(fixed.max(axis=0)), _scale_uint8(moving.max(axis=0)), np.zeros(fixed.shape[1:], dtype=np.uint8)], axis=-1)
    Image.fromarray(overlay).save(output_dir / "before_overlay_max_projection.png")


def _write_registration_pngs(run_dir: Path, fixed: np.ndarray, moving: np.ndarray, registered: np.ndarray) -> None:
    fixed_mip = fixed.max(axis=0)
    moving_mip = moving.max(axis=0)
    registered_mip = registered.max(axis=0)
    Image.fromarray(_scale_uint8(registered_mip)).save(run_dir / "registered_reference_max_projection.png")
    before = np.stack([_scale_uint8(fixed_mip), _scale_uint8(moving_mip), np.zeros(fixed.shape[1:], dtype=np.uint8)], axis=-1)
    after = np.stack([_scale_uint8(fixed_mip), _scale_uint8(registered_mip), np.zeros(fixed.shape[1:], dtype=np.uint8)], axis=-1)
    Image.fromarray(before).save(run_dir / "before_overlay_max_projection.png")
    Image.fromarray(after).save(run_dir / "after_overlay_max_projection.png")
    delta = np.abs(_scale_uint8(fixed_mip).astype(np.int16) - _scale_uint8(registered_mip).astype(np.int16)).astype(np.uint8)
    Image.fromarray(delta).save(run_dir / "after_absdiff_max_projection.png")


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a.astype(np.float64, copy=False) - float(np.mean(a))
    b0 = b.astype(np.float64, copy=False) - float(np.mean(b))
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a0 * b0) / denom)


if __name__ == "__main__":
    main()
