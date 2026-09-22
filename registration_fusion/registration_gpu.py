from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import affine_transform as gpu_affine_transform
from scipy.optimize import minimize


EPS = np.float32(1e-6)
VALID_GPU_REGISTRATION_MODES = {
    "translation",
    "rigid",
    "similarity-7dof",
    "scale-9dof",
    "affine-12dof",
}


@dataclass(frozen=True)
class GpuRegistrationResult:
    registered: np.ndarray
    mode: str
    stage_modes: tuple[str, ...]
    initial_shift_zyx: tuple[float, float, float]
    output_to_input_matrix_zyx: np.ndarray
    output_to_input_offset_zyx: np.ndarray
    correlation: float

    @property
    def shift_zyx(self) -> tuple[float, float, float]:
        return self.initial_shift_zyx


def require_cuda_gpu() -> None:
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError("A CUDA-capable GPU is required for this migrated dual-view pipeline") from exc
    if device_count < 1:
        raise RuntimeError("A CUDA-capable GPU is required for this migrated dual-view pipeline")


def register_stack_pair_gpu(
    fixed_zcyx: np.ndarray,
    moving_zcyx: np.ndarray,
    *,
    mode: str = "translation",
    ftol: float = 1e-4,
    max_iterations: int = 50,
    downsample: int = 2,
) -> GpuRegistrationResult:
    """Register ``moving`` onto ``fixed`` with GPU phase initialization and GPU affine resampling."""
    require_cuda_gpu()
    if mode not in VALID_GPU_REGISTRATION_MODES:
        valid = ", ".join(sorted(VALID_GPU_REGISTRATION_MODES))
        raise ValueError(f"Unsupported GPU registration mode {mode!r}; expected one of: {valid}")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if downsample <= 0:
        raise ValueError("downsample must be positive")

    fixed = _as_zcyx(fixed_zcyx, "fixed")
    moving = _as_zcyx(moving_zcyx, "moving")
    if fixed.shape != moving.shape:
        raise ValueError(f"Expected matching ZCYX shapes, got fixed={fixed.shape}, moving={moving.shape}")

    fixed_ref = cp.asarray(fixed.max(axis=1), dtype=cp.float32)
    moving_ref = cp.asarray(moving.max(axis=1), dtype=cp.float32)
    initial_shift = estimate_translation_gpu(fixed_ref, moving_ref)

    matrix_zyx, offset_zyx, stage_modes = _fit_transform(
        cp.asnumpy(fixed_ref),
        cp.asnumpy(moving_ref),
        mode=mode,
        initial_shift_zyx=initial_shift,
        ftol=ftol,
        max_iterations=max_iterations,
        downsample=downsample,
    )
    registered = apply_output_to_input_transform_gpu(moving, matrix_zyx, offset_zyx)
    return GpuRegistrationResult(
        registered=registered,
        mode=mode,
        stage_modes=stage_modes,
        initial_shift_zyx=initial_shift,
        output_to_input_matrix_zyx=matrix_zyx,
        output_to_input_offset_zyx=offset_zyx,
        correlation=_correlation(fixed.astype(np.float32, copy=False), registered),
    )


def register_translation_gpu(fixed_zcyx: np.ndarray, moving_zcyx: np.ndarray) -> GpuRegistrationResult:
    return register_stack_pair_gpu(fixed_zcyx, moving_zcyx, mode="translation", max_iterations=0)


def estimate_translation_gpu(fixed_zyx: cp.ndarray, moving_zyx: cp.ndarray) -> tuple[float, float, float]:
    """Return the integer-pixel shift that maps ``moving`` onto ``fixed``."""
    if fixed_zyx.shape != moving_zyx.shape:
        raise ValueError(f"Expected matching shapes, got {fixed_zyx.shape} and {moving_zyx.shape}")
    fixed = fixed_zyx.astype(cp.float32, copy=False) - cp.mean(fixed_zyx)
    moving = moving_zyx.astype(cp.float32, copy=False) - cp.mean(moving_zyx)
    product = cp.fft.fftn(fixed) * cp.conj(cp.fft.fftn(moving))
    product /= cp.maximum(cp.abs(product), EPS)
    corr = cp.abs(cp.fft.ifftn(product))
    max_index = cp.unravel_index(cp.argmax(corr), corr.shape)
    shift = [float(cp.asnumpy(value)) for value in max_index]
    for axis, size in enumerate(fixed_zyx.shape):
        if shift[axis] > size // 2:
            shift[axis] -= size
    return float(shift[0]), float(shift[1]), float(shift[2])


def apply_output_to_input_transform_gpu(
    moving_zcyx: np.ndarray,
    matrix_zyx: np.ndarray,
    offset_zyx: np.ndarray,
) -> np.ndarray:
    moving = _as_zcyx(moving_zcyx, "moving")
    matrix_gpu = cp.asarray(matrix_zyx, dtype=cp.float32)
    offset_gpu = cp.asarray(offset_zyx, dtype=cp.float32)
    output = np.empty(moving.shape, dtype=np.float32)

    for channel in range(moving.shape[1]):
        moving_gpu = cp.asarray(moving[:, channel], dtype=cp.float32)
        registered_gpu = gpu_affine_transform(
            moving_gpu,
            matrix_gpu,
            offset_gpu,
            output_shape=moving.shape[0:1] + moving.shape[2:4],
            order=1,
            mode="constant",
            cval=0.0,
        )
        output[:, channel] = cp.asnumpy(registered_gpu)
    return output


def _fit_transform(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    *,
    mode: str,
    initial_shift_zyx: tuple[float, float, float],
    ftol: float,
    max_iterations: int,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if mode == "translation" and max_iterations == 0:
        matrix, offset = _output_to_input_from_model(
            np.eye(3, dtype=np.float32),
            np.asarray(initial_shift_zyx),
            fixed_zyx.shape,
        )
        return matrix, offset, ("translation",)

    fixed_ds = fixed_zyx[::downsample, ::downsample, ::downsample].astype(np.float32, copy=False)
    moving_ds = moving_zyx[::downsample, ::downsample, ::downsample].astype(np.float32, copy=False)
    initial_shift_ds = np.asarray(initial_shift_zyx, dtype=np.float32) / np.float32(downsample)
    fixed_gpu = cp.asarray(fixed_ds, dtype=cp.float32)
    moving_gpu = cp.asarray(moving_ds, dtype=cp.float32)

    fixed_centered = fixed_gpu - cp.mean(fixed_gpu)
    fixed_norm = cp.linalg.norm(fixed_centered)
    if float(cp.asnumpy(fixed_norm)) == 0.0:
        raise ValueError("fixed stack is constant; registration requires image contrast")

    def objective_for_model(model_matrix: np.ndarray, model_translation: np.ndarray) -> float:
        matrix, offset = _output_to_input_from_model(model_matrix, model_translation, fixed_ds.shape)
        registered = gpu_affine_transform(
            moving_gpu,
            cp.asarray(matrix, dtype=cp.float32),
            cp.asarray(offset, dtype=cp.float32),
            output_shape=fixed_ds.shape,
            order=1,
            mode="constant",
            cval=0.0,
        )
        registered_centered = registered - cp.mean(registered)
        denom = fixed_norm * cp.linalg.norm(registered_centered)
        if float(cp.asnumpy(denom)) == 0.0:
            return 1.0
        corr = cp.sum(fixed_centered * registered_centered) / denom
        return -float(cp.asnumpy(corr))

    if mode == "affine-12dof":
        stage_modes = ("translation", "rigid", "scale-9dof", "affine-12dof")
        model_matrix_ds = np.eye(3, dtype=np.float32)
        model_translation_ds = np.zeros(3, dtype=np.float32)
        for stage_mode in stage_modes:
            model_matrix_ds, model_translation_ds = _fit_stage_delta(
                stage_mode,
                model_matrix_ds,
                model_translation_ds,
                initial_shift_ds if stage_mode == "translation" else np.zeros(3, dtype=np.float32),
                objective_for_model,
                ftol=ftol,
                max_iterations=max_iterations,
            )
    else:
        stage_modes = (mode,)
        initial_params = _initial_params(mode, initial_shift_ds)
        if max_iterations > 0:
            result = minimize(
                lambda params: objective_for_model(*_model_from_params(mode, params)),
                initial_params,
                method="Powell",
                options={"maxiter": max_iterations, "ftol": ftol, "xtol": ftol, "disp": False},
            )
            params = result.x.astype(np.float32, copy=False)
        else:
            params = initial_params
        model_matrix_ds, model_translation_ds = _model_from_params(mode, params)

    model_translation = model_translation_ds * np.float32(downsample)
    matrix, offset = _output_to_input_from_model(model_matrix_ds, model_translation, fixed_zyx.shape)
    return matrix, offset, stage_modes


def _fit_stage_delta(
    stage_mode: str,
    base_matrix: np.ndarray,
    base_translation: np.ndarray,
    initial_translation: np.ndarray,
    objective_for_model,
    *,
    ftol: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    initial_params = _initial_params(stage_mode, initial_translation)

    def compose(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_matrix, delta_translation = _model_from_params(stage_mode, params)
        return (
            (delta_matrix @ base_matrix).astype(np.float32),
            (delta_matrix @ base_translation + delta_translation).astype(np.float32),
        )

    if max_iterations > 0:
        result = minimize(
            lambda params: objective_for_model(*compose(params)),
            initial_params,
            method="Powell",
            options={"maxiter": max_iterations, "ftol": ftol, "xtol": ftol, "disp": False},
        )
        params = result.x.astype(np.float32, copy=False)
    else:
        params = initial_params
    return compose(params)


def _initial_params(mode: str, initial_shift_zyx: np.ndarray) -> np.ndarray:
    if mode == "translation":
        return initial_shift_zyx.astype(np.float32, copy=True)
    if mode == "rigid":
        return np.r_[np.zeros(3, dtype=np.float32), initial_shift_zyx]
    if mode == "similarity-7dof":
        return np.r_[np.zeros(4, dtype=np.float32), initial_shift_zyx]
    if mode == "scale-9dof":
        return np.r_[np.zeros(6, dtype=np.float32), initial_shift_zyx]
    return np.r_[np.zeros(9, dtype=np.float32), initial_shift_zyx]


def _model_from_params(mode: str, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float32)
    if mode == "translation":
        return np.eye(3, dtype=np.float32), params[0:3]
    if mode == "rigid":
        return _rotation_zyx(params[0:3]), params[3:6]
    if mode == "similarity-7dof":
        return np.exp(params[3]) * _rotation_zyx(params[0:3]), params[4:7]
    if mode == "scale-9dof":
        scales = np.diag(np.exp(params[3:6]).astype(np.float32))
        return _rotation_zyx(params[0:3]) @ scales, params[6:9]
    return (np.eye(3, dtype=np.float32) + params[0:9].reshape(3, 3)).astype(np.float32), params[9:12]


def _rotation_zyx(angles: np.ndarray) -> np.ndarray:
    rz, ry, rx = [float(value) for value in angles]
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    rot_z = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_x = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _output_to_input_from_model(
    moving_to_fixed_matrix: np.ndarray,
    moving_to_fixed_translation: np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    center = (np.asarray(shape_zyx, dtype=np.float32) - np.float32(1.0)) / np.float32(2.0)
    inverse = np.linalg.inv(moving_to_fixed_matrix).astype(np.float32)
    offset = center - inverse @ (center + moving_to_fixed_translation.astype(np.float32))
    return inverse, offset.astype(np.float32)


def _as_zcyx(stack: np.ndarray, name: str) -> np.ndarray:
    data = np.asarray(stack)
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]
    if data.ndim != 4:
        raise ValueError(f"Expected {name} to be ZYX or ZCYX, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return data


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a.astype(np.float64, copy=False) - float(np.mean(a))
    b0 = b.astype(np.float64, copy=False) - float(np.mean(b))
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0:
        return 0.0
    return float(np.sum(a0 * b0) / denom)
