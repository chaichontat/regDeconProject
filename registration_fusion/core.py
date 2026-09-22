from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from skimage.registration import phase_cross_correlation


VALID_MODES = {
    "translation",
    "rigid",
    "similarity-7dof",
    "scale-9dof",
    "affine-12dof",
}


@dataclass(frozen=True)
class RegistrationResult:
    registered: np.ndarray
    transform: sitk.Transform
    initial_shift_zyx: tuple[float, float, float]
    final_metric_value: float


def register_stack_pair(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    mode: str = "rigid",
    ftol: float = 1e-4,
    max_iterations: int = 200,
    sampling_percentage: float = 0.2,
) -> RegistrationResult:
    """Register ``moving`` onto ``fixed`` and return the resampled moving stack."""
    if mode not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Unsupported registration mode {mode!r}; expected one of: {valid}")
    if fixed.ndim != 3 or moving.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got fixed={fixed.shape}, moving={moving.shape}")
    if fixed.shape != moving.shape:
        raise ValueError(f"Expected matching shapes, got fixed={fixed.shape}, moving={moving.shape}")
    if not 0 < sampling_percentage <= 1:
        raise ValueError("sampling_percentage must be in the interval (0, 1]")

    fixed_float = _as_nonconstant_float(fixed, "fixed")
    moving_float = _as_nonconstant_float(moving, "moving")
    fixed_image = sitk.GetImageFromArray(fixed_float)
    moving_image = sitk.GetImageFromArray(moving_float)

    initial_shift_zyx = estimate_mip_translation(fixed_float, moving_float)
    transform = _make_transform(mode, fixed.shape, initial_shift_zyx)

    if max_iterations > 0:
        transform = _refine_transform(
            fixed_image,
            moving_image,
            transform,
            ftol=ftol,
            max_iterations=max_iterations,
            sampling_percentage=sampling_percentage,
        )

    registered_image = sitk.Resample(
        moving_image,
        fixed_image,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    registered = sitk.GetArrayFromImage(registered_image)
    metric_value = _correlation(fixed_float, registered)
    return RegistrationResult(
        registered=registered,
        transform=transform,
        initial_shift_zyx=initial_shift_zyx,
        final_metric_value=metric_value,
    )


def estimate_mip_translation(fixed: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """Estimate the moving-to-fixed shift from XY and ZX maximum projections."""
    xy_shift, _, _ = phase_cross_correlation(fixed.max(axis=0), moving.max(axis=0), upsample_factor=1)
    zx_shift, _, _ = phase_cross_correlation(fixed.max(axis=1), moving.max(axis=1), upsample_factor=1)

    dy_xy, dx = (float(v) for v in xy_shift)
    dz, dx_zx = (float(v) for v in zx_shift)
    if abs(dx - dx_zx) <= 2:
        x_shift = 0.5 * (dx + dx_zx)
    else:
        x_shift = dx
    return dz, dy_xy, x_shift


def _as_nonconstant_float(stack: np.ndarray, name: str) -> np.ndarray:
    data = np.asarray(stack, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} stack contains NaN or infinite values")
    if float(data.max()) == float(data.min()):
        raise ValueError(f"{name} stack is constant; registration requires image contrast")
    return data


def _make_transform(
    mode: str,
    shape_zyx: tuple[int, int, int],
    initial_shift_zyx: tuple[float, float, float],
) -> sitk.Transform:
    center_xyz = (
        (shape_zyx[2] - 1) / 2.0,
        (shape_zyx[1] - 1) / 2.0,
        (shape_zyx[0] - 1) / 2.0,
    )
    # SimpleITK transforms map fixed output coordinates to moving input coordinates.
    offset_xyz = tuple(-v for v in reversed(initial_shift_zyx))

    if mode == "translation":
        transform = sitk.TranslationTransform(3)
        transform.SetOffset(offset_xyz)
        return transform
    if mode == "rigid":
        transform = sitk.Euler3DTransform()
        transform.SetCenter(center_xyz)
        transform.SetTranslation(offset_xyz)
        return transform
    if mode == "similarity-7dof":
        transform = sitk.Similarity3DTransform()
        transform.SetCenter(center_xyz)
        transform.SetTranslation(offset_xyz)
        return transform
    if mode == "scale-9dof":
        transform = sitk.ScaleVersor3DTransform()
        transform.SetCenter(center_xyz)
        transform.SetTranslation(offset_xyz)
        return transform

    transform = sitk.AffineTransform(3)
    transform.SetCenter(center_xyz)
    transform.SetTranslation(offset_xyz)
    return transform


def _refine_transform(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    transform: sitk.Transform,
    *,
    ftol: float,
    max_iterations: int,
    sampling_percentage: float,
) -> sitk.Transform:
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    if sampling_percentage < 1:
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_percentage, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=ftol,
        numberOfIterations=max_iterations,
        relaxationFactor=0.5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=False)
    return registration.Execute(fixed_image, moving_image)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a.astype(np.float64, copy=False) - float(np.mean(a))
    b0 = b.astype(np.float64, copy=False) - float(np.mean(b))
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0:
        return 0.0
    return float(np.sum(a0 * b0) / denom)
