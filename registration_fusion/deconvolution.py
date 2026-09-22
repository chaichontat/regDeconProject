from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import convolve as gpu_convolve

from .registration_gpu import GpuRegistrationResult, register_translation_gpu, require_cuda_gpu
from .simulate_dual_view import (
    DEFAULT_PSF_PATH,
    PsfCropConfig,
    crop_psf_like_fishtools,
    rotate_psf_for_view,
    transpose_psf_for_orthogonal_view,
)


EPS = np.float32(1e-6)


@dataclass(frozen=True)
class WbProjectorConfig:
    alpha: float = 0.02
    beta: float = 0.02
    n: int = 10
    sigma_g: float = 1.7


def deconvolve_dual_view_gpu(
    view_a_zcyx: np.ndarray,
    view_b_zcyx: np.ndarray,
    psf_a_zyx: np.ndarray,
    psf_b_zyx: np.ndarray,
    *,
    iterations: int = 1,
    projector_config: WbProjectorConfig = WbProjectorConfig(),
) -> np.ndarray:
    """Run alternating dual-view WB Richardson-Lucy deconvolution on the GPU."""
    require_cuda_gpu()
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    view_a = _as_zcyx(view_a_zcyx, "view_a")
    view_b = _as_zcyx(view_b_zcyx, "view_b")
    if view_a.shape != view_b.shape:
        raise ValueError(f"Expected matching ZCYX shapes, got view_a={view_a.shape}, view_b={view_b.shape}")

    image_a = cp.maximum(cp.asarray(view_a, dtype=cp.float32), EPS)
    image_b = cp.maximum(cp.asarray(view_b, dtype=cp.float32), EPS)
    forward_a, back_a = make_wb_projectors_gpu(psf_a_zyx, projector_config)
    forward_b, back_b = make_wb_projectors_gpu(psf_b_zyx, projector_config)
    estimate = cp.maximum((image_a + image_b) * np.float32(0.5), EPS)

    for _ in range(iterations):
        estimate *= _correction(image_a, estimate, forward_a, back_a)
        estimate = cp.maximum(estimate, EPS)
        estimate *= _correction(image_b, estimate, forward_b, back_b)
        estimate = cp.maximum(estimate, EPS)

    return cp.asnumpy(estimate)


def default_dual_view_psfs(
    psf_path: Path = DEFAULT_PSF_PATH,
    crop: PsfCropConfig = PsfCropConfig(),
    *,
    view_angle_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    psf_a = crop_psf_like_fishtools(psf_path, crop)
    if np.isclose(view_angle_deg, 90.0):
        return psf_a, transpose_psf_for_orthogonal_view(psf_a)
    return psf_a, rotate_psf_for_view(psf_a, view_angle_deg)


def make_wb_projectors_gpu(
    psf_zyx: np.ndarray,
    config: WbProjectorConfig = WbProjectorConfig(),
) -> tuple[cp.ndarray, cp.ndarray]:
    psf = cp.asarray(_normalized_psf(psf_zyx), dtype=cp.float32)
    forward = psf[:, cp.newaxis, :, :]
    return forward, _wiener_butterworth_back_projector(psf, config)[:, cp.newaxis, :, :]


def _correction(
    image: cp.ndarray,
    estimate: cp.ndarray,
    forward_projector: cp.ndarray,
    back_projector: cp.ndarray,
) -> cp.ndarray:
    blurred = cp.maximum(gpu_convolve(estimate, forward_projector, mode="reflect"), EPS)
    ratio = image / blurred
    return gpu_convolve(ratio, back_projector, mode="reflect")


def _wiener_butterworth_back_projector(psf: cp.ndarray, config: WbProjectorConfig) -> cp.ndarray:
    # MATLAB BackProjector.m builds the Wiener term from flipPSF(PSF_fp), then
    # returns fftshift(real(ifftn(OTF_bp))) as the spatial back-projector.
    flipped_psf = cp.flip(psf)
    otf_flip = cp.fft.fftn(cp.fft.ifftshift(flipped_psf))
    otf_flip /= cp.maximum(cp.max(cp.abs(otf_flip)), EPS)
    wiener = otf_flip / (cp.abs(otf_flip) ** 2 + np.float32(config.alpha))
    kz = cp.fft.fftfreq(psf.shape[0])
    ky = cp.fft.fftfreq(psf.shape[1])
    kx = cp.fft.fftfreq(psf.shape[2])
    kk = cp.sqrt((cp.array(cp.meshgrid(kz, ky, kx, indexing="ij")) ** 2).sum(axis=0))
    cutoff = np.float32(1.0 / (0.5 * 2.355 * config.sigma_g))
    eps = np.float32(np.sqrt(1.0 / (config.beta**2) - 1.0))
    butterworth = 1.0 / cp.sqrt(1.0 + eps**2 * (kk / cutoff) ** (2 * config.n))
    return cp.fft.fftshift(cp.real(cp.fft.ifftn(wiener * butterworth))).astype(cp.float32)


def _normalized_psf(psf: np.ndarray) -> np.ndarray:
    data = np.asarray(psf, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D PSF, got shape {data.shape}")
    total = float(data.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF sum must be finite and positive")
    return data / total


def _as_zcyx(stack: np.ndarray, name: str) -> np.ndarray:
    data = np.asarray(stack)
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]
    if data.ndim != 4:
        raise ValueError(f"Expected {name} to be ZYX or ZCYX, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return data


__all__ = [
    "GpuRegistrationResult",
    "WbProjectorConfig",
    "deconvolve_dual_view_gpu",
    "default_dual_view_psfs",
    "make_wb_projectors_gpu",
    "register_translation_gpu",
]
