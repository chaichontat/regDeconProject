from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FlatfieldProfile:
    flatfield_yx: np.ndarray
    source_path: str


def load_flatfield_profile(path: Path | str) -> FlatfieldProfile:
    """Load only the multiplicative BaSiC flatfield from a pickle payload."""
    p = Path(path)
    loaded = pickle.loads(p.read_bytes())
    basic = loaded.get("basic") if isinstance(loaded, dict) else loaded
    if basic is None or not hasattr(basic, "flatfield"):
        raise ValueError(f"BaSiC profile at {p} does not contain a flatfield")
    flatfield = np.asarray(basic.flatfield, dtype=np.float32)
    validate_flatfield(flatfield)
    return FlatfieldProfile(flatfield_yx=flatfield, source_path=str(p))


def validate_flatfield(flatfield_yx: np.ndarray) -> None:
    flatfield = np.asarray(flatfield_yx)
    if flatfield.ndim != 2:
        raise ValueError(f"flatfield must be a 2D YX array, got shape {flatfield.shape}")
    if not np.all(np.isfinite(flatfield)):
        raise ValueError("flatfield contains non-finite values")
    if np.any(flatfield <= 0):
        raise ValueError("flatfield must be strictly positive")


def flatfield_metadata(profile: FlatfieldProfile | None) -> dict[str, object] | None:
    if profile is None:
        return None
    return {
        "source_path": profile.source_path,
        "shape_yx": list(profile.flatfield_yx.shape),
        "mode": "multiply_by_precomputed_inverse_flatfield_during_fusion",
    }
