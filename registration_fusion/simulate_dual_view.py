from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage


DEFAULT_SHAPE_ZYXC = (500, 1920, 1920, 3)
DEFAULT_PSF_PATH = Path("/working/fishtools/data/PSF GL.tif")


@dataclass(frozen=True)
class PsfCropConfig:
    step: int = 6
    max_z: int = 7
    size: int = 31
    center: int = 50


@dataclass(frozen=True)
class SimulationConfig:
    shape_zyxc: tuple[int, int, int, int] = DEFAULT_SHAPE_ZYXC
    spots_per_channel: int = 2000
    seed: int = 0
    min_amplitude: float = 8_000.0
    max_amplitude: float = 60_000.0
    background: float = 100.0
    read_noise_sigma: float = 0.0
    psf_path: Path = DEFAULT_PSF_PATH
    psf_crop: PsfCropConfig = PsfCropConfig()
    view_angle_deg: float = 90.0
    compression: int | str | None = None


SPOT_DTYPE = np.dtype(
    [
        ("channel", np.int16),
        ("z", np.int32),
        ("y", np.int32),
        ("x", np.int32),
        ("amplitude", np.float32),
    ]
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic dual-view light-sheet data.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--psf", default=DEFAULT_PSF_PATH, type=Path)
    parser.add_argument("--shape", default="500,1920,1920,3", help="Z,Y,X,C")
    parser.add_argument("--spots-per-channel", default=2000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--background", default=100.0, type=float)
    parser.add_argument("--read-noise-sigma", default=0.0, type=float)
    parser.add_argument("--min-amplitude", default=8_000.0, type=float)
    parser.add_argument("--max-amplitude", default=60_000.0, type=float)
    parser.add_argument("--psf-step", default=6, type=int)
    parser.add_argument("--psf-max-z", default=7, type=int)
    parser.add_argument("--psf-size", default=31, type=int)
    parser.add_argument("--psf-center", default=50, type=int)
    parser.add_argument(
        "--view-angle-deg",
        default=90.0,
        type=float,
        help="Angle between view-A and view-B detection axes; objective-native simulation currently supports 90.",
    )
    parser.add_argument("--compression", type=_parse_compression, help="TIFF compression name or numeric code")
    args = parser.parse_args(argv)

    config = SimulationConfig(
        shape_zyxc=_parse_shape(args.shape),
        spots_per_channel=args.spots_per_channel,
        seed=args.seed,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude,
        background=args.background,
        read_noise_sigma=args.read_noise_sigma,
        psf_path=args.psf,
        psf_crop=PsfCropConfig(
            step=args.psf_step,
            max_z=args.psf_max_z,
            size=args.psf_size,
            center=args.psf_center,
        ),
        view_angle_deg=args.view_angle_deg,
        compression=args.compression,
    )
    generate_dataset(args.output_dir, config)
    return 0


def generate_dataset(output_dir: Path, config: SimulationConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    psf_a = crop_psf_like_fishtools(config.psf_path, config.psf_crop)
    if not np.isclose(config.view_angle_deg, 90.0):
        raise ValueError("objective-native simulation currently supports only a 90 degree view angle")
    psf_b_object_grid = transpose_psf_for_orthogonal_view(psf_a)
    spots = generate_spots(config)
    view_b_spots = transpose_spots_for_orthogonal_view(spots)

    metadata = _metadata(config)
    z_size, y_size, x_size, _ = config.shape_zyxc
    object_shape = (z_size, y_size, x_size)
    view_b_native_shape = (y_size, z_size, x_size)
    _write_ground_truth(output_dir / "ground_truth.tif", spots, config, metadata)
    _write_view(
        output_dir / "view_a.tif",
        spots,
        psf_a,
        config,
        metadata | {"coordinate_system": "view_a_objective_native", "shape_zyxc": list(config.shape_zyxc)},
        shape_zyx=object_shape,
        view_index=0,
    )
    _write_view(
        output_dir / "view_b.tif",
        view_b_spots,
        psf_a,
        config,
        metadata
        | {
            "coordinate_system": "view_b_objective_native",
            "shape_zyxc": [view_b_native_shape[0], view_b_native_shape[1], view_b_native_shape[2], config.shape_zyxc[3]],
            "object_axis_mapping": {"z": "object_y", "y": "object_z", "x": "object_x"},
        },
        shape_zyx=view_b_native_shape,
        view_index=1,
    )
    _write_view(
        output_dir / "view_b_object_grid.tif",
        spots,
        psf_b_object_grid,
        config,
        metadata | {"coordinate_system": "object_grid", "shape_zyxc": list(config.shape_zyxc)},
        shape_zyx=object_shape,
        view_index=1,
    )
    tifffile.imwrite(
        output_dir / "psf_cropped_view_a.tif",
        psf_a.astype(np.float32),
        photometric="minisblack",
        compression=config.compression,
    )
    tifffile.imwrite(
        output_dir / "psf_cropped_view_b.tif",
        psf_b_object_grid.astype(np.float32),
        photometric="minisblack",
        compression=config.compression,
    )
    tifffile.imwrite(
        output_dir / "psf_cropped_view_b_native.tif",
        psf_a.astype(np.float32),
        photometric="minisblack",
        compression=config.compression,
    )
    write_spots_csv(output_dir / "spots.csv", spots)
    (output_dir / "simulation.json").write_text(
        json.dumps(
            _jsonable_config(config)
            | {
                "spot_count": int(spots.size),
                "ground_truth": "ground_truth.tif",
                "objective_native_views": ["view_a.tif", "view_b.tif"],
                "object_grid_views": ["view_a.tif", "view_b_object_grid.tif"],
                "view_b_native_shape_zyxc": [
                    view_b_native_shape[0],
                    view_b_native_shape[1],
                    view_b_native_shape[2],
                    config.shape_zyxc[3],
                ],
            },
            indent=2,
        )
    )


def crop_psf_like_fishtools(path: Path, config: PsfCropConfig = PsfCropConfig()) -> np.ndarray:
    psf_raw = tifffile.imread(path).astype(np.float32, copy=False)
    if psf_raw.ndim != 3 or psf_raw.shape[1] != psf_raw.shape[2]:
        raise ValueError(f"Expected PSF shaped (z, y, x) with square y/x, got {psf_raw.shape}")

    z_indices = _center_index(config.center, psf_raw.shape[0], config.step)
    z_crop = (len(z_indices) - config.max_z) // 2
    if z_crop > 0:
        z_indices = z_indices[z_crop:-z_crop]
    z_indices = z_indices[: config.max_z]

    xy_crop = (psf_raw.shape[1] - config.size) // 2
    if xy_crop < 0:
        raise ValueError(f"PSF size {psf_raw.shape[1]} is smaller than requested crop {config.size}")
    cropped = psf_raw[z_indices][::-1, xy_crop : xy_crop + config.size, xy_crop : xy_crop + config.size]
    total = float(cropped.sum())
    if total <= 0:
        raise ValueError("Cropped PSF has non-positive sum")
    return (cropped / total).astype(np.float32, copy=False)


def orthogonal_psf(psf_zyx: np.ndarray) -> np.ndarray:
    """Rotate the view-A PSF so axial blur runs along Y in the output grid."""
    return transpose_psf_for_orthogonal_view(psf_zyx)


def transpose_psf_for_orthogonal_view(psf_zyx: np.ndarray) -> np.ndarray:
    """Represent a 90-degree objective-native PSF in the view-A object grid."""
    psf = np.asarray(psf_zyx, dtype=np.float32)
    if psf.ndim != 3:
        raise ValueError(f"Expected a 3D PSF, got shape {psf.shape}")
    transposed = np.swapaxes(psf, 0, 1).copy()
    total = float(transposed.sum())
    if total <= 0:
        raise ValueError("Transposed PSF has non-positive sum")
    return transposed


def transpose_spots_for_orthogonal_view(spots: np.ndarray) -> np.ndarray:
    """Map object-grid spots into the native coordinate frame of the orthogonal view."""
    transformed = spots.copy()
    transformed["z"] = spots["y"]
    transformed["y"] = spots["z"]
    transformed["x"] = spots["x"]
    return transformed


def rotate_psf_for_view(psf_zyx: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a detection PSF in the Z/Y plane for a second view."""
    if not np.isfinite(angle_deg):
        raise ValueError("view angle must be finite")
    psf = np.asarray(psf_zyx, dtype=np.float32)
    if psf.ndim != 3:
        raise ValueError(f"Expected a 3D PSF, got shape {psf.shape}")
    rotated = ndimage.rotate(
        psf,
        angle=float(angle_deg),
        axes=(1, 0),
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    rotated = np.clip(rotated, 0, None).astype(np.float32, copy=False)
    total = float(rotated.sum())
    if total <= 0:
        raise ValueError("Rotated PSF has non-positive sum")
    return rotated / np.float32(total)


def generate_spots(config: SimulationConfig) -> np.ndarray:
    z_size, y_size, x_size, channels = config.shape_zyxc
    rng = np.random.default_rng(config.seed)
    spots = np.empty(config.spots_per_channel * channels, dtype=SPOT_DTYPE)
    margin_z = max(config.psf_crop.max_z, config.psf_crop.size) // 2 + 1
    margin_xy = config.psf_crop.size // 2 + 1

    row = 0
    for channel in range(channels):
        count = config.spots_per_channel
        spots["channel"][row : row + count] = channel
        spots["z"][row : row + count] = rng.integers(margin_z, z_size - margin_z, size=count)
        spots["y"][row : row + count] = rng.integers(margin_xy, y_size - margin_xy, size=count)
        spots["x"][row : row + count] = rng.integers(margin_xy, x_size - margin_xy, size=count)
        spots["amplitude"][row : row + count] = rng.uniform(
            config.min_amplitude,
            config.max_amplitude,
            size=count,
        ).astype(np.float32)
        row += count
    return spots


def write_spots_csv(path: Path, spots: np.ndarray) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["channel", "z", "y", "x", "amplitude"])
        for spot in spots:
            writer.writerow(
                [
                    int(spot["channel"]),
                    int(spot["z"]),
                    int(spot["y"]),
                    int(spot["x"]),
                    f"{float(spot['amplitude']):.6f}",
                ]
            )


def _write_view(
    path: Path,
    spots: np.ndarray,
    psf: np.ndarray,
    config: SimulationConfig,
    metadata: dict[str, object],
    *,
    shape_zyx: tuple[int, int, int],
    view_index: int,
) -> None:
    z_size, y_size, x_size = shape_zyx
    channels = config.shape_zyxc[3]
    first_description = json.dumps(metadata | {"axes": "QYX", "shape": [z_size * channels, y_size, x_size]})
    mip = np.zeros((channels, y_size, x_size), dtype=np.float32)
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for z_index in range(z_size):
            for channel in range(channels):
                plane = _render_plane(
                    spots,
                    psf,
                    config,
                    z_index,
                    channel,
                    shape_zyx=shape_zyx,
                    view_index=view_index,
                )
                np.maximum(mip[channel], plane, out=mip[channel])
                tif.write(
                    plane,
                    photometric="minisblack",
                    compression=config.compression,
                    description=first_description if z_index == 0 and channel == 0 else None,
                )
    _write_max_projection_png(path.with_suffix(".png"), mip)


def _render_plane(
    spots: np.ndarray,
    psf: np.ndarray,
    config: SimulationConfig,
    z_index: int,
    channel: int,
    *,
    shape_zyx: tuple[int, int, int],
    view_index: int,
) -> np.ndarray:
    _, y_size, x_size = shape_zyx
    plane = np.full((y_size, x_size), config.background, dtype=np.float32)
    radius_z = psf.shape[0] // 2
    radius_y = psf.shape[1] // 2
    radius_x = psf.shape[2] // 2
    channel_spots = spots[spots["channel"] == channel]
    z0_all = channel_spots["z"] - radius_z
    z1_all = z0_all + psf.shape[0]
    in_z = (z_index >= z0_all) & (z_index < z1_all)

    for spot in channel_spots[in_z]:
        kernel_z = int(z_index - (int(spot["z"]) - radius_z))
        y0 = int(spot["y"]) - radius_y
        y1 = y0 + psf.shape[1]
        x0 = int(spot["x"]) - radius_x
        x1 = x0 + psf.shape[2]
        dst_y0 = max(0, y0)
        dst_y1 = min(y_size, y1)
        dst_x0 = max(0, x0)
        dst_x1 = min(x_size, x1)
        src_y0 = dst_y0 - y0
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x0 = dst_x0 - x0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        plane[dst_y0:dst_y1, dst_x0:dst_x1] += (
            float(spot["amplitude"]) * psf[kernel_z, src_y0:src_y1, src_x0:src_x1]
        )

    if config.read_noise_sigma > 0:
        rng = np.random.default_rng(config.seed + view_index * 1_000_003 + channel * 10_007 + z_index)
        plane += rng.normal(0.0, config.read_noise_sigma, size=plane.shape).astype(np.float32)
    return np.clip(plane, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _write_ground_truth(
    path: Path,
    spots: np.ndarray,
    config: SimulationConfig,
    metadata: dict[str, object],
) -> None:
    z_size, y_size, x_size, channels = config.shape_zyxc
    first_description = json.dumps(
        metadata | {"coordinate_system": "object_grid", "axes": "QYX", "shape": [z_size * channels, y_size, x_size]}
    )
    mip = np.zeros((channels, y_size, x_size), dtype=np.float32)
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for z_index in range(z_size):
            for channel in range(channels):
                plane = np.zeros((y_size, x_size), dtype=np.float32)
                in_plane = spots[(spots["channel"] == channel) & (spots["z"] == z_index)]
                for spot in in_plane:
                    plane[int(spot["y"]), int(spot["x"])] += float(spot["amplitude"])
                np.maximum(mip[channel], plane, out=mip[channel])
                tif.write(
                    np.clip(plane, 0, np.iinfo(np.uint16).max).astype(np.uint16),
                    photometric="minisblack",
                    compression=config.compression,
                    description=first_description if z_index == 0 and channel == 0 else None,
                )
    _write_max_projection_png(path.with_suffix(".png"), mip)


def _write_max_projection_png(path: Path, mip_cyx: np.ndarray) -> None:
    if mip_cyx.shape[0] == 1:
        image = _scale_uint8(mip_cyx[0])
    elif mip_cyx.shape[0] == 3:
        image = np.stack([_scale_uint8(mip_cyx[channel]) for channel in range(3)], axis=-1)
    else:
        image = _scale_uint8(np.max(mip_cyx, axis=0))
    Image.fromarray(image).save(path)


def _scale_uint8(image: np.ndarray) -> np.ndarray:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [0.1, 99.9])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (np.asarray(image, dtype=np.float32) - np.float32(low)) / np.float32(high - low)
    return np.clip(scaled * np.float32(255.0), 0, 255).astype(np.uint8)


def _center_index(center: int, nz: int, step: int) -> list[int]:
    indices: list[int] = []
    current = center
    while current > 0:
        indices.insert(0, current)
        current -= step
    indices.extend(list(range(center, nz, step))[1 : len(indices)])
    return indices


def _metadata(config: SimulationConfig) -> dict[str, object]:
    return {
        "axes": "QYX",
        "channels": [f"channel_{idx}" for idx in range(config.shape_zyxc[3])],
        "simulation": _jsonable_config(config),
        "plane_order": "z-major, channel-minor; reshape pages to (z, c, y, x)",
    }


def _jsonable_config(config: SimulationConfig) -> dict[str, object]:
    data = asdict(config)
    data["psf_path"] = str(config.psf_path)
    return data


def _parse_shape(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("--shape must have four comma-separated integers: Z,Y,X,C")
    if any(part <= 0 for part in parts):
        raise ValueError("--shape dimensions must be positive")
    return parts


def _parse_compression(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
