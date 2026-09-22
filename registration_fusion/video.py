from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from xml.etree.ElementTree import Element

import cv2
import numpy as np
import zarr
from aicspylibczi import CziFile
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


@dataclass(frozen=True)
class ZarrVideoResult:
    output_path: str
    input_path: str
    shape_zcyx: tuple[int, int, int, int]
    channels: tuple[int, ...]
    frame_count: int
    output_width: int
    output_height: int
    fps: float
    low: float
    high: float


def write_zarr_slice_video(
    input_zarr: Path,
    output_video: Path,
    *,
    channels: tuple[int, ...] | None = None,
    height: int = 1080,
    width: int | None = None,
    fps: float = 24.0,
    z_step: int = 1,
    max_frames: int | None = None,
    sample_frames: int = 64,
    sample_stride: int | None = None,
    crf: int = 18,
    preset: str = "medium",
    scale_bar_um: float | None = None,
    pixel_size_um: float | None = None,
) -> ZarrVideoResult:
    """Render a ZCYX Zarr as a streaming Z-slice MP4 without loading the stack."""
    if height <= 0:
        raise ValueError("height must be positive")
    if width is not None and width <= 0:
        raise ValueError("width must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if z_step <= 0:
        raise ValueError("z_step must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if sample_frames <= 0:
        raise ValueError("sample_frames must be positive")
    if sample_stride is not None and sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if scale_bar_um is not None and scale_bar_um <= 0:
        raise ValueError("scale_bar_um must be positive")
    if pixel_size_um is not None and pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")

    array = zarr.open_array(input_zarr, mode="r")
    shape = tuple(int(axis) for axis in array.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected ZCYX Zarr array, got shape {shape}")
    z_count, channel_count, source_y, source_x = shape
    selected_channels = _selected_channels(channel_count, channels)
    frame_indices = tuple(range(0, z_count, z_step))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError("No frames selected")

    output_width, output_height = _video_size(source_y, source_x, height=height, width=width)
    display_pixel_size_um = None
    if scale_bar_um is not None:
        source_pixel_size_um = pixel_size_um if pixel_size_um is not None else _infer_zarr_x_pixel_size_um(array)
        if source_pixel_size_um is None:
            raise ValueError("scale_bar_um requires pixel_size_um or a Zarr source_czi attr with CZI scaling metadata")
        display_pixel_size_um = source_x * source_pixel_size_um / output_width

    low, high = _sample_intensity_range(
        array,
        frame_indices=frame_indices,
        channels=selected_channels,
        source_y=source_y,
        sample_frames=sample_frames,
        sample_stride=sample_stride,
    )

    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    input_pix_fmt = "rgb24" if scale_bar_um is not None else "gray"
    process = _start_ffmpeg(
        output_video,
        width=output_width,
        height=output_height,
        fps=fps,
        crf=crf,
        preset=preset,
        input_pix_fmt=input_pix_fmt,
    )
    stream = process.stdin
    if stream is None:
        process.kill()
        process.wait()
        raise RuntimeError("ffmpeg stdin pipe was not created")
    try:
        _write_frames(
            stream,
            array,
            frame_indices=frame_indices,
            channels=selected_channels,
            width=output_width,
            height=output_height,
            low=low,
            high=high,
            scale_bar_um=scale_bar_um,
            display_pixel_size_um=display_pixel_size_um,
        )
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        stream.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    return ZarrVideoResult(
        output_path=str(output_video),
        input_path=str(input_zarr),
        shape_zcyx=shape,
        channels=selected_channels,
        frame_count=len(frame_indices),
        output_width=output_width,
        output_height=output_height,
        fps=float(fps),
        low=float(low),
        high=float(high),
    )


def _selected_channels(channel_count: int, channels: tuple[int, ...] | None) -> tuple[int, ...]:
    selected = tuple(range(channel_count)) if not channels else tuple(int(channel) for channel in channels)
    for channel in selected:
        if channel < 0 or channel >= channel_count:
            raise ValueError(f"Channel {channel} is outside available range 0..{channel_count - 1}")
    return selected


def _video_size(source_y: int, source_x: int, *, height: int, width: int | None) -> tuple[int, int]:
    output_height = _even(height)
    output_width = _even(round(source_x * (output_height / source_y))) if width is None else _even(width)
    if output_width < 2 or output_height < 2:
        raise ValueError("Output video dimensions must be at least 2x2")
    return output_width, output_height


def _even(value: int) -> int:
    return max(2, int(value) - int(value) % 2)


def _sample_intensity_range(
    array: zarr.Array,
    *,
    frame_indices: tuple[int, ...],
    channels: tuple[int, ...],
    source_y: int,
    sample_frames: int,
    sample_stride: int | None,
) -> tuple[float, float]:
    if sample_stride is None:
        sample_stride = max(1, source_y // 512)
    sampled_indices = np.linspace(0, len(frame_indices) - 1, min(sample_frames, len(frame_indices)), dtype=int)
    samples: list[np.ndarray] = []
    for index in sampled_indices:
        frame = _read_plane(array, frame_indices[int(index)], channels, spatial_stride=sample_stride)
        finite = frame[np.isfinite(frame)]
        if finite.size:
            samples.append(finite.astype(np.float32, copy=False).reshape(-1))
    if not samples:
        return 0.0, 1.0

    sample = np.concatenate(samples)
    low, high = np.percentile(sample, [0.1, 99.9])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(sample))
        high = float(np.max(sample))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _infer_zarr_x_pixel_size_um(array: zarr.Array) -> float | None:
    attrs = array.attrs.asdict()
    source_czi = attrs.get("source_czi")
    if not source_czi:
        return None
    try:
        return _czi_x_pixel_size_um(Path(str(source_czi)))
    except Exception as exc:
        raise ValueError(
            f"Could not infer pixel size from source_czi={source_czi!r}; pass pixel_size_um explicitly"
        ) from exc


def _czi_x_pixel_size_um(czi_path: Path) -> float | None:
    value_m = _czi_axis_scale_m(CziFile(czi_path).meta, "X")
    return None if value_m is None else float(value_m * 1_000_000.0)


def _czi_axis_scale_m(root: Element, axis: str) -> float | None:
    for distance in root.iter():
        if _local_name(distance.tag) != "Distance" or distance.attrib.get("Id") != axis:
            continue
        for child in distance.iter():
            if _local_name(child.tag) == "Value" and child.text:
                return float(child.text)

    fallback_tag = f"Scaling{axis}"
    for element in root.iter():
        if _local_name(element.tag) == fallback_tag and element.text:
            return float(element.text)
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _write_frames(
    stream: BinaryIO,
    array: zarr.Array,
    *,
    frame_indices: tuple[int, ...],
    channels: tuple[int, ...],
    width: int,
    height: int,
    low: float,
    high: float,
    scale_bar_um: float | None,
    display_pixel_size_um: float | None,
) -> None:
    for z_index in frame_indices:
        plane = _read_plane(array, z_index, channels, spatial_stride=1)
        frame = _scale_uint8(plane, low=low, high=high)
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if scale_bar_um is not None:
            if display_pixel_size_um is None:
                raise RuntimeError("scale bar rendering requires display_pixel_size_um")
            resized = _draw_scale_bar(resized, scale_bar_um, display_pixel_size_um)
        stream.write(np.ascontiguousarray(resized).tobytes())


def _read_plane(array: zarr.Array, z_index: int, channels: tuple[int, ...], *, spatial_stride: int) -> np.ndarray:
    data = np.asarray(array[z_index, list(channels), ::spatial_stride, ::spatial_stride], dtype=np.float32)
    if data.shape[0] == 1:
        return data[0]
    return np.max(data, axis=0)


def _scale_uint8(image: np.ndarray, *, low: float, high: float) -> np.ndarray:
    scaled = (np.asarray(image, dtype=np.float32) - np.float32(low)) / np.float32(high - low)
    return np.clip(scaled * np.float32(255.0), 0, 255).astype(np.uint8)


def _scale_bar_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, 22)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_scale_bar(frame: np.ndarray, length_um: float, display_pixel_size_um: float) -> np.ndarray:
    bar_width = int(round(length_um / display_pixel_size_um))
    if bar_width <= 0:
        raise ValueError(f"Scale bar is too short for displayed pixel size: {length_um:g} um")
    if bar_width > frame.shape[1] - 40:
        raise ValueError(f"Scale bar is too long for frame width: {length_um:g} um")

    rgb = np.repeat(frame[:, :, np.newaxis], 3, axis=2) if frame.ndim == 2 else frame
    image = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _scale_bar_font()

    margin = 28
    bar_height = 3
    text_gap = 8
    alpha = 204
    x1 = frame.shape[1] - margin
    x0 = x1 - bar_width
    label = f"{length_um:g} \u00b5m"
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    y1 = frame.shape[0] - margin - text_height - text_gap
    y0 = y1 - bar_height
    text_x = x0 + (bar_width - text_width) // 2
    text_y = y1 + text_gap

    shadow_offset = 2
    draw.rectangle(
        (x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset),
        fill=(0, 0, 0, alpha),
    )
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, alpha))
    draw.text((text_x + 1, text_y + 1), label, fill=(0, 0, 0, alpha), font=font)
    draw.text((text_x, text_y), label, fill=(255, 255, 255, alpha), font=font)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _start_ffmpeg(
    output_video: Path,
    *,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
    input_pix_fmt: str,
) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        input_pix_fmt,
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)
