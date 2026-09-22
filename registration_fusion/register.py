from __future__ import annotations

import argparse
from pathlib import Path

from .core import VALID_MODES, register_stack_pair
from .io import read_tiff_stack, write_tiff_stack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register one StackB volume onto one StackA volume.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing StackA/StackB TIFFs")
    parser.add_argument("--fixed", type=Path, help="Fixed StackA TIFF path")
    parser.add_argument("--moving", type=Path, help="Moving StackB TIFF path")
    parser.add_argument("--output", type=Path, help="Registered StackB output TIFF path")
    parser.add_argument("--mode", default="rigid", choices=sorted(VALID_MODES))
    parser.add_argument("--ftol", default=1e-4, type=float)
    parser.add_argument("--max-iterations", default=200, type=int)
    parser.add_argument("--sampling-percentage", default=0.2, type=float)
    parser.add_argument("--bit-depth", default=16, choices=[16, 32], type=int)
    args = parser.parse_args(argv)

    fixed_path, moving_path = _resolve_inputs(args.data_dir, args.fixed, args.moving)
    output_path = args.output or _default_output_path(args.data_dir, moving_path)
    fixed = read_tiff_stack(fixed_path)
    moving = read_tiff_stack(moving_path)
    result = register_stack_pair(
        fixed,
        moving,
        mode=args.mode,
        ftol=args.ftol,
        max_iterations=args.max_iterations,
        sampling_percentage=args.sampling_percentage,
    )
    write_tiff_stack(output_path, result.registered, bit_depth=args.bit_depth)
    dz, dy, dx = result.initial_shift_zyx
    print(
        f"wrote {output_path} "
        f"(initial shift z/y/x={dz:.1f}/{dy:.1f}/{dx:.1f}, "
        f"corr={result.final_metric_value:.4f})"
    )
    return 0


def _resolve_inputs(
    data_dir: Path | None,
    fixed: Path | None,
    moving: Path | None,
) -> tuple[Path, Path]:
    if fixed and moving:
        return fixed, moving
    if fixed or moving:
        raise ValueError("--fixed and --moving must be provided together")
    if data_dir is None:
        raise ValueError("Provide either --data-dir or both --fixed and --moving")

    candidates = [
        (data_dir / "StackA.tif", data_dir / "StackB.tif"),
        (data_dir / "StackA_0.tif", data_dir / "StackB_0.tif"),
    ]
    for fixed_path, moving_path in candidates:
        if fixed_path.exists() and moving_path.exists():
            return fixed_path, moving_path
    raise FileNotFoundError(
        f"Could not find StackA/StackB TIFFs in {data_dir}; expected "
        "StackA.tif + StackB.tif or StackA_0.tif + StackB_0.tif"
    )


def _default_output_path(data_dir: Path | None, moving_path: Path) -> Path:
    if data_dir is not None:
        return data_dir / "results" / "StackB_reg.tif"
    return moving_path.with_name(f"{moving_path.stem}_reg.tif")


if __name__ == "__main__":
    raise SystemExit(main())
