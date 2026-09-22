from __future__ import annotations

from pathlib import Path

from registration_fusion.deconvolution_preview import PreviewConfig, write_deconvolved_max_projection_png


def main() -> None:
    write_deconvolved_max_projection_png(
        Path("synthetic_dual_view_500x1920x1920x3_50000spots_22610"),
        output_png=Path("/tmp/deconvolved_profile_max_projection.png"),
        config=PreviewConfig(chunk_depth=64, halo_z=20, max_chunks=1),
    )


if __name__ == "__main__":
    main()
