reg_deconProject
=============

A code collection of the registration and deconvolution project corresponding to the paper: 

 Min Guo, *et al*. "[Rapid image deconvolution and multiview fusion for optical microscopy](https://doi.org/10.1038/s41587-020-0560-x)." Nature Biotechnology 38.11 (2020): 1337-1346.

This repository covers most functions and implementations reported in the paper. All codes, except the deep learning module `DenseDeconNet`, now run in MATLAB. Users may refer to the [code description](./Code_description.pdf) for more details and use the repository under the [license agreement](./LICENSE.pdf).  

The dependency C++/CUDA library for the codes has been compiled and attached in the release package. User can also find the source code of the C++/CUDA library at repository [**microImageLib**](https://github.com/eguomin/microImageLib).
In addition, the diSPIM data processing programs have been split to another repository [**diSPIMFusion**](https://github.com/eguomin/diSPIMFusion) and will be maintained independently.

Python GPU workflow
===================

The migrated Python path lives in `registration_fusion/` and is intended to be run from the
`seq` conda environment. It processes one dual-view timepoint at a time. TIFF inputs may be
plain `ZYX` stacks or page-major multi-channel stacks stored as `(Z*C, Y, X)`; pass
`--channels` for the latter.

1. Generate a deterministic dual-view test dataset, if needed:

```bash
conda run -n seq python -m registration_fusion.simulate_dual_view \
  --output-dir synthetic_dual_view_500x1920x1920x3_50000spots_22610 \
  --spots-per-channel 50000 \
  --view-angle-deg 90 \
  --compression 22610
```

This writes `ground_truth.tif`, objective-native `view_a.tif` and `view_b.tif`,
the reoriented object-grid `view_b_object_grid.tif`, matching max-projection PNG previews,
cropped view PSFs, `spots.csv`, and `simulation.json`. The default PSF is `/working/fishtools/data/PSF GL.tif`, cropped
with the same defaults used by fishtools deconvolution (`step=6`, `max_z=7`, `size=31`,
`center=50`). The simulator currently models the orthogonal 90-degree case. View B is
generated from the objective perspective by transposing object Z/Y into the view-B native
frame, convolving with the same native objective PSF as view A, then writing a separate
object-grid view for registration/deconvolution.

Synthetic stitching tests reuse this simulator and slice `view_a.tif` into overlapping
3D tiles. The Python stitching core in `registration_fusion/stitching.py` currently covers
regular or connected irregular in-memory ZCYX tile grids: coarse log phase-correlation on
expected adjacent overlap slabs, GPU downsampling of large coarse-correlation inputs to at
most 1,024 voxels per axis by default, Fourier-wrap candidate selection by overlap NCC,
optional fine translation refinement, least-squares global tile placement, and separable
linear weight blending. The stitching path requires CUDA through CuPy for phase correlation,
NCC scoring, fine translation refinement through the existing GPU registration path, and weighted fusion.
It applies the shifts and weights from a selected reference channel to every channel. For
CZI mosaics with low-information edge tiles, pass nominal tile origins from the CZI tile
bounding boxes and a minimum NCC threshold; low-NCC pair shifts fall back to those stage
coordinates while high-NCC pairs still use image-based CUDA registration. File-backed CLI
paths are provided below for CZI mosaics and extracted TIFF tile manifests.

The CZI chunked writer streams raw mosaic tiles directly to a stitched Zarr:

```bash
conda run -n seq python -m registration_fusion.cli inspect-czi-stitch \
  /path/to/acquisition.czi \
  --metadata-output results/acquisition_stitch_plan.json \
  --channel 0 \
  --channels 2

conda run -n seq python -m registration_fusion.cli estimate-czi-placement \
  /path/to/acquisition.czi \
  --output results/acquisition_placement.json \
  --metadata-output results/acquisition_placement_summary.json \
  --channel 0 \
  --reference-z-start 244 \
  --reference-z-count 8

conda run -n seq python -m registration_fusion.cli stitch-czi \
  /path/to/acquisition.czi \
  --output results/acquisition_stitched.zarr \
  --metadata-output results/acquisition_stitched.json \
  --placement-input results/acquisition_placement.json \
  --channel 0 \
  --apply-channel 0 \
  --apply-channel 1 \
  --z-start 0 \
  --chunk-depth 2 \
  --reference-z-start 244 \
  --reference-z-count 8
```

Use `--max-chunks 1` for a smoke run (`--max-chunks` must be positive), or add `--benchmark-only` to measure the same
read/fuse path without writing a Zarr. The metadata JSON records pair NCCs, fallback counts,
solved origins, output channel geometry validation, output shape/bytes, measured read/fuse/write timings, and a projected
full-run time when only part of the volume is processed. Benchmark-only runs with
`--metadata-output` also write a `.progress.jsonl` sidecar unless `--progress-output` is set.
Reuse a previously estimated
placement with `--placement-input results/acquisition_placement.json` to stitch additional
channels or rerun the full file without recomputing pair shifts. Full writes record completed
Z chunks in `results/acquisition_stitched.progress.jsonl` by default; pass `--resume` to skip
completed chunks after an interrupted run, or `--progress-output` to choose a different log.
Resume requires that progress log; it will not infer completed chunks from the Zarr alone.
Pass `--preview-output results/acquisition_preview.png` with smoke or benchmark runs to write
a downsampled max-projection QC image of the processed chunks before committing to a full
write. Existing Zarr outputs are not replaced unless `--overwrite` is passed. CZI placement
downsamples coarse phase-correlation inputs with `--coarse-max-size 1024` and uses subpixel
fine refinement by default (`--fine-upsample-factor 10`); set the latter to `1` for
coarse-only debug runs, or raise it to give the fine optimizer more iterations. Written Zarr arrays store self-describing attributes with axes,
source CZI, source Z range, output channel mapping, output origin, overlap, and placement
summary; resume runs validate those attributes before appending chunks. Requested output channels are checked
against the CZI channel dimension, must be unique, and must match the placement channel's
mosaic geometry before any chunk processing starts.
Fresh full writes also check target filesystem free space against the estimated uncompressed
output size before creating the Zarr.
For manual correction, export `results/acquisition_placement.json` with
`export-czi-placement-origins`, edit the CSV origins, and write a new placement JSON with
`apply-czi-placement-origins`. The chunked CZI writer requires zero Z origins; manual edits
should adjust Y/X tile origins only.

For extracted TIFF tiles, write a CSV manifest with columns `path,index_z,index_y,index_x`.
Paths may be relative to the manifest. Optional `origin_z,origin_y,origin_x` columns provide
nominal stage coordinates for `--fallback-min-ncc`. The stitched tile Zarr also records
axes, source manifest, tile indices, overlap, reference channel, and refinement settings in
its array attributes. Tile indices in the manifest must be unique. `inspect-tile-manifest`
reads TIFF series metadata to estimate output size without loading tile pixel data.

```bash
conda run -n seq python -m registration_fusion.cli inspect-tile-manifest tiles.csv \
  --metadata-output results/tiles_stitch_plan.json \
  --channels 2 \
  --overlap-z 4 \
  --overlap-y 290 \
  --overlap-x 288

conda run -n seq python -m registration_fusion.cli stitch-tiles tiles.csv \
  --output results/tiles_stitched.zarr \
  --metadata-output results/tiles_stitched.json \
  --channels 2 \
  --overlap-z 4 \
  --overlap-y 290 \
  --overlap-x 288
```

2. Register view B onto view A:

```bash
conda run -n seq python -m registration_fusion.cli register \
  --fixed synthetic_dual_view_500x1920x1920x3_50000spots_22610/view_a.tif \
  --moving synthetic_dual_view_500x1920x1920x3_50000spots_22610/view_b_object_grid.tif \
  --output results/view_b_registered.tif \
  --metadata-output results/registration.json \
  --png-dir results/previews \
  --channels 3 \
  --mode rigid \
  --compression 22610
```

Registration modes are `translation`, `rigid`, `similarity-7dof`, `scale-9dof`, and
`affine-12dof`. The migrated registration code uses CUDA through CuPy for phase-correlation
initialization, affine resampling, and metric evaluation. The metadata JSON records the
initial shift, output-to-input transform matrix, offset, final correlation, and max-projection
PNG paths. For `affine-12dof`, the GPU path mirrors the MATLAB `affMethod = 7` sequence:
translation, rigid, 9-DOF scale, then 12-DOF affine. The registration command writes
max-projection PNG previews for the fixed input, moving input, and registered output.

For large acquisitions, assume the inputs are root Zarr arrays shaped `(Z, C, Y, X)`.
Use the Zarr registration path instead of `register` or `dual-view`. It reads only a
downsampled channel-max reference for optimization and applies the final affine transform to
the full moving view in Z chunks:

```bash
conda run -n seq python -m registration_fusion.cli register-zarr \
  --fixed data/view_a.zarr \
  --moving data/view_b_object_grid.zarr \
  --output results/view_b_registered.zarr \
  --metadata-output results/registration_zarr.json \
  --png-output results/view_b_registered.png \
  --mode affine-12dof \
  --reference-downsample 16 \
  --chunk-depth 2
```

Tune `--reference-downsample` to keep the optimization reference small enough for GPU memory,
and tune `--chunk-depth` to keep full-resolution resampling below host and GPU memory limits.
At 2 TB scale, start conservatively with `--chunk-depth 1` or `2`; increasing it only improves
throughput if memory and Zarr chunk/cache buffers leave enough headroom. The output Zarr uses
the input Y/X chunking and one channel per chunk.

For synthetic registration tests, a deterministic 12-DOF perturbation can be applied directly
to a Zarr view:

```bash
conda run -n seq python -m registration_fusion.cli perturb-zarr \
  --input data/view_b_object_grid.zarr \
  --output results/view_b_object_grid_12dof_seed123.zarr \
  --metadata-output results/view_b_object_grid_12dof_seed123.json \
  --png-output results/view_b_object_grid_12dof_seed123.png \
  --seed 123 \
  --chunk-depth 2
```

3. Register and deconvolve in one pass:

```bash
conda run -n seq python -m registration_fusion.cli dual-view \
  --fixed synthetic_dual_view_500x1920x1920x3_50000spots_22610/view_a.tif \
  --moving synthetic_dual_view_500x1920x1920x3_50000spots_22610/view_b_object_grid.tif \
  --output results/deconvolved.tif \
  --registered-output results/view_b_registered.tif \
  --metadata-output results/dual_view_run.json \
  --png-dir results/previews \
  --channels 3 \
  --registration-mode rigid \
  --view-angle-deg 90 \
  --iterations 1 \
  --compression 22610
```

The deconvolution implements the dual-view alternating Richardson-Lucy update described in
Guo et al. with the Wiener-Butterworth unmatched back projector used in fishtools. The
default WB parameters are `alpha=0.02`, `beta=0.02`, `n=10`, and `sigma_g=1.7`.
For the orthogonal case, `--view-angle-deg 90` uses the exact Z/Y transpose of the
native objective PSF before building the second-view forward and back projectors.
The `dual-view` command writes max-projection PNG previews for view A, view B, and the
deconvolved output. Three-channel inputs are written as RGB max projections; other channel
counts are collapsed to a grayscale max projection.

For large registered datasets, use the Zarr chunked deconvolved preview command to validate
the registration/deconvolution behavior without materializing the full deconvolved volume:

```bash
conda run -n seq python -m registration_fusion.cli deconvolved-preview-zarr \
  --view-a data/view_a.zarr \
  --view-b results/view_b_registered.zarr \
  --psf-a synthetic_dual_view_500x1920x1920x3_50000spots_22610/psf_cropped_view_a.tif \
  --psf-b synthetic_dual_view_500x1920x1920x3_50000spots_22610/psf_cropped_view_b.tif \
  --output results/deconvolved_registered.png \
  --background 0 \
  --chunk-depth 32 \
  --halo-z 20 \
  --iterations 1
```

This command streams Z chunks with halo overlap and writes the max-projection PNG plus JSON
metrics. The full TIFF-producing `dual-view` command is still an in-memory path and should
not be used for terabyte-scale acquisitions.

The reflective diSPIM and reflective LLS MATLAB scripts are a different forward model:
`DeconReflectiveDiSPIM.m` uses `angle = 45`, and `DeconReflectiveLLS.m` uses `angle = 31`
inside illumination and sensitivity calculations. That is not just a PSF-orientation
parameter. Migrating those examples requires the spatially varying illumination/sensitivity
loops from the reflective MATLAB scripts, not only `--view-angle-deg`.

## Real CZI Benchmark

The current full-file benchmark was run with CUDA on
`/home/chaichontat/nvme/lightsheet/20260523-fullHCR/5x-zoom-561-638.czi` using the saved
placement `/home/chaichontat/nvme/lightsheet/20260523-fullHCR/5x-zoom-561-638_placement.json`
(`sha256=fb018d67a92a38c1ed3dbdc0cf820900b86d345e4e8d665e19793b0e85dddc0e`).

```bash
conda run -n seq python -m registration_fusion.cli stitch-czi \
  /home/chaichontat/nvme/lightsheet/20260523-fullHCR/5x-zoom-561-638.czi \
  --metadata-output /tmp/5x_zoom_561_638_coarse1024_fine10_benchmark_full_processing_v1.json \
  --placement-input /home/chaichontat/nvme/lightsheet/20260523-fullHCR/5x-zoom-561-638_placement.json \
  --channel 0 \
  --apply-channel 0 \
  --apply-channel 1 \
  --chunk-depth 2 \
  --benchmark-only
```

This benchmark processed all `274` two-plane chunks without writing the stitched Zarr. It
measured `670.84s` read time, `394.77s` fuse time, and `1260.61s` total processing-loop
time. The output shape would be `[548, 2, 10104, 6824]`, with an estimated materialized
size of `151,137,733,632` bytes. Placement used `coarse_max_size=1024` and
`fine_upsample_factor=10`, with `36` adjacent pairs, `24` nominal fallbacks, and median
pair NCC `0.5018`. A full materialized write has not been run because it creates the
151 GB Zarr output and needs explicit approval for the target path.

After approving that write, materialize the stitched Zarr with:

```bash
conda run -n seq python -m registration_fusion.cli stitch-czi \
  /home/chaichontat/nvme/lightsheet/20260523-fullHCR/5x-zoom-561-638.czi \
  --output /home/chaichontat/nvme/lightsheet/20260523-fullHCR/5x-zoom-561-638_stitched.zarr \
  --metadata-output /home/chaichontat/nvme/lightsheet/20260523-fullHCR/5x-zoom-561-638_stitched.json \
  --placement-input /home/chaichontat/nvme/lightsheet/20260523-fullHCR/5x-zoom-561-638_placement.json \
  --channel 0 \
  --apply-channel 0 \
  --apply-channel 1 \
  --chunk-depth 2
```

4. Verify the migration:

```bash
conda run -n seq pytest -q
conda run -n seq python -m compileall -q registration_fusion tests
conda run -n seq python -m pip wheel . --no-deps -w /tmp/regfusion_wheel_check
```

The tests require CUDA access and should not be skipped. If they fail with
`cudaErrorNoDevice`, rerun with permissions that expose the GPU.
The wheel check verifies that the installable package exposes the `registration-fusion`
console script.

Deconvolution validation is metric-based on deterministic synthetic spots. The GPU test
constructs a known sparse ground-truth volume, blurs it with complementary view-A and view-B
PSFs, runs dual-view WB deconvolution, and asserts that the result improves correlation to
ground truth, narrows the mean spot FWHM, and localizes every spot within one voxel.
