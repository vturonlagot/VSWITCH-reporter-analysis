# OFFON Reporter Analysis Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19463387.svg)](https://doi.org/10.5281/zenodo.19463387)

Image analysis and single-cell quantification pipeline for the split-mNG OFFON reporter system, used to measure viral protease activity at single-cell resolution by live fluorescence microscopy.

---

## Overview

The pipeline processes multi-dimensional fluorescence microscopy images (zarr format) through three sequential scripts:

| Script | Purpose |
|--------|---------|
| `1-segmentation_tracking.py` | Nuclear segmentation (Cellpose) and single-cell tracking (ultrack) |
| `2-trajectory_extraction.py` | Extraction and normalization of mNG and BFP fluorescence trajectories |
| `3-activation_analysis.py` | Sigmoid fitting, activation detection, and response group classification |

The reporter uses a split-mNG system in which reconstitution of mNG fluorescence reports on viral protease activity. The mNG/BFP ratio is used to normalize for cell-to-cell variation in reporter expression.

---

## Requirements

Python 3.9+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Scripts 1 runs on GPU (tested on NVIDIA A100 via SLURM). Scripts 2 and 3 run on CPU and can be executed locally or on HPC.

---

## Usage

### Script 1 — Segmentation and tracking
Processes a single field of view (FOV). Designed to be run as a SLURM array job.

```bash
# Single FOV
python 1-segmentation_tracking.py \
    --zarr-path /path/to/data.zarr \
    --row <row> --well <well> --fov <fov> \
    --output-dir /path/to/script1/output/

# Submit as SLURM array (edit the paths and SLURM settings inside
# submit_array.sh, and ZARR_PATH/OUTPUT_DIR inside 1-segmentation_tracking.py, first)
bash submit_array.sh
```

Rows, wells, and FOVs are auto-detected from the zarr store. `generate_fov_list.py`
walks the store automatically; pass `--rows`/`--wells` to restrict to a subset.
The local batch path in script 1 auto-detects too (set the `ROWS`/`WELLS`
constants to a list only if you want to process a subset).

Channel indices for script 1 are set with the `PHASE_CHANNEL_IDX`,
`DAPI_CHANNEL_IDX`, and `GFP_CHANNEL_IDX` constants at the top of the file.

### Script 2 — Trajectory extraction
Reads the per-FOV tracking output from script 1.

```bash
python 2-trajectory_extraction.py \
    --input /path/to/script1/output/ \
    --output /path/to/script2/output/ \
    --well all          # 'all' auto-detects wells from script 1's output; or list them: B1 B2 C1
```

### Script 3 — Activation analysis and figures
Reads the trajectory output from script 2 (`--analysis-dir`) and the
tracking output from script 1 (`--tracking-dir`).

```bash
python 3-activation_analysis.py \
    --analysis-dir /path/to/script2/output/ \
    --tracking-dir /path/to/script1/output/ \
    --output-dir /path/to/script3/output/ \
    --well all          # 'all' auto-detects wells; or list them: B1 B2 C1
```

Paths also have defaults defined in each script's `DEFAULTS` block / config
constants, so the flags above are optional once those are edited. Run any
script with `--help` for the full list of options.

---

## Input data format

Input images are expected as a zarr store with the layout:

```
store[row][well][fov]['0']  →  shape (T, C, Z, Y, X)
```

Channel indices are configured per script: script 1 uses the
`PHASE_CHANNEL_IDX` / `DAPI_CHANNEL_IDX` / `GFP_CHANNEL_IDX` constants at the top
of the file, and script 3 accepts `--nucleus-channel`, `--mng-channel`, and
`--bfp-channel` flags (run `python 3-activation_analysis.py --help` for defaults).

---

## Output

- **Script 1:** Per-FOV tracking CSVs (`ultrack_tracks.csv`) and segmentation masks
- **Script 2:** Per-well trajectory CSVs with normalized mNG/BFP ratios
- **Script 3:** Per-well activation summary CSVs and publication-quality figures (PNG/PDF/SVG)

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Vincent Turon-Lagot - https://orcid.org/0000-0003-2983-0684

Arias Lab, Biohub SF
