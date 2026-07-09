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

Python 3.9+ is recommended. Create a dedicated environment named `vswitch_analysis`,
activate it, then install the dependencies into it:

```bash
# Create the environment (conda)
conda create -n vswitch_analysis python=3.10

# Activate it — do this BEFORE installing so packages land in the env
conda activate vswitch_analysis

# Install dependencies
pip install -r requirements.txt
```

You must activate the environment (`conda activate vswitch_analysis`) in every new
shell before running any script — including inside SLURM batch jobs, which do not
inherit your interactive shell's setup.

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

# Submit as SLURM array (edit ZARR_PATH, CONDA_ENV, and the SLURM settings at
# the top of submit_1-segmentation_tracking.sh first). The launcher loads conda and activates the
# environment itself, so no manual activation is needed.
bash submit_1-segmentation_tracking.sh              # generate FOV list + submit
bash submit_1-segmentation_tracking.sh --dry-run    # generate list only, print it, don't submit
```

Rows, wells, and FOVs are auto-detected from the zarr store. `submit_1-segmentation_tracking.sh`
first runs `1-segmentation_tracking.py --list-fovs`, which walks the store and
writes `fov_list.txt` (one `row well fov` per line); the SLURM array then indexes
into that file. Pass `--rows`/`--wells` to `--list-fovs` to restrict to a subset.
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

Channel indices are read from the store by script 1 only, via the
`PHASE_CHANNEL_IDX` / `DAPI_CHANNEL_IDX` / `GFP_CHANNEL_IDX` constants at the top
of the file. Scripts 2 and 3 operate on script 1's per-FOV outputs (tracking
CSVs and saved MIP arrays), so they do not take channel arguments.

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
