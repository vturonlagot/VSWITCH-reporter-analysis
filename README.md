# OFFON Reporter Analysis Pipeline

Image analysis and single-cell quantification pipeline for the split-mNG OFFON reporter system, used to measure viral protease activity at single-cell resolution by live fluorescence microscopy.

---

## Overview

The pipeline processes multi-dimensional fluorescence microscopy images (zarr format) through three sequential scripts:

| Script | Purpose |
|--------|---------|
| `1-OFFON_reporter_image_analysis_ultrack_vTEST.py` | Nuclear segmentation (Cellpose) and single-cell tracking (ultrack) |
| `2-OFFON_reporter_GFP_trajectories_analysis_ultrack_TEST.py` | Extraction and normalization of mNG and BFP fluorescence trajectories |
| `3-OFFON_reporter_activation_group_analysis_ultrack_vTEST_light.py` | Sigmoid fitting, activation detection, and response group classification |

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
python 1-OFFON_reporter_image_analysis_ultrack_vTEST.py \
    --zarr-path /path/to/data.zarr \
    --row <row> --well <well> --fov <fov> \
    --output-dir /path/to/output/

# Submit as SLURM array (edit paths inside submit_array.sh first)
bash submit_array.sh
```

### Script 2 — Trajectory extraction
```bash
python 2-OFFON_reporter_GFP_trajectories_analysis_ultrack_TEST.py \
    --input-dir /path/to/script1/output/ \
    --zarr-path /path/to/data.zarr \
    --output-dir /path/to/script2/output/
```

### Script 3 — Activation analysis and figures
```bash
python 3-OFFON_reporter_activation_group_analysis_ultrack_vTEST_light.py \
    --input-dir /path/to/script2/output/ \
    --output-dir /path/to/script3/output/
```

Run any script with `--help` for the full list of options.

---

## Input data format

Input images are expected as a zarr store with the layout:

```
store[row][well][fov]['0']  →  shape (T, C, Z, Y, X)
```

Channel order is specified via `--mng-channel` and `--bfp-channel` arguments (default: 0 and 1).

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
