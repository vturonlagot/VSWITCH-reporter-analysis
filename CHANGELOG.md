# Changelog

Changes made while validating the repository by re-running the analysis and
confirming it reproduces the published datasets.

## Environment and launchers

- **Standardized on a `vswitch_analysis` conda environment.** The setup is
  documented in the README (Requirements). The activation must happen in every
  new shell, including SLURM batch jobs, which do not inherit the interactive
  shell's setup.
- **`submit_1-segmentation_tracking.sh` is now self-contained.** It loads conda
  (`module load anaconda` + conda init + `conda activate vswitch_analysis`) in
  both the outer launcher and the sbatch heredoc, so no manual activation is
  needed. Previously the batch job silently failed with `python: command not
  found` because conda was never initialized in the non-interactive shell.
- **Single `ZARR_PATH` variable** in `submit_1-segmentation_tracking.sh`, passed
  via `--zarr-path` to the FOV-listing and segmentation steps. Previously the
  generator ran with no `--zarr-path` and fell back to a placeholder path,
  failing with a zarr `PathNotFoundError`.

## Merged `generate_fov_list.py` into `1-segmentation_tracking.py`

- Added a `--list-fovs` mode (with `--out`, `--rows`, `--wells`) that enumerates
  `(row, well, fov)` from the zarr store and writes the task list, then exits
  before any GPU/Cellpose init so it runs cheaply on the launch node.
- `submit_1-segmentation_tracking.sh` now calls
  `python 1-segmentation_tracking.py --list-fovs ...` instead of the separate
  generator, and `generate_fov_list.py` was deleted (its helpers were already
  duplicated in script 1).

## requirements.txt — corrected cellpose pin

- Changed the cellpose pin from `cellpose>=3.0,<4.0` to `cellpose==4.0.7`.
  The paper analysis ran with **cellpose 4.0.7** (Cellpose-SAM).
- Root cause of an observed 0-nuclei bug: with identical script parameters
  (`model_type='nuclei'`, `DIAMETER=100`, `CELLPOSE_THRESHOLD=1.0`), cellpose
  3.1.1.3 detected 0 nuclei while 4.0.7 works. Cellpose 4.x (Cellpose-SAM)
  ignores `model_type` and handles the ~25px nuclei at `diameter=100`; the 3.x
  `nuclei` model rescales to its ~17px training diameter, shrinking the nuclei
  away at `diameter=100`. The parameters were never the problem — only the
  cellpose version.

## Activation calling — control-well fixed threshold

- Activation is defined on the `mng_bfp_ratio` (mNG normalized by BFP) using a
  FIXED threshold derived from an uninfected control well
  (`control_mean + n_sd × SD`), with a **per-row** control well (B* wells
  calibrate from B1, C* wells from C1). `submit_2-trajectory_extraction.sh`
  passes `--n-sd 3` and a `--control-map`.
- The **per-cell adaptive threshold** (each track's own
  `baseline_mean + n_sd × SD` over its first frames) was removed from
  `2-trajectory_extraction.py`. In low-signal control wells that bar sits near
  each cell's own noise floor and over-calls activation (control wells reported
  ~10× more activating tracks than the paper). The paper only ever used the
  control-well fixed threshold; the adaptive path was a footgun.
- Verified reproduction (n_sd 3, per-row control well):
  - B1: threshold 0.6044, 4.0% activating (paper 0.6134, ~3.7%)
  - C1: threshold 0.5681, 2.83% activating (paper 0.5689, ~3%)
  The small B1 offset is the known ~1% segmentation jitter shifting the control
  distribution; C1 matches to <0.2%.

## Matched the published time axis

- `2-trajectory_extraction.py` default `end_timepoint` changed 96 → 48 (the
  paper analysed the first 48 frames).
- Imaging started 1.5 h post-infection, so `3-activation_analysis.py` gained
  `--imaging-start-hpi` (default 1.5). The offset is added to the `Timepoint`
  column of the two time-series exports — activation overview panel B
  (cumulative % activated) and panel F (mean ± SD trajectories) — *after* the
  raw-frame-index filtering, so filtering is unaffected and only the reported
  axis shifts. The offset is deliberately NOT applied to panel C (activation
  timepoint by response group), whose published medians match the re-run at raw
  frame indices.
