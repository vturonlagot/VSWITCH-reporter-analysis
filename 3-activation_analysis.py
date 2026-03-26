"""
OFFON Reporter Activation Group Analysis - Version 15
======================================================
Supports multiple well formats (B2, B3, C2, C3, etc.)
Adds statistical test for max mNG intensity between activation groups
Adds individual figure export for publication

USAGE
=====
Basic analysis for a single well:
    python script.py --well C2

Two-pass IQR comparison (default):
    python script.py --well C2

Single-pass with manual thresholds:
    python script.py --well C2 --skip-iqr-comparison --baseline-intensity-min 20 --baseline-intensity-max 60

COMMAND-LINE OPTIONS
====================
Input/Output:
    --analysis-dir PATH       Directory containing activation analysis results
    --tracking-dir PATH       Directory containing ultrack tracking data
    --output-dir PATH         Output directory for results
    --well WELL               Well to analyze: B2, B3, C2, C3, or 'all' (default: C2)
    --exclude-fovs FOV [FOV ...]
                              Exclude specific FOVs from analysis

Timepoint Filtering:
    --timepoint-min N         Minimum timepoint to include (default: None = start from 0)
    --timepoint-max N         Maximum timepoint to include (default: None = use all)

Activation Group Classification (Percentile-based, default):
    --early-pct VALUE         Percentile threshold for early activators (default: 10)
    --average-pct-low VALUE   Lower percentile for average activators (default: 25)
    --average-pct-high VALUE  Upper percentile for average activators (default: 75)
    --late-pct VALUE          Percentile threshold for late activators (default: 90)

Activation Group Classification (Fixed thresholds, legacy):
    --early-min VALUE         Minimum timepoint for early activators (default: 6)
    --early-max VALUE         Maximum timepoint for early activators (default: 14)
    --average-min VALUE       Minimum timepoint for average activators (default: 16)
    --average-max VALUE       Maximum timepoint for average activators (default: 23)
    --late-min VALUE          Minimum timepoint for late activators (default: 27)

Baseline Analysis:
    --baseline-start N        Start timepoint for baseline calculation (default: 0)
    --baseline-end N          End timepoint for baseline calculation (default: 5)
    --threshold VALUE         mNG intensity threshold for activation (default: 40)
    --min-pre-activation-frames N
                              Minimum frames before activation (default: 2)

IQR Filtering:
    --skip-iqr-comparison     Skip two-pass IQR comparison, use single-pass mode
    --iqr-percentile-low VALUE
                              Lower percentile for IQR filtering (default: 25)
    --iqr-percentile-high VALUE
                              Upper percentile for IQR filtering (default: 75)
    --baseline-intensity-min VALUE
                              Manual minimum baseline intensity (for single-pass mode)
    --baseline-intensity-max VALUE
                              Manual maximum baseline intensity (for single-pass mode)

Normalization:
    --correction-method METHOD
                              Normalization method: baseline_bfp, mean_bfp, ratio, regression
                              (default: baseline_bfp)

Figure Output:
    --save-pdf                Save figures in PDF format (in addition to PNG)
    --save-svg                Save figures in SVG format (in addition to PNG)
    --save-individual         Save individual figures (not just panels)

Visualization (requires napari):
    --view-napari             Launch napari viewer
    --view-cell               View specific cell
    --view-grid               View grid of cells
    --fov N                   FOV to visualize (default: 0)
    --track-id N              Track ID to visualize
    --group GROUP             Group to visualize: early, average, late
    --n-examples N            Number of example cells (default: 5)
    --n-per-group N           Number of cells per group for grid (default: 3)
    --crop-size N             Crop size for cell view (default: 200)
    --grid-crop-size N        Crop size for grid view (default: 150)

EXAMPLES
========
# Basic two-pass IQR comparison analysis (default)
python script.py --well C2

# Save individual figures for publication (PNG + SVG)
python script.py --well C2 --save-individual --save-svg

# All formats: panels + individual figures in PNG, PDF, and SVG
python script.py --well C2 --save-individual --save-svg --save-pdf

# Custom IQR percentile range
python script.py --well C2 --iqr-percentile-low 20 --iqr-percentile-high 80

# Single-pass mode with manual baseline filtering
python script.py --well C2 --skip-iqr-comparison --baseline-intensity-min 25 --baseline-intensity-max 55

# Single-pass mode without any baseline filtering
python script.py --well C2 --skip-iqr-comparison

# Custom percentile-based group classification
python script.py --well C2 --early-pct 15 --late-pct 85

# Exclude problematic FOVs
python script.py --well C2 --exclude-fovs 3 5 7

OUTPUT FILES
============
Figures - Panels (in output_dir/figures/):
    - well_XX_activation_overview*.png        6-panel activation overview
    - well_XX_trajectories_panel*.png         6-panel trajectory analysis
    - well_XX_baseline_analysis_panel*.png    6-panel baseline correlation analysis
    - well_XX_max_gfp_distribution_*.png      Max mNG distribution by group
    - well_XX_max_gfp_statistical_*.png       Statistical comparison of max mNG
    - well_XX_activation_timing_*.png         Activation timing distribution
    - well_XX_bfp_analysis_panel*.png         BFP vs mNG correlation analysis
    - well_XX_bfp_stability*.png              BFP signal stability verification
    - well_XX_bfp_activation_aligned*.png     BFP dynamics aligned to activation
    - well_XX_reconstitution_dynamics*.png    Split-mNG reconstitution analysis
    - well_XX_IQR_comparison.png              Comparison of unfiltered vs IQR-filtered

Figures - Individual (in output_dir/figures/individual/, requires --save-individual):
    - well_XX_activation_histogram*.png       Activation time distribution
    - well_XX_cumulative_activation*.png      Cumulative activation curve
    - well_XX_example_trajectories*.png       Example mNG trajectories
    - well_XX_population_dynamics*.png        Population mean ± SD over time
    - well_XX_group_trajectories*.png         Group trajectories overlaid
    - well_XX_baseline_correlation*.png       Baseline vs activation correlation
    - well_XX_baseline_boxplot*.png           Baseline intensity by group
    - well_XX_max_gfp_boxplot*.png            Max mNG intensity by group

Data files (in output_dir/):
    - well_XX_activation_groups*.csv          Classified activation data
    - well_XX_measurements_final*.csv         Final measurements with all metrics
    - well_XX_napari_lookup*.csv              Lookup table for napari visualization
    - well_XX_max_gfp_statistics*.csv         Statistical test results for max mNG

Note: Files with *_unfiltered and *_iqr_filtered_QXX-QXX suffixes are generated
in two-pass IQR comparison mode (default).

ANALYSIS MODES
==============
1. Two-pass IQR comparison (default):
   - First pass: Analyze all cells without filtering
   - Calculate IQR thresholds from baseline intensity distribution
   - Second pass: Analyze cells within IQR range
   - Generate comparison figure showing impact of filtering

2. Single-pass mode (--skip-iqr-comparison):
   - Optionally apply manual baseline intensity thresholds
   - Run analysis once with specified parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import norm
from skimage.measure import regionprops_table
from sklearn.mixture import GaussianMixture
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')
try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
# ==================== PUBLICATION STYLE CONFIGURATION ====================
def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.title_fontsize': 10,
        'lines.linewidth': 1.5, 'lines.markersize': 6,
        'axes.linewidth': 1.0, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1, 'savefig.transparent': False, 'savefig.facecolor': 'white',
        'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
    })

COLORS = {'early': '#2166AC', 'average': '#4DAF4A', 'late': '#D62728', 'gray': '#7F7F7F', 'light_gray': '#D3D3D3'}
RESPONSE_COLORS = {'low': '#6BAED6', 'medium': '#FD8D3C', 'high': '#BD0026', 'unfit': '#BDBDBD'}


def _pairwise_sig_brackets(ax, data_list, positions=None, labels=None, bonferroni=True,
                            min_n=3, fontsize=9, lw=1.2):
    """
    Draw pairwise Mann-Whitney U significance brackets + stats table on a boxplot axes.

    All pairs are shown: significant ones in black (*, **, ***) and
    non-significant ones in gray (ns).  A compact stats table is drawn in the
    bottom-left corner showing the Bonferroni-corrected p-value for each pair.

    Parameters
    ----------
    ax : matplotlib Axes
    data_list : list of array-like  — one entry per box, matching x positions.
    positions : list of float, optional  — x-positions (default: 1, 2, ..., n).
    labels : list of str, optional  — group names used in the table (default: positional).
    bonferroni : bool  — apply Bonferroni correction for number of comparisons.
    min_n : int  — minimum group size to include in a comparison.
    """
    from itertools import combinations
    n = len(data_list)
    if positions is None:
        positions = list(range(1, n + 1))
    if labels is None:
        labels = [str(i + 1) for i in range(n)]

    pairs = [(i, j) for i, j in combinations(range(n), 2)
             if len(data_list[i]) >= min_n and len(data_list[j]) >= min_n]
    if not pairs:
        return

    p_raw = []
    for i, j in pairs:
        try:
            _, p = stats.mannwhitneyu(data_list[i], data_list[j], alternative='two-sided')
        except Exception:
            p = 1.0
        p_raw.append(p)

    n_comp = len(pairs)
    p_corr = [min(p * n_comp, 1.0) for p in p_raw] if (bonferroni and n_comp > 1) else p_raw

    def _stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'

    results = [(pairs[k], _stars(p_c), p_c, p_raw[k])
               for k, p_c in enumerate(p_corr)]

    # Sort brackets: narrower pairs drawn at lower height first
    results_sorted = sorted(results, key=lambda x: x[0][1] - x[0][0])

    # Base y from data max (robust to scatter points)
    all_vals = np.concatenate([np.asarray(d).ravel() for d in data_list if len(d) > 0])
    y_data_max = float(np.nanmax(all_vals)) if len(all_vals) > 0 else 1.0
    y_data_min = float(np.nanmin(all_vals)) if len(all_vals) > 0 else 0.0
    y_range = max(y_data_max - y_data_min, 1e-9)
    bar_y = y_data_max + y_range * 0.06
    step  = y_range * 0.13
    tick  = y_range * 0.03

    for (i, j), stars, p_c, p_r in results_sorted:
        x1, x2 = positions[i], positions[j]
        is_sig = stars != 'ns'
        color  = 'k' if is_sig else '#888888'
        lw_use = lw if is_sig else lw * 0.7
        ax.plot([x1, x1, x2, x2],
                [bar_y, bar_y + tick, bar_y + tick, bar_y],
                lw=lw_use, c=color, clip_on=False)
        ax.text((x1 + x2) / 2, bar_y + tick * 1.1, stars,
                ha='center', va='bottom', fontsize=fontsize,
                fontweight='bold' if is_sig else 'normal',
                color=color)
        bar_y += step

    ax.set_ylim(y_data_min - y_range * 0.05, bar_y + step * 0.3)

    # ── Stats table (bottom-left corner, axes coordinates) ───────────────────
    header = f"{'Comparison':<18} {'p (Bonf.)':<10} {'sig.'}"
    rows   = [header, '─' * len(header)]
    for (i, j), stars, p_c, _ in sorted(results, key=lambda x: x[0]):
        label_i = labels[i] if i < len(labels) else str(i)
        label_j = labels[j] if j < len(labels) else str(j)
        cmp_str = f'{label_i} vs {label_j}'
        rows.append(f'{cmp_str:<18} {p_c:<10.3g} {stars}')
    table_text = '\n'.join(rows)
    ax.text(0.02, 0.02, table_text,
            transform=ax.transAxes,
            ha='left', va='bottom', fontsize=6.5,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#f5f5f5',
                      edgecolor='#bbbbbb', alpha=0.92))


DEFAULTS = {
    'analysis_dir': "/path/to/your/output/2-trajectories",
    'tracking_dir': "/path/to/your/output/1-nuclear_analysis",
    'output_dir': "/path/to/your/output/3-activation_analysis",
    'well': 'B2',
    'early_min': 6, 'early_max': 13,
    'average_min': 15, 'average_max': 27,
    'late_min': 39,
    'min_pre_activation_frames': 2,
    'baseline_intensity_min': None,  # Changed to None for auto IQR
    'baseline_intensity_max': None,  # Changed to None for auto IQR
    'timepoint_min': None,  # None = start from 0
    'timepoint_max': 48,  # None = use all timepoints
}

AVAILABLE_WELLS = ['B1', 'B2', 'B3', 'C1', 'C2', 'C3']


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze early/average/late activators (ultrack)')
    parser.add_argument('--analysis-dir', type=str, default=DEFAULTS['analysis_dir'])
    parser.add_argument('--tracking-dir', type=str, default=DEFAULTS['tracking_dir'])
    parser.add_argument('--output-dir', type=str, default=DEFAULTS['output_dir'])
    parser.add_argument('--well', type=str, nargs='+', default=AVAILABLE_WELLS,
                        help=f"Well(s) to analyze: {', '.join(AVAILABLE_WELLS)}, or 'all' (default: all wells)")
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        help='Pool wells by condition. Format: NAME:WELL1,WELL2 '
                             'e.g. --conditions DENV:B2,B3 ZIKV:C2,C3. '
                             'Runs a pooled analysis per condition in addition to per-well analysis.')
    parser.add_argument('--exclude-fovs', nargs='+', type=int, default=None)
    parser.add_argument('--early-min', type=float, default=DEFAULTS['early_min'])
    parser.add_argument('--early-max', type=float, default=DEFAULTS['early_max'])
    parser.add_argument('--average-min', type=float, default=DEFAULTS['average_min'])
    parser.add_argument('--average-max', type=float, default=DEFAULTS['average_max'])
    parser.add_argument('--late-min', type=float, default=DEFAULTS['late_min'])
    parser.add_argument('--early-pct', type=float, default=10,
                        help='Percentile threshold for early activators (default: 10, meaning earliest 10%%)')
    parser.add_argument('--average-pct-low', type=float, default=25,
                        help='Lower percentile for average activators (default: 25)')
    parser.add_argument('--average-pct-high', type=float, default=75,
                        help='Upper percentile for average activators (default: 75)')
    parser.add_argument('--late-pct', type=float, default=90,
                        help='Percentile threshold for late activators (default: 90, meaning latest 10%%)')
    parser.add_argument('--classification-method', type=str, default='sd',
                        choices=['percentile', 'gmm', 'fixed', 'sd'],
                        help='Activation group classification method (default: percentile). '
                             'sd: mean ± sd_multiplier*SD, unbiased for sigmoid/unimodal distributions')
    parser.add_argument('--sd-multiplier', type=float, default=1.0,
                        help='SD multiplier for sd classification method (default: 1.0). '
                             'e.g. 1.0 → early/late = beyond ±1 SD from mean activation time')
    parser.add_argument('--response-sd-multiplier', type=float, default=1.0,
                        help='SD multiplier for response amplitude grouping (default: 1.0)')
    parser.add_argument('--response-method', type=str, default='tertile',
                        choices=['sd', 'tertile'],
                        help='Method for low/medium/high grouping: '
                             'tertile = equal thirds by percentile (default), '
                             'sd = mean ± sd_multiplier * SD')

    parser.add_argument('--response-r2-min', type=float, default=0.7,
                        help='Minimum sigmoid R² to include a cell in response grouping (default: 0.7)')
    parser.add_argument('--gmm-max-components', type=int, default=5,
                        help='Max Gaussian components to test via BIC (default: 5)')
    parser.add_argument('--gmm-force-components', type=int, default=None,
                        help='Force specific number of GMM components (skip BIC)')
    parser.add_argument('--gmm-covariance-type', type=str, default='full',
                        choices=['full', 'tied', 'diag', 'spherical'],
                        help='GMM covariance type (default: full)')
    parser.add_argument('--min-pre-activation-frames', type=int, default=DEFAULTS['min_pre_activation_frames'])
    parser.add_argument('--view-napari', action='store_true')
    parser.add_argument('--view-death-napari', action='store_true',
                        help='Open Napari to inspect dying vs surviving cell examples '
                             '(requires --fov; use --zarr-path/--zarr-row/--zarr-well '
                             'to overlay raw images)')
    parser.add_argument('--annotate-death-napari', action='store_true',
                        help='Open interactive Napari annotator: click nuclei and label '
                             'them death/division/alive to build a ground-truth CSV. '
                             'Mode dropdown in right panel; u=undo  Ctrl-S=save. '
                             'Requires --fov.  Output goes to --annotation-csv.')
    parser.add_argument('--annotation-csv', type=str, default=None,
                        help='Output path for manual death annotations '
                             '(default: death_annotations_fov<N>.csv in the output dir)')
    parser.add_argument('--train-death-classifier', action='store_true',
                        help='Train a Random Forest death classifier from --annotation-csv '
                             'and save it to --classifier-model')
    parser.add_argument('--apply-death-classifier', action='store_true',
                        help='Detect cell death using the trained model at --classifier-model '
                             'instead of the heuristic compute_cell_death()')
    parser.add_argument('--classifier-model', type=str, default=None,
                        help='Path to the trained death classifier .pkl file '
                             '(default: <output_dir>/death_classifier.pkl)')
    parser.add_argument('--view-classifier-napari', action='store_true',
                        help='View classifier predictions for one FOV in Napari')
    parser.add_argument('--death-prob-threshold', type=float, default=0.5,
                        help='Probability threshold for the classifier to call a death '
                             '(default: 0.5; lower = more sensitive, higher = more specific)')
    parser.add_argument('--fov', type=int, default=0)
    parser.add_argument('--n-examples', type=int, default=5)
    parser.add_argument('--group', type=str, choices=['early', 'average', 'late'], default=None)
    parser.add_argument('--death-response-group', type=str, default=None,
                        choices=['low', 'medium', 'high'],
                        help='Restrict dying-cell sample to one response group')
    parser.add_argument('--zarr-path', type=str, default=None,
                        help='Path to the .zarr store for raw image overlay in Napari')
    parser.add_argument('--zarr-row', type=str, default=None,
                        help='Row key inside the zarr store (e.g. B)')
    parser.add_argument('--zarr-well', type=str, default=None,
                        help='Well key inside the zarr store (e.g. 03)')
    parser.add_argument('--nucleus-channel', type=int, default=0,
                        help='Channel index for the nucleus image in the zarr store (default: 0)')
    parser.add_argument('--bfp-channel', type=int, default=None,
                        help='Channel index for BFP in the zarr store (omit to skip)')
    parser.add_argument('--mng-channel', type=int, default=None,
                        help='Channel index for mNG in the zarr store (omit to skip)')
    parser.add_argument('--view-grid', action='store_true')
    parser.add_argument('--view-response-grid', action='store_true',
                        help='View napari grid of low/medium/high response group cells')
    parser.add_argument('--n-per-group', type=int, default=3)
    parser.add_argument('--grid-crop-size', type=int, default=150)
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--save-svg', action='store_true')
    parser.add_argument('--save-individual', action='store_true', help='Save individual figures (not just panels)')
    parser.add_argument('--baseline-start', type=int, default=0)
    parser.add_argument('--baseline-end', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=50)
    parser.add_argument('--baseline-intensity-min', type=float, default=DEFAULTS['baseline_intensity_min'])
    parser.add_argument('--baseline-intensity-max', type=float, default=DEFAULTS['baseline_intensity_max'])
    parser.add_argument('--timepoint-min', type=int, default=DEFAULTS['timepoint_min'],
                        help='Minimum timepoint to include in analysis (default: None = start from 0)')
    parser.add_argument('--timepoint-max', type=int, default=DEFAULTS['timepoint_max'],
                        help='Maximum timepoint to include in analysis (default: None = use all)')
    # NEW: Option to skip IQR comparison and use manual thresholds
    parser.add_argument('--skip-iqr-comparison', action='store_true',
                        help='Skip two-pass IQR comparison, use manual thresholds only')
    parser.add_argument('--iqr-percentile-low', type=float, default=25,
                        help='Lower percentile for IQR filtering (default: 25)')
    parser.add_argument('--iqr-percentile-high', type=float, default=75,
                        help='Upper percentile for IQR filtering (default: 75)')
    parser.add_argument('--spatial-n-neighbors', type=int, default=10,
                        help='Number of nearest neighbours for spatial weight matrix (default: 10)')
    parser.add_argument('--spatial-n-permutations', type=int, default=999,
                        help='Number of permutations for Moran\'s I significance test (default: 999)')
    parser.add_argument('--cluster-r2-min', type=float, default=0.7,
                        help='Minimum sigmoid R² to include a cell in trajectory clustering (default: 0.7)')
    parser.add_argument('--cluster-k-min', type=int, default=2,
                        help='Minimum K to test in K-means cluster sweep (default: 2)')
    parser.add_argument('--cluster-k-max', type=int, default=7,
                        help='Maximum K (exclusive) to test in K-means cluster sweep (default: 7)')
    parser.add_argument('--cluster-no-umap', action='store_true',
                        help='Skip UMAP and use PCA 2-D embedding for clustering (default: try UMAP first)')
    parser.add_argument('--umap-n-neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                        help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--cluster-max-zscore', type=float, default=4.0,
                        help='Max absolute z-score in any feature before a cell is '
                             'flagged as a feature-space outlier and excluded from '
                             'K-means (default: 4.0; set to 0 to disable)')
    parser.add_argument('--cluster-min-frac', type=float, default=0.05,
                        help='Minimum fraction of inlier cells that each K-means cluster '
                             'must contain; K solutions with smaller clusters are rejected '
                             '(default: 0.05; set to 0 to disable)')
    return parser.parse_args()


# ==================== WELL PARSING FUNCTIONS ====================
def parse_well(well_str):
    normalized = well_str.strip().upper()
    if len(normalized) >= 2 and normalized[0].isalpha() and normalized[1:].isdigit():
        return (normalized[0], int(normalized[1:]), normalized)
    if normalized.isdigit():
        return ('C', int(normalized), f"C{normalized}")
    # Non-standard name (e.g. condition name like "DENV") — return as-is
    return (well_str.strip(), 0, well_str.strip())


def get_well_dir_pattern(well_info):
    row, col, full_name = well_info
    return f"well_{full_name}_FOV*"


def get_available_wells(tracking_dir):
    tracking_path = Path(tracking_dir)
    wells = set()
    for d in tracking_path.glob("well_*_FOV*"):
        parts = d.name.split("_")
        if len(parts) >= 2:
            wells.add(parts[1])
    return sorted(wells)

def get_timepoint_range(timepoint_min, timepoint_max, data_max=50):
    """Helper to get consistent timepoint range values."""
    t_min = timepoint_min if timepoint_min is not None else 0
    t_max = timepoint_max if timepoint_max is not None else data_max
    return t_min, t_max

# ==================== UTILITY FUNCTIONS ====================

def save_figure(fig, output_dir, name, save_pdf=False, save_svg=False, subdir=None):
    if subdir:
        fig_dir = output_dir / "figures" / subdir
    else:
        fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.png")
    if save_pdf:
        fig.savefig(fig_dir / f"{name}.pdf", bbox_inches='tight', facecolor='white')
    if save_svg:
        fig.savefig(fig_dir / f"{name}.svg", bbox_inches='tight', facecolor='white')

# ==================== DATA LOADING FUNCTIONS ====================
def load_activation_data(analysis_dir, well, exclude_fovs=None):
    _, _, full_name = parse_well(well)
    
    filepath = analysis_dir / f"well_{full_name}_all_tracks.csv"
    if not filepath.exists():
        filepath = analysis_dir / f"well_{full_name}_all_tracks_activation.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Activation data not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    def extract_fov(track_id):
        parts = str(track_id).split('_')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                if len(parts) >= 3:
                    return int(parts[1])
        return 0
    
    df['fov'] = df['unique_track_id'].apply(extract_fov)
    
    if exclude_fovs:
        df = df[~df['fov'].isin(exclude_fovs)]
        print(f"Excluded FOVs {exclude_fovs}: {len(df)} tracks remaining")
    
    return df


def load_measurements(tracking_dir, well, exclude_fovs=None):
    _, _, full_name = parse_well(well)
    
    all_meas = []
    pattern = f"well_{full_name}_FOV*"
    fov_dirs = list(Path(tracking_dir).glob(pattern))
    
    print(f"Found {len(fov_dirs)} FOV directories for well {full_name}")
    
    for fov_dir in sorted(fov_dirs):
        fov = int(fov_dir.name.split("FOV")[-1])
        
        if exclude_fovs and fov in exclude_fovs:
            continue
        
        meas_file = fov_dir / "nuclear_measurements.csv"
        if meas_file.exists():
            df = pd.read_csv(meas_file)
            df['fov'] = fov
            df['unique_track_id'] = f"{full_name}_{fov}_" + df['track_id'].astype(str)
            all_meas.append(df)
    
    return pd.concat(all_meas, ignore_index=True) if all_meas else None


def load_condition_data(analysis_dir, wells, exclude_fovs=None):
    """Load and pool activation data from multiple wells into a single DataFrame."""
    dfs = []
    for well in wells:
        try:
            df = load_activation_data(analysis_dir, well, exclude_fovs)
            print(f"  Loaded {len(df)} tracks from well {well}")
            dfs.append(df)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    if not dfs:
        raise FileNotFoundError(f"No activation data found for wells: {wells}")
    return pd.concat(dfs, ignore_index=True)


def load_condition_measurements(tracking_dir, wells, exclude_fovs=None):
    """Load and pool measurements from multiple wells into a single DataFrame."""
    dfs = []
    for well in wells:
        df = load_measurements(tracking_dir, well, exclude_fovs)
        if df is not None:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

def filter_by_timepoint_range(df_meas, df_act, timepoint_min=None, timepoint_max=None):
    """Filter measurements and activation data to a specific timepoint range."""
    if timepoint_min is None and timepoint_max is None:
        return df_meas, df_act
    
    df_meas_filtered = df_meas.copy()
    
    if timepoint_min is not None:
        df_meas_filtered = df_meas_filtered[df_meas_filtered['timepoint'] >= timepoint_min]
        print(f"  Filtered to timepoints >= {timepoint_min}")
    
    if timepoint_max is not None:
        df_meas_filtered = df_meas_filtered[df_meas_filtered['timepoint'] <= timepoint_max]
        print(f"  Filtered to timepoints <= {timepoint_max}")
    
    # Also filter activation data - exclude cells that activate outside the range
    df_act_filtered = df_act.copy()
    if timepoint_min is not None:
        df_act_filtered = df_act_filtered[
            (df_act_filtered['activation_timepoint'] >= timepoint_min) | 
            (df_act_filtered['activation_timepoint'].isna()) |
            (df_act_filtered['activates'] == False)
        ]
    
    if timepoint_max is not None:
        df_act_filtered = df_act_filtered[
            (df_act_filtered['activation_timepoint'] <= timepoint_max) | 
            (df_act_filtered['activation_timepoint'].isna()) |
            (df_act_filtered['activates'] == False)
        ]
    
    # Only keep tracks that still have measurements
    valid_tracks = df_meas_filtered['unique_track_id'].unique()
    df_act_filtered = df_act_filtered[df_act_filtered['unique_track_id'].isin(valid_tracks)]
    
    n_meas_before = len(df_meas)
    n_meas_after = len(df_meas_filtered)
    n_act_before = len(df_act)
    n_act_after = len(df_act_filtered)
    
    pct_meas = f" ({n_meas_after/n_meas_before*100:.1f}% retained)" if n_meas_before > 0 else ""
    pct_act = f" ({n_act_after/n_act_before*100:.1f}% retained)" if n_act_before > 0 else ""
    print(f"  Measurements: {n_meas_before} -> {n_meas_after}{pct_meas}")
    print(f"  Tracks: {n_act_before} -> {n_act_after}{pct_act}")
    
    return df_meas_filtered, df_act_filtered


def load_division_events(tracking_dir, well, exclude_fovs=None):
    """
    Extract cell division events from ultrack_tracks.csv files.

    Ultrack records divisions via a parent column (tried in order):
      'parent_track_id', 'parent_id'
    Root tracks have value -1, 0, or NaN depending on ultrack version.

    Returns a DataFrame with one row per daughter track:
        fov                    : field of view
        unique_track_id        : daughter cell
        parent_unique_track_id : parent cell
        division_timepoint     : first timepoint of daughter (= the division event)
    Returns None if no usable lineage data is found.
    """
    _, _, full_name = parse_well(well)
    fov_dirs = sorted(Path(tracking_dir).glob(f"well_{full_name}_FOV*"))
    print(f"  Scanning {len(fov_dirs)} FOV director{'y' if len(fov_dirs)==1 else 'ies'} for division data")

    # ultrack uses different column names across versions
    PARENT_COL_CANDIDATES = ['parent_track_id', 'parent_id']
    # ultrack uses -1 (v1) or 0 (some builds) or NaN for root tracks
    ROOT_SENTINEL_VALUES  = {-1, 0}

    records = []

    for fov_dir in fov_dirs:
        fov = int(fov_dir.name.split("FOV")[-1])
        if exclude_fovs and fov in exclude_fovs:
            continue

        track_file = fov_dir / "ultrack_tracks.csv"
        if not track_file.exists():
            print(f"    FOV {fov}: ultrack_tracks.csv not found — skipping")
            continue

        ti = pd.read_csv(track_file)
        print(f"    FOV {fov}: {len(ti)} rows, columns = {list(ti.columns)}")

        # Normalise timepoint column
        if 't' in ti.columns:
            ti = ti.rename(columns={'t': 'timepoint'})

        # Normalise track_id column
        if 'track_id' not in ti.columns:
            if 'id' in ti.columns:
                ti = ti.rename(columns={'id': 'track_id'})
            else:
                print(f"    FOV {fov}: no track_id / id column — skipping")
                continue

        # Find parent column
        parent_col = next((c for c in PARENT_COL_CANDIDATES if c in ti.columns), None)
        if parent_col is None:
            print(f"    FOV {fov}: no parent column found ({PARENT_COL_CANDIDATES}) — "
                  f"division tracking may not have been enabled")
            continue

        # One row per track: first timepoint + parent id
        per_track = ti.groupby('track_id').agg(
            division_timepoint=('timepoint', 'min'),
            parent_raw=(parent_col, 'first')
        ).reset_index()

        n_unique_parents = per_track['parent_raw'].nunique()
        print(f"    FOV {fov}: {len(per_track)} tracks, "
              f"parent column '{parent_col}', "
              f"{n_unique_parents} unique parent values "
              f"(sample: {sorted(per_track['parent_raw'].dropna().unique()[:5].tolist())})")

        # Keep only true daughters (parent is not a root sentinel)
        daughters = per_track[
            per_track['parent_raw'].notna() &
            ~per_track['parent_raw'].isin(ROOT_SENTINEL_VALUES)
        ].copy()

        print(f"    FOV {fov}: {len(daughters)} daughter tracks identified")
        if len(daughters) == 0:
            continue

        daughters['fov'] = fov
        daughters['unique_track_id'] = (
            f"{full_name}_{fov}_" + daughters['track_id'].astype(int).astype(str)
        )
        daughters['parent_unique_track_id'] = (
            f"{full_name}_{fov}_" + daughters['parent_raw'].astype(int).astype(str)
        )
        records.append(
            daughters[['fov', 'unique_track_id', 'parent_unique_track_id', 'division_timepoint']]
        )

    if not records:
        print("  No division events found — check diagnostics above")
        return None

    df_div = pd.concat(records, ignore_index=True)
    print(f"  Total: {len(df_div)} division events across {df_div['fov'].nunique()} FOV(s)")
    return df_div


def load_script2_uninfected(analysis_dir, well, df_meas):
    """
    Load non-activating (uninfected) cells from script 2 output and compute their motility.

    Script 2 applies quality filters (min track duration, max position jump, gap fraction,
    area CV) before classifying cells as non-activating, making these a more rigorous
    uninfected reference group than script 3's own activates==False cells.

    Returns a DataFrame with motility columns (mean_speed, net_displacement,
    total_path_length, straightness) indexed by unique_track_id, or None if the
    script 2 output file is not found.
    """
    if 'centroid-0' not in df_meas.columns or 'centroid-1' not in df_meas.columns:
        print("  Skipping script 2 uninfected load: no centroid columns in df_meas")
        return None

    _, _, full_name = parse_well(well)
    path = Path(analysis_dir) / f"well_{full_name}_all_tracks.csv"
    if not path.exists():
        print(f"  Script 2 all_tracks not found at {path} — uninfected reference unavailable")
        return None

    df_s2 = pd.read_csv(path)
    if 'activates' not in df_s2.columns or 'unique_track_id' not in df_s2.columns:
        print("  Script 2 all_tracks missing required columns (activates, unique_track_id)")
        return None

    non_act_ids = set(df_s2.loc[~df_s2['activates'], 'unique_track_id'])
    print(f"  Script 2: {len(df_s2)} total tracks, {len(non_act_ids)} non-activating (uninfected)")

    pos = (df_meas.loc[df_meas['unique_track_id'].isin(non_act_ids),
                       ['unique_track_id', 'timepoint', 'centroid-0', 'centroid-1']]
           .sort_values(['unique_track_id', 'timepoint'])
           .copy())

    if len(pos) == 0:
        print("  No position data in df_meas matches script 2 uninfected track IDs")
        return None

    pos['dy']   = pos.groupby('unique_track_id')['centroid-0'].diff()
    pos['dx']   = pos.groupby('unique_track_id')['centroid-1'].diff()
    pos['step'] = np.sqrt(pos['dy'] ** 2 + pos['dx'] ** 2)

    agg = pos.groupby('unique_track_id')['step'].agg(
        mean_speed='mean',
        total_path_length='sum'
    ).reset_index()

    first_pos = pos.groupby('unique_track_id')[['centroid-0', 'centroid-1']].first()
    last_pos  = pos.groupby('unique_track_id')[['centroid-0', 'centroid-1']].last()
    net_disp  = np.sqrt(
        (last_pos['centroid-0'] - first_pos['centroid-0']) ** 2 +
        (last_pos['centroid-1'] - first_pos['centroid-1']) ** 2
    ).rename('net_displacement').reset_index()

    agg = agg.merge(net_disp, on='unique_track_id', how='left')
    agg['straightness'] = (agg['net_displacement'] / agg['total_path_length']).clip(0, 1)

    print(f"  Motility computed for {len(agg)} script 2 uninfected cells")
    return agg


def extract_bfp_measurements(tracking_dir, well, exclude_fovs=None):
    _, _, full_name = parse_well(well)
    
    all_meas = []
    pattern = f"well_{full_name}_FOV*"
    fov_dirs = list(Path(tracking_dir).glob(pattern))
    
    for fov_dir in sorted(fov_dirs):
        fov = int(fov_dir.name.split("FOV")[-1])
        
        if exclude_fovs and fov in exclude_fovs:
            continue
        
        masks_file = fov_dir / "tracked_masks.npy"
        bfp_file = fov_dir / "dapi_mips.npy"
        
        if not masks_file.exists() or not bfp_file.exists():
            print(f"  Skipping FOV {fov}: missing files")
            continue
        
        print(f"  Processing FOV {fov}...")
        
        tracked_masks = np.load(masks_file)
        bfp_mips = np.load(bfp_file)
        n_timepoints = tracked_masks.shape[0]
        
        for t in range(n_timepoints):
            mask = tracked_masks[t]
            bfp = bfp_mips[t]
            
            if mask.max() == 0:
                continue
            
            props = regionprops_table(
                mask, intensity_image=bfp,
                properties=['label', 'centroid', 'area', 'mean_intensity', 
                           'max_intensity', 'min_intensity']
            )
            
            df_t = pd.DataFrame(props)
            df_t = df_t.rename(columns={
                'label': 'track_id',
                'centroid-0': 'centroid_y',
                'centroid-1': 'centroid_x',
                'mean_intensity': 'bfp_mean_intensity',
                'max_intensity': 'bfp_max_intensity',
                'min_intensity': 'bfp_min_intensity',
                'area': 'area_pixels'
            })
            
            df_t['timepoint'] = t
            df_t['fov'] = fov
            df_t['unique_track_id'] = f"{full_name}_{fov}_" + df_t['track_id'].astype(str)
            all_meas.append(df_t)
    
    if not all_meas:
        return None
    
    df_bfp = pd.concat(all_meas, ignore_index=True)
    print(f"\nExtracted BFP measurements for {len(df_bfp)} observations")
    return df_bfp


# ==================== CLASSIFICATION FUNCTIONS ====================
def classify_activators(df, df_meas, early_min=None, early_max=None, average_min=None,
                        average_max=None, late_min=None, min_pre_activation_frames=2,
                        use_percentile=True, early_pct=10, average_pct_low=25,
                        average_pct_high=75, late_pct=90,
                        method='percentile',
                        gmm_max_components=5, gmm_force_components=None,
                        gmm_covariance_type='full',
                        sd_multiplier=1.0):
    """
    Classify activators into early/average/late groups.

    method='percentile' (default): percentile-based thresholds
    method='gmm': Gaussian Mixture Model (falls back to percentile if GMM fails)
    method='fixed': legacy fixed timepoint thresholds
    method='sd': mean ± sd_multiplier * SD (most unbiased for unimodal/sigmoid distributions)
    """

    # ---- GMM path ----
    if method == 'gmm':
        result = classify_activators_gmm(
            df, df_meas,
            min_pre_activation_frames=min_pre_activation_frames,
            max_components=gmm_max_components,
            force_components=gmm_force_components,
            covariance_type=gmm_covariance_type
        )
        if result is not None:
            return result
        # Fallback to percentile
        print("  GMM failed or returned 1 component — falling back to percentile method")
        method = 'percentile'

    # ---- SD path ----
    if method == 'sd':
        df_act = df[df['activates'] == True].copy()

        track_start_times = df_meas.groupby('unique_track_id')['timepoint'].min().to_dict()
        df_act['track_start_timepoint'] = df_act['unique_track_id'].map(track_start_times)
        df_act['pre_activation_frames'] = df_act['activation_timepoint'] - df_act['track_start_timepoint']
        df_act['sufficient_pre_tracking'] = df_act['pre_activation_frames'] >= min_pre_activation_frames

        activation_times = df_act['activation_timepoint'].dropna()
        mean_t = activation_times.mean()
        std_t = activation_times.std()
        early_threshold = mean_t - sd_multiplier * std_t
        late_threshold  = mean_t + sd_multiplier * std_t

        df_act['activation_group'] = 'average'
        df_act.loc[df_act['activation_timepoint'] <  early_threshold, 'activation_group'] = 'early'
        df_act.loc[df_act['activation_timepoint'] >  late_threshold,  'activation_group'] = 'late'

        print(f"\nSD-based activation groups (multiplier={sd_multiplier}):")
        print(f"  Mean activation time: {mean_t:.1f}, SD: {std_t:.1f}")
        print(f"  Early  (t < {early_threshold:.1f}): {(df_act['activation_group'] == 'early').sum()}")
        print(f"  Average ({early_threshold:.1f} ≤ t ≤ {late_threshold:.1f}): {(df_act['activation_group'] == 'average').sum()}")
        print(f"  Late   (t > {late_threshold:.1f}): {(df_act['activation_group'] == 'late').sum()}")

        df_act.attrs['classification_thresholds'] = {
            'method': 'sd',
            'mean': mean_t, 'std': std_t, 'sd_multiplier': sd_multiplier,
            'early_threshold': early_threshold, 'late_threshold': late_threshold,
        }
        df_act.attrs.pop('classification_info', None)
        return df_act

    # ---- Percentile / Fixed path (original code, unchanged) ----
    df_act = df[df['activates'] == True].copy()

    track_start_times = df_meas.groupby('unique_track_id')['timepoint'].min().to_dict()
    df_act['track_start_timepoint'] = df_act['unique_track_id'].map(track_start_times)
    df_act['pre_activation_frames'] = df_act['activation_timepoint'] - df_act['track_start_timepoint']
    df_act['sufficient_pre_tracking'] = df_act['pre_activation_frames'] >= min_pre_activation_frames

    df_act['activation_group'] = 'other'

    if method == 'percentile' or (method != 'fixed' and use_percentile):
        activation_times = df_act['activation_timepoint'].dropna()

        early_threshold = activation_times.quantile(early_pct / 100)
        average_low_threshold = activation_times.quantile(average_pct_low / 100)
        average_high_threshold = activation_times.quantile(average_pct_high / 100)
        late_threshold = activation_times.quantile(late_pct / 100)

        df_act.loc[df_act['activation_timepoint'] <= early_threshold, 'activation_group'] = 'early'
        df_act.loc[(df_act['activation_timepoint'] > average_low_threshold) &
                   (df_act['activation_timepoint'] <= average_high_threshold), 'activation_group'] = 'average'
        df_act.loc[df_act['activation_timepoint'] > late_threshold, 'activation_group'] = 'late'

        print(f"\nPercentile-based activation groups:")
        print(f"  Early (≤{early_pct}%, t≤{early_threshold:.1f}): {(df_act['activation_group'] == 'early').sum()}")
        print(f"  Average ({average_pct_low}-{average_pct_high}%, {average_low_threshold:.1f}<t≤{average_high_threshold:.1f}): {(df_act['activation_group'] == 'average').sum()}")
        print(f"  Late (>{late_pct}%, t>{late_threshold:.1f}): {(df_act['activation_group'] == 'late').sum()}")
        print(f"  Other: {(df_act['activation_group'] == 'other').sum()}")

        df_act.attrs['classification_thresholds'] = {
            'method': 'percentile',
            'early_pct': early_pct, 'early_threshold': early_threshold,
            'average_pct_low': average_pct_low, 'average_low_threshold': average_low_threshold,
            'average_pct_high': average_pct_high, 'average_high_threshold': average_high_threshold,
            'late_pct': late_pct, 'late_threshold': late_threshold
        }
        # Clear any previous GMM info so filtered run doesn't use stale attrs
        df_act.attrs.pop('classification_info', None)

    return df_act


def classify_activators_gmm(df, df_meas, min_pre_activation_frames=2,
                             max_components=5, force_components=None,
                             covariance_type='full', random_state=42):
    """
    Classify activators into early/average/late groups using Gaussian Mixture Models.

    Fits GMMs with 1..max_components and selects the best by BIC, unless
    force_components is set.  Clusters are mapped to groups by sorting
    component means: lowest mean → early, highest → late, middle → average.

    Returns df_act with 'activation_group' column, same interface as
    classify_activators().
    """
    df_act = df[df['activates'] == True].copy()

    # Pre-activation tracking (same as percentile method)
    track_start_times = df_meas.groupby('unique_track_id')['timepoint'].min().to_dict()
    df_act['track_start_timepoint'] = df_act['unique_track_id'].map(track_start_times)
    df_act['pre_activation_frames'] = df_act['activation_timepoint'] - df_act['track_start_timepoint']
    df_act['sufficient_pre_tracking'] = df_act['pre_activation_frames'] >= min_pre_activation_frames

    activation_times = df_act['activation_timepoint'].dropna().values
    n_cells = len(activation_times)

    if n_cells < 10:
        print("WARNING: Too few activating cells for GMM (<10). Falling back to percentile.")
        return None  # Caller handles fallback

    X = activation_times.reshape(-1, 1)

    # ---- Model selection via BIC ----
    bic_scores = []
    models = []

    if force_components is not None:
        n_range = [force_components]
        print(f"\nGMM: Forced {force_components} components (BIC selection skipped)")
    else:
        n_range = range(1, max_components + 1)

    for n in n_range:
        gmm = GaussianMixture(
            n_components=n, covariance_type=covariance_type,
            n_init=10, random_state=random_state, max_iter=300
        )
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        models.append(gmm)

    best_idx = int(np.argmin(bic_scores))
    best_gmm = models[best_idx]
    best_n = n_range[best_idx] if isinstance(n_range, range) else n_range[best_idx]

    if best_n == 1 and force_components is None:
        print("WARNING: BIC selected 1 component (unimodal). Falling back to percentile.")
        return None

    print(f"\nGMM Model Selection:")
    if force_components is None:
        for n, bic in zip(n_range, bic_scores):
            marker = " <-- best" if n == best_n else ""
            print(f"  {n} components: BIC = {bic:.1f}{marker}")
    print(f"  Selected: {best_n} components ({covariance_type} covariance)")

    # ---- Predict cluster assignments ----
    labels = best_gmm.predict(X)
    posteriors = best_gmm.predict_proba(X)

    # ---- Map clusters to early/average/late by sorting means ----
    means = best_gmm.means_.flatten()
    sorted_indices = np.argsort(means)

    group_mapping = {}
    if best_n == 2:
        group_mapping[sorted_indices[0]] = 'early'
        group_mapping[sorted_indices[1]] = 'late'
    elif best_n >= 3:
        group_mapping[sorted_indices[0]] = 'early'
        group_mapping[sorted_indices[-1]] = 'late'
        for idx in sorted_indices[1:-1]:
            group_mapping[idx] = 'average'
    else:
        # Shouldn't reach here (n=1 handled above), but just in case
        group_mapping[0] = 'average'

    # Build label array aligned with df_act rows that have valid activation times
    valid_mask = df_act['activation_timepoint'].notna()
    df_act['activation_group'] = 'other'
    df_act.loc[valid_mask, 'activation_group'] = [group_mapping.get(l, 'other') for l in labels]

    # Store posterior probability of assigned cluster
    max_posteriors = posteriors.max(axis=1)
    df_act.loc[valid_mask, 'gmm_posterior'] = max_posteriors

    # ---- Extract GMM parameters ----
    stds = np.sqrt(best_gmm.covariances_.flatten()) if covariance_type in ('spherical', 'diag') \
        else np.sqrt(np.array([best_gmm.covariances_[i][0, 0] for i in range(best_n)]))
    weights = best_gmm.weights_

    # ---- Print summary ----
    print(f"\nGMM Cluster Summary:")
    print(f"  {'Cluster':<10} {'Group':<10} {'Mean':>8} {'Std':>8} {'Weight':>8} {'N cells':>8}")
    print(f"  {'-'*54}")
    for i in sorted_indices:
        group = group_mapping[i]
        n_in_cluster = (labels == i).sum()
        print(f"  {i:<10} {group:<10} {means[i]:8.1f} {stds[i]:8.1f} {weights[i]:8.3f} {n_in_cluster:>8}")

    groups = ['early', 'average', 'late']
    print(f"\nFinal group counts:")
    for g in groups:
        count = (df_act['activation_group'] == g).sum()
        if count > 0:
            print(f"  {g.capitalize()}: {count}")
    other_count = (df_act['activation_group'] == 'other').sum()
    if other_count > 0:
        print(f"  Other: {other_count}")

    mean_posterior = max_posteriors.mean()
    low_conf = (max_posteriors < 0.7).sum()
    print(f"\nClassification confidence:")
    print(f"  Mean posterior: {mean_posterior:.3f}")
    print(f"  Low confidence (<0.7): {low_conf} cells ({low_conf/n_cells*100:.1f}%)")

    # Store metadata for diagnostics
    df_act.attrs['classification_info'] = {
        'method': 'gmm',
        'n_components': best_n,
        'bic_scores': list(zip(list(n_range), bic_scores)),
        'means': means[sorted_indices].tolist(),
        'stds': stds[sorted_indices].tolist(),
        'weights': weights[sorted_indices].tolist(),
        'group_mapping': {int(k): v for k, v in group_mapping.items()},
        'covariance_type': covariance_type,
        'gmm_model': best_gmm,
        'posteriors': posteriors,
        'labels': labels,
        'sorted_indices': sorted_indices,
    }

    return df_act


def plot_gmm_diagnostics(df_act, df_meas, output_dir, well, save_pdf=False, save_svg=False,
                          save_individual=False, suffix="", timepoint_min=None, timepoint_max=None,
                          # Percentile args for comparison panel:
                          early_pct=10, average_pct_low=25, average_pct_high=75, late_pct=90):
    """
    Generate a 2x3 diagnostic figure for GMM-based classification.

    Panels:
      A - BIC curve (if model selection was performed)
      B - Histogram + GMM component densities
      C - Posterior probability of assigned cluster
      D - Cluster assignment vs baseline intensity
      E - Comparison with percentile classification (heatmap)
      F - Summary statistics text
    """
    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    info = df_act.attrs.get('classification_info', {})
    if info.get('method') != 'gmm':
        print("  No GMM info found, skipping diagnostics")
        return

    # Ensure posteriors/labels match current df_act (e.g. after IQR filter, attrs can be stale)
    valid_mask = df_act['activation_timepoint'].notna() & (df_act['activates'] == True)
    n_valid = valid_mask.sum()
    posteriors = info['posteriors']
    labels = info['labels']
    if len(posteriors) != n_valid or len(labels) != n_valid:
        print(f"  GMM diagnostics skipped: classification_info size ({len(posteriors)}) != current activating cells ({n_valid})")
        return

    gmm = info['gmm_model']
    bic_data = info['bic_scores']  # list of (n, bic)
    sorted_indices = info['sorted_indices']
    group_mapping = info['group_mapping']

    n_components = info['n_components']
    means = gmm.means_.flatten()
    covariance_type = info['covariance_type']
    if covariance_type in ('spherical', 'diag'):
        stds = np.sqrt(gmm.covariances_.flatten())
    else:
        stds = np.sqrt(np.array([gmm.covariances_[i][0, 0] for i in range(n_components)]))
    weights = gmm.weights_

    groups = ['early', 'average', 'late']
    df_groups = df_act[df_act['activation_group'].isin(groups)].copy()
    activation_times = df_act[df_act['activates'] == True]['activation_timepoint'].dropna().values

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'GMM Classification Diagnostics — Well {full_name}{suffix}',
                 fontsize=14, fontweight='bold', y=0.98)

    # ---- Panel A: BIC curve ----
    ax = axes[0, 0]
    if len(bic_data) > 1:
        ns, bics = zip(*bic_data)
        ax.plot(ns, bics, 'o-', color='#1f77b4', linewidth=2, markersize=8)
        best_n_idx = int(np.argmin(bics))
        ax.plot(ns[best_n_idx], bics[best_n_idx], 'o', color='#E31A1C', markersize=14,
                zorder=5, label=f'Best: {ns[best_n_idx]} comp.')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('BIC')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xticks(list(ns))
    else:
        ax.text(0.5, 0.5, f'Forced: {n_components} components\n(BIC selection skipped)',
                transform=ax.transAxes, ha='center', va='center', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title('A   BIC model selection', loc='left', fontweight='bold')

    # ---- Panel B: Histogram + GMM components ----
    ax = axes[0, 1]
    bins = np.arange(t_min, t_max + 1, 1)
    ax.hist(activation_times, bins=bins, density=True, alpha=0.5, color='#AAAAAA',
            edgecolor='white', label='Data')

    x_dense = np.linspace(activation_times.min() - 2, activation_times.max() + 2, 500)
    total_pdf = np.zeros_like(x_dense)

    for i in sorted_indices:
        group = group_mapping.get(int(i), 'other')
        color = COLORS.get(group, COLORS['gray'])
        component_pdf = weights[i] * norm.pdf(x_dense, means[i], stds[i])
        total_pdf += component_pdf
        ax.plot(x_dense, component_pdf, color=color, linewidth=2,
                label=f'{group.capitalize()} (μ={means[i]:.1f})')
        ax.fill_between(x_dense, component_pdf, alpha=0.15, color=color)

    ax.plot(x_dense, total_pdf, 'k--', linewidth=2, alpha=0.7, label='Mixture')
    ax.set_xlabel('Activation timepoint')
    ax.set_ylabel('Density')
    ax.set_title('B   GMM fit', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)

    # ---- Panel C: Posterior probabilities ----
    ax = axes[0, 2]
    valid_mask = df_act['activation_timepoint'].notna() & (df_act['activates'] == True)
    valid_act = df_act[valid_mask]
    act_times_valid = valid_act['activation_timepoint'].values
    max_post = posteriors.max(axis=1)

    for group in groups:
        gmask = valid_act['activation_group'].values == group
        if gmask.sum() > 0:
            ax.scatter(act_times_valid[gmask], max_post[gmask],
                      color=COLORS[group], alpha=0.5, s=25, label=group.capitalize(),
                      edgecolors='white', linewidths=0.3)

    # Cells classified as 'other'
    omask = valid_act['activation_group'].values == 'other'
    if omask.sum() > 0:
        ax.scatter(act_times_valid[omask], max_post[omask],
                  color=COLORS['gray'], alpha=0.3, s=15, label='Other')

    ax.axhline(0.7, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Activation timepoint')
    ax.set_ylabel('Posterior probability')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title('C   Classification confidence', loc='left', fontweight='bold')
    ax.legend(loc='lower left', fontsize=7)

    # ---- Panel D: Cluster assignment vs baseline intensity ----
    ax = axes[1, 0]
    if 'baseline_intensity' in df_groups.columns:
        for group in groups:
            gdata = df_groups[df_groups['activation_group'] == group]
            if len(gdata) > 0:
                ax.scatter(gdata['activation_timepoint'], gdata['baseline_intensity'],
                          color=COLORS[group], alpha=0.5, s=30, label=group.capitalize(),
                          edgecolors='white', linewidths=0.3)
        ax.set_xlabel('Activation timepoint')
        ax.set_ylabel('Baseline mNG intensity')
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Baseline intensity\nnot yet computed',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_title('D   Groups in 2D', loc='left', fontweight='bold')

    # ---- Panel E: Comparison with percentile method ----
    ax = axes[1, 1]

    # Run percentile classification internally for comparison
    pct_groups = pd.Series('other', index=valid_act.index)
    act_series = valid_act['activation_timepoint']
    early_th = act_series.quantile(early_pct / 100)
    avg_lo = act_series.quantile(average_pct_low / 100)
    avg_hi = act_series.quantile(average_pct_high / 100)
    late_th = act_series.quantile(late_pct / 100)
    pct_groups[act_series <= early_th] = 'early'
    pct_groups[(act_series > avg_lo) & (act_series <= avg_hi)] = 'average'
    pct_groups[act_series > late_th] = 'late'

    gmm_groups = valid_act['activation_group']

    all_labels = ['early', 'average', 'late', 'other']
    confusion = np.zeros((len(all_labels), len(all_labels)), dtype=int)
    for i, gl in enumerate(all_labels):
        for j, pl in enumerate(all_labels):
            confusion[i, j] = ((gmm_groups == gl) & (pct_groups == pl)).sum()

    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels([l.capitalize() for l in all_labels], rotation=45, ha='right')
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels([l.capitalize() for l in all_labels])
    ax.set_xlabel('Percentile method')
    ax.set_ylabel('GMM method')

    # Add cell counts to heatmap
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            val = confusion[i, j]
            if val > 0:
                text_color = 'white' if val > confusion.max() * 0.5 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                       fontsize=9, fontweight='bold', color=text_color)

    ax.set_title('E   GMM vs Percentile', loc='left', fontweight='bold')

    # Agreement percentage
    agree = sum(confusion[i, i] for i in range(len(all_labels)))
    total = confusion.sum()
    if total > 0:
        ax.text(0.98, 0.02, f'Agreement: {agree/total*100:.1f}%',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # ---- Panel F: Summary text ----
    ax = axes[1, 2]
    ax.axis('off')

    txt = "GMM CLASSIFICATION\n" + "=" * 28 + "\n\n"
    txt += f"Components: {n_components}\n"
    txt += f"Covariance: {covariance_type}\n\n"

    txt += f"{'Group':<8} {'Mean':>6} {'Std':>6} {'Wt':>6} {'N':>5}\n"
    txt += "-" * 34 + "\n"
    for i in sorted_indices:
        group = group_mapping.get(int(i), 'other')
        n_in = (labels == i).sum()
        txt += f"{group:<8} {means[i]:6.1f} {stds[i]:6.1f} {weights[i]:6.3f} {n_in:5d}\n"

    txt += "\n" + "=" * 28 + "\n"
    txt += f"Total cells: {len(activation_times)}\n"
    txt += f"Mean posterior: {max_post.mean():.3f}\n"
    low_conf = (max_post < 0.7).sum()
    txt += f"Low conf (<0.7): {low_conf} ({low_conf/len(max_post)*100:.1f}%)\n"

    if len(bic_data) > 1:
        txt += f"\nBest BIC: {min(b for _, b in bic_data):.0f}\n"
        txt += f"  (at {n_components} components)"

    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8, fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', edgecolor='#4682b4', alpha=0.9))
    ax.set_title('F   Summary', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    name = f"well_{full_name}_gmm_diagnostics{suffix_clean}"
    save_figure(fig, output_dir, name, save_pdf, save_svg)
    plt.close()

    # ---- Individual figures ----
    if save_individual:
        # Histogram + GMM fit (standalone)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(activation_times, bins=bins, density=True, alpha=0.5, color='#AAAAAA',
                edgecolor='white', label='Data')
        total_pdf = np.zeros_like(x_dense)
        for i in sorted_indices:
            group = group_mapping.get(int(i), 'other')
            color = COLORS.get(group, COLORS['gray'])
            component_pdf = weights[i] * norm.pdf(x_dense, means[i], stds[i])
            total_pdf += component_pdf
            ax.plot(x_dense, component_pdf, color=color, linewidth=2,
                    label=f'{group.capitalize()} (μ={means[i]:.1f})')
            ax.fill_between(x_dense, component_pdf, alpha=0.15, color=color)
        ax.plot(x_dense, total_pdf, 'k--', linewidth=2, alpha=0.7, label='Mixture')
        ax.set_xlabel('Activation timepoint')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_gmm_fit{suffix_clean}",
                    save_pdf, save_svg, subdir="individual")
        plt.close()

        # Posterior probability (standalone)
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in groups:
            gmask = valid_act['activation_group'].values == group
            if gmask.sum() > 0:
                ax.scatter(act_times_valid[gmask], max_post[gmask],
                          color=COLORS[group], alpha=0.5, s=25, label=group.capitalize(),
                          edgecolors='white', linewidths=0.3)
        omask = valid_act['activation_group'].values == 'other'
        if omask.sum() > 0:
            ax.scatter(act_times_valid[omask], max_post[omask],
                      color=COLORS['gray'], alpha=0.3, s=15, label='Other')
        ax.axhline(0.7, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('Activation timepoint')
        ax.set_ylabel('Posterior probability')
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_gmm_posteriors{suffix_clean}",
                    save_pdf, save_svg, subdir="individual")
        plt.close()

    print(f"  Saved GMM diagnostics")


# ==================== BASELINE CALCULATIONS ====================
def calculate_baseline_intensity(df_act, df_meas, baseline_frames=(0, 5)):
    start_t, end_t = baseline_frames
    baseline_values = {}
    
    for track_id in df_act['unique_track_id'].unique():
        track_data = df_meas[df_meas['unique_track_id'] == track_id]
        baseline_data = track_data[(track_data['timepoint'] >= start_t) & (track_data['timepoint'] <= end_t)]
        
        if len(baseline_data) > 0:
            baseline_values[track_id] = baseline_data['mean_intensity'].mean()
        elif len(track_data) > 0:
            baseline_values[track_id] = track_data.nsmallest(3, 'timepoint')['mean_intensity'].mean()
        else:
            baseline_values[track_id] = np.nan
    
    df_act = df_act.copy()
    df_act['baseline_intensity'] = df_act['unique_track_id'].map(baseline_values)
    return df_act


def calculate_baseline_bfp(df_act, df_meas, baseline_frames=(0, 5)):
    start_t, end_t = baseline_frames
    baseline_values = {}
    
    for track_id in df_act['unique_track_id'].unique():
        track_data = df_meas[df_meas['unique_track_id'] == track_id]
        baseline_data = track_data[(track_data['timepoint'] >= start_t) & (track_data['timepoint'] <= end_t)]
        
        if len(baseline_data) > 0 and 'bfp_mean_intensity' in baseline_data.columns:
            baseline_values[track_id] = baseline_data['bfp_mean_intensity'].mean()
        else:
            baseline_values[track_id] = np.nan
    
    df_act = df_act.copy()
    df_act['baseline_bfp'] = df_act['unique_track_id'].map(baseline_values)
    return df_act


def calculate_iqr_thresholds(df_act, percentile_low=25, percentile_high=75):
    """Calculate IQR thresholds from baseline intensity distribution."""
    valid_baseline = df_act['baseline_intensity'].dropna()
    
    q_low = valid_baseline.quantile(percentile_low / 100)
    q_high = valid_baseline.quantile(percentile_high / 100)
    median_val = valid_baseline.median()
    iqr = q_high - q_low
    
    return {
        'q_low': q_low,
        'q_high': q_high,
        'median': median_val,
        'iqr': iqr,
        'percentile_low': percentile_low,
        'percentile_high': percentile_high,
        'n_total': len(valid_baseline)
    }


def filter_by_baseline_intensity(df_act, df_meas, baseline_min, baseline_max):
    n_before = len(df_act)
    mask = (df_act['baseline_intensity'] >= baseline_min) & (df_act['baseline_intensity'] <= baseline_max)
    df_act_filtered = df_act[mask].copy()
    n_after = len(df_act_filtered)
    
    valid_track_ids = df_act_filtered['unique_track_id'].unique()
    df_meas_filtered = df_meas[df_meas['unique_track_id'].isin(valid_track_ids)].copy()
    
    print(f"\n" + "="*60)
    print("BASELINE INTENSITY FILTERING")
    print("="*60)
    print(f"  Range: {baseline_min:.1f} - {baseline_max:.1f}")
    print(f"  Before: {n_before}, After: {n_after}, Removed: {n_before - n_after}")
    print(f"  Retention: {n_after/n_before*100:.1f}%")
    
    return df_act_filtered, df_meas_filtered


def merge_bfp_with_gfp(df_meas, df_bfp):
    df_merged = df_meas.merge(
        df_bfp[['unique_track_id', 'timepoint', 'bfp_mean_intensity', 'bfp_max_intensity', 'bfp_min_intensity']],
        on=['unique_track_id', 'timepoint'], how='left'
    )
    print(f"Merged measurements: {len(df_merged)} rows")
    return df_merged


def normalize_mng_by_baseline_bfp(df_meas, baseline_frames=(0, 5)):
    df_meas = df_meas.copy()
    start_t, end_t = baseline_frames
    
    baseline_bfp_per_cell = df_meas[
        (df_meas['timepoint'] >= start_t) & (df_meas['timepoint'] <= end_t)
    ].groupby('unique_track_id')['bfp_mean_intensity'].mean()
    
    df_meas['baseline_bfp_cell'] = df_meas['unique_track_id'].map(baseline_bfp_per_cell)
    df_meas['fraction_reconstituted'] = df_meas['mean_intensity'] / df_meas['baseline_bfp_cell']
    
    median_baseline_bfp = baseline_bfp_per_cell.median()
    df_meas['mean_intensity_corrected'] = (df_meas['mean_intensity'] / df_meas['baseline_bfp_cell']) * median_baseline_bfp
    
    print(f"  Normalized by baseline BFP (median = {median_baseline_bfp:.1f})")
    return df_meas


def classify_by_response(df_act, r2_min=0.7, sd_multiplier=1.0, method='sd'):
    """
    Classify activating cells into low/medium/high responders based on sigmoid plateau value.

    Plateau value = sigmoid_baseline + sigmoid_amplitude (fitted asymptote of the sigmoid).
    Only cells with sigmoid_r2 >= r2_min are classified; others get 'unfit'.

    method='sd'      : groups defined by mean ± sd_multiplier * SD (default)
    method='tertile' : groups defined by 33rd / 66th percentile cuts (equal-sized thirds)
    """
    df = df_act.copy()
    df['plateau_value'] = np.nan
    df['response_group'] = 'unfit'

    if 'sigmoid_baseline' not in df.columns or 'sigmoid_amplitude' not in df.columns:
        print("  Warning: sigmoid_baseline/sigmoid_amplitude not found — response grouping unavailable")
        return df

    df['plateau_value'] = df['sigmoid_baseline'] + df['sigmoid_amplitude']

    good_fit = pd.Series(True, index=df.index)
    if 'sigmoid_r2' in df.columns:
        good_fit = df['sigmoid_r2'] >= r2_min

    plateau_vals = df.loc[good_fit, 'plateau_value'].dropna()

    if len(plateau_vals) < 3:
        print("  Warning: too few cells with good sigmoid fits for response grouping")
        return df

    if method == 'tertile':
        low_threshold  = plateau_vals.quantile(1 / 3)
        high_threshold = plateau_vals.quantile(2 / 3)
        method_label   = 'tertile (33rd/66th percentile)'
    else:
        mean_p = plateau_vals.mean()
        std_p  = plateau_vals.std()
        low_threshold  = mean_p - sd_multiplier * std_p
        high_threshold = mean_p + sd_multiplier * std_p
        method_label   = f'SD multiplier={sd_multiplier}'

    df.loc[good_fit, 'response_group'] = 'medium'
    df.loc[good_fit & (df['plateau_value'] <  low_threshold),  'response_group'] = 'low'
    df.loc[good_fit & (df['plateau_value'] >= high_threshold), 'response_group'] = 'high'

    print(f"\nResponse-based groups ({method_label}, R²≥{r2_min}):")
    print(f"  Low    (plateau < {low_threshold:.3f}): {(df['response_group'] == 'low').sum()}")
    print(f"  Medium ({low_threshold:.3f} ≤ plateau < {high_threshold:.3f}): {(df['response_group'] == 'medium').sum()}")
    print(f"  High   (plateau ≥ {high_threshold:.3f}): {(df['response_group'] == 'high').sum()}")
    print(f"  Unfit  (R² < {r2_min}): {(df['response_group'] == 'unfit').sum()}")

    df.attrs['response_thresholds'] = {
        'method': method, 'low_threshold': low_threshold,
        'high_threshold': high_threshold, 'r2_min': r2_min,
    }
    return df


# ==================== AUC & SURVIVAL FUNCTIONS ====================

def compute_auc(df_act, df_meas, signal_col='mean_intensity', baseline_frames=(0, 5)):
    """
    Compute area under the mNG/BFP ratio curve above baseline per cell.
    Uses trapezoidal integration of (signal - baseline), clipped at zero.
    """
    _ycol = signal_col if signal_col in df_meas.columns else 'mean_intensity'
    start_t, end_t = baseline_frames
    auc_values = {}

    for track_id in df_act['unique_track_id']:
        track_data = df_meas[df_meas['unique_track_id'] == track_id].sort_values('timepoint')
        if _ycol not in track_data.columns or len(track_data) < 2:
            auc_values[track_id] = np.nan
            continue
        baseline_data = track_data[
            (track_data['timepoint'] >= start_t) & (track_data['timepoint'] <= end_t)
        ][_ycol].dropna()
        if len(baseline_data) == 0:
            auc_values[track_id] = np.nan
            continue
        baseline = baseline_data.mean()
        signal = track_data[_ycol].ffill().fillna(baseline)
        signal_above = (signal - baseline).clip(lower=0)
        auc_values[track_id] = np.trapz(signal_above.values, track_data['timepoint'].values)

    df_act = df_act.copy()
    df_act['auc'] = df_act['unique_track_id'].map(auc_values)
    n = df_act['auc'].notna().sum()
    print(f"  AUC computed for {n}/{len(df_act)} cells — median: {df_act['auc'].median():.2f}")
    return df_act


def compute_activation_kinetics(df_act, df_meas, signal_col='mean_intensity',
                                baseline_frames=(0, 5), threshold_fraction=0.5):
    """
    Compute per-cell activation kinetics and add columns to df_act:

      rise_time       : frames from activation onset to plateau.
                        Uses plateau_t - activation_start_t when both are present,
                        falls back to plateau_t - activation_timepoint.
                        NaN when plateau_t is absent.

      duration_active : number of frames where the signal exceeds
                        baseline + threshold_fraction * (plateau_value - baseline).
                        Uses the mNG/BFP ratio (or mNG intensity) from df_meas.
                        Falls back to signal max when plateau_value is NaN.
    """
    df = df_act.copy()
    start_t, end_t = baseline_frames

    # --- rise_time from sigmoid-fit columns (already in df_act from script 2) ---
    if 'plateau_t' in df.columns:
        ref_col = 'activation_start_t' if 'activation_start_t' in df.columns else 'activation_timepoint'
        df['rise_time'] = (df['plateau_t'] - df[ref_col]).clip(lower=0)
    else:
        df['rise_time'] = np.nan

    # --- duration_active from the measured trajectory ---
    _ycol = signal_col if signal_col in df_meas.columns else 'mean_intensity'
    duration_values = {}
    for _, row in df.iterrows():
        tid = row['unique_track_id']
        track = df_meas[df_meas['unique_track_id'] == tid].sort_values('timepoint')
        if _ycol not in track.columns or len(track) < 3:
            duration_values[tid] = np.nan
            continue
        baseline_vals = track[
            (track['timepoint'] >= start_t) & (track['timepoint'] <= end_t)
        ][_ycol].dropna()
        if len(baseline_vals) == 0:
            duration_values[tid] = np.nan
            continue
        baseline = baseline_vals.mean()
        plateau = row.get('plateau_value', np.nan)
        if pd.isna(plateau):
            plateau = track[_ycol].max()
        cutoff = baseline + threshold_fraction * (plateau - baseline)
        duration_values[tid] = int((track[_ycol] > cutoff).sum())

    df['duration_active'] = df['unique_track_id'].map(duration_values)

    n_rise = df['rise_time'].notna().sum()
    n_dur  = df['duration_active'].notna().sum()
    med_dur = df['duration_active'].median()
    print(f"  Kinetics: rise_time for {n_rise} cells, "
          f"duration_active for {n_dur} cells (median={med_dur:.1f} frames)")
    return df


def compute_track_duration(df_act, df_meas):
    """
    Compute per-cell track duration from df_meas and add columns to df_act:

      track_start_t  : first timepoint the cell is observed
      track_end_t    : last timepoint the cell is observed
      track_duration : track_end_t - track_start_t (proxy for cell survival)

    Cells that disappear early may have died or detached due to viral cytopathic effect.
    """
    gb = df_meas.groupby('unique_track_id')['timepoint']
    start_t = gb.min()
    end_t   = gb.max()

    df = df_act.copy()
    df['track_start_t']  = df['unique_track_id'].map(start_t)
    df['track_end_t']    = df['unique_track_id'].map(end_t)
    df['track_duration'] = df['track_end_t'] - df['track_start_t']

    n = df['track_duration'].notna().sum()
    print(f"  Track duration computed for {n}/{len(df)} cells "
          f"(median={df['track_duration'].median():.1f} frames)")
    return df


def compute_division_status(df_act, df_divisions):
    """
    Annotate df_act with lineage information from load_division_events().

    Adds columns:
        is_daughter            : True if cell was born from a tracked division
        division_timepoint     : frame at which cell was born (NaN if not a daughter)
        parent_unique_track_id : parent cell id (NaN if not a daughter)
        parent_activated       : True if parent cell had activates==True
        parent_activation_t    : activation_timepoint of parent (NaN if not activated)
        parent_response_group  : response_group of parent (NaN if unavailable)
    """
    df = df_act.copy()

    # Initialise all columns to safe defaults before any merge
    df['is_daughter']            = False
    df['division_timepoint']     = np.nan
    df['parent_unique_track_id'] = np.nan
    df['parent_activated']       = False
    df['parent_activation_t']    = np.nan
    df['parent_response_group']  = np.nan

    if df_divisions is None or len(df_divisions) == 0:
        print("  No division data — division columns set to defaults")
        return df

    # Merge division table into df_act
    df = df.merge(
        df_divisions[['unique_track_id', 'division_timepoint', 'parent_unique_track_id']],
        on='unique_track_id', how='left', suffixes=('', '_div')
    )
    # Prefer the merged columns, drop init defaults
    if 'division_timepoint_div' in df.columns:
        df['division_timepoint'] = df['division_timepoint_div'].fillna(df['division_timepoint'])
        df.drop(columns=['division_timepoint_div'], inplace=True)
    if 'parent_unique_track_id_div' in df.columns:
        df['parent_unique_track_id'] = df['parent_unique_track_id_div'].fillna(df['parent_unique_track_id'])
        df.drop(columns=['parent_unique_track_id_div'], inplace=True)
    df['is_daughter'] = df['division_timepoint'].notna()

    # Build a lookup table from the activation data
    act_idx = df_act.set_index('unique_track_id')

    def _lookup(parent_ids, col, default):
        return [
            act_idx.at[pid, col] if isinstance(pid, str) and pid in act_idx.index else default
            for pid in parent_ids
        ]

    df['parent_activated']    = _lookup(df['parent_unique_track_id'], 'activates', False)
    df['parent_activation_t'] = _lookup(df['parent_unique_track_id'], 'activation_timepoint', np.nan)
    if 'response_group' in df_act.columns:
        df['parent_response_group'] = _lookup(df['parent_unique_track_id'], 'response_group', np.nan)

    n_daughters = int(df['is_daughter'].sum())
    n_from_act  = int((df['is_daughter'] & df['parent_activated']).sum())
    print(f"  Division status: {n_daughters} daughter cells "
          f"({n_from_act} born from activated parents)")
    return df


def detect_death_by_topology(df_meas,
                              search_radius=100,
                              min_fragments=3,
                              max_fragment_area_fraction=0.45,
                              min_frames_before_end=5,
                              search_window=3,
                              border_margin=80):
    """
    Detect nuclear fragmentation by analysing ultrack track topology.

    When Cellpose segments a fragmenting nucleus it detects multiple small
    objects; ultrack creates new short tracks for each fragment.  By
    inspecting what tracks START near the location where another track ENDS
    we can classify the ending event:

        Division      : parent ends → exactly 2 new tracks start within
                        search_radius, each with area ≈ 40–70 % of parent
        Fragmentation : parent ends → ≥ min_fragments new small tracks start,
                        OR 2 children where neither looks like a normal daughter
        Abrupt end    : track vanishes with no nearby children AND the final
                        position is > border_margin pixels from any FOV edge
                        (tracks ending near the edge likely exited the FOV).

    Parameters
    ----------
    df_meas                    : full per-timepoint measurement DataFrame
    search_radius              : pixel radius to search for child tracks (default 100)
    min_fragments              : minimum small-child count for fragmentation (default 3)
    max_fragment_area_fraction : children must be < this fraction of parent area
                                 to count as fragments, not daughters (default 0.45)
    min_frames_before_end      : ignore tracks ending in the last N frames (default 5)
    search_window              : frames after track end to look for children (default 3)
    border_margin              : pixel margin from inferred FOV boundary; abrupt-end
                                 tracks whose last centroid falls within this margin
                                 are assumed to have exited the FOV and are NOT
                                 flagged as deaths (default 80)

    Returns
    -------
    dict  {unique_track_id : (death_timepoint, death_type)}
          death_type is 'fragmentation' or 'abrupt_end'
    """
    if 'centroid-0' not in df_meas.columns or 'area_pixels' not in df_meas.columns:
        print("  Warning: topology detection requires centroid-0/1 and area_pixels")
        return {}

    t_max = df_meas['timepoint'].max()

    # Infer approximate FOV boundaries from the range of centroids observed.
    # Tracks ending within border_margin of any edge are likely FOV exits.
    y_fov_min = df_meas['centroid-0'].min()
    y_fov_max = df_meas['centroid-0'].max()
    x_fov_min = df_meas['centroid-1'].min()
    x_fov_max = df_meas['centroid-1'].max()

    def _near_border(y, x):
        return (y < y_fov_min + border_margin or
                y > y_fov_max - border_margin or
                x < x_fov_min + border_margin or
                x > x_fov_max - border_margin)

    # Pre-compute per-track summary: last timepoint, last centroid, mean area
    track_last = (df_meas.sort_values('timepoint')
                  .groupby('unique_track_id')
                  .last()[['timepoint', 'centroid-0', 'centroid-1', 'area_pixels']]
                  .rename(columns={'timepoint': 't_last',
                                   'centroid-0': 'y_last',
                                   'centroid-1': 'x_last',
                                   'area_pixels': 'area_last'}))
    track_mean_area = df_meas.groupby('unique_track_id')['area_pixels'].mean()
    track_last['mean_area'] = track_mean_area

    # Per-track: first timepoint and first centroid (to identify child tracks)
    track_first = (df_meas.sort_values('timepoint')
                   .groupby('unique_track_id')
                   .first()[['timepoint', 'centroid-0', 'centroid-1', 'area_pixels']]
                   .rename(columns={'timepoint': 't_first',
                                    'centroid-0': 'y_first',
                                    'centroid-1': 'x_first',
                                    'area_pixels': 'area_first'}))

    results = {}

    # Only examine tracks that end well before the experiment ends
    ending_tracks = track_last[track_last['t_last'] < t_max - min_frames_before_end]

    for tid, row in ending_tracks.iterrows():
        t_end       = row['t_last']
        y_end       = row['y_last']
        x_end       = row['x_last']
        parent_area = row['mean_area']

        # Find tracks that START within search_window frames after this track ends
        # and within search_radius pixels of the last centroid
        window_mask = (
            (track_first['t_first'] >= t_end) &
            (track_first['t_first'] <= t_end + search_window) &
            (track_first.index != tid)
        )
        candidates = track_first[window_mask].copy()

        # Compute distance to each candidate
        if len(candidates) > 0:
            dy = candidates['y_first'] - y_end
            dx = candidates['x_first'] - x_end
            dist = np.sqrt(dy**2 + dx**2)
            nearby = candidates[dist <= search_radius].copy()
            nearby['dist'] = dist[dist <= search_radius]
            nearby['area_frac'] = nearby['area_first'] / max(parent_area, 1)
        else:
            nearby = candidates  # empty

        n_nearby = len(nearby)

        if n_nearby == 0:
            # Track ends with no detectable children.
            # Only flag as death if the ending position is well inside the FOV
            # (near-border endings are almost certainly FOV exits).
            if not _near_border(y_end, x_end):
                results[tid] = (t_end, 'abrupt_end')

        elif n_nearby == 1:
            # Single child — ambiguous (partial tracking handoff?); skip
            pass

        elif n_nearby == 2:
            # Check if the two children look like division daughters
            # (each roughly 40–70 % of the parent area)
            both_daughter = (nearby['area_frac'] > 0.35).all() and \
                            (nearby['area_frac'] < 0.75).all()
            if not both_daughter:
                # Two children but wrong sizes → fragmentation
                results[tid] = (t_end, 'fragmentation')
            # else: normal division — do NOT flag as death

        else:  # n_nearby >= 3
            # Many children → fragmentation if enough are genuinely small
            small_fracs = nearby['area_frac'] <= max_fragment_area_fraction
            if small_fracs.sum() >= min_fragments:
                results[tid] = (t_end, 'fragmentation')

    n_frag   = sum(1 for _, t in results.values() if t == 'fragmentation')
    n_abrupt = sum(1 for _, t in results.values() if t == 'abrupt_end')
    n_fov_exit = len(ending_tracks) - len(results) - (
        sum(1 for row in ending_tracks.itertuples()
            if _near_border(row.y_last, row.x_last))
    )
    print(f"  Topology detection: {n_frag} fragmentation, {n_abrupt} abrupt-end "
          f"(interior), FOV-exit filtered out "
          f"(out of {len(ending_tracks)} tracks ending before t_max-{min_frames_before_end})")
    return results


def _first_sustained(mask_series, timepoints, min_duration):
    """Return the first timepoint where mask_series is True for min_duration consecutive frames.
    Returns np.nan if no such run exists."""
    count = 0
    start_idx = None
    vals = mask_series.values
    tps  = timepoints.values
    for i, val in enumerate(vals):
        if val:
            if count == 0:
                start_idx = i
            count += 1
            if count >= min_duration:
                return tps[start_idx]
        else:
            count = 0
            start_idx = None
    return np.nan


def compute_cell_death(df_act, df_meas,
                       solidity_drop_threshold=0.20,
                       min_frames_baseline=3,
                       smooth_window=3,
                       min_sustained_frames=5,
                       min_track_length=15):
    """
    Estimate a probable cell-death timepoint from nuclear morphology and track topology.

    Strategy (in order of reliability)
    ------------------------------------
    a. BFP intensity CV spike  [requires bfp_cv column from script 1]
       CV = std/mean within the nuclear BFP mask.  Fragmentation makes the
       intensity patchy → CV rises sharply.  Division keeps CV low (uniform
       daughter nuclei).  Most reliable, division-immune signal.

    b. Solidity drop  [requires solidity column from script 1]
       Fragmenting nuclei become irregular (low solidity). Division daughters
       are round (high solidity). Threshold 0.20, sustained for 5 frames.
       More conservative than area-based methods to avoid false positives.

    c. Track topology  [always available]
       detect_death_by_topology() inspects what tracks start near the location
       where a track ends:
         • Fragmentation  — ≥3 small child tracks appear (nuclear fragments)
         • Abrupt end     — track disappears with no children AND ending
                            position is well inside the FOV (not a FOV exit)

    Area-collapse and swelling detection are intentionally omitted: area
    changes alone are unreliable because ultrack often continues one daughter
    as the parent track after division, creating a persistent ~50% area drop
    that is indistinguishable from condensation/fragmentation by area alone.

    Adds to df_act
    --------------
    probable_death_timepoint     : float (NaN = survived / not detected)
    death_type                   : str  ('bfp_fragmentation' | 'fragmentation' |
                                         'abrupt_end' | NaN)
    frames_post_activation_death : float (NaN if no death or not activated)
    """
    has_shape  = 'solidity' in df_meas.columns
    has_bfp_cv = 'bfp_cv'  in df_meas.columns

    if has_bfp_cv:
        print("  BFP intensity CV available — using as primary fragmentation signal")
    if has_shape:
        print("  Solidity available — using sustained drop (≥0.20, ≥5 frames) as "
              "fragmentation signal (division-resistant)")
    if not has_bfp_cv and not has_shape:
        print("  Warning: no BFP CV or solidity — only topology detection available. "
              "Re-run script 1 to add these columns.")

    df = df_act.copy()
    df['probable_death_timepoint']    = np.nan
    df['death_type']                  = np.nan
    df['frames_post_activation_death'] = np.nan

    # Sort measurements once
    meas_sorted = df_meas.sort_values(['unique_track_id', 'timepoint'])

    for tid, track in meas_sorted.groupby('unique_track_id'):
        if len(track) < max(min_frames_baseline + 2, min_track_length):
            continue

        track = track.sort_values('timepoint').reset_index(drop=True)

        # ── Baseline (first N frames of the track) ───────────────────────────
        baseline = track.head(min_frames_baseline)

        if has_shape:
            base_solidity = baseline['solidity'].mean()
            sol_s = (track['solidity']
                     .rolling(smooth_window, center=True, min_periods=1).mean())
        else:
            base_solidity = np.nan
            sol_s = None

        if has_bfp_cv:
            base_bfp_cv = baseline['bfp_cv'].mean()
            bfp_cv_s = (track['bfp_cv']
                        .rolling(smooth_window, center=True, min_periods=1).mean())
        else:
            base_bfp_cv = np.nan
            bfp_cv_s = None

        death_tp   = np.nan
        death_type = np.nan

        # ── (a) BFP intensity CV spike — primary fragmentation signal ────────
        if has_bfp_cv and base_bfp_cv > 0:
            bfp_cv_mask = bfp_cv_s > (base_bfp_cv * 2.0)
            first_bfp_cv = _first_sustained(bfp_cv_mask, track['timepoint'],
                                            min_sustained_frames)
            if not np.isnan(first_bfp_cv):
                death_tp   = first_bfp_cv
                death_type = 'bfp_fragmentation'

        # ── (b) Solidity drop — secondary fragmentation signal ───────────────
        # Threshold is intentionally conservative (0.20) and requires more
        # sustained frames (5) to avoid transient noise from crowded cells.
        # NOTE: this signal is skipped when bfp_cv already fired.
        if has_shape and np.isnan(death_tp):
            sol_drop_mask = sol_s < (base_solidity - solidity_drop_threshold)
            first_sol_drop = _first_sustained(sol_drop_mask, track['timepoint'],
                                              min_sustained_frames)
            if not np.isnan(first_sol_drop):
                death_tp   = first_sol_drop
                death_type = 'fragmentation'

        # ── Store morphology result ───────────────────────────────────────────
        if not np.isnan(death_tp):
            mask = df['unique_track_id'] == tid
            df.loc[mask, 'probable_death_timepoint'] = death_tp
            df.loc[mask, 'death_type'] = death_type
            if 'activation_timepoint' in df.columns:
                act_tp = df.loc[mask, 'activation_timepoint']
                if act_tp.notna().any():
                    df.loc[mask, 'frames_post_activation_death'] = death_tp - act_tp.values[0]

    # ── Topology-based detection (track ending analysis) ─────────────────────
    # Applied to any track NOT already flagged by BFP CV or solidity signals.
    # Uses FOV-boundary filtering to suppress FOV-exit false positives.
    topology_results = detect_death_by_topology(df_meas)
    topo_applied = 0
    for tid, (death_tp_topo, dtype_topo) in topology_results.items():
        mask = df['unique_track_id'] == tid
        if mask.any() and df.loc[mask, 'probable_death_timepoint'].isna().all():
            df.loc[mask, 'probable_death_timepoint'] = death_tp_topo
            df.loc[mask, 'death_type'] = dtype_topo
            if 'activation_timepoint' in df.columns:
                act_tp = df.loc[mask, 'activation_timepoint']
                if act_tp.notna().any():
                    df.loc[mask, 'frames_post_activation_death'] = (
                        death_tp_topo - act_tp.values[0])
            topo_applied += 1
    if topo_applied:
        print(f"  Topology detection added {topo_applied} additional death events")

    n_dead   = df['probable_death_timepoint'].notna().sum()
    n_total  = len(df)
    type_counts = df['death_type'].value_counts().to_dict()
    print(f"  Cell death detected: {n_dead}/{n_total} cells "
          f"({100*n_dead/n_total:.1f}%) — types: {type_counts}")
    return df


# ==================== DEATH CLASSIFIER ====================

def extract_death_features(df_meas, min_track_length=5):
    """
    Extract per-track scalar features for the death Random Forest classifier.

    Returns a DataFrame with one row per unique_track_id.
    NaN is used for features that require unavailable columns (solidity, bfp_cv).
    Missing values are handled by the classifier pipeline (filled with -1).

    Features
    --------
    Track-level:
        track_length       – number of frames tracked
        t_last_norm        – (t_last - t_min) / (t_max - t_min); 1.0 = survived to end
        ends_before_end    – 1 if track ends ≥5 frames before experiment end
        near_border        – 1 if last centroid is within 80 px of inferred FOV edge
    Area dynamics:
        final_area_frac    – mean area (last 3 fr) / mean area (first 3 fr)
        min_area_frac      – min smoothed area / baseline area
        max_area_frac      – max smoothed area / baseline area
        area_cv            – coefficient of variation of area over the track
        area_slope_last10  – linear slope of area/baseline in the last ~10 frames
    Shape (requires solidity column):
        min_solidity       – minimum solidity over the track
        final_solidity     – mean solidity in last 3 frames
        solidity_slope     – linear slope of solidity in the last ~10 frames
    BFP texture (requires bfp_cv column):
        max_bfp_cv_ratio   – max(bfp_cv) / baseline bfp_cv
        final_bfp_cv_ratio – mean bfp_cv (last 3 fr) / baseline bfp_cv
    Phase contrast (requires phase_cv / phase_mean columns):
        max_phase_cv_ratio     – max(phase_cv) / baseline phase_cv
        final_phase_cv_ratio   – mean phase_cv (last 3 fr) / baseline phase_cv
        final_phase_mean_ratio – mean phase_mean (last 3 fr) / baseline phase_mean
    Topology:
        n_nearby_starters  – tracks starting within 3 fr / 100 px of this track's end
        child_area_frac    – mean area fraction of nearby starters vs parent area
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.spatial import cKDTree

    has_shape     = 'solidity'   in df_meas.columns
    has_bfp_cv    = 'bfp_cv'     in df_meas.columns
    has_phase     = 'phase_cv'   in df_meas.columns
    has_centroids = ('centroid-0' in df_meas.columns and
                     'centroid-1' in df_meas.columns)

    t_max   = df_meas['timepoint'].max()
    t_min   = df_meas['timepoint'].min()
    t_range = max(float(t_max - t_min), 1.0)

    BORDER        = 80
    TOPO_RADIUS   = 100
    TOPO_WINDOW   = 3

    if has_centroids:
        y_fov_min = df_meas['centroid-0'].min()
        y_fov_max = df_meas['centroid-0'].max()
        x_fov_min = df_meas['centroid-1'].min()
        x_fov_max = df_meas['centroid-1'].max()

        # Pre-build topology lookup: track_last / track_first positions
        tl = (df_meas.sort_values('timepoint')
              .groupby('unique_track_id')
              .last()[['timepoint', 'centroid-0', 'centroid-1', 'area_pixels']])
        tf = (df_meas.sort_values('timepoint')
              .groupby('unique_track_id')
              .first()[['timepoint', 'centroid-0', 'centroid-1', 'area_pixels']])
        mean_area_map = df_meas.groupby('unique_track_id')['area_pixels'].mean()

        # Group starters by their first timepoint for fast lookup
        starters_by_t = {}
        for uid, row in tf.iterrows():
            t = int(row['timepoint'])
            starters_by_t.setdefault(t, []).append(
                (uid, float(row['centroid-0']), float(row['centroid-1']),
                 float(row['area_pixels'])))
    else:
        tl = tf = mean_area_map = starters_by_t = None

    rows = []
    meas_sorted = df_meas.sort_values(['unique_track_id', 'timepoint'])

    for tid, track in meas_sorted.groupby('unique_track_id'):
        track = track.sort_values('timepoint').reset_index(drop=True)
        n = len(track)
        if n < min_track_length:
            continue

        tp   = track['timepoint'].values.astype(float)
        area = track['area_pixels'].values.astype(float)
        area_s = uniform_filter1d(area, size=min(3, n), mode='nearest')

        n_base   = min(3, n)
        base_area = area_s[:n_base].mean()
        if base_area <= 0:
            continue

        t_last     = float(tp[-1])
        t_last_norm = (t_last - t_min) / t_range
        ends_before_end = int(t_last < t_max - 5)

        final_area_frac = float(area_s[-min(3, n):].mean() / base_area)
        min_area_frac   = float(area_s.min() / base_area)
        max_area_frac   = float(area_s.max() / base_area)
        area_cv         = float(area.std() / area.mean()) if area.mean() > 0 else 0.0

        n_tail = max(3, min(10, n // 3))
        tail_tp   = tp[-n_tail:]
        tail_norm = area_s[-n_tail:] / base_area
        if len(tail_tp) > 1 and tail_tp[-1] > tail_tp[0]:
            area_slope = float(np.polyfit(tail_tp, tail_norm, 1)[0])
        else:
            area_slope = 0.0

        # ── Border proximity ─────────────────────────────────────────────────
        near_border = 0
        if has_centroids:
            y_last = float(track['centroid-0'].iloc[-1])
            x_last = float(track['centroid-1'].iloc[-1])
            near_border = int(
                y_last < y_fov_min + BORDER or y_last > y_fov_max - BORDER or
                x_last < x_fov_min + BORDER or x_last > x_fov_max - BORDER)
        # ── Solidity ─────────────────────────────────────────────────────────
        min_solidity   = np.nan
        final_solidity = np.nan
        solidity_slope = np.nan
        if has_shape:
            sol   = track['solidity'].values.astype(float)
            sol_s = uniform_filter1d(sol, size=min(3, n), mode='nearest')
            min_solidity   = float(sol_s.min())
            final_solidity = float(sol_s[-min(3, n):].mean())
            if len(tail_tp) > 1 and tail_tp[-1] > tail_tp[0]:
                solidity_slope = float(np.polyfit(tail_tp, sol_s[-n_tail:], 1)[0])

        # ── BFP CV ───────────────────────────────────────────────────────────
        max_bfp_cv_ratio   = np.nan
        final_bfp_cv_ratio = np.nan
        if has_bfp_cv:
            cv   = track['bfp_cv'].values.astype(float)
            cv_s = uniform_filter1d(cv, size=min(3, n), mode='nearest')
            base_cv = cv_s[:n_base].mean()
            if base_cv > 0:
                max_bfp_cv_ratio   = float(cv_s.max() / base_cv)
                final_bfp_cv_ratio = float(cv_s[-min(3, n):].mean() / base_cv)

        # ── Phase contrast ────────────────────────────────────────────────────
        max_phase_cv_ratio   = np.nan
        final_phase_cv_ratio = np.nan
        final_phase_mean_ratio = np.nan
        if has_phase:
            pcv   = track['phase_cv'].values.astype(float)
            pcv_s = uniform_filter1d(pcv, size=min(3, n), mode='nearest')
            base_pcv = pcv_s[:n_base].mean()
            if base_pcv > 0:
                max_phase_cv_ratio   = float(pcv_s.max() / base_pcv)
                final_phase_cv_ratio = float(pcv_s[-min(3, n):].mean() / base_pcv)
            pm    = track['phase_mean'].values.astype(float)
            pm_s  = uniform_filter1d(pm, size=min(3, n), mode='nearest')
            base_pm = pm_s[:n_base].mean()
            if abs(base_pm) > 0:
                final_phase_mean_ratio = float(pm_s[-min(3, n):].mean() / base_pm)

        # ── Topology ─────────────────────────────────────────────────────────
        n_nearby_starters = 0
        child_area_frac   = 0.0
        if has_centroids and starters_by_t is not None and tid in tl.index:
            t_end       = int(tl.loc[tid, 'timepoint'])
            y_end       = float(tl.loc[tid, 'centroid-0'])
            x_end       = float(tl.loc[tid, 'centroid-1'])
            parent_area = float(mean_area_map.get(tid, 1.0))

            nearby_areas = []
            for dt in range(TOPO_WINDOW + 1):
                for (uid2, y2, x2, a2) in starters_by_t.get(t_end + dt, []):
                    if uid2 == tid:
                        continue
                    if np.sqrt((y2 - y_end)**2 + (x2 - x_end)**2) <= TOPO_RADIUS:
                        nearby_areas.append(a2 / max(parent_area, 1.0))

            n_nearby_starters = len(nearby_areas)
            child_area_frac   = float(np.mean(nearby_areas)) if nearby_areas else 0.0

        rows.append({
            'unique_track_id':    tid,
            'track_length':       n,
            't_last_norm':        t_last_norm,
            'ends_before_end':    ends_before_end,
            'near_border':        near_border,
            'final_area_frac':    final_area_frac,
            'min_area_frac':      min_area_frac,
            'max_area_frac':      max_area_frac,
            'area_cv':            area_cv,
            'area_slope_last10':  area_slope,
            'min_solidity':       min_solidity,
            'final_solidity':     final_solidity,
            'solidity_slope':     solidity_slope,
            'max_bfp_cv_ratio':     max_bfp_cv_ratio,
            'final_bfp_cv_ratio':   final_bfp_cv_ratio,
            'max_phase_cv_ratio':   max_phase_cv_ratio,
            'final_phase_cv_ratio': final_phase_cv_ratio,
            'final_phase_mean_ratio': final_phase_mean_ratio,
            'n_nearby_starters':    n_nearby_starters,
            'child_area_frac':    child_area_frac,
        })

    return pd.DataFrame(rows).set_index('unique_track_id')


def train_death_classifier(annotation_csv, df_meas,
                           model_output_path='death_classifier.pkl',
                           alive_sample=150,
                           division_sample=80,
                           random_seed=42):
    """
    Train a Random Forest classifier to detect cell death.

    Labels
    ------
    Positive (death=1):
        Manual annotations labelled 'death' in annotation_csv.

    Negative (death=0) — combined:
        • Manual 'division' and 'alive' annotations from annotation_csv
        • Auto-labelled alive: tracks that reach the final frame (t_last == t_max)
          (sampled up to alive_sample to balance classes)
        • Auto-labelled division: tracks ending with exactly 2 children
          whose area fractions are both 0.30–0.75 (sampled up to
          division_sample)

    The Random Forest is wrapped in an sklearn Pipeline that imputes NaN
    features with -1 before fitting.  The fitted pipeline is saved as a
    joblib pickle to model_output_path.

    Prints
    ------
    Class breakdown, 5-fold stratified cross-validation ROC-AUC and
    accuracy, and the top-10 most important features.

    Returns
    -------
    (pipeline, feature_names, cv_scores_dict)
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold, cross_validate
        import joblib
    except ImportError:
        raise ImportError(
            "scikit-learn and joblib are required. "
            "Run: pip install scikit-learn joblib")

    annotation_csv = Path(annotation_csv)
    if not annotation_csv.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {annotation_csv}")

    df_ann = pd.read_csv(annotation_csv)
    required = {'unique_track_id', 'annotated_label'}
    if not required.issubset(df_ann.columns):
        raise ValueError(f"Annotation CSV must have columns: {required}")

    print(f"\n  Training death classifier")
    print(f"  Annotation CSV: {annotation_csv}  ({len(df_ann)} rows)")

    # ── Extract features for every track ─────────────────────────────────────
    print("  Extracting per-track features …")
    feat = extract_death_features(df_meas, min_track_length=5)
    FEATURE_COLS = [c for c in feat.columns]   # all numeric feature columns

    t_max = df_meas['timepoint'].max()

    # ── Manual annotation labels ──────────────────────────────────────────────
    manual_deaths   = set(df_ann.loc[df_ann['annotated_label'] == 'death',
                                     'unique_track_id'])
    manual_nondeath = set(df_ann.loc[df_ann['annotated_label'].isin(['division', 'alive']),
                                     'unique_track_id'])
    all_manual      = manual_deaths | manual_nondeath

    # ── Auto-label alive: tracks reaching t_max ───────────────────────────────
    track_t_last = df_meas.groupby('unique_track_id')['timepoint'].max()
    auto_alive_pool = track_t_last[
        (track_t_last >= t_max) &          # must reach the final frame exactly
        (~track_t_last.index.isin(all_manual))
    ].index.tolist()
    rng = np.random.default_rng(random_seed)
    auto_alive = list(rng.choice(auto_alive_pool,
                                  size=min(alive_sample, len(auto_alive_pool)),
                                  replace=False))

    # ── Auto-label division: topology → 2 normal-sized children ──────────────
    auto_div_pool = []
    if 'n_nearby_starters' in feat.columns and 'child_area_frac' in feat.columns:
        div_cands = feat[
            (feat['n_nearby_starters'] == 2) &
            (feat['child_area_frac'] >= 0.30) &
            (feat['child_area_frac'] <= 0.75) &
            (feat['ends_before_end'] == 1) &
            (~feat.index.isin(all_manual))
        ]
        auto_div_pool = div_cands.index.tolist()
    auto_div = list(rng.choice(auto_div_pool,
                                size=min(division_sample, len(auto_div_pool)),
                                replace=False))

    # ── Assemble labelled dataset ─────────────────────────────────────────────
    label_map = {}
    for tid in manual_deaths:
        label_map[tid] = 1
    for tid in manual_nondeath:
        label_map[tid] = 0
    for tid in auto_alive:
        label_map[tid] = 0
    for tid in auto_div:
        label_map[tid] = 0

    labelled = feat.loc[feat.index.isin(label_map)].copy()
    labelled['label'] = labelled.index.map(label_map)
    labelled = labelled.dropna(subset=['label'])

    n_death = (labelled['label'] == 1).sum()
    n_alive = (labelled['label'] == 0).sum()
    print(f"  Training set: {n_death} deaths, {n_alive} non-deaths "
          f"({len(manual_deaths)} manual, {len(auto_alive)} auto-alive, "
          f"{len(auto_div)} auto-division)")

    if n_death < 5:
        raise ValueError(
            f"Only {n_death} death examples — annotate more cells before training.")
    if n_alive < 5:
        raise ValueError(
            f"Only {n_alive} non-death examples — run on a larger dataset or annotate more.")

    X = labelled[FEATURE_COLS].values
    y = labelled['label'].values.astype(int)

    # ── Train ─────────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1.0)),
        ('clf',     RandomForestClassifier(
                        n_estimators=300,
                        max_features='sqrt',
                        class_weight='balanced',
                        random_state=random_seed,
                        n_jobs=-1)),
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(5, n_death), shuffle=True,
                         random_state=random_seed)
    cv_res = cross_validate(pipeline, X, y, cv=cv,
                            scoring=['roc_auc', 'balanced_accuracy'],
                            return_train_score=False)
    print(f"  5-fold CV  ROC-AUC: {cv_res['test_roc_auc'].mean():.3f} "
          f"± {cv_res['test_roc_auc'].std():.3f}   "
          f"balanced-acc: {cv_res['test_balanced_accuracy'].mean():.3f} "
          f"± {cv_res['test_balanced_accuracy'].std():.3f}")

    # Final fit on all data
    pipeline.fit(X, y)

    # Feature importances
    importances = pipeline.named_steps['clf'].feature_importances_
    fi = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    print("  Top features:")
    for fname, fimp in fi[:10]:
        bar = '█' * int(fimp * 40)
        print(f"    {fname:<22}  {fimp:.3f}  {bar}")

    # Save model
    import joblib
    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'feature_cols': FEATURE_COLS},
                model_output_path)
    print(f"  Model saved → {model_output_path}")

    return pipeline, FEATURE_COLS, cv_res


def apply_death_classifier(df_act, df_meas, model_path,
                           death_prob_threshold=0.5,
                           min_track_length=15):
    """
    Apply a trained death classifier to all tracks.

    For each track predicted as a death (probability ≥ death_prob_threshold):
        • The death timepoint is estimated as the last timepoint of the track
          (the nucleus became undetectable at track end, which is ≈ the death
          event).  A more precise onset can be obtained once bfp_cv data is
          available.
        • death_type is set to 'classifier_death'.

    Adds to df_act
    --------------
    probable_death_timepoint  – float (NaN = survived / not detected)
    death_type                – 'classifier_death' | NaN
    death_probability         – float [0, 1]  (probability of death class)
    frames_post_activation_death – float (NaN if not activated or no death)
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required.  Run: pip install joblib")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Run with --train-death-classifier first.")

    saved      = joblib.load(model_path)
    pipeline   = saved['pipeline']
    feat_cols  = saved['feature_cols']

    print(f"\n  Applying death classifier: {model_path}")
    print("  Extracting features …")
    feat = extract_death_features(df_meas, min_track_length=min_track_length)

    # Predict on all tracks that appear in df_act
    act_tids  = set(df_act['unique_track_id'].unique())
    feat_sub  = feat.loc[feat.index.isin(act_tids)].copy()

    # Align columns: model may have been trained with a different feature set
    for col in feat_cols:
        if col not in feat_sub.columns:
            feat_sub[col] = np.nan
    X = feat_sub[feat_cols].values

    proba  = pipeline.predict_proba(X)[:, 1]   # P(death)
    preds  = (proba >= death_prob_threshold).astype(int)

    n_pred_dead  = preds.sum()
    n_total      = len(feat_sub)
    print(f"  Predicted dead: {n_pred_dead}/{n_total} tracks "
          f"({100*n_pred_dead/n_total:.1f}%) at threshold {death_prob_threshold}")

    # Map probabilities and predictions back to df_act
    proba_series = pd.Series(proba, index=feat_sub.index, name='death_probability')
    death_mask   = pd.Series(preds.astype(bool), index=feat_sub.index)

    # Last timepoint per track (used as death timepoint estimate)
    t_last_map = df_meas.groupby('unique_track_id')['timepoint'].max()

    df = df_act.copy()
    df['probable_death_timepoint']     = np.nan
    df['death_type']                   = np.nan
    df['death_probability']            = np.nan
    df['frames_post_activation_death'] = np.nan

    for tid in feat_sub.index:
        mask = df['unique_track_id'] == tid
        if not mask.any():
            continue
        p = float(proba_series.loc[tid])
        df.loc[mask, 'death_probability'] = p
        if death_mask.loc[tid]:
            t_death = float(t_last_map.get(tid, np.nan))
            df.loc[mask, 'probable_death_timepoint'] = t_death
            df.loc[mask, 'death_type'] = 'classifier_death'
            if 'activation_timepoint' in df.columns:
                act_tp = df.loc[mask, 'activation_timepoint']
                if act_tp.notna().any():
                    df.loc[mask, 'frames_post_activation_death'] = (
                        t_death - act_tp.values[0])

    type_counts = df['death_type'].value_counts().to_dict()
    print(f"  Death detection complete — types: {type_counts}")
    return df


def compute_motility(df_act, df_meas):
    """
    Compute per-track motility metrics from centroid positions in df_meas.

    Columns added to df_act:
      mean_speed            – mean per-frame Euclidean displacement (pixels / frame)
      net_displacement      – straight-line distance first → last centroid
      total_path_length     – cumulative sum of per-frame displacements
      straightness          – net_displacement / total_path_length  (1=straight line, ~0=diffusive)
      pre_activation_speed  – mean speed before activation_timepoint (activated cells only)
      post_activation_speed – mean speed after  activation_timepoint (activated cells only)
    """
    if 'centroid-0' not in df_meas.columns or 'centroid-1' not in df_meas.columns:
        print("  Warning: centroid columns not found in df_meas — skipping motility computation")
        return df_act

    pos = (df_meas[['unique_track_id', 'timepoint', 'centroid-0', 'centroid-1']]
           .sort_values(['unique_track_id', 'timepoint'])
           .copy())

    # Vectorised per-frame displacement
    pos['dy']   = pos.groupby('unique_track_id')['centroid-0'].diff()
    pos['dx']   = pos.groupby('unique_track_id')['centroid-1'].diff()
    pos['step'] = np.sqrt(pos['dy'] ** 2 + pos['dx'] ** 2)

    # Aggregate per track (diff() leaves NaN for first row of each group — mean/sum skip it)
    agg = pos.groupby('unique_track_id')['step'].agg(
        mean_speed='mean',
        total_path_length='sum'
    ).reset_index()

    # Net displacement: Euclidean distance between first and last centroid
    first_pos = pos.groupby('unique_track_id')[['centroid-0', 'centroid-1']].first()
    last_pos  = pos.groupby('unique_track_id')[['centroid-0', 'centroid-1']].last()
    net_disp  = np.sqrt(
        (last_pos['centroid-0'] - first_pos['centroid-0']) ** 2 +
        (last_pos['centroid-1'] - first_pos['centroid-1']) ** 2
    ).rename('net_displacement').reset_index()

    agg = agg.merge(net_disp, on='unique_track_id', how='left')
    agg['straightness'] = (agg['net_displacement'] / agg['total_path_length']).clip(0, 1)

    df_act = df_act.merge(agg, on='unique_track_id', how='left')

    # Pre- / post-activation speed for activated cells
    if 'activation_timepoint' in df_act.columns:
        act_map = df_act.set_index('unique_track_id')[['activation_timepoint', 'activates']]
        pos = pos.join(act_map, on='unique_track_id')

        # step[i] = movement from prev timepoint → current timepoint
        pos['prev_t'] = pos.groupby('unique_track_id')['timepoint'].shift(1)

        activated_pos = pos[pos['activates'] == True].copy()

        pre_speed = (
            activated_pos[activated_pos['prev_t'] < activated_pos['activation_timepoint']]
            .groupby('unique_track_id')['step'].mean()
            .rename('pre_activation_speed')
        )
        post_speed = (
            activated_pos[activated_pos['prev_t'] >= activated_pos['activation_timepoint']]
            .groupby('unique_track_id')['step'].mean()
            .rename('post_activation_speed')
        )

        df_act = df_act.join(pre_speed,  on='unique_track_id')
        df_act = df_act.join(post_speed, on='unique_track_id')
    else:
        df_act['pre_activation_speed']  = np.nan
        df_act['post_activation_speed'] = np.nan

    n_ok = df_act['mean_speed'].notna().sum()
    print(f"  Motility computed for {n_ok} / {len(df_act)} tracks")
    return df_act


def compute_survival_hazard(df_all, df_meas, timepoint_max=None, smooth_window=3):
    """
    Kaplan-Meier survival curve and discrete hazard function.

    Events   : activating cells  → event_time = activation_timepoint
    Censored : non-activating    → censored at last observed timepoint

    Returns a DataFrame with one row per timepoint containing:
        n_risk, n_events, n_censored, survival, survival_lower, survival_upper,
        hazard (discrete), hazard_smooth (rolling mean)
    """
    last_t  = df_meas.groupby('unique_track_id')['timepoint'].max()
    entry_t = df_meas.groupby('unique_track_id')['timepoint'].min()

    rows = []
    for _, row in df_all.iterrows():
        tid = row['unique_track_id']
        t_entry = entry_t.get(tid, 0)
        if row.get('activates', False) and not pd.isna(row.get('activation_timepoint')):
            rows.append({'entry': t_entry, 'time': row['activation_timepoint'], 'event': 1})
        else:
            rows.append({'entry': t_entry, 'time': last_t.get(tid, t_entry), 'event': 0})

    df_ev = pd.DataFrame(rows).dropna()
    t_max = int(timepoint_max if timepoint_max is not None else df_ev['time'].max())

    results = []
    S = 1.0
    var_sum = 0.0

    for t in range(0, t_max + 1):
        n_risk     = int(((df_ev['entry'] <= t) & (df_ev['time'] >= t)).sum())
        n_events   = int(((df_ev['time'] == t) & (df_ev['event'] == 1)).sum())
        n_censored = int(((df_ev['time'] == t) & (df_ev['event'] == 0)).sum())

        if n_risk > 0 and n_events > 0:
            h_t = n_events / n_risk
            S  *= (1 - h_t)
            denom = n_risk * (n_risk - n_events)
            if denom > 0:
                var_sum += n_events / denom
        else:
            h_t = 0.0

        se = S * np.sqrt(var_sum)
        results.append({
            'timepoint':       t,
            'n_risk':          n_risk,
            'n_events':        n_events,
            'n_censored':      n_censored,
            'survival':        S,
            'survival_lower':  max(0.0, S - 1.96 * se),
            'survival_upper':  min(1.0, S + 1.96 * se),
            'hazard':          h_t,
        })

    df_surv = pd.DataFrame(results)
    df_surv['hazard_smooth'] = (
        df_surv['hazard']
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
    )
    print(f"  Survival analysis: {int(df_ev['event'].sum())} events, "
          f"{int((df_ev['event']==0).sum())} censored, "
          f"final S(t) = {S*100:.1f}%")
    return df_surv


# ==================== PLOTTING FUNCTIONS ====================
def plot_activation_overview(df_act, df_meas, output_dir, well, threshold=40, save_pdf=False, save_svg=False, suffix="", timepoint_min=None, timepoint_max=None, signal_col='mean_intensity', df_all=None):
    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    _ycol = signal_col if signal_col in df_meas.columns else 'mean_intensity'
    _ylabel = 'mNG/BFP ratio' if _ycol == 'mng_bfp_ratio' else 'Mean mNG Intensity'
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # df_all contains all tracked cells (activating + non-activating) for correct totals
    _df_ref = df_all if df_all is not None else df_act
    df_activating = df_act[df_act['activates'] == True]
    df_non_activating = _df_ref[_df_ref['activates'] == False]
    n_activating, n_total = len(df_activating), len(_df_ref)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'Activation Overview — Well {full_name}{suffix}', fontsize=14, fontweight='bold', y=0.98)
    
    activation_times = df_activating['activation_timepoint'].dropna()
    
    # Panel A: Histogram
    ax = axes[0, 0]
    if len(activation_times) > 0:
        median_t = activation_times.median()
        ax.hist(activation_times, bins=np.arange(t_min, t_max + 1, 1), color='#4DAF4A', edgecolor='white', alpha=0.8)
        ax.axvline(median_t, color='#E31A1C', linestyle='--', linewidth=2, label=f'Median: {median_t:.1f}')
        ax.legend(loc='upper right')
    ax.set_xlabel('Activation Timepoint')
    ax.set_ylabel('Count')
    ax.set_title(f'A   Activation Times (n={n_activating})', loc='left', fontweight='bold')
    
    # Panel B: Cumulative
    ax = axes[0, 1]
    t50 = None
    if len(activation_times) > 0:
        timepoints = np.arange(t_min, t_max + 1)
        cumulative_pct = [(activation_times <= t).sum() / n_total * 100 for t in timepoints]
        ax.plot(timepoints, cumulative_pct, color='#2166AC', linewidth=2.5)
        ax.fill_between(timepoints, 0, cumulative_pct, color='#2166AC', alpha=0.1)
        
        final_pct = cumulative_pct[-1]
        half_final = final_pct / 2
        for i, pct in enumerate(cumulative_pct):
            if pct >= half_final:
                t50 = timepoints[i-1] + (half_final - cumulative_pct[i-1]) / (pct - cumulative_pct[i-1]) if i > 0 else timepoints[i]
                break
        if t50:
            ax.axvline(t50, color='#E31A1C', linestyle='--', linewidth=2, label=f't50 = {t50:.1f}')
            ax.legend(loc='lower right')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Cumulative % Activated')
    ax.set_ylim(0, 100)
    ax.set_title('B   Cumulative Activation', loc='left', fontweight='bold')
    
    # Panel C: Example trajectories
    ax = axes[0, 2]
    if len(df_activating) > 0:
        for _, row in df_activating.sample(min(10, len(df_activating))).iterrows():
            data = df_meas[df_meas['unique_track_id'] == row['unique_track_id']].sort_values('timepoint')
            if len(data) > 0 and _ycol in data.columns:
                ax.plot(data['timepoint'], data[_ycol], color='#4DAF4A', alpha=0.6, linewidth=1.2)
    if len(df_non_activating) > 0:
        for _, row in df_non_activating.sample(min(5, len(df_non_activating))).iterrows():
            data = df_meas[df_meas['unique_track_id'] == row['unique_track_id']].sort_values('timepoint')
            if len(data) > 0 and _ycol in data.columns:
                ax.plot(data['timepoint'], data[_ycol], color='gray', alpha=0.5, linewidth=1)
    if _ycol == 'mean_intensity':
        ax.axhline(threshold, color='#E31A1C', linestyle='--', linewidth=1.5, label=f'Threshold={threshold}')
        ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel(_ylabel)
    ax.set_title('C   Example Trajectories', loc='left', fontweight='bold')
    ax.set_xlim(t_min, t_max)

    # Panel D: Max intensity
    ax = axes[1, 0]
    if 'max_intensity' in df_act.columns:
        max_act = df_activating['max_intensity'].dropna()
        max_non = df_non_activating['max_intensity'].dropna()
        bins = np.linspace(0, max(max_act.max() if len(max_act) > 0 else 100, max_non.max() if len(max_non) > 0 else 100), 40)
        if len(max_act) > 0:
            ax.hist(max_act, bins=bins, alpha=0.7, color='#4DAF4A', label='Activating', edgecolor='white')
        if len(max_non) > 0:
            ax.hist(max_non, bins=bins, alpha=0.7, color='gray', label='Non-activating', edgecolor='white')
        ax.axvline(threshold, color='#E31A1C', linestyle='--', linewidth=1.5)
        ax.legend(loc='upper right')
    ax.set_xlabel('Max mNG Intensity')
    ax.set_ylabel('Count')
    ax.set_title('D   Max Intensity Distribution', loc='left', fontweight='bold')
    
    # Panel E: By FOV
    ax = axes[1, 1]
    if 'fov' in _df_ref.columns:
        fov_stats = _df_ref.groupby('fov').agg({'activates': ['sum', 'count']})
        fov_stats.columns = ['n_act', 'n_total']
        fov_stats['pct'] = fov_stats['n_act'] / fov_stats['n_total'] * 100
        ax.bar(fov_stats.index, fov_stats['pct'], color='#1f77b4', edgecolor='black', alpha=0.8)
        ax.axhline(fov_stats['pct'].mean(), color='#E31A1C', linestyle='--', linewidth=1.5)
    ax.set_xlabel('FOV')
    ax.set_ylabel('% Activating')
    ax.set_title('E   Activation by FOV', loc='left', fontweight='bold')
    
    # Panel F: Population mean
    ax = axes[1, 2]
    if len(df_activating) > 0:
        act_data = df_meas[df_meas['unique_track_id'].isin(df_activating['unique_track_id'])]
        if _ycol in act_data.columns:
            mean_traj = act_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
            ax.plot(mean_traj.index, mean_traj['mean'], color='#4DAF4A', linewidth=2.5, label='Activating')
            ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'], mean_traj['mean'] + mean_traj['std'],
                           color='#4DAF4A', alpha=0.2)
    if len(df_non_activating) > 0:
        non_data = df_meas[df_meas['unique_track_id'].isin(df_non_activating['unique_track_id'])]
        if _ycol in non_data.columns:
            mean_traj = non_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
            ax.plot(mean_traj.index, mean_traj['mean'], color='gray', linewidth=2, label='Non-activating')
            ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'], mean_traj['mean'] + mean_traj['std'],
                           color='gray', alpha=0.2)
    if _ycol == 'mean_intensity':
        ax.axhline(threshold, color='#E31A1C', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel(_ylabel)
    ax.set_title('F   Population Mean ± SD', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(t_min, t_max)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_activation_overview" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "")
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()

    return t50


def plot_activation_wave(df_act, output_dir, well, save_pdf=False, save_svg=False,
                         suffix="", timepoint_min=None, timepoint_max=None):
    """
    Population-level activation kinetics: onset, T50, and plateau.

    Panel A — Cumulative activation curve (% of all cells) with sigmoid fit and
              three landmark timepoints marked:
                T_onset  (10 % of activating cells reached)
                T_50     (50 % reached — inflection of sigmoid)
                T_plateau(90 % reached)
    Panel B — Instantaneous activation rate (new activations / timepoint, smoothed)
              with peak rate marked.
    Panel C — Cumulative curves split by response group (low / medium / high).
    Panel D — Summary statistics table.
    """
    from scipy.optimize import curve_fit

    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    df_act_all = df_act                                        # all tracked cells
    df_act_on  = df_act[df_act['activates'] == True]
    n_all      = len(df_act_all)
    n_act      = len(df_act_on)

    if n_act == 0:
        print("  Skipping activation wave plot: no activating cells")
        return None

    act_times = df_act_on['activation_timepoint'].dropna().values
    timepoints = np.arange(t_min, t_max + 1)

    # Cumulative counts (% of all cells)
    cum_pct_all  = np.array([(act_times <= t).sum() / n_all * 100 for t in timepoints])
    # Cumulative counts (% of activating cells only — pure timing curve)
    cum_pct_act  = np.array([(act_times <= t).sum() / n_act * 100 for t in timepoints])

    # ── Sigmoid fit to cumulative count of activating cells ──────────────────
    def _logistic(t, L, k, t0):
        return L / (1.0 + np.exp(-k * (t - t0)))

    fit_ok = False
    try:
        popt, _ = curve_fit(_logistic, timepoints, cum_pct_act,
                            p0=[100.0, 0.3, float(np.median(act_times))],
                            bounds=([0, 0, t_min], [200, 5, t_max]),
                            maxfev=8000)
        L_fit, k_fit, t0_fit = popt
        fit_curve = _logistic(timepoints, *popt)
        # Landmark definitions from logistic parameters:
        #   p % of L  →  t = t0 - ln((1-p/100)/(p/100)) / k
        def _t_at_pct(pct):
            return t0_fit - np.log((100 - pct) / pct) / k_fit
        t5_fit        = _t_at_pct(5)
        t_onset_fit   = _t_at_pct(10)
        t50_fit       = t0_fit          # exact midpoint
        t_plateau_fit = _t_at_pct(90)
        t95_fit       = _t_at_pct(95)
        fit_ok = True
    except Exception:
        fit_curve = None

    # ── Empirical landmarks (percentiles of activation times) ────────────────
    t5_emp        = float(np.percentile(act_times,  5))
    t_onset_emp   = float(np.percentile(act_times, 10))
    t50_emp       = float(np.percentile(act_times, 50))
    t_plateau_emp = float(np.percentile(act_times, 90))
    t95_emp       = float(np.percentile(act_times, 95))

    # Choose which landmarks to display (fit preferred when available)
    t5        = t5_fit        if fit_ok else t5_emp
    t_onset   = t_onset_fit   if fit_ok else t_onset_emp
    t50       = t50_fit       if fit_ok else t50_emp
    t_plateau = t_plateau_fit if fit_ok else t_plateau_emp
    t95       = t95_fit       if fit_ok else t95_emp

    # ── Instantaneous rate (new activations per timepoint) ───────────────────
    rate = np.array([(act_times == t).sum() for t in timepoints], dtype=float)
    # Gaussian smooth (σ = 1.5 frames)
    from scipy.ndimage import gaussian_filter1d
    rate_smooth = gaussian_filter1d(rate, sigma=1.5)
    peak_t = timepoints[np.argmax(rate_smooth)]
    peak_r = rate_smooth.max()

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f'Activation Wave Kinetics — Well {full_name}{(" — " + suffix) if suffix else ""}',
        fontsize=14, fontweight='bold', y=0.98)

    # 5 landmarks: T5, T10(onset), T50, T90(plateau), T95
    # colour ramp from light orange → red → light blue
    landmark_colors = {
        't5':      '#FDAE61',
        'onset':   '#F4A582',
        't50':     '#E31A1C',
        'plateau': '#92C5DE',
        't95':     '#4393C3',
    }
    landmarks = [
        ('T₅',       t5,        landmark_colors['t5'],      '5 %'),
        ('T₁₀',      t_onset,   landmark_colors['onset'],  '10 %'),
        ('T₅₀',      t50,       landmark_colors['t50'],    '50 %'),
        ('T₉₀',      t_plateau, landmark_colors['plateau'],'90 %'),
        ('T₉₅',      t95,       landmark_colors['t95'],    '95 %'),
    ]

    # ── Panel A: Cumulative activation (% of ALL cells) with landmarks ───────
    ax = axes[0, 0]
    ax.plot(timepoints, cum_pct_all, color='#2166AC', linewidth=2.5,
            label=f'Observed  (n={n_act}/{n_all})')
    ax.fill_between(timepoints, 0, cum_pct_all, color='#2166AC', alpha=0.08)

    if fit_ok:
        fit_curve_all = fit_curve * (n_act / n_all)
        ax.plot(timepoints, fit_curve_all, color='#E31A1C', linewidth=1.8,
                linestyle='--', alpha=0.8, label='Sigmoid fit')

    for lbl, t_val, col, pct_str in landmarks:
        if t_min <= t_val <= t_max:
            y_val = (act_times <= t_val).sum() / n_all * 100
            ax.axvline(t_val, color=col, linestyle=':', linewidth=2.0,
                       label=f'{lbl} = {t_val:.1f}  ({pct_str})')
            ax.axhline(y_val, color=col, linestyle=':', linewidth=0.8, alpha=0.45)
            ax.scatter([t_val], [y_val], color=col, s=55, zorder=5)

    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Cumulative activated (% of all cells)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, max(cum_pct_all.max() * 1.15, 5))
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_title('A   Cumulative Activation', loc='left', fontweight='bold')

    # ── Panel B: Instantaneous activation rate ────────────────────────────────
    ax = axes[0, 1]
    ax.bar(timepoints, rate, color='#74C476', alpha=0.5, width=0.8,
           label='New activations / frame')
    ax.plot(timepoints, rate_smooth, color='#238B45', linewidth=2.5,
            label='Smoothed (σ=1.5)')
    ax.axvline(peak_t, color='#E31A1C', linestyle='--', linewidth=2.0,
               label=f'Peak = t{peak_t:.0f}  ({peak_r:.1f} cells/frame)')
    # Shade 5–95 % zone
    ax.axvspan(t5, t95, alpha=0.07, color='#2166AC',
               label=f'5–95 % zone ({t95 - t5:.1f} frames)')
    # Shade 10–90 % inner zone slightly darker
    ax.axvspan(t_onset, t_plateau, alpha=0.07, color='#2166AC',
               label=f'10–90 % zone ({t_plateau - t_onset:.1f} frames)')
    for lbl, t_val, col, _ in landmarks:
        if t_min <= t_val <= t_max:
            ax.axvline(t_val, color=col, linestyle=':', linewidth=1.5)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('New activations / timepoint')
    ax.set_xlim(t_min, t_max)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_title('B   Instantaneous Activation Rate', loc='left', fontweight='bold')

    # ── Panel C: Cumulative by response group ─────────────────────────────────
    ax = axes[1, 0]
    r_groups = ['low', 'medium', 'high']
    r_labels = ['Low', 'Medium', 'High']
    group_t50s = {}
    for g, lbl in zip(r_groups, r_labels):
        sub = df_act_on[df_act_on['response_group'] == g] \
            if 'response_group' in df_act_on.columns else pd.DataFrame()
        if len(sub) < 3:
            continue
        t_sub = sub['activation_timepoint'].dropna().values
        cum_g  = np.array([(t_sub <= t).sum() / len(t_sub) * 100 for t in timepoints])
        ax.plot(timepoints, cum_g, color=RESPONSE_COLORS[g], linewidth=2.2,
                label=f'{lbl} (n={len(t_sub)})')
        # Mark T50 per group
        t50_g = float(np.percentile(t_sub, 50))
        y50_g = (t_sub <= t50_g).sum() / len(t_sub) * 100
        ax.scatter([t50_g], [y50_g], color=RESPONSE_COLORS[g], s=60,
                   zorder=5, marker='D')
        group_t50s[lbl] = t50_g

    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Cumulative activated (% within group)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('C   Cumulative by Response Group  (◆ = T₅₀)', loc='left', fontweight='bold')

    # ── Panel D: Summary statistics ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')

    lines = [
        "Population Activation Kinetics",
        "=" * 32,
        f"  Total cells tracked :  {n_all}",
        f"  Cells activating    :  {n_act}  ({n_act/n_all*100:.1f} %)",
        "",
        "Landmark timepoints",
        "-" * 32,
        f"  T₅        ( 5 %)  :  {t5:.1f}",
        f"  T₁₀       (10 %)  :  {t_onset:.1f}",
        f"  T₅₀       (50 %)  :  {t50:.1f}",
        f"  T₉₀       (90 %)  :  {t_plateau:.1f}",
        f"  T₉₅       (95 %)  :  {t95:.1f}",
        f"  5–95 % span      :  {t95 - t5:.1f} frames",
        f"  10–90 % span     :  {t_plateau - t_onset:.1f} frames",
        "",
        "Activation rate",
        "-" * 32,
        f"  Peak rate        :  {peak_r:.2f} cells/frame",
        f"  Peak timepoint   :  {peak_t:.0f}",
    ]
    if fit_ok:
        lines += [
            "",
            "Sigmoid fit  (% activating)",
            "-" * 32,
            f"  Plateau (L)      :  {L_fit:.1f} %",
            f"  Steepness (k)    :  {k_fit:.4f}",
            f"  Midpoint (t₀)    :  {t0_fit:.1f}",
        ]
    if group_t50s:
        lines += ["", "T₅₀ by response group", "-" * 32]
        for lbl, v in group_t50s.items():
            lines.append(f"  {lbl:<10}  :  {v:.1f}")
    lines += [
        "",
        "Method: Bonferroni-corrected definitions",
        f"  T_onset / T_plateau = {'' if fit_ok else 'empirical '}10th / 90th pct",
    ]

    ax.text(0.04, 0.97, '\n'.join(lines),
            transform=ax.transAxes, fontsize=8.5, fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff',
                      edgecolor='#4682b4', alpha=0.9))
    ax.set_title('D   Summary', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_activation_wave{suffix_clean}"
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()
    print("  Saved activation wave kinetics plot")

    return {'t5': t5, 't_onset': t_onset, 't50': t50,
            't_plateau': t_plateau, 't95': t95,
            'peak_rate': peak_r, 'peak_t': peak_t,
            'fit_ok': fit_ok,
            'sigmoid_L': L_fit if fit_ok else None,
            'sigmoid_k': k_fit if fit_ok else None,
            'sigmoid_t0': t0_fit if fit_ok else None}


def plot_group_trajectories(df_act, df_meas, output_dir, well, early_min, early_max,
                            average_min, average_max, late_min, save_pdf=False, save_svg=False, suffix="", timepoint_min=None, timepoint_max=None, signal_col='mean_intensity'):
    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    _ycol = signal_col if signal_col in df_meas.columns else 'mean_intensity'
    _ylabel = 'mNG/BFP ratio' if _ycol == 'mng_bfp_ratio' else 'mNG intensity (a.u.)'
    groups = ['early', 'average', 'late']
    group_labels = ['Early', 'Average', 'Late']
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Activation Group Trajectories — Well {full_name}{suffix}', fontsize=14, fontweight='bold', y=0.98)
    
    # Identify good-fit cells (sigmoid_r2 >= 0.5) for use in mean trajectory.
    # Cells with poor fits (noise spikes mis-classified as early activators, etc.)
    # are still plotted as individual traces but in grey and excluded from the mean.
    has_r2 = 'sigmoid_r2' in df_act.columns
    good_fit_ids = set(
        df_act.loc[df_act['sigmoid_r2'] >= 0.5, 'unique_track_id']
        if has_r2 else df_act['unique_track_id']
    )

    # Top row: Individual trajectories per group
    for idx, group in enumerate(groups):
        ax = axes[0, idx]
        df_grp     = df_act[df_act['activation_group'] == group]
        track_ids  = df_grp['unique_track_id'].values
        good_ids   = [t for t in track_ids if t in good_fit_ids]
        poor_ids   = [t for t in track_ids if t not in good_fit_ids]

        # poor-fit cells: grey, thin, behind everything
        for tid in poor_ids[:50]:
            data = df_meas[df_meas['unique_track_id'] == tid].sort_values('timepoint')
            if len(data) > 0 and _ycol in data.columns:
                ax.plot(data['timepoint'], data[_ycol], alpha=0.3, color='grey',
                        linewidth=0.6, zorder=1)

        # good-fit cells: group colour
        for tid in good_ids[:50]:
            data = df_meas[df_meas['unique_track_id'] == tid].sort_values('timepoint')
            if len(data) > 0 and _ycol in data.columns:
                ax.plot(data['timepoint'], data[_ycol], alpha=0.25, color=COLORS[group],
                        linewidth=0.8, zorder=2)

        # mean ± SD from good-fit cells only
        good_data = df_meas[df_meas['unique_track_id'].isin(good_ids)]
        if len(good_data) > 0 and _ycol in good_data.columns:
            mean_traj = good_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
            ax.plot(mean_traj.index, mean_traj['mean'], color='black', linewidth=2.5,
                    label=f'Mean (n={len(good_ids)})', zorder=3)
            ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                            mean_traj['mean'] + mean_traj['std'], alpha=0.25, color='black',
                            zorder=2)

        title_str = f'{chr(65+idx)}   {group_labels[idx]} (n={len(track_ids)}'
        if poor_ids:
            title_str += f', {len(poor_ids)} poor-fit in grey'
        title_str += ')'
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel(_ylabel)
        ax.set_title(title_str, loc='left', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(t_min, t_max)

    # Panel D: All groups overlaid — mean ± SD from good-fit cells only
    ax = axes[1, 0]
    for group in groups:
        df_grp   = df_act[df_act['activation_group'] == group]
        good_ids = [t for t in df_grp['unique_track_id'].values if t in good_fit_ids]
        good_data = df_meas[df_meas['unique_track_id'].isin(good_ids)]
        if len(good_data) > 0 and _ycol in good_data.columns:
            mean_traj = good_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
            ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2,
                    label=f'{group.capitalize()} (n={len(good_ids)})')
            ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                            mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel(_ylabel)
    ax.set_title('D   Group comparison (good-fit cells)', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(t_min, t_max)

    # Panel E: Activation time distribution
    ax = axes[1, 1]
    bins = np.arange(t_min, t_max + 2, 2)
    for group in groups:
        group_data = df_act[df_act['activation_group'] == group]['activation_timepoint']
        ax.hist(group_data, bins=bins, alpha=0.6, color=COLORS[group], label=group.capitalize(), edgecolor='white')
    ax.set_xlabel('Activation time (frames)')
    ax.set_ylabel('Count')
    ax.set_title('E   Activation time distribution', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # Panel F: Per-cell normalized mean trajectories (0 = baseline, 1 = plateau)
    # Normalization per cell: norm(t) = (signal(t) - cell_baseline) / cell_amplitude
    #   cell_baseline = sigmoid_baseline if fit available, else mean of first 5 frames
    #   cell_amplitude = sigmoid_amplitude if fit available, else (max_signal - baseline)
    # This removes cell-to-cell variability in absolute levels so timing group
    # separation becomes clearly visible — unlike the raw overlay in Panel D.
    ax = axes[1, 2]
    has_sigmoid = ('sigmoid_baseline' in df_act.columns and 'sigmoid_amplitude' in df_act.columns
                   and 'sigmoid_r2' in df_act.columns)
    for group in groups:
        df_grp = df_act[df_act['activation_group'] == group]
        norm_trajs = []
        for _, row in df_grp.iterrows():
            tid  = row['unique_track_id']
            cell = df_meas[df_meas['unique_track_id'] == tid].sort_values('timepoint')
            if len(cell) == 0 or _ycol not in cell.columns:
                continue
            sig = cell[_ycol].values
            tp  = cell['timepoint'].values

            # Determine per-cell baseline and amplitude
            if has_sigmoid and row.get('sigmoid_r2', 0) >= 0.5:
                bl   = row['sigmoid_baseline']
                amp  = row['sigmoid_amplitude']
            else:
                # Fallback: mean of first 5 frames as baseline
                early_mask = tp <= (tp[0] + 4)
                bl  = sig[early_mask].mean() if early_mask.any() else sig[0]
                amp = sig.max() - bl

            if amp <= 0:
                continue

            norm_sig = (sig - bl) / amp
            norm_trajs.append(pd.Series(norm_sig, index=tp))

        if not norm_trajs:
            continue
        pivot     = pd.concat(norm_trajs, axis=1)
        mean_norm = pivot.mean(axis=1)
        std_norm  = pivot.std(axis=1)
        tp_idx    = mean_norm.index.values
        ax.plot(tp_idx, mean_norm.values, color=COLORS[group], linewidth=2,
                label=f'{group.capitalize()} (n={len(norm_trajs)})')
        ax.fill_between(tp_idx, (mean_norm - std_norm).values, (mean_norm + std_norm).values,
                        alpha=0.15, color=COLORS[group])

    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax.axhline(1, color='grey', linewidth=0.7, linestyle='--')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Normalised signal  (0 = baseline, 1 = plateau)')
    ax.set_title('F   Group comparison (normalised)', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(t_min, t_max)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_trajectories_panel" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "")
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()


def plot_response_groups(df_act, df_meas, output_dir, well, save_pdf=False, save_svg=False,
                         suffix="", timepoint_min=None, timepoint_max=None, signal_col='mean_intensity'):
    """
    Overview figure for response-amplitude groups (low / medium / high).

    Panel A — plateau value distribution with group boundaries
    Panel B — plateau value by response_group (box + scatter)
    Panel C — activation timepoint by response_group (box + scatter)
    Panel D — activation timepoint vs plateau value, colored by response_group
    Panel E — activation timepoint vs plateau value, colored by activation_group
    Panel F — cross-tabulation heatmap: activation_group vs response_group
    """
    if 'response_group' not in df_act.columns or 'plateau_value' not in df_act.columns:
        print("  Skipping response group plots: classify_by_response not run")
        return

    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    _ycol   = signal_col if signal_col in df_meas.columns else 'mean_intensity'
    _ylabel = 'mNG/BFP ratio' if _ycol == 'mng_bfp_ratio' else 'mNG intensity (a.u.)'
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    r_groups  = ['low', 'medium', 'high']
    r_labels  = ['Low', 'Medium', 'High']
    a_groups  = ['early', 'average', 'late']
    a_labels  = ['Early', 'Average', 'Late']

    df_fit = df_act[df_act['response_group'].isin(r_groups)].copy()
    thresholds = df_act.attrs.get('response_thresholds', {})

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'Response Amplitude Groups — Well {full_name}{suffix}',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel A: plateau value distribution
    ax = axes[0, 0]
    if len(df_fit) > 0:
        ax.hist(df_fit['plateau_value'].dropna(), bins=30, color='#BDBDBD', edgecolor='white', alpha=0.8)
        if 'low_threshold' in thresholds:
            ax.axvline(thresholds['low_threshold'],  color=RESPONSE_COLORS['low'],  linestyle='--', linewidth=2, label=f"Low boundary ({thresholds['low_threshold']:.2f})")
            ax.axvline(thresholds['high_threshold'], color=RESPONSE_COLORS['high'], linestyle='--', linewidth=2, label=f"High boundary ({thresholds['high_threshold']:.2f})")
            ax.legend(fontsize=8)
    ax.set_xlabel('Plateau value (sigmoid asymptote)')
    ax.set_ylabel('Count')
    ax.set_title('A   Plateau Distribution', loc='left', fontweight='bold')

    # Panel B: plateau value by response_group
    ax = axes[0, 1]
    data_b = [df_fit[df_fit['response_group'] == g]['plateau_value'].dropna().values for g in r_groups]
    bp = ax.boxplot(data_b, labels=r_labels, patch_artist=True, widths=0.6)
    for patch, g in zip(bp['boxes'], r_groups):
        patch.set_facecolor(RESPONSE_COLORS[g]); patch.set_alpha(0.7)
    for i, (g, d) in enumerate(zip(r_groups, data_b)):
        if len(d) > 0:
            ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)), d,
                       alpha=0.4, color=RESPONSE_COLORS[g], s=20)
    _pairwise_sig_brackets(ax, data_b, labels=r_labels)
    ax.set_ylabel('Plateau value')
    ax.set_title('B   Plateau by Response Group', loc='left', fontweight='bold')

    # Panel C: activation timepoint by response_group
    ax = axes[0, 2]
    data_c = [df_fit[df_fit['response_group'] == g]['activation_timepoint'].dropna().values for g in r_groups]
    bp = ax.boxplot(data_c, labels=r_labels, patch_artist=True, widths=0.6)
    for patch, g in zip(bp['boxes'], r_groups):
        patch.set_facecolor(RESPONSE_COLORS[g]); patch.set_alpha(0.7)
    for i, (g, d) in enumerate(zip(r_groups, data_c)):
        if len(d) > 0:
            ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)), d,
                       alpha=0.4, color=RESPONSE_COLORS[g], s=20)
    _pairwise_sig_brackets(ax, data_c, labels=r_labels)
    ax.set_ylabel('Activation timepoint')
    ax.set_title('C   Activation Time by Response Group', loc='left', fontweight='bold')

    # Panel D: scatter activation_timepoint vs plateau, colored by response_group
    ax = axes[1, 0]
    for g, label in zip(r_groups, r_labels):
        sub = df_fit[df_fit['response_group'] == g]
        if len(sub) > 0:
            ax.scatter(sub['activation_timepoint'], sub['plateau_value'],
                       color=RESPONSE_COLORS[g], alpha=0.6, s=25, label=label, edgecolors='none')
    ax.set_xlabel('Activation timepoint')
    ax.set_ylabel('Plateau value')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('D   Timing vs Amplitude (response groups)', loc='left', fontweight='bold')

    # Panel E: same scatter, colored by activation_group
    ax = axes[1, 1]
    for g, label in zip(a_groups, a_labels):
        sub = df_fit[df_fit['activation_group'] == g] if 'activation_group' in df_fit.columns else pd.DataFrame()
        if len(sub) > 0:
            ax.scatter(sub['activation_timepoint'], sub['plateau_value'],
                       color=COLORS[g], alpha=0.6, s=25, label=label, edgecolors='none')
    ax.set_xlabel('Activation timepoint')
    ax.set_ylabel('Plateau value')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('E   Timing vs Amplitude (timing groups)', loc='left', fontweight='bold')

    # Panel F: cross-tabulation heatmap
    ax = axes[1, 2]
    if 'activation_group' in df_fit.columns:
        ct = pd.crosstab(
            df_fit['activation_group'].where(df_fit['activation_group'].isin(a_groups)),
            df_fit['response_group'].where(df_fit['response_group'].isin(r_groups))
        ).reindex(index=a_groups, columns=r_groups, fill_value=0)
        im = ax.imshow(ct.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(r_groups))); ax.set_xticklabels(r_labels)
        ax.set_yticks(range(len(a_groups))); ax.set_yticklabels(a_labels)
        ax.set_xlabel('Response group'); ax.set_ylabel('Timing group')
        for i in range(len(a_groups)):
            for j in range(len(r_groups)):
                ax.text(j, i, ct.values[i, j], ha='center', va='center', fontsize=11,
                        color='white' if ct.values[i, j] > ct.values.max() * 0.6 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Cell count')
        try:
            _, p_chi, _, _ = stats.chi2_contingency(ct.values)
            ax.set_xlabel(f'Response group  (χ² p={p_chi:.3g})')
        except Exception:
            ax.set_xlabel('Response group')
    ax.set_title('F   Timing × Amplitude Cross-tab', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_response_groups{suffix_clean}"
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()
    print(f"  Saved response groups overview")

    # Average trajectories per response group
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, label in zip(r_groups, r_labels):
        ids = df_fit[df_fit['response_group'] == g]['unique_track_id']
        if len(ids) == 0:
            continue
        group_data = df_meas[df_meas['unique_track_id'].isin(ids)]
        if _ycol not in group_data.columns:
            continue
        traj = group_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
        ax.plot(traj.index, traj['mean'], color=RESPONSE_COLORS[g], linewidth=2.5, label=f'{label} (n={len(ids)})')
        ax.fill_between(traj.index, traj['mean'] - traj['std'], traj['mean'] + traj['std'],
                        color=RESPONSE_COLORS[g], alpha=0.15)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel(_ylabel)
    ax.set_xlim(t_min, t_max)
    ax.legend(loc='upper left')
    ax.set_title(f'Mean ± SD trajectories by response group — Well {full_name}')
    plt.tight_layout()
    save_figure(fig, fig_dir, f"well_{full_name}_response_trajectories{suffix_clean}", save_pdf, save_svg)
    plt.close()
    print(f"  Saved response group trajectories")


def plot_response_kinetics(df_act, df_meas, output_dir, well, save_pdf=False, save_svg=False,
                           suffix="", timepoint_min=None, timepoint_max=None,
                           signal_col='mean_intensity'):
    """
    Kinetic comparison across response-amplitude groups (low / medium / high).

    Panel A — max_slope by response group    (activation speed: peak rate of signal rise)
    Panel B — sigmoid_k by response group    (steepness of the fitted sigmoid)
    Panel C — rise_time by response group    (frames from onset to plateau)
    Panel D — duration_active by response group (frames signal stays > 50 % of plateau)
    Panel E — activation-aligned mean ± SD trajectories per response group
    Panel F — scatter: max_slope vs plateau_value, colored by response group
    """
    if 'response_group' not in df_act.columns:
        print("  Skipping response kinetics: classify_by_response not run")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    _ycol   = signal_col if signal_col in df_meas.columns else 'mean_intensity'
    _ylabel = 'mNG/BFP ratio' if _ycol == 'mng_bfp_ratio' else 'mNG intensity (a.u.)'
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    r_groups = ['low', 'medium', 'high']
    r_labels = ['Low', 'Medium', 'High']

    # Good-fit cells only for sigmoid-derived columns
    df_fit = df_act[df_act['response_group'].isin(r_groups)].copy()
    if 'sigmoid_r2' in df_fit.columns:
        df_fit = df_fit[df_fit['sigmoid_r2'] >= 0.7]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f'Response Group Kinetics — Well {full_name}{(" — " + suffix) if suffix else ""}',
                 fontsize=14, fontweight='bold', y=0.98)

    def _boxplot_panel(ax, col, ylabel, letter, title):
        data = [df_fit[df_fit['response_group'] == g][col].dropna().values
                if col in df_fit.columns else np.array([])
                for g in r_groups]
        if not any(len(d) > 0 for d in data):
            ax.text(0.5, 0.5, f'No {col} data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')
            return
        bp = ax.boxplot(data, labels=r_labels, patch_artist=True, widths=0.6)
        for patch, g in zip(bp['boxes'], r_groups):
            patch.set_facecolor(RESPONSE_COLORS[g])
            patch.set_alpha(0.7)
        for i, (g, d) in enumerate(zip(r_groups, data)):
            if len(d) > 0:
                ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)),
                           d, alpha=0.4, color=RESPONSE_COLORS[g], s=20)
        _pairwise_sig_brackets(ax, data, labels=r_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')

    _boxplot_panel(axes[0, 0], 'max_slope',       'Max slope (signal/frame)',   'A', 'Activation Speed')
    _boxplot_panel(axes[0, 1], 'sigmoid_k',        'Sigmoid steepness (k)',      'B', 'Sigmoid Steepness')
    _boxplot_panel(axes[0, 2], 'rise_time',        'Rise time (frames)',         'C', 'Rise Time (onset → plateau)')
    _boxplot_panel(axes[1, 0], 'duration_active',  'Frames above 50 % plateau', 'D', 'Duration Active')

    # Panel E: activation-aligned trajectories by response group (mean ± SEM)
    ax = axes[1, 1]
    act_times  = df_act.set_index('unique_track_id')['activation_timepoint'].to_dict()
    t_max_meas = int(df_meas['timepoint'].max()) if len(df_meas) > 0 else 50
    align_range = np.arange(-10, min(t_max_meas, 50))   # extend to show full plateau
    for g, label in zip(r_groups, r_labels):
        ids = df_fit[df_fit['response_group'] == g]['unique_track_id'].values
        if len(ids) == 0:
            continue
        buckets = {t: [] for t in align_range}
        for tid in ids:
            t_act = act_times.get(tid, np.nan)
            if pd.isna(t_act):
                continue
            track = df_meas[df_meas['unique_track_id'] == tid]
            if _ycol not in track.columns:
                continue
            for _, row in track.iterrows():
                t_rel = int(row['timepoint'] - t_act)
                if t_rel in buckets:
                    buckets[t_rel].append(row[_ycol])
        means, sems, valid_t = [], [], []
        for t in align_range:
            vals = np.array([v for v in buckets[t] if not np.isnan(v)])
            if len(vals) >= 3:
                means.append(vals.mean())
                sems.append(vals.std() / np.sqrt(len(vals)))
                valid_t.append(t)
        if valid_t:
            means = np.array(means)
            sems  = np.array(sems)
            ax.plot(valid_t, means, color=RESPONSE_COLORS[g], linewidth=2.5,
                    label=f'{label} (n={len(ids)})')
            ax.fill_between(valid_t, means - sems, means + sems,
                            color=RESPONSE_COLORS[g], alpha=0.2)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('Time relative to activation (frames)')
    ax.set_ylabel(_ylabel)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('E   Aligned Trajectories by Response Group (mean ± SEM)', loc='left', fontweight='bold')

    # Panel F: max_slope vs plateau_value scatter
    ax = axes[1, 2]
    if 'max_slope' in df_fit.columns and 'plateau_value' in df_fit.columns:
        for g, label in zip(r_groups, r_labels):
            sub = df_fit[df_fit['response_group'] == g].dropna(subset=['max_slope', 'plateau_value'])
            if len(sub) > 0:
                ax.scatter(sub['max_slope'], sub['plateau_value'],
                           color=RESPONSE_COLORS[g], alpha=0.6, s=25,
                           label=label, edgecolors='none')
        df_valid = df_fit.dropna(subset=['max_slope', 'plateau_value'])
        if len(df_valid) > 5:
            r, p = stats.pearsonr(df_valid['max_slope'], df_valid['plateau_value'])
            ax.text(0.03, 0.97, f'r = {r:.2f}  p = {p:.3g}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.set_xlabel('Max slope (signal/frame)')
        ax.set_ylabel('Plateau value')
        ax.legend(loc='lower right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No max_slope / plateau_value data',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_title('F   Amplitude vs Max Speed', loc='left', fontweight='bold')

    # Panel G: sigmoid_k vs plateau_value scatter
    ax = axes[2, 0]
    if 'sigmoid_k' in df_fit.columns and 'plateau_value' in df_fit.columns:
        for g, label in zip(r_groups, r_labels):
            sub = df_fit[df_fit['response_group'] == g].dropna(subset=['sigmoid_k', 'plateau_value'])
            if len(sub) > 0:
                ax.scatter(sub['sigmoid_k'], sub['plateau_value'],
                           color=RESPONSE_COLORS[g], alpha=0.6, s=25,
                           label=label, edgecolors='none')
        df_valid = df_fit.dropna(subset=['sigmoid_k', 'plateau_value'])
        if len(df_valid) > 5:
            r, p = stats.pearsonr(df_valid['sigmoid_k'], df_valid['plateau_value'])
            ax.text(0.03, 0.97, f'r = {r:.2f}  p = {p:.3g}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.set_xlabel('Sigmoid steepness k')
        ax.set_ylabel('Plateau value')
        ax.legend(loc='lower right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No sigmoid_k / plateau_value data',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_title('G   Amplitude vs Sigmoid Steepness', loc='left', fontweight='bold')

    # Panel H: activation_timepoint vs plateau_value — tests timing independence
    ax = axes[2, 1]
    if 'activation_timepoint' in df_fit.columns and 'plateau_value' in df_fit.columns:
        for g, label in zip(r_groups, r_labels):
            sub = df_fit[df_fit['response_group'] == g].dropna(
                subset=['activation_timepoint', 'plateau_value'])
            if len(sub) > 0:
                ax.scatter(sub['activation_timepoint'], sub['plateau_value'],
                           color=RESPONSE_COLORS[g], alpha=0.6, s=25,
                           label=label, edgecolors='none')
        df_valid = df_fit.dropna(subset=['activation_timepoint', 'plateau_value'])
        if len(df_valid) > 5:
            r, p = stats.pearsonr(df_valid['activation_timepoint'], df_valid['plateau_value'])
            ax.text(0.03, 0.97, f'r = {r:.2f}  p = {p:.3g}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.set_xlabel('Activation timepoint (frame)')
        ax.set_ylabel('Plateau value')
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No activation_timepoint / plateau_value data',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_title('H   Amplitude vs Timing of Infection', loc='left', fontweight='bold')

    # Panel I: activation_timepoint distribution by response group (violin)
    ax = axes[2, 2]
    if 'activation_timepoint' in df_fit.columns:
        data_i = [df_fit[df_fit['response_group'] == g]['activation_timepoint'].dropna().values
                  for g in r_groups]
        non_empty = [(d, l, g) for d, l, g in zip(data_i, r_labels, r_groups) if len(d) >= 3]
        if non_empty:
            data_ne, labels_ne, groups_ne = zip(*non_empty)
            parts = ax.violinplot(list(data_ne), positions=range(1, len(data_ne) + 1),
                                  showmedians=True, showextrema=True)
            for pc, g in zip(parts['bodies'], groups_ne):
                pc.set_facecolor(RESPONSE_COLORS[g])
                pc.set_alpha(0.6)
            for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                if part in parts:
                    parts[part].set_color('black')
                    parts[part].set_linewidth(1.2)
            ax.set_xticks(range(1, len(data_ne) + 1))
            ax.set_xticklabels(list(labels_ne))
            # Pairwise MWU with Bonferroni for violin panel
            from itertools import combinations as _comb
            _pairs = [(i, j) for i, j in _comb(range(len(data_ne)), 2)
                      if len(data_ne[i]) >= 3 and len(data_ne[j]) >= 3]
            _n_comp = len(_pairs)
            _sig_lines = []
            for _i, _j in _pairs:
                try:
                    _, _p = stats.mannwhitneyu(data_ne[_i], data_ne[_j], alternative='two-sided')
                    _p_corr = min(_p * _n_comp, 1.0) if _n_comp > 1 else _p
                    _star = ('***' if _p_corr < 0.001 else '**' if _p_corr < 0.01
                             else '*' if _p_corr < 0.05 else None)
                    if _star:
                        _sig_lines.append(f'{list(labels_ne)[_i]} vs {list(labels_ne)[_j]}: {_star}')
                except Exception:
                    pass
            if _sig_lines:
                ax.text(0.97, 0.97, '\n'.join(_sig_lines), transform=ax.transAxes,
                        ha='right', va='top', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.set_ylabel('Activation timepoint (frame)')
    else:
        ax.text(0.5, 0.5, 'No activation_timepoint data',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_title('I   Infection Timing by Response Group', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir, f"well_{full_name}_response_kinetics{suffix_clean}", save_pdf, save_svg)
    plt.close()
    print(f"  Saved response kinetics panel")


def plot_preinf_predictors(df_act, df_meas, output_dir, well,
                           save_pdf=False, save_svg=False, suffix=""):
    """
    Pre-infection predictors of response level (low / medium / high).

    Asks: what is different about a cell BEFORE it gets infected that predicts
    whether it will be a low, medium, or high responder?

    Panel A — BFP baseline by response group       (cell content / size proxy)
    Panel B — Pre-activation speed by response group
    Panel C — Daughter-cell fraction by response group (stacked bar)
    Panel D — BFP baseline vs plateau_value scatter (Pearson r)
    Panel E — Pre-activation speed vs plateau_value scatter (Pearson r)
    Panel F — Spatial map: cell positions colored by response group
    """
    if 'response_group' not in df_act.columns:
        print("  Skipping pre-infection predictors: response_group not computed")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    r_groups = ['low', 'medium', 'high']
    r_labels = ['Low', 'Medium', 'High']
    df_r = df_act[df_act['response_group'].isin(r_groups)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f'Pre-Infection Predictors of Response Level — Well {full_name}'
        f'{(" — " + suffix) if suffix else ""}',
        fontsize=14, fontweight='bold', y=0.98)

    def _boxplot(ax, col, ylabel, letter, title, note=None):
        data = [df_r[df_r['response_group'] == g][col].dropna().values for g in r_groups]
        if not any(len(d) > 0 for d in data):
            ax.text(0.5, 0.5, f'No {col} data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')
            return
        bp = ax.boxplot(data, labels=r_labels, patch_artist=True, widths=0.6)
        for patch, g in zip(bp['boxes'], r_groups):
            patch.set_facecolor(RESPONSE_COLORS[g])
            patch.set_alpha(0.7)
        for i, (g, d) in enumerate(zip(r_groups, data)):
            if len(d) > 0:
                ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)),
                           d, alpha=0.4, color=RESPONSE_COLORS[g], s=20, zorder=3)
        _pairwise_sig_brackets(ax, data, labels=r_labels)
        if note:
            ax.text(0.03, 0.03, note, transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=7, color='#555555')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')

    def _scatter(ax, xcol, ycol, xlabel, ylabel, letter, title):
        df_v = df_r.dropna(subset=[xcol, ycol])
        if len(df_v) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')
            return
        for g, label in zip(r_groups, r_labels):
            sub = df_v[df_v['response_group'] == g]
            if len(sub) > 0:
                ax.scatter(sub[xcol], sub[ycol],
                           color=RESPONSE_COLORS[g], alpha=0.55, s=25,
                           label=label, edgecolors='none')
        r_val, p_val = stats.pearsonr(df_v[xcol], df_v[ycol])
        ax.text(0.03, 0.97, f'r = {r_val:.2f}  p = {p_val:.3g}\nn = {len(df_v)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')

    # ── Panel A: BFP baseline by response group ───────────────────────────────
    bfp_col = 'baseline_bfp' if df_r['baseline_bfp'].notna().sum() > 10 else None
    if bfp_col:
        n_bfp = df_r[bfp_col].notna().sum()
        _boxplot(axes[0, 0], bfp_col, 'BFP baseline (a.u.)',
                 'A', 'Cell Content at Infection (BFP)', note=f'n={n_bfp} cells with BFP')
    else:
        _boxplot(axes[0, 0], 'baseline_intensity', 'mNG baseline (a.u.)',
                 'A', 'mNG Baseline at Infection')

    # ── Panel B: pre-activation speed by response group ───────────────────────
    _boxplot(axes[0, 1], 'pre_activation_speed', 'Mean speed before activation (px/frame)',
             'B', 'Pre-Infection Motility')

    # ── Panel C: daughter-cell fraction by response group ─────────────────────
    ax = axes[0, 2]
    if 'is_daughter' in df_r.columns:
        daughter_frac = (df_r.groupby('response_group')['is_daughter']
                         .apply(lambda x: x.fillna(False).mean() * 100)
                         .reindex(r_groups, fill_value=0))
        founder_frac  = 100 - daughter_frac
        x = np.arange(len(r_groups))
        bars_f = ax.bar(x, founder_frac.values,
                        color=[RESPONSE_COLORS[g] for g in r_groups],
                        alpha=0.5, label='Founder', width=0.6)
        bars_d = ax.bar(x, daughter_frac.values, bottom=founder_frac.values,
                        color=[RESPONSE_COLORS[g] for g in r_groups],
                        alpha=0.9, label='Daughter', width=0.6,
                        hatch='//')
        ax.set_xticks(x)
        ax.set_xticklabels(r_labels)
        ax.set_ylabel('% of cells')
        ax.set_ylim(0, 110)
        for i, (f, d) in enumerate(zip(founder_frac.values, daughter_frac.values)):
            ax.text(i, 102, f'{d:.0f}%', ha='center', va='bottom', fontsize=8)
        ax.legend(fontsize=8, loc='upper left')
        # Chi-square test
        try:
            contingency = np.array([
                [int(df_r[df_r['response_group'] == g]['is_daughter'].fillna(False).astype(bool).sum()),
                 int((~df_r[df_r['response_group'] == g]['is_daughter'].fillna(False).astype(bool)).sum())]
                for g in r_groups
            ])
            _, p_chi, _, _ = stats.chi2_contingency(contingency)
            ax.text(0.97, 0.97, f'χ² p={p_chi:.3g}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        except Exception as e:
            print(f"  Warning: chi2 test failed: {e}")
    else:
        ax.text(0.5, 0.5, 'No is_daughter data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.set_title('C   Daughter-Cell Fraction by Response Group', loc='left', fontweight='bold')

    # ── Panel D: BFP baseline vs plateau_value scatter ────────────────────────
    if bfp_col:
        _scatter(axes[1, 0], bfp_col, 'plateau_value',
                 'BFP baseline (a.u.)', 'Plateau value', 'D', 'Cell Content vs Response Level')
    else:
        _scatter(axes[1, 0], 'baseline_intensity', 'plateau_value',
                 'mNG baseline (a.u.)', 'Plateau value', 'D', 'mNG Baseline vs Response Level')

    # ── Panel E: pre-activation speed vs plateau_value scatter ───────────────
    _scatter(axes[1, 1], 'pre_activation_speed', 'plateau_value',
             'Pre-activation speed (px/frame)', 'Plateau value',
             'E', 'Pre-Infection Motility vs Response Level')

    # ── Panel F: spatial map colored by response group ────────────────────────
    ax = axes[1, 2]
    # Use mean centroid across all tracked timepoints per cell as position
    if 'centroid-0' in df_meas.columns and 'centroid-1' in df_meas.columns:
        pos = (df_meas.groupby('unique_track_id')[['centroid-0', 'centroid-1']]
               .mean()
               .rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}))
        df_pos = df_r[['unique_track_id', 'response_group']].merge(
            pos.reset_index(), on='unique_track_id', how='left')
        df_pos = df_pos.dropna(subset=['x', 'y'])
        for g, label in zip(r_groups, r_labels):
            sub = df_pos[df_pos['response_group'] == g]
            if len(sub) > 0:
                ax.scatter(sub['x'], sub['y'],
                           color=RESPONSE_COLORS[g], alpha=0.65, s=20,
                           label=f'{label} (n={len(sub)})', edgecolors='none')
        ax.set_xlabel('x position (px)')
        ax.set_ylabel('y position (px)')
        ax.invert_yaxis()   # image convention: y increases downward
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'No centroid data in df_meas',
                transform=ax.transAxes, ha='center', va='center', color='gray')
    ax.set_title('F   Spatial Distribution by Response Group', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir,
                f"well_{full_name}_preinf_predictors{suffix_clean}",
                save_pdf, save_svg)
    plt.close()
    print(f"  Saved pre-infection predictors panel")


def plot_track_duration_by_response(df_act, output_dir, well, save_pdf=False, save_svg=False,
                                    suffix="", timepoint_max=None):
    """
    Compare cell track duration across response-amplitude groups (low / medium / high).

    Cells that disappear early are presumed dead or detached — a proxy for viral
    cytopathic effect. Higher mNG/BFP ratio (= more replication) may correlate
    with shorter survival.

    Panel A — track_duration by response group  (boxplot + scatter + KW p)
    Panel B — track_end_t by response group     (when does the cell disappear?)
    Panel C — retention curve: fraction of cells still tracked vs time, by group
    Panel D — plateau_value vs track_duration scatter (replication level vs survival)
    """
    if 'track_duration' not in df_act.columns:
        print("  Skipping track duration plot: compute_track_duration not run")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    r_groups = ['low', 'medium', 'high']
    r_labels = ['Low', 'Medium', 'High']
    df_fit = df_act[df_act['response_group'].isin(r_groups)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f'Cell Track Duration by Response Group — Well {full_name}{(" — " + suffix) if suffix else ""}',
                 fontsize=14, fontweight='bold', y=0.98)

    def _boxplot_panel(ax, col, ylabel, letter, title):
        data = [df_fit[df_fit['response_group'] == g][col].dropna().values for g in r_groups]
        if not any(len(d) > 0 for d in data):
            ax.text(0.5, 0.5, f'No {col} data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')
            return
        bp = ax.boxplot(data, labels=r_labels, patch_artist=True, widths=0.6)
        for patch, g in zip(bp['boxes'], r_groups):
            patch.set_facecolor(RESPONSE_COLORS[g])
            patch.set_alpha(0.7)
        for i, (g, d) in enumerate(zip(r_groups, data)):
            if len(d) > 0:
                ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)),
                           d, alpha=0.4, color=RESPONSE_COLORS[g], s=20)
        _pairwise_sig_brackets(ax, data, labels=r_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')

    # Panel A: track_duration by response group
    _boxplot_panel(axes[0, 0], 'track_duration', 'Track duration (frames)',
                   'A', 'Track Duration by Response Group')

    # Panel B: track_end_t by response group
    _boxplot_panel(axes[0, 1], 'track_end_t', 'Last observed timepoint',
                   'B', 'Track End Time by Response Group')

    # Panel C: retention curve — fraction still tracked at each timepoint
    ax = axes[1, 0]
    t_end = int(timepoint_max) if timepoint_max is not None else int(df_fit['track_end_t'].max())
    timepoints = np.arange(0, t_end + 1)
    for g, label in zip(r_groups, r_labels):
        sub = df_fit[df_fit['response_group'] == g].dropna(subset=['track_start_t', 'track_end_t'])
        if len(sub) == 0:
            continue
        n_total = len(sub)
        retention = [
            ((sub['track_start_t'] <= t) & (sub['track_end_t'] >= t)).sum() / n_total
            for t in timepoints
        ]
        ax.plot(timepoints, retention, color=RESPONSE_COLORS[g], linewidth=2.5,
                label=f'{label} (n={n_total})')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Fraction of cells still tracked')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, t_end)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_title('C   Cell Retention Over Time', loc='left', fontweight='bold')

    # Panel D: plateau_value vs track_duration scatter
    ax = axes[1, 1]
    df_valid = df_fit.dropna(subset=['plateau_value', 'track_duration'])
    if len(df_valid) > 0:
        for g, label in zip(r_groups, r_labels):
            sub = df_valid[df_valid['response_group'] == g]
            if len(sub) > 0:
                ax.scatter(sub['track_duration'], sub['plateau_value'],
                           color=RESPONSE_COLORS[g], alpha=0.6, s=25,
                           label=label, edgecolors='none')
        if len(df_valid) > 5:
            r, p = stats.pearsonr(df_valid['track_duration'], df_valid['plateau_value'])
            ax.text(0.03, 0.97, f'r = {r:.2f}  p = {p:.3g}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.set_xlabel('Track duration (frames)')
        ax.set_ylabel('Plateau value (mNG/BFP)')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_title('D   Replication Level vs Cell Survival', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir, f"well_{full_name}_track_duration_by_response{suffix_clean}",
                save_pdf, save_svg)
    plt.close()
    print(f"  Saved track duration by response group panel")


def plot_division_analysis(df_act, df_divisions, output_dir, well,
                           save_pdf=False, save_svg=False, suffix="",
                           timepoint_min=None, timepoint_max=None):
    """
    Examine the relationship between cell division and viral reporter activation.

    Panel A — activation rate for non-daughters / daughters of uninfected / daughters of infected
    Panel B — activation timepoint for the same 3 origin groups
    Panel C — time from birth to activation for daughter cells (can be negative if parent
               was already activated at division — viral inheritance)
    Panel D — response group breakdown: daughters of activated vs non-activated parents
    Panel E — post-division survival (track_end_t − division_timepoint) by response group
    Panel F — parent vs daughter plateau_value (is response amplitude inherited?)
    """
    if df_divisions is None or 'is_daughter' not in df_act.columns:
        print("  Skipping division analysis: no division data available")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    r_groups = ['low', 'medium', 'high']
    r_labels = ['Low', 'Medium', 'High']

    mask_nd  = ~df_act['is_daughter'].fillna(False).astype(bool)
    mask_dna = (df_act['is_daughter'].fillna(False).astype(bool) &
                ~df_act['parent_activated'].fillna(False).astype(bool))
    mask_da  = (df_act['is_daughter'].fillna(False).astype(bool) &
                df_act['parent_activated'].fillna(False).astype(bool))

    origin_masks  = [mask_nd,  mask_dna,  mask_da]
    origin_labels = ['Non-\ndaughter', 'Born from\nnon-infected', 'Born from\ninfected']
    origin_colors = ['#7F7F7F', '#6BAED6', '#D62728']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f'Cell Division & Infection — Well {full_name}{(" — " + suffix) if suffix else ""}',
        fontsize=14, fontweight='bold', y=0.98
    )

    # ── Panel A: activation rate by origin ──────────────────────────────────
    ax = axes[0, 0]
    rates, ns = [], []
    for mask in origin_masks:
        sub = df_act[mask]
        ns.append(len(sub))
        rates.append(float(sub['activates'].mean() * 100) if len(sub) > 0 else 0.0)
    bars = ax.bar(origin_labels, rates, color=origin_colors, alpha=0.8,
                  edgecolor='white', width=0.6)
    for bar, n, r in zip(bars, ns, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.5, f'n={n}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Activation rate (%)')
    ax.set_ylim(0, max(rates + [1]) * 1.3)
    ax.set_title('A   Activation Rate by Cell Origin', loc='left', fontweight='bold')

    # ── Panel B: activation timepoint by origin ──────────────────────────────
    ax = axes[0, 1]
    data_b = [
        df_act[mask & (df_act['activates'] == True)]['activation_timepoint'].dropna().values
        for mask in origin_masks
    ]
    bp = ax.boxplot(data_b, labels=origin_labels, patch_artist=True, widths=0.6)
    for patch, col in zip(bp['boxes'], origin_colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    for i, (d, col) in enumerate(zip(data_b, origin_colors)):
        if len(d) > 0:
            ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)),
                       d, alpha=0.4, color=col, s=20)
    _pairwise_sig_brackets(ax, data_b, labels=origin_labels)
    ax.set_ylabel('Activation timepoint')
    ax.set_title('B   Activation Time by Cell Origin', loc='left', fontweight='bold')

    # ── Panel C: delay from birth to activation ───────────────────────────────
    ax = axes[0, 2]
    daughters_act = df_act[
        df_act['is_daughter'].fillna(False).astype(bool) &
        (df_act['activates'] == True)
    ].dropna(subset=['activation_timepoint', 'division_timepoint'])
    if len(daughters_act) > 0:
        delay = daughters_act['activation_timepoint'] - daughters_act['division_timepoint']
        ax.hist(delay, bins=20, color='#4393C3', edgecolor='white', alpha=0.85)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Birth timepoint')
        ax.axvline(delay.median(), color='#D62728', linestyle='-', linewidth=2,
                   label=f'Median = {delay.median():.1f}')
        # Fraction with negative delay (activated before/at birth = inherited)
        frac_inherited = (delay <= 0).mean() * 100
        ax.text(0.03, 0.97, f'{frac_inherited:.0f}% inherited\n(delay ≤ 0)',
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.legend(fontsize=8)
        ax.set_xlabel('Activation − Birth timepoint (frames)')
        ax.set_ylabel('Count')
    else:
        ax.text(0.5, 0.5, 'No activated daughters', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.set_title('C   Time from Birth to Activation', loc='left', fontweight='bold')

    # ── Panel D: response group distribution by parent status ─────────────────
    ax = axes[1, 0]
    groups_d = {
        'Born from\nnon-infected': df_act[mask_dna & df_act['response_group'].isin(r_groups)],
        'Born from\ninfected':     df_act[mask_da  & df_act['response_group'].isin(r_groups)],
    }
    x = np.arange(len(r_groups))
    width = 0.35
    for j, (lbl, sub) in enumerate(groups_d.items()):
        if len(sub) == 0:
            continue
        counts = [len(sub[sub['response_group'] == g]) for g in r_groups]
        total  = sum(counts)
        pcts   = [c / total * 100 if total > 0 else 0 for c in counts]
        color  = '#6BAED6' if 'non' in lbl else '#D62728'
        ax.bar(x + (j - 0.5) * width, pcts, width,
               label=f'{lbl} (n={total})', color=color, alpha=0.8, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(r_labels)
    ax.set_ylabel('Percentage (%)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('D   Response Group by Parent Infection', loc='left', fontweight='bold')

    # ── Panel E: post-division survival by response group ─────────────────────
    ax = axes[1, 1]
    daughters_rg = df_act[
        df_act['is_daughter'].fillna(False).astype(bool) &
        df_act['response_group'].isin(r_groups)
    ].copy()
    if 'track_end_t' in daughters_rg.columns:
        daughters_rg['post_div_survival'] = (
            daughters_rg['track_end_t'] - daughters_rg['division_timepoint']
        )
        data_e = [daughters_rg[daughters_rg['response_group'] == g]['post_div_survival']
                  .dropna().values for g in r_groups]
        if any(len(d) > 0 for d in data_e):
            bp = ax.boxplot(data_e, labels=r_labels, patch_artist=True, widths=0.6)
            for patch, g in zip(bp['boxes'], r_groups):
                patch.set_facecolor(RESPONSE_COLORS[g]); patch.set_alpha(0.7)
            for i, (g, d) in enumerate(zip(r_groups, data_e)):
                if len(d) > 0:
                    ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)),
                               d, alpha=0.4, color=RESPONSE_COLORS[g], s=20)
            _pairwise_sig_brackets(ax, data_e, labels=r_labels)
        ax.set_ylabel('Frames from birth to track end')
    else:
        ax.text(0.5, 0.5, 'Run compute_track_duration first',
                transform=ax.transAxes, ha='center', va='center', color='gray')
    ax.set_title('E   Post-Division Survival by Response Group', loc='left', fontweight='bold')

    # ── Panel F: parent vs daughter plateau value ─────────────────────────────
    ax = axes[1, 2]
    if 'plateau_value' in df_act.columns:
        pv_lookup = df_act.set_index('unique_track_id')['plateau_value']
        daughters_pv = df_act[df_act['is_daughter'].fillna(False).astype(bool)].copy()
        daughters_pv['parent_plateau'] = daughters_pv['parent_unique_track_id'].map(pv_lookup)
        valid_pv = daughters_pv.dropna(subset=['plateau_value', 'parent_plateau'])
        if len(valid_pv) > 0:
            for g in r_groups:
                sub = valid_pv[valid_pv.get('response_group', pd.Series()) == g] \
                    if 'response_group' in valid_pv.columns else pd.DataFrame()
                if len(sub) > 0:
                    ax.scatter(sub['parent_plateau'], sub['plateau_value'],
                               color=RESPONSE_COLORS[g], alpha=0.6, s=30,
                               label=g.capitalize(), edgecolors='none')
            lims = [min(valid_pv['parent_plateau'].min(), valid_pv['plateau_value'].min()),
                    max(valid_pv['parent_plateau'].max(), valid_pv['plateau_value'].max())]
            ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.4, label='Identity')
            if len(valid_pv) > 3:
                r, p = stats.pearsonr(valid_pv['parent_plateau'], valid_pv['plateau_value'])
                ax.text(0.03, 0.97, f'r = {r:.2f}  p = {p:.3g}',
                        transform=ax.transAxes, ha='left', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
            ax.set_xlabel('Parent plateau value')
            ax.set_ylabel('Daughter plateau value')
            ax.legend(fontsize=7, loc='lower right')
        else:
            ax.text(0.5, 0.5, 'No matched parent–daughter\nplateau values',
                    transform=ax.transAxes, ha='center', va='center', color='gray')
    ax.set_title('F   Parent vs Daughter Response Amplitude', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir, f"well_{full_name}_division_analysis{suffix_clean}", save_pdf, save_svg)
    plt.close()
    print(f"  Saved division analysis panel")


def plot_motility_analysis(df_act, df_meas, output_dir, well,
                           save_pdf=False, save_svg=False, suffix="",
                           timepoint_min=None, timepoint_max=None,
                           signal_col='mean_intensity', df_uninfected=None):
    """
    Compare cell motility between infected and uninfected cells, and before vs after activation.

    Panel A – Mean speed by response group (uninfected + low/medium/high)  [KW p]
    Panel B – Pre- vs post-activation speed for activated cells  [Wilcoxon p]
    Panel C – Mean speed ± SEM vs time relative to activation (trajectory aligned to t_act=0)
    Panel D – Straightness by response group  [KW p]
    Panel E – Mean speed vs plateau_value scatter (infected cells, coloured by response group)
    Panel F – Net displacement: uninfected vs infected  [Mann-Whitney p]

    If df_uninfected is provided (from load_script2_uninfected), it is used as the reference
    group for panels A, D, F instead of script 3's own activates==False cells.
    """
    if 'mean_speed' not in df_act.columns:
        print("  Skipping motility plot: compute_motility not run")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    r_groups = ['low', 'medium', 'high']
    r_labels = ['Low', 'Medium', 'High']
    never_color = '#BDBDBD'

    # Reference group: use script 2 quality-filtered uninfected cells if available,
    # otherwise fall back to script 3's own activates==False cells
    if df_uninfected is not None and len(df_uninfected) > 0:
        ref_speed       = df_uninfected['mean_speed'].dropna().values
        ref_straight    = df_uninfected['straightness'].dropna().values
        ref_net_disp    = df_uninfected['net_displacement'].dropna().values
        ref_label       = 'Uninfected\n(script 2)'
        ref_source_note = f'n={len(df_uninfected)}'
    else:
        ref_speed       = df_act[~df_act['activates']]['mean_speed'].dropna().values
        ref_straight    = df_act[~df_act['activates']]['straightness'].dropna().values
        ref_net_disp    = df_act[~df_act['activates']]['net_displacement'].dropna().values
        ref_label       = 'Never\nActivated'
        ref_source_note = ''

    all_labels = [ref_label] + r_labels
    all_colors = [never_color] + [RESPONSE_COLORS[g] for g in r_groups]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f'Cell Motility Analysis — Well {full_name}{(" — " + suffix) if suffix else ""}',
        fontsize=14, fontweight='bold', y=0.98)

    def _mw_p(a, b):
        if len(a) < 3 or len(b) < 3:
            return None
        try:
            _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            return p
        except Exception:
            return None

    def _boxplot(ax, data_list, labels, colors, ylabel, letter, title):
        non_empty = [(d, l, c) for d, l, c in zip(data_list, labels, colors) if len(d) > 0]
        if not non_empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')
            return
        data_ne, labels_ne, colors_ne = zip(*non_empty)
        bp = ax.boxplot(list(data_ne), labels=list(labels_ne), patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors_ne):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for i, (d, c) in enumerate(zip(data_ne, colors_ne)):
            ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)), d,
                       alpha=0.3, color=c, s=15, zorder=3)
        _pairwise_sig_brackets(ax, list(data_ne), labels=list(labels_ne))
        ax.set_ylabel(ylabel)
        ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')

    # ── Panel A: mean speed by response group ────────────────────────────────
    ax = axes[0, 0]
    data_a = [ref_speed] + [
        df_act[df_act['response_group'] == g]['mean_speed'].dropna().values
        for g in r_groups
    ]
    title_a = f'Mean Speed by Response Group{(" — " + ref_source_note) if ref_source_note else ""}'
    _boxplot(ax, data_a, all_labels, all_colors,
             'Mean speed (px / frame)', 'A', title_a)

    # ── Panel B: pre vs post activation speed (activated cells) ─────────────
    ax = axes[0, 1]
    df_paired = df_act[df_act['activates'] == True][
        ['pre_activation_speed', 'post_activation_speed']
    ].dropna()
    if len(df_paired) >= 3:
        pre_vals  = df_paired['pre_activation_speed'].values
        post_vals = df_paired['post_activation_speed'].values
        bp = ax.boxplot([pre_vals, post_vals],
                        labels=['Pre-activation', 'Post-activation'],
                        patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], ['#74C476', '#E6550D']):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for i, (vals, c) in enumerate(zip([pre_vals, post_vals], ['#31A354', '#A63603'])):
            ax.scatter(np.random.normal(i + 1, 0.07, size=len(vals)), vals,
                       alpha=0.3, color=c, s=15, zorder=3)
        try:
            _, p_wc = stats.wilcoxon(pre_vals, post_vals)
            ax.text(0.97, 0.97, f'Wilcoxon p={p_wc:.3g}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        except Exception:
            pass
        ax.set_ylabel('Mean speed (px / frame)')
    else:
        ax.text(0.5, 0.5, 'Insufficient activated cells', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.set_title('B   Speed Before vs After Activation', loc='left', fontweight='bold')

    # ── Panel C: speed trajectory aligned to activation timepoint ───────────
    ax = axes[0, 2]
    if 'centroid-0' in df_meas.columns and 'centroid-1' in df_meas.columns:
        act_cells = df_act[df_act['activates'] == True][
            ['unique_track_id', 'activation_timepoint']
        ].set_index('unique_track_id')['activation_timepoint'].to_dict()

        pos_df = (df_meas[['unique_track_id', 'timepoint', 'centroid-0', 'centroid-1']]
                  .sort_values(['unique_track_id', 'timepoint'])
                  .copy())
        pos_df['dy']   = pos_df.groupby('unique_track_id')['centroid-0'].diff()
        pos_df['dx']   = pos_df.groupby('unique_track_id')['centroid-1'].diff()
        pos_df['step'] = np.sqrt(pos_df['dy'] ** 2 + pos_df['dx'] ** 2)
        pos_df['prev_t'] = pos_df.groupby('unique_track_id')['timepoint'].shift(1)
        pos_df = pos_df.dropna(subset=['step', 'prev_t'])

        pos_df['t_act'] = pos_df['unique_track_id'].map(act_cells)
        act_pos = pos_df.dropna(subset=['t_act']).copy()
        act_pos['rel_t'] = (act_pos['prev_t'] - act_pos['t_act']).astype(int)

        if len(act_pos) > 0:
            aligned = act_pos.groupby('rel_t')['step'].agg(['mean', 'sem', 'count']).reset_index()
            aligned = aligned[aligned['count'] >= 5]  # at least 5 cells per bin
            ax.plot(aligned['rel_t'], aligned['mean'], color='#E6550D', linewidth=1.5)
            ax.fill_between(aligned['rel_t'],
                            aligned['mean'] - aligned['sem'],
                            aligned['mean'] + aligned['sem'],
                            alpha=0.25, color='#E6550D')
            ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Activation')
            ax.set_xlabel('Time relative to activation (frames)')
            ax.set_ylabel('Mean speed (px / frame)')
            ax.legend(fontsize=8)
    ax.set_title('C   Speed Aligned to Activation', loc='left', fontweight='bold')

    # ── Panel D: straightness by response group ──────────────────────────────
    ax = axes[1, 0]
    data_d = [ref_straight] + [
        df_act[df_act['response_group'] == g]['straightness'].dropna().values
        for g in r_groups
    ]
    _boxplot(ax, data_d, all_labels, all_colors,
             'Straightness (0–1)', 'D', 'Track Straightness by Response Group')

    # ── Panel E: mean speed vs plateau_value scatter ─────────────────────────
    ax = axes[1, 1]
    df_fit = df_act[df_act['response_group'].isin(r_groups)].dropna(
        subset=['mean_speed', 'plateau_value'])
    for g, label in zip(r_groups, r_labels):
        sub = df_fit[df_fit['response_group'] == g]
        ax.scatter(sub['plateau_value'], sub['mean_speed'],
                   color=RESPONSE_COLORS[g], alpha=0.5, s=20, label=label)
    if len(df_fit) >= 5:
        from scipy.stats import pearsonr
        r_val, p_val = pearsonr(df_fit['plateau_value'], df_fit['mean_speed'])
        ax.text(0.97, 0.97, f'r={r_val:.2f}, p={p_val:.3g}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    ax.set_xlabel('Plateau value (replication level)')
    ax.set_ylabel('Mean speed (px / frame)')
    ax.legend(fontsize=8)
    ax.set_title('E   Speed vs Replication Level', loc='left', fontweight='bold')

    # ── Panel F: net displacement — uninfected vs infected ───────────────────
    ax = axes[1, 2]
    nd_infected = df_act[df_act['activates'] == True]['net_displacement'].dropna().values
    data_f   = [d for d in [ref_net_disp, nd_infected] if len(d) > 0]
    labels_f = [l for d, l in zip([ref_net_disp, nd_infected], [ref_label, 'Infected'])
                if len(d) > 0]
    colors_f = [c for d, c in zip([ref_net_disp, nd_infected], [never_color, '#BD0026'])
                if len(d) > 0]
    if data_f:
        bp = ax.boxplot(data_f, labels=labels_f, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors_f):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for i, (d, c) in enumerate(zip(data_f, colors_f)):
            ax.scatter(np.random.normal(i + 1, 0.07, size=len(d)), d,
                       alpha=0.3, color=c, s=15, zorder=3)
        pval = _mw_p(ref_net_disp, nd_infected)
        if pval is not None:
            ax.text(0.97, 0.97, f'MW p={pval:.3g}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    ax.set_ylabel('Net displacement (px)')
    ax.set_title(f'F   Net Displacement: {ref_label.replace(chr(10), " ")} vs Infected',
                 loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir, f"well_{full_name}_motility_analysis{suffix_clean}", save_pdf, save_svg)
    plt.close()
    print(f"  Saved motility analysis panel")


def plot_response_distribution_over_time(df_act, df_meas, output_dir, well,
                                         save_pdf=False, save_svg=False,
                                         suffix="", timepoint_max=None,
                                         r2_min=0.7):
    """
    At each timepoint, show the fraction of ALL tracked cells in four categories:

      Uninfected – not yet activated at this timepoint, or never activated
      Low        – activated, plateau_value in bottom tertile of infected cells
      Medium     – activated, plateau_value in middle tertile
      High       – activated, plateau_value in top tertile

    Thresholds are computed from the sigmoid plateau_value of confirmed infected
    cells only (activates=True, sigmoid_r2 >= r2_min), using 33rd/66th percentile
    cuts.  This keeps the uninfected cluster entirely out of the threshold
    calculation.  A cell is classified as Uninfected until its activation_timepoint,
    then transitions permanently to its response group.

    Panel A – Stacked area: % of all tracked cells in each category over time.
    Panel B – plateau_value histogram of activated cells with threshold lines,
              validating the threshold choice.
    """
    r_groups   = ['low', 'medium', 'high']
    categories = ['uninfected'] + r_groups
    cat_colors = ['#BDBDBD'] + [RESPONSE_COLORS[g] for g in r_groups]
    cat_labels = ['Uninfected', 'Low', 'Medium', 'High']

    # ── Compute thresholds from plateau_value of well-fit infected cells ──────
    # Only THRESHOLDS use the sigmoid R² filter — all activated cells are classified
    all_activated = df_act[df_act['activates'].fillna(False).astype(bool)].copy()
    if len(all_activated) == 0:
        print("  Skipping response distribution over time: no activated cells")
        return

    fit_mask = all_activated['plateau_value'].notna()
    if 'sigmoid_r2' in all_activated.columns:
        fit_mask &= all_activated['sigmoid_r2'] >= r2_min

    plateau_vals = all_activated.loc[fit_mask, 'plateau_value']
    if len(plateau_vals) < 6:
        print("  Skipping response distribution over time: too few cells with good sigmoid fits")
        return

    low_thr  = plateau_vals.quantile(1 / 3)
    high_thr = plateau_vals.quantile(2 / 3)
    print(f"  Plateau thresholds from {len(plateau_vals)} well-fit infected cells "
          f"(R²≥{r2_min}): low={low_thr:.3f}, high={high_thr:.3f}")

    # ── Classify ALL activated cells ──────────────────────────────────────────
    # For cells with a good plateau_value, use it directly.
    # For cells with a poor sigmoid fit, fall back to the peak measured signal
    # as a proxy (it approximates the asymptote the sigmoid would reach).
    sig_col = next((c for c in ['mng_bfp_ratio', 'mean_intensity']
                    if c in df_meas.columns), None)
    if sig_col:
        peak_signal = df_meas.groupby('unique_track_id')[sig_col].max()
        all_activated['_proxy'] = all_activated['plateau_value'].fillna(
            all_activated['unique_track_id'].map(peak_signal))
    else:
        all_activated['_proxy'] = all_activated['plateau_value']

    def _grp(v):
        if pd.isna(v):   return 'low'   # last-resort fallback
        if v < low_thr:  return 'low'
        if v < high_thr: return 'medium'
        return 'high'

    all_activated['_grp'] = all_activated['_proxy'].apply(_grp)

    n_proxy = all_activated['plateau_value'].isna().sum()
    print(f"  Classified {len(all_activated)} activated cells "
          f"({len(all_activated) - n_proxy} via plateau_value, {n_proxy} via peak-signal proxy)")

    group_map = all_activated.set_index('unique_track_id')['_grp'].to_dict()
    act_time  = all_activated.set_index('unique_track_id')['activation_timepoint'].dropna().to_dict()

    # ── Classify every measurement row ───────────────────────────────────────
    df_class = df_meas[['unique_track_id', 'timepoint']].copy()
    df_class['_grp']  = df_class['unique_track_id'].map(group_map)   # NaN if not in map
    df_class['_tact'] = df_class['unique_track_id'].map(act_time)    # NaN if not in map

    # A row is infected (and past activation) when _grp is set AND t >= _tact
    infected_mask = (
        df_class['_grp'].notna() &
        (df_class['timepoint'] >= df_class['_tact'].fillna(np.inf))
    )
    df_class['category'] = 'uninfected'
    df_class.loc[infected_mask, 'category'] = df_class.loc[infected_mask, '_grp']

    # ── Percentage at each timepoint ─────────────────────────────────────────
    t_max = int(timepoint_max) if timepoint_max is not None else int(df_meas['timepoint'].max())
    timepoints = np.arange(0, t_max + 1)

    counts = (
        df_class[df_class['timepoint'].isin(timepoints)]
        .groupby(['timepoint', 'category'])
        .size()
        .unstack(fill_value=0)
    )
    for c in categories:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts.reindex(timepoints, fill_value=0)
    totals = counts.sum(axis=1).replace(0, np.nan)
    pct_df = (counts[categories].div(totals, axis=0) * 100).fillna(0)

    # Sanity-check diagnostic
    infected_pct = (pct_df['low'] + pct_df['medium'] + pct_df['high'])
    print(f"  Total unique tracks in df_meas: {df_meas['unique_track_id'].nunique()}")
    print(f"  Peak infected fraction: {infected_pct.max():.1f}%  |  "
          f"Min uninfected fraction: {pct_df['uninfected'].min():.1f}%")

    # ── Dynamic classification: current signal vs plateau thresholds ─────────
    # Cells are classified at each timepoint by their ACTUAL current signal value
    # (not their final plateau), so they genuinely start as "low" at activation
    # and may transition to "medium" or "high" as the signal rises over time.
    # The same low_thr / high_thr (from plateau_value percentiles) are reused so
    # the steady-state composition matches the plateau-based classification.
    act_ids = set(act_time.keys())
    if sig_col is not None:
        df_dyn = df_meas[['unique_track_id', 'timepoint', sig_col]].copy()
        df_dyn['_tact'] = df_dyn['unique_track_id'].map(act_time)
        active_mask_dyn = (
            df_dyn['unique_track_id'].isin(act_ids) &
            (df_dyn['timepoint'] >= df_dyn['_tact'].fillna(np.inf))
        )
        dyn_sig = df_dyn.loc[active_mask_dyn, sig_col]
        dyn_cat = pd.cut(dyn_sig,
                         bins=[-np.inf, low_thr, high_thr, np.inf],
                         labels=['low', 'medium', 'high'])
        df_dyn['category'] = 'uninfected'
        df_dyn.loc[active_mask_dyn, 'category'] = dyn_cat.astype(str)

        dyn_counts = (
            df_dyn[df_dyn['timepoint'].isin(timepoints)]
            .groupby(['timepoint', 'category'])
            .size()
            .unstack(fill_value=0)
        )
        for c in categories:          # all 4: uninfected + r_groups
            if c not in dyn_counts.columns:
                dyn_counts[c] = 0
        dyn_counts = dyn_counts.reindex(timepoints, fill_value=0)

        # Panel A data: all categories as % of total tracked cells
        dyn_totals  = dyn_counts[categories].sum(axis=1).replace(0, np.nan)
        dyn_pct_df  = (dyn_counts[categories].div(dyn_totals, axis=0) * 100).fillna(0)

        # Panel B data: low/medium/high as % of infected cells only
        dyn_inf_counts = dyn_counts[r_groups]
        dyn_inf_totals = dyn_inf_counts.sum(axis=1).replace(0, np.nan)
        dyn_inf_pct    = (dyn_inf_counts.div(dyn_inf_totals, axis=0) * 100).fillna(0)
        dyn_inf_n      = dyn_inf_counts.sum(axis=1)
        use_dynamic = True
    else:
        use_dynamic = False

    # ── Plot ─────────────────────────────────────────────────────────────────
    set_publication_style()
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f'Infection Level Distribution Over Time — Well {full_name}'
        f'{(" — " + suffix) if suffix else ""}\n'
        f'Classification: instantaneous {sig_col or "signal"} vs plateau thresholds '
        f'(low<{low_thr:.2f}, high>{high_thr:.2f}; '
        f'33rd/66th pct of {len(plateau_vals)} well-fit cells, R²≥{r2_min})',
        fontsize=11, fontweight='bold', y=1.03)

    # Panel A: stacked area — all tracked cells, same dynamic classification as Panel B
    ax = axes[0]
    src_pct = dyn_pct_df if use_dynamic else pct_df
    ax.stackplot(timepoints,
                 [src_pct[c].values for c in categories],
                 labels=cat_labels, colors=cat_colors, alpha=0.85)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('% of tracked cells')
    ax.set_title('A   Current Signal Level (All Tracked Cells)', loc='left', fontweight='bold')
    ax.legend(loc='center right', fontsize=9)
    # Annotate final percentages
    cumsum = 0
    for c, label, col in zip(categories, cat_labels, cat_colors):
        pct_final = src_pct[c].iloc[-1]
        if pct_final >= 3:
            ax.text(t_max - 0.5, cumsum + pct_final / 2,
                    f'{pct_final:.0f}%', va='center', ha='right',
                    fontsize=8, color='white', fontweight='bold')
        cumsum += pct_final

    # Panel B: stacked area — infected cells only, classified by CURRENT signal
    # Cells start as "low" at activation (signal just above baseline) and
    # transition to "medium" / "high" as their signal rises toward the plateau.
    ax = axes[1]
    if use_dynamic:
        valid_tp = dyn_inf_n[dyn_inf_n >= 3].index
        if len(valid_tp) > 0:
            ax.stackplot(valid_tp,
                         [dyn_inf_pct.loc[valid_tp, g].values for g in r_groups],
                         labels=['Low', 'Medium', 'High'],
                         colors=[RESPONSE_COLORS[g] for g in r_groups],
                         alpha=0.85)
            ax.set_xlim(0, t_max)
            ax.set_ylim(0, 100)
            # Annotate final percentages
            last_tp = valid_tp[-1]
            cumsum = 0
            for g in r_groups:
                pct_final = dyn_inf_pct.loc[last_tp, g]
                if pct_final >= 3:
                    ax.text(last_tp - 0.5, cumsum + pct_final / 2,
                            f'{pct_final:.0f}%', va='center', ha='right',
                            fontsize=8, color='white', fontweight='bold')
                cumsum += pct_final
            ax.legend(loc='center right', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Insufficient infected cells', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
    else:
        ax.text(0.5, 0.5, 'No signal column available', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('% of infected cells')
    ax.set_title('B   Current Signal Level (Infected Only)', loc='left', fontweight='bold')
    ax.text(0.02, 0.97,
            f'Thresholds: low<{low_thr:.2f}, high>{high_thr:.2f}\n'
            f'(33rd/66th pct of plateau values)',
            transform=ax.transAxes, va='top', fontsize=7, color='#555555')

    # Panel C: plateau_value distribution of infected cells with thresholds
    ax = axes[2]
    ax.hist(plateau_vals, bins=40, color='#969696', alpha=0.75,
            edgecolor='white', linewidth=0.5)
    ax.axvline(low_thr,  color=RESPONSE_COLORS['low'],    linestyle='--',
               linewidth=1.5, label=f'33rd pct = {low_thr:.2f}')
    ax.axvline(high_thr, color=RESPONSE_COLORS['high'],   linestyle='--',
               linewidth=1.5, label=f'66th pct = {high_thr:.2f}')
    x_min = plateau_vals.min()
    x_max = plateau_vals.max() * 1.05
    ax.axvspan(x_min,    low_thr,  alpha=0.15, color=RESPONSE_COLORS['low'])
    ax.axvspan(low_thr,  high_thr, alpha=0.15, color=RESPONSE_COLORS['medium'])
    ax.axvspan(high_thr, x_max,    alpha=0.15, color=RESPONSE_COLORS['high'])
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Plateau value (sigmoid asymptote)')
    ax.set_ylabel('Number of infected cells')
    ax.set_title('C   Plateau Value Distribution — Infected Cells', loc='left', fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_figure(fig, fig_dir,
                f"well_{full_name}_response_distribution_over_time{suffix_clean}",
                save_pdf, save_svg)
    plt.close()
    print(f"  Saved response distribution over time panel")


def plot_auc(df_act, output_dir, well, save_pdf=False, save_svg=False,
             suffix="", timepoint_min=None, timepoint_max=None):
    """
    AUC analysis — 6-panel figure:
    A  AUC distribution
    B  AUC by timing group (early/average/late)
    C  AUC by response group (low/medium/high)
    D  Activation timepoint vs AUC (scatter)
    E  Plateau value vs AUC (scatter)
    F  AUC cumulative distribution by timing group
    """
    if 'auc' not in df_act.columns:
        print("  Skipping AUC plots: auc not computed")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    groups_a = ['early', 'average', 'late']
    labels_a = ['Early', 'Average', 'Late']
    groups_r = ['low', 'medium', 'high']
    labels_r = ['Low', 'Medium', 'High']

    df_v = df_act[df_act['auc'].notna()]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'AUC Analysis — Well {full_name}{suffix}', fontsize=14, fontweight='bold', y=0.98)

    # Panel A: distribution
    ax = axes[0, 0]
    ax.hist(df_v['auc'], bins=30, color='#74ADD1', edgecolor='white', alpha=0.8)
    ax.axvline(df_v['auc'].median(), color='#E31A1C', linestyle='--', linewidth=2,
               label=f"Median: {df_v['auc'].median():.1f}")
    ax.legend()
    ax.set_xlabel('AUC (signal × frames above baseline)')
    ax.set_ylabel('Count')
    ax.set_title('A   AUC Distribution', loc='left', fontweight='bold')

    # Panel B: by timing group
    ax = axes[0, 1]
    data_b = [df_v[df_v['activation_group'] == g]['auc'].values for g in groups_a]
    bp = ax.boxplot(data_b, labels=labels_a, patch_artist=True, widths=0.6)
    for patch, g in zip(bp['boxes'], groups_a):
        patch.set_facecolor(COLORS[g]); patch.set_alpha(0.7)
    for i, (g, d) in enumerate(zip(groups_a, data_b)):
        if len(d) > 0:
            ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)), d,
                       alpha=0.4, color=COLORS[g], s=20)
    ax.set_ylabel('AUC')
    ax.set_title('B   AUC by Timing Group', loc='left', fontweight='bold')

    # Panel C: by response group
    ax = axes[0, 2]
    if 'response_group' in df_v.columns:
        data_c = [df_v[df_v['response_group'] == g]['auc'].values for g in groups_r]
        bp = ax.boxplot(data_c, labels=labels_r, patch_artist=True, widths=0.6)
        for patch, g in zip(bp['boxes'], groups_r):
            patch.set_facecolor(RESPONSE_COLORS[g]); patch.set_alpha(0.7)
        for i, (g, d) in enumerate(zip(groups_r, data_c)):
            if len(d) > 0:
                ax.scatter(np.random.normal(i + 1, 0.08, size=len(d)), d,
                           alpha=0.4, color=RESPONSE_COLORS[g], s=20)
    ax.set_ylabel('AUC')
    ax.set_title('C   AUC by Response Group', loc='left', fontweight='bold')

    # Panel D: timing vs AUC
    ax = axes[1, 0]
    for g, label in zip(groups_a, labels_a):
        sub = df_v[df_v['activation_group'] == g]
        if len(sub) > 0:
            ax.scatter(sub['activation_timepoint'], sub['auc'],
                       color=COLORS[g], alpha=0.6, s=25, label=label, edgecolors='none')
    valid_d = df_v[['activation_timepoint', 'auc']].dropna()
    if len(valid_d) > 2:
        r, p = stats.pearsonr(valid_d['activation_timepoint'], valid_d['auc'])
        ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3f}',
                transform=ax.transAxes, fontsize=9, va='top')
    ax.set_xlabel('Activation timepoint')
    ax.set_ylabel('AUC')
    ax.legend(fontsize=8)
    ax.set_title('D   Timing vs AUC', loc='left', fontweight='bold')

    # Panel E: plateau vs AUC
    ax = axes[1, 1]
    if 'plateau_value' in df_v.columns and 'response_group' in df_v.columns:
        for g, label in zip(groups_r, labels_r):
            sub = df_v[df_v['response_group'] == g].dropna(subset=['plateau_value', 'auc'])
            if len(sub) > 0:
                ax.scatter(sub['plateau_value'], sub['auc'],
                           color=RESPONSE_COLORS[g], alpha=0.6, s=25, label=label, edgecolors='none')
        valid_e = df_v[['plateau_value', 'auc']].dropna()
        if len(valid_e) > 2:
            r2, p2 = stats.pearsonr(valid_e['plateau_value'], valid_e['auc'])
            ax.text(0.05, 0.95, f'r = {r2:.2f}, p = {p2:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top')
        ax.legend(fontsize=8)
    ax.set_xlabel('Plateau value')
    ax.set_ylabel('AUC')
    ax.set_title('E   Plateau vs AUC', loc='left', fontweight='bold')

    # Panel F: AUC CDF by timing group
    ax = axes[1, 2]
    for g, label in zip(groups_a, labels_a):
        sub = df_v[df_v['activation_group'] == g]['auc'].dropna().sort_values()
        if len(sub) > 0:
            cdf = np.arange(1, len(sub) + 1) / len(sub) * 100
            ax.plot(sub.values, cdf, color=COLORS[g], linewidth=2, label=label)
    ax.set_xlabel('AUC')
    ax.set_ylabel('Cumulative %')
    ax.legend(fontsize=8)
    ax.set_title('F   AUC CDF by Timing Group', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir, f"well_{full_name}_auc{suffix_clean}", save_pdf, save_svg)
    plt.close()
    print("  Saved AUC analysis")


def plot_survival_hazard(df_surv, output_dir, well, save_pdf=False, save_svg=False, suffix=""):
    """
    Kaplan-Meier survival curve and hazard function — 3-panel figure:
    A  KM survival curve  S(t) with 95% CI
    B  Cumulative activation  1 - S(t) with 95% CI
    C  Discrete + smoothed hazard h(t)
    """
    if df_surv is None or len(df_surv) == 0:
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Survival & Hazard Analysis — Well {full_name}{suffix}',
                 fontsize=14, fontweight='bold')

    # Panel A: KM survival
    ax = axes[0]
    ax.step(df_surv['timepoint'], df_surv['survival'] * 100,
            where='post', color='#2166AC', linewidth=2.5, label='KM estimate')
    ax.fill_between(df_surv['timepoint'],
                    df_surv['survival_lower'] * 100, df_surv['survival_upper'] * 100,
                    step='post', color='#2166AC', alpha=0.15, label='95% CI')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('% not yet activated')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title('A   Kaplan-Meier Survival', loc='left', fontweight='bold')

    # Panel B: cumulative activation
    ax = axes[1]
    ax.step(df_surv['timepoint'], (1 - df_surv['survival']) * 100,
            where='post', color='#4DAF4A', linewidth=2.5)
    ax.fill_between(df_surv['timepoint'],
                    (1 - df_surv['survival_upper']) * 100,
                    (1 - df_surv['survival_lower']) * 100,
                    step='post', color='#4DAF4A', alpha=0.15)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Cumulative % activated')
    ax.set_ylim(0, 100)
    ax.set_title('B   Cumulative Activation (KM)', loc='left', fontweight='bold')

    # Panel C: hazard function
    ax = axes[2]
    ax.bar(df_surv['timepoint'], df_surv['hazard'] * 100,
           color='#BDBDBD', alpha=0.6, width=0.8, label='Discrete h(t)')
    ax.plot(df_surv['timepoint'], df_surv['hazard_smooth'] * 100,
            color='#D62728', linewidth=2.5, label='Smoothed')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Hazard (%/frame)')
    ax.legend(fontsize=8)
    ax.set_title('C   Hazard Function h(t)', loc='left', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, fig_dir, f"well_{full_name}_survival_hazard{suffix_clean}", save_pdf, save_svg)
    plt.close()
    print("  Saved survival & hazard analysis")


# ==================== SPATIAL PROPAGATION FUNCTIONS ====================

def compute_cell_positions(df, df_meas, baseline_frames=(0, 5)):
    """
    Add x_pos, y_pos columns to df using mean centroid during baseline frames.
    Works on any df containing unique_track_id (df_act or df_all_tracks).
    """
    if 'centroid-0' not in df_meas.columns:
        print("  Warning: centroid columns not found — spatial analysis unavailable")
        df = df.copy()
        df['x_pos'] = np.nan
        df['y_pos'] = np.nan
        return df

    start_t, end_t = baseline_frames
    baseline = df_meas[
        (df_meas['timepoint'] >= start_t) & (df_meas['timepoint'] <= end_t)
    ]
    pos = baseline.groupby('unique_track_id')[['centroid-0', 'centroid-1']].mean()
    df = df.copy()
    df['y_pos'] = df['unique_track_id'].map(pos['centroid-0'])
    df['x_pos'] = df['unique_track_id'].map(pos['centroid-1'])
    return df


def compute_spatial_stats(df_act, n_neighbors=10, n_permutations=999):
    """
    Per-FOV spatial autocorrelation of activation timing using Moran's I.

    Weight matrix: row-standardised k-nearest neighbours.
    Significance: permutation test (shuffle activation times within FOV).

    Also returns:
      pairs  — all pairwise (distance, |ΔT|) data for distance-ΔT plot
      lag    — spatial lag table (own activation time, mean-neighbour time)
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist, squareform

    if 'x_pos' not in df_act.columns:
        print("  Warning: run compute_cell_positions first — spatial stats skipped")
        return None

    per_fov  = {}
    all_pairs = []
    all_lag   = []

    for fov, fov_df in df_act.groupby('fov'):
        sub = fov_df.dropna(subset=['x_pos', 'y_pos', 'activation_timepoint']).copy()
        n   = len(sub)
        k   = min(n_neighbors, n - 1)
        if n < 4:
            print(f"  FOV {fov}: only {n} activating cells — skipping")
            continue

        positions = sub[['x_pos', 'y_pos']].values
        times     = sub['activation_timepoint'].values.astype(float)

        # Row-standardised KNN weight matrix
        tree = cKDTree(positions)
        _, indices = tree.query(positions, k=k + 1)   # k+1 includes self at index 0
        W = np.zeros((n, n))
        for i in range(n):
            neighbors = indices[i, 1:]
            if len(neighbors) > 0:
                W[i, neighbors] = 1.0 / len(neighbors)

        # Moran's I (W already row-standardised so S0 = N)
        def _morans_i(t):
            z = t - t.mean()
            denom = np.dot(z, z)
            if denom == 0:
                return np.nan
            return n * float(np.einsum('ij,i,j', W, z, z)) / denom

        obs_I  = _morans_i(times)
        null_I = np.array([_morans_i(np.random.permutation(times))
                           for _ in range(n_permutations)])
        p_val  = (np.sum(null_I >= obs_I) + 1) / (n_permutations + 1)

        per_fov[fov] = {
            'morans_i': obs_I, 'p_value': p_val,
            'null_dist': null_I, 'n_cells': n,
        }
        print(f"  FOV {fov} (n={n}): Moran's I = {obs_I:.3f}, p = {p_val:.3f}")

        # Pairwise distance vs |ΔT|
        dist_mat  = squareform(pdist(positions))
        triu_idx  = np.triu_indices(n, k=1)
        all_pairs.append(pd.DataFrame({
            'distance': dist_mat[triu_idx],
            'delta_t':  np.abs(times[triu_idx[0]] - times[triu_idx[1]]),
            'fov': fov,
        }))

        # Spatial lag: own activation time vs mean-neighbour activation time
        for i in range(n):
            nbrs = indices[i, 1:]
            if len(nbrs) > 0:
                all_lag.append({
                    'own_t': times[i],
                    'lag_t': times[nbrs].mean(),
                    'fov': fov,
                })

    return {
        'per_fov': per_fov,
        'pairs':   pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(),
        'lag':     pd.DataFrame(all_lag) if all_lag else pd.DataFrame(),
    }


def plot_spatial_propagation(df_act, df_all, spatial_stats, output_dir, well,
                              save_pdf=False, save_svg=False, suffix="",
                              timepoint_min=None, timepoint_max=None):
    """
    Figure 1 — well_*_spatial_maps.png
        One panel per FOV: cells coloured by activation time, non-activating in grey.
        Moran's I + p-value annotated on each panel.

    Figure 2 — well_*_spatial_propagation.png
        A  Binned distance vs mean |ΔActivation time|
        B  Spatial lag: own activation time vs mean-neighbour time
        C  Moran's I per FOV (red bar = p < 0.05)
    """
    if spatial_stats is None or len(spatial_stats['per_fov']) == 0:
        print("  Skipping spatial propagation plots: no stats available")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    sc = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)

    # ── Figure 1: spatial maps ──────────────────────────────────────────────
    fovs   = sorted(df_act['fov'].unique())
    n_fovs = len(fovs)
    n_cols = min(3, n_fovs)
    n_rows = int(np.ceil(n_fovs / n_cols))
    cmap   = plt.cm.plasma
    norm   = plt.Normalize(vmin=t_min, vmax=t_max)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle(f'Spatial Activation Maps — Well {full_name}{suffix}',
                 fontsize=14, fontweight='bold')

    for idx, fov in enumerate(fovs):
        ax = axes[idx // n_cols][idx % n_cols]

        # Non-activating cells
        if df_all is not None and 'x_pos' in df_all.columns:
            non_act = df_all[(df_all['fov'] == fov) &
                             (df_all['activates'] == False)].dropna(subset=['x_pos', 'y_pos'])
            if len(non_act) > 0:
                ax.scatter(non_act['x_pos'], non_act['y_pos'],
                           c='#BDBDBD', s=12, alpha=0.5, linewidths=0)

        # Activating cells coloured by activation time
        act = df_act[df_act['fov'] == fov].dropna(subset=['x_pos', 'y_pos', 'activation_timepoint'])
        if len(act) > 0:
            ax.scatter(act['x_pos'], act['y_pos'],
                       c=act['activation_timepoint'], cmap=cmap, norm=norm,
                       s=30, alpha=0.9, linewidths=0.3, edgecolors='white')

        # Moran's I annotation
        if fov in spatial_stats['per_fov']:
            sf  = spatial_stats['per_fov'][fov]
            p_s = f"p={sf['p_value']:.3f}" if sf['p_value'] >= 0.001 else "p<0.001"
            ax.text(0.05, 0.97, f"I={sf['morans_i']:.2f}, {p_s}",
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        ax.set_title(f'FOV {fov}  (n={len(act)} activating)', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')

    for idx in range(n_fovs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(),
                 label='Activation timepoint', shrink=0.6, pad=0.02)
    plt.tight_layout()
    save_figure(fig, fig_dir, f"well_{full_name}_spatial_maps{sc}", save_pdf, save_svg)
    plt.close()
    print("  Saved spatial activation maps")

    # ── Figure 2: statistics ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Spatial Propagation Analysis — Well {full_name}{suffix}',
                 fontsize=14, fontweight='bold')

    # Panel A: distance vs |ΔT| binned
    ax = axes[0]
    pairs = spatial_stats['pairs'].dropna()
    if len(pairs) > 10:
        pairs['dist_bin'] = pd.qcut(pairs['distance'], q=10, duplicates='drop')
        binned      = pairs.groupby('dist_bin')['delta_t'].agg(['mean', 'sem'])
        bin_centers = [b.mid for b in binned.index]
        ax.errorbar(bin_centers, binned['mean'], yerr=binned['sem'],
                    fmt='o-', color='#2166AC', linewidth=2, markersize=5, capsize=3)
        lr = stats.linregress(pairs['distance'], pairs['delta_t'])
        x_line = np.linspace(pairs['distance'].min(), pairs['distance'].max(), 100)
        ax.plot(x_line, lr.slope * x_line + lr.intercept,
                '--', color='#D62728', linewidth=1.5,
                label=f'r={lr.rvalue:.2f}, p={lr.pvalue:.3f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Distance between cells (px)')
    ax.set_ylabel('Mean |ΔActivation time| (frames)')
    ax.set_title('A   Distance vs ΔActivation Time', loc='left', fontweight='bold')

    # Panel B: spatial lag
    ax = axes[1]
    lag = spatial_stats['lag'].dropna()
    if len(lag) > 2:
        fov_list   = sorted(lag['fov'].unique())
        fov_colors = plt.cm.tab10(np.linspace(0, 0.9, len(fov_list)))
        for fov_i, col_i in zip(fov_list, fov_colors):
            sub = lag[lag['fov'] == fov_i]
            ax.scatter(sub['own_t'], sub['lag_t'],
                       color=col_i, alpha=0.5, s=20, edgecolors='none',
                       label=f'FOV {fov_i}')
        t_range = [lag[['own_t', 'lag_t']].min().min(),
                   lag[['own_t', 'lag_t']].max().max()]
        ax.plot(t_range, t_range, 'k--', linewidth=1, alpha=0.4)
        r, p = stats.pearsonr(lag['own_t'], lag['lag_t'])
        ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3f}',
                transform=ax.transAxes, fontsize=9, va='top')
        ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel('Cell activation time (frames)')
    ax.set_ylabel('Mean neighbour activation time (frames)')
    ax.set_title('B   Spatial Lag Correlation', loc='left', fontweight='bold')

    # Panel C: Moran's I per FOV
    ax = axes[2]
    pf = spatial_stats['per_fov']
    if pf:
        fov_names = list(pf.keys())
        I_vals    = [pf[f]['morans_i'] for f in fov_names]
        p_vals    = [pf[f]['p_value']  for f in fov_names]
        null_95   = [np.percentile(pf[f]['null_dist'], 95) for f in fov_names]
        bar_cols  = ['#D62728' if p < 0.05 else '#BDBDBD' for p in p_vals]
        ax.bar([str(f) for f in fov_names], I_vals,
               color=bar_cols, alpha=0.8, edgecolor='black', width=0.5)
        ax.scatter([str(f) for f in fov_names], null_95,
                   color='black', marker='_', s=120, zorder=5,
                   label='95th pct null')
        ax.axhline(0, color='black', linewidth=0.8)
        legend_elems = [Patch(facecolor='#D62728', label='p < 0.05'),
                        Patch(facecolor='#BDBDBD', label='p ≥ 0.05')]
        ax.legend(handles=legend_elems, fontsize=8)
        ax.set_xlabel('FOV')
        ax.set_ylabel("Moran's I")
    ax.set_title("C   Moran's I per FOV", loc='left', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, fig_dir, f"well_{full_name}_spatial_propagation{sc}", save_pdf, save_svg)
    plt.close()
    print("  Saved spatial propagation analysis")


# ==================== TRAJECTORY CLUSTERING FUNCTIONS ====================

def compute_trajectory_clusters(df_act, r2_min=0.7, n_clusters_range=(2, 7),
                                  use_umap=True, umap_n_neighbors=15, umap_min_dist=0.1,
                                  max_feature_zscore=4.0, min_cluster_frac=0.05,
                                  random_state=42):
    """
    Cluster activating cells by their kinetic fingerprint using UMAP + K-means.

    Features used (all from sigmoid fit in script 2):
        activation_timepoint, sigmoid_k, plateau_value, sigmoid_baseline, max_slope

    Steps:
      1. Keep only cells with sigmoid_r2 >= r2_min (good fits).
      2. StandardScaler-normalise the feature matrix.
      3. Flag feature-space outliers: cells with max |z-score| > max_feature_zscore
         in ANY feature are labelled traj_cluster = -2 and excluded from embedding
         and K-means (prevents tiny outlier micro-clusters from dominating the
         silhouette score and producing a degenerate K=2 solution).
      4. PCA (always computed — loadings available for interpretation).
      5. UMAP 2-D embedding on inliers (falls back to PCA 2-D if umap-learn absent).
      6. K-means sweep: for each K, reject solutions where any cluster has fewer
         than min_cluster_frac * n_inliers cells. Pick K with highest silhouette
         score among valid solutions.
      7. Write columns back to df_act:
           traj_cluster  (-2 = feature outlier, -1 = poor fit, ≥0 = cluster id)
           embed_x / embed_y  (2-D coords; NaN for outliers/poor fits)
           embedding_method  ('umap' or 'pca')

    Returns
    -------
    df_act      : copy with new columns
    cluster_info: dict with keys 'method', 'n_clusters', 'silhouette_scores',
                  'pca', 'feature_cols', 'scaler', 'n_outliers'
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    FEATURE_COLS = ['activation_timepoint', 'sigmoid_k', 'plateau_value',
                    'sigmoid_baseline', 'max_slope']

    df_out = df_act.copy()
    df_out['traj_cluster']      = -1
    df_out['embed_x']           = np.nan
    df_out['embed_y']           = np.nan
    df_out['embedding_method']  = 'none'

    # Require sigmoid_r2 and at least 2 feature columns
    available = [c for c in FEATURE_COLS if c in df_out.columns]
    if len(available) < 2:
        print(f"  Warning: only {len(available)} feature columns available — skipping clustering")
        return df_out, None

    good_fit = (
        (df_out['sigmoid_r2'] >= r2_min) if 'sigmoid_r2' in df_out.columns
        else pd.Series(True, index=df_out.index)
    )
    df_fit = df_out.loc[good_fit, available].dropna()
    idx    = df_fit.index

    if len(idx) < max(n_clusters_range):
        print(f"  Warning: only {len(idx)} well-fit cells — skipping clustering")
        return df_out, None

    # 1. Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_fit.values)

    # 2. Flag feature-space outliers before embedding
    max_abs_z   = np.abs(X_scaled).max(axis=1)   # worst-case z across all features
    inlier_mask = max_abs_z <= max_feature_zscore
    n_outliers  = (~inlier_mask).sum()
    if n_outliers:
        outlier_idx = idx[~inlier_mask]
        df_out.loc[outlier_idx, 'traj_cluster'] = -2
        print(f"  Flagged {n_outliers} feature-space outliers (max |z| > {max_feature_zscore})")

    X_inlier = X_scaled[inlier_mask]
    idx_in   = idx[inlier_mask]

    if len(idx_in) < max(n_clusters_range):
        print(f"  Warning: only {len(idx_in)} inlier cells after outlier removal — skipping K-means")
        return df_out, None

    print(f"  Clustering {len(idx_in)} inlier cells on features: {available}")

    # 3. PCA (always, for loadings/scree)
    pca   = PCA(random_state=random_state)
    X_pca = pca.fit_transform(X_inlier)

    # 4. UMAP or PCA-2D embedding (on inliers only)
    method  = 'pca'
    X_embed = X_pca[:, :2]
    if use_umap:
        try:
            import umap
            n_nbrs  = min(umap_n_neighbors, len(idx_in) - 1)
            reducer = umap.UMAP(n_neighbors=n_nbrs, min_dist=umap_min_dist,
                                 n_components=2, random_state=random_state)
            X_embed = reducer.fit_transform(X_inlier)
            method  = 'umap'
            print(f"  UMAP embedding computed  (n_neighbors={n_nbrs})")
        except ImportError:
            print("  umap-learn not installed — falling back to PCA 2-D embedding")

    # 5. K-means sweep with minimum-cluster-size guard
    min_cluster_size = max(1, int(min_cluster_frac * len(idx_in)))
    sil_scores       = {}
    for k in range(n_clusters_range[0], n_clusters_range[1]):
        km  = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X_embed)
        counts = np.bincount(lbl)
        if counts.min() < min_cluster_size:
            print(f"  K={k}: smallest cluster={counts.min()} < min_size={min_cluster_size} — skipped")
            continue
        sil = silhouette_score(X_embed, lbl)
        sil_scores[k] = sil

    if not sil_scores:
        print("  No valid K found (all cluster solutions have a micro-cluster). "
              "Try --cluster-min-frac 0 to disable the guard.")
        return df_out, None

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"  Silhouette scores: { {k: f'{v:.3f}' for k, v in sil_scores.items()} }")
    print(f"  Best K = {best_k}  (silhouette = {sil_scores[best_k]:.3f})")

    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels   = km_final.fit_predict(X_embed)

    # 6. Write back to df_out (inliers only)
    df_out.loc[idx_in, 'traj_cluster']     = labels
    df_out.loc[idx_in, 'embed_x']          = X_embed[:, 0]
    df_out.loc[idx_in, 'embed_y']          = X_embed[:, 1]
    df_out.loc[idx_in, 'embedding_method'] = method

    cluster_info = {
        'method':            method,
        'n_clusters':        best_k,
        'silhouette_scores': sil_scores,
        'pca':               pca,
        'feature_cols':      available,
        'scaler':            scaler,
        'n_outliers':        int(n_outliers),
    }
    return df_out, cluster_info


def plot_trajectory_clusters(df_act, df_meas, cluster_info, output_dir, well,
                              save_pdf=False, save_svg=False, suffix="",
                              signal_col='mean_intensity',
                              timepoint_min=None, timepoint_max=None):
    """
    Three figures:

    1. well_*_trajectory_clusters.png — 6-panel embedding scatter
       A  coloured by K-means cluster label
       B  coloured by activation timing group (early/average/late)
       C  coloured by response group (low/medium/high)
       D  coloured by activation_timepoint (continuous)
       E  coloured by plateau_value (continuous)
       F  coloured by sigmoid_k / steepness (continuous)

    2. well_*_trajectory_cluster_profiles.png — per-cluster mean ± SD trajectories

    3. well_*_trajectory_cluster_features.png — PCA loadings + per-cluster feature boxes
    """
    if cluster_info is None or df_act['traj_cluster'].eq(-1).all():
        print("  Skipping trajectory cluster plots: clustering not available")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    sc       = f"_{suffix}" if suffix else ""
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    fig_dir  = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    method  = cluster_info['method'].upper()
    xlab    = f"{method} 1"
    ylab    = f"{method} 2"
    n_outliers = cluster_info.get('n_outliers', 0)

    # traj_cluster >= 0 → inlier cells assigned to a cluster
    # traj_cluster == -2 → feature-space outliers (shown separately)
    df_fit      = df_act[df_act['traj_cluster'] >= 0].copy()
    df_outliers = df_act[df_act['traj_cluster'] == -2].copy()
    n_clusters  = cluster_info['n_clusters']
    cluster_palette = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))

    # ── Figure 1: embedding scatter (6 panels) ─────────────────────────────
    n_out_label = f"  ({n_outliers} outliers excluded)" if n_outliers else ""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'Trajectory Clustering — Well {full_name}{suffix}  ({method}){n_out_label}',
                 fontsize=14, fontweight='bold', y=0.98)

    ex, ey = df_fit['embed_x'].values, df_fit['embed_y'].values

    def _add_outlier_scatter(ax):
        """Overlay outlier points as grey X markers (no embed coords → skip)."""
        if n_outliers:
            ax.text(0.02, 0.98, f'{n_outliers} outlier(s) not shown',
                    transform=ax.transAxes, fontsize=7, va='top', color='grey')

    # A — cluster labels
    ax = axes[0, 0]
    for k in sorted(df_fit['traj_cluster'].unique()):
        m  = df_fit['traj_cluster'] == k
        ax.scatter(ex[m], ey[m], c=[cluster_palette[k]], s=15, alpha=0.7,
                   label=f'Cluster {k}  (n={m.sum()})', rasterized=True)
    _add_outlier_scatter(ax)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.legend(fontsize=7, markerscale=1.5)
    ax.set_title('A   K-means Clusters', loc='left', fontweight='bold')

    # B — timing group
    ax = axes[0, 1]
    for grp, col in COLORS.items():
        if grp not in ('early', 'average', 'late'):
            continue
        m = df_fit['activation_group'] == grp if 'activation_group' in df_fit.columns else np.zeros(len(df_fit), dtype=bool)
        if m.sum():
            ax.scatter(ex[m], ey[m], c=col, s=15, alpha=0.7, label=grp.capitalize(), rasterized=True)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.legend(fontsize=7, markerscale=1.5)
    ax.set_title('B   Timing Group', loc='left', fontweight='bold')
    _add_outlier_scatter(ax)

    # C — response group
    ax = axes[0, 2]
    for grp, col in RESPONSE_COLORS.items():
        if grp == 'unfit':
            continue
        m = df_fit['response_group'] == grp if 'response_group' in df_fit.columns else np.zeros(len(df_fit), dtype=bool)
        if m.sum():
            ax.scatter(ex[m], ey[m], c=col, s=15, alpha=0.7, label=grp.capitalize(), rasterized=True)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.legend(fontsize=7, markerscale=1.5)
    ax.set_title('C   Response Group', loc='left', fontweight='bold')
    _add_outlier_scatter(ax)

    # D — activation timepoint (continuous)
    ax = axes[1, 0]
    sc_d = ax.scatter(ex, ey, c=df_fit['activation_timepoint'].values, cmap='plasma',
                      s=15, alpha=0.7, rasterized=True)
    plt.colorbar(sc_d, ax=ax, label='Activation timepoint')
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title('D   Activation Timepoint', loc='left', fontweight='bold')
    _add_outlier_scatter(ax)

    # E — plateau value (continuous)
    ax = axes[1, 1]
    if 'plateau_value' in df_fit.columns:
        sc_e = ax.scatter(ex, ey, c=df_fit['plateau_value'].values, cmap='YlOrRd',
                          s=15, alpha=0.7, rasterized=True)
        plt.colorbar(sc_e, ax=ax, label='Plateau value')
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title('E   Plateau Value', loc='left', fontweight='bold')
    _add_outlier_scatter(ax)

    # F — sigmoid_k (steepness, continuous)
    ax = axes[1, 2]
    if 'sigmoid_k' in df_fit.columns:
        sc_f = ax.scatter(ex, ey, c=df_fit['sigmoid_k'].values, cmap='cool',
                          s=15, alpha=0.7, rasterized=True)
        plt.colorbar(sc_f, ax=ax, label='sigmoid_k (steepness)')
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title('F   Sigmoid Steepness (k)', loc='left', fontweight='bold')
    _add_outlier_scatter(ax)

    plt.tight_layout()
    save_figure(fig, fig_dir, f"well_{full_name}_trajectory_clusters{sc}", save_pdf, save_svg)
    plt.close()
    print("  Saved trajectory cluster embedding")

    # ── Figure 2: per-cluster mean ± SD trajectories ───────────────────────
    _ycol = signal_col if signal_col in df_meas.columns else 'mean_intensity'
    ncols  = min(n_clusters, 4)
    nrows  = (n_clusters + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                               squeeze=False)
    fig2.suptitle(f'Cluster Trajectories — Well {full_name}{suffix}',
                  fontsize=14, fontweight='bold', y=0.98)

    for k in sorted(df_fit['traj_cluster'].unique()):
        r, c = divmod(k, ncols)
        ax   = axes2[r][c]
        ids  = df_fit[df_fit['traj_cluster'] == k]['unique_track_id'].values
        traj = df_meas[df_meas['unique_track_id'].isin(ids)].copy()
        traj = traj[(traj['timepoint'] >= t_min) & (traj['timepoint'] <= t_max)]

        if len(traj):
            pivot = traj.pivot_table(index='timepoint', columns='unique_track_id',
                                     values=_ycol, aggfunc='mean')
            mean_t = pivot.mean(axis=1)
            std_t  = pivot.std(axis=1)
            tp     = mean_t.index.values
            ax.fill_between(tp, (mean_t - std_t).values, (mean_t + std_t).values,
                            alpha=0.25, color=cluster_palette[k])
            ax.plot(tp, mean_t.values, color=cluster_palette[k], linewidth=2)

        n_k  = len(ids)
        # cluster summary stats
        sub  = df_fit[df_fit['traj_cluster'] == k]
        t50  = sub['activation_timepoint'].median()
        plat = sub['plateau_value'].median() if 'plateau_value' in sub.columns else np.nan
        k_m  = sub['sigmoid_k'].median() if 'sigmoid_k' in sub.columns else np.nan
        ax.set_title(f'Cluster {k}  (n={n_k})\n'
                     f'median T½={t50:.1f}  plateau={plat:.2f}  k={k_m:.3f}',
                     fontsize=8, loc='left')
        ax.set_xlabel('Timepoint'); ax.set_ylabel(_ycol.replace('_', ' '))
        ax.axvline(t50, color=cluster_palette[k], linestyle='--', linewidth=0.8, alpha=0.6)

    # Hide empty panels
    for k in range(n_clusters, nrows * ncols):
        r, c = divmod(k, ncols)
        axes2[r][c].set_visible(False)

    plt.tight_layout()
    save_figure(fig2, fig_dir, f"well_{full_name}_trajectory_cluster_profiles{sc}", save_pdf, save_svg)
    plt.close()
    print("  Saved cluster trajectory profiles")

    # ── Figure 3: PCA loadings + per-cluster feature distributions ─────────
    feat_cols  = cluster_info['feature_cols']
    pca_obj    = cluster_info['pca']
    n_feat     = len(feat_cols)
    n_comp_show = min(3, pca_obj.n_components_)

    # layout: top row = loadings heatmap + scree; bottom rows = feature boxes
    n_box_rows = (n_feat + 2) // 3
    total_rows = 1 + n_box_rows
    fig3, axes3 = plt.subplots(total_rows, 3, figsize=(13, 3.5 * total_rows))
    fig3.suptitle(f'Cluster Feature Analysis — Well {full_name}{suffix}',
                  fontsize=14, fontweight='bold', y=0.98)

    # Top-left: loadings heatmap (PC1-3 × features)
    ax = axes3[0, 0]
    loadings = pca_obj.components_[:n_comp_show, :]   # shape (n_comp, n_feat)
    im = ax.imshow(loadings, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels([f.replace('_', '\n') for f in feat_cols], fontsize=7)
    ax.set_yticks(range(n_comp_show))
    ax.set_yticklabels([f'PC{i+1}' for i in range(n_comp_show)])
    plt.colorbar(im, ax=ax, shrink=0.8, label='Loading')
    ax.set_title('A   PCA Loadings (PC1-3)', loc='left', fontweight='bold')

    # Top-middle: scree plot
    ax = axes3[0, 1]
    evr = pca_obj.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    ax.bar(range(1, len(evr) + 1), evr * 100, color='steelblue', alpha=0.7)
    ax.plot(range(1, len(evr) + 1), cumvar * 100, 'o-', color='tomato', linewidth=1.5)
    ax.axhline(80, linestyle='--', color='grey', linewidth=0.8)
    ax.set_xlabel('PC'); ax.set_ylabel('Variance explained (%)')
    ax.set_title('B   Scree Plot', loc='left', fontweight='bold')

    # Top-right: silhouette score bar chart
    ax = axes3[0, 2]
    ks   = list(cluster_info['silhouette_scores'].keys())
    sils = [cluster_info['silhouette_scores'][k] for k in ks]
    bar_colors = ['tomato' if k == cluster_info['n_clusters'] else 'steelblue' for k in ks]
    ax.bar(ks, sils, color=bar_colors, alpha=0.8)
    ax.set_xlabel('Number of clusters K')
    ax.set_ylabel('Silhouette score')
    ax.set_xticks(ks)
    ax.set_title('C   Cluster Selection', loc='left', fontweight='bold')

    # Bottom panels: box plot per feature (one panel per feature)
    panel_idx = 3   # index into flattened axes3
    axes3_flat = axes3.flatten()
    for fi, feat in enumerate(feat_cols):
        if panel_idx >= len(axes3_flat):
            break
        ax = axes3_flat[panel_idx]
        panel_idx += 1
        if feat not in df_fit.columns:
            ax.set_visible(False)
            continue
        data = [df_fit[df_fit['traj_cluster'] == k][feat].dropna().values
                for k in sorted(df_fit['traj_cluster'].unique())]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, k in zip(bp['boxes'], sorted(df_fit['traj_cluster'].unique())):
            patch.set_facecolor(cluster_palette[k])
            patch.set_alpha(0.7)
        ax.set_xticklabels([f'C{k}' for k in sorted(df_fit['traj_cluster'].unique())])
        ax.set_ylabel(feat.replace('_', ' '))
        letter = chr(ord('D') + fi)
        ax.set_title(f'{letter}   {feat.replace("_", " ").title()}', loc='left', fontweight='bold')

    for i in range(panel_idx, len(axes3_flat)):
        axes3_flat[i].set_visible(False)

    plt.tight_layout()
    save_figure(fig3, fig_dir, f"well_{full_name}_trajectory_cluster_features{sc}", save_pdf, save_svg)
    plt.close()
    print("  Saved cluster feature analysis")


def plot_baseline_analysis(df_act, df_meas, output_dir, well, baseline_frames=(0, 5), save_pdf=False, save_svg=False, suffix="", timepoint_min=None, timepoint_max=None):
    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    groups = ['early', 'average', 'late']
    group_labels = ['Early', 'Average', 'Late']
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    if 'baseline_intensity' not in df_act.columns:
        df_act = calculate_baseline_intensity(df_act, df_meas, baseline_frames)
    
    df_groups = df_act[df_act['activation_group'].isin(groups)].copy()
    df_groups['baseline_log10'] = np.log10(df_groups['baseline_intensity'])
    
    valid_data = df_groups.dropna(subset=['baseline_intensity', 'activation_timepoint'])
    
    # Calculate correlations
    corr_log10, p_value_log10 = stats.pearsonr(valid_data['baseline_log10'], valid_data['activation_timepoint'])
    corr_abs, p_value_abs = stats.pearsonr(valid_data['baseline_intensity'], valid_data['activation_timepoint'])
    
    # Calculate IQR
    q25 = valid_data['baseline_intensity'].quantile(0.25)
    q75 = valid_data['baseline_intensity'].quantile(0.75)
    median_baseline = valid_data['baseline_intensity'].median()
    iqr = q75 - q25
    
    print(f"Correlation (log₁₀ baseline vs activation): r = {corr_log10:.3f}, p = {p_value_log10:.2e}")
    print(f"Correlation (absolute baseline vs activation): r = {corr_abs:.3f}, p = {p_value_abs:.2e}")
    print(f"Baseline intensity IQR (middle 50%): {q25:.1f} - {q75:.1f} (median: {median_baseline:.1f})")
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'Baseline mNG Analysis — Well {full_name}{suffix}', fontsize=14, fontweight='bold', y=0.98)
    
    # Panel A: Boxplot with IQR
    ax = axes[0, 0]
    data = [df_groups[df_groups['activation_group'] == g]['baseline_intensity'].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=group_labels, patch_artist=True, widths=0.6)
    for patch, group in zip(bp['boxes'], groups):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.7)
    for i, (group, d) in enumerate(zip(groups, data)):
        x = np.random.normal(i + 1, 0.08, size=len(d))
        ax.scatter(x, d, alpha=0.4, color=COLORS[group], s=20)
    
    ax.axhspan(q25, q75, alpha=0.1, color='#FFD700', zorder=0)
    ax.axhline(q25, color='#FF8C00', linestyle='--', linewidth=1, alpha=0.6, label=f'Q25: {q25:.0f}')
    ax.axhline(q75, color='#FF8C00', linestyle='--', linewidth=1, alpha=0.6, label=f'Q75: {q75:.0f}')
    _pairwise_sig_brackets(ax, data, labels=group_labels)
    ax.set_ylabel('Baseline mNG intensity (a.u.)')
    ax.set_title('A   Baseline by group', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    
    # Panel B: Violin (skip empty groups to avoid violinplot error)
    ax = axes[0, 1]
    valid_data_b = [d for d in data if len(d) > 0]
    valid_groups_b = [g for g, d in zip(groups, data) if len(d) > 0]
    valid_labels_b = [group_labels[groups.index(g)] for g in valid_groups_b]
    if valid_data_b:
        positions_b = list(range(1, len(valid_data_b) + 1))
        parts = ax.violinplot(valid_data_b, positions=positions_b, showmeans=True, showmedians=True)
        for i, (pc, group) in enumerate(zip(parts['bodies'], valid_groups_b)):
            pc.set_facecolor(COLORS[group])
            pc.set_alpha(0.6)
        ax.set_xticks(positions_b)
        ax.set_xticklabels(valid_labels_b)
    else:
        ax.text(0.5, 0.5, 'No group data', transform=ax.transAxes, ha='center', va='center', fontsize=11)
    ax.set_ylabel('Baseline mNG intensity (a.u.)')
    ax.set_title('B   Distribution', loc='left', fontweight='bold')
    
    # Panel C: Scatter correlation (LOG10)
    ax = axes[0, 2]
    for group in groups:
        gdata = df_groups[df_groups['activation_group'] == group]
        ax.scatter(gdata['baseline_log10'], gdata['activation_timepoint'],
                  color=COLORS[group], alpha=0.6, s=40, label=group.capitalize(), edgecolors='white', linewidths=0.5)
    
    valid = df_groups.dropna(subset=['baseline_log10', 'activation_timepoint'])
    z = np.polyfit(valid['baseline_log10'], valid['activation_timepoint'], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(valid['baseline_log10'].min(), valid['baseline_log10'].max(), 100)
    ax.plot(x_line, p_fit(x_line), 'k--', linewidth=2, alpha=0.7)
    ax.text(0.02, 0.98, f'r = {corr_log10:.2f}\np = {p_value_log10:.2e}', transform=ax.transAxes, ha='left', va='top', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax.set_xlabel('Baseline mNG intensity ($\log_{10}$)')
    ax.set_ylabel('Activation time (frames)')
    ax.set_title('C   Correlation (log scale)', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel D: Scatter correlation (ABSOLUTE)
    ax = axes[1, 0]
    
    y_min, y_max = df_groups['activation_timepoint'].min(), df_groups['activation_timepoint'].max()
    ax.axvspan(q25, q75, alpha=0.15, color='#FFD700', label=f'IQR: {q25:.0f}-{q75:.0f}', zorder=0)
    ax.axvline(median_baseline, color='#FF8C00', linestyle=':', linewidth=1.5, alpha=0.8, zorder=1)
    
    for group in groups:
        gdata = df_groups[df_groups['activation_group'] == group]
        ax.scatter(gdata['baseline_intensity'], gdata['activation_timepoint'],
                  color=COLORS[group], alpha=0.6, s=40, label=group.capitalize(), edgecolors='white', linewidths=0.5, zorder=2)
    
    valid = df_groups.dropna(subset=['baseline_intensity', 'activation_timepoint'])
    z = np.polyfit(valid['baseline_intensity'], valid['activation_timepoint'], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(valid['baseline_intensity'].min(), valid['baseline_intensity'].max(), 100)
    ax.plot(x_line, p_fit(x_line), 'k--', linewidth=2, alpha=0.7, zorder=3)
    ax.text(0.02, 0.98, f'r = {corr_abs:.2f}\np = {p_value_abs:.2e}', transform=ax.transAxes, ha='left', va='top', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax.set_xlabel('Baseline mNG intensity (a.u.)')
    ax.set_ylabel('Activation time (frames)')
    ax.set_title('D   Correlation (absolute)', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    
    # Panel E: Low vs high baseline trajectories
    ax = axes[1, 1]
    median_baseline = df_groups['baseline_intensity'].median()
    df_low = df_groups[df_groups['baseline_intensity'] <= median_baseline]
    df_high = df_groups[df_groups['baseline_intensity'] > median_baseline]
    
    for df_subset, label, color, ls in [(df_low, f'Low (n={len(df_low)})', '#1f77b4', '-'),
                                         (df_high, f'High (n={len(df_high)})', '#d62728', '--')]:
        track_ids = df_subset['unique_track_id'].values
        subset_data = df_meas[df_meas['unique_track_id'].isin(track_ids)]
        mean_traj = subset_data.groupby('timepoint')['mean_intensity'].agg(['mean', 'std'])
        ax.plot(mean_traj.index, mean_traj['mean'], color=color, linewidth=2, linestyle=ls, label=label)
        ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                       mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=color)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('mNG intensity (a.u.)')
    ax.set_title('E   Low vs High baseline', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(t_min, t_max)
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Correlation Summary\n" + "="*25 + "\n\n"
    summary_text += f"Log₁₀ baseline:\n  r = {corr_log10:.3f}\n  p = {p_value_log10:.2e}\n\n"
    summary_text += f"Absolute baseline:\n  r = {corr_abs:.3f}\n  p = {p_value_abs:.2e}\n\n"
    summary_text += "="*25 + "\n"
    summary_text += f"n = {len(valid_data)} cells\n\n"
    summary_text += "Baseline Intensity\n" + "-"*20 + "\n"
    summary_text += f"  Median: {median_baseline:.1f}\n"
    summary_text += f"  IQR (middle 50%):\n"
    summary_text += f"    {q25:.1f} - {q75:.1f}\n"
    summary_text += f"  IQR width: {iqr:.1f}"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9, fontfamily='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', edgecolor='#4682b4', alpha=0.9))
    ax.set_title('F   Summary', loc='left', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_baseline_analysis_panel" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "")
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()
    
    iqr_stats = {'q25': q25, 'q75': q75, 'median': median_baseline, 'iqr': iqr}
    return df_act, corr_log10, p_value_log10, corr_abs, p_value_abs, iqr_stats


def calculate_max_gfp_top6(df_act, df_meas, window=6):
    """
    Calculate max mNG intensity per cell as the mean of the <window> highest
    intensity timepoints observed after the activation timepoint.
    Adds 'max_gfp_top6' column to df_act.
    """
    max_gfp_values = {}
    for _, row in df_act.iterrows():
        tid   = row['unique_track_id']
        t_act = row.get('activation_timepoint', np.nan)
        track = df_meas[df_meas['unique_track_id'] == tid]
        if 'mean_intensity' not in track.columns or len(track) == 0:
            max_gfp_values[tid] = np.nan
            continue
        if not pd.isna(t_act):
            track = track[track['timepoint'] >= t_act]
        vals = track['mean_intensity'].dropna()
        if len(vals) == 0:
            max_gfp_values[tid] = np.nan
            continue
        max_gfp_values[tid] = vals.nlargest(min(window, len(vals))).mean()
    df_act = df_act.copy()
    df_act['max_gfp_top6'] = df_act['unique_track_id'].map(max_gfp_values)
    return df_act


def plot_max_gfp_distribution(df_act, df_meas, output_dir, well, save_pdf=False, save_svg=False,
                              save_individual=False, suffix="", timepoint_min=None, timepoint_max=None):
    """
    Plot the distribution of max mNG intensities (top 6 average) by activation group.
    Includes normality tests and overlaid normal distribution fits.
    Also includes ungrouped (all cells) analysis.
    """
    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    groups = ['early', 'average', 'late']
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Calculate max mNG if not already done
    if 'max_gfp_top6' not in df_act.columns:
        df_act = calculate_max_gfp_top6(df_act, df_meas)
    
    df_groups = df_act[df_act['activation_group'].isin(groups)].copy()
    
    # ========== FIGURE 1: BY GROUP (existing) ==========
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'Max mNG Distribution by Group — Well {full_name}{suffix}', fontsize=14, fontweight='bold', y=0.98)
    
    # Panel A: Overlaid histograms by group
    ax = axes[0, 0]
    all_max_gfp = df_groups['max_gfp_top6'].dropna()
    bins = np.linspace(all_max_gfp.min(), all_max_gfp.max(), 40)
    
    for group in groups:
        group_data = df_groups[df_groups['activation_group'] == group]['max_gfp_top6'].dropna()
        if len(group_data) > 0:
            ax.hist(group_data, bins=bins, alpha=0.5, color=COLORS[group], 
                   label=f'{group.capitalize()} (n={len(group_data)})', edgecolor='white')
    
    ax.set_xlabel('Max mNG intensity (top 6 avg)')
    ax.set_ylabel('Count')
    ax.set_title('A   Distribution by group', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel B: Stacked histogram
    ax = axes[0, 1]
    data_by_group = [df_groups[df_groups['activation_group'] == g]['max_gfp_top6'].dropna().values for g in groups]
    colors_list = [COLORS[g] for g in groups]
    ax.hist(data_by_group, bins=bins, stacked=True, color=colors_list, 
           label=[g.capitalize() for g in groups], edgecolor='white', alpha=0.8)
    ax.set_xlabel('Max mNG intensity (top 6 avg)')
    ax.set_ylabel('Count')
    ax.set_title('B   Stacked distribution', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel C: KDE plot with normal fit overlay
    ax = axes[0, 2]
    from scipy.stats import norm, shapiro, normaltest, gaussian_kde
    
    normality_results = {}
    for group in groups:
        group_data = df_groups[df_groups['activation_group'] == group]['max_gfp_top6'].dropna()
        if len(group_data) > 3:
            # KDE plot
            kde = gaussian_kde(group_data)
            x_range = np.linspace(group_data.min(), group_data.max(), 200)
            ax.plot(x_range, kde(x_range), color=COLORS[group], linewidth=2, label=f'{group.capitalize()}')
            
            # Normal fit overlay (dashed)
            mu, std = group_data.mean(), group_data.std()
            normal_fit = norm.pdf(x_range, mu, std)
            # Scale to match KDE
            normal_fit_scaled = normal_fit * (kde(x_range).max() / normal_fit.max())
            ax.plot(x_range, normal_fit_scaled, color=COLORS[group], linewidth=1.5, linestyle='--', alpha=0.7)
            
            # Normality tests
            if len(group_data) >= 8:
                try:
                    shapiro_stat, shapiro_p = shapiro(group_data[:5000])  # Shapiro limited to 5000
                except:
                    shapiro_stat, shapiro_p = np.nan, np.nan
                try:
                    dagostino_stat, dagostino_p = normaltest(group_data)
                except:
                    dagostino_stat, dagostino_p = np.nan, np.nan
                normality_results[group] = {
                    'shapiro_p': shapiro_p, 
                    'dagostino_p': dagostino_p,
                    'mean': mu, 
                    'std': std,
                    'n': len(group_data)
                }
    
    ax.set_xlabel('Max mNG intensity (top 6 avg)')
    ax.set_ylabel('Density')
    ax.set_title('C   KDE with normal fit (dashed)', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel D: Q-Q plots
    ax = axes[1, 0]
    from scipy.stats import probplot
    
    for i, group in enumerate(groups):
        group_data = df_groups[df_groups['activation_group'] == group]['max_gfp_top6'].dropna()
        if len(group_data) > 3:
            (osm, osr), (slope, intercept, r) = probplot(group_data, dist="norm")
            ax.scatter(osm, osr, alpha=0.4, s=15, color=COLORS[group], label=f'{group.capitalize()} (r²={r**2:.3f})')
    
    # Add reference line
    ax.plot([-3, 3], [-3, 3], 'k--', linewidth=1, alpha=0.5, transform=ax.transData)
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.set_title('D   Q-Q plot (normality check)', loc='left', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    
    # Panel E: Boxplot with individual points
    ax = axes[1, 1]
    data = [df_groups[df_groups['activation_group'] == g]['max_gfp_top6'].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=[g.capitalize() for g in groups], patch_artist=True, widths=0.6)
    for patch, group in zip(bp['boxes'], groups):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.7)
    for i, (group, d) in enumerate(zip(groups, data)):
        if len(d) > 0:
            x = np.random.normal(i + 1, 0.08, size=len(d))
            ax.scatter(x, d, alpha=0.4, color=COLORS[group], s=20)
    ax.set_ylabel('Max mNG intensity (top 6 avg)')
    ax.set_title('E   Boxplot by group', loc='left', fontweight='bold')
    
    # Panel F: Summary statistics and normality test results
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Max mNG Distribution Summary\n" + "="*30 + "\n\n"
    
    for group in groups:
        if group in normality_results:
            res = normality_results[group]
            summary_text += f"{group.capitalize()} (n={res['n']}):\n"
            summary_text += f"  Mean: {res['mean']:.1f}\n"
            summary_text += f"  Std:  {res['std']:.1f}\n"
            summary_text += f"  Shapiro p: {res['shapiro_p']:.3e}\n"
            summary_text += f"  D'Agostino p: {res['dagostino_p']:.3e}\n"
            
            # Interpretation
            is_normal = res['shapiro_p'] > 0.05 and res['dagostino_p'] > 0.05
            summary_text += f"  Normal: {'Yes' if is_normal else 'No'}\n\n"
    
    summary_text += "="*30 + "\n"
    summary_text += "p > 0.05 suggests normality"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8, fontfamily='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', edgecolor='#4682b4', alpha=0.9))
    ax.set_title('F   Normality tests', loc='left', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_max_gfp_distribution_by_group" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "")
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()
    
    # ========== FIGURE 2: ALL CELLS COMBINED (NEW) ==========
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9))
    fig2.suptitle(f'Max mNG Distribution — All Cells Combined — Well {full_name}{suffix}', fontsize=14, fontweight='bold', y=0.98)
    
    all_data = df_groups['max_gfp_top6'].dropna()
    n_all = len(all_data)
    
    # Calculate normality for all cells
    mu_all, std_all = all_data.mean(), all_data.std()
    try:
        shapiro_stat_all, shapiro_p_all = shapiro(all_data[:5000])
    except:
        shapiro_stat_all, shapiro_p_all = np.nan, np.nan
    try:
        dagostino_stat_all, dagostino_p_all = normaltest(all_data)
    except:
        dagostino_stat_all, dagostino_p_all = np.nan, np.nan
    
    normality_results['all_cells'] = {
        'shapiro_p': shapiro_p_all,
        'dagostino_p': dagostino_p_all,
        'mean': mu_all,
        'std': std_all,
        'n': n_all
    }
    
    # Panel A: Histogram with normal fit
    ax = axes2[0, 0]
    bins_all = np.linspace(all_data.min(), all_data.max(), 50)
    ax.hist(all_data, bins=bins_all, alpha=0.7, color='#1f77b4', edgecolor='white', density=True, label=f'All cells (n={n_all})')
    
    # Overlay normal distribution
    x_range = np.linspace(all_data.min(), all_data.max(), 200)
    normal_fit = norm.pdf(x_range, mu_all, std_all)
    ax.plot(x_range, normal_fit, 'r-', linewidth=2.5, label=f'Normal fit\nμ={mu_all:.1f}, σ={std_all:.1f}')
    
    ax.set_xlabel('Max mNG intensity (top 6 avg)')
    ax.set_ylabel('Density')
    ax.set_title('A   Histogram with normal fit', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel B: KDE vs Normal
    ax = axes2[0, 1]
    kde_all = gaussian_kde(all_data)
    ax.plot(x_range, kde_all(x_range), 'b-', linewidth=2.5, label='KDE (actual)')
    ax.plot(x_range, normal_fit, 'r--', linewidth=2, label='Normal fit')
    ax.fill_between(x_range, kde_all(x_range), alpha=0.3, color='blue')
    ax.set_xlabel('Max mNG intensity (top 6 avg)')
    ax.set_ylabel('Density')
    ax.set_title('B   KDE vs Normal distribution', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel C: Q-Q plot
    ax = axes2[0, 2]
    (osm, osr), (slope, intercept, r) = probplot(all_data, dist="norm")
    ax.scatter(osm, osr, alpha=0.5, s=20, color='#1f77b4', edgecolors='white', linewidths=0.3)
    
    # Add perfect fit line
    line_x = np.array([osm.min(), osm.max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r-', linewidth=2, label=f'Fit (r²={r**2:.4f})')
    
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.set_title('C   Q-Q plot (all cells)', loc='left', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    
    # Panel D: Cumulative distribution
    ax = axes2[1, 0]
    sorted_data = np.sort(all_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cumulative, 'b-', linewidth=2, label='Empirical CDF')
    
    # Theoretical normal CDF
    theoretical_cdf = norm.cdf(sorted_data, mu_all, std_all)
    ax.plot(sorted_data, theoretical_cdf, 'r--', linewidth=2, label='Normal CDF')
    
    ax.set_xlabel('Max mNG intensity (top 6 avg)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('D   CDF comparison', loc='left', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    
    # Panel E: Log-transformed analysis
    ax = axes2[1, 1]
    log_data = np.log10(all_data.clip(lower=0.1))
    mu_log, std_log = log_data.mean(), log_data.std()
    
    bins_log = np.linspace(log_data.min(), log_data.max(), 50)
    ax.hist(log_data, bins=bins_log, alpha=0.7, color='#2ca02c', edgecolor='white', density=True, label='Log₁₀ transformed')
    
    x_log = np.linspace(log_data.min(), log_data.max(), 200)
    normal_log = norm.pdf(x_log, mu_log, std_log)
    ax.plot(x_log, normal_log, 'r-', linewidth=2.5, label=f'Normal fit\nμ={mu_log:.2f}, σ={std_log:.2f}')
    
    # Test normality of log-transformed data
    try:
        shapiro_log, shapiro_p_log = shapiro(log_data[:5000])
    except:
        shapiro_log, shapiro_p_log = np.nan, np.nan
    try:
        dagostino_log, dagostino_p_log = normaltest(log_data)
    except:
        dagostino_log, dagostino_p_log = np.nan, np.nan
    
    normality_results['all_cells_log10'] = {
        'shapiro_p': shapiro_p_log,
        'dagostino_p': dagostino_p_log,
        'mean': mu_log,
        'std': std_log,
        'n': n_all
    }
    
    ax.set_xlabel('Max mNG intensity ($\log_{10}$)')
    ax.set_ylabel('Density')
    ax.set_title('E   Log-transformed distribution', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel F: Summary
    ax = axes2[1, 2]
    ax.axis('off')
    
    is_normal_all = shapiro_p_all > 0.05 and dagostino_p_all > 0.05
    is_normal_log = shapiro_p_log > 0.05 and dagostino_p_log > 0.05
    
    summary_all = "ALL CELLS COMBINED\n" + "="*30 + "\n\n"
    summary_all += f"n = {n_all} cells\n\n"
    summary_all += "LINEAR SCALE:\n"
    summary_all += f"  Mean: {mu_all:.1f}\n"
    summary_all += f"  Std:  {std_all:.1f}\n"
    summary_all += f"  CV:   {std_all/mu_all*100:.1f}%\n"
    summary_all += f"  Shapiro p:    {shapiro_p_all:.3e}\n"
    summary_all += f"  D'Agostino p: {dagostino_p_all:.3e}\n"
    summary_all += f"  Normal: {'YES' if is_normal_all else 'NO'}\n\n"
    summary_all += "LOG₁₀ SCALE:\n"
    summary_all += f"  Mean: {mu_log:.3f}\n"
    summary_all += f"  Std:  {std_log:.3f}\n"
    summary_all += f"  Shapiro p:    {shapiro_p_log:.3e}\n"
    summary_all += f"  D'Agostino p: {dagostino_p_log:.3e}\n"
    summary_all += f"  Normal: {'YES' if is_normal_log else 'NO'}\n\n"
    summary_all += "="*30 + "\n"
    summary_all += "p > 0.05 suggests normality"
    
    ax.text(0.05, 0.95, summary_all, transform=ax.transAxes, fontsize=8, fontfamily='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', edgecolor='#4682b4', alpha=0.9))
    ax.set_title('F   Summary statistics', loc='left', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name2 = f"well_{full_name}_max_gfp_distribution_all_cells" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "")
    save_figure(fig2, fig_dir, name2, save_pdf, save_svg)
    plt.close()
    
    if save_individual:
        suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
        # Histogram by group
        fig, ax = plt.subplots(figsize=(5, 4))
        for group in groups:
            group_data = df_groups[df_groups['activation_group'] == group]['max_gfp_top6'].dropna()
            if len(group_data) > 0:
                ax.hist(group_data, bins=bins, alpha=0.5, color=COLORS[group],
                       label=f'{group.capitalize()} (n={len(group_data)})', edgecolor='white')
        ax.set_xlabel('Max mNG intensity (top 6 avg)')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_hist_by_group{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Stacked histogram
        fig, ax = plt.subplots(figsize=(5, 4))
        data_by_group_hist = [df_groups[df_groups['activation_group'] == g]['max_gfp_top6'].dropna().values for g in groups]
        colors_list = [COLORS[g] for g in groups]
        ax.hist(data_by_group_hist, bins=bins, stacked=True, color=colors_list,
                label=[g.capitalize() for g in groups], edgecolor='white', alpha=0.8)
        ax.set_xlabel('Max mNG intensity (top 6 avg)')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_stacked_hist{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # KDE with normal fit by group
        fig, ax = plt.subplots(figsize=(5, 4))
        for group in groups:
            group_data = df_groups[df_groups['activation_group'] == group]['max_gfp_top6'].dropna()
            if len(group_data) > 3:
                kde = gaussian_kde(group_data)
                x_range_grp = np.linspace(group_data.min(), group_data.max(), 200)
                ax.plot(x_range_grp, kde(x_range_grp), color=COLORS[group], linewidth=2, label=f'{group.capitalize()}')
                mu, std = group_data.mean(), group_data.std()
                normal_fit_grp = norm.pdf(x_range_grp, mu, std)
                if normal_fit_grp.max() > 0:
                    normal_fit_scaled = normal_fit_grp * (kde(x_range_grp).max() / normal_fit_grp.max())
                    ax.plot(x_range_grp, normal_fit_scaled, color=COLORS[group], linewidth=1.5, linestyle='--', alpha=0.7)
        ax.set_xlabel('Max mNG intensity (top 6 avg)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_kde_by_group{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Q-Q plots by group
        fig, ax = plt.subplots(figsize=(5, 4))
        for group in groups:
            group_data = df_groups[df_groups['activation_group'] == group]['max_gfp_top6'].dropna()
            if len(group_data) > 3:
                (osm, osr), (slope, intercept, r) = probplot(group_data, dist="norm")
                ax.scatter(osm, osr, alpha=0.4, s=15, color=COLORS[group], label=f'{group.capitalize()} (r²={r**2:.3f})')
        ax.plot([-3, 3], [-3, 3], 'k--', linewidth=1, alpha=0.5, transform=ax.transData)
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Sample quantiles')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_qq_by_group{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Boxplot with individual points
        fig, ax = plt.subplots(figsize=(5, 4))
        data_box = [df_groups[df_groups['activation_group'] == g]['max_gfp_top6'].dropna().values for g in groups]
        bp = ax.boxplot(data_box, labels=[g.capitalize() for g in groups], patch_artist=True, widths=0.6)
        for patch, group in zip(bp['boxes'], groups):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(0.7)
        for i, (group, d) in enumerate(zip(groups, data_box)):
            if len(d) > 0:
                x = np.random.normal(i + 1, 0.08, size=len(d))
                ax.scatter(x, d, alpha=0.4, color=COLORS[group], s=20)
        ax.set_ylabel('Max mNG intensity (top 6 avg)')
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_boxplot{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # All cells: histogram with normal fit
        fig, ax = plt.subplots(figsize=(5, 4))
        bins_all = np.linspace(all_data.min(), all_data.max(), 50)
        ax.hist(all_data, bins=bins_all, alpha=0.7, color='#1f77b4', edgecolor='white', density=True, label=f'All cells (n={n_all})')
        x_range_all = np.linspace(all_data.min(), all_data.max(), 200)
        normal_fit_all = norm.pdf(x_range_all, mu_all, std_all)
        ax.plot(x_range_all, normal_fit_all, 'r-', linewidth=2.5, label=f'Normal fit\nμ={mu_all:.1f}, σ={std_all:.1f}')
        ax.set_xlabel('Max mNG intensity (top 6 avg)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_all_hist_normal{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # All cells: KDE vs normal
        fig, ax = plt.subplots(figsize=(5, 4))
        kde_all = gaussian_kde(all_data)
        ax.plot(x_range_all, kde_all(x_range_all), 'b-', linewidth=2.5, label='KDE (actual)')
        ax.plot(x_range_all, normal_fit_all, 'r--', linewidth=2, label='Normal fit')
        ax.fill_between(x_range_all, kde_all(x_range_all), alpha=0.3, color='blue')
        ax.set_xlabel('Max mNG intensity (top 6 avg)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_all_kde_vs_normal{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # All cells: Q-Q plot
        fig, ax = plt.subplots(figsize=(5, 4))
        (osm, osr), (slope, intercept, r) = probplot(all_data, dist="norm")
        ax.scatter(osm, osr, alpha=0.5, s=20, color='#1f77b4', edgecolors='white', linewidths=0.3)
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r-', linewidth=2, label=f'Fit (r²={r**2:.4f})')
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Sample quantiles')
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_all_qq{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # All cells: CDF comparison
        fig, ax = plt.subplots(figsize=(5, 4))
        sorted_data = np.sort(all_data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cumulative, 'b-', linewidth=2, label='Empirical CDF')
        theoretical_cdf = norm.cdf(sorted_data, mu_all, std_all)
        ax.plot(sorted_data, theoretical_cdf, 'r--', linewidth=2, label='Normal CDF')
        ax.set_xlabel('Max mNG intensity (top 6 avg)')
        ax.set_ylabel('Cumulative probability')
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_all_cdf{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # All cells: log-transformed distribution
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(log_data, bins=np.linspace(log_data.min(), log_data.max(), 50), alpha=0.7, color='#2ca02c', edgecolor='white', density=True, label='Log₁₀ transformed')
        ax.plot(x_log, normal_log, 'r-', linewidth=2.5, label=f'Normal fit\nμ={mu_log:.2f}, σ={std_log:.2f}')
        ax.set_xlabel('Max mNG intensity ($\log_{10}$)')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_max_gfp_all_log_transform{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
    
    # Print results
    print(f"\nMax mNG Distribution Analysis:")
    print(f"  BY GROUP:")
    for group in groups:
        if group in normality_results:
            res = normality_results[group]
            is_norm = res['shapiro_p'] > 0.05 and res['dagostino_p'] > 0.05
            print(f"    {group.capitalize()}: mean={res['mean']:.1f}, std={res['std']:.1f}, Shapiro p={res['shapiro_p']:.3e}, Normal={'Yes' if is_norm else 'No'}")
    
    print(f"  ALL CELLS COMBINED:")
    print(f"    Linear: mean={mu_all:.1f}, std={std_all:.1f}, Shapiro p={shapiro_p_all:.3e}, Normal={'Yes' if is_normal_all else 'No'}")
    print(f"    Log₁₀:  mean={mu_log:.3f}, std={std_log:.3f}, Shapiro p={shapiro_p_log:.3e}, Normal={'Yes' if is_normal_log else 'No'}")
    
    return df_act, normality_results

def statistical_comparison_max_gfp(df_act, output_dir, well, save_pdf=False, save_svg=False,
                                   save_individual=False, suffix="", timepoint_min=None, timepoint_max=None):
    """
    Statistical comparison of max mNG intensity between activation groups.
    Performs both parametric (ANOVA) and non-parametric (Kruskal-Wallis) tests
    with appropriate post-hoc analyses.
    """
    from scipy.stats import kruskal, mannwhitneyu, f_oneway, levene, shapiro
    
    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    groups = ['early', 'average', 'late']
    group_labels = ['Early', 'Average', 'Late']
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Calculate max mNG if not already done
    if 'max_gfp_top6' not in df_act.columns:
        raise ValueError("Run calculate_max_gfp_top6 first")
    
    # Extract data per group
    data_by_group = {}
    for group in groups:
        data = df_act[df_act['activation_group'] == group]['max_gfp_top6'].dropna().values
        if len(data) > 0:
            data_by_group[group] = data
    
    if len(data_by_group) < 2:
        print("Not enough groups with data for statistical comparison")
        return None
    
    results = {'groups': {}, 'tests': {}, 'posthoc': {}}
    
    # ========== Descriptive Statistics ==========
    print(f"\n{'='*60}")
    print(f"STATISTICAL COMPARISON: Max mNG Intensity")
    print(f"{'='*60}")
    
    for group in groups:
        if group in data_by_group:
            d = data_by_group[group]
            results['groups'][group] = {
                'n': len(d),
                'mean': np.mean(d),
                'std': np.std(d),
                'median': np.median(d),
                'iqr': np.percentile(d, 75) - np.percentile(d, 25),
                'q25': np.percentile(d, 25),
                'q75': np.percentile(d, 75)
            }
            print(f"\n{group.capitalize()} (n={len(d)}):")
            print(f"  Mean ± SD: {np.mean(d):.1f} ± {np.std(d):.1f}")
            print(f"  Median [IQR]: {np.median(d):.1f} [{np.percentile(d, 25):.1f}-{np.percentile(d, 75):.1f}]")
    
    # ========== Check Assumptions ==========
    print(f"\n{'-'*40}")
    print("Assumption Checks:")
    print(f"{'-'*40}")
    
    # Normality (Shapiro-Wilk)
    normality_ok = True
    for group in groups:
        if group in data_by_group and len(data_by_group[group]) >= 8:
            stat, p = shapiro(data_by_group[group][:5000])
            results['groups'][group]['shapiro_p'] = p
            is_normal = p > 0.05
            normality_ok = normality_ok and is_normal
            print(f"  {group.capitalize()} normality (Shapiro): p={p:.3e} {'✓' if is_normal else '✗'}")
    
    # Homogeneity of variances (Levene's test)
    valid_groups = [data_by_group[g] for g in groups if g in data_by_group]
    if len(valid_groups) >= 2:
        levene_stat, levene_p = levene(*valid_groups)
        results['tests']['levene'] = {'statistic': levene_stat, 'p': levene_p}
        variance_ok = levene_p > 0.05
        print(f"  Equal variances (Levene): p={levene_p:.3e} {'✓' if variance_ok else '✗'}")
    else:
        variance_ok = False
        levene_p = np.nan
    
    # ========== Omnibus Tests ==========
    print(f"\n{'-'*40}")
    print("Omnibus Tests (Overall Group Differences):")
    print(f"{'-'*40}")
    
    # Kruskal-Wallis (non-parametric)
    kw_stat, kw_p = kruskal(*valid_groups)
    results['tests']['kruskal_wallis'] = {'statistic': kw_stat, 'p': kw_p}
    print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.3e} {'***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'}")
    
    # One-way ANOVA (parametric)
    anova_stat, anova_p = f_oneway(*valid_groups)
    results['tests']['anova'] = {'statistic': anova_stat, 'p': anova_p}
    print(f"  One-way ANOVA: F={anova_stat:.2f}, p={anova_p:.3e} {'***' if anova_p < 0.001 else '**' if anova_p < 0.01 else '*' if anova_p < 0.05 else 'ns'}")
    
    # ========== Post-hoc Tests ==========
    print(f"\n{'-'*40}")
    print("Post-hoc Pairwise Comparisons:")
    print(f"{'-'*40}")
    
    # Dunn's test (post-hoc for Kruskal-Wallis)
    # Manual implementation with Bonferroni correction
    comparisons = [('early', 'average'), ('early', 'late'), ('average', 'late')]
    n_comparisons = len(comparisons)
    
    print("\n  Mann-Whitney U with Bonferroni correction:")
    for g1, g2 in comparisons:
        if g1 in data_by_group and g2 in data_by_group:
            stat, p_raw = mannwhitneyu(data_by_group[g1], data_by_group[g2], alternative='two-sided')
            p_corrected = min(p_raw * n_comparisons, 1.0)  # Bonferroni
            
            # Effect size (rank-biserial correlation)
            n1, n2 = len(data_by_group[g1]), len(data_by_group[g2])
            r = 1 - (2 * stat) / (n1 * n2)  # rank-biserial correlation
            
            results['posthoc'][f'{g1}_vs_{g2}'] = {
                'statistic': stat, 'p_raw': p_raw, 'p_corrected': p_corrected, 'effect_size_r': r
            }
            
            sig = '***' if p_corrected < 0.001 else '**' if p_corrected < 0.01 else '*' if p_corrected < 0.05 else 'ns'
            print(f"    {g1.capitalize()} vs {g2.capitalize()}: U={stat:.0f}, p_raw={p_raw:.3e}, p_adj={p_corrected:.3e} {sig}, r={r:.3f}")
    
    # ========== Effect Sizes ==========
    print(f"\n{'-'*40}")
    print("Effect Sizes:")
    print(f"{'-'*40}")
    
    # Eta-squared for Kruskal-Wallis
    n_total = sum(len(d) for d in valid_groups)
    eta_squared = (kw_stat - len(valid_groups) + 1) / (n_total - len(valid_groups))
    eta_squared = max(0, eta_squared)  # Can't be negative
    results['tests']['kruskal_wallis']['eta_squared'] = eta_squared
    print(f"  η² (Kruskal-Wallis) = {eta_squared:.3f}")
    print(f"    Interpretation: {'large' if eta_squared > 0.14 else 'medium' if eta_squared > 0.06 else 'small'} effect")
    
    # Cohen's d for pairwise comparisons
    print("\n  Cohen's d (pairwise):")
    for g1, g2 in comparisons:
        if g1 in data_by_group and g2 in data_by_group:
            d1, d2 = data_by_group[g1], data_by_group[g2]
            pooled_std = np.sqrt(((len(d1)-1)*np.std(d1)**2 + (len(d2)-1)*np.std(d2)**2) / (len(d1)+len(d2)-2))
            cohens_d = (np.mean(d1) - np.mean(d2)) / pooled_std if pooled_std > 0 else 0
            results['posthoc'][f'{g1}_vs_{g2}']['cohens_d'] = cohens_d
            interp = 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
            print(f"    {g1.capitalize()} vs {g2.capitalize()}: d={cohens_d:.3f} ({interp})")
    
    # ========== Create Figure ==========
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'Statistical Comparison: Max mNG Intensity — Well {full_name}{suffix}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Panel A: Boxplot with significance bars
    ax = axes[0, 0]
    data_list = [data_by_group.get(g, []) for g in groups]
    positions = [1, 2, 3]
    bp = ax.boxplot([d for d in data_list if len(d) > 0], 
                    positions=[p for p, d in zip(positions, data_list) if len(d) > 0],
                    patch_artist=True, widths=0.6)
    
    valid_idx = 0
    for i, (group, d) in enumerate(zip(groups, data_list)):
        if len(d) > 0:
            bp['boxes'][valid_idx].set_facecolor(COLORS[group])
            bp['boxes'][valid_idx].set_alpha(0.7)
            x_jitter = np.random.normal(positions[i], 0.08, size=len(d))
            ax.scatter(x_jitter, d, alpha=0.4, color=COLORS[group], s=20, zorder=3)
            valid_idx += 1
    
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel('Max mNG intensity (top 6 avg)')
    ax.set_title('A   Distribution by group', loc='left', fontweight='bold')
    
    # Add significance bars
    y_max = max(max(d) for d in data_list if len(d) > 0)
    y_step = y_max * 0.08
    
    sig_pairs = [(0, 1, 'early_vs_average'), (1, 2, 'average_vs_late'), (0, 2, 'early_vs_late')]
    for idx, (i, j, key) in enumerate(sig_pairs):
        if key in results['posthoc']:
            p_adj = results['posthoc'][key]['p_corrected']
            if p_adj < 0.05:
                y_bar = y_max + y_step * (idx + 1)
                ax.plot([positions[i], positions[j]], [y_bar, y_bar], 'k-', linewidth=1)
                sig_text = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*'
                ax.text((positions[i] + positions[j]) / 2, y_bar, sig_text, ha='center', va='bottom', fontsize=12)
    
    ax.set_ylim(top=y_max + y_step * 4)
    
    # Panel B: Violin plot
    ax = axes[0, 1]
    valid_data = [data_by_group[g] for g in groups if g in data_by_group]
    valid_positions = [i+1 for i, g in enumerate(groups) if g in data_by_group]
    parts = ax.violinplot(valid_data, positions=valid_positions, showmeans=True, showmedians=True)
    for i, (pc, group) in enumerate(zip(parts['bodies'], [g for g in groups if g in data_by_group])):
        pc.set_facecolor(COLORS[group])
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(group_labels)
    ax.set_ylabel('Max mNG intensity (top 6 avg)')
    ax.set_title('B   Violin plot', loc='left', fontweight='bold')
    
    # Panel C: Mean ± SEM bar plot
    ax = axes[0, 2]
    means = [results['groups'][g]['mean'] for g in groups if g in results['groups']]
    sems = [results['groups'][g]['std'] / np.sqrt(results['groups'][g]['n']) for g in groups if g in results['groups']]
    valid_groups_list = [g for g in groups if g in results['groups']]
    
    bars = ax.bar(range(len(means)), means, yerr=sems, capsize=5, 
                  color=[COLORS[g] for g in valid_groups_list], alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([g.capitalize() for g in valid_groups_list])
    ax.set_ylabel('Max mNG intensity (mean ± SEM)')
    ax.set_title('C   Mean comparison', loc='left', fontweight='bold')
    
    # Panel D: Effect size visualization
    ax = axes[1, 0]
    effect_sizes = []
    labels = []
    for g1, g2 in comparisons:
        key = f'{g1}_vs_{g2}'
        if key in results['posthoc']:
            effect_sizes.append(results['posthoc'][key]['cohens_d'])
            labels.append(f'{g1[:1].upper()} vs {g2[:1].upper()}')
    
    colors_bars = ['#d62728' if abs(e) > 0.8 else '#ff7f0e' if abs(e) > 0.5 else '#2ca02c' for e in effect_sizes]
    ax.barh(range(len(effect_sizes)), effect_sizes, color=colors_bars, alpha=0.8, edgecolor='black')
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d")
    ax.set_title('D   Effect sizes', loc='left', fontweight='bold')
    
    # Panel E: Cumulative distribution
    ax = axes[1, 1]
    for group in groups:
        if group in data_by_group:
            sorted_data = np.sort(data_by_group[group])
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cumulative, color=COLORS[group], linewidth=2, label=group.capitalize())
    ax.set_xlabel('Max mNG intensity (top 6 avg)')
    ax.set_ylabel('Cumulative proportion')
    ax.set_title('E   Cumulative distribution', loc='left', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    
    # Panel F: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = "STATISTICAL SUMMARY\n" + "="*30 + "\n\n"
    summary += "Omnibus Tests:\n"
    summary += f"  Kruskal-Wallis: H={kw_stat:.1f}\n"
    summary += f"    p = {kw_p:.2e} {'***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'}\n"
    summary += f"    η² = {eta_squared:.3f}\n\n"
    summary += "Post-hoc (Bonferroni):\n"
    for g1, g2 in comparisons:
        key = f'{g1}_vs_{g2}'
        if key in results['posthoc']:
            p = results['posthoc'][key]['p_corrected']
            d = results['posthoc'][key]['cohens_d']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            summary += f"  {g1[:1].upper()}-{g2[:1].upper()}: p={p:.2e} {sig}\n"
            summary += f"       d={d:.2f}\n"
    
    summary += "\n" + "="*30 + "\n"
    summary += "* p<0.05, ** p<0.01, *** p<0.001"
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=8, fontfamily='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', edgecolor='#4682b4', alpha=0.9))
    ax.set_title('F   Test results', loc='left', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_max_gfp_statistical_comparison" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "")
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()
    
    if save_individual:
        suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
        # Boxplot with significance bars
        fig, ax = plt.subplots(figsize=(6, 4))
        data_list = [data_by_group.get(g, []) for g in groups]
        positions = [1, 2, 3]
        bp = ax.boxplot([d for d in data_list if len(d) > 0],
                        positions=[p for p, d in zip(positions, data_list) if len(d) > 0],
                        patch_artist=True, widths=0.6)
        valid_idx = 0
        for i, (group, d) in enumerate(zip(groups, data_list)):
            if len(d) > 0:
                bp['boxes'][valid_idx].set_facecolor(COLORS[group])
                bp['boxes'][valid_idx].set_alpha(0.7)
                x_jitter = np.random.normal(positions[i], 0.08, size=len(d))
                ax.scatter(x_jitter, d, alpha=0.4, color=COLORS[group], s=20, zorder=3)
                valid_idx += 1
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('Max mNG intensity (top 6 avg)')
        y_max = max(max(d) for d in data_list if len(d) > 0)
        y_step = y_max * 0.08
        sig_pairs = [(0, 1, 'early_vs_average'), (1, 2, 'average_vs_late'), (0, 2, 'early_vs_late')]
        for idx, (i, j, key) in enumerate(sig_pairs):
            if key in results['posthoc']:
                p_adj = results['posthoc'][key]['p_corrected']
                if p_adj < 0.05:
                    y_bar = y_max + y_step * (idx + 1)
                    ax.plot([positions[i], positions[j]], [y_bar, y_bar], 'k-', linewidth=1)
                    sig_text = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*'
                    ax.text((positions[i] + positions[j]) / 2, y_bar, sig_text, ha='center', va='bottom', fontsize=12)
        ax.set_ylim(top=y_max + y_step * 4)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_stat_max_gfp_boxplot_sig{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Violin plot
        fig, ax = plt.subplots(figsize=(5, 4))
        valid_data = [data_by_group[g] for g in groups if g in data_by_group]
        valid_positions = [i+1 for i, g in enumerate(groups) if g in data_by_group]
        parts = ax.violinplot(valid_data, positions=valid_positions, showmeans=True, showmedians=True)
        for i, (pc, group) in enumerate(zip(parts['bodies'], [g for g in groups if g in data_by_group])):
            pc.set_facecolor(COLORS[group])
            pc.set_alpha(0.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('Max mNG intensity (top 6 avg)')
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_stat_max_gfp_violin{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Mean ± SEM bar plot
        fig, ax = plt.subplots(figsize=(5, 4))
        means = [results['groups'][g]['mean'] for g in groups if g in results['groups']]
        sems = [results['groups'][g]['std'] / np.sqrt(results['groups'][g]['n']) for g in groups if g in results['groups']]
        valid_groups_list = [g for g in groups if g in results['groups']]
        ax.bar(range(len(means)), means, yerr=sems, capsize=5,
               color=[COLORS[g] for g in valid_groups_list], alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels([g.capitalize() for g in valid_groups_list])
        ax.set_ylabel('Max mNG intensity (mean ± SEM)')
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_stat_max_gfp_mean_sem{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Effect size visualization
        fig, ax = plt.subplots(figsize=(5, 4))
        effect_sizes = []
        labels = []
        for g1, g2 in comparisons:
            key = f'{g1}_vs_{g2}'
            if key in results['posthoc']:
                effect_sizes.append(results['posthoc'][key]['cohens_d'])
                labels.append(f'{g1[:1].upper()} vs {g2[:1].upper()}')
        colors_bars = ['#d62728' if abs(e) > 0.8 else '#ff7f0e' if abs(e) > 0.5 else '#2ca02c' for e in effect_sizes]
        ax.barh(range(len(effect_sizes)), effect_sizes, color=colors_bars, alpha=0.8, edgecolor='black')
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(-0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Cohen's d")
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_stat_max_gfp_effect_sizes{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Cumulative distribution by group
        fig, ax = plt.subplots(figsize=(5, 4))
        for group in groups:
            if group in data_by_group:
                sorted_data = np.sort(data_by_group[group])
                cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(sorted_data, cumulative, color=COLORS[group], linewidth=2, label=group.capitalize())
        ax.set_xlabel('Max mNG intensity (top 6 avg)')
        ax.set_ylabel('Cumulative proportion')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_stat_max_gfp_cdf{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
    
    # Save results to CSV
    stats_df = pd.DataFrame([
        {'comparison': 'omnibus_kruskal_wallis', 'statistic': kw_stat, 'p_value': kw_p, 'effect_size': eta_squared, 'effect_type': 'eta_squared'},
        {'comparison': 'omnibus_anova', 'statistic': anova_stat, 'p_value': anova_p, 'effect_size': np.nan, 'effect_type': np.nan},
    ])
    for g1, g2 in comparisons:
        key = f'{g1}_vs_{g2}'
        if key in results['posthoc']:
            stats_df = pd.concat([stats_df, pd.DataFrame([{
                'comparison': key,
                'statistic': results['posthoc'][key]['statistic'],
                'p_value': results['posthoc'][key]['p_corrected'],
                'effect_size': results['posthoc'][key]['cohens_d'],
                'effect_type': 'cohens_d'
            }])], ignore_index=True)
    
    suffix_clean = suffix.replace(' ', '_').replace('(', '').replace(')', '') if suffix else ""
    stats_filename = f"well_{full_name}_max_gfp_statistics_{suffix_clean}.csv" if suffix_clean else f"well_{full_name}_max_gfp_statistics.csv"
    stats_df.to_csv(output_dir / stats_filename, index=False)
    print(f"\nSaved: {stats_filename}")
    
    return results

def verify_bfp_stability(df_meas, df_act, output_dir, well, save_pdf=False, save_svg=False,
                          save_individual=False, suffix="", timepoint_min=None, timepoint_max=None):
    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    groups = ['early', 'average', 'late']
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    bfp_stats = df_meas.groupby('unique_track_id')['bfp_mean_intensity'].agg(['mean', 'std', 'min', 'max'])
    bfp_stats['cv'] = bfp_stats['std'] / bfp_stats['mean'] * 100
    
    mng_stats = df_meas.groupby('unique_track_id')['mean_intensity'].agg(['mean', 'std'])
    mng_stats['cv'] = mng_stats['std'] / mng_stats['mean'] * 100
    
    print(f"BFP stability: CV = {bfp_stats['cv'].mean():.1f}% ± {bfp_stats['cv'].std():.1f}%")
    print(f"mNG variability: CV = {mng_stats['cv'].mean():.1f}% ± {mng_stats['cv'].std():.1f}%")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'BFP Stability Analysis — Well {full_name}{suffix}', fontsize=14, fontweight='bold', y=0.98)

    def _scatter_bfp_mng(ax, df, xcol, ycol, xlabel, ylabel, letter, title, log_log=True,
                         color_by_group=True):
        """Scatter plot of xcol vs ycol from df_act, optionally colored by activation group."""
        df_v = df.dropna(subset=[xcol, ycol])
        df_v = df_v[(df_v[xcol] > 0) & (df_v[ycol] > 0)]
        if len(df_v) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')
            return
        if color_by_group:
            for group in groups:
                sub = df_v[df_v['activation_group'] == group] if 'activation_group' in df_v.columns else pd.DataFrame()
                if len(sub) > 0:
                    ax.scatter(sub[xcol], sub[ycol], color=COLORS[group],
                               alpha=0.55, s=22, label=group.capitalize(), edgecolors='none')
            # Cells not in any activation group (non-activating)
            other = df_v[~df_v['activation_group'].isin(groups)] if 'activation_group' in df_v.columns else df_v
            if len(other) > 0:
                ax.scatter(other[xcol], other[ycol], color='#BDBDBD',
                           alpha=0.35, s=15, label='Not activated', edgecolors='none')
        else:
            ax.scatter(df_v[xcol], df_v[ycol], color='#1a3a6b', alpha=1.0, s=18, edgecolors='none')
        # Regression line and Pearson r (on log scale if requested)
        x_vals = np.log10(df_v[xcol]) if log_log else df_v[xcol].values
        y_vals = np.log10(df_v[ycol]) if log_log else df_v[ycol].values
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        if valid.sum() > 5:
            r, p = stats.pearsonr(x_vals[valid], y_vals[valid])
            z = np.polyfit(x_vals[valid], y_vals[valid], 1)
            x_line = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 100)
            y_line = np.poly1d(z)(x_line)
            if log_log:
                ax.plot(10 ** x_line, 10 ** y_line, 'k--', linewidth=1.5, alpha=0.7)
            else:
                ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.7)
            ax.text(0.04, 0.96, f'r = {r:.2f}  p = {p:.3g}\nn = {valid.sum()}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))
        if log_log:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if color_by_group:
            ax.legend(loc='lower right', fontsize=7, markerscale=1.3)
        ax.set_title(f'{letter}   {title}', loc='left', fontweight='bold')

    # Panel A: BFP trajectories
    ax = axes[0, 0]
    for group in groups:
        track_ids = df_act[df_act['activation_group'] == group]['unique_track_id'].values
        group_data = df_meas[df_meas['unique_track_id'].isin(track_ids)]
        if len(group_data) > 0 and 'bfp_mean_intensity' in group_data.columns:
            mean_traj = group_data.groupby('timepoint')['bfp_mean_intensity'].agg(['mean', 'std'])
            ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2, label=group.capitalize())
            ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                           mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('BFP intensity (a.u.)')
    ax.set_title('A   BFP trajectories', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(t_min, t_max)

    # Panel B: mNG trajectories
    ax = axes[0, 1]
    for group in groups:
        track_ids = df_act[df_act['activation_group'] == group]['unique_track_id'].values
        group_data = df_meas[df_meas['unique_track_id'].isin(track_ids)]
        if len(group_data) > 0:
            mean_traj = group_data.groupby('timepoint')['mean_intensity'].agg(['mean', 'std'])
            ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2, label=group.capitalize())
            ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                           mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('mNG intensity (a.u.)')
    ax.set_title('B   mNG trajectories', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(t_min, t_max)

    # Panel C: CV histogram
    ax = axes[0, 2]
    ax.hist(bfp_stats['cv'].dropna(), bins=30, alpha=0.7, color='blue', label=f'BFP (mean={bfp_stats["cv"].mean():.1f}%)', edgecolor='white')
    ax.hist(mng_stats['cv'].dropna(), bins=30, alpha=0.7, color='green', label=f'mNG (mean={mng_stats["cv"].mean():.1f}%)', edgecolor='white')
    ax.set_xlabel('CV (%)')
    ax.set_ylabel('Count')
    ax.set_title('C   Signal variability (CV)', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Panel D: Individual BFP traces
    ax = axes[1, 0]
    sample_tracks = df_act['unique_track_id'].sample(min(20, len(df_act)), random_state=42).values
    for track_id in sample_tracks:
        track_data = df_meas[df_meas['unique_track_id'] == track_id].sort_values('timepoint')
        if len(track_data) > 0 and 'bfp_mean_intensity' in track_data.columns:
            bfp_norm = track_data['bfp_mean_intensity'] / track_data['bfp_mean_intensity'].iloc[0]
            ax.plot(track_data['timepoint'], bfp_norm, alpha=0.4, linewidth=1, color='blue')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('BFP (normalized to t=0)')
    ax.set_title('D   Individual BFP traces', loc='left', fontweight='bold')
    ax.set_ylim(0.5, 1.5)
    ax.set_xlim(t_min, t_max)

    # Panel E: Baseline mNG vs Baseline BFP
    ax = axes[1, 1]
    if 'baseline_bfp' in df_act.columns and 'baseline_intensity' in df_act.columns:
        _scatter_bfp_mng(ax, df_act, 'baseline_bfp', 'baseline_intensity',
                         'Baseline BFP (a.u., log)', 'Baseline mNG (a.u., log)',
                         'E', 'Baseline mNG vs Baseline BFP', log_log=True,
                         color_by_group=False)
    else:
        ax.text(0.5, 0.5, 'baseline_bfp not available', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        ax.set_title('E   Baseline mNG vs Baseline BFP', loc='left', fontweight='bold')

    # Panel F: Max mNG vs Baseline BFP
    ax = axes[1, 2]
    max_col = 'max_gfp_top6' if 'max_gfp_top6' in df_act.columns else \
              'max_intensity'  if 'max_intensity'  in df_act.columns else None
    if max_col and 'baseline_bfp' in df_act.columns:
        _scatter_bfp_mng(ax, df_act, 'baseline_bfp', max_col,
                         'Baseline BFP (a.u., log)',
                         'Max mNG (a.u., log)',
                         'F', 'Max mNG vs Baseline BFP', log_log=True,
                         color_by_group=False)
    else:
        ax.text(0.5, 0.5, 'baseline_bfp / max mNG not available',
                transform=ax.transAxes, ha='center', va='center', color='gray')
        ax.set_title('F   Max mNG vs Baseline BFP', loc='left', fontweight='bold')

    # Publication-ready axis styling: black spines on all sides, inward ticks
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')
        ax.tick_params(axis='both', which='major', direction='in', length=4,
                       width=0.8, color='black', labelsize=9, labelcolor='black')
        ax.tick_params(axis='both', which='minor', direction='in', length=2,
                       width=0.6, color='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_bfp_stability" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "")
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()
    
    if save_individual:
        suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
        # BFP trajectories by group
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in groups:
            track_ids = df_act[df_act['activation_group'] == group]['unique_track_id'].values
            group_data = df_meas[df_meas['unique_track_id'].isin(track_ids)]
            if len(group_data) > 0 and 'bfp_mean_intensity' in group_data.columns:
                mean_traj = group_data.groupby('timepoint')['bfp_mean_intensity'].agg(['mean', 'std'])
                ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2, label=group.capitalize())
                ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                               mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('BFP intensity (a.u.)')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(t_min, t_max)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_bfp_trajectories_by_group{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # mNG trajectories by group
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in groups:
            track_ids = df_act[df_act['activation_group'] == group]['unique_track_id'].values
            group_data = df_meas[df_meas['unique_track_id'].isin(track_ids)]
            if len(group_data) > 0:
                mean_traj = group_data.groupby('timepoint')['mean_intensity'].agg(['mean', 'std'])
                ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2, label=group.capitalize())
                ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                               mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('mNG intensity (a.u.)')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(t_min, t_max)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_mng_trajectories_by_group{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # CV histogram
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(bfp_stats['cv'].dropna(), bins=30, alpha=0.7, color='blue', label=f'BFP (mean={bfp_stats["cv"].mean():.1f}%)', edgecolor='white')
        ax.hist(mng_stats['cv'].dropna(), bins=30, alpha=0.7, color='green', label=f'mNG (mean={mng_stats["cv"].mean():.1f}%)', edgecolor='white')
        ax.set_xlabel('CV (%)')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_cv_histogram_bfp_mng{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Individual normalized BFP traces
        fig, ax = plt.subplots(figsize=(6, 4))
        sample_tracks = df_act['unique_track_id'].sample(min(20, len(df_act)), random_state=42).values
        for track_id in sample_tracks:
            track_data = df_meas[df_meas['unique_track_id'] == track_id].sort_values('timepoint')
            if len(track_data) > 0 and 'bfp_mean_intensity' in track_data.columns:
                bfp_norm = track_data['bfp_mean_intensity'] / track_data['bfp_mean_intensity'].iloc[0]
                ax.plot(track_data['timepoint'], bfp_norm, alpha=0.4, linewidth=1, color='blue')
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('BFP (normalized)')
        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(t_min, t_max)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_bfp_individual_traces{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
    
    return bfp_stats, mng_stats


def analyze_bfp_relative_to_activation(df_meas, df_act, output_dir, well, save_pdf=False, save_svg=False,
                                        save_individual=False, suffix="", timepoint_min=None, timepoint_max=None):
    set_publication_style()
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    _, _, full_name = parse_well(well)
    groups = ['early', 'average', 'late']
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    activation_times = df_act.set_index('unique_track_id')['activation_timepoint'].to_dict()
    
    aligned_data = []
    for track_id in df_act[df_act['activation_group'].isin(groups)]['unique_track_id'].unique():
        if track_id not in activation_times or np.isnan(activation_times[track_id]):
            continue
        
        t_act = activation_times[track_id]
        track_data = df_meas[df_meas['unique_track_id'] == track_id].copy()
        if len(track_data) == 0 or 'bfp_mean_intensity' not in track_data.columns:
            continue
        
        track_data['time_relative'] = track_data['timepoint'] - t_act
        group = df_act[df_act['unique_track_id'] == track_id]['activation_group'].values[0]
        track_data['group'] = group
        
        pre_act = track_data[track_data['time_relative'] < 0]['bfp_mean_intensity']
        if len(pre_act) > 0:
            track_data['bfp_normalized'] = track_data['bfp_mean_intensity'] / pre_act.mean()
            aligned_data.append(track_data)
    
    if not aligned_data:
        return None, None
    
    df_aligned = pd.concat(aligned_data, ignore_index=True)
    print(f"  Aligned {df_aligned['unique_track_id'].nunique()} cells")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'BFP Dynamics — Well {full_name}{suffix}', fontsize=14, fontweight='bold', y=0.98)
    
    time_range = np.arange(-15, 25)
    
    # Panel A: BFP aligned
    ax = axes[0, 0]
    for group in groups:
        group_data = df_aligned[df_aligned['group'] == group]
        means, stds, valid_times = [], [], []
        for t in time_range:
            t_data = group_data[group_data['time_relative'] == t]['bfp_normalized']
            if len(t_data) >= 5:
                means.append(t_data.mean())
                stds.append(t_data.std())
                valid_times.append(t)
        if valid_times:
            means, stds = np.array(means), np.array(stds)
            ax.plot(valid_times, means, color=COLORS[group], linewidth=2, label=group.capitalize())
            ax.fill_between(valid_times, means - stds, means + stds, color=COLORS[group], alpha=0.15)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Activation')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time relative to activation')
    ax.set_ylabel('BFP (normalized)')
    ax.set_title('A   BFP aligned', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-15, 24)
    
    # Panel B: mNG aligned
    ax = axes[0, 1]
    for group in groups:
        group_data = df_aligned[df_aligned['group'] == group]
        means, stds, valid_times = [], [], []
        for t in time_range:
            t_data = group_data[group_data['time_relative'] == t]['mean_intensity']
            if len(t_data) >= 5:
                means.append(t_data.mean())
                stds.append(t_data.std())
                valid_times.append(t)
        if valid_times:
            means, stds = np.array(means), np.array(stds)
            ax.plot(valid_times, means, color=COLORS[group], linewidth=2, label=group.capitalize())
            ax.fill_between(valid_times, means - stds, means + stds, color=COLORS[group], alpha=0.15)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time relative to activation')
    ax.set_ylabel('mNG intensity (a.u.)')
    ax.set_title('B   mNG aligned', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(-15, 24)
    
    # Panel C: BFP change boxplot
    ax = axes[1, 0]
    bfp_changes = []
    for track_id in df_aligned['unique_track_id'].unique():
        track_data = df_aligned[df_aligned['unique_track_id'] == track_id].sort_values('time_relative')
        group = track_data['group'].iloc[0]
        pre = track_data[(track_data['time_relative'] >= -10) & (track_data['time_relative'] < 0)]
        post = track_data[(track_data['time_relative'] >= 0) & (track_data['time_relative'] <= 10)]
        if len(pre) >= 3 and len(post) >= 3:
            pct_change = (post['bfp_mean_intensity'].mean() - pre['bfp_mean_intensity'].mean()) / pre['bfp_mean_intensity'].mean() * 100
            bfp_changes.append({'track_id': track_id, 'group': group, 'pct_change': pct_change})
    
    df_changes = pd.DataFrame(bfp_changes)
    if len(df_changes) > 0:
        data = [df_changes[df_changes['group'] == g]['pct_change'].dropna().values for g in groups]
        data = [d for d in data if len(d) > 0]
        if data:
            bp = ax.boxplot(data, labels=[g.capitalize() for g in groups if len(df_changes[df_changes['group'] == g]) > 0],
                           patch_artist=True, widths=0.6)
            for i, (patch, group) in enumerate(zip(bp['boxes'], groups)):
                patch.set_facecolor(COLORS[group])
                patch.set_alpha(0.7)
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('BFP change (%)')
    ax.set_title('C   BFP change post-activation', loc='left', fontweight='bold')
    
    # Panel D: Individual examples
    ax = axes[1, 1]
    for group in groups:
        group_tracks = df_aligned[df_aligned['group'] == group]['unique_track_id'].unique()
        sample_tracks = np.random.choice(group_tracks, min(5, len(group_tracks)), replace=False)
        for track_id in sample_tracks:
            track_data = df_aligned[df_aligned['unique_track_id'] == track_id].sort_values('time_relative')
            ax.plot(track_data['time_relative'], track_data['bfp_normalized'], color=COLORS[group], alpha=0.4, linewidth=1)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time relative to activation')
    ax.set_ylabel('BFP (normalized)')
    ax.set_title('D   Individual examples', loc='left', fontweight='bold')
    ax.set_xlim(-15, 24)
    ax.set_ylim(0.5, 1.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"well_{full_name}_bfp_activation_aligned" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "")
    save_figure(fig, fig_dir, name, save_pdf, save_svg)
    plt.close()
    
    if save_individual:
        suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
        time_range = np.arange(-15, 25)
        # BFP normalized, aligned to activation (by group)
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in groups:
            group_data = df_aligned[df_aligned['group'] == group]
            means, stds, valid_times = [], [], []
            for t in time_range:
                t_data = group_data[group_data['time_relative'] == t]['bfp_normalized']
                if len(t_data) >= 5:
                    means.append(t_data.mean())
                    stds.append(t_data.std())
                    valid_times.append(t)
            if valid_times:
                means, stds = np.array(means), np.array(stds)
                ax.plot(valid_times, means, color=COLORS[group], linewidth=2, label=group.capitalize())
                ax.fill_between(valid_times, means - stds, means + stds, color=COLORS[group], alpha=0.15)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Activation')
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time relative to activation')
        ax.set_ylabel('BFP (normalized)')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-15, 24)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_bfp_aligned_normalized{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # mNG intensity aligned to activation (by group)
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in groups:
            group_data = df_aligned[df_aligned['group'] == group]
            means, stds, valid_times = [], [], []
            for t in time_range:
                t_data = group_data[group_data['time_relative'] == t]['mean_intensity']
                if len(t_data) >= 5:
                    means.append(t_data.mean())
                    stds.append(t_data.std())
                    valid_times.append(t)
            if valid_times:
                means, stds = np.array(means), np.array(stds)
                ax.plot(valid_times, means, color=COLORS[group], linewidth=2, label=group.capitalize())
                ax.fill_between(valid_times, means - stds, means + stds, color=COLORS[group], alpha=0.15)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time relative to activation')
        ax.set_ylabel('mNG intensity (a.u.)')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(-15, 24)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_mng_aligned_to_activation{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # BFP percent change post-activation boxplot
        if len(df_changes) > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            data_bfp = [df_changes[df_changes['group'] == g]['pct_change'].dropna().values for g in groups]
            data_bfp = [d for d in data_bfp if len(d) > 0]
            groups_with_data = [g for g in groups if len(df_changes[df_changes['group'] == g]) > 0]
            if data_bfp and groups_with_data:
                bp = ax.boxplot(data_bfp, labels=[g.capitalize() for g in groups_with_data],
                                patch_artist=True, widths=0.6)
                for patch, group in zip(bp['boxes'], groups_with_data):
                    patch.set_facecolor(COLORS[group])
                    patch.set_alpha(0.7)
                ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_ylabel('BFP change (%)')
            plt.tight_layout()
            save_figure(fig, output_dir, f"well_{full_name}_bfp_change_post_activation{suffix_clean}", save_pdf, save_svg, subdir="individual")
            plt.close()
        # Individual BFP examples aligned to activation
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in groups:
            group_tracks = df_aligned[df_aligned['group'] == group]['unique_track_id'].unique()
            sample_tracks = np.random.choice(group_tracks, min(5, len(group_tracks)), replace=False)
            for track_id in sample_tracks:
                track_data = df_aligned[df_aligned['unique_track_id'] == track_id].sort_values('time_relative')
                ax.plot(track_data['time_relative'], track_data['bfp_normalized'], color=COLORS[group], alpha=0.4, linewidth=1)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time relative to activation')
        ax.set_ylabel('BFP (normalized)')
        ax.set_xlim(-15, 24)
        ax.set_ylim(0.5, 1.5)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_bfp_aligned_individual{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
    
    return df_aligned, df_changes


def create_napari_lookup_table(df_act, df_meas, output_dir, well, early_min, suffix=""):
    _, _, full_name = parse_well(well)
    positions = []
    
    for _, row in df_act.iterrows():
        track_id = row['unique_track_id']
        track_data = df_meas[df_meas['unique_track_id'] == track_id].sort_values('timepoint')
        if len(track_data) > 0:
            t_act = row['activation_timepoint']
            if not np.isnan(t_act):
                pos_at_act = track_data[track_data['timepoint'] == int(t_act)]
                if len(pos_at_act) > 0:
                    y_at_act = pos_at_act['centroid-0'].values[0]
                    x_at_act = pos_at_act['centroid-1'].values[0]
                else:
                    y_at_act = x_at_act = np.nan
            else:
                y_at_act = x_at_act = np.nan
            
            positions.append({
                'unique_track_id': track_id,
                'fov': row['fov'],
                'track_id_local': int(track_id.split('_')[-1]),
                'activation_group': row['activation_group'],
                'response_group': row.get('response_group', 'unfit'),
                'plateau_value': row.get('plateau_value', np.nan),
                'activation_timepoint': row['activation_timepoint'],
                'max_intensity': row['max_intensity'],
                'y_at_activation': y_at_act,
                'x_at_activation': x_at_act,
            })
    
    df_lookup = pd.DataFrame(positions).sort_values(['activation_group', 'activation_timepoint'])
    filename = f"well_{full_name}_napari_lookup" + (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else "") + ".csv"
    df_lookup.to_csv(output_dir / filename, index=False)
    print(f"Saved: {filename}")
    return df_lookup

def load_images_for_fov(tracking_dir, well, fov):
    """Load image data for a specific FOV."""
    _, _, full_name = parse_well(well)
    fov_dir = Path(tracking_dir) / f"well_{full_name}_FOV{fov}"
    
    images = {}
    
    # Load mNG MIPs
    gfp_file = fov_dir / "gfp_mips.npy"
    if gfp_file.exists():
        images['gfp'] = np.load(gfp_file)
        print(f"    mNG: {images['gfp'].shape}")
    
    # Load BFP/DAPI MIPs
    bfp_file = fov_dir / "dapi_mips.npy"
    if bfp_file.exists():
        images['bfp'] = np.load(bfp_file)
        print(f"    BFP: {images['bfp'].shape}")
    
    # Load tracked masks
    masks_file = fov_dir / "tracked_masks.npy"
    if masks_file.exists():
        images['masks'] = np.load(masks_file)
        print(f"    Masks: {images['masks'].shape}")
    
    return images

def extract_cell_timelapse(images, df_meas, track_id, crop_size=150):
    """
    Extract a time-lapse crop for a single cell across all timepoints.
    
    Returns:
        dict with 'gfp', 'bfp', 'masks' arrays of shape (T, crop_size, crop_size)
    """
    track_data = df_meas[df_meas['unique_track_id'] == track_id].sort_values('timepoint')
    
    if len(track_data) == 0:
        return None
    
    # Determine number of timepoints from images
    n_timepoints = None
    for key in ['gfp', 'bfp', 'masks']:
        if key in images and images[key] is not None:
            n_timepoints = images[key].shape[0]
            img_shape = images[key].shape[1:3]
            break
    
    if n_timepoints is None:
        return None
    
    half = crop_size // 2
    
    # Initialize output arrays
    result = {}
    for key in ['gfp', 'bfp', 'masks']:
        if key in images and images[key] is not None:
            result[key] = np.zeros((n_timepoints, crop_size, crop_size), dtype=images[key].dtype)
    
    # For each timepoint, get centroid and extract crop
    for t in range(n_timepoints):
        # Find centroid at this timepoint
        t_data = track_data[track_data['timepoint'] == t]
        
        if len(t_data) > 0:
            cy = int(t_data.iloc[0]['centroid-0'])
            cx = int(t_data.iloc[0]['centroid-1'])
        else:
            # Interpolate or use nearest timepoint
            if len(track_data) > 0:
                # Use nearest available timepoint
                nearest_idx = (track_data['timepoint'] - t).abs().idxmin()
                cy = int(track_data.loc[nearest_idx, 'centroid-0'])
                cx = int(track_data.loc[nearest_idx, 'centroid-1'])
            else:
                continue
        
        # Extract crop for each channel
        for key in result.keys():
            img = images[key][t]
            
            # Calculate crop bounds with boundary handling
            y_start = max(0, cy - half)
            y_end = min(img.shape[0], cy + half)
            x_start = max(0, cx - half)
            x_end = min(img.shape[1], cx + half)
            
            crop = img[y_start:y_end, x_start:x_end]
            
            # Place in output array (centered)
            out_y_start = half - (cy - y_start)
            out_x_start = half - (cx - x_start)
            
            result[key][t, out_y_start:out_y_start+crop.shape[0], 
                          out_x_start:out_x_start+crop.shape[1]] = crop
    
    return result

def get_cell_crop(images, df_meas, track_id, timepoint, crop_size=200):
    """Extract a crop around a cell at a specific timepoint."""
    track_data = df_meas[(df_meas['unique_track_id'] == track_id) & 
                         (df_meas['timepoint'] == timepoint)]
    
    if len(track_data) == 0:
        return None, None, None
    
    row = track_data.iloc[0]
    cy, cx = int(row['centroid-0']), int(row['centroid-1'])
    
    half = crop_size // 2
    
    crops = {}
    for key, img_stack in images.items():
        if img_stack is None:
            continue
        
        t = min(timepoint, img_stack.shape[0] - 1)
        img = img_stack[t]
        
        # Handle boundary conditions
        y_start = max(0, cy - half)
        y_end = min(img.shape[0], cy + half)
        x_start = max(0, cx - half)
        x_end = min(img.shape[1], cx + half)
        
        crop = img[y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if crop.shape != (crop_size, crop_size):
            padded = np.zeros((crop_size, crop_size), dtype=crop.dtype)
            py_start = half - (cy - y_start)
            px_start = half - (cx - x_start)
            padded[py_start:py_start+crop.shape[0], px_start:px_start+crop.shape[1]] = crop
            crop = padded
        
        crops[key] = crop
    
    return crops, cy, cx


def view_cell_grid(tracking_dir, well, df_act, df_meas, n_per_group=3, crop_size=150,
                   exclude_fovs=None, group_col='activation_group'):
    """
    View a grid of cells in napari: n_per_group cells per group.
    group_col: 'activation_group' (early/average/late) or 'response_group' (low/medium/high)
    Each cell shows full time-lapse with mNG and BFP channels.

    Layout in napari:
        - 9 cells arranged in a 3x3 spatial grid
        - Each cell has full T timepoints
        - mNG and BFP as separate layers
        - Navigate through time with the slider
    """
    if not NAPARI_AVAILABLE:
        print("ERROR: napari is not installed. Install with: pip install napari[all]")
        return

    _, _, full_name = parse_well(well)
    if group_col == 'response_group':
        groups = ['low', 'medium', 'high']
    else:
        groups = ['early', 'average', 'late']
    
    print(f"\n{'='*60}")
    print(f"CREATING CELL GRID VIEW")
    print(f"{'='*60}")
    print(f"  {n_per_group} cells per group, crop size: {crop_size}x{crop_size}")
    
    # Select cells from each group
    selected_cells = []
    for group in groups:
        group_cells = df_act[df_act[group_col] == group].copy()

        if len(group_cells) == 0:
            print(f"  Warning: No cells in {group} group")
            continue

        # Sort by activation timepoint for reproducibility
        group_cells = group_cells.sort_values('activation_timepoint')
        
        # Sample cells (evenly spaced through the distribution)
        n_available = len(group_cells)
        n_sample = min(n_per_group, n_available)
        
        if n_sample < n_available:
            indices = np.linspace(0, n_available - 1, n_sample, dtype=int)
            sampled = group_cells.iloc[indices]
        else:
            sampled = group_cells
        
        for _, row in sampled.iterrows():
            selected_cells.append({
                'track_id': row['unique_track_id'],
                'fov': row['fov'],
                'group': group,
                'activation_t': row['activation_timepoint'],
                'max_intensity': row.get('max_intensity', np.nan)
            })
        
        print(f"  {group.capitalize()}: selected {n_sample} cells")
    
    if len(selected_cells) == 0:
        print("ERROR: No cells selected")
        return
    
    # Group cells by FOV to minimize image loading
    cells_by_fov = {}
    for cell in selected_cells:
        fov = cell['fov']
        if fov not in cells_by_fov:
            cells_by_fov[fov] = []
        cells_by_fov[fov].append(cell)
    
    print(f"\n  Loading images from {len(cells_by_fov)} FOVs...")
    
    # Extract time-lapse crops for each cell
    cell_data = []  # List of dicts with 'gfp', 'bfp', 'group', 'track_id', 'activation_t'
    
    for fov, cells in cells_by_fov.items():
        print(f"  FOV {fov}:")
        images = load_images_for_fov(tracking_dir, well, fov)
        
        if not images:
            print(f"    Warning: Could not load images")
            continue
        
        for cell in cells:
            track_id = cell['track_id']
            
            timelapse = extract_cell_timelapse(images, df_meas, track_id, crop_size)
            
            if timelapse is None:
                print(f"    Warning: Could not extract timelapse for {track_id}")
                continue
            
            cell_data.append({
                'gfp': timelapse.get('gfp'),
                'bfp': timelapse.get('bfp'),
                'masks': timelapse.get('masks'),
                'group': cell['group'],
                'track_id': track_id,
                'activation_t': cell['activation_t'],
                'fov': fov
            })
            print(f"    Extracted: {track_id} ({cell['group']})")
    
    if len(cell_data) == 0:
        print("ERROR: No cell data extracted")
        return
    
    # Organize cells into grid: rows = groups, cols = cells within group
    grid_cells = {group: [] for group in groups}
    for cell in cell_data:
        grid_cells[cell['group']].append(cell)
    
    # Determine grid dimensions
    n_rows = len([g for g in groups if len(grid_cells[g]) > 0])
    n_cols = max(len(grid_cells[g]) for g in groups) if n_rows > 0 else 0
    
    if n_rows == 0 or n_cols == 0:
        print("ERROR: Empty grid")
        return
    
    # Get dimensions from first cell
    first_cell = cell_data[0]
    n_timepoints = first_cell['gfp'].shape[0] if first_cell['gfp'] is not None else 1
    
    print(f"\n  Grid: {n_rows} rows x {n_cols} cols, {n_timepoints} timepoints")
    
    # Create montage arrays: (T, n_rows * crop_size, n_cols * crop_size)
    montage_height = n_rows * crop_size
    montage_width = n_cols * crop_size
    
    gfp_montage = np.zeros((n_timepoints, montage_height, montage_width), dtype=np.float32)
    bfp_montage = np.zeros((n_timepoints, montage_height, montage_width), dtype=np.float32)
    has_bfp = False
    
    # Also create a labels layer to identify cells
    labels_montage = np.zeros((montage_height, montage_width), dtype=np.int32)
    
    # Fill in the montage
    cell_info = []  # For annotation
    row_idx = 0
    
    for group in groups:
        if len(grid_cells[group]) == 0:
            continue
        
        for col_idx, cell in enumerate(grid_cells[group][:n_cols]):
            y_start = row_idx * crop_size
            x_start = col_idx * crop_size
            
            if cell['gfp'] is not None:
                gfp_montage[:, y_start:y_start+crop_size, x_start:x_start+crop_size] = cell['gfp']
            
            if cell['bfp'] is not None:
                bfp_montage[:, y_start:y_start+crop_size, x_start:x_start+crop_size] = cell['bfp']
                has_bfp = True
            
            # Label this cell region
            cell_label = row_idx * n_cols + col_idx + 1
            labels_montage[y_start:y_start+crop_size, x_start:x_start+crop_size] = cell_label
            
            cell_info.append({
                'label': cell_label,
                'group': group,
                'track_id': cell['track_id'],
                'activation_t': cell['activation_t'],
                'row': row_idx,
                'col': col_idx,
                'y_center': y_start + crop_size // 2,
                'x_center': x_start + crop_size // 2
            })
        
        row_idx += 1
    
    # Create napari viewer
    print(f"\n  Launching napari...")
    viewer = napari.Viewer(title=f"Cell Grid - Well {full_name} ({n_rows}x{n_cols})")
    
    # Add mNG layer
    gfp_contrast = [0, np.percentile(gfp_montage[gfp_montage > 0], 99.5)] if gfp_montage.max() > 0 else [0, 1]
    viewer.add_image(gfp_montage, name='mNG', colormap='green', 
                     contrast_limits=gfp_contrast, blending='additive')
    
    # Add BFP layer
    if has_bfp:
        bfp_contrast = [0, np.percentile(bfp_montage[bfp_montage > 0], 99.5)] if bfp_montage.max() > 0 else [0, 1]
        viewer.add_image(bfp_montage, name='BFP', colormap='blue', 
                         contrast_limits=bfp_contrast, blending='additive', visible=True)
    
    # Add grid lines as shapes
    grid_lines = []
    # Horizontal lines
    for i in range(1, n_rows):
        y = i * crop_size
        grid_lines.append(np.array([[y, 0], [y, montage_width]]))
    # Vertical lines
    for j in range(1, n_cols):
        x = j * crop_size
        grid_lines.append(np.array([[0, x], [montage_height, x]]))
    
    if grid_lines:
        viewer.add_shapes(grid_lines, shape_type='line', edge_color='white', 
                         edge_width=2, name='Grid', opacity=0.5)
    
    # Add text annotations for each cell
    text_positions = []
    text_strings = []
    text_colors = []
    
    group_colors_rgb = {
        'early': [0.13, 0.40, 0.67],    # Blue
        'average': [0.30, 0.69, 0.29],  # Green  
        'late': [0.84, 0.15, 0.16]      # Red
    }
    
    for info in cell_info:
        # Position text at top-left of each cell
        text_positions.append([info['y_center'] - crop_size//2 + 10, 
                               info['x_center'] - crop_size//2 + 5])
        text_strings.append(f"{info['group'][:1].upper()}\nt={info['activation_t']:.0f}")
        text_colors.append(group_colors_rgb.get(info['group'], [1, 1, 1]))
    
    # Add points with text
    text_positions = np.array(text_positions)
    viewer.add_points(text_positions, name='Labels', size=1, 
                     face_color=text_colors, opacity=0)
    
    # Add text as properties
    text_layer = viewer.add_points(
        text_positions, 
        name='Cell Info',
        text={'string': text_strings, 'color': 'white', 'size': 10},
        size=1,
        face_color='transparent',
        border_color='transparent'
    )
    
    # Add activation timepoint markers
    activation_points = []
    activation_colors = []
    
    for info in cell_info:
        if not np.isnan(info['activation_t']):
            # Point at (t, y, x)
            activation_points.append([info['activation_t'], info['y_center'], info['x_center']])
            activation_colors.append(group_colors_rgb.get(info['group'], [1, 1, 1]))
    
    if activation_points:
        viewer.add_points(
            np.array(activation_points),
            name='Activation Points',
            size=15,
            face_color=activation_colors,
            border_color='white',
            border_width=0.1,
            symbol='x',
            opacity=0.8
        )
    
    # Print cell info
    print(f"\n  Cell Grid Contents:")
    print(f"  " + "-"*50)
    for info in cell_info:
        print(f"    [{info['row']},{info['col']}] {info['group']:8s} | t_act={info['activation_t']:5.1f} | {info['track_id']}")
    print(f"  " + "-"*50)
    
    print(f"\n  Controls:")
    print(f"    - Use slider to navigate through timepoints")
    print(f"    - Toggle mNG/BFP visibility in layer list")
    print(f"    - 'X' markers show activation timepoints")
    print(f"    - Grid: rows=groups (E/A/L), cols=cells within group")
    
    napari.run()
    
    return cell_info

def view_napari_browser(tracking_dir, well, df_act, df_meas, fov=0, n_examples=5, 
                        exclude_fovs=None):
    """Launch napari viewer to browse cells interactively."""
    if not NAPARI_AVAILABLE:
        print("ERROR: napari is not installed. Install with: pip install napari[all]")
        return
    
    _, _, full_name = parse_well(well)
    
    print(f"\nLoading FOV {fov} for napari browsing...")
    images = load_images_for_fov(tracking_dir, well, fov)
    
    if not images:
        print("ERROR: Could not load images")
        return
    
    # Launch napari
    viewer = napari.Viewer(title=f"Well {full_name} - FOV {fov}")
    
    if 'gfp' in images:
        viewer.add_image(images['gfp'], name='mNG', colormap='green',
                        contrast_limits=[0, np.percentile(images['gfp'], 99.5)])
    
    if 'bfp' in images:
        viewer.add_image(images['bfp'], name='BFP', colormap='blue',
                        contrast_limits=[0, np.percentile(images['bfp'], 99.5)],
                        visible=False)
    
    if 'masks' in images:
        viewer.add_labels(images['masks'], name='Segmentation', opacity=0.3)
    
    # Add points for cells colored by group
    groups = ['early', 'average', 'late']
    group_colors = {'early': 'blue', 'average': 'green', 'late': 'red'}

    fov_cells = df_act[df_act['fov'] == fov]

    for group in groups:
        group_cells = fov_cells[fov_cells['activation_group'] == group]

        if len(group_cells) == 0:
            continue

        points = []
        for _, row in group_cells.iterrows():
            track_id = row['unique_track_id']
            activation_t = row['activation_timepoint']

            if np.isnan(activation_t):
                continue

            # Get position at activation
            track_data = df_meas[(df_meas['unique_track_id'] == track_id) &
                                 (df_meas['timepoint'] == int(activation_t))]

            if len(track_data) > 0:
                cy = track_data.iloc[0]['centroid-0']
                cx = track_data.iloc[0]['centroid-1']
                points.append([activation_t, cy, cx])

        if points:
            points = np.array(points)
            viewer.add_points(points, name=f'{group.capitalize()} cells',
                            face_color=group_colors[group], size=20, opacity=0.7)

    # Add tracks for all measured cells in this FOV, colored to match their nucleus label
    fov_meas = df_meas[df_meas['fov'] == fov]
    fov_track_uids = fov_meas['unique_track_id'].unique()

    if len(fov_track_uids) > 0:
        local_ids = sorted(set(int(uid.split('_')[-1]) for uid in fov_track_uids))
        n = len(local_ids)
        id_to_rank = {lid: i for i, lid in enumerate(local_ids)}

        tracks_data = []
        rank_vals = []
        for uid in fov_track_uids:
            local_id = int(uid.split('_')[-1])
            rank_val = id_to_rank[local_id] / max(n - 1, 1)
            cell_meas = fov_meas[fov_meas['unique_track_id'] == uid].sort_values('timepoint')
            for _, row in cell_meas.iterrows():
                tracks_data.append([local_id, int(row['timepoint']),
                                    row['centroid-0'], row['centroid-1']])
                rank_vals.append(rank_val)

        if tracks_data:
            tracks_array = np.array(tracks_data)
            if 'masks' in images:
                from napari.utils.colormaps import Colormap as NapariColormap, AVAILABLE_COLORMAPS
                seg_layer = viewer.layers['Segmentation']
                colors = [seg_layer.get_color(lid) for lid in local_ids]
                controls = [i / max(n - 1, 1) for i in range(n)]
                # Colormap requires at least 2 control points
                if n == 1:
                    colors = [colors[0], colors[0]]
                    controls = [0.0, 1.0]
                track_cmap = NapariColormap(colors=colors, controls=controls,
                                            name='label_match')
                AVAILABLE_COLORMAPS['label_match'] = track_cmap
                viewer.add_tracks(
                    tracks_array,
                    properties={'label_color': np.array(rank_vals)},
                    color_by='label_color',
                    colormap='label_match',
                    name='Tracks',
                    tail_width=3,
                    tail_length=30,
                )
            else:
                viewer.add_tracks(tracks_array, name='Tracks',
                                  tail_width=3, tail_length=30)

    print(f"\nNapari viewer launched with:")
    print(f"  - mNG channel")
    print(f"  - BFP channel (hidden by default)")
    print(f"  - Segmentation masks")
    print(f"  - Tracks colored to match nucleus labels ({len(fov_track_uids)} cells)")
    print(f"  - Cell positions colored by activation group")
    print(f"\nUse the time slider to navigate through timepoints.")
    
    napari.run()

# ==================== COMPARISON PLOTTING ====================
def plot_comparison_figure(results_unfiltered, results_filtered, output_dir, well, iqr_thresholds, save_pdf=False, save_svg=False,
                           save_individual=False, suffix=""):
    """Create a comparison figure showing unfiltered vs IQR-filtered results."""
    set_publication_style()
    _, _, full_name = parse_well(well)
    
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Unpack results
    (df_act_uf, df_meas_uf, corr_orig_uf, p_orig_uf, corr_corr_uf, p_corr_uf,
     corr_bfp_gfp_uf, p_bfp_gfp_uf, corr_bfp_mng_uf, p_bfp_mng_uf, 
     corr_abs_uf, p_abs_uf, iqr_stats_uf) = results_unfiltered
    
    (df_act_f, df_meas_f, corr_orig_f, p_orig_f, corr_corr_f, p_corr_f,
     corr_bfp_gfp_f, p_bfp_gfp_f, corr_bfp_mng_f, p_bfp_mng_f,
     corr_abs_f, p_abs_f, iqr_stats_f) = results_filtered
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, left=0.06, right=0.98, top=0.92, bottom=0.06)
    fig.suptitle(f'Impact of IQR Baseline Filtering — Well {full_name}', fontsize=16, fontweight='bold', y=0.97)
    
    groups = ['early', 'average', 'late']
    
    # Panel A: Baseline distribution with IQR highlighted
    ax = fig.add_subplot(gs[0, 0])
    baseline_uf = df_act_uf['baseline_intensity'].dropna()
    ax.hist(baseline_uf, bins=50, alpha=0.7, color='#1f77b4', edgecolor='white', label='All cells')
    ax.axvspan(iqr_thresholds['q_low'], iqr_thresholds['q_high'], alpha=0.3, color='#FFD700', 
               label=f"IQR ({iqr_thresholds['percentile_low']}-{iqr_thresholds['percentile_high']}%)")
    ax.axvline(iqr_thresholds['q_low'], color='#FF8C00', linestyle='--', linewidth=2)
    ax.axvline(iqr_thresholds['q_high'], color='#FF8C00', linestyle='--', linewidth=2)
    ax.set_xlabel('Baseline mNG intensity')
    ax.set_ylabel('Count')
    ax.set_title('A   Baseline distribution', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel B: Cell counts comparison
    ax = fig.add_subplot(gs[0, 1])
    n_uf = [len(df_act_uf[df_act_uf['activation_group'] == g]) for g in groups]
    n_f = [len(df_act_f[df_act_f['activation_group'] == g]) for g in groups]
    x = np.arange(len(groups))
    width = 0.35
    bars1 = ax.bar(x - width/2, n_uf, width, label='Unfiltered', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, n_f, width, label='IQR filtered', color='#ff7f0e', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylabel('Cell count')
    ax.set_title('B   Cells per group', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Add retention percentage
    for i, (nuf, nf) in enumerate(zip(n_uf, n_f)):
        if nuf > 0:
            pct = nf / nuf * 100
            ax.text(i + width/2, nf + 1, f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # Panel C: Correlation comparison (log10)
    ax = fig.add_subplot(gs[0, 2])
    corr_values = [abs(corr_orig_uf), abs(corr_orig_f)]
    bars = ax.bar(['Unfiltered', 'IQR filtered'], corr_values, color=['#1f77b4', '#ff7f0e'], alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, corr_values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'r={val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('|Correlation| ($\log_{10}$ baseline)')
    ax.set_title('C   Baseline-activation correlation', loc='left', fontweight='bold')
    ax.set_ylim(0, max(corr_values) * 1.3)
    
    # Panel D: Summary stats
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    
    summary = f"IQR FILTERING SUMMARY\n" + "="*25 + "\n\n"
    summary += f"Thresholds:\n"
    summary += f"  Q{iqr_thresholds['percentile_low']}: {iqr_thresholds['q_low']:.1f}\n"
    summary += f"  Q{iqr_thresholds['percentile_high']}: {iqr_thresholds['q_high']:.1f}\n\n"
    summary += f"Cell retention:\n"
    summary += f"  {len(df_act_f)}/{len(df_act_uf)} ({len(df_act_f)/len(df_act_uf)*100:.1f}%)\n\n"
    summary += f"Correlation change:\n"
    summary += f"  Log₁₀: {corr_orig_uf:.3f} → {corr_orig_f:.3f}\n"
    summary += f"  Abs:   {corr_abs_uf:.3f} → {corr_abs_f:.3f}"
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9, fontfamily='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', edgecolor='#4682b4', alpha=0.9))
    ax.set_title('D   Summary', loc='left', fontweight='bold')
    
    # Panel E: Scatter - Unfiltered
    ax = fig.add_subplot(gs[1, 0:2])
    df_groups_uf = df_act_uf[df_act_uf['activation_group'].isin(groups)].copy()
    df_groups_uf['baseline_log10'] = np.log10(df_groups_uf['baseline_intensity'])
    
    for group in groups:
        gdata = df_groups_uf[df_groups_uf['activation_group'] == group]
        ax.scatter(gdata['baseline_log10'], gdata['activation_timepoint'],
                  color=COLORS[group], alpha=0.5, s=30, label=group.capitalize())
    
    valid = df_groups_uf.dropna(subset=['baseline_log10', 'activation_timepoint'])
    if len(valid) > 2:
        z = np.polyfit(valid['baseline_log10'], valid['activation_timepoint'], 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(valid['baseline_log10'].min(), valid['baseline_log10'].max(), 100)
        ax.plot(x_line, p_fit(x_line), 'k--', linewidth=2)
    
    # Show IQR bounds in log space
    ax.axvline(np.log10(iqr_thresholds['q_low']), color='#FF8C00', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(np.log10(iqr_thresholds['q_high']), color='#FF8C00', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(np.log10(iqr_thresholds['q_low']), np.log10(iqr_thresholds['q_high']), alpha=0.1, color='#FFD700')
    
    ax.text(0.02, 0.98, f'r = {corr_orig_uf:.3f}\nn = {len(valid)}', transform=ax.transAxes, ha='left', va='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax.set_xlabel('Baseline mNG intensity ($\log_{10}$)')
    ax.set_ylabel('Activation time (frames)')
    ax.set_title('E   UNFILTERED — All cells', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel F: Scatter - Filtered
    ax = fig.add_subplot(gs[1, 2:4])
    df_groups_f = df_act_f[df_act_f['activation_group'].isin(groups)].copy()
    df_groups_f['baseline_log10'] = np.log10(df_groups_f['baseline_intensity'])
    
    for group in groups:
        gdata = df_groups_f[df_groups_f['activation_group'] == group]
        ax.scatter(gdata['baseline_log10'], gdata['activation_timepoint'],
                  color=COLORS[group], alpha=0.5, s=30, label=group.capitalize())
    
    valid = df_groups_f.dropna(subset=['baseline_log10', 'activation_timepoint'])
    if len(valid) > 2:
        z = np.polyfit(valid['baseline_log10'], valid['activation_timepoint'], 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(valid['baseline_log10'].min(), valid['baseline_log10'].max(), 100)
        ax.plot(x_line, p_fit(x_line), 'k--', linewidth=2)
    
    ax.text(0.02, 0.98, f'r = {corr_orig_f:.3f}\nn = {len(valid)}', transform=ax.transAxes, ha='left', va='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax.set_xlabel('Baseline mNG intensity ($\log_{10}$)')
    ax.set_ylabel('Activation time (frames)')
    ax.set_title(f'F   IQR FILTERED — Q{iqr_thresholds["percentile_low"]}-Q{iqr_thresholds["percentile_high"]}', loc='left', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel G: Trajectory comparison - Unfiltered
    ax = fig.add_subplot(gs[2, 0:2])
    for group in groups:
        track_ids = df_act_uf[df_act_uf['activation_group'] == group]['unique_track_id'].values
        group_data = df_meas_uf[df_meas_uf['unique_track_id'].isin(track_ids)]
        if len(group_data) > 0:
            mean_traj = group_data.groupby('timepoint')['mean_intensity'].agg(['mean', 'std'])
            ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2, 
                   label=f'{group.capitalize()} (n={len(track_ids)})')
            ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                           mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('mNG intensity (a.u.)')
    ax.set_title('G   Trajectories — UNFILTERED', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    # Panel H: Trajectory comparison - Filtered
    ax = fig.add_subplot(gs[2, 2:4])
    for group in groups:
        track_ids = df_act_f[df_act_f['activation_group'] == group]['unique_track_id'].values
        group_data = df_meas_f[df_meas_f['unique_track_id'].isin(track_ids)]
        if len(group_data) > 0:
            mean_traj = group_data.groupby('timepoint')['mean_intensity'].agg(['mean', 'std'])
            ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2, 
                   label=f'{group.capitalize()} (n={len(track_ids)})')
            ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                           mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('mNG intensity (a.u.)')
    ax.set_title('H   Trajectories — IQR FILTERED', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    save_figure(fig, fig_dir, f"well_{full_name}_IQR_comparison", save_pdf, save_svg)
    plt.close()
    
    if save_individual:
        suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
        # Panel A: Baseline distribution with IQR highlighted
        fig, ax = plt.subplots(figsize=(5, 4))
        baseline_uf = df_act_uf['baseline_intensity'].dropna()
        ax.hist(baseline_uf, bins=50, alpha=0.7, color='#1f77b4', edgecolor='white', label='All cells')
        ax.axvspan(iqr_thresholds['q_low'], iqr_thresholds['q_high'], alpha=0.3, color='#FFD700',
                   label=f"IQR ({iqr_thresholds['percentile_low']}-{iqr_thresholds['percentile_high']}%)")
        ax.axvline(iqr_thresholds['q_low'], color='#FF8C00', linestyle='--', linewidth=2)
        ax.axvline(iqr_thresholds['q_high'], color='#FF8C00', linestyle='--', linewidth=2)
        ax.set_xlabel('Baseline mNG intensity')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_iqr_baseline_distribution{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Panel B: Cell counts comparison
        fig, ax = plt.subplots(figsize=(5, 4))
        n_uf = [len(df_act_uf[df_act_uf['activation_group'] == g]) for g in groups]
        n_f = [len(df_act_f[df_act_f['activation_group'] == g]) for g in groups]
        x = np.arange(len(groups))
        width = 0.35
        ax.bar(x - width/2, n_uf, width, label='Unfiltered', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, n_f, width, label='IQR filtered', color='#ff7f0e', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in groups])
        ax.set_ylabel('Cell count')
        ax.legend(loc='upper right', fontsize=8)
        for i, (nuf, nf) in enumerate(zip(n_uf, n_f)):
            if nuf > 0:
                pct = nf / nuf * 100
                ax.text(i + width/2, nf + 1, f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_iqr_cell_counts{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Panel C: Correlation comparison
        fig, ax = plt.subplots(figsize=(5, 4))
        corr_values = [abs(corr_orig_uf), abs(corr_orig_f)]
        bars = ax.bar(['Unfiltered', 'IQR filtered'], corr_values, color=['#1f77b4', '#ff7f0e'], alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, corr_values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'r={val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('|Correlation| ($\log_{10}$ baseline)')
        ax.set_ylim(0, max(corr_values) * 1.3)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_iqr_correlation_comparison{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Panel E: Scatter - Unfiltered
        fig, ax = plt.subplots(figsize=(6, 4))
        df_groups_uf = df_act_uf[df_act_uf['activation_group'].isin(groups)].copy()
        df_groups_uf['baseline_log10'] = np.log10(df_groups_uf['baseline_intensity'])
        for group in groups:
            gdata = df_groups_uf[df_groups_uf['activation_group'] == group]
            ax.scatter(gdata['baseline_log10'], gdata['activation_timepoint'],
                      color=COLORS[group], alpha=0.5, s=30, label=group.capitalize())
        valid = df_groups_uf.dropna(subset=['baseline_log10', 'activation_timepoint'])
        if len(valid) > 2:
            z = np.polyfit(valid['baseline_log10'], valid['activation_timepoint'], 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(valid['baseline_log10'].min(), valid['baseline_log10'].max(), 100)
            ax.plot(x_line, p_fit(x_line), 'k--', linewidth=2)
        ax.axvline(np.log10(iqr_thresholds['q_low']), color='#FF8C00', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(np.log10(iqr_thresholds['q_high']), color='#FF8C00', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(np.log10(iqr_thresholds['q_low']), np.log10(iqr_thresholds['q_high']), alpha=0.1, color='#FFD700')
        ax.text(0.02, 0.98, f'r = {corr_orig_uf:.3f}\nn = {len(valid)}', transform=ax.transAxes, ha='left', va='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        ax.set_xlabel('Baseline mNG intensity ($\log_{10}$)')
        ax.set_ylabel('Activation time (frames)')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_iqr_scatter_unfiltered{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Panel F: Scatter - Filtered
        fig, ax = plt.subplots(figsize=(6, 4))
        df_groups_f = df_act_f[df_act_f['activation_group'].isin(groups)].copy()
        df_groups_f['baseline_log10'] = np.log10(df_groups_f['baseline_intensity'])
        for group in groups:
            gdata = df_groups_f[df_groups_f['activation_group'] == group]
            ax.scatter(gdata['baseline_log10'], gdata['activation_timepoint'],
                      color=COLORS[group], alpha=0.5, s=30, label=group.capitalize())
        valid = df_groups_f.dropna(subset=['baseline_log10', 'activation_timepoint'])
        if len(valid) > 2:
            z = np.polyfit(valid['baseline_log10'], valid['activation_timepoint'], 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(valid['baseline_log10'].min(), valid['baseline_log10'].max(), 100)
            ax.plot(x_line, p_fit(x_line), 'k--', linewidth=2)
        ax.text(0.02, 0.98, f'r = {corr_orig_f:.3f}\nn = {len(valid)}', transform=ax.transAxes, ha='left', va='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        ax.set_xlabel('Baseline mNG intensity ($\log_{10}$)')
        ax.set_ylabel('Activation time (frames)')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_iqr_scatter_filtered{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Panel G: Trajectories - Unfiltered
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in groups:
            track_ids = df_act_uf[df_act_uf['activation_group'] == group]['unique_track_id'].values
            group_data = df_meas_uf[df_meas_uf['unique_track_id'].isin(track_ids)]
            if len(group_data) > 0:
                mean_traj = group_data.groupby('timepoint')['mean_intensity'].agg(['mean', 'std'])
                ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2,
                       label=f'{group.capitalize()} (n={len(track_ids)})')
                ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                               mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('mNG intensity (a.u.)')
        ax.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_iqr_trajectories_unfiltered{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
        # Panel H: Trajectories - Filtered
        fig, ax = plt.subplots(figsize=(6, 4))
        for group in groups:
            track_ids = df_act_f[df_act_f['activation_group'] == group]['unique_track_id'].values
            group_data = df_meas_f[df_meas_f['unique_track_id'].isin(track_ids)]
            if len(group_data) > 0:
                mean_traj = group_data.groupby('timepoint')['mean_intensity'].agg(['mean', 'std'])
                ax.plot(mean_traj.index, mean_traj['mean'], color=COLORS[group], linewidth=2,
                       label=f'{group.capitalize()} (n={len(track_ids)})')
                ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'],
                               mean_traj['mean'] + mean_traj['std'], alpha=0.15, color=COLORS[group])
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('mNG intensity (a.u.)')
        ax.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        save_figure(fig, output_dir, f"well_{full_name}_iqr_trajectories_filtered{suffix_clean}", save_pdf, save_svg, subdir="individual")
        plt.close()
    
    print(f"\nSaved: well_{full_name}_IQR_comparison.png")


def plot_cell_death(df_act, df_meas, output_dir, well,
                   save_pdf=False, save_svg=False, suffix=""):
    """
    Visualise cell death events inferred from nuclear morphology.

    Panel A — Nuclear area over time: mean ± SEM for surviving vs dying cells
    Panel B — Death event timing: histogram of probable_death_timepoint
    Panel C — Death type breakdown by response group (stacked bar)
    Panel D — Frames from activation to death (infected dying cells only)
    Panel E — Nuclear area trajectories aligned to death timepoint (mean ± SEM)
    Panel F — Cumulative death fraction over time by response group
    """
    if 'probable_death_timepoint' not in df_act.columns:
        print("  Skipping cell death plot: compute_cell_death not run")
        return

    set_publication_style()
    _, _, full_name = parse_well(well)
    suffix_clean = f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}" if suffix else ""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    r_groups = ['low', 'medium', 'high']
    r_labels = ['Low', 'Medium', 'High']
    dead_mask    = df_act['probable_death_timepoint'].notna()
    df_dead      = df_act[dead_mask].copy()
    df_surviving = df_act[~dead_mask].copy()

    # Uninfected tracks = in df_meas but not in df_act (non-activated cells)
    act_ids   = set(df_act['unique_track_id'])
    uninf_ids = set(df_meas['unique_track_id']) - act_ids

    # Non-activating track death detection — same solidity-first logic as
    # compute_cell_death, plus a minimum track length of 15 frames.
    _MIN_TRACK_LEN     = 15
    _MIN_SUSTAINED     = 3
    _SOL_DROP_THR      = 0.15   # same as solidity_drop_threshold default
    _has_shape_uninf   = 'solidity' in df_meas.columns
    uninf_death_tps    = {}
    if len(uninf_ids) > 0:
        meas_s = df_meas[df_meas['unique_track_id'].isin(uninf_ids)].sort_values(
            ['unique_track_id', 'timepoint'])
        for tid, track in meas_s.groupby('unique_track_id'):
            track = track.reset_index(drop=True)
            if len(track) < _MIN_TRACK_LEN:
                continue
            base_area = track.head(3)['area_pixels'].mean()
            if base_area <= 0:
                continue
            area_s = track['area_pixels'].rolling(3, center=True, min_periods=1).mean()

            if _has_shape_uninf:
                base_sol = track.head(3)['solidity'].mean()
                sol_s_u  = track['solidity'].rolling(3, center=True, min_periods=1).mean()
            else:
                base_sol = np.nan
                sol_s_u  = None

            tp = np.nan

            # (a) Fragmentation — sustained solidity drop
            if _has_shape_uninf:
                sol_mask = sol_s_u < (base_sol - _SOL_DROP_THR)
                first_sol = _first_sustained(sol_mask, track['timepoint'], _MIN_SUSTAINED)
                if not np.isnan(first_sol):
                    tp = first_sol

            # (b) Area collapse — require solidity confirmation
            collapse  = area_s < 0.5 * base_area
            first_col = _first_sustained(collapse, track['timepoint'], _MIN_SUSTAINED)
            if not np.isnan(first_col):
                accept = False
                if _has_shape_uninf:
                    post = track[track['timepoint'] >= first_col].head(_MIN_SUSTAINED)
                    if len(post) > 0:
                        accept = (base_sol - sol_s_u.loc[post.index].mean()) >= _SOL_DROP_THR
                else:
                    accept = True
                if accept and (np.isnan(tp) or first_col < tp):
                    tp = first_col

            # (c) Swelling
            spike       = area_s > 2.0 * base_area
            first_spike = _first_sustained(spike, track['timepoint'], _MIN_SUSTAINED)
            if not np.isnan(first_spike) and (np.isnan(tp) or first_spike < tp):
                tp = first_spike

            if not np.isnan(tp):
                uninf_death_tps[tid] = tp

    n_uninf_dead  = len(uninf_death_tps)
    n_uninf_total = len(uninf_ids)
    n_uninf_long  = sum(
        1 for tid in uninf_ids
        if len(df_meas[df_meas['unique_track_id'] == tid]) >= _MIN_TRACK_LEN
    )
    print(f"  Plotting: {dead_mask.sum()} dying / {(~dead_mask).sum()} surviving infected cells; "
          f"{n_uninf_dead}/{n_uninf_long} non-activating cells (≥{_MIN_TRACK_LEN} frames) "
          f"flagged as dying  [{n_uninf_total} total non-activating tracks]")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f'Cell Death Analysis (Nuclear Morphology) — Well {full_name}'
        f'{(" — " + suffix) if suffix else ""}',
        fontsize=13, fontweight='bold', y=0.98)

    # ── Panel A: mean area over time — dying vs surviving vs uninfected ───────
    ax = axes[0, 0]
    if 'area_pixels' in df_meas.columns:
        dead_ids = set(df_dead['unique_track_id'])
        surv_ids = set(df_surviving['unique_track_id'])
        groups_a = [
            (dead_ids,   'Infected dying',     '#D62728'),
            (surv_ids,   'Infected surviving', '#1F77B4'),
            (uninf_ids,  'Non-activating',      '#2CA02C'),
        ]
        for ids, label, color in groups_a:
            sub = df_meas[df_meas['unique_track_id'].isin(ids)]
            if len(sub) == 0:
                continue
            grp = sub.groupby('timepoint')['area_pixels'].agg(['mean', 'sem', 'count'])
            grp = grp[grp['count'] >= 3]
            ls = '--' if label == 'Non-activating' else '-'
            ax.plot(grp.index, grp['mean'], color=color, linewidth=2,
                    linestyle=ls, label=f'{label} (n={len(ids)})')
            ax.fill_between(grp.index,
                            grp['mean'] - grp['sem'],
                            grp['mean'] + grp['sem'],
                            alpha=0.15, color=color)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Nuclear area (px²)')
        ax.legend(fontsize=9)
    ax.set_title('A   Nuclear Area Over Time', loc='left', fontweight='bold')

    # ── Panel B: death timing histogram ──────────────────────────────────────
    ax = axes[0, 1]
    if len(df_dead) > 0:
        death_tps = df_dead['probable_death_timepoint'].dropna()
        ax.hist(death_tps, bins=20, color='#D62728', alpha=0.75, edgecolor='white')
        ax.axvline(death_tps.median(), color='black', linestyle='--',
                   linewidth=1.5, label=f'Median = {death_tps.median():.0f}')
        ax.set_xlabel('Timepoint of probable death')
        ax.set_ylabel('Number of cells')
        ax.legend(fontsize=9)
    ax.set_title('B   Death Event Timing', loc='left', fontweight='bold')

    # ── Panel C: death type by response group ────────────────────────────────
    ax = axes[0, 2]
    death_types = ['condensation', 'swelling', 'necrosis', 'unknown']
    type_colors = {'condensation': '#FF7F0E', 'swelling': '#D62728',
                   'necrosis': '#9467BD', 'unknown': '#7F7F7F'}
    df_r_dead = df_dead[df_dead['response_group'].isin(r_groups)]
    if len(df_r_dead) > 0:
        bottoms = np.zeros(len(r_groups))
        group_totals = [len(df_act[df_act['response_group'] == g]) for g in r_groups]
        for dt in death_types:
            fracs = [
                100 * len(df_r_dead[(df_r_dead['response_group'] == g) &
                                     (df_r_dead['death_type'] == dt)]) / max(tot, 1)
                for g, tot in zip(r_groups, group_totals)
            ]
            ax.bar(r_labels, fracs, bottom=bottoms,
                   color=type_colors[dt], alpha=0.8, label=dt.capitalize(), width=0.6)
            bottoms += np.array(fracs)
        ax.set_ylabel('% of cells')
        ax.legend(fontsize=8, loc='upper right')
    ax.set_title('C   Death Type by Response Group', loc='left', fontweight='bold')

    # ── Panel D: frames from activation to death ─────────────────────────────
    ax = axes[1, 0]
    df_act_dead = df_dead[df_dead['activates'] == True].dropna(subset=['frames_post_activation_death'])
    if len(df_act_dead) >= 3:
        for g, label in zip(r_groups, r_labels):
            sub = df_act_dead[df_act_dead['response_group'] == g]['frames_post_activation_death']
            if len(sub) > 0:
                ax.scatter(np.full(len(sub), label), sub,
                           color=RESPONSE_COLORS[g], alpha=0.6, s=25, zorder=3)
        bp_data = [df_act_dead[df_act_dead['response_group'] == g]['frames_post_activation_death']
                   .dropna().values for g in r_groups]
        bp = ax.boxplot(bp_data, labels=r_labels, patch_artist=True, widths=0.5)
        for patch, g in zip(bp['boxes'], r_groups):
            patch.set_facecolor(RESPONSE_COLORS[g])
            patch.set_alpha(0.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel('Frames from activation to death')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    ax.set_title('D   Time from Infection to Death', loc='left', fontweight='bold')

    # ── Panel E: area trajectories aligned to death ───────────────────────────
    ax = axes[1, 1]
    if 'area_pixels' in df_meas.columns and len(df_dead) > 0:
        death_time_map = df_dead.set_index('unique_track_id')['probable_death_timepoint'].to_dict()
        align_range = np.arange(-10, 8)
        buckets = {t: [] for t in align_range}
        for tid, t_death in death_time_map.items():
            track = df_meas[df_meas['unique_track_id'] == tid].sort_values('timepoint')
            if len(track) == 0:
                continue
            base_area = track.head(3)['area_pixels'].mean()
            if base_area <= 0:
                continue
            for _, row in track.iterrows():
                t_rel = int(row['timepoint'] - t_death)
                if t_rel in buckets:
                    buckets[t_rel].append(row['area_pixels'] / base_area)
        means, sems, valid_t = [], [], []
        for t in align_range:
            vals = np.array([v for v in buckets[t] if not np.isnan(v)])
            if len(vals) >= 3:
                means.append(vals.mean())
                sems.append(vals.std() / np.sqrt(len(vals)))
                valid_t.append(t)
        if valid_t:
            means = np.array(means)
            sems  = np.array(sems)
            ax.plot(valid_t, means, color='#D62728', linewidth=2)
            ax.fill_between(valid_t, means - sems, means + sems, alpha=0.2, color='#D62728')
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(1, color='gray', linestyle=':', linewidth=1)
            ax.set_xlabel('Time relative to death event (frames)')
            ax.set_ylabel('Normalised nuclear area\n(relative to cell baseline)')
    ax.set_title('E   Area Trajectory Aligned to Death (mean ± SEM)', loc='left', fontweight='bold')

    # ── Panel F: cumulative death fraction over time by response group ────────
    ax = axes[1, 2]
    t_max = int(df_meas['timepoint'].max()) if len(df_meas) > 0 else 50
    timepoints = np.arange(0, t_max + 1)
    for g, label in zip(r_groups, r_labels):
        sub = df_act[df_act['response_group'] == g]
        n_total_g = len(sub)
        if n_total_g == 0:
            continue
        dead_tps = sub['probable_death_timepoint'].dropna().values
        cum_frac = np.array([
            100 * (dead_tps <= t).sum() / n_total_g for t in timepoints
        ])
        ax.plot(timepoints, cum_frac, color=RESPONSE_COLORS[g],
                linewidth=2, label=f'{label} (n={n_total_g})')
    # Also show overall infected
    n_all = len(df_act)
    all_dead_tps = df_act['probable_death_timepoint'].dropna().values
    cum_all = np.array([100 * (all_dead_tps <= t).sum() / n_all for t in timepoints])
    ax.plot(timepoints, cum_all, color='black', linewidth=1.5,
            linestyle='--', label=f'All infected (n={n_all})')
    # Non-activating cells reference line
    if n_uninf_total > 0:
        uninf_dead_arr = np.array(list(uninf_death_tps.values()))
        cum_uninf = np.array([
            100 * (uninf_dead_arr <= t).sum() / n_uninf_total
            if len(uninf_dead_arr) > 0 else 0
            for t in timepoints
        ])
        ax.plot(timepoints, cum_uninf, color='#2CA02C', linewidth=1.5,
                linestyle=':', label=f'Non-activating (n={n_uninf_total}, ≥15fr)')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Cumulative % dead')
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    ax.set_title('F   Cumulative Death Fraction Over Time', loc='left', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir, f"well_{full_name}_cell_death{suffix_clean}", save_pdf, save_svg)
    plt.close()
    print(f"  Saved cell death panel")


# ==================== NAPARI DEATH VIEWER ====================
def view_death_examples_napari(df_act, df_meas, fov,
                                zarr_path=None, zarr_row=None, zarr_well=None,
                                nucleus_channel=0,
                                bfp_channel=None,
                                mng_channel=None,
                                n_dying=8, n_surviving=8,
                                response_group=None,
                                random_seed=42):
    """
    Open a Napari viewer showing examples of dying and non-dying tracked cells.

    Layers added
    ------------
    - Nucleus channel (MIP over Z)  — if zarr_path / zarr_row / zarr_well provided
    - Dying cell tracks             — one layer per response group (group colour)
    - Surviving cell tracks         — grey
    - Death markers                 — yellow star; appears at the detected death
                                      timepoint and remains visible afterwards

    Parameters
    ----------
    df_act        : DataFrame returned by compute_cell_death (must have
                    'probable_death_timepoint', 'response_group', 'fov')
    df_meas       : per-timepoint measurement DataFrame (must have 'centroid-0',
                    'centroid-1', 'timepoint', 'fov', 'unique_track_id')
    fov           : integer FOV to visualise
    zarr_path     : path to the .zarr store (str or Path); optional
    zarr_row      : row key inside the zarr store, e.g. 'B'
    zarr_well     : well key inside the zarr store, e.g. '03'
    nucleus_channel : channel index for the nucleus image (default 0)
    n_dying       : how many dying cells to sample
    n_surviving   : how many surviving cells to sample
    response_group: if given, restrict dying sample to this group
                    ('low' | 'medium' | 'high')
    random_seed   : for reproducible sampling

    Usage example
    -------------
    # After run_complete_analysis has populated df_act with death columns:
    view_death_examples_napari(
        df_act, df_meas, fov=3,
        zarr_path="/path/to/experiment.zarr",
        zarr_row="B", zarr_well="03",
        nucleus_channel=0,
    )
    """
    try:
        import napari
    except ImportError:
        raise ImportError("napari is not installed. Run: pip install napari[all]")

    import zarr as _zarr

    rng = np.random.default_rng(random_seed)

    # ── Filter to this FOV ───────────────────────────────────────────────────
    fov_act  = df_act[df_act['fov'] == fov].copy()
    fov_meas = df_meas[df_meas['fov'] == fov].copy()

    if response_group is not None:
        dying_pool = fov_act[
            fov_act['probable_death_timepoint'].notna() &
            (fov_act['response_group'] == response_group)
        ]
    else:
        dying_pool = fov_act[fov_act['probable_death_timepoint'].notna()]

    surv_pool = fov_act[fov_act['probable_death_timepoint'].isna()]

    n_d = min(n_dying,    len(dying_pool))
    n_s = min(n_surviving, len(surv_pool))

    if n_d == 0:
        print(f"  Warning: no dying cells found in FOV {fov}"
              + (f" for group '{response_group}'" if response_group else ""))
    if n_s == 0:
        print(f"  Warning: no surviving cells found in FOV {fov}")

    sample_dying = dying_pool.sample(n_d, random_state=random_seed) if n_d > 0 else dying_pool.iloc[:0]
    sample_surv  = surv_pool.sample(n_s,  random_state=random_seed) if n_s > 0 else surv_pool.iloc[:0]

    selected_ids = set(sample_dying['unique_track_id']) | set(sample_surv['unique_track_id'])
    meas_sel = fov_meas[fov_meas['unique_track_id'].isin(selected_ids)].copy()

    death_tp_map = dict(zip(fov_act['unique_track_id'],
                            fov_act['probable_death_timepoint']))
    group_map    = dict(zip(fov_act['unique_track_id'],
                            fov_act['response_group']))

    # ── Load images if zarr path provided ───────────────────────────────────
    channel_imgs = {}   # name -> (colormap, (T, Y, X) MIP array)
    if zarr_path is not None and zarr_row is not None and zarr_well is not None:
        try:
            store    = _zarr.open(str(zarr_path), mode='r')
            data_arr = store[zarr_row][zarr_well][str(fov)]['0']
            n_ch = data_arr.shape[1]
            print(f"  Zarr array shape for FOV {fov}: {data_arr.shape}  ({n_ch} channels)")
        except Exception as e:
            print(f"  Warning: could not open zarr array — {e}")
            data_arr = None

        if data_arr is not None:
            channels_to_load = [
                ('Nucleus', nucleus_channel, 'gray'),
                ('BFP',     bfp_channel,    'bop blue'),
                ('mNG',     mng_channel,    'green'),
            ]
            for ch_name, ch_idx, cmap in channels_to_load:
                if ch_idx is None:
                    continue
                try:
                    mip = np.array(data_arr[:, ch_idx]).max(axis=1)
                    channel_imgs[ch_name] = (cmap, mip)
                    print(f"  Loaded {ch_name} (channel {ch_idx}): "
                          f"min={mip.min():.0f}  max={mip.max():.0f}")
                except Exception as e:
                    print(f"  Warning: could not load {ch_name} (channel {ch_idx}) — {e}")

    # ── Napari viewer ────────────────────────────────────────────────────────
    viewer = napari.Viewer(title=f'Cell death inspector — FOV {fov}')

    for ch_name, (cmap, imgs) in channel_imgs.items():
        p_low  = float(np.percentile(imgs,  1))
        p_high = float(np.percentile(imgs, 99.5))
        viewer.add_image(imgs, name=f'{ch_name} (MIP)', colormap=cmap,
                         blending='additive',
                         contrast_limits=[p_low, p_high])

    # ── Surviving tracks — grey ──────────────────────────────────────────────
    if n_s > 0:
        surv_ids  = set(sample_surv['unique_track_id'])
        surv_meas = meas_sel[meas_sel['unique_track_id'].isin(surv_ids)].copy()
        surv_imap = {uid: i for i, uid in enumerate(sorted(surv_ids))}
        surv_meas['_tid'] = surv_meas['unique_track_id'].map(surv_imap)
        surv_data = surv_meas[['_tid', 'timepoint', 'centroid-0', 'centroid-1']].values

        viewer.add_tracks(surv_data, name=f'Surviving (n={n_s})',
                          tail_length=15, color_by='track_id',
                          colormap='gray')

    # ── Dying tracks — one layer per response group ──────────────────────────
    # Use napari.utils.colormaps.Colormap with both endpoints set to the same
    # colour so every track in the layer appears as a flat uniform colour.
    from napari.utils.colormaps import Colormap as NapariColormap, AVAILABLE_COLORMAPS

    dying_ids = set(sample_dying['unique_track_id'])
    for grp in ['low', 'medium', 'high']:
        grp_ids = {uid for uid in dying_ids if group_map.get(uid) == grp}
        if not grp_ids:
            continue
        grp_meas = meas_sel[meas_sel['unique_track_id'].isin(grp_ids)].copy()
        grp_imap = {uid: i for i, uid in enumerate(sorted(grp_ids))}
        grp_meas['_tid'] = grp_meas['unique_track_id'].map(grp_imap)
        grp_data = grp_meas[['_tid', 'timepoint', 'centroid-0', 'centroid-1']].values
        color_hex = RESPONSE_COLORS.get(grp, '#D62728')
        r, g_c, b = [int(color_hex[i:i+2], 16)/255 for i in (1, 3, 5)]
        # Register a flat (uniform) colormap so every track gets the group colour.
        # Must be registered by name before passing the name string to add_tracks.
        cmap_name = f'death_{grp}'
        AVAILABLE_COLORMAPS[cmap_name] = NapariColormap(
            colors=[[r, g_c, b, 1.0], [r, g_c, b, 1.0]],
            name=cmap_name,
        )
        viewer.add_tracks(grp_data, name=f'Dying {grp} (n={len(grp_ids)})',
                          tail_length=15, color_by='track_id',
                          colormap=cmap_name)

    # ── Death marker points ──────────────────────────────────────────────────
    death_rows = []
    for uid in dying_ids:
        dtp = death_tp_map.get(uid, np.nan)
        if np.isnan(dtp):
            continue
        cell_meas = meas_sel[meas_sel['unique_track_id'] == uid].sort_values('timepoint')
        # Snap to closest measured timepoint
        idx = (cell_meas['timepoint'] - dtp).abs().idxmin()
        row = cell_meas.loc[idx]
        death_rows.append([int(dtp), float(row['centroid-0']), float(row['centroid-1'])])

    if death_rows:
        death_coords = np.array(death_rows)   # (N, 3): [t, y, x]
        death_tps    = death_coords[:, 0]

        death_layer = viewer.add_points(
            death_coords,
            name='Death markers (★)',
            symbol='star',
            size=22,
            face_color='yellow',
            border_color='red',
            border_width=0.15,
            opacity=0.95,
            shown=np.zeros(len(death_rows), dtype=bool),   # hidden until reached
        )

        # ── Callback: reveal marker once its timepoint is reached ────────────
        @viewer.dims.events.current_step.connect
        def _show_death_markers(event):
            t = viewer.dims.current_step[0]
            death_layer.shown = death_tps <= t

        # Initialise for the current step
        _show_death_markers(None)

    print(f"\n  Napari viewer ready — FOV {fov}")
    print(f"    Dying   : {n_d} tracks  ({', '.join(str(uid) for uid in list(dying_ids)[:3])}{'...' if n_d > 3 else ''})")
    print(f"    Surviving: {n_s} tracks")
    print(f"    Death markers: yellow ★ appears at detected death timepoint")
    print(f"    Tip: use the time slider at the bottom to scrub through frames")

    napari.run()
    return viewer


# ==================== NAPARI CLASSIFIER VIEWER ====================
def view_classifier_predictions_napari(classifier_csv, df_meas, fov,
                                        zarr_path=None, zarr_row=None, zarr_well=None,
                                        nucleus_channel=0, bfp_channel=None,
                                        mng_channel=None):
    """
    Show classifier-predicted deaths for one FOV in Napari.

    Layers
    ------
    Nucleus (BFP) / mNG  — raw images (if zarr provided)
    Dead tracks          — red; all tracks the classifier flagged as deaths
    Alive tracks         — gray; all other tracked cells
    Death probability    — same tracks coloured by P(death) 0→1 (plasma colormap)
    Death markers (★)    — yellow star appearing at the predicted death timepoint
    """
    try:
        import napari
    except ImportError:
        raise ImportError("napari is not installed. Run: pip install napari[all]")
    from napari.utils.colormaps import AVAILABLE_COLORMAPS
    from napari.utils.colormaps import Colormap as NapariColormap

    # ── Load classifier results ───────────────────────────────────────────────
    classifier_csv = Path(classifier_csv)
    if not classifier_csv.exists():
        raise FileNotFoundError(f"Classifier results CSV not found: {classifier_csv}\n"
                                f"Run --apply-death-classifier first.")
    df_clf = pd.read_csv(classifier_csv)
    fov_clf = df_clf[df_clf['fov'] == fov].copy()
    if fov_clf.empty:
        raise ValueError(f"No classifier results for FOV {fov} in {classifier_csv}")

    meas_fov = df_meas[df_meas['fov'] == fov].copy()

    dead_tids  = set(fov_clf.loc[fov_clf['probable_death_timepoint'].notna(),
                                 'unique_track_id'])
    prob_map   = fov_clf.set_index('unique_track_id')['death_probability'].to_dict() \
                 if 'death_probability' in fov_clf.columns else {}

    n_dead  = len(dead_tids)
    n_total = fov_clf['unique_track_id'].nunique()
    print(f"\n  Classifier viewer — FOV {fov}: {n_dead}/{n_total} predicted dead")

    viewer = napari.Viewer(title=f"Classifier Predictions — FOV {fov}")

    # ── Images ────────────────────────────────────────────────────────────────
    if zarr_path is not None:
        try:
            import zarr
            store    = zarr.open(zarr_path, mode='r')
            well_key = zarr_well
            if zarr_row and well_key not in store[zarr_row]:
                stripped = str(int(zarr_well))
                if stripped in store[zarr_row]:
                    well_key = stripped
            data_arr = store[zarr_row][well_key][str(fov)]['0']
            T, C, Z, Y, X = data_arr.shape

            def _mip(ch):
                return np.stack([np.max(data_arr[t, ch, :, :, :], axis=0)
                                 for t in range(T)], axis=0)

            nuc = _mip(nucleus_channel)
            viewer.add_image(nuc, name='Nucleus (BFP)', colormap='cyan',
                             blending='additive',
                             contrast_limits=[nuc.min(), np.percentile(nuc, 99.5)])
            if mng_channel is not None and mng_channel != nucleus_channel:
                mng = _mip(mng_channel)
                viewer.add_image(mng, name='mNG', colormap='green',
                                 blending='additive',
                                 contrast_limits=[mng.min(), np.percentile(mng, 99.5)])
        except Exception as e:
            print(f"  Warning: could not load images: {e}")

    # ── Build track arrays ────────────────────────────────────────────────────
    all_uids   = meas_fov['unique_track_id'].unique()
    uid_to_int = {uid: i for i, uid in enumerate(all_uids)}

    dead_rows, alive_rows = [], []
    prob_rows, prob_vals  = [], []

    for tid, grp in meas_fov.sort_values('timepoint').groupby('unique_track_id'):
        tid_int = uid_to_int[tid]
        p       = float(prob_map.get(tid, 0.0))
        for _, row in grp.iterrows():
            entry = [tid_int, int(row['timepoint']),
                     float(row['centroid-0']), float(row['centroid-1'])]
            prob_rows.append(entry)
            prob_vals.append(p)
            if tid in dead_tids:
                dead_rows.append(entry)
            else:
                alive_rows.append(entry)

    # Register colormaps
    for cname, colors in [
        ('dead_red',  [[0, 0, 0, 1], [0.9, 0.1, 0.1, 1]]),
        ('alive_gray',[[0, 0, 0, 1], [0.55, 0.55, 0.55, 1]]),
    ]:
        if cname not in AVAILABLE_COLORMAPS:
            AVAILABLE_COLORMAPS[cname] = NapariColormap(
                colors=colors, name=cname)

    if alive_rows:
        viewer.add_tracks(np.array(alive_rows, dtype=float),
                          name=f'Alive ({n_total - n_dead})',
                          colormap='alive_gray', opacity=0.4, tail_length=5)
    if dead_rows:
        viewer.add_tracks(np.array(dead_rows, dtype=float),
                          name=f'Dead — classifier ({n_dead})',
                          colormap='dead_red', opacity=0.9, tail_length=8)

    # Probability layer (plasma gradient)
    if prob_rows and prob_map:
        arr_prob  = np.array(prob_rows, dtype=float)
        prob_arr  = np.array(prob_vals, dtype=float)
        viewer.add_tracks(arr_prob,
                          properties={'P(death)': prob_arr},
                          name='Death probability (plasma)',
                          color_by='P(death)', colormap='plasma',
                          opacity=0.7, tail_length=5)

    # ── Death markers ─────────────────────────────────────────────────────────
    death_rows = []
    for _, row in fov_clf.iterrows():
        if pd.isna(row.get('probable_death_timepoint')):
            continue
        tid = row['unique_track_id']
        dtp = row['probable_death_timepoint']
        cell = meas_fov[meas_fov['unique_track_id'] == tid].sort_values('timepoint')
        if cell.empty:
            continue
        idx = (cell['timepoint'] - dtp).abs().idxmin()
        r   = cell.loc[idx]
        death_rows.append([int(dtp), float(r['centroid-0']), float(r['centroid-1'])])

    if death_rows:
        death_coords = np.array(death_rows)
        death_tps    = death_coords[:, 0]
        death_layer  = viewer.add_points(
            death_coords, name='Death markers (★)',
            symbol='star', size=22,
            face_color='yellow', border_color='red', border_width=0.15,
            opacity=0.95, shown=np.zeros(len(death_rows), dtype=bool))

        @viewer.dims.events.current_step.connect
        def _show_markers(event):
            death_layer.shown = death_tps <= viewer.dims.current_step[0]
        _show_markers(None)

    print(f"    Red tracks  = {n_dead} classifier-predicted deaths")
    print(f"    Gray tracks = {n_total - n_dead} surviving cells")
    print(f"    Plasma layer = death probability (0→1)")
    print(f"    Yellow ★ appears at predicted death timepoint")
    print(f"    Tip: toggle layers on/off with the eye icon")

    napari.run()
    return viewer


# ==================== NAPARI DEATH ANNOTATOR ====================
def annotate_cell_death_napari(df_meas, fov,
                                zarr_path=None, zarr_row=None, zarr_well=None,
                                nucleus_channel=0,
                                bfp_channel=None,
                                mng_channel=None,
                                output_csv=None,
                                max_click_distance=40):
    """
    Interactive Napari tool for manually annotating cell death / division events.

    Workflow
    --------
    1. Scrub through time with the slider to find an event.
    2. Press the key for the label you want:
           d  →  death      (mark at the current timepoint)
           v  →  division   (mark at the current timepoint)
           a  →  alive      (cell survives to end; timepoint ignored)
    3. Left-click on the nucleus — the nearest nucleus centroid within
       max_click_distance pixels is auto-selected and annotated.
    4. Repeat.  Re-clicking an already-annotated cell overwrites it.
    5. Press  u         to undo the last annotation.
       Press  Ctrl-S    to save immediately.
       Close the viewer — annotations are auto-saved on close.

    Output CSV columns
    ------------------
    unique_track_id, fov, annotated_label, annotated_timepoint
        annotated_timepoint  is the timepoint of the event (NaN for 'alive').

    Parameters
    ----------
    df_meas            : per-timepoint measurement DataFrame (centroid-0/1, timepoint,
                         fov, unique_track_id required)
    fov                : integer FOV to annotate
    zarr_path/row/well : zarr image store location (optional; shows BFP/mNG if given)
    nucleus_channel    : channel index for the nucleus image (default 0)
    bfp_channel        : BFP channel index (None = same as nucleus_channel)
    mng_channel        : mNG channel index (None = skip)
    output_csv         : path to save annotations (default: ./death_annotations_fov{fov}.csv)
    max_click_distance : maximum distance (pixels) to snap to nearest nucleus (default 40)
    """
    try:
        import napari
    except ImportError:
        raise ImportError("napari is not installed. Run: pip install napari[all]")
    from scipy.spatial import cKDTree
    from napari.utils.colormaps import AVAILABLE_COLORMAPS
    from napari.utils.colormaps import Colormap as NapariColormap

    if output_csv is None:
        output_csv = f"death_annotations_fov{fov}.csv"
    output_csv = Path(output_csv)

    # ── Filter to the requested FOV ──────────────────────────────────────────
    meas_fov = df_meas[df_meas['fov'] == fov].copy()
    if meas_fov.empty:
        raise ValueError(f"No measurements found for FOV {fov}")

    timepoints = sorted(meas_fov['timepoint'].unique())
    t_min, t_max = int(timepoints[0]), int(timepoints[-1])
    n_frames = t_max - t_min + 1

    # ── Build per-timepoint KDTrees for fast nearest-nucleus lookup ──────────
    kd_trees   = {}   # timepoint → cKDTree
    tid_index  = {}   # timepoint → array of unique_track_ids (same order as tree)
    for t, frame in meas_fov.groupby('timepoint'):
        coords = frame[['centroid-0', 'centroid-1']].values
        kd_trees[t]  = cKDTree(coords)
        tid_index[t] = frame['unique_track_id'].values

    # ── Open Napari viewer ───────────────────────────────────────────────────
    viewer = napari.Viewer(title=f"Death Annotator — FOV {fov}")

    # ── Load images (optional) ───────────────────────────────────────────────
    if zarr_path is not None:
        try:
            import zarr
            store = zarr.open(zarr_path, mode='r')
            # Try the well key as given, then stripped of leading zeros
            well_key = zarr_well
            if well_key not in store[zarr_row]:
                stripped = str(int(zarr_well))
                if stripped in store[zarr_row]:
                    well_key = stripped
                    print(f"  Note: using well key '{well_key}' ('{zarr_well}' not found)")
                else:
                    avail = list(store[zarr_row].keys())
                    raise KeyError(f"Well '{zarr_well}' not in zarr store. "
                                   f"Available: {avail}")
            data_arr = store[zarr_row][well_key][str(fov)]['0']
            T, C, Z, Y, X = data_arr.shape

            def _load_mip(ch_idx):
                frames = []
                for t in range(T):
                    frames.append(np.max(data_arr[t, ch_idx, :, :, :], axis=0))
                return np.stack(frames, axis=0)   # (T, Y, X)

            # Nucleus / BFP channel
            nuc_mip = _load_mip(nucleus_channel)
            viewer.add_image(nuc_mip, name='Nucleus (BFP)',
                             colormap='cyan', blending='additive',
                             contrast_limits=[nuc_mip.min(), np.percentile(nuc_mip, 99.5)])

            # mNG channel
            if mng_channel is not None and mng_channel != nucleus_channel:
                try:
                    mng_mip = _load_mip(mng_channel)
                    viewer.add_image(mng_mip, name='mNG',
                                     colormap='green', blending='additive',
                                     contrast_limits=[mng_mip.min(), np.percentile(mng_mip, 99.5)])
                except Exception as e:
                    print(f"  Warning: could not load mNG channel {mng_channel}: {e}")
        except Exception as e:
            print(f"  Warning: could not load zarr images: {e}")

    # ── Build ALL-tracks layer (thin, low opacity — for reference) ───────────
    # Napari requires integer track IDs, but unique_track_id is a string
    # (e.g. "C2_3_17").  Build a stable int↔string mapping.
    all_uids = meas_fov['unique_track_id'].unique()
    uid_to_int = {uid: i for i, uid in enumerate(all_uids)}

    all_track_data = []
    for tid, grp in meas_fov.sort_values('timepoint').groupby('unique_track_id'):
        tid_int = uid_to_int[tid]
        for _, row in grp.iterrows():
            all_track_data.append([tid_int,
                                   int(row['timepoint']),
                                   float(row['centroid-0']),
                                   float(row['centroid-1'])])
    if all_track_data:
        arr = np.array(all_track_data, dtype=float)   # (N, 4): track_id, t, y, x
        viewer.add_tracks(arr, name='All tracks',
                          colormap='gray', opacity=0.35, tail_length=3)

    # ── Annotation state ─────────────────────────────────────────────────────
    # Each entry: unique_track_id → {'label': str, 'timepoint': int|NaN, 'y': float, 'x': float}
    annotations   = {}
    current_mode  = {'label': 'death'}   # mutable so closures can mutate it
    undo_stack    = []                   # list of unique_track_ids for undo

    # Colour / symbol per label
    LABEL_COLOR  = {'death': 'red',    'division': 'cyan',  'alive': 'lime'}
    LABEL_SYMBOL = {'death': 'star',   'division': 'disc',  'alive': 'cross'}
    LABEL_SIZE   = {'death': 22,       'division': 18,      'alive': 15}

    # ── Annotation points layer ───────────────────────────────────────────────
    ann_layer = viewer.add_points(
        np.empty((0, 3)),
        name='Annotations  [d/v/a + click]',
        ndim=3,
        symbol='star',
        size=22,
        face_color='red',
        border_color='white',
        border_width=0.1,
        opacity=0.9,
    )

    def _refresh_annotation_layer():
        """Rebuild the annotation points layer from the annotations dict."""
        if not annotations:
            ann_layer.data = np.empty((0, 3))
            return
        pts, colors, symbols, sizes = [], [], [], []
        for tid, ann in annotations.items():
            tp = ann['timepoint'] if not np.isnan(ann['timepoint']) else t_min
            pts.append([tp, ann['y'], ann['x']])
            lbl = ann['label']
            colors.append(LABEL_COLOR[lbl])
            symbols.append(LABEL_SYMBOL[lbl])
            sizes.append(LABEL_SIZE[lbl])
        ann_layer.data    = np.array(pts)
        ann_layer.face_color = colors
        ann_layer.symbol  = symbols
        ann_layer.size    = sizes

    def _save_csv(silent=False):
        rows = []
        for tid, ann in annotations.items():
            rows.append({
                'unique_track_id':      tid,
                'fov':                  fov,
                'annotated_label':      ann['label'],
                'annotated_timepoint':  ann['timepoint'],
            })
        if rows:
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            if not silent:
                print(f"  Saved {len(rows)} annotations → {output_csv}")
        else:
            if not silent:
                print("  No annotations to save yet.")

    def _status():
        lbl = current_mode['label']
        color_name = {'death': 'RED', 'division': 'CYAN', 'alive': 'GREEN'}[lbl]
        return (f"Mode: {color_name} [{lbl.upper()}]  |  "
                f"{len(annotations)} annotated  |  "
                f"Mode dropdown in right panel; u=undo  Ctrl-S=save")

    # Initialise status bar
    viewer.status = _status()

    # ── Helper actions (shared by dock buttons and key bindings) ─────────────
    def _do_undo():
        if undo_stack:
            tid = undo_stack.pop()
            removed = annotations.pop(tid, None)
            _refresh_annotation_layer()
            viewer.status = (f"  Undo: removed '{removed['label'] if removed else '?'}' "
                             f"for track {tid}  |  " + _status())
        else:
            viewer.status = "  Nothing to undo.  |  " + _status()

    # ── Key bindings (undo + save only — mode switching via dock widget) ─────
    @viewer.bind_key('u')
    def _undo_key(v):
        _do_undo()

    @viewer.bind_key('Control-s')
    def _save_key(v):
        _save_csv()
        v.status = "  Saved.  |  " + _status()

    # ── Mode selector dock widget (no keyboard conflicts) ─────────────────────
    try:
        from magicgui.widgets import ComboBox, PushButton, Container

        mode_combo = ComboBox(
            choices=['death', 'division', 'alive'],
            value='death',
            label='Annotation mode',
        )
        def _on_mode_change(choice):
            current_mode['label'] = choice
            viewer.status = _status()

        mode_combo.changed.connect(_on_mode_change)

        undo_btn = PushButton(text='Undo last  [u]')
        save_btn = PushButton(text='Save now  [Ctrl-S]')

        def _on_undo_click(_):
            _do_undo()

        def _on_save_click(_):
            _save_csv()
            viewer.status = '  Saved.  |  ' + _status()

        undo_btn.changed.connect(_on_undo_click)
        save_btn.changed.connect(_on_save_click)

        panel = Container(widgets=[mode_combo, undo_btn, save_btn])
        viewer.window.add_dock_widget(panel.native, name='Death Annotator',
                                      area='right')
        print("  Mode panel added to the RIGHT dock — select mode from the dropdown.")
        print("  Click on any nucleus in the viewer to annotate it.")
    except Exception as e:
        print(f"  Warning: could not create mode dock widget ({e})")
        print("  Falling back to key bindings — but these may conflict with napari.")

    # ── Mouse click callback ─────────────────────────────────────────────────
    @viewer.mouse_drag_callbacks.append
    def _on_click(v, event):
        # Only handle the initial press, not drag/release
        if event.type != 'mouse_press' or event.button != 1:
            return

        # Napari cursor.position gives (t, y, x) in world/data coordinates
        pos = v.cursor.position
        if len(pos) < 3:
            return
        t_click = int(round(pos[0]))
        y_click, x_click = float(pos[-2]), float(pos[-1])

        # Clamp to valid timepoint range
        t_click = max(t_min, min(t_max, t_click))
        # Snap to nearest available timepoint in df_meas
        t_snap = min(timepoints, key=lambda t: abs(t - t_click))

        if t_snap not in kd_trees:
            v.status = f"  No data at t={t_snap}.  |  " + _status()
            return

        dist, idx = kd_trees[t_snap].query([y_click, x_click])
        if dist > max_click_distance:
            v.status = (f"  No nucleus within {max_click_distance} px "
                        f"(nearest: {dist:.0f} px).  |  " + _status())
            return

        tid   = str(tid_index[t_snap][idx])
        label = current_mode['label']
        ann_tp = float(t_snap) if label != 'alive' else np.nan

        annotations[tid] = {'label': label, 'timepoint': ann_tp,
                            'y': y_click, 'x': x_click}
        undo_stack.append(tid)
        _refresh_annotation_layer()
        _save_csv(silent=True)   # auto-save after every annotation

        tp_str = f"t={int(t_snap)}" if not np.isnan(ann_tp) else "t=N/A"
        v.status = (f"  Annotated track {tid} as '{label}' at {tp_str}.  |  "
                    + _status())

    # ── Print usage summary ───────────────────────────────────────────────────
    print(f"\n  Death Annotator — FOV {fov}")
    print(f"    Tracks:     {meas_fov['unique_track_id'].nunique()} in this FOV")
    print(f"    Output CSV: {output_csv}")
    print(f"    Mode: select from the dropdown in the RIGHT dock panel")
    print(f"           u = undo   |  Ctrl-S = save now")
    print(f"    Left-click on a nucleus to annotate it (snaps within {max_click_distance} px)")
    print(f"    Annotations are auto-saved after every click.\n")

    napari.run()
    _save_csv()   # final save on close
    return annotations


# ==================== PRISM CSV EXPORT ====================
def export_prism_csvs(df_act, df_meas, df_all_tracks, output_dir, well,
                      timepoint_min=None, timepoint_max=None, signal_col='mean_intensity',
                      suffix=""):
    """
    Export GraphPad Prism-ready CSV files for the publication panels:

    bfp_stability  Panel E — Baseline mNG vs Baseline BFP (scatter)
    bfp_stability  Panel F — Max mNG vs Baseline BFP (scatter)
    activation_overview  Panel B — Cumulative % activated (XY curve)
    activation_overview  Panel F — Population mean ± SD trajectories (XY)
    response_groups  Panel B — Plateau value by response group (column)
    response_groups  Panel C — Activation timepoint by response group (column)
    """
    _, _, full_name = parse_well(well)
    t_min, t_max = get_timepoint_range(timepoint_min, timepoint_max, 50)
    suffix_clean = (f"_{suffix.replace(' ', '_').replace('(', '').replace(')', '')}"
                    if suffix else "")

    csv_dir = output_dir / "prism_csv"
    csv_dir.mkdir(exist_ok=True)

    _ycol = signal_col if signal_col in df_meas.columns else 'mean_intensity'
    _ylabel = 'mNG_BFP_ratio' if _ycol == 'mng_bfp_ratio' else 'mNG_intensity'

    # ------------------------------------------------------------------
    # bfp_stability  Panel E — Baseline mNG vs Baseline BFP
    # ------------------------------------------------------------------
    if 'baseline_bfp' in df_act.columns and 'baseline_intensity' in df_act.columns:
        df_e = df_act[['baseline_bfp', 'baseline_intensity']].dropna()
        df_e = df_e[(df_e['baseline_bfp'] > 0) & (df_e['baseline_intensity'] > 0)].copy()
        df_e.columns = ['Baseline_BFP', 'Baseline_mNG']
        df_e.to_csv(csv_dir / f"well_{full_name}_bfp_stability_panelE_baseline_mNG_vs_BFP{suffix_clean}.csv",
                    index=False)
        print(f"  Saved Panel E CSV ({len(df_e)} rows)")
    else:
        print("  Panel E skipped: baseline_bfp or baseline_intensity not in df_act")

    # ------------------------------------------------------------------
    # bfp_stability  Panel F — Max mNG vs Baseline BFP
    # ------------------------------------------------------------------
    max_col = ('max_gfp_top6' if 'max_gfp_top6' in df_act.columns
               else 'max_intensity' if 'max_intensity' in df_act.columns else None)
    if max_col and 'baseline_bfp' in df_act.columns:
        df_f = df_act[['baseline_bfp', max_col]].dropna()
        df_f = df_f[(df_f['baseline_bfp'] > 0) & (df_f[max_col] > 0)].copy()
        df_f.columns = ['Baseline_BFP', 'Max_mNG']
        df_f.to_csv(csv_dir / f"well_{full_name}_bfp_stability_panelF_max_mNG_vs_BFP{suffix_clean}.csv",
                    index=False)
        print(f"  Saved Panel F CSV ({len(df_f)} rows)")
    else:
        print("  Panel F skipped: baseline_bfp or max intensity column not available")

    # ------------------------------------------------------------------
    # activation_overview  Panel B — Cumulative % activated
    # ------------------------------------------------------------------
    _df_ref = df_all_tracks if df_all_tracks is not None and len(df_all_tracks) > 0 else df_act
    df_activating = df_act[df_act['activates'] == True]
    n_total = len(_df_ref)
    if len(df_activating) > 0 and n_total > 0:
        activation_times = df_activating['activation_timepoint'].dropna()
        timepoints = np.arange(t_min, t_max + 1)
        cumulative_pct = [(activation_times <= t).sum() / n_total * 100 for t in timepoints]
        df_b = pd.DataFrame({'Timepoint': timepoints, 'Cumulative_pct_activated': cumulative_pct})
        df_b.to_csv(csv_dir / f"well_{full_name}_activation_overview_panelB_cumulative{suffix_clean}.csv",
                    index=False)
        print(f"  Saved Panel B (cumulative) CSV ({len(df_b)} rows)")
    else:
        print("  Panel B skipped: no activating cells found")

    # ------------------------------------------------------------------
    # activation_overview  Panel F — Population mean ± SD trajectories
    # ------------------------------------------------------------------
    df_non_activating = _df_ref[_df_ref['activates'] == False]
    rows_f = []
    act_ids  = set(df_activating['unique_track_id'])
    non_ids  = set(df_non_activating['unique_track_id'])
    act_data = df_meas[df_meas['unique_track_id'].isin(act_ids)]
    non_data = df_meas[df_meas['unique_track_id'].isin(non_ids)]
    if _ycol in df_meas.columns:
        act_traj = act_data.groupby('timepoint')[_ycol].agg(['mean', 'std']).rename(
            columns={'mean': f'Activating_{_ylabel}_Mean', 'std': f'Activating_{_ylabel}_SD'})
        non_traj = non_data.groupby('timepoint')[_ycol].agg(['mean', 'std']).rename(
            columns={'mean': f'NonActivating_{_ylabel}_Mean', 'std': f'NonActivating_{_ylabel}_SD'})
        df_panel_f = act_traj.join(non_traj, how='outer').reset_index().rename(columns={'timepoint': 'Timepoint'})
        df_panel_f = df_panel_f[(df_panel_f['Timepoint'] >= t_min) & (df_panel_f['Timepoint'] <= t_max)]
        df_panel_f.to_csv(csv_dir / f"well_{full_name}_activation_overview_panelF_mean_trajectories{suffix_clean}.csv",
                          index=False)
        print(f"  Saved Panel F (trajectories) CSV ({len(df_panel_f)} rows)")
    else:
        print(f"  Panel F skipped: signal column '{_ycol}' not in df_meas")

    # ------------------------------------------------------------------
    # response_groups  Panel B — Plateau value by response group
    # ------------------------------------------------------------------
    if 'response_group' in df_act.columns and 'plateau_value' in df_act.columns:
        r_groups = ['low', 'medium', 'high']
        series_b = {g.capitalize(): df_act[df_act['response_group'] == g]['plateau_value'].dropna().values
                    for g in r_groups}
        max_len = max(len(v) for v in series_b.values()) if series_b else 0
        if max_len > 0:
            df_resp_b = pd.DataFrame({k: pd.Series(v) for k, v in series_b.items()})
            df_resp_b.to_csv(csv_dir / f"well_{full_name}_response_groups_panelB_plateau_by_group{suffix_clean}.csv",
                             index=False)
            counts_b = {k: len(v) for k, v in series_b.items()}
            print(f"  Saved Panel B (plateau) CSV — {counts_b}")
    else:
        print("  Response group Panels B/C skipped: response_group or plateau_value not in df_act")

    # ------------------------------------------------------------------
    # response_groups  Panel C — Activation timepoint by response group
    # ------------------------------------------------------------------
    if 'response_group' in df_act.columns and 'activation_timepoint' in df_act.columns:
        r_groups = ['low', 'medium', 'high']
        series_c = {g.capitalize(): df_act[df_act['response_group'] == g]['activation_timepoint'].dropna().values
                    for g in r_groups}
        max_len = max(len(v) for v in series_c.values()) if series_c else 0
        if max_len > 0:
            df_resp_c = pd.DataFrame({k: pd.Series(v) for k, v in series_c.items()})
            df_resp_c.to_csv(csv_dir / f"well_{full_name}_response_groups_panelC_activation_time_by_group{suffix_clean}.csv",
                             index=False)
            counts_c = {k: len(v) for k, v in series_c.items()}
            print(f"  Saved Panel C (activation time) CSV — {counts_c}")

    print(f"\nPrism CSVs written to: {csv_dir}")


# ==================== MAIN ANALYSIS FUNCTION ====================
def run_complete_analysis(args, analysis_dir, tracking_dir, output_dir, well, exclude_fovs,
                          baseline_min=None, baseline_max=None, suffix="", skip_bfp=False, save_individual=False,
                          timepoint_min=None, timepoint_max=None,
                          condition_name=None, wells=None):
    """
    Run complete analysis with optional baseline filtering.

    Parameters:
    -----------
    baseline_min, baseline_max : float or None
        If None, no filtering is applied. Otherwise, filter to this range.
    suffix : str
        Suffix to add to output filenames (e.g., "unfiltered" or "iqr_filtered")
    skip_bfp : bool
        If True, skip BFP extraction (useful for second pass where BFP is already loaded)
    condition_name : str or None
        When set, used as the output identifier instead of the well name.
    wells : list or None
        When set, pool data from all listed wells (overrides single-well loading).
    """
    if condition_name is not None:
        full_name = condition_name
        display_well = condition_name   # used as 'well' argument to plot functions
    else:
        _, _, full_name = parse_well(well)
        display_well = well

    print(f"\n{'='*60}")
    print(f"RUNNING ANALYSIS: {suffix if suffix else 'default'}")
    print(f"{'='*60}")

    print("\nLoading data...")
    if wells is not None:
        df_act = load_condition_data(analysis_dir, wells, exclude_fovs)
        df_meas = load_condition_measurements(tracking_dir, wells, exclude_fovs)
    else:
        df_act = load_activation_data(analysis_dir, well, exclude_fovs)
        df_meas = load_measurements(tracking_dir, well, exclude_fovs)
    
    if df_meas is None:
        raise ValueError("Could not load measurements!")
    
    # Apply timepoint filtering if specified
    if timepoint_min is not None or timepoint_max is not None:
        print(f"\nFiltering timepoint range: {timepoint_min} - {timepoint_max}")
        df_meas, df_act = filter_by_timepoint_range(df_meas, df_act, timepoint_min, timepoint_max)
    
    # Save all tracks (activating + non-activating) before classify_activators filters to activating only
    df_all_tracks = df_act.copy()

    # Restrict df_meas to script-2 quality-filtered tracks (activated + non-activated).
    # The raw tracking output contains ALL detected nuclei; the activation analysis only covers
    # quality-filtered cells.  Using the full tracking df_meas as denominator in plots would
    # heavily underestimate the infection rate.  Filter now so BOTH passes use the correct denominator.
    _script2_ids = set(df_all_tracks['unique_track_id'])
    n_meas_before = df_meas['unique_track_id'].nunique()
    df_meas = df_meas[df_meas['unique_track_id'].isin(_script2_ids)].copy()
    n_meas_after = df_meas['unique_track_id'].nunique()
    print(f"  Restricted df_meas from {n_meas_before} → {n_meas_after} quality-analyzed tracks "
          f"(removed {n_meas_before - n_meas_after} unanalyzed tracking tracks)")

    df_act = classify_activators(
        df_act, df_meas, args.early_min, args.early_max,
        args.average_min, args.average_max, args.late_min,
        args.min_pre_activation_frames,
        use_percentile=True, early_pct=args.early_pct,
        average_pct_low=args.average_pct_low,
        average_pct_high=args.average_pct_high,
        late_pct=args.late_pct,
        method=args.classification_method,
        gmm_max_components=args.gmm_max_components,
        gmm_force_components=args.gmm_force_components,
        gmm_covariance_type=args.gmm_covariance_type,
        sd_multiplier=args.sd_multiplier
    )

    # Generate GMM diagnostics if using GMM
    if args.classification_method == 'gmm' and df_act.attrs.get('classification_info', {}).get('method') == 'gmm':
        plot_gmm_diagnostics(
            df_act, df_meas, output_dir, display_well,
            save_pdf=args.save_pdf, save_svg=args.save_svg,
            save_individual=save_individual, suffix=suffix,
            timepoint_min=timepoint_min, timepoint_max=timepoint_max,
            early_pct=args.early_pct, average_pct_low=args.average_pct_low,
            average_pct_high=args.average_pct_high, late_pct=args.late_pct
        )

    baseline_frames = (args.baseline_start, args.baseline_end)
    df_act = calculate_baseline_intensity(df_act, df_meas, baseline_frames)

    # Apply baseline filtering if thresholds provided
    if baseline_min is not None and baseline_max is not None:
        # Save non-activating cell measurements before IQR filter removes them from df_meas
        _non_act_ids = set(df_all_tracks[df_all_tracks['activates'] == False]['unique_track_id'])
        _df_meas_non_act = df_meas[df_meas['unique_track_id'].isin(_non_act_ids)].copy()

        df_act, df_meas = filter_by_baseline_intensity(df_act, df_meas, baseline_min, baseline_max)
        # Re-classify after filtering using percentiles on the FILTERED population
        df_act = classify_activators(
            df_act, df_meas, args.early_min, args.early_max,
            args.average_min, args.average_max, args.late_min,
            args.min_pre_activation_frames,
            use_percentile=True, early_pct=args.early_pct,
            average_pct_low=args.average_pct_low,
            average_pct_high=args.average_pct_high,
            late_pct=args.late_pct,
            method=args.classification_method,
            gmm_max_components=args.gmm_max_components,
            gmm_force_components=args.gmm_force_components,
            gmm_covariance_type=args.gmm_covariance_type,
            sd_multiplier=args.sd_multiplier
        )

        if args.classification_method == 'gmm' and df_act.attrs.get('classification_info', {}).get('method') == 'gmm':
            plot_gmm_diagnostics(
                df_act, df_meas, output_dir, well,
                save_pdf=args.save_pdf, save_svg=args.save_svg,
                save_individual=save_individual, suffix=suffix + "_post_filter",
                timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                early_pct=args.early_pct, average_pct_low=args.average_pct_low,
                average_pct_high=args.average_pct_high, late_pct=args.late_pct
            )

        # Restore non-activating measurements and trim df_all_tracks to reflect the IQR filter
        if len(_df_meas_non_act) > 0:
            df_meas = pd.concat([df_meas, _df_meas_non_act], ignore_index=True)
        df_all_tracks = df_all_tracks[
            (df_all_tracks['activates'] == False) |
            (df_all_tracks['unique_track_id'].isin(df_act['unique_track_id']))
        ]
    else:
        print("\n  No baseline intensity filtering applied")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Load division events from ultrack lineage data
    print("\n" + "="*60)
    print("LOADING DIVISION EVENTS")
    print("="*60)
    if wells is not None:
        _div_dfs = [load_division_events(tracking_dir, w, exclude_fovs) for w in wells]
        _div_dfs = [d for d in _div_dfs if d is not None and len(d) > 0]
        df_divisions = pd.concat(_div_dfs, ignore_index=True) if _div_dfs else None
    else:
        df_divisions = load_division_events(tracking_dir, well, exclude_fovs)

    # Extract BFP early so mNG/BFP ratio can be used as the primary signal
    print("\n" + "="*60)
    print("EXTRACTING BFP MEASUREMENTS")
    print("="*60)
    if wells is not None:
        _bfp_dfs = [extract_bfp_measurements(tracking_dir, w, exclude_fovs) for w in wells]
        _bfp_dfs = [d for d in _bfp_dfs if d is not None and len(d) > 0]
        df_bfp = pd.concat(_bfp_dfs, ignore_index=True) if _bfp_dfs else None
    else:
        df_bfp = extract_bfp_measurements(tracking_dir, well, exclude_fovs)
    has_ratio = False
    if df_bfp is not None:
        df_meas = merge_bfp_with_gfp(df_meas, df_bfp)
        df_act = calculate_baseline_bfp(df_act, df_meas, baseline_frames)
        df_meas['mng_bfp_ratio'] = np.where(
            df_meas['bfp_mean_intensity'] > 0,
            df_meas['mean_intensity'] / df_meas['bfp_mean_intensity'],
            np.nan
        )
        signal_col = 'mng_bfp_ratio'
        has_ratio = True
    else:
        signal_col = 'mean_intensity'

    # Classify by response amplitude (plateau value from sigmoid fit)
    print("\nClassifying by response amplitude...")
    df_act = classify_by_response(df_act, r2_min=args.response_r2_min,
                                  sd_multiplier=args.response_sd_multiplier,
                                  method=args.response_method)

    # Compute AUC per cell
    print("\nComputing AUC per cell...")
    df_act = compute_auc(df_act, df_meas, signal_col=signal_col, baseline_frames=baseline_frames)

    # Compute survival and hazard (uses all tracked cells including non-activating)
    print("\nComputing survival and hazard functions...")
    df_surv = compute_survival_hazard(df_all_tracks, df_meas, timepoint_max=timepoint_max)

    # Generate plots (ratio used as primary signal when BFP is available)
    print("\nGenerating activation overview...")
    t50 = plot_activation_overview(df_act, df_meas, output_dir, display_well, threshold=args.threshold,
                                    save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
                                    timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                                    signal_col=signal_col, df_all=df_all_tracks)

    print("\nGenerating activation wave kinetics plot...")
    plot_activation_wave(df_all_tracks, output_dir, display_well,
                         save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
                         timepoint_min=timepoint_min, timepoint_max=timepoint_max)

    print("\nGenerating trajectory plots...")
    plot_group_trajectories(df_act, df_meas, output_dir, display_well, args.early_min, args.early_max,
                            args.average_min, args.average_max, args.late_min,
                            args.save_pdf, args.save_svg, suffix=suffix,
                            timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                            signal_col=signal_col)

    print("\nGenerating response group plots...")
    plot_response_groups(df_act, df_meas, output_dir, display_well,
                         save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
                         timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                         signal_col=signal_col)

    print("\nExporting Prism CSV files...")
    export_prism_csvs(df_act, df_meas, df_all_tracks, output_dir, display_well,
                      timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                      signal_col=signal_col, suffix=suffix)

    print("\nComputing activation kinetics per cell...")
    df_act = compute_activation_kinetics(df_act, df_meas, signal_col=signal_col,
                                         baseline_frames=baseline_frames)

    print("\nComputing track duration per cell...")
    df_act = compute_track_duration(df_act, df_meas)

    print("\nComputing division status per cell...")
    df_act = compute_division_status(df_act, df_divisions)

    print("\nComputing cell death events from nuclear morphology...")
    df_act = compute_cell_death(df_act, df_meas)

    print("\nComputing motility metrics per cell...")
    df_act = compute_motility(df_act, df_meas)

    print("\nLoading script 2 uninfected reference cells for motility analysis...")
    if wells is not None:
        _uninf_dfs = [load_script2_uninfected(analysis_dir, w, df_meas) for w in wells]
        _uninf_dfs = [d for d in _uninf_dfs if d is not None and len(d) > 0]
        df_uninfected = pd.concat(_uninf_dfs, ignore_index=True) if _uninf_dfs else None
    else:
        df_uninfected = load_script2_uninfected(analysis_dir, well, df_meas)

    print("\nGenerating response kinetics plots...")
    plot_response_kinetics(df_act, df_meas, output_dir, display_well,
                           save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
                           timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                           signal_col=signal_col)

    print("\nGenerating response group distribution over time...")
    plot_response_distribution_over_time(df_act, df_meas, output_dir, display_well,
                                         save_pdf=args.save_pdf, save_svg=args.save_svg,
                                         suffix=suffix, timepoint_max=timepoint_max,
                                         r2_min=args.response_r2_min)

    print("\nGenerating pre-infection predictor plots...")
    plot_preinf_predictors(df_act, df_meas, output_dir, display_well,
                           save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix)

    print("\nGenerating track duration by response group plots...")
    plot_track_duration_by_response(df_act, output_dir, display_well,
                                    save_pdf=args.save_pdf, save_svg=args.save_svg,
                                    suffix=suffix, timepoint_max=timepoint_max)

    print("\nGenerating division analysis plots...")
    plot_division_analysis(df_act, df_divisions, output_dir, display_well,
                           save_pdf=args.save_pdf, save_svg=args.save_svg,
                           suffix=suffix,
                           timepoint_min=timepoint_min, timepoint_max=timepoint_max)

    print("\nGenerating motility analysis plots...")
    plot_motility_analysis(df_act, df_meas, output_dir, display_well,
                           save_pdf=args.save_pdf, save_svg=args.save_svg,
                           suffix=suffix,
                           timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                           signal_col=signal_col,
                           df_uninfected=df_uninfected)

    print("\nGenerating cell death analysis plots...")
    plot_cell_death(df_act, df_meas, output_dir, display_well,
                    save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix)

    print("\nPlotting AUC analysis...")
    plot_auc(df_act, output_dir, display_well,
             save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
             timepoint_min=timepoint_min, timepoint_max=timepoint_max)

    print("\nPlotting survival & hazard...")
    plot_survival_hazard(df_surv, output_dir, display_well,
                         save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix)

    print("\nComputing cell positions for spatial analysis...")
    df_act = compute_cell_positions(df_act, df_meas, baseline_frames)
    df_all_tracks = compute_cell_positions(df_all_tracks, df_meas, baseline_frames)

    print("\nComputing spatial propagation statistics...")
    spatial_stats = compute_spatial_stats(df_act,
                                          n_neighbors=args.spatial_n_neighbors,
                                          n_permutations=args.spatial_n_permutations)

    print("\nPlotting spatial propagation...")
    plot_spatial_propagation(df_act, df_all_tracks, spatial_stats, output_dir, display_well,
                             save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
                             timepoint_min=timepoint_min, timepoint_max=timepoint_max)

    print("\nClustering cells by kinetic fingerprint...")
    df_act, cluster_info = compute_trajectory_clusters(
        df_act,
        r2_min=args.cluster_r2_min,
        n_clusters_range=(args.cluster_k_min, args.cluster_k_max),
        use_umap=not args.cluster_no_umap,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        max_feature_zscore=args.cluster_max_zscore,
        min_cluster_frac=args.cluster_min_frac,
    )

    print("\nPlotting trajectory clusters...")
    plot_trajectory_clusters(df_act, df_meas, cluster_info, output_dir, display_well,
                              save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
                              signal_col=signal_col,
                              timepoint_min=timepoint_min, timepoint_max=timepoint_max)

    print("\nAnalyzing baseline intensity...")
    df_act, corr_original, p_original, corr_abs, p_abs, iqr_stats = plot_baseline_analysis(
        df_act, df_meas, output_dir, display_well, baseline_frames, args.save_pdf, args.save_svg, suffix=suffix,
        timepoint_min=timepoint_min, timepoint_max=timepoint_max)

    print("\nAnalyzing max mNG distribution...")
    df_act, max_gfp_normality_results = plot_max_gfp_distribution(
        df_act, df_meas, output_dir, display_well, args.save_pdf, args.save_svg,
        save_individual=False, suffix=suffix,
        timepoint_min=timepoint_min, timepoint_max=timepoint_max
    )

    print("\nStatistical comparison of max mNG between groups...")
    max_gfp_stats = statistical_comparison_max_gfp(
        df_act, output_dir, display_well, args.save_pdf, args.save_svg,
        save_individual=False, suffix=suffix,
        timepoint_min=timepoint_min, timepoint_max=timepoint_max
    )

    # BFP-specific analyses and correlations
    corr_bfp_gfp, p_bfp_gfp = np.nan, np.nan
    corr_bfp_mng_change, p_bfp_mng_change = np.nan, np.nan
    corr_corrected, p_corrected = np.nan, np.nan

    if has_ratio:
        verify_bfp_stability(df_meas, df_act, output_dir, display_well, args.save_pdf, args.save_svg,
                            save_individual=save_individual, suffix=suffix,
                            timepoint_min=timepoint_min, timepoint_max=timepoint_max)

        # Correlations
        df_groups = df_act[df_act['activation_group'].isin(['early', 'average', 'late'])].copy()
        valid_bfp = df_groups.dropna(subset=['baseline_bfp', 'baseline_intensity'])
        if len(valid_bfp) > 2:
            corr_bfp_gfp, p_bfp_gfp = stats.pearsonr(np.log10(valid_bfp['baseline_bfp']), np.log10(valid_bfp['baseline_intensity']))

        # BFP decrease vs mNG increase
        start_t, end_t = baseline_frames
        late_t = timepoint_max - 5 if timepoint_max is not None else 40
        changes = []
        for track_id in df_act[df_act['activation_group'].isin(['early', 'average', 'late'])]['unique_track_id'].unique():
            track_data = df_meas[df_meas['unique_track_id'] == track_id]
            baseline = track_data[(track_data['timepoint'] >= start_t) & (track_data['timepoint'] <= end_t)]
            late = track_data[track_data['timepoint'] >= late_t]
            if len(baseline) > 0 and len(late) > 0 and 'bfp_mean_intensity' in baseline.columns:
                bfp_decrease = baseline['bfp_mean_intensity'].mean() - late['bfp_mean_intensity'].mean()
                mng_increase = late['mean_intensity'].mean() - baseline['mean_intensity'].mean()
                changes.append({'bfp_decrease': bfp_decrease, 'mng_increase': mng_increase})
        if len(changes) > 10:
            df_changes = pd.DataFrame(changes)
            corr_bfp_mng_change, p_bfp_mng_change = stats.pearsonr(df_changes['bfp_decrease'], df_changes['mng_increase'])

        # Normalize mNG by baseline BFP
        df_meas = normalize_mng_by_baseline_bfp(df_meas, baseline_frames)

        # Calculate corrected baseline
        baseline_corrected = {}
        for track_id in df_act['unique_track_id'].unique():
            track_data = df_meas[df_meas['unique_track_id'] == track_id]
            baseline_data = track_data[(track_data['timepoint'] >= start_t) & (track_data['timepoint'] <= end_t)]
            if len(baseline_data) > 0 and 'mean_intensity_corrected' in baseline_data.columns:
                baseline_corrected[track_id] = baseline_data['mean_intensity_corrected'].mean()
        df_act['baseline_intensity_corrected'] = df_act['unique_track_id'].map(baseline_corrected)

        # Corrected correlation
        valid_corr = df_groups.dropna(subset=['baseline_intensity_corrected', 'activation_timepoint']) if 'baseline_intensity_corrected' in df_groups.columns else pd.DataFrame()
        if len(valid_corr) == 0:
            df_groups['baseline_intensity_corrected'] = df_groups['unique_track_id'].map(baseline_corrected)
            valid_corr = df_groups.dropna(subset=['baseline_intensity_corrected', 'activation_timepoint'])
        if len(valid_corr) > 2:
            valid_corr['baseline_corrected_log10'] = np.log10(valid_corr['baseline_intensity_corrected'].clip(lower=0.1))
            corr_corrected, p_corrected = stats.pearsonr(valid_corr['baseline_corrected_log10'], valid_corr['activation_timepoint'])
    
    # Save data
    df_lookup = create_napari_lookup_table(df_act, df_meas, output_dir, display_well, args.early_min, suffix=suffix)
    
    suffix_clean = suffix.replace(' ', '_').replace('(', '').replace(')', '') if suffix else ""
    df_act.to_csv(output_dir / f"well_{full_name}_activation_groups_{suffix_clean}.csv" if suffix_clean else output_dir / f"well_{full_name}_activation_groups.csv", index=False)
    df_meas.to_csv(output_dir / f"well_{full_name}_measurements_final_{suffix_clean}.csv" if suffix_clean else output_dir / f"well_{full_name}_measurements_final.csv", index=False)
    
    return (df_act, df_meas, corr_original, p_original, corr_corrected, p_corrected, 
            corr_bfp_gfp, p_bfp_gfp, corr_bfp_mng_change, p_bfp_mng_change, corr_abs, p_abs, iqr_stats)


def run_iqr_comparison_analysis(args, analysis_dir, tracking_dir, output_dir, well, exclude_fovs,
                                condition_name=None, wells=None):
    """
    Run two-pass analysis: first unfiltered, then with IQR-based filtering.
    """
    full_name = condition_name if condition_name is not None else parse_well(well)[2]
    display_well = condition_name if condition_name is not None else well

    print("\n" + "="*70)
    print("PASS 1: UNFILTERED ANALYSIS")
    print("="*70)

    # First pass: No filtering
    results_unfiltered = run_complete_analysis(
        args, analysis_dir, tracking_dir, output_dir, well, exclude_fovs,
        baseline_min=None, baseline_max=None, suffix="unfiltered", save_individual=args.save_individual,
        timepoint_min=args.timepoint_min, timepoint_max=args.timepoint_max,
        condition_name=condition_name, wells=wells
    )
    
    # Extract IQR from unfiltered results
    df_act_uf = results_unfiltered[0]
    iqr_stats = results_unfiltered[-1]
    
    # Calculate IQR thresholds from unfiltered data
    iqr_thresholds = calculate_iqr_thresholds(
        df_act_uf, 
        percentile_low=args.iqr_percentile_low,
        percentile_high=args.iqr_percentile_high
    )
    
    print("\n" + "="*70)
    print("IQR THRESHOLDS CALCULATED FROM UNFILTERED DATA")
    print("="*70)
    print(f"  Percentile range: {iqr_thresholds['percentile_low']}-{iqr_thresholds['percentile_high']}%")
    print(f"  Baseline intensity range: {iqr_thresholds['q_low']:.1f} - {iqr_thresholds['q_high']:.1f}")
    print(f"  Median: {iqr_thresholds['median']:.1f}")
    print(f"  IQR width: {iqr_thresholds['iqr']:.1f}")
    print(f"  Total cells: {iqr_thresholds['n_total']}")
    
    print("\n" + "="*70)
    print("PASS 2: IQR-FILTERED ANALYSIS")
    print("="*70)
    
    # Second pass: With IQR filtering
    results_filtered = run_complete_analysis(
        args, analysis_dir, tracking_dir, output_dir, well, exclude_fovs,
        baseline_min=iqr_thresholds['q_low'],
        baseline_max=iqr_thresholds['q_high'],
        suffix=f"iqr_filtered_Q{int(args.iqr_percentile_low)}-Q{int(args.iqr_percentile_high)}", save_individual=args.save_individual,
        timepoint_min=args.timepoint_min, timepoint_max=args.timepoint_max,
        condition_name=condition_name, wells=wells
    )
    
    # Create comparison figure
    print("\n" + "="*70)
    print("GENERATING COMPARISON FIGURE")
    print("="*70)
    
    plot_comparison_figure(results_unfiltered, results_filtered, output_dir, display_well,
                           iqr_thresholds, args.save_pdf, args.save_svg,
                           save_individual=args.save_individual, suffix="")
    
    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    corr_orig_uf, corr_abs_uf = results_unfiltered[2], results_unfiltered[10]
    corr_orig_f, corr_abs_f = results_filtered[2], results_filtered[10]
    
    n_uf = len(results_unfiltered[0])
    n_f = len(results_filtered[0])
    
    print(f"\n  UNFILTERED (n={n_uf}):")
    print(f"    Correlation ($\log_{10}$): r = {corr_orig_uf:.4f}")
    print(f"    Correlation (abs):   r = {corr_abs_uf:.4f}")
    
    print(f"\n  IQR FILTERED (n={n_f}, {n_f/n_uf*100:.1f}% retained):")
    print(f"    Correlation ($\log_{10}$): r = {corr_orig_f:.4f}")
    print(f"    Correlation (abs):   r = {corr_abs_f:.4f}")
    
    print(f"\n  CHANGE:")
    print(f"    Log₁₀ correlation: {corr_orig_uf:.4f} → {corr_orig_f:.4f} (Δ = {corr_orig_f - corr_orig_uf:+.4f})")
    print(f"    Abs correlation:   {corr_abs_uf:.4f} → {corr_abs_f:.4f} (Δ = {corr_abs_f - corr_abs_uf:+.4f})")
    
    if corr_orig_uf != 0:
        pct_change = (abs(corr_orig_f) - abs(corr_orig_uf)) / abs(corr_orig_uf) * 100
        print(f"    Relative change:   {pct_change:+.1f}%")
    
    return results_unfiltered, results_filtered, iqr_thresholds


# ==================== MAIN ====================
if __name__ == "__main__":
    args = parse_args()
    set_publication_style()
    
    analysis_dir = Path(args.analysis_dir)
    tracking_dir = Path(args.tracking_dir)
    exclude_fovs = set(args.exclude_fovs) if args.exclude_fovs else None

    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "output_v13"

    print("="*70)
    print("OFFON REPORTER ACTIVATION ANALYSIS v12")
    print("With IQR Comparison Mode")
    print("="*70)

    # Detect available wells
    available_wells = get_available_wells(tracking_dir)
    if available_wells:
        print(f"Detected wells: {', '.join(available_wells)}")

    # Resolve well list
    if len(args.well) == 1 and args.well[0].lower() == 'all':
        wells = available_wells if available_wells else AVAILABLE_WELLS
    else:
        wells = args.well

    # For single-well modes use the first well only
    well = wells[0]
    _, _, full_name = parse_well(well)
    print(f"\nOutput: {output_dir}")
    
    if args.view_grid or args.view_response_grid or args.view_napari or args.view_death_napari or args.annotate_death_napari or args.view_classifier_napari:
        # Visualization mode
        print("\nVisualization mode - loading data...")
        df_act = load_activation_data(analysis_dir, well, exclude_fovs)
        df_meas = load_measurements(tracking_dir, well, exclude_fovs)
        df_act = classify_activators(
            df_act, df_meas, args.early_min, args.early_max,
            args.average_min, args.average_max, args.late_min,
            args.min_pre_activation_frames,
            use_percentile=True, early_pct=args.early_pct,
            average_pct_low=args.average_pct_low,
            average_pct_high=args.average_pct_high,
            late_pct=args.late_pct,
            method=args.classification_method,
            gmm_max_components=args.gmm_max_components,
            gmm_force_components=args.gmm_force_components,
            gmm_covariance_type=args.gmm_covariance_type,
            sd_multiplier=args.sd_multiplier
        )

        baseline_frames = (args.baseline_start, args.baseline_end)
        df_act = calculate_baseline_intensity(df_act, df_meas, baseline_frames)
        df_act = classify_by_response(df_act, r2_min=args.response_r2_min,
                                      sd_multiplier=args.response_sd_multiplier,
                                      method=args.response_method)

        output_dir.mkdir(parents=True, exist_ok=True)

        if args.view_grid:
            view_cell_grid(tracking_dir, well, df_act, df_meas,
                          n_per_group=args.n_per_group, crop_size=args.grid_crop_size,
                          exclude_fovs=exclude_fovs, group_col='activation_group')

        elif args.view_response_grid:
            view_cell_grid(tracking_dir, well, df_act, df_meas,
                          n_per_group=args.n_per_group, crop_size=args.grid_crop_size,
                          exclude_fovs=exclude_fovs, group_col='response_group')

        elif args.view_napari:
            view_napari_browser(tracking_dir, well, df_act, df_meas,
                                fov=args.fov, n_examples=args.n_examples,
                                exclude_fovs=exclude_fovs)

        elif args.view_death_napari:
            print("\nComputing cell death events for death viewer...")
            df_act = classify_activators(df_act, df_meas)
            df_act = compute_cell_death(df_act, df_meas)
            view_death_examples_napari(
                df_act, df_meas,
                fov=args.fov,
                zarr_path=args.zarr_path,
                zarr_row=args.zarr_row,
                zarr_well=args.zarr_well,
                nucleus_channel=args.nucleus_channel,
                bfp_channel=args.bfp_channel,
                mng_channel=args.mng_channel,
                n_dying=args.n_examples,
                n_surviving=args.n_examples,
                response_group=args.death_response_group,
            )

        elif args.annotate_death_napari:
            print("\nStarting interactive death annotator...")
            ann_csv = args.annotation_csv
            if ann_csv is None:
                ann_csv = str(Path(output_dir) / f"death_annotations_fov{args.fov}.csv")
            annotate_cell_death_napari(
                df_meas,
                fov=args.fov,
                zarr_path=args.zarr_path,
                zarr_row=args.zarr_row,
                zarr_well=args.zarr_well,
                nucleus_channel=args.nucleus_channel,
                bfp_channel=args.bfp_channel,
                mng_channel=args.mng_channel,
                output_csv=ann_csv,
            )
            print(f"\nAnnotations saved to: {ann_csv}")

        elif args.view_classifier_napari:
            print("\nLoading classifier predictions for Napari viewer...")
            classifier_csv = args.annotation_csv
            if classifier_csv is None:
                classifier_csv = str(Path(output_dir) / f'death_classifier_results_{full_name}.csv')
            view_classifier_predictions_napari(
                classifier_csv=classifier_csv,
                df_meas=df_meas,
                fov=args.fov,
                zarr_path=args.zarr_path,
                zarr_row=args.zarr_row,
                zarr_well=args.zarr_well,
                nucleus_channel=args.nucleus_channel,
                bfp_channel=args.bfp_channel,
                mng_channel=args.mng_channel,
            )

    elif args.train_death_classifier:
        print("\nTraining death classifier …")
        df_meas = load_measurements(tracking_dir, well, exclude_fovs)
        ann_csv = args.annotation_csv
        if ann_csv is None:
            raise ValueError("--annotation-csv is required for --train-death-classifier")
        model_path = args.classifier_model
        if model_path is None:
            model_path = str(Path(output_dir) / 'death_classifier.pkl')
        train_death_classifier(
            annotation_csv=ann_csv,
            df_meas=df_meas,
            model_output_path=model_path,
            random_seed=42,
        )

    elif args.apply_death_classifier:
        print("\nApplying death classifier …")
        df_act  = load_activation_data(analysis_dir, well, exclude_fovs)
        df_meas = load_measurements(tracking_dir, well, exclude_fovs)
        df_act  = classify_activators(
            df_act, df_meas, args.early_min, args.early_max,
            args.average_min, args.average_max, args.late_min,
            args.min_pre_activation_frames,
            use_percentile=True, early_pct=args.early_pct,
            average_pct_low=args.average_pct_low,
            average_pct_high=args.average_pct_high,
        )
        df_act = classify_by_response(df_act)
        model_path = args.classifier_model
        if model_path is None:
            model_path = str(Path(output_dir) / 'death_classifier.pkl')
        df_act = apply_death_classifier(
            df_act, df_meas,
            model_path=model_path,
            death_prob_threshold=args.death_prob_threshold,
            min_track_length=15,
        )
        # Save results alongside existing analysis outputs
        out_path = Path(output_dir) / f'death_classifier_results_{full_name}.csv'
        df_act.to_csv(out_path, index=False)
        print(f"\nResults saved → {out_path}")

        # Plot cumulative death curves using classifier predictions
        plot_cell_death(df_act, df_meas, output_dir, well,
                        save_pdf=args.save_pdf, save_svg=args.save_svg)

    # Parse --conditions argument into a dict: {name: [well1, well2, ...]}
    conditions = {}
    if args.conditions:
        for cond_str in args.conditions:
            if ':' not in cond_str:
                print(f"WARNING: Skipping malformed condition '{cond_str}' (expected NAME:WELL1,WELL2)")
                continue
            cond_name, cond_wells_str = cond_str.split(':', 1)
            conditions[cond_name.strip()] = [w.strip() for w in cond_wells_str.split(',')]
        print(f"\nConditions defined: {conditions}")

    elif args.skip_iqr_comparison:
        # Single-pass mode with manual thresholds
        print("\nSingle-pass mode (IQR comparison disabled)")
        for well in wells:
            _, _, full_name = parse_well(well)
            print(f"\n{'='*70}\nWell: {full_name}\n{'='*70}")
            try:
                if args.baseline_intensity_min is not None and args.baseline_intensity_max is not None:
                    print(f"Manual baseline intensity range: {args.baseline_intensity_min} - {args.baseline_intensity_max}")
                    results = run_complete_analysis(
                        args, analysis_dir, tracking_dir, output_dir, well, exclude_fovs,
                        baseline_min=args.baseline_intensity_min,
                        baseline_max=args.baseline_intensity_max,
                        save_individual=args.save_individual,
                        timepoint_min=args.timepoint_min, timepoint_max=args.timepoint_max
                    )
                else:
                    print("No baseline filtering applied")
                    results = run_complete_analysis(
                        args, analysis_dir, tracking_dir, output_dir, well, exclude_fovs,
                        baseline_min=None, baseline_max=None,
                        save_individual=args.save_individual,
                        timepoint_min=args.timepoint_min, timepoint_max=args.timepoint_max
                    )
            except Exception as e:
                print(f"\nERROR well {well}: {e}")

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

    else:
        # Two-pass IQR comparison mode (default)
        print(f"\nTwo-pass IQR comparison mode")
        print(f"IQR percentile range: {args.iqr_percentile_low}-{args.iqr_percentile_high}%")
        for well in wells:
            _, _, full_name = parse_well(well)
            print(f"\n{'='*70}\nWell: {full_name}\n{'='*70}")
            try:
                results_uf, results_f, iqr_thresholds = run_iqr_comparison_analysis(
                    args, analysis_dir, tracking_dir, output_dir, well, exclude_fovs
                )
            except Exception as e:
                print(f"\nERROR well {well}: {e}")

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nOutputs saved to: {output_dir}")

    # --- Condition-level pooled analysis ---
    if conditions:
        print("\n" + "="*70)
        print("POOLED CONDITION ANALYSIS")
        print("="*70)
        for cond_name, cond_wells in conditions.items():
            print(f"\n{'='*70}\nCondition: {cond_name} (wells: {', '.join(cond_wells)})\n{'='*70}")
            try:
                if args.skip_iqr_comparison:
                    if args.baseline_intensity_min is not None and args.baseline_intensity_max is not None:
                        run_complete_analysis(
                            args, analysis_dir, tracking_dir, output_dir, cond_wells[0], exclude_fovs,
                            baseline_min=args.baseline_intensity_min,
                            baseline_max=args.baseline_intensity_max,
                            save_individual=args.save_individual,
                            timepoint_min=args.timepoint_min, timepoint_max=args.timepoint_max,
                            condition_name=cond_name, wells=cond_wells
                        )
                    else:
                        run_complete_analysis(
                            args, analysis_dir, tracking_dir, output_dir, cond_wells[0], exclude_fovs,
                            baseline_min=None, baseline_max=None,
                            save_individual=args.save_individual,
                            timepoint_min=args.timepoint_min, timepoint_max=args.timepoint_max,
                            condition_name=cond_name, wells=cond_wells
                        )
                else:
                    run_iqr_comparison_analysis(
                        args, analysis_dir, tracking_dir, output_dir, cond_wells[0], exclude_fovs,
                        condition_name=cond_name, wells=cond_wells
                    )
            except Exception as e:
                print(f"\nERROR condition {cond_name}: {e}")

        print("\n" + "="*70)
        print("CONDITION ANALYSIS COMPLETE")
        print("="*70)

    print("\nDone!")