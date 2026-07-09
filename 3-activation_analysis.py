"""
OFFON Reporter Activation Group Analysis
========================================
Sigmoid fitting, activation detection, and response-group classification for
the split-mNG OFFON reporter. Reads the trajectory output from script 2
(--analysis-dir) and the tracking output from script 1 (--tracking-dir), then
writes per-well (and optionally pooled-condition) activation summary CSVs and
publication figures. CPU-only.

USAGE
=====
Single well:
    python 3-activation_analysis.py --well B2

All wells (auto-detected):
    python 3-activation_analysis.py --well all

Pooled condition (produces well_<NAME>_... panels in addition to per-well):
    python 3-activation_analysis.py --well all --conditions DENV:B2,B3

COMMAND-LINE OPTIONS
====================
Input/Output:
    --analysis-dir PATH       Script 2 trajectory output directory
    --tracking-dir PATH       Script 1 ultrack tracking output directory
    --output-dir PATH         Output directory for figures and CSVs
    --well WELL [WELL ...]    Well(s) to analyze, or 'all' (default: all wells)
    --conditions NAME:W1,W2   Pool wells into a named condition (repeatable),
                              e.g. --conditions DENV:B2,B3 ZIKV:C2,C3
    --exclude-fovs FOV [...]  Exclude specific FOVs from analysis

Timepoint Filtering:
    --timepoint-min N         Minimum timepoint to include (default: None = from 0)
    --timepoint-max N         Maximum timepoint to include (default: None = all)
    --imaging-start-hpi H     Hours post-infection at imaging start; offsets the
                              time axis of the cumulative (panel B) and
                              mean-trajectory (panel F) CSVs (default: 1.5)

Activation-time Group Classification:
    --classification-method M  percentile | fixed | sd  (default: sd)
    --sd-multiplier VALUE      SD multiplier for the 'sd' method (default: 1.0)
    # percentile method:
    --early-pct / --average-pct-low / --average-pct-high / --late-pct
    # fixed method:
    --early-min / --early-max / --average-min / --average-max / --late-min

Response-amplitude Grouping (low/medium/high):
    --response-method M        tertile | sd  (default: tertile)
    --response-sd-multiplier V SD multiplier for the 'sd' method (default: 1.0)
    --response-r2-min VALUE    Minimum sigmoid R² to include a cell (default: 0.7)

Baseline / Activation:
    --baseline-start N         Start frame for baseline (default: 0)
    --baseline-end N           End frame for baseline (default: 5)
    --threshold VALUE          mNG intensity activation threshold (default: 50)
    --min-pre-activation-frames N   Minimum frames before activation

Figure Output:
    --save-pdf                 Also save figures as PDF
    --save-svg                 Also save figures as SVG

Run `python 3-activation_analysis.py --help` for the full, authoritative list.

OUTPUT
======
Figures are written to output_dir/figures/ (PNG, plus PDF/SVG with --save-pdf /
--save-svg). Per-well and per-condition activation summary CSVs and the Prism
time-series exports are written to output_dir/.
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
    'imaging_start_hpi': 1.5,  # hours post-infection when imaging started (time-axis offset)
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
                        choices=['percentile', 'fixed', 'sd'],
                        help='Activation group classification method (default: sd). '
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
    parser.add_argument('--min-pre-activation-frames', type=int, default=DEFAULTS['min_pre_activation_frames'])
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--save-svg', action='store_true')
    parser.add_argument('--baseline-start', type=int, default=0)
    parser.add_argument('--baseline-end', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=50)
    parser.add_argument('--timepoint-min', type=int, default=DEFAULTS['timepoint_min'],
                        help='Minimum timepoint to include in analysis (default: None = start from 0)')
    parser.add_argument('--timepoint-max', type=int, default=DEFAULTS['timepoint_max'],
                        help='Maximum timepoint to include in analysis (default: None = use all)')
    parser.add_argument('--imaging-start-hpi', type=float, default=DEFAULTS['imaging_start_hpi'],
                        help='Hours post-infection at which imaging started. Added as an offset '
                             'to the time axis of the cumulative (panel B) and mean-trajectory '
                             '(panel F) CSVs so timepoints read as hours post-infection (default: 1.5)')
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
                        sd_multiplier=1.0):
    """
    Classify activators into early/average/late groups.

    method='percentile' (default): percentile-based thresholds
    method='fixed': legacy fixed timepoint thresholds
    method='sd': mean ± sd_multiplier * SD (most unbiased for unimodal/sigmoid distributions)
    """

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


def merge_bfp_with_gfp(df_meas, df_bfp):
    df_merged = df_meas.merge(
        df_bfp[['unique_track_id', 'timepoint', 'bfp_mean_intensity', 'bfp_max_intensity', 'bfp_min_intensity']],
        on=['unique_track_id', 'timepoint'], how='left'
    )
    print(f"Merged measurements: {len(df_merged)} rows")
    return df_merged


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


# ==================== PRISM CSV EXPORT ====================
def export_prism_csvs(df_act, df_meas, df_all_tracks, output_dir, well,
                      timepoint_min=None, timepoint_max=None, signal_col='mean_intensity',
                      suffix="", imaging_start_hpi=0.0):
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
        df_b = pd.DataFrame({'Timepoint': timepoints + imaging_start_hpi,
                             'Cumulative_pct_activated': cumulative_pct})
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
        df_panel_f['Timepoint'] = df_panel_f['Timepoint'] + imaging_start_hpi
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
                          suffix="unfiltered", timepoint_min=None, timepoint_max=None,
                          condition_name=None, wells=None):
    """
    Run the published activation analysis for a single well or a pooled condition.

    Parameters
    ----------
    suffix : str
        Suffix added to output filenames.  The published panels use "unfiltered".
    condition_name : str or None
        When set, used as the output identifier instead of the well name.
    wells : list or None
        When set, pool data from all listed wells (overrides single-well loading).
    """
    if condition_name is not None:
        full_name = condition_name
        display_well = condition_name
    else:
        _, _, full_name = parse_well(well)
        display_well = well

    print(f"\n{'='*60}")
    print(f"RUNNING ANALYSIS: {full_name}")
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

    # Restrict df_meas to the script-2 quality-filtered tracks so plot/export denominators are correct.
    # The raw tracking output contains ALL detected nuclei; the activation analysis only covers
    # quality-filtered cells.
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
        sd_multiplier=args.sd_multiplier
    )

    baseline_frames = (args.baseline_start, args.baseline_end)
    df_act = calculate_baseline_intensity(df_act, df_meas, baseline_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Extract BFP so the mNG/BFP ratio can be used as the primary signal
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

    # Generate published figures (ratio used as primary signal when BFP is available)
    print("\nGenerating activation overview...")
    plot_activation_overview(df_act, df_meas, output_dir, display_well, threshold=args.threshold,
                             save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
                             timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                             signal_col=signal_col, df_all=df_all_tracks)

    print("\nGenerating response group plots...")
    plot_response_groups(df_act, df_meas, output_dir, display_well,
                         save_pdf=args.save_pdf, save_svg=args.save_svg, suffix=suffix,
                         timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                         signal_col=signal_col)

    print("\nExporting Prism CSV files...")
    export_prism_csvs(df_act, df_meas, df_all_tracks, output_dir, display_well,
                      timepoint_min=timepoint_min, timepoint_max=timepoint_max,
                      signal_col=signal_col, suffix=suffix,
                      imaging_start_hpi=args.imaging_start_hpi)

    if has_ratio:
        print("\nVerifying BFP stability...")
        verify_bfp_stability(df_meas, df_act, output_dir, display_well,
                             args.save_pdf, args.save_svg, suffix=suffix,
                             timepoint_min=timepoint_min, timepoint_max=timepoint_max)

    # Save per-cell tables
    suffix_clean = suffix.replace(' ', '_').replace('(', '').replace(')', '') if suffix else ""
    if suffix_clean:
        df_act.to_csv(output_dir / f"well_{full_name}_activation_groups_{suffix_clean}.csv", index=False)
        df_meas.to_csv(output_dir / f"well_{full_name}_measurements_final_{suffix_clean}.csv", index=False)
    else:
        df_act.to_csv(output_dir / f"well_{full_name}_activation_groups.csv", index=False)
        df_meas.to_csv(output_dir / f"well_{full_name}_measurements_final.csv", index=False)

    return df_act, df_meas


# ==================== MAIN ====================
if __name__ == "__main__":
    args = parse_args()
    set_publication_style()

    analysis_dir = Path(args.analysis_dir)
    tracking_dir = Path(args.tracking_dir)
    exclude_fovs = set(args.exclude_fovs) if args.exclude_fovs else None

    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "output_v13"

    print("="*70)
    print("OFFON REPORTER ACTIVATION ANALYSIS")
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

    print(f"\nOutput: {output_dir}")

    # Parse --conditions into {name: [well1, well2, ...]}
    conditions = {}
    if args.conditions:
        for cond_str in args.conditions:
            if ':' not in cond_str:
                print(f"WARNING: Skipping malformed condition '{cond_str}' (expected NAME:WELL1,WELL2)")
                continue
            cond_name, cond_wells_str = cond_str.split(':', 1)
            conditions[cond_name.strip()] = [w.strip() for w in cond_wells_str.split(',')]
        print(f"\nConditions defined: {conditions}")

    # Per-well analysis
    for well in wells:
        _, _, full_name = parse_well(well)
        print(f"\n{'='*70}\nWell: {full_name}\n{'='*70}")
        try:
            run_complete_analysis(
                args, analysis_dir, tracking_dir, output_dir, well, exclude_fovs,
                suffix="unfiltered",
                timepoint_min=args.timepoint_min, timepoint_max=args.timepoint_max
            )
        except Exception as e:
            print(f"\nERROR well {well}: {e}")

    print("\n" + "="*70)
    print("PER-WELL ANALYSIS COMPLETE")
    print("="*70)

    # Pooled condition analysis
    if conditions:
        print("\n" + "="*70)
        print("POOLED CONDITION ANALYSIS")
        print("="*70)
        for cond_name, cond_wells in conditions.items():
            print(f"\n{'='*70}\nCondition: {cond_name} (wells: {', '.join(cond_wells)})\n{'='*70}")
            try:
                run_complete_analysis(
                    args, analysis_dir, tracking_dir, output_dir, cond_wells[0], exclude_fovs,
                    suffix="unfiltered",
                    timepoint_min=args.timepoint_min, timepoint_max=args.timepoint_max,
                    condition_name=cond_name, wells=cond_wells
                )
            except Exception as e:
                print(f"\nERROR condition {cond_name}: {e}")

        print("\n" + "="*70)
        print("CONDITION ANALYSIS COMPLETE")
        print("="*70)

    print(f"\nOutputs saved to: {output_dir}")
    print("\nDone!")
