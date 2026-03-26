"""
OFF→ON mNG Reporter Activation Analysis
Adapted for ultrack tracking output
Publication-quality figures

USAGE
=====
Basic analysis for a single well:
    python script.py --well C2

Analyze all available wells:
    python script.py --well all

COMMAND-LINE OPTIONS
====================
Input/Output:
    -i, --input PATH          Input directory containing ultrack output
    -o, --output PATH         Output directory for results
    -w, --well WELL           Well to analyze: B3, B4, C3, C4, or 'all' (default: C2)

Activation Detection:
    -t, --threshold VALUE     mNG intensity threshold for activation (default: 35)
    --min-duration N          Minimum track duration in frames (default: 30)
    --sustained / --no-sustained
                              Require sustained activation above threshold (default: True)
    --sustained-window N      Number of consecutive frames above threshold (default: 6)
    --min-pre-activation-frames N
                              Minimum frames before activation to include track (default: 0)
    --bin-size N              Bin size for grouping activation times (default: 5)

Quality Filtering:
    --filter-quality / --no-filter-quality
                              Apply track quality filters (default: True)
    --max-position-jump VALUE Maximum allowed position jump between frames (default: 50)
    --max-intensity-jump VALUE
                              Maximum relative intensity jump (default: 2.0)
    --max-gap-fraction VALUE  Maximum fraction of missing timepoints (default: 0.1)
    --max-area-cv VALUE       Maximum coefficient of variation for nuclear area (default: 0.5)

FOV Exclusion:
    --exclude-fovs FOV [FOV ...]
                              Exclude specific FOVs. Format: FOV numbers or WELL:FOV1,FOV2
                              Examples: --exclude-fovs 3 5
                                        --exclude-fovs C2:3,5 C3:1

Figure Output:
    --save-pdf                Save figures in PDF format (in addition to PNG)
    --save-svg                Save figures in SVG format (in addition to PNG)
    --save-individual         Save individual figures (not just panels)
    --no-style                Disable publication style formatting

Exploration:
    --explore-thresholds      Run threshold exploration analysis

EXAMPLES
========
# Basic analysis with default threshold
python script.py --well C3

# Custom threshold with SVG output for publication (panels only)
python script.py --well C2 --threshold 40 --save-svg

# Save individual figures for publication (PNG + SVG)
python script.py --well C2 --save-individual --save-svg

# All formats: panels + individual figures in PNG, PDF, and SVG
python script.py --well C2 --save-individual --save-svg --save-pdf

# Analyze all wells, excluding problematic FOVs
python script.py --well all --exclude-fovs C2:3,5 C3:1

# Explore different thresholds to find optimal value
python script.py --well C2 --explore-thresholds

# Strict quality filtering
python script.py --well C2 --max-position-jump 30 --max-intensity-jump 1.5

# Relaxed activation criteria (no sustained requirement)
python script.py --well C2 --no-sustained --threshold 40

OUTPUT FILES
============
Figures - Panels (in output_dir/figures/panels/):
    - well_XX_activation_analysis_panel.png   6-panel analysis overview
    - well_XX_trajectories_by_bin_panel.png   Trajectories grouped by activation time
    - well_XX_summary_panel.png               3-panel publication summary

Figures - Individual (in output_dir/figures/individual/, requires --save-individual):
    - well_XX_activation_histogram.png        Activation time distribution
    - well_XX_cumulative_activation.png       Cumulative activation curve
    - well_XX_example_trajectories.png        Example mNG trajectories
    - well_XX_max_intensity_distribution.png  Max intensity by activation status
    - well_XX_activation_by_fov.png           Activation percentage per FOV
    - well_XX_population_dynamics.png         Population mean ± SD over time

Data files (in output_dir/):
    - well_XX_all_tracks.csv                  All analyzed tracks with activation status
    - well_XX_activating.csv                  Activating tracks with time bins
    - well_XX_non_activating.csv              Non-activating tracks
    - well_XX_track_quality.csv               Quality metrics for all tracks
    - well_XX_cumulative_activation.csv       Cumulative activation percentage per timepoint
    - well_XX_activation_windows.csv          Early/middle/late activation windows
    - well_XX_traj_tXX_XX.csv                 Trajectory data by activation time bin
    - summary.csv                             Summary statistics across all wells
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats
import argparse
import time
import warnings
warnings.filterwarnings('ignore')
try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
try:
    from skimage.measure import regionprops_table
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ==================== PUBLICATION STYLE CONFIGURATION ====================
def setup_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,
        'savefig.facecolor': 'white',
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })

COLORS = {
    'activating': '#4DAF4A',
    'non_activating': '#7F7F7F',
    'threshold': '#E31A1C',
    'median': '#2166AC',
    'highlight': '#D62728',
    'fill_alpha': 0.15,
    'line_alpha': 0.7,
}

FIGURE_SIZES = {
    'panel_2x3': (14, 9),
    'panel_1x3': (14, 4),
    'panel_2x2': (10, 8),
    'single': (5, 4),
    'single_wide': (6, 4),
}

DEFAULTS = {
    'input_dir': "/hpc/projects/arias_group/Vincent_Turon-Lagot/Imaging_Experiments/20260321_A549_OFFON18_OFFON20_48hrs_dragonfly/Image_analysis/1-nuclear_analysis/nuclear_analysis_output_ultrack",
    'output_dir': "/hpc/projects/arias_group/Vincent_Turon-Lagot/Imaging_Experiments/20260321_A549_OFFON18_OFFON20_48hrs_dragonfly/Image_analysis/2-mNG_trajectories_analysis/output",
    'well': 'B3',
    'threshold': 40,
    'min_duration': 30,
    'sustained': True,
    'sustained_window': 6,
    'bin_size': 5,
    'min_pre_activation_frames': 0,
    'start_timepoint': None,  # NEW
    'end_timepoint': 96,    # NEW
    'min_activation_timepoint': 0,
}

AVAILABLE_WELLS = ['B3', 'B4', 'C3', 'C4']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze OFF→ON mNG reporter activation from ultrack tracking data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py --well B2 --threshold 800
  python script.py --well C3
  python script.py --well all --explore-thresholds
  
Available wells: B2, B3, C2, C3 (or 'all' for all wells)
        """
    )
    
    parser.add_argument('-i', '--input', type=str, default=DEFAULTS['input_dir'])
    parser.add_argument('-o', '--output', type=str, default=DEFAULTS['output_dir'])
    parser.add_argument('-w', '--well', type=str, nargs='+', default=[DEFAULTS['well']],
                        help=f"Well(s) to analyze: {', '.join(AVAILABLE_WELLS)}, or 'all'")
    parser.add_argument('-t', '--threshold', type=float, default=DEFAULTS['threshold'])
    parser.add_argument('--min-duration', type=int, default=DEFAULTS['min_duration'])
    parser.add_argument('--sustained', dest='sustained', action='store_true', default=True)
    parser.add_argument('--no-sustained', dest='sustained', action='store_false')
    parser.add_argument('--sustained-window', type=int, default=DEFAULTS['sustained_window'])
    parser.add_argument('--bin-size', type=int, default=DEFAULTS['bin_size'])
    parser.add_argument('--min-pre-activation-frames', type=int, default=DEFAULTS['min_pre_activation_frames'])
    parser.add_argument('--start-timepoint', type=int, default=DEFAULTS['start_timepoint'],
                        help='First timepoint to include in analysis (default: None = use all)')
    parser.add_argument('--end-timepoint', type=int, default=DEFAULTS['end_timepoint'],
                        help='Last timepoint to include in analysis (default: None = use all)')
    parser.add_argument('--min-activation-timepoint', type=int, default=DEFAULTS['min_activation_timepoint'],
                        help='Minimum timepoint for valid activation (cells activating earlier are excluded, default: 0)')
    parser.add_argument('--n-sd', type=float, default=3.0,
                        help='Number of SDs above baseline mean used as per-cell activation threshold (default: 3)')
    parser.add_argument('--filter-quality', action='store_true', default=True)
    parser.add_argument('--no-filter-quality', dest='filter_quality', action='store_false')
    parser.add_argument('--max-position-jump', type=float, default=50)
    parser.add_argument('--max-intensity-jump', type=float, default=2.0)
    parser.add_argument('--max-gap-fraction', type=float, default=0.1)
    parser.add_argument('--max-area-cv', type=float, default=0.5)
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--save-svg', action='store_true')
    parser.add_argument('--save-individual', action='store_true', help='Save individual figures (not just panels)')
    parser.add_argument('--no-style', action='store_true')
    parser.add_argument('--explore-thresholds', action='store_true')
    parser.add_argument('--explore-n-sd', action='store_true',
                        help='Sweep n_sd values across all wells to calibrate activation threshold')
    parser.add_argument('--n-sd-values', nargs='+', type=float, default=None,
                        help='n_sd values to test (default: 1 2 3 4 5 6 7 8 10)')
    parser.add_argument('--drift-correction', choices=['none', 'population', 'control'],
                        default='none',
                        help='Correct systematic fluorescence drift over time: '
                             '"population" subtracts per-timepoint median within each well '
                             '(assumes low MOI); '
                             '"control" subtracts the drift curve from the uninfected control well '
                             '(see --control-well). Default: none')
    parser.add_argument('--control-well', type=str, default='B2',
                        help='Uninfected control well used for drift correction when '
                             '--drift-correction=control (default: B2)')
    parser.add_argument('--control-threshold', action='store_true', default=False,
                        help='Derive activation threshold from the full signal distribution '
                             'of the uninfected control well (mean + n_sd × SD across all '
                             'cells and timepoints) instead of per-cell baseline thresholds')
    parser.add_argument('--exclude-fovs', nargs='+', default=None)
    parser.add_argument('--view-napari', action='store_true')
    parser.add_argument('--fov', type=int, default=0)

    return parser.parse_args()


def parse_well(well_str):
    """Parse well string and return (row, col, full_name)."""
    well_str = well_str.strip().upper()
    if len(well_str) >= 2 and well_str[0].isalpha() and well_str[1:].isdigit():
        return (well_str[0], int(well_str[1:]), well_str)
    if well_str.isdigit():
        return ('C', int(well_str), f"C{well_str}")
    raise ValueError(f"Invalid well format: {well_str}")


def parse_exclude_fovs(exclude_arg, current_well):
    if exclude_arg is None:
        return set()
    excluded = set()
    for item in exclude_arg:
        if ':' in item:
            well_part, fov_part = item.split(':')
            if well_part.upper() == current_well.upper():
                fovs = [int(f.strip()) for f in fov_part.split(',')]
                excluded.update(fovs)
        else:
            excluded.add(int(item))
    return excluded


def setup_paths(args):
    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir = Path(args.output) if args.output else input_dir.parent / "offon_analysis_ultrack"
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "figures" / "panels").mkdir(exist_ok=True, parents=True)
    (output_dir / "figures" / "individual").mkdir(exist_ok=True, parents=True)
    return input_dir, output_dir


def safe_save_csv(df, filepath, max_retries=3):
    filepath = Path(filepath)
    for attempt in range(max_retries):
        try:
            temp_path = filepath.with_suffix('.csv.tmp')
            df.to_csv(temp_path, index=False)
            temp_path.rename(filepath)
            return True
        except OSError as e:
            print(f"  Warning: Save attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    return False


def save_figure(fig, output_dir, name, save_pdf=False, save_svg=False, subdir="panels"):
    fig_dir = output_dir / "figures" / subdir
    fig_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    if save_pdf:
        fig.savefig(fig_dir / f"{name}.pdf", bbox_inches='tight', facecolor='white')
    if save_svg:
        fig.savefig(fig_dir / f"{name}.svg", bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_well_data(output_dir, well, exclude_fovs=None):
    """Load and merge all FOV data for a single well."""
    row, col, full_name = parse_well(well)
    
    print(f"\n{'='*60}")
    print(f"Loading data for Well {full_name}")
    print(f"{'='*60}")
    
    if exclude_fovs:
        print(f"Excluding FOVs: {sorted(exclude_fovs)}")
    
    all_measurements = []
    all_track_info = []
    
    pattern = f"well_{full_name}_FOV*"
    fov_dirs = list(output_dir.glob(pattern))
    
    print(f"Found {len(fov_dirs)} FOVs (pattern: {pattern})")
    
    if not fov_dirs:
        all_dirs = [d.name for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('well_')]
        print(f"Available directories: {all_dirs[:10]}")
        raise ValueError(f"No FOV directories found for well {full_name}")
    
    for fov_dir in sorted(fov_dirs):
        fov = int(fov_dir.name.split("FOV")[-1])
        
        if exclude_fovs and fov in exclude_fovs:
            print(f"  FOV {fov}: EXCLUDED")
            continue
        
        meas_file = fov_dir / "nuclear_measurements.csv"
        if meas_file.exists():
            df = pd.read_csv(meas_file)
            df['fov'] = fov
            df['well'] = full_name
            df['unique_track_id'] = f"{full_name}_{fov}_" + df['track_id'].astype(str)
            all_measurements.append(df)
            print(f"  FOV {fov}: {df['track_id'].nunique()} tracks, {len(df)} measurements")
        
        track_file = fov_dir / "ultrack_tracks.csv"
        if track_file.exists():
            ti = pd.read_csv(track_file)
            if 't' in ti.columns:
                ti = ti.rename(columns={'t': 'timepoint'})
            track_summary = ti.groupby('track_id').agg({
                'timepoint': ['min', 'max', 'count']
            }).reset_index()
            track_summary.columns = ['track_id', 'start_t', 'end_t', 'n_timepoints']
            track_summary['fov'] = fov
            track_summary['well'] = full_name
            track_summary['unique_track_id'] = f"{full_name}_{fov}_" + track_summary['track_id'].astype(str)
            all_track_info.append(track_summary)
    
    if not all_measurements:
        raise ValueError(f"No measurement files found for well {full_name}!")
    
    df_all = pd.concat(all_measurements, ignore_index=True)
    df_tracks = pd.concat(all_track_info, ignore_index=True) if all_track_info else pd.DataFrame()
    
    print(f"\nTotal: {df_all['unique_track_id'].nunique()} unique tracks, {len(df_all)} measurements")
    return df_all, df_tracks

def filter_timepoint_range(df_all, df_tracks, start_timepoint=None, end_timepoint=None, verbose=True):
    """
    Filter data to include only measurements within the specified timepoint range.
    
    Parameters:
    -----------
    df_all : DataFrame
        All measurements data
    df_tracks : DataFrame
        Track summary data
    start_timepoint : int or None
        First timepoint to include (inclusive). None means no lower bound.
    end_timepoint : int or None
        Last timepoint to include (inclusive). None means no upper bound.
    verbose : bool
        Whether to print filtering information
    
    Returns:
    --------
    df_all_filtered, df_tracks_filtered : tuple of DataFrames
    """
    if start_timepoint is None and end_timepoint is None:
        return df_all, df_tracks
    
    original_measurements = len(df_all)
    original_tracks = df_all['unique_track_id'].nunique()
    
    # Build timepoint mask
    mask = pd.Series(True, index=df_all.index)
    if start_timepoint is not None:
        mask &= df_all['timepoint'] >= start_timepoint
    if end_timepoint is not None:
        mask &= df_all['timepoint'] <= end_timepoint
    
    df_all_filtered = df_all[mask].copy()
    
    # Update df_tracks if it exists
    if len(df_tracks) > 0:
        # Keep only tracks that have measurements in the filtered range
        valid_tracks = df_all_filtered['unique_track_id'].unique()
        df_tracks_filtered = df_tracks[df_tracks['unique_track_id'].isin(valid_tracks)].copy()
    else:
        df_tracks_filtered = df_tracks
    
    if verbose:
        t_range = f"[{start_timepoint if start_timepoint else 'start'} - {end_timepoint if end_timepoint else 'end'}]"
        print(f"\nTimepoint filtering {t_range}:")
        print(f"  Measurements: {original_measurements} → {len(df_all_filtered)}")
        print(f"  Tracks with data in range: {original_tracks} → {df_all_filtered['unique_track_id'].nunique()}")
    
    return df_all_filtered, df_tracks_filtered

def extract_bfp_measurements(input_dir, well, exclude_fovs=None):
    """Extract BFP (DAPI) mean intensity per tracked nucleus from raw image files."""
    if not SKIMAGE_AVAILABLE:
        print("WARNING: scikit-image not available; cannot extract BFP measurements.")
        return None
    _, _, full_name = parse_well(well)
    all_meas = []
    fov_dirs = sorted(Path(input_dir).glob(f"well_{full_name}_FOV*"))
    for fov_dir in fov_dirs:
        fov = int(fov_dir.name.split("FOV")[-1])
        if exclude_fovs and fov in exclude_fovs:
            continue
        masks_file = fov_dir / "tracked_masks.npy"
        bfp_file   = fov_dir / "dapi_mips.npy"
        if not masks_file.exists() or not bfp_file.exists():
            continue
        tracked_masks = np.load(masks_file)
        bfp_mips      = np.load(bfp_file)
        for t in range(tracked_masks.shape[0]):
            mask = tracked_masks[t]
            if mask.max() == 0:
                continue
            props = regionprops_table(mask, intensity_image=bfp_mips[t],
                                      properties=['label', 'mean_intensity'])
            df_t = pd.DataFrame(props).rename(columns={
                'label': 'track_id',
                'mean_intensity': 'bfp_mean_intensity',
            })
            df_t['timepoint'] = t
            df_t['fov'] = fov
            df_t['unique_track_id'] = f"{full_name}_{fov}_" + df_t['track_id'].astype(str)
            all_meas.append(df_t)
    if not all_meas:
        print("WARNING: No BFP measurements could be extracted.")
        return None
    df_bfp = pd.concat(all_meas, ignore_index=True)
    print(f"Extracted BFP measurements: {len(df_bfp)} observations")
    return df_bfp


def merge_bfp_with_mng(df_all, df_bfp):
    """Left-join BFP measurements onto the mNG measurement table."""
    df_merged = df_all.merge(
        df_bfp[['unique_track_id', 'timepoint', 'bfp_mean_intensity']],
        on=['unique_track_id', 'timepoint'], how='left'
    )
    print(f"Merged BFP into measurements: {len(df_merged)} rows")
    return df_merged


def compute_mng_bfp_ratio(df_all):
    """Add mng_bfp_ratio column (mNG / BFP) to the measurements table."""
    df_all = df_all.copy()
    df_all['mng_bfp_ratio'] = np.where(
        df_all['bfp_mean_intensity'] > 0,
        df_all['mean_intensity'] / df_all['bfp_mean_intensity'],
        np.nan
    )
    valid = df_all['mng_bfp_ratio'].notna().sum()
    print(f"Computed mNG/BFP ratio: {valid}/{len(df_all)} valid values")
    return df_all


def compute_track_quality_metrics(df_all, verbose=True):
    if verbose:
        print("\nComputing track quality metrics...")
    
    quality_metrics = []
    for track_id in df_all['unique_track_id'].unique():
        track_data = df_all[df_all['unique_track_id'] == track_id].sort_values('timepoint')
        if len(track_data) < 2:
            continue
        
        timepoints = track_data['timepoint'].values
        intensities = track_data['mean_intensity'].values
        
        intensity_diffs = np.diff(intensities)
        max_intensity_jump = np.max(np.abs(intensity_diffs))
        mean_intensity = np.mean(intensities)
        max_relative_intensity_jump = max_intensity_jump / mean_intensity if mean_intensity > 0 else np.inf
        
        has_position = 'centroid-0' in track_data.columns and 'centroid-1' in track_data.columns
        if has_position:
            y_pos = track_data['centroid-0'].values
            x_pos = track_data['centroid-1'].values
            dx, dy = np.diff(x_pos), np.diff(y_pos)
            displacements = np.sqrt(dx**2 + dy**2)
            max_position_jump = np.max(displacements)
        else:
            max_position_jump = np.nan
        
        expected = set(range(int(timepoints.min()), int(timepoints.max()) + 1))
        actual = set(timepoints.astype(int))
        gap_fraction = len(expected - actual) / len(expected) if len(expected) > 0 else 0
        
        area_cv = np.nan
        if 'area_pixels' in track_data.columns:
            areas = track_data['area_pixels'].values
            area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else np.inf
        
        quality_metrics.append({
            'unique_track_id': track_id,
            'track_duration': len(track_data),
            'max_intensity_jump': max_intensity_jump,
            'max_relative_intensity_jump': max_relative_intensity_jump,
            'max_position_jump': max_position_jump,
            'gap_fraction': gap_fraction,
            'area_cv': area_cv,
        })
    
    return pd.DataFrame(quality_metrics)


def filter_tracks_by_quality(df_all, df_quality, max_position_jump=50,
                              max_relative_intensity_jump=2.0, max_gap_fraction=0.1,
                              max_area_cv=0.5, verbose=True):
    n_original = len(df_quality)
    mask = pd.Series(True, index=df_quality.index)
    
    if max_position_jump and not df_quality['max_position_jump'].isna().all():
        mask &= df_quality['max_position_jump'] <= max_position_jump
    if max_relative_intensity_jump:
        mask &= df_quality['max_relative_intensity_jump'] <= max_relative_intensity_jump
    if max_gap_fraction:
        mask &= df_quality['gap_fraction'] <= max_gap_fraction
    if max_area_cv and not df_quality['area_cv'].isna().all():
        mask &= (df_quality['area_cv'] <= max_area_cv) | df_quality['area_cv'].isna()
    
    good_tracks = df_quality[mask]['unique_track_id'].values
    df_filtered = df_all[df_all['unique_track_id'].isin(good_tracks)].copy()
    df_quality_filtered = df_quality[mask].copy()
    
    n_filtered = n_original - len(good_tracks)
    if verbose:
        print(f"\nQuality filtering: {n_original} → {len(good_tracks)} tracks ({n_filtered} removed)")
    
    filter_stats = {'n_original': n_original, 'n_kept': len(good_tracks), 'n_filtered': n_filtered}
    return df_filtered, df_quality_filtered, filter_stats


def correct_drift(df_all, signal_col, method='population', df_control=None):
    """
    Subtract per-timepoint drift from single-cell trajectories.

    Parameters
    ----------
    method : 'population' or 'control'
        'population' — subtract the per-timepoint median of df_all itself.
                       Assumes most cells are NOT activating (robust for low MOI).
        'control'    — subtract the per-timepoint median of df_control (e.g. B2
                       uninfected well). Gold standard when a negative control exists.
    df_control : DataFrame
        Required when method='control'. Must contain 'timepoint' and signal_col.

    The correction preserves the absolute signal level at t=0 (only relative drift
    over time is removed).
    """
    if signal_col not in df_all.columns:
        print(f"WARNING: drift correction skipped — '{signal_col}' not in columns")
        return df_all

    if method == 'population':
        drift = df_all.groupby('timepoint')[signal_col].median()
        print(f"  Drift correction (population median): "
              f"t0={drift.iloc[0]:.4g}, t_end={drift.iloc[-1]:.4g}, "
              f"total drift={drift.iloc[-1]-drift.iloc[0]:.4g}")
    elif method == 'control':
        if df_control is None or signal_col not in df_control.columns:
            print("WARNING: drift correction method='control' requires df_control — skipping")
            return df_all
        drift = df_control.groupby('timepoint')[signal_col].median()
        print(f"  Drift correction (control well): "
              f"t0={drift.iloc[0]:.4g}, t_end={drift.iloc[-1]:.4g}, "
              f"total drift={drift.iloc[-1]-drift.iloc[0]:.4g}")
    else:
        return df_all

    baseline = drift.iloc[0]
    correction = df_all['timepoint'].map(drift) - baseline  # zero at t=0
    df_corrected = df_all.copy()
    df_corrected[signal_col] = df_corrected[signal_col] - correction
    return df_corrected


def analyze_activation(df_all, df_tracks, threshold, min_duration,
                       require_sustained=True, sustained_window=3,
                       min_pre_activation_frames=2, min_activation_timepoint=0,
                       n_baseline_frames=3, n_sd=3, signal_col='mean_intensity',
                       fixed_threshold=None, verbose=True):
    use_ratio = (signal_col == 'mng_bfp_ratio' and signal_col in df_all.columns
                 and fixed_threshold is None)
    if verbose:
        print(f"\n{'='*60}")
        if use_ratio:
            print(f"Analyzing activation using mNG/BFP ratio")
            print(f"  Per-cell threshold: baseline mean + {n_sd} × SD (first {n_baseline_frames} frames)")
        else:
            print(f"Analyzing activation (threshold={threshold}, min_activation_t={min_activation_timepoint})")
        print(f"{'='*60}")

    results = []
    n_insufficient_pre_tracking = 0
    n_no_baseline = 0

    for track_id in df_all['unique_track_id'].unique():
        track_data = df_all[df_all['unique_track_id'] == track_id].sort_values('timepoint')

        if len(track_data) < min_duration:
            continue

        fov = track_data['fov'].iloc[0]
        timepoints = track_data['timepoint'].values
        start_timepoint = timepoints[0]

        # --- Determine per-cell or global threshold ---
        if fixed_threshold is not None:
            cell_threshold = fixed_threshold
            signal_values = track_data[signal_col].values
        elif use_ratio:
            baseline_vals = track_data.head(n_baseline_frames)[signal_col].dropna()
            if len(baseline_vals) >= 2:
                cell_threshold = baseline_vals.mean() + n_sd * baseline_vals.std()
            elif len(baseline_vals) == 1:
                cell_threshold = baseline_vals.iloc[0]
            else:
                cell_threshold = np.nan
                n_no_baseline += 1
            signal_values = track_data[signal_col].values
        else:
            cell_threshold = threshold
            signal_values = track_data['mean_intensity'].values

        if np.isnan(cell_threshold):
            results.append({
                'unique_track_id': track_id, 'fov': fov,
                'track_duration': len(track_data),
                'start_timepoint': start_timepoint, 'end_timepoint': timepoints[-1],
                'mean_intensity': track_data['mean_intensity'].mean(),
                'max_intensity': track_data['mean_intensity'].max(),
                'min_intensity': track_data['mean_intensity'].min(),
                'activation_timepoint': None, 'pre_activation_frames': None,
                'insufficient_pre_tracking': False, 'too_early_activation': False,
                'had_early_high_signal': False, 'activates': False,
                'activation_threshold': np.nan,
            })
            continue

        above_threshold = signal_values >= cell_threshold
        activation_timepoint = None

        if require_sustained:
            for i in range(len(above_threshold) - sustained_window + 1):
                if all(above_threshold[i:i+sustained_window]):
                    if timepoints[i] >= min_activation_timepoint:
                        activation_timepoint = timepoints[i]
                        break
        else:
            for i, (t, above) in enumerate(zip(timepoints, above_threshold)):
                if above and t >= min_activation_timepoint:
                    activation_timepoint = t
                    break

        pre_activation_frames = activation_timepoint - start_timepoint if activation_timepoint is not None else None
        insufficient_pre_tracking = (activation_timepoint is not None and
                                      pre_activation_frames is not None and
                                      pre_activation_frames < min_pre_activation_frames)
        if insufficient_pre_tracking:
            n_insufficient_pre_tracking += 1

        had_early_high_signal = False
        if min_activation_timepoint > 0:
            early_mask = timepoints < min_activation_timepoint
            if any(above_threshold[early_mask] if any(early_mask) else []):
                had_early_high_signal = True

        activates = (activation_timepoint is not None and not insufficient_pre_tracking)

        results.append({
            'unique_track_id': track_id,
            'fov': fov,
            'track_duration': len(track_data),
            'start_timepoint': start_timepoint,
            'end_timepoint': timepoints[-1],
            'mean_intensity': track_data['mean_intensity'].mean(),
            'max_intensity': track_data['mean_intensity'].max(),
            'min_intensity': track_data['mean_intensity'].min(),
            'activation_timepoint': activation_timepoint,
            'pre_activation_frames': pre_activation_frames,
            'insufficient_pre_tracking': insufficient_pre_tracking,
            'too_early_activation': False,
            'had_early_high_signal': had_early_high_signal,
            'activates': activates,
            'activation_threshold': cell_threshold,
        })
    
    df_activation = pd.DataFrame(results)
    df_activating = df_activation[df_activation['activates']].copy()
    df_non_activating = df_activation[~df_activation['activates']].copy()
    
    if verbose:
        print(f"\nTracks analyzed: {len(df_activation)}")
        print(f"  Activating: {len(df_activating)} ({100*len(df_activating)/len(df_activation):.1f}%)")
        if n_insufficient_pre_tracking > 0:
            print(f"  Excluded (insufficient pre-tracking): {n_insufficient_pre_tracking}")
        if use_ratio and n_no_baseline > 0:
            print(f"  Skipped (no baseline data): {n_no_baseline}")
        if len(df_activating) > 0:
            print(f"  Median activation time: {df_activating['activation_timepoint'].median():.1f}")
        if use_ratio and len(df_activation) > 0:
            median_thresh = df_activation['activation_threshold'].median()
            print(f"  Median per-cell threshold: {median_thresh:.4f}")

    return df_activation, df_activating, df_non_activating


def group_by_activation_time(df_activating, df_all, bin_size=5):
    if len(df_activating) == 0:
        return {}, df_activating
    
    max_t = df_activating['activation_timepoint'].max()
    bins = np.arange(0, max_t + bin_size + 1, bin_size)
    df_activating = df_activating.copy()
    df_activating['activation_bin'] = pd.cut(
        df_activating['activation_timepoint'], 
        bins=bins, 
        labels=[f"t{int(b)}-{int(b+bin_size-1)}" for b in bins[:-1]],
        include_lowest=True
    )
    
    grouped = {}
    for bin_label in df_activating['activation_bin'].dropna().unique():
        tracks = df_activating[df_activating['activation_bin'] == bin_label]['unique_track_id'].tolist()
        df_bin = df_all[df_all['unique_track_id'].isin(tracks)].copy()
        df_bin['activation_bin'] = bin_label
        grouped[bin_label] = df_bin
    
    return grouped, df_activating


def generate_cumulative_activation_table(df_activating, df_activation, output_dir, well, max_timepoint=50):
    n_total = len(df_activation)
    if len(df_activating) == 0 or n_total == 0:
        return None
    
    cumulative_data = []
    for t in range(max_timepoint + 1):
        n_activated = (df_activating['activation_timepoint'] <= t).sum()
        cumulative_data.append({
            'timepoint': t,
            'n_activated': int(n_activated),
            'n_total': n_total,
            'cumulative_pct': round(n_activated / n_total * 100, 2)
        })
    
    df_cumulative = pd.DataFrame(cumulative_data)
    safe_save_csv(df_cumulative, output_dir / f"well_{well}_cumulative_activation.csv")
    return df_cumulative


def calculate_activation_windows(df_activating, df_activation, output_dir, well,
                                  min_activation_timepoint=0, verbose=True):
    """
    Calculate timeframe windows for early (first 10%), middle (50%), and late (last 10%) activation.
    
    Parameters:
    -----------
    df_activating : DataFrame
        DataFrame containing activating cells with 'activation_timepoint' column
    df_activation : DataFrame
        DataFrame containing all analyzed cells
    output_dir : Path
        Output directory for saving results
    min_activation_timepoint : int
        Minimum activation timepoint to include (filters out early artifacts, default=6)
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    dict : Dictionary containing activation window statistics
    """
    
    if len(df_activating) == 0:
        print("  No activating cells - skipping activation windows calculation")
        return None
    
    # Filter out cells activating before min_activation_timepoint (likely artifacts)
    df_filtered = df_activating[df_activating['activation_timepoint'] >= min_activation_timepoint].copy()
    n_filtered_out = len(df_activating) - len(df_filtered)
    
    if len(df_filtered) == 0:
        print(f"  No cells remaining after filtering (removed {n_filtered_out} cells activating before t={min_activation_timepoint})")
        return None
    
    # Sort by activation timepoint
    activation_times = df_filtered['activation_timepoint'].sort_values().values
    n_cells = len(activation_times)
    
    # Calculate percentile indices
    # First 10% (0-10th percentile)
    idx_10 = int(np.ceil(n_cells * 0.10)) - 1
    idx_10 = max(0, idx_10)
    
    # Middle 50% (25th-75th percentile)
    idx_25 = int(np.ceil(n_cells * 0.25)) - 1
    idx_75 = int(np.ceil(n_cells * 0.75)) - 1
    idx_25 = max(0, idx_25)
    idx_75 = min(n_cells - 1, idx_75)
    
    # Last 10% (90-100th percentile)
    idx_90 = int(np.ceil(n_cells * 0.90)) - 1
    idx_90 = min(n_cells - 1, idx_90)
    
    # Extract timepoints for each window
    early_10_start = activation_times[0]
    early_10_end = activation_times[idx_10]
    
    middle_50_start = activation_times[idx_25]
    middle_50_end = activation_times[idx_75]
    
    late_10_start = activation_times[idx_90]
    late_10_end = activation_times[-1]
    
    # Count cells in each window
    n_early_10 = idx_10 + 1
    n_middle_50 = idx_75 - idx_25 + 1
    n_late_10 = n_cells - idx_90
    
    # Calculate median and mean for each window
    early_cells = activation_times[:idx_10 + 1]
    middle_cells = activation_times[idx_25:idx_75 + 1]
    late_cells = activation_times[idx_90:]
    
    results = {
        'total_cells_analyzed': len(df_activation),
        'total_activating': len(df_activating),
        'filtered_out_early': n_filtered_out,
        'cells_after_filtering': n_cells,
        'min_activation_timepoint_filter': min_activation_timepoint,
        
        'early_10_pct': {
            'window_start': float(early_10_start),
            'window_end': float(early_10_end),
            'n_cells': int(n_early_10),
            'median': float(np.median(early_cells)),
            'mean': float(np.mean(early_cells)),
        },
        'middle_50_pct': {
            'window_start': float(middle_50_start),
            'window_end': float(middle_50_end),
            'n_cells': int(n_middle_50),
            'median': float(np.median(middle_cells)),
            'mean': float(np.mean(middle_cells)),
        },
        'late_10_pct': {
            'window_start': float(late_10_start),
            'window_end': float(late_10_end),
            'n_cells': int(n_late_10),
            'median': float(np.median(late_cells)),
            'mean': float(np.mean(late_cells)),
        },
        
        # Overall statistics
        'overall_median': float(np.median(activation_times)),
        'overall_mean': float(np.mean(activation_times)),
        'overall_std': float(np.std(activation_times)),
        'overall_min': float(activation_times[0]),
        'overall_max': float(activation_times[-1]),
    }
    
    if verbose:
        print(f"\n  Activation Windows Analysis (excluding t < {min_activation_timepoint}):")
        print(f"  " + "="*55)
        print(f"  Cells filtered out (t < {min_activation_timepoint}): {n_filtered_out}")
        print(f"  Cells analyzed: {n_cells}")
        print(f"  ")
        print(f"  Early 10% (first activators):")
        print(f"    Timeframe: {early_10_start:.1f} - {early_10_end:.1f}")
        print(f"    N cells: {n_early_10} | Median: {results['early_10_pct']['median']:.1f}")
        print(f"  ")
        print(f"  Middle 50% (25th-75th percentile):")
        print(f"    Timeframe: {middle_50_start:.1f} - {middle_50_end:.1f}")
        print(f"    N cells: {n_middle_50} | Median: {results['middle_50_pct']['median']:.1f}")
        print(f"  ")
        print(f"  Late 10% (last activators):")
        print(f"    Timeframe: {late_10_start:.1f} - {late_10_end:.1f}")
        print(f"    N cells: {n_late_10} | Median: {results['late_10_pct']['median']:.1f}")
        print(f"  ")
        print(f"  Overall: median={results['overall_median']:.1f}, mean={results['overall_mean']:.1f} ± {results['overall_std']:.1f}")
    
    # Save to CSV
    windows_df = pd.DataFrame([
        {
            'window': 'early_10_pct',
            'percentile_range': '0-10%',
            'timeframe_start': results['early_10_pct']['window_start'],
            'timeframe_end': results['early_10_pct']['window_end'],
            'n_cells': results['early_10_pct']['n_cells'],
            'median_activation_time': results['early_10_pct']['median'],
            'mean_activation_time': results['early_10_pct']['mean'],
        },
        {
            'window': 'middle_50_pct',
            'percentile_range': '25-75%',
            'timeframe_start': results['middle_50_pct']['window_start'],
            'timeframe_end': results['middle_50_pct']['window_end'],
            'n_cells': results['middle_50_pct']['n_cells'],
            'median_activation_time': results['middle_50_pct']['median'],
            'mean_activation_time': results['middle_50_pct']['mean'],
        },
        {
            'window': 'late_10_pct',
            'percentile_range': '90-100%',
            'timeframe_start': results['late_10_pct']['window_start'],
            'timeframe_end': results['late_10_pct']['window_end'],
            'n_cells': results['late_10_pct']['n_cells'],
            'median_activation_time': results['late_10_pct']['median'],
            'mean_activation_time': results['late_10_pct']['mean'],
        },
    ])
    
    # Add metadata row
    metadata_df = pd.DataFrame([{
        'window': 'METADATA',
        'percentile_range': f'min_t_filter={min_activation_timepoint}',
        'timeframe_start': results['overall_min'],
        'timeframe_end': results['overall_max'],
        'n_cells': n_cells,
        'median_activation_time': results['overall_median'],
        'mean_activation_time': results['overall_mean'],
    }])
    
    windows_df = pd.concat([windows_df, metadata_df], ignore_index=True)
    safe_save_csv(windows_df, output_dir / f"well_{well}_activation_windows.csv")
    print(f"  Saved activation windows table")
    
    return results


def explore_thresholds(df_all, df_tracks, min_duration, min_pre_activation_frames, 
                       output_dir, save_pdf=False, save_svg=False):
    print("\n" + "="*60)
    print("THRESHOLD EXPLORATION")
    print("="*60)
    
    thresholds = np.percentile(df_all['mean_intensity'], [50, 60, 70, 75, 80, 85, 90, 95])
    thresholds = np.unique(np.round(thresholds, -1))
    
    results = []
    for thresh in thresholds:
        _, df_act, df_non = analyze_activation(
            df_all, df_tracks, thresh, min_duration, True, 3, 
            min_pre_activation_frames, verbose=False
        )
        n_act, n_non = len(df_act), len(df_non)
        pct = 100 * n_act / (n_act + n_non) if (n_act + n_non) > 0 else 0
        median_t = df_act['activation_timepoint'].median() if n_act > 0 else np.nan
        print(f"Threshold {thresh:.0f}: {n_act} activating ({pct:.1f}%), median t={median_t:.1f}")
        results.append({'threshold': thresh, 'n_activating': n_act, 'pct': pct, 'median_t': median_t})
    
    res_df = pd.DataFrame(results)
    safe_save_csv(res_df, output_dir / "threshold_exploration.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['panel_1x3'])
    axes[0].hist(df_all['mean_intensity'], bins=80, color=COLORS['activating'], alpha=0.7)
    axes[0].set_xlabel('Mean mNG Intensity')
    axes[0].set_ylabel('Count')
    axes[1].plot(res_df['threshold'], res_df['pct'], 'o-', color=COLORS['median'])
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('% Activating')
    axes[2].plot(res_df['threshold'], res_df['median_t'], 'o-', color=COLORS['highlight'])
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('Median activation time')
    plt.tight_layout()
    save_figure(fig, output_dir, "threshold_exploration_panel", save_pdf, save_svg, subdir="panels")


def explore_n_sd(input_dir, output_dir, wells, threshold, min_duration,
                 sustained, sustained_window, min_pre_activation_frames,
                 start_timepoint=None, end_timepoint=None, min_activation_timepoint=0,
                 filter_quality=True, max_position_jump=50, max_intensity_jump=2.0,
                 max_gap_fraction=0.1, max_area_cv=0.5,
                 n_sd_values=None, save_pdf=False, save_svg=False,
                 drift_correction='none', control_well='B2'):
    """
    Sweep n_sd values across all wells to calibrate the per-cell activation threshold.

    For each well and each n_sd, runs activation detection and records % activating cells.
    Use the uninfected control well (e.g. B2) to find the n_sd where false positives
    drop to ~0%, then pick the lowest n_sd satisfying that criterion.
    """
    if n_sd_values is None:
        n_sd_values = [1, 2, 3, 4, 5, 6, 7, 8, 10]

    print("\n" + "="*60)
    print("N_SD THRESHOLD EXPLORATION")
    print("="*60)

    results = []

    for well in wells:
        print(f"\nLoading well {well}...")
        try:
            _, _, full_name = parse_well(well)
            df_all_w, df_tracks_w = load_well_data(input_dir, well)
            df_all_w, df_tracks_w = filter_timepoint_range(
                df_all_w, df_tracks_w, start_timepoint, end_timepoint, verbose=False
            )

            df_bfp = extract_bfp_measurements(input_dir, well)
            if df_bfp is not None:
                df_all_w = merge_bfp_with_mng(df_all_w, df_bfp)
                df_all_w = compute_mng_bfp_ratio(df_all_w)
                signal_col = 'mng_bfp_ratio'
            else:
                signal_col = 'mean_intensity'

            if drift_correction != 'none':
                df_ctrl = None
                if drift_correction == 'control':
                    df_ctrl_raw, _ = load_well_data(input_dir, control_well)
                    df_ctrl_raw, _ = filter_timepoint_range(df_ctrl_raw, df_ctrl_raw, start_timepoint, end_timepoint, verbose=False)
                    df_ctrl_bfp = extract_bfp_measurements(input_dir, control_well)
                    if df_ctrl_bfp is not None:
                        df_ctrl_raw = merge_bfp_with_mng(df_ctrl_raw, df_ctrl_bfp)
                        df_ctrl_raw = compute_mng_bfp_ratio(df_ctrl_raw)
                    df_ctrl = df_ctrl_raw
                df_all_w = correct_drift(df_all_w, signal_col, method=drift_correction, df_control=df_ctrl)

            if filter_quality:
                df_quality_w = compute_track_quality_metrics(df_all_w, verbose=False)
                df_all_w, _, _ = filter_tracks_by_quality(
                    df_all_w, df_quality_w, max_position_jump, max_intensity_jump,
                    max_gap_fraction, max_area_cv
                )

            for n_sd in n_sd_values:
                _, df_act, df_non = analyze_activation(
                    df_all_w, df_tracks_w, threshold, min_duration,
                    sustained, sustained_window,
                    min_pre_activation_frames, min_activation_timepoint,
                    n_sd=n_sd, signal_col=signal_col, verbose=False
                )
                n_act = len(df_act)
                n_tot = n_act + len(df_non)
                pct = 100 * n_act / n_tot if n_tot > 0 else 0
                results.append({'well': full_name, 'n_sd': n_sd,
                                 'n_activating': n_act, 'n_total': n_tot,
                                 'pct_activating': pct})
                print(f"  n_sd={n_sd:4.1f}: {n_act}/{n_tot} activating ({pct:.1f}%)")

        except Exception as e:
            print(f"  ERROR for well {well}: {e}")
            import traceback; traceback.print_exc()
            continue

    if not results:
        print("No results collected.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    safe_save_csv(res_df, output_dir / "n_sd_exploration.csv")

    # Plot
    wells_found = res_df['well'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(wells_found)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for well_name, color in zip(wells_found, colors):
        wd = res_df[res_df['well'] == well_name].sort_values('n_sd')
        ax.plot(wd['n_sd'], wd['pct_activating'], 'o-',
                label=well_name, color=color, linewidth=2)
    ax.set_xlabel('n_sd  (threshold = per-cell baseline + n_sd × SD)')
    ax.set_ylabel('% Activating cells')
    ax.set_title('Activation rate vs threshold stringency')
    ax.legend(title='Well')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_dir, "n_sd_exploration_panel", save_pdf, save_svg, subdir="panels")

    print(f"\nResults saved to {output_dir / 'n_sd_exploration.csv'}")
    return res_df


def plot_activation_analysis(df_all, df_activation, df_activating, df_non_activating,
                             threshold, output_dir, well, save_pdf=False, save_svg=False,
                             signal_col='mean_intensity', suffix=''):
    n_total = len(df_activation)
    
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZES['panel_2x3'])
    fig.suptitle(f'Activation Analysis — Well {well}', fontsize=14, fontweight='bold', y=0.98)
    
    # A: Activation time histogram
    ax = axes[0, 0]
    if len(df_activating) > 0:
        ax.hist(df_activating['activation_timepoint'], bins=np.arange(0, 50, 1), 
                color=COLORS['activating'], alpha=0.8, edgecolor='white')
        median_val = df_activating['activation_timepoint'].median()
        ax.axvline(median_val, color=COLORS['threshold'], linestyle='--', label=f'Median: {median_val:.1f}')
        ax.legend()
    ax.set_xlabel('Activation Timepoint')
    ax.set_ylabel('Count')
    ax.set_title(f'A   Activation times (n={len(df_activating)})', loc='left', fontweight='bold')
    
    # B: Cumulative activation
    ax = axes[0, 1]
    if len(df_activating) > 0:
        timepoints = np.arange(0, 50)
        cumulative_pct = [(df_activating['activation_timepoint'] <= t).sum() / n_total * 100 for t in timepoints]
        ax.plot(timepoints, cumulative_pct, linewidth=2.5, color=COLORS['median'])
        ax.fill_between(timepoints, 0, cumulative_pct, alpha=0.15, color=COLORS['median'])
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Cumulative % Activated')
    ax.set_ylim(0, 100)
    ax.set_title('B   Cumulative activation', loc='left', fontweight='bold')
    
    # C: Example trajectories
    ax = axes[0, 2]
    _ycol = signal_col if signal_col in df_all.columns else 'mean_intensity'
    _ylabel = 'mNG/BFP ratio' if _ycol == 'mng_bfp_ratio' else 'Mean mNG Intensity'
    _thresh_label = f'Median threshold={threshold:.4g}'
    if len(df_activating) > 0:
        for i, tid in enumerate(df_activating.sample(min(3, len(df_activating)))['unique_track_id']):
            data = df_all[df_all['unique_track_id'] == tid].sort_values('timepoint')
            ax.plot(data['timepoint'], data[_ycol], color=COLORS['activating'], alpha=0.6,
                    label='Activating' if i == 0 else None)
    if len(df_non_activating) > 0:
        for i, tid in enumerate(df_non_activating.sample(min(3, len(df_non_activating)))['unique_track_id']):
            data = df_all[df_all['unique_track_id'] == tid].sort_values('timepoint')
            ax.plot(data['timepoint'], data[_ycol], color=COLORS['non_activating'], alpha=0.5,
                    label='Non-activating' if i == 0 else None)
    ax.axhline(threshold, color=COLORS['threshold'], linestyle='--', label=_thresh_label)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel(_ylabel)
    ax.set_title('C   Example trajectories', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    # D: Max intensity distribution
    ax = axes[1, 0]
    if len(df_activating) > 0:
        ax.hist(df_activating['max_intensity'], bins=40, alpha=0.7, label='Activating', color=COLORS['activating'])
    if len(df_non_activating) > 0:
        ax.hist(df_non_activating['max_intensity'], bins=40, alpha=0.5, label='Non-activating', color=COLORS['non_activating'])
    ax.axvline(threshold, color=COLORS['threshold'], linestyle='--')
    ax.set_xlabel('Maximum mNG Intensity')
    ax.set_ylabel('Count')
    ax.set_title('D   Max intensity distribution', loc='left', fontweight='bold')
    ax.legend()
    
    # E: Activation by FOV
    ax = axes[1, 1]
    fov_stats = df_activation.groupby('fov').agg({'activates': ['sum', 'count']}).reset_index()
    fov_stats.columns = ['fov', 'n_act', 'n_total']
    fov_stats['pct'] = 100 * fov_stats['n_act'] / fov_stats['n_total']
    ax.bar(fov_stats['fov'].astype(str), fov_stats['pct'], color=COLORS['median'], alpha=0.8)
    ax.axhline(fov_stats['pct'].mean(), color=COLORS['threshold'], linestyle='--')
    ax.set_xlabel('FOV')
    ax.set_ylabel('% Activating')
    ax.set_title('E   Activation by FOV', loc='left', fontweight='bold')
    
    # F: Population mean
    ax = axes[1, 2]
    if len(df_activating) > 0:
        act_data = df_all[df_all['unique_track_id'].isin(df_activating['unique_track_id'])]
        mean_act = act_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
        ax.fill_between(mean_act.index, mean_act['mean'] - mean_act['std'], mean_act['mean'] + mean_act['std'],
                        alpha=0.15, color=COLORS['activating'])
        ax.plot(mean_act.index, mean_act['mean'], color=COLORS['activating'], linewidth=2.5, label='Activating')
    if len(df_non_activating) > 0:
        non_data = df_all[df_all['unique_track_id'].isin(df_non_activating['unique_track_id'])]
        mean_non = non_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
        ax.fill_between(mean_non.index, mean_non['mean'] - mean_non['std'], mean_non['mean'] + mean_non['std'],
                        alpha=0.15, color=COLORS['non_activating'])
        ax.plot(mean_non.index, mean_non['mean'], color=COLORS['non_activating'], linewidth=2, label='Non-activating')
    ax.axhline(threshold, color=COLORS['threshold'], linestyle='--', alpha=0.7)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel(_ylabel)
    ax.set_title('F   Population mean ± SD', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_dir, f"well_{well}_activation_analysis_panel{suffix}", save_pdf, save_svg, subdir="panels")


def plot_grouped_trajectories(grouped_data, threshold, output_dir, well, save_pdf=False, save_svg=False,
                              signal_col='mean_intensity', suffix=''):
    if not grouped_data:
        return
    
    n_groups = len(grouped_data)
    n_cols = min(4, n_groups)
    n_rows = int(np.ceil(n_groups / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows), squeeze=False)
    fig.suptitle(f'Trajectories by Activation Time — Well {well}', fontsize=14, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()
    
    _ycol = signal_col if signal_col in next(iter(grouped_data.values())).columns else 'mean_intensity'
    _ylabel = 'mNG/BFP ratio' if _ycol == 'mng_bfp_ratio' else 'mNG (a.u.)'
    for idx, (bin_label, df_group) in enumerate(sorted(grouped_data.items())):
        ax = axes_flat[idx]
        for tid in df_group['unique_track_id'].unique():
            data = df_group[df_group['unique_track_id'] == tid].sort_values('timepoint')
            ax.plot(data['timepoint'], data[_ycol], alpha=0.3, linewidth=0.8, color=COLORS['activating'])

        mean_traj = df_group.groupby('timepoint')[_ycol].agg(['mean', 'std'])
        ax.plot(mean_traj.index, mean_traj['mean'], color='black', linewidth=2.5)
        ax.fill_between(mean_traj.index, mean_traj['mean'] - mean_traj['std'], mean_traj['mean'] + mean_traj['std'],
                        alpha=0.2, color='black')
        ax.axhline(threshold, color=COLORS['threshold'], linestyle='--', alpha=0.6)
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel(_ylabel)
        ax.set_title(f'{bin_label} (n={df_group["unique_track_id"].nunique()})', fontweight='bold')
    
    for idx in range(len(grouped_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_dir, f"well_{well}_trajectories_by_bin_panel{suffix}", save_pdf, save_svg, subdir="panels")


def plot_summary_figure(df_all, df_activating, df_non_activating, threshold, output_dir, well, save_pdf=False, save_svg=False,
                        signal_col='mean_intensity', suffix=''):
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['panel_1x3'])
    fig.suptitle(f'Activation Summary — Well {well}', fontsize=14, fontweight='bold', y=0.98)
    
    # Panel A: Example trajectories
    ax = axes[0]
    _ycol = signal_col if signal_col in df_all.columns else 'mean_intensity'
    _ylabel = 'mNG/BFP ratio' if _ycol == 'mng_bfp_ratio' else 'mNG intensity'
    if len(df_activating) > 0:
        for tid in df_activating.sample(min(10, len(df_activating)))['unique_track_id']:
            data = df_all[df_all['unique_track_id'] == tid].sort_values('timepoint')
            ax.plot(data['timepoint'], data[_ycol], color=COLORS['activating'], alpha=0.5)
    if len(df_non_activating) > 0:
        for tid in df_non_activating.sample(min(5, len(df_non_activating)))['unique_track_id']:
            data = df_all[df_all['unique_track_id'] == tid].sort_values('timepoint')
            ax.plot(data['timepoint'], data[_ycol], color=COLORS['non_activating'], alpha=0.4)
    ax.axhline(threshold, color=COLORS['threshold'], linestyle='--')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel(_ylabel)
    ax.set_title('A   Individual trajectories', loc='left', fontweight='bold')
    
    # Panel B: Cumulative activation
    ax = axes[1]
    if len(df_activating) > 0:
        total = len(df_activating) + len(df_non_activating)
        sorted_times = np.sort(df_activating['activation_timepoint'].values)
        cumulative = np.arange(1, len(sorted_times) + 1) / total * 100
        ax.plot(sorted_times, cumulative, linewidth=2.5, color=COLORS['median'])
        ax.fill_between(sorted_times, 0, cumulative, alpha=0.15, color=COLORS['median'])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Cumulative activation (%)')
    ax.set_ylim(0, 105)
    ax.set_title('B   Cumulative activation', loc='left', fontweight='bold')
    
    # Panel C: Population dynamics
    ax = axes[2]
    if len(df_activating) > 0:
        act_data = df_all[df_all['unique_track_id'].isin(df_activating['unique_track_id'])]
        mean_act = act_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
        ax.fill_between(mean_act.index, mean_act['mean'] - mean_act['std'], mean_act['mean'] + mean_act['std'],
                        alpha=0.15, color=COLORS['activating'])
        ax.plot(mean_act.index, mean_act['mean'], color=COLORS['activating'], linewidth=2.5, label='Activating')
    if len(df_non_activating) > 0:
        non_data = df_all[df_all['unique_track_id'].isin(df_non_activating['unique_track_id'])]
        mean_non = non_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
        ax.fill_between(mean_non.index, mean_non['mean'] - mean_non['std'], mean_non['mean'] + mean_non['std'],
                        alpha=0.15, color=COLORS['non_activating'])
        ax.plot(mean_non.index, mean_non['mean'], color=COLORS['non_activating'], linewidth=2, label='Non-activating')
    ax.axhline(threshold, color=COLORS['threshold'], linestyle='--', alpha=0.6)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel(_ylabel)
    ax.set_title('C   Population mean ± SD', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_dir, f"well_{well}_summary_panel{suffix}", save_pdf, save_svg, subdir="panels")

def plot_individual_figures(df_all, df_activation, df_activating, df_non_activating,
                            threshold, output_dir, well, save_pdf=False, save_svg=False,
                            signal_col='mean_intensity', suffix=''):
    """Save each analysis plot as a separate publication-ready figure."""
    
    n_total = len(df_activation)
    
    # 1. Activation time histogram
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    if len(df_activating) > 0:
        ax.hist(df_activating['activation_timepoint'], bins=np.arange(0, 50, 1),
                color=COLORS['activating'], alpha=0.8, edgecolor='white')
        median_val = df_activating['activation_timepoint'].median()
        ax.axvline(median_val, color=COLORS['threshold'], linestyle='--', 
                   label=f'Median: {median_val:.1f}')
        ax.legend()
    ax.set_xlabel('Activation Timepoint')
    ax.set_ylabel('Count')
    plt.tight_layout()
    save_figure(fig, output_dir, f"well_{well}_activation_histogram", save_pdf, save_svg, subdir="individual")
    
    # 2. Cumulative activation curve
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    if len(df_activating) > 0:
        timepoints = np.arange(0, 50)
        cumulative_pct = [(df_activating['activation_timepoint'] <= t).sum() / n_total * 100 for t in timepoints]
        ax.plot(timepoints, cumulative_pct, linewidth=2.5, color=COLORS['median'])
        ax.fill_between(timepoints, 0, cumulative_pct, alpha=0.15, color=COLORS['median'])
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Cumulative % Activated')
    ax.set_ylim(0, 105)
    plt.tight_layout()
    save_figure(fig, output_dir, f"well_{well}_cumulative_activation", save_pdf, save_svg, subdir="individual")
    
    # 3. Example trajectories
    _ycol = signal_col if signal_col in df_all.columns else 'mean_intensity'
    _ylabel = 'mNG/BFP ratio' if _ycol == 'mng_bfp_ratio' else 'Mean mNG Intensity'
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_wide'])
    if len(df_activating) > 0:
        for i, tid in enumerate(df_activating.sample(min(3, len(df_activating)))['unique_track_id']):
            data = df_all[df_all['unique_track_id'] == tid].sort_values('timepoint')
            ax.plot(data['timepoint'], data[_ycol], color=COLORS['activating'], alpha=0.6,
                    label='Activating' if i == 0 else None)
    if len(df_non_activating) > 0:
        for i, tid in enumerate(df_non_activating.sample(min(3, len(df_non_activating)))['unique_track_id']):
            data = df_all[df_all['unique_track_id'] == tid].sort_values('timepoint')
            ax.plot(data['timepoint'], data[_ycol], color=COLORS['non_activating'], alpha=0.5,
                    label='Non-activating' if i == 0 else None)
    ax.axhline(threshold, color=COLORS['threshold'], linestyle='--', label=f'Median threshold={threshold:.4g}')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel(_ylabel)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    save_figure(fig, output_dir, f"well_{well}_example_trajectories{suffix}", save_pdf, save_svg, subdir="individual")
    
    # 4. Max intensity distribution
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    if len(df_activating) > 0:
        ax.hist(df_activating['max_intensity'], bins=40, alpha=0.7, label='Activating', color=COLORS['activating'])
    if len(df_non_activating) > 0:
        ax.hist(df_non_activating['max_intensity'], bins=40, alpha=0.5, label='Non-activating', color=COLORS['non_activating'])
    ax.axvline(threshold, color=COLORS['threshold'], linestyle='--')
    ax.set_xlabel('Maximum mNG Intensity')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, f"well_{well}_max_intensity_distribution", save_pdf, save_svg, subdir="individual")
    
    # 5. Activation by FOV
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    fov_stats = df_activation.groupby('fov').agg({'activates': ['sum', 'count']}).reset_index()
    fov_stats.columns = ['fov', 'n_act', 'n_total']
    fov_stats['pct'] = 100 * fov_stats['n_act'] / fov_stats['n_total']
    ax.bar(fov_stats['fov'].astype(str), fov_stats['pct'], color=COLORS['median'], alpha=0.8)
    ax.axhline(fov_stats['pct'].mean(), color=COLORS['threshold'], linestyle='--')
    ax.set_xlabel('FOV')
    ax.set_ylabel('% Activating')
    plt.tight_layout()
    save_figure(fig, output_dir, f"well_{well}_activation_by_fov", save_pdf, save_svg, subdir="individual")
    
    # 6. Population dynamics (mean ± SD)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_wide'])
    if len(df_activating) > 0:
        act_data = df_all[df_all['unique_track_id'].isin(df_activating['unique_track_id'])]
        mean_act = act_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
        ax.fill_between(mean_act.index, mean_act['mean'] - mean_act['std'],
                        mean_act['mean'] + mean_act['std'], alpha=0.15, color=COLORS['activating'])
        ax.plot(mean_act.index, mean_act['mean'], color=COLORS['activating'], linewidth=2.5, label='Activating')
    if len(df_non_activating) > 0:
        non_data = df_all[df_all['unique_track_id'].isin(df_non_activating['unique_track_id'])]
        mean_non = non_data.groupby('timepoint')[_ycol].agg(['mean', 'std'])
        ax.fill_between(mean_non.index, mean_non['mean'] - mean_non['std'],
                        mean_non['mean'] + mean_non['std'], alpha=0.15, color=COLORS['non_activating'])
        ax.plot(mean_non.index, mean_non['mean'], color=COLORS['non_activating'], linewidth=2, label='Non-activating')
    ax.axhline(threshold, color=COLORS['threshold'], linestyle='--', alpha=0.6)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel(_ylabel)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    save_figure(fig, output_dir, f"well_{well}_population_dynamics{suffix}", save_pdf, save_svg, subdir="individual")
    
    print(f"  Saved 6 individual figures to figures/individual/")

def load_images_for_fov(input_dir, well, fov):
    """Load image data for a specific FOV."""
    _, _, full_name = parse_well(well)
    fov_dir = Path(input_dir) / f"well_{full_name}_FOV{fov}"

    images = {}

    gfp_file = fov_dir / "gfp_mips.npy"
    if gfp_file.exists():
        images['gfp'] = np.load(gfp_file)
        print(f"    mNG: {images['gfp'].shape}")

    bfp_file = fov_dir / "dapi_mips.npy"
    if bfp_file.exists():
        images['bfp'] = np.load(bfp_file)
        print(f"    BFP: {images['bfp'].shape}")

    masks_file = fov_dir / "tracked_masks.npy"
    if masks_file.exists():
        images['masks'] = np.load(masks_file)
        print(f"    Masks: {images['masks'].shape}")

    return images


def view_napari_browser(input_dir, well, df_all, df_activation, fov=0):
    """Launch napari viewer to browse cells interactively."""
    if not NAPARI_AVAILABLE:
        print("ERROR: napari is not installed. Install with: pip install napari[all]")
        return

    _, _, full_name = parse_well(well)

    print(f"\nLoading FOV {fov} for napari browsing...")
    images = load_images_for_fov(input_dir, well, fov)

    if not images:
        print("ERROR: Could not load images")
        return

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

    # Add points at activation timepoint for activating cells
    fov_act = df_activation[(df_activation['fov'] == fov) & df_activation['activates']]
    act_points = []
    for _, row in fov_act.iterrows():
        track_id = row['unique_track_id']
        activation_t = row['activation_timepoint']
        if activation_t is None or pd.isna(activation_t):
            continue
        track_data = df_all[(df_all['unique_track_id'] == track_id) &
                            (df_all['timepoint'] == int(activation_t))]
        if len(track_data) > 0:
            cy = track_data.iloc[0]['centroid-0']
            cx = track_data.iloc[0]['centroid-1']
            act_points.append([int(activation_t), cy, cx])

    if act_points:
        viewer.add_points(np.array(act_points), name='Activating cells',
                          face_color='lime', size=20, opacity=0.7)

    # Add tracks for all measured cells in this FOV, colored to match their nucleus label
    fov_meas = df_all[df_all['fov'] == fov]
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
    print(f"  - Activating cell positions ({len(act_points)} cells)")
    print(f"\nUse the time slider to navigate through timepoints.")

    napari.run()


def characterize_activation_kinetics(df_activation, df_all, signal_col='mng_bfp_ratio',
                                     n_baseline_frames=3):
    """
    Fit a sigmoid to each activating cell's trajectory and extract kinetic parameters.

    Sigmoid model: f(t) = baseline + amplitude / (1 + exp(-k * (t - t0)))

    Derived metrics
    ---------------
    activation_start_t  : timepoint where signal reaches 10 % of amplitude above baseline
                          (= t0 - ln(9)/k)
    max_slope           : steepest slope of the sigmoid in signal/frame
                          (= k * amplitude / 4, at the inflection point t0)
    plateau_t           : timepoint where signal reaches 90 % of amplitude above baseline
                          (= t0 + ln(9)/k)
    sigmoid_r2          : R² of the sigmoid fit (quality indicator; < 0.7 → poor fit)
    """
    from scipy.optimize import curve_fit

    def sigmoid(t, baseline, amplitude, k, t0):
        return baseline + amplitude / (1.0 + np.exp(-k * (t - t0)))

    _ycol = signal_col if signal_col in df_all.columns else 'mean_intensity'

    records = []
    for _, row in df_activation[df_activation['activates']].iterrows():
        track_id = row['unique_track_id']
        track_data = df_all[df_all['unique_track_id'] == track_id].sort_values('timepoint')

        t = track_data['timepoint'].values.astype(float)
        y = track_data[_ycol].values.astype(float)
        valid = ~np.isnan(y)
        t, y = t[valid], y[valid]

        nan_row = {
            'unique_track_id': track_id,
            'sigmoid_baseline': np.nan, 'sigmoid_amplitude': np.nan,
            'sigmoid_k': np.nan, 'sigmoid_t0': np.nan,
            'activation_start_t': np.nan, 'max_slope': np.nan,
            'plateau_t': np.nan, 'sigmoid_r2': np.nan,
        }

        if len(t) < 6:
            records.append(nan_row)
            continue

        baseline_guess  = float(np.nanmean(y[:n_baseline_frames]))
        amplitude_guess = max(float(np.max(y)) - baseline_guess, 1e-6)
        t0_guess        = float(row['activation_timepoint']) if not pd.isna(row['activation_timepoint']) else float(np.median(t))
        k_guess         = 0.5

        try:
            popt, _ = curve_fit(
                sigmoid, t, y,
                p0=[baseline_guess, amplitude_guess, k_guess, t0_guess],
                bounds=([0, 0, 1e-3, t.min()], [np.inf, np.inf, 5.0, t.max()]),
                maxfev=10000,
            )
            baseline, amplitude, k, t0 = popt

            t_start  = t0 - np.log(9.0) / k   # 10 % of amplitude
            t_plateau = t0 + np.log(9.0) / k   # 90 % of amplitude
            max_slope = k * amplitude / 4.0

            y_pred  = sigmoid(t, *popt)
            ss_res  = np.sum((y - y_pred) ** 2)
            ss_tot  = np.sum((y - y.mean()) ** 2)
            r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            records.append({
                'unique_track_id':   track_id,
                'sigmoid_baseline':  baseline,
                'sigmoid_amplitude': amplitude,
                'sigmoid_k':         k,
                'sigmoid_t0':        t0,
                'activation_start_t': t_start,
                'max_slope':         max_slope,
                'plateau_t':         t_plateau,
                'sigmoid_r2':        r2,
            })
        except Exception:
            records.append(nan_row)

    df_kinetics = pd.DataFrame(records)

    if len(df_kinetics) > 0:
        good = df_kinetics['sigmoid_r2'] >= 0.7
        print(f"\nSigmoid kinetics fitted for {len(df_kinetics)} activating cells")
        print(f"  Good fits (R² ≥ 0.7): {good.sum()} ({100*good.mean():.0f}%)")
        if good.sum() > 0:
            print(f"  Median activation start : {df_kinetics.loc[good, 'activation_start_t'].median():.1f}")
            print(f"  Median max slope        : {df_kinetics.loc[good, 'max_slope'].median():.4f} /frame")
            print(f"  Median plateau          : {df_kinetics.loc[good, 'plateau_t'].median():.1f}")

    return df_kinetics


def get_available_wells(input_dir):
    """Scan input directory to find available wells."""
    input_path = Path(input_dir)
    wells = set()
    for d in input_path.glob("well_*_FOV*"):
        parts = d.name.split("_")
        if len(parts) >= 2:
            wells.add(parts[1])
    return sorted(wells)


def run_analysis(input_dir, output_dir, well, threshold, min_duration,
                 sustained, sustained_window, bin_size, min_pre_activation_frames,
                 start_timepoint=None, end_timepoint=None, min_activation_timepoint=0,
                 n_sd=3.0,
                 filter_quality=True, max_position_jump=50, max_intensity_jump=2.0,
                 max_gap_fraction=0.1, max_area_cv=0.5, exclude_fovs=None,
                 save_pdf=False, save_svg=False, save_individual=False,
                 drift_correction='none', control_well='B2', control_threshold=False):
    """Run analysis for a single well."""

    _, _, full_name = parse_well(well)

    df_all, df_tracks = load_well_data(input_dir, well, exclude_fovs)

    df_all, df_tracks = filter_timepoint_range(df_all, df_tracks, start_timepoint, end_timepoint)

    # Extract BFP and compute per-cell mNG/BFP ratio for threshold calculation
    print("\nExtracting BFP measurements for ratio-based activation detection...")
    df_bfp = extract_bfp_measurements(input_dir, well, exclude_fovs)
    if df_bfp is not None:
        df_all = merge_bfp_with_mng(df_all, df_bfp)
        df_all = compute_mng_bfp_ratio(df_all)
        signal_col = 'mng_bfp_ratio'
    else:
        print("WARNING: Falling back to raw mNG intensity with fixed threshold.")
        signal_col = 'mean_intensity'

    # Drift correction
    if drift_correction != 'none':
        print(f"\nApplying drift correction (method={drift_correction})...")
        df_control = None
        if drift_correction == 'control':
            print(f"  Loading control well {control_well}...")
            df_ctrl_raw, _ = load_well_data(input_dir, control_well)
            df_ctrl_raw, _ = filter_timepoint_range(df_ctrl_raw, df_ctrl_raw, start_timepoint, end_timepoint, verbose=False)
            df_ctrl_bfp = extract_bfp_measurements(input_dir, control_well)
            if df_ctrl_bfp is not None:
                df_ctrl_raw = merge_bfp_with_mng(df_ctrl_raw, df_ctrl_bfp)
                df_ctrl_raw = compute_mng_bfp_ratio(df_ctrl_raw)
            df_control = df_ctrl_raw
        df_all = correct_drift(df_all, signal_col, method=drift_correction, df_control=df_control)

    df_quality = compute_track_quality_metrics(df_all)
    safe_save_csv(df_quality, output_dir / f"well_{full_name}_track_quality.csv")

    filter_stats = None
    if filter_quality:
        df_all, df_quality, filter_stats = filter_tracks_by_quality(
            df_all, df_quality, max_position_jump, max_intensity_jump,
            max_gap_fraction, max_area_cv
        )

    # Compute population-level threshold from uninfected control well if requested
    fixed_threshold = None
    if control_threshold and signal_col == 'mng_bfp_ratio':
        print(f"\nComputing activation threshold from control well {control_well}...")
        df_ctrl_raw, _ = load_well_data(input_dir, control_well)
        df_ctrl_raw, _ = filter_timepoint_range(df_ctrl_raw, df_ctrl_raw, start_timepoint, end_timepoint, verbose=False)
        df_ctrl_bfp = extract_bfp_measurements(input_dir, control_well)
        if df_ctrl_bfp is not None:
            df_ctrl_raw = merge_bfp_with_mng(df_ctrl_raw, df_ctrl_bfp)
            df_ctrl_raw = compute_mng_bfp_ratio(df_ctrl_raw)
            ctrl_vals = df_ctrl_raw['mng_bfp_ratio'].dropna()
            fixed_threshold = ctrl_vals.mean() + n_sd * ctrl_vals.std()
            print(f"  Control distribution: mean={ctrl_vals.mean():.4f}, SD={ctrl_vals.std():.4f}")
            print(f"  Fixed threshold (mean + {n_sd}×SD): {fixed_threshold:.4f}")
        else:
            print("  WARNING: No BFP data found for control well — falling back to per-cell threshold.")

    df_activation, df_activating, df_non_activating = analyze_activation(
        df_all, df_tracks, threshold, min_duration, sustained, sustained_window,
        min_pre_activation_frames, min_activation_timepoint,
        n_sd=n_sd, signal_col=signal_col, fixed_threshold=fixed_threshold,
    )

    # Derive a single representative threshold value for plot annotations
    if signal_col == 'mng_bfp_ratio' and 'activation_threshold' in df_activation.columns:
        plot_threshold = df_activation['activation_threshold'].median()
    else:
        plot_threshold = threshold

    # Fit sigmoid kinetics for activating cells
    print("\nFitting sigmoid kinetics for activating cells...")
    df_kinetics = characterize_activation_kinetics(df_activation, df_all, signal_col=signal_col)
    df_activation = df_activation.merge(df_kinetics, on='unique_track_id', how='left')
    df_activating = df_activating.merge(df_kinetics, on='unique_track_id', how='left')

    grouped_data, df_activating_binned = group_by_activation_time(df_activating, df_all, bin_size)

    print("  Generating cumulative activation table...")
    generate_cumulative_activation_table(df_activating, df_activation, output_dir, full_name)

    print("  Calculating activation windows...")
    calculate_activation_windows(df_activating, df_activation, output_dir, full_name)

    print("\nGenerating figures...")
    # Always generate mNG intensity plots
    plot_activation_analysis(df_all, df_activation, df_activating, df_non_activating,
                             threshold, output_dir, full_name, save_pdf, save_svg,
                             signal_col='mean_intensity')
    plot_grouped_trajectories(grouped_data, threshold, output_dir, full_name, save_pdf, save_svg,
                              signal_col='mean_intensity')
    plot_summary_figure(df_all, df_activating, df_non_activating, threshold,
                        output_dir, full_name, save_pdf, save_svg, signal_col='mean_intensity')
    if save_individual:
        plot_individual_figures(df_all, df_activation, df_activating, df_non_activating,
                                threshold, output_dir, full_name, save_pdf, save_svg,
                                signal_col='mean_intensity')

    # Also generate mNG/BFP ratio plots when ratio data is available
    if signal_col == 'mng_bfp_ratio':
        print("  Generating mNG/BFP ratio figures...")
        plot_activation_analysis(df_all, df_activation, df_activating, df_non_activating,
                                 plot_threshold, output_dir, full_name, save_pdf, save_svg,
                                 signal_col='mng_bfp_ratio', suffix='_ratio')
        plot_grouped_trajectories(grouped_data, plot_threshold, output_dir, full_name, save_pdf, save_svg,
                                  signal_col='mng_bfp_ratio', suffix='_ratio')
        plot_summary_figure(df_all, df_activating, df_non_activating, plot_threshold,
                            output_dir, full_name, save_pdf, save_svg,
                            signal_col='mng_bfp_ratio', suffix='_ratio')
        if save_individual:
            plot_individual_figures(df_all, df_activation, df_activating, df_non_activating,
                                    plot_threshold, output_dir, full_name, save_pdf, save_svg,
                                    signal_col='mng_bfp_ratio', suffix='_ratio')
    
    print("\nSaving data files...")
    safe_save_csv(df_activation, output_dir / f"well_{full_name}_all_tracks.csv")
    if len(df_activating_binned) > 0:
        safe_save_csv(df_activating_binned, output_dir / f"well_{full_name}_activating.csv")
    safe_save_csv(df_non_activating, output_dir / f"well_{full_name}_non_activating.csv")
    
    for bin_label, df_group in grouped_data.items():
        safe_label = str(bin_label).replace("-", "_")
        safe_save_csv(df_group, output_dir / f"well_{full_name}_traj_{safe_label}.csv")
    
    return {
        'well': full_name,
        'total_tracks': filter_stats['n_original'] if filter_stats else len(df_activation),
        'filtered': filter_stats['n_filtered'] if filter_stats else 0,
        'analyzed': len(df_activation),
        'activating': len(df_activating),
        'non_activating': len(df_non_activating),
        'pct_activating': 100 * len(df_activating) / len(df_activation) if len(df_activation) > 0 else 0,
        'median_activation_t': df_activating['activation_timepoint'].median() if len(df_activating) > 0 else None,
        'threshold': threshold,
        'start_timepoint': start_timepoint,  # NEW
        'end_timepoint': end_timepoint,      # NEW
        'min_activation_timepoint': min_activation_timepoint,  # NEW
    }


if __name__ == "__main__":
    args = parse_args()
    
    if not args.no_style:
        setup_publication_style()
    
    print("="*70)
    print("OFF→ON REPORTER ANALYSIS (ultrack) - v4")
    print("="*70)
    
    input_dir, output_dir = setup_paths(args)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")

    if args.start_timepoint is not None or args.end_timepoint is not None:
        t_start = args.start_timepoint if args.start_timepoint else "0"
        t_end = args.end_timepoint if args.end_timepoint else "end"
        print(f"Timepoint range: [{t_start} - {t_end}]")
    if args.min_activation_timepoint > 0:
        print(f"Min activation timepoint: {args.min_activation_timepoint}")
    
    available_wells = get_available_wells(input_dir)
    if available_wells:
        print(f"Detected wells: {', '.join(available_wells)}")
    
    if len(args.well) == 1 and args.well[0].lower() == 'all':
        wells = available_wells if available_wells else AVAILABLE_WELLS
    else:
        wells = args.well
    
    if args.explore_thresholds:
        well_info = parse_well(wells[0])
        exclude_fovs = parse_exclude_fovs(args.exclude_fovs, well_info[2])
        df_all, df_tracks = load_well_data(input_dir, wells[0], exclude_fovs)
        explore_thresholds(df_all, df_tracks, args.min_duration,
                          args.min_pre_activation_frames, output_dir,
                          args.save_pdf, args.save_svg)
        exit()

    if args.explore_n_sd:
        explore_n_sd(
            input_dir, output_dir, wells,
            args.threshold, args.min_duration,
            args.sustained, args.sustained_window,
            args.min_pre_activation_frames,
            args.start_timepoint, args.end_timepoint, args.min_activation_timepoint,
            args.filter_quality,
            args.max_position_jump, args.max_intensity_jump,
            args.max_gap_fraction, args.max_area_cv,
            n_sd_values=args.n_sd_values,
            save_pdf=args.save_pdf, save_svg=args.save_svg,
            drift_correction=args.drift_correction,
            control_well=args.control_well,
        )
        exit()

    if args.view_napari:
        well_info = parse_well(wells[0])
        _, _, full_name = well_info
        exclude_fovs = parse_exclude_fovs(args.exclude_fovs, full_name)
        print("\nVisualization mode - loading data...")
        df_all, _ = load_well_data(input_dir, wells[0], exclude_fovs)
        act_file = output_dir / f"well_{full_name}_all_tracks.csv"
        if not act_file.exists():
            print(f"ERROR: No activation data found at {act_file}")
            print("Run the analysis first (without --view-napari) to generate the data.")
            exit(1)
        df_activation = pd.read_csv(act_file)
        view_napari_browser(input_dir, wells[0], df_all, df_activation, fov=args.fov)
        exit()

    summaries = []
    for well in wells:
        try:
            well_info = parse_well(well)
            exclude_fovs = parse_exclude_fovs(args.exclude_fovs, well_info[2])
            summary = run_analysis(
                input_dir, output_dir, well, args.threshold, args.min_duration,
                args.sustained, args.sustained_window, args.bin_size,
                args.min_pre_activation_frames,
                args.start_timepoint, args.end_timepoint, args.min_activation_timepoint,
                args.n_sd,
                args.filter_quality,
                args.max_position_jump, args.max_intensity_jump,
                args.max_gap_fraction, args.max_area_cv, exclude_fovs,
                args.save_pdf, args.save_svg, args.save_individual,
                drift_correction=args.drift_correction,
                control_well=args.control_well,
                control_threshold=args.control_threshold,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"\nERROR well {well}: {e}")
            import traceback
            traceback.print_exc()
    
    if summaries:
        df_summary = pd.DataFrame(summaries)
        safe_save_csv(df_summary, output_dir / "summary.csv")
        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(df_summary.to_string(index=False))
        print(f"\nResults saved to: {output_dir}")