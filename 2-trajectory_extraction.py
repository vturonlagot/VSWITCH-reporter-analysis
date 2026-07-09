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
    -w, --well WELL [WELL ...] Well(s) to analyze: B1, B2, B3, C1, C2, C3, or
                              'all' (default: B1)

Activation Threshold (control-well fixed threshold on mNG/BFP ratio):
    --n-sd VALUE              SD multiplier for the control-well fixed threshold,
                              control_mean + n_sd*SD (default: 3.0)
    --control-well WELL       Uninfected control well used to derive the threshold
                              (default: B2)
    --control-map C:T1,T2 ... Per-target control assignment, e.g.
                              --control-map B1:B1,B2,B3 C1:C1,C2,C3 makes B1 the
                              control for the B wells and C1 for the C wells

Activation Detection:
    -t, --threshold VALUE     mNG intensity threshold for activation (default: 40)
    --min-duration N          Minimum track duration in frames (default: 30)
    --sustained / --no-sustained
                              Require sustained activation above threshold (default: True)
    --sustained-window N      Number of consecutive frames above threshold (default: 6)
    --min-pre-activation-frames N
                              Minimum frames before activation to include track (default: 0)
    --bin-size N              Bin size for grouping activation times (default: 5)

Timepoint Filtering:
    --start-timepoint N       First frame to include (default: None = from 0)
    --end-timepoint N         Last frame to include (default: 48)
    --min-activation-timepoint N
                              Ignore activations before this frame (default: 0)

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

EXAMPLES
========
# Basic analysis with default threshold
python script.py --well C3

# Custom threshold with SVG output for publication (panels only)
python script.py --well C2 --threshold 40 --save-svg

# Analyze all wells, excluding problematic FOVs
python script.py --well all --exclude-fovs C2:3,5 C3:1

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
    'input_dir': "/path/to/your/output/1-nuclear_analysis",
    'output_dir': "/path/to/your/output/2-trajectories",
    'well': 'B1',
    'threshold': 40,
    'min_duration': 30,
    'sustained': True,
    'sustained_window': 6,
    'bin_size': 5,
    'min_pre_activation_frames': 0,
    'start_timepoint': None,  # NEW
    'end_timepoint': 48,    # NEW
    'min_activation_timepoint': 0,
}

AVAILABLE_WELLS = ['B1', 'B2', 'B3', 'C1', 'C2', 'C3']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze OFF→ON mNG reporter activation from ultrack tracking data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py --well B2 --threshold 800
  python script.py --well C3
  python script.py --well all

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
                        help='Number of SDs above the control-well mean mNG/BFP ratio used as '
                             'the fixed activation threshold (default: 3)')
    parser.add_argument('--filter-quality', action='store_true', default=True)
    parser.add_argument('--no-filter-quality', dest='filter_quality', action='store_false')
    parser.add_argument('--max-position-jump', type=float, default=50)
    parser.add_argument('--max-intensity-jump', type=float, default=2.0)
    parser.add_argument('--max-gap-fraction', type=float, default=0.1)
    parser.add_argument('--max-area-cv', type=float, default=0.5)
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--save-svg', action='store_true')
    parser.add_argument('--control-well', type=str, default='B2',
                        help='Default uninfected control well used to compute the fixed '
                             'activation threshold (mean + n_sd × SD of the mNG/BFP ratio). '
                             'Used for any well not covered by --control-map (default: B2)')
    parser.add_argument('--control-map', nargs='+', default=None,
                        help='Explicit control-well assignments as CONTROL:TARGET1,TARGET2,... '
                             'entries, e.g. --control-map B1:B1,B2,B3 C1:C1,C2,C3 makes B1 the '
                             'control for the B wells and C1 the control for the C wells. Wells '
                             'not listed fall back to --control-well.')
    parser.add_argument('--exclude-fovs', nargs='+', default=None)

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


def parse_control_map(entries):
    """Parse CONTROL:TARGET1,TARGET2,... entries into a {target: control} dict.

    e.g. ['B1:B1,B2,B3', 'C1:C1,C2,C3'] -> {'B1':'B1','B2':'B1','B3':'B1',
    'C1':'C1','C2':'C1','C3':'C1'}. Well names are normalized to upper case.
    """
    mapping = {}
    if not entries:
        return mapping
    for entry in entries:
        if ':' not in entry:
            raise ValueError(
                f"Invalid --control-map entry '{entry}'. Expected CONTROL:TARGET1,TARGET2,..."
            )
        control_part, targets_part = entry.split(':', 1)
        control = parse_well(control_part)[2]
        for tgt in targets_part.split(','):
            tgt = tgt.strip()
            if tgt:
                mapping[parse_well(tgt)[2]] = control
    return mapping


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


def analyze_activation(df_all, df_tracks, threshold, min_duration,
                       require_sustained=True, sustained_window=3,
                       min_pre_activation_frames=2, min_activation_timepoint=0,
                       signal_col='mean_intensity',
                       fixed_threshold=None, verbose=True):
    use_ratio = (signal_col == 'mng_bfp_ratio' and signal_col in df_all.columns)
    if use_ratio and fixed_threshold is None:
        raise ValueError(
            "Ratio-based activation requires a control-well fixed threshold "
            "(fixed_threshold). Per-cell adaptive thresholds are not supported."
        )
    if verbose:
        print(f"\n{'='*60}")
        if use_ratio:
            print(f"Analyzing activation using mNG/BFP ratio")
            print(f"  Fixed control-well threshold: {fixed_threshold:.4f}")
        else:
            print(f"Analyzing activation (threshold={threshold}, min_activation_t={min_activation_timepoint})")
        print(f"{'='*60}")

    results = []
    n_insufficient_pre_tracking = 0

    for track_id in df_all['unique_track_id'].unique():
        track_data = df_all[df_all['unique_track_id'] == track_id].sort_values('timepoint')

        if len(track_data) < min_duration:
            continue

        fov = track_data['fov'].iloc[0]
        timepoints = track_data['timepoint'].values
        start_timepoint = timepoints[0]

        # --- Determine activation threshold (control-well fixed) ---
        if use_ratio:
            cell_threshold = fixed_threshold
            signal_values = track_data[signal_col].values
        else:
            cell_threshold = threshold
            signal_values = track_data['mean_intensity'].values

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
        if len(df_activating) > 0:
            print(f"  Median activation time: {df_activating['activation_timepoint'].median():.1f}")
        if use_ratio and len(df_activation) > 0:
            print(f"  Fixed control-well threshold: {fixed_threshold:.4f}")

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
                 save_pdf=False, save_svg=False,
                 control_well='B2'):
    """Run analysis for a single well."""

    _, _, full_name = parse_well(well)

    df_all, df_tracks = load_well_data(input_dir, well, exclude_fovs)

    df_all, df_tracks = filter_timepoint_range(df_all, df_tracks, start_timepoint, end_timepoint)

    # Extract BFP and compute the mNG/BFP ratio used for activation detection
    print("\nExtracting BFP measurements for ratio-based activation detection...")
    df_bfp = extract_bfp_measurements(input_dir, well, exclude_fovs)
    if df_bfp is not None:
        df_all = merge_bfp_with_mng(df_all, df_bfp)
        df_all = compute_mng_bfp_ratio(df_all)
        signal_col = 'mng_bfp_ratio'
    else:
        print("WARNING: Falling back to raw mNG intensity with fixed threshold.")
        signal_col = 'mean_intensity'

    df_quality = compute_track_quality_metrics(df_all)
    safe_save_csv(df_quality, output_dir / f"well_{full_name}_track_quality.csv")

    filter_stats = None
    if filter_quality:
        df_all, df_quality, filter_stats = filter_tracks_by_quality(
            df_all, df_quality, max_position_jump, max_intensity_jump,
            max_gap_fraction, max_area_cv
        )

    # Compute the population-level activation threshold from the uninfected
    # control well (mean + n_sd × SD of the mNG/BFP ratio across all cells and
    # timepoints). This control-well fixed threshold is the only supported mode.
    fixed_threshold = None
    if signal_col == 'mng_bfp_ratio':
        print(f"\nComputing activation threshold from control well {control_well}...")
        df_ctrl_raw, _ = load_well_data(input_dir, control_well)
        df_ctrl_raw, _ = filter_timepoint_range(df_ctrl_raw, df_ctrl_raw, start_timepoint, end_timepoint, verbose=False)
        df_ctrl_bfp = extract_bfp_measurements(input_dir, control_well)
        if df_ctrl_bfp is None:
            raise RuntimeError(
                f"No BFP data for control well {control_well}; cannot compute the "
                f"control-well activation threshold."
            )
        df_ctrl_raw = merge_bfp_with_mng(df_ctrl_raw, df_ctrl_bfp)
        df_ctrl_raw = compute_mng_bfp_ratio(df_ctrl_raw)
        ctrl_vals = df_ctrl_raw['mng_bfp_ratio'].dropna()
        fixed_threshold = ctrl_vals.mean() + n_sd * ctrl_vals.std()
        print(f"  Control distribution: mean={ctrl_vals.mean():.4f}, SD={ctrl_vals.std():.4f}")
        print(f"  Fixed threshold (mean + {n_sd}×SD): {fixed_threshold:.4f}")

    df_activation, df_activating, df_non_activating = analyze_activation(
        df_all, df_tracks, threshold, min_duration, sustained, sustained_window,
        min_pre_activation_frames, min_activation_timepoint,
        signal_col=signal_col, fixed_threshold=fixed_threshold,
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

    print("\nSaving data files...")
    safe_save_csv(df_activation, output_dir / f"well_{full_name}_all_tracks.csv")
    if len(df_activating_binned) > 0:
        safe_save_csv(df_activating_binned, output_dir / f"well_{full_name}_activating.csv")
    safe_save_csv(df_non_activating, output_dir / f"well_{full_name}_non_activating.csv")
    
    for bin_label, df_group in grouped_data.items():
        safe_label = str(bin_label).replace("-", "_")
        safe_save_csv(df_group, output_dir / f"well_{full_name}_traj_{safe_label}.csv")
    
    summary = {
        'well': full_name,
        'control_well': control_well,
        'total_tracks': filter_stats['n_original'] if filter_stats else len(df_activation),
        'filtered': filter_stats['n_filtered'] if filter_stats else 0,
        'analyzed': len(df_activation),
        'activating': len(df_activating),
        'non_activating': len(df_non_activating),
        'pct_activating': 100 * len(df_activating) / len(df_activation) if len(df_activation) > 0 else 0,
        'median_activation_t': df_activating['activation_timepoint'].median() if len(df_activating) > 0 else None,
        'activation_threshold': fixed_threshold,
        'threshold': threshold,
        'start_timepoint': start_timepoint,
        'end_timepoint': end_timepoint,
        'min_activation_timepoint': min_activation_timepoint,
    }

    # Write a per-well summary file so concurrent per-well jobs don't clobber a
    # single shared summary.csv (which is rebuilt from these — see __main__).
    safe_save_csv(pd.DataFrame([summary]), output_dir / f"well_{full_name}_summary.csv")

    return summary


if __name__ == "__main__":
    args = parse_args()

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
    
    control_map = parse_control_map(args.control_map)

    summaries = []
    for well in wells:
        try:
            well_info = parse_well(well)
            full_name = well_info[2]
            exclude_fovs = parse_exclude_fovs(args.exclude_fovs, full_name)
            control_well = control_map.get(full_name, args.control_well)
            summary = run_analysis(
                input_dir, output_dir, well, args.threshold, args.min_duration,
                args.sustained, args.sustained_window, args.bin_size,
                args.min_pre_activation_frames,
                args.start_timepoint, args.end_timepoint, args.min_activation_timepoint,
                args.n_sd,
                args.filter_quality,
                args.max_position_jump, args.max_intensity_jump,
                args.max_gap_fraction, args.max_area_cv, exclude_fovs,
                args.save_pdf, args.save_svg,
                control_well=control_well,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"\nERROR well {well}: {e}")
            import traceback
            traceback.print_exc()

    if summaries:
        # Rebuild summary.csv from every per-well summary file present in the output
        # directory. This keeps the combined summary complete even when wells are
        # processed by separate (e.g. per-well SLURM array) jobs, where a plain
        # overwrite would leave only the last well.
        summary_files = sorted(output_dir.glob("well_*_summary.csv"))
        df_summary = pd.concat(
            [pd.read_csv(f) for f in summary_files], ignore_index=True
        ) if summary_files else pd.DataFrame(summaries)
        df_summary = df_summary.drop_duplicates(subset='well', keep='last').sort_values('well')
        safe_save_csv(df_summary, output_dir / "summary.csv")
        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(df_summary.to_string(index=False))
        print(f"\nResults saved to: {output_dir}")