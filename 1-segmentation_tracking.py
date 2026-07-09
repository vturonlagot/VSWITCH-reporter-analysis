# napari is imported lazily (only when the interactive viewer is actually
# used) so that headless / HPC SLURM runs do not require it to be installed.
import zarr
import numpy as np
from cellpose import models
from skimage.measure import regionprops_table
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ultrack imports
from ultrack import MainConfig, Tracker
from ultrack.utils import labels_to_contours

# ==================== CONFIGURATION ====================
# These are defaults only. They can be overridden at runtime with
# --zarr-path and --output-dir (see the argument parser at the bottom).
# The output directory is created in __main__, not at import time.
ZARR_PATH = "/path/to/your/data.zarr"
OUTPUT_DIR = Path("/path/to/your/output/1-nuclear_analysis")

# Channel indices
PHASE_CHANNEL_IDX = 0
DAPI_CHANNEL_IDX = 2
GFP_CHANNEL_IDX = 1

# Cellpose parameters for 2D
CELLPOSE_MODEL = 'nuclei'
DIAMETER = 100
FLOW_THRESHOLD = 0.4
CELLPOSE_THRESHOLD = 1.0
USE_GPU = True

# Ultrack parameters
ULTRACK_MIN_AREA = 200       # Minimum nucleus area in pixels
ULTRACK_MAX_AREA = 10000     # Maximum nucleus area in pixels
ULTRACK_MAX_DISTANCE = 50    # Maximum distance for linking between frames
ULTRACK_MIN_FRONTIER = 0.5   # For labels input, helps remove irrelevant segments

# Confluence / merged-nucleus mitigation
SPLIT_MERGED_NUCLEI = True        # Watershed-split segments that are too large
SPLIT_AREA_THRESHOLD = 3500       # Segments above this area (px) are candidates for splitting
SPLIT_MIN_DISTANCE = 10           # Minimum distance (px) between watershed seed peaks
#   — single nucleus median=2046px², 95th pct=3078px², so 3500 catches merged pairs (~4092px²)
#     without flagging large singles; min_distance ~nucleus_radius/2.5
#   — shape filter: segments below this circularity are excluded even after splitting
SHAPE_CIRCULARITY_MIN = 0.3       # 0 = any shape, 1 = perfect circle; set 0 to disable

# Measurement options
MEASURE_IN_3D = True

# ==================== HELPER FUNCTIONS ====================

def get_rows(zarr_store):
    """Return sorted list of row keys present in the zarr store (auto-detected)."""
    return sorted([k for k in zarr_store.keys() if not k.startswith('.')])


def get_wells(zarr_store, row):
    """Return sorted list of well keys under a given row (auto-detected)."""
    try:
        return sorted([k for k in zarr_store[row].keys() if not k.startswith('.')])
    except KeyError:
        return []


def get_fovs(zarr_store, row, well):
    """Return sorted list of available FOV indices for a given row/well."""
    try:
        keys = list(zarr_store[row][well].keys())
        return sorted([int(k) for k in keys if k.isdigit()])
    except KeyError:
        return []


def create_mip(image_3d, axis=0):
    """Create maximum intensity projection along specified axis (default: Z)."""
    return np.max(image_3d, axis=axis)


def normalize_for_cellpose(image_2d):
    """Normalize image for Cellpose."""
    img = image_2d.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-10)
    return img


def segment_nuclei_2d(image_2d, model, diameter, flow_threshold, cellpose_threshold):
    """Segment nuclei in a 2D MIP using Cellpose."""
    normalized = normalize_for_cellpose(image_2d)
    
    masks, flows, styles = model.eval(
        normalized,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellpose_threshold,
        do_3D=False,
    )
    return masks


def split_large_nuclei(masks, area_threshold=3000, min_distance=15, circularity_min=0.3):
    """
    Post-process Cellpose masks to mitigate confluent-cell merging.

    Two steps:
    1. Watershed splitting: segments larger than area_threshold are split using
       the distance-transform local maxima as seeds.
    2. Shape filtering: segments with circularity < circularity_min are removed
       (likely residual merges that watershed couldn't cleanly separate).

    Parameters
    ----------
    masks           : 2D label array from Cellpose
    area_threshold  : pixel area above which a segment is a split candidate
    min_distance    : minimum peak distance for watershed seeds (~ nucleus radius)
    circularity_min : lower bound on 4π·area/perimeter²; set 0.0 to skip filtering
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops

    result = np.zeros_like(masks)
    next_label = 1
    n_candidates = 0
    n_split = 0
    n_filtered = 0

    for region in regionprops(masks):
        label = region.label
        area  = region.area
        bbox  = region.bbox           # (min_row, min_col, max_row, max_col)

        # Crop to bounding box for efficiency
        sl = (slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3]))
        patch = (masks[sl] == label)

        if area > area_threshold:
            n_candidates += 1
            # Distance transform watershed
            distance = ndi.distance_transform_edt(patch)
            coords   = peak_local_max(distance, min_distance=min_distance, labels=patch)
            if len(coords) > 1:
                n_split += 1
                markers = np.zeros_like(patch, dtype=int)
                for idx, (r, c) in enumerate(coords, start=1):
                    markers[r, c] = idx
                markers, _ = ndi.label(markers > 0)
                split_patch = watershed(-distance, markers, mask=patch)
            else:
                split_patch = patch.astype(int)  # couldn't split — keep as-is
        else:
            split_patch = patch.astype(int)

        # Write each fragment back, optionally filtering by circularity
        for frag_id in np.unique(split_patch):
            if frag_id == 0:
                continue
            frag_mask = split_patch == frag_id
            if circularity_min > 0:
                frag_area = frag_mask.sum()
                perim = regionprops((frag_mask).astype(np.uint8))[0].perimeter
                circ  = (4 * np.pi * frag_area / perim ** 2) if perim > 0 else 0.0
                if circ < circularity_min:
                    n_filtered += 1
                    continue  # discard non-circular fragment
            result[sl][frag_mask] = next_label
            next_label += 1

    n_before = masks.max()
    n_after  = next_label - 1
    print(f"  split_large_nuclei: {n_before} → {n_after} segments "
          f"({n_candidates} candidates, {n_split} split, {n_filtered} filtered by shape)")
    return result


def create_ultrack_config():
    """Create ultrack configuration."""
    config = MainConfig()
    
    # Segmentation config - for labels input
    config.segmentation_config.min_area = ULTRACK_MIN_AREA
    config.segmentation_config.max_area = ULTRACK_MAX_AREA
    config.segmentation_config.min_frontier = ULTRACK_MIN_FRONTIER
    
    # Linking config
    config.linking_config.max_distance = ULTRACK_MAX_DISTANCE
    config.linking_config.max_neighbors = 5

    # Tracking config - adjust weights for your use case
    config.tracking_config.appear_weight = -0.5
    config.tracking_config.disappear_weight = -0.5
    config.tracking_config.division_weight = 0.0   # No cost for divisions; was -0.1 (still too penalizing)
    
    # Solver config - use Gurobi if available, otherwise use windowed CBC
    # Option A: If you have Gurobi installed (recommended)
    config.tracking_config.solver_name = 'GUROBI'
    
    # Option B: Workaround for CBC crashes - solve in smaller time windows
    #config.tracking_config.window_size = 10      # Solve 10 frames at a time
    #config.tracking_config.overlap_size = 2      # Overlap windows by 2 frames for continuity
    #config.tracking_config.time_limit = 300      # 5 min timeout per window
    
    # Data config
    config.data_config.working_dir = OUTPUT_DIR
    config.data_config.n_workers = 4
    
    return config


def run_ultrack(labels_stack, config, fov_output):
    """Run ultrack tracking on a stack of label images."""
    print("\nRunning ultrack tracking...")
    
    # Labels should be shape (T, Y, X) for 2D or (T, Z, Y, X) for 3D
    print(f"Labels stack shape: {labels_stack.shape}")
    
    # Create tracker
    tracker = Tracker(config=config)
    
    # Track using labels directly
    # ultrack will create foreground/contours internally
    tracker.track(
        labels=labels_stack,
        overwrite=True
    )
    
    # Export results
    tracks_df, graph = tracker.to_tracks_layer()
    
    # Convert to our format
    print(f"Ultrack found {len(tracks_df['track_id'].unique())} tracks")
    
    # Export segmentation masks with track IDs
    tracked_labels = tracker.to_zarr(
        chunks=(1, labels_stack.shape[-2], labels_stack.shape[-1]),
        overwrite=True
    )
    tracked_masks = np.array(tracked_labels)
    
    return tracked_masks, tracks_df, graph


def measure_intensity_2d(masks_2d, intensity_2d, background_percentile=5):
    """Measure intensity in 2D MIP."""
    if masks_2d.max() == 0:
        return pd.DataFrame()
    
    bg_mask = masks_2d == 0
    if bg_mask.sum() > 100:
        background = np.percentile(intensity_2d[bg_mask], background_percentile)
    else:
        background = np.percentile(intensity_2d, background_percentile)
    
    corrected = intensity_2d.astype(np.float32) - background
    corrected = np.clip(corrected, 0, None)
    
    props = regionprops_table(
        masks_2d,
        intensity_image=corrected,
        properties=['label', 'area', 'centroid', 'mean_intensity', 'max_intensity', 'min_intensity',
                    'eccentricity', 'solidity', 'perimeter',
                    'axis_major_length', 'axis_minor_length']
    )

    df = pd.DataFrame(props)
    df['integrated_intensity'] = df['mean_intensity'] * df['area']
    df['circularity'] = np.where(
        df['perimeter'] > 0,
        4 * np.pi * df['area'] / (df['perimeter'] ** 2),
        np.nan
    )
    df['aspect_ratio'] = np.where(
        df['axis_minor_length'] > 0,
        df['axis_major_length'] / df['axis_minor_length'],
        np.nan
    )
    df['background'] = background
    df.rename(columns={'area': 'area_pixels'}, inplace=True)
    
    return df


def measure_bfp_cv(masks_2d, bfp_2d):
    """
    Compute BFP intensity coefficient of variation (std/mean) per nucleus.

    When a nucleus fragments, the BFP signal within the (still-intact)
    segmentation mask becomes patchy → CV rises sharply.  This is
    division-resistant: daughter nuclei have uniform BFP distribution.

    Returns a DataFrame with columns [label, bfp_mean, bfp_std, bfp_cv].
    """
    if masks_2d.max() == 0:
        return pd.DataFrame(columns=['label', 'bfp_mean', 'bfp_std', 'bfp_cv'])

    rows = []
    img = bfp_2d.astype(np.float32)
    for label in np.unique(masks_2d):
        if label == 0:
            continue
        pixels = img[masks_2d == label]
        mean_v = float(pixels.mean())
        std_v  = float(pixels.std())
        cv     = std_v / mean_v if mean_v > 0 else np.nan
        rows.append({'label': label, 'bfp_mean': mean_v,
                     'bfp_std': std_v, 'bfp_cv': cv})
    return pd.DataFrame(rows)


def measure_phase_features(masks_2d, phase_2d):
    """
    Measure phase contrast intensity statistics per nucleus.

    Phase contrast texture (CV, std) increases as cells become granular
    or fragment, making it a useful morphological death indicator.

    Returns a DataFrame with columns [label, phase_mean, phase_std, phase_cv].
    """
    if masks_2d.max() == 0:
        return pd.DataFrame(columns=['label', 'phase_mean', 'phase_std', 'phase_cv'])

    rows = []
    img = phase_2d.astype(np.float32)
    for label in np.unique(masks_2d):
        if label == 0:
            continue
        pixels = img[masks_2d == label]
        mean_v = float(pixels.mean())
        std_v  = float(pixels.std())
        cv     = std_v / abs(mean_v) if mean_v != 0 else np.nan
        rows.append({'label': label, 'phase_mean': mean_v,
                     'phase_std': std_v, 'phase_cv': cv})
    return pd.DataFrame(rows)


def measure_intensity_3d_with_2d_mask(masks_2d, intensity_3d, background_percentile=5):
    """Use 2D mask to measure intensity across full 3D stack."""
    if masks_2d.max() == 0:
        return pd.DataFrame()
    
    mid_z = intensity_3d.shape[0] // 2
    bg_mask = masks_2d == 0
    if bg_mask.sum() > 100:
        background = np.percentile(intensity_3d[mid_z][bg_mask], background_percentile)
    else:
        background = np.percentile(intensity_3d, background_percentile)
    
    results = []
    for label in np.unique(masks_2d):
        if label == 0:
            continue
        
        nucleus_mask_2d = masks_2d == label
        intensity_column = intensity_3d[:, nucleus_mask_2d]
        
        corrected = intensity_column.astype(np.float32) - background
        corrected = np.clip(corrected, 0, None)
        
        props_2d = regionprops_table(
            (masks_2d == label).astype(int),
            properties=['centroid', 'area', 'eccentricity', 'solidity',
                        'perimeter', 'axis_major_length', 'axis_minor_length']
        )
        _area = props_2d['area'][0]
        _perim = props_2d['perimeter'][0]
        _maj   = props_2d['axis_major_length'][0]
        _min   = props_2d['axis_minor_length'][0]

        results.append({
            'label': label,
            'area_pixels': _area,
            'centroid-0': props_2d['centroid-0'][0],
            'centroid-1': props_2d['centroid-1'][0],
            'mean_intensity': corrected.mean(),
            'max_intensity': corrected.max(),
            'min_intensity': corrected.min(),
            'integrated_intensity': corrected.sum(),
            'integrated_intensity_mip': corrected.max(axis=0).sum(),
            'background': background,
            'z_profile_max': corrected.mean(axis=1).argmax(),
            'eccentricity':   props_2d['eccentricity'][0],
            'solidity':       props_2d['solidity'][0],
            'perimeter':      _perim,
            'axis_major_length': _maj,
            'axis_minor_length': _min,
            'circularity':    (4 * np.pi * _area / (_perim ** 2)) if _perim > 0 else np.nan,
            'aspect_ratio':   (_maj / _min) if _min > 0 else np.nan,
        })
    
    return pd.DataFrame(results)


def process_fov(zarr_store, row, well, fov, model, output_dir):
    """Process a single FOV using MIP-based segmentation and ultrack tracking."""
    print(f"\n{'='*80}")
    print(f"Processing Well {row}/{well}, FOV {fov}")
    print(f"{'='*80}")
    
    fov_output = output_dir / f"well_{row}{well}_FOV{fov}"
    fov_output.mkdir(exist_ok=True)
    
    # Get data info
    data_arr = zarr_store[row][well][str(fov)]['0']
    n_timepoints, n_channels, n_z, n_y, n_x = data_arr.shape
    print(f"Data shape: {data_arr.shape}")
    print(f"Timepoints: {n_timepoints}, Z-slices: {n_z}")
    
    # ==================== CREATE MIPs ====================
    print("\nCreating MIPs...")
    dapi_mips = []
    gfp_mips = []
    
    for t in tqdm(range(n_timepoints), desc="Creating MIPs"):
        dapi_3d = data_arr[t, DAPI_CHANNEL_IDX, :, :, :]
        gfp_3d = data_arr[t, GFP_CHANNEL_IDX, :, :, :]
        
        dapi_mips.append(create_mip(dapi_3d))
        gfp_mips.append(create_mip(gfp_3d))
    
    dapi_mips = np.array(dapi_mips)
    gfp_mips = np.array(gfp_mips)
    
    # ==================== SEGMENT ====================
    print("\nSegmenting nuclei (2D Cellpose)...")
    all_masks = []
    
    for t in tqdm(range(n_timepoints), desc="Segmenting"):
        masks = segment_nuclei_2d(
            dapi_mips[t], model, DIAMETER, FLOW_THRESHOLD, CELLPOSE_THRESHOLD
        )
        if SPLIT_MERGED_NUCLEI:
            masks = split_large_nuclei(
                masks,
                area_threshold=SPLIT_AREA_THRESHOLD,
                min_distance=SPLIT_MIN_DISTANCE,
                circularity_min=SHAPE_CIRCULARITY_MIN,
            )
        all_masks.append(masks)

    all_masks = np.array(all_masks)

    nuclei_counts = [all_masks[t].max() for t in range(n_timepoints)]
    print(f"Nuclei per timepoint (first 10): {nuclei_counts[:10]}")
    print(f"Total nuclei detected: {sum(nuclei_counts)}")
    
    if sum(nuclei_counts) == 0:
        print("\nERROR: No nuclei detected!")
        return dapi_mips, gfp_mips, all_masks, pd.DataFrame()
    
    # Save raw segmentation
    np.save(fov_output / "dapi_mips.npy", dapi_mips)
    np.save(fov_output / "gfp_mips.npy", gfp_mips)
    np.save(fov_output / "segmentation_masks_raw.npy", all_masks)
    
    # ==================== ULTRACK TRACKING ====================
    # Create config with FOV-specific working directory
    config = create_ultrack_config()
    config.data_config.working_dir = fov_output
    
    tracked_masks, tracks_df, graph = run_ultrack(all_masks, config, fov_output)
    
    n_tracks = len(tracks_df['track_id'].unique())
    print(f"Total tracks: {n_tracks}")
    
    if n_tracks > 0:
        track_lengths = tracks_df.groupby('track_id').size()
        print(f"Track lengths: min={track_lengths.min()}, max={track_lengths.max()}, "
              f"median={track_lengths.median():.1f}")
    
    np.save(fov_output / "tracked_masks.npy", tracked_masks)
    tracks_df.to_csv(fov_output / "ultrack_tracks.csv", index=False)
    
    # ==================== MEASURE INTENSITY ====================
    print(f"\nMeasuring GFP intensity ({'3D stack' if MEASURE_IN_3D else 'MIP'})...")
    measurements = []
    
    for t in tqdm(range(n_timepoints), desc="Measuring"):
        if MEASURE_IN_3D:
            gfp_3d = data_arr[t, GFP_CHANNEL_IDX, :, :, :]
            df_t = measure_intensity_3d_with_2d_mask(tracked_masks[t], gfp_3d)
        else:
            df_t = measure_intensity_2d(tracked_masks[t], gfp_mips[t])

        if len(df_t) > 0:
            # BFP (nuclear channel) intensity CV — measures fragmentation heterogeneity
            bfp_mip_t = np.max(data_arr[t, DAPI_CHANNEL_IDX, :, :, :], axis=0)
            df_bfp = measure_bfp_cv(tracked_masks[t], bfp_mip_t)
            if len(df_bfp) > 0:
                df_t = df_t.merge(df_bfp, on='label', how='left')

            # Phase contrast features — cytoplasmic texture/granularity
            phase_mip_t = np.max(data_arr[t, PHASE_CHANNEL_IDX, :, :, :], axis=0)
            df_phase = measure_phase_features(tracked_masks[t], phase_mip_t)
            if len(df_phase) > 0:
                df_t = df_t.merge(df_phase, on='label', how='left')

            df_t['timepoint'] = t
            df_t['row'] = row
            df_t['well'] = well
            df_t['fov'] = fov
            measurements.append(df_t)
    
    if not measurements:
        print("No measurements collected!")
        return dapi_mips, gfp_mips, tracked_masks, pd.DataFrame()
    
    df_measurements = pd.concat(measurements, ignore_index=True)
    df_measurements.rename(columns={'label': 'track_id'}, inplace=True)
    df_measurements.to_csv(fov_output / "nuclear_measurements.csv", index=False)
    
    # ==================== SUMMARY STATS ====================
    print("\nGenerating summary statistics...")
    
    track_stats = df_measurements.groupby('track_id').agg({
        'mean_intensity': ['mean', 'std', 'min', 'max'],
        'integrated_intensity': ['mean', 'std'],
        'area_pixels': ['mean', 'std'],
        'timepoint': ['min', 'max', 'count']
    }).reset_index()
    
    track_stats.columns = ['_'.join(col).strip('_') for col in track_stats.columns]
    track_stats.rename(columns={
        'timepoint_min': 'start_t',
        'timepoint_max': 'end_t',
        'timepoint_count': 'n_timepoints'
    }, inplace=True)
    track_stats['track_duration'] = track_stats['end_t'] - track_stats['start_t'] + 1
    track_stats['row'] = row
    track_stats['well'] = well
    track_stats['fov'] = fov
    track_stats.to_csv(fov_output / "track_summary_stats.csv", index=False)
    
    # ==================== PLOTS ====================
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Intensity trajectories
    ax = axes[0, 0]
    long_tracks = track_stats[track_stats['n_timepoints'] >= n_timepoints * 0.5]
    if len(long_tracks) > 0:
        top_tracks = long_tracks.nlargest(min(10, len(long_tracks)), 'n_timepoints')['track_id'].values
        for track_id in top_tracks:
            data_track = df_measurements[df_measurements['track_id'] == track_id].sort_values('timepoint')
            ax.plot(data_track['timepoint'], data_track['mean_intensity'],
                    label=f'Track {track_id}', alpha=0.7, linewidth=1.5)
        if len(top_tracks) <= 10:
            ax.legend(fontsize=7)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Mean GFP Intensity')
    ax.set_title(f'GFP Trajectories (Well {row}{well}, FOV {fov})')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Intensity distribution
    ax = axes[0, 1]
    ax.hist(track_stats['mean_intensity_mean'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(track_stats['mean_intensity_mean'].median(), color='red', linestyle='--',
               label=f"Median: {track_stats['mean_intensity_mean'].median():.1f}")
    ax.set_xlabel('Mean GFP Intensity')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('Distribution of Mean Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Track duration
    ax = axes[0, 2]
    ax.hist(track_stats['n_timepoints'], bins=range(1, n_timepoints+2), edgecolor='black', alpha=0.7)
    ax.set_xlabel('Track Duration (timepoints)')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('Track Duration Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Nucleus count over time
    ax = axes[1, 0]
    nuclei_per_t = df_measurements.groupby('timepoint')['track_id'].nunique()
    ax.plot(nuclei_per_t.index, nuclei_per_t.values, marker='o', markersize=4)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Number of Nuclei')
    ax.set_title('Nucleus Count Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Population mean over time
    ax = axes[1, 1]
    mean_per_t = df_measurements.groupby('timepoint')['mean_intensity'].agg(['mean', 'std'])
    ax.fill_between(mean_per_t.index,
                    mean_per_t['mean'] - mean_per_t['std'],
                    mean_per_t['mean'] + mean_per_t['std'], alpha=0.3)
    ax.plot(mean_per_t.index, mean_per_t['mean'], marker='o', markersize=4)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Mean GFP Intensity')
    ax.set_title('Population Mean Over Time (±1 SD)')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Example segmentation overlay
    ax = axes[1, 2]
    mid_t = n_timepoints // 2
    ax.imshow(dapi_mips[mid_t], cmap='gray')
    ax.contour(tracked_masks[mid_t], colors='cyan', linewidths=0.5)
    ax.set_title(f'Segmentation Overlay (t={mid_t})')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(fov_output / "analysis_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {fov_output}")
    
    return dapi_mips, gfp_mips, tracked_masks, df_measurements


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nuclear tracking pipeline — single-FOV mode for SLURM arrays")
    parser.add_argument('--row',  default=None, help="Row letter (e.g. B)")
    parser.add_argument('--well', default=None, help="Well number (e.g. 3)")
    parser.add_argument('--fov',  type=int, default=None, help="FOV index")
    parser.add_argument('--zarr-path', default=None,
                        help="Path to the input zarr store (overrides ZARR_PATH constant)")
    parser.add_argument('--output-dir', default=None,
                        help="Output directory (overrides OUTPUT_DIR constant)")
    parser.add_argument('--list-fovs', action='store_true',
                        help="Enumerate (row, well, fov) from the zarr store, write them "
                             "to --out, and exit (used by submit_1-segmentation_tracking.sh). No GPU needed.")
    parser.add_argument('--out', default=str(Path(__file__).parent / 'fov_list.txt'),
                        help="Output path for the FOV list (with --list-fovs)")
    parser.add_argument('--rows', nargs='+', default=None,
                        help="With --list-fovs: restrict to these rows (default: auto-detect)")
    parser.add_argument('--wells', nargs='+', default=None,
                        help="With --list-fovs: restrict to these wells (default: auto-detect)")
    args = parser.parse_args()

    # CLI args override the config constants at the top of the file
    if args.zarr_path is not None:
        ZARR_PATH = args.zarr_path

    # ---- FOV-listing mode: enumerate the store and exit (no GPU/Cellpose) ----
    if args.list_fovs:
        store = zarr.open(ZARR_PATH, mode='r')
        list_rows = args.rows if args.rows is not None else get_rows(store)
        print(f"Rows to scan: {list_rows}")
        tasks = []
        for row in list_rows:
            row = row.strip()
            wells = args.wells if args.wells is not None else get_wells(store, row)
            for well in wells:
                well = str(well).strip()
                fovs = get_fovs(store, row, well)
                if not fovs:
                    print(f"  Skipping {row}{well} — no FOVs found in zarr store")
                    continue
                for fov in fovs:
                    line = f"{row} {well} {fov}".replace('\r', '').replace('\n', '')
                    if len(line.split()) != 3:
                        raise ValueError(f"Malformed task line (hidden characters?): {repr(line)}")
                    tasks.append(line)
                    print(f"  Found: {row}/{well} FOV {fov}")
        with open(args.out, 'w', newline='\n') as f:
            f.write('\n'.join(tasks) + '\n')
        print(f"\nWrote {len(tasks)} tasks to {args.out}")
        exit()

    if args.output_dir is not None:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Single-FOV mode (SLURM array path): --row, --well, --fov are required.
    if args.row is None or args.well is None or args.fov is None:
        parser.error("--row, --well and --fov are required (single-FOV mode). "
                     "Use --list-fovs to enumerate the store.")
    _row, _well, _fov = args.row, args.well, args.fov

    print("="*80)
    print("NUCLEAR TRACKING PIPELINE (MIP + Cellpose + Ultrack)")
    print(f"  Single-FOV mode: {_row}/{_well} FOV {_fov}")
    print("="*80)

    print("\nLoading zarr store...")
    zarr_store = zarr.open(ZARR_PATH, mode='r')

    print("\nChecking GPU...")
    import torch
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ No GPU - using CPU")

    print("\nInitializing Cellpose...")
    model = models.CellposeModel(gpu=USE_GPU, model_type=CELLPOSE_MODEL)

    # ---- SLURM array path: process exactly one FOV then exit ----
    process_fov(zarr_store, _row, _well, _fov, model, OUTPUT_DIR)
    print(f"\nDone! Results in {OUTPUT_DIR / f'well_{_row}{_well}_FOV{_fov}'}")