import napari
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
ZARR_PATH = "/hpc/projects/arias_group/Vincent_Turon-Lagot/Imaging_Experiments/20260321_A549_OFFON18_OFFON20_48hrs_dragonfly/2-register/20260321_A549_OFFON18_OFFON20_48hrs_dragonfly_registered.zarr"
OUTPUT_DIR = Path("/hpc/projects/arias_group/Vincent_Turon-Lagot/Imaging_Experiments/20260321_A549_OFFON18_OFFON20_48hrs_dragonfly/Image_analysis/1-nuclear_analysis/nuclear_analysis_output_ultrack")
OUTPUT_DIR.mkdir(exist_ok=True)

# Channel indices
PHASE_CHANNEL_IDX = 0
DAPI_CHANNEL_IDX = 2
GFP_CHANNEL_IDX = 1

# Rows, Wells and FOVs (FOVs are auto-detected from the zarr store)
ROWS  = ['B']
WELLS = ['3', '4']

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

# Processing options
PROCESS_SINGLE_FOV_FIRST = False
TEST_ROW = 'B'
TEST_WELL = '2'
TEST_FOV = 3

# Measurement options
MEASURE_IN_3D = True

# Debug mode
DEBUG = False
INSPECT_CHANNELS_ONLY = False

# Cellpose parameter assessment
ASSESS_CELLPOSE = False
ASSESS_DIAMETERS       = [70, 85, 100, 115, 130]
ASSESS_FLOW_THRESHOLDS = [0.3, 0.4, 0.6]
ASSESS_TIMEPOINTS      = [0, 48, -1]  # timepoints to sweep: 0 = first, -1 = last (mid = e.g. 48)

# ==================== HELPER FUNCTIONS ====================

def get_fovs(zarr_store, row, well):
    """Return sorted list of available FOV indices for a given row/well."""
    try:
        keys = list(zarr_store[row][well].keys())
        return sorted([int(k) for k in keys if k.isdigit()])
    except KeyError:
        return []


def inspect_channels(zarr_store, row, well, fov, output_dir):
    """Inspect all channels to identify DAPI and GFP."""
    from skimage.io import imsave
    
    print("\n" + "="*60)
    print("CHANNEL INSPECTION")
    print("="*60)
    
    data_arr = zarr_store[row][well][str(fov)]['0']
    n_timepoints, n_channels, n_z, n_y, n_x = data_arr.shape
    print(f"Data shape: {data_arr.shape}")
    print(f"Number of channels: {n_channels}")
    
    t = 0
    print(f"\nAnalyzing timepoint {t}:")
    
    for ch in range(n_channels):
        ch_data = data_arr[t, ch, :, :, :]
        mip = np.max(ch_data, axis=0)
        
        print(f"\n  Channel {ch}:")
        print(f"    3D shape: {ch_data.shape}, dtype: {ch_data.dtype}")
        print(f"    3D range: [{ch_data.min()}, {ch_data.max()}]")
        print(f"    MIP range: [{mip.min()}, {mip.max()}]")
        
        if mip.max() > 0:
            mip_norm = ((mip - mip.min()) / (mip.max() - mip.min() + 1e-10) * 65535).astype(np.uint16)
        else:
            mip_norm = mip.astype(np.uint16)
        
        save_path = output_dir / f"channel_{ch}_mip_t0.tif"
        imsave(save_path, mip_norm)
        print(f"    Saved: {save_path}")
    
    print("\n" + "="*60)


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


# ==================== CELLPOSE PARAMETER ASSESSMENT ====================

def assess_cellpose_params(zarr_store, row, well, fov, model, output_dir,
                           diameters, flow_thresholds, timepoint=0):
    """
    Run a grid search over Cellpose diameter × flow_threshold on one timepoint.

    Saves a comparison figure:  rows = flow_threshold, cols = diameter
    Each panel shows the DAPI MIP with coloured nucleus outlines and the
    detected nucleus count.
    """
    data_arr  = zarr_store[row][well][str(fov)]['0']
    dapi_3d   = data_arr[timepoint, DAPI_CHANNEL_IDX, :, :, :]
    dapi_mip  = create_mip(dapi_3d)

    n_rows = len(flow_thresholds)
    n_cols = len(diameters)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(f"Cellpose parameter sweep — {row}{well} FOV{fov} t={timepoint}",
                 fontsize=14, y=1.01)

    vmin = np.percentile(dapi_mip, 1)
    vmax = np.percentile(dapi_mip, 99.5)

    for ri, flow_thr in enumerate(flow_thresholds):
        for ci, diam in enumerate(diameters):
            masks = segment_nuclei_2d(dapi_mip, model, diam, flow_thr, CELLPOSE_THRESHOLD)
            n_nuclei = masks.max()

            ax = axes[ri][ci]
            ax.imshow(dapi_mip, cmap='gray', vmin=vmin, vmax=vmax)

            # Draw nucleus outlines
            if masks.max() > 0:
                ax.contour(masks, levels=np.arange(0.5, masks.max() + 1), colors='red', linewidths=0.5)

            ax.set_title(f"diam={diam}  flow={flow_thr}\nn={n_nuclei}", fontsize=9)
            ax.axis('off')

    fig.tight_layout()
    out_path = output_dir / f"cellpose_param_sweep_{row}{well}_FOV{fov}_t{timepoint}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved parameter sweep → {out_path}")
    return out_path


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nuclear tracking pipeline — single-FOV mode for SLURM arrays")
    parser.add_argument('--row',  default=None, help="Row letter (e.g. B)")
    parser.add_argument('--well', default=None, help="Well number (e.g. 3)")
    parser.add_argument('--fov',  type=int, default=None, help="FOV index")
    args = parser.parse_args()

    # Single-FOV mode: CLI args override config constants
    SINGLE_FOV_MODE = args.fov is not None
    if SINGLE_FOV_MODE:
        _row  = args.row  if args.row  is not None else TEST_ROW
        _well = args.well if args.well is not None else TEST_WELL
        _fov  = args.fov

    print("="*80)
    print("NUCLEAR TRACKING PIPELINE (MIP + Cellpose + Ultrack)")
    if SINGLE_FOV_MODE:
        print(f"  Single-FOV mode: {_row}/{_well} FOV {_fov}")
    print("="*80)

    print("\nLoading zarr store...")
    zarr_store = zarr.open(ZARR_PATH, mode='r')

    if INSPECT_CHANNELS_ONLY:
        OUTPUT_DIR.mkdir(exist_ok=True)
        inspect_channels(zarr_store, TEST_ROW, TEST_WELL, TEST_FOV, OUTPUT_DIR)
        print("\nExiting after channel inspection.")
        exit()

    if ASSESS_CELLPOSE:
        OUTPUT_DIR.mkdir(exist_ok=True)
        print("\nInitializing Cellpose for parameter sweep...")
        import torch
        USE_GPU = torch.cuda.is_available()
        model = models.CellposeModel(gpu=USE_GPU, model_type=CELLPOSE_MODEL)
        n_timepoints = zarr_store[TEST_ROW][TEST_WELL][str(TEST_FOV)]['0'].shape[0]
        for tp in ASSESS_TIMEPOINTS:
            tp_actual = tp if tp >= 0 else n_timepoints + tp
            assess_cellpose_params(
                zarr_store, TEST_ROW, TEST_WELL, TEST_FOV, model, OUTPUT_DIR,
                diameters=ASSESS_DIAMETERS,
                flow_thresholds=ASSESS_FLOW_THRESHOLDS,
                timepoint=tp_actual,
            )
        print("\nExiting after Cellpose assessment. Check the saved PNGs, then set ASSESS_CELLPOSE = False.")
        exit()
    
    print("\nChecking GPU...")
    import torch
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ No GPU - using CPU")
    
    print("\nInitializing Cellpose...")
    model = models.CellposeModel(gpu=USE_GPU, model_type=CELLPOSE_MODEL)
    
    if SINGLE_FOV_MODE:
        # ---- SLURM array path: process exactly one FOV then exit ----
        process_fov(zarr_store, _row, _well, _fov, model, OUTPUT_DIR)
        print(f"\nDone! Results in {OUTPUT_DIR / f'well_{_row}{_well}_FOV{_fov}'}")

    elif PROCESS_SINGLE_FOV_FIRST:
        dapi_mips, gfp_mips, tracked_masks, df = process_fov(
            zarr_store, TEST_ROW, TEST_WELL, TEST_FOV, model, OUTPUT_DIR
        )

        print("\nOpening napari...")
        viewer = napari.Viewer()

        viewer.add_image(dapi_mips, name='DAPI MIP', colormap='blue', blending='additive',
                        contrast_limits=[np.percentile(dapi_mips, 1), np.percentile(dapi_mips, 99)])
        viewer.add_image(gfp_mips, name='GFP MIP', colormap='green', blending='additive',
                        contrast_limits=[np.percentile(gfp_mips, 1), np.percentile(gfp_mips, 99)])
        viewer.add_labels(tracked_masks, name='Tracked Nuclei')

        napari.run()

    else:
        # ---- Local batch path: process all wells/FOVs sequentially ----
        all_measurements = []
        all_stats = []

        for row in ROWS:
            for well in WELLS:
                fovs = get_fovs(zarr_store, row, well)
                if not fovs:
                    print(f"  Skipping {row}{well} — no FOVs found in zarr store")
                    continue
                print(f"  {row}{well}: {len(fovs)} FOVs detected {fovs}")
                for fov in fovs:
                    try:
                        _, _, _, df = process_fov(zarr_store, row, well, fov, model, OUTPUT_DIR)
                        if len(df) > 0:
                            all_measurements.append(df)
                            stats = pd.read_csv(OUTPUT_DIR / f"well_{row}{well}_FOV{fov}" / "track_summary_stats.csv")
                            all_stats.append(stats)
                    except Exception as e:
                        print(f"\nERROR: Well {row}/{well}, FOV {fov}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        if all_measurements:
            pd.concat(all_measurements, ignore_index=True).to_csv(OUTPUT_DIR / "all_measurements.csv", index=False)
            pd.concat(all_stats, ignore_index=True).to_csv(OUTPUT_DIR / "all_track_stats.csv", index=False)

        print(f"\nDone! Results in {OUTPUT_DIR}")