"""
generate_fov_list.py
--------------------
Enumerate all (row, well, fov) combinations present in the zarr store
and write them to a plain-text task list consumed by the SLURM array.

Rows, wells, and FOVs are auto-detected from the zarr store by default.
Use --rows / --wells to restrict to a subset.

Usage:
    python generate_fov_list.py --zarr-path /path/to/data.zarr
    python generate_fov_list.py --zarr-path /path/to/data.zarr --rows B C --wells 1 2 3
    python generate_fov_list.py --out /path/to/fov_list.txt
"""

import zarr
import argparse
from pathlib import Path

# Default zarr path (overridable with --zarr-path)
ZARR_PATH = "/path/to/your/data.zarr"


def get_rows(store):
    """Sorted row keys present in the zarr store."""
    return sorted([k for k in store.keys() if not k.startswith('.')])


def get_wells(store, row):
    """Sorted well keys under a given row."""
    try:
        return sorted([k for k in store[row].keys() if not k.startswith('.')])
    except KeyError:
        return []


def get_fovs(store, row, well):
    """Sorted FOV indices (integer keys) under a given row/well."""
    try:
        return sorted([int(k) for k in store[row][well].keys() if k.isdigit()])
    except KeyError:
        return []


parser = argparse.ArgumentParser()
parser.add_argument('--out', default=str(Path(__file__).parent / 'fov_list.txt'))
parser.add_argument('--zarr-path', default=ZARR_PATH,
                    help="Path to the input zarr store (overrides the ZARR_PATH constant)")
parser.add_argument('--rows', nargs='+', default=None,
                    help="Restrict to these rows (default: auto-detect all rows)")
parser.add_argument('--wells', nargs='+', default=None,
                    help="Restrict to these wells (default: auto-detect all wells)")
args = parser.parse_args()

store = zarr.open(args.zarr_path, mode='r')

rows = args.rows if args.rows is not None else get_rows(store)
print(f"Rows to scan: {rows}")

tasks = []
for row in rows:
    row = row.strip()
    wells = args.wells if args.wells is not None else get_wells(store, row)
    for well in wells:
        well = well.strip()
        fovs = get_fovs(store, row, well)
        if not fovs:
            print(f"  Skipping {row}{well} — no FOVs found in zarr store")
            continue
        for fov in fovs:
            line = f"{row} {well} {fov}".replace('\r', '').replace('\n', '')
            # Validate: must be exactly 3 whitespace-separated fields
            if len(line.split()) != 3:
                raise ValueError(f"Malformed task line (hidden characters?): {repr(line)}")
            tasks.append(line)
            print(f"  Found: {row}/{well} FOV {fov}")

with open(args.out, 'w', newline='\n') as f:
    f.write('\n'.join(tasks) + '\n')

# Post-write verification
bad = []
with open(args.out) as f:
    for i, line in enumerate(f, 1):
        parts = line.strip().split()
        if len(parts) != 3:
            bad.append((i, repr(line)))

if bad:
    print("\nWARNING: corrupted lines in output file:")
    for lineno, content in bad:
        print(f"  Line {lineno}: {content}")
else:
    print(f"\nWrote {len(tasks)} tasks to {args.out} (all lines verified OK)")
