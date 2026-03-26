"""
generate_fov_list.py
--------------------
Enumerate all (row, well, fov) combinations present in the zarr store
and write them to a plain-text task list consumed by the SLURM array.

Usage:
    python generate_fov_list.py              # writes fov_list.txt next to this script
    python generate_fov_list.py --out /path/to/fov_list.txt
"""

import zarr
import argparse
from pathlib import Path

# ---- same constants as the main script ----
ZARR_PATH = "/hpc/projects/arias_group/Vincent_Turon-Lagot/Imaging_Experiments/20260321_A549_OFFON18_OFFON20_48hrs_dragonfly/2-register/20260321_A549_OFFON18_OFFON20_48hrs_dragonfly_registered.zarr"
ROWS  = ['B', 'C']
WELLS = ['1', '2', '3']
# -------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--out', default=str(Path(__file__).parent / 'fov_list.txt'))
args = parser.parse_args()

store = zarr.open(ZARR_PATH, mode='r')

tasks = []
for row in ROWS:
    row = row.strip()
    for well in WELLS:
        well = well.strip()
        try:
            keys = sorted([int(k) for k in store[row][well].keys() if k.isdigit()])
        except KeyError:
            print(f"  Skipping {row}{well} — not found in zarr store")
            continue
        for fov in keys:
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
