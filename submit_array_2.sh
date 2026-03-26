#!/usr/bin/env bash
# =============================================================================
# submit_array_2.sh  —  SLURM array launcher for script 2 (mNG trajectory analysis)
#
# One job per well. No GPU needed — CPU-only data analysis.
#
# Usage:
#   bash submit_array_2.sh              # submit all wells
#   bash submit_array_2.sh --dry-run    # print tasks without submitting
# =============================================================================

set -euo pipefail

# ---- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/2-OFFON_reporter_GFP_trajectories_analysis_ultrack_TEST.py"
LOG_DIR="$SCRIPT_DIR/slurm_logs_2"

# ---- wells to process ----
WELLS=(B1 B2 B3 C1 C2 C3)

# ---- SLURM resource settings ----
PARTITION="cpu"          # CPU partition (no GPU needed)
CPUS=4
MEM="32G"
TIME="02:00:00"
CONDA_ENV="/path/to/your/conda_env"

# ---- script 2 arguments (passed through to every job) ----
EXTRA_ARGS="--n-sd 10"    # stdev threshold (already the default, but explicit here)
# Add any other flags here, e.g.:
# EXTRA_ARGS="--n-sd 3 --min-duration 30 --save-svg"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

N_TASKS=${#WELLS[@]}
mkdir -p "$LOG_DIR"

echo "Wells to process (${N_TASKS}):"
for i in "${!WELLS[@]}"; do
    printf "  %2d  %s\n" "$((i+1))" "${WELLS[$i]}"
done

if $DRY_RUN; then
    echo ""
    echo "Dry-run — skipping submission."
    exit 0
fi

# ---- submit array job ----
# SLURM_ARRAY_TASK_ID is 0-indexed here for simplicity with bash arrays
sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=offon_traj
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=0-$((N_TASKS-1))
#SBATCH --output=${LOG_DIR}/job_%A_task_%a.out
#SBATCH --error=${LOG_DIR}/job_%A_task_%a.err

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

WELLS=(${WELLS[*]})
WELL=\${WELLS[\${SLURM_ARRAY_TASK_ID}]}

echo "Task \${SLURM_ARRAY_TASK_ID}: well=\${WELL}"
echo "Node: \$(hostname)"

python "${MAIN_SCRIPT}" --well "\${WELL}" ${EXTRA_ARGS}
EOF

echo ""
echo "Submitted ${N_TASKS} tasks."
echo "Logs → ${LOG_DIR}/"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/job_*_task_0.out"
