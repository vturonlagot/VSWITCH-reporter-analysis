#!/usr/bin/env bash
# =============================================================================
# submit_array.sh  —  SLURM array launcher for the nuclear tracking pipeline
#
# Usage:
#   bash submit_array.sh              # generate task list + submit
#   bash submit_array.sh --dry-run    # generate list only, don't submit
# =============================================================================

set -euo pipefail

# ---- paths (edit if needed) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/1-OFFON_reporter_image_analysis_ultrack_vTEST.py"
GEN_SCRIPT="$SCRIPT_DIR/generate_fov_list.py"
FOV_LIST="$SCRIPT_DIR/fov_list.txt"
LOG_DIR="$SCRIPT_DIR/slurm_logs"

# ---- SLURM resource settings (adjust for your cluster) ----
PARTITION="gpu"          # partition/queue name
GRES="gpu:1"             # GPU resource (e.g. gpu:a100:1 if you need a specific type)
CPUS=4                   # CPUs per task (matches n_workers in ultrack config)
MEM="32G"                # RAM per task
TIME="08:00:00"          # wall-clock limit per FOV
CONDA_ENV="/hpc/mydata/vincent.turon-lagot/conda_env/imageprocessing"
MAX_PARALLEL=8           # max simultaneously running array tasks (%N in --array)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ---- Step 1: generate FOV list ----
echo "Generating FOV list..."
mkdir -p "$LOG_DIR"
python "$GEN_SCRIPT" --out "$FOV_LIST"

N_TASKS=$(wc -l < "$FOV_LIST")
if [[ "$N_TASKS" -eq 0 ]]; then
    echo "ERROR: fov_list.txt is empty — nothing to submit."
    exit 1
fi
echo "Found $N_TASKS FOVs."

if $DRY_RUN; then
    echo "Dry-run — skipping submission."
    python3 -c "
with open('$FOV_LIST') as f:
    for i, line in enumerate(f, 1):
        print(f'{i:3d}  {line.rstrip()}')
"
    exit 0
fi

# ---- Step 2: submit array job ----
# Each task reads its own line from fov_list.txt (1-indexed via SLURM_ARRAY_TASK_ID)
sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=nuclear_tracking
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --array=1-${N_TASKS}%${MAX_PARALLEL}
#SBATCH --output=${LOG_DIR}/job_%A_task_%a.out
#SBATCH --error=${LOG_DIR}/job_%A_task_%a.err

# Activate environment
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Gurobi license (not sourced from .bashrc in batch jobs)
export GRB_LICENSE_FILE="\$HOME/gurobi.lic"

# Read this task's row/well/fov from the list (SLURM_ARRAY_TASK_ID is 1-indexed)
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${FOV_LIST}" | tr -d '\r')
ROW=\$(echo "\$LINE"  | awk '{print \$1}')
WELL=\$(echo "\$LINE" | awk '{print \$2}')
FOV=\$(echo "\$LINE"  | awk '{print \$3}')

# Validate — fail loudly if any field is missing
if [[ -z "\$ROW" || -z "\$WELL" || -z "\$FOV" ]]; then
    echo "ERROR: malformed line \${SLURM_ARRAY_TASK_ID} in fov_list.txt: '\${LINE}'"
    exit 1
fi

echo "Task \${SLURM_ARRAY_TASK_ID}: row=\${ROW} well=\${WELL} fov=\${FOV}"
echo "Node: \$(hostname)  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python "${MAIN_SCRIPT}" --row "\${ROW}" --well "\${WELL}" --fov "\${FOV}"
EOF

echo "Submitted ${N_TASKS} tasks (max ${MAX_PARALLEL} running at once)."
echo "Logs → ${LOG_DIR}/"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/job_*_task_1.out"
