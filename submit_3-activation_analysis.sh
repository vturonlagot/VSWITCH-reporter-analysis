#!/usr/bin/env bash
# =============================================================================
# submit_3-activation_analysis.sh  —  SLURM launcher for script 3 (activation analysis + figures)
#
# One job for the whole plate. Script 3 reads the trajectory output from
# script 2 (--analysis-dir) and the tracking output from script 1
# (--tracking-dir), then writes figures/tables to --output-dir. CPU-only.
#
# Usage:
#   bash submit_3-activation_analysis.sh              # submit the analysis job
#   bash submit_3-activation_analysis.sh --dry-run    # print the invocation without submitting
# =============================================================================

set -euo pipefail

# ---- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/3-activation_analysis.py"
LOG_DIR="$SCRIPT_DIR/slurm_logs_3"

BASE="/path/to/your/output"
ANALYSIS_DIR="$BASE/2-trajectories"          # script 2 output
TRACKING_DIR="$BASE/1-nuclear_analysis"      # script 1 output
OUTPUT_DIR="$BASE/3-activation_analysis"     # script 3 output

# ---- wells to process ----
# 'all' auto-detects wells from the tracking directory; or list them: B1 B2 C1
WELLS="all"

# ---- SLURM resource settings ----
PARTITION="cpu"          # CPU partition (no GPU needed)
CPUS=8
MEM="64G"
TIME="04:00:00"
CONDA_ENV="vswitch_analysis"

# ---- extra script 3 arguments (passed through) ----
# Run `python 3-activation_analysis.py --help` for the full option list.
#
# --conditions pools wells into a named condition (NAME:WELL1,WELL2), producing
# well_{NAME}_... panels. Edit to match your plate layout; the example below
# pools B2+B3 into a "DENV" condition.
EXTRA_ARGS="--conditions DENV:B2,B3"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$LOG_DIR"

echo "Script 3 — activation analysis"
echo "  analysis-dir : $ANALYSIS_DIR"
echo "  tracking-dir : $TRACKING_DIR"
echo "  output-dir   : $OUTPUT_DIR"
echo "  wells        : $WELLS"

if $DRY_RUN; then
    echo ""
    echo "Dry-run — would run:"
    echo "  python $MAIN_SCRIPT \\"
    echo "      --analysis-dir $ANALYSIS_DIR \\"
    echo "      --tracking-dir $TRACKING_DIR \\"
    echo "      --output-dir $OUTPUT_DIR \\"
    echo "      --well $WELLS ${EXTRA_ARGS}"
    exit 0
fi

# ---- submit job ----
sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=offon_activation
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/job_%j.out
#SBATCH --error=${LOG_DIR}/job_%j.err

module load anaconda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Node: \$(hostname)"

python "${MAIN_SCRIPT}" \
    --analysis-dir "${ANALYSIS_DIR}" \
    --tracking-dir "${TRACKING_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --well ${WELLS} ${EXTRA_ARGS}
EOF

echo ""
echo "Submitted script 3."
echo "Logs → ${LOG_DIR}/"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/job_*.out"
