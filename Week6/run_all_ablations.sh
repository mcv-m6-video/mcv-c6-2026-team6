#!/bin/bash
# ============================================================
#  run_all_ablations.sh
#  Submit all ablation experiments as a SLURM array job.
#  Each experiment runs sequentially in its own job slot.
#
#  Usage:
#    bash run_all_ablations.sh
#
#  To add a new experiment, add its config name to EXPERIMENTS.
# ============================================================

# ── List of experiment names (must match config/<name>.json) ──
EXPERIMENTS=(
    "baseline"
    "lstm"
    "lstm_2layer"
    "tcn"
    "transformer"
    "focal_loss"
)

echo "Submitting ${#EXPERIMENTS[@]} ablation jobs..."

for EXP in "${EXPERIMENTS[@]}"; do
    JOB_ID=$(sbatch --parsable slurm/job_single.sh "$EXP")
    echo "  Submitted $EXP  →  SLURM job $JOB_ID"
done

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       logs/<experiment_name>.out"
