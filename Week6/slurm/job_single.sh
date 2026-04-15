#!/bin/bash
# ============================================================
#  slurm/job_single.sh
#  Runs ONE ablation experiment.
#  Called by run_all_ablations.sh with the experiment name.
#
#  Usage (manual):
#    sbatch slurm/job_single.sh baseline
#    sbatch slurm/job_single.sh lstm
# ============================================================

#SBATCH --job-name=bas_ablation
#SBATCH --partition=mhigh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out  # logs/bas_ablation_<jobid>.out
#SBATCH --error=logs/%x_%j.err

if [ -z "$1" ]; then
    echo "ERROR: No experiment name provided."
    echo "Usage: sbatch slurm/job_single.sh <experiment_name>"
    exit 1
fi

EXPERIMENT=$1
echo "============================================"
echo "  Experiment : $EXPERIMENT"
echo "  Node       : $(hostname)"
echo "  Start      : $(date)"
echo "============================================"

mkdir -p logs

cd /ghome/group06/Ouss/c6

python main_spotting.py --model "$EXPERIMENT"

echo "============================================"
echo "  Done : $(date)"
echo "============================================"
