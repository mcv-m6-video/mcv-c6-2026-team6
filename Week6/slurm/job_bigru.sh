#!/bin/bash
# ============================================================
#  slurm/job_bigru.sh
#  Runs ONE BiGRU ablation experiment.
#
#  Usage:
#    sbatch slurm/job_bigru.sh bigru_hd512_l1_dr01_lr8e4_ce
# ============================================================

#SBATCH --job-name=bigru_ablation
#SBATCH --partition=mhigh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

if [ -z "$1" ]; then
    echo "ERROR: No experiment name provided."
    echo "Usage: sbatch slurm/job_bigru.sh <experiment_name>"
    exit 1
fi

EXPERIMENT=$1
echo "============================================"
echo "  Experiment : $EXPERIMENT"
echo "  Node       : $(hostname)"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Start      : $(date)"
echo "============================================"

mkdir -p logs
cd /ghome/group06/Ouss/c6

python main_spotting.py --model "$EXPERIMENT"

EXIT_CODE=$?
echo "============================================"
echo "  Exit code : $EXIT_CODE"
echo "  Done      : $(date)"
echo "============================================"
exit $EXIT_CODE
