#!/bin/bash
# ============================================================
#  run_bigru_ablations.sh
#  Submits BiGRU ablation jobs from config/experiment_list.txt
#
#  Features:
#  - Skips experiments that already have results.json (done)
#  - Can submit a specific stage only: --stage 1
#  - Dry run mode: --dry-run
#
#  Usage:
#    bash run_bigru_ablations.sh              # submit all pending
#    bash run_bigru_ablations.sh --stage 1    # submit stage 1 only
#    bash run_bigru_ablations.sh --dry-run    # print without submitting
# ============================================================

SAVE_ROOT="/data-fast/data-server/aclapes/SN-BAS-2025_savedata"
LIST_FILE="config/experiment_list.txt"
SLURM_SCRIPT="slurm/job_bigru.sh"
STAGE_FILTER=""
DRY_RUN=false

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --stage)   STAGE_FILTER="$2"; shift ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [ ! -f "$LIST_FILE" ]; then
    echo "ERROR: $LIST_FILE not found."
    echo "Run: python generate_configs.py first."
    exit 1
fi

mkdir -p logs

submitted=0
skipped=0
total=0

while IFS= read -r EXP; do
    [ -z "$EXP" ] && continue

    # Stage filter
    if [ -n "$STAGE_FILTER" ]; then
        STAGE=$(python3 -c "
import json, sys
try:
    cfg = json.load(open('config/$EXP.json'))
    print(cfg.get('stage', '?'))
except: print('?')
")
        [ "$STAGE" != "$STAGE_FILTER" ] && continue
    fi

    total=$((total + 1))

    # Skip if results already exist
    RESULTS="$SAVE_ROOT/$EXP/results.json"
    if [ -f "$RESULTS" ]; then
        echo "  [DONE]    $EXP"
        skipped=$((skipped + 1))
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  [WOULD SUBMIT] $EXP"
    else
        JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT" "$EXP")
        echo "  [SUBMITTED] $EXP  →  job $JOB_ID"
        submitted=$((submitted + 1))
    fi

done < "$LIST_FILE"

echo ""
echo "Total: $total | Submitted: $submitted | Already done: $skipped"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  python scripts/aggregate_bigru.py"
