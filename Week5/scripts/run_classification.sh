#!/usr/bin/env bash
# Run main_classification.py from starter/ (config/ and data/soccernetball/ are relative to that dir).
#
# From C6_Week5:
#   CUDA_VISIBLE_DEVICES=5 bash scripts/run_classification.sh --model baseline_c6_store

set -euo pipefail
_C6="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${_C6}/starter"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
exec python main_classification.py "$@"
