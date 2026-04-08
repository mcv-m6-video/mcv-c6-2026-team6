#!/usr/bin/env bash
# Redirect Hugging Face / PyTorch caches into C6_Week5/cache (not $HOME).
#
# Usage:
#   source /data/113-2/users/kpurkayastha/MCV/C6_Week5/scripts/setup_cache_env.sh
#
# From C6_Week5:
#   source scripts/setup_cache_env.sh

_C6="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${_C6}/cache"
mkdir -p "${CACHE_DIR}"

export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/transformers"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"

echo "Cache redirected to: ${CACHE_DIR}"
