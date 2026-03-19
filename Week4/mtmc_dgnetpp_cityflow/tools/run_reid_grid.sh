#!/usr/bin/env bash
set -euo pipefail

# Run 2x3 ReID grid:
# backbones: resnet18, resnet34
# image sizes: 128x128, 256x256, 512x512

DATA_ROOT="${DATA_ROOT:-../AI_CITY_CHALLENGE_2022_TRAIN}"
CONFIG="${CONFIG:-configs/default.yaml}"
TRAIN_INDEX="${TRAIN_INDEX:-./outputs/reid_data/train_index.csv}"
SEQ="${SEQ:-S03}"
MTSF="${MTSF:-mtsc_tc_ssd512.txt}"

BACKBONES=(resnet18 resnet34)
SIZES=(128 256 512)

mkdir -p outputs/experiments

for bb in "${BACKBONES[@]}"; do
  for sz in "${SIZES[@]}"; do
    EXP="${bb}_${sz}"
    MODEL_DIR="./outputs/experiments/${EXP}/reid_model"
    TRACK_FILE="./outputs/experiments/${EXP}/track1.txt"

    echo "[INFO] Training ${EXP}"
    python tools/train_reid.py \
      --config "${CONFIG}" \
      --train-index "${TRAIN_INDEX}" \
      --output-dir "${MODEL_DIR}" \
      --backbone "${bb}" \
      --image-size "${sz}" "${sz}"

    echo "[INFO] Running MTMC ${EXP}"
    python tools/run_mtmc.py \
      --config "${CONFIG}" \
      --data-root "${DATA_ROOT}" \
      --sequence "${SEQ}" \
      --checkpoint "${MODEL_DIR}/best.pt" \
      --mtsc-file "${MTSF}" \
      --output-file "${TRACK_FILE}"

    echo "[INFO] Done ${EXP} -> ${TRACK_FILE}"
  done
done

echo "[INFO] All experiments finished."
