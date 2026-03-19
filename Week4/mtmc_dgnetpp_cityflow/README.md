# MTMC Vehicle Tracking (CityFlowV2) with DG-Net-PP-Inspired ReID

This repository provides a practical multi-camera tracking baseline for CityFlowV2-style data.
It is inspired by DG-Net-PP ideas (identity/style disentanglement) and adapted for vehicle ReID + MTMC association.

Implementation note: this code is a clean, task-focused reimplementation for CityFlow vehicles and does not copy source files from DG-Net-PP.

## What Is Implemented

- ReID crop extraction from CityFlow `gt/gt.txt` (training identities).
- DG-style vehicle ReID network:
  - shared CNN backbone
  - identity embedding branch
  - style embedding branch
  - decoder for reconstruction supervision
- ReID training with identity classification + triplet + reconstruction loss.
- Tracklet-level feature extraction from MTSC files.
- Cross-camera association using cosine similarity + temporal consistency.
- CityFlow Track-1 submission export format:
  - `camera_id obj_id frame_id xmin ymin width height xworld yworld`

## Repository Goals

- Keep the experimental split explicit and reproducible.
- Make the project runnable from the repository root with minimal manual editing.
- Provide a clean baseline that is easy to tune and extend.

## Repository Layout

```text
mtmc_dgnetpp_cityflow/
├── configs/              # experiment configuration
├── docs/                 # references and troubleshooting
├── src/mtmc/             # library code
├── tools/                # runnable entry scripts
├── outputs/              # generated artifacts (ignored by git)
├── Makefile              # convenience commands
├── README.md
└── requirements.txt
```

## Dataset Layout

Expected root: `../AI_CITY_CHALLENGE_2022_TRAIN` (configurable)

## Split Used Here

- Train: `S01`, `S04`
- Test: `S03`

This split is intentional and should be preserved during both training and evaluation.

## Setup

```bash
cd mtmc_dgnetpp_cityflow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The same `requirements.txt` covers both the MTMC pipeline and the patched CityFlow evaluator dependencies.

## Quick Start

If you prefer short commands, use the included `Makefile`:

```bash
make prepare
make train
make run
make eval-s03
```

List available targets with:

```bash
make help
```

## 1) Prepare ReID Crops

```bash
python tools/prepare_reid_crops.py \
  --data-root ../AI_CITY_CHALLENGE_2022_TRAIN \
  --sequences S01 S04 \
  --output-dir ./outputs/reid_data
```

This creates:

- `outputs/reid_data/crops/...jpg`
- `outputs/reid_data/train_index.csv`

## 2) Train DG-Style ReID Model

```bash
python tools/train_reid.py \
  --config configs/default.yaml \
  --train-index ./outputs/reid_data/train_index.csv \
  --output-dir ./outputs/reid_model
```

You can override backbone and image size per run:

```bash
python tools/train_reid.py \
  --config configs/default.yaml \
  --train-index ./outputs/reid_data/train_index.csv \
  --output-dir ./outputs/reid_model_resnet34_512 \
  --backbone resnet34 \
  --image-size 512 512
```

Supported backbones:

- `resnet18`
- `resnet34`

Main checkpoint:

- `outputs/reid_model/best.pt`

## 3) Run MTMC Association on S03

```bash
python tools/run_mtmc.py \
  --config configs/default.yaml \
  --data-root ../AI_CITY_CHALLENGE_2022_TRAIN \
  --sequence S03 \
  --checkpoint ./outputs/reid_model/best.pt \
  --output-file ./outputs/track1.txt
```

You can quickly test different MTSC inputs without editing config:

```bash
python tools/run_mtmc.py \
  --config configs/default.yaml \
  --data-root ../AI_CITY_CHALLENGE_2022_TRAIN \
  --sequence S03 \
  --checkpoint ./outputs/reid_model/best.pt \
  --mtsc-file mtsc_tc_ssd512.txt \
  --output-file ./outputs/track1.txt
```

## 4) Evaluate

```bash
cd ../AI_CITY_CHALLENGE_2022_TRAIN/eval
awk '$1>=10 && $1<=15' ground_truth_train.txt > ground_truth_s03.txt
python eval.py ground_truth_s03.txt ../../mtmc_dgnetpp_cityflow/outputs/track1.txt --dstype train
```

This evaluator has been patched to work offline: it auto-loads `roi.jpg` from local dataset folders and does not download from Drive.
If you use a custom ROI directory, pass `--roidir <path>`.

Adjust ground truth file according to your local split protocol.

## 6) Run 2x3 Comparison Grid (ResNet18/34 x 128/256/512)

A helper script is included to run all 6 experiments end-to-end (train + MTMC export):

```bash
bash tools/run_reid_grid.sh
```

Outputs are stored under:

- `outputs/experiments/resnet18_128/...`
- `outputs/experiments/resnet18_256/...`
- `outputs/experiments/resnet18_512/...`
- `outputs/experiments/resnet34_128/...`
- `outputs/experiments/resnet34_256/...`
- `outputs/experiments/resnet34_512/...`

Environment overrides (optional):

- `DATA_ROOT`
- `CONFIG`
- `TRAIN_INDEX`
- `SEQ`
- `MTSF`

Example:

```bash
DATA_ROOT=../AI_CITY_CHALLENGE_2022_TRAIN SEQ=S03 bash tools/run_reid_grid.sh
```

## Notes

- This is a strong baseline scaffold, not a leaderboard-tuned solution.
- You can swap `mtsc_file` in config to another baseline tracker file.
- `xworld yworld` are currently set to `-1 -1` in export (placeholders).
- MTMC tuning knobs live in `configs/default.yaml` under `mtmc`:
  - `min_sim`
  - `min_track_length`
  - `min_mean_area`
  - `keep_only_multicam`
  - `min_global_cameras`
  - `min_global_rows`
  - `max_global_ids`

## 5) Fast Post-Filter Tuning (No Re-Extraction)

If `track1.txt` has too many rows / low precision, quickly filter by global-id quality:

```bash
python tools/postfilter_track1.py \
  --input ./outputs/track1.txt \
  --output ./outputs/track1_filtered.txt \
  --min-global-cameras 5 \
  --min-global-rows 300 \
  --max-global-ids 18
```

Then evaluate `track1_filtered.txt` on `ground_truth_s03.txt`.

## Reproducibility Notes

- The repository default configuration is stored in `configs/default.yaml`.
- Generated artifacts under `outputs/` are ignored by git.
- The evaluator is expected at `../AI_CITY_CHALLENGE_2022_TRAIN/eval` and has been patched for offline ROI loading and modern dependency compatibility.

## References

See `docs/REFERENCES.md` for the dataset, DG-Net-PP, and tracking baseline references used by this repository.

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues encountered while setting up or evaluating this project.
