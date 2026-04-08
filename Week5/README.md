# C6 Week 5 — Ball Action Classification (SN-BAS-2025)

> **Project 2, Task 1**
> Action classification on the [SoccerNet Ball Action Spotting 2025 (SN-BAS-2025)](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025) dataset.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
  - [Step 1 – Store Clip Features (first run)](#step-1--store-clip-features-first-run)
  - [Step 2 – Train & Evaluate](#step-2--train--evaluate)
  - [Step 3 – Report AP Scores](#step-3--report-ap-scores)
  - [Step 4 – Compute MACs / Parameter Count](#step-4--compute-macs--parameter-count)
- [Available Models & Experiments](#available-models--experiments)
- [Results](#results)
- [Support](#support)

---

## Overview

This repository contains the starter code and extended experiments for **Task 1: Ball Action Classification** in the C6 Video Analysis module. Given a sequence of video frames from soccer matches, the model must classify which of 12 ball-action classes (e.g., passes, shots, headers) occurs in each clip.

Key features implemented on top of the baseline:

- **Temporal encoders:** `maxpool` (baseline), `gru` (bidirectional GRU), `transformer`, and `attention` pooling
- **Loss functions:** `bce` (baseline), `focal` (Focal Loss γ=2), `weighted` (frequency-weighted BCE)
- **Augmentation:** `CenterCrop(224)`, `RandomResizedCrop`, temporal jitter, frame dropout
- **Training extras:** label smoothing, differential learning rates, test-time augmentation (TTA), gradient checkpointing

---

## Project Structure

```
c6w4/
├── starter/                    # Main source code
│   ├── main_classification.py  # Entry point: train + evaluate
│   ├── extract_frames_snb.py   # Extract video frames to disk
│   ├── download_frames_snb.py  # Alternative: download pre-extracted frames
│   ├── config/                 # JSON configuration files (one per experiment)
│   ├── dataset/
│   │   ├── datasets.py         # Dataset loaders (train/val/test splits)
│   │   └── frame.py            # Frame-level reading utilities
│   ├── model/
│   │   ├── model_classification.py  # Model definition (backbones, temporal heads, losses)
│   │   └── modules.py               # Base classes (ABCModel, FCLayers, etc.)
│   └── util/
│       ├── eval_classification.py   # Average Precision evaluation
│       ├── dataset.py               # Dataset utility functions
│       └── io.py                    # JSON I/O helpers
├── scripts/
│   ├── run_classification.sh    # Wrapper to run from project root
│   ├── prepare_sn_bas_data.sh   # Download & unzip SN-BAS-2025 splits
│   ├── extract_split_zips.py    # Python helper for zip extraction
│   ├── verify_data_layout.py    # Sanity-check extracted dataset layout
│   ├── compute_macs.py          # Measure FLOPs and parameter count
│   └── report_ap10_ap12.py      # Report AP10 / AP12 on the test split
├── runs/                        # Training outputs (checkpoints, loss logs)
├── weights/                     # Best model checkpoints (saved manually)
├── cache/                       # Preprocessed clip cache
├── environment.yml              # Conda environment definition
├── requirements-c6-pip.txt      # Pip dependencies (with thop + pyzipper)
└── commands.txt                 # Full step-by-step commands reference
```

---

## Requirements

- Python 3.11
- CUDA-capable GPU (experiments used CUDA 12.1)
- Conda (recommended)

Key Python packages:

| Package | Version |
|---|---|
| torch | 2.3.1 |
| torchvision | 0.18.1 |
| timm | 1.0.3 |
| numpy | 1.26.4 |
| scikit-learn | 1.6.1 |
| opencv-contrib-python | 4.11.0.86 |
| huggingface_hub | 0.29.3 |
| pyzipper | ≥ 0.3.6 |

---

## Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate c6
```

### 2. Install PyTorch with the correct CUDA wheel

If PyTorch with CUDA is missing after environment creation:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining pip dependencies

```bash
pip install -r requirements-c6-pip.txt
```

---

## Dataset Setup

### 1. Set up environment variables

Add the following to your shell (or paste at the start of every session):

```bash
export C6_ROOT=/path/to/c6w4        # adjust to your checkout path
export CUDA_VISIBLE_DEVICES=0       # adjust to your GPU index
source "${C6_ROOT}/scripts/setup_cache_env.sh"
```

### 2. Authenticate with Hugging Face

```bash
export HF_TOKEN="your_huggingface_token"
huggingface-cli login --token "${HF_TOKEN}"
```

### 3. Download the SN-BAS-2025 dataset

```bash
huggingface-cli download SoccerNet/SN-BAS-2025 \
    --repo-type dataset \
    --local-dir "${C6_ROOT}/data/hf_sn_bas"
```

> **Note:** Access requires accepting the SoccerNet NDA. Store the NDA password in `${C6_ROOT}/sn_bas_pass.txt` (already git-ignored):
> ```bash
> printf '%s' 'YOUR_NDA_PASSWORD' > "${C6_ROOT}/sn_bas_pass.txt"
> chmod 600 "${C6_ROOT}/sn_bas_pass.txt"
> ```

### 4. Extract the dataset splits

```bash
export SN_BAS_ZIP_PASSWORD=$(cat "${C6_ROOT}/sn_bas_pass.txt")
bash "${C6_ROOT}/scripts/prepare_sn_bas_data.sh"
```

Verify the layout afterwards:

```bash
python3 "${C6_ROOT}/scripts/verify_data_layout.py"
# Expected output: all X games ok
```

### 5. Extract video frames

This is a one-time step (large, takes a while):

```bash
cd "${C6_ROOT}/starter"
python extract_frames_snb.py \
    --video_dir "${C6_ROOT}/data/sn_bas_2025" \
    -o "${C6_ROOT}/data/frames"
```

---

## Configuration

All experiment settings live in `starter/config/<model_name>.json`. You must update at least these three paths before running:

| Parameter | Description |
|---|---|
| `frame_dir` | Directory where extracted frames are stored |
| `save_dir` | Directory for checkpoints and logs |
| `labels_dir` | Directory where dataset label files are stored |

Other notable parameters:

| Parameter | Description |
|---|---|
| `store_mode` | `"store"` on first run (caches clips); `"load"` on subsequent runs |
| `feature_arch` | Backbone: `rny002`, `rny004`, `rny008`, `rny002_gsf`, `hiera_base` |
| `temporal_arch` | Temporal head: `maxpool`, `gru`, `transformer`, `attention` |
| `loss_type` | Loss function: `bce`, `focal`, `weighted` |
| `clip_len` | Number of frames per clip |
| `stride` | Sample 1 out of every N frames |
| `num_epochs` | Total training epochs |
| `learning_rate` | Initial learning rate |
| `device` | `"cuda"` or `"cpu"` |

See `starter/config/README.md` for a full parameter reference.

---

## Running the Code

All commands below assume you are in `${C6_ROOT}/starter` and have activated the `c6` conda environment.

### Step 1 – Store Clip Features (first run)

Run once per configuration to preprocess and cache clips (requires `store_mode: "store"` in the config):

```bash
python main_classification.py --model baseline_c6_store
```

After this completes, switch `store_mode` to `"load"` in the config (or use a separate `_store` config variant as provided).

### Step 2 – Train & Evaluate

```bash
python main_classification.py --model <model_name>
```

For example:

```bash
# Baseline
python main_classification.py --model baseline_c6

# With bidirectional GRU temporal encoder
python main_classification.py --model exp_gru

# With Focal Loss + GRU (best configuration)
python main_classification.py --model exp_focal_gru_best
```

You can also run from the project root using the wrapper script:

```bash
cd "${C6_ROOT}"
CUDA_VISIBLE_DEVICES=0 bash scripts/run_classification.sh --model baseline_c6
```

Checkpoints and loss logs are saved to `runs/<model_name>/`.

### Step 3 – Report AP Scores

Compute AP10 and AP12 on the test split:

```bash
cd "${C6_ROOT}/starter"
python3 "${C6_ROOT}/scripts/report_ap10_ap12.py" --model baseline_c6

# With rare-class masking at inference
python3 "${C6_ROOT}/scripts/report_ap10_ap12.py" --model baseline_c6 --mask_rare_inference
```

### Step 4 – Compute MACs / Parameter Count

```bash
cd "${C6_ROOT}"
python3 scripts/compute_macs.py --model baseline_c6
python3 scripts/compute_macs.py --model ablation_rny004
```

### Save the Best Checkpoint

```bash
cp "${C6_ROOT}/runs/<model_name>/checkpoints/checkpoint_best.pt" "${C6_ROOT}/weights/"
```

---

## Available Models & Experiments

| Config name | Description |
|---|---|
| `baseline_c6` | Baseline — RNY002 + max-pool + BCE |
| `exp_gru` | Bidirectional GRU temporal encoder |
| `exp_transformer` | Transformer temporal encoder |
| `exp_attention` | Learned attention pooling |
| `exp_focal` | Focal Loss (γ=2, α=0.25) |
| `exp_focal_gru` | Focal Loss + GRU |
| `exp_focal_gru_best` | Best overall: Focal + GRU + augmentation |
| `exp_weighted` | Frequency-weighted BCE |
| `exp_hiera` | Meta Hiera-Base backbone |
| `ablation_rny004` | Ablation: larger backbone (RNY004) |
| `ablation_rny008` | Ablation: largest backbone (RNY008) |
| `ablation_stride1` | Ablation: stride=1 (dense sampling) |
| `ablation_clip40` | Ablation: clip length = 40 |
| `ablation_overlap050` | Ablation: clip overlap = 0.5 |
| `ablation_lr0004` | Ablation: learning rate = 0.0004 |
| `ablation_bs2` | Ablation: batch size = 2 |

`_store` variants (e.g. `baseline_c6_store`) are identical configs with `store_mode: "store"` for the initial caching run.

---

## Results

Training metrics (loss curves) are saved as JSON in `runs/<model_name>/loss.json`. Final AP scores are printed to stdout and can be re-computed at any time with `report_ap10_ap12.py`.

Example output format:

```
+----------------------+-----------+
| Class                |    AP (%) |
+======================+===========+
| PASS                 |     72.14 |
| ...                  |       ... |
+----------------------+-----------+
+----------------------------+-------+
| AP12 (all)                 | 55.30 |
| AP10 (excl. FREEKICK+GOAL) | 58.12 |
+----------------------------+-------+
```

---

## Support

For questions related to the code, please contact:

- **Albert Clapés** – [aclapes@ub.edu](mailto:aclapes@ub.edu)
- **Artur Xarles** – [arturxe@gmail.com](mailto:arturxe@gmail.com)
